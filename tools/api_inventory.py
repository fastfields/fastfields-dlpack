#!/usr/bin/env python3
"""Inventory the fastfields Python exposure: hub API vs. binding vs. wrappers.

Answers, reproducibly, the question "which operations of the C++ hub can a
Python user actually reach, and through which layer?". The stack is

    fastfields-lib (C++ hub)   <- source of truth: include/fastfields/api/*.h
      -> fastfields.dlpack     <- the raw nanobind binding (this package)
        -> fastfields.{numpy,cupy,torch}
          -> fastfields.auto   <- passthrough to one of those wrappers

and each arrow is a place an operation can quietly fail to appear. Before this
script that was re-derived by hand (and by ad-hoc regex) every time somebody
asked; the counts disagreed with each other and with the tree.

Usage
-----
    python tools/api_inventory.py                 # table + summary
    python tools/api_inventory.py --format json   # machine-readable
    python tools/api_inventory.py --check         # exit 1 on an undoc'd gap

What each column means
----------------------
hub
    A public entry point declared in the hub's ``include/fastfields/api/*.h``.
    Found by parsing the headers of the ``_fastfields_lib`` source tree --
    the commit pinned in ``fastfields-lib.sha`` and fetched at build time --
    or of ``--lib-dir``, so it tracks the pinned hub, not a remembered list.
    Only
    ``api/*.h`` is read: ``api/cpu/`` and ``api/cuda/`` are the per-backend
    dtype-dispatch layers, not the device-dispatching hub surface, and
    ``checks.h`` declares inline helpers rather than ops.
dlpack
    Exported by ``fastfields.dlpack`` (its ``__all__``).
numpy / cupy / torch
    Exposed by that wrapper package's ``__all__``, counting the in-place
    (trailing ``_``) and out-of-place spellings of one hub op as the same op:
    ``field_adddiag`` and ``field_adddiag_`` both reach ``field_adddiag_``,
    ``dt_euclidean_`` reaches ``dt_euclidean``. A wrapper is only reported
    when it is importable; otherwise its column reads ``n/a``.
auto
    Exposed by ``fastfields.auto``. ``fastfields.auto`` dispatches on the type
    of the first array argument, so an op whose first argument is a *shape*
    rather than an array (``field_diag``, ``field_kernel``, ``flow_diag``,
    ``flow_kernel``) cannot be dispatched there by construction and is listed
    as a documented gap, not a hole.

By default the wrapper layers are read from the *installed* packages. Pass
``--src-root DIR`` to read sibling source checkouts (``DIR/fastfields-numpy``,
...) instead, which is what a developer looking at five working trees wants,
and which needs none of them installed.

Deliberate gaps live in ``EXPECTED_GAPS`` below, each with the reason. That
list is the documentation: ``--check`` fails on a gap that is *not* in it, so a
new hub function (or a wrapper that drops one) is a red check rather than a
discovery months later.
"""

from __future__ import annotations

import argparse
import ast
import glob
import importlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

#: Wrapper layers, in stack order, and the sibling repo each one ships from.
WRAPPERS = (
    ("numpy", "fastfields.numpy", "fastfields-numpy"),
    ("cupy", "fastfields.cupy", "fastfields-cupy"),
    ("torch", "fastfields.torch", "fastfields-torch"),
    ("auto", "fastfields.auto", "fastfields"),
)

#: Gaps that are deliberate: ``(layer, hub op) -> why``. ``--check`` ignores
#: these and fails on anything else, so adding a line here is a decision
#: that shows up in review rather than a silent omission.
EXPECTED_GAPS: dict[tuple[str, str], str] = {
    # -- fastfields.dlpack ---------------------------------------------------
    # (none: the binding covers the whole hub surface;
    #  tests/test_api_coverage.py turns that into a check.)
    #
    # -- numpy / cupy: no autograd, so no use for the adjoints ---------------
    **{
        (layer, op): (
            "no autograd in this wrapper: the *_backward adjoints exist to "
            "differentiate through pushpull, which only fastfields.torch "
            "does. Still reachable raw via fastfields.dlpack."
        )
        for layer in ("numpy", "cupy")
        for op in (
            "pull_backward",
            "push_backward",
            "count_backward",
            "grad_backward",
        )
    },
    # -- torch: the adjoints are *used*, not re-exported ---------------------
    **{
        ("torch", op): (
            "wired into the autograd.Function for the forward op rather than "
            "exported as a function; calling the forward op and "
            "``.backward()`` is the supported spelling."
        )
        for op in (
            "pull_backward",
            "push_backward",
            "count_backward",
            "grad_backward",
            "sym_matvec_backward",
        )
    },
    # -- torch: solver entries that need an autograd decision first ----------
    **{
        ("torch", op): (
            "in-place preconditioner solve; wrapping it means deciding how "
            "autograd treats a mutated buffer (API_CONTRACT.md, 'In-place "
            "policy'). The out-of-place *_precond is exposed."
        )
        for op in ("field_precond_", "flow_precond_")
    },
    ("torch", "field_cg"): (
        "iterative solver whose gradient is a solve with the same operator "
        "-- a real autograd.Function, not a mechanical wrapper. field_relax "
        "is the in-place solver exposed today."
    ),
    # -- cupy: the op cannot run on a cupy array at all ----------------------
    ("cupy", "field_cg"): (
        "CPU-only in fastfields-lib (fastfields-lib#34): the hub rejects "
        "CUDA tensors, so a cupy wrapper could only ever raise. field_relax "
        "runs on CUDA and solves the same system."
    ),
    # -- fastfields.auto: cannot dispatch without an array -------------------
    **{
        ("auto", op): (
            "allocates from a *shape*, so there is no array to dispatch on; "
            "call it on a concrete backend (fastfields.numpy/.cupy/.torch)."
        )
        for op in ("field_diag", "field_kernel", "flow_diag", "flow_kernel")
    },
    **{
        ("auto", op): (
            "adjoint of a pushpull op; only fastfields.torch has a use for "
            "it and it is wired into autograd there, so there is nothing "
            "uniform to dispatch."
        )
        for op in (
            "pull_backward",
            "push_backward",
            "count_backward",
            "grad_backward",
        )
    },
}

#: Wrapper spellings that are the same hub op under a different name. Both
#: in-place/out-of-place spellings of an accumulate primitive, and the
#: trailing-underscore in-place forms the wrappers add, collapse onto the hub
#: entry point they call.
_ALIAS_RE = re.compile(r"^(?P<stem>.+?)_?$")


def hub_api(lib_dir: str) -> dict[str, str]:
    """Return ``{op: header}`` for every public hub entry point.

    Parameters
    ----------
    lib_dir
        A ``fastfields-lib`` source tree (the directory holding ``include/``).

    Returns
    -------
    dict
        Operation name -> the ``api/*.h`` basename that declares it.
    """
    pattern = os.path.join(lib_dir, "include", "fastfields", "api", "*.h")
    headers = sorted(glob.glob(pattern))
    if not headers:
        raise SystemExit(
            f"no hub headers under {pattern!r}: build the package once "
            "to fetch the pinned fastfields-lib into _fastfields_lib, or "
            "pass --lib-dir"
        )
    api: dict[str, str] = {}
    for path in headers:
        name = os.path.basename(path)
        if name == "checks.h":
            # Inline device predicates / assertions, not operations.
            continue
        src = open(path, encoding="utf-8").read()
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
        src = re.sub(r"//.*", "", src)
        # Every hub entry point is a file-scope `void <name>(`; the
        # templated helpers live in core/ and the backend api/{cpu,cuda}/.
        for op in re.findall(r"(?m)^void\s+(\w+)\s*\(", src):
            api[op] = name
    return api


def _module_all(mod_name: str) -> list[str] | None:
    """``__all__`` of an importable module, or None if it will not import."""
    try:
        mod = importlib.import_module(mod_name)
    except Exception:  # ImportError, or cupy raising with no GPU present
        return None
    return list(getattr(mod, "__all__", ()))


def _source_all(repo_dir: str, mod_name: str) -> list[str] | None:
    """``__all__`` read from a source checkout, without importing it.

    Lets the inventory run over five sibling working trees with nothing
    installed and no compiled extension present.
    """
    rel = os.path.join(*mod_name.split("."), "__init__.py")
    path = os.path.join(repo_dir, rel)
    if not os.path.exists(path):
        return None
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "__all__" for t in node.targets
        ):
            return [
                e.value
                for e in node.value.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)
            ]
    return None


def layer_names(src_root: str | None) -> dict[str, list[str] | None]:
    """Collect ``__all__`` for the binding and every wrapper layer."""
    out: dict[str, list[str] | None] = {}
    if src_root:
        out["dlpack"] = _source_all(
            os.path.join(src_root, "fastfields-dlpack"), "fastfields.dlpack"
        )
        for key, mod, repo in WRAPPERS:
            out[key] = _source_all(os.path.join(src_root, repo), mod)
    else:
        out["dlpack"] = _source_all(REPO, "fastfields.dlpack") or _module_all(
            "fastfields.dlpack"
        )
        for key, mod, _repo in WRAPPERS:
            out[key] = _module_all(mod)
    return out


def reachable(names: list[str], api: dict[str, str]) -> dict[str, set[str]]:
    """Map each hub op to the exported names that reach it.

    A wrapper may spell one hub op two ways. The hub's accumulate primitives
    exist only in place (``field_adddiag_``), and the wrappers add an
    out-of-place spelling (``field_adddiag``, "copy then accumulate") that
    calls the very same entry point; conversely ``dt_euclidean`` /
    ``spline_coeff`` exist only out of place in the hub, and the wrappers add
    an in-place ``dt_euclidean_`` / ``spline_coeff_``. So an exported name
    also reaches the hub op that differs from it by a trailing underscore.

    That aliasing is only safe where the hub does *not* itself declare both
    spellings. It does for four pairs -- ``sym_solve``/``sym_solve_``,
    ``sym_invert``/``sym_invert_``, ``{field,flow}_precond`` /
    ``{field,flow}_precond_`` -- which are genuinely different operations
    (one allocates, one overwrites its input). Aliasing across those would
    report the in-place op as covered by a wrapper that only has the
    out-of-place one, which is exactly the kind of false "yes" this script
    exists to stop. Hence: alias only when the exported name is not itself a
    hub entry point.
    """
    hits: dict[str, set[str]] = {op: set() for op in api}
    for name in names:
        if name in hits:
            hits[name].add(name)
            continue
        for cand in (name + "_", name[:-1] if name.endswith("_") else name):
            if cand in hits:
                hits[cand].add(name)
    return hits


def build(lib_dir: str, src_root: str | None) -> dict:
    api = hub_api(lib_dir)
    layers = layer_names(src_root)
    rows = []
    for op, header in sorted(api.items(), key=lambda kv: (kv[1], kv[0])):
        row = {"op": op, "header": header, "layers": {}}
        for layer, names in layers.items():
            if names is None:
                row["layers"][layer] = None  # layer unavailable
            else:
                hit = reachable(names, api)[op]
                row["layers"][layer] = sorted(hit)
        rows.append(row)
    return {
        "lib_dir": lib_dir,
        "api_size": len(api),
        "rows": rows,
        "layers": {
            k: (None if v is None else len(v)) for k, v in layers.items()
        },
    }


def gaps(report: dict) -> list[tuple[str, str]]:
    """Every (layer, op) pair not reached, minus the documented ones."""
    out = []
    for row in report["rows"]:
        for layer, hit in row["layers"].items():
            if hit is None or hit:
                continue
            if (layer, row["op"]) in EXPECTED_GAPS:
                continue
            out.append((layer, row["op"]))
    return out


_LAYERS = ("dlpack", "numpy", "cupy", "torch", "auto")


def print_table(report: dict) -> None:
    width = max(len(r["op"]) for r in report["rows"]) + 2
    head = "op".ljust(width) + "".join(x.ljust(8) for x in _LAYERS)
    print(head)
    print("-" * len(head))
    header = None
    for row in report["rows"]:
        if row["header"] != header:
            header = row["header"]
            print(f"# api/{header}")
        cells = []
        for layer in _LAYERS:
            hit = row["layers"].get(layer)
            if hit is None:
                cells.append("n/a")
            elif hit:
                cells.append("yes")
            elif (layer, row["op"]) in EXPECTED_GAPS:
                cells.append("--")  # documented, deliberate
            else:
                cells.append("MISSING")
        print(row["op"].ljust(width) + "".join(c.ljust(8) for c in cells))
    print()
    print(f"{report['api_size']} public hub functions")
    for layer in _LAYERS:
        rows = report["rows"]
        if report["layers"].get(layer) is None:
            print(f"  {layer:<7} not available (not importable)")
            continue
        yes = sum(1 for r in rows if r["layers"][layer])
        exp = sum(
            1
            for r in rows
            if not r["layers"][layer] and (layer, r["op"]) in EXPECTED_GAPS
        )
        bad = len(rows) - yes - exp
        print(
            f"  {layer:<7} {yes:>3} reached, {exp:>2} documented gaps, "
            f"{bad:>2} undocumented"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--lib-dir",
        default=os.environ.get("FASTFIELDS_DIR")
        or os.path.join(REPO, "_fastfields_lib"),
        help="fastfields-lib source tree (default: $FASTFIELDS_DIR, else "
        "the ./_fastfields_lib checkout of the pinned commit)",
    )
    p.add_argument(
        "--src-root",
        default=None,
        help="directory holding sibling source checkouts "
        "(fastfields-numpy/, fastfields-cupy/, ...); default is to read the "
        "installed packages",
    )
    p.add_argument("--format", choices=("table", "json"), default="table")
    p.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if any layer misses a hub op that is not listed "
        "in EXPECTED_GAPS",
    )
    args = p.parse_args(argv)

    report = build(args.lib_dir, args.src_root)
    if args.format == "json":
        report["gaps"] = [list(g) for g in gaps(report)]
        report["expected_gaps"] = {
            f"{k[0]}:{k[1]}": v for k, v in EXPECTED_GAPS.items()
        }
        print(json.dumps(report, indent=2))
    else:
        print_table(report)

    if args.check:
        bad = gaps(report)
        if bad:
            print("\nundocumented gaps:", file=sys.stderr)
            for layer, op in bad:
                print(f"  {layer}: {op}", file=sys.stderr)
            print(
                "\nEither expose the op in that layer, or add it to "
                "EXPECTED_GAPS in this script with the reason.",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
