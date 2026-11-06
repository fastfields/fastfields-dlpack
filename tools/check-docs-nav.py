#!/usr/bin/env python3
"""Check a zensical.toml's top-level nav against the canonical fastfields menu.

The fastfields documentation is one site assembled from four repositories, each
building its own section with its own toolchain (see DOCS-THEME.md in the
fastfields/.github repository). zensical has no include mechanism, so the menu
is duplicated in four `zensical.toml` files -- and it drifted: at one point no
two of them agreed on the top-level entries, and the landing page carried no
per-namespace entries at all.

This script is the guard. It is identical in every publisher repository and
each one runs it against its own config, because a check that lives in only
one repo cannot see another repo's drift.

What it enforces:

  1. `nav`'s top-level labels are exactly CANONICAL_NAV, in that order. A site
     may expand any entry into a nested list of its own pages -- that is what
     makes the menu useful locally -- but it may not add, remove, reorder or
     relabel a top-level entry.
  2. `site_name` is "fastfields". Every section is the same product; the page's
     own `# heading` is what says which part of it you are reading.

What it deliberately does not enforce: anything about the nested children, the
theme block, or whether a target is relative or absolute. Those are prose
conventions in DOCS-THEME.md, and the nested lists are the one part of the menu
a section is meant to own.

Usage:  check-docs-nav.py --check [zensical.toml ...]
        check-docs-nav.py --selftest
"""

from __future__ import annotations

import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    sys.exit(
        "check-docs-nav.py: needs Python 3.11+ for tomllib "
        f"(running {sys.version.split()[0]})."
    )

# The canonical top-level menu: one entry per fastfields namespace, plus Home
# and the wheel index. The namespace order matches the Packages table on the
# landing page -- keep the two consistent.
CANONICAL_NAV = [
    "Home",
    "fastfields.auto",
    "fastfields.numpy",
    "fastfields.torch",
    "fastfields.cupy",
    "fastfields.helpers",
    "fastfields.dlpack",
    "libfastfields",
    "Wheel index",
]

CANONICAL_SITE_NAME = "fastfields"


def nav_labels(nav: object, problems: list[str]) -> list[str]:
    """Extract the ordered top-level labels from a parsed `nav` value."""
    if not isinstance(nav, list):
        problems.append(f"`nav` must be an array of tables, found {type(nav).__name__}.")
        return []

    labels: list[str] = []
    for i, entry in enumerate(nav):
        if isinstance(entry, dict):
            if len(entry) != 1:
                problems.append(
                    f"`nav` entry {i} has {len(entry)} keys "
                    f"({', '.join(map(repr, entry))}); each entry needs exactly one label."
                )
            labels.extend(entry)
        elif isinstance(entry, str):
            # A bare path carries no label, so the menu would render a title
            # derived from the page -- unreviewable and not one of the nine.
            problems.append(
                f"`nav` entry {i} is the bare path {entry!r}; "
                "every top-level entry must be a labelled table."
            )
        else:
            problems.append(
                f"`nav` entry {i} is a {type(entry).__name__}; expected a labelled table."
            )
    return labels


def check_config(path: str) -> list[str]:
    """Return a list of problems with one zensical.toml (empty means it passes)."""
    problems: list[str] = []
    try:
        with open(path, "rb") as fh:
            config = tomllib.load(fh)
    except FileNotFoundError:
        return [f"{path}: no such file."]
    except tomllib.TOMLDecodeError as exc:
        return [f"{path}: is not valid TOML: {exc}"]

    project = config.get("project")
    if not isinstance(project, dict):
        return [f"{path}: has no [project] table."]

    if "nav" not in project:
        problems.append("has no `nav`; the canonical menu is not optional.")
    else:
        labels = nav_labels(project["nav"], problems)
        if labels != CANONICAL_NAV:
            problems.append(
                "top-level `nav` labels do not match the canonical menu.\n"
                + _diff(CANONICAL_NAV, labels)
            )

    site_name = project.get("site_name")
    if site_name != CANONICAL_SITE_NAME:
        problems.append(
            f"site_name is {site_name!r}, expected {CANONICAL_SITE_NAME!r} "
            "-- every section documents the same product."
        )

    return [f"{path}: {p}" for p in problems]


def _diff(want: list[str], got: list[str]) -> str:
    """A positional want/got table, which is what makes a failure actionable."""
    lines = [f"    {'':>3}  {'expected':<24}  found"]
    for i in range(max(len(want), len(got))):
        w = want[i] if i < len(want) else "-"
        g = got[i] if i < len(got) else "-"
        lines.append(f"    {i + 1:>3}  {w:<24}  {g}{'' if w == g else '   <-- differs'}")
    return "\n".join(lines)


SELFTEST_CASES = [
    # (name, toml, expected-to-pass)
    (
        "canonical",
        """
        [project]
        site_name = "fastfields"
        nav = [
          {"Home" = "index.md"},
          {"fastfields.auto" = ["auto.md", "api/auto.md"]},
          {"fastfields.numpy" = "https://fastfields.github.io/python/numpy/"},
          {"fastfields.torch" = "https://fastfields.github.io/python/torch/"},
          {"fastfields.cupy" = "https://fastfields.github.io/python/cupy/"},
          {"fastfields.helpers" = "https://fastfields.github.io/python/helpers/"},
          {"fastfields.dlpack" = "https://fastfields.github.io/dlpack/"},
          {"libfastfields" = "https://fastfields.github.io/lib/"},
          {"Wheel index" = "https://fastfields.github.io/whl/"},
        ]
        """,
        True,
    ),
    (
        "relabelled entry",
        """
        [project]
        site_name = "fastfields"
        nav = [
          {"Home" = "index.md"},
          {"fastfields.auto" = "a.md"},
          {"fastfields.numpy" = "b.md"},
          {"fastfields.torch" = "c.md"},
          {"fastfields.cupy" = "d.md"},
          {"fastfields.helpers" = "e.md"},
          {"Raw DLPack binding" = "f.md"},
          {"libfastfields" = "g.md"},
          {"Wheel index" = "h.md"},
        ]
        """,
        False,
    ),
    (
        "extra top-level entry",
        """
        [project]
        site_name = "fastfields"
        nav = [
          {"Home" = "index.md"},
          {"fastfields.auto" = "a.md"},
          {"fastfields.numpy" = "b.md"},
          {"fastfields.torch" = "c.md"},
          {"fastfields.cupy" = "d.md"},
          {"fastfields.helpers" = "e.md"},
          {"fastfields.dlpack" = "f.md"},
          {"libfastfields" = "g.md"},
          {"Packaging" = "monorepo.md"},
          {"Wheel index" = "h.md"},
        ]
        """,
        False,
    ),
    (
        "reordered namespaces",
        """
        [project]
        site_name = "fastfields"
        nav = [
          {"Home" = "index.md"},
          {"fastfields.numpy" = "b.md"},
          {"fastfields.auto" = "a.md"},
          {"fastfields.torch" = "c.md"},
          {"fastfields.cupy" = "d.md"},
          {"fastfields.helpers" = "e.md"},
          {"fastfields.dlpack" = "f.md"},
          {"libfastfields" = "g.md"},
          {"Wheel index" = "h.md"},
        ]
        """,
        False,
    ),
    (
        "wrong site_name",
        """
        [project]
        site_name = "fastfields-lib"
        nav = [
          {"Home" = "index.md"},
          {"fastfields.auto" = "a.md"},
          {"fastfields.numpy" = "b.md"},
          {"fastfields.torch" = "c.md"},
          {"fastfields.cupy" = "d.md"},
          {"fastfields.helpers" = "e.md"},
          {"fastfields.dlpack" = "f.md"},
          {"libfastfields" = "g.md"},
          {"Wheel index" = "h.md"},
        ]
        """,
        False,
    ),
    (
        "missing nav",
        """
        [project]
        site_name = "fastfields"
        """,
        False,
    ),
    (
        "bare path at top level",
        """
        [project]
        site_name = "fastfields"
        nav = [
          "index.md",
          {"fastfields.auto" = "a.md"},
          {"fastfields.numpy" = "b.md"},
          {"fastfields.torch" = "c.md"},
          {"fastfields.cupy" = "d.md"},
          {"fastfields.helpers" = "e.md"},
          {"fastfields.dlpack" = "f.md"},
          {"libfastfields" = "g.md"},
          {"Wheel index" = "h.md"},
        ]
        """,
        False,
    ),
]


def selftest() -> int:
    """Check the checker: every case below must land on the expected verdict."""
    import os
    import tempfile
    import textwrap

    failures = 0
    for name, toml, should_pass in SELFTEST_CASES:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "zensical.toml")
            with open(path, "w") as fh:
                fh.write(textwrap.dedent(toml))
            problems = check_config(path)
        passed = not problems
        if passed != should_pass:
            failures += 1
            verdict = "passed" if passed else "failed"
            want = "pass" if should_pass else "fail"
            print(f"  selftest '{name}': {verdict}, expected it to {want}", file=sys.stderr)
        else:
            print(f"  selftest '{name}': ok ({'passes' if passed else 'rejected'})")

    if failures:
        print(f"check-docs-nav.py: {failures} selftest case(s) wrong.", file=sys.stderr)
        return 1
    print(f"check-docs-nav.py: all {len(SELFTEST_CASES)} selftest cases behave.")
    return 0


def main(argv: list[str]) -> int:
    args = [a for a in argv if a != "--check"]
    if "--selftest" in args:
        if len(args) != 1:
            print("check-docs-nav.py: --selftest takes no other arguments.", file=sys.stderr)
            return 2
        return selftest()

    paths = args or ["zensical.toml"]
    problems: list[str] = []
    for path in paths:
        problems.extend(check_config(path))

    if problems:
        print("check-docs-nav.py: the documentation menu has drifted.\n", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        print(
            "\n  The canonical menu is one entry per fastfields namespace, defined in\n"
            "  CANONICAL_NAV in this script and documented in DOCS-THEME.md in the\n"
            "  fastfields/.github repository. All four sections must agree, so a change\n"
            "  to the menu is a change to every publisher repo.",
            file=sys.stderr,
        )
        return 1

    for path in paths:
        print(f"check-docs-nav.py: {path} matches the canonical menu.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
