"""The binding must cover the whole hub API.

``fastfields.dlpack`` is the only way into ``fastfields-lib`` from Python:
every friendly wrapper (numpy / cupy / torch) and ``fastfields.auto`` sits on
top of it, so an op that is not bound here is unreachable from Python no matter
what the wrappers do. Nothing in the build notices: the hub declares a
function, ``libfastfields.so`` exports it, and the extension simply never
mentions it.

That is how the solver/preconditioner family -- ``field_forward``,
``{field,flow}_precond[_]``, ``flow_forward``, ``field_cg`` -- stayed unbound
while two wrappers advertised ``field_forward``/``field_precond`` and quietly
recomputed them in Python out of ``*_matvec`` + ``sym_solve``.

So this test re-derives the hub's public surface from the headers of the
pinned ``_fastfields_lib`` submodule -- the same parse
``tools/api_inventory.py`` does, imported from it so there is one
implementation -- and asserts the binding exports all of it. A new hub entry
point therefore turns red here rather than being discovered later.

Skipped when the submodule is not checked out (a plain sdist install has the
compiled extension but no C++ sources).
"""

from __future__ import annotations

import os
import sys

import pytest

import fastfields.dlpack as ff

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB = os.path.join(_REPO, "_fastfields_lib")

sys.path.insert(0, os.path.join(_REPO, "tools"))


def _hub_api():
    if not os.path.isdir(os.path.join(_LIB, "include")):
        pytest.skip(f"no fastfields-lib source tree at {_LIB}")
    try:
        import api_inventory
    except ImportError:  # pragma: no cover - tools/ missing from an sdist
        pytest.skip("tools/api_inventory.py not available")
    return api_inventory.hub_api(_LIB)


def test_every_hub_function_is_bound():
    api = _hub_api()
    assert api, "parsed no functions out of the hub headers -- parser broken?"
    missing = sorted(op for op in api if not hasattr(ff, op))
    assert not missing, (
        "hub functions with no binding in fastfields.dlpack: "
        f"{missing}. Bind them in src/ext.cpp and re-export them from "
        "fastfields/dlpack/__init__.py, or -- if one is deliberately not "
        "exposed -- record it in EXPECTED_GAPS in tools/api_inventory.py "
        "with the reason."
    )


def test_every_hub_function_is_in_dunder_all():
    # hasattr() is not enough: the wrappers and `from ... import *` go through
    # __all__, and a binding missing from it is invisible to tooling.
    api = _hub_api()
    missing = sorted(op for op in api if op not in ff.__all__)
    assert not missing, (
        f"bound but absent from fastfields.dlpack.__all__: {missing}"
    )


def test_dunder_all_has_no_dangling_names():
    missing = sorted(n for n in ff.__all__ if not hasattr(ff, n))
    assert not missing, f"__all__ names that do not exist: {missing}"


def test_inventory_check_passes():
    """``api_inventory.py --check`` must be clean for the binding layer.

    The wrapper layers are only checked when they happen to be installed
    alongside; this asserts the binding column, which is this repo's business.
    """
    api = _hub_api()
    import api_inventory

    hits = api_inventory.reachable(list(ff.__all__), api)
    unreached = sorted(op for op, names in hits.items() if not names)
    assert not unreached, f"hub ops no exported name reaches: {unreached}"
