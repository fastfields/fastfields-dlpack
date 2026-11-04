"""The regularised-system bindings: ``*_forward``/``*_precond``/``field_cg``.

These bind the hub's ``(H + L)`` family, where ``H`` is the per-voxel
compact-symmetric data-term Hessian and ``L`` the field/flow regulariser:
``*_forward`` applies the operator, ``*_precond[_]`` applies the Jacobi
approximation of its inverse, and ``field_cg`` solves the system.

Each is checked against the composition a caller would otherwise have to write
out of the already-bound primitives -- ``sym_matvec`` + ``*_matvec`` for the
forward, ``sym_solve`` + ``*_diag`` for the preconditioner. The composition is
the independent reference: it is a different code path through the library
(two traversals and a temporary instead of one fused kernel), so agreement
to the bit is real evidence that these entry points compute what their
documentation says. It is also exactly the composition the numpy/cupy wrappers
used to carry in Python, which is why these ops stayed unbound unnoticed.
"""

import numpy as np
import pytest

import fastfields.dlpack as ff

NDIM = 2
SPATIAL = (5, 6)


def _field_operands(channels=2, seed=0):
    rng = np.random.default_rng(seed)
    npack = channels * (channels + 1) // 2
    vec = rng.standard_normal(SPATIAL + (channels,))
    hes = np.zeros(SPATIAL + (npack,))
    # Diagonally dominant -> positive definite, so the solves are well posed.
    for c in range(channels):
        hes[..., c] = 3.0 + c + rng.random(SPATIAL)
    hes[..., channels:] = 0.1 * rng.standard_normal(
        SPATIAL + (npack - channels,)
    )
    pen = {
        "voxel_size": [1.0, 1.2],
        "absolute": [0.1] * channels,
        "membrane": [0.5] * channels,
        "bending": [0.05] * channels,
        "bound": 3,
        "ndim": NDIM,
    }
    return hes, vec, pen


def _flow_operands(seed=0):
    rng = np.random.default_rng(seed)
    npack = NDIM * (NDIM + 1) // 2
    vec = rng.standard_normal(SPATIAL + (NDIM,))
    hes = np.zeros(SPATIAL + (npack,))
    for c in range(NDIM):
        hes[..., c] = 3.0 + c + rng.random(SPATIAL)
    hes[..., NDIM:] = 0.1 * rng.standard_normal(SPATIAL + (npack - NDIM,))
    pen = {
        "voxel_size": [1.0, 1.2],
        "absolute": 0.1,
        "membrane": 0.5,
        "bending": 0.0,
        "shears": 0.2,
        "div": 0.1,
        "bound": 3,
        "ndim": NDIM,
    }
    return hes, vec, pen


# --------------------------------------------------------------------------- #
# forward: out = (H + L) @ inp                                                #
# --------------------------------------------------------------------------- #


def test_field_forward_matches_sym_matvec_plus_field_matvec():
    hes, vec, pen = _field_operands()
    out = np.empty_like(vec)
    ff.field_forward(out, hes, vec, **pen)

    ref = np.empty_like(vec)
    ff.sym_matvec(ref, hes, vec)
    reg = np.zeros_like(vec)
    ff.field_matvec(reg, vec, **pen)
    np.testing.assert_allclose(out, ref + reg, rtol=1e-12, atol=1e-12)


def test_flow_forward_matches_sym_matvec_plus_flow_matvec():
    hes, vec, pen = _flow_operands()
    out = np.empty_like(vec)
    ff.flow_forward(out, hes, vec, **pen)

    ref = np.empty_like(vec)
    ff.sym_matvec(ref, hes, vec)
    reg = np.zeros_like(vec)
    ff.flow_matvec(reg, vec, **pen)
    np.testing.assert_allclose(out, ref + reg, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# preconditioner: out = (H + diag(L)) \ grd                                   #
# --------------------------------------------------------------------------- #


def test_field_precond_matches_sym_solve_with_field_diag():
    hes, grd, pen = _field_operands()
    out = np.empty_like(grd)
    ff.field_precond(out, hes, grd, **pen)

    diag = np.zeros_like(grd)
    ff.field_diag(diag, **pen)
    ref = np.empty_like(grd)
    ff.sym_solve(ref, hes, grd, diag)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_flow_precond_matches_sym_solve_with_flow_diag():
    hes, grd, pen = _flow_operands()
    out = np.empty_like(grd)
    ff.flow_precond(out, hes, grd, **pen)

    diag = np.zeros_like(grd)
    ff.flow_diag(diag, **pen)
    ref = np.empty_like(grd)
    ff.sym_solve(ref, hes, grd, diag)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("family", ["field", "flow"])
def test_precond_inplace_agrees_with_out_of_place(family):
    if family == "field":
        hes, grd, pen = _field_operands()
        out_of_place, in_place = ff.field_precond, ff.field_precond_
    else:
        hes, grd, pen = _flow_operands()
        out_of_place, in_place = ff.flow_precond, ff.flow_precond_

    expected = np.empty_like(grd)
    out_of_place(expected, hes, grd, **pen)

    sol = grd.copy()
    in_place(sol, hes, **pen)
    np.testing.assert_allclose(sol, expected, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# field_cg                                                                    #
# --------------------------------------------------------------------------- #


def test_field_cg_solves_the_system_field_forward_applies():
    hes, grd, pen = _field_operands(seed=3)
    sol = np.zeros_like(grd)  # cold start
    nb_iter, residual = ff.field_cg(
        sol, hes, grd, nb_iter=200, tol=1e-12, **pen
    )

    assert 0 < nb_iter <= 200
    assert residual < 1e-10

    # The reported residual must be the real one: re-apply the operator.
    applied = np.empty_like(grd)
    ff.field_forward(applied, hes, sol, **pen)
    rel = np.linalg.norm(applied - grd) / np.linalg.norm(grd)
    assert rel < 1e-9
    np.testing.assert_allclose(residual, rel, rtol=1e-3, atol=1e-12)


def test_field_cg_warm_start_is_honoured():
    """`sol` is the warm start, not scratch: starting at the answer is free."""
    hes, grd, pen = _field_operands(seed=4)
    sol = np.zeros_like(grd)
    ff.field_cg(sol, hes, grd, nb_iter=200, tol=1e-12, **pen)

    warm = sol.copy()
    nb_iter, residual = ff.field_cg(
        warm, hes, grd, nb_iter=200, tol=1e-8, **pen
    )
    assert nb_iter <= 2, (
        "starting from the solution should converge immediately, "
        f"took {nb_iter} iterations"
    )
    np.testing.assert_allclose(warm, sol, rtol=1e-6, atol=1e-8)


def test_field_cg_respects_nb_iter_cap():
    hes, grd, pen = _field_operands(seed=5)
    sol = np.zeros_like(grd)
    nb_iter, _ = ff.field_cg(sol, hes, grd, nb_iter=3, tol=0.0, **pen)
    assert nb_iter <= 3


def test_field_cg_beats_the_same_number_of_relaxation_sweeps():
    """Sanity: CG is the Krylov alternative to field_relax on one system."""
    hes, grd, pen = _field_operands(seed=6)

    cg = np.zeros_like(grd)
    ff.field_cg(cg, hes, grd, nb_iter=8, tol=0.0, **pen)

    gs = np.zeros_like(grd)
    ff.field_relax(gs, hes, grd, nb_iter=8, **pen)

    def rel(x):
        applied = np.empty_like(grd)
        ff.field_forward(applied, hes, x, **pen)
        return np.linalg.norm(applied - grd) / np.linalg.norm(grd)

    assert rel(cg) <= rel(gs)


# --------------------------------------------------------------------------- #
# argument checking still applies to the new entries                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "call",
    [
        lambda a: ff.field_forward(a, a, a, ndim=1),
        lambda a: ff.field_precond(a, a, a, ndim=1),
        lambda a: ff.field_precond_(a, a, ndim=1),
        lambda a: ff.flow_forward(a, a, a, ndim=1),
        lambda a: ff.flow_precond(a, a, a, ndim=1),
        lambda a: ff.flow_precond_(a, a, ndim=1),
        lambda a: ff.field_cg(a, a, a, ndim=1),
    ],
)
def test_zero_d_argument_is_rejected(call):
    # The raw layer indexes the trailing axis before any of the library's own
    # checks run; a 0-d array must be refused, not read off a null pointer.
    with pytest.raises(ValueError):
        call(np.array(1.0))
