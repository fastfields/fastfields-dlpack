"""fastfields.dlpack: nanobind DLPack bindings to the fastfields-lib C++ lib.

All functions accept any array object exposing ``__dlpack__`` (numpy, torch,
cupy, ...). Tensors marked in-place / as outputs are written through their
DLPack data pointers.

Argument checking
-----------------
This is the raw layer and it stays sharp: it does not reshape, cast, allocate
or copy anything for you. It does reject *structurally* malformed arrays --
a null data pointer, a 0-d array, a vector (multi-lane) dtype, arrays of a
single call spread across different devices -- with a ``ValueError`` naming the
function and the argument, instead of passing them on to the kernels. Shape and
dtype *contracts* (matching ranks, matching batch shapes, ``ndim`` vs. rank)
are enforced one layer down, by ``fastfields-lib``, and also surface as
``ValueError``.

Device coverage
---------------
Every op here runs on CPU and, when this package was built with CUDA
(``backend`` starting with ``"cu"``), on CUDA -- with one exception:
``field_cg`` is CPU-only, and raises ``ValueError`` for CUDA arrays rather
than falling back to the host (fastfields-lib#34). ``backend`` tells you which
build you have: ``"cpu"``, ``"cu128"``, ... or ``None`` if unknown.

CUDA lifetime invariant
-----------------------
**Every array passed to a CUDA op must be kept alive by the caller until the
stream it was submitted on has been synchronized.**

These bindings are asynchronous on CUDA: a call enqueues work on ``stream`` and
returns immediately, and nothing in this package holds a reference to your
arrays -- the DLPack tensors handed to C++ are plain views over memory the
Python objects own.

If an input, output or temporary is dropped before the stream is synchronized,
its memory goes back to the framework's allocator (CuPy's memory pool,
PyTorch's caching allocator) while a kernel may still be reading or writing
it. The allocator can then hand that same block to an unrelated allocation:
silent corruption, not a crash. The frameworks differ in *when* such reuse is
possible -- PyTorch's caching allocator records the stream a block was used on,
CuPy's pool does not -- so code that happens to work under one is not portable
to the other.

In practice:

.. code-block:: python

    out = cupy.empty_like(inp)
    ff.pull(out, inp, grid, stream=stream.ptr)
    stream.synchronize()        # only now may inp / grid / out be dropped
    del inp, grid               # (or let them go out of scope)

The friendly wrappers satisfy this for you, which is why it is easy to forget:
they submit on the framework's *own* current stream (``fastfields.torch`` via
``_util.stream_ptr``, ``fastfields.cupy`` via ``_util.current_stream_ptr``), so
the framework's ordering rules cover the memory they allocated, and they keep
the operands referenced for the whole call -- ``fastfields.torch`` beyond it,
through autograd's ``save_for_backward``. Calling ``fastfields.dlpack``
directly, or passing a stream of your own, means taking the invariant on
yourself. On CPU it holds trivially -- the calls are synchronous.
"""

from __future__ import annotations

import ctypes
import os
import re as _re
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
_lib_dir = os.path.join(_this_dir, "lib")

# Shared-library suffix used by the fastfields-lib Makefile, per platform.
if sys.platform == "darwin":
    _LIBEXT = "dylib"
elif sys.platform == "win32":
    _LIBEXT = "dll"
else:
    _LIBEXT = "so"

# cpu first, then the hub library that depends on it.
_NATIVE_LIBS = ("libfastfields-cpu." + _LIBEXT, "libfastfields." + _LIBEXT)


def _preload_native_libs() -> None:
    """Make the shipped shared libraries loadable by the ``_core`` extension.

    On ELF/Mach-O we preload cpu first, then the hub library, by absolute path
    with ``RTLD_LOCAL | RTLD_NOW``: this registers each under its soname so the
    loader reuses these copies when ``_core`` is imported, regardless of the
    RUNPATH baked into libfastfields, and resolving eagerly turns a broken
    build into an error here rather than a crash on the first call.

    ``RTLD_LOCAL`` (the default; spelled out for the record) is deliberate:
    ``_core`` links against ``libfastfields`` directly -- it needs no symbol to
    be visible in the *global* namespace -- so ``RTLD_GLOBAL`` would only serve
    to export every ``ff::`` symbol process-wide, where it can collide with
    another library's. Object *reuse* does not depend on the flag: the loader
    matches an already-loaded object by soname either way.

    On Windows there is no ``RTLD_*`` and no rpath, so we add the lib dir to
    the DLL search path (``_core`` and the hub DLL then resolve their
    dependencies from there) and also load the DLLs eagerly so a missing
    dependency surfaces here rather than as an opaque extension-import failure.
    """
    if sys.platform == "win32":
        if os.path.isdir(_lib_dir):
            os.add_dll_directory(_lib_dir)
        for name in _NATIVE_LIBS:
            path = os.path.join(_lib_dir, name)
            if os.path.exists(path):
                ctypes.CDLL(path)
        return
    mode = ctypes.RTLD_LOCAL | os.RTLD_NOW
    for name in _NATIVE_LIBS:
        path = os.path.join(_lib_dir, name)
        if os.path.exists(path):
            ctypes.CDLL(path, mode=mode)


_preload_native_libs()

from . import _core  # noqa: E402  (must follow the preload above)

# -- version + compute backend ---------------------------------------------
# ``_version.py`` is generated at build time by versioningit; a bare source
# tree that was never built has no such file.
try:
    from ._version import __version__
except ImportError:  # pragma: no cover - source tree without a build
    __version__ = "0+unknown"


_BACKEND_RE = _re.compile(r"\A(cpu|cu[0-9]+)\Z")


def _parse_backend(version: str) -> str | None:
    """Extract the compute-backend label from a PEP 440 version string.

    This distribution is the only backend-specific one in the stack (it ships
    the compiled ``libfastfields*``), and it records which build it is in the
    version's *local* segment, PyTorch-style: ``0.0.0.dev1+cpu``. A build made
    off-tag carries the git distance in the same segment
    (``0.0.0.dev1+cpu.3.gdeadbee``), so only the leading component is the
    backend.

    Parameters
    ----------
    version : str
        A PEP 440 version string.

    Returns
    -------
    str or None
        ``"cpu"``, ``"cu128"``, ... or ``None`` for a build with no backend
        label (e.g. an unversioned source checkout).
    """
    try:
        from packaging.version import InvalidVersion, parse
    except ImportError:  # pragma: no cover - packaging always installed
        return None
    try:
        local = parse(version).local
    except InvalidVersion:  # pragma: no cover
        return None
    if not local:
        return None
    head = local.split(".")[0]
    return head if _BACKEND_RE.match(head) else None


#: Compute backend this build of the native libraries was compiled for --
#: ``"cpu"``, ``"cu128"``, ... or ``None`` if unknown. Mirrors the role of
#: ``torch.version.cuda``: a friendly derived attribute so callers never have
#: to parse ``__version__`` themselves.
backend = _parse_backend(__version__)

# Re-export every binding under its pythonic (== ff::) name.
dt_euclidean = _core.dt_euclidean
dt_l1 = _core.dt_l1
dt_spline_table = _core.dt_spline_table
dt_spline_brent = _core.dt_spline_brent
dt_spline_gaussnewton = _core.dt_spline_gaussnewton
dt_mesh = _core.dt_mesh

sym_matvec = _core.sym_matvec
sym_matvec_backward = _core.sym_matvec_backward
sym_addmatvec_ = _core.sym_addmatvec_
sym_submatvec_ = _core.sym_submatvec_
sym_solve = _core.sym_solve
sym_solve_ = _core.sym_solve_
sym_invert = _core.sym_invert
sym_invert_ = _core.sym_invert_

resample = _core.resample
restriction = _core.restriction
spline_coeff = _core.spline_coeff

pull = _core.pull
push = _core.push
count = _core.count
grad = _core.grad
pull_backward = _core.pull_backward
push_backward = _core.push_backward
count_backward = _core.count_backward
grad_backward = _core.grad_backward

field_matvec = _core.field_matvec
field_diag = _core.field_diag
field_relax = _core.field_relax
field_kernel = _core.field_kernel
field_matvec_rls = _core.field_matvec_rls
field_diag_rls = _core.field_diag_rls
field_relax_rls = _core.field_relax_rls
flow_matvec = _core.flow_matvec
flow_diag = _core.flow_diag
flow_relax = _core.flow_relax
flow_kernel = _core.flow_kernel
flow_matvec_rls = _core.flow_matvec_rls
flow_diag_rls = _core.flow_diag_rls
flow_relax_rls = _core.flow_relax_rls

# The regularised system (H + L), where H is the per-voxel data-term Hessian:
# `*_forward` applies it, `*_precond[_]` applies the Jacobi approximation of
# its inverse, and `*_relax` / `field_cg` solve it. Each is one kernel -- a
# Python-side composition out of `*_matvec`/`*_diag` and `sym_matvec`/
# `sym_solve` computes the same values but allocates a full-size temporary for
# the regulariser result and traverses the field twice.
field_forward = _core.field_forward
field_precond = _core.field_precond
field_precond_ = _core.field_precond_
flow_forward = _core.flow_forward
flow_precond = _core.flow_precond
flow_precond_ = _core.flow_precond_

#: Conjugate-gradient solver for ``(H + L) x = g`` on a multi-channel field.
#:
#: **CPU only.** The hub has no CUDA implementation to forward to -- the
#: solver's dot products need a device-side reduction the CUDA backend does not
#: expose yet (fastfields-lib#34) -- so passing CUDA arrays raises
#: ``ValueError`` from C++ instead of silently running on the host. This is the
#: one op in this module that is not available on every device
#: ``fastfields.dlpack`` was built for; branch on ``backend`` (``"cpu"`` /
#: ``"cu128"`` / ``None``) if you need to pick a solver up front, or on the
#: array's device.
#:
#: It is also the one op that returns a value: ``(iterations_run,
#: final_relative_residual)``. Those are diagnostics about the call, not field
#: data, so they come back as a tuple rather than through caller-allocated
#: output arrays; ``sol`` is still written in place like every other output
#: here.
field_cg = _core.field_cg

# In-place-only accumulate primitives (jitfields op '+' / '-'). These
# read-modify-write the caller's `out`; the trailing underscore marks that, as
# for sym_addmatvec_ / sym_submatvec_. There is no out-of-place counterpart at
# this level -- the numpy/torch/cupy wrappers get it by cloning first.
field_addmatvec_ = _core.field_addmatvec_
field_submatvec_ = _core.field_submatvec_
field_adddiag_ = _core.field_adddiag_
field_subdiag_ = _core.field_subdiag_
field_addkernel_ = _core.field_addkernel_
field_subkernel_ = _core.field_subkernel_
flow_addmatvec_ = _core.flow_addmatvec_
flow_submatvec_ = _core.flow_submatvec_
flow_adddiag_ = _core.flow_adddiag_
flow_subdiag_ = _core.flow_subdiag_
flow_addkernel_ = _core.flow_addkernel_
flow_subkernel_ = _core.flow_subkernel_

__all__ = [
    "__version__",
    "backend",
    "dt_euclidean",
    "dt_l1",
    "dt_spline_table",
    "dt_spline_brent",
    "dt_spline_gaussnewton",
    "dt_mesh",
    "sym_matvec",
    "sym_matvec_backward",
    "sym_addmatvec_",
    "sym_submatvec_",
    "sym_solve",
    "sym_solve_",
    "sym_invert",
    "sym_invert_",
    "resample",
    "restriction",
    "spline_coeff",
    # pushpull
    "pull",
    "push",
    "count",
    "grad",
    "pull_backward",
    "push_backward",
    "count_backward",
    "grad_backward",
    # regularisers -- field
    "field_matvec",
    "field_diag",
    "field_relax",
    "field_kernel",
    "field_matvec_rls",
    "field_diag_rls",
    "field_relax_rls",
    "field_addmatvec_",
    "field_submatvec_",
    "field_adddiag_",
    "field_subdiag_",
    "field_addkernel_",
    "field_subkernel_",
    # regularisers -- flow
    "flow_matvec",
    "flow_diag",
    "flow_relax",
    "flow_kernel",
    "flow_matvec_rls",
    "flow_diag_rls",
    "flow_relax_rls",
    "flow_addmatvec_",
    "flow_submatvec_",
    "flow_adddiag_",
    "flow_subdiag_",
    "flow_addkernel_",
    "flow_subkernel_",
    # regularised system (H + L): forward / preconditioner / solver
    "field_forward",
    "field_precond",
    "field_precond_",
    "flow_forward",
    "flow_precond",
    "flow_precond_",
    "field_cg",
]
