# fastfields-dlpack

**fastfields-dlpack** is the low-level core of the fastfields project. It imports
as `fastfields.dlpack` and exposes the field operators as raw, in-place calls
that work directly on any array — NumPy, PyTorch or CuPy — sharing memory with
zero copies.

Most people don't use this package directly. The friendly, array-returning
wrappers — [`fastfields.numpy`](https://fastfields.github.io/fastfields-numpy/),
[`fastfields.torch`](https://fastfields.github.io/fastfields-torch/),
[`fastfields.cupy`](https://fastfields.github.io/fastfields-cupy/) and the
unified [`fastfields.auto`](https://fastfields.github.io/fastfields/) — are built
on top of it and are what you normally want. Reach for `fastfields.dlpack` when
you want the thinnest possible layer and are happy to manage output buffers
yourself.

## Install

```sh
pip install fastfields-dlpack \
    --extra-index-url https://fastfields.github.io/whl/cpu/
```

## Use it

Functions write results in place or into a pre-allocated output you pass in:

```python
import numpy as np
import fastfields.dlpack as ff

x = np.array([0, np.inf, np.inf, 0, np.inf], dtype=np.float32)
ff.dt_euclidean(x)          # in-place squared Euclidean distance transform
```

## What's inside

The same operation families as the higher-level packages, in their raw in-place
form: distance transforms (`dt_euclidean`, `dt_l1`, the `dt_spline_*` and
`dt_mesh` point distances), positive-definite linear algebra (`sym_matvec`,
`sym_addmatvec_`, `sym_solve`, `sym_invert`, …), resampling (`resample`,
`restriction`, `spline_coeff`), spline gather/scatter (`pull`, `push`, `count`,
`grad` and their adjoints) and the spatial regularisers (`field_*` for a
multi-channel field, `flow_*` for a vector flow). The `Spline` and `Bound`
enums document the integer order/boundary arguments.

Every public entry point of the C++ hub is bound — all 58 of them, checked by
`tests/test_api_coverage.py`. `tools/api_inventory.py` prints the whole
hub → binding → wrapper matrix if you want to see which layer exposes what:

```sh
python tools/api_inventory.py              # table
python tools/api_inventory.py --check      # non-zero on an undocumented gap
```

### Solving the regularised system

The regulariser families each come with the operators an outer optimisation
loop needs on `(H + L)`, where `H` is the per-voxel data-term Hessian in
compact-symmetric packing and `L` the regulariser:

| call | what it does |
| --- | --- |
| `field_forward` / `flow_forward` | apply `(H + L) x` |
| `field_precond` / `flow_precond` | apply `(H + diag(L))⁻¹ g` (Jacobi) |
| `field_precond_` / `flow_precond_` | the same, in place |
| `field_relax` / `flow_relax` | Gauss-Seidel sweeps on `(H + L) x = g` |
| `field_cg` | preconditioned conjugate gradients on the same system |

`field_cg` is the only operation here that is **CPU-only** (the CUDA backend
has no device-side reduction for the solver's dot products yet) and the only
one that returns a value: `(iterations_run, final_relative_residual)`.

See the [API reference](api/index.md) for full signatures and options.

## On CUDA

The same calls work on CuPy and PyTorch CUDA arrays and take the stream to
submit on. They are asynchronous, so the caller must keep every array alive
until that stream is synchronized — see [CUDA usage](cuda.md) for the full
contract.
