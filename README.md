# fastfields-dlpack

nanobind bindings from DLPack to the
[`fastfields-lib`](https://github.com/fastfields/fastfields-lib) C++
library. Imports as **`fastfields.dlpack`** (a PEP 420 namespace subpackage; this
distribution ships only `fastfields/dlpack/` and never a `fastfields/__init__.py`,
so it merges with `fastfields-numpy`/`-torch`/`-cupy`/`fastfields`).

All functions accept any array exposing `__dlpack__` (numpy, torch, cupy).
Arguments documented as in-place / outputs are written through their DLPack
data pointers.

```python
import numpy as np
import fastfields.dlpack as ff

x = np.array([0, np.inf, np.inf, 0, np.inf], dtype=np.float32)
ff.dt_euclidean(x)          # in-place squared Euclidean distance transform
```

On CUDA arrays the calls are asynchronous: they enqueue work on the `stream`
you pass and return. **Every array must be kept alive by the caller until that
stream is synchronized** — see [`docs/cuda.md`](./docs/cuda.md).

Structurally malformed arrays (0-d, null data, vector dtypes, or a single call
mixing devices) are rejected with a `ValueError` naming the function and the
argument; shape/dtype contracts are checked by `fastfields-lib` one layer down.

## Build

```bash
pip install .
```

(Editable installs can break the native-namespace merge across the sibling
`fastfields-*` distributions; prefer a regular `pip install .`.)

The build needs the fastfields-lib C++ source tree, because the extension
compiles against its headers. `fastfields-lib.sha` records the exact commit it
is pinned to (a full SHA, never a tag -- a tag can be moved); `setup.py` fetches
that commit into `./_fastfields_lib` if it is not already there. **This is the
one step that needs the network.** It replaced a git submodule, which is what
makes `pip install` from an sdist work at all: git does not carry submodule
contents in a source distribution.

To build against your own fastfields-lib working tree instead, point
`FASTFIELDS_DIR` at it:

```bash
FASTFIELDS_DIR=/path/to/fastfields-lib pip install .
```

Precedence is `$FASTFIELDS_DIR` (or `$FASTFIELDS_INCLUDE`), then an existing
`./_fastfields_lib`, then the fetch. The first two are used exactly as they are
-- nothing is fetched into them, reset, or overwritten; a checkout at a commit
other than the pinned one is reported as a warning and then used. Offline with
no local tree, the build fails with a message naming the SHA it wanted, where
from, and `FASTFIELDS_DIR`.

`setup.py` then builds the shared libraries via `make` if they are missing,
ships them inside the wheel under `fastfields/dlpack/lib/`, and compiles the
nanobind extension (`fastfields.dlpack._core`) against them with an
`$ORIGIN/lib` rpath.

Integer `spline` arguments run 0=Nearest .. 7, and `bound` arguments run
0=Zero,1=Replicate,2=DCT1,3=DCT2,4=DST1,5=DST2,6=DFT,7=NoCheck. The `Spline`
and `Bound` enums that document these, plus the argument-normalisation
helpers built on them, live in the separate, dependency-free
[`fastfields-helpers`](https://github.com/fastfields/fastfields-helpers)
package rather than here -- this package (`fastfields.dlpack`) takes plain
integers and does not depend on it.
