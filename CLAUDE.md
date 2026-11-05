# fastfields-dlpack  (imports as `fastfields.dlpack`)

**nanobind** bindings from DLPack to the `fastfields-lib` C++/CUDA library. This
is the base of the Python stack: the raw, in-place bindings that every friendly
wrapper (numpy/cupy/torch/auto) sits on.

```
… ─ lib ─ dlpack ← (you are here) ─ {numpy,cupy,torch} ─ fastfields
fastfields-helpers (independent sibling, no dependency either way)
```

- The `fastfields-lib` C++ tree is **pinned by commit SHA in
  `fastfields-lib.sha`**, which `setup.py` fetches into `./_fastfields_lib` at
  build time. It used to be a git submodule; it is not any more, because git
  does not put submodule contents in an sdist, so `pip install` from a source
  distribution could never build. A full 40-char SHA only -- a tag can be
  moved. `FASTFIELDS_DIR` (or a pre-existing `./_fastfields_lib`) overrides the
  fetch and is used untouched, so a C++ working tree is never reset by a
  build.
- Distribution name `fastfields-dlpack`; imports as **`fastfields.dlpack`**.

## Philosophy / role
- Thin, close-to-the-metal bindings. Functions accept **any `__dlpack__`
  object** (numpy / torch / cupy) and operate **in place** / write through
  pre-allocated output arrays via the DLPack data pointer.
- Exposes the same operation families as `fastfields-lib`: distance transforms &
  point-to-spline/mesh distance, posdef (compact-symmetric) linear algebra,
  resampling (`resample`/`restriction`/`spline_coeff`), pushpull, regularisers.
- Integer `spline` arguments run 0=Nearest … 7, and `bound` arguments run
  0=Zero,1=Replicate,2=DCT1,3=DCT2,4=DST1,5=DST2,6=DFT,7=NoCheck. The `Spline`
  / `Bound` enums that document these values, and the pure-Python argument
  normalisers built on them, **used to live here** (`fastfields/dlpack/
  _helpers.py`) but were extracted to the separate, dependency-free
  `fastfields-helpers` package so a one-line Python fix ships without a
  compiled-wheel release cycle. **This package takes plain integers and has
  no dependency on `fastfields-helpers`** — they are independent siblings,
  not a layering. Do not re-add that import here.

## Layout
- `src/ext.cpp` — the nanobind extension source (builds as
  `fastfields.dlpack._core`).
- `fastfields/dlpack/__init__.py` — Python surface; `lib/*.so` holds the shipped
  shared libraries.
- `setup.py` — custom `build_ext` (no CMake). `tests/test_bindings.py`.

## Build & test
```
pip install .                     # regular install (NOT editable — see caveat)
python -m pytest tests/ -q        # import from a neutral cwd
```
`setup.py` fetches the pinned C++ tree if `./_fastfields_lib` is absent, then
builds `libfastfields.so` + `libfastfields-cpu.so` via its Makefile **if
missing** (bare `make`, relying on that Makefile's `.DEFAULT_GOAL := all`),
copies them into
`fastfields/dlpack/lib/`, and compiles the nanobind extension against them with
an `$ORIGIN/lib` rpath. It respects `CXX` (default `clang++`); the extension is
built `-std=c++17` even though the C++ libs are C++11.

## Conventions & caveats
- **PEP 420 namespace package**: this distribution ships only
  `fastfields/dlpack/` and **never a `fastfields/__init__.py`**, so it merges
  with `fastfields-numpy`/`-cupy`/`-torch`/`fastfields` into one `fastfields`
  namespace. Do not add a top-level `__init__.py`.
- **Editable installs break the native-namespace merge** across the sibling
  distributions — prefer a regular `pip install .`. Import from a neutral cwd
  (not the repo root) so the installed package, not the source dir, is found.
- Ruff config in `pyproject.toml`: line-length 79, select B/E/F/I/W.

## Pointers
- Hierarchy: `/home/user/.github/profile/README.md`.
- Underlying C++ API + status: `/home/user/fastfields-lib/MIGRATION.md`.
