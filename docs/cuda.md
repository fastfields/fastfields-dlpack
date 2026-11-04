# CUDA usage

`fastfields.dlpack` runs the same operations on CPU and CUDA arrays: it looks at
the device of the arrays you pass and dispatches to the matching backend. Two
things behave differently on CUDA, and both are contracts you have to keep --
this page is the statement of record for them.

One operation is not available on CUDA at all: `field_cg` raises `ValueError`
for CUDA arrays rather than quietly running on the host, because the hub has no
CUDA implementation of the solver's dot-product reductions yet
(fastfields-lib#34). Everything else in the module runs on either device.
`backend` tells you which build you have installed -- `"cpu"`, `"cu128"`, ...
or `None` if the version carries no label.

## Streams

Every binding takes a trailing `stream` argument: a CUDA stream handle as an
integer, `0` meaning the default/CPU stream. The value is forwarded to the CUDA
backend unchanged, so the kernels are queued on *your* stream and stay ordered
with respect to the rest of the work you put there.

Pass the stream your framework is currently using, not a fresh one:

```python
import cupy
import fastfields.dlpack as ff

stream = cupy.cuda.get_current_stream()
ff.pull(out, inp, grid, stream=stream.ptr)
```

The friendly wrappers do exactly this for you (`fastfields.cupy` forwards
`cupy.cuda.get_current_stream().ptr`, `fastfields.torch` the current stream of
the tensors' device).

## The lifetime invariant

!!! warning "Keep every array alive until the stream is synchronized"

    An array passed to a CUDA op -- input, output or temporary -- must stay
    alive on the Python side until the stream it was submitted on has been
    synchronized.

The calls are **asynchronous**: they enqueue work on `stream` and return before
it has run. Nothing in this package holds a reference to your arrays; the
DLPack tensors handed to C++ are plain views over memory the Python objects
own. When the last Python reference to an array goes away, its memory returns
to the framework's allocator -- CuPy's memory pool, PyTorch's caching allocator
-- which is free to hand that same block to the next allocation *while a
fastfields kernel is still reading or writing it*. The result is silent
corruption of unrelated data, not an error.

```python
# WRONG: `grid` is dropped while the kernel may still be reading it.
def sample(inp, coords, stream):
    grid = build_grid(coords)                 # allocates from the pool
    out = cupy.empty(...)
    ff.pull(out, inp, grid, stream=stream.ptr)
    return out                                # `grid` freed here, kernel still queued

# RIGHT: synchronize before letting the operands go...
def sample(inp, coords, stream):
    grid = build_grid(coords)
    out = cupy.empty(...)
    ff.pull(out, inp, grid, stream=stream.ptr)
    stream.synchronize()
    return out

# ...or keep them alive as long as the result is unsynchronized.
def sample(inp, coords, stream):
    grid = build_grid(coords)
    out = cupy.empty(...)
    ff.pull(out, inp, grid, stream=stream.ptr)
    return out, grid          # caller holds `grid` until it synchronizes
```

Note that the two frameworks do **not** give the same guarantees here. PyTorch's
caching allocator records the stream a block was last used on and will not
recycle it for another stream without synchronization; CuPy's memory pool tracks
no such thing. Code that appears to work under one framework can therefore
corrupt memory under the other, and the same code can start failing when the
allocation pattern around it changes. Do not rely on the allocator; hold the
reference.

Returning the arrays (or keeping them in a live object) is the usual fix, and it
is what the wrappers do: `fastfields.torch` additionally holds its operands past
the call through autograd's `save_for_backward`, and every wrapper keeps them
referenced for the duration of the op.

On CPU none of this applies: the calls are synchronous, so an array that is
alive at call time is alive for the whole operation.

## Device consistency

All arrays in a single call must live on the same device (same device type
*and* device id). A mixed call is rejected with a `ValueError` naming the
offending argument, rather than dispatched to one backend that would then read
the other operand's pointer:

```python
>>> ff.sym_matvec(out_gpu, hessian_cpu, inp_gpu)
ValueError: fastfields.sym_matvec(): argument 'hessian' is on
(device_type=1, device_id=0) but 'out' is on (device_type=2, device_id=0);
all arrays in a call must be on the same device
```
