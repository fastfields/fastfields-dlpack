// nanobind bindings for fastfields-lib.
//
// We accept untyped `nb::ndarray<>` arguments: nanobind auto-imports any
// object exposing `__dlpack__` (numpy / torch / cupy / ...). Each ndarray is
// converted to a DLTensor by the per-call `Args` checker below and handed to
// the C++ library, which operates in place / writes outputs through the
// DLTensor pointers.
//
// ---------------------------------------------------------------------------
// Lifetime contract (matters on CUDA)
// ---------------------------------------------------------------------------
// Every binding is *asynchronous* with respect to the caller when it runs on a
// CUDA device: it enqueues work on `stream` and returns without
// synchronizing. Nothing here keeps a reference to the caller's arrays -- the
// DLTensors are stack views over memory the Python objects own, and nanobind
// drops its references as soon as the call returns.
//
// The caller therefore MUST keep every array passed to a CUDA op alive until
// that stream has been synchronized. Letting an input, output or temporary go
// out of scope earlier returns its memory to the framework's allocator (cupy's
// memory pool, torch's caching allocator), which may hand the same block to an
// unrelated allocation while the kernel is still reading or writing it --
// silent corruption, not a crash. The two allocators differ in exactly *when*
// that reuse becomes possible (torch records stream ownership, cupy's pool does
// not), so "it happened to work with framework X" is not portable.
//
// On CPU the same rule holds trivially: the calls are synchronous, so an array
// that is alive at call time is alive for the whole operation.
//
// See `fastfields/dlpack/__init__.py` and `docs/cuda.md` for the user-facing
// statement of this contract.
//
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

// The public headers share the FF_LIB_BOUND_SPLINE_T guard, so they
// co-include cleanly and provide the correctly-namespaced ff:: declarations.
#include "fastfields/api/distance.h"   // ff::dt_euclidean / dt_l1 / dt_spline_* / dt_mesh
#include "fastfields/api/posdef.h"     // ff::sym_matvec / sym_solve / sym_invert / ...
#include "fastfields/api/resize.h"     // ff::resample  + dlpack.h (DLTensor)
#include "fastfields/api/restrict.h"   // ff::restriction
#include "fastfields/api/splinc.h"     // ff::spline_coeff
#include "fastfields/api/pushpull.h"   // ff::pull / push / count / grad
#include "fastfields/api/reg_field.h"  // ff::field_matvec / field_diag
#include "fastfields/api/reg_flow.h"   // ff::flow_matvec / flow_diag
#include "fastfields/api/solve_field.h"  // ff::field_cg

namespace nb = nanobind;
using namespace nb::literals;
using arr = nb::ndarray<>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Fill a DLTensor that views the memory of a nanobind ndarray. The shape and
// stride pointers alias nanobind's internal storage, which stays alive for the
// duration of the wrapped call (the ndarray argument outlives it), so no copy
// is required. Strides are already in elements (DLPack convention).
static DLTensor to_dltensor(arr &a) {
    DLTensor t;
    std::memset(&t, 0, sizeof(t));
    t.data = a.data_handle();
    t.device.device_type = (DLDeviceType)a.device_type();
    t.device.device_id = a.device_id();
    t.ndim = (int32_t)a.ndim();
    t.dtype.code = a.dtype().code;
    t.dtype.bits = a.dtype().bits;
    t.dtype.lanes = a.dtype().lanes;
    t.shape = const_cast<int64_t *>(a.shape_ptr());
    t.strides = const_cast<int64_t *>(a.stride_ptr());
    t.byte_offset = 0;
    return t;
}

// A DLTensor with null data: the library treats this as "absent" for the
// optional weight (posdef) and nearest_vertex (mesh) arguments.
static DLTensor null_dltensor() {
    DLTensor t;
    std::memset(&t, 0, sizeof(t));
    t.data = nullptr;
    return t;
}

// ---------------------------------------------------------------------------
// Argument sanity checks
// ---------------------------------------------------------------------------
//
// This is a raw layer and stays sharp: it hands the caller's memory straight to
// the kernels and does not second-guess shapes. `fastfields-lib` validates the
// *semantic* contracts (matching dtypes, matching batch shapes, `ndim` vs.
// rank, all operands on one device) and raises `std::invalid_argument` when
// they are violated.
//
// What it cannot validate is whether a DLTensor is well-formed enough to be
// inspected at all. Several entry points read `t.shape[t.ndim - 1]` -- the
// coordinate dimension of `loc`/`grid`, the axis `dt_euclidean` / `dt_l1` /
// `spline_coeff` sweep, the channel count of the posdef operands -- *before*
// any of their own checks run. For a 0-d array that index is `shape[-1]`, and
// nanobind leaves the shape/stride pointers null when there is no dimension to
// describe, so the read is off the end of a null pointer. Likewise every entry
// dereferences `t.data`. These are the preconditions of the library's own
// checks, so they belong here, on the way in.
//
// Deliberately *not* policy: any rank the library can index is accepted, any
// dtype it dispatches on, any strides (including non-contiguous and negative),
// any device. Only structurally malformed tensors are rejected -- and the
// rank check is per-argument, so the unbatched form of dt_spline_*/dt_mesh,
// where the per-point outputs are genuinely 0-d, keeps working (see
// `Args::batch_shaped`). The cost is a handful of integer comparisons per
// call, against millisecond-scale kernels.

// Upper bound on the rank we accept. NumPy's own maximum is 64 and no
// fastfields op comes close, so this only catches a garbage `ndim` (negative,
// or absurdly large) that would otherwise index shape/stride arrays out of
// bounds.
static constexpr int32_t FF_MAX_NDIM = 64;

[[noreturn]] static void arg_error(const char *op, const char *name,
                                   const std::string &what) {
    throw std::invalid_argument(std::string("fastfields.") + op +
                                "(): argument '" + name + "' " + what);
}

static std::string device_str(const DLDevice &d) {
    return "(device_type=" + std::to_string((int)d.device_type) +
           ", device_id=" + std::to_string((int)d.device_id) + ")";
}

// Per-call argument checker: converts each ndarray to a DLTensor, checks it is
// well-formed, and requires every tensor in the call to sit on one device.
// One instance per binding, named for the op so errors say which call failed.
class Args {
  public:
    // Tag for an argument whose rank *is* the batch rank: the per-point
    // outputs of dt_spline_* / dt_mesh (`time`, `dist`, `nearest_vertex`). A
    // single unbatched point has no batch dimension, so those are legitimately
    // 0-d -- and nothing reads their trailing dimension, so the rank check is
    // relaxed for them and for them only.
    static constexpr bool batch_shaped = true;

    explicit Args(const char *op) : op_(op) {}

    // A required tensor argument.
    DLTensor operator()(arr &a, const char *name, bool allow_scalar = false) {
        DLTensor t = to_dltensor(a);
        check(t, name, allow_scalar);
        return t;
    }

    // An optional tensor argument. Absent -> a null-data placeholder, which
    // the library reads as "not provided"; present -> checked as usual (in
    // particular a null data pointer is rejected rather than silently
    // demoting the argument to "absent").
    DLTensor opt(std::optional<arr> &a, const char *name,
                 bool allow_scalar = false) {
        if (!a.has_value())
            return null_dltensor();
        return (*this)(*a, name, allow_scalar);
    }

  private:
    void check(const DLTensor &t, const char *name, bool allow_scalar) {
        if (t.data == nullptr)
            arg_error(op_, name, "has a null data pointer");
        if (t.ndim < 0)
            arg_error(op_, name, "has a negative rank");
        if (t.ndim < 1 && !allow_scalar)
            arg_error(op_, name,
                      "is a 0-d array; it is indexed along its last axis, so "
                      "it must have rank >= 1");
        if (t.ndim > FF_MAX_NDIM)
            arg_error(op_, name,
                      "has rank " + std::to_string(t.ndim) + ", above the " +
                          std::to_string(FF_MAX_NDIM) + " supported");
        // A 0-d array has no shape/stride arrays to point at, so only demand
        // them once there is a dimension to describe.
        if (t.ndim > 0 && (t.shape == nullptr || t.strides == nullptr))
            arg_error(op_, name, "has no shape/stride information");
        if (t.dtype.lanes != 1)
            arg_error(op_, name,
                      "has a vector dtype (lanes=" +
                          std::to_string((int)t.dtype.lanes) +
                          "); only scalar dtypes are supported");
        if (t.dtype.bits != 8 && t.dtype.bits != 16 && t.dtype.bits != 32 &&
            t.dtype.bits != 64)
            arg_error(op_, name,
                      "has an unsupported dtype width (" +
                          std::to_string((int)t.dtype.bits) +
                          " bits); expected 8, 16, 32 or 64");
        // Device consistency: the library dispatches the whole op on one
        // operand's device, so a mismatch means the wrong backend reads the
        // other operand's pointer -- a device pointer read as host memory, or
        // vice versa. Catch it here, naming both arguments.
        if (!seen_) {
            seen_ = true;
            dev_ = t.device;
            ref_ = name;
        } else if (t.device.device_type != dev_.device_type ||
                   t.device.device_id != dev_.device_id) {
            arg_error(op_, name,
                      "is on " + device_str(t.device) + " but '" +
                          std::string(ref_) + "' is on " + device_str(dev_) +
                          "; all arrays in a call must be on the same device");
        }
    }

    const char *op_;
    const char *ref_ = nullptr;
    bool seen_ = false;
    DLDevice dev_{};
};

// A `const double *` view of an optional Python sequence, or nullptr when the
// sequence is absent/empty (the library reads a null pointer as "use the
// default": all-ones voxel size, or a disabled penalty). Used by the
// regulariser bindings for voxel_size / absolute / membrane / bending.
static const double *vec_ptr(std::optional<std::vector<double>> &v) {
    return (v.has_value() && !v->empty()) ? v->data() : nullptr;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
NB_MODULE(_core, m) {
    m.doc() = "nanobind bindings to fastfields-lib (DLPack in/out).\n"
              "Arrays may be any object exposing __dlpack__ (numpy/torch/cupy).\n"
              "spline order: 0=Nearest 1=Linear 2=Quadratic 3=Cubic ... 7.\n"
              "bound: 0=Zero 1=Replicate 2=DCT1 3=DCT2 4=DST1 5=DST2 6=DFT 7=NoCheck.";

    // ----- distance.h -----
    m.def(
        "dt_euclidean",
        [](arr inp_out, double voxel_spacing, intptr_t stream) {
            Args A("dt_euclidean");
            DLTensor t = A(inp_out, "inp_out");
            ff::dt_euclidean(t, voxel_spacing, stream);
        },
        "inp_out"_a, "voxel_spacing"_a = 1.0, "stream"_a = 0,
        "In-place Euclidean distance transform along the last axis "
        "(float32/float64; 0 at features, +inf elsewhere).");

    m.def(
        "dt_l1",
        [](arr inp_out, double voxel_spacing, intptr_t stream) {
            Args A("dt_l1");
            DLTensor t = A(inp_out, "inp_out");
            ff::dt_l1(t, voxel_spacing, stream);
        },
        "inp_out"_a, "voxel_spacing"_a = 1.0, "stream"_a = 0,
        "In-place L1 distance transform along the last axis.");

    m.def(
        "dt_spline_table",
        [](arr time, arr dist, arr loc, arr coeff, arr times, int8_t spline,
           int8_t bound, intptr_t stream) {
            Args A("dt_spline_table");
            DLTensor t = A(time, "time", Args::batch_shaped),
                     d = A(dist, "dist", Args::batch_shaped),
                     l = A(loc, "loc"), c = A(coeff, "coeff"),
                     ts = A(times, "times");
            ff::dt_spline_table(t, d, l, c, ts, spline, bound, stream);
        },
        "time"_a, "dist"_a, "loc"_a, "coeff"_a, "times"_a, "spline"_a = 3,
        "bound"_a = 3, "stream"_a = 0,
        "Point-to-spline squared distance via a dictionary of candidate "
        "times. loc (*batch,D), coeff (*batch,npoints,D), times "
        "(*batch,ntimes); time and dist are (*batch,) -- 0-d for a single "
        "point. Writes the best time and its squared distance.");

    m.def(
        "dt_spline_brent",
        [](arr time, arr dist, arr loc, arr coeff, int64_t max_iter, double tol,
           double step, int8_t spline, int8_t bound, intptr_t stream) {
            Args A("dt_spline_brent");
            DLTensor t = A(time, "time", Args::batch_shaped),
                     d = A(dist, "dist", Args::batch_shaped),
                     l = A(loc, "loc"), c = A(coeff, "coeff");
            ff::dt_spline_brent(t, d, l, c, max_iter, tol, step, spline, bound,
                              stream);
        },
        "time"_a, "dist"_a, "loc"_a, "coeff"_a, "max_iter"_a, "tol"_a, "step"_a,
        "spline"_a = 3, "bound"_a = 3, "stream"_a = 0,
        "Point-to-spline squared distance via Brent's method. Refines the "
        "`time`/`dist` already in the buffers (seed them with a coarse "
        "dt_spline_table pass); `step` is the initial bracket half-width "
        "and must be wide enough to bracket the optimum.");

    m.def(
        "dt_spline_gaussnewton",
        [](arr time, arr dist, arr loc, arr coeff, int64_t max_iter, double tol,
           int8_t spline, int8_t bound, intptr_t stream) {
            Args A("dt_spline_gaussnewton");
            DLTensor t = A(time, "time", Args::batch_shaped),
                     d = A(dist, "dist", Args::batch_shaped),
                     l = A(loc, "loc"), c = A(coeff, "coeff");
            ff::dt_spline_gaussnewton(t, d, l, c, max_iter, tol, spline, bound,
                                    stream);
        },
        "time"_a, "dist"_a, "loc"_a, "coeff"_a, "max_iter"_a, "tol"_a,
        "spline"_a = 3, "bound"_a = 3, "stream"_a = 0,
        "Point-to-spline squared distance via Gauss-Newton optimization. "
        "Refines the `time`/`dist` already in the buffers (seed them with "
        "a coarse dt_spline_table pass).");

    m.def(
        "dt_mesh",
        [](arr dist, std::optional<arr> nearest_vertex, arr loc, arr vertices,
           arr faces, bool signed_, bool naive, intptr_t stream) {
            Args A("dt_mesh");
            DLTensor d = A(dist, "dist", Args::batch_shaped);
            DLTensor nv = A.opt(nearest_vertex, "nearest_vertex",
                                Args::batch_shaped);
            DLTensor l = A(loc, "loc"), v = A(vertices, "vertices"),
                     f = A(faces, "faces");
            ff::dt_mesh(d, nv, l, v, f, signed_, naive, stream);
        },
        "dist"_a, "nearest_vertex"_a.none() = nb::none(), "loc"_a, "vertices"_a,
        "faces"_a, "signed_"_a = true, "naive"_a = false, "stream"_a = 0,
        "Point-to-triangular-mesh distance. `dist` is the Euclidean distance "
        "(NOT squared, unlike dt_spline_*), negative inside the surface when "
        "signed_. loc (*batch,D), vertices (N,D), faces (M,D); dist and the "
        "optional nearest_vertex are (*batch,) -- 0-d for a single point.");

    // ----- posdef.h -----
    m.def(
        "sym_matvec",
        [](arr out, arr hessian, arr inp, intptr_t stream) {
            Args A("sym_matvec");
            DLTensor o = A(out, "out"), h = A(hessian, "hessian"),
                     i = A(inp, "inp");
            ff::sym_matvec(o, h, i, stream);
        },
        "out"_a, "hessian"_a, "inp"_a, "stream"_a = 0,
        "out = H @ inp (H is compact-symmetric, diagonal-then-rows packed).");

    m.def(
        "sym_matvec_backward",
        [](arr out, arr grd, arr inp, intptr_t stream) {
            Args A("sym_matvec_backward");
            DLTensor o = A(out, "out"), g = A(grd, "grd"),
                     i = A(inp, "inp");
            ff::sym_matvec_backward(o, g, i, stream);
        },
        "out"_a, "grd"_a, "inp"_a, "stream"_a = 0,
        "Backward of sym_matvec wrt the matrix.");

    m.def(
        "sym_addmatvec_",
        [](arr out, arr hessian, arr inp, intptr_t stream) {
            Args A("sym_addmatvec_");
            DLTensor o = A(out, "out"), h = A(hessian, "hessian"),
                     i = A(inp, "inp");
            ff::sym_addmatvec_(o, h, i, stream);
        },
        "out"_a, "hessian"_a, "inp"_a, "stream"_a = 0, "out += H @ inp.");

    m.def(
        "sym_submatvec_",
        [](arr out, arr hessian, arr inp, intptr_t stream) {
            Args A("sym_submatvec_");
            DLTensor o = A(out, "out"), h = A(hessian, "hessian"),
                     i = A(inp, "inp");
            ff::sym_submatvec_(o, h, i, stream);
        },
        "out"_a, "hessian"_a, "inp"_a, "stream"_a = 0, "out -= H @ inp.");

    m.def(
        "sym_solve",
        [](arr out, arr hessian, arr inp, std::optional<arr> weight,
           intptr_t stream) {
            Args A("sym_solve");
            DLTensor o = A(out, "out"), h = A(hessian, "hessian"),
                     i = A(inp, "inp");
            DLTensor w = A.opt(weight, "weight");
            ff::sym_solve(o, h, i, w, stream);
        },
        "out"_a, "hessian"_a, "inp"_a, "weight"_a.none() = nb::none(),
        "stream"_a = 0, "out = (H + diag(weight)) \\ inp (weight optional).");

    m.def(
        "sym_solve_",
        [](arr inp_out, arr hessian, std::optional<arr> weight, intptr_t stream) {
            Args A("sym_solve_");
            DLTensor io = A(inp_out, "inp_out"), h = A(hessian, "hessian");
            DLTensor w = A.opt(weight, "weight");
            ff::sym_solve_(io, h, w, stream);
        },
        "inp_out"_a, "hessian"_a, "weight"_a.none() = nb::none(),
        "stream"_a = 0,
        "In-place: inp_out = (H + diag(weight)) \\ inp_out (weight optional).");

    m.def(
        "sym_invert",
        [](arr out, arr hessian, intptr_t stream) {
            Args A("sym_invert");
            DLTensor o = A(out, "out"), h = A(hessian, "hessian");
            ff::sym_invert(o, h, stream);
        },
        "out"_a, "hessian"_a, "stream"_a = 0,
        "out = inv(H) (both compact-symmetric).");

    m.def(
        "sym_invert_",
        [](arr hessian, intptr_t stream) {
            Args A("sym_invert_");
            DLTensor h = A(hessian, "hessian");
            ff::sym_invert_(h, stream);
        },
        "hessian"_a, "stream"_a = 0,
        "In-place: hessian = inv(hessian) (compact-symmetric).");

    // ----- resize.h / restrict.h (scale as a Python sequence) -----
    auto resize_like = [](arr &out, arr &inp, int8_t spline, int8_t bound,
                          double shift, std::optional<std::vector<double>> &scale,
                          int ndim, intptr_t stream, bool restriction) {
        Args A(restriction ? "restriction" : "resample");
        DLTensor o = A(out, "out"), i = A(inp, "inp");
        const double *scale_ptr = nullptr;
        if (scale.has_value() && !scale->empty())
            scale_ptr = scale->data();
        if (restriction)
            ff::restriction(o, i, spline, bound, shift, scale_ptr, ndim, stream);
        else
            ff::resample(o, i, spline, bound, shift, scale_ptr, ndim, stream);
    };

    m.def(
        "resample",
        [resize_like](arr out, arr inp, int8_t spline, int8_t bound,
                      double shift, std::optional<std::vector<double>> scale,
                      int ndim, intptr_t stream) {
            resize_like(out, inp, spline, bound, shift, scale, ndim, stream,
                        false);
        },
        "out"_a, "inp"_a, "spline"_a = 2, "bound"_a = 3, "shift"_a = 0.0,
        "scale"_a.none() = nb::none(), "ndim"_a = 1, "stream"_a = 0,
        "Spline resample (prolongation). scale is a per-dim sequence of length "
        "ndim (input-index per output-index).");

    m.def(
        "restriction",
        [resize_like](arr out, arr inp, int8_t spline, int8_t bound,
                      double shift, std::optional<std::vector<double>> scale,
                      int ndim, intptr_t stream) {
            resize_like(out, inp, spline, bound, shift, scale, ndim, stream,
                        true);
        },
        "out"_a, "inp"_a, "spline"_a = 2, "bound"_a = 3, "shift"_a = 0.0,
        "scale"_a.none() = nb::none(), "ndim"_a = 1, "stream"_a = 0,
        "Restriction (adjoint of resample). out is accumulated into and must be "
        "pre-zeroed by the caller.");

    // ----- splinc.h -----
    m.def(
        "spline_coeff",
        [](arr inp_out, int8_t spline, int8_t bound, intptr_t stream) {
            Args A("spline_coeff");
            DLTensor t = A(inp_out, "inp_out");
            ff::spline_coeff(t, spline, bound, stream);
        },
        "inp_out"_a, "spline"_a = 3, "bound"_a = 3, "stream"_a = 0,
        "In-place spline-coefficient prefilter along the last axis "
        "(orders 0/1 are no-ops).");

    // ----- pushpull.h -----
    // Channel-last, x-first coordinate convention. `grid` holds sampling
    // coordinates in voxels; D (its last dim) is the spatial rank (1/2/3).
    m.def(
        "pull",
        [](arr out, arr inp, arr grid, int8_t spline, int8_t bound,
           int8_t extrapolate, intptr_t stream) {
            Args A("pull");
            DLTensor o = A(out, "out"), i = A(inp, "inp"),
                     g = A(grid, "grid");
            ff::pull(o, i, g, spline, bound, extrapolate, stream);
        },
        "out"_a, "inp"_a, "grid"_a, "spline"_a = 2, "bound"_a = 3,
        "extrapolate"_a = 1, "stream"_a = 0,
        "Sample (pull) a spline-encoded volume at grid coordinates. "
        "inp (*batch,*inshape,C), grid (*batch,*outshape,D), "
        "out (*batch,*outshape,C).");

    m.def(
        "push",
        [](arr out, arr inp, arr grid, int8_t spline, int8_t bound,
           int8_t extrapolate, intptr_t stream) {
            Args A("push");
            DLTensor o = A(out, "out"), i = A(inp, "inp"),
                     g = A(grid, "grid");
            ff::push(o, i, g, spline, bound, extrapolate, stream);
        },
        "out"_a, "inp"_a, "grid"_a, "spline"_a = 2, "bound"_a = 3,
        "extrapolate"_a = 1, "stream"_a = 0,
        "Splat (push) values into a volume; adjoint of pull. out "
        "(*batch,*inshape,C) is accumulated into and must be pre-zeroed.");

    m.def(
        "count",
        [](arr out, arr grid, int8_t spline, int8_t bound, int8_t extrapolate,
           intptr_t stream) {
            Args A("count");
            DLTensor o = A(out, "out"), g = A(grid, "grid");
            ff::count(o, g, spline, bound, extrapolate, stream);
        },
        "out"_a, "grid"_a, "spline"_a = 2, "bound"_a = 3, "extrapolate"_a = 1,
        "stream"_a = 0,
        "Splat ones (push of an all-ones input). out (*batch,*inshape,1) must "
        "be pre-zeroed.");

    m.def(
        "grad",
        [](arr out, arr inp, arr grid, int8_t spline, int8_t bound,
           int8_t extrapolate, bool abs, intptr_t stream) {
            Args A("grad");
            DLTensor o = A(out, "out"), i = A(inp, "inp"),
                     g = A(grid, "grid");
            ff::grad(o, i, g, spline, bound, extrapolate, abs, stream);
        },
        "out"_a, "inp"_a, "grid"_a, "spline"_a = 2, "bound"_a = 3,
        "extrapolate"_a = 1, "abs"_a = false, "stream"_a = 0,
        "Sample spatial gradients of a spline-encoded volume. "
        "out (*batch,*outshape,C,D).");

    // ----- pushpull.h : backward passes -----
    // Adjoints wrt *both* the field and the sampling coordinates. `out` is
    // the gradient wrt the forward `inp`, `gout` the gradient wrt `grid`,
    // `ginp` the incoming gradient (shaped like the forward op's output).
    // Pre-zero `out`: pull_backward / grad_backward scatter into it.
    m.def(
        "pull_backward",
        [](arr out, arr gout, arr inp, arr ginp, arr grid, int8_t spline,
           int8_t bound, int8_t extrapolate, intptr_t stream) {
            Args A("pull_backward");
            DLTensor o = A(out, "out"), go = A(gout, "gout"),
                     i = A(inp, "inp"), gi = A(ginp, "ginp"),
                     g = A(grid, "grid");
            ff::pull_backward(o, go, i, gi, g, spline, bound, extrapolate,
                              stream);
        },
        "out"_a, "gout"_a, "inp"_a, "ginp"_a, "grid"_a, "spline"_a = 2,
        "bound"_a = 3, "extrapolate"_a = 1, "stream"_a = 0,
        "Adjoint of pull. out (*batch,*inshape,C) is accumulated into and "
        "must be pre-zeroed; gout (*batch,*outshape,D) is overwritten; "
        "ginp (*batch,*outshape,C).");

    m.def(
        "push_backward",
        [](arr out, arr gout, arr inp, arr ginp, arr grid, int8_t spline,
           int8_t bound, int8_t extrapolate, intptr_t stream) {
            Args A("push_backward");
            DLTensor o = A(out, "out"), go = A(gout, "gout"),
                     i = A(inp, "inp"), gi = A(ginp, "ginp"),
                     g = A(grid, "grid");
            ff::push_backward(o, go, i, gi, g, spline, bound, extrapolate,
                              stream);
        },
        "out"_a, "gout"_a, "inp"_a, "ginp"_a, "grid"_a, "spline"_a = 2,
        "bound"_a = 3, "extrapolate"_a = 1, "stream"_a = 0,
        "Adjoint of push. out (*batch,*outshape,C) and gout "
        "(*batch,*outshape,D) are overwritten; ginp (*batch,*inshape,C).");

    m.def(
        "count_backward",
        [](arr gout, arr ginp, arr grid, int8_t spline, int8_t bound,
           int8_t extrapolate, intptr_t stream) {
            Args A("count_backward");
            DLTensor go = A(gout, "gout"), gi = A(ginp, "ginp"),
                     g = A(grid, "grid");
            ff::count_backward(go, gi, g, spline, bound, extrapolate, stream);
        },
        "gout"_a, "ginp"_a, "grid"_a, "spline"_a = 2, "bound"_a = 3,
        "extrapolate"_a = 1, "stream"_a = 0,
        "Adjoint of count wrt grid. gout (*batch,*outshape,D) is "
        "overwritten; ginp (*batch,*inshape,1).");

    m.def(
        "grad_backward",
        [](arr out, arr gout, arr inp, arr ginp, arr grid, int8_t spline,
           int8_t bound, int8_t extrapolate, bool abs, intptr_t stream) {
            Args A("grad_backward");
            DLTensor o = A(out, "out"), go = A(gout, "gout"),
                     i = A(inp, "inp"), gi = A(ginp, "ginp"),
                     g = A(grid, "grid");
            ff::grad_backward(o, go, i, gi, g, spline, bound, extrapolate,
                              abs, stream);
        },
        "out"_a, "gout"_a, "inp"_a, "ginp"_a, "grid"_a, "spline"_a = 2,
        "bound"_a = 3, "extrapolate"_a = 1, "abs"_a = false, "stream"_a = 0,
        "Adjoint of grad. out (*batch,*inshape,C) is accumulated into and "
        "must be pre-zeroed; gout (*batch,*outshape,D) is overwritten; "
        "ginp (*batch,*outshape,C,D). `abs` must match the forward call.");

    // ----- reg_field.h (multi-channel field; per-channel penalty vectors) -----
    // voxel_size is a length-ndim sequence; absolute/membrane/bending are
    // length-C sequences (any may be omitted -> that penalty is disabled).
    m.def(
        "field_matvec",
        [](arr out, arr inp, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_matvec");
            DLTensor o = A(out, "out"), i = A(inp, "inp");
            ff::field_matvec(o, i, vec_ptr(voxel_size), vec_ptr(absolute),
                             vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                             stream);
        },
        "out"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "Apply a spatial regulariser to a multi-channel field "
        "(*batch,*spatial,C).");

    m.def(
        "field_diag",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_diag");
            DLTensor o = A(out, "out");
            ff::field_diag(o, vec_ptr(voxel_size), vec_ptr(absolute),
                           vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                           stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "Diagonal (preconditioner) of the field regulariser operator.");

    m.def(
        "field_relax",
        [](arr sol, arr hes, arr grd,
           std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           int nb_iter, intptr_t stream) {
            Args A("field_relax");
            DLTensor s = A(sol, "sol"), h = A(hes, "hes"),
                     g = A(grd, "grd");
            ff::field_relax(s, h, g, vec_ptr(voxel_size), vec_ptr(absolute),
                            vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                            nb_iter, stream);
        },
        "sol"_a, "hes"_a, "grd"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "nb_iter"_a = 1, "stream"_a = 0,
        "In-place relaxation sweeps solving (H + L) x = g for a "
        "multi-channel field.");

    m.def(
        "field_kernel",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_kernel");
            DLTensor o = A(out, "out");
            ff::field_kernel(o, vec_ptr(voxel_size), vec_ptr(absolute),
                             vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                             stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "Materialise the Toeplitz convolution kernel of the field "
        "regulariser.");

    // --- the regularised system: forward, preconditioner, CG solver ---
    //
    // These three take the per-voxel compact-symmetric data-term Hessian `H`
    // alongside the regulariser `L`, and are the operators an outer
    // optimisation loop needs: `field_forward` applies (H + L), `field_precond`
    // approximates its inverse by the Jacobi diagonal (H + diag(L))^-1, and
    // `field_relax` / `field_cg` solve the system. Each is a single kernel; a
    // Python-side composition of field_matvec/field_diag + sym_matvec/sym_solve
    // computes the same thing but allocates a full-size temporary for the
    // regulariser result and walks the field twice.
    m.def(
        "field_forward",
        [](arr out, arr hes, arr inp,
           std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_forward");
            DLTensor o = A(out, "out"), h = A(hes, "hes"), i = A(inp, "inp");
            ff::field_forward(o, h, i, vec_ptr(voxel_size), vec_ptr(absolute),
                              vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                              stream);
        },
        "out"_a, "hes"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "out = (H + L) @ inp for a multi-channel field: the forward "
        "application of the regularised system field_relax/field_cg solve. "
        "hes is the compact-symmetric per-voxel Hessian "
        "(*batch,*spatial,C*(C+1)/2); out and inp are (*batch,*spatial,C).");

    m.def(
        "field_precond",
        [](arr out, arr hes, arr grd,
           std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_precond");
            DLTensor o = A(out, "out"), h = A(hes, "hes"), g = A(grd, "grd");
            ff::field_precond(o, h, g, vec_ptr(voxel_size), vec_ptr(absolute),
                              vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                              stream);
        },
        "out"_a, "hes"_a, "grd"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "out = (H + diag(L)) \\ grd: the Jacobi preconditioner of the "
        "regularised field system. hes is the compact-symmetric per-voxel "
        "Hessian; out and grd are (*batch,*spatial,C).");

    m.def(
        "field_precond_",
        [](arr sol, arr hes, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_precond_");
            DLTensor s = A(sol, "sol"), h = A(hes, "hes");
            ff::field_precond_(s, h, vec_ptr(voxel_size), vec_ptr(absolute),
                               vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                               stream);
        },
        "sol"_a, "hes"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "In-place field_precond: sol holds grd on entry and "
        "(H + diag(L)) \\ grd on exit.");

    // ----- solve_field.h -----
    //
    // CPU ONLY. The hub has no FF_CUDA::field_cg to forward to -- the solver's
    // dot products need a device-side reduction the cuda backend does not
    // expose yet (fastfields-lib#34) -- so a CUDA tensor raises ValueError
    // from the hub rather than silently running on the host. Check
    // `fastfields.dlpack.backend` if you need to branch before the call.
    //
    // Unlike every other binding here this one *returns* something: the two
    // `int*`/`double*` out-parameters of ff::field_cg (iterations run, final
    // relative residual) are convergence diagnostics about the call rather
    // than field data, so they come back as a Python tuple instead of as
    // caller-allocated 0-d arrays. The arrays themselves stay caller-allocated
    // as everywhere else.
    m.def(
        "field_cg",
        [](arr sol, arr hes, arr grd,
           std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           int nb_iter, double tol, intptr_t stream) {
            Args A("field_cg");
            DLTensor s = A(sol, "sol"), h = A(hes, "hes"), g = A(grd, "grd");
            int nb_iter_out = 0;
            double residual_out = 0.0;
            ff::field_cg(s, h, g, vec_ptr(voxel_size), vec_ptr(absolute),
                         vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                         nb_iter, tol, &nb_iter_out, &residual_out, stream);
            return nb::make_tuple(nb_iter_out, residual_out);
        },
        "sol"_a, "hes"_a, "grd"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "nb_iter"_a = 32, "tol"_a = 1e-8, "stream"_a = 0,
        "Solve (H + L) x = g by Jacobi-preconditioned conjugate gradients, "
        "in place: `sol` is the warm start on entry (pass zeros for a cold "
        "start) and the solution on exit. Returns (iterations_run, "
        "final_relative_residual). CPU ONLY -- a CUDA tensor raises "
        "ValueError (fastfields-lib#34).");

    // --- RLS/JRLS (weighted) variants ---
    // `wgt` selects the mode via its trailing dimension: 1 (one weight shared
    // -- "joint" -- across all channels, JRLS) or C (genuine per-channel
    // weight, RLS). This is the jitfields/nitorch convention; see
    // fastfields-cpu-lib#65, where the dispatch predicate had it backwards.
    m.def(
        "field_matvec_rls",
        [](arr out, arr inp, arr wgt,
           std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_matvec_rls");
            DLTensor o = A(out, "out"), i = A(inp, "inp"),
                     w = A(wgt, "wgt");
            ff::field_matvec_rls(o, i, w, vec_ptr(voxel_size),
                                 vec_ptr(absolute), vec_ptr(membrane),
                                 vec_ptr(bending), bound, ndim, stream);
        },
        "out"_a, "inp"_a, "wgt"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "RLS/JRLS-weighted variant of field_matvec: `wgt` is "
        "(*batch,*spatial,1) for JRLS (one weight shared across channels) "
        "or (*batch,*spatial,C) for RLS (a genuine per-channel weight).");

    m.def(
        "field_diag_rls",
        [](arr out, arr wgt, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_diag_rls");
            DLTensor o = A(out, "out"), w = A(wgt, "wgt");
            ff::field_diag_rls(o, w, vec_ptr(voxel_size), vec_ptr(absolute),
                               vec_ptr(membrane), vec_ptr(bending), bound,
                               ndim, stream);
        },
        "out"_a, "wgt"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "Diagonal (preconditioner) of the RLS/JRLS-weighted field "
        "regulariser operator, same `wgt` conventions as field_matvec_rls.");

    m.def(
        "field_relax_rls",
        [](arr sol, arr hes, arr grd, arr wgt,
           std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           int nb_iter, intptr_t stream) {
            Args A("field_relax_rls");
            DLTensor s = A(sol, "sol"), h = A(hes, "hes"),
                     g = A(grd, "grd"), w = A(wgt, "wgt");
            ff::field_relax_rls(s, h, g, w, vec_ptr(voxel_size),
                                vec_ptr(absolute), vec_ptr(membrane),
                                vec_ptr(bending), bound, ndim, nb_iter,
                                stream);
        },
        "sol"_a, "hes"_a, "grd"_a, "wgt"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "nb_iter"_a = 1, "stream"_a = 0,
        "In-place RLS/JRLS-weighted relaxation sweeps solving (H + L(w)) x = "
        "g for a multi-channel field, same `wgt` conventions as "
        "field_matvec_rls.");

    // --- in-place accumulate variants (restored jitfields op '+'/'-') ---
    m.def(
        "field_addmatvec_",
        [](arr out, arr inp, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_addmatvec_");
            DLTensor o = A(out, "out"), i = A(inp, "inp");
            ff::field_addmatvec_(o, i, vec_ptr(voxel_size), vec_ptr(absolute),
                                 vec_ptr(membrane), vec_ptr(bending), bound,
                                 ndim, stream);
        },
        "out"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "In-place: out += L(inp) for a multi-channel field. Accumulates into "
        "the caller's `out` (jitfields op '+'); out-of-place is a caller-side "
        "copy of `out` followed by this call.");

    m.def(
        "field_submatvec_",
        [](arr out, arr inp, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_submatvec_");
            DLTensor o = A(out, "out"), i = A(inp, "inp");
            ff::field_submatvec_(o, i, vec_ptr(voxel_size), vec_ptr(absolute),
                                 vec_ptr(membrane), vec_ptr(bending), bound,
                                 ndim, stream);
        },
        "out"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "In-place: out -= L(inp) for a multi-channel field. Accumulates into "
        "the caller's `out` (jitfields op '-'); out-of-place is a caller-side "
        "copy of `out` followed by this call.");

    m.def(
        "field_adddiag_",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_adddiag_");
            DLTensor o = A(out, "out");
            ff::field_adddiag_(o, vec_ptr(voxel_size), vec_ptr(absolute),
                               vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                               stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "In-place: out += diag(L) of the field regulariser (jitfields op '+').");

    m.def(
        "field_subdiag_",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_subdiag_");
            DLTensor o = A(out, "out");
            ff::field_subdiag_(o, vec_ptr(voxel_size), vec_ptr(absolute),
                               vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                               stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "In-place: out -= diag(L) of the field regulariser (jitfields op '-').");

    m.def(
        "field_addkernel_",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_addkernel_");
            DLTensor o = A(out, "out");
            ff::field_addkernel_(o, vec_ptr(voxel_size), vec_ptr(absolute),
                               vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                               stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "In-place: out += the stencil K of the field regulariser (jitfields op '+').");

    m.def(
        "field_subkernel_",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           std::optional<std::vector<double>> absolute,
           std::optional<std::vector<double>> membrane,
           std::optional<std::vector<double>> bending, int8_t bound, int ndim,
           intptr_t stream) {
            Args A("field_subkernel_");
            DLTensor o = A(out, "out");
            ff::field_subkernel_(o, vec_ptr(voxel_size), vec_ptr(absolute),
                               vec_ptr(membrane), vec_ptr(bending), bound, ndim,
                               stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a.none() = nb::none(), "membrane"_a.none() = nb::none(),
        "bending"_a.none() = nb::none(), "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "In-place: out -= the stencil K of the field regulariser (jitfields op '-').");

    // ----- reg_flow.h (vector flow field; scalar penalties) -----
    m.def(
        "flow_matvec",
        [](arr out, arr inp, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_matvec");
            DLTensor o = A(out, "out"), i = A(inp, "inp");
            ff::flow_matvec(o, i, vec_ptr(voxel_size), absolute, membrane,
                            bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "Apply a spatial regulariser to a vector flow field.");

    m.def(
        "flow_diag",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_diag");
            DLTensor o = A(out, "out");
            ff::flow_diag(o, vec_ptr(voxel_size), absolute, membrane, bending,
                          shears, div, bound, ndim, stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(), "absolute"_a = 0.0,
        "membrane"_a = 0.0, "bending"_a = 0.0, "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "Diagonal (preconditioner) of the flow regulariser operator.");

    m.def(
        "flow_relax",
        [](arr sol, arr hes, arr grd,
           std::optional<std::vector<double>> voxel_size, double absolute,
           double membrane, double bending, double shears, double div,
           int8_t bound, int ndim, int nb_iter, intptr_t stream) {
            Args A("flow_relax");
            DLTensor s = A(sol, "sol"), h = A(hes, "hes"),
                     g = A(grd, "grd");
            ff::flow_relax(s, h, g, vec_ptr(voxel_size), absolute, membrane,
                           bending, shears, div, bound, ndim, nb_iter, stream);
        },
        "sol"_a, "hes"_a, "grd"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0, "bound"_a = 3, "ndim"_a = 1,
        "nb_iter"_a = 1, "stream"_a = 0,
        "In-place relaxation sweeps solving (H + L) x = g for the flow field.");

    m.def(
        "flow_kernel",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_kernel");
            DLTensor o = A(out, "out");
            ff::flow_kernel(o, vec_ptr(voxel_size), absolute, membrane,
                            bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(), "absolute"_a = 0.0,
        "membrane"_a = 0.0, "bending"_a = 0.0, "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "Materialise the Toeplitz convolution kernel of the flow regulariser.");

    // --- the regularised system: forward and preconditioner ---
    // Same role as field_forward / field_precond, for the vector flow field.
    // There is no flow_cg: the hub ships a CG driver for the field family
    // only (api/solve_field.h).
    m.def(
        "flow_forward",
        [](arr out, arr hes, arr inp,
           std::optional<std::vector<double>> voxel_size, double absolute,
           double membrane, double bending, double shears, double div,
           int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_forward");
            DLTensor o = A(out, "out"), h = A(hes, "hes"), i = A(inp, "inp");
            ff::flow_forward(o, h, i, vec_ptr(voxel_size), absolute, membrane,
                             bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "hes"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0, "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "out = (H + L) @ inp for the flow field: the forward application of "
        "the regularised system flow_relax solves. hes is the symmetric "
        "per-voxel Hessian (*batch,*spatial,ndim*(ndim+1)/2); out and inp are "
        "(*batch,*spatial,ndim).");

    m.def(
        "flow_precond",
        [](arr out, arr hes, arr grd,
           std::optional<std::vector<double>> voxel_size, double absolute,
           double membrane, double bending, double shears, double div,
           int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_precond");
            DLTensor o = A(out, "out"), h = A(hes, "hes"), g = A(grd, "grd");
            ff::flow_precond(o, h, g, vec_ptr(voxel_size), absolute, membrane,
                             bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "hes"_a, "grd"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0, "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "out = (H + diag(L)) \\ grd: the Jacobi preconditioner of the "
        "regularised flow system.");

    m.def(
        "flow_precond_",
        [](arr sol, arr hes, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_precond_");
            DLTensor s = A(sol, "sol"), h = A(hes, "hes");
            ff::flow_precond_(s, h, vec_ptr(voxel_size), absolute, membrane,
                              bending, shears, div, bound, ndim, stream);
        },
        "sol"_a, "hes"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0, "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "In-place flow_precond: sol holds grd on entry and "
        "(H + diag(L)) \\ grd on exit.");

    // --- RLS/JRLS (weighted) variants ---
    // Unlike the field family, the flow weight map is always *joint*: the
    // flow components are the components of one displacement vector, so a
    // single weight is shared across all of them and `wgt` must have a
    // trailing size-1 axis. `bending` is not wired with weighting at the loop
    // level (as in jitfields) and the library rejects a non-zero value.
    m.def(
        "flow_matvec_rls",
        [](arr out, arr inp, arr wgt,
           std::optional<std::vector<double>> voxel_size, double absolute,
           double membrane, double bending, double shears, double div,
           int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_matvec_rls");
            DLTensor o = A(out, "out"), i = A(inp, "inp"),
                     w = A(wgt, "wgt");
            ff::flow_matvec_rls(o, i, w, vec_ptr(voxel_size), absolute,
                                membrane, bending, shears, div, bound, ndim,
                                stream);
        },
        "out"_a, "inp"_a, "wgt"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0, "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "RLS/JRLS-weighted variant of flow_matvec: `wgt` is "
        "(*batch,*spatial,1), one weight shared across the flow components. "
        "`bending` is not supported with weighting.");

    m.def(
        "flow_diag_rls",
        [](arr out, arr wgt, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_diag_rls");
            DLTensor o = A(out, "out"), w = A(wgt, "wgt");
            ff::flow_diag_rls(o, w, vec_ptr(voxel_size), absolute, membrane,
                              bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "wgt"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0, "bound"_a = 3, "ndim"_a = 1,
        "stream"_a = 0,
        "Diagonal (preconditioner) of the RLS/JRLS-weighted flow regulariser "
        "operator, same `wgt` conventions as flow_matvec_rls.");

    m.def(
        "flow_relax_rls",
        [](arr sol, arr hes, arr grd, arr wgt,
           std::optional<std::vector<double>> voxel_size, double absolute,
           double membrane, double bending, double shears, double div,
           int8_t bound, int ndim, int nb_iter, intptr_t stream) {
            Args A("flow_relax_rls");
            DLTensor s = A(sol, "sol"), h = A(hes, "hes"),
                     g = A(grd, "grd"), w = A(wgt, "wgt");
            ff::flow_relax_rls(s, h, g, w, vec_ptr(voxel_size), absolute,
                               membrane, bending, shears, div, bound, ndim,
                               nb_iter, stream);
        },
        "sol"_a, "hes"_a, "grd"_a, "wgt"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0, "bound"_a = 3, "ndim"_a = 1,
        "nb_iter"_a = 1, "stream"_a = 0,
        "In-place RLS/JRLS-weighted relaxation sweeps solving "
        "(H + L(w)) x = g for the flow field, same `wgt` conventions as "
        "flow_matvec_rls.");

    // --- in-place accumulate variants (restored jitfields op '+'/'-') ---
    m.def(
        "flow_addmatvec_",
        [](arr out, arr inp, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_addmatvec_");
            DLTensor o = A(out, "out"), i = A(inp, "inp");
            ff::flow_addmatvec_(o, i, vec_ptr(voxel_size), absolute, membrane,
                                bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "In-place: out += L(inp) for a vector flow field (jitfields op '+').");

    m.def(
        "flow_submatvec_",
        [](arr out, arr inp, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_submatvec_");
            DLTensor o = A(out, "out"), i = A(inp, "inp");
            ff::flow_submatvec_(o, i, vec_ptr(voxel_size), absolute, membrane,
                                bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "inp"_a, "voxel_size"_a.none() = nb::none(),
        "absolute"_a = 0.0, "membrane"_a = 0.0, "bending"_a = 0.0,
        "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "In-place: out -= L(inp) for a vector flow field (jitfields op '-').");

    m.def(
        "flow_adddiag_",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_adddiag_");
            DLTensor o = A(out, "out");
            ff::flow_adddiag_(o, vec_ptr(voxel_size), absolute, membrane,
                              bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(), "absolute"_a = 0.0,
        "membrane"_a = 0.0, "bending"_a = 0.0, "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "In-place: out += diag(L) of the flow regulariser (jitfields op '+').");

    m.def(
        "flow_subdiag_",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_subdiag_");
            DLTensor o = A(out, "out");
            ff::flow_subdiag_(o, vec_ptr(voxel_size), absolute, membrane,
                              bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(), "absolute"_a = 0.0,
        "membrane"_a = 0.0, "bending"_a = 0.0, "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "In-place: out -= diag(L) of the flow regulariser (jitfields op '-').");

    m.def(
        "flow_addkernel_",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_addkernel_");
            DLTensor o = A(out, "out");
            ff::flow_addkernel_(o, vec_ptr(voxel_size), absolute, membrane,
                              bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(), "absolute"_a = 0.0,
        "membrane"_a = 0.0, "bending"_a = 0.0, "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "In-place: out += the stencil K of the flow regulariser (jitfields op '+').");

    m.def(
        "flow_subkernel_",
        [](arr out, std::optional<std::vector<double>> voxel_size,
           double absolute, double membrane, double bending, double shears,
           double div, int8_t bound, int ndim, intptr_t stream) {
            Args A("flow_subkernel_");
            DLTensor o = A(out, "out");
            ff::flow_subkernel_(o, vec_ptr(voxel_size), absolute, membrane,
                              bending, shears, div, bound, ndim, stream);
        },
        "out"_a, "voxel_size"_a.none() = nb::none(), "absolute"_a = 0.0,
        "membrane"_a = 0.0, "bending"_a = 0.0, "shears"_a = 0.0, "div"_a = 0.0,
        "bound"_a = 3, "ndim"_a = 1, "stream"_a = 0,
        "In-place: out -= the stencil K of the flow regulariser (jitfields op '-').");
}
