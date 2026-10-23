// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D — Deterministic Reduction Substrate (Package 2) — implementation
//
// Read `deterministic_reduce.cuh` first: it states the ordering guarantee that
// this file implements.  Everything here is subordinate to that contract.
//
// Structural rules obeyed by every line below:
//   * No atomics anywhere on a decision path.  (The one atomic in this file is
//     the deliberately-named DIAGNOSTIC counter-example.)
//   * No cudaMalloc / cudaMemcpy / synchronize / printf in any launcher.
//   * Every FP op is an explicit round-to-nearest-even intrinsic, so neither
//     `--use_fast_math` nor FMA contraction can move the result.
//   * Launch geometry is a compile-time constant, so no autotuning or device
//     query can change the produced bits.
// ═══════════════════════════════════════════════════════════════════════════

#include "deterministic_reduce.cuh"

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

namespace prism_dr {

// ───────────────────────────────────────────────────────────────────────────
// Arithmetic traits.
//
// EVERY operation routes through these.  `__fadd_rn` / `__fmul_rn` /
// `__dadd_rn` / `__dmul_rn` are documented to be emitted as a single
// round-to-nearest-even instruction that the compiler must NOT contract into
// an FMA and must NOT reassociate.  Writing `a + b` instead would be
// correct today (build.rs compiles this TU IEEE-strict) and silently wrong
// the day somebody re-adds `--use_fast_math`.
// ───────────────────────────────────────────────────────────────────────────

template <typename T>
struct Arith;

template <>
struct Arith<float> {
    typedef float value_type;

    __device__ __forceinline__ static float add(float a, float b) {
        return __fadd_rn(a, b);
    }
    __device__ __forceinline__ static float mul(float a, float b) {
        return __fmul_rn(a, b);
    }
    __device__ __forceinline__ static bool is_nan(float a) {
        // Explicit bit test rather than isnan(): immune to any future
        // fast-math flag that would let the compiler assume no-NaN.
        const uint32_t u = __float_as_uint(a);
        return (u & 0x7F800000u) == 0x7F800000u && (u & 0x007FFFFFu) != 0u;
    }
    __device__ __forceinline__ static float pos_inf() {
        return __uint_as_float(0x7F800000u);
    }
    __device__ __forceinline__ static float neg_inf() {
        return __uint_as_float(0xFF800000u);
    }
    __device__ __forceinline__ static float zero() { return 0.0f; }
    __device__ __forceinline__ static float quiet_nan() {
        return __uint_as_float(0x7FC00000u);
    }
};

template <>
struct Arith<double> {
    typedef double value_type;

    __device__ __forceinline__ static double add(double a, double b) {
        return __dadd_rn(a, b);
    }
    __device__ __forceinline__ static double mul(double a, double b) {
        return __dmul_rn(a, b);
    }
    __device__ __forceinline__ static bool is_nan(double a) {
        const uint64_t u = (uint64_t)__double_as_longlong(a);
        return (u & 0x7FF0000000000000ull) == 0x7FF0000000000000ull &&
               (u & 0x000FFFFFFFFFFFFFull) != 0ull;
    }
    __device__ __forceinline__ static double pos_inf() {
        return __longlong_as_double((long long)0x7FF0000000000000ull);
    }
    __device__ __forceinline__ static double neg_inf() {
        return __longlong_as_double((long long)0xFFF0000000000000ull);
    }
    __device__ __forceinline__ static double zero() { return 0.0; }
    __device__ __forceinline__ static double quiet_nan() {
        return __longlong_as_double((long long)0x7FF8000000000000ull);
    }
};

// ───────────────────────────────────────────────────────────────────────────
// Operator identity, element transform, and binary combine.
//
// NaN POLICY — deliberate and load-bearing:
//
//   SUM / SUMSQ:  IEEE addition already propagates NaN, so a single NaN force
//                 component makes the whole reduced value NaN.  Correct: the
//                 governor MUST see that, and gear 3 is the NaN sentinel.
//
//   MIN / MAX:    CUDA's fminf/fmaxf are NaN-SUPPRESSING — fminf(a, NaN)
//                 returns a.  Using them here would HIDE a NaN force from the
//                 physical-stability governor, which is precisely the class of
//                 event the governor exists to catch.  So min/max below are
//                 explicitly NaN-PROPAGATING.  This is a deviation from
//                 IEEE-754 minNum/maxNum and it is intentional.
//
//   Signed zero:  `a < b ? a : b` returns b when a == b, so min(-0.0, +0.0)
//                 yields +0.0.  Deterministic, and signed zero carries no
//                 physical meaning for any governor input.  Documented, not
//                 special-cased.
// ───────────────────────────────────────────────────────────────────────────

template <typename T, int OP>
struct Op;

template <typename T>
struct Op<T, PRISM_DR_OP_SUM> {
    __device__ __forceinline__ static T identity() { return Arith<T>::zero(); }
    __device__ __forceinline__ static T transform(T x) { return x; }
    __device__ __forceinline__ static T combine(T a, T b) {
        return Arith<T>::add(a, b);
    }
};

template <typename T>
struct Op<T, PRISM_DR_OP_SUMSQ> {
    __device__ __forceinline__ static T identity() { return Arith<T>::zero(); }
    __device__ __forceinline__ static T transform(T x) {
        return Arith<T>::mul(x, x);
    }
    __device__ __forceinline__ static T combine(T a, T b) {
        return Arith<T>::add(a, b);
    }
};

template <typename T>
struct Op<T, PRISM_DR_OP_MIN> {
    __device__ __forceinline__ static T identity() { return Arith<T>::pos_inf(); }
    __device__ __forceinline__ static T transform(T x) { return x; }
    __device__ __forceinline__ static T combine(T a, T b) {
        if (Arith<T>::is_nan(a) || Arith<T>::is_nan(b)) {
            return Arith<T>::quiet_nan();
        }
        return (a < b) ? a : b;
    }
};

template <typename T>
struct Op<T, PRISM_DR_OP_MAX> {
    __device__ __forceinline__ static T identity() { return Arith<T>::neg_inf(); }
    __device__ __forceinline__ static T transform(T x) { return x; }
    __device__ __forceinline__ static T combine(T a, T b) {
        if (Arith<T>::is_nan(a) || Arith<T>::is_nan(b)) {
            return Arith<T>::quiet_nan();
        }
        return (a > b) ? a : b;
    }
};

// ───────────────────────────────────────────────────────────────────────────
// PHASE 1 — fixed cyclic partition, strictly sequential per-partition fold.
//
// partial[j] = FOLD_{k>=0, increasing}  in[j + k*P]
//
// The partition is CYCLIC (stride P) rather than contiguous-block for one
// reason: at any instruction the 32 lanes of a warp touch 32 CONSECUTIVE
// addresses, so the load coalesces into a single transaction.  A contiguous
// partition would have each lane walking its own far-apart run, costing ~32x
// the memory transactions for identical determinism.  The ordering guarantee
// is unaffected — thread j always folds the same elements in the same order
// either way.
// ───────────────────────────────────────────────────────────────────────────

template <typename T, int OP>
__global__ void phase1_kernel(const T* __restrict__ in,
                              uint32_t n,
                              T* __restrict__ partial)
{
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= PRISM_DR_PARTITIONS) return;

    T acc = Op<T, OP>::identity();
    for (uint32_t i = j; i < n; i += PRISM_DR_PARTITIONS) {
        acc = Op<T, OP>::combine(acc, Op<T, OP>::transform(in[i]));
    }
    partial[j] = acc;
}

// float4 xyz sum-of-squares.  Per-vector fold order is x, then y, then z; the
// .w lane is never read into the accumulator.
__global__ void phase1_f4xyz_sumsq_kernel(const float4* __restrict__ in,
                                          uint32_t n_vec,
                                          float* __restrict__ partial)
{
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= PRISM_DR_PARTITIONS) return;

    float acc = 0.0f;
    for (uint32_t i = j; i < n_vec; i += PRISM_DR_PARTITIONS) {
        const float4 q = in[i];
        acc = __fadd_rn(acc, __fmul_rn(q.x, q.x));
        acc = __fadd_rn(acc, __fmul_rn(q.y, q.y));
        acc = __fadd_rn(acc, __fmul_rn(q.z, q.z));
    }
    partial[j] = acc;
}

// ───────────────────────────────────────────────────────────────────────────
// PHASE 2 — fixed-shape balanced binary tree over the P partials.
//
// P is a compile-time power of two, so every level is full and the tree shape
// is a constant.  Within a level all writes target distinct indices, so which
// thread performs which position is irrelevant to the value — only the LEVEL
// STRUCTURE matters, and that is fixed.
//
// The whole tree runs in ONE block so the level barrier is __syncthreads().
// A multi-block tree would need either a grid-wide barrier (cooperative
// launch — incompatible with the captured-graph node model) or a second
// kernel per level.  P = 2048 partials is a trivial amount of work for one
// block; the parallelism that matters is in phase 1.
// ───────────────────────────────────────────────────────────────────────────

template <typename T, int OP>
__global__ void phase2_kernel(const T* __restrict__ partial,
                              T* __restrict__ out)
{
    __shared__ T s[PRISM_DR_PARTITIONS];

    const unsigned int t = threadIdx.x;

    for (unsigned int i = t; i < PRISM_DR_PARTITIONS; i += blockDim.x) {
        s[i] = partial[i];
    }
    __syncthreads();

    for (unsigned int stride = PRISM_DR_PARTITIONS >> 1; stride >= 1u; stride >>= 1) {
        for (unsigned int i = t; i < stride; i += blockDim.x) {
            s[i] = Op<T, OP>::combine(s[i], s[i + stride]);
        }
        __syncthreads();
    }

    if (t == 0u) {
        *out = s[0];
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Sentinel + quantization kernels (single-thread, stream-ordered).
// ───────────────────────────────────────────────────────────────────────────

__global__ void write_sentinel_f32_kernel(float* __restrict__ out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (out == nullptr) return;
    *out = __uint_as_float(PRISM_DR_SENTINEL_NOT_MEASURED_F32_BITS);
}

__global__ void write_sentinel_f64_kernel(double* __restrict__ out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (out == nullptr) return;
    *out = __longlong_as_double((long long)PRISM_DR_SENTINEL_NOT_MEASURED_F64_BITS);
}

__global__ void quantize_inplace_f32_kernel(float* __restrict__ value,
                                            unsigned int mantissa_bits_kept)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (value == nullptr) return;
    *value = prism_dr_quantize_f32(*value, mantissa_bits_kept);
}

__global__ void quantize_inplace_f64_kernel(double* __restrict__ value,
                                            unsigned int mantissa_bits_kept)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (value == nullptr) return;
    *value = prism_dr_quantize_f64(*value, mantissa_bits_kept);
}

// ───────────────────────────────────────────────────────────────────────────
// DIAGNOSTIC counter-example.  NOT a decision path.  See header.
// ───────────────────────────────────────────────────────────────────────────

__global__ void DIAGNOSTIC_atomic_sum_f32_kernel(const float* __restrict__ in,
                                                 uint32_t n,
                                                 float* __restrict__ out)
{
    const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x & 31u;

    float local = 0.0f;
    if (tid < n) local = in[tid];

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xFFFFFFFFu, local, offset);
    }
    if (lane == 0u) {
        atomicAdd(out, local);  // arrival-order dependent — the whole point
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Dispatch helper: op code -> template instantiation.
// ───────────────────────────────────────────────────────────────────────────

template <typename T>
static int launch_reduce(const T* d_in,
                         uint32_t n,
                         int op,
                         T* d_partial,
                         T* d_out,
                         cudaStream_t s)
{
    dim3 grid1(PRISM_DR_PHASE1_GRID);
    dim3 blk1(PRISM_DR_PHASE1_BLOCK);
    dim3 grid2(1);
    dim3 blk2(PRISM_DR_PHASE2_BLOCK);

    switch (op) {
        case PRISM_DR_OP_SUM:
            phase1_kernel<T, PRISM_DR_OP_SUM><<<grid1, blk1, 0, s>>>(d_in, n, d_partial);
            phase2_kernel<T, PRISM_DR_OP_SUM><<<grid2, blk2, 0, s>>>(d_partial, d_out);
            break;
        case PRISM_DR_OP_SUMSQ:
            phase1_kernel<T, PRISM_DR_OP_SUMSQ><<<grid1, blk1, 0, s>>>(d_in, n, d_partial);
            phase2_kernel<T, PRISM_DR_OP_SUMSQ><<<grid2, blk2, 0, s>>>(d_partial, d_out);
            break;
        case PRISM_DR_OP_MIN:
            phase1_kernel<T, PRISM_DR_OP_MIN><<<grid1, blk1, 0, s>>>(d_in, n, d_partial);
            phase2_kernel<T, PRISM_DR_OP_MIN><<<grid2, blk2, 0, s>>>(d_partial, d_out);
            break;
        case PRISM_DR_OP_MAX:
            phase1_kernel<T, PRISM_DR_OP_MAX><<<grid1, blk1, 0, s>>>(d_in, n, d_partial);
            phase2_kernel<T, PRISM_DR_OP_MAX><<<grid2, blk2, 0, s>>>(d_partial, d_out);
            break;
        default:
            return PRISM_DR_RC_BAD_OP;
    }
    return (int)cudaGetLastError();
}

}  // namespace prism_dr

// ═══════════════════════════════════════════════════════════════════════════
// extern "C" surface
// ═══════════════════════════════════════════════════════════════════════════

extern "C" size_t prism_dr_scratch_bytes_f32(void)
{
    return (size_t)PRISM_DR_PARTITIONS * sizeof(float);
}

extern "C" size_t prism_dr_scratch_bytes_f64(void)
{
    return (size_t)PRISM_DR_PARTITIONS * sizeof(double);
}

extern "C" uint32_t prism_dr_partitions(void)
{
    return PRISM_DR_PARTITIONS;
}

extern "C" uint32_t prism_dr_schema_version(void)
{
    return PRISM_DR_SCHEMA_VERSION;
}

extern "C" int prism_dr_write_not_measured_f32(float* d_out, void* stream)
{
    if (d_out == nullptr) return PRISM_DR_RC_NULL_ARG;
    prism_dr::write_sentinel_f32_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(d_out);
    return (int)cudaGetLastError();
}

extern "C" int prism_dr_write_not_measured_f64(double* d_out, void* stream)
{
    if (d_out == nullptr) return PRISM_DR_RC_NULL_ARG;
    prism_dr::write_sentinel_f64_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(d_out);
    return (int)cudaGetLastError();
}

extern "C" int prism_dr_reduce_f32(const float* d_in,
                                   uint32_t     n,
                                   int          op,
                                   void*        d_scratch,
                                   size_t       scratch_bytes,
                                   float*       d_out,
                                   void*        stream)
{
    // PRECONDITION ORDER IS LOAD-BEARING.
    //
    // `n == 0` must be classified BEFORE the input/scratch null checks. A
    // zero-length device allocation legitimately has a null base pointer
    // (cudarc returns one), and with n == 0 neither `d_in` nor `d_scratch` is
    // ever dereferenced — so reporting NULL_ARG there would mislabel a
    // perfectly normal "nothing to measure" step as a programming error, and
    // the caller would skip the sentinel write, leaving the decision slot
    // holding the PREVIOUS step's value. Stale data that looks like a fresh
    // measurement is exactly the failure this module exists to prevent.
    //
    // `d_out` is still required up front: without it there is nowhere to put
    // the sentinel, and silently returning EMPTY_INPUT with no write would
    // reintroduce the same ambiguity.
    if (d_out == nullptr) {
        return PRISM_DR_RC_NULL_ARG;
    }
    if (op < 0 || op >= PRISM_DR_OP_COUNT) {
        return PRISM_DR_RC_BAD_OP;
    }

    cudaStream_t s = (cudaStream_t)stream;

    if (n == 0u) {
        // Explicit NOT-MEASURED sentinel; never the operator identity.
        prism_dr::write_sentinel_f32_kernel<<<1, 1, 0, s>>>(d_out);
        const int rc = (int)cudaGetLastError();
        return (rc != 0) ? rc : PRISM_DR_RC_EMPTY_INPUT;
    }

    if (d_in == nullptr || d_scratch == nullptr) {
        return PRISM_DR_RC_NULL_ARG;
    }
    if (scratch_bytes < prism_dr_scratch_bytes_f32()) {
        return PRISM_DR_RC_SCRATCH_TOO_SMALL;
    }

    return prism_dr::launch_reduce<float>(
        d_in, n, op, (float*)d_scratch, d_out, s);
}

extern "C" int prism_dr_reduce_f64(const double* d_in,
                                   uint32_t      n,
                                   int           op,
                                   void*         d_scratch,
                                   size_t        scratch_bytes,
                                   double*       d_out,
                                   void*         stream)
{
    // Same precondition ordering rationale as prism_dr_reduce_f32.
    if (d_out == nullptr) {
        return PRISM_DR_RC_NULL_ARG;
    }
    if (op < 0 || op >= PRISM_DR_OP_COUNT) {
        return PRISM_DR_RC_BAD_OP;
    }

    cudaStream_t s = (cudaStream_t)stream;

    if (n == 0u) {
        prism_dr::write_sentinel_f64_kernel<<<1, 1, 0, s>>>(d_out);
        const int rc = (int)cudaGetLastError();
        return (rc != 0) ? rc : PRISM_DR_RC_EMPTY_INPUT;
    }

    if (d_in == nullptr || d_scratch == nullptr) {
        return PRISM_DR_RC_NULL_ARG;
    }
    if (scratch_bytes < prism_dr_scratch_bytes_f64()) {
        return PRISM_DR_RC_SCRATCH_TOO_SMALL;
    }

    return prism_dr::launch_reduce<double>(
        d_in, n, op, (double*)d_scratch, d_out, s);
}

extern "C" int prism_dr_reduce_f4_xyz_sumsq(const void* d_in_float4,
                                            uint32_t    n_vec,
                                            void*       d_scratch,
                                            size_t      scratch_bytes,
                                            float*      d_out,
                                            void*       stream)
{
    // Same precondition ordering rationale as prism_dr_reduce_f32.
    if (d_out == nullptr) {
        return PRISM_DR_RC_NULL_ARG;
    }

    cudaStream_t s = (cudaStream_t)stream;

    if (n_vec == 0u) {
        prism_dr::write_sentinel_f32_kernel<<<1, 1, 0, s>>>(d_out);
        const int rc = (int)cudaGetLastError();
        return (rc != 0) ? rc : PRISM_DR_RC_EMPTY_INPUT;
    }

    if (d_in_float4 == nullptr || d_scratch == nullptr) {
        return PRISM_DR_RC_NULL_ARG;
    }
    if (scratch_bytes < prism_dr_scratch_bytes_f32()) {
        return PRISM_DR_RC_SCRATCH_TOO_SMALL;
    }

    dim3 grid1(PRISM_DR_PHASE1_GRID);
    dim3 blk1(PRISM_DR_PHASE1_BLOCK);
    prism_dr::phase1_f4xyz_sumsq_kernel<<<grid1, blk1, 0, s>>>(
        (const float4*)d_in_float4, n_vec, (float*)d_scratch);

    dim3 grid2(1);
    dim3 blk2(PRISM_DR_PHASE2_BLOCK);
    prism_dr::phase2_kernel<float, PRISM_DR_OP_SUM><<<grid2, blk2, 0, s>>>(
        (const float*)d_scratch, d_out);

    return (int)cudaGetLastError();
}

extern "C" int prism_dr_quantize_inplace_f32(float*   d_value,
                                             uint32_t mantissa_bits_kept,
                                             void*    stream)
{
    if (d_value == nullptr) return PRISM_DR_RC_NULL_ARG;
    prism_dr::quantize_inplace_f32_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
        d_value, mantissa_bits_kept);
    return (int)cudaGetLastError();
}

extern "C" int prism_dr_quantize_inplace_f64(double*  d_value,
                                             uint32_t mantissa_bits_kept,
                                             void*    stream)
{
    if (d_value == nullptr) return PRISM_DR_RC_NULL_ARG;
    prism_dr::quantize_inplace_f64_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(
        d_value, mantissa_bits_kept);
    return (int)cudaGetLastError();
}

extern "C" int prism_dr_DIAGNOSTIC_nondeterministic_atomic_sum_f32(
    const float* d_in,
    uint32_t     n,
    float*       d_out,
    void*        stream)
{
    if (d_in == nullptr || d_out == nullptr) return PRISM_DR_RC_NULL_ARG;
    if (n == 0u) return PRISM_DR_RC_EMPTY_INPUT;

    const uint32_t block = 256u;
    const uint32_t grid  = (n + block - 1u) / block;
    prism_dr::DIAGNOSTIC_atomic_sum_f32_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        d_in, n, d_out);
    return (int)cudaGetLastError();
}
