// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D — Deterministic Reduction Substrate (Package 2)
//
// Purpose: produce GOVERNOR DECISION INPUTS whose value is a pure function of
// the input array contents, independent of thread scheduling, block
// scheduling, SM count, occupancy, stream concurrency, or atomic arrival
// order.
//
// ───────────────────────────────────────────────────────────────────────────
// WHY THIS EXISTS
// ───────────────────────────────────────────────────────────────────────────
// Float addition is NOT associative.  (a+b)+c != a+(b+c) in IEEE-754 whenever
// the operands span enough magnitude for a rounding to differ.  An
// atomicAdd-based reduction sums in ARRIVAL order, and arrival order is a
// function of the scheduler, not of the data.  Two bit-identical runs of the
// same binary on the same input therefore produce sums that differ in the low
// mantissa bits.
//
// That is normally harmless.  It is NOT harmless here.  The Coupled Temporal /
// Gearbox governor COMPARES reduced quantities (||F||, energy drift) against
// thresholds.  A one-ulp difference straddling a threshold flips the
// comparison, which flips the authorized gear, which changes the integration
// timestep, which changes the trajectory.  The provenance record then
// faithfully records two different decisions for the same input — destroying
// the reproducibility of a controller whose entire purpose is auditable
// decisions.
//
// The existing `zstr_force_stage_f4_kernel` (zstr_kernels.cu:203-209) uses
// exactly the forbidden pattern:  per-warp `atomicAdd(fn_ptr, local_sumsq)`.
// That is acceptable for an EVIDENCE/telemetry value.  It must NEVER be
// reused for a DECISION input.  This translation unit is the replacement.
//
// ───────────────────────────────────────────────────────────────────────────
// THE ORDERING GUARANTEE  (this is the contract — read it before using)
// ───────────────────────────────────────────────────────────────────────────
// Let `in[0..n)` be the input and `P = PRISM_DR_PARTITIONS` (a COMPILE-TIME
// constant, deliberately independent of n, of the launch geometry, and of the
// device).  Define
//
//   partial[j] = FOLD_{k = 0,1,2,...}  in[j + k*P]        (j in [0, P))
//
// where FOLD is a STRICTLY SEQUENTIAL left fold in increasing k, seeded with
// the operator identity.  Thread j — and only thread j — evaluates partial[j],
// and it evaluates it in that one order.  No other thread contributes to
// partial[j], so no scheduling decision can reorder it.
//
// Then reduce `partial[0..P)` with a FIXED-SHAPE binary tree:
//
//   for stride = P/2, P/4, ..., 2, 1:
//       for all i in [0, stride):   partial[i] <- partial[i] (op) partial[i+stride]
//       barrier
//
// Every write within a level targets a distinct index, so the assignment of
// positions to threads inside a level is irrelevant to the value.  The tree
// SHAPE is fixed by P alone.  P is a constant.  Therefore the shape is fixed.
//
// CONSEQUENCE — the exact guarantee we provide, stated without hedging:
//
//   G1. For a fixed input array and fixed n, the returned scalar is
//       BIT-IDENTICAL across runs, across streams, across concurrent load,
//       across devices of the same architecture, and across graph replays.
//   G2. The summation order is reproducible OFF-DEVICE.  The host reference
//       in `deterministic_reduce.rs::reference_reduce_f32/f64` replays the
//       identical order and yields the identical bits, so a recorded decision
//       can be re-derived on a CPU from a recorded input snapshot.
//   G3. The launch geometry is FIXED (P/PRISM_DR_PHASE1_BLOCK blocks for
//       phase 1, exactly one block for phase 2).  Nothing about the result
//       depends on a tuning parameter that a future edit might change without
//       noticing it changed the numbers.
//
// WHAT WE DO **NOT** GUARANTEE (stated so nobody assumes it):
//   - The result is NOT the correctly-rounded infinite-precision sum.  It is a
//     specific, reproducible, well-conditioned approximation.  Pairwise
//     summation of P partials bounds the error growth at O(log P) * eps
//     instead of the O(n) * eps of a naive serial sum, so it is also more
//     ACCURATE than a single sequential fold — but exactness is not claimed.
//   - Bit-identity does NOT hold if `n` changes, because the partition
//     changes.  n is part of the decision provenance and must be recorded.
//   - Bit-identity across DIFFERENT CUDA architectures is not claimed (the
//     `_rn` intrinsics are IEEE-defined, so it will in practice hold, but we
//     do not assert what we have not measured on more than sm_120).
//
// ───────────────────────────────────────────────────────────────────────────
// HOW THE ARITHMETIC IS PINNED
// ───────────────────────────────────────────────────────────────────────────
// Two independent belts, because either one alone is a silent-breakage risk:
//
//   B1. build.rs compiles THIS translation unit with the IEEE-strict entry
//       (`compile_to_static_archive_ieee754`), which omits `--use_fast_math`
//       and passes `-fmad=false --ftz=false --prec-div=true --prec-sqrt=true`.
//       Every other .cu in this crate keeps `--use_fast_math`; only the
//       decision path is strict.
//   B2. Every arithmetic operation below is written with the explicit
//       round-to-nearest-even intrinsics `__fadd_rn/__fmul_rn` (f32) and
//       `__dadd_rn/__dmul_rn` (f64).  These are documented to never be
//       contracted into an FMA and never reassociated.  So even if a future
//       edit to build.rs re-adds `--use_fast_math` to this file, the
//       arithmetic — and therefore the recorded decisions — do not move.
//
// ───────────────────────────────────────────────────────────────────────────
// HOT-PATH / GRAPH-CAPTURE PROPERTIES
// ───────────────────────────────────────────────────────────────────────────
//   - No cudaMalloc, no cudaFree, no cudaMemcpy, no synchronization, no host
//     callback, no printf on any launcher path.  Caller supplies scratch.
//
//     VERIFIED, not asserted: the complete set of CUDA runtime symbols the
//     compiled archive references is
//         cudaLaunchKernel, cudaGetKernel, cudaGetLastError,
//         cudaPushCallConfiguration, cudaPopCallConfiguration,
//         __cudaRegisterFatBinary{,End}, __cudaRegisterFunction,
//         __cudaUnregisterFatBinary
//     i.e. kernel launch, error query, and module registration — nothing else.
//     Re-check after any edit with:
//         nm -u libdeterministic_reduce.a | grep -oE 'cuda[A-Za-z_]+' | sort -u
//
//   - Every launcher is therefore a fixed sequence of stream-ordered kernel
//     launches with launch geometry known at capture time  ==>
//     cudaStreamCapture-safe, and a graph replay re-executes the identical
//     node sequence with identical parameters over the same pointer-stable
//     buffers, which is why guarantee G1 extends to replays.
//   - `n` is a HOST argument.  Under graph capture it is baked at capture
//     time.  If the element count can change between replays the graph must be
//     re-captured (or the caller must pad to a fixed n).  This is intentional:
//     a decision input whose element count silently changed is a different
//     measurement and must not masquerade as the same one.
// ═══════════════════════════════════════════════════════════════════════════

#ifndef PRISM_DETERMINISTIC_REDUCE_CUH
#define PRISM_DETERMINISTIC_REDUCE_CUH

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>

// ───────────────────────────────────────────────────────────────────────────
// Schema version.  Bump when the PARTITION COUNT, the tree shape, the
// intrinsic set, or the sentinel encoding changes — i.e. whenever a recorded
// decision from an older binary would no longer be reproducible by this one.
// The Rust mirror asserts on this value so a stale reader fails loudly.
// ───────────────────────────────────────────────────────────────────────────
#define PRISM_DR_SCHEMA_VERSION 1u

// ───────────────────────────────────────────────────────────────────────────
// Fixed reduction geometry.  THESE ARE PART OF THE NUMERICAL CONTRACT.
// Changing any of them changes the produced bits.  Bump
// PRISM_DR_SCHEMA_VERSION if you touch them.
// ───────────────────────────────────────────────────────────────────────────

// Number of partial accumulators.  Power of two so the phase-2 tree is
// perfectly balanced (no ragged final level, no special-casing, no
// shape dependence on n).
//   2048 f64 partials = 16 KiB of static __shared__ — comfortably inside the
//   48 KiB per-block static limit on every arch we build for, with margin.
#define PRISM_DR_PARTITIONS 2048u

// Phase-1 block size.  PARTITIONS / PHASE1_BLOCK = 8 blocks.  This is a
// SCHEDULING parameter only: the value of partial[j] does not depend on it,
// because j alone determines which elements thread j folds and in what order.
#define PRISM_DR_PHASE1_BLOCK 256u

// Phase-2 block size.  Exactly one block.  Threads cooperate over a
// grid-stride loop within each tree level, so this too is scheduling-only.
#define PRISM_DR_PHASE2_BLOCK 512u

#define PRISM_DR_PHASE1_GRID (PRISM_DR_PARTITIONS / PRISM_DR_PHASE1_BLOCK)

// ───────────────────────────────────────────────────────────────────────────
// Operators
// ───────────────────────────────────────────────────────────────────────────
#define PRISM_DR_OP_SUM   0
#define PRISM_DR_OP_SUMSQ 1
#define PRISM_DR_OP_MIN   2
#define PRISM_DR_OP_MAX   3
#define PRISM_DR_OP_COUNT 4

// ───────────────────────────────────────────────────────────────────────────
// Return codes.  Negative = this module's own contract violations.
// Positive = a raw `cudaError_t` forwarded verbatim (same convention as
// energy_monitor.cu / gearbox.cu).  Zero = success.
// ───────────────────────────────────────────────────────────────────────────
#define PRISM_DR_RC_OK                 0
#define PRISM_DR_RC_NULL_ARG          (-1)
#define PRISM_DR_RC_EMPTY_INPUT       (-2)
#define PRISM_DR_RC_BAD_OP            (-3)
#define PRISM_DR_RC_SCRATCH_TOO_SMALL (-4)

// ───────────────────────────────────────────────────────────────────────────
// NOT-MEASURED SENTINEL
//
// Invariant: "NO fabricated values as measurements.  Not-measured gets an
// explicit sentinel."  An empty input must NOT silently yield the operator
// identity (0 for sum, +inf for min) — that is a fabricated measurement that
// a threshold comparison would happily consume.
//
// So on an empty input the launcher writes a QUIET NaN WITH A DISTINCTIVE
// PAYLOAD and returns PRISM_DR_RC_EMPTY_INPUT.  The payload matters: it lets a
// downstream reader distinguish "we never measured this" from "the physics
// produced a NaN", which are completely different incidents.  A NaN arising
// from the data will essentially never carry this exact payload.
// ───────────────────────────────────────────────────────────────────────────
#define PRISM_DR_SENTINEL_NOT_MEASURED_F32_BITS 0x7FC0DEADu
#define PRISM_DR_SENTINEL_NOT_MEASURED_F64_BITS 0x7FF800000000DEADull

// ───────────────────────────────────────────────────────────────────────────
// QUANTIZATION — significand-grid rounding
//
// See the recommendation block in `deterministic_reduce.rs`.  Summary: this is
// DEFENCE IN DEPTH for decision inputs this module did not itself produce.  It
// is NOT a substitute for the deterministic reduction and it does NOT by
// itself make a threshold comparison stable (a value sitting on a quantization
// boundary still flips).  Use it in addition to, never instead of, G1.
//
// `mantissa_bits_kept` retains the top N explicit significand bits and rounds
// the remainder to nearest, ties-to-even, ON THE RETAINED GRID.  The grid is
// RELATIVE (uniform in ulp, so scale-free across the 30+ decades a force norm
// can span) rather than absolute decimal — an absolute decimal grid would be
// meaningless for a quantity whose magnitude is not known a priori.
//
// Carry out of the significand into the exponent is intentional and correct:
// rounding 1.9999 up to 2.0 crosses a binade and that is the right answer.
// NaN and Inf pass through UNCHANGED — a quantizer must never launder a NaN
// into a finite number.
// ───────────────────────────────────────────────────────────────────────────

// Source-literal defaults.  Policy-resolvable on the Rust side; these are the
// documented fallbacks and the resolved value + its origin are recorded with
// the decision (see `QuantizationPolicy` in deterministic_reduce.rs).
//   f32: keep 16 of 23 ==> relative quantum <= 2^-17  ~ 7.6e-6
//   f64: keep 34 of 52 ==> relative quantum <= 2^-35  ~ 2.9e-11
#define PRISM_DR_MANTISSA_BITS_F32 23u
#define PRISM_DR_MANTISSA_BITS_F64 52u
#define PRISM_DR_DEFAULT_QUANT_BITS_F32 16u
#define PRISM_DR_DEFAULT_QUANT_BITS_F64 34u

#ifdef __CUDACC__

// Device-only: __float_as_uint / __double_as_longlong are device intrinsics.
// The bit-exact host mirror lives in `deterministic_reduce.rs::quantize_f32`
// and the integration test asserts the two agree on every probe value.
__device__ __forceinline__ float
prism_dr_quantize_f32(float x, unsigned int mantissa_bits_kept)
{
    // NaN / Inf pass through untouched.
    const uint32_t bits = __float_as_uint(x);
    if ((bits & 0x7F800000u) == 0x7F800000u) {
        return x;
    }
    if (mantissa_bits_kept >= PRISM_DR_MANTISSA_BITS_F32) {
        return x;
    }
    const uint32_t drop = PRISM_DR_MANTISSA_BITS_F32 - mantissa_bits_kept;
    const uint32_t lsb  = 1u << drop;
    const uint32_t half = lsb >> 1;
    const uint32_t rem  = bits & (lsb - 1u);
    uint32_t kept = bits & ~(lsb - 1u);
    // Round to nearest; ties to even ON THE RETAINED GRID.
    if (rem > half || (rem == half && (kept & lsb) != 0u)) {
        kept += lsb;  // may carry into the exponent — correct.
    }
    return __uint_as_float(kept);
}

__device__ __forceinline__ double
prism_dr_quantize_f64(double x, unsigned int mantissa_bits_kept)
{
    const uint64_t bits = (uint64_t)__double_as_longlong(x);
    if ((bits & 0x7FF0000000000000ull) == 0x7FF0000000000000ull) {
        return x;
    }
    if (mantissa_bits_kept >= PRISM_DR_MANTISSA_BITS_F64) {
        return x;
    }
    const uint32_t drop = PRISM_DR_MANTISSA_BITS_F64 - mantissa_bits_kept;
    const uint64_t lsb  = 1ull << drop;
    const uint64_t half = lsb >> 1;
    const uint64_t rem  = bits & (lsb - 1ull);
    uint64_t kept = bits & ~(lsb - 1ull);
    if (rem > half || (rem == half && (kept & lsb) != 0ull)) {
        kept += lsb;
    }
    return __longlong_as_double((long long)kept);
}

#endif  // __CUDACC__

// ═══════════════════════════════════════════════════════════════════════════
// extern "C" host launchers
// ═══════════════════════════════════════════════════════════════════════════

#ifdef __cplusplus
extern "C" {
#endif

/// Scratch bytes required by the f32 / f64 reductions.  Constant — depends
/// only on PRISM_DR_PARTITIONS, never on n.  Allocate ONCE per pipeline and
/// keep the pointer stable for the campaign (graph-capture requirement).
size_t prism_dr_scratch_bytes_f32(void);
size_t prism_dr_scratch_bytes_f64(void);

/// Compile-time geometry, exported so the Rust mirror can assert agreement
/// rather than duplicating literals that could drift apart silently.
uint32_t prism_dr_partitions(void);
uint32_t prism_dr_schema_version(void);

/// Deterministic reduction over `d_in[0..n)`.
///
///   op          one of PRISM_DR_OP_*
///   d_scratch   device buffer of >= prism_dr_scratch_bytes_f32() bytes
///   d_out       device scalar receiving the result
///   stream      cudaStream_t (may be under capture)
///
/// ALIASING: `d_in`, `d_scratch`, and `d_out` must be pairwise
/// non-overlapping.  The kernels are compiled with `--restrict` and declare
/// `__restrict__` on these pointers, so an overlap is undefined behaviour
/// rather than a merely wrong answer.  The Rust wrapper enforces this by
/// construction (separate `CudaSlice` allocations); a direct C caller must
/// enforce it itself.
///
/// Returns PRISM_DR_RC_OK, a negative PRISM_DR_RC_*, or a positive
/// cudaError_t.  On PRISM_DR_RC_EMPTY_INPUT the launcher still writes the
/// NOT-MEASURED sentinel into *d_out so the slot is never ambiguous — but the
/// CALLER MUST CHECK THE RETURN CODE and must not treat the sentinel as a
/// measurement.
///
/// PRECONDITION ORDER (contractual, not incidental):
///   1. `d_out` null            -> PRISM_DR_RC_NULL_ARG
///   2. `op` out of range       -> PRISM_DR_RC_BAD_OP
///   3. `n == 0`                -> sentinel written, PRISM_DR_RC_EMPTY_INPUT
///   4. `d_in`/`d_scratch` null -> PRISM_DR_RC_NULL_ARG
///   5. scratch too small       -> PRISM_DR_RC_SCRATCH_TOO_SMALL
///
/// Step 3 precedes step 4 deliberately.  A zero-length device allocation
/// legitimately has a null base pointer, and with n == 0 neither `d_in` nor
/// `d_scratch` is dereferenced.  Reporting NULL_ARG there would mislabel a
/// normal "nothing to measure" step as a programming error AND skip the
/// sentinel write, leaving the decision slot holding the previous step's
/// value — stale data indistinguishable from a fresh measurement.
int prism_dr_reduce_f32(const float* d_in,
                        uint32_t     n,
                        int          op,
                        void*        d_scratch,
                        size_t       scratch_bytes,
                        float*       d_out,
                        void*        stream);

int prism_dr_reduce_f64(const double* d_in,
                        uint32_t      n,
                        int           op,
                        void*         d_scratch,
                        size_t        scratch_bytes,
                        double*       d_out,
                        void*         stream);

/// Deterministic  sum_i (x_i^2 + y_i^2 + z_i^2)  over a float4 array,
/// SKIPPING the .w lane.
///
/// This is the governor's force-magnitude input.  It exists as its own entry
/// because the padded-float4 force layout makes the obvious call site wrong:
/// reducing the buffer as a flat float array folds the .w padding into
/// ||F||^2.  Per-vector fold order is x, then y, then z — fixed.
int prism_dr_reduce_f4_xyz_sumsq(const void* d_in_float4,
                                 uint32_t    n_vec,
                                 void*       d_scratch,
                                 size_t      scratch_bytes,
                                 float*      d_out,
                                 void*       stream);

/// In-place significand-grid quantization of a device scalar.  <<<1,1>>>,
/// stream-ordered, capture-safe.  Intended to run immediately before the
/// governor's threshold comparison so the compared value and the RECORDED
/// value are the same object.
int prism_dr_quantize_inplace_f32(float*       d_value,
                                  uint32_t     mantissa_bits_kept,
                                  void*        stream);

int prism_dr_quantize_inplace_f64(double*      d_value,
                                  uint32_t     mantissa_bits_kept,
                                  void*        stream);

/// Write the NOT-MEASURED sentinel into a device scalar.  Exposed so callers
/// can pre-poison a decision slot before the producing node runs: a slot that
/// still reads as the sentinel afterwards proves the producer did not run,
/// rather than leaving stale data that looks like a fresh measurement.
int prism_dr_write_not_measured_f32(float* d_out, void* stream);
int prism_dr_write_not_measured_f64(double* d_out, void* stream);

/// ══ DIAGNOSTIC ONLY — NEVER FOR A DECISION INPUT ══
///
/// A warp-shuffle + atomicAdd sum, i.e. the exact nondeterministic pattern
/// this module exists to replace (see zstr_force_stage_f4_kernel).  It is
/// compiled in so the determinism test can EXHIBIT the run-to-run variance
/// side by side with the deterministic result on the same data, instead of
/// merely asserting that the deterministic path is self-consistent.
///
/// `d_out` must be zeroed by the caller before each call.
/// Guarded by name, not by #ifdef, so that any grep for decision-path callers
/// finds it immediately.
int prism_dr_DIAGNOSTIC_nondeterministic_atomic_sum_f32(const float* d_in,
                                                        uint32_t     n,
                                                        float*       d_out,
                                                        void*        stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // PRISM_DETERMINISTIC_REDUCE_CUH
