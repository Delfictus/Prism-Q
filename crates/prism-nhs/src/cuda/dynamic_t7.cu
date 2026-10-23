// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / Dynamic T7 — substrate-aware noise-floor calibration (impl)
// ═══════════════════════════════════════════════════════════════════════════
//
// M1.2.20.C-C / T19 — F64 Calibration Epoch refactor.  Pre-Phase-3 this
// kernel triplet wrote samples into a 500-element f32 buffer and reduced
// after PRISM_DYNT7_N_MIN with f32 math.  Operator's "Native Statistical
// Grounding" mandate (2026-05-02) replaces that with two f64
// accumulators (sum_kl, sum_sq_kl) plus a u32 count, computed in a
// single-thread fixed order during the 100-frame cold-hold burn-in.
// At count == N_MIN a
// single-thread reduce kernel computes:
//
//   μ        = sum_kl    / N
//   σ²       = sum_sq_kl / N − μ²
//   σ        = sqrt(max(σ², 0))
//
// then writes (μ, σ) to adj.noise_floor_mu[0] / sigma[0] (cast f64→f32
// at the boundary).  If σ² <= 0 the kernel sets `adj.lqi_flags |=
// LQI_T7_VARIANCE_ZERO` (Bit-31), and the Adjudicator step kernel
// thereafter forces VIOLATION (operator's Lineage Protection — a
// degenerate cold-hold floor cannot be trusted to gate anything).
// ═══════════════════════════════════════════════════════════════════════════

#include "dynamic_t7.cuh"
#include "adjudicator.cuh"  // pulls in InterferometricAdjudicatorFfi
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>
#include <cstdio>  // Amendment 3.10 — kernel-level printf triage

// ─── M1.2.20.C-C / T19 — CalibrationStateF64 (F2-pool buffer) ─────────────
//
// Layout: 8 + 8 + 4 + 4 = 24 bytes; the host allocates 32 bytes for
// natural alignment.  Pre-capture cuMemsetD8 zero-init guarantees the
// first kernel launch sees count = 0.

struct __align__(8) CalibrationStateF64 {
    double   sum_kl;       // 0..8    ordered f64 accumulator
    double   sum_sq_kl;    // 8..16   ordered f64 accumulator
    uint32_t count;        // 16..20  sample count — frozen at N_MIN
    uint32_t applied;      // 20..24  set to 1 by reduce kernel; gates re-apply
};

static_assert(sizeof(CalibrationStateF64) == 24, "CalibrationStateF64 must be 24 B");

// ─── Accumulate kernel ────────────────────────────────────────────────────
//
// Single-thread launch.  Reads adj.current_divergence (the captured-graph
// per-frame KL-divergence scalar set by the Adjudicator step kernel),
// accumulates it into sum_kl + (kl² into sum_sq_kl), increments count.
// Frozen once count >= N_MIN — subsequent launches no-op.

extern "C"
__global__ void prism_dynamic_t7_capture_kernel(
    const InterferometricAdjudicatorFfi* __restrict__ adj,
    CalibrationStateF64* __restrict__    state
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Freeze after burn-in — the accumulators stay frozen at their
    // 100-sample values for the rest of the campaign.
    const uint32_t n_now = state->count;
    if (n_now >= PRISM_DYNT7_N_MIN) return;

    // adj->current_divergence is f32 from the kernel side; cast to f64
    // for accumulation precision (operator §1 — "do not compromise on
    // the f64 precision for the calibration accumulator").
    const double kl = static_cast<double>(adj->current_divergence);

    // Single-thread launch, so direct ordered accumulation is both
    // deterministic and equivalent to the former atomics.
    state->sum_kl    += kl;
    state->sum_sq_kl += kl * kl;
    state->count      = n_now + 1u;
    __threadfence();
}

// ─── Reduce + Apply kernel ────────────────────────────────────────────────
//
// Single-thread.  Gated on count >= N_MIN AND applied == 0 (idempotent —
// fires once per campaign, then the `applied` interlock prevents
// re-entry on subsequent captured-graph replays).  Computes μ and σ
// with f64 precision, casts to f32 at the FFI boundary, sets
// `lqi_flags` Bit-31 if σ² <= 0 (degenerate floor — Lineage Protection).

extern "C"
__global__ void prism_dynamic_t7_reduce_kernel(
    CalibrationStateF64* __restrict__ state,
    InterferometricAdjudicatorFfi* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t n        = state->count;
    if (n < PRISM_DYNT7_N_MIN) return;

    // Idempotency interlock. Re-firing this kernel is a no-op because
    // the FFI fields are written deterministically from the accumulators.
    const uint32_t already = state->applied;
    if (already != 0u) return;

    const double inv_n = 1.0 / static_cast<double>(n);
    const double mu    = state->sum_kl    * inv_n;
    const double m2_n  = state->sum_sq_kl * inv_n;
    const double var   = m2_n - mu * mu;

    // Operator §1 LQI Bit-31 Quarantine — variance reduced to zero
    // means the cold-hold KL stream was perfectly stationary
    // (numerically degenerate).  We cannot threshold against a zero
    // floor; flag the run for Lineage Protection.
    if (!(var > 0.0)) {
        // Bit-31 of adjudication_reason_flags = LQI_T7_VARIANCE_ZERO
        // (Lineage Protection — operator §1, retained from M1.2.20.C-C).
        adj->adjudication_reason_flags |= 0x80000000u;
        __threadfence();
        // Still write a sane fallback so subsequent captured-graph
        // replays don't read uninitialised noise_floor values.
        adj->noise_floor_mu[0]    = static_cast<float>(mu);
        adj->noise_floor_sigma[0] = 1.0e-3f;  // bootstrap σ
        // Mark applied so we don't re-enter.
        state->applied = 1u;
        return;
    }

    const double sigma = sqrt(var);
    adj->noise_floor_mu[0]    = static_cast<float>(mu);
    adj->noise_floor_sigma[0] = static_cast<float>(sigma);
    __threadfence();

    state->applied = 1u;
}

// ─── Host launcher: enqueues both kernels in order ───────────────────────
//
// Captured-graph dependency: capture → reduce.  Both retire on the
// adjudicator's stream so subsequent reads of adj.noise_floor_mu/sigma
// see the post-reduce values.

extern "C"
int prism_dynamic_t7_launch(
    const void* adj,
    void*       state_dev,
    void*       /* idx_dev_unused — kept for ABI compat */,
    void*       /* stats_dev_unused — kept for ABI compat */,
    void*       stream
) {
    if (adj == nullptr || state_dev == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const InterferometricAdjudicatorFfi* adj_ptr =
        static_cast<const InterferometricAdjudicatorFfi*>(adj);
    CalibrationStateF64* state_ptr =
        static_cast<CalibrationStateF64*>(state_dev);

    // Capture: ordered f64 accumulation in a single-thread kernel.
    prism_dynamic_t7_capture_kernel<<<1, 1, 0, s>>>(adj_ptr, state_ptr);

    // Reduce + Apply: gated on count >= N_MIN, idempotent via `applied`.
    prism_dynamic_t7_reduce_kernel<<<1, 1, 0, s>>>(
        state_ptr,
        const_cast<InterferometricAdjudicatorFfi*>(adj_ptr)
    );

    return static_cast<int>(cudaGetLastError());
}
