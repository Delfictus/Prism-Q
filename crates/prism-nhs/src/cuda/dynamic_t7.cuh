// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / Dynamic T7 — substrate-aware noise-floor calibration
//
// Operator-authorised Wave 3 / Path B (2026-05-02): replace the locked
// 4LPK T7 priors with values derived natively from the running substrate's
// thermal-equilibrium Δ_AB distribution.  Three captured kernels run inside
// the V2 pipeline's CUgraph downstream of the Adjudicator step:
//
//   1. Capture: 1 thread; reads adj->current_divergence, writes to
//      acc[atomicAdd(idx, 1)] with saturation at PRISM_DYNT7_N_SAMPLES.
//      No-op once the window is full.
//   2. Reduce:  256-thread block; computes mean + stddev across acc[0..n].
//      Block-level shared-memory reduction.
//   3. Apply:   1 thread; if (n >= PRISM_DYNT7_N_MIN), writes mean →
//      adj->noise_floor_mu[0] and stddev → adj->noise_floor_sigma[0].
//      Adjudicator threshold = μ[0] + 3σ[0] adapts on the next launch.
//
// All three are stream-ordered after the Adjudicator step kernel — by
// the time Capture runs, current_divergence is fully written and visible
// (the Adjudicator's __threadfence() guarantees L2 visibility before
// adjudication_code is stamped, and the captured-graph dependency chain
// orders Capture after Adjudicator).
//
// Buffers are pre-allocated in the F2 pool and pointer-stable for the
// pipeline's lifetime; the captured graph bakes their addresses in.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// Number of cold-equilibrium Δ_AB samples accumulated before the noise
// floor is considered "calibrated".  500 is enough to estimate μ + σ
// to ≈4-5% precision under stationarity.  Memory footprint: 500 × 4 B =
// 2 KB per pipeline.
#define PRISM_DYNT7_N_SAMPLES 500u

// Minimum samples before Apply writes to the FFI struct.  Below this
// threshold the kernel keeps the locked 4LPK priors (or whatever the
// orchestrator set via apply_t7_calibration / set_noise_floor_constants
// at build time).
#define PRISM_DYNT7_N_MIN     100u

#ifdef __cplusplus
extern "C" {
#endif

/// Single-launch C-ABI entry: enqueues all three kernels (capture →
/// reduce → apply) on `stream` in order.  Returns cudaError_t cast to int.
///
/// Pointer contract:
///   - `adj`        — *const InterferometricAdjudicatorFfi (read for capture,
///                    mutated by apply at offsets 0..4 and 24..28).
///   - `acc_dev`    — F2-pooled f32[PRISM_DYNT7_N_SAMPLES] device buffer.
///   - `idx_dev`    — F2-pooled u32 (single counter, atomic write target).
///   - `stats_dev`  — F2-pooled f32[2] (mean, stddev outputs).
///   - `stream`     — captured-graph md_stream.
int prism_dynamic_t7_launch(
    const void* adj,
    void*       acc_dev,
    void*       idx_dev,
    void*       stats_dev,
    void*       stream
);

#ifdef __cplusplus
}
#endif
