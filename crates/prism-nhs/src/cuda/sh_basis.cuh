// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / RECT-3.1 — Spherical Harmonics Y_lm Evaluator (Lmax=5)
// ═══════════════════════════════════════════════════════════════════════
//
// Per the Production Architecture mandate Phase 1 Deliverable 1.1
// (operator directive 2026-04-29). Straight-line PTX evaluation of
// 36 real spherical harmonics Y_lm for Lmax=5. Designed to feed the
// SO(3) projection kernel (Deliverable 1.2) which accumulates the
// per-spike Y_lm * intensity contributions into a_lm coefficients
// via WMMA Tensor Core fragments.
//
// **Straight-line execution contract** (mandate §4.1):
//
//   - Constant-time per call: no recursion, no per-thread branching.
//   - Every thread in a warp executes the EXACT same instruction
//     sequence regardless of (theta, phi) inputs. 100% warp efficiency
//     during harmonic evaluation.
//
// **Convention** (chosen + locked):
//
//   Y_l^m(θ, φ) = K_lm * P_l^|m|(cos θ) * angular(m, φ)
//
//     angular(0, φ)   = 1
//     angular(m>0, φ) = cos(m*φ)
//     angular(m<0, φ) = sin(|m|*φ)
//
//     K_lm (m=0)  = sqrt((2l+1) / (4π))
//     K_lm (m≠0)  = sqrt((2l+1) / (2π) * (l-|m|)! / (l+|m|)!)
//                   (the sqrt(2) factor for m≠0 is merged into K)
//
//     P_l^m(x) = associated Legendre, "physical" convention
//                (no Condon-Shortley (-1)^m phase).
//
// The CL = Σ_m |a_lm|² rotational power spectrum is invariant under
// the choice of phase convention (Condon-Shortley vs. not), so the
// G11 invariance gate passes regardless. The Rust-side CPU reference
// in `sh_basis.rs` uses the same convention bit-for-bit.
//
// **Index layout** (for both `K_LM` and the 36-float output):
//
//   Y[ l*(l+1) + m ]   for l ∈ [0, 5], m ∈ [-l, l]
//
//   l=0 → idx 0         (1 entry)
//   l=1 → idx 1, 2, 3   (3 entries)
//   l=2 → idx 4..8      (5 entries)
//   l=3 → idx 9..15     (7 entries)
//   l=4 → idx 16..24    (9 entries)
//   l=5 → idx 25..35    (11 entries)
//
// Total: 1+3+5+7+9+11 = 36.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_SH_BASIS_CUH
#define PRISM_NHS_SH_BASIS_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace prism_nhs { namespace sh_basis {

constexpr int LMAX = 5;
constexpr int N_COEFFS = (LMAX + 1) * (LMAX + 1);  // 36

// ─────────────────────────────────────────────────────────────────────
// Straight-line evaluator — 36 real Y_lm values for one (θ, φ).
//
// Caller passes a const pointer to a 36-float K_LM normalization
// table. The K_LM table is populated by `prism_sh_basis_init`
// once per process; the table's symbol lives in `sh_basis.cu`
// (`__device__ float k_lm[N_COEFFS]`) and external translation units
// either reference it via `prism_sh_get_k_lm_dev_ptr` (host-side
// FFI getter) or carry their own copy.
//
// The helper is `__device__ __forceinline__` so the entire
// 36-coefficient computation is inlined into the calling kernel —
// no function-call overhead, no register-spill risk, no
// translation-unit linkage issue.
// ─────────────────────────────────────────────────────────────────────
__device__ __forceinline__ void prism_sh_eval_lmax5(
    float theta,
    float phi,
    const float* __restrict__ k_lm,
    float Y[N_COEFFS]
) {
    // ── Precompute powers of cos(θ) and sin(θ). ────────────────────
    float ct, st;
    sincosf(theta, &st, &ct);
    const float ct2 = ct * ct;
    const float ct3 = ct2 * ct;
    const float ct4 = ct3 * ct;
    const float ct5 = ct4 * ct;
    const float st2 = st * st;
    const float st3 = st2 * st;
    const float st4 = st3 * st;
    const float st5 = st4 * st;

    // ── Associated Legendre polynomials P_l^|m|(cos θ).
    // ── Physical convention (no (-1)^m Condon-Shortley phase).
    const float P_0_0 = 1.0f;

    const float P_1_0 = ct;
    const float P_1_1 = st;

    const float P_2_0 = 0.5f * (3.0f * ct2 - 1.0f);
    const float P_2_1 = 3.0f * ct * st;
    const float P_2_2 = 3.0f * st2;

    const float P_3_0 = 0.5f * (5.0f * ct3 - 3.0f * ct);
    const float P_3_1 = 1.5f * (5.0f * ct2 - 1.0f) * st;
    const float P_3_2 = 15.0f * ct * st2;
    const float P_3_3 = 15.0f * st3;

    const float P_4_0 = 0.125f * (35.0f * ct4 - 30.0f * ct2 + 3.0f);
    const float P_4_1 = 2.5f * (7.0f * ct3 - 3.0f * ct) * st;
    const float P_4_2 = 7.5f * (7.0f * ct2 - 1.0f) * st2;
    const float P_4_3 = 105.0f * ct * st3;
    const float P_4_4 = 105.0f * st4;

    const float P_5_0 = 0.125f * (63.0f * ct5 - 70.0f * ct3 + 15.0f * ct);
    const float P_5_1 = (15.0f / 8.0f) * (21.0f * ct4 - 14.0f * ct2 + 1.0f) * st;
    const float P_5_2 = 52.5f * (3.0f * ct3 - ct) * st2;
    const float P_5_3 = 52.5f * (9.0f * ct2 - 1.0f) * st3;
    const float P_5_4 = 945.0f * ct * st4;
    const float P_5_5 = 945.0f * st5;

    // ── Angular factors via angle-addition recurrence.
    // Avoids 6 separate sincosf calls (m=1..5); replaces with 1
    // sincosf + 4 multiply-add pairs.
    float sin_phi, cos_phi;
    sincosf(phi, &sin_phi, &cos_phi);

    const float cos_2phi = cos_phi * cos_phi - sin_phi * sin_phi;
    const float sin_2phi = 2.0f * sin_phi * cos_phi;

    const float cos_3phi = cos_phi * cos_2phi - sin_phi * sin_2phi;
    const float sin_3phi = sin_phi * cos_2phi + cos_phi * sin_2phi;

    const float cos_4phi = cos_phi * cos_3phi - sin_phi * sin_3phi;
    const float sin_4phi = sin_phi * cos_3phi + cos_phi * sin_3phi;

    const float cos_5phi = cos_phi * cos_4phi - sin_phi * sin_4phi;
    const float sin_5phi = sin_phi * cos_4phi + cos_phi * sin_4phi;

    // ── Final assembly: Y_lm = K_lm * P_l^|m| * angular(m, φ).
    // 36 fused-multiply-adds; no branches.

    Y[0]  = k_lm[0]  * P_0_0;                  // (0, 0)

    Y[1]  = k_lm[1]  * P_1_1 * sin_phi;        // (1, -1)
    Y[2]  = k_lm[2]  * P_1_0;                  // (1, 0)
    Y[3]  = k_lm[3]  * P_1_1 * cos_phi;        // (1, 1)

    Y[4]  = k_lm[4]  * P_2_2 * sin_2phi;       // (2, -2)
    Y[5]  = k_lm[5]  * P_2_1 * sin_phi;        // (2, -1)
    Y[6]  = k_lm[6]  * P_2_0;                  // (2, 0)
    Y[7]  = k_lm[7]  * P_2_1 * cos_phi;        // (2, 1)
    Y[8]  = k_lm[8]  * P_2_2 * cos_2phi;       // (2, 2)

    Y[9]  = k_lm[9]  * P_3_3 * sin_3phi;       // (3, -3)
    Y[10] = k_lm[10] * P_3_2 * sin_2phi;       // (3, -2)
    Y[11] = k_lm[11] * P_3_1 * sin_phi;        // (3, -1)
    Y[12] = k_lm[12] * P_3_0;                  // (3, 0)
    Y[13] = k_lm[13] * P_3_1 * cos_phi;        // (3, 1)
    Y[14] = k_lm[14] * P_3_2 * cos_2phi;       // (3, 2)
    Y[15] = k_lm[15] * P_3_3 * cos_3phi;       // (3, 3)

    Y[16] = k_lm[16] * P_4_4 * sin_4phi;       // (4, -4)
    Y[17] = k_lm[17] * P_4_3 * sin_3phi;       // (4, -3)
    Y[18] = k_lm[18] * P_4_2 * sin_2phi;       // (4, -2)
    Y[19] = k_lm[19] * P_4_1 * sin_phi;        // (4, -1)
    Y[20] = k_lm[20] * P_4_0;                  // (4, 0)
    Y[21] = k_lm[21] * P_4_1 * cos_phi;        // (4, 1)
    Y[22] = k_lm[22] * P_4_2 * cos_2phi;       // (4, 2)
    Y[23] = k_lm[23] * P_4_3 * cos_3phi;       // (4, 3)
    Y[24] = k_lm[24] * P_4_4 * cos_4phi;       // (4, 4)

    Y[25] = k_lm[25] * P_5_5 * sin_5phi;       // (5, -5)
    Y[26] = k_lm[26] * P_5_4 * sin_4phi;       // (5, -4)
    Y[27] = k_lm[27] * P_5_3 * sin_3phi;       // (5, -3)
    Y[28] = k_lm[28] * P_5_2 * sin_2phi;       // (5, -2)
    Y[29] = k_lm[29] * P_5_1 * sin_phi;        // (5, -1)
    Y[30] = k_lm[30] * P_5_0;                  // (5, 0)
    Y[31] = k_lm[31] * P_5_1 * cos_phi;        // (5, 1)
    Y[32] = k_lm[32] * P_5_2 * cos_2phi;       // (5, 2)
    Y[33] = k_lm[33] * P_5_3 * cos_3phi;       // (5, 3)
    Y[34] = k_lm[34] * P_5_4 * cos_4phi;       // (5, 4)
    Y[35] = k_lm[35] * P_5_5 * cos_5phi;       // (5, 5)
}

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration.
// ─────────────────────────────────────────────────────────────────────

extern "C" {

/// Sentinel `0x5BAC` ("SBAC" — SH BASis Constant). Pinned by Rust-
/// side `link_probe_returns_sentinel` test.
uint32_t prism_sh_basis_link_probe(void);

/// Populate the device-side `K_LM` table. Must be called once per
/// process lifetime (per device) before any call to
/// `prism_sh_eval_run`. Subsequent calls are idempotent (safe to
/// re-invoke, but redundant).
cudaError_t prism_sh_basis_init(cudaStream_t stream);

/// Evaluate `n_points` real spherical harmonic vectors. Inputs:
/// `d_theta_phi` is planar `[N][2]` (theta_0, phi_0, theta_1,
/// phi_1, ...). Outputs: `d_Y_out` is planar `[N][36]` real Y_lm
/// values. One thread per point.
cudaError_t prism_sh_eval_run(
    const float* d_theta_phi,
    uint32_t     n_points,
    float*       d_Y_out,
    cudaStream_t stream
);

/// Retrieve a device pointer to the K_LM[36] normalization table
/// populated by `prism_sh_basis_init`. The pointer is stable for the
/// process lifetime and may be passed to downstream kernels (e.g.
/// `prism_so3_project_run`) that need to evaluate Y_lm in their own
/// translation units.
///
/// Returns `cudaErrorInvalidValue` if `out_dev_ptr` is null;
/// otherwise the result of `cudaGetSymbolAddress`.
cudaError_t prism_sh_basis_get_k_lm_dev_ptr(const float** out_dev_ptr);

}  // extern "C"

}}  // namespace prism_nhs::sh_basis

#endif  // PRISM_NHS_SH_BASIS_CUH
