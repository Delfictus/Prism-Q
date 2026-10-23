// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Rectification Phase 2 — Pre-Rank Adjudicator (CUDA impl)
// ═══════════════════════════════════════════════════════════════════════
//
// See `pre_rank.cuh` for rationale + caller contract.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════

#include "pre_rank.cuh"
#include "gpu_invariant.cuh"  // gpu_hard_assert for SAD-PATH guards

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace prism_nhs { namespace pre_rank {

// ═══════════════════════════════════════════════════════════════════════
// __global__ kernels
// ═══════════════════════════════════════════════════════════════════════

/// One thread per cluster. Computes
/// `(max[0]-min[0]) * (max[1]-min[1]) * (max[2]-min[2])`. Returns
/// 0.0 for degenerate AABBs (any axis with `max <= min`) — the
/// adjudicator treats zero-volume clusters as Prune.
__global__ void prism_compute_aabb_volumes_kernel(
    const ClusterAabb* __restrict__ d_aabbs,
    uint32_t                        n_clusters,
    float* __restrict__             d_volumes_out
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_clusters) return;

    const ClusterAabb a = d_aabbs[tid];
    const float ex = a.max[0] - a.min[0];
    const float ey = a.max[1] - a.min[1];
    const float ez = a.max[2] - a.min[2];

    if (ex <= 0.0f || ey <= 0.0f || ez <= 0.0f) {
        d_volumes_out[tid] = 0.0f;
        return;
    }
    d_volumes_out[tid] = ex * ey * ez;
}

/// One thread per cluster. Computes
/// `intensity_sums[c] / volumes[c]` with zero-volume safety. NaN
/// inputs propagate to NaN outputs — the adjudicator catches NaN
/// as VIOLATION (Case 2) per the §2.3 SAD-PATH guard.
__global__ void prism_compute_energy_density_kernel(
    const float* __restrict__ d_intensity_sums,
    const float* __restrict__ d_volumes,
    uint32_t                  n_clusters,
    float* __restrict__       d_densities_out
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_clusters) return;

    const float intensity = d_intensity_sums[tid];
    const float volume = d_volumes[tid];

    if (volume <= 0.0f) {
        d_densities_out[tid] = 0.0f;
        return;
    }
    d_densities_out[tid] = intensity / volume;
}

/// 3-way adjudicator. One thread per cluster. Implements the
/// mandate's classification ladder:
///
///   1. NaN / Inf in density or flux → VIOLATION (Case 2). The §2.3
///      SAD-PATH guard: `if (isnan(divergence_val)) {
///      *d_adjudication_code = 2; __threadfence(); }`.
///   2. Both observables below threshold → PRUNE (Case 0).
///   3. At least one above threshold → CONSTRUCT (Case 1).
__global__ void prism_pre_rank_adjudicator_kernel(
    const float* __restrict__ d_densities,
    const float* __restrict__ d_fluxes,
    uint32_t                  n_clusters,
    float                     threshold_rho,
    float                     threshold_phi,
    uint32_t* __restrict__    d_adjudication_codes_out
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_clusters) return;

    const float density = d_densities[tid];
    const float flux = d_fluxes[tid];

    // SAD-PATH guard (§2.3): non-finite observables route to
    // VIOLATION. `isnan` covers NaN; `isfinite` is FALSE on Inf
    // and NaN both — equivalently combined check.
    if (!isfinite(density) || !isfinite(flux)) {
        d_adjudication_codes_out[tid] = ADJ_VIOLATION;
        __threadfence();  // ensure the violation code is visible
                          // to the SWITCH-node consumer before
                          // any subsequent observation reads it.
        return;
    }

    // Strict inequality on BOTH observables for Prune. The
    // mandate's Logic block is "if density < T_rho && flux < T_phi
    // → Prune; else → Construct". A density EXACTLY at the
    // threshold is NOT below it (strict `<`), so it routes to
    // Construct. This matches the mandate's pseudocode (`<`).
    if (density < threshold_rho && flux < threshold_phi) {
        d_adjudication_codes_out[tid] = ADJ_PRUNE;
    } else {
        d_adjudication_codes_out[tid] = ADJ_CONSTRUCT;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// extern "C" host orchestration
// ═══════════════════════════════════════════════════════════════════════

extern "C" {

uint32_t prism_pre_rank_link_probe(void) {
    // Sentinel: 0xPREE (literally the hex letters P-R-E-E aren't
    // all valid hex; using 0x9255A4 as "PRE-RANK" mnemonic).
    return 0x9255A4u;
}

cudaError_t prism_compute_aabb_volumes(
    const ClusterAabb* d_aabbs,
    uint32_t           n_clusters,
    float*             d_volumes_out,
    cudaStream_t       stream
) {
    if (d_aabbs == nullptr || d_volumes_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (n_clusters == 0u) {
        return cudaSuccess;
    }
    constexpr uint32_t TPB = 256u;
    const uint32_t blocks = (n_clusters + TPB - 1u) / TPB;
    prism_compute_aabb_volumes_kernel<<<blocks, TPB, 0, stream>>>(
        d_aabbs, n_clusters, d_volumes_out
    );
    return cudaGetLastError();
}

cudaError_t prism_compute_energy_density(
    const float* d_intensity_sums,
    const float* d_volumes,
    uint32_t     n_clusters,
    float*       d_densities_out,
    cudaStream_t stream
) {
    if (d_intensity_sums == nullptr || d_volumes == nullptr || d_densities_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (n_clusters == 0u) {
        return cudaSuccess;
    }
    constexpr uint32_t TPB = 256u;
    const uint32_t blocks = (n_clusters + TPB - 1u) / TPB;
    prism_compute_energy_density_kernel<<<blocks, TPB, 0, stream>>>(
        d_intensity_sums, d_volumes, n_clusters, d_densities_out
    );
    return cudaGetLastError();
}

cudaError_t prism_pre_rank_adjudicator(
    const float* d_densities,
    const float* d_fluxes,
    uint32_t     n_clusters,
    float        threshold_rho,
    float        threshold_phi,
    uint32_t*    d_adjudication_codes_out,
    cudaStream_t stream
) {
    if (d_densities == nullptr || d_fluxes == nullptr || d_adjudication_codes_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (n_clusters == 0u) {
        return cudaSuccess;
    }
    constexpr uint32_t TPB = 256u;
    const uint32_t blocks = (n_clusters + TPB - 1u) / TPB;
    prism_pre_rank_adjudicator_kernel<<<blocks, TPB, 0, stream>>>(
        d_densities, d_fluxes, n_clusters,
        threshold_rho, threshold_phi,
        d_adjudication_codes_out
    );
    return cudaGetLastError();
}

}  // extern "C"

}}  // namespace prism_nhs::pre_rank
