// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Rectification Phase 2 — Shift-Left MAR Pre-Rank Adjudicator
// ═══════════════════════════════════════════════════════════════════════
//
// Per the PRISM-4D Pipeline Rectification mandate §1 / §4.2 (operator
// directive 2026-04-29). Replaces "late-stage adjudication" — where
// expensive 2+2+2+2 manifold construction was performed for clusters
// that ultimately represent thermal background noise — with an
// EARLY-STAGE classifier that runs immediately after the LBVH AABB
// reduction but BEFORE any TIDE / KCC kernel.
//
// The adjudicator writes a 3-way `d_adjudication_code` per cluster
// that the cudaGraphConditionalNode (F1) SWITCH variant consumes:
//
//   Case 0 (PRUNE)     — cluster is diffuse/dry; route to no-op
//                        sub-graph; site is dropped from manifest.
//   Case 1 (CONSTRUCT) — cluster has burst signal; route to the
//                        Interferometric Bisimulation sub-graph.
//   Case 2 (VIOLATION) — invariant violated (NaN propagation, pool
//                        exhaustion); route to Abort sub-graph.
//
// The mandate's three Pruning Observables (POs) are:
//
//   ρ_E (Local Energy Density)  = Σ(intensity) / AABB_volume
//   τ   (Temporal Coherence)    = Shannon entropy of frame_index
//                                 within Morton-ordered sequence
//   Φ_W (Solvation Flux)        = |mean(wd_change)| within leaf set
//
// **Scope of this commit (Phase 2 minimum-viable)**: the
// `prism_pre_rank_adjudicator_kernel` consumes pre-computed ρ_E and
// Φ_W per cluster (taken as input arrays). Two helper kernels
// produce ρ_E from per-cluster intensity sums and AABB volumes:
//
//   prism_compute_aabb_volume_kernel(aabbs[N], out_volumes[N])
//   prism_compute_energy_density_kernel(intensity_sums[N], volumes[N], out_densities[N])
//
// Φ_W is consumed as upstream-computed input. The wd_change source
// is the M1 typed producer's per-spike water-density derivative
// (post-LBVH-3 wire-in). For tests we exercise the adjudicator with
// synthetic Φ_W values.
//
// Temporal coherence τ is deferred to the LBVH-3 commit alongside
// the segmented Shannon-entropy reduce — its computation needs the
// Morton-sorted spike sequence.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_PRE_RANK_CUH
#define PRISM_NHS_PRE_RANK_CUH

#include <cstdint>
#include <cuda_runtime.h>

namespace prism_nhs { namespace pre_rank {

// ─────────────────────────────────────────────────────────────────────
// Adjudication code constants — must match the cudaGraphConditionalNode
// SWITCH branch indices and the Rust-side enum mirrors.
// ─────────────────────────────────────────────────────────────────────
constexpr uint32_t ADJ_PRUNE     = 0u;  // route to no-op sub-graph
constexpr uint32_t ADJ_CONSTRUCT = 1u;  // route to Bisimulation sub-graph
constexpr uint32_t ADJ_VIOLATION = 2u;  // route to Abort sub-graph

// ─────────────────────────────────────────────────────────────────────
// AABB representation in this lane. 24 bytes, layout-pinned. The
// Rust mirror carries `#[repr(C)]` and a `mem::size_of` test.
// ─────────────────────────────────────────────────────────────────────
struct ClusterAabb {
    float min[3];
    float max[3];
};
static_assert(sizeof(ClusterAabb) == 24, "ClusterAabb FFI layout drift");

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration entry points.
//
// All single-stream, single-step kernels. No internal sync, no
// default-stream usage. The captured graph (M1.2.4-style) wraps
// the chain { compute_volume → compute_density → adjudicator } as
// a sequence of stream-ordered nodes.
// ─────────────────────────────────────────────────────────────────────

extern "C" {

/// Sentinel `0xPREE` (PRE-rank). Pinned by the Rust-side
/// `link_probe_returns_sentinel` test.
uint32_t prism_pre_rank_link_probe(void);

/// Compute per-cluster AABB volume = (max[0]-min[0]) *
/// (max[1]-min[1]) * (max[2]-min[2]). Negative or empty AABBs
/// produce 0.0 (degenerate cluster — adjudicator treats as Prune).
cudaError_t prism_compute_aabb_volumes(
    const ClusterAabb* d_aabbs,
    uint32_t           n_clusters,
    float*             d_volumes_out,
    cudaStream_t       stream
);

/// Compute per-cluster energy density:
///   density[c] = intensity_sums[c] / volumes[c]   if volumes[c] > 0
///                0.0                              otherwise
cudaError_t prism_compute_energy_density(
    const float* d_intensity_sums,
    const float* d_volumes,
    uint32_t     n_clusters,
    float*       d_densities_out,
    cudaStream_t stream
);

/// 3-way pre-rank adjudicator. One thread per cluster. Reads
/// per-cluster `density` and `flux`, classifies into 3 branches:
///
///   Case 0 (PRUNE):     density < T_rho AND flux < T_phi
///   Case 1 (CONSTRUCT): otherwise (one of the observables exceeds
///                       its threshold)
///   Case 2 (VIOLATION): density or flux is NaN/Inf — invariant
///                       violation per §2.3 SAD-PATH guard.
///
/// The output `d_adjudication_codes` is the SWITCH-node selector
/// array. The cudaGraphConditionalNode reads it via a graph-output
/// dependency edge (post-Phase-2 captured-graph wire-in).
cudaError_t prism_pre_rank_adjudicator(
    const float* d_densities,
    const float* d_fluxes,
    uint32_t     n_clusters,
    float        threshold_rho,
    float        threshold_phi,
    uint32_t*    d_adjudication_codes_out,
    cudaStream_t stream
);

}  // extern "C"

}}  // namespace prism_nhs::pre_rank

#endif  // PRISM_NHS_PRE_RANK_CUH
