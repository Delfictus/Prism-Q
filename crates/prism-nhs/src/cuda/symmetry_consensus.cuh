// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / G28 — SISR Symmetry Consensus (Spatially-Indexed Symmetric
//                  Reflection)
// ═══════════════════════════════════════════════════════════════════════════
//
// Bilateral-truth gate for homodimer targets (e.g., 7C8R Mpro).  For every
// candidate cluster A on the dimer, the kernel reflects its AABB centroid
// across the dyad axis (R, t in __constant__ memory) and checks whether ANY
// other cluster's AABB centroid sits within `epsilon_sym` of the reflected
// target.  No partner ⇒ the cluster failed bilateral consensus and a bit is
// set in the `force_prune_mask` u64.  The Adjudicator step kernel reads the
// mask and overrides its decision to PRISM_ADJ_PRUNE on a hit.
//
// Capture-window placement: AFTER prism_so3_project_run (Node B) and BEFORE
// prism_interferometric_adjudicator_step (Node C) on md_stream.  Brute-force
// O(N²) pairwise centroid distance check; n_clusters ≤ 64 (mask bit limit).
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace symmetry_consensus {

/// Maximum clusters supported per launch (1 bit per cluster in the mask).
constexpr uint32_t SISR_MAX_CLUSTERS = 64;

}} // namespace prism_nhs::symmetry_consensus

// ─── C-ABI host-side surface ────────────────────────────────────────────────
//
// Both functions return cudaError_t cast to int; 0 == cudaSuccess.

#ifdef __cplusplus
extern "C" {
#endif

/// Initialize the dyad-axis transform in __constant__ memory.
/// `R_row_major` is a 3×3 rotation/reflection matrix (9 floats, row-major);
/// `t` is a 3-vector translation (3 floats).
/// For 7C8R the BIOMT-2 transform is:
///   R = [[-0.5, 0.866025, 0], [0.866025, 0.5, 0], [0, 0, -1]]
///   t = [0, 0, 30.07767]
/// Single-thread kernel; safe to call outside capture (initialization only).
int prism_sisr_init_dyad(const float* R_row_major,
                          const float* t,
                          void* stream);

/// Launch the SISR kernel: each thread handles one cluster centroid.
/// `n_clusters_dev` is a pinned/device u32 buffer holding the live cluster
/// count (RECT-3.4.1 device-resident veto: kernel early-exits when *n < 2).
/// `out_force_prune_mask` is a single device-side u64 buffer (8 B); the
/// kernel zeros it then ORs in bits for clusters with no Chain-B partner.
/// `epsilon_sym_angstrom` is the partner-search tolerance (typical: 1.5 Å).
int prism_sisr_launch(const void* tiles,                  // *const ContactShellTile
                      const void* n_clusters_dev,          // *const u32
                      void*       out_force_prune_mask,    // *mut u64
                      float       epsilon_sym_angstrom,
                      void*       stream);

#ifdef __cplusplus
}
#endif
