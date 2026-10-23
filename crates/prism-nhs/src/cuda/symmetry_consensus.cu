// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / G28 — SISR Symmetry Consensus kernel (impl)
// ═══════════════════════════════════════════════════════════════════════════
//
// Operator directive (Amendment 3.4):
//   - C2 symmetry of the 7C8R dimer centered at the origin: the canonical
//     reflection is (x,y,z) → (-x,-y,z) (180° rotation about Z).  We
//     parameterise R/t so non-Z dyads (or BIOMT-driven dimers like 7C8R's
//     own asymmetric biological assembly) can swap in a different transform
//     without recompile.
//   - Squared-distance comparison: ‖Δ‖² < ε² avoids sqrtf latency.
//   - n_clusters ≤ 64 (one bit per cluster in the u64 mask). Brute-force
//     O(N²) outperforms LBVH traversal at this scale (Blackwell L1 hits
//     vs. tree-walk cache misses).  LBVH-restricted search marked as future
//     optimisation for n > 256.
//
// Capture-window contract:
//   - Caller invokes prism_sisr_launch INSIDE cuStreamBeginCapture on
//     md_stream, AFTER prism_so3_project_run, BEFORE
//     prism_interferometric_adjudicator_step.
//   - The kernel records as a single launch node in the in-progress CUgraph.
//   - The host pre-allocates the u64 mask buffer in the F2 pool; this kernel
//     is responsible for zeroing it before each frame's bit-OR phase.
// ═══════════════════════════════════════════════════════════════════════════

#include "symmetry_consensus.cuh"
#include "so3_project.cuh"  // ContactShellTile
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>  // Amendment 3.10 — kernel-level printf triage

using prism_nhs::so3_project::ContactShellTile;
using prism_nhs::symmetry_consensus::SISR_MAX_CLUSTERS;

// ─── Constant memory: dyad-axis transform ───────────────────────────────────
// 9 floats (row-major 3×3 rotation) + 3 floats (translation) = 48 bytes.
__constant__ static float c_dyad_R[9] = {
    -1.0f, 0.0f, 0.0f,
     0.0f,-1.0f, 0.0f,
     0.0f, 0.0f, 1.0f,
};
__constant__ static float c_dyad_t[3] = { 0.0f, 0.0f, 0.0f };

// ─── Init kernel: copies R/t into __constant__ memory ──────────────────────
// Trivial 1-thread kernel; called via prism_sisr_init_dyad() outside capture.
extern "C"
__global__ void prism_sisr_init_dyad_kernel(
    const float* __restrict__ R_in,   // 9 floats, row-major
    const float* __restrict__ t_in    // 3 floats
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // Note: __constant__ writes from device kernels are not supported;
    // we use cudaMemcpyToSymbol from the host instead. This kernel is
    // retained for symmetry with other prism_*_init kernels but the
    // actual write happens in prism_sisr_init_dyad host wrapper.
}

// ─── SISR kernel: per-cluster bilateral-truth gate ─────────────────────────
//
// Thread layout:
//   Thread 0 of block 0 zeros *out_force_prune_mask before any threads
//   compute reflections (single-block launch with __syncthreads()).
//   Each thread i (i < n_clusters) handles one cluster centroid.
//
// One block, blockDim.x = SISR_MAX_CLUSTERS = 64.  This matches the mask bit
// width and lets us use a single __syncthreads() barrier between the zero-
// init and the bit-OR phase.
extern "C"
__global__ void prism_sisr_kernel(
    const ContactShellTile* __restrict__ tiles,
    const uint32_t* __restrict__         n_clusters_dev,        // RECT-3.4.1
    unsigned long long* __restrict__      out_force_prune_mask,  // u64
    float                                 epsilon_sym_squared
) {
    const uint32_t i = threadIdx.x;

    // Phase 1: thread 0 zeros the mask.
    if (i == 0) {
        *out_force_prune_mask = 0ull;
    }
    __syncthreads();

    // RECT-3.4.1 — Device-resident n_clusters veto (Amendment 3.4.4):
    // single-cluster operation has no symmetric partner, so SISR's brute-force
    // search would set bit 0 every frame and collapse the F1 SWITCH to PRUNE.
    // Reading n_clusters from a pinned device buffer keeps the captured graph
    // topology immutable while still gating the prune-mask write.  The mask
    // stays zero (Phase 1) → Adjudicator does not override → Δ_AB-based
    // decision propagates → "Burst" path becomes reachable.
    const uint32_t n_clusters = *n_clusters_dev;
    if (n_clusters < 2u) {
        __threadfence_block();
        return;
    }

    // Phase 2: clusters beyond n_clusters bail.
    if (i >= n_clusters || i >= SISR_MAX_CLUSTERS) return;

    // Read cluster i centroid (AABB midpoint).
    const ContactShellTile& tile_i = tiles[i];
    const float cx = 0.5f * (tile_i.aabb_min[0] + tile_i.aabb_max[0]);
    const float cy = 0.5f * (tile_i.aabb_min[1] + tile_i.aabb_max[1]);
    const float cz = 0.5f * (tile_i.aabb_min[2] + tile_i.aabb_max[2]);

    // Reflect across dyad: target = R · c + t.
    const float tx = c_dyad_R[0]*cx + c_dyad_R[1]*cy + c_dyad_R[2]*cz + c_dyad_t[0];
    const float ty = c_dyad_R[3]*cx + c_dyad_R[4]*cy + c_dyad_R[5]*cz + c_dyad_t[1];
    const float tz = c_dyad_R[6]*cx + c_dyad_R[7]*cy + c_dyad_R[8]*cz + c_dyad_t[2];

    // Brute-force partner search (n_clusters ≤ 64, fits in L1 in one warp pass).
    bool found_partner = false;
    for (uint32_t j = 0; j < n_clusters; ++j) {
        if (j == i) continue;
        const ContactShellTile& tile_j = tiles[j];
        const float jx = 0.5f * (tile_j.aabb_min[0] + tile_j.aabb_max[0]);
        const float jy = 0.5f * (tile_j.aabb_min[1] + tile_j.aabb_max[1]);
        const float jz = 0.5f * (tile_j.aabb_min[2] + tile_j.aabb_max[2]);
        const float dx = jx - tx;
        const float dy = jy - ty;
        const float dz = jz - tz;
        const float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < epsilon_sym_squared) { found_partner = true; break; }
    }

    // No partner ⇒ set bit i in the mask. atomicOr on u64 is supported
    // on Pascal+ (sm_60+); Blackwell sm_120 supports it natively.
    if (!found_partner) {
        atomicOr(out_force_prune_mask, 1ull << i);
    }
}

// ─── C-ABI host-side launchers ─────────────────────────────────────────────

extern "C"
int prism_sisr_init_dyad(const float* R_row_major,
                          const float* t,
                          void*        stream)
{
    if (R_row_major == nullptr || t == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t rc;
    // cudaMemcpyToSymbolAsync is the canonical path for writing __constant__
    // memory; safe outside the capture window (called once at pipeline build).
    rc = cudaMemcpyToSymbolAsync(c_dyad_R, R_row_major, 9 * sizeof(float),
                                  0, cudaMemcpyHostToDevice,
                                  static_cast<cudaStream_t>(stream));
    if (rc != cudaSuccess) return static_cast<int>(rc);
    rc = cudaMemcpyToSymbolAsync(c_dyad_t, t, 3 * sizeof(float),
                                  0, cudaMemcpyHostToDevice,
                                  static_cast<cudaStream_t>(stream));
    return static_cast<int>(rc);
}

extern "C"
int prism_sisr_launch(const void* tiles,
                       const void* n_clusters_dev,
                       void*       out_force_prune_mask,
                       float       epsilon_sym_angstrom,
                       void*       stream)
{
    if (tiles == nullptr || out_force_prune_mask == nullptr
        || n_clusters_dev == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    // RECT-3.4.1: kernel reads n_clusters from device pointer at execution.
    // No host-side cluster-count check — the captured graph node fires every
    // launch; the kernel's own early-exit handles n_clusters < 2.
    // Single block, blockDim = SISR_MAX_CLUSTERS so the __syncthreads
    // barrier between mask-zero and bit-OR is well-defined.
    prism_sisr_kernel<<<1, SISR_MAX_CLUSTERS, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<const ContactShellTile*>(tiles),
        static_cast<const uint32_t*>(n_clusters_dev),
        static_cast<unsigned long long*>(out_force_prune_mask),
        epsilon_sym_angstrom * epsilon_sym_angstrom
    );
    return static_cast<int>(cudaGetLastError());
}

// Exact module state transfer at an all-worker quiescent boundary.
// Caller owns the context, holds every producer and synchronizes before entry.
extern "C" size_t prism_checkpoint_symmetry_consensus_bytes() { return sizeof(c_dyad_R) + sizeof(c_dyad_t); }
extern "C" int prism_checkpoint_symmetry_consensus_capture(unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_symmetry_consensus_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, c_dyad_R, sizeof(c_dyad_R), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(c_dyad_R);
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, c_dyad_t, sizeof(c_dyad_t), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(c_dyad_t);
    return (int)cudaStreamSynchronize(stream);
}
extern "C" int prism_checkpoint_symmetry_consensus_restore(const unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_symmetry_consensus_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyToSymbolAsync(c_dyad_R, bytes + offset, sizeof(c_dyad_R), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(c_dyad_R);
    rc = cudaMemcpyToSymbolAsync(c_dyad_t, bytes + offset, sizeof(c_dyad_t), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(c_dyad_t);
    return (int)cudaStreamSynchronize(stream);
}
