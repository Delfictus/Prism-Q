// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / GhostPhaseLattice4D — static-archive wrapper (host orchestrator)
// ═══════════════════════════════════════════════════════════════════════════
//
// This translation unit is compiled by the prism-nhs build.rs via
// `compile_to_static_archive(..., "src/cuda/ghost_lattice_kernel.cu",
// "ghost_lattice_kernel", &out_dir)`. It includes the device-only kernels
// from `ghost_lattice_kernel_nvrtc.cu` (the same source NVRTC compiles when
// the static archive is unreachable) and provides the extern "C" host
// orchestrator declared in `ghost_lattice_kernel.cuh`.
//
// The wrapper:
//   * pins the GhostPhaseLatticeNode struct layout via static_assert,
//   * launches `ghost_lattice_init_parent`, `ghost_lattice_edge_kernel`, and
//     `ghost_lattice_path_compress` against the supplied stream,
//   * synchronises the stream once at the end,
//   * returns the first non-success `cudaError_t` (or 0 on full success).
// ═══════════════════════════════════════════════════════════════════════════

#include "ghost_lattice_kernel.cuh"
#include "ghost_lattice_kernel_nvrtc.cu" // pulls in the __global__ kernels

#include <cstddef>
#include <cstdint>

// ── Layout assertions (Rust ↔ CUDA byte parity) ──────────────────────────
static_assert(sizeof(GhostPhaseLatticeNodeHost) == 208,
              "GhostPhaseLatticeNodeHost size drift — must be 208 bytes "
              "(byte-pinned to Rust GhostPhaseLatticeNode in "
              "ghost_phase_lattice.rs).");
static_assert(alignof(GhostPhaseLatticeNodeHost) == 8,
              "GhostPhaseLatticeNodeHost alignment drift — must be 8 bytes.");
static_assert(offsetof(GhostPhaseLatticeNodeHost, tile_index)        ==   0, "tile_index offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, stream_id)         ==   4, "stream_id offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, site_id)           ==   8, "site_id offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, frame_idx)         ==  16, "frame_idx offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, step_idx)          ==  24, "step_idx offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, protocol_phase)    ==  32, "protocol_phase offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, step_bucket)       ==  36, "step_bucket offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, ccns_phase_bin)    ==  40, "ccns_phase_bin offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, gear_id)           ==  44, "gear_id offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, aabb_min)          ==  48, "aabb_min offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, aabb_max)          ==  60, "aabb_max offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, centroid_xyz)      ==  72, "centroid_xyz offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, kl_divergence)     ==  84, "kl_divergence offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, thermo_flux)       ==  88, "thermo_flux offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, water_density)     ==  96, "water_density offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, causal_lead_residue) == 100, "causal_lead_residue offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, so3_power_spectrum) == 104, "so3_power_spectrum offset");
static_assert(offsetof(GhostPhaseLatticeNodeHost, so3_plane_status)  == 200, "so3_plane_status offset");
static_assert(sizeof(GhostPhaseLatticeAcceptedEdgeHost) == 56,
              "accepted Ghost lattice edge size drift");
static_assert(alignof(GhostPhaseLatticeAcceptedEdgeHost) == 8,
              "accepted Ghost lattice edge alignment drift");
static_assert(offsetof(GhostPhaseLatticeAcceptedEdgeHost, from_node_index) == 0, "edge from offset");
static_assert(offsetof(GhostPhaseLatticeAcceptedEdgeHost, to_node_index) == 4, "edge to offset");
static_assert(offsetof(GhostPhaseLatticeAcceptedEdgeHost, from_step_idx) == 8, "edge from step offset");
static_assert(offsetof(GhostPhaseLatticeAcceptedEdgeHost, to_step_idx) == 16, "edge to step offset");
static_assert(offsetof(GhostPhaseLatticeAcceptedEdgeHost, temporal_delta_steps) == 24, "edge delta offset");
static_assert(offsetof(GhostPhaseLatticeAcceptedEdgeHost, total_score) == 32, "edge total score offset");
static_assert(offsetof(GhostPhaseLatticeAcceptedEdgeHost, from_protocol_phase) == 48, "edge phase offset");
static_assert(offsetof(GhostPhaseLatticeAcceptedEdgeHost, schema_version) == 52, "edge schema offset");

// Same plane-weight + score-quant constants the device kernel uses — so any
// host-side audit of the score-sum atomics matches what the kernel wrote.
static_assert(GHOST_LATTICE_W_GEO + GHOST_LATTICE_W_CAUS +
              GHOST_LATTICE_W_THER + GHOST_LATTICE_W_CHEM == 1.0f,
              "GhostPhaseLattice4D plane weights must sum to 1.0 — directive Part II.2.");

// ── Host orchestrator ────────────────────────────────────────────────────

extern "C" int prism_ghost_phase_lattice_count(
    uint64_t       nodes_dev,
    uint32_t       n_nodes,
    uint64_t       permutation_dev,
    uint64_t       cell_first_dev,
    uint64_t       cell_count_dev,
    uint32_t       n_cells,
    uint64_t       cell_table_dev,
    uint64_t       parent_dev,
    uint64_t       edge_score_sum_dev,
    uint64_t       edge_count_dev,
    uint64_t       pair_count_dev,
    uint64_t       phase_legal_count_dev,
    uint64_t       aabb_overlap_count_dev,
    float          spatial_cell_size_a,
    uint64_t       max_temporal_edge_steps,
    float          so3_threshold,
    void*          stream
) {
    if (n_nodes == 0u) {
        return 0;
    }

    cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);

    // ── Reinterpret raw u64 device pointers as typed device pointers. ──
    // cudarc gives us CUdeviceptr (a uintptr_t-sized opaque type) which we
    // surface as u64 across the FFI boundary. Casting back is well-defined
    // for any object pointer the CUDA runtime accepts.
    const unsigned char* d_nodes_bytes = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(nodes_dev));
    const unsigned int* d_permutation = reinterpret_cast<const unsigned int*>(
        static_cast<uintptr_t>(permutation_dev));
    const unsigned int* d_cell_first  = reinterpret_cast<const unsigned int*>(
        static_cast<uintptr_t>(cell_first_dev));
    const unsigned int* d_cell_count  = reinterpret_cast<const unsigned int*>(
        static_cast<uintptr_t>(cell_count_dev));
    const int* d_cell_table = reinterpret_cast<const int*>(
        static_cast<uintptr_t>(cell_table_dev));
    int* d_parent = reinterpret_cast<int*>(static_cast<uintptr_t>(parent_dev));
    unsigned long long* d_edge_score_sum = reinterpret_cast<unsigned long long*>(
        static_cast<uintptr_t>(edge_score_sum_dev));
    unsigned long long* d_edge_count = reinterpret_cast<unsigned long long*>(
        static_cast<uintptr_t>(edge_count_dev));
    unsigned long long* d_pair_count = reinterpret_cast<unsigned long long*>(
        static_cast<uintptr_t>(pair_count_dev));
    unsigned long long* d_phase_legal = reinterpret_cast<unsigned long long*>(
        static_cast<uintptr_t>(phase_legal_count_dev));
    unsigned long long* d_aabb_overlap = reinterpret_cast<unsigned long long*>(
        static_cast<uintptr_t>(aabb_overlap_count_dev));

    // ── 1. Init parent[i] = i ───────────────────────────────────────
    {
        const unsigned int block = 256u;
        const unsigned int grid  = (n_nodes + block - 1u) / block;
        ghost_lattice_init_parent<<<grid, block, 0, cu_stream>>>(
            d_parent, n_nodes);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }

    // ── 2. Edge adjudication (one thread per cell) ───────────────────
    if (n_cells > 0u) {
        const unsigned int block = 128u;
        const unsigned int grid  = (n_cells + block - 1u) / block;
        ghost_lattice_edge_kernel<<<grid, block, 0, cu_stream>>>(
            d_nodes_bytes,
            n_nodes,
            d_permutation,
            d_cell_first,
            d_cell_count,
            n_cells,
            d_cell_table,
            d_parent,
            d_edge_score_sum,
            d_edge_count,
            nullptr,
            0ULL,
            nullptr,
            d_pair_count,
            d_phase_legal,
            d_aabb_overlap,
            spatial_cell_size_a,
            max_temporal_edge_steps,
            so3_threshold,
            0u);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }

    // ── 3. Path compression (every node → its canonical root) ───────
    {
        const unsigned int block = 256u;
        const unsigned int grid  = (n_nodes + block - 1u) / block;
        ghost_lattice_path_compress<<<grid, block, 0, cu_stream>>>(
            d_parent, n_nodes);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            return static_cast<int>(e);
        }
    }

    // Single sync at the end so the host can read parent_dev /
    // edge_count_dev / etc. immediately on return. The Rust caller will
    // also re-sync the stream defensively before downloads.
    cudaError_t sync_err = cudaStreamSynchronize(cu_stream);
    return (sync_err == cudaSuccess) ? 0 : static_cast<int>(sync_err);
}

extern "C" int prism_ghost_phase_lattice_write(
    uint64_t       nodes_dev,
    uint32_t       n_nodes,
    uint64_t       permutation_dev,
    uint64_t       cell_first_dev,
    uint64_t       cell_count_dev,
    uint32_t       n_cells,
    uint64_t       cell_table_dev,
    uint64_t       edge_score_sum_dev,
    uint64_t       edge_count_dev,
    uint64_t       accepted_edges_dev,
    uint64_t       accepted_edge_capacity,
    uint64_t       accepted_edge_overflow_dev,
    float          spatial_cell_size_a,
    uint64_t       max_temporal_edge_steps,
    float          so3_threshold,
    void*          stream
) {
    if (n_nodes == 0u) return 0;
    cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);
    const unsigned char* d_nodes_bytes = reinterpret_cast<const unsigned char*>(
        static_cast<uintptr_t>(nodes_dev));
    const unsigned int* d_permutation = reinterpret_cast<const unsigned int*>(
        static_cast<uintptr_t>(permutation_dev));
    const unsigned int* d_cell_first = reinterpret_cast<const unsigned int*>(
        static_cast<uintptr_t>(cell_first_dev));
    const unsigned int* d_cell_count = reinterpret_cast<const unsigned int*>(
        static_cast<uintptr_t>(cell_count_dev));
    const int* d_cell_table = reinterpret_cast<const int*>(
        static_cast<uintptr_t>(cell_table_dev));
    unsigned long long* d_edge_score_sum = reinterpret_cast<unsigned long long*>(
        static_cast<uintptr_t>(edge_score_sum_dev));
    unsigned long long* d_edge_count = reinterpret_cast<unsigned long long*>(
        static_cast<uintptr_t>(edge_count_dev));
    unsigned char* d_accepted_edges = reinterpret_cast<unsigned char*>(
        static_cast<uintptr_t>(accepted_edges_dev));
    unsigned long long* d_accepted_edge_overflow =
        reinterpret_cast<unsigned long long*>(
            static_cast<uintptr_t>(accepted_edge_overflow_dev));

    if (n_cells > 0u) {
        const unsigned int block = 128u;
        const unsigned int grid = (n_cells + block - 1u) / block;
        ghost_lattice_edge_kernel<<<grid, block, 0, cu_stream>>>(
            d_nodes_bytes, n_nodes, d_permutation, d_cell_first, d_cell_count,
            n_cells, d_cell_table, nullptr, d_edge_score_sum, d_edge_count,
            d_accepted_edges, accepted_edge_capacity, d_accepted_edge_overflow,
            nullptr, nullptr, nullptr, spatial_cell_size_a,
            max_temporal_edge_steps, so3_threshold, 1u);
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return static_cast<int>(e);
    }
    cudaError_t sync_err = cudaStreamSynchronize(cu_stream);
    return (sync_err == cudaSuccess) ? 0 : static_cast<int>(sync_err);
}
