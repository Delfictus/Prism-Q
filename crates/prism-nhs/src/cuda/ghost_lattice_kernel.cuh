// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / GhostPhaseLattice4D — host header
// ═══════════════════════════════════════════════════════════════════════════
//
// Companion to `ghost_lattice_kernel.cu`. Declares:
//
//   * `GhostPhaseLatticeNodeHost` — byte-pinned mirror of the Rust
//     `GhostPhaseLatticeNode` struct in `ghost_phase_lattice.rs`. Both this
//     header and the wrapper .cu carry static_asserts on size + offset.
//   * `prism_ghost_phase_lattice_run` — the extern "C" host orchestrator
//     called from Rust. Takes raw device pointers (cast from
//     `cudarc::CudaSlice::cu_device_ptr() -> u64`) and a CUstream pointer.
//
// The kernel symbols themselves (`ghost_lattice_init_parent`,
// `ghost_lattice_edge_kernel`, `ghost_lattice_path_compress`) are linked
// from the static archive via the `#include` of
// `ghost_lattice_kernel_nvrtc.cu` inside the wrapper .cu — they are NOT
// surfaced as host-callable symbols here.
//
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// Same struct layout as the Rust `GhostPhaseLatticeNode`. Compile-time
// asserts in the wrapper .cu pin every offset.
struct __align__(8) GhostPhaseLatticeNodeHost {
    uint32_t tile_index;          //   0
    uint16_t stream_id;           //   4
    uint16_t _pad0;               //   6
    uint32_t site_id;             //   8
    uint32_t _pad1;               //  12
    uint64_t frame_idx;           //  16
    uint64_t step_idx;            //  24

    uint8_t  protocol_phase;      //  32
    uint8_t  _pad2[3];            //  33
    uint32_t step_bucket;         //  36
    uint16_t ccns_phase_bin;      //  40
    uint16_t _pad3;               //  42
    uint32_t gear_id;             //  44

    float    aabb_min[3];         //  48
    float    aabb_max[3];         //  60
    float    centroid_xyz[3];     //  72

    float    kl_divergence;       //  84
    float    thermo_flux[2];      //  88
    float    water_density;       //  96
    uint32_t causal_lead_residue; // 100

    float    so3_power_spectrum[4][6]; // 104
    uint8_t  so3_plane_status;    // 200
    uint8_t  _pad4[7];            // 201..208
};

struct __align__(8) GhostPhaseLatticeAcceptedEdgeHost {
    uint32_t from_node_index;       //  0
    uint32_t to_node_index;         //  4
    uint64_t from_step_idx;         //  8
    uint64_t to_step_idx;           // 16
    uint64_t temporal_delta_steps;  // 24
    float    total_score;           // 32
    float    so3_score;             // 36
    float    causal_score;          // 40
    float    thermo_score;          // 44
    uint8_t  from_protocol_phase;   // 48
    uint8_t  to_protocol_phase;     // 49
    uint8_t  common_so3_plane_mask; // 50
    uint8_t  predicate_flags;       // 51
    uint32_t schema_version;        // 52
};

extern "C" {

/// Run the full lattice pipeline: init parent → edge adjudication → path
/// compress. Returns 0 on success or a non-zero `cudaError_t` value on the
/// first failed launch / sync.
///
/// Pointers are device pointers obtained via `cudarc::CudaSlice::cu_device_ptr`
/// in Rust (a u64 representation of a CUdeviceptr). The kernel does not
/// retain them past `cudaStreamSynchronize` on the supplied stream.
int prism_ghost_phase_lattice_count(
    uint64_t       nodes_dev,
    uint32_t       n_nodes,
    uint64_t       permutation_dev,
    uint64_t       cell_first_dev,
    uint64_t       cell_count_dev,
    uint32_t       n_cells,
    uint64_t       cell_table_dev,        // 5 ints/cell: cx,cy,cz,phase,bucket
    uint64_t       parent_dev,
    uint64_t       edge_score_sum_dev,    // 1 × u64
    uint64_t       edge_count_dev,        // 1 × u64
    uint64_t       pair_count_dev,        // 1 × u64
    uint64_t       phase_legal_count_dev, // 1 × u64
    uint64_t       aabb_overlap_count_dev,// 1 × u64
    float          spatial_cell_size_a,
    uint64_t       max_temporal_edge_steps,
    float          so3_threshold,
    void*          stream                 // CUstream
);

/// Second deterministic pass. Re-evaluates the identical predicate and emits
/// exactly `accepted_edge_capacity` rows, where capacity is the pass-1 count.
/// It does not mutate union-find state or pass-1 funnel telemetry.
int prism_ghost_phase_lattice_write(
    uint64_t       nodes_dev,
    uint32_t       n_nodes,
    uint64_t       permutation_dev,
    uint64_t       cell_first_dev,
    uint64_t       cell_count_dev,
    uint32_t       n_cells,
    uint64_t       cell_table_dev,
    uint64_t       edge_score_sum_dev,    // pass-2 1 × u64
    uint64_t       edge_count_dev,        // pass-2 1 × u64
    uint64_t       accepted_edges_dev,    // pass-1 count × 56-byte rows
    uint64_t       accepted_edge_capacity,
    uint64_t       accepted_edge_overflow_dev, // 1 × u64
    float          spatial_cell_size_a,
    uint64_t       max_temporal_edge_steps,
    float          so3_threshold,
    void*          stream
);

} // extern "C"
