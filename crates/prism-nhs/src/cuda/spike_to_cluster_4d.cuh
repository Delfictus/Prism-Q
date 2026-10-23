// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / M1 — SpikeToCluster4D shared header
// ═══════════════════════════════════════════════════════════════════════
//
// Shared header for the M1 producer transform. Defines:
//
//   * FFI-stable C++ struct shapes that mirror the Rust-side types
//     in `crates/prism-nhs/src/spike_to_cluster_4d.rs` and
//     `entangled_manifold.rs` byte-for-byte. Layouts MUST match.
//
//   * The canonical `__host__ __device__` cluster-assignment helper
//     `spike_to_cell_id`. Per the M1.2 contract, this function is
//     called from BOTH the GPU kernel AND V3's CPU reference
//     implementation (which lands in M1.3) so bit-exact equivalence
//     is by construction.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_M1_SPIKE_TO_CLUSTER_4D_CUH
#define PRISM_NHS_M1_SPIKE_TO_CLUSTER_4D_CUH

#include <cstdint>

// PRISM_HD: makes a function callable from BOTH host and device when
// compiled with nvcc, and host-only when included from a plain C++
// translation unit. This is the V3 "same function in both paths"
// mechanism (M1.2 contract §4f, M1.3 V3 verification gate).
#ifdef __CUDACC__
  #define PRISM_HD __host__ __device__ __forceinline__
#else
  #define PRISM_HD inline
#endif

namespace prism_nhs { namespace m1 {

// ---------------------------------------------------------------------
// FFI-stable struct shapes
// ---------------------------------------------------------------------
//
// LAYOUT CONTRACT — these must match Rust-side types byte-for-byte:
//
//   Aabb                 = [f32; 3] min + [f32; 3] max         = 24 B
//   ManifoldViewAabbFfi  = Aabb + u32 support_count + u32 frame = 32 B
//
// Both are #[repr(C)] on the Rust side; both are POD here. A pin
// test on the Rust side asserts size_of::<ManifoldViewAabbFfi>() == 32
// and align_of == 4 (see spike_to_cluster_4d::tests).

struct Aabb {
    float min[3];
    float max[3];
};
static_assert(sizeof(Aabb) == 24, "Aabb FFI layout drift");

struct ManifoldViewAabbFfi {
    Aabb aabb;
    uint32_t support_count;
    uint32_t frame;
};
static_assert(sizeof(ManifoldViewAabbFfi) == 32, "ManifoldViewAabbFfi FFI layout drift");

// ---------------------------------------------------------------------
// Spatial-hash parameters
// ---------------------------------------------------------------------
//
// The M1 producer uses a uniform-grid spatial hash with cell size =
// epsilon (the clustering radius). The 27-cell neighborhood of a
// query point exactly covers all points within epsilon, matching the
// legacy `gpu_cluster_backend.rs` convention.
//
// `bbox_min` and `bbox_max` are the axis-aligned bounding box over
// the input spike positions, computed by a bbox-scan kernel before
// the main assignment kernel. `cell_size` is the epsilon used at
// view-construction time.

struct SpatialHashParams {
    float    bbox_min[3];
    float    bbox_max[3];
    float    cell_size;
    int32_t  grid_dim[3];
    uint32_t num_cells;
};

// ---------------------------------------------------------------------
// Sentinels — must match Rust-side constants
// ---------------------------------------------------------------------
//
// UNCLUSTERED_CLUSTER_ID matches `crate::site_manifest::ClusterId::UNCLUSTERED`
// (= u32::MAX). A spike whose position falls outside the spatial-hash
// bbox, or whose cell has no merge edge to a non-empty neighbor, gets
// this id and contributes to `background_count` (see Conservation of
// Mass invariant in §5 of the M1 readiness report).

constexpr uint32_t UNCLUSTERED_CLUSTER_ID = 0xFFFFFFFFu;

// ---------------------------------------------------------------------
// Canonical cluster-cell assignment helper
// ---------------------------------------------------------------------
//
// `spike_to_cell_id` is the M1 producer's atomic per-spike → per-cell
// hash function. Called from the assignment kernel
// (`assign_clusters_and_count`) per spike, and called from V3's CPU
// reference implementation (M1.3) per spike. Bit-exact equivalence is
// by construction — both paths invoke this single function.
//
// Returns `UNCLUSTERED_CLUSTER_ID` for any spike whose position falls
// outside the spatial-hash bbox; otherwise returns the linear
// row-major cell index `cz * grid_dim[1] * grid_dim[0] + cy * grid_dim[0] + cx`.
//
// This helper computes the *cell* id, not yet the connected-components
// cluster id. The full cluster id is produced after a union-find pass
// over the 27-cell neighborhood graph (lands in M1.2.x). For the
// foundation commit, only this helper is non-stub.

PRISM_HD uint32_t spike_to_cell_id(
    const float pos[3],
    const SpatialHashParams& params
) {
    // Use float arithmetic with floor cast; matches the legacy
    // gpu_cluster_backend.rs `assign_cells` semantics exactly.
    const float dx = (pos[0] - params.bbox_min[0]) / params.cell_size;
    const float dy = (pos[1] - params.bbox_min[1]) / params.cell_size;
    const float dz = (pos[2] - params.bbox_min[2]) / params.cell_size;
    const int32_t cx = static_cast<int32_t>(dx);
    const int32_t cy = static_cast<int32_t>(dy);
    const int32_t cz = static_cast<int32_t>(dz);
    if (cx < 0 || cx >= params.grid_dim[0]) return UNCLUSTERED_CLUSTER_ID;
    if (cy < 0 || cy >= params.grid_dim[1]) return UNCLUSTERED_CLUSTER_ID;
    if (cz < 0 || cz >= params.grid_dim[2]) return UNCLUSTERED_CLUSTER_ID;
    return static_cast<uint32_t>(
        cz * params.grid_dim[0] * params.grid_dim[1]
      + cy * params.grid_dim[0]
      + cx
    );
}

// ---------------------------------------------------------------------
// extern "C" host orchestration entry points
// ---------------------------------------------------------------------
//
// Forward declarations of the host orchestration functions exported
// to Rust. Their definitions live in spike_to_cluster_4d.cu. Per the
// M1.2 contract, every function takes a `cudaStream_t` parameter
// (no default-stream launches, §4c) and returns a `cudaError_t` for
// caller-side error handling.
//
// At M1.2.1 (foundation commit) these are stubs that immediately
// return `cudaSuccess` without launching kernels. Real
// implementations land in subsequent intra-lane commits:
//
//   M1.2.x: assign_clusters_and_count kernel (with warp-aggregated atomics)
//   M1.2.x: cub::DeviceRadixSort::SortPairs to segment-order spike positions
//   M1.2.x: per-cluster AABB segmented reduction (cub::DeviceSegmentedReduce
//           with custom BoundingBox operator — NOT atomicMin/atomicMax loops)
//   M1.2.x: cub::DeviceReduce::Sum on per-cluster counters → total_attributed
//   M1.2.x: cudaStream{Begin,End}Capture wrapping the sequence

#include <cuda_runtime.h>

extern "C" {

// Run the full M1 producer kernel sequence on the given stream.
// Inputs are device pointers; outputs are device pointers (caller is
// responsible for allocating both). At M1.2.1 this is a stub that
// returns cudaSuccess immediately.
cudaError_t prism_m1_spike_to_cluster_4d_run(
    // --- inputs (device pointers) ---
    const float* d_spike_positions,           // [num_spikes][3]
    uint32_t num_spikes,
    const SpatialHashParams* h_params,        // host-passed (small struct)
    cudaStream_t stream,
    // --- outputs (device pointers; pre-allocated by caller) ---
    uint32_t* d_cluster_id_per_spike,         // [num_spikes]
    uint64_t* d_per_cluster_count,            // [num_clusters]
    uint64_t* d_total_attributed_scalar,      // [1]
    uint64_t* d_background_count_scalar,      // [1]
    Aabb* d_per_cluster_aabb,                 // [num_clusters]
    uint32_t num_clusters
);

// Probe function — confirms the static archive linked correctly and
// the FFI ABI is consistent. Returns 0xC0FFEE.
uint32_t prism_m1_link_probe(void);

}  // extern "C"

}}  // namespace prism_nhs::m1

#endif  // PRISM_NHS_M1_SPIKE_TO_CLUSTER_4D_CUH
