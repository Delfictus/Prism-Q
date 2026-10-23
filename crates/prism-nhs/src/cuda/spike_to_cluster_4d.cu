// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / M1 — SpikeToCluster4D producer (CUDA + CUB host orchestration)
// ═══════════════════════════════════════════════════════════════════════
//
// On-device producer for the M1 lane. Reads the engine's already-on-
// device spike buffer, performs voxel-to-cluster attribution and
// per-cluster integer count tallies in VRAM, then runs CUB host-side
// reductions to produce the per-cluster AABB and the
// `total_attributed` scalar required for the §5 Conservation-of-Mass
// audit.
//
// M1.2 contract §4 architectural constraints honored here:
//   §4a: warp-aggregated atomics on contended counters
//   §4b: pinned-memory output (Rust side; FFI accessor reads from pinned)
//   §4c: every function takes cudaStream_t — no default-stream launches
//   §4d: no cudaDeviceSynchronize; sync at FFI boundary only
//   §4e: cudaStreamBeginCapture/EndCapture wrapping the kernel sequence
//        (lands in M1.2.4 — Rust side)
//   §4f: shared __host__ __device__ helper in .cuh (V3-ready)
//   C4 : NO cudaLaunchCooperativeKernel
//   C6 : per-cluster AABB segmented reduction uses
//        cub::DeviceSegmentedReduce, NOT atomicMin/atomicMax loops.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
//              -I/usr/local/cuda/include -I/usr/local/cuda/include/cccl
//
// Status at M1.2.3: kernel + CUB orchestration. The producer runs:
//   1. prism_m1_assign_clusters_and_count            (M1.2.2 kernel)
//   2. cub::DeviceRadixSort::SortPairs               (sort by cluster_id)
//   3. cub::DeviceReduce::Sum                        (total_attributed)
//   4. cub::DeviceScan::ExclusiveSum                 (segment offsets)
//   5. prism_m1_pack_offset_terminator               (helper kernel)
//   6. cub::DeviceSegmentedReduce::Reduce<AabbUnion> (per-cluster AABB)
// All scratch buffers come from cudaMallocAsync against the default
// mempool — implicit caching across invocations and capture-compatible.
//
// ═══════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include "spike_to_cluster_4d.cuh"

namespace prism_nhs { namespace m1 {

// ═══════════════════════════════════════════════════════════════════════
// __global__ kernel definitions
// ═══════════════════════════════════════════════════════════════════════

/// Per-spike cluster-id assignment + atomic per-cluster spike-count
/// tally + atomic background-count tally.
///
/// One thread per spike. Hashes the spike position to a 4D voxel cell-id
/// via the canonical __host__ __device__ helper `spike_to_cell_id` (in
/// the shared `.cuh`), writes the cell-id into
/// `d_cluster_id_per_spike[tid]`, and atomically increments either
/// `d_per_cluster_count[cell_id]` (in-bbox) or
/// `*d_background_count_scalar` (UNCLUSTERED — out-of-bbox).
///
/// Determinism class: AtomicsAffected (per
/// `crates/prism-nhs/src/spike_to_cluster_4d.rs::ConservationScalars`
/// type-level docs). Atomic ordering within a cluster depends on
/// warp-scheduling order, so the position of a particular spike's
/// cell-id within the cluster's spike list is non-deterministic. The
/// integer counts themselves (`d_per_cluster_count[c]`,
/// `*d_background_count_scalar`) ARE deterministic — atomicAdd of
/// constant 1ULL is order-independent in its sum, so the counts are
/// BitExact across replicates given fixed input.
__global__ void prism_m1_assign_clusters_and_count(
    const float* __restrict__ d_spike_positions,
    uint32_t                  num_spikes,
    SpatialHashParams         params,
    uint32_t* __restrict__    d_cluster_id_per_spike,
    uint64_t* __restrict__    d_per_cluster_count,
    uint64_t* __restrict__    d_background_count_scalar
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_spikes) return;

    // Layout: planar [num_spikes][3] (x0,y0,z0,x1,y1,z1,...). Matches
    // the legacy `gpu_cluster_backend.rs` spike-buffer convention.
    const float pos[3] = {
        d_spike_positions[3u * tid + 0u],
        d_spike_positions[3u * tid + 1u],
        d_spike_positions[3u * tid + 2u],
    };

    const uint32_t cell_id = spike_to_cell_id(pos, params);
    d_cluster_id_per_spike[tid] = cell_id;

    // atomicAdd on `unsigned long long` is the canonical 64-bit atomic
    // primitive on sm_60+ (sm_120 included). The reinterpret_cast is
    // safe by ABI: on every CUDA-supported platform `uint64_t` and
    // `unsigned long long` are the same underlying type.
    if (cell_id == UNCLUSTERED_CLUSTER_ID) {
        atomicAdd(reinterpret_cast<unsigned long long*>(d_background_count_scalar),
                  static_cast<unsigned long long>(1));
    } else {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_per_cluster_count[cell_id]),
                  static_cast<unsigned long long>(1));
    }
}

/// Initialize a `uint32_t` array to the identity permutation
/// `[0, 1, 2, ..., n-1]`. Used as the value array for
/// `cub::DeviceRadixSort::SortPairs` so the sort produces a permutation
/// of spike-indices ordered by cluster_id ascending.
__global__ void prism_m1_init_spike_indices(
    uint32_t* __restrict__ d_spike_indices,
    uint32_t               num_spikes
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_spikes) return;
    d_spike_indices[tid] = tid;
}

/// Single-thread helper that copies `total_attributed` (a u64 scalar)
/// into the `num_clusters`-th slot of the `u32` segment-offsets array,
/// completing the (num_clusters+1)-element offsets array required by
/// `cub::DeviceSegmentedReduce::Reduce` (begin = offsets[0..N],
/// end = offsets[1..N+1]).
///
/// Narrowing u64 → u32: total_attributed ≤ num_spikes which is u32 by
/// the FFI signature, so the cast is bounded. Asserted at the FFI
/// boundary on the Rust side.
__global__ void prism_m1_pack_offset_terminator(
    const uint64_t* __restrict__ d_total_attributed_scalar,
    uint32_t* __restrict__       d_segment_offsets,
    uint32_t                     num_clusters
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    d_segment_offsets[num_clusters] =
        static_cast<uint32_t>(*d_total_attributed_scalar);
}

// ═══════════════════════════════════════════════════════════════════════
// Device-side functors for cub::DeviceSegmentedReduce
// ═══════════════════════════════════════════════════════════════════════

/// Permutation+lift functor. Given a sequential index `i` into the
/// SORTED order, looks up the original spike-index via
/// `sorted_spike_indices[i]`, fetches its position from the original
/// `positions` buffer, and lifts it to a degenerate `Aabb` whose `min`
/// and `max` both equal that single position.
///
/// This functor is the input transform of the `transform_iterator` we
/// hand to `cub::DeviceSegmentedReduce`, replacing what would otherwise
/// be a separate gather kernel. Two birds: positions are fetched on
/// demand (no scratch sorted_positions buffer) and lifted into the
/// reduction's value type at the same call site.
struct GatherToDegenerateAabb {
    const float*    positions;             // [num_spikes][3] planar
    const uint32_t* sorted_spike_indices;  // [num_spikes]

    __host__ __device__ Aabb operator()(uint32_t i) const {
        const uint32_t spike_id = sorted_spike_indices[i];
        const float x = positions[3u * spike_id + 0u];
        const float y = positions[3u * spike_id + 1u];
        const float z = positions[3u * spike_id + 2u];
        Aabb out;
        out.min[0] = x; out.min[1] = y; out.min[2] = z;
        out.max[0] = x; out.max[1] = y; out.max[2] = z;
        return out;
    }
};

/// Narrowing cast functor `uint64_t -> uint32_t`. Used as the unary
/// transform in front of `cub::DeviceScan::ExclusiveSum` so the scan's
/// accumulator type is u32 (matching `d_segment_offsets`) while the
/// per-cluster count buffer remains u64. The narrowing is bounded:
/// total spikes ≤ u32 by the FFI surface, so per-cluster counts can
/// never exceed u32.
struct U64ToU32Cast {
    __host__ __device__ uint32_t operator()(uint64_t v) const {
        return static_cast<uint32_t>(v);
    }
};

/// Element-wise AABB union. Associative + commutative; suitable as the
/// reduction op for `cub::DeviceSegmentedReduce::Reduce`.
struct AabbUnion {
    __host__ __device__ Aabb operator()(const Aabb& a, const Aabb& b) const {
        Aabb out;
        out.min[0] = fminf(a.min[0], b.min[0]);
        out.min[1] = fminf(a.min[1], b.min[1]);
        out.min[2] = fminf(a.min[2], b.min[2]);
        out.max[0] = fmaxf(a.max[0], b.max[0]);
        out.max[1] = fmaxf(a.max[1], b.max[1]);
        out.max[2] = fmaxf(a.max[2], b.max[2]);
        return out;
    }
};

// ═══════════════════════════════════════════════════════════════════════
// extern "C" host orchestration entry points
// ═══════════════════════════════════════════════════════════════════════

extern "C" {

uint32_t prism_m1_link_probe(void) {
    // Sentinel value verifies the static archive linked correctly and
    // the FFI ABI is round-tripping: Rust calls this, expects 0xC0FFEE.
    // Pinned by `link_probe_returns_sentinel` test in
    // spike_to_cluster_4d.rs.
    return 0xC0FFEEu;
}

cudaError_t prism_m1_spike_to_cluster_4d_run(
    const float* d_spike_positions,
    uint32_t num_spikes,
    const SpatialHashParams* h_params,
    cudaStream_t stream,
    uint32_t* d_cluster_id_per_spike,
    uint64_t* d_per_cluster_count,
    uint64_t* d_total_attributed_scalar,
    uint64_t* d_background_count_scalar,
    Aabb* d_per_cluster_aabb,
    uint32_t num_clusters
) {
    // ───────────────────────────────────────────────────────────────────
    // M1.2.3 orchestration — kernel + CUB.
    //
    //   1. cudaMemsetAsync zeros the three integer count outputs.
    //   2. `prism_m1_assign_clusters_and_count` is launched
    //      one-thread-per-spike with 256 threads/block.
    //   3. Allocate scratch via cudaMallocAsync against the default
    //      mempool. The mempool caches freed allocations, giving
    //      "persistent across invocations for fixed shape" behavior
    //      without an explicit Rust-side workspace cache. Captured-
    //      graph-compatible since CUDA 11.4 (M1.2.4 wraps capture
    //      around this sequence).
    //   4. CUB chain:
    //        a. DeviceRadixSort::SortPairs  (cluster_id, spike_index)
    //        b. DeviceReduce::Sum           (per_cluster_count → total)
    //        c. DeviceScan::ExclusiveSum    (per_cluster_count → offsets)
    //        d. pack_offset_terminator      (offsets[N] = total)
    //        e. DeviceSegmentedReduce       (transform-iter → AABB)
    //   5. cudaFreeAsync — returns scratch to the mempool.
    //
    // No cudaDeviceSynchronize anywhere (§4d). Caller calls
    // cudaStreamSynchronize at the FFI boundary when it needs the
    // result.
    // ───────────────────────────────────────────────────────────────────

    cudaError_t err;

    // ── Step 1: zero the integer count outputs. ─────────────────────
    err = cudaMemsetAsync(
        d_per_cluster_count, 0,
        static_cast<size_t>(num_clusters) * sizeof(uint64_t),
        stream
    );
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(d_background_count_scalar, 0, sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(d_total_attributed_scalar, 0, sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    if (num_spikes == 0) {
        // No spikes — counts stay zero, conservation (0 + 0 == 0)
        // holds trivially. Per-cluster AABB stays whatever the caller
        // initialized to (unspecified — caller must not consume it
        // when num_spikes == 0).
        return cudaSuccess;
    }

    constexpr uint32_t THREADS_PER_BLOCK = 256u;

    // ── Step 2: per-spike cluster-id + count kernel. ─────────────────
    {
        const uint32_t blocks =
            (num_spikes + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        prism_m1_assign_clusters_and_count<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            d_spike_positions,
            num_spikes,
            *h_params,
            d_cluster_id_per_spike,
            d_per_cluster_count,
            d_background_count_scalar
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) return err;
    }

    // ── Step 3: scratch allocation via mempool. ──────────────────────
    //
    // Buffers (sizes in bytes):
    //   d_input_indices       : num_spikes      * 4   (u32 identity)
    //   d_sorted_cluster_ids  : num_spikes      * 4   (u32, sort dst)
    //   d_sorted_spike_indices: num_spikes      * 4   (u32, sort dst)
    //   d_segment_offsets     : (num_clusters+1)* 4   (u32, scan dst + terminator)
    //   d_cub_scratch_*       : query-result bytes per CUB call
    //
    // A single contiguous allocation per buffer keeps fragmentation
    // bounded; the default mempool reuses freed extents on the next
    // call with the same shape, so amortized cost across a run is the
    // first-call cost.

    uint32_t* d_input_indices        = nullptr;
    uint32_t* d_sorted_cluster_ids   = nullptr;
    uint32_t* d_sorted_spike_indices = nullptr;
    uint32_t* d_segment_offsets      = nullptr;
    void*     d_cub_scratch_sort     = nullptr;
    void*     d_cub_scratch_reduce   = nullptr;
    void*     d_cub_scratch_scan     = nullptr;
    void*     d_cub_scratch_segred   = nullptr;

    // RAII-style cleanup: any failure goto FAILED_FREE which calls
    // cudaFreeAsync on every non-null pointer and returns `err`.
#define M1_TRY(call)                                          \
    do {                                                      \
        err = (call);                                         \
        if (err != cudaSuccess) goto FAILED_FREE;             \
    } while (0)

    M1_TRY(cudaMallocAsync(reinterpret_cast<void**>(&d_input_indices),
                           static_cast<size_t>(num_spikes) * sizeof(uint32_t),
                           stream));
    M1_TRY(cudaMallocAsync(reinterpret_cast<void**>(&d_sorted_cluster_ids),
                           static_cast<size_t>(num_spikes) * sizeof(uint32_t),
                           stream));
    M1_TRY(cudaMallocAsync(reinterpret_cast<void**>(&d_sorted_spike_indices),
                           static_cast<size_t>(num_spikes) * sizeof(uint32_t),
                           stream));
    M1_TRY(cudaMallocAsync(reinterpret_cast<void**>(&d_segment_offsets),
                           (static_cast<size_t>(num_clusters) + 1u) * sizeof(uint32_t),
                           stream));

    // ── Step 4: init identity permutation on the input-indices. ──────
    {
        const uint32_t blocks =
            (num_spikes + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        prism_m1_init_spike_indices<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            d_input_indices, num_spikes
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto FAILED_FREE;
    }

    // ── Step 5a: DeviceRadixSort::SortPairs. ─────────────────────────
    {
        size_t bytes_sort = 0;
        M1_TRY(cub::DeviceRadixSort::SortPairs(
            nullptr, bytes_sort,
            d_cluster_id_per_spike, d_sorted_cluster_ids,
            d_input_indices,        d_sorted_spike_indices,
            static_cast<int>(num_spikes),
            0, sizeof(uint32_t) * 8,  // begin_bit, end_bit (full key range)
            stream
        ));
        M1_TRY(cudaMallocAsync(&d_cub_scratch_sort, bytes_sort, stream));
        M1_TRY(cub::DeviceRadixSort::SortPairs(
            d_cub_scratch_sort, bytes_sort,
            d_cluster_id_per_spike, d_sorted_cluster_ids,
            d_input_indices,        d_sorted_spike_indices,
            static_cast<int>(num_spikes),
            0, sizeof(uint32_t) * 8,
            stream
        ));
    }

    // ── Step 5b: DeviceReduce::Sum (per_cluster_count → total). ──────
    {
        size_t bytes_red = 0;
        M1_TRY(cub::DeviceReduce::Sum(
            nullptr, bytes_red,
            d_per_cluster_count, d_total_attributed_scalar,
            static_cast<int>(num_clusters),
            stream
        ));
        M1_TRY(cudaMallocAsync(&d_cub_scratch_reduce, bytes_red, stream));
        M1_TRY(cub::DeviceReduce::Sum(
            d_cub_scratch_reduce, bytes_red,
            d_per_cluster_count, d_total_attributed_scalar,
            static_cast<int>(num_clusters),
            stream
        ));
    }

    // ── Step 5c: DeviceScan::ExclusiveSum (counts → begin offsets). ──
    {
        size_t bytes_scan = 0;
        // ExclusiveSum operates on uint32 here. Per-cluster counts are
        // u64 in the FFI; offsets target is u32 (sized by num_spikes
        // which is u32). The narrowing is bounded: per-cluster count ≤
        // num_spikes ≤ u32::MAX. Lift the input iterator through a
        // U64→U32 cast functor (struct, not lambda — a
        // `__host__ __device__` lambda would require the
        // `--extended-lambda` nvcc flag, which we deliberately do not
        // depend on).
        auto count_to_u32 = thrust::make_transform_iterator(
            d_per_cluster_count,
            U64ToU32Cast{}
        );
        M1_TRY(cub::DeviceScan::ExclusiveSum(
            nullptr, bytes_scan,
            count_to_u32, d_segment_offsets,
            static_cast<int>(num_clusters),
            stream
        ));
        M1_TRY(cudaMallocAsync(&d_cub_scratch_scan, bytes_scan, stream));
        M1_TRY(cub::DeviceScan::ExclusiveSum(
            d_cub_scratch_scan, bytes_scan,
            count_to_u32, d_segment_offsets,
            static_cast<int>(num_clusters),
            stream
        ));
    }

    // ── Step 5d: pack offsets[N] = total_attributed. ────────────────
    {
        prism_m1_pack_offset_terminator<<<1, 1, 0, stream>>>(
            d_total_attributed_scalar,
            d_segment_offsets,
            num_clusters
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto FAILED_FREE;
    }

    // ── Step 5e: DeviceSegmentedReduce (transform-iter → AABB). ─────
    {
        // Input iterator: `i in [0..num_spikes)` → permuted via
        // `sorted_spike_indices[i]` → degenerate Aabb at that position.
        // The "input segment c" is sorted_spike_indices[offsets[c]..offsets[c+1]],
        // matching cub::DeviceSegmentedReduce's begin/end_offsets ABI.
        auto input_iter = thrust::make_transform_iterator(
            thrust::counting_iterator<uint32_t>(0u),
            GatherToDegenerateAabb{ d_spike_positions, d_sorted_spike_indices }
        );

        // Initial value: the AABB-union identity ([+inf]³, [-inf]³).
        Aabb identity_aabb;
        identity_aabb.min[0] = identity_aabb.min[1] = identity_aabb.min[2] =  FLT_MAX;
        identity_aabb.max[0] = identity_aabb.max[1] = identity_aabb.max[2] = -FLT_MAX;

        size_t bytes_segred = 0;
        M1_TRY(cub::DeviceSegmentedReduce::Reduce(
            nullptr, bytes_segred,
            input_iter,
            d_per_cluster_aabb,
            static_cast<int>(num_clusters),
            d_segment_offsets,        // begin_offsets[c]
            d_segment_offsets + 1,    // end_offsets[c]   (= begin_offsets[c+1])
            AabbUnion{},
            identity_aabb,
            stream
        ));
        M1_TRY(cudaMallocAsync(&d_cub_scratch_segred, bytes_segred, stream));
        M1_TRY(cub::DeviceSegmentedReduce::Reduce(
            d_cub_scratch_segred, bytes_segred,
            input_iter,
            d_per_cluster_aabb,
            static_cast<int>(num_clusters),
            d_segment_offsets,
            d_segment_offsets + 1,
            AabbUnion{},
            identity_aabb,
            stream
        ));
    }

    // ── Step 6: free scratch via cudaFreeAsync. ─────────────────────
    err = cudaSuccess;
FAILED_FREE:
    // cudaFreeAsync on a null pointer is a no-op (CUDA spec); on a
    // valid pointer it returns the allocation to the default mempool
    // for reuse on the next invocation. Errors during free are
    // surfaced ONLY if the orchestration was otherwise successful;
    // a free-error after an earlier failure does not overwrite the
    // earlier diagnostic.
    {
        cudaError_t free_err = cudaSuccess;
        if (d_cub_scratch_segred)   free_err = cudaFreeAsync(d_cub_scratch_segred, stream);
        if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
        if (d_cub_scratch_scan)     free_err = cudaFreeAsync(d_cub_scratch_scan, stream);
        if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
        if (d_cub_scratch_reduce)   free_err = cudaFreeAsync(d_cub_scratch_reduce, stream);
        if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
        if (d_cub_scratch_sort)     free_err = cudaFreeAsync(d_cub_scratch_sort, stream);
        if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
        if (d_segment_offsets)      free_err = cudaFreeAsync(d_segment_offsets, stream);
        if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
        if (d_sorted_spike_indices) free_err = cudaFreeAsync(d_sorted_spike_indices, stream);
        if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
        if (d_sorted_cluster_ids)   free_err = cudaFreeAsync(d_sorted_cluster_ids, stream);
        if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
        if (d_input_indices)        free_err = cudaFreeAsync(d_input_indices, stream);
        if (err == cudaSuccess && free_err != cudaSuccess) err = free_err;
    }
    return err;

#undef M1_TRY
}

}  // extern "C"

}}  // namespace prism_nhs::m1
