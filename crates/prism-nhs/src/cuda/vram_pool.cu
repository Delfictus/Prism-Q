// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / F2 — Stream-Ordered Memory Pool + VRAM Audit (CUDA impl)
// ═══════════════════════════════════════════════════════════════════════
//
// See `vram_pool.cuh` for the layout / concurrency / API contract.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════

#include "vram_pool.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace vram_pool {

// ═══════════════════════════════════════════════════════════════════════
// __global__ telemetry kernels (single-thread launch contract)
// ═══════════════════════════════════════════════════════════════════════

/// Zero every field of the audit struct. Single-thread launch.
__global__ void prism_vram_audit_zero_kernel(VramAudit* __restrict__ d_audit) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    d_audit->current_allocated_bytes = 0ULL;
    d_audit->peak_high_water_mark = 0ULL;
    d_audit->pool_exhaustion_flag = VRAM_AUDIT_OK;
}

/// Record a single allocation. Three atomic ops in a single-thread
/// launch:
///   1. atomicAdd: bump `current_allocated_bytes`.
///   2. atomicMax: update `peak_high_water_mark` if exceeded.
///   3. atomicCAS: set `pool_exhaustion_flag` 0→1 if budget exceeded.
///
/// All three ops are necessary even though only one thread runs:
/// concurrent invocations from different streams (different host
/// threads) DO race against each other, and the atomics serialize
/// those cross-stream interleavings deterministically.
__global__ void prism_vram_audit_record_alloc_kernel(
    VramAudit* __restrict__ d_audit,
    unsigned long long alloc_size,
    unsigned long long budget
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;

    // (1) Increment current_allocated_bytes. atomicAdd returns the
    //     PRE-add value; new_current = pre + alloc_size.
    const unsigned long long pre = atomicAdd(
        &d_audit->current_allocated_bytes,
        alloc_size
    );
    const unsigned long long new_current = pre + alloc_size;

    // (2) Update high-water mark if exceeded. atomicMax on u64 is
    //     supported on sm_60+ (Blackwell sm_120 covered).
    atomicMax(&d_audit->peak_high_water_mark, new_current);

    // (3) Set the exhaustion flag if `new_current > budget`. The
    //     atomicCAS 0→1 is idempotent — repeated calls with an
    //     already-set flag are no-ops, so the F1 SWITCH node sees a
    //     stable VIOLATION signal regardless of how many subsequent
    //     allocations happen (they're either successful and counted,
    //     or fail at the cudaMallocFromPoolAsync boundary).
    if (new_current > budget) {
        atomicCAS(&d_audit->pool_exhaustion_flag,
                  VRAM_AUDIT_OK,
                  VRAM_AUDIT_VIOLATION);
    }
}

/// Record a single free. Single-thread atomicAdd of the additive
/// inverse (CUDA has no `atomicSub` for u64; the
/// `~free_size + 1ULL` form is the bit-perfect two's-complement
/// negative under modular u64 arithmetic, so the subtraction is
/// exact for any free_size ≤ current_allocated_bytes).
__global__ void prism_vram_audit_record_free_kernel(
    VramAudit* __restrict__ d_audit,
    unsigned long long free_size
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    const unsigned long long neg = (~free_size) + 1ULL;  // == -free_size mod 2^64
    atomicAdd(&d_audit->current_allocated_bytes, neg);
}

// ═══════════════════════════════════════════════════════════════════════
// extern "C" host orchestration
// ═══════════════════════════════════════════════════════════════════════

extern "C" {

uint32_t prism_vram_pool_link_probe(void) {
    // Sentinel: 0xF2A4 = "F2 Audit". Rust-side
    // `link_probe_returns_sentinel` pins this value.
    return 0xF2A4u;
}

cudaError_t prism_vram_pool_create(
    int32_t device_id,
    void** out_pool
) {
    if (out_pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    *out_pool = nullptr;

    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = device_id;

    cudaMemPool_t pool = nullptr;
    cudaError_t err = cudaMemPoolCreate(&pool, &props);
    if (err != cudaSuccess) {
        return err;
    }

    // ReleaseThreshold = UINT64_MAX. The driver never returns pool
    // memory to the OS while the pool exists. Critical for
    // captured-graph execution: a release-threshold-induced free
    // during graph replay would corrupt the graph's mempool node
    // dependencies.
    uint64_t release_threshold = UINT64_MAX;
    err = cudaMemPoolSetAttribute(
        pool,
        cudaMemPoolAttrReleaseThreshold,
        &release_threshold
    );
    if (err != cudaSuccess) {
        // Best-effort cleanup; report the original error.
        (void)cudaMemPoolDestroy(pool);
        return err;
    }

    // ReuseAllowOpportunistic = false. Reuse only after stream-order
    // completes (not opportunistically). The captured graph models
    // explicit stream dependencies; opportunistic reuse can violate
    // those models.
    int32_t reuse_opportunistic = 0;
    err = cudaMemPoolSetAttribute(
        pool,
        cudaMemPoolReuseAllowOpportunistic,
        &reuse_opportunistic
    );
    if (err != cudaSuccess) {
        (void)cudaMemPoolDestroy(pool);
        return err;
    }

    // ReuseAllowInternalDependencies = false. Same reasoning.
    int32_t reuse_internal = 0;
    err = cudaMemPoolSetAttribute(
        pool,
        cudaMemPoolReuseAllowInternalDependencies,
        &reuse_internal
    );
    if (err != cudaSuccess) {
        (void)cudaMemPoolDestroy(pool);
        return err;
    }

    *out_pool = static_cast<void*>(pool);
    return cudaSuccess;
}

cudaError_t prism_vram_pool_destroy(void* pool) {
    if (pool == nullptr) {
        return cudaSuccess;  // no-op on null handle
    }
    return cudaMemPoolDestroy(static_cast<cudaMemPool_t>(pool));
}

cudaError_t prism_vram_pool_alloc_async(
    void*        pool,
    uint64_t     size,
    cudaStream_t stream,
    void**       out_ptr
) {
    if (out_ptr == nullptr) {
        return cudaErrorInvalidValue;
    }
    *out_ptr = nullptr;
    if (pool == nullptr) {
        return cudaErrorInvalidValue;
    }
    return cudaMallocFromPoolAsync(
        out_ptr,
        static_cast<size_t>(size),
        static_cast<cudaMemPool_t>(pool),
        stream
    );
}

cudaError_t prism_vram_pool_memory_snapshot(
    void* pool, VramPoolMemorySnapshot* out_snapshot
) {
    if (pool == nullptr || out_snapshot == nullptr) return cudaErrorInvalidValue;
    // Publish no partial struct on a failed query. The caller retains the error
    // instead of interpreting an unavailable counter as a measured zero.
    VramPoolMemorySnapshot sampled = {};
    const cudaMemPool_t handle = static_cast<cudaMemPool_t>(pool);
    cudaError_t rc = cudaMemPoolGetAttribute(handle, cudaMemPoolAttrUsedMemCurrent,
                                            &sampled.used_current_bytes);
    if (rc != cudaSuccess) return rc;
    rc = cudaMemPoolGetAttribute(handle, cudaMemPoolAttrUsedMemHigh,
                                &sampled.used_high_bytes);
    if (rc != cudaSuccess) return rc;
    rc = cudaMemPoolGetAttribute(handle, cudaMemPoolAttrReservedMemCurrent,
                                &sampled.reserved_current_bytes);
    if (rc != cudaSuccess) return rc;
    rc = cudaMemPoolGetAttribute(handle, cudaMemPoolAttrReservedMemHigh,
                                &sampled.reserved_high_bytes);
    if (rc != cudaSuccess) return rc;
    *out_snapshot = sampled;
    return cudaSuccess;
}

cudaError_t prism_vram_pool_free_async(
    void*        ptr,
    cudaStream_t stream
) {
    if (ptr == nullptr) {
        return cudaSuccess;  // no-op on null pointer
    }
    return cudaFreeAsync(ptr, stream);
}

cudaError_t prism_vram_audit_init(
    VramAudit*   d_audit,
    cudaStream_t stream
) {
    if (d_audit == nullptr) {
        return cudaErrorInvalidValue;
    }
    prism_vram_audit_zero_kernel<<<1, 1, 0, stream>>>(d_audit);
    return cudaGetLastError();
}

cudaError_t prism_vram_audit_record_alloc(
    VramAudit*   d_audit,
    uint64_t     alloc_size,
    uint64_t     budget,
    cudaStream_t stream
) {
    if (d_audit == nullptr) {
        return cudaErrorInvalidValue;
    }
    prism_vram_audit_record_alloc_kernel<<<1, 1, 0, stream>>>(
        d_audit,
        static_cast<unsigned long long>(alloc_size),
        static_cast<unsigned long long>(budget)
    );
    return cudaGetLastError();
}

cudaError_t prism_vram_audit_record_free(
    VramAudit*   d_audit,
    uint64_t     free_size,
    cudaStream_t stream
) {
    if (d_audit == nullptr) {
        return cudaErrorInvalidValue;
    }
    prism_vram_audit_record_free_kernel<<<1, 1, 0, stream>>>(
        d_audit,
        static_cast<unsigned long long>(free_size)
    );
    return cudaGetLastError();
}

}  // extern "C"

}}  // namespace prism_nhs::vram_pool
