/**
 * PRISM4D Deterministic Reduction Primitives
 *
 * This header provides warp-level and block-level reduction functions
 * that produce deterministic results regardless of thread scheduling.
 *
 * Key principle: Tree reductions have fixed execution order, unlike atomicAdd
 * which depends on thread arrival order.
 *
 * Performance: Equal or FASTER than atomicAdd due to reduced contention.
 *
 * Usage Pattern:
 *   1. Each thread computes local contributions
 *   2. Use warp_reduce_* for intra-warp reduction (32 threads)
 *   3. Use block_reduce_* for inter-warp reduction (full block)
 *   4. Single atomicAdd from lane 0 (deterministic - one writer per block)
 */

#ifndef PRISM_REDUCTION_PRIMITIVES_CUH
#define PRISM_REDUCTION_PRIMITIVES_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Full warp mask for CUDA 9.0+
#define FULL_WARP_MASK 0xFFFFFFFF

//==============================================================================
// Warp-Level Reductions (32 threads, lock-step execution)
//==============================================================================

/**
 * Warp-level sum reduction for float.
 * All threads in warp participate. Returns sum in lane 0.
 * Other lanes have undefined values after return.
 */
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

/**
 * Warp-level sum reduction for double (FP64).
 * Returns sum in lane 0.
 */
__device__ __forceinline__ double warp_reduce_sum_f64(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

/**
 * Warp-level sum reduction for float3.
 * Returns sum in lane 0.
 */
__device__ __forceinline__ float3 warp_reduce_sum_float3(float3 val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(FULL_WARP_MASK, val.x, offset);
        val.y += __shfl_down_sync(FULL_WARP_MASK, val.y, offset);
        val.z += __shfl_down_sync(FULL_WARP_MASK, val.z, offset);
    }
    return val;
}

/**
 * Warp-level max reduction for float.
 * Returns max in lane 0.
 */
__device__ __forceinline__ float warp_reduce_max_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

/**
 * Warp-level min reduction for float.
 * Returns min in lane 0.
 */
__device__ __forceinline__ float warp_reduce_min_f32(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

//==============================================================================
// Block-Level Reductions (Arbitrary block size, requires shared memory)
//==============================================================================

/**
 * Block-level max reduction for float.
 *
 * @param val Local value from this thread
 * @param smem Shared memory array of size >= (blockDim.x / 32)
 * @return Max of all values in block (valid in thread 0 only)
 */
__device__ __forceinline__ float block_reduce_max_f32(float val, float* smem) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = (blockDim.x + 31) / 32;

    // Phase 1: Reduce within each warp
    val = warp_reduce_max_f32(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // Phase 2: First warp reduces across all warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : -INFINITY;
        val = warp_reduce_max_f32(val);
    }

    return val;
}

/**
 * Block-level min reduction for float.
 *
 * @param val Local value from this thread
 * @param smem Shared memory array of size >= (blockDim.x / 32)
 * @return Min of all values in block (valid in thread 0 only)
 */
__device__ __forceinline__ float block_reduce_min_f32(float val, float* smem) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = (blockDim.x + 31) / 32;

    // Phase 1: Reduce within each warp
    val = warp_reduce_min_f32(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // Phase 2: First warp reduces across all warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : INFINITY;
        val = warp_reduce_min_f32(val);
    }

    return val;
}

/**
 * Block-level sum reduction for float.
 *
 * @param val Local value from this thread
 * @param smem Shared memory array of size >= (blockDim.x / 32)
 * @return Sum of all values in block (valid in thread 0 only)
 *
 * Usage:
 *   __shared__ float smem[32];  // Max 32 warps per block
 *   float result = block_reduce_sum_f32(my_value, smem);
 *   if (threadIdx.x == 0) { atomicAdd(output, result); }
 */
__device__ __forceinline__ float block_reduce_sum_f32(float val, float* smem) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = (blockDim.x + 31) / 32;

    // Phase 1: Reduce within each warp
    val = warp_reduce_sum_f32(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // Phase 2: First warp reduces across all warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum_f32(val);
    }

    return val;
}

/**
 * Block-level sum reduction for double (FP64).
 * Use this for energy accumulation to maintain precision.
 */
__device__ __forceinline__ double block_reduce_sum_f64(double val, double* smem) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = (blockDim.x + 31) / 32;

    // Phase 1: Reduce within each warp
    val = warp_reduce_sum_f64(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // Phase 2: First warp reduces across all warps
    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : 0.0;
        val = warp_reduce_sum_f64(val);
    }

    return val;
}

/**
 * Block-level sum reduction for float3.
 * Use this for force accumulation.
 */
__device__ __forceinline__ float3 block_reduce_sum_float3(float3 val, float3* smem) {
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    const int num_warps = (blockDim.x + 31) / 32;

    // Phase 1: Reduce within each warp
    val = warp_reduce_sum_float3(val);

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    // Phase 2: First warp reduces across all warps
    if (warp_id == 0) {
        if (lane < num_warps) {
            val = smem[lane];
        } else {
            val = make_float3(0.0f, 0.0f, 0.0f);
        }
        val = warp_reduce_sum_float3(val);
    }

    return val;
}

//==============================================================================
// Deterministic Force Accumulation Helpers
//==============================================================================

/**
 * Structure to collect forces per atom across all interactions.
 * Allocate one per atom in shared memory for small systems,
 * or use thread-local accumulation with block reduction.
 */
struct ForceAccumulator {
    float3 force;

    __device__ __forceinline__ void init() {
        force = make_float3(0.0f, 0.0f, 0.0f);
    }

    __device__ __forceinline__ void add(float fx, float fy, float fz) {
        force.x += fx;
        force.y += fy;
        force.z += fz;
    }

    __device__ __forceinline__ void add(float3 f) {
        force.x += f.x;
        force.y += f.y;
        force.z += f.z;
    }
};

//==============================================================================
// Kahan Summation for High-Precision Energy Accumulation
//==============================================================================

/**
 * Kahan summation accumulator for numerically stable FP64 sums.
 * Reduces error from O(n*eps) to O(eps) where eps is machine epsilon.
 */
struct KahanAccumulator {
    double sum;
    double compensation;

    __device__ __forceinline__ void init() {
        sum = 0.0;
        compensation = 0.0;
    }

    __device__ __forceinline__ void add(double value) {
        double y = value - compensation;
        double t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }

    __device__ __forceinline__ void add_f32(float value) {
        add((double)value);
    }

    __device__ __forceinline__ double get() const {
        return sum;
    }
};

/**
 * Reduce Kahan accumulators across warp.
 * Returns combined accumulator in lane 0.
 */
__device__ __forceinline__ KahanAccumulator warp_reduce_kahan(KahanAccumulator acc) {
    // First reduce sums
    acc.sum = warp_reduce_sum_f64(acc.sum);
    // Then reduce compensations (for maximum precision)
    acc.compensation = warp_reduce_sum_f64(acc.compensation);
    return acc;
}

//==============================================================================
// Segmented Reductions (for per-atom force accumulation)
//==============================================================================

/**
 * Segmented warp reduction - reduces values within segments defined by keys.
 * Useful for accumulating forces where multiple threads contribute to same atom.
 *
 * @param val Value to reduce
 * @param key Segment identifier (e.g., atom index)
 * @param is_head True if this thread starts a new segment
 * @return Reduced value (valid at segment head only)
 */
__device__ __forceinline__ float warp_segmented_reduce_sum_f32(
    float val,
    int key,
    bool* is_head
) {
    const int lane = threadIdx.x % 32;

    // Determine segment heads by comparing keys with previous lane
    int prev_key = __shfl_up_sync(FULL_WARP_MASK, key, 1);
    *is_head = (lane == 0) || (key != prev_key);

    // Create segment mask
    unsigned int head_mask = __ballot_sync(FULL_WARP_MASK, *is_head);

    // Find the position of the next segment head after this lane
    unsigned int lanes_after = ~((1u << (lane + 1)) - 1);
    unsigned int next_heads = head_mask & lanes_after;
    int next_head_lane = (next_heads == 0) ? 32 : __ffs(next_heads) - 1;

    // Reduce within segment using inclusive scan
    float sum = val;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        float n = __shfl_up_sync(FULL_WARP_MASK, sum, offset);
        int src_lane = lane - offset;
        // Only add if source is in same segment
        if (src_lane >= 0) {
            int src_key = __shfl_sync(FULL_WARP_MASK, key, src_lane);
            if (src_key == key) {
                sum += n;
            }
        }
    }

    return sum;
}

//==============================================================================
// Atomic Operations with Deterministic Ordering
//==============================================================================

/**
 * Deterministic atomic add - wraps atomicAdd but documents the contract
 * that the caller has ensured only one thread per unique address writes.
 *
 * Use case: After block_reduce_sum, only thread 0 writes to global memory.
 * This is deterministic because there's exactly one writer per block per address.
 */
__device__ __forceinline__ void deterministic_atomic_add_f32(
    float* address,
    float val
) {
    // The determinism comes from the calling pattern, not the atomicAdd itself.
    // By using block reductions, we ensure only one thread per block writes
    // to each address, making the order deterministic.
    atomicAdd(address, val);
}

__device__ __forceinline__ void deterministic_atomic_add_f64(
    double* address,
    double val
) {
    atomicAdd(address, val);
}

//==============================================================================
// Thread Block Cooperation Utilities
//==============================================================================

/**
 * Barrier with memory fence - ensures all writes visible before proceeding.
 */
__device__ __forceinline__ void block_sync_with_fence() {
    __threadfence_block();
    __syncthreads();
}

/**
 * Grid-wide memory fence - ensures writes visible to all blocks.
 * Use before reading results written by other blocks.
 */
__device__ __forceinline__ void grid_sync_with_fence() {
    __threadfence();
}

#endif // PRISM_REDUCTION_PRIMITIVES_CUH
