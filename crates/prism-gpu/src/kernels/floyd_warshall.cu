/**
 * Floyd-Warshall All-Pairs Shortest Paths CUDA Kernel
 *
 * ASSUMPTIONS:
 * - Input: Distance matrix stored as row-major float array (n×n)
 * - MAX_VERTICES = 100,000 (enforced by caller)
 * - Precision: f32 for distance values
 * - Block size: 32×32 threads (warp-aligned for optimal coalescing)
 * - Algorithm: Blocked Floyd-Warshall with 3 phases per iteration
 * - Requires: sm_86 (RTX 3060 Ampere architecture)
 *
 * ALGORITHM OVERVIEW:
 * For each pivot k in 0..n:
 *   Phase 1: Update diagonal block containing pivot k
 *   Phase 2: Update row/column blocks dependent on pivot block
 *   Phase 3: Update remaining independent blocks
 *
 * PERFORMANCE TARGETS:
 * - DSJC500 (500 vertices): < 1.5 seconds
 * - Memory throughput: > 80% peak bandwidth
 * - SM occupancy: > 75%
 *
 * REFERENCE: PRISM GPU Plan §4.4 (Phase 4 APSP Kernel)
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

// Maximum supported graph size (caller must enforce)
#define MAX_VERTICES 100000

// Block size for tiled algorithm (32×32 for warp alignment)
#define BLOCK_SIZE 32

/**
 * Phase 1: Update the diagonal block containing pivot k
 *
 * This block is self-dependent and must be updated first.
 * Uses shared memory to cache the pivot block for fast access.
 *
 * @param dist     Distance matrix (n×n, row-major)
 * @param n        Number of vertices
 * @param k        Current pivot vertex
 * @param block_id Block index of the diagonal block
 */
extern "C" __global__ void floyd_warshall_phase1(
    float* __restrict__ dist,
    const int n,
    const int k,
    const int block_id
) {
    // Shared memory for pivot block (BLOCK_SIZE × BLOCK_SIZE)
    __shared__ float s_dist[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global indices for this thread's element
    const int block_start = block_id * BLOCK_SIZE;
    const int i = block_start + ty;
    const int j = block_start + tx;

    // Load distance into shared memory
    float d = FLT_MAX;
    if (i < n && j < n) {
        d = dist[i * n + j];
    }
    s_dist[ty][tx] = d;
    __syncthreads();

    // Compute shortest paths through pivot k within this block
    // Pivot k maps to thread index (k - block_start)
    const int k_local = k - block_start;

    if (k_local >= 0 && k_local < BLOCK_SIZE) {
        // Relaxation: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
        float d_ik = s_dist[ty][k_local];
        float d_kj = s_dist[k_local][tx];

        if (d_ik < FLT_MAX && d_kj < FLT_MAX) {
            float new_dist = d_ik + d_kj;
            if (new_dist < d) {
                d = new_dist;
                s_dist[ty][tx] = d;
            }
        }
    }
    __syncthreads();

    // Write back to global memory
    if (i < n && j < n) {
        dist[i * n + j] = s_dist[ty][tx];
    }
}

/**
 * Phase 2: Update row blocks dependent on pivot block
 *
 * Each row block depends on the pivot block's row data.
 * Uses shared memory to cache pivot block column and current block.
 *
 * @param dist         Distance matrix (n×n, row-major)
 * @param n            Number of vertices
 * @param k            Current pivot vertex
 * @param pivot_block  Block index containing pivot k
 */
extern "C" __global__ void floyd_warshall_phase2_row(
    float* __restrict__ dist,
    const int n,
    const int k,
    const int pivot_block
) {
    // Shared memory for pivot block column and current block
    __shared__ float s_pivot[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_curr[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int pivot_start = pivot_block * BLOCK_SIZE;
    const int block_id = blockIdx.x;
    const int block_start = block_id * BLOCK_SIZE;

    // Skip if this is the pivot block (handled by phase 1)
    if (block_id == pivot_block) {
        return;
    }

    const int i = pivot_start + ty;
    const int j = block_start + tx;

    // Load pivot block column (fixed i, variable j in pivot block)
    float pivot_val = FLT_MAX;
    if (i < n && (pivot_start + tx) < n) {
        pivot_val = dist[i * n + (pivot_start + tx)];
    }
    s_pivot[ty][tx] = pivot_val;

    // Load current block
    float curr_val = FLT_MAX;
    if (i < n && j < n) {
        curr_val = dist[i * n + j];
    }
    s_curr[ty][tx] = curr_val;
    __syncthreads();

    // Relaxation using pivot block
    const int k_local = k - pivot_start;
    if (k_local >= 0 && k_local < BLOCK_SIZE) {
        float d_ik = s_pivot[ty][k_local];
        float d_kj = dist[k * n + j]; // Load k-th row element directly

        if (i < n && j < n && d_ik < FLT_MAX && d_kj < FLT_MAX) {
            float new_dist = d_ik + d_kj;
            if (new_dist < curr_val) {
                curr_val = new_dist;
            }
        }
    }

    // Write back
    if (i < n && j < n) {
        dist[i * n + j] = curr_val;
    }
}

/**
 * Phase 2: Update column blocks dependent on pivot block
 *
 * Each column block depends on the pivot block's column data.
 * Uses shared memory to cache pivot block row and current block.
 *
 * @param dist         Distance matrix (n×n, row-major)
 * @param n            Number of vertices
 * @param k            Current pivot vertex
 * @param pivot_block  Block index containing pivot k
 */
extern "C" __global__ void floyd_warshall_phase2_col(
    float* __restrict__ dist,
    const int n,
    const int k,
    const int pivot_block
) {
    __shared__ float s_pivot[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_curr[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int pivot_start = pivot_block * BLOCK_SIZE;
    const int block_id = blockIdx.x;
    const int block_start = block_id * BLOCK_SIZE;

    if (block_id == pivot_block) {
        return;
    }

    const int i = block_start + ty;
    const int j = pivot_start + tx;

    // Load pivot block row (variable i in pivot block, fixed j)
    float pivot_val = FLT_MAX;
    if ((pivot_start + ty) < n && j < n) {
        pivot_val = dist[(pivot_start + ty) * n + j];
    }
    s_pivot[ty][tx] = pivot_val;

    // Load current block
    float curr_val = FLT_MAX;
    if (i < n && j < n) {
        curr_val = dist[i * n + j];
    }
    s_curr[ty][tx] = curr_val;
    __syncthreads();

    // Relaxation
    const int k_local = k - pivot_start;
    if (k_local >= 0 && k_local < BLOCK_SIZE) {
        float d_ik = dist[i * n + k]; // Load i-th row element directly
        float d_kj = s_pivot[k_local][tx];

        if (i < n && j < n && d_ik < FLT_MAX && d_kj < FLT_MAX) {
            float new_dist = d_ik + d_kj;
            if (new_dist < curr_val) {
                curr_val = new_dist;
            }
        }
    }

    if (i < n && j < n) {
        dist[i * n + j] = curr_val;
    }
}

/**
 * Phase 3: Update remaining independent blocks
 *
 * These blocks depend only on their corresponding row and column blocks,
 * which have been updated in Phase 2.
 *
 * @param dist         Distance matrix (n×n, row-major)
 * @param n            Number of vertices
 * @param k            Current pivot vertex
 * @param pivot_block  Block index containing pivot k
 */
extern "C" __global__ void floyd_warshall_phase3(
    float* __restrict__ dist,
    const int n,
    const int k,
    const int pivot_block
) {
    __shared__ float s_row[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_col[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int pivot_start = pivot_block * BLOCK_SIZE;
    const int row_block = blockIdx.y;
    const int col_block = blockIdx.x;

    // Skip if this is pivot row or column (handled by phase 2)
    if (row_block == pivot_block || col_block == pivot_block) {
        return;
    }

    const int row_start = row_block * BLOCK_SIZE;
    const int col_start = col_block * BLOCK_SIZE;

    const int i = row_start + ty;
    const int j = col_start + tx;

    // Load row block element (from pivot column)
    float row_val = FLT_MAX;
    if (i < n && (pivot_start + tx) < n) {
        row_val = dist[i * n + (pivot_start + tx)];
    }
    s_row[ty][tx] = row_val;

    // Load column block element (from pivot row)
    float col_val = FLT_MAX;
    if ((pivot_start + ty) < n && j < n) {
        col_val = dist[(pivot_start + ty) * n + j];
    }
    s_col[ty][tx] = col_val;
    __syncthreads();

    // Current element
    float curr_val = FLT_MAX;
    if (i < n && j < n) {
        curr_val = dist[i * n + j];
    }

    // Relaxation using cached row and column blocks
    const int k_local = k - pivot_start;
    if (k_local >= 0 && k_local < BLOCK_SIZE) {
        float d_ik = s_row[ty][k_local];
        float d_kj = s_col[k_local][tx];

        if (d_ik < FLT_MAX && d_kj < FLT_MAX) {
            float new_dist = d_ik + d_kj;
            if (new_dist < curr_val) {
                curr_val = new_dist;
            }
        }
    }

    if (i < n && j < n) {
        dist[i * n + j] = curr_val;
    }
}

/**
 * Host entry point for blocked Floyd-Warshall algorithm
 *
 * Executes n iterations (one per pivot vertex), each with 3 kernel phases.
 *
 * @param dist  Distance matrix (n×n, row-major, device pointer)
 * @param n     Number of vertices
 * @return      0 on success, -1 on error
 */
extern "C" int floyd_warshall_blocked(float* dist, int n) {
    if (n <= 0 || n > MAX_VERTICES) {
        return -1;
    }

    const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);

    // Iterate through all pivots
    for (int k = 0; k < n; k++) {
        const int pivot_block = k / BLOCK_SIZE;

        // Phase 1: Update diagonal block
        dim3 grid1(1, 1);
        floyd_warshall_phase1<<<grid1, block_dim>>>(dist, n, k, pivot_block);
        cudaDeviceSynchronize();

        // Phase 2: Update row and column blocks
        dim3 grid2(num_blocks, 1);
        floyd_warshall_phase2_row<<<grid2, block_dim>>>(dist, n, k, pivot_block);
        floyd_warshall_phase2_col<<<grid2, block_dim>>>(dist, n, k, pivot_block);
        cudaDeviceSynchronize();

        // Phase 3: Update remaining blocks
        dim3 grid3(num_blocks, num_blocks);
        floyd_warshall_phase3<<<grid3, block_dim>>>(dist, n, k, pivot_block);
        cudaDeviceSynchronize();
    }

    return 0;
}
