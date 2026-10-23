// ═══════════════════════════════════════════════════════════════════════
// PRISM-TWIN: Tensor Core Cross-Correlation (WMMA)
//
// Computes full residue×residue cross-correlation matrix between
// twin simulation spike trains using CUDA Tensor Cores.
//
// CCF = spike_matrix_A × spike_matrix_B^T (normalized)
//
// VERSION A: __global__ kernel (standalone, post-simulation)
// VERSION B: __device__ function (fuseable into persistent kernel)
//
// WMMA fragments: 16×16×16, FP16 input, FP32 accumulator
// RTX 5080: SM 12.0 (Blackwell), full WMMA support
// ═══════════════════════════════════════════════════════════════════════

#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

// ─────────────────────────────────────────────────────────────────────
// VERSION A: Standalone global kernel
// Each warp computes one 16×16 tile of the output CCF matrix
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void tensor_ccf_compute(
    const half* __restrict__ spike_matrix_a,  // [n_res_padded × n_bins_padded], row-major
    const half* __restrict__ spike_matrix_b,  // [n_res_padded × n_bins_padded], row-major
    float* __restrict__ ccf_output,           // [n_res × n_res], row-major
    const float* __restrict__ norm_a,         // [n_res]
    const float* __restrict__ norm_b,         // [n_res]
    int n_res,
    int n_res_padded,
    int n_bins_padded
) {
    // Each warp handles one 16×16 output tile
    // Grid: (ceil(n_res_padded/16), ceil(n_res_padded/16))
    // Block: (32, 4) = 128 threads = 4 warps per block
    // Each warp in the block handles a different tile column offset

    int warp_id = threadIdx.y;
    int warp_row = blockIdx.y * WMMA_M;
    int warp_col = (blockIdx.x * blockDim.y + warp_id) * WMMA_N;

    if (warp_row >= n_res_padded || warp_col >= n_res_padded) return;

    // Declare fragments
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Zero accumulator
    fill_fragment(c_frag, 0.0f);

    // Accumulate over time bins (K dimension)
    for (int k = 0; k < n_bins_padded; k += WMMA_K) {
        // A tile: rows [warp_row..+16), cols [k..+16)
        load_matrix_sync(a_frag,
            &spike_matrix_a[warp_row * n_bins_padded + k],
            n_bins_padded);

        // B tile: rows [warp_col..+16), cols [k..+16)
        // col_major means B^T is computed automatically
        load_matrix_sync(b_frag,
            &spike_matrix_b[warp_col * n_bins_padded + k],
            n_bins_padded);

        // Tensor Core MMA: c += a × b^T
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store to shared memory for normalization
    __shared__ float tile_buf[4][WMMA_M * WMMA_N];  // 4 warps
    store_matrix_sync(&tile_buf[warp_id][0], c_frag, WMMA_N, mem_row_major);
    __syncthreads();

    // Normalize and write to global memory
    int lane = threadIdx.x;
    for (int i = lane; i < WMMA_M * WMMA_N; i += WARP_SIZE) {
        int local_row = i / WMMA_N;
        int local_col = i % WMMA_N;
        int global_row = warp_row + local_row;
        int global_col = warp_col + local_col;

        if (global_row < n_res && global_col < n_res) {
            float na = norm_a[global_row];
            float nb = norm_b[global_col];
            float denom = na * nb + 1e-8f;
            ccf_output[global_row * n_res + global_col] =
                tile_buf[warp_id][i] / denom;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────
// VERSION B: Device function (for persistent kernel fusion)
// Same math, parameterized by block assignment
// ─────────────────────────────────────────────────────────────────────

__device__ void tensor_ccf_compute_device(
    const half* __restrict__ spike_matrix_a,
    const half* __restrict__ spike_matrix_b,
    float* __restrict__ ccf_output,
    const float* __restrict__ norm_a,
    const float* __restrict__ norm_b,
    int n_res,
    int n_res_padded,
    int n_bins_padded,
    int tile_row,      // which output tile row this call handles
    int tile_col       // which output tile col this call handles
) {
    int warp_row = tile_row * WMMA_M;
    int warp_col = tile_col * WMMA_N;

    if (warp_row >= n_res_padded || warp_col >= n_res_padded) return;

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < n_bins_padded; k += WMMA_K) {
        load_matrix_sync(a_frag,
            &spike_matrix_a[warp_row * n_bins_padded + k],
            n_bins_padded);
        load_matrix_sync(b_frag,
            &spike_matrix_b[warp_col * n_bins_padded + k],
            n_bins_padded);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Direct store + normalize (no shared memory in device version)
    float tile_buf[WMMA_M * WMMA_N];
    store_matrix_sync(tile_buf, c_frag, WMMA_N, mem_row_major);

    int lane = threadIdx.x % WARP_SIZE;
    for (int i = lane; i < WMMA_M * WMMA_N; i += WARP_SIZE) {
        int local_row = i / WMMA_N;
        int local_col = i % WMMA_N;
        int global_row = warp_row + local_row;
        int global_col = warp_col + local_col;

        if (global_row < n_res && global_col < n_res) {
            float na = norm_a[global_row];
            float nb = norm_b[global_col];
            float denom = na * nb + 1e-8f;
            ccf_output[global_row * n_res + global_col] =
                tile_buf[i] / denom;
        }
    }
}


// ─────────────────────────────────────────────────────────────────────
// HELPER: Compute per-row norms (for normalization)
// ─────────────────────────────────────────────────────────────────────

extern "C" __global__ void compute_spike_norms(
    const half* __restrict__ spike_matrix,   // [n_res_padded × n_bins_padded]
    float* __restrict__ norms,               // [n_res]
    int n_res,
    int n_bins_padded
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_res) return;

    float sum_sq = 0.0f;
    for (int k = 0; k < n_bins_padded; k++) {
        float val = __half2float(spike_matrix[row * n_bins_padded + k]);
        sum_sq += val * val;
    }
    norms[row] = sqrtf(sum_sq);
}
