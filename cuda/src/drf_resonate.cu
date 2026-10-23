#include "prismcode_cuda.h"

#include "../kernels/drf_resonate.cu"

namespace {

cudaError_t launch_shape(uint64_t count, dim3* grid, dim3* block) {
    if (grid == nullptr || block == nullptr) {
        return cudaErrorInvalidValue;
    }
    *block = dim3(PRISMCODE_CUDA_THREADS_PER_BLOCK, 1u, 1u);
    const uint64_t blocks =
        (count + PRISMCODE_CUDA_THREADS_PER_BLOCK - 1u) /
        PRISMCODE_CUDA_THREADS_PER_BLOCK;
    if (blocks > 0x7fffffffu) {
        return cudaErrorInvalidValue;
    }
    *grid = dim3(static_cast<unsigned int>(blocks), 1u, 1u);
    return cudaSuccess;
}

}  // namespace

extern "C" cudaError_t prismcode_cuda_drf_resonate(
    const uint16_t* d_channel_vectors,
    const float* d_channel_weights,
    PrismcodeDrfState* d_states,
    uint64_t row_count,
    uint32_t channels_per_row,
    cudaStream_t stream
) {
    if (row_count == 0u) {
        return cudaSuccess;
    }
    if (d_channel_vectors == nullptr || d_states == nullptr || channels_per_row == 0u) {
        return cudaErrorInvalidValue;
    }

    dim3 grid;
    dim3 block;
    cudaError_t err = launch_shape(row_count, &grid, &block);
    if (err != cudaSuccess) {
        return err;
    }

    prismcode_cuda_kernels::prismcode_drf_resonate_kernel<<<grid, block, 0, stream>>>(
        d_channel_vectors,
        d_channel_weights,
        d_states,
        row_count,
        channels_per_row
    );
    return cudaGetLastError();
}
