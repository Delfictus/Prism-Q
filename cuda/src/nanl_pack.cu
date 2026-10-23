#include "prismcode_cuda.h"

#include "../kernels/nanl_pack.cu"

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

extern "C" cudaError_t prismcode_cuda_nanl_pack(
    const PrismcodeMemoryRow* d_rows,
    PrismcodeNanlPackedRow* d_out,
    uint64_t row_count,
    cudaStream_t stream
) {
    if (row_count == 0u) {
        return cudaSuccess;
    }
    if (d_rows == nullptr || d_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    dim3 grid;
    dim3 block;
    cudaError_t err = launch_shape(row_count, &grid, &block);
    if (err != cudaSuccess) {
        return err;
    }

    prismcode_cuda_kernels::prismcode_nanl_pack_kernel<<<grid, block, 0, stream>>>(
        d_rows,
        d_out,
        row_count
    );
    return cudaGetLastError();
}
