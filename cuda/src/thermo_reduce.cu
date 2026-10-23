#include "prismcode_cuda.h"

#include "../kernels/thermo_reduce.cu"

namespace {

cudaError_t launch_shape(uint64_t count, dim3* grid, dim3* block) {
    if (grid == nullptr || block == nullptr) {
        return cudaErrorInvalidValue;
    }
    *block = dim3(PRISMCODE_CUDA_THREADS_PER_BLOCK, 1u, 1u);
    uint64_t blocks =
        (count + PRISMCODE_CUDA_THREADS_PER_BLOCK - 1u) /
        PRISMCODE_CUDA_THREADS_PER_BLOCK;
    if (blocks > 4096u) {
        blocks = 4096u;
    }
    *grid = dim3(static_cast<unsigned int>(blocks), 1u, 1u);
    return cudaSuccess;
}

}  // namespace

extern "C" cudaError_t prismcode_cuda_thermo_reduce(
    const PrismcodeThermoSample* d_samples,
    PrismcodeThermoSummary* d_summary,
    uint64_t sample_count,
    cudaStream_t stream
) {
    if (d_summary == nullptr) {
        return cudaErrorInvalidValue;
    }

    prismcode_cuda_kernels::prismcode_thermo_zero_kernel<<<1, 1, 0, stream>>>(d_summary);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess || sample_count == 0u) {
        return err;
    }
    if (d_samples == nullptr) {
        return cudaErrorInvalidValue;
    }

    dim3 grid;
    dim3 block;
    err = launch_shape(sample_count, &grid, &block);
    if (err != cudaSuccess) {
        return err;
    }

    prismcode_cuda_kernels::prismcode_thermo_reduce_kernel<<<grid, block, 0, stream>>>(
        d_samples,
        d_summary,
        sample_count
    );
    return cudaGetLastError();
}
