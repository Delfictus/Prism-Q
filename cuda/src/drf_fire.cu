#include "prismcode_cuda.h"

#include "../kernels/drf_fire.cu"

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

extern "C" cudaError_t prismcode_cuda_drf_fire(
    PrismcodeDrfState* d_states,
    PrismcodeEventRecord* d_events,
    uint64_t row_count,
    double threshold_floor,
    double refractory_period,
    uint64_t event_time_tick,
    cudaStream_t stream
) {
    if (row_count == 0u) {
        return cudaSuccess;
    }
    if (d_states == nullptr || d_events == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (threshold_floor < 0.0 || refractory_period < 0.0) {
        return cudaErrorInvalidValue;
    }

    dim3 grid;
    dim3 block;
    cudaError_t err = launch_shape(row_count, &grid, &block);
    if (err != cudaSuccess) {
        return err;
    }

    prismcode_cuda_kernels::prismcode_drf_fire_kernel<<<grid, block, 0, stream>>>(
        d_states,
        d_events,
        row_count,
        threshold_floor,
        refractory_period,
        event_time_tick
    );
    return cudaGetLastError();
}
