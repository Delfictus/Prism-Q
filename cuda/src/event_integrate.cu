#include "prismcode_cuda.h"

#include "../kernels/event_integrate.cu"

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

extern "C" cudaError_t prismcode_cuda_event_integrate(
    const PrismcodeInstruction* d_instructions,
    const PrismcodeMemoryRow* d_rows,
    PrismcodeEventRecord* d_events,
    uint64_t instruction_count,
    uint64_t row_count,
    uint64_t event_time_base,
    cudaStream_t stream
) {
    if (instruction_count == 0u) {
        return cudaSuccess;
    }
    if (d_instructions == nullptr || d_events == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (row_count != 0u && d_rows == nullptr) {
        return cudaErrorInvalidValue;
    }

    dim3 grid;
    dim3 block;
    cudaError_t err = launch_shape(instruction_count, &grid, &block);
    if (err != cudaSuccess) {
        return err;
    }

    prismcode_cuda_kernels::prismcode_event_integrate_kernel<<<grid, block, 0, stream>>>(
        d_instructions,
        d_rows,
        d_events,
        instruction_count,
        row_count,
        event_time_base
    );
    return cudaGetLastError();
}
