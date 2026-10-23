#include "prismcode_cuda.h"

namespace prismcode_cuda_kernels {

__global__ void prismcode_tensor_reduce_sum_kernel(
    const float* in,
    float* block_sums,
    uint64_t count
) {
    __shared__ float scratch[PRISMCODE_CUDA_THREADS_PER_BLOCK];
    const unsigned int local = threadIdx.x;
    const uint64_t global = static_cast<uint64_t>(blockIdx.x) * blockDim.x + local;
    scratch[local] = global < count ? in[global] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            scratch[local] += scratch[local + stride];
        }
        __syncthreads();
    }

    if (local == 0u) {
        block_sums[blockIdx.x] = scratch[0];
    }
}

}  // namespace prismcode_cuda_kernels
