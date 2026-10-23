#include "prismcode_cuda.h"

#include "../kernels/tensor_ops.cu"

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

bool tensor_op_requires_rhs(uint32_t op) {
    return op == PRISMCODE_TENSOR_OP_ADD ||
           op == PRISMCODE_TENSOR_OP_MUL ||
           op == PRISMCODE_TENSOR_OP_AXPBY;
}

}  // namespace

extern "C" cudaError_t prismcode_cuda_tensor_ops(
    const float* d_lhs,
    const float* d_rhs,
    float* d_out,
    PrismcodeTensorOpParams params,
    cudaStream_t stream
) {
    if (params.element_count == 0u) {
        return cudaSuccess;
    }
    if (d_lhs == nullptr || d_out == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (tensor_op_requires_rhs(params.op) && d_rhs == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (params.op > PRISMCODE_TENSOR_OP_CLAMP) {
        return cudaErrorInvalidValue;
    }
    if (params.op == PRISMCODE_TENSOR_OP_CLAMP && params.clamp_min > params.clamp_max) {
        return cudaErrorInvalidValue;
    }

    dim3 grid;
    dim3 block;
    cudaError_t err = launch_shape(params.element_count, &grid, &block);
    if (err != cudaSuccess) {
        return err;
    }

    prismcode_cuda_kernels::prismcode_tensor_ops_kernel<<<grid, block, 0, stream>>>(
        d_lhs,
        d_rhs,
        d_out,
        params
    );
    return cudaGetLastError();
}
