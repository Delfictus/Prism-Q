#include "prismcode_cuda_device.cuh"

namespace prismcode_cuda_kernels {

__global__ void prismcode_tensor_ops_kernel(
    const float* __restrict__ lhs,
    const float* __restrict__ rhs,
    float* __restrict__ out,
    PrismcodeTensorOpParams params
) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= params.element_count) {
        return;
    }

    const float x = lhs[idx];
    const float y = rhs == nullptr ? 0.0f : rhs[idx];
    float value = x;

    switch (params.op) {
        case PRISMCODE_TENSOR_OP_ADD:
            value = params.alpha * x + params.beta * y + params.gamma;
            break;
        case PRISMCODE_TENSOR_OP_MUL:
            value = params.alpha * x * y + params.gamma;
            break;
        case PRISMCODE_TENSOR_OP_AXPBY:
            value = params.alpha * x + params.beta * y;
            break;
        case PRISMCODE_TENSOR_OP_RELU:
            value = x > 0.0f ? x : 0.0f;
            break;
        case PRISMCODE_TENSOR_OP_CLAMP:
            value = x < params.clamp_min ? params.clamp_min : x;
            value = value > params.clamp_max ? params.clamp_max : value;
            break;
        case PRISMCODE_TENSOR_OP_COPY:
        default:
            value = x;
            break;
    }

    out[idx] = value;
}

}  // namespace prismcode_cuda_kernels
