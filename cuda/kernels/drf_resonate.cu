#include "prismcode_cuda_device.cuh"

#include <math.h>

namespace prismcode_cuda_kernels {

__global__ void prismcode_drf_resonate_kernel(
    const uint16_t* __restrict__ channel_vectors,
    const float* __restrict__ channel_weights,
    PrismcodeDrfState* __restrict__ states,
    uint64_t row_count,
    uint32_t channels_per_row
) {
    const uint64_t row_idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_idx >= row_count) {
        return;
    }

    double weighted_sum = 0.0;
    double weight_norm = 0.0;
    const uint64_t base = row_idx * static_cast<uint64_t>(channels_per_row);
    for (uint32_t channel = 0; channel < channels_per_row; ++channel) {
        const float weight = channel_weights == nullptr ? 1.0f : channel_weights[channel];
        weighted_sum += static_cast<double>(channel_vectors[base + channel]) * weight;
        weight_norm += fabs(static_cast<double>(weight));
    }

    const double denom = weight_norm > 0.0 ? 65535.0 * weight_norm : 1.0;
    const double resonance = weighted_sum / denom;

    PrismcodeDrfState state = states[row_idx];
    state.row_index = row_idx;
    state.resonance = resonance;
    if (state.threshold <= 0.0) {
        state.threshold = 0.5;
    }
    if (state.refractory < 0.0) {
        state.refractory = 0.0;
    }
    state.flags |= PRISMCODE_EVENT_FLAG_VALID;
    states[row_idx] = state;
}

}  // namespace prismcode_cuda_kernels
