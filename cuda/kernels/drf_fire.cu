#include "prismcode_cuda_device.cuh"

namespace prismcode_cuda_kernels {

__global__ void prismcode_drf_fire_kernel(
    PrismcodeDrfState* __restrict__ states,
    PrismcodeEventRecord* __restrict__ events,
    uint64_t row_count,
    double threshold_floor,
    double refractory_period,
    uint64_t event_time_tick
) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= row_count) {
        return;
    }

    PrismcodeDrfState state = states[idx];
    const double threshold = state.threshold > threshold_floor ? state.threshold : threshold_floor;
    const bool can_fire = state.refractory <= 0.0;
    const bool fired = can_fire && state.resonance >= threshold;

    PrismcodeEventRecord event = {};
    event.row_index = state.row_index;
    event.event_time_tick = event_time_tick;
    event.source_hash_lo = prismcode_hash_pair(state.row_index, static_cast<uint64_t>(state.fire_count));
    event.source_hash_hi = prismcode_hash_pair(event.source_hash_lo, static_cast<uint64_t>(event_time_tick));
    event.resonance = static_cast<float>(state.resonance);
    event.semantic_status = PRISMCODE_STATUS_ACTIVE;
    event.flags = PRISMCODE_EVENT_FLAG_VALID;

    if (fired) {
        state.fire_count += 1u;
        state.refractory = refractory_period;
        state.last_fire_intensity = static_cast<float>(state.resonance);
        event.intensity = state.last_fire_intensity;
        event.opcode_code = PRISMCODE_OP_FIRE_DRF_EVENT;
        event.event_kind = PRISMCODE_EVENT_KIND_DRF_FIRE;
        event.flags |= PRISMCODE_EVENT_FLAG_FIRED;
    } else {
        if (state.refractory > 0.0) {
            state.refractory -= 1.0;
            if (state.refractory < 0.0) {
                state.refractory = 0.0;
            }
        }
        event.intensity = 0.0f;
        event.opcode_code = PRISMCODE_OP_SUPPRESS_DRF_EVENT;
        event.event_kind = PRISMCODE_EVENT_KIND_DRF_SUPPRESS;
        event.flags |= PRISMCODE_EVENT_FLAG_SUPPRESSED;
    }

    states[idx] = state;
    events[idx] = event;
}

}  // namespace prismcode_cuda_kernels
