#include "prismcode_cuda_device.cuh"

namespace prismcode_cuda_kernels {

__global__ void prismcode_event_integrate_kernel(
    const PrismcodeInstruction* __restrict__ instructions,
    const PrismcodeMemoryRow* __restrict__ rows,
    PrismcodeEventRecord* __restrict__ events,
    uint64_t instruction_count,
    uint64_t row_count,
    uint64_t event_time_base
) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= instruction_count) {
        return;
    }

    const PrismcodeInstruction ins = instructions[idx];
    const bool row_valid = ins.row_index < row_count;
    const PrismcodeMemoryRow row = row_valid ? rows[ins.row_index] : PrismcodeMemoryRow{};
    const float status_scale = prismcode_status_scale(ins.semantic_status);
    const float resonance = row_valid
        ? static_cast<float>(
            0.5 * row.drf_resonance_fusion_score +
            0.25 * row.nrf_resonance_score +
            0.25 * row.dstw_temporal_waveform_score
        )
        : 0.0f;

    PrismcodeEventRecord event = {};
    event.row_index = ins.row_index;
    event.event_time_tick = event_time_base + idx;
    event.source_hash_lo = ins.payload_hash_lo;
    event.source_hash_hi = ins.payload_hash_hi;
    event.intensity = status_scale * (1.0f + resonance);
    event.resonance = resonance;
    event.opcode_code = ins.opcode_code;
    event.event_kind = prismcode_event_kind(ins.opcode_code, ins.opcode_family);
    event.semantic_status = ins.semantic_status;
    event.flags = PRISMCODE_EVENT_FLAG_INTEGRATED;
    event.flags |= row_valid ? PRISMCODE_EVENT_FLAG_VALID : PRISMCODE_EVENT_FLAG_SOURCE_INVALID;

    events[idx] = event;
}

}  // namespace prismcode_cuda_kernels
