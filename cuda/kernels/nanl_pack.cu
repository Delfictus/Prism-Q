#include "prismcode_cuda_device.cuh"

namespace prismcode_cuda_kernels {

__global__ void prismcode_nanl_pack_kernel(
    const PrismcodeMemoryRow* __restrict__ rows,
    PrismcodeNanlPackedRow* __restrict__ out,
    uint64_t row_count
) {
    const uint64_t idx = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= row_count) {
        return;
    }

    const PrismcodeMemoryRow row = rows[idx];
    const bool has_token = row.nanl_token_u64 != 0ULL;
    const bool binding_complete = row.missing_binding_field_count == 0;
    const uint64_t event_count =
        row.nrf_event_count_compiled + row.neuro_event_count_compiled;

    PrismcodeNanlPackedRow packed = {};
    packed.row_index = row.row_index;
    packed.nanl_token_u64 = row.nanl_token_u64;
    packed.packed_fingerprint_lo = prismcode_hash_pair(
        row.nanl_token_u64,
        row.row_hash_lo ^ row.row_index
    );
    packed.packed_fingerprint_hi = prismcode_hash_pair(
        row.row_hash_hi,
        row.nanl_token_u64 ^ event_count
    );
    packed.validity_flags = (has_token && binding_complete)
        ? PRISMCODE_EVENT_FLAG_VALID
        : PRISMCODE_EVENT_FLAG_SOURCE_INVALID;
    packed.event_hint = prismcode_saturating_u32(event_count);
    packed.instruction_opcode_hint = PRISMCODE_OP_LOAD_NANL_STRAND;

    out[idx] = packed;
}

}  // namespace prismcode_cuda_kernels
