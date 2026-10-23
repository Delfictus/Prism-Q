#ifndef PRISMCODE_CUDA_DEVICE_CUH
#define PRISMCODE_CUDA_DEVICE_CUH

#include "prismcode_cuda.h"

#include <cuda_runtime.h>
#include <stdint.h>

namespace prismcode_cuda_kernels {

__device__ __forceinline__ uint64_t prismcode_mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__device__ __forceinline__ uint64_t prismcode_hash_pair(uint64_t a, uint64_t b) {
    return prismcode_mix64(a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2)));
}

__device__ __forceinline__ uint32_t prismcode_saturating_u32(uint64_t value) {
    return value > 0xffffffffULL ? 0xffffffffu : static_cast<uint32_t>(value);
}

__device__ __forceinline__ uint32_t prismcode_event_kind(uint32_t opcode_code, uint32_t opcode_family) {
    if (opcode_code == PRISMCODE_OP_FIRE_DRF_EVENT) {
        return PRISMCODE_EVENT_KIND_DRF_FIRE;
    }
    if (opcode_code == PRISMCODE_OP_SUPPRESS_DRF_EVENT) {
        return PRISMCODE_EVENT_KIND_DRF_SUPPRESS;
    }
    if (opcode_code == PRISMCODE_OP_EMIT_SPIKE) {
        return PRISMCODE_EVENT_KIND_SPIKE;
    }
    if (opcode_code == PRISMCODE_OP_EMIT_RUNTIME_EVENT) {
        return PRISMCODE_EVENT_KIND_RUNTIME;
    }
    if (opcode_family == PRISMCODE_FAMILY_VALIDATION) {
        return PRISMCODE_EVENT_KIND_VALIDATION;
    }
    return PRISMCODE_EVENT_KIND_GENERIC;
}

__device__ __forceinline__ float prismcode_status_scale(uint32_t semantic_status) {
    switch (semantic_status) {
        case PRISMCODE_STATUS_ACTIVE:
            return 1.0f;
        case PRISMCODE_STATUS_PARTIAL:
            return 0.5f;
        case PRISMCODE_STATUS_PENDING:
            return 0.25f;
        default:
            return 0.0f;
    }
}

}  // namespace prismcode_cuda_kernels

#endif
