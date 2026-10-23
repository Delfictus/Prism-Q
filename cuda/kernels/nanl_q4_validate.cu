#include "prismcode_cuda.h"

namespace prismcode_cuda_kernels {

__device__ uint64_t prismcode_q4_mix64(uint64_t hash, uint8_t value, uint64_t index) {
    hash ^= static_cast<uint64_t>(value) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    hash ^= index * 0x100000001b3ULL;
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53ULL;
    hash ^= hash >> 33;
    return hash;
}

__global__ void prismcode_nanl_q4_validate_kernel(
    const uint8_t* q4_bytes,
    uint64_t* per_block_hashes,
    uint32_t* per_block_failures,
    uint64_t len
) {
    __shared__ uint64_t hash_scratch[PRISMCODE_CUDA_THREADS_PER_BLOCK];
    __shared__ uint32_t fail_scratch[PRISMCODE_CUDA_THREADS_PER_BLOCK];

    const unsigned int local = threadIdx.x;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + local;
    uint64_t hash = 0x514456414c494431ULL;
    uint32_t fail = 0u;

    if (index < len) {
        const uint8_t packed = q4_bytes[index];
        // Packed NANL-Q4 bytes contain four two-bit quaternary symbols.
        // Any byte is syntactically valid at this layer; constraint scoring
        // remains a host-side deterministic policy check.
        hash = prismcode_q4_mix64(hash, packed, index);
        fail = 0u;
    }

    hash_scratch[local] = hash;
    fail_scratch[local] = fail;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            hash_scratch[local] ^= hash_scratch[local + stride];
            fail_scratch[local] += fail_scratch[local + stride];
        }
        __syncthreads();
    }

    if (local == 0u) {
        per_block_hashes[blockIdx.x] = hash_scratch[0];
        per_block_failures[blockIdx.x] = fail_scratch[0];
    }
}

}  // namespace prismcode_cuda_kernels
