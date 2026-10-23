#include "prismcode_cuda.h"

namespace prismcode_cuda_kernels {

__device__ uint64_t prismcode_nanl_bin_mix64(uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return value;
}

__global__ void prismcode_nanl_bin_pack_u64_kernel(
    const uint64_t* tokens,
    uint64_t* packed,
    uint64_t count
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }

    const uint64_t token = tokens[index] & 0x3ULL;
    const uint64_t class_bits = (tokens[index] >> 2u) & 0x3fULL;
    const uint64_t lane = index & 0xffULL;
    packed[index] = prismcode_nanl_bin_mix64(
        0x4e414e4c42494e31ULL ^ token ^ (class_bits << 8u) ^ (lane << 32u)
    );
}

}  // namespace prismcode_cuda_kernels
