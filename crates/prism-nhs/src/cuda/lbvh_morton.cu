// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / LBVH — Morton 30-bit encoder (CUDA implementation)
// ═══════════════════════════════════════════════════════════════════════
//
// Phase 1 of the LBVH lane: encode every spike position to a 30-bit
// Morton code packed in a u32. The output is the input to the
// Karras 2012 parallel binary-radix-tree builder (subsequent commit)
// which sorts by Morton code and constructs the LBVH topology.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
//
// ═══════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cstdint>

#include "lbvh_morton.cuh"

namespace prism_nhs { namespace lbvh {

// ═══════════════════════════════════════════════════════════════════════
// __global__ kernel: per-position 30-bit Morton encoding
// ═══════════════════════════════════════════════════════════════════════

/// One thread per position. Reads three floats (x, y, z), normalizes
/// to the unit cube via `bbox`, quantizes each axis to 10 bits, and
/// emits the interleaved Morton code at `d_codes_out[tid]`.
///
/// The math is the canonical helper chain from the .cuh:
/// `prism_morton_quantize_10bit` (per-axis quantize) →
/// `prism_morton_30bit_encode` (interleave). Both helpers are
/// `__host__ __device__` and bit-equivalent on CPU and GPU, so the
/// V3-style "single source of truth" guarantee holds: a CPU
/// reference written in Rust that calls the same Rust port of these
/// helpers produces bit-exact identical output.
__global__ void prism_morton_30bit_encode_kernel(
    const float* __restrict__ d_positions,
    uint32_t                  num_positions,
    MortonBboxParams          bbox,
    uint32_t* __restrict__    d_codes_out
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_positions) return;

    const float px = d_positions[3u * tid + 0u];
    const float py = d_positions[3u * tid + 1u];
    const float pz = d_positions[3u * tid + 2u];

    const uint32_t qx = prism_morton_quantize_10bit(px, bbox.min[0], bbox.max[0]);
    const uint32_t qy = prism_morton_quantize_10bit(py, bbox.min[1], bbox.max[1]);
    const uint32_t qz = prism_morton_quantize_10bit(pz, bbox.min[2], bbox.max[2]);

    d_codes_out[tid] = prism_morton_30bit_encode(qx, qy, qz);
}

// ═══════════════════════════════════════════════════════════════════════
// extern "C" host orchestration
// ═══════════════════════════════════════════════════════════════════════

extern "C" {

uint32_t prism_lbvh_link_probe(void) {
    // Sentinel: Rust-side `lbvh::link_probe()` test pins this value.
    // Confirms (a) the static archive linked correctly, (b) the FFI
    // ABI is round-tripping, (c) the gpu feature is enabled.
    return 0xC0DEu;
}

cudaError_t prism_morton_30bit_encode_run(
    const float* d_positions,
    uint32_t num_positions,
    const MortonBboxParams* h_bbox,
    cudaStream_t stream,
    uint32_t* d_codes_out
) {
    if (num_positions == 0u) {
        // Vacuous success: no positions to encode. Caller's output
        // buffer is left as-is. Avoiding a zero-block kernel launch
        // keeps the host round-trip cost at zero for empty inputs.
        return cudaSuccess;
    }

    constexpr uint32_t THREADS_PER_BLOCK = 256u;
    const uint32_t blocks =
        (num_positions + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;

    // Pass `*h_bbox` by value into the kernel parameter — the host
    // pointer's lifetime is bounded by this call (kernel argument
    // copy happens before the kernel returns control to the host).
    prism_morton_30bit_encode_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_positions,
        num_positions,
        *h_bbox,
        d_codes_out
    );
    return cudaGetLastError();
}

}  // extern "C"

}}  // namespace prism_nhs::lbvh
