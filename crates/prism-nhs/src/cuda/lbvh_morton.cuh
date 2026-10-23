// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / LBVH — Morton 30-bit encoder shared header
// ═══════════════════════════════════════════════════════════════════════
//
// Shared header for the LBVH lane's Morton-encoding kernel. Defines:
//
//   * `MortonBboxParams` — FFI-stable POD that mirrors the Rust-side
//     bbox parameters byte-for-byte. Layouts MUST match (Rust side
//     uses `#[repr(C)]`).
//
//   * `prism_morton_quantize_10bit` — canonical `__host__ __device__`
//     position-to-quantized-index helper. Both the GPU kernel and any
//     CPU reference call this single function, so bit-exact equivalence
//     is by construction (M1.2 contract §4f, V3-style verification).
//
//   * `prism_morton_30bit_encode` — canonical `__host__ __device__`
//     bit-interleaving helper. Same dual-callable pattern as the
//     quantize helper.
//
//   * `prism_expand_bits_30` — internal helper exposed for
//     bit-equivalent CPU-side reference tests.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_LBVH_MORTON_CUH
#define PRISM_NHS_LBVH_MORTON_CUH

#include <cstdint>

#ifdef __CUDACC__
  #define PRISM_LBVH_HD __host__ __device__ __forceinline__
#else
  #define PRISM_LBVH_HD inline
#endif

namespace prism_nhs { namespace lbvh {

// ─────────────────────────────────────────────────────────────────────
// FFI-stable bbox parameters.
//
// LAYOUT CONTRACT — must match Rust-side `MortonBboxParams`
// byte-for-byte: 24 bytes (6 × f32), no padding.
// ─────────────────────────────────────────────────────────────────────
struct MortonBboxParams {
    float min[3];
    float max[3];
};
static_assert(sizeof(MortonBboxParams) == 24, "MortonBboxParams FFI drift");

// ─────────────────────────────────────────────────────────────────────
// Morton 30-bit encoding constants.
//
// 10 bits per axis × 3 axes = 30 bits per code, packed into a u32.
// MORTON_GRID_RES = 2^10 = 1024 cells per axis, max coord = 1023.
// ─────────────────────────────────────────────────────────────────────
constexpr uint32_t PRISM_MORTON_BITS_PER_AXIS = 10u;
constexpr uint32_t PRISM_MORTON_MAX_COORD = (1u << PRISM_MORTON_BITS_PER_AXIS) - 1u; // 1023

// ─────────────────────────────────────────────────────────────────────
// Quantize one float coordinate ∈ [bbox_min, bbox_max] to a 10-bit
// integer index ∈ [0, 1023]. Out-of-bbox inputs are clamped.
//
// Uses round-to-nearest-even semantics via `roundf`-equivalent. Values
// in `[0, 1]` of the unit-cube space are mapped onto the 1024-cell
// grid; the corner case at exactly `bbox_max` clamps to 1023 rather
// than wrapping to 1024.
//
// V3 contract: this function is the SINGLE source of quantization
// truth. CPU reference and GPU kernel both call it directly.
// ─────────────────────────────────────────────────────────────────────
PRISM_LBVH_HD uint32_t prism_morton_quantize_10bit(
    float coord,
    float bbox_min,
    float bbox_max
) {
    const float span = bbox_max - bbox_min;
    if (span <= 0.0f) {
        // Degenerate bbox along this axis (zero or inverted span).
        // Map every input to the grid origin so the encoder remains
        // total without producing NaN-derived indices.
        return 0u;
    }
    const float u = (coord - bbox_min) / span;
    // Clamp to [0, 1] to handle out-of-bbox inputs and avoid negative
    // or > 1023 quantized values.
    float uc = u;
    if (uc < 0.0f) uc = 0.0f;
    if (uc > 1.0f) uc = 1.0f;
    // Quantize to [0, 1023] with round-to-nearest. The +0.5f bias
    // before truncation realizes round-to-nearest for non-negative
    // values (which `uc` is, by clamp).
    uint32_t q = static_cast<uint32_t>(uc * 1023.0f + 0.5f);
    if (q > PRISM_MORTON_MAX_COORD) q = PRISM_MORTON_MAX_COORD;
    return q;
}

// ─────────────────────────────────────────────────────────────────────
// Insert two zero bits between each bit of a 10-bit value. Output is
// a 30-bit value with bits at positions 0, 3, 6, ..., 27.
//
// The classic "bit twiddling" sequence for Morton encoding from Karras
// 2012 / Lauterbach et al. 2009. Each step doubles the gap between
// adjacent input bits.
//
// Input MUST be in [0, 1023]; behavior on larger values is masked
// silently by the bitmask sequence (high bits beyond 10 are dropped).
// ─────────────────────────────────────────────────────────────────────
PRISM_LBVH_HD uint32_t prism_expand_bits_30(uint32_t v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v <<  8)) & 0x0300F00Fu;
    v = (v | (v <<  4)) & 0x030C30C3u;
    v = (v | (v <<  2)) & 0x09249249u;
    return v;
}

// ─────────────────────────────────────────────────────────────────────
// Encode three 10-bit coordinates `(qx, qy, qz)` into one 30-bit
// Morton code packed in a u32. Bit layout (LSB → MSB):
//
//   bit 0 = qx[0],  bit 1 = qy[0],  bit 2 = qz[0],
//   bit 3 = qx[1],  bit 4 = qy[1],  bit 5 = qz[1],
//   ...
//   bit 27 = qx[9], bit 28 = qy[9], bit 29 = qz[9].
//
// The high 2 bits (30, 31) are zero. The (Z, Y, X) interleave order is
// the standard convention used by all major LBVH implementations
// (Karras 2012, NVIDIA's LBVH sample, OptiX BVH).
//
// Input contract: `qx`, `qy`, `qz` ∈ [0, 1023]. Higher values produce
// undefined high-bit interactions but no side effects (silent mask).
// ─────────────────────────────────────────────────────────────────────
PRISM_LBVH_HD uint32_t prism_morton_30bit_encode(
    uint32_t qx,
    uint32_t qy,
    uint32_t qz
) {
    return prism_expand_bits_30(qx)
         | (prism_expand_bits_30(qy) << 1)
         | (prism_expand_bits_30(qz) << 2);
}

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration entry points
// ─────────────────────────────────────────────────────────────────────
//
// Definitions live in lbvh_morton.cu. Each function takes a
// `cudaStream_t` (no default-stream launches per M1.2 contract §4c)
// and returns a `cudaError_t` for caller-side error handling.

#include <cuda_runtime.h>

extern "C" {

/// Encode `num_positions` 3-D positions to 30-bit Morton codes.
///
/// Layout: `d_positions` is planar `[N][3]` (x0, y0, z0, x1, y1, z1, ...).
/// `d_codes_out` receives one `uint32_t` per position.
///
/// Returns `cudaSuccess` on success or the underlying CUDA error.
/// `num_positions == 0` is a no-op success (no kernel launched).
cudaError_t prism_morton_30bit_encode_run(
    const float* d_positions,
    uint32_t num_positions,
    const MortonBboxParams* h_bbox,  // host-passed (small POD, copied by value into kernel arg)
    cudaStream_t stream,
    uint32_t* d_codes_out
);

/// Probe function returning the canonical sentinel `0xC0DE`. Used by
/// the Rust-side link-probe test to confirm the static archive linked
/// correctly and the FFI ABI is round-tripping.
uint32_t prism_lbvh_link_probe(void);

}  // extern "C"

}}  // namespace prism_nhs::lbvh

#endif  // PRISM_NHS_LBVH_MORTON_CUH
