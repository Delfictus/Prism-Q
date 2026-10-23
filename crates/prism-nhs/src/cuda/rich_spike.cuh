// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Continuous Learning Architecture — RichSpike (64-byte) Schema
// ═══════════════════════════════════════════════════════════════════════
//
// Per Continuous Learning Architecture mandate §1.1 (operator
// directive 2026-04-29). Replaces the legacy 16-byte spike with a
// "Rich Spike" 64-byte schema that captures the full state of the
// neuromorphic event generator at the moment of emission.
//
// **Why 64 bytes**: aligns to the Blackwell sm_120 cache line.
// Prevents partial line writes; a single LDG.E.128 pulls a quarter
// of the spike (16 bytes / "plane") into registers in one
// transaction.
//
// **Why aligned(64)**: prevents unaligned-access traps under the
// PTX trap policy (Rectification §2). Unaligned access on Blackwell
// is a hardware fault under strict alignment policies; the 64-byte
// alignment guarantees every spike read/write is line-aligned.
//
// **Layout** (4 × 16-byte planes):
//
//   Plane 1 — Spatiotemporal (16 B):
//     float    x, y, z          (12 B)
//     uint32_t t_frame          (4 B)  Frame index within CCNS phase.
//
//   Plane 2 — Thermodynamic Gradients (16 B):
//     float    water_density    (4 B)  Local rho_w.
//     float    wd_change        (4 B)  d(rho_w)/dt — solvation flux.
//     float    vib_energy       (4 B)  Local vibrational stress.
//     uint32_t intensity_packed (4 B)  Top 8 bits = percentile,
//                                       bottom 24 bits = intensity.
//
//   Plane 3 — Causal & Neuromorphic Metadata (16 B):
//     int32_t  residue_id       (4 B)  Authoritative anchor (per
//                                       Mandate §M2 identity tuple).
//     int32_t  cluster_id       (4 B)  LBVH-assigned cluster
//                                       (atomic assignment).
//     float    causal_lag       (4 B)  Cross-correlation lag.
//     uint32_t n_excited        (4 B)  Multi-neuron excitation count.
//
//   Plane 4 — Provenance & Chemical Tags (16 B):
//     uint32_t origin_phase     (4 B)  CCNS phase (Cold/Heat/Warm/Cool).
//     uint32_t spike_source     (4 B)  LIF / UV / EFP / LADD / COFIRE.
//     uint32_t chem_flags       (4 B)  Bit-packed pharmacophore.
//     float    kinetic_delta    (4 B)  ΔE excursion value.
//
// **Composite sort key**: per Mandate §1.3, the 128-bit Karras-tree
// sort key is `[30-bit Morton | 32-bit residue_id | 66-bit feature_hash]`.
// The feature_hash is computed by `prism_rich_spike_feature_hash`
// below — a `__host__ __device__` helper so CPU reference and GPU
// kernel produce bit-equivalent hashes (V3 contract).
//
// Compilation: nvcc -arch=sm_120 (no special flags; alignas(64) is
// standard C++11+).
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_RICH_SPIKE_CUH
#define PRISM_NHS_RICH_SPIKE_CUH

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
  #define PRISM_RS_HD __host__ __device__ __forceinline__
#else
  #define PRISM_RS_HD inline
#endif

namespace prism_nhs { namespace rich_spike {

// ─────────────────────────────────────────────────────────────────────
// RichSpike — 64-byte cache-line-aligned event record.
//
// LAYOUT CONTRACT (FFI-stable):
//   sizeof(RichSpike)  == 64 bytes (static_assert below)
//   alignof(RichSpike) == 64 bytes (static_assert below)
//
// Rust mirror is `#[repr(C, align(64))]` in
// `crates/prism-nhs/src/rich_spike.rs`. Layout drift breaks FFI.
// ─────────────────────────────────────────────────────────────────────
struct alignas(64) RichSpike {
    // Plane 1 — Spatiotemporal Coordinates (16 B)
    float    x;
    float    y;
    float    z;
    uint32_t t_frame;

    // Plane 2 — Thermodynamic Gradients (16 B)
    float    water_density;
    float    wd_change;
    float    vib_energy;
    uint32_t intensity_packed;

    // Plane 3 — Causal & Neuromorphic Metadata (16 B)
    int32_t  residue_id;
    int32_t  cluster_id;
    float    causal_lag;
    uint32_t n_excited;

    // Plane 4 — Provenance & Chemical Tags (16 B)
    uint32_t origin_phase;
    uint32_t spike_source;
    uint32_t chem_flags;
    float    kinetic_delta;
};
static_assert(sizeof(RichSpike) == 64, "RichSpike layout drift: not 64 B");
static_assert(alignof(RichSpike) == 64, "RichSpike alignment drift: not 64 B");

// Additive GPU-only view. Original RichSpike ABI/archive remains64B.
struct NativeV2SeedLayout {
    uint32_t format, causal_lag_bits, chem_flags, reserved;
};
struct NativeV2Seed32 {
    float x,y,z,water_density,wd_change,vib_energy;
    uint32_t intensity_packed;
    int16_t residue_id;
    uint16_t phase_source;
};
static_assert(sizeof(NativeV2SeedLayout)==16,"V2 seed descriptor ABI");
static_assert(sizeof(NativeV2Seed32)==32,"V2 seed view ABI");
PRISM_RS_HD bool valid_native_v2_seed_layout(NativeV2SeedLayout layout) {
    return layout.format<=1u && layout.reserved==0u;
}
PRISM_RS_HD RichSpike load_native_v2_seed(const RichSpike* raw,uint32_t i,NativeV2SeedLayout layout) {
    if(layout.format==0u) return raw[i];
    const NativeV2Seed32 c=reinterpret_cast<const NativeV2Seed32*>(raw)[i];
    RichSpike s{};
    s.x=c.x;s.y=c.y;s.z=c.z;s.water_density=c.water_density;s.wd_change=c.wd_change;
    s.vib_energy=c.vib_energy;s.intensity_packed=c.intensity_packed;
    s.residue_id=static_cast<int32_t>(c.residue_id);
    s.origin_phase=static_cast<uint32_t>(c.phase_source&1023u);
    s.spike_source=static_cast<uint32_t>(c.phase_source>>10);
    // Bit-preserving memcpy is valid in CUDA device and ordinary C++ compilation.
    memcpy(&s.causal_lag,&layout.causal_lag_bits,sizeof(float));
    s.chem_flags=layout.chem_flags;
    // These fields are absent from the GPU-consumption view, not absent from
    // the archived RichSpike. Current SO3/GASP/Ghost consumers never read them.
    return s;
}

// ─────────────────────────────────────────────────────────────────────
// Sentinel / unset values for RichSpike fields.
//
// `RICH_SPIKE_UNCLUSTERED_ID` matches `UNCLUSTERED_CLUSTER_ID`
// from `spike_to_cluster_4d.cuh` semantically; we use the signed
// int32 form here because `cluster_id` is `int32_t` (allows -1
// sentinels in addition to 0xFFFFFFFF).
// ─────────────────────────────────────────────────────────────────────
constexpr int32_t  RICH_SPIKE_UNCLUSTERED_ID = -1;
constexpr int32_t  RICH_SPIKE_UNRESOLVED_RESIDUE = -1;

// ─────────────────────────────────────────────────────────────────────
// Intensity-pack helpers (8-bit percentile + 24-bit intensity).
//
// The packed layout (MSB → LSB):
//   bits 31..24 : 8-bit percentile rank (0..255)
//   bits 23..0  : 24-bit intensity value (0..16777215)
//
// Both helpers are `__host__ __device__` so CPU + GPU produce
// identical packed/unpacked values.
// ─────────────────────────────────────────────────────────────────────
PRISM_RS_HD uint32_t prism_rich_spike_pack_intensity(uint32_t percentile_8, uint32_t intensity_24) {
    const uint32_t p = percentile_8 & 0xFFu;
    const uint32_t i = intensity_24 & 0x00FFFFFFu;
    return (p << 24) | i;
}
PRISM_RS_HD uint32_t prism_rich_spike_unpack_percentile(uint32_t packed) {
    return (packed >> 24) & 0xFFu;
}
PRISM_RS_HD uint32_t prism_rich_spike_unpack_intensity(uint32_t packed) {
    return packed & 0x00FFFFFFu;
}

// ─────────────────────────────────────────────────────────────────────
// 64-bit feature hash for RichSpike.
//
// Per Mandate §1.3: the 128-bit composite Karras sort key is
// `[30-bit Morton | 32-bit residue_id | 66-bit feature_hash]`.
// The 66-bit hash space is more than the 64-bit hash this helper
// produces — the high 2 bits of the conceptual 66-bit hash come
// from a separate spatial-collision discriminator (chain_id +
// atom_index) handled at sort-key assembly time. This helper
// produces the 64-bit *feature*-only component.
//
// The hash blends every non-spatial field of the RichSpike using
// SplitMix64 — a fast, well-distributed mixer that is bit-equivalent
// across CPU and GPU (no platform-specific math).
//
// V3 contract: `prism_rich_spike_feature_hash` is `__host__
// __device__` and the Rust port `cpu_rich_spike_feature_hash` calls
// the same algorithm. Any divergence is a bug.
// ─────────────────────────────────────────────────────────────────────
PRISM_RS_HD uint64_t prism_splitmix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);
    return x;
}

PRISM_RS_HD uint64_t prism_rich_spike_feature_hash(const RichSpike& s) {
    // Mix every non-spatial field. The spatial (x, y, z) is
    // discriminated by the Morton code in the upper 30 bits of the
    // composite key, so we exclude it from the feature hash to keep
    // hash collisions tied to genuine biophysical similarity rather
    // than spatial coincidence.
    uint64_t h = 0xCBF29CE484222325ULL;  // FNV offset basis as starter
    h ^= prism_splitmix64(static_cast<uint64_t>(s.t_frame));
    // Plane 2 — bitcast floats to u32 so the hash is bit-exact on
    // CPU and GPU regardless of FPU rounding mode.
    union { float f; uint32_t u; } cvt;
    cvt.f = s.water_density;     h ^= prism_splitmix64(static_cast<uint64_t>(cvt.u));
    cvt.f = s.wd_change;         h ^= prism_splitmix64(static_cast<uint64_t>(cvt.u));
    cvt.f = s.vib_energy;        h ^= prism_splitmix64(static_cast<uint64_t>(cvt.u));
    h ^= prism_splitmix64(static_cast<uint64_t>(s.intensity_packed));
    // Plane 3
    h ^= prism_splitmix64(static_cast<uint64_t>(static_cast<uint32_t>(s.residue_id)));
    h ^= prism_splitmix64(static_cast<uint64_t>(static_cast<uint32_t>(s.cluster_id)));
    cvt.f = s.causal_lag;        h ^= prism_splitmix64(static_cast<uint64_t>(cvt.u));
    h ^= prism_splitmix64(static_cast<uint64_t>(s.n_excited));
    // Plane 4
    h ^= prism_splitmix64(static_cast<uint64_t>(s.origin_phase));
    h ^= prism_splitmix64(static_cast<uint64_t>(s.spike_source));
    h ^= prism_splitmix64(static_cast<uint64_t>(s.chem_flags));
    cvt.f = s.kinetic_delta;     h ^= prism_splitmix64(static_cast<uint64_t>(cvt.u));
    // Final mix to bury any remaining linearity.
    return prism_splitmix64(h);
}

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration — link probe.
// (Kernels that consume RichSpike land in CLA-1b alongside the
// Morton encoder upgrade.)
// ─────────────────────────────────────────────────────────────────────
extern "C" {

/// Sentinel `0x6164` ("ad" — for "rich-spike"). Pinned by Rust-side
/// `link_probe_returns_sentinel` test.
uint32_t prism_rich_spike_link_probe(void);

}  // extern "C"

}}  // namespace prism_nhs::rich_spike

#endif  // PRISM_NHS_RICH_SPIKE_CUH
