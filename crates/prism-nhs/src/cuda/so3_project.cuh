// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / RECT-3.1.c — SO(3) Projection Kernel + ContactShellTile
//                          (4-plane, WMMA tf32, 1280 B / align(128))
// ═══════════════════════════════════════════════════════════════════════
//
// Per the Production Architecture mandate Phase 1 Deliverable 1.2 and
// the RECT-3.1.c Tensor Core mandate (operator directives 2026-04-29).
// Consumes RichSpike clusters and writes one ContactShellTile per
// cluster. The tile carries FOUR independent SH expansions — Geometry,
// Causality, Thermodynamics, Chemistry — preserving the 2+2+2+2
// Entangled Graph plane separation through the Tensor Core reduction.
//
// **Why 1280 B / align(128)**: each plane has its own 64-float a_lm
// buffer (256 B = 4 × 16-float WMMA fragment columns) and 8-float
// power spectrum (32 B). 4 planes × 288 B = 1152 B for SH state.
// Header (16) + AABB (32) + lossless tag aggregates (32) + counters
// (16) + padding to 10×128 = 1280 B total.
//
// **Per-plane weights** (applied in shared memory before WMMA):
//   - Geometry      (G): w = 1
//   - Causality     (C): w = |causal_lag|
//   - Thermodynamics(T): w = water_density
//   - Chemistry     (H): w = popcount(chem_flags)
//
// **WMMA pipeline**:
//   - Fragment shape m=16, n=16, k=8 with tf32 inputs / fp32 accum.
//   - "Ones" A fragment + data B fragment → C[0][j] = column sum.
//   - Bank-padded shared memory (stride = 49 floats, +1 col pad).
//   - Per-plane × per-col-group reduction (4 planes × 3 col-groups =
//     12 fragment loops per spike tile of 16).
//   - Explicit `__syncwarp()` before/after every wmma::* call.
//   - Explicit f32 → tf32 down-conversion via __float_to_tf32 prior
//     to shared-memory store, so the precision loss is visible to
//     anyone reading the kernel rather than buried in load_matrix_sync.
//
// **Lossless tag propagation**: `agg_spike_source`, `agg_origin_phase`,
// `agg_chem_flags` are populated via shared-memory `atomicOr` over
// every spike in the cluster. No metadata is shaved during the
// Morton/LBVH/SO(3) chain.
//
// **Rotational invariance** (G11): each plane's `C_l = Σ_m |a_lm|²`
// is invariant under SO(3) rotations of the spike cloud. Verified by
// the Rust-side G11 test for the Geometry plane within the
// rigorously-bounded tf32 tolerance (see `so3_project.rs` test
// `g11_rotation_invariance` and the error-bound block in its docs).
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_SO3_PROJECT_CUH
#define PRISM_NHS_SO3_PROJECT_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "rich_spike.cuh"
#include "sh_basis.cuh"

namespace prism_nhs { namespace so3_project {

using prism_nhs::rich_spike::RichSpike;
using prism_nhs::sh_basis::LMAX;
using prism_nhs::sh_basis::N_COEFFS;

// ─────────────────────────────────────────────────────────────────────
// ContactShellTile — 1280-byte, 128-byte-aligned execution tile.
//
// LAYOUT CONTRACT (FFI-stable):
//   sizeof(ContactShellTile)  == 1280 bytes (10 × 128)
//   alignof(ContactShellTile) == 128 bytes
//
//   Offset  Field                  Size  Notes
//   ─────────────────────────────────────────────────────────────
//   0       phase                  4     CCNS phase tag
//   4       stream_id              4     Producing stream
//   8       cluster_id             4     int32 (matches RichSpike)
//   12      frame                  4     Frame index
//
//   16      geo_alm[64]            256   Plane G a_lm; 36 valid + 28 pad
//   272     geo_power_spectrum[8]  32    Plane G C_l; 6 valid + 2 pad
//
//   304     caus_alm[64]           256   Plane C a_lm
//   560     caus_power_spectrum[8] 32    Plane C C_l
//
//   592     therm_alm[64]          256   Plane T a_lm
//   848     therm_power_spectrum[8] 32   Plane T C_l
//
//   880     chem_alm[64]           256   Plane H a_lm
//   1136    chem_power_spectrum[8] 32    Plane H C_l
//
//   1168    aabb_min[4]            16    xyz + 1 pad
//   1184    aabb_max[4]            16    xyz + 1 pad
//
//   1200    agg_spike_source       4     bitOR of every RichSpike.spike_source
//   1204    agg_origin_phase       4     bitOR of every RichSpike.origin_phase
//   1208    agg_chem_flags         4     bitOR of every RichSpike.chem_flags
//   1212    agg_pad                4     keeps the next field 4-aligned
//
//   1216    sum_w_geo              4     Σ w_G  (= spike_count)
//   1220    sum_w_caus             4     Σ |causal_lag|
//   1224    sum_w_therm            4     Σ water_density
//   1228    sum_w_chem             4     Σ popcount(chem_flags)
//
//   1232    spike_count            4     N spikes
//   1236    adjudication_code      4     Filled by pre_rank
//   1240    reserved[2]            8     Future-use telemetry
//
//   1248    _pad[32]               32    Pad to 1280 B (10 × 128)
//   ─────────────────────────────────────────────────────────────
//   Total                          1280
//
// Plane indexing (used in kernel + Rust accessors):
//   PLANE_GEO    = 0
//   PLANE_CAUS   = 1
//   PLANE_THERM  = 2
//   PLANE_CHEM   = 3
// ─────────────────────────────────────────────────────────────────────

constexpr int N_PLANES = 4;
constexpr int PLANE_GEO   = 0;
constexpr int PLANE_CAUS  = 1;
constexpr int PLANE_THERM = 2;
constexpr int PLANE_CHEM  = 3;

struct alignas(128) ContactShellTile {
    // Header (16 B)
    uint32_t phase;
    uint32_t stream_id;
    int32_t  cluster_id;
    uint32_t frame;

    // Plane G (Geometry)
    float    geo_alm[64];
    float    geo_power_spectrum[8];
    // Plane C (Causality)
    float    caus_alm[64];
    float    caus_power_spectrum[8];
    // Plane T (Thermodynamics)
    float    therm_alm[64];
    float    therm_power_spectrum[8];
    // Plane H (Chemistry)
    float    chem_alm[64];
    float    chem_power_spectrum[8];

    // AABB (xyz + pad).
    float    aabb_min[4];
    float    aabb_max[4];

    // Lossless aggregate provenance — bitwise OR over every spike.
    uint32_t agg_spike_source;
    uint32_t agg_origin_phase;
    uint32_t agg_chem_flags;
    uint32_t agg_pad;

    // Per-plane sum-of-weights. Useful for downstream normalisation
    // and for the F1 SWITCH adjudicator's empty-plane gate.
    float    sum_w_geo;
    float    sum_w_caus;
    float    sum_w_therm;
    float    sum_w_chem;

    // Counters / control.
    uint32_t spike_count;
    uint32_t adjudication_code;
    uint32_t reserved[2];

    // Padding to 10 × 128 = 1280 B.
    uint8_t  _pad[32];
};
static_assert(sizeof(ContactShellTile) == 1280,
    "ContactShellTile layout drift: not 1280 B");
static_assert(alignof(ContactShellTile) == 128,
    "ContactShellTile alignment drift: not 128 B");

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration.
//
// Caller protocol (unchanged from RECT-3.1.b):
//   1. `prism_sh_basis_init` once per process.
//   2. `prism_sh_basis_get_k_lm_dev_ptr` once per process.
//   3. Pack RichSpikes; build CSR cluster_offsets[N+1].
//   4. Allocate `n_clusters × ContactShellTile` device output.
//   5. Call `prism_so3_project_run`.
//
// The kernel writes one 1280-byte tile per cluster carrying the four
// SH plane decompositions, the AABB, the lossless aggregate tags, and
// the per-plane sum-of-weights.
// ─────────────────────────────────────────────────────────────────────

extern "C" {

/// Sentinel `0x53033` ("S0(3)"). Pinned by Rust-side
/// `link_probe_returns_sentinel` test.
uint32_t prism_so3_project_link_probe(void);

/// Run the WMMA-accelerated SO(3) projection kernel.
cudaError_t prism_so3_project_run(
    const RichSpike*  d_spikes,
    const uint32_t*   d_cluster_offsets,
    uint32_t          n_clusters,
    const float*      d_k_lm,
    ContactShellTile* d_tiles_out,
    uint32_t          frame_id,
    uint32_t          source_stream_id,
    cudaStream_t      stream
);

// ─────────────────────────────────────────────────────────────────────
// M1.2.20.C-A — Gradient Gasp Kernel + LUT host entry points.
// ─────────────────────────────────────────────────────────────────────

/// Populate the __constant__ d_residue_to_calpha[1024] lookup table
/// from the per-residue Cα atom index array on the host.  Stream-
/// ordered cudaMemcpyToSymbolAsync.  `n` clamped to 1024.  Sentinel
/// 0xFFFFFFFFu in unset entries causes the gasp kernel to pass the
/// spike through without displacement.
int prism_so3_set_residue_to_calpha(
    const uint32_t* host_table,
    uint32_t        n,
    void*           stream);

/// Legacy compatibility setter only. Not valid for concurrent production streams.
/// Production captured GASP receives an engine-owned live clock pointer instead.
int prism_so3_set_current_md_step(
    uint32_t        step,
    void*           stream);

/// Launch prism_apply_gradient_gasp_kernel.
/// Reads adjudicator FFI fields (gasp_gain_eta @136, force_burst_step
/// @140, d_dt @120) via byte-offset arithmetic; reads per-spike
/// residue_id, looks up Cα atom index, fetches f_anchor + m_anchor,
/// computes Δr = η_eff · Q_s · (f/m) · dt², writes the perturbed
/// RichSpike to d_spikes_out (struct-copy with x/y/z modified) AND
/// atomicAdds m·Δr into d_com_shift[3] for the post-pass Momentum
/// Guard.  Pass `d_com_shift == nullptr` to disable the COM accumulation
/// (e.g., legacy / Phase 1 calls without the Momentum Guard).
int prism_apply_gradient_gasp_launch(
    const void*  d_spikes_in,
    void*        d_spikes_out,
    const void*  d_forces,        /* [n_atoms × 3] f32 */
    const void*  d_masses,        /* [n_atoms]     f32 */
    const void*  adj_base,        /* InterferometricAdjudicatorFfi*  */
    void*        d_com_shift,     /* [3] f32 — Σ m·Δr accumulator, nullable */
    void*        d_total_mass,    /* [1] f32 — Σ m accumulator (Path Ω Option A), nullable */
    uint32_t     current_step,
    uint32_t     n_spikes,
    uint32_t     n_atoms,
    void*        stream);

/// Strict captured production ABI; pointer must remain live through graph teardown.
int prism_apply_gradient_gasp_clock_launch(
    const void* d_spikes_in, void* d_spikes_out,
    const void* d_forces, const void* d_masses, const void* adj_base,
    void* d_com_shift, void* d_total_mass,
    const uint32_t* d_stream_md_step, uint32_t n_spikes, uint32_t n_atoms,
    void* stream);

/// Compact production storage; raw metadata remains in the source population.
int prism_apply_gradient_gasp_positions_clock_launch(
    const void* d_spikes_in, void* d_positions_out,
    const void* d_forces, const void* d_masses, const void* adj_base,
    void* d_com_shift, void* d_total_mass,
    const uint32_t* d_stream_md_step, uint32_t n_spikes, uint32_t n_atoms,
    void* stream);
int prism_apply_com_positions_correction_launch(
    void* d_positions, const void* d_com_correction, uint32_t n_spikes, void* stream);
cudaError_t prism_so3_project_positions_run(
    const RichSpike* d_spikes, const float* d_positions,
    const uint32_t* d_cluster_offsets, uint32_t n_clusters,
    const float* d_k_lm, ContactShellTile* d_tiles_out,
    uint32_t frame_id, uint32_t source_stream_id, cudaStream_t stream);

/// **Path Ω Option A** — Single-thread post-pass kernel that reads
/// d_com_shift[3] and d_total_mass[1], computes the COM correction
/// vector `correction = Σ m·Δr / Σ m`, writes it to d_com_correction[3]
/// (consumed by `prism_apply_com_correction_kernel` downstream).  Also
/// writes adj.momentum_violation_flag = 1 when the correction
/// magnitude exceeds 1.0 Å (loose threshold; the strict 1e-4 Å gate
/// pre-Option-A was incompatible with any non-trivial gasp because
/// random-walk Σ scales as √n_spikes × Δr × m, ≫ 1e-4 for 500k spikes).
/// The new threshold catches genuinely unphysical kicks (>1 Å COM
/// drift) without blocking the gasp's intended perturbation.
int prism_momentum_guard_check_launch(
    const void*  d_com_shift,         /* [3] f32 — Σ m·Δr */
    const void*  d_total_mass,        /* [1] f32 — Σ m */
    void*        d_com_correction,    /* [3] f32 — output: correction vector */
    void*        adj_base,            /* InterferometricAdjudicatorFfi* */
    void*        stream);

/// **Path Ω Option A** — Per-spike COM-correction pass.  Reads
/// d_com_correction[3] (computed by the post-pass momentum-guard kernel)
/// and subtracts it from each spike's (x, y, z) in d_spikes_inout.
/// Result: the perturbed manifold has Σ m·(pos_perturbed - pos_relaxed)
/// = 0 by construction, so the SO(3) KL between relaxed and perturbed
/// captures relative structural divergence (the discovery signal) and
/// not the global rigid drift the gradient gasp inherently produces.
int prism_apply_com_correction_launch(
    void*        d_spikes_inout,      /* [n_spikes] RichSpike — in-place modify */
    const void*  d_com_correction,    /* [3] f32 */
    uint32_t     n_spikes,
    void*        stream);

}  // extern "C"

}}  // namespace prism_nhs::so3_project


namespace prism_nhs { namespace so3_project {
extern "C" cudaError_t prism_so3_project_seed_run(
    const RichSpike*  d_spikes,
    const uint32_t*   d_cluster_offsets,
    uint32_t          n_clusters,
    const float*      d_k_lm,
    ContactShellTile* d_tiles_out,
    uint32_t          frame_id,
    uint32_t          source_stream_id,
    cudaStream_t      stream,
    prism_nhs::rich_spike::NativeV2SeedLayout seed_layout);
}}

namespace prism_nhs { namespace so3_project {
extern "C" cudaError_t prism_so3_project_positions_seed_run(
    const RichSpike* d_spikes, const float* d_positions,
    const uint32_t* d_cluster_offsets, uint32_t n_clusters,
    const float* d_k_lm, ContactShellTile* d_tiles_out,
    uint32_t frame_id, uint32_t source_stream_id, cudaStream_t stream,
    prism_nhs::rich_spike::NativeV2SeedLayout seed_layout);
}}

namespace prism_nhs { namespace so3_project {
extern "C" int prism_apply_gradient_gasp_positions_seed_clock_launch(
    const void* d_spikes_in, void* d_spikes_out,
    const void* d_forces, const void* d_masses, const void* adj_base,
    void* d_com_shift, void* d_total_mass,
    const uint32_t* d_stream_md_step, uint32_t n_spikes, uint32_t n_atoms,
    void* stream,
    prism_nhs::rich_spike::NativeV2SeedLayout seed_layout);
}}
#endif  // PRISM_NHS_SO3_PROJECT_CUH
