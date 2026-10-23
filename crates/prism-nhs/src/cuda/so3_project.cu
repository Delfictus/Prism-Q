// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / RECT-3.1.c — SO(3) Projection Kernel (CUDA + WMMA tf32)
// ═══════════════════════════════════════════════════════════════════════
//
// See `so3_project.cuh` for layout + caller contract. This file
// implements the WMMA-refactored kernel mandated by the operator
// directive 2026-04-29 (RECT-3.1.c Tensor Core mandate).
//
// **Kernel summary (per block / per cluster, single warp)**:
//
//   Pass 1 — Centroid + AABB (scalar warp-shuffle).
//
//   Pass 2 — Per-tile projection over `TILE_ROWS = 16` spikes:
//     2.a  Lanes 0..15 evaluate Y_lm relative to the cluster
//          centroid; explicitly down-convert to tf32 via
//          `__float_to_tf32`; write 4 plane-weighted rows into
//          `s_plane_tile[plane][row][col]` (bank-padded shared mem,
//          stride 49 floats).
//     2.b  Lossless tag aggregation: `atomicOr` into
//          `s_agg_spike_source / origin_phase / chem_flags`.
//          Atomic per-plane `sum_w` accumulators.
//     2.c  __syncwarp() — full warp barrier before WMMA load.
//     2.d  For each (plane, col-group): load b_data fragment from the
//          tile (two 8-row slices, k=8 each), mma_sync into a
//          fresh fp32 c_acc, store back to a 16×17 staging buffer,
//          then lanes 0..15 each add the corresponding column-sum
//          into the per-plane accumulator `s_plane_acc`.
//          Each wmma::* is bracketed by __syncwarp().
//
//   Pass 3 — Tile write (lanes cooperate to populate the 1280-byte
//   ContactShellTile, including per-plane C_l = Σ_m |a_lm|², AABB,
//   lossless tags, and counters).
//
// **Bank-conflict-free shared memory**:
//   - Plane tile stride = N_COEFFS_PADDED + 1 = 49 floats. 49 mod 32
//     = 17 (coprime to 32 in the bank-stride sense), so consecutive
//     rows do NOT collide on the 32-bank, 4-byte-wide layout.
//   - WMMA staging buffer 16×17 (stride 17 floats) — same trick.
//
// **tf32 precision-aware accumulation**:
//   - Inputs explicitly down-converted via the inline PTX
//     `cvt.rna.tf32.f32` op (exposed by mma.h's __float_to_tf32).
//   - Accumulator fragment is `fragment<accumulator, 16, 16, 8, float>`
//     so the FMA chain runs in fp32 even though operands are tf32.
//   - The per-spike multiplication (Y_lm · weight) happens in fp32
//     and is THEN truncated to tf32 — the operator's "explicit
//     downcast" mandate is satisfied at the shared-memory boundary.
//
// **Warp-synchronous discipline**:
//   - Every wmma::load_matrix_sync, wmma::mma_sync, and
//     wmma::store_matrix_sync is preceded and followed by an
//     explicit `__syncwarp()`.
//   - We do NOT rely on the implicit Volta+ warp-synchronous behaviour
//     that the operator forbids.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════

#include "so3_project.cuh"

#include <cuda_runtime.h>
#include <mma.h>
#include <cstdint>
#include <cfloat>
#include <cstdio>  // Amendment 3.10 — kernel-level printf triage
#include <cmath>

namespace prism_nhs { namespace so3_project {

namespace wmma = nvcuda::wmma;

// ─────────────────────────────────────────────────────────────────────
// Compile-time geometry.
// ─────────────────────────────────────────────────────────────────────
constexpr int BLOCK_SIZE       = 32;   // single warp per cluster
constexpr int WARP_SIZE        = 32;
constexpr int TILE_ROWS        = 16;   // wmma m dim (= rows reduced per call pair)
constexpr int N_COEFFS_PADDED  = 48;   // 36 padded up to multiple of 16 for wmma n dim
constexpr int N_COL_GROUPS     = N_COEFFS_PADDED / 16;  // 3
// WMMA tf32 load_matrix_sync REQUIRES the leading dimension to be a
// multiple of 4 (16-byte / 128-bit row alignment for the underlying
// `ldmatrix.sync` PTX op). Bank padding must therefore be a multiple
// of 4. We use 4 so the stride 52 = 48+4 is 4-aligned AND has
// gcd(52/4, 32)=1 in the bank-stride sense (52 mod 32 = 20, and rows
// 0..7 hit banks 0/20/8/28/16/4/24/12 — all distinct).
constexpr int BANK_PAD         = 4;
constexpr int TILE_STRIDE      = N_COEFFS_PADDED + BANK_PAD;  // 52, ldm-aligned
constexpr int STAGE_STRIDE     = 16 + BANK_PAD;               // 20, ldm-aligned

static_assert(N_COEFFS <= N_COEFFS_PADDED,
    "N_COEFFS must fit inside N_COEFFS_PADDED");
static_assert(BLOCK_SIZE == WARP_SIZE,
    "RECT-3.1.c kernel is single-warp per block");

// ─────────────────────────────────────────────────────────────────────
// Warp-level scalar reductions (full mask).
// ─────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    }
    return v;
}
__device__ __forceinline__ float warp_reduce_min(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fminf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    }
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────
// Explicit f32 → tf32 downcast (rounds-to-nearest-even on the lower
// 13 mantissa bits via inline PTX). Mirror of mma.h's
// __float_to_tf32 — re-declared here so the precision loss is
// readable in this translation unit.
// ─────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float prism_f32_to_tf32(float in) {
    float out;
    asm volatile(
        "{\n  .reg .b32 t;"
        "\n   cvt.rna.tf32.f32 t, %1;"
        "\n   mov.b32 %0, t;\n}\n"
        : "=f"(out) : "f"(in));
    return out;
}

// Compact scratch changes storage only. The raw RichSpike remains authoritative
// for all metadata; both paths use the same numerical kernel bodies below.
using prism_nhs::rich_spike::NativeV2SeedLayout;
using prism_nhs::rich_spike::load_native_v2_seed;
using prism_nhs::rich_spike::valid_native_v2_seed_layout;
struct SpikePosition { float x, y, z; };
static_assert(sizeof(SpikePosition) == 3 * sizeof(float), "XYZ scratch must be 12 bytes");
static_assert(alignof(SpikePosition) == alignof(float), "XYZ scratch alignment");

__device__ __forceinline__ void store_perturbed(RichSpike* out, uint32_t i, const RichSpike& s) {
    out[i] = s;
}
__device__ __forceinline__ void store_perturbed(SpikePosition* out, uint32_t i, const RichSpike& s) {
    out[i] = {s.x, s.y, s.z};
}
template<bool PositionOverlay>
__device__ __forceinline__ RichSpike load_projected_spike(
    const RichSpike* raw, const SpikePosition* xyz, uint32_t i, NativeV2SeedLayout layout) {
    RichSpike s = load_native_v2_seed(raw,i,layout);
    if (PositionOverlay) {
        const SpikePosition p = xyz[i];
        s.x = p.x; s.y = p.y; s.z = p.z;
    }
    return s;
}

// ─────────────────────────────────────────────────────────────────────
// SO(3) projection kernel — one cluster per block, one warp per block.
// ─────────────────────────────────────────────────────────────────────
template<bool PositionOverlay>
__global__ void prism_so3_project_manifold_kernel(
    const RichSpike*  __restrict__ d_spikes,
    const SpikePosition* __restrict__ d_positions,
    const uint32_t*   __restrict__ d_cluster_offsets,
    uint32_t                       n_clusters,
    const float*      __restrict__ d_k_lm,
    ContactShellTile* __restrict__ d_tiles_out,
    uint32_t                       frame_id,
    uint32_t                       source_stream_id,
    NativeV2SeedLayout seed_layout
) {
    const uint32_t cluster_id = blockIdx.x;
    if (cluster_id >= n_clusters) return;

    const uint32_t start = d_cluster_offsets[cluster_id];
    const uint32_t end   = d_cluster_offsets[cluster_id + 1u];
    const uint32_t n     = end - start;

    const uint32_t lane = threadIdx.x;  // 0..31

    // ── Shared memory ─────────────────────────────────────────────
    // Per-plane data tile (4 × 16 × 49 floats = 12544 B = 12.25 KB).
    __shared__ float s_plane_tile[N_PLANES][TILE_ROWS][TILE_STRIDE];
    // WMMA store staging buffer (16 × 17 floats = 1088 B = 1.0625 KB).
    // Reused across (plane, col-group); explicit __syncwarp() guards
    // the reuse.
    __shared__ float s_wmma_stage[TILE_ROWS][STAGE_STRIDE];
    // Per-plane a_lm accumulator across all tiles (4 × 48 floats = 768 B).
    __shared__ float s_plane_acc[N_PLANES][N_COEFFS_PADDED];
    // Centroid + AABB broadcast.
    __shared__ float s_centroid[3];
    __shared__ float s_aabb_min[3];
    __shared__ float s_aabb_max[3];
    // Lossless tag aggregators.
    __shared__ uint32_t s_agg_spike_source;
    __shared__ uint32_t s_agg_origin_phase;
    __shared__ uint32_t s_agg_chem_flags;
    __shared__ float    s_sum_w[N_PLANES];

    // ── Init shared accumulators ──────────────────────────────────
    // Stride loop covers all N_COEFFS_PADDED slots even though
    // N_COEFFS_PADDED (48) > WARP_SIZE (32).
    #pragma unroll
    for (int p = 0; p < N_PLANES; ++p) {
        for (int idx = lane; idx < N_COEFFS_PADDED; idx += WARP_SIZE) {
            s_plane_acc[p][idx] = 0.0f;
        }
    }
    if (lane == 0u) {
        s_agg_spike_source = 0u;
        s_agg_origin_phase = 0u;
        s_agg_chem_flags   = 0u;
        #pragma unroll
        for (int p = 0; p < N_PLANES; ++p) s_sum_w[p] = 0.0f;
    }
    __syncwarp();

    // ── Empty-cluster fast path ──────────────────────────────────
    if (n == 0u) {
        if (lane == 0u) {
            ContactShellTile& tile = d_tiles_out[cluster_id];
            tile.phase             = 0u;
            tile.stream_id         = source_stream_id;
            tile.cluster_id        = static_cast<int32_t>(cluster_id);
            tile.frame             = frame_id;
            #pragma unroll
            for (int k = 0; k < 64; ++k) {
                tile.geo_alm[k]   = 0.0f;
                tile.caus_alm[k]  = 0.0f;
                tile.therm_alm[k] = 0.0f;
                tile.chem_alm[k]  = 0.0f;
            }
            // Epsilon-padded zeros: Adjudicator's log(C_l) must stay
            // finite even on empty clusters. Cross-Agent FFI Mandate
            // §Dependency 2 (KL-Divergence Absolute-Zero Fatality).
            constexpr float KL_EPS_FP = 1e-7f;
            #pragma unroll
            for (int k = 0; k <= static_cast<int>(LMAX); ++k) {
                tile.geo_power_spectrum[k]   = KL_EPS_FP;
                tile.caus_power_spectrum[k]  = KL_EPS_FP;
                tile.therm_power_spectrum[k] = KL_EPS_FP;
                tile.chem_power_spectrum[k]  = KL_EPS_FP;
            }
            #pragma unroll
            for (int k = static_cast<int>(LMAX) + 1; k < 8; ++k) {
                tile.geo_power_spectrum[k]   = 0.0f;
                tile.caus_power_spectrum[k]  = 0.0f;
                tile.therm_power_spectrum[k] = 0.0f;
                tile.chem_power_spectrum[k]  = 0.0f;
            }
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                tile.aabb_min[k] = 0.0f;
                tile.aabb_max[k] = 0.0f;
            }
            tile.agg_spike_source  = 0u;
            tile.agg_origin_phase  = 0u;
            tile.agg_chem_flags    = 0u;
            tile.agg_pad           = 0u;
            tile.sum_w_geo         = 0.0f;
            tile.sum_w_caus        = 0.0f;
            tile.sum_w_therm       = 0.0f;
            tile.sum_w_chem        = 0.0f;
            tile.spike_count       = 0u;
            tile.adjudication_code = 0u;
            tile.reserved[0]       = 0u;
            tile.reserved[1]       = 0u;
            #pragma unroll
            for (int k = 0; k < 32; ++k) tile._pad[k] = 0u;
        }
        return;
    }

    // ── Pass 1 — centroid + AABB ──────────────────────────────────
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    float mn_x = +FLT_MAX, mn_y = +FLT_MAX, mn_z = +FLT_MAX;
    float mx_x = -FLT_MAX, mx_y = -FLT_MAX, mx_z = -FLT_MAX;
    for (uint32_t i = start + lane; i < end; i += WARP_SIZE) {
        const RichSpike s = load_projected_spike<PositionOverlay>(d_spikes, d_positions, i, seed_layout);
        sx += s.x; sy += s.y; sz += s.z;
        mn_x = fminf(mn_x, s.x); mn_y = fminf(mn_y, s.y); mn_z = fminf(mn_z, s.z);
        mx_x = fmaxf(mx_x, s.x); mx_y = fmaxf(mx_y, s.y); mx_z = fmaxf(mx_z, s.z);
    }
    sx = warp_reduce_sum(sx);
    sy = warp_reduce_sum(sy);
    sz = warp_reduce_sum(sz);
    mn_x = warp_reduce_min(mn_x); mn_y = warp_reduce_min(mn_y); mn_z = warp_reduce_min(mn_z);
    mx_x = warp_reduce_max(mx_x); mx_y = warp_reduce_max(mx_y); mx_z = warp_reduce_max(mx_z);
    if (lane == 0u) {
        const float inv_n = 1.0f / static_cast<float>(n);
        s_centroid[0] = sx * inv_n;
        s_centroid[1] = sy * inv_n;
        s_centroid[2] = sz * inv_n;
        s_aabb_min[0] = mn_x; s_aabb_min[1] = mn_y; s_aabb_min[2] = mn_z;
        s_aabb_max[0] = mx_x; s_aabb_max[1] = mx_y; s_aabb_max[2] = mx_z;
    }
    __syncwarp();

    const float cx = s_centroid[0];
    const float cy = s_centroid[1];
    const float cz = s_centroid[2];

    // ── WMMA fragment setup (constant across iterations) ─────────
    // A is "ones" so that C[i][j] = sum_k 1·B[k][j] = column-sum of B.
    // We use row-major for both A and B for natural tile-load.
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_ones;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_data;
    wmma::fragment<wmma::accumulator, 16, 16, 8, float>                                c_acc;

    // 1.0f survives the cvt.rna.tf32.f32 conversion bit-exact.
    wmma::fill_fragment(a_ones, 1.0f);

    // ── Pass 2 — projection in 16-spike tiles ────────────────────
    for (uint32_t tile_base = start; tile_base < end; tile_base += TILE_ROWS) {
        const uint32_t tile_end   = (tile_base + TILE_ROWS < end) ? (tile_base + TILE_ROWS) : end;
        const uint32_t n_in_tile  = tile_end - tile_base;

        // Zero the per-plane data tile (every cell — the tile_size
        // may be < TILE_ROWS at the last iteration, and unused rows
        // must contribute zero to the WMMA reduction).
        // 32 lanes split 4*16*49 = 3136 floats → 98 floats/lane.
        constexpr int TILE_FLOATS = N_PLANES * TILE_ROWS * TILE_STRIDE;
        for (int idx = lane; idx < TILE_FLOATS; idx += WARP_SIZE) {
            const int p = idx / (TILE_ROWS * TILE_STRIDE);
            const int rem = idx - p * (TILE_ROWS * TILE_STRIDE);
            const int r = rem / TILE_STRIDE;
            const int c = rem - r * TILE_STRIDE;
            s_plane_tile[p][r][c] = 0.0f;
        }
        __syncwarp();

        // Lanes 0..n_in_tile-1 evaluate one spike each. Per-plane
        // weights are reduced with the same fixed shuffle tree as the
        // centroid path; unordered f32 shared updates would make sum_w
        // order-dependent under high-density clusters.
        float w_g_local = 0.0f;
        float w_c_local = 0.0f;
        float w_t_local = 0.0f;
        float w_h_local = 0.0f;
        if (lane < n_in_tile) {
            const RichSpike s = load_projected_spike<PositionOverlay>(d_spikes, d_positions, tile_base + lane, seed_layout);

            // Lossless tag aggregation (always — even for centroid-coincident
            // spikes which contribute Y_l>0 = 0 but still carry tags).
            atomicOr(&s_agg_spike_source, s.spike_source);
            atomicOr(&s_agg_origin_phase, s.origin_phase);
            atomicOr(&s_agg_chem_flags,   s.chem_flags);

            const float dx = s.x - cx;
            const float dy = s.y - cy;
            const float dz = s.z - cz;
            const float r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= 1e-12f) {
                const float r     = sqrtf(r2);
                const float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dz / r)));
                const float phi   = atan2f(dy, dx);

                float Y[N_COEFFS];
                prism_nhs::sh_basis::prism_sh_eval_lmax5(theta, phi, d_k_lm, Y);

                // Per-plane weights.
                const float w_g = 1.0f;
                const float w_c = fabsf(s.causal_lag);
                const float w_t = s.water_density;
                const float w_h = static_cast<float>(__popc(s.chem_flags));
                w_g_local = w_g;
                w_c_local = w_c;
                w_t_local = w_t;
                w_h_local = w_h;

                // Write tf32-down-converted, weighted Y_lm rows into
                // each plane's tile slot at row=lane, col=k. Pad cols
                // beyond N_COEFFS were already zero-init'd.
                #pragma unroll
                for (int k = 0; k < N_COEFFS; ++k) {
                    const float y = Y[k];
                    s_plane_tile[PLANE_GEO]  [lane][k] = prism_f32_to_tf32(y * w_g);
                    s_plane_tile[PLANE_CAUS] [lane][k] = prism_f32_to_tf32(y * w_c);
                    s_plane_tile[PLANE_THERM][lane][k] = prism_f32_to_tf32(y * w_t);
                    s_plane_tile[PLANE_CHEM] [lane][k] = prism_f32_to_tf32(y * w_h);
                }
            }
        }
        const float w_g_tile = warp_reduce_sum(w_g_local);
        const float w_c_tile = warp_reduce_sum(w_c_local);
        const float w_t_tile = warp_reduce_sum(w_t_local);
        const float w_h_tile = warp_reduce_sum(w_h_local);
        if (lane == 0u) {
            s_sum_w[PLANE_GEO]   += w_g_tile;
            s_sum_w[PLANE_CAUS]  += w_c_tile;
            s_sum_w[PLANE_THERM] += w_t_tile;
            s_sum_w[PLANE_CHEM]  += w_h_tile;
        }
        __syncwarp();

        // ── WMMA reduction: per (plane, col-group), two mma_sync
        // calls (k=8 contracts → 2 calls cover 16 rows).
        #pragma unroll
        for (int p = 0; p < N_PLANES; ++p) {
            #pragma unroll
            for (int g = 0; g < N_COL_GROUPS; ++g) {
                const int col_start = g * 16;

                // Fresh accumulator for this (plane, col-group).
                wmma::fill_fragment(c_acc, 0.0f);

                __syncwarp();

                // First half (rows 0..7): ldm = TILE_STRIDE, B layout = row-major.
                wmma::load_matrix_sync(
                    b_data,
                    &s_plane_tile[p][0][col_start],
                    static_cast<unsigned>(TILE_STRIDE)
                );
                __syncwarp();
                wmma::mma_sync(c_acc, a_ones, b_data, c_acc);
                __syncwarp();

                // Second half (rows 8..15).
                wmma::load_matrix_sync(
                    b_data,
                    &s_plane_tile[p][8][col_start],
                    static_cast<unsigned>(TILE_STRIDE)
                );
                __syncwarp();
                wmma::mma_sync(c_acc, a_ones, b_data, c_acc);
                __syncwarp();

                // Store to staging buffer (row 0 contains the column-sum).
                wmma::store_matrix_sync(
                    &s_wmma_stage[0][0],
                    c_acc,
                    static_cast<unsigned>(STAGE_STRIDE),
                    wmma::mem_row_major
                );
                __syncwarp();

                // Lanes 0..15 each accumulate one column into the
                // per-plane persistent accumulator. Lanes 16..31 idle.
                if (lane < 16) {
                    s_plane_acc[p][col_start + lane] += s_wmma_stage[0][lane];
                }
                __syncwarp();
            }
        }
    }

    // ── Pass 2.5 — Per-plane L2 normalization ─────────────────────
    //
    // Per Cross-Agent FFI Mandate §Dependency 1 (operator directive
    // 2026-04-29 Part 2): a_lm coefficients must be L2-normalized
    // before global write so that:
    //   (a) downstream Q_s · a_lm scaling cannot drive tf32 to Inf
    //       under 10^7+ spike bursts, and
    //   (b) `C_l = Σ_m |a_lm/L2|² = original_C_l / L2²` becomes a
    //       probability distribution, ready for the KL-divergence
    //       Adjudicator (Σ_l C_l = 1 for non-empty planes).
    //
    // L2 norm is preserved under SO(3) rotation so this normalization
    // does NOT break the G11 invariance gate.
    //
    // Empty-plane guard: if L2 < L2_EPS, leave a_lm at zero (skip
    // normalization). The tile-write epsilon-pad (1e-7f) below keeps
    // `log(C_l)` finite in Claude-2's Adjudicator either way.
    constexpr float L2_EPS = 1e-6f;
    __shared__ float s_l2_norm[N_PLANES];
    if (lane < N_PLANES) {
        const int p = lane;
        float sum_sq = 0.0f;
        #pragma unroll
        for (int k = 0; k < N_COEFFS_PADDED; ++k) {
            const float a = s_plane_acc[p][k];
            sum_sq += a * a;
        }
        s_l2_norm[p] = sqrtf(sum_sq);
    }
    __syncwarp();

    // Apply normalization. Stride loop handles all N_COEFFS_PADDED
    // slots (32 lanes < 48 indices, so a single `lane < N` guard
    // would silently leave slots 32..47 un-normalized — we hit this
    // bug during RECT-3.1.c bring-up, caught by Σ C_l = 5.29 ≠ 1).
    #pragma unroll
    for (int p = 0; p < N_PLANES; ++p) {
        const float l2 = s_l2_norm[p];
        const float inv_l2 = (l2 >= L2_EPS) ? (1.0f / l2) : 0.0f;
        for (int idx = lane; idx < N_COEFFS_PADDED; idx += WARP_SIZE) {
            s_plane_acc[p][idx] *= inv_l2;
        }
    }
    __syncwarp();

    // ── Pass 3 — tile write (cooperative across lanes) ───────────
    ContactShellTile* tile_p = &d_tiles_out[cluster_id];

    // Lanes 0..15 cooperate to write per-plane a_lm[64] (28 padding
    // slots remain zero; lanes 16..31 idle).
    if (lane < 16) {
        #pragma unroll
        for (int p = 0; p < N_PLANES; ++p) {
            // 4 column-groups × 16 lanes = 64 slots.
            for (int g = 0; g < 4; ++g) {
                const int idx = g * 16 + lane;
                const float v = (idx < N_COEFFS_PADDED)
                    ? s_plane_acc[p][idx]
                    : 0.0f;
                // Route to the right plane field.
                switch (p) {
                    case PLANE_GEO:   tile_p->geo_alm  [idx] = (idx < 64) ? v : 0.0f; break;
                    case PLANE_CAUS:  tile_p->caus_alm [idx] = (idx < 64) ? v : 0.0f; break;
                    case PLANE_THERM: tile_p->therm_alm[idx] = (idx < 64) ? v : 0.0f; break;
                    case PLANE_CHEM:  tile_p->chem_alm [idx] = (idx < 64) ? v : 0.0f; break;
                }
            }
        }
    }

    // Per-plane C_l = Σ_m |a_lm|² for l=0..LMAX. Only lane 0 does this
    // (cheap: 36 sq-add ops × 4 planes = 144 fma). Epsilon-pad each
    // value via fmaxf(c_l, KL_EPS) so the downstream Adjudicator
    // (Claude-2) never feeds log(0) → -Inf into the F1 SWITCH selector.
    //
    // Cross-Agent FFI Mandate §Dependency 2 (operator directive
    // 2026-04-29 Part 2.2): "MUST NOT allow a true zero to be written
    // to the ContactShellTile."
    constexpr float KL_EPS         = 1e-7f;
    constexpr float MAX_ENERGY     = 100.0f;  // post-norm Σ C_l = 1.0 nominal; 100× headroom
    if (lane == 0u) {
        // Header
        tile_p->phase      = 0u;
        tile_p->stream_id  = source_stream_id;
        tile_p->cluster_id = static_cast<int32_t>(cluster_id);
        tile_p->frame      = frame_id;

        // C_l per plane (epsilon-padded).
        float total_energy = 0.0f;
        #pragma unroll
        for (int p = 0; p < N_PLANES; ++p) {
            float c_l_buf[8];
            #pragma unroll
            for (int l = 0; l <= static_cast<int>(LMAX); ++l) {
                float c_l = 0.0f;
                for (int m = -l; m <= l; ++m) {
                    const int idx = l * (l + 1) + m;
                    const float a = s_plane_acc[p][idx];
                    c_l += a * a;
                }
                c_l_buf[l] = fmaxf(c_l, KL_EPS);
                total_energy += c_l_buf[l];
            }
            #pragma unroll
            for (int l = static_cast<int>(LMAX) + 1; l < 8; ++l) {
                c_l_buf[l] = 0.0f;
            }
            switch (p) {
                case PLANE_GEO:
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) tile_p->geo_power_spectrum[l] = c_l_buf[l];
                    break;
                case PLANE_CAUS:
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) tile_p->caus_power_spectrum[l] = c_l_buf[l];
                    break;
                case PLANE_THERM:
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) tile_p->therm_power_spectrum[l] = c_l_buf[l];
                    break;
                case PLANE_CHEM:
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) tile_p->chem_power_spectrum[l] = c_l_buf[l];
                    break;
            }
        }

        // PTX hard-trap on energy-budget overflow. Cross-Agent FFI
        // Mandate §3 PTX Invariant Traps: if normalized Σ_p Σ_l C_l
        // exceeds MAX_ENERGY (4 planes × Σ_l C_l ≤ 4 × 1 = 4 for
        // non-empty planes; 4 × 6 × KL_EPS ≈ 2.4e-6 for empty), the
        // L2 normalization or accumulation has broken — abort the
        // SM rather than feed garbage to the Adjudicator.
        if (total_energy > MAX_ENERGY) {
            asm volatile("trap;");
        }

        // AABB
        tile_p->aabb_min[0] = s_aabb_min[0];
        tile_p->aabb_min[1] = s_aabb_min[1];
        tile_p->aabb_min[2] = s_aabb_min[2];
        tile_p->aabb_min[3] = 0.0f;
        tile_p->aabb_max[0] = s_aabb_max[0];
        tile_p->aabb_max[1] = s_aabb_max[1];
        tile_p->aabb_max[2] = s_aabb_max[2];
        tile_p->aabb_max[3] = 0.0f;

        // Lossless aggregates
        tile_p->agg_spike_source = s_agg_spike_source;
        tile_p->agg_origin_phase = s_agg_origin_phase;
        tile_p->agg_chem_flags   = s_agg_chem_flags;
        tile_p->agg_pad          = 0u;

        // Per-plane sum_w
        tile_p->sum_w_geo   = s_sum_w[PLANE_GEO];
        tile_p->sum_w_caus  = s_sum_w[PLANE_CAUS];
        tile_p->sum_w_therm = s_sum_w[PLANE_THERM];
        tile_p->sum_w_chem  = s_sum_w[PLANE_CHEM];

        // Counters
        tile_p->spike_count       = n;
        tile_p->adjudication_code = 0u;
        tile_p->reserved[0]       = 0u;
        tile_p->reserved[1]       = 0u;

        // Padding
        #pragma unroll
        for (int k = 0; k < 32; ++k) tile_p->_pad[k] = 0u;

        // Cross-Agent FFI Mandate §Dependency 3: __threadfence() after
        // the final global write so the ContactShellTile is L2-coherent
        // before the downstream Adjudicator / ASC kernel reads it.
        // Required for in-graph WHILE-node correctness — the captured
        // graph cannot rely on inter-kernel implicit synchronization
        // when both producer and consumer are in the same DAG node.
        __threadfence();
    }
}

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration
// ─────────────────────────────────────────────────────────────────────

extern "C" {

uint32_t prism_so3_project_link_probe(void) {
    return 0x53033u;
}

cudaError_t prism_so3_project_run(
    const RichSpike*  d_spikes,
    const uint32_t*   d_cluster_offsets,
    uint32_t          n_clusters,
    const float*      d_k_lm,
    ContactShellTile* d_tiles_out,
    uint32_t          frame_id,
    uint32_t          source_stream_id,
    cudaStream_t      stream
) {
    if (n_clusters == 0u) return cudaSuccess;
    if (d_spikes == nullptr || d_cluster_offsets == nullptr ||
        d_k_lm == nullptr || d_tiles_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    prism_so3_project_manifold_kernel<false><<<n_clusters, BLOCK_SIZE, 0, stream>>>(
        d_spikes, nullptr, d_cluster_offsets, n_clusters,
        d_k_lm, d_tiles_out, frame_id, source_stream_id, NativeV2SeedLayout{}
    );
    return cudaGetLastError();
}

cudaError_t prism_so3_project_seed_run(
    const RichSpike*  d_spikes,
    const uint32_t*   d_cluster_offsets,
    uint32_t          n_clusters,
    const float*      d_k_lm,
    ContactShellTile* d_tiles_out,
    uint32_t          frame_id,
    uint32_t          source_stream_id,
    cudaStream_t      stream,
    NativeV2SeedLayout seed_layout) {
    if (!valid_native_v2_seed_layout(seed_layout)) return cudaErrorInvalidValue;
    if (n_clusters == 0u) return cudaSuccess;
    if (d_spikes == nullptr || d_cluster_offsets == nullptr ||
        d_k_lm == nullptr || d_tiles_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    prism_so3_project_manifold_kernel<false><<<n_clusters, BLOCK_SIZE, 0, stream>>>(
        d_spikes, nullptr, d_cluster_offsets, n_clusters,
        d_k_lm, d_tiles_out, frame_id, source_stream_id, seed_layout
    );
    return cudaGetLastError();
}

// Same projector, with coordinates from compact scratch and every other field
// from the unmodified raw population. No metadata/plane is omitted.
cudaError_t prism_so3_project_positions_run(
    const RichSpike* d_spikes, const float* d_positions,
    const uint32_t* d_cluster_offsets, uint32_t n_clusters,
    const float* d_k_lm, ContactShellTile* d_tiles_out,
    uint32_t frame_id, uint32_t source_stream_id, cudaStream_t stream)
{
    if (n_clusters == 0u) return cudaSuccess;
    if (d_spikes == nullptr || d_positions == nullptr || d_cluster_offsets == nullptr ||
        d_k_lm == nullptr || d_tiles_out == nullptr) return cudaErrorInvalidValue;
    prism_so3_project_manifold_kernel<true><<<n_clusters, BLOCK_SIZE, 0, stream>>>(
        d_spikes, reinterpret_cast<const SpikePosition*>(d_positions), d_cluster_offsets,
        n_clusters, d_k_lm, d_tiles_out, frame_id, source_stream_id, NativeV2SeedLayout{});
    return cudaGetLastError();
}

cudaError_t prism_so3_project_positions_seed_run(
    const RichSpike* d_spikes, const float* d_positions,
    const uint32_t* d_cluster_offsets, uint32_t n_clusters,
    const float* d_k_lm, ContactShellTile* d_tiles_out,
    uint32_t frame_id, uint32_t source_stream_id, cudaStream_t stream,
    NativeV2SeedLayout seed_layout)
{
    if (!valid_native_v2_seed_layout(seed_layout)) return cudaErrorInvalidValue;
    if (n_clusters == 0u) return cudaSuccess;
    if (d_spikes == nullptr || d_positions == nullptr || d_cluster_offsets == nullptr ||
        d_k_lm == nullptr || d_tiles_out == nullptr) return cudaErrorInvalidValue;
    prism_so3_project_manifold_kernel<true><<<n_clusters, BLOCK_SIZE, 0, stream>>>(
        d_spikes, reinterpret_cast<const SpikePosition*>(d_positions), d_cluster_offsets,
        n_clusters, d_k_lm, d_tiles_out, frame_id, source_stream_id, seed_layout);
    return cudaGetLastError();
}

}  // extern "C"

// ════════════════════════════════════════════════════════════════════
// M1.2.20.C-A — Gradient Gasp Kernel + Anchor LUTs
//
// Per the operator's "Six Pillars" rulings 2026-05-02:
//
//   Ruling 1 — Q_s(λ) LUT for {260, 280, 305, 320 nm}:
//                 {0.75, 1.00, 0.35, 0.05}
//   Ruling 2 — η_eff = η_base / max(vib_energy, 1e-3)
//   Ruling 3 — d_forces from NhsAmberFusedEngine (per-atom f32×3)
//   Ruling 4 — Cα anchor: atom_idx = d_residue_to_calpha[residue_id]
//   Ruling 5 — 10× amplification when current_step == force_burst_step
//   Ruling 6 — d_masses from NhsAmberFusedEngine (per-atom f32)
//
// Math:
//   Δr = η_eff · Q_s · (f_anchor / m_anchor) · dt²
//   pos_perturbed = pos_relaxed + Δr
//
// The kernel is single-block, n-thread-per-spike (clamped to 256 per
// block; multiple blocks for >256 spike clusters).  Each thread:
//   1. Loads its RichSpike (64 B coalesced).
//   2. Looks up Cα atom index in __constant__ memory.
//   3. Reads f_anchor (3 floats) + m_anchor (1 float) via __ldg().
//   4. Computes Δr.
//   5. Writes the perturbed RichSpike to d_spikes_out (64 B coalesced).
//
// Phase 1 deliverable: kernel + LUTs + host populator + launcher.
// The kernel is NOT yet wired into the captured graph — that's Phase 2's
// parallel-stream Path C refactor.  This commit lands the math + the
// FFI plumbing end-to-end so the Phase 2 wire-in is mechanical.
// ════════════════════════════════════════════════════════════════════

// Ruling 1 — Q_s(λ) Transition Dipole Moment LUT.
// Indexed by the 2-bit UV code packed into RichSpike::intensity_packed
// upper nibble (bits 30-31): 0 = 260 nm, 1 = 280 nm, 2 = 305 nm,
// 3 = 320 nm.  __constant__ memory ⇒ broadcast load (1 cycle).
__constant__ float d_mu01_lut[4] = {
    0.75f,  // 260 nm — backbone / nucleic resonance
    1.00f,  // 280 nm — peak aromatic trigger (TRP/TYR)
    0.35f,  // 305 nm — near-UV tail
    0.05f,  // 320 nm — threshold baseline
};

// Ruling 4 — Residue → Cα atom index LUT.
// Sized to 1024 entries to cover targets up to ~1k residues without
// reallocation.  Host populates via prism_so3_set_residue_to_calpha;
// unpopulated entries default to 0xFFFFFFFFu (sentinel — kernel
// treats as "no Cα available; skip displacement on this spike").
__constant__ uint32_t d_residue_to_calpha[1024];

// Legacy compatibility clock only. Production captures read an engine-owned
// ProtocolState.current_step pointer through prism_apply_gradient_gasp_clock_launch.
// This global is deliberately not written by production multi-stream orchestration.
__constant__ uint32_t d_current_md_step;

template<typename Output>
__global__ void prism_apply_gradient_gasp_kernel(
    const RichSpike*    __restrict__ d_spikes_in,
    Output*             __restrict__ d_spikes_out,
    const float*        __restrict__ d_forces,           // [n_atoms × 3]
    const float*        __restrict__ d_masses,           // [n_atoms]
    const uint8_t*      __restrict__ adj_base,           // FFI offset arithmetic
    float*              __restrict__ d_com_shift,        // nullable; deterministic COM kernel handles non-null case
    float*              __restrict__ d_total_mass,       // nullable; deterministic COM kernel handles non-null case
    const uint32_t*                  d_stream_md_step,
    uint32_t                         n_spikes,
    uint32_t                         n_atoms,
    NativeV2SeedLayout seed_layout
) {
    // Production reads the live, per-engine device clock. The null path
    // exists only for the explicitly retained legacy ABI launcher.
    const uint32_t current_step = d_stream_md_step != nullptr
        ? __ldg(d_stream_md_step) : d_current_md_step;
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_spikes) return;

    // Load the input spike (64-byte coalesced).
    const RichSpike spike_in = load_native_v2_seed(d_spikes_in,i,seed_layout);

    // Read FFI handles from the adjudicator state.
    //   gasp_gain_eta    @ offset 136 (f32)
    //   force_burst_step @ offset 140 (u32)
    //   d_dt             @ offset 120 (*mut f32) → dereference for current dt
    const float    eta_base = *reinterpret_cast<const float*   >(adj_base + 136);
    const uint32_t burst_at = *reinterpret_cast<const uint32_t*>(adj_base + 140);
    float* const* p_d_dt    =  reinterpret_cast<float* const*>(adj_base + 120);
    const float dt = (*p_d_dt != nullptr) ? __ldg(*p_d_dt) : 0.002f;  // fallback 2 fs

    // Resolve Cα anchor.  residue_id is signed (RichSpike::UNRESOLVED_RESIDUE
    // = -1); reject negative or out-of-range and skip displacement.
    const int32_t res_id = spike_in.residue_id;
    if (res_id < 0 || res_id >= 1024) {
        store_perturbed(d_spikes_out, i, spike_in);  // pass-through for unresolved spikes
        return;
    }
    const uint32_t atom_idx = d_residue_to_calpha[res_id];
    if (atom_idx == 0xFFFFFFFFu || atom_idx >= n_atoms) {
        store_perturbed(d_spikes_out, i, spike_in);  // pass-through for unmapped residues
        return;
    }

    // Read per-atom force + mass via __ldg() (read-only cache hint).
    const uint32_t f_off = atom_idx * 3u;
    const float fx = __ldg(d_forces + f_off + 0u);
    const float fy = __ldg(d_forces + f_off + 1u);
    const float fz = __ldg(d_forces + f_off + 2u);
    const float mass = fmaxf(__ldg(d_masses + atom_idx), 1.0e-3f);

    // Q_s(λ) lookup — top 2 bits of intensity_packed select the wavelength.
    const uint32_t uv_code = (spike_in.intensity_packed >> 30) & 0x3u;
    const float Q_s = d_mu01_lut[uv_code];

    // **M1.2.20.C-G / T22 — Gasp Gain Recalibration & Singularity Guard.**
    // Pre-G the formula `1.0 / max(vib, 1e-3)` blew up to 1000× gain
    // during the 50K cold-hold (vib_energy ≈ 0 in frozen states),
    // catastrophic Σ m·Δr → uniform momentum_violation_flag = 1 → all
    // 69 v9B records came back code=VIOLATION.  Operator §2 fix:
    //   • η_base lowered 1.0 → 0.5 (anticipatory safety)
    //   • soft vib floor raised 1e-3 → 1e-2 (10× lower max gain)
    //   • hard saturation cap η_eff ≤ 5.0 (prevents single-spike runaway)
    constexpr float VIB_SOFT_FLOOR = 1.0e-2f;
    constexpr float ETA_HARD_CAP   = 5.0f;
    const float soft_vib = fmaxf(spike_in.vib_energy, VIB_SOFT_FLOOR);
    float eta_eff = (0.5f * eta_base) / soft_vib;
    const bool gain_saturated = (eta_eff > ETA_HARD_CAP);
    eta_eff = fminf(eta_eff, ETA_HARD_CAP);
    // Stamp the GAIN_SATURATION bit on adj.adjudication_reason_flags
    // (offset 156 post-M1.2.26.RECOUPLE; relocated from 148 to make
    // room for total_kl_mu/sigma at 148/152). Bit 3 = 0x8. Atomic OR
    // survives the 256-thread block contention. Adjudicator step
    // kernel reads this for the CSR-Q forensic readback.
    if (gain_saturated) {
        atomicOr(reinterpret_cast<uint32_t*>(
            const_cast<uint8_t*>(adj_base) + 156), 0x8u);
    }

    // Ruling 5 — 10× amplification when this is the burst step.
    if (current_step == burst_at) {
        eta_eff *= 10.0f;
    }

    // Δr = η_eff · Q_s · (f / m) · dt²
    const float scale = eta_eff * Q_s * (dt * dt) / mass;
    const float dx = scale * fx;
    const float dy = scale * fy;
    const float dz = scale * fz;

    // COM accumulation is intentionally absent here. The launcher routes
    // non-null COM requests to prism_apply_gradient_gasp_com_kernel, which
    // uses one fixed block and a deterministic reduction tree before the
    // momentum guard consumes d_com_shift/d_total_mass.
    (void)d_com_shift;
    (void)d_total_mass;

    // Write the perturbed spike (struct-copy with x/y/z replaced).
    // The COM correction is applied LATER by
    // `prism_apply_com_correction_kernel` after the post-pass
    // computes the global correction vector.  Writing provisional
    // (uncorrected) positions here is fine — they're transient until
    // the correction pass runs on the same stream.
    RichSpike out = spike_in;
    out.x = spike_in.x + dx;
    out.y = spike_in.y + dy;
    out.z = spike_in.z + dz;
    store_perturbed(d_spikes_out, i, out);
}

template<typename Output>
__global__ void prism_apply_gradient_gasp_com_kernel(
    const RichSpike*    __restrict__ d_spikes_in,
    Output*             __restrict__ d_spikes_out,
    const float*        __restrict__ d_forces,
    const float*        __restrict__ d_masses,
    const uint8_t*      __restrict__ adj_base,
    float*              __restrict__ d_com_shift,
    float*              __restrict__ d_total_mass,
    const uint32_t*                  d_stream_md_step,
    uint32_t                         n_spikes,
    uint32_t                         n_atoms,
    NativeV2SeedLayout seed_layout
) {
    constexpr uint32_t TPB = 256u;
    const uint32_t lane = threadIdx.x;
    if (blockIdx.x != 0u || lane >= TPB) return;

    const uint32_t current_step = d_stream_md_step != nullptr
        ? __ldg(d_stream_md_step) : d_current_md_step;
    const float    eta_base = *reinterpret_cast<const float*   >(adj_base + 136);
    const uint32_t burst_at = *reinterpret_cast<const uint32_t*>(adj_base + 140);
    float* const* p_d_dt    =  reinterpret_cast<float* const*>(adj_base + 120);
    const float dt = (*p_d_dt != nullptr) ? __ldg(*p_d_dt) : 0.002f;

    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    double sm = 0.0;
    uint32_t local_reason_flags = 0u;

    for (uint32_t i = lane; i < n_spikes; i += TPB) {
        const RichSpike spike_in = load_native_v2_seed(d_spikes_in,i,seed_layout);
        const int32_t res_id = spike_in.residue_id;
        if (res_id < 0 || res_id >= 1024) {
            store_perturbed(d_spikes_out, i, spike_in);
            continue;
        }

        const uint32_t atom_idx = d_residue_to_calpha[res_id];
        if (atom_idx == 0xFFFFFFFFu || atom_idx >= n_atoms) {
            store_perturbed(d_spikes_out, i, spike_in);
            continue;
        }

        const uint32_t f_off = atom_idx * 3u;
        const float fx = __ldg(d_forces + f_off + 0u);
        const float fy = __ldg(d_forces + f_off + 1u);
        const float fz = __ldg(d_forces + f_off + 2u);
        const float mass = fmaxf(__ldg(d_masses + atom_idx), 1.0e-3f);

        const uint32_t uv_code = (spike_in.intensity_packed >> 30) & 0x3u;
        const float Q_s = d_mu01_lut[uv_code];

        constexpr float VIB_SOFT_FLOOR = 1.0e-2f;
        constexpr float ETA_HARD_CAP   = 5.0f;
        const float soft_vib = fmaxf(spike_in.vib_energy, VIB_SOFT_FLOOR);
        float eta_eff = (0.5f * eta_base) / soft_vib;
        if (eta_eff > ETA_HARD_CAP) {
            local_reason_flags |= 0x8u;
        }
        eta_eff = fminf(eta_eff, ETA_HARD_CAP);

        if (current_step == burst_at) {
            eta_eff *= 10.0f;
        }

        const float scale = eta_eff * Q_s * (dt * dt) / mass;
        const float dx = scale * fx;
        const float dy = scale * fy;
        const float dz = scale * fz;

        sx += static_cast<double>(mass) * static_cast<double>(dx);
        sy += static_cast<double>(mass) * static_cast<double>(dy);
        sz += static_cast<double>(mass) * static_cast<double>(dz);
        sm += static_cast<double>(mass);

        RichSpike out = spike_in;
        out.x = spike_in.x + dx;
        out.y = spike_in.y + dy;
        out.z = spike_in.z + dz;
        store_perturbed(d_spikes_out, i, out);
    }

    __shared__ double s_x[TPB];
    __shared__ double s_y[TPB];
    __shared__ double s_z[TPB];
    __shared__ double s_m[TPB];
    __shared__ uint32_t s_flags[TPB];
    s_x[lane] = sx;
    s_y[lane] = sy;
    s_z[lane] = sz;
    s_m[lane] = sm;
    s_flags[lane] = local_reason_flags;
    __syncthreads();

    #pragma unroll
    for (uint32_t stride = TPB >> 1; stride > 0u; stride >>= 1) {
        if (lane < stride) {
            s_x[lane] += s_x[lane + stride];
            s_y[lane] += s_y[lane + stride];
            s_z[lane] += s_z[lane + stride];
            s_m[lane] += s_m[lane + stride];
            s_flags[lane] |= s_flags[lane + stride];
        }
        __syncthreads();
    }

    if (lane == 0u) {
        d_com_shift[0] = static_cast<float>(s_x[0]);
        d_com_shift[1] = static_cast<float>(s_y[0]);
        d_com_shift[2] = static_cast<float>(s_z[0]);
        d_total_mass[0] = static_cast<float>(s_m[0]);
        if (s_flags[0] != 0u) {
            *reinterpret_cast<uint32_t*>(const_cast<uint8_t*>(adj_base) + 156) |= s_flags[0];
        }
        __threadfence();
    }
}

// ─────────────────────────────────────────────────────────────────────
// M1.2.20.C-B — PTX Momentum Guard check kernel.
//
// Single-thread post-pass that reads the [3 × f32] com_shift accumulator
// populated by prism_apply_gradient_gasp_kernel, computes the
// translation magnitude `|Σ m·Δr|`, and writes `adj.momentum_violation_flag`
// (FFI offset 144) = 1 when the magnitude exceeds 1.0e-4 Å.  Operator's
// Zero-Trust §3 invariant — a legitimate gasp is an EXPANSION, not a
// translation; if the protein walks during the perturbation the SO(3)
// power spectrum becomes ungrounded noise and the Adjudicator must
// override to VIOLATION (Case 2 / abort).
// ─────────────────────────────────────────────────────────────────────

extern "C"
__global__ void prism_momentum_guard_check_kernel(
    const float*    __restrict__ d_com_shift,         // [3]   Σ m·Δr  (input)
    const float*    __restrict__ d_total_mass,        // [1]   Σ m     (input)
    float*          __restrict__ d_com_correction,    // [3]   output: correction vector
    uint8_t*        __restrict__ adj_base             // InterferometricAdjudicatorFfi*
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const float sx = d_com_shift[0];
    const float sy = d_com_shift[1];
    const float sz = d_com_shift[2];
    const float total_m = d_total_mass[0];

    // Path Ω Option A — COM correction vector.  Guard against zero/tiny
    // total mass (only when the gasp produced no valid Δr at all, e.g.
    // every spike had unresolved residue → pass-through).  In that case
    // the correction is zero and the apply kernel is a no-op.
    constexpr float TOTAL_MASS_FLOOR = 1.0f;  // amu — well below any real protein
    const float inv_m = (total_m > TOTAL_MASS_FLOOR) ? (1.0f / total_m) : 0.0f;

    const float cx = sx * inv_m;
    const float cy = sy * inv_m;
    const float cz = sz * inv_m;

    d_com_correction[0] = cx;
    d_com_correction[1] = cy;
    d_com_correction[2] = cz;

    // Loosened violation threshold — see header comment.  Pre-Option-A
    // this was 1e-4 Å, which fired every chunk because the raw |Σ m·Δr|
    // is a √n random walk that vastly exceeds 1e-4 for 500k spikes.
    // The CORRECTION magnitude (per-atom-mass-averaged drift) is the
    // physically meaningful quantity; >1 Å indicates the gasp is wildly
    // off (real bug), not normal random-walk accumulation.
    const float corr_mag2 = cx*cx + cy*cy + cz*cz;
    constexpr float MAX_CORRECTION_AA = 1.0f;        // 1 Å COM drift cap
    constexpr float MAX_CORRECTION_SQ = MAX_CORRECTION_AA * MAX_CORRECTION_AA;

    uint32_t* p_flag = reinterpret_cast<uint32_t*>(adj_base + 144);
    *p_flag = (corr_mag2 > MAX_CORRECTION_SQ) ? 1u : 0u;
    __threadfence();
}

extern "C"
int prism_momentum_guard_check_launch(
    const void*  d_com_shift,
    const void*  d_total_mass,
    void*        d_com_correction,
    void*        adj_base,
    void*        stream)
{
    if (d_com_shift == nullptr || d_total_mass == nullptr ||
        d_com_correction == nullptr || adj_base == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    prism_momentum_guard_check_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<const float*>(d_com_shift),
        static_cast<const float*>(d_total_mass),
        static_cast<float*>(d_com_correction),
        static_cast<uint8_t*>(adj_base)
    );
    return static_cast<int>(cudaGetLastError());
}

// ─────────────────────────────────────────────────────────────────────
// Path Ω Option A — Per-spike COM correction kernel.
//
// Reads d_com_correction[3] (computed by the post-pass momentum-guard
// kernel) and subtracts it from each spike's (x, y, z) in
// d_spikes_inout.  This is what makes the perturbed manifold COM-locked
// to the relaxed manifold so the downstream SO(3) KL captures relative
// structural divergence (the actual discovery signal) and not the
// global rigid drift the gradient gasp inherently produces.
//
// 256 threads/block, n_spikes/256 blocks.  Each thread modifies one
// spike in place.  Pure read-modify-write; no atomics.
// ─────────────────────────────────────────────────────────────────────

template<typename Output>
__global__ void prism_apply_com_correction_kernel(
    Output*           __restrict__ d_spikes_inout,
    const float*      __restrict__ d_com_correction,
    uint32_t                       n_spikes
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_spikes) return;

    const float cx = d_com_correction[0];
    const float cy = d_com_correction[1];
    const float cz = d_com_correction[2];

    Output s = d_spikes_inout[i];
    s.x -= cx;
    s.y -= cy;
    s.z -= cz;
    d_spikes_inout[i] = s;
}

extern "C"
int prism_apply_com_correction_launch(
    void*        d_spikes_inout,
    const void*  d_com_correction,
    uint32_t     n_spikes,
    void*        stream)
{
    if (d_spikes_inout == nullptr || d_com_correction == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_spikes == 0u) return static_cast<int>(cudaSuccess);
    constexpr uint32_t TPB = 256u;
    const uint32_t blocks = (n_spikes + TPB - 1u) / TPB;
    prism_apply_com_correction_kernel<RichSpike><<<blocks, TPB, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<RichSpike*>(d_spikes_inout),
        static_cast<const float*>(d_com_correction),
        n_spikes
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int prism_apply_com_positions_correction_launch(
    void* d_positions, const void* d_com_correction, uint32_t n_spikes, void* stream)
{
    if (d_positions == nullptr || d_com_correction == nullptr)
        return static_cast<int>(cudaErrorInvalidValue);
    if (n_spikes == 0u) return static_cast<int>(cudaSuccess);
    constexpr uint32_t TPB = 256u;
    const uint32_t blocks = (n_spikes + TPB - 1u) / TPB;
    prism_apply_com_correction_kernel<SpikePosition><<<blocks, TPB, 0,
        static_cast<cudaStream_t>(stream)>>>(static_cast<SpikePosition*>(d_positions),
        static_cast<const float*>(d_com_correction), n_spikes);
    return static_cast<int>(cudaGetLastError());
}

// ─── Host helpers ───────────────────────────────────────────────────

extern "C"
int prism_so3_set_residue_to_calpha(
    const uint32_t* host_table,   // [n] residue → Cα atom index
    uint32_t        n,
    void*           stream)
{
    if (host_table == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    if (n == 0u) return static_cast<int>(cudaSuccess);
    if (n > 1024u) n = 1024u;  // clamp to LUT capacity
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_residue_to_calpha,
        host_table,
        n * sizeof(uint32_t),
        /*offset*/ 0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

// Legacy compatibility setter only; production must never use a module-global
// clock to communicate per-stream state. Complete the copy before its stack
// source expires. This fence has no production per-chunk cost.
extern "C"
int prism_so3_set_current_md_step(
    uint32_t        step,
    void*           stream)
{
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_current_md_step,
        &step,
        sizeof(uint32_t),
        /*offset*/ 0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    if (rc == cudaSuccess) rc = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    return static_cast<int>(rc);
}

template<typename Output>
static int prism_apply_gradient_gasp_launch_impl(
    const void*  d_spikes_in,
    void*        d_spikes_out,
    const void*  d_forces,
    const void*  d_masses,
    const void*  adj_base,
    void*        d_com_shift,    // [3 × f32] — Σ m·Δr accumulator (nullable)
    void*        d_total_mass,   // [1 × f32] — Σ m accumulator (Path Ω Option A, nullable)
    const uint32_t* d_stream_md_step,
    uint32_t     n_spikes,
    uint32_t     n_atoms,
    void*        stream,
    NativeV2SeedLayout seed_layout = NativeV2SeedLayout{})
{
    if (d_spikes_in == nullptr || d_spikes_out == nullptr ||
        d_forces == nullptr || d_masses == nullptr || adj_base == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if ((d_com_shift == nullptr) != (d_total_mass == nullptr)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_spikes == 0u || n_atoms == 0u) return static_cast<int>(cudaSuccess);

    constexpr uint32_t TPB = 256u;  // threads-per-block
    if (d_com_shift != nullptr && d_total_mass != nullptr) {
        prism_apply_gradient_gasp_com_kernel<Output><<<1u, TPB, 0,
            static_cast<cudaStream_t>(stream)>>>(
            static_cast<const RichSpike*>(d_spikes_in),
            static_cast<Output*>(d_spikes_out),
            static_cast<const float*>(d_forces),
            static_cast<const float*>(d_masses),
            static_cast<const uint8_t*>(adj_base),
            static_cast<float*>(d_com_shift),
            static_cast<float*>(d_total_mass),
            d_stream_md_step,
            n_spikes,
            n_atoms, seed_layout
        );
        return static_cast<int>(cudaGetLastError());
    }

    const uint32_t blocks = (n_spikes + TPB - 1u) / TPB;
    prism_apply_gradient_gasp_kernel<Output><<<blocks, TPB, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<const RichSpike*>(d_spikes_in),
        static_cast<Output*>(d_spikes_out),
        static_cast<const float*>(d_forces),
        static_cast<const float*>(d_masses),
        static_cast<const uint8_t*>(adj_base),
        static_cast<float*>(d_com_shift),
        static_cast<float*>(d_total_mass),
        d_stream_md_step,
        n_spikes,
        n_atoms, seed_layout
    );
    return static_cast<int>(cudaGetLastError());
}

// Strict production entrypoint: graph nodes retain the owning engine's
// ProtocolState.current_step pointer. No module-global clock fallback.
extern "C"
int prism_apply_gradient_gasp_clock_launch(
    const void* d_spikes_in, void* d_spikes_out,
    const void* d_forces, const void* d_masses, const void* adj_base,
    void* d_com_shift, void* d_total_mass,
    const uint32_t* d_stream_md_step, uint32_t n_spikes, uint32_t n_atoms,
    void* stream)
{
    if (d_stream_md_step == nullptr ||
        (reinterpret_cast<uintptr_t>(d_stream_md_step) % alignof(uint32_t)) != 0u)
        return static_cast<int>(cudaErrorInvalidValue);
    return prism_apply_gradient_gasp_launch_impl<RichSpike>(d_spikes_in, d_spikes_out,
        d_forces, d_masses, adj_base, d_com_shift, d_total_mass,
        d_stream_md_step, n_spikes, n_atoms, stream);
}

// Compact production entrypoint: same body, XYZ-only output.
extern "C"
int prism_apply_gradient_gasp_positions_clock_launch(
    const void* d_spikes_in, void* d_spikes_out,
    const void* d_forces, const void* d_masses, const void* adj_base,
    void* d_com_shift, void* d_total_mass,
    const uint32_t* d_stream_md_step, uint32_t n_spikes, uint32_t n_atoms,
    void* stream)
{
    // A populated projection must never consume unwritten position scratch.
    if (n_spikes > 0u && n_atoms == 0u)
        return static_cast<int>(cudaErrorInvalidValue);
    if (d_stream_md_step == nullptr ||
        (reinterpret_cast<uintptr_t>(d_stream_md_step) % alignof(uint32_t)) != 0u)
        return static_cast<int>(cudaErrorInvalidValue);
    return prism_apply_gradient_gasp_launch_impl<SpikePosition>(d_spikes_in, d_spikes_out,
        d_forces, d_masses, adj_base, d_com_shift, d_total_mass,
        d_stream_md_step, n_spikes, n_atoms, stream);
}

extern "C"
int prism_apply_gradient_gasp_positions_seed_clock_launch(
    const void* d_spikes_in, void* d_spikes_out,
    const void* d_forces, const void* d_masses, const void* adj_base,
    void* d_com_shift, void* d_total_mass,
    const uint32_t* d_stream_md_step, uint32_t n_spikes, uint32_t n_atoms,
    void* stream,
    NativeV2SeedLayout seed_layout)
{
    if (!valid_native_v2_seed_layout(seed_layout)) return static_cast<int>(cudaErrorInvalidValue);
    // A populated projection must never consume unwritten position scratch.
    if (n_spikes > 0u && n_atoms == 0u)
        return static_cast<int>(cudaErrorInvalidValue);
    if (d_stream_md_step == nullptr ||
        (reinterpret_cast<uintptr_t>(d_stream_md_step) % alignof(uint32_t)) != 0u)
        return static_cast<int>(cudaErrorInvalidValue);
    return prism_apply_gradient_gasp_launch_impl<SpikePosition>(d_spikes_in, d_spikes_out,
        d_forces, d_masses, adj_base, d_com_shift, d_total_mass,
        d_stream_md_step, n_spikes, n_atoms, stream, seed_layout);
}

// Legacy ABI only. Production captured pipelines must use clock_launch.
extern "C"
int prism_apply_gradient_gasp_launch(
    const void* d_spikes_in, void* d_spikes_out,
    const void* d_forces, const void* d_masses, const void* adj_base,
    void* d_com_shift, void* d_total_mass,
    uint32_t /*legacy captured scalar*/, uint32_t n_spikes, uint32_t n_atoms,
    void* stream)
{
    return prism_apply_gradient_gasp_launch_impl<RichSpike>(d_spikes_in, d_spikes_out,
        d_forces, d_masses, adj_base, d_com_shift, d_total_mass,
        nullptr, n_spikes, n_atoms, stream);
}

// Explicit diagnostic entrypoint, never inserted into production graphs.
__global__ void prism_v2_seed_consumed_fields_kernel(const RichSpike* seed,uint32_t* out,
    uint32_t count,NativeV2SeedLayout layout) {
    const uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const RichSpike s=load_native_v2_seed(seed,i,layout);uint32_t* r=out+12ull*i;
    r[0]=__float_as_uint(s.x);r[1]=__float_as_uint(s.y);r[2]=__float_as_uint(s.z);
    r[3]=__float_as_uint(s.water_density);r[4]=__float_as_uint(s.wd_change);r[5]=__float_as_uint(s.vib_energy);
    r[6]=s.intensity_packed;r[7]=static_cast<uint32_t>(s.residue_id);r[8]=__float_as_uint(s.causal_lag);
    r[9]=s.origin_phase;r[10]=s.spike_source;r[11]=s.chem_flags;
}
extern "C" int prism_v2_seed_consumed_fields_launch(const RichSpike* seed,uint32_t* out,
    uint32_t count,void* stream,NativeV2SeedLayout layout) {
    if(!seed||!out||!valid_native_v2_seed_layout(layout))return static_cast<int>(cudaErrorInvalidValue);
    if(count==0)return 0;
    prism_v2_seed_consumed_fields_kernel<<<(count+255u)/256u,256,0,static_cast<cudaStream_t>(stream)>>>(seed,out,count,layout);
    return static_cast<int>(cudaGetLastError());
}

}}  // namespace prism_nhs::so3_project

// Exact module state transfer at an all-worker quiescent boundary.
// Caller owns the context, holds every producer and synchronizes before entry.
extern "C" size_t prism_checkpoint_so3_project_bytes() { return sizeof(prism_nhs::so3_project::d_mu01_lut) + sizeof(prism_nhs::so3_project::d_residue_to_calpha) + sizeof(prism_nhs::so3_project::d_current_md_step); }
extern "C" int prism_checkpoint_so3_project_capture(unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_so3_project_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, prism_nhs::so3_project::d_mu01_lut, sizeof(prism_nhs::so3_project::d_mu01_lut), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(prism_nhs::so3_project::d_mu01_lut);
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, prism_nhs::so3_project::d_residue_to_calpha, sizeof(prism_nhs::so3_project::d_residue_to_calpha), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(prism_nhs::so3_project::d_residue_to_calpha);
    rc = cudaMemcpyFromSymbolAsync(bytes + offset, prism_nhs::so3_project::d_current_md_step, sizeof(prism_nhs::so3_project::d_current_md_step), 0, cudaMemcpyDeviceToHost, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(prism_nhs::so3_project::d_current_md_step);
    return (int)cudaStreamSynchronize(stream);
}
extern "C" int prism_checkpoint_so3_project_restore(const unsigned char* bytes, size_t length, cudaStream_t stream) {
    if (!bytes || length != prism_checkpoint_so3_project_bytes()) return (int)cudaErrorInvalidValue;
    size_t offset = 0; cudaError_t rc;
    rc = cudaMemcpyToSymbolAsync(prism_nhs::so3_project::d_mu01_lut, bytes + offset, sizeof(prism_nhs::so3_project::d_mu01_lut), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(prism_nhs::so3_project::d_mu01_lut);
    rc = cudaMemcpyToSymbolAsync(prism_nhs::so3_project::d_residue_to_calpha, bytes + offset, sizeof(prism_nhs::so3_project::d_residue_to_calpha), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(prism_nhs::so3_project::d_residue_to_calpha);
    rc = cudaMemcpyToSymbolAsync(prism_nhs::so3_project::d_current_md_step, bytes + offset, sizeof(prism_nhs::so3_project::d_current_md_step), 0, cudaMemcpyHostToDevice, stream);
    if (rc != cudaSuccess) return (int)rc;
    offset += sizeof(prism_nhs::so3_project::d_current_md_step);
    return (int)cudaStreamSynchronize(stream);
}
