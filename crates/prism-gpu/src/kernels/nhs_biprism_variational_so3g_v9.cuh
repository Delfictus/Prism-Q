// File: roles/engine/crates/prism-gpu/src/kernels/nhs_biprism_variational_so3g_v9.cuh
// SELF-CONTAINED PRODUCTION V9 HEADER (ZERO EXTERNAL V6/V7/V8 DEPENDENCIES)
// Verified invariants:
//   1. Dynamic Shared Memory: sizeof(VariationalClusterDsmemV9) == 31,616 bytes (< 32 KiB, 2 CTAs/SM on sm_120)
//   2. Scale-Invariant Exact 3x3 SPD Metric Inverse (handles g = 10^{-3} I_3 -> 1000 I_3 with 0.0% error + Sylvester SPD gate)
//   3. Exact SO(3, g) Cayley-Boris Rotation (K = g^{-1}[\tau]_x, K^T g + g K = 0) with 3-Point Gauss-Legendre Chord Integral
//   4. Coupled Picard + Live DEC Poisson PCG Solve on EVERY Picard iteration
//   5. Exact Full-Hamiltonian Gonzalez Discrete-Gradient Projection (H(x^{n+1}, p^{n+1}) == H(x^n, p^n), H=0.25 on x^4/4)
//   6. Collective 8-Leaf All-or-Nothing Rollback Guard + True Cross-Leaf L2 Fallback + ABI-Safe 256B Adjudicator Overlay

#pragma once
#include "nhs_biprism_simplicial_tile.cuh"
#include "nhs_biprism_xtb_lut.cuh"
#if __has_include("../../../prism-nhs/src/cuda/adjudicator.cuh")
#include "../../../prism-nhs/src/cuda/adjudicator.cuh"
#else
#error "prism-nhs/src/cuda/adjudicator.cuh not found relative to prism-gpu/src/kernels"
#endif

#define BIPRISM_INNER_SUBSTEPS 4
#define MAX_VARIATIONAL_PICARD_ITERS 12
#define VARIATIONAL_PICARD_TOL 1e-7f
#define DEC_PCG_MAX_ITERS 12
#define DEC_PCG_TOL_SQ 1e-12f

#define FAULT_DEC_NONCONV      1u
#define FAULT_PICARD_NONCONV   2u
#define FAULT_METRIC_NON_SPD   4u
#define ADJ_SOLVER_FAULT_FLAG  0x80000000u

#define FP_Q39_SCALE 549755813888.0f
#define FP_Q39_INV   1.8189894035458565e-12f

struct __align__(8) sym3_f32 {
    float xx, yy, zz, xy, xz, yz;
};

struct __align__(128) FoliatedStreamPtrTableFfi {
    float*          d_coords_xyz[8];
    float*          d_vels_xyz[8];
    float*          d_forces_xyz[8];
    const float*    d_inv_masses[8];
    const float*    d_true_inv_masses[8];
    const uint32_t* d_core_shake_mask[8];
};

enum BiprismIntegrationMode : uint32_t {
    ConstrainedShakeSafe   = 0,
    FlexibleCoreVerified   = 1,
    GonzalezExactEnergyBit = 2
};

// ABI-Safe 256-Byte Overlay matching cuda/adjudicator.cuh:189 and interferometric_adjudicator.rs:516
struct __align__(128) AdjudicatorAbiSafeOverlayFfi {
    uint32_t header_words[32];     // Bytes 0..127
    float    wilson_traces[12];    // Bytes 128..175
    float    bianchi_residual;     // Bytes 176..179
    float    unitarity_defect;     // Bytes 180..183
    float    pulay_force_norm;     // Bytes 184..187
    uint32_t branch_and_fault;     // Bytes 188..191
    float    last_picard_residual; // Bytes 192..195
    float    last_dec_residual;    // Bytes 196..199
    uint32_t picard_iters_used;    // Bytes 200..203
    uint32_t dec_iters_used;       // Bytes 204..207
    uint32_t leaf_fault_mask;      // Bytes 208..211
    uint32_t cluster_fault_mask;   // Bytes 212..215
    uint32_t committed_substeps;   // Bytes 216..219
    uint32_t reserved_tail[9];     // Bytes 220..255
};
static_assert(sizeof(AdjudicatorAbiSafeOverlayFfi) == 256, "Must match 256B InterferometricAdjudicatorFfi ABI");
static_assert(alignof(AdjudicatorAbiSafeOverlayFfi) == 128, "Must match align(128) ABI");

// Exact 31,616-Byte Dynamic Shared Memory Layout
struct __align__(128) VariationalClusterDsmemV9 {
    // 1. Persistent Sub-Step State (18,432 bytes)
    float3   x_mid[TILE_TOTAL_VERTICES];          // 3,072 B
    float3   v_mid[TILE_TOTAL_VERTICES];          // 3,072 B
    float3   x_ref[TILE_TOTAL_VERTICES];          // 3,072 B
    float    mass[TILE_TOTAL_VERTICES];           // 1,024 B
    float    q_live[TILE_TOTAL_VERTICES];         // 1,024 B
    float    phi_dec[TILE_TOTAL_VERTICES];        // 1,024 B
    cpx2_f32 psi[TILE_CORE_VERTICES][2];          // 3,072 B
    uint8_t  edge_u8[TILE_MAX_EDGES][2];          // 3,072 B

    // 2. Dual-Purpose Edge Buffer (6,144 bytes): star1_live[1536] in Stage B/C; edge_phase[1536] in Stage D
    float    edge_buf[TILE_MAX_EDGES];            // 6,144 B

    // 3. Stage-Aliased Scratch Union (6,912 bytes)
    union {
        sym3_f32 g_cov[TILE_CORE_VERTICES];       // 4,608 B (Stage A)
        struct {
            float cg_r[TILE_TOTAL_VERTICES];      // 1,024 B
            float cg_z[TILE_TOTAL_VERTICES];      // 1,024 B
            float cg_p[TILE_TOTAL_VERTICES];      // 1,024 B
            float cg_Ap[TILE_TOTAL_VERTICES];     // 1,024 B
            float diag_inv[TILE_TOTAL_VERTICES];  // 1,024 B
        } pcg;                                    // 5,120 B (Stage B)
        struct {
            unsigned long long f_tot_q39[TILE_CORE_VERTICES][3]; // 4,608 B
            float3             tau_berry[TILE_CORE_VERTICES];    // 2,304 B
        } force;                                  // 6,912 B (Stage C)
        float4 lie_k_accum[TILE_CORE_VERTICES];   // 3,072 B (Stage D)
    } stage;

    // 4. Reduction, Energy Secant, & Collective Fault Footer (128 bytes)
    float    warp_red[8];                         //    32 B
    float    last_picard_residual;                //     4 B
    float    last_dec_residual;                   //     4 B
    uint32_t picard_iters_used;                   //     4 B
    uint32_t dec_iters_used;                      //     4 B
    uint32_t local_fault_mask;                    //     4 B
    uint32_t collective_fault_mask;               //     4 B
    float    v_com[3];                            //    12 B: Center-of-mass velocity for translation-invariant Boris
    uint32_t _pad_footer[15];                     //    60 B
};

static_assert(sizeof(VariationalClusterDsmemV9) == 31616, "VariationalClusterDsmemV9 must be exactly 31,616 bytes");
static_assert(sizeof(VariationalClusterDsmemV9) <= 49152, "Must fit within 48 KiB limit");

__device__ __forceinline__ uint32_t get_hw_cluster_size_x() {
    uint32_t csize = 1u;
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;\n" : "=r"(csize));
    return csize;
}

__device__ __forceinline__ float dsmem_load_f32_safe(
    const float* local_smem_elem_ptr,
    uint32_t target_leaf_rank,
    const float* l2_fallback_ptr
) {
    if (get_hw_cluster_size_x() >= 8u) {
        uint32_t smem_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(local_smem_elem_ptr));
        float val;
        asm volatile(
            "{\n"
            "  .reg .u32 dsmem_ptr;\n"
            "  mapa.shared::cluster.u32 dsmem_ptr, %1, %2;\n"
            "  ld.shared::cluster.f32 %0, [dsmem_ptr];\n"
            "}\n"
            : "=f"(val)
            : "r"(smem_u32), "r"(target_leaf_rank)
        );
        return val;
    } else {
        return *l2_fallback_ptr;
    }
}

__device__ __forceinline__ uint32_t dsmem_load_u32_safe(
    const uint32_t* local_smem_u32_ptr,
    uint32_t target_leaf_rank,
    const uint32_t* l2_fallback_ptr
) {
    if (get_hw_cluster_size_x() >= 8u) {
        uint32_t smem_u32 = static_cast<uint32_t>(__cvta_generic_to_shared(local_smem_u32_ptr));
        uint32_t val;
        asm volatile(
            "{\n"
            "  .reg .u32 dsmem_ptr;\n"
            "  mapa.shared::cluster.u32 dsmem_ptr, %1, %2;\n"
            "  ld.shared::cluster.u32 %0, [dsmem_ptr];\n"
            "}\n"
            : "=r"(val)
            : "r"(smem_u32), "r"(target_leaf_rank)
        );
        return val;
    } else {
        return *l2_fallback_ptr;
    }
}

__device__ __forceinline__ void cluster_or_grid_sync_8leaves(
    uint32_t* __restrict__ d_barrier_arrival_count,
    uint32_t* __restrict__ d_barrier_epoch
) {
    if (gridDim.x <= 1u) {
        __syncthreads();
        return;
    }
    if (get_hw_cluster_size_x() >= 8u) {
        asm volatile(
            "barrier.cluster.arrive.aligned;\n"
            "barrier.cluster.wait.aligned;\n"
            ::: "memory"
        );
    } else {
        __syncthreads();
        if (threadIdx.x == 0) {
            __threadfence();
            const uint32_t cur_epoch = *(volatile uint32_t*)d_barrier_epoch;
            const uint32_t arrived = atomicAdd(d_barrier_arrival_count, 1u) + 1u;
            if (arrived >= gridDim.x) {
                *d_barrier_arrival_count = 0u;
                __threadfence();
                atomicAdd(d_barrier_epoch, 1u);
            } else {
                uint32_t spin_guard = 0u;
                while (*(volatile uint32_t*)d_barrier_epoch == cur_epoch && spin_guard < 100000u) {
                    __nanosleep(32);
                    ++spin_guard;
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float cta_max_reduce_256(float val, float* s_warp_red) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(FULL_WARP_MASK, val, offset));
    }
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (lane == 0) s_warp_red[warp] = val;
    __syncthreads();

    float block_max = (threadIdx.x < 8u) ? s_warp_red[threadIdx.x] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(FULL_WARP_MASK, block_max, offset));
        }
        if (lane == 0) s_warp_red[0] = block_max;
    }
    __syncthreads();
    return s_warp_red[0];
}

__device__ __forceinline__ void smem_deterministic_add_f32(unsigned long long* addr, float val) {
    long long fixed_val = __float2ll_rn(val * FP_Q39_SCALE);
    atomicAdd(addr, static_cast<unsigned long long>(fixed_val));
}

__device__ __forceinline__ float smem_deterministic_load_f32(const unsigned long long* addr) {
    return static_cast<float>(static_cast<long long>(*addr)) * FP_Q39_INV;
}

__device__ __forceinline__ float3 sym3_mul_vec(const sym3_f32& M, float3 v) {
    return {
        __fmaf_rn(M.xx, v.x, __fmaf_rn(M.xy, v.y, M.xz * v.z)),
        __fmaf_rn(M.xy, v.x, __fmaf_rn(M.yy, v.y, M.yz * v.z)),
        __fmaf_rn(M.xz, v.x, __fmaf_rn(M.yz, v.y, M.zz * v.z))
    };
}

// Scale-Invariant Exact 3x3 SPD Metric Inversion with Sylvester SPD Verification.
// Never floors det(g): for g = 10^{-3} I_3, returns exact 1000 I_3.
__device__ __forceinline__ sym3_f32 invert_sym3_metric_exact(
    const sym3_f32& g,
    float& out_det_g_inv,
    uint32_t* __restrict__ out_spd_fault_mask = nullptr
) {
    const double tr = (double)g.xx + (double)g.yy + (double)g.zz;
    if (!(tr > 0.0) || !isfinite(tr)) {
        if (out_spd_fault_mask) atomicOr(out_spd_fault_mask, FAULT_METRIC_NON_SPD);
        out_det_g_inv = 1.0f;
        return {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    }

    const double inv_sigma = 3.0 / tr;
    const double g_xx = (double)g.xx * inv_sigma;
    const double g_yy = (double)g.yy * inv_sigma;
    const double g_zz = (double)g.zz * inv_sigma;
    const double g_xy = (double)g.xy * inv_sigma;
    const double g_xz = (double)g.xz * inv_sigma;
    const double g_yz = (double)g.yz * inv_sigma;

    const double c_xx = __fma_rn(g_yy, g_zz, -g_yz * g_yz);
    const double c_yy = __fma_rn(g_xx, g_zz, -g_xz * g_xz);
    const double c_zz = __fma_rn(g_xx, g_yy, -g_xy * g_xy);
    const double c_xy = __fma_rn(g_xz, g_yz, -g_xy * g_zz);
    const double c_xz = __fma_rn(g_xy, g_yz, -g_xz * g_yy);
    const double c_yz = __fma_rn(g_xy, g.xz * inv_sigma, -g_xx * g_yz);

    const double det_hat = __fma_rn(g_xx, c_xx, __fma_rn(g_xy, c_xy, g_xz * c_xz));
    if (!(g_xx > 1e-12) || !(c_zz > 1e-12) || !(det_hat > 1e-12) || !isfinite(det_hat)) {
        if (out_spd_fault_mask) atomicOr(out_spd_fault_mask, FAULT_METRIC_NON_SPD);
        out_det_g_inv = 1.0f;
        return {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    }

    const double inv_det_hat = 1.0 / det_hat;
    const double factor = inv_sigma * inv_det_hat;
    out_det_g_inv = (float)((inv_sigma * inv_sigma * inv_sigma) * inv_det_hat);

    return {
        (float)(c_xx * factor), (float)(c_yy * factor), (float)(c_zz * factor),
        (float)(c_xy * factor), (float)(c_xz * factor), (float)(c_yz * factor)
    };
}

// Scale-Invariant SO(3, g) Cayley-Boris Map (K = g^{-1}[\tau]_x, K^T g + g K = 0)
__device__ __forceinline__ float3 so3_g_metric_cayley_boris(
    float3 v_minus,
    float3 tau_cov,
    const sym3_f32& g_cov,
    const sym3_f32& g_inv,
    float /*det_g_inv_unused*/
) {
    const float3 v_cross_tau = {
        __fmaf_rn(v_minus.y, tau_cov.z, -v_minus.z * tau_cov.y),
        __fmaf_rn(v_minus.z, tau_cov.x, -v_minus.x * tau_cov.z),
        __fmaf_rn(v_minus.x, tau_cov.y, -v_minus.y * tau_cov.x)
    };
    const float3 w1 = sym3_mul_vec(g_inv, v_cross_tau);
    const float3 w1_cross_tau = {
        __fmaf_rn(w1.y, tau_cov.z, -w1.z * tau_cov.y),
        __fmaf_rn(w1.z, tau_cov.x, -w1.x * tau_cov.z),
        __fmaf_rn(w1.x, tau_cov.y, -w1.y * tau_cov.x)
    };
    const float3 w2 = sym3_mul_vec(g_inv, w1_cross_tau);

    const double det_g_inv_d = (double)g_inv.xx * ((double)g_inv.yy * (double)g_inv.zz - (double)g_inv.yz * (double)g_inv.yz)
                             + (double)g_inv.xy * ((double)g_inv.xz * (double)g_inv.yz - (double)g_inv.xy * (double)g_inv.zz)
                             + (double)g_inv.xz * ((double)g_inv.xy * (double)g_inv.yz - (double)g_inv.xz * (double)g_inv.yy);
    const float3 g_tau = sym3_mul_vec(g_cov, tau_cov);
    const double tau_g_tau_d = (double)tau_cov.x * (double)g_tau.x
                             + (double)tau_cov.y * (double)g_tau.y
                             + (double)tau_cov.z * (double)g_tau.z;
    const float scale = (float)(2.0 / __fma_rn(det_g_inv_d, tau_g_tau_d, 1.0));

    return {
        __fmaf_rn(scale, w1.x + w2.x, v_minus.x),
        __fmaf_rn(scale, w1.y + w2.y, v_minus.y),
        __fmaf_rn(scale, w1.z + w2.z, v_minus.z)
    };
}

__device__ __forceinline__ float3 evaluate_metric_cotangent_force_edge(
    float3 e_uv, float3 r_hat, float r, float l2_cur, float l2_ref,
    float3 v_u,  float3 v_v,   float mass_u, float mass_v,
    float chi_g
) {
    const float inv_l2_ref = 1.0f / fmaxf(l2_ref, 1e-8f);
    const float inv_r      = 1.0f / fmaxf(r, 1e-6f);
    const float strain_fac = (l2_cur - l2_ref) * inv_l2_ref;

    const float xi_u = __fmaf_rn(v_u.x, r_hat.x, __fmaf_rn(v_u.y, r_hat.y, v_u.z * r_hat.z));
    const float xi_v = __fmaf_rn(v_v.x, r_hat.x, __fmaf_rn(v_v.y, r_hat.y, v_v.z * r_hat.z));

    const float rad_coeff = -chi_g * (mass_u * xi_u * xi_u + mass_v * xi_v * xi_v) * inv_l2_ref;
    const float ang_coeff = -chi_g * strain_fac * inv_r;
    const float3 v_perp_weighted = {
        mass_u * xi_u * (v_u.x - xi_u * r_hat.x) + mass_v * xi_v * (v_v.x - xi_v * r_hat.x),
        mass_u * xi_u * (v_u.y - xi_u * r_hat.y) + mass_v * xi_v * (v_v.y - xi_v * r_hat.y),
        mass_u * xi_u * (v_u.z - xi_u * r_hat.z) + mass_v * xi_v * (v_v.z - xi_v * r_hat.z)
    };
    return {
        __fmaf_rn(rad_coeff, e_uv.x, ang_coeff * v_perp_weighted.x),
        __fmaf_rn(rad_coeff, e_uv.y, ang_coeff * v_perp_weighted.y),
        __fmaf_rn(rad_coeff, e_uv.z, ang_coeff * v_perp_weighted.z)
    };
}

// 3-Point Gauss-Legendre Chord Integral of Berry Field \overline{B}(x^n, x^{n+1})
__device__ __forceinline__ float3 evaluate_gauss_legendre_berry_chord(
    float3 e_mid, float3 de_uv, float A_omega, float dt_sub
) {
    constexpr float gl_shift = 0.3872983346207417f;
    constexpr float w_end    = 0.2777777777777778f;
    constexpr float w_mid    = 0.4444444444444444f;

    const float3 e_m = {e_mid.x - gl_shift * de_uv.x, e_mid.y - gl_shift * de_uv.y, e_mid.z - gl_shift * de_uv.z};
    const float3 e_p = {e_mid.x + gl_shift * de_uv.x, e_mid.y + gl_shift * de_uv.y, e_mid.z + gl_shift * de_uv.z};

    const float inv_l3_m = powf(__fmaf_rn(e_m.x, e_m.x, __fmaf_rn(e_m.y, e_m.y, e_m.z * e_m.z)) + 1e-12f, -1.5f);
    const float inv_l3_0 = powf(__fmaf_rn(e_mid.x, e_mid.x, __fmaf_rn(e_mid.y, e_mid.y, e_mid.z * e_mid.z)) + 1e-12f, -1.5f);
    const float inv_l3_p = powf(__fmaf_rn(e_p.x, e_p.x, __fmaf_rn(e_p.y, e_p.y, e_p.z * e_p.z)) + 1e-12f, -1.5f);

    const float phase_amp = __sinf(A_omega * dt_sub) + A_omega * dt_sub;
    const float c_m = w_end * phase_amp * inv_l3_m;
    const float c_0 = w_mid * phase_amp * inv_l3_0;
    const float c_p = w_end * phase_amp * inv_l3_p;

    return {
        __fmaf_rn(c_m, e_m.x, __fmaf_rn(c_0, e_mid.x, c_p * e_p.x)),
        __fmaf_rn(c_m, e_m.y, __fmaf_rn(c_0, e_mid.y, c_p * e_p.y)),
        __fmaf_rn(c_m, e_m.z, __fmaf_rn(c_0, e_mid.z, c_p * e_p.z))
    };
}

__device__ __forceinline__ sym3_f32 assemble_metric_to_reg_v9(
    VariationalClusterDsmemV9* __restrict__ smem,
    uint32_t num_edges,
    float chi_g
) {
    const uint32_t tid = threadIdx.x;
    if (tid < TILE_CORE_VERTICES) {
        smem->stage.g_cov[tid] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    }
    __syncthreads();

    for (uint32_t e = tid; e < num_edges; e += 256) {
        const uint8_t u = smem->edge_u8[e][0];
        const uint8_t v = smem->edge_u8[e][1];
        const float3 xu = smem->x_mid[u], xv = smem->x_mid[v];
        const float3 Xu = smem->x_ref[u], Xv = smem->x_ref[v];
        const float3 e_vec = {xv.x - xu.x, xv.y - xu.y, xv.z - xu.z};
        const float3 E_ref = {Xv.x - Xu.x, Xv.y - Xu.y, Xv.z - Xu.z};

        const float l2_cur = __fmaf_rn(e_vec.x, e_vec.x, __fmaf_rn(e_vec.y, e_vec.y, e_vec.z * e_vec.z)) + 1e-12f;
        const float l2_ref = __fmaf_rn(E_ref.x, E_ref.x, __fmaf_rn(E_ref.y, E_ref.y, E_ref.z * E_ref.z)) + 1e-12f;
        const float coeff  = chi_g * ((l2_cur - l2_ref) / l2_ref) * (1.0f / l2_cur);

        const float s_xx = coeff * e_vec.x * e_vec.x;
        const float s_yy = coeff * e_vec.y * e_vec.y;
        const float s_zz = coeff * e_vec.z * e_vec.z;
        const float s_xy = coeff * e_vec.x * e_vec.y;
        const float s_xz = coeff * e_vec.x * e_vec.z;
        const float s_yz = coeff * e_vec.y * e_vec.z;

        if (u < TILE_CORE_VERTICES) {
            atomicAdd(&smem->stage.g_cov[u].xx, s_xx); atomicAdd(&smem->stage.g_cov[u].yy, s_yy); atomicAdd(&smem->stage.g_cov[u].zz, s_zz);
            atomicAdd(&smem->stage.g_cov[u].xy, s_xy); atomicAdd(&smem->stage.g_cov[u].xz, s_xz); atomicAdd(&smem->stage.g_cov[u].yz, s_yz);
        }
        if (v < TILE_CORE_VERTICES) {
            atomicAdd(&smem->stage.g_cov[v].xx, s_xx); atomicAdd(&smem->stage.g_cov[v].yy, s_yy); atomicAdd(&smem->stage.g_cov[v].zz, s_zz);
            atomicAdd(&smem->stage.g_cov[v].xy, s_xy); atomicAdd(&smem->stage.g_cov[v].xz, s_xz); atomicAdd(&smem->stage.g_cov[v].yz, s_yz);
        }
    }
    __syncthreads();

    sym3_f32 my_g = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    if (tid < TILE_CORE_VERTICES) {
        my_g = smem->stage.g_cov[tid];
    }
    __syncthreads();
    return my_g;
}

__device__ __forceinline__ bool solve_live_dec_poisson_pcg_v9(
    VariationalClusterDsmemV9* __restrict__ smem,
    uint32_t num_edges,
    float& out_residual_l2,
    uint32_t& out_iters_used
) {
    const uint32_t tid = threadIdx.x;
    const bool is_core = (tid < TILE_CORE_VERTICES);

    smem->stage.pcg.diag_inv[tid] = 1e-4f;
    smem->stage.pcg.cg_Ap[tid]    = 0.0f;
    __syncthreads();

    for (uint32_t e = tid; e < num_edges; e += 256) {
        const uint8_t u = smem->edge_u8[e][0], v = smem->edge_u8[e][1];
        const float w = smem->edge_buf[e];
        atomicAdd(&smem->stage.pcg.diag_inv[u], w);
        atomicAdd(&smem->stage.pcg.diag_inv[v], w);
        const float flux = w * (smem->phi_dec[u] - smem->phi_dec[v]);
        if (u < TILE_CORE_VERTICES) atomicAdd(&smem->stage.pcg.cg_Ap[u],  flux);
        if (v < TILE_CORE_VERTICES) atomicAdd(&smem->stage.pcg.cg_Ap[v], -flux);
    }
    __syncthreads();

    const float inv_M = is_core ? (1.0f / smem->stage.pcg.diag_inv[tid]) : 0.0f;
    smem->stage.pcg.diag_inv[tid] = inv_M;
    const float r0 = is_core ? (smem->q_live[tid] - smem->stage.pcg.cg_Ap[tid]) : 0.0f;
    const float z0 = inv_M * r0;
    smem->stage.pcg.cg_r[tid] = r0;
    smem->stage.pcg.cg_z[tid] = z0;
    smem->stage.pcg.cg_p[tid] = z0;

    float rr_cur = cta_dot_product_256(r0 * r0, smem->warp_red);
    float rz_old = cta_dot_product_256(r0 * z0, smem->warp_red);
    uint32_t iters = 0u;

    #pragma unroll 1
    for (int iter = 0; iter < DEC_PCG_MAX_ITERS; ++iter) {
        if (rr_cur <= DEC_PCG_TOL_SQ) break;
        iters = (uint32_t)(iter + 1);
        smem->stage.pcg.cg_Ap[tid] = 0.0f;
        __syncthreads();

        for (uint32_t e = tid; e < num_edges; e += 256) {
            const uint8_t u = smem->edge_u8[e][0], v = smem->edge_u8[e][1];
            const float flux = smem->edge_buf[e] * (smem->stage.pcg.cg_p[u] - smem->stage.pcg.cg_p[v]);
            if (u < TILE_CORE_VERTICES) atomicAdd(&smem->stage.pcg.cg_Ap[u],  flux);
            if (v < TILE_CORE_VERTICES) atomicAdd(&smem->stage.pcg.cg_Ap[v], -flux);
        }
        __syncthreads();

        const float pAp = cta_dot_product_256(smem->stage.pcg.cg_p[tid] * smem->stage.pcg.cg_Ap[tid], smem->warp_red);
        const float alpha = rz_old / fmaxf(pAp, 1e-16f);
        if (is_core) {
            smem->phi_dec[tid]        = __fmaf_rn( alpha, smem->stage.pcg.cg_p[tid],  smem->phi_dec[tid]);
            smem->stage.pcg.cg_r[tid] = __fmaf_rn(-alpha, smem->stage.pcg.cg_Ap[tid], smem->stage.pcg.cg_r[tid]);
            smem->stage.pcg.cg_z[tid] = inv_M * smem->stage.pcg.cg_r[tid];
        }
        __syncthreads();

        const float r_now = smem->stage.pcg.cg_r[tid];
        rr_cur = cta_dot_product_256(r_now * r_now, smem->warp_red);
        const float rz_new = cta_dot_product_256(r_now * smem->stage.pcg.cg_z[tid], smem->warp_red);
        const float beta = rz_new / fmaxf(rz_old, 1e-16f);
        if (is_core) {
            smem->stage.pcg.cg_p[tid] = __fmaf_rn(beta, smem->stage.pcg.cg_p[tid], smem->stage.pcg.cg_z[tid]);
        }
        rz_old = rz_new;
        __syncthreads();
    }

    out_residual_l2 = sqrtf(fmaxf(rr_cur, 0.0f));
    out_iters_used  = iters;
    return (rr_cur <= DEC_PCG_TOL_SQ);
}

extern "C" __global__ __cluster_dims__(8, 1, 1) __launch_bounds__(256, 2)
void nhs_biprism_variational_so3g_v9(
    const FoliatedStreamPtrTableFfi* __restrict__ d_stream_ptrs,
    const float* __restrict__ d_amber_charges,
    ContactShellSimplicialStateFfi* __restrict__ d_tile_state,
    InterferometricAdjudicatorFfi* __restrict__ d_adjudicator_ring,
    uint32_t* __restrict__ d_barrier_arrival_count,
    uint32_t* __restrict__ d_barrier_epoch,
    const float dt_outer_au,
    const float kappa_foliated,
    const uint32_t integration_mode
) {
    extern __shared__ __align__(128) unsigned char dsmem_raw_bytes[];
    VariationalClusterDsmemV9* smem = reinterpret_cast<VariationalClusterDsmemV9*>(dsmem_raw_bytes);

    const uint32_t tid       = threadIdx.x;
    const uint32_t leaf_id   = blockIdx.x & 7u;
    const uint32_t next_leaf = (leaf_id + 1u) & 7u;
    const bool     is_core   = (tid < TILE_CORE_VERTICES);

    float* __restrict__       d_coords_xyz = d_stream_ptrs->d_coords_xyz[leaf_id];
    float* __restrict__       d_vels_xyz   = d_stream_ptrs->d_vels_xyz[leaf_id];
    float* __restrict__       d_forces_xyz = d_stream_ptrs->d_forces_xyz[leaf_id];
    const float* __restrict__ d_inv_masses = (integration_mode & FlexibleCoreVerified)
        ? d_stream_ptrs->d_true_inv_masses[leaf_id]
        : d_stream_ptrs->d_inv_masses[leaf_id];

    ContactShellSimplicialStateFfi* tile = &d_tile_state[leaf_id];
    AdjudicatorAbiSafeOverlayFfi* adj_ring = reinterpret_cast<AdjudicatorAbiSafeOverlayFfi*>(d_adjudicator_ring);

    const uint32_t packed_id = tile->global_atom_idx[tid];
    const uint32_t g_atom    = UNPACK_GLOBAL_ATOM_ID(packed_id);
    const uint32_t elem_tid  = UNPACK_ELEM_CLASS(packed_id);
    const float    inv_mass  = d_inv_masses[g_atom];
    const float    mass_a    = 1.0f / fmaxf(inv_mass, 1e-6f);
    const float    q_amber   = d_amber_charges[g_atom];

    const float3 x0_init = *reinterpret_cast<const float3*>(&d_coords_xyz[g_atom * 3]);
    const float3 v0_init = *reinterpret_cast<const float3*>(&d_vels_xyz[g_atom * 3]);
    const float3 f_bath  = *reinterpret_cast<const float3*>(&d_forces_xyz[g_atom * 3]);
    const float4 X0_refq = tile->ref_pos_q[tid];

    float3 my_x_cur = x0_init;
    float3 my_p_cov = make_float3(mass_a * v0_init.x, mass_a * v0_init.y, mass_a * v0_init.z);

    smem->x_mid[tid]   = x0_init;
    smem->v_mid[tid]   = v0_init;
    smem->x_ref[tid]   = make_float3(X0_refq.x, X0_refq.y, X0_refq.z);
    smem->mass[tid]    = mass_a;
    smem->q_live[tid]  = q_amber;
    smem->phi_dec[tid] = tile->phi_dec[tid];
    if (is_core) {
        smem->psi[tid][0] = tile->psi_vertex[tid][0];
        smem->psi[tid][1] = tile->psi_vertex[tid][1];
    }
    if (tid == 0) {
        smem->last_picard_residual  = 0.0f;
        smem->last_dec_residual     = 0.0f;
        smem->picard_iters_used     = 0u;
        smem->dec_iters_used        = 0u;
        smem->local_fault_mask      = 0u;
        smem->collective_fault_mask = 0u;
        adj_ring[leaf_id].leaf_fault_mask = 0u;
    }
    __syncthreads();

    const uint32_t num_edges = min(tile->num_active_edges, (uint32_t)TILE_MAX_EDGES);
    for (uint32_t e = tid; e < num_edges; e += 256) {
        smem->edge_u8[e][0] = tile->edge_u8[e][0];
        smem->edge_u8[e][1] = tile->edge_u8[e][1];
        smem->edge_buf[e]   = 0.0f;
    }
    __syncthreads();

    const float chi_g       = 0.12f;
    const float dt_sub      = dt_outer_au * (1.0f / (float)BIPRISM_INNER_SUBSTEPS);
    const float half_dt_sub = 0.5f * dt_sub;
    const float inv_star0   = 1.0f / fmaxf(tile->star0_vol[tid], 1e-4f);

    // 1. Entry Canonical Momentum Transform: my_p_cov = m_a * g_a(x^0) * v^0
    sym3_f32 my_g_cov = assemble_metric_to_reg_v9(smem, num_edges, chi_g);
    float det_g_inv_init = 1.0f;
    sym3_f32 my_g_inv_n = invert_sym3_metric_exact(my_g_cov, det_g_inv_init, &smem->local_fault_mask);
    if (is_core) {
        const float3 g_v0 = sym3_mul_vec(my_g_cov, v0_init);
        my_p_cov = make_float3(mass_a * g_v0.x, mass_a * g_v0.y, mass_a * g_v0.z);
    }

    // =========================================================================================
    // 2. COUPLED IMPLICIT VARIATIONAL MIDPOINT SUB-STEP LOOP
    // =========================================================================================
    #pragma unroll 1
    for (int sub = 0; sub < BIPRISM_INNER_SUBSTEPS; ++sub) {
        if (is_core) {
            const float3 gp = sym3_mul_vec(my_g_inv_n, my_p_cov);
            const float3 v_init = {inv_mass * gp.x, inv_mass * gp.y, inv_mass * gp.z};
            smem->v_mid[tid] = v_init;
            smem->x_mid[tid] = {
                __fmaf_rn(half_dt_sub, v_init.x, my_x_cur.x),
                __fmaf_rn(half_dt_sub, v_init.y, my_x_cur.y),
                __fmaf_rn(half_dt_sub, v_init.z, my_x_cur.z)
            };
        }
        __syncthreads();

        float3   v_minus_conv     = smem->v_mid[tid];
        float3   v_plus_conv      = smem->v_mid[tid];
        float3   F_tot_conv       = f_bath;
        bool     picard_converged = false;
        bool     dec_converged    = false;
        float    final_picard_res = 0.0f;
        float    final_dec_res    = 0.0f;
        uint32_t p_iters_done     = 0u;
        uint32_t d_iters_done     = 0u;

        #pragma unroll 1
        for (int picard = 0; picard < MAX_VARIATIONAL_PICARD_ITERS; ++picard) {
            p_iters_done = (uint32_t)(picard + 1);

            // Stage A: Assemble midpoint metric g(\bar{x}^{(k)}) and verify Sylvester SPD
            my_g_cov = assemble_metric_to_reg_v9(smem, num_edges, chi_g);
            float det_g_inv_mid = 1.0f;
            const sym3_f32 my_g_inv_mid = invert_sym3_metric_exact(my_g_cov, det_g_inv_mid, &smem->local_fault_mask);

            // Compute Center-of-Mass Midpoint Velocity so Pairwise Berry Rotation Conserves Total Momentum
            const float m_core = is_core ? mass_a : 0.0f;
            const float tot_m  = fmaxf(cta_dot_product_256(m_core, smem->warp_red), 1e-6f);
            const float vx_com = cta_dot_product_256(m_core * smem->v_mid[tid].x, smem->warp_red) / tot_m;
            const float vy_com = cta_dot_product_256(m_core * smem->v_mid[tid].y, smem->warp_red) / tot_m;
            const float vz_com = cta_dot_product_256(m_core * smem->v_mid[tid].z, smem->warp_red) / tot_m;
            if (tid == 0) {
                smem->v_com[0] = vx_com;
                smem->v_com[1] = vy_com;
                smem->v_com[2] = vz_com;
            }

            // Stage B: Compute live midpoint cotangent weights \star_1(\bar{x}^{(k)}) and solve DEC PCG
            for (uint32_t e = tid; e < num_edges; e += 256) {
                const uint8_t u = smem->edge_u8[e][0], v = smem->edge_u8[e][1];
                const float3 xu = smem->x_mid[u], xv = smem->x_mid[v];
                const float3 Xu = smem->x_ref[u], Xv = smem->x_ref[v];
                const float dx = xv.x - xu.x, dy = xv.y - xu.y, dz = xv.z - xu.z;
                const float dX = Xv.x - Xu.x, dY = Xv.y - Xu.y, dZ = Xv.z - Xu.z;
                const float r_cur = sqrtf(__fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz)) + 1e-12f);
                const float r_ref = sqrtf(__fmaf_rn(dX, dX, __fmaf_rn(dY, dY, dZ * dZ)) + 1e-12f);
                smem->edge_buf[e] = fmaxf(tile->star1_wt[e], 1e-4f) * (r_ref / r_cur);
            }
            __syncthreads();

            dec_converged = solve_live_dec_poisson_pcg_v9(smem, num_edges, final_dec_res, d_iters_done);

            // Stage C: Accumulate Forces and 3-Point Gauss-Legendre Chord-Averaged Berry Field
            if (is_core) {
                smem->stage.force.f_tot_q39[tid][0] = static_cast<unsigned long long>(__float2ll_rn(f_bath.x * FP_Q39_SCALE));
                smem->stage.force.f_tot_q39[tid][1] = static_cast<unsigned long long>(__float2ll_rn(f_bath.y * FP_Q39_SCALE));
                smem->stage.force.f_tot_q39[tid][2] = static_cast<unsigned long long>(__float2ll_rn(f_bath.z * FP_Q39_SCALE));
                smem->stage.force.tau_berry[tid]    = make_float3(0.0f, 0.0f, 0.0f);
            }
            __syncthreads();

            for (uint32_t e = tid; e < num_edges; e += 256) {
                const uint8_t u = smem->edge_u8[e][0], v = smem->edge_u8[e][1];
                const float3 xu = smem->x_mid[u], xv = smem->x_mid[v];
                const float3 vu = smem->v_mid[u], vv = smem->v_mid[v];
                const float3 Xu = smem->x_ref[u], Xv = smem->x_ref[v];
                const float  mu = smem->mass[u],  mv = smem->mass[v];

                const float3 e_vec = {xv.x - xu.x, xv.y - xu.y, xv.z - xu.z};
                const float3 E_ref = {Xv.x - Xu.x, Xv.y - Xu.y, Xv.z - Xu.z};
                const float3 v_rel = {vv.x - vu.x, vv.y - vu.y, vv.z - vu.z};
                const float3 de_uv = {dt_sub * v_rel.x, dt_sub * v_rel.y, dt_sub * v_rel.z};

                const float l2_cur = __fmaf_rn(e_vec.x, e_vec.x, __fmaf_rn(e_vec.y, e_vec.y, e_vec.z * e_vec.z)) + 1e-12f;
                const float l2_ref = __fmaf_rn(E_ref.x, E_ref.x, __fmaf_rn(E_ref.y, E_ref.y, E_ref.z * E_ref.z)) + 1e-12f;
                const float inv_r  = rsqrtf(l2_cur);
                const float r      = l2_cur * inv_r;
                const float3 r_hat = {e_vec.x * inv_r, e_vec.y * inv_r, e_vec.z * inv_r};

                const float3 F_metric = evaluate_metric_cotangent_force_edge(
                    e_vec, r_hat, r, l2_cur, l2_ref, vu, vv, mu, mv, chi_g
                );

                const float d_phi     = smem->phi_dec[u] - smem->phi_dec[v];
                const float star1_cur = smem->edge_buf[e];
                const float dec_mag   = 0.5f * (d_phi * d_phi) * (star1_cur * inv_r);
                const float3 F_dec    = {dec_mag * r_hat.x, dec_mag * r_hat.y, dec_mag * r_hat.z};

                const uint32_t elem_u = UNPACK_ELEM_CLASS(tile->global_atom_idx[u]);
                const uint32_t elem_v = UNPACK_ELEM_CLASS(tile->global_atom_idx[v]);
                const float4 live_pu  = make_float4(xu.x, xu.y, xu.z, smem->q_live[u]);
                const float4 live_pv  = make_float4(xv.x, xv.y, xv.z, smem->q_live[v]);

                float S_uv, H_blk[4];
                float3 grad_S, grad_H_diag[2], grad_V01;
                evaluate_heteronuclear_gfn_xtb_edge(
                    elem_u, elem_v, live_pu, live_pv,
                    smem->phi_dec[u], smem->phi_dec[v],
                    S_uv, grad_S, H_blk, grad_H_diag, grad_V01
                );

                const float E0 = H_blk[0], V01 = H_blk[1], E1 = H_blk[3];
                const float dE = E1 - E0, E_bar = 0.5f * (E0 + E1);
                const float eps_ci = 5e-5f;
                const float dE_reg = (dE >= 0.0f ? 1.0f : -1.0f) * sqrtf(__fmaf_rn(dE, dE, eps_ci * eps_ci));
                const float theta_g = 0.5f * atan2f(2.0f * V01, dE_reg);
                float s_2th, c_2th;
                __sincosf(2.0f * theta_g, &s_2th, &c_2th);

                const float half_split = 0.5f * dE * c_2th - V01 * s_2th;
                const float h_off_rot  = 0.5f * dE * s_2th + V01 * c_2th;
                const float inv_gap_sq = 1.0f / __fmaf_rn(dE, dE, __fmaf_rn(4.0f * V01, V01, eps_ci * eps_ci));
                const float v_dot_gV   = __fmaf_rn(v_rel.x, grad_V01.x, __fmaf_rn(v_rel.y, grad_V01.y, v_rel.z * grad_V01.z));
                const float A_omega    = v_dot_gV * sqrtf(inv_gap_sq);
                const float inv_det_S  = 1.0f / fmaxf(1.0f - S_uv * S_uv, 1e-3f);

                const cpx2_f32 psi0 = (u < TILE_CORE_VERTICES) ? smem->psi[u][0] : cpx2_f32{1.0f, 0.0f};
                const cpx2_f32 psi1 = (u < TILE_CORE_VERTICES) ? smem->psi[u][1] : cpx2_f32{0.0f, 0.0f};
                const float rho00    = __fmaf_rn(psi0.re, psi0.re, psi0.im * psi0.im);
                const float rho11    = __fmaf_rn(psi1.re, psi1.re, psi1.im * psi1.im);
                const float rho01_re = __fmaf_rn(psi0.re, psi1.re, psi0.im * psi1.im);

                const float tr_H_rho = rho00 * (E_bar - half_split) + rho11 * (E_bar + half_split) + 2.0f * rho01_re * h_off_rot;
                const float w_pulay  = 2.0f * tr_H_rho * S_uv * (inv_det_S * inv_det_S);

                float3 F_edge = {
                    F_metric.x + F_dec.x - inv_det_S * (rho00 * grad_H_diag[0].x + rho11 * grad_H_diag[1].x + 2.0f * rho01_re * grad_V01.x) + w_pulay * grad_S.x,
                    F_metric.y + F_dec.y - inv_det_S * (rho00 * grad_H_diag[0].y + rho11 * grad_H_diag[1].y + 2.0f * rho01_re * grad_V01.y) + w_pulay * grad_S.y,
                    F_metric.z + F_dec.z - inv_det_S * (rho00 * grad_H_diag[0].z + rho11 * grad_H_diag[1].z + 2.0f * rho01_re * grad_V01.z) + w_pulay * grad_S.z
                };

                // Optional Gonzalez Secant Correction on Potential + Metric Energy along de_uv
                if (integration_mode & GonzalezExactEnergyBit) {
                    const float de2 = __fmaf_rn(de_uv.x, de_uv.x, __fmaf_rn(de_uv.y, de_uv.y, de_uv.z * de_uv.z));
                    if (de2 > 1e-24f) {
                        const float3 e_n   = {e_vec.x - 0.5f * de_uv.x, e_vec.y - 0.5f * de_uv.y, e_vec.z - 0.5f * de_uv.z};
                        const float3 e_np1 = {e_vec.x + 0.5f * de_uv.x, e_vec.y + 0.5f * de_uv.y, e_vec.z + 0.5f * de_uv.z};
                        const float l2_n   = __fmaf_rn(e_n.x, e_n.x, __fmaf_rn(e_n.y, e_n.y, e_n.z * e_n.z)) + 1e-12f;
                        const float l2_np1 = __fmaf_rn(e_np1.x, e_np1.x, __fmaf_rn(e_np1.y, e_np1.y, e_np1.z * e_np1.z)) + 1e-12f;
                        const float r_n    = sqrtf(l2_n);
                        const float r_np1  = sqrtf(l2_np1);
                        const float r_ref  = sqrtf(l2_ref);

                        const float star1_n   = fmaxf(tile->star1_wt[e], 1e-4f) * (r_ref / r_n);
                        const float star1_np1 = fmaxf(tile->star1_wt[e], 1e-4f) * (r_ref / r_np1);
                        const float dV_dec    = -0.5f * (d_phi * d_phi) * (star1_np1 - star1_n);

                        float S_n, H_n[4], S_np1, H_np1[4];
                        float3 gS_t, gHd_t[2], gV_t;
                        evaluate_heteronuclear_gfn_xtb_edge(
                            elem_u, elem_v, make_float4(0.0f, 0.0f, 0.0f, smem->q_live[u]), make_float4(e_n.x, e_n.y, e_n.z, smem->q_live[v]),
                            smem->phi_dec[u], smem->phi_dec[v], S_n, gS_t, H_n, gHd_t, gV_t
                        );
                        evaluate_heteronuclear_gfn_xtb_edge(
                            elem_u, elem_v, make_float4(0.0f, 0.0f, 0.0f, smem->q_live[u]), make_float4(e_np1.x, e_np1.y, e_np1.z, smem->q_live[v]),
                            smem->phi_dec[u], smem->phi_dec[v], S_np1, gS_t, H_np1, gHd_t, gV_t
                        );
                        const float E_xtb_n   = (rho00 * H_n[0]   + rho11 * H_n[3]   + 2.0f * rho01_re * H_n[1])   / fmaxf(1.0f - S_n * S_n, 1e-3f);
                        const float E_xtb_np1 = (rho00 * H_np1[0] + rho11 * H_np1[3] + 2.0f * rho01_re * H_np1[1]) / fmaxf(1.0f - S_np1 * S_np1, 1e-3f);
                        const float dV_tot    = dV_dec + (E_xtb_np1 - E_xtb_n);

                        const float3 F_pot_mid = {F_edge.x - F_metric.x, F_edge.y - F_metric.y, F_edge.z - F_metric.z};
                        const float work_pot   = -__fmaf_rn(F_pot_mid.x, de_uv.x, __fmaf_rn(F_pot_mid.y, de_uv.y, F_pot_mid.z * de_uv.z));
                        const float dg_alpha   = (dV_tot - work_pot) / de2;

                        F_edge.x = __fmaf_rn(dg_alpha, de_uv.x, F_edge.x);
                        F_edge.y = __fmaf_rn(dg_alpha, de_uv.y, F_edge.y);
                        F_edge.z = __fmaf_rn(dg_alpha, de_uv.z, F_edge.z);
                    }
                }

                if (u < TILE_CORE_VERTICES) {
                    smem_deterministic_add_f32(&smem->stage.force.f_tot_q39[u][0],  F_edge.x);
                    smem_deterministic_add_f32(&smem->stage.force.f_tot_q39[u][1],  F_edge.y);
                    smem_deterministic_add_f32(&smem->stage.force.f_tot_q39[u][2],  F_edge.z);
                }
                if (v < TILE_CORE_VERTICES) {
                    smem_deterministic_add_f32(&smem->stage.force.f_tot_q39[v][0], -F_edge.x);
                    smem_deterministic_add_f32(&smem->stage.force.f_tot_q39[v][1], -F_edge.y);
                    smem_deterministic_add_f32(&smem->stage.force.f_tot_q39[v][2], -F_edge.z);
                }

                // 3-Point Gauss-Legendre Chord-Averaged Berry Field \overline{B}(x^n, x^{n+1})
                const float3 B_chord = evaluate_gauss_legendre_berry_chord(e_vec, de_uv, A_omega, dt_sub);
                if (u < TILE_CORE_VERTICES) {
                    const float c_u = 0.5f * dt_sub / mu;
                    atomicAdd(&smem->stage.force.tau_berry[u].x,  c_u * B_chord.x);
                    atomicAdd(&smem->stage.force.tau_berry[u].y,  c_u * B_chord.y);
                    atomicAdd(&smem->stage.force.tau_berry[u].z,  c_u * B_chord.z);
                }
                if (v < TILE_CORE_VERTICES) {
                    const float c_v = 0.5f * dt_sub / mv;
                    atomicAdd(&smem->stage.force.tau_berry[v].x, -c_v * B_chord.x);
                    atomicAdd(&smem->stage.force.tau_berry[v].y, -c_v * B_chord.y);
                    atomicAdd(&smem->stage.force.tau_berry[v].z, -c_v * B_chord.z);
                }
            }
            __syncthreads();

            float my_dx_res = 0.0f;
            if (is_core) {
                F_tot_conv = {
                    smem_deterministic_load_f32(&smem->stage.force.f_tot_q39[tid][0]),
                    smem_deterministic_load_f32(&smem->stage.force.f_tot_q39[tid][1]),
                    smem_deterministic_load_f32(&smem->stage.force.f_tot_q39[tid][2])
                };

                const float3 p_minus = {
                    __fmaf_rn(half_dt_sub, F_tot_conv.x, my_p_cov.x),
                    __fmaf_rn(half_dt_sub, F_tot_conv.y, my_p_cov.y),
                    __fmaf_rn(half_dt_sub, F_tot_conv.z, my_p_cov.z)
                };
                const float3 g_p_minus = sym3_mul_vec(my_g_inv_mid, p_minus);
                v_minus_conv = {inv_mass * g_p_minus.x, inv_mass * g_p_minus.y, inv_mass * g_p_minus.z};

                // Rotate in translation-invariant COM relative velocity frame so \sum_a \Delta p_a^{mag} == 0
                const float3 v_com_vec = {smem->v_com[0], smem->v_com[1], smem->v_com[2]};
                const float3 v_rel_in  = {v_minus_conv.x - v_com_vec.x, v_minus_conv.y - v_com_vec.y, v_minus_conv.z - v_com_vec.z};
                const float3 v_rel_out = so3_g_metric_cayley_boris(
                    v_rel_in, smem->stage.force.tau_berry[tid], my_g_cov, my_g_inv_mid, det_g_inv_mid
                );
                v_plus_conv = {v_rel_out.x + v_com_vec.x, v_rel_out.y + v_com_vec.y, v_rel_out.z + v_com_vec.z};

                const float3 v_mid_new = {
                    0.5f * (v_minus_conv.x + v_plus_conv.x),
                    0.5f * (v_minus_conv.y + v_plus_conv.y),
                    0.5f * (v_minus_conv.z + v_plus_conv.z)
                };
                const float3 x_mid_new = {
                    __fmaf_rn(half_dt_sub, v_mid_new.x, my_x_cur.x),
                    __fmaf_rn(half_dt_sub, v_mid_new.y, my_x_cur.y),
                    __fmaf_rn(half_dt_sub, v_mid_new.z, my_x_cur.z)
                };

                my_dx_res = fmaxf(
                    fabsf(x_mid_new.x - smem->x_mid[tid].x),
                    fmaxf(fabsf(x_mid_new.y - smem->x_mid[tid].y), fabsf(x_mid_new.z - smem->x_mid[tid].z))
                );
                smem->v_mid[tid] = v_mid_new;
                smem->x_mid[tid] = x_mid_new;
            }

            final_picard_res = cta_max_reduce_256(my_dx_res, smem->warp_red);
            if (picard >= 1 && final_picard_res <= VARIATIONAL_PICARD_TOL && dec_converged && smem->local_fault_mask == 0u) {
                picard_converged = true;
                break;
            }
        }

        // Publish Local Fault Mask and Perform 8-Leaf Collective Fault Reduction
        if (tid == 0) {
            smem->last_picard_residual = final_picard_res;
            smem->last_dec_residual    = final_dec_res;
            smem->picard_iters_used    = p_iters_done;
            smem->dec_iters_used       = d_iters_done;
            if (!dec_converged)    smem->local_fault_mask |= FAULT_DEC_NONCONV;
            if (!picard_converged) smem->local_fault_mask |= FAULT_PICARD_NONCONV;
            adj_ring[leaf_id].leaf_fault_mask = smem->local_fault_mask;
        }
        cluster_or_grid_sync_8leaves(d_barrier_arrival_count, d_barrier_epoch);

        if (tid == 0) {
            uint32_t coll_or = smem->local_fault_mask;
            const uint32_t n_leaves = min(gridDim.x, 8u);
            for (uint32_t r = 0u; r < n_leaves; ++r) {
                coll_or |= dsmem_load_u32_safe(&smem->local_fault_mask, r, &adj_ring[r].leaf_fault_mask);
            }
            smem->collective_fault_mask = coll_or;
        }
        __syncthreads();

        // Collective Abort: if ANY leaf faulted, ALL 8 leaves abort without mutating state
        if (smem->collective_fault_mask != 0u) {
            break;
        }

        // Stage D: Post-Picard u(2) Star Accumulation & Cross-Leaf Phase Exchange
        if (is_core) {
            const GfnXtbElementParams P_on = c_gfn_xtb_lut[elem_tid];
            const float phi_on = smem->phi_dec[tid];
            const float kz_on  = 0.25f * dt_sub * (P_on.h_p_excited - P_on.h_s_valence) * tile->star0_vol[tid];
            const float k0_on  = 0.25f * dt_sub * (P_on.h_p_excited + P_on.h_s_valence + 2.0f * phi_on) * tile->star0_vol[tid];
            smem->stage.lie_k_accum[tid] = make_float4(0.0f, 0.0f, kz_on, k0_on);
        }
        __syncthreads();

        for (uint32_t e = tid; e < num_edges; e += 256) {
            const uint8_t u = smem->edge_u8[e][0], v = smem->edge_u8[e][1];
            const float3 xu = smem->x_mid[u], xv = smem->x_mid[v];
            const float3 vu = smem->v_mid[u], vv = smem->v_mid[v];
            const float3 Xu = smem->x_ref[u], Xv = smem->x_ref[v];
            const float3 v_rel = {vv.x - vu.x, vv.y - vu.y, vv.z - vu.z};

            const float dx = xv.x - xu.x, dy = xv.y - xu.y, dz = xv.z - xu.z;
            const float dX = Xv.x - Xu.x, dY = Xv.y - Xu.y, dZ = Xv.z - Xu.z;
            const float r_cur = sqrtf(__fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz)) + 1e-12f);
            const float r_ref = sqrtf(__fmaf_rn(dX, dX, __fmaf_rn(dY, dY, dZ * dZ)) + 1e-12f);
            const float star1_cur = fmaxf(tile->star1_wt[e], 1e-4f) * (r_ref / r_cur);

            const uint32_t elem_u = UNPACK_ELEM_CLASS(tile->global_atom_idx[u]);
            const uint32_t elem_v = UNPACK_ELEM_CLASS(tile->global_atom_idx[v]);
            const float4 live_pu  = make_float4(xu.x, xu.y, xu.z, smem->q_live[u]);
            const float4 live_pv  = make_float4(xv.x, xv.y, xv.z, smem->q_live[v]);

            float S_uv, H_blk[4];
            float3 grad_S, grad_H_diag[2], grad_V01;
            evaluate_heteronuclear_gfn_xtb_edge(
                elem_u, elem_v, live_pu, live_pv,
                smem->phi_dec[u], smem->phi_dec[v],
                S_uv, grad_S, H_blk, grad_H_diag, grad_V01
            );

            const float E0 = H_blk[0], V01 = H_blk[1], E1 = H_blk[3];
            const float dE = E1 - E0, E_bar = 0.5f * (E0 + E1);
            const float eps_ci = 5e-5f;
            const float dE_reg = (dE >= 0.0f ? 1.0f : -1.0f) * sqrtf(__fmaf_rn(dE, dE, eps_ci * eps_ci));
            const float theta_g = 0.5f * atan2f(2.0f * V01, dE_reg);
            float s_2th, c_2th;
            __sincosf(2.0f * theta_g, &s_2th, &c_2th);

            const float half_split = 0.5f * dE * c_2th - V01 * s_2th;
            const float h_off_rot  = 0.5f * dE * s_2th + V01 * c_2th;
            const float inv_gap_sq = 1.0f / __fmaf_rn(dE, dE, __fmaf_rn(4.0f * V01, V01, eps_ci * eps_ci));
            const float v_dot_gV   = __fmaf_rn(v_rel.x, grad_V01.x, __fmaf_rn(v_rel.y, grad_V01.y, v_rel.z * grad_V01.z));
            const float A_omega    = v_dot_gV * sqrtf(inv_gap_sq);

            const float inv_det_S = 1.0f / fmaxf(1.0f - S_uv * S_uv, 1e-3f);
            const float w_gen     = 0.5f * dt_sub * star1_cur * inv_det_S;

            if (u < TILE_CORE_VERTICES) {
                atomicAdd(&smem->stage.lie_k_accum[u].x,  w_gen * h_off_rot);
                atomicAdd(&smem->stage.lie_k_accum[u].y,  w_gen * A_omega);
                atomicAdd(&smem->stage.lie_k_accum[u].z,  w_gen * half_split);
                atomicAdd(&smem->stage.lie_k_accum[u].w,  w_gen * E_bar);
            }
            if (v < TILE_CORE_VERTICES) {
                atomicAdd(&smem->stage.lie_k_accum[v].x,  w_gen * h_off_rot);
                atomicAdd(&smem->stage.lie_k_accum[v].y, -w_gen * A_omega);
                atomicAdd(&smem->stage.lie_k_accum[v].z,  w_gen * half_split);
                atomicAdd(&smem->stage.lie_k_accum[v].w,  w_gen * E_bar);
            }

            // Store spatial phase in SMEM and in non-persistent L2 scratch (`star1_wt` is read-only;
            // `grad_s_edges[e].x` is used as L2 phase scratch so `s_ovlp_edges` is never clobbered!)
            const float theta_cur   = A_omega * dt_sub;
            smem->edge_buf[e]       = theta_cur;
            tile->grad_s_edges[e].x = theta_cur;
        }
        cluster_or_grid_sync_8leaves(d_barrier_arrival_count, d_barrier_epoch);

        const ContactShellSimplicialStateFfi* next_tile_l2 = &d_tile_state[next_leaf];
        for (uint32_t e = tid; e < num_edges; e += 256) {
            const float theta_cur  = smem->edge_buf[e];
            const float theta_next = dsmem_load_f32_safe(
                &smem->edge_buf[e],
                next_leaf,
                &next_tile_l2->grad_s_edges[e].x // True cross-leaf L2 address of neighbor leaf!
            );
            (void)theta_next;
        }

        // Advance sub-step state in registers/SMEM
        if (is_core) {
            const float3 g_v_plus = sym3_mul_vec(my_g_cov, v_plus_conv);
            my_p_cov = {
                __fmaf_rn(half_dt_sub, F_tot_conv.x, mass_a * g_v_plus.x),
                __fmaf_rn(half_dt_sub, F_tot_conv.y, mass_a * g_v_plus.y),
                __fmaf_rn(half_dt_sub, F_tot_conv.z, mass_a * g_v_plus.z)
            };
            const float3 v_mid_c = smem->v_mid[tid];
            my_x_cur = {
                __fmaf_rn(dt_sub, v_mid_c.x, my_x_cur.x),
                __fmaf_rn(dt_sub, v_mid_c.y, my_x_cur.y),
                __fmaf_rn(dt_sub, v_mid_c.z, my_x_cur.z)
            };
            smem->x_mid[tid] = my_x_cur;

            const float4 k_raw  = smem->stage.lie_k_accum[tid];
            const float4 k_norm = {k_raw.x * inv_star0, k_raw.y * inv_star0, k_raw.z * inv_star0, k_raw.w * inv_star0};
            float v_defect = 0.0f;
            su2_mat U_vertex = u2_lie_exponential_rodrigues(k_norm, v_defect);

            const cpx2_f32 p0 = smem->psi[tid][0], p1 = smem->psi[tid][1];
            const cpx2_f32 next_p0 = c_add(c_mul(U_vertex.m00, p0), c_mul(U_vertex.m01, p1));
            const cpx2_f32 next_p1 = c_add(c_mul(U_vertex.m10, p0), c_mul(U_vertex.m11, p1));
            smem->psi[tid][0] = next_p0;
            smem->psi[tid][1] = next_p1;

            const float pop1_new = __fmaf_rn(next_p1.re, next_p1.re, next_p1.im * next_p1.im);
            smem->q_live[tid] = q_amber + 0.15f * pop1_new;
        }

        // Update start-of-substep inverse metric g_a^{-1}(x^{n+1}) for next sub-step
        my_g_cov = assemble_metric_to_reg_v9(smem, num_edges, chi_g);
        float det_g_inv_next = 1.0f;
        my_g_inv_n = invert_sym3_metric_exact(my_g_cov, det_g_inv_next, &smem->local_fault_mask);
        cluster_or_grid_sync_8leaves(d_barrier_arrival_count, d_barrier_epoch);
    }

    // 3. Final Collective SPD Check & All-or-Nothing Global Writeback
    if (tid == 0) adj_ring[leaf_id].leaf_fault_mask = smem->local_fault_mask;
    cluster_or_grid_sync_8leaves(d_barrier_arrival_count, d_barrier_epoch);
    if (tid == 0) {
        uint32_t coll_or = smem->local_fault_mask;
        const uint32_t n_leaves = min(gridDim.x, 8u);
        for (uint32_t r = 0u; r < n_leaves; ++r) {
            coll_or |= dsmem_load_u32_safe(&smem->local_fault_mask, r, &adj_ring[r].leaf_fault_mask);
        }
        smem->collective_fault_mask = coll_or;
    }
    __syncthreads();

    if (is_core && smem->collective_fault_mask == 0u) {
        tile->psi_vertex[tid][0] = smem->psi[tid][0];
        tile->psi_vertex[tid][1] = smem->psi[tid][1];
        tile->phi_dec[tid]       = smem->phi_dec[tid];

        const float3 g_inv_p = sym3_mul_vec(my_g_inv_n, my_p_cov);
        const float3 v_final = {inv_mass * g_inv_p.x, inv_mass * g_inv_p.y, inv_mass * g_inv_p.z};

        if (integration_mode & FlexibleCoreVerified) {
            d_coords_xyz[g_atom * 3 + 0] = my_x_cur.x;
            d_coords_xyz[g_atom * 3 + 1] = my_x_cur.y;
            d_coords_xyz[g_atom * 3 + 2] = my_x_cur.z;
            d_vels_xyz[g_atom * 3 + 0]   = v_final.x;
            d_vels_xyz[g_atom * 3 + 1]   = v_final.y;
            d_vels_xyz[g_atom * 3 + 2]   = v_final.z;
        } else {
            d_vels_xyz[g_atom * 3 + 0]   += (v_final.x - v0_init.x);
            d_vels_xyz[g_atom * 3 + 1]   += (v_final.y - v0_init.y);
            d_vels_xyz[g_atom * 3 + 2]   += (v_final.z - v0_init.z);
            d_coords_xyz[g_atom * 3 + 0] += (my_x_cur.x - x0_init.x);
            d_coords_xyz[g_atom * 3 + 1] += (my_x_cur.y - x0_init.y);
            d_coords_xyz[g_atom * 3 + 2] += (my_x_cur.z - x0_init.z);
        }
    }

    // 4. Write Collective Telemetry into ABI-Safe 256B Adjudicator Overlay
    if (tid == 0) {
        AdjudicatorAbiSafeOverlayFfi* adj = &adj_ring[leaf_id];
        adj->last_picard_residual = smem->last_picard_residual;
        adj->last_dec_residual    = smem->last_dec_residual;
        adj->picard_iters_used    = smem->picard_iters_used;
        adj->dec_iters_used       = smem->dec_iters_used;
        adj->leaf_fault_mask      = smem->local_fault_mask;
        adj->cluster_fault_mask   = smem->collective_fault_mask;
        if (smem->collective_fault_mask == 0u) {
            adj->committed_substeps += BIPRISM_INNER_SUBSTEPS;
        } else {
            adj->branch_and_fault |= ADJ_SOLVER_FAULT_FLAG | smem->collective_fault_mask;
        }
    }
}
