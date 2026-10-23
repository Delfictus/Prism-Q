// File: roles/engine/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh
// Diagonal-DEC tile and transactional oriented-face seeder.
// Bounded extracted-C++ tests verify unique IDs/edges, six-edge incidence, and
// symmetric oriented-face adjacency after prospective-slot reservation.
// They do not establish a complete 3D manifold or Delaunay tessellation:
// random/KRAS controls retain 619/393 edge-plus-vertex link defects and
// empty-sphere/coverage failures. Evidence: INTERNAL_QC/v10_manifold_revision/paths.
// The inner operator uses frozen-base w(x)=base*L0/r; Phase2 geometry refresh
// derivatives and subsequent consumers are separate numerical obligations.

#pragma once
#ifndef NHS_BIPRISM_SIMPLICIAL_TILE_CUH_INCLUDED
#define NHS_BIPRISM_SIMPLICIAL_TILE_CUH_INCLUDED

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include "nhs_biprism_xtb_lut.cuh"
#include "nhs_biprism_coordinate_precision.cuh"
#include "nhs_biprism_coordinate_geometry.cuh"
#include "nhs_biprism_convergence_abi.cuh"
#include "nhs_biprism_electronic_discrete.cuh"

#define TILE_CORE_VERTICES  192
#define TILE_HALO_VERTICES  64
#define TILE_TOTAL_VERTICES 256
#define TILE_MAX_TETS       1024
#define TILE_MAX_EDGES      1536
#define TILE_HASH_SLOTS     2048
#define FULL_WARP_MASK      0xffffffffu

struct __align__(8) cpx2_f32 {
    float re;
    float im;
};

struct __align__(16) su2_mat {
    cpx2_f32 m00, m01, m10, m11;
};

__device__ __forceinline__ cpx2_f32 c_mul(cpx2_f32 a, cpx2_f32 b) {
    return {__fmaf_rn(a.re, b.re, -a.im * b.im), __fmaf_rn(a.re, b.im, a.im * b.re)};
}
__device__ __forceinline__ cpx2_f32 c_conj(cpx2_f32 a) { return {a.re, -a.im}; }
__device__ __forceinline__ cpx2_f32 c_add(cpx2_f32 a, cpx2_f32 b) { return {a.re + b.re, a.im + b.im}; }
__device__ __forceinline__ cpx2_f32 c_scale(cpx2_f32 a, float s) { return {a.re * s, a.im * s}; }

__device__ __forceinline__ su2_mat su2_mul(const su2_mat& A, const su2_mat& B) {
    return {
        c_add(c_mul(A.m00, B.m00), c_mul(A.m01, B.m10)),
        c_add(c_mul(A.m00, B.m01), c_mul(A.m01, B.m11)),
        c_add(c_mul(A.m10, B.m00), c_mul(A.m11, B.m10)),
        c_add(c_mul(A.m10, B.m01), c_mul(A.m11, B.m11))
    };
}

__device__ __forceinline__ su2_mat su2_dagger(const su2_mat& A) {
    return {c_conj(A.m00), c_conj(A.m10), c_conj(A.m01), c_conj(A.m11)};
}

__device__ __forceinline__ su2_mat su2_polar_project(su2_mat U) {
    float norm_sq = __fmaf_rn(U.m00.re, U.m00.re, __fmaf_rn(U.m00.im, U.m00.im,
                    __fmaf_rn(U.m01.re, U.m01.re, __fmaf_rn(U.m01.im, U.m01.im,
                    __fmaf_rn(U.m10.re, U.m10.re, __fmaf_rn(U.m10.im, U.m10.im,
                    __fmaf_rn(U.m11.re, U.m11.re, U.m11.im * U.m11.im)))))));
    float inv_scale = rsqrtf(fmaxf(norm_sq * 0.5f, 1e-12f));
    U.m00 = c_scale(U.m00, inv_scale); U.m01 = c_scale(U.m01, inv_scale);
    U.m10 = c_scale(U.m10, inv_scale); U.m11 = c_scale(U.m11, inv_scale);

    #pragma unroll
    for (int iter = 0; iter < 3; ++iter) {
        su2_mat G = su2_mul(su2_dagger(U), U);
        su2_mat M = {
            {0.5f * (3.0f - G.m00.re), -0.5f * G.m00.im},
            {-0.5f * G.m01.re,         -0.5f * G.m01.im},
            {-0.5f * G.m10.re,         -0.5f * G.m10.im},
            {0.5f * (3.0f - G.m11.re), -0.5f * G.m11.im}
        };
        U = su2_mul(U, M);
    }
    return U;
}

__device__ __forceinline__ su2_mat u2_lie_exponential_rodrigues(float4 k_u2, float& out_defect) {
    const float k0 = k_u2.w;
    const float kx = k_u2.x, ky = k_u2.y, kz = k_u2.z;
    const float theta2 = __fmaf_rn(kx, kx, __fmaf_rn(ky, ky, kz * kz));
    const float theta  = sqrtf(theta2);

    float s_th, c_th;
    __sincosf(theta, &s_th, &c_th);
    const float sinc_th = (theta > 1e-4f) ? (s_th / theta) : (1.0f - 0.16666667f * theta2);

    float s_k0, c_k0;
    __sincosf(-k0, &s_k0, &c_k0);
    const cpx2_f32 phase = {c_k0, s_k0};

    const float q0 = c_th;
    const float qx = -sinc_th * kx;
    const float qy = -sinc_th * ky;
    const float qz = -sinc_th * kz;

    su2_mat U = {
        c_mul(phase, {q0,  qz}),
        c_mul(phase, {qy,  qx}),
        c_mul(phase, {-qy, qx}),
        c_mul(phase, {q0, -qz})
    };

    su2_mat G = su2_mul(su2_dagger(U), U);
    su2_mat M = {
        {0.5f * (3.0f - G.m00.re), -0.5f * G.m00.im},
        {-0.5f * G.m01.re,         -0.5f * G.m01.im},
        {-0.5f * G.m10.re,         -0.5f * G.m10.im},
        {0.5f * (3.0f - G.m11.re), -0.5f * G.m11.im}
    };
    U = su2_mul(U, M);

    su2_mat G_fin = su2_mul(su2_dagger(U), U);
    const float d00 = G_fin.m00.re - 1.0f, d11 = G_fin.m11.re - 1.0f;
    out_defect = sqrtf(__fmaf_rn(d00, d00, __fmaf_rn(d11, d11, 2.0f * __fmaf_rn(G_fin.m01.re, G_fin.m01.re, G_fin.m01.im * G_fin.m01.im))));
    return U;
}

__device__ __forceinline__ float cta_dot_product_256(float val, float* s_warp_red) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(FULL_WARP_MASK, val, offset);
    }
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (lane == 0) s_warp_red[warp] = val;
    __syncthreads();

    float block_sum = (threadIdx.x < 8u) ? s_warp_red[threadIdx.x] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            block_sum += __shfl_xor_sync(FULL_WARP_MASK, block_sum, offset);
        }
        if (lane == 0) s_warp_red[0] = block_sum;
    }
    __syncthreads();
    const float result = s_warp_red[0];
    // Every warp must consume the published result before a following
    // reduction may reuse the shared scratch array.
    __syncthreads();
    return result;
}

#include "nhs_biprism_dec_pcg.cuh"

struct __align__(128) ContactShellSimplicialStateFfi {
    uint32_t global_atom_idx[TILE_TOTAL_VERTICES];
    float4   ref_pos_q[TILE_TOTAL_VERTICES];
    uint8_t  tet_v[TILE_MAX_TETS][4];
    uint16_t tet_edges[TILE_MAX_TETS][6];
    uint16_t tet_opp_tet[TILE_MAX_TETS][4];
    uint8_t  edge_u8[TILE_MAX_EDGES][2];
    su2_mat  u_spatial_edge[TILE_MAX_EDGES];
    su2_mat  v_leaf_vertex[TILE_TOTAL_VERTICES];
    cpx2_f32 psi_vertex[TILE_TOTAL_VERTICES][2];
    float    phi_dec[TILE_TOTAL_VERTICES];
    float    star0_vol[TILE_TOTAL_VERTICES];
    float    star1_wt[TILE_MAX_EDGES];
    float3   grad_star1_u[TILE_MAX_EDGES];
    float    s_ovlp_edges[TILE_MAX_EDGES];
    float3   grad_s_edges[TILE_MAX_EDGES];
    float    h_cov_edges[TILE_MAX_EDGES][4];
    float3   grad_h_diag_u[TILE_MAX_EDGES][2];
    float3   grad_h_cpl_u[TILE_MAX_EDGES];
    uint32_t num_active_tets;
    uint32_t num_active_edges;
    uint32_t pachner_flips_step;
    uint32_t active_pocket_id;
};

__device__ __forceinline__ float exact_laguerre_insphere_4x4(
    float4 a, float4 b, float4 c, float4 d, float4 e
) {
    const float ax = a.x - e.x, ay = a.y - e.y, az = a.z - e.z;
    const float bx = b.x - e.x, by = b.y - e.y, bz = b.z - e.z;
    const float cx = c.x - e.x, cy = c.y - e.y, cz = c.z - e.z;
    const float dx = d.x - e.x, dy = d.y - e.y, dz = d.z - e.z;

    const float aw = __fmaf_rn(ax, ax, __fmaf_rn(ay, ay, az * az)) - (a.w - e.w);
    const float bw = __fmaf_rn(bx, bx, __fmaf_rn(by, by, bz * bz)) - (b.w - e.w);
    const float cw = __fmaf_rn(cx, cx, __fmaf_rn(cy, cy, cz * cz)) - (c.w - e.w);
    const float dw = __fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz)) - (d.w - e.w);

    const float ab_xy = __fmaf_rn(ax, by, -ay * bx);
    const float ab_xz = __fmaf_rn(ax, bz, -az * bx);
    const float ab_xw = __fmaf_rn(ax, bw, -aw * bx);
    const float ab_yz = __fmaf_rn(ay, bz, -az * by);
    const float ab_yw = __fmaf_rn(ay, bw, -aw * by);
    const float ab_zw = __fmaf_rn(az, bw, -aw * bz);

    const float cd_xy = __fmaf_rn(cx, dy, -cy * dx);
    const float cd_xz = __fmaf_rn(cx, dz, -cz * dx);
    const float cd_xw = __fmaf_rn(cx, dw, -cw * dx);
    const float cd_yz = __fmaf_rn(cy, dz, -cz * dy);
    const float cd_yw = __fmaf_rn(cy, dw, -cw * dy);
    const float cd_zw = __fmaf_rn(cz, dw, -cw * dz);

    return __fmaf_rn( ab_xy, cd_zw,
           __fmaf_rn(-ab_xz, cd_yw,
           __fmaf_rn( ab_xw, cd_yz,
           __fmaf_rn( ab_yz, cd_xw,
           __fmaf_rn(-ab_yw, cd_xz, ab_zw * cd_xy)))));
}

// =============================================================================================
// CIRCUMCENTRIC / WELL-CENTERED DIAGONAL DEC HODGE WEIGHT EVALUATOR ON A TETRAHEDRON
// Computes positive dual cell volume contribution \star_0^{(T)} and positive base dual-area-to-length
// weights w_{ab}^{(T)} for the 6 edges of tetrahedron T = (v0, v1, v2, v3).
// =============================================================================================
__device__ __forceinline__ void evaluate_diagonal_dec_tet_weights(
    float4 x0, float4 x1, float4 x2, float4 x3,
    float4 X0, float4 X1, float4 X2, float4 X3,
    float sigma_0,
    float& out_warped_vol,
    float& out_w01, float& out_w02, float& out_w03,
    float& out_w12, float& out_w13, float& out_w23
) {
    const float3 e01 = {x1.x - x0.x, x1.y - x0.y, x1.z - x0.z};
    const float3 e02 = {x2.x - x0.x, x2.y - x0.y, x2.z - x0.z};
    const float3 e03 = {x3.x - x0.x, x3.y - x0.y, x3.z - x0.z};
    const float3 e12 = {x2.x - x1.x, x2.y - x1.y, x2.z - x1.z};
    const float3 e13 = {x3.x - x1.x, x3.y - x1.y, x3.z - x1.z};
    const float3 e23 = {x3.x - x2.x, x3.y - x2.y, x3.z - x2.z};

    const float3 E01 = {X1.x - X0.x, X1.y - X0.y, X1.z - X0.z};
    const float3 E02 = {X2.x - X0.x, X2.y - X0.y, X2.z - X0.z};
    const float3 E03 = {X3.x - X0.x, X3.y - X0.y, X3.z - X0.z};

    const float3 N3 = {e01.y * e02.z - e01.z * e02.y, e01.z * e02.x - e01.x * e02.z, e01.x * e02.y - e01.y * e02.x};
    const float3 N2 = {e03.y * e01.z - e03.z * e01.y, e03.z * e01.x - e03.x * e01.z, e03.x * e01.y - e03.y * e01.x};
    const float3 N1 = {e02.y * e03.z - e02.z * e03.y, e02.z * e03.x - e02.x * e03.z, e02.x * e03.y - e02.y * e03.x};
    const float3 N0 = {e13.y * e12.z - e13.z * e12.y, e13.z * e12.x - e13.x * e12.z, e13.x * e12.y - e13.y * e12.x};

    const float det6V = __fmaf_rn(e01.x, N1.x, __fmaf_rn(e01.y, N1.y, e01.z * N1.z));
    const float det6V_ref = E01.x * (E02.y * E03.z - E02.z * E03.y)
                          - E01.y * (E02.x * E03.z - E02.z * E03.x)
                          + E01.z * (E02.x * E03.y - E02.y * E03.x);

    const float V_T     = fabsf(det6V) * (1.0f / 6.0f);
    const float V_ref   = fmaxf(fabsf(det6V_ref) * (1.0f / 6.0f), 1e-5f);
    const float J_piola = fminf(fmaxf(V_T / V_ref, 0.05f), 20.0f);

    const float l2_sum = (e01.x*e01.x + e01.y*e01.y + e01.z*e01.z) + (e02.x*e02.x + e02.y*e02.y + e02.z*e02.z)
                       + (e03.x*e03.x + e03.y*e03.y + e03.z*e03.z) + (e12.x*e12.x + e12.y*e12.y + e12.z*e12.z)
                       + (e13.x*e13.x + e13.y*e13.y + e13.z*e13.z) + (e23.x*e23.x + e23.y*e23.y + e23.z*e23.z);
    const float l_rms2    = l2_sum * (1.0f / 6.0f);
    const float reg_denom = __fmaf_rn(V_T, V_T, (sigma_0 * sigma_0) * (l_rms2 * l_rms2 * l_rms2));

    const float inv_36V_reg = (V_T / fmaxf(reg_denom, 1e-16f)) * (1.0f / 36.0f) * J_piola;
    out_warped_vol = fmaxf(V_T * J_piola, 1e-5f);

    // Positive weight floor gives a semidefinite graph Laplacian; SPD additionally needs boundary/gauge conditions.
    const float spd_floor = 1e-3f * sqrtf(l_rms2);
    const float dot_N2_N3 = -(N2.x * N3.x + N2.y * N3.y + N2.z * N3.z);
    const float dot_N1_N3 = -(N1.x * N3.x + N1.y * N3.y + N1.z * N3.z);
    const float dot_N1_N2 = -(N1.x * N2.x + N1.y * N2.y + N1.z * N2.z);
    const float dot_N0_N3 = -(N0.x * N3.x + N0.y * N3.y + N0.z * N3.z);
    const float dot_N0_N2 = -(N0.x * N2.x + N0.y * N2.y + N0.z * N2.z);
    const float dot_N0_N1 = -(N0.x * N1.x + N0.y * N1.y + N0.z * N1.z);

    out_w01 = fmaxf(dot_N2_N3 * inv_36V_reg, spd_floor);
    out_w02 = fmaxf(dot_N1_N3 * inv_36V_reg, spd_floor);
    out_w03 = fmaxf(dot_N1_N2 * inv_36V_reg, spd_floor);
    out_w12 = fmaxf(dot_N0_N3 * inv_36V_reg, spd_floor);
    out_w13 = fmaxf(dot_N0_N2 * inv_36V_reg, spd_floor);
    out_w23 = fmaxf(dot_N0_N1 * inv_36V_reg, spd_floor);
}

// Compensated preparation keeps the legacy physical expressions and original
// represented constants, evaluating coordinate differences/radial geometry in
// FP64 and rounding only the declared FP32 coefficient outputs.
__device__ __forceinline__ BiprismCoordinate3F64 bp_p2_coordinate_difference(
    const BiprismCoordinateOwnerFfi& owner,const ContactShellSimplicialStateFfi* tile,
    uint32_t u,uint32_t v,BiprismCoordinatePlane plane
) {
    return bp_coord_difference(bp_coord_load(owner,plane,UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[v])),
        bp_coord_load(owner,plane,UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u])));
}
__device__ __forceinline__ void evaluate_diagonal_dec_tet_weights_compensated(
    const BiprismCoordinateOwnerFfi& owner,const ContactShellSimplicialStateFfi* tile,
    uint32_t v0,uint32_t v1,uint32_t v2,uint32_t v3,float sigma_input,
    float& out_warped_vol,float& out_w01,float& out_w02,float& out_w03,
    float& out_w12,float& out_w13,float& out_w23
) {
    const double sigma_0=double(sigma_input);
    const auto e01=bp_p2_coordinate_difference(owner,tile,v0,v1,BiprismCoordinatePlane::Current);
    const auto e02=bp_p2_coordinate_difference(owner,tile,v0,v2,BiprismCoordinatePlane::Current);
    const auto e03=bp_p2_coordinate_difference(owner,tile,v0,v3,BiprismCoordinatePlane::Current);
    const auto e12=bp_p2_coordinate_difference(owner,tile,v1,v2,BiprismCoordinatePlane::Current);
    const auto e13=bp_p2_coordinate_difference(owner,tile,v1,v3,BiprismCoordinatePlane::Current);
    const auto e23=bp_p2_coordinate_difference(owner,tile,v2,v3,BiprismCoordinatePlane::Current);
    const auto E01=bp_p2_coordinate_difference(owner,tile,v0,v1,BiprismCoordinatePlane::MeshReference);
    const auto E02=bp_p2_coordinate_difference(owner,tile,v0,v2,BiprismCoordinatePlane::MeshReference);
    const auto E03=bp_p2_coordinate_difference(owner,tile,v0,v3,BiprismCoordinatePlane::MeshReference);
    const BiprismCoordinate3F64 N3 = {e01.y * e02.z - e01.z * e02.y, e01.z * e02.x - e01.x * e02.z, e01.x * e02.y - e01.y * e02.x};
    const BiprismCoordinate3F64 N2 = {e03.y * e01.z - e03.z * e01.y, e03.z * e01.x - e03.x * e01.z, e03.x * e01.y - e03.y * e01.x};
    const BiprismCoordinate3F64 N1 = {e02.y * e03.z - e02.z * e03.y, e02.z * e03.x - e02.x * e03.z, e02.x * e03.y - e02.y * e03.x};
    const BiprismCoordinate3F64 N0 = {e13.y * e12.z - e13.z * e12.y, e13.z * e12.x - e13.x * e12.z, e13.x * e12.y - e13.y * e12.x};

    const double det6V = __fma_rn(e01.x, N1.x, __fma_rn(e01.y, N1.y, e01.z * N1.z));
    const double det6V_ref = E01.x * (E02.y * E03.z - E02.z * E03.y)
                          - E01.y * (E02.x * E03.z - E02.z * E03.x)
                          + E01.z * (E02.x * E03.y - E02.y * E03.x);

    const double V_T     = fabs(det6V) * double(1.0f / 6.0f);
    const double V_ref   = fmax(fabs(det6V_ref) * double(1.0f / 6.0f), double(1e-5f));
    const double J_piola = fmin(fmax(V_T / V_ref, double(0.05f)), double(20.0f));

    const double l2_sum = (e01.x*e01.x + e01.y*e01.y + e01.z*e01.z) + (e02.x*e02.x + e02.y*e02.y + e02.z*e02.z)
                       + (e03.x*e03.x + e03.y*e03.y + e03.z*e03.z) + (e12.x*e12.x + e12.y*e12.y + e12.z*e12.z)
                       + (e13.x*e13.x + e13.y*e13.y + e13.z*e13.z) + (e23.x*e23.x + e23.y*e23.y + e23.z*e23.z);
    const double l_rms2    = l2_sum * double(1.0f / 6.0f);
    const double reg_denom = __fma_rn(V_T, V_T, (sigma_0 * sigma_0) * (l_rms2 * l_rms2 * l_rms2));

    const double inv_36V_reg = (V_T / fmax(reg_denom, double(1e-16f))) * double(1.0f / 36.0f) * J_piola;
    out_warped_vol = fmax(V_T * J_piola, double(1e-5f));

    // Positive weight floor gives a semidefinite graph Laplacian; SPD additionally needs boundary/gauge conditions.
    const double spd_floor = double(1e-3f) * sqrt(l_rms2);
    const double dot_N2_N3 = -(N2.x * N3.x + N2.y * N3.y + N2.z * N3.z);
    const double dot_N1_N3 = -(N1.x * N3.x + N1.y * N3.y + N1.z * N3.z);
    const double dot_N1_N2 = -(N1.x * N2.x + N1.y * N2.y + N1.z * N2.z);
    const double dot_N0_N3 = -(N0.x * N3.x + N0.y * N3.y + N0.z * N3.z);
    const double dot_N0_N2 = -(N0.x * N2.x + N0.y * N2.y + N0.z * N2.z);
    const double dot_N0_N1 = -(N0.x * N1.x + N0.y * N1.y + N0.z * N1.z);

    out_w01 = fmax(dot_N2_N3 * inv_36V_reg, spd_floor);
    out_w02 = fmax(dot_N1_N3 * inv_36V_reg, spd_floor);
    out_w03 = fmax(dot_N1_N2 * inv_36V_reg, spd_floor);
    out_w12 = fmax(dot_N0_N3 * inv_36V_reg, spd_floor);
    out_w13 = fmax(dot_N0_N2 * inv_36V_reg, spd_floor);
    out_w23 = fmax(dot_N0_N1 * inv_36V_reg, spd_floor);
}

__device__ __forceinline__ double bp_p2_laguerre_compensated(
    const BiprismCoordinateOwnerFfi& owner,const ContactShellSimplicialStateFfi* tile,
    const float4* q,uint32_t a,uint32_t b,uint32_t c,uint32_t d,uint32_t e
) {
    const auto A=bp_p2_coordinate_difference(owner,tile,e,a,BiprismCoordinatePlane::Current);
    const auto B=bp_p2_coordinate_difference(owner,tile,e,b,BiprismCoordinatePlane::Current);
    const auto C=bp_p2_coordinate_difference(owner,tile,e,c,BiprismCoordinatePlane::Current);
    const auto D=bp_p2_coordinate_difference(owner,tile,e,d,BiprismCoordinatePlane::Current);
    const double ax=A.x,ay=A.y,az=A.z,bx=B.x,by=B.y,bz=B.z;
    const double cx=C.x,cy=C.y,cz=C.z,dx=D.x,dy=D.y,dz=D.z;
    const double aw=__fma_rn(ax,ax,__fma_rn(ay,ay,az*az))-(double(q[a].w)-double(q[e].w));
    const double bw=__fma_rn(bx,bx,__fma_rn(by,by,bz*bz))-(double(q[b].w)-double(q[e].w));
    const double cw=__fma_rn(cx,cx,__fma_rn(cy,cy,cz*cz))-(double(q[c].w)-double(q[e].w));
    const double dw=__fma_rn(dx,dx,__fma_rn(dy,dy,dz*dz))-(double(q[d].w)-double(q[e].w));
    const double ab_xy = __fma_rn(ax, by, -ay * bx);
    const double ab_xz = __fma_rn(ax, bz, -az * bx);
    const double ab_xw = __fma_rn(ax, bw, -aw * bx);
    const double ab_yz = __fma_rn(ay, bz, -az * by);
    const double ab_yw = __fma_rn(ay, bw, -aw * by);
    const double ab_zw = __fma_rn(az, bw, -aw * bz);

    const double cd_xy = __fma_rn(cx, dy, -cy * dx);
    const double cd_xz = __fma_rn(cx, dz, -cz * dx);
    const double cd_xw = __fma_rn(cx, dw, -cw * dx);
    const double cd_yz = __fma_rn(cy, dz, -cz * dy);
    const double cd_yw = __fma_rn(cy, dw, -cw * dy);
    const double cd_zw = __fma_rn(cz, dw, -cw * dz);

    return __fma_rn( ab_xy, cd_zw,
           __fma_rn(-ab_xz, cd_yw,
           __fma_rn( ab_xw, cd_yz,
           __fma_rn( ab_yz, cd_xw,
           __fma_rn(-ab_yw, cd_xz, ab_zw * cd_xy)))));
}


// Shared-Memory Hash Helpers for Incidence-Verified Seeding
// Replace `sort4_u8` through `nhs_seed_simplicial_tile_from_warp_nlist_v1`
// in `roles/engine/crates/prism-gpu/src/kernels/nhs_biprism_simplicial_tile.cuh`:

__device__ __forceinline__ void sort4_u8(uint8_t& a, uint8_t& b, uint8_t& c, uint8_t& d) {
    uint8_t t;
    if (a > b) { t = a; a = b; b = t; }
    if (c > d) { t = c; c = d; d = t; }
    if (a > c) { t = a; a = c; c = t; }
    if (b > d) { t = b; b = d; d = t; }
    if (b > c) { t = b; b = c; c = t; }
}

// Sorts 3 vertices (a, b, c) into ascending order fa < fb < fc and returns permutation parity (0 = even, 1 = odd)
__device__ __forceinline__ uint32_t sort3_u8_with_parity(uint8_t& a, uint8_t& b, uint8_t& c) {
    uint32_t parity = 0u;
    uint8_t t;
    if (a > b) { t = a; a = b; b = t; parity ^= 1u; }
    if (b > c) { t = b; b = c; c = t; parity ^= 1u; }
    if (a > b) { t = a; a = b; b = t; parity ^= 1u; }
    return parity;
}

// Compute canonical 24-bit face key (fa << 16) | (fb << 8) | fc and induced boundary orientation side \in {0, 1}
// for local face `f` (opposite vertex `v_f`) of a positively oriented tetrahedron (v0, v1, v2, v3).
__device__ __forceinline__ void get_oriented_face_key_and_side(
    uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3,
    uint32_t f,
    uint32_t& out_fkey,
    uint32_t& out_side
) {
    // Induced outward boundary triangles of positively oriented (v0, v1, v2, v3):
    //   f=0 (opp v0): +(v1, v2, v3)
    //   f=1 (opp v1): +(v0, v3, v2)
    //   f=2 (opp v2): +(v0, v1, v3)
    //   f=3 (opp v3): +(v0, v2, v1)
    uint8_t a, b, c;
    if (f == 0u)      { a = v1; b = v2; c = v3; }
    else if (f == 1u) { a = v0; b = v3; c = v2; }
    else if (f == 2u) { a = v0; b = v1; c = v3; }
    else              { a = v0; b = v2; c = v1; }

    out_side = sort3_u8_with_parity(a, b, c);
    out_fkey = ((uint32_t)a << 16u) | ((uint32_t)b << 8u) | (uint32_t)c;
}

// A candidate probes all four faces before committing any hash key. Reserve earlier
// prospective empty slots locally so distinct face keys cannot select the same slot.
__device__ __forceinline__ bool face_slot_reserved_for_other_candidate_key(
    uint32_t slot, uint32_t key, uint32_t prior_faces,
    const uint32_t* __restrict__ candidate_slots,
    const uint32_t* __restrict__ candidate_keys
) {
    #pragma unroll
    for (uint32_t prior = 0u; prior < prior_faces; ++prior) {
        if (candidate_slots[prior] == slot && candidate_keys[prior] != key) {
            return true;
        }
    }
    return false;
}

// 30,720-Byte (< 32 KiB) Shared Memory Workspace for Transactional Oriented-Face Seeding
struct __align__(16) SimplicialSeederSmem {
    float4   pos_q[TILE_TOTAL_VERTICES];        // 4,096 B
    union {
        uint32_t unique_atoms[TILE_TOTAL_VERTICES]; // 1,024 B (Step 1)
        uint8_t  knn[TILE_TOTAL_VERTICES][8];       // 2,048 B (Step 2 & 3)
    };
    uint16_t edge_keys[TILE_HASH_SLOTS];        // 4,096 B: (u << 8) | v, 0xFFFFu = empty
    uint16_t edge_vals[TILE_HASH_SLOTS];        // 4,096 B: edge index 0..1535
    uint32_t face_keys[TILE_HASH_SLOTS];        // 8,192 B: (fa << 16) | (fb << 8) | fc, 0xFFFFFFFFu = empty
    uint32_t face_sides[TILE_HASH_SLOTS];       // 8,192 B: low 16b = side0 (t<<2|f), high 16b = side1 (t<<2|f)
};
static_assert(sizeof(SimplicialSeederSmem) == 30720, "Seeder SMEM must be 30,720 bytes (< 32 KiB)");

// Read-only probe of edge hash table: returns true if edge (u_in, v_in) is already registered
__device__ __forceinline__ bool edge_hash_contains(
    uint8_t u_in, uint8_t v_in,
    const uint16_t* __restrict__ s_edge_keys
) {
    const uint8_t u = (u_in < v_in) ? u_in : v_in;
    const uint8_t v = (u_in < v_in) ? v_in : u_in;
    const uint16_t key = ((uint16_t)u << 8u) | (uint16_t)v;
    uint32_t slot = ((uint32_t)key * 2654435761u) & (TILE_HASH_SLOTS - 1u);
    for (uint32_t probe = 0u; probe < TILE_HASH_SLOTS; ++probe) {
        const uint16_t k = s_edge_keys[slot];
        if (k == key) return true;
        if (k == 0xFFFFu) return false;
        slot = (slot + 1u) & (TILE_HASH_SLOTS - 1u);
    }
    return false;
}

// Insert or lookup edge (u_in, v_in) in edge hash table
__device__ __forceinline__ uint16_t edge_hash_get_or_insert(
    uint8_t u_in, uint8_t v_in,
    uint16_t* __restrict__ s_edge_keys,
    uint16_t* __restrict__ s_edge_vals,
    uint32_t& io_num_edges,
    ContactShellSimplicialStateFfi* __restrict__ tile
) {
    const uint8_t u = (u_in < v_in) ? u_in : v_in;
    const uint8_t v = (u_in < v_in) ? v_in : u_in;
    const uint16_t key = ((uint16_t)u << 8u) | (uint16_t)v;
    uint32_t slot = ((uint32_t)key * 2654435761u) & (TILE_HASH_SLOTS - 1u);
    for (uint32_t probe = 0u; probe < TILE_HASH_SLOTS; ++probe) {
        const uint16_t k = s_edge_keys[slot];
        if (k == key) {
            return s_edge_vals[slot];
        }
        if (k == 0xFFFFu) {
            const uint16_t eid = (uint16_t)(io_num_edges++);
            s_edge_keys[slot] = key;
            s_edge_vals[slot] = eid;
            tile->edge_u8[eid][0] = u;
            tile->edge_u8[eid][1] = v;
            const su2_mat I_su2 = {{1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
            tile->u_spatial_edge[eid] = I_su2;
            tile->star1_wt[eid]       = 0.1f;
            tile->grad_star1_u[eid]   = make_float3(0.0f, 0.0f, 0.0f);
            tile->s_ovlp_edges[eid]   = 0.0f;
            tile->grad_s_edges[eid]   = make_float3(0.0f, 0.0f, 0.0f);
            return eid;
        }
        slot = (slot + 1u) & (TILE_HASH_SLOTS - 1u);
    }
    return 0xFFFFu;
}

// =============================================================================================
// TRANSACTIONAL ORIENTED-FACE SIMPLICIAL SEEDER (`nhs_seed_simplicial_tile_from_warp_nlist_v1`)
// Checks the following incidence predicates; edge/vertex link and coverage predicates remain separate:
//   1. 256 strictly distinct global atom indices.
//   2. Strictly positive volume (\det > 1e-3) and positive orientation.
//   3. Zero duplicate tetrahedra and zero non-manifold faces: every triangle face F has at most
//      one tet on side 0 (+1) and at most one tet on side 1 (-1) (\deg(F) \in {1, 2}).
//   4. 100% exact 6-edge incidence in `tet_edges[t][0..5]`.
//   5. 100% symmetric 1-to-1 face adjacency in `tet_opp_tet[t][0..3]`.
// =============================================================================================
extern "C" __global__ __launch_bounds__(256, 2) void nhs_seed_simplicial_tile_from_warp_nlist_v1(
    const float* __restrict__ d_amber_coords_xyz,
    const float* __restrict__ d_amber_charges,
    const uint8_t* __restrict__ d_atom_elem_classes,
    const int32_t* __restrict__ d_neighbor_list,
    const int32_t* __restrict__ d_neighbor_counts,
    ContactShellSimplicialStateFfi* __restrict__ d_tile_state,
    const uint32_t center_pocket_atom_id,
    const uint32_t stream_leaf_id
) {
    __shared__ SimplicialSeederSmem smem;

    const uint32_t tid  = threadIdx.x;
    const uint32_t leaf = (gridDim.x >= 8u) ? (blockIdx.x & 7u) : stream_leaf_id;
    ContactShellSimplicialStateFfi* tile = &d_tile_state[leaf];

    // Step 1: Select 256 strictly distinct global atom IDs
    if (tid == 0) {
        smem.unique_atoms[0] = center_pocket_atom_id;
        uint32_t count = 1u;
        const int32_t n_neigh = (d_neighbor_counts != nullptr) ? d_neighbor_counts[center_pocket_atom_id] : 0;
        for (int32_t k = 0; k < n_neigh && count < TILE_TOTAL_VERTICES; ++k) {
            const int32_t cand_i = (d_neighbor_list != nullptr)
                ? d_neighbor_list[center_pocket_atom_id * 1024u + (uint32_t)k]
                : -1;
            if (cand_i >= 0) {
                const uint32_t cand = (uint32_t)cand_i;
                bool dup = false;
                for (uint32_t j = 0u; j < count; ++j) {
                    if (smem.unique_atoms[j] == cand) { dup = true; break; }
                }
                if (!dup) smem.unique_atoms[count++] = cand;
            }
        }
        uint32_t fallback_id = (center_pocket_atom_id > 128u) ? (center_pocket_atom_id - 128u) : 0u;
        while (count < TILE_TOTAL_VERTICES) {
            bool dup = false;
            for (uint32_t j = 0u; j < count; ++j) {
                if (smem.unique_atoms[j] == fallback_id) { dup = true; break; }
            }
            if (!dup) smem.unique_atoms[count++] = fallback_id;
            ++fallback_id;
        }
    }
    __syncthreads();

    const uint32_t g_id = smem.unique_atoms[tid];
    const uint32_t elem_cls = (d_atom_elem_classes != nullptr) ? (uint32_t)(d_atom_elem_classes[g_id] & 7u) : (tid & 3u);
    const float3 p = *reinterpret_cast<const float3*>(&d_amber_coords_xyz[g_id * 3u]);
    const float  q = d_amber_charges[g_id];
    smem.pos_q[tid] = make_float4(p.x, p.y, p.z, q);

    tile->global_atom_idx[tid] = PACK_ATOM_ELEM(g_id, elem_cls);
    tile->ref_pos_q[tid]       = smem.pos_q[tid];
    tile->phi_dec[tid]         = q * 0.07957747f;
    tile->star0_vol[tid]       = 1.0f;
    tile->psi_vertex[tid][0]   = {1.0f, 0.0f};
    tile->psi_vertex[tid][1]   = {0.0f, 0.0f};
    const su2_mat I_su2 = {{1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
    tile->v_leaf_vertex[tid]   = I_su2;

    for (uint32_t h = tid; h < TILE_HASH_SLOTS; h += 256) {
        smem.edge_keys[h]  = 0xFFFFu;
        smem.edge_vals[h]  = 0xFFFFu;
        smem.face_keys[h]  = 0xFFFFFFFFu;
        smem.face_sides[h] = 0xFFFFFFFFu;
    }
    __syncthreads();

    // Step 2: Parallel 6-KNN computation across all 256 threads
    uint8_t knn0 = 0u, knn1 = 1u, knn2 = 2u, knn3 = 3u, knn4 = 4u, knn5 = 5u;
    float d0 = 1e30f, d1 = 1e30f, d2 = 1e30f, d3 = 1e30f, d4 = 1e30f, d5 = 1e30f;
    const float4 my_p = smem.pos_q[tid];

    for (uint32_t j = 0; j < TILE_TOTAL_VERTICES; ++j) {
        if (j == tid) continue;
        const float dx = smem.pos_q[j].x - my_p.x;
        const float dy = smem.pos_q[j].y - my_p.y;
        const float dz = smem.pos_q[j].z - my_p.z;
        const float dist2 = __fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz)) + (float)j * 1e-7f;
        const uint8_t uj = (uint8_t)j;
        if (dist2 < d0)      { d5=d4; knn5=knn4; d4=d3; knn4=knn3; d3=d2; knn3=knn2; d2=d1; knn2=knn1; d1=d0; knn1=knn0; d0=dist2; knn0=uj; }
        else if (dist2 < d1) { d5=d4; knn5=knn4; d4=d3; knn4=knn3; d3=d2; knn3=knn2; d2=d1; knn2=knn1; d1=dist2; knn1=uj; }
        else if (dist2 < d2) { d5=d4; knn5=knn4; d4=d3; knn4=knn3; d3=d2; knn3=knn2; d2=dist2; knn2=uj; }
        else if (dist2 < d3) { d5=d4; knn5=knn4; d4=d3; knn4=knn3; d3=dist2; knn3=uj; }
        else if (dist2 < d4) { d5=d4; knn5=knn4; d4=dist2; knn4=uj; }
        else if (dist2 < d5) { d5=dist2; knn5=uj; }
    }
    smem.knn[tid][0] = knn0; smem.knn[tid][1] = knn1; smem.knn[tid][2] = knn2;
    smem.knn[tid][3] = knn3; smem.knn[tid][4] = knn4; smem.knn[tid][5] = knn5;
    __syncthreads();

    // Step 3: Transactional Oriented-Face Admission (Checks all 4 oriented half-faces & 6 edges before commit)
    if (tid == 0) {
        uint32_t num_tets  = 0u;
        uint32_t num_edges = 0u;
        uint32_t num_faces = 0u;

        // 5 local KNN combinations per vertex: 4 faces of the 5-clique {i, n0, n1, n2, n3} + 2 sliding
        // The same five KNN triples are selected without addressable local arrays.
        for (uint32_t round = 0u; round < 5u; ++round) {
            const uint8_t ca = (round < 3u) ? 0u : ((round == 3u) ? 1u : 2u);
            const uint8_t cb = (round < 2u) ? 1u : ((round < 4u) ? 2u : 3u);
            const uint8_t cc = (round == 0u) ? 2u : ((round == 4u) ? 4u : 3u);
            for (uint32_t i = 0u; i < TILE_TOTAL_VERTICES; ++i) {
                if (num_tets >= TILE_MAX_TETS || num_edges > (TILE_MAX_EDGES - 6u) || num_faces > (TILE_HASH_SLOTS - 16u)) {
                    break;
                }
                uint8_t v0 = (uint8_t)i;
                uint8_t v1 = smem.knn[i][ca];
                uint8_t v2 = smem.knn[i][cb];
                uint8_t v3 = smem.knn[i][cc];

                sort4_u8(v0, v1, v2, v3);
                if (v0 == v1 || v1 == v2 || v2 == v3) continue;

                const float4 p0 = smem.pos_q[v0], p1 = smem.pos_q[v1], p2 = smem.pos_q[v2], p3 = smem.pos_q[v3];
                const float3 e01 = {p1.x - p0.x, p1.y - p0.y, p1.z - p0.z};
                const float3 e02 = {p2.x - p0.x, p2.y - p0.y, p2.z - p0.z};
                const float3 e03 = {p3.x - p0.x, p3.y - p0.y, p3.z - p0.z};
                const float det6V = e01.x * (e02.y * e03.z - e02.z * e03.y)
                                  - e01.y * (e02.x * e03.z - e02.z * e03.x)
                                  + e01.z * (e02.x * e03.y - e02.y * e03.x);
                if (fabsf(det6V) <= 1e-3f) continue;
                if (det6V < 0.0f) {
                    const uint8_t tmp = v2; v2 = v3; v3 = tmp;
                }

                // 3a. Verify all 4 oriented half-faces (fkey_f, side_f) are unoccupied (rejects duplicates & non-manifold faces!)
                uint32_t fkeys[4], fsides[4], fslots[4];
                uint32_t new_faces = 0u;
                bool manifold_ok = true;

                #pragma unroll
                for (uint32_t f = 0u; f < 4u; ++f) {
                    get_oriented_face_key_and_side(v0, v1, v2, v3, f, fkeys[f], fsides[f]);
                    uint32_t slot = (fkeys[f] * 2654435761u) & (TILE_HASH_SLOTS - 1u);
                    bool found_slot = false;
                    for (uint32_t probe = 0u; probe < TILE_HASH_SLOTS; ++probe) {
                        const uint32_t k = smem.face_keys[slot];
                        if (k == fkeys[f]) {
                            const uint32_t packed_sides = smem.face_sides[slot];
                            const uint16_t my_side_val = (fsides[f] == 0u)
                                ? (uint16_t)(packed_sides & 0xFFFFu)
                                : (uint16_t)(packed_sides >> 16u);
                            if (my_side_val != 0xFFFFu) {
                                // Half-face already occupied -> duplicate tet or non-manifold 3rd tet!
                                manifold_ok = false;
                            }
                            fslots[f] = slot;
                            found_slot = true;
                            break;
                        } else if (k == 0xFFFFFFFFu) {
                            if (face_slot_reserved_for_other_candidate_key(
                                    slot, fkeys[f], f, fslots, fkeys)) {
                                slot = (slot + 1u) & (TILE_HASH_SLOTS - 1u);
                                continue;
                            }
                            fslots[f] = slot;
                            ++new_faces;
                            found_slot = true;
                            break;
                        }
                        slot = (slot + 1u) & (TILE_HASH_SLOTS - 1u);
                    }
                    if (!found_slot || !manifold_ok) { manifold_ok = false; break; }
                }
                if (!manifold_ok || (num_faces + new_faces) > (TILE_HASH_SLOTS - 16u)) continue;

                // 3b. Verify all 6 canonical edges fit within TILE_MAX_EDGES (1536)
                const uint8_t eu[6] = {v0, v0, v0, v1, v1, v2};
                const uint8_t ev[6] = {v1, v2, v3, v2, v3, v3};
                uint32_t new_edges = 0u;
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    if (!edge_hash_contains(eu[k], ev[k], smem.edge_keys)) {
                        ++new_edges;
                    }
                }
                if ((num_edges + new_edges) > TILE_MAX_EDGES) continue;

                // 3c. Commit Tetrahedron `t_idx` & Symmetrically Link `tet_opp_tet`
                const uint32_t t_idx = num_tets++;
                tile->tet_v[t_idx][0] = v0;
                tile->tet_v[t_idx][1] = v1;
                tile->tet_v[t_idx][2] = v2;
                tile->tet_v[t_idx][3] = v3;

                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    tile->tet_edges[t_idx][k] = edge_hash_get_or_insert(
                        eu[k], ev[k], smem.edge_keys, smem.edge_vals, num_edges, tile
                    );
                }

                #pragma unroll
                for (uint32_t f = 0u; f < 4u; ++f) {
                    tile->tet_opp_tet[t_idx][f] = 0xFFFFu;
                    const uint32_t slot = fslots[f];
                    if (smem.face_keys[slot] == 0xFFFFFFFFu) {
                        smem.face_keys[slot] = fkeys[f];
                        ++num_faces;
                    }
                    uint32_t packed = smem.face_sides[slot];
                    const uint16_t my_tf = (uint16_t)((t_idx << 2u) | f);
                    const uint16_t opp_tf = (fsides[f] == 0u)
                        ? (uint16_t)(packed >> 16u)
                        : (uint16_t)(packed & 0xFFFFu);

                    if (fsides[f] == 0u) {
                        packed = (packed & 0xFFFF0000u) | (uint32_t)my_tf;
                    } else {
                        packed = (packed & 0x0000FFFFu) | ((uint32_t)my_tf << 16u);
                    }
                    smem.face_sides[slot] = packed;

                    if (opp_tf != 0xFFFFu) {
                        const uint16_t t_other = opp_tf >> 2u;
                        const uint16_t f_other = opp_tf & 3u;
                        tile->tet_opp_tet[t_idx][f]         = t_other;
                        tile->tet_opp_tet[t_other][f_other] = (uint16_t)t_idx;
                    }
                }
            }
        }

        tile->num_active_tets    = num_tets;
        tile->num_active_edges   = num_edges;
        tile->pachner_flips_step = 0u;
        tile->active_pocket_id   = center_pocket_atom_id;
    }
}

struct __align__(16) SimplicialTilePhase2Smem {
    float4   cur_pos_q[TILE_TOTAL_VERTICES];       // 4,096 B
    float    star0[TILE_TOTAL_VERTICES];           // 1,024 B
    float    star1[TILE_MAX_EDGES];                // 6,144 B
    union {
        struct {
            float4 ref_pos_w[TILE_TOTAL_VERTICES];
            float phi[TILE_TOTAL_VERTICES],cg_r[TILE_TOTAL_VERTICES];
            float cg_z[TILE_TOTAL_VERTICES],cg_p[TILE_TOTAL_VERTICES];
            float cg_Ap[TILE_TOTAL_VERTICES],diag_inv[TILE_TOTAL_VERTICES];
        };
        // Compensated preparation reloads reference pairs from their owner.
        // Its PCG arrays are uninitialized until all tetrahedron batches have
        // been reduced. Barriers bracket this disjoint temporary lifetime.
        float tet_terms[7][256];
    };
    uint8_t  edge_u[TILE_MAX_EDGES];               // 1,536 B
    uint8_t  edge_v[TILE_MAX_EDGES];               // 1,536 B
    float    warp_red[8];                          //    32 B
    uint32_t delaunay_violations;                  //     4 B
};
static_assert(sizeof(SimplicialTilePhase2Smem) <= 32768, "Phase 2 SMEM must fit in 32 KiB");

// =============================================================================================
// PHASE 2 KERNEL (`nhs_biprism_simplicial_tile_v1`)
// Preserves 100% incidence invariants while assembling circumcentric Diagonal DEC weights
// (\star_0, \star_1), exact edge force derivatives `grad_star1_u[e]` for all active edges,
// and solving the DEC Poisson equation (d_0^T \star_1 d_0) \phi = q_{live}.
// =============================================================================================
extern "C" __global__ __launch_bounds__(256, 2) void nhs_biprism_simplicial_tile_v1(
    const float* __restrict__ d_amber_coords_xyz,
    const float* __restrict__ d_amber_charges,
    ContactShellSimplicialStateFfi* __restrict__ d_tile_state,
    const uint32_t stream_leaf_id
) {
    __shared__ SimplicialTilePhase2Smem smem;
    const uint32_t tid  = threadIdx.x;
    const uint32_t leaf = (gridDim.x >= 8u) ? (blockIdx.x & 7u) : stream_leaf_id;
    ContactShellSimplicialStateFfi* tile = &d_tile_state[leaf];

    const uint32_t packed_id = tile->global_atom_idx[tid];
    const uint32_t g_atom    = UNPACK_GLOBAL_ATOM_ID(packed_id);
    const float3 pos = *reinterpret_cast<const float3*>(&d_amber_coords_xyz[g_atom * 3u]);
    const float  q_amber = d_amber_charges[g_atom];
    const bool   is_core = (tid < TILE_CORE_VERTICES);

    const cpx2_f32 p1_init = tile->psi_vertex[tid][1];
    const float pop1 = is_core ? __fmaf_rn(p1_init.re, p1_init.re, p1_init.im * p1_init.im) : 0.0f;
    const float q_live = q_amber + 0.15f * pop1;

    smem.cur_pos_q[tid] = make_float4(pos.x, pos.y, pos.z, q_live);
    smem.ref_pos_w[tid] = tile->ref_pos_q[tid];
    smem.star0[tid]     = 1e-5f;
    smem.diag_inv[tid]  = 1e-4f;
    if (tid == 0) smem.delaunay_violations = 0u;

    const uint32_t num_edges = min(tile->num_active_edges, (uint32_t)TILE_MAX_EDGES);
    for (uint32_t e = tid; e < num_edges; e += 256) {
        smem.star1[e]  = 0.0f;
        smem.edge_u[e] = tile->edge_u8[e][0];
        smem.edge_v[e] = tile->edge_u8[e][1];
    }
    __syncthreads();

    // 1. Evaluate Diagonal DEC Dual Volumes \star_0 and Base Dual-Area Edge Weights \star_1^{base}
    //    without mutating `tet_v` in-place (preserves 100% `tet_edges` and `tet_opp_tet` validity!)
    const uint32_t num_tets = min(tile->num_active_tets, (uint32_t)TILE_MAX_TETS);
    for (uint32_t t = tid; t < num_tets; t += 256) {
        const uint8_t v0 = tile->tet_v[t][0], v1 = tile->tet_v[t][1];
        const uint8_t v2 = tile->tet_v[t][2], v3 = tile->tet_v[t][3];

        // Check true face-adjacent neighbor `opp_t = tile->tet_opp_tet[t][0]` for Laguerre insphere telemetry
        const uint16_t opp_t = tile->tet_opp_tet[t][0];
        if (opp_t < num_tets && t < opp_t) {
            // Find the vertex in `opp_t` opposite the shared face {v1, v2, v3}
            uint8_t v_opp = tile->tet_v[opp_t][0];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t cand = tile->tet_v[opp_t][k];
                if (cand != v1 && cand != v2 && cand != v3) { v_opp = cand; break; }
            }
            const float insphere = exact_laguerre_insphere_4x4(
                smem.cur_pos_q[v0], smem.cur_pos_q[v1],
                smem.cur_pos_q[v2], smem.cur_pos_q[v3],
                smem.cur_pos_q[v_opp]
            );
            if (insphere > 1e-4f) {
                atomicAdd(&smem.delaunay_violations, 1u);
            }
        }

        float warped_vol = 0.0f;
        float w01, w02, w03, w12, w13, w23;
        evaluate_diagonal_dec_tet_weights(
            smem.cur_pos_q[v0], smem.cur_pos_q[v1], smem.cur_pos_q[v2], smem.cur_pos_q[v3],
            smem.ref_pos_w[v0], smem.ref_pos_w[v1], smem.ref_pos_w[v2], smem.ref_pos_w[v3],
            1e-4f, warped_vol, w01, w02, w03, w12, w13, w23
        );

        const float q_vol = 0.25f * warped_vol;
        atomicAdd(&smem.star0[v0], q_vol);
        atomicAdd(&smem.star0[v1], q_vol);
        atomicAdd(&smem.star0[v2], q_vol);
        atomicAdd(&smem.star0[v3], q_vol);

        const uint16_t id0 = tile->tet_edges[t][0], id1 = tile->tet_edges[t][1], id2 = tile->tet_edges[t][2];
        const uint16_t id3 = tile->tet_edges[t][3], id4 = tile->tet_edges[t][4], id5 = tile->tet_edges[t][5];
        if (id0 < num_edges) atomicAdd(&smem.star1[id0], w01);
        if (id1 < num_edges) atomicAdd(&smem.star1[id1], w02);
        if (id2 < num_edges) atomicAdd(&smem.star1[id2], w03);
        if (id3 < num_edges) atomicAdd(&smem.star1[id3], w12);
        if (id4 < num_edges) atomicAdd(&smem.star1[id4], w13);
        if (id5 < num_edges) atomicAdd(&smem.star1[id5], w23);
    }
    __syncthreads();

    // 2. Finalize Live Diagonal DEC Edge Weights (\star_1)_e(x) and Exact Analytical Edge Derivatives
    for (uint32_t e = tid; e < num_edges; e += 256) {
        const uint8_t u = smem.edge_u[e], v = smem.edge_v[e];
        const float4 xu = smem.cur_pos_q[u], xv = smem.cur_pos_q[v];
        const float4 Xu = smem.ref_pos_w[u], Xv = smem.ref_pos_w[v];

        const float dx = xv.x - xu.x, dy = xv.y - xu.y, dz = xv.z - xu.z;
        const float dX = Xv.x - Xu.x, dY = Xv.y - Xu.y, dZ = Xv.z - Xu.z;
        const float l2_cur = __fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz)) + 1e-12f;
        const float l2_ref = __fmaf_rn(dX, dX, __fmaf_rn(dY, dY, dZ * dZ)) + 1e-12f;
        const float inv_r  = rsqrtf(l2_cur);
        const float r_cur  = l2_cur * inv_r;
        const float r_ref  = sqrtf(l2_ref);
        const float3 r_hat = {dx * inv_r, dy * inv_r, dz * inv_r};

        const float star1_base = fmaxf(smem.star1[e], 1e-4f);
        const float star1_live = star1_base * (r_ref / r_cur);
        smem.star1[e]     = star1_live;
        tile->star1_wt[e] = star1_base; // Persist base weight so Phase 3 `star1_wt[e] * (r_ref / r_cur)` matches!

        // Exact derivative \partial (\star_1)_e(x) / \partial x_u = +(\star_1)_e(x) / r_e(x) * \hat{e}_{uv}
        const float dstar1_dr = star1_live * inv_r;
        tile->grad_star1_u[e] = {dstar1_dr * r_hat.x, dstar1_dr * r_hat.y, dstar1_dr * r_hat.z};

        atomicAdd(&smem.diag_inv[u], star1_live);
        atomicAdd(&smem.diag_inv[v], star1_live);
    }
    __syncthreads();

    // 3. Bounded DEC initialization. Phase3 independently verifies the true
    // residual before using this provisional field in propagation.
    smem.phi[tid] = is_core ? tile->phi_dec[tid] : (q_live * 0.07957747f);
    __syncthreads();
    float initial_dec_residual = CUDART_INF_F;
    uint32_t initial_dec_iters = 0u;
    biprism_dec_pcg_256(
        smem.edge_u, smem.edge_v, 1u, smem.star1, num_edges, q_live,
        smem.phi, smem.cg_r, smem.cg_z, smem.cg_p, smem.cg_Ap,
        smem.diag_inv, smem.warp_red, initial_dec_residual, initial_dec_iters
    );

    tile->phi_dec[tid]   = smem.phi[tid];
    tile->star0_vol[tid] = smem.star0[tid];
    if (tid == 0) tile->pachner_flips_step = smem.delaunay_violations;

    for (uint32_t e = tid; e < num_edges; e += 256) {
        const uint8_t u = smem.edge_u[e], v = smem.edge_v[e];
        const uint32_t elem_u = UNPACK_ELEM_CLASS(tile->global_atom_idx[u]);
        const uint32_t elem_v = UNPACK_ELEM_CLASS(tile->global_atom_idx[v]);
        float S_uv, E0, V01, E1;
        float3 grad_S, grad_E0, grad_E1, grad_V01;
        evaluate_heteronuclear_gfn_xtb_edge_scalar(
            elem_u, elem_v, smem.cur_pos_q[u], smem.cur_pos_q[v],
            smem.phi[u], smem.phi[v],
            S_uv, grad_S, E0, V01, E1, grad_E0, grad_E1, grad_V01
        );
        tile->s_ovlp_edges[e]     = S_uv;
        tile->grad_s_edges[e]     = grad_S;
        tile->h_cov_edges[e][0]   = E0;
        tile->h_cov_edges[e][1]   = V01;
        tile->h_cov_edges[e][2]   = V01;
        tile->h_cov_edges[e][3]   = E1;
        tile->grad_h_diag_u[e][0] = grad_E0;
        tile->grad_h_diag_u[e][1] = grad_E1;
        tile->grad_h_cpl_u[e]     = grad_V01;
    }
}

// Explicit compensated preparation with a real convergence fault channel.
// Its owner must refuse the observation/propagation window after any fault.
extern "C" __global__ __launch_bounds__(256, 2) void nhs_biprism_simplicial_tile_compensated_v1(
    const float* __restrict__ d_amber_coords_xyz,
    const float* __restrict__ d_amber_charges,
    ContactShellSimplicialStateFfi* __restrict__ d_tile_state,
    const uint32_t stream_leaf_id,
    const BiprismCoordinateOwnerFfi* __restrict__ d_coordinate_owners,
    BiprismConvergenceRecordFfi* __restrict__ d_convergence
) {
    __shared__ SimplicialTilePhase2Smem smem;
    const uint32_t tid  = threadIdx.x;
    const uint32_t leaf = (gridDim.x >= 8u) ? (blockIdx.x & 7u) : stream_leaf_id;
    if(leaf>=8u)return;
    bool valid=blockDim.x==256u&&d_amber_coords_xyz&&d_amber_charges&&d_tile_state&&d_coordinate_owners&&d_convergence
        &&(reinterpret_cast<uintptr_t>(d_coordinate_owners)&15u)==0u;
    if(valid){const auto* o=&d_coordinate_owners[leaf];
        valid=bp_coord_owner_valid(o,leaf,o->atom_count,o->generation,BP_COORD_ATOMIC_BOHR,
            BP_COORD_ALL_INITIALIZED,d_amber_coords_xyz)
            &&d_convergence[leaf].abi_magic==0x42505631u&&d_convergence[leaf].leaf_id==leaf
            &&d_convergence[leaf].terminal_fault_latch==0u;}
    if(!valid){if(tid==0u&&d_convergence)atomicOr(&d_convergence[leaf].terminal_fault_latch,2u);return;}
    const auto& owner=d_coordinate_owners[leaf];
    ContactShellSimplicialStateFfi* tile = &d_tile_state[leaf];
    const uint32_t num_edges=tile->num_active_edges,num_tets=tile->num_active_tets;
    const uint32_t packed_id=tile->global_atom_idx[tid],g_atom=UNPACK_GLOBAL_ATOM_ID(packed_id);
    valid=g_atom<owner.atom_count&&num_edges>0u&&num_edges<=TILE_MAX_EDGES&&num_tets>0u&&num_tets<=TILE_MAX_TETS;
    if(!__syncthreads_and(valid)){if(tid==0u)atomicOr(&d_convergence[leaf].terminal_fault_latch,2u);return;}
    const auto current=bp_coord_load(owner,BiprismCoordinatePlane::Current,g_atom);
    const auto reference=bp_coord_load(owner,BiprismCoordinatePlane::MeshReference,g_atom);
    const float3 pos={current.x.hi,current.y.hi,current.z.hi}; // projection only; geometry reloads both components
    const float q_amber=d_amber_charges[g_atom];
    valid=bp_coord_finite(current)&&bp_coord_finite(reference)&&isfinite(q_amber)
        &&reference.x.hi==tile->ref_pos_q[tid].x&&reference.y.hi==tile->ref_pos_q[tid].y&&reference.z.hi==tile->ref_pos_q[tid].z;
    for(uint32_t t=tid;t<num_tets;t+=256u){
        for(uint32_t j=0u;j<6u;++j)valid=valid&&tile->tet_edges[t][j]<num_edges;
        for(uint32_t j=0u;j<4u;++j)valid=valid&&(tile->tet_opp_tet[t][j]==0xffffu||tile->tet_opp_tet[t][j]<num_tets);
    }
    if(!__syncthreads_and(valid)){if(tid==0u)atomicOr(&d_convergence[leaf].terminal_fault_latch,2u);return;}
    const bool   is_core = (tid < TILE_CORE_VERTICES);

    const cpx2_f32 p1_init = tile->psi_vertex[tid][1];
    const float pop1 = is_core ? __fmaf_rn(p1_init.re, p1_init.re, p1_init.im * p1_init.im) : 0.0f;
    const float q_live = q_amber + 0.15f * pop1;
    if(!__syncthreads_and(isfinite(pop1)&&isfinite(q_live))){if(tid==0u)atomicOr(&d_convergence[leaf].terminal_fault_latch,2u);return;}

    smem.cur_pos_q[tid] = make_float4(pos.x, pos.y, pos.z, q_live);
    smem.star0[tid]     = 1e-5f;
    if (tid == 0) smem.delaunay_violations = 0u;

    for (uint32_t e = tid; e < num_edges; e += 256) {
        smem.star1[e]  = 0.0f;
        smem.edge_u[e] = tile->edge_u8[e][0];
        smem.edge_v[e] = tile->edge_u8[e][1];
    }
    __syncthreads();

    // 1. Evaluate Diagonal DEC Dual Volumes \star_0 and Base Dual-Area Edge Weights \star_1^{base}
    //    without mutating `tet_v` in-place (preserves 100% `tet_edges` and `tet_opp_tet` validity!)
    for (uint32_t base = 0; base < num_tets; base += 256u) {
        const uint32_t t=base+tid;
        if(t<num_tets){
        const uint8_t v0 = tile->tet_v[t][0], v1 = tile->tet_v[t][1];
        const uint8_t v2 = tile->tet_v[t][2], v3 = tile->tet_v[t][3];

        // Check true face-adjacent neighbor `opp_t = tile->tet_opp_tet[t][0]` for Laguerre insphere telemetry
        const uint16_t opp_t = tile->tet_opp_tet[t][0];
        if (opp_t < num_tets && t < opp_t) {
            // Find the vertex in `opp_t` opposite the shared face {v1, v2, v3}
            uint8_t v_opp = tile->tet_v[opp_t][0];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint8_t cand = tile->tet_v[opp_t][k];
                if (cand != v1 && cand != v2 && cand != v3) { v_opp = cand; break; }
            }
            const double insphere=bp_p2_laguerre_compensated(owner,tile,smem.cur_pos_q,v0,v1,v2,v3,v_opp);
            if (insphere > 1e-4f) {
                atomicAdd(&smem.delaunay_violations, 1u);
            }
        }

        float warped_vol = 0.0f;
        float w01, w02, w03, w12, w13, w23;
        evaluate_diagonal_dec_tet_weights_compensated(owner,tile,v0,v1,v2,v3,
            1e-4f,warped_vol,w01,w02,w03,w12,w13,w23);
        valid=valid&&isfinite(warped_vol)&&isfinite(w01)&&isfinite(w02)&&isfinite(w03)&&isfinite(w12)&&isfinite(w13)&&isfinite(w23);

        smem.tet_terms[0][tid]=0.25f*warped_vol;
        smem.tet_terms[1][tid]=w01;smem.tet_terms[2][tid]=w02;smem.tet_terms[3][tid]=w03;
        smem.tet_terms[4][tid]=w12;smem.tet_terms[5][tid]=w13;smem.tet_terms[6][tid]=w23;
        }
        __syncthreads();
        const uint32_t count=min(256u,num_tets-base);
        // Each destination has one writer and sums the represented provider
        // terms in increasing tetrahedron order. Scheduling cannot alter O.
        float volume=smem.star0[tid];
        #pragma unroll 1
        for(uint32_t j=0;j<count;++j){
            const auto* v=tile->tet_v[base+j];
            if(v[0]==tid||v[1]==tid||v[2]==tid||v[3]==tid)volume+=smem.tet_terms[0][j];
        }
        smem.star0[tid]=volume;
        #pragma unroll 1
        for(uint32_t e=tid;e<num_edges;e+=256u){
            float weight=smem.star1[e];
            #pragma unroll 1
            for(uint32_t j=0;j<count;++j){
                #pragma unroll
                for(uint32_t k=0;k<6u;++k)
                    if(tile->tet_edges[base+j][k]==e)weight+=smem.tet_terms[k+1u][j];
            }
            smem.star1[e]=weight;
        }
        __syncthreads(); // All term readers finish before the next batch/PCG.
    }

    valid=valid&&isfinite(smem.star0[tid]);
    for(uint32_t e=tid;e<num_edges;e+=256u)valid=valid&&isfinite(smem.star1[e]);
    if(!__syncthreads_and(valid)){if(tid==0u)atomicOr(&d_convergence[leaf].terminal_fault_latch,2u);return;}

    // 2. Finalize Live Diagonal DEC Edge Weights (\star_1)_e(x) and Exact Analytical Edge Derivatives
    for (uint32_t e = tid; e < num_edges; e += 256) {
        const uint8_t u = smem.edge_u[e], v = smem.edge_v[e];
        const auto current_edge=bp_p2_coordinate_difference(owner,tile,u,v,BiprismCoordinatePlane::Current);
        const auto reference_edge=bp_p2_coordinate_difference(owner,tile,u,v,BiprismCoordinatePlane::MeshReference);
        const float star1_base=fmaxf(smem.star1[e],1e-4f);
        const float star1_live=bp_geometry_live_weight(star1_base,current_edge,reference_edge);
        smem.star1[e]=star1_live;
        tile->star1_wt[e]=star1_base;
        tile->grad_star1_u[e]=bp_geometry_live_weight_gradient(star1_live,current_edge);
        valid=valid&&isfinite(star1_live)&&star1_live>0.f;


    }
    __syncthreads();

    if(!__syncthreads_and(valid)){if(tid==0u)atomicOr(&d_convergence[leaf].terminal_fault_latch,2u);return;}

    // 3. Bounded DEC initialization. Phase3 independently verifies the true
    // residual before using this provisional field in propagation.
    smem.phi[tid] = is_core ? tile->phi_dec[tid] : (q_live * 0.07957747f);
    __syncthreads();
    float diagonal=1e-4f;
    #pragma unroll 1
    for(uint32_t e=0;e<num_edges;++e)
        if(smem.edge_u[e]==tid||smem.edge_v[e]==tid)diagonal+=smem.star1[e];
    smem.diag_inv[tid]=diagonal;
    __syncthreads();
    float initial_dec_residual = CUDART_INF_F;
    uint32_t initial_dec_iters = 0u;
    biprism_dec_pcg_256(
        smem.edge_u, smem.edge_v, 1u, smem.star1, num_edges, q_live,
        smem.phi, smem.cg_r, smem.cg_z, smem.cg_p, smem.cg_Ap,
        smem.diag_inv, smem.warp_red, initial_dec_residual, initial_dec_iters
    );

    tile->phi_dec[tid]   = smem.phi[tid];
    tile->star0_vol[tid] = smem.star0[tid];
    if (tid == 0) tile->pachner_flips_step = smem.delaunay_violations;

    for (uint32_t e = tid; e < num_edges; e += 256) {
        const uint8_t u = smem.edge_u[e], v = smem.edge_v[e];
        const uint32_t elem_u = UNPACK_ELEM_CLASS(tile->global_atom_idx[u]);
        const uint32_t elem_v = UNPACK_ELEM_CLASS(tile->global_atom_idx[v]);
        float S_uv, E0, V01, E1;
        float3 grad_S, grad_E0, grad_E1, grad_V01;
        const auto a=c_gfn_xtb_lut[elem_u&7u],b=c_gfn_xtb_lut[elem_v&7u];
        const auto xu=bp_coord_value(bp_coord_load(owner,BiprismCoordinatePlane::Current,UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u])));
        const auto xv=bp_coord_value(bp_coord_load(owner,BiprismCoordinatePlane::Current,UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[v])));
        BiprismElectronicEdgeProvider<double> p;
        const bool ok=bp_ed_evaluate_edge(
            BiprismElectronicElementParams<double>{double(a.zeta_slater),double(a.h_s_valence),double(a.h_p_excited),double(a.hubbard_eta)},
            BiprismElectronicElementParams<double>{double(b.zeta_slater),double(b.h_s_valence),double(b.h_p_excited),double(b.hubbard_eta)},
            BiprismElectronicVector<double>{xu.x,xu.y,xu.z},BiprismElectronicVector<double>{xv.x,xv.y,xv.z},
            double(smem.cur_pos_q[u].w),double(smem.cur_pos_q[v].w),p);
        valid=valid&&ok;if(!ok)continue;
        S_uv=float(p.S);E0=float(p.E0);E1=float(p.E1);V01=float(p.V01);
        grad_S={float(p.grad_S_u.x),float(p.grad_S_u.y),float(p.grad_S_u.z)};
        grad_E0={float(p.grad_E0_u.x),float(p.grad_E0_u.y),float(p.grad_E0_u.z)};
        grad_E1={float(p.grad_E1_u.x),float(p.grad_E1_u.y),float(p.grad_E1_u.z)};
        grad_V01={float(p.grad_V01_u.x),float(p.grad_V01_u.y),float(p.grad_V01_u.z)};
        valid=valid&&isfinite(S_uv)&&isfinite(E0)&&isfinite(E1)&&isfinite(V01)
            &&isfinite(grad_S.x)&&isfinite(grad_S.y)&&isfinite(grad_S.z)
            &&isfinite(grad_E0.x)&&isfinite(grad_E0.y)&&isfinite(grad_E0.z)
            &&isfinite(grad_E1.x)&&isfinite(grad_E1.y)&&isfinite(grad_E1.z)
            &&isfinite(grad_V01.x)&&isfinite(grad_V01.y)&&isfinite(grad_V01.z);
        tile->s_ovlp_edges[e]     = S_uv;
        tile->grad_s_edges[e]     = grad_S;
        tile->h_cov_edges[e][0]   = E0;
        tile->h_cov_edges[e][1]   = V01;
        tile->h_cov_edges[e][2]   = V01;
        tile->h_cov_edges[e][3]   = E1;
        tile->grad_h_diag_u[e][0] = grad_E0;
        tile->grad_h_diag_u[e][1] = grad_E1;
        tile->grad_h_cpl_u[e]     = grad_V01;
    }
    valid=valid&&isfinite(smem.phi[tid])&&isfinite(smem.star0[tid]);
    if(!__syncthreads_and(valid)&&tid==0u)atomicOr(&d_convergence[leaf].terminal_fault_latch,2u);
}

#endif // NHS_BIPRISM_SIMPLICIAL_TILE_CUH_INCLUDED

