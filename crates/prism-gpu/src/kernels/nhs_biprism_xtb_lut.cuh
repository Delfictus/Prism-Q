// File: roles/engine/crates/prism-gpu/src/kernels/nhs_biprism_xtb_lut.cuh
#pragma once
#ifndef NHS_BIPRISM_XTB_LUT_CUH_INCLUDED
#define NHS_BIPRISM_XTB_LUT_CUH_INCLUDED

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

#define UNPACK_GLOBAL_ATOM_ID(packed)  ((packed) & 0x00FFFFFFu)
#define UNPACK_ELEM_CLASS(packed)      (((packed) >> 24u) & 0x7u)
#define PACK_ATOM_ELEM(id, elem_cls)   (((id) & 0x00FFFFFFu) | (((elem_cls) & 0x7u) << 24u))

struct __align__(16) GfnXtbElementParams {
    float zeta_slater;
    float h_s_valence;
    float h_p_excited;
    float hubbard_eta;
};

__constant__ GfnXtbElementParams c_gfn_xtb_lut[8] = {
    {1.240f, -0.500f, -0.120f, 0.472f}, // 0: H
    {1.625f, -0.558f, -0.392f, 0.380f}, // 1: C
    {1.950f, -0.682f, -0.445f, 0.435f}, // 2: N
    {2.275f, -0.825f, -0.510f, 0.495f}, // 3: O
    {1.600f, -0.540f, -0.350f, 0.310f}, // 4: P
    {1.820f, -0.610f, -0.380f, 0.345f}, // 5: S
    {2.425f, -0.910f, -0.560f, 0.540f}, // 6: F/Cl
    {1.350f, -0.410f, -0.220f, 0.280f}  // 7: Metal/Other
};

// Scalarized Zero-Stack-Frame GFN-xTB Evaluator.
// Note: Electrostatic potential \phi is coupled exclusively through the variational DEC functional
// V_{DEC} = -0.5 \sum_e \star_1 (\phi_u - \phi_v)^2 + \sum_u q_u \phi_u so that \partial E_{tot}/\partial \phi_u == 0
// holds identically whenever the DEC Poisson residual (d_0^T \star_1 d_0 \phi - q)_u == 0.
__device__ __forceinline__ void evaluate_heteronuclear_gfn_xtb_edge_scalar(
    uint32_t elem_u, uint32_t elem_v,
    float4 pu, float4 pv,
    float /*phi_u_unused*/, float /*phi_v_unused*/,
    float& out_S_uv,
    float3& out_grad_S_u,
    float& out_E0, float& out_V01, float& out_E1,
    float3& out_grad_E0_u, float3& out_grad_E1_u,
    float3& out_grad_V01_u
) {
    const GfnXtbElementParams P_u = c_gfn_xtb_lut[elem_u & 7u];
    const GfnXtbElementParams P_v = c_gfn_xtb_lut[elem_v & 7u];

    const float dx = pv.x - pu.x, dy = pv.y - pu.y, dz = pv.z - pu.z;
    const float r2 = __fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz)) + 1e-10f;
    const float inv_r = rsqrtf(r2);
    const float r = r2 * inv_r;
    const float3 r_hat = {dx * inv_r, dy * inv_r, dz * inv_r};

    const float zeta = (2.0f * P_u.zeta_slater * P_v.zeta_slater) / (P_u.zeta_slater + P_v.zeta_slater);
    const float zr = zeta * r;
    const float exp_zr = __expf(-zr);
    out_S_uv = __fmaf_rn(0.33333333f, zr * zr, 1.0f + zr) * exp_zr;
    const float dS_dr = -(0.33333333f * zeta * zeta * r) * (1.0f + zr) * exp_zr;
    out_grad_S_u = {-dS_dr * r_hat.x, -dS_dr * r_hat.y, -dS_dr * r_hat.z};

    const float a_ko = (1.0f / P_u.hubbard_eta) + (1.0f / P_v.hubbard_eta);
    const float inv_ko_denom = rsqrtf(__fmaf_rn(a_ko, a_ko, r2));
    const float dgamma_dr = -r * (inv_ko_denom * inv_ko_denom * inv_ko_denom);

    const float dq_gamma = (pv.w - pu.w) * inv_ko_denom;
    out_E0  = P_u.h_s_valence - 0.5f * dq_gamma;
    out_E1  = P_v.h_p_excited + 0.5f * dq_gamma;

    const float k_wh = -0.875f * (P_u.h_s_valence + P_v.h_p_excited);
    out_V01 = k_wh * out_S_uv;
    const float dV01_dr = k_wh * dS_dr;

    const float dE_ko_dr = 0.5f * (pv.w - pu.w) * dgamma_dr;
    out_grad_E0_u  = { dE_ko_dr * r_hat.x,  dE_ko_dr * r_hat.y,  dE_ko_dr * r_hat.z};
    out_grad_E1_u  = {-dE_ko_dr * r_hat.x, -dE_ko_dr * r_hat.y, -dE_ko_dr * r_hat.z};
    out_grad_V01_u = {-dV01_dr  * r_hat.x, -dV01_dr  * r_hat.y, -dV01_dr  * r_hat.z};
}

__device__ __forceinline__ void evaluate_heteronuclear_gfn_xtb_edge(
    uint32_t elem_u, uint32_t elem_v,
    float4 pu, float4 pv,
    float phi_u, float phi_v,
    float& out_S_uv,
    float3& out_grad_S_u,
    float  out_H_blk[4],
    float3 out_grad_H_diag_u[2],
    float3& out_grad_V01_u
) {
    float E0, V01, E1;
    evaluate_heteronuclear_gfn_xtb_edge_scalar(
        elem_u, elem_v, pu, pv, phi_u, phi_v,
        out_S_uv, out_grad_S_u, E0, V01, E1,
        out_grad_H_diag_u[0], out_grad_H_diag_u[1], out_grad_V01_u
    );
    out_H_blk[0] = E0;
    out_H_blk[1] = V01;
    out_H_blk[2] = V01;
    out_H_blk[3] = E1;
}

#endif // NHS_BIPRISM_XTB_LUT_CUH_INCLUDED
