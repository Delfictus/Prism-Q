#pragma once
#include "nhs_biprism_electronic_discrete.cuh"

// A local derivative extension, never an accepted physical energy. Keep the
// canonical overlap branch of the evaluated base while measuring a symmetric
// endpoint-energy derivative. Independent actions still use the original max.
__host__ __device__ __forceinline__ bool bp_v10_endpoint_branch_model(
    uint32_t energy_bank,uint32_t phase,uint32_t mode,uint32_t tangent,uint32_t refinement
) {
    return (energy_bank&2u)!=0u&&phase==2u&&mode==1u&&tangent!=0u&&refinement>=6u;
}
template<class R> __host__ __device__ __forceinline__ R bp_v10_endpoint_branch_energy(
    R E0,R E1,R V01,R S,BiprismElectronicDensity<R> rho,R base_raw_D,bool model
) {
    if(!model)return bp_ed_edge_energy(E0,E1,V01,S,rho);
    const R D=base_raw_D>R(1e-3f)?R(1)-S*S:R(1e-3f);
    if(!isfinite(base_raw_D)||!isfinite(D)||!(D>R(0)))return R(NAN);
    const R N=bp_ed_fma(rho.r00,E0,bp_ed_fma(rho.r11,E1,R(2)*rho.r01_re*V01));
    return N/D;
}
