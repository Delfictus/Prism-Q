// Common electronic discrete operators for the existing Biprism energy functional.
// The owner must solve the nonlinear endpoint-density system and the coupled
// coordinate/DEC equations. These helpers do not certify whole-step conservation.
#pragma once
#ifndef NHS_BIPRISM_ELECTRONIC_DISCRETE_CUH_INCLUDED
#define NHS_BIPRISM_ELECTRONIC_DISCRETE_CUH_INCLUDED
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// One algebra is instantiated in FP32 by production and in FP64 by controls.
// Host and device use explicit round-to-nearest FMA; no fast-math requirement.
__host__ __device__ __forceinline__ float bp_ed_fma(float a, float b, float c) {
#ifdef __CUDA_ARCH__
    return __fmaf_rn(a, b, c);
#else
    return fmaf(a, b, c);
#endif
}
__host__ __device__ __forceinline__ double bp_ed_fma(double a, double b, double c) {
#ifdef __CUDA_ARCH__
    return __fma_rn(a, b, c);
#else
    return fma(a, b, c);
#endif
}
template<class R> __host__ __device__ __forceinline__ R bp_ed_abs(R a) { return a < R(0) ? -a : a; }
template<class R> __host__ __device__ __forceinline__ R bp_ed_max(R a, R b) { return a > b ? a : b; }
template<class R> struct BiprismElectronicComplex { R re, im; };
template<class R> struct BiprismElectronicSpinor { BiprismElectronicComplex<R> s0, s1; };
template<class R> struct BiprismElectronicDensity { R r00, r11, r01_re, r01_im; };
// H_01=h01_re+i*h01_im; H_10=conj(H_01). rho_01=psi_0*conj(psi_1).
template<class R> struct BiprismElectronicHermitian { R h00, h11, h01_re, h01_im; };
template<class R> struct BiprismElectronicVector { R x, y, z; };
template<class R> struct BiprismElectronicEndpointData {
    BiprismElectronicSpinor<R> psi_bar;       // (psi_n+psi_np1)/2, never normalized
    BiprismElectronicDensity<R> rho_bar;    // (rho_n+rho_np1)/2, not rho(psi_bar)
    R q_n, q_np1, q_bar;
};
template<class R> struct BiprismElectronicEdgeDiscrete {
    BiprismElectronicHermitian<R> owner_h;  // direct edge energy belongs to u only
    R charge_derivative_u, charge_derivative_v; // dE_edge/dq, before factor0.15
};
template<class R> struct BiprismElectronicCayleyResult {
    BiprismElectronicSpinor<R> psi_np1;
    R norm_defect;
    R scaled_equation_residual;
};
// These values are passed from the existing c_gfn_xtb_lut; no second LUT owner.
template<class R> struct BiprismElectronicElementParams {
    R zeta_slater, h_s_valence, h_p_excited, hubbard_eta;
};
template<class R> struct BiprismElectronicEdgeProvider {
    R S, E0, E1, V01, gamma;
    BiprismElectronicVector<R> grad_S_u, grad_E0_u, grad_E1_u, grad_V01_u, grad_gamma_u;
};
__host__ __device__ __forceinline__ float bp_ed_sqrt(float x) { return sqrtf(x); }
__host__ __device__ __forceinline__ double bp_ed_sqrt(double x) { return sqrt(x); }
__host__ __device__ __forceinline__ float bp_ed_exp(float x) { return expf(x); }
__host__ __device__ __forceinline__ double bp_ed_exp(double x) { return exp(x); }
// Shared provider formula: coordinates, q and affine charge terms stay in R.
// FP64 callers must not first pack q into float4. The actual source's FP32
// coefficients/regularizers are promoted, preserving their represented values.
// Precise sqrt/exp replace the old approximate FP32 intrinsic realization;
// this is a precision change, not a bitwise-identity claim for the old provider.
template<class R> __host__ __device__ __forceinline__ bool
bp_ed_evaluate_edge(BiprismElectronicElementParams<R> u,BiprismElectronicElementParams<R> v,
    BiprismElectronicVector<R> xu,BiprismElectronicVector<R> xv,R qu,R qv,
    BiprismElectronicEdgeProvider<R>& out) {
    if(!isfinite(xu.x)||!isfinite(xu.y)||!isfinite(xu.z)||!isfinite(xv.x)||!isfinite(xv.y)||!isfinite(xv.z)
        ||!isfinite(qu)||!isfinite(qv)||!isfinite(u.zeta_slater)||!isfinite(v.zeta_slater)
        ||!(u.zeta_slater>R(0))||!(v.zeta_slater>R(0))||!isfinite(u.h_s_valence)||!isfinite(u.h_p_excited)
        ||!isfinite(v.h_s_valence)||!isfinite(v.h_p_excited)||!isfinite(u.hubbard_eta)||!isfinite(v.hubbard_eta)
        ||!(u.hubbard_eta>R(0))||!(v.hubbard_eta>R(0)))return false;
    const R dx=xv.x-xu.x,dy=xv.y-xu.y,dz=xv.z-xu.z;
    const R r2=bp_ed_fma(dx,dx,bp_ed_fma(dy,dy,dz*dz))+R(1e-10f);
    if(!isfinite(r2)||!(r2>R(0)))return false;
    const R r=bp_ed_sqrt(r2),inv_r=R(1)/r;
    const R zeta=(R(2)*u.zeta_slater*v.zeta_slater)/(u.zeta_slater+v.zeta_slater);
    const R zr=zeta*r,c=R(0.33333333f),ex=bp_ed_exp(-zr);
    const R S=bp_ed_fma(c,zr*zr,R(1)+zr)*ex;
    // Exact derivative of (1+zr+c*zr^2)*exp(-zr), including represented c.
    // The old -c*zeta*zr*(1+zr)*exp assumes 3*c==1 and is not that derivative.
    const R dS_dr=zeta*zr*((R(2)*c-R(1))-c*zr)*ex;
    const R ako=R(1)/u.hubbard_eta+R(1)/v.hubbard_eta;
    const R gamma=R(1)/bp_ed_sqrt(bp_ed_fma(ako,ako,r2));
    const R dgamma_dr=-r*gamma*gamma*gamma;
    const R dq=qv-qu;
    const R E0=u.h_s_valence-R(.5)*dq*gamma;
    const R E1=v.h_p_excited+R(.5)*dq*gamma;
    const R kwh=R(-0.875f)*(u.h_s_valence+v.h_p_excited);
    const R V01=kwh*S,dV_dr=kwh*dS_dr,dE_dr=R(.5)*dq*dgamma_dr;
    const BiprismElectronicVector<R> n={dx*inv_r,dy*inv_r,dz*inv_r};
    out={S,E0,E1,V01,gamma,
        {-dS_dr*n.x,-dS_dr*n.y,-dS_dr*n.z},
        {dE_dr*n.x,dE_dr*n.y,dE_dr*n.z},
        {-dE_dr*n.x,-dE_dr*n.y,-dE_dr*n.z},
        {-dV_dr*n.x,-dV_dr*n.y,-dV_dr*n.z},
        {-dgamma_dr*n.x,-dgamma_dr*n.y,-dgamma_dr*n.z}};
    return isfinite(S)&&isfinite(E0)&&isfinite(E1)&&isfinite(V01)&&isfinite(gamma)
        &&isfinite(dS_dr)&&isfinite(dV_dr)&&isfinite(dE_dr)&&isfinite(dgamma_dr);
}
using BiprismElectronicComplexF32 = BiprismElectronicComplex<float>;
using BiprismElectronicSpinorF32 = BiprismElectronicSpinor<float>;
using BiprismElectronicDensityF32 = BiprismElectronicDensity<float>;
using BiprismElectronicHermitianF32 = BiprismElectronicHermitian<float>;
using BiprismElectronicVectorF32 = BiprismElectronicVector<float>;

template<class R> __host__ __device__ __forceinline__ BiprismElectronicComplex<R>
bp_ed_add(BiprismElectronicComplex<R> a, BiprismElectronicComplex<R> b) { return {a.re+b.re,a.im+b.im}; }
template<class R> __host__ __device__ __forceinline__ BiprismElectronicComplex<R>
bp_ed_sub(BiprismElectronicComplex<R> a, BiprismElectronicComplex<R> b) { return {a.re-b.re,a.im-b.im}; }
template<class R> __host__ __device__ __forceinline__ BiprismElectronicComplex<R>
bp_ed_scale(BiprismElectronicComplex<R> a,R b) { return {a.re*b,a.im*b}; }
template<class R> __host__ __device__ __forceinline__ BiprismElectronicComplex<R>
bp_ed_mul(BiprismElectronicComplex<R> a,BiprismElectronicComplex<R> b) {
    return {bp_ed_fma(a.re,b.re,-a.im*b.im),bp_ed_fma(a.re,b.im,a.im*b.re)};
}
template<class R> __host__ __device__ __forceinline__ R bp_ed_norm2(BiprismElectronicComplex<R> a) {
    return bp_ed_fma(a.re,a.re,a.im*a.im);
}
template<class R> __host__ __device__ __forceinline__ bool bp_ed_finite(BiprismElectronicComplex<R> a) {
    return isfinite(a.re)&&isfinite(a.im);
}
template<class R> __host__ __device__ __forceinline__ bool bp_ed_finite(BiprismElectronicSpinor<R> a) {
    return bp_ed_finite(a.s0)&&bp_ed_finite(a.s1);
}
template<class R> __host__ __device__ __forceinline__ BiprismElectronicDensity<R>
bp_ed_density(BiprismElectronicSpinor<R> p) {
    return {bp_ed_norm2(p.s0),bp_ed_norm2(p.s1),
        bp_ed_fma(p.s0.re,p.s1.re,p.s0.im*p.s1.im),
        bp_ed_fma(p.s0.im,p.s1.re,-p.s0.re*p.s1.im)};
}
template<class R> __host__ __device__ __forceinline__ BiprismElectronicEndpointData<R>
bp_ed_endpoint_data(BiprismElectronicSpinor<R> p0,BiprismElectronicSpinor<R> p1,R q_amber) {
    const auto r0=bp_ed_density(p0),r1=bp_ed_density(p1);
    BiprismElectronicEndpointData<R> out;
    out.psi_bar={bp_ed_scale(bp_ed_add(p0.s0,p1.s0),R(.5)),bp_ed_scale(bp_ed_add(p0.s1,p1.s1),R(.5))};
    out.rho_bar={R(.5)*(r0.r00+r1.r00),R(.5)*(r0.r11+r1.r11),R(.5)*(r0.r01_re+r1.r01_re),R(.5)*(r0.r01_im+r1.r01_im)};
    out.q_n=bp_ed_fma(R(0.15f),r0.r11,q_amber);
    out.q_np1=bp_ed_fma(R(0.15f),r1.r11,q_amber);
    out.q_bar=R(.5)*(out.q_n+out.q_np1);
    return out;
}
// omega is the actual vertex star0 weight; i*omega*dpsi/dt=G. DEC phi is
// evaluated at endpoint-average q with the actual fixed Dirichlet halo values.
template<class R> __host__ __device__ __forceinline__ BiprismElectronicHermitian<R>
bp_ed_onsite_dec_hamiltonian(R omega,R eps_s,R eps_p,R phi_bar) {
    return {omega*eps_s,bp_ed_fma(R(0.15f),phi_bar,omega*eps_p),R(0),R(0)};
}
template<class R> __host__ __device__ __forceinline__ BiprismElectronicHermitian<R>
bp_ed_hadd(BiprismElectronicHermitian<R> a,BiprismElectronicHermitian<R> b) {
    return {a.h00+b.h00,a.h11+b.h11,a.h01_re+b.h01_re,a.h01_im+b.h01_im};
}
// E0,E1,V01 must be the SAME unrotated provider values used in V_e=N/D,
// evaluated at q_bar; gamma is that provider's Ohno coefficient. No star1
// multiplier is added to the direct electronic energy or its derivative.
template<class R> __host__ __device__ __forceinline__ BiprismElectronicEdgeDiscrete<R>
bp_ed_edge_discrete(R E0,R E1,R V01,R S,R gamma,BiprismElectronicDensity<R> owner_rho_bar) {
    const R inv_D=R(1)/bp_ed_max(R(1)-S*S,R(1e-3f));
    const R chain=R(.5)*gamma*(owner_rho_bar.r11-owner_rho_bar.r00)*inv_D;
    return {{E0*inv_D,E1*inv_D,V01*inv_D,R(0)},-chain,chain};
}
// Apply only to electronically dynamic endpoints. Fixed halo charge has no
// state derivative, although its owner-density contribution remains in E.
template<class R> __host__ __device__ __forceinline__ void
bp_ed_add_charge_chain(BiprismElectronicHermitian<R>& H,R derivative_q) {
    H.h11=bp_ed_fma(R(0.15f),derivative_q,H.h11);
}
template<class R> __host__ __device__ __forceinline__ R
bp_ed_edge_energy(R E0,R E1,R V01,R S,BiprismElectronicDensity<R> rho) {
    const R N=bp_ed_fma(rho.r00,E0,bp_ed_fma(rho.r11,E1,R(2)*rho.r01_re*V01));
    return N/bp_ed_max(R(1)-S*S,R(1e-3f));
}
template<class R> __host__ __device__ __forceinline__ BiprismElectronicSpinor<R>
bp_ed_apply(BiprismElectronicHermitian<R> H,BiprismElectronicSpinor<R> p) {
    const BiprismElectronicComplex<R> h01={H.h01_re,H.h01_im},h10={H.h01_re,-H.h01_im};
    return {bp_ed_add(bp_ed_scale(p.s0,H.h00),bp_ed_mul(h01,p.s1)),
            bp_ed_add(bp_ed_mul(h10,p.s0),bp_ed_scale(p.s1,H.h11))};
}
template<class R> __host__ __device__ __forceinline__ R
bp_ed_pairing_real(BiprismElectronicSpinor<R> a,BiprismElectronicSpinor<R> b) {
    return bp_ed_fma(a.s0.re,b.s0.re,bp_ed_fma(a.s0.im,b.s0.im,bp_ed_fma(a.s1.re,b.s1.re,a.s1.im*b.s1.im)));
}
template<class R> __host__ __device__ __forceinline__ R
bp_ed_discrete_electronic_work(BiprismElectronicHermitian<R> H_energy,
    BiprismElectronicSpinor<R> p0,BiprismElectronicSpinor<R> p1) {
    const BiprismElectronicSpinor<R> pm={bp_ed_scale(bp_ed_add(p0.s0,p1.s0),R(.5)),bp_ed_scale(bp_ed_add(p0.s1,p1.s1),R(.5))};
    const BiprismElectronicSpinor<R> dp={bp_ed_sub(p1.s0,p0.s0),bp_ed_sub(p1.s1,p0.s1)};
    return R(2)*bp_ed_pairing_real(bp_ed_apply(H_energy,pm),dp);
}
// Scaled direct solution of (omega I+i dt H/2)psi1=(omega I-i dt H/2)psi0.
// No eigenbasis rotation, renormalization, determinant floor or hidden retry.
// false means unsupported/nonfinite arithmetic; the owner must fault/refuse.
// A true result still carries equation/norm diagnostics for owner admission.
template<class R> __host__ __device__ __forceinline__ bool
bp_ed_hermitian_cayley(BiprismElectronicSpinor<R> p0,BiprismElectronicHermitian<R> H,
    R omega,R dt,BiprismElectronicCayleyResult<R>& out) {
    out={p0,R(0),R(0)};
    if(!bp_ed_finite(p0)||!isfinite(H.h00)||!isfinite(H.h11)||!isfinite(H.h01_re)||!isfinite(H.h01_im)
        ||!isfinite(omega)||!(omega>R(0))||!isfinite(dt)) return false;
    const R n0=bp_ed_norm2(p0.s0)+bp_ed_norm2(p0.s1);
    if(!isfinite(n0))return false;
    const R a=R(.5)*dt*H.h00,d=R(.5)*dt*H.h11,tr=R(.5)*dt*H.h01_re,ti=R(.5)*dt*H.h01_im;
    if(!isfinite(a)||!isfinite(d)||!isfinite(tr)||!isfinite(ti))return false;
    const R scale=bp_ed_max(omega,bp_ed_max(bp_ed_abs(a),bp_ed_max(bp_ed_abs(d),bp_ed_max(bp_ed_abs(tr),bp_ed_abs(ti)))));
    const R w=omega/scale,aa=a/scale,dd=d/scale,rr=tr/scale,ii=ti/scale;
    const BiprismElectronicComplex<R> A={w,aa},B={-ii,rr},C={ii,rr},D={w,dd};
    const BiprismElectronicComplex<R> det=bp_ed_sub(bp_ed_mul(A,D),bp_ed_mul(B,C));
    const R ds=bp_ed_max(bp_ed_abs(det.re),bp_ed_abs(det.im));
    if(!isfinite(ds)||!(ds>R(0)))return false;
    const R dr=det.re/ds,di=det.im/ds,den=bp_ed_fma(dr,dr,di*di);
    const auto b0=bp_ed_scale(p0.s0,w),b1=bp_ed_scale(p0.s1,w);
    const auto n_0=bp_ed_sub(bp_ed_mul(D,b0),bp_ed_mul(B,b1));
    const auto n_1=bp_ed_sub(bp_ed_mul(A,b1),bp_ed_mul(C,b0));
    const R n0r=n_0.re/ds,n0i=n_0.im/ds,n1r=n_1.re/ds,n1i=n_1.im/ds;
    const BiprismElectronicComplex<R> y0={bp_ed_fma(n0r,dr,n0i*di)/den,bp_ed_fma(n0i,dr,-n0r*di)/den};
    const BiprismElectronicComplex<R> y1={bp_ed_fma(n1r,dr,n1i*di)/den,bp_ed_fma(n1i,dr,-n1r*di)/den};
    const BiprismElectronicSpinor<R> p1={bp_ed_sub(bp_ed_scale(y0,R(2)),p0.s0),bp_ed_sub(bp_ed_scale(y1,R(2)),p0.s1)};
    if(!bp_ed_finite(p1))return false;
    const auto sum0=bp_ed_add(p1.s0,p0.s0),sum1=bp_ed_add(p1.s1,p0.s1);
    const auto delta0=bp_ed_sub(p1.s0,p0.s0),delta1=bp_ed_sub(p1.s1,p0.s1);
    const auto hs0=bp_ed_add(bp_ed_scale(sum0,aa),bp_ed_mul(BiprismElectronicComplex<R>{rr,ii},sum1));
    const auto hs1=bp_ed_add(bp_ed_mul(BiprismElectronicComplex<R>{rr,-ii},sum0),bp_ed_scale(sum1,dd));
    const auto res0=bp_ed_add(bp_ed_scale(delta0,w),BiprismElectronicComplex<R>{-hs0.im,hs0.re});
    const auto res1=bp_ed_add(bp_ed_scale(delta1,w),BiprismElectronicComplex<R>{-hs1.im,hs1.re});
    out.psi_np1=p1;
    out.norm_defect=bp_ed_abs(bp_ed_norm2(p1.s0)+bp_ed_norm2(p1.s1)-n0);
    out.scaled_equation_residual=bp_ed_max(bp_ed_max(bp_ed_abs(res0.re),bp_ed_abs(res0.im)),bp_ed_max(bp_ed_abs(res1.re),bp_ed_abs(res1.im)));
    return isfinite(out.norm_defect)&&isfinite(out.scaled_equation_residual);
}
// Coordinate derivative at fixed endpoint-average density/charge, same provider
// gradients w.r.t. x_u and same clamped D as the energy. At D's kink use the
// already selected clamped-side derivative, without changing numerator terms.
template<class R> __host__ __device__ __forceinline__ BiprismElectronicVector<R>
bp_ed_edge_coordinate_force(R E0,R E1,R V01,R S,BiprismElectronicDensity<R> rho,
    BiprismElectronicVector<R> grad_E0,BiprismElectronicVector<R> grad_E1,
    BiprismElectronicVector<R> grad_V01,BiprismElectronicVector<R> grad_S) {
    const R raw_D=R(1)-S*S,inv_D=R(1)/bp_ed_max(raw_D,R(1e-3f));
    const R N=bp_ed_fma(rho.r00,E0,bp_ed_fma(rho.r11,E1,R(2)*rho.r01_re*V01));
    const R pulay=raw_D>R(1e-3f)?R(2)*N*S*inv_D*inv_D:R(0);
    return {
        -inv_D*bp_ed_fma(rho.r00,grad_E0.x,bp_ed_fma(rho.r11,grad_E1.x,R(2)*rho.r01_re*grad_V01.x))-pulay*grad_S.x,
        -inv_D*bp_ed_fma(rho.r00,grad_E0.y,bp_ed_fma(rho.r11,grad_E1.y,R(2)*rho.r01_re*grad_V01.y))-pulay*grad_S.y,
        -inv_D*bp_ed_fma(rho.r00,grad_E0.z,bp_ed_fma(rho.r11,grad_E1.z,R(2)*rho.r01_re*grad_V01.z))-pulay*grad_S.z};
}
// Preserve the existing C1 connection coefficient in numerator-H units:
// kappa=star1*grad_V01/(2*D*sqrt(dE^2+4V01^2+eps_ci^2)).
// The root's actual provider owns the geometry, eps_ci and gradient convention.
template<class R> __host__ __device__ __forceinline__ BiprismElectronicVector<R>
bp_ed_connection_coefficient(R star1,R S,R gap_inverse_sqrt,BiprismElectronicVector<R> grad_V01) {
    const R c=R(.5)*star1*gap_inverse_sqrt/bp_ed_max(R(1)-S*S,R(1e-3f));
    return {c*grad_V01.x,c*grad_V01.y,c*grad_V01.z};
}
template<class R> __host__ __device__ __forceinline__ R
bp_ed_dot(BiprismElectronicVector<R> a,BiprismElectronicVector<R> b) {
    return bp_ed_fma(a.x,b.x,bp_ed_fma(a.y,b.y,a.z*b.z));
}
template<class R> __host__ __device__ __forceinline__ void
bp_ed_add_connection(BiprismElectronicHermitian<R>& Hu,BiprismElectronicHermitian<R>& Hv,
    BiprismElectronicVector<R> kappa,BiprismElectronicVector<R> v_rel,bool dynamic_u,bool dynamic_v) {
    const R lambda=bp_ed_dot(kappa,v_rel);
    // sigma_y has H01=-i: numerator Hconn_u=+lambda sigma_y; v gets the opposite.
    if(dynamic_u)Hu.h01_im-=lambda;
    if(dynamic_v)Hv.h01_im+=lambda;
}
template<class R> __host__ __device__ __forceinline__ R
bp_ed_connection_energy_pairing(BiprismElectronicHermitian<R> H_energy,BiprismElectronicSpinor<R> psi_bar) {
    const auto G=bp_ed_apply(H_energy,psi_bar);
    const BiprismElectronicSpinor<R> sigma_y_psi={{psi_bar.s1.im,-psi_bar.s1.re},{-psi_bar.s0.im,psi_bar.s0.re}};
    // Im[(H_energy psi_bar)^dagger sigma_y psi_bar]; connection excluded from G.
    return bp_ed_fma(G.s0.re,sigma_y_psi.s0.im,bp_ed_fma(-G.s0.im,sigma_y_psi.s0.re,
           bp_ed_fma(G.s1.re,sigma_y_psi.s1.im,-G.s1.im*sigma_y_psi.s1.re)));
}
template<class R> __host__ __device__ __forceinline__ BiprismElectronicVector<R>
bp_ed_reciprocal_connection_force(BiprismElectronicVector<R> kappa,
    BiprismElectronicHermitian<R> Hu_energy,BiprismElectronicSpinor<R> u_bar,R omega_u,bool dynamic_u,
    BiprismElectronicHermitian<R> Hv_energy,BiprismElectronicSpinor<R> v_bar,R omega_v,bool dynamic_v) {
    const R zu=dynamic_u?bp_ed_connection_energy_pairing(Hu_energy,u_bar)/omega_u:R(0);
    const R zv=dynamic_v?bp_ed_connection_energy_pairing(Hv_energy,v_bar)/omega_v:R(0);
    const R c=R(2)*(zu-zv);
    // Deposit +F at coordinate u and -F at coordinate v. An excluded/fixed
    // coordinate carries boundary work; do not claim closed-subsystem balance.
    return {c*kappa.x,c*kappa.y,c*kappa.z};
}
#endif
