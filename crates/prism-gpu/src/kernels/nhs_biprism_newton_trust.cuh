#pragma once
#include <cmath>
#include <stdint.h>
#ifdef __CUDACC__
#define BP_NT_HD __host__ __device__ __forceinline__
#define BP_NT_LOOP _Pragma("unroll 1")
#else
#define BP_NT_HD inline
#define BP_NT_LOOP
#endif
// Solve (I + lambda T^T T)y = -Q^T F in the measured (<=N) column space.
// The identity makes this SPD even for dependent input directions. All mutable
// storage is explicitly supplied; no local arrays or physical state are used.
template<uint32_t N> BP_NT_HD bool bp_newton_regularized(uint32_t n,double lambda,const volatile double* gram,
    const volatile double* c,volatile double* factor,volatile double* y) {
    if(n==0u||n>N||!std::isfinite(lambda)||lambda<0.)return false;
    BP_NT_LOOP for(uint32_t i=0;i<n;++i){
        BP_NT_LOOP for(uint32_t j=0;j<=i;++j){
            double a=(i==j?1.:0.)+lambda*gram[i*N+j];
            BP_NT_LOOP for(uint32_t k=0;k<j;++k)a-=factor[i*N+k]*factor[j*N+k];
            if(i==j){if(!(a>0.)||!std::isfinite(a))return false;factor[i*N+j]=sqrt(a);}
            else factor[i*N+j]=a/factor[j*N+j];
        }
        double a=-c[i];BP_NT_LOOP for(uint32_t k=0;k<i;++k)a-=factor[i*N+k]*y[k];y[i]=a/factor[i*N+i];
    }
    BP_NT_LOOP for(int i=int(n)-1;i>=0;--i){double a=y[i];BP_NT_LOOP for(uint32_t k=uint32_t(i)+1;k<n;++k)a-=factor[k*N+i]*y[k];y[i]=a/factor[i*N+i];if(!std::isfinite(y[i]))return false;}
    return true;
}
template<uint32_t N> BP_NT_HD double bp_newton_input_norm(uint32_t n,const volatile double* gram,const volatile double* y){
    double out=0.;BP_NT_LOOP for(uint32_t i=0;i<n;++i){double row=0.;BP_NT_LOOP for(uint32_t j=0;j<n;++j)row+=gram[i*N+j]*y[j];out+=y[i]*row;}return out;
}
// Lagrange multiplier search changes only the proposed correction. Original
// equations and physical acceptance remain the caller's independent checks.
template<uint32_t N> BP_NT_HD bool bp_newton_trust(uint32_t n,double radius2,const volatile double* gram,const volatile double* c,
    volatile double* factor,volatile double* y,volatile double& lambda){
    lambda=0.;if(!(radius2>0.)||!std::isfinite(radius2)||!bp_newton_regularized<N>(n,0.,gram,c,factor,y))return false;
    if(bp_newton_input_norm<N>(n,gram,y)<=radius2)return true;
    double lo=0.,hi=1.;
    BP_NT_LOOP for(uint32_t i=0;i<64u;++i){if(!bp_newton_regularized<N>(n,hi,gram,c,factor,y))return false;
        if(bp_newton_input_norm<N>(n,gram,y)<=radius2)break;lo=hi;hi*=4.;}
    if(bp_newton_input_norm<N>(n,gram,y)>radius2)return false;
    BP_NT_LOOP for(uint32_t i=0;i<40u;++i){const double mid=.5*(lo+hi);if(!bp_newton_regularized<N>(n,mid,gram,c,factor,y))return false;
        if(bp_newton_input_norm<N>(n,gram,y)>radius2)lo=mid;else hi=mid;}
    lambda=hi;return bp_newton_regularized<N>(n,hi,gram,c,factor,y);
}
// Historical 16-column controls keep their original packing.
BP_NT_HD bool bp_newton_trust16(uint32_t n,double radius2,const volatile double* gram,const volatile double* c,
    volatile double* factor,volatile double* y,volatile double& lambda){
    return bp_newton_trust<16>(n,radius2,gram,c,factor,y,lambda);
}
#undef BP_NT_HD
#undef BP_NT_LOOP
