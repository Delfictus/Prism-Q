#pragma once
#include <math.h>
#include <stdint.h>
#ifdef __CUDACC__
#define BP_MF_INLINE __host__ __device__ __forceinline__
#else
#define BP_MF_INLINE inline
#endif
// FP64 realization of x+sin(x), the existing magnetic field amplitude.
// A scalar three-part pi/2 reduction avoids libdevice's40B local slow-path
// array. For |x|>=2^54, RN(x+sin(x))=x because |sin(x)|<=1. This is a
// rounding identity, not a physical phase limit or force clipping.
// Smaller arguments reduce to [-pi/4,pi/4]. Taylor remainders are below
// 1e-19 (sine) and4e-19 (cosine); rounding is independently checked with
// MPFR over binary64 exponent ranges and reduction boundaries.
namespace biprism_magnetic_f64 {
BP_MF_INLINE double madd(double a,double b,double c) {
#ifdef __CUDA_ARCH__
    return __fma_rn(a,b,c);
#else
    return fma(a,b,c);
#endif
}
BP_MF_INLINE double phase_plus_sine(double x) {
    if(!isfinite(x))return x-x;
    if(x==0.||fabs(x)>=0x1p54)return x;
    const double p0=0x1.921fb54442d18p+0,p1=0x1.1a62633145c07p-54,p2=-0x1.f1976b7ed8fbcp-110;
    const long long n0=llrint(x*0x1.45f306dc9c883p-1);
    long long n=n0;
    double r=madd(-double(n),p0,x);
    r=madd(-double(n),p1,r);r=madd(-double(n),p2,r);
    // The rounded quotient may be off by at most2 at the largest input.
    // These corrections retain the same scalar ownership and no local array.
    #pragma unroll 1
    for(int k=0;k<3;++k){
        if(r>0x1.921fb54442d18p-1){++n;r=madd(-1.,p0,r);r=madd(-1.,p1,r);r=madd(-1.,p2,r);}
        else if(r< -0x1.921fb54442d18p-1){--n;r=madd(1.,p0,r);r=madd(1.,p1,r);r=madd(1.,p2,r);}
        else break;
    }
    const double z=r*r;
    double p;
    if((n&1ll)==0){
        p=0x1.952c77030ad4ap-49;
        p=madd(z,p,-0x1.ae7f3e733b81fp-41);
        p=madd(z,p,0x1.6124613a86d09p-33);
        p=madd(z,p,-0x1.ae64567f544e4p-26);
        p=madd(z,p,0x1.71de3a556c734p-19);
        p=madd(z,p,-0x1.a01a01a01a01ap-13);
        p=madd(z,p,0x1.1111111111111p-7);
        p=madd(z,p,-0x1.5555555555555p-3);
        p=madd(r*z,p,r);
    }else{
        p=0x1.ae7f3e733b81fp-45;
        p=madd(z,p,-0x1.93974a8c07c9dp-37);
        p=madd(z,p,0x1.1eed8eff8d898p-29);
        p=madd(z,p,-0x1.27e4fb7789f5cp-22);
        p=madd(z,p,0x1.a01a01a01a01ap-16);
        p=madd(z,p,-0x1.6c16c16c16c17p-10);
        p=madd(z,p,0x1.5555555555555p-5);
        p=madd(z,p,-0x1.0000000000000p-1);
        p=madd(z,p,1.);
    }
    if(n&2ll)p=-p;
    return x+p;
}
} // namespace biprism_magnetic_f64
#undef BP_MF_INLINE
