#pragma once

// Frozen-metric algebra for Biprism. This header does not implement a moving-
// metric nuclear integrator, electronic-structure model, or gauge transport.
// Arithmetic is binary32, including diagnostics. Callers MUST check status;
// failed results deliberately expose NaN payloads. No clamping, polar projection,
// input repair, or Euclidean state normalization is performed.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <float.h>
#include <stdint.h>

namespace nhs_biprism_metric {

struct Complex { float re, im; };
struct Mat2 { Complex a00, a01, a10, a11; };
// A Hermitian matrix is represented without a redundant lower triangle.
struct Hermitian2 { float d0, d1; Complex upper; };
// Symmetric tensor in the declared Cartesian coordinate basis.
struct Symmetric3 { float xx, yy, zz, xy, xz, yz; };

enum class Status : uint32_t {
    Ok = 0, NonfiniteInput = 1, NonpositiveMetric = 2,
    IllConditioned = 3, ArithmeticRange = 4, ResidualRejected = 5,
    InvalidMass = 6
};

// Frozen acceptance domain, not a production trajectory threshold. Reciprocal
// Frobenius condition estimates accompany residuals; a small residual alone is
// insufficient. Inputs outside the domain are rejected rather than modified.
constexpr float kMinReciprocalCondition = 1.0e-4f;
constexpr float kMetricRelativeTolerance = 1.0e-6f;
constexpr float kEquationRelativeTolerance = 1.0e-6f;

// A scalar used in a later multiplicative rescaling must be normal. Accepting
// an underflowed intermediate could lose a physically significant product
// before a later large factor restores its scale. This bounded FP32 API rejects
// that domain explicitly rather than silently returning an identity map.
__device__ __forceinline__ bool normal_scale(float x) {
    return isfinite(x) && fabsf(x)>=FLT_MIN;
}

struct ElectronicResult {
    Mat2 u;
    float metric_defect_absolute;
    float metric_defect_relative;
    float equation_residual_relative;
    float reciprocal_condition_s;
    float reciprocal_condition_a;
    Status status;
};
struct MagneticResult {
    float3 velocity;
    float metric_energy_defect_relative;
    float equation_residual_relative;
    float reciprocal_condition_g;
    Status status;
};

__device__ __forceinline__ Complex add(Complex a, Complex b) {
    return {a.re + b.re, a.im + b.im};
}
__device__ __forceinline__ Complex sub(Complex a, Complex b) {
    return {a.re - b.re, a.im - b.im};
}
__device__ __forceinline__ Complex mul(Complex a, Complex b) {
    return {fmaf(a.re,b.re,-a.im*b.im), fmaf(a.re,b.im,a.im*b.re)};
}
__device__ __forceinline__ Complex scale(Complex a, float x) {
    return {a.re*x,a.im*x};
}
__device__ __forceinline__ Complex conj(Complex a) { return {a.re,-a.im}; }
__device__ __forceinline__ float norm2(Complex a) { return fmaf(a.re,a.re,a.im*a.im); }
__device__ __forceinline__ bool finite(Complex a) { return isfinite(a.re) && isfinite(a.im); }
__device__ __forceinline__ bool finite(float3 a) { return isfinite(a.x)&&isfinite(a.y)&&isfinite(a.z); }
__device__ __forceinline__ float maxabs(Complex a) { return fmaxf(fabsf(a.re),fabsf(a.im)); }
__device__ __forceinline__ float maxabs(float3 a) { return fmaxf(fabsf(a.x),fmaxf(fabsf(a.y),fabsf(a.z))); }
__device__ __forceinline__ float maxabs(Mat2 a) {
    return fmaxf(fmaxf(maxabs(a.a00),maxabs(a.a01)),fmaxf(maxabs(a.a10),maxabs(a.a11)));
}
__device__ __forceinline__ Mat2 scale(Mat2 a,float x) {
    return {scale(a.a00,x),scale(a.a01,x),scale(a.a10,x),scale(a.a11,x)};
}
__device__ __forceinline__ Mat2 sub(Mat2 a,Mat2 b) {
    return {sub(a.a00,b.a00),sub(a.a01,b.a01),sub(a.a10,b.a10),sub(a.a11,b.a11)};
}
__device__ __forceinline__ Mat2 mul(Mat2 a,Mat2 b) {
    return {add(mul(a.a00,b.a00),mul(a.a01,b.a10)),
            add(mul(a.a00,b.a01),mul(a.a01,b.a11)),
            add(mul(a.a10,b.a00),mul(a.a11,b.a10)),
            add(mul(a.a10,b.a01),mul(a.a11,b.a11))};
}
__device__ __forceinline__ Mat2 dagger(Mat2 a) { return {conj(a.a00),conj(a.a10),conj(a.a01),conj(a.a11)}; }
__device__ __forceinline__ float frobenius(Mat2 a) {
    float s=maxabs(a);
    if(s==0.0f) return 0.0f;
    if(!isfinite(s)) return CUDART_INF_F;
    a=scale(a,1.0f/s);
    return s*sqrtf(norm2(a.a00)+norm2(a.a01)+norm2(a.a10)+norm2(a.a11));
}
__device__ __forceinline__ bool finite(Mat2 a) { return finite(a.a00)&&finite(a.a01)&&finite(a.a10)&&finite(a.a11); }
__device__ __forceinline__ Mat2 matrix(Hermitian2 a) { return {{a.d0,0.0f},a.upper,conj(a.upper),{a.d1,0.0f}}; }
__device__ __forceinline__ Complex divide(Complex a,Complex b) {
    // Scaling the denominator prevents squaring a tiny determinant.
    float s=maxabs(b);
    Complex q=scale(b,1.0f/s), p=scale(a,1.0f/s);
    return scale(mul(p,conj(q)),1.0f/norm2(q));
}
__device__ __forceinline__ ElectronicResult bad_electronic(Status status) {
    Complex n={CUDART_NAN_F,CUDART_NAN_F};
    return {{n,n,n,n},CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F,status};
}

// U = (S + i dt H/2)^(-1) (S - i dt H/2), in the ORIGINAL
// nonorthogonal coefficient basis, with hbar=1 and compatible H/time units.
// S is fixed HPD and H fixed Hermitian for this map. The mean energy and every
// complex off-diagonal component participate. U is S-unitary, not generally
// Euclidean-unitary. Time-dependent S needs a separately specified transport.
__device__ __forceinline__ ElectronicResult electronic_cayley(
        Hermitian2 s_input,Hermitian2 h_input,float dt) {
    Mat2 s=matrix(s_input), h=matrix(h_input);
    if(!finite(s)||!finite(h)||!isfinite(dt)) return bad_electronic(Status::NonfiniteInput);
    float ss=maxabs(s);
    if(!(ss>0.0f)) return bad_electronic(Status::NonpositiveMetric);
    Mat2 sn=scale(s,1.0f/ss);
    if(!finite(sn)) return bad_electronic(Status::ArithmeticRange);
    float ds=fmaf(sn.a00.re,sn.a11.re,-norm2(sn.a01));
    if(!(sn.a00.re>0.0f&&sn.a11.re>0.0f&&ds>0.0f)) return bad_electronic(Status::NonpositiveMetric);
    float snorm=frobenius(sn), rcs=ds/(snorm*snorm);
    if(!(rcs>=kMinReciprocalCondition)) return bad_electronic(Status::IllConditioned);
    if(dt==0.0f) return {{{1.0f,0.0f},{0.0f,0.0f},{0.0f,0.0f},{1.0f,0.0f}},0.0f,0.0f,0.0f,rcs,rcs,Status::Ok};
    float half_dt=0.5f*dt;
    float hs=maxabs(h), ah=fabsf(half_dt)*hs;
    if(!normal_scale(half_dt)||(hs>0.0f&&!normal_scale(ah))) return bad_electronic(Status::ArithmeticRange);
    float as=fmaxf(ss,ah);
    float hcoef=ah/as;
    if(hs>0.0f&&!normal_scale(hcoef)) return bad_electronic(Status::ArithmeticRange);
    Mat2 sscaled=scale(s,1.0f/as);
    Mat2 hscaled=hs==0.0f ? h : scale(scale(h,1.0f/hs),copysignf(hcoef,dt));
    Mat2 a={add(sscaled.a00,{-hscaled.a00.im,hscaled.a00.re}),
            add(sscaled.a01,{-hscaled.a01.im,hscaled.a01.re}),
            add(sscaled.a10,{-hscaled.a10.im,hscaled.a10.re}),
            add(sscaled.a11,{-hscaled.a11.im,hscaled.a11.re})};
    Mat2 b={sub(sscaled.a00,{-hscaled.a00.im,hscaled.a00.re}),
            sub(sscaled.a01,{-hscaled.a01.im,hscaled.a01.re}),
            sub(sscaled.a10,{-hscaled.a10.im,hscaled.a10.re}),
            sub(sscaled.a11,{-hscaled.a11.im,hscaled.a11.re})};
    if(!finite(a)||!finite(b)) return bad_electronic(Status::ArithmeticRange);
    Complex det=sub(mul(a.a00,a.a11),mul(a.a01,a.a10));
    float anorm=frobenius(a);
    float rca=sqrtf(norm2(det))/(anorm*anorm);
    if(!(rca>=kMinReciprocalCondition)) return bad_electronic(Status::IllConditioned);
    Mat2 u={divide(sub(mul(a.a11,b.a00),mul(a.a01,b.a10)),det),
            divide(sub(mul(a.a11,b.a01),mul(a.a01,b.a11)),det),
            divide(sub(mul(a.a00,b.a10),mul(a.a10,b.a00)),det),
            divide(sub(mul(a.a00,b.a11),mul(a.a10,b.a01)),det)};
    if(!finite(u)) return bad_electronic(Status::ArithmeticRange);
    float md=frobenius(sub(mul(dagger(u),mul(sn,u)),sn));
    float mr=md/snorm;
    float er=frobenius(sub(mul(a,u),b))/(anorm*frobenius(u)+frobenius(b));
    float ma=md*ss;
    if(!isfinite(mr)||!isfinite(er)||!isfinite(ma)) return bad_electronic(Status::ArithmeticRange);
    if(mr>kMetricRelativeTolerance||er>kEquationRelativeTolerance) {
        auto result=bad_electronic(Status::ResidualRejected);
        result.metric_defect_absolute=ma; result.metric_defect_relative=mr;
        result.equation_residual_relative=er; result.reciprocal_condition_s=rcs;
        result.reciprocal_condition_a=rca; return result;
    }
    return {u,ma,mr,er,rcs,rca,Status::Ok};
}

__device__ __forceinline__ float dot(float3 a,float3 b) { return fmaf(a.x,b.x,fmaf(a.y,b.y,a.z*b.z)); }
__device__ __forceinline__ float3 scale(float3 a,float s) { return {a.x*s,a.y*s,a.z*s}; }
__device__ __forceinline__ float3 add(float3 a,float3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__device__ __forceinline__ float3 sub(float3 a,float3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ __forceinline__ float3 cross(float3 a,float3 b) {
    return {fmaf(a.y,b.z,-a.z*b.y),fmaf(a.z,b.x,-a.x*b.z),fmaf(a.x,b.y,-a.y*b.x)};
}
__device__ __forceinline__ float3 mul(Symmetric3 a,float3 v) {
    return {fmaf(a.xx,v.x,fmaf(a.xy,v.y,a.xz*v.z)),
            fmaf(a.xy,v.x,fmaf(a.yy,v.y,a.yz*v.z)),
            fmaf(a.xz,v.x,fmaf(a.yz,v.y,a.zz*v.z))};
}
__device__ __forceinline__ float frobenius(Symmetric3 a) {
    return sqrtf(a.xx*a.xx+a.yy*a.yy+a.zz*a.zz+2.0f*(a.xy*a.xy+a.xz*a.xz+a.yz*a.yz));
}
__device__ __forceinline__ MagneticResult bad_magnetic(Status status) {
    return {{CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F},CUDART_NAN_F,CUDART_NAN_F,CUDART_NAN_F,status};
}

// Solve m g(v+ - v-)/dt = ((v+ + v-)/2) x B, with frozen SPD
// full 3x3 g. C_B v = v x B fixes the rotation sign. Cholesky whitening
// y=L^T v converts the skew congruence to C_{L^T B / det L}; transforming
// velocity as well as the field is essential. A scaled rational rotation
// avoids squaring a large Berry field. This is a frozen magnetic subflow,
// not a claim that Boris is symplectic for arbitrary moving g or B(x).
__device__ __forceinline__ MagneticResult magnetic_cayley(
        Symmetric3 input,float3 b,float3 velocity,float mass,float dt) {
    float gs=fmaxf(fmaxf(fabsf(input.xx),fabsf(input.yy)),fmaxf(fabsf(input.zz),
             fmaxf(fabsf(input.xy),fmaxf(fabsf(input.xz),fabsf(input.yz)))));
    if(!isfinite(input.xx)||!isfinite(input.yy)||!isfinite(input.zz)||
       !isfinite(input.xy)||!isfinite(input.xz)||!isfinite(input.yz)||
       !finite(b)||!finite(velocity)||!isfinite(mass)||!isfinite(dt)) return bad_magnetic(Status::NonfiniteInput);
    if(!(mass>0.0f)) return bad_magnetic(Status::InvalidMass);
    if(!(gs>0.0f)) return bad_magnetic(Status::NonpositiveMetric);
    Symmetric3 g={input.xx/gs,input.yy/gs,input.zz/gs,input.xy/gs,input.xz/gs,input.yz/gs};
    if(!(g.xx>0.0f)) return bad_magnetic(Status::NonpositiveMetric);
    float l00=sqrtf(g.xx),l10=g.xy/l00,l20=g.xz/l00;
    float d11=fmaf(-l10,l10,g.yy);
    if(!(d11>0.0f)) return bad_magnetic(Status::NonpositiveMetric);
    float l11=sqrtf(d11),l21=fmaf(-l20,l10,g.yz)/l11;
    float d22=fmaf(-l21,l21,fmaf(-l20,l20,g.zz));
    if(!(d22>0.0f)) return bad_magnetic(Status::NonpositiveMetric);
    float l22=sqrtf(d22),ldet=l00*l11*l22;
    float j00=1.0f/l00,j11=1.0f/l11,j22=1.0f/l22;
    float j10=-l10*j00*j11,j21=-l21*j11*j22;
    float j20=fmaf(l21,l10,-l20*l11)/ldet;
    Symmetric3 gi={j00*j00+j10*j10+j20*j20,j11*j11+j21*j21,j22*j22,
                  fmaf(j10,j11,j20*j21),j20*j22,j21*j22};
    float gn=frobenius(g),rcg=1.0f/(gn*frobenius(gi));
    if(!isfinite(rcg)) return bad_magnetic(Status::ArithmeticRange);
    if(!(rcg>=kMinReciprocalCondition)) return bad_magnetic(Status::IllConditioned);
    float vs=maxabs(velocity),bs=maxabs(b);
    if(dt==0.0f||bs==0.0f||vs==0.0f) return {velocity,0.0f,0.0f,rcg,Status::Ok};
    float3 v=scale(velocity,1.0f/vs),bn=scale(b,1.0f/bs);
    if(!finite(v)||!finite(bn)) return bad_magnetic(Status::ArithmeticRange);
    float3 y={fmaf(l00,v.x,fmaf(l10,v.y,l20*v.z)),fmaf(l11,v.y,l21*v.z),l22*v.z};
    float half_dt=0.5f*dt,half_dt_over_mass=half_dt/mass,field_scale=bs/gs;
    if(!normal_scale(half_dt)||!normal_scale(half_dt_over_mass)||!normal_scale(field_scale)) return bad_magnetic(Status::ArithmeticRange);
    float hc=half_dt_over_mass*field_scale;
    if(!normal_scale(hc)) return bad_magnetic(Status::ArithmeticRange);
    float3 bt={fmaf(l00,bn.x,fmaf(l10,bn.y,l20*bn.z)),fmaf(l11,bn.y,l21*bn.z),l22*bn.z};
    float3 t=scale(bt,hc/ldet);
    if(!finite(t)||!isfinite(hc)) return bad_magnetic(Status::ArithmeticRange);
    float ts=maxabs(t);
    float3 yp=y;
    if(ts>0.0f) {
        float3 q=scale(t,1.0f/ts);
        float q2=dot(q,q),qn=sqrtf(q2);
        float3 axis=scale(q,1.0f/qn);
        float c,s;
        if(ts>1.0f) {
            float it=1.0f/ts,den=fmaf(it,it,q2);
            c=(it*it-q2)/den; s=2.0f*it*qn/den;
        } else {
            float tm=ts*qn,den=fmaf(tm,tm,1.0f);
            c=(1.0f-tm*tm)/den; s=2.0f*tm/den;
        }
        float3 parallel=scale(axis,dot(axis,y));
        yp=add(parallel,add(scale(sub(y,parallel),c),scale(cross(y,axis),s)));
    }
    float3 vp;
    vp.z=yp.z/l22;
    vp.y=fmaf(-l21,vp.z,yp.y)/l11;
    vp.x=fmaf(-l10,vp.y,fmaf(-l20,vp.z,yp.x))/l00;
    float3 result=scale(vp,vs);
    if(!finite(vp)||!finite(result)) return bad_magnetic(Status::ArithmeticRange);
    float e0=dot(v,mul(g,v)),e1=dot(vp,mul(g,vp));
    float mr=fabsf(e1-e0)/e0;
    float eqscale=fmaxf(1.0f,fabsf(hc));
    float gcoef=1.0f/eqscale,bcoef=hc/eqscale;
    float3 residual=sub(scale(mul(g,sub(vp,v)),gcoef),scale(cross(add(vp,v),bn),bcoef));
    float er=sqrtf(dot(residual,residual))/
        ((gn*gcoef+sqrtf(2.0f*dot(bn,bn))*fabsf(bcoef))*(sqrtf(dot(vp,vp))+sqrtf(dot(v,v))));
    if(!isfinite(mr)||!isfinite(er)) return bad_magnetic(Status::ArithmeticRange);
    if(mr>kMetricRelativeTolerance||er>kEquationRelativeTolerance) {
        auto bad=bad_magnetic(Status::ResidualRejected);
        bad.metric_energy_defect_relative=mr; bad.equation_residual_relative=er;
        bad.reciprocal_condition_g=rcg; return bad;
    }
    return {result,mr,er,rcg,Status::Ok};
}

} // namespace nhs_biprism_metric
