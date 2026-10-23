#pragma once
#include <math.h>

#ifdef __CUDACC__
#define BP_NC_INLINE __host__ __device__ __forceinline__
#else
#define BP_NC_INLINE inline
#endif

// Precision realization of (G-C_tau) v_plus = (G+C_tau) v_minus,
// C_tau v = v cross tau. G is the same represented covariance used by
// the force/energy owners. No normalization, metric floor, or work credit.
// This small host/device primitive is checked against an independent exact
// rational 3x3 solve; integrated acceptance still belongs to C3/C4.
namespace biprism_nuclear_f64 {
BP_NC_INLINE double madd(double a,double b,double c) {
#ifdef __CUDA_ARCH__
    return __fma_rn(a,b,c);
#else
    return fma(a,b,c);
#endif
}
template<class V> BP_NC_INLINE V cross(V a,V b) {
    return {madd(a.y,b.z,-a.z*b.y),madd(a.z,b.x,-a.x*b.z),madd(a.x,b.y,-a.y*b.x)};
}
template<class V> BP_NC_INLINE V multiply(double xx,double yy,double zz,double xy,double xz,double yz,V v) {
    return {madd(xx,v.x,madd(xy,v.y,xz*v.z)),
            madd(xy,v.x,madd(yy,v.y,yz*v.z)),
            madd(xz,v.x,madd(yz,v.y,zz*v.z))};
}
template<class G,class V> BP_NC_INLINE V cayley(G g,V velocity,V tau) {
    const double scale=fmax(double(g.xx),fmax(double(g.yy),double(g.zz)));
    if(!(scale>0.)||!isfinite(scale))return {NAN,NAN,NAN};
    const double inv=1./scale;
    const double sx=sqrt(double(g.xx)*inv),sy=sqrt(double(g.yy)*inv),sz=sqrt(double(g.zz)*inv);
    const double xy=((double(g.xy)*inv)/sx)/sy;
    const double xz=((double(g.xz)*inv)/sx)/sz;
    const double yz=((double(g.yz)*inv)/sy)/sz;
    const double xx=madd(-yz,yz,1.),yy=madd(-xz,xz,1.),zz=madd(-xy,xy,1.);
    const double cxy=madd(xz,yz,-xy),cxz=madd(xy,yz,-xz),cyz=madd(xy,xz,-yz);
    const double det=madd(xy,cxy,madd(xz,cxz,xx));
    if(!(det>0.)||!isfinite(det))return {NAN,NAN,NAN};
    const double id=1./det;
    const V t={((tau.x*inv)/sy)/sz,((tau.y*inv)/sx)/sz,((tau.z*inv)/sx)/sy};
    const V v={sx*velocity.x,sy*velocity.y,sz*velocity.z};
    const V w1=multiply(xx*id,yy*id,zz*id,cxy*id,cxz*id,cyz*id,cross(v,t));
    const V w2=multiply(xx*id,yy*id,zz*id,cxy*id,cxz*id,cyz*id,cross(w1,t));
    const V gt=multiply(1.,1.,1.,xy,xz,yz,t);
    const double omega2=id*madd(t.x,gt.x,madd(t.y,gt.y,t.z*gt.z));
    const double factor=2./(1.+omega2);
    return {madd(factor,(w1.x+w2.x)/sx,velocity.x),
            madd(factor,(w1.y+w2.y)/sy,velocity.y),
            madd(factor,(w1.z+w2.z)/sz,velocity.z)};
}
} // namespace biprism_nuclear_f64
#undef BP_NC_INLINE
