#pragma once
#include <stdint.h>
#include <math.h>

// Numerical history only. These planes never own a physical endpoint, force,
// work credit, or acceptance residual. They are separate from all C1--C4 data.
struct BiprismNuclearSecantFfi {
    double dx[4][3][192];
    double dr[4][3][192];
    double alpha_dx[4],alpha_dr[4];
    double gram[4][4],rhs[4],gamma[4];
};
static_assert(sizeof(BiprismNuclearSecantFfi)==37120,"Four-column numerical history ABI");

#if defined(__CUDACC__)
#define BP_ITER_HD __host__ __device__ __forceinline__
#else
#define BP_ITER_HD inline
#endif
// Regularized normal equation for the bounded four-column least-squares
// predictor. Regularization changes only the next guess; raw C3/C4 equations
// decide acceptance independently. Storage is caller-owned, never a local array.
BP_ITER_HD bool bp_nuclear_secant_solve(volatile BiprismNuclearSecantFfi* w,uint32_t n) {
    if(n>4u)return false;
    for(uint32_t i=0u;i<4u;++i)w->gamma[i]=0.;
    if(n==0u)return true;
    double scale=0.;
    for(uint32_t i=0u;i<n;++i){
        if(!isfinite(w->rhs[i])||!isfinite(w->gram[i][i])||w->gram[i][i]<0.)return false;
        scale=fmax(scale,w->gram[i][i]);
    }
    if(!(scale>0.))return true;
    for(uint32_t i=0u;i<n;++i)w->gram[i][i]+=scale*1e-12;
    for(uint32_t i=0u;i<n;++i){
        const double pivot=w->gram[i][i];
        if(!isfinite(pivot)||!(pivot>0.))return false;
        for(uint32_t j=i+1u;j<n;++j){
            const double factor=w->gram[j][i]/pivot;
            for(uint32_t k=i+1u;k<n;++k)w->gram[j][k]-=factor*w->gram[i][k];
            w->rhs[j]-=factor*w->rhs[i];
        }
    }
    for(int i=int(n)-1;i>=0;--i){
        double value=w->rhs[i];
        for(uint32_t j=uint32_t(i)+1u;j<n;++j)value-=w->gram[i][j]*w->gamma[j];
        w->gamma[i]=value/w->gram[i][i];
        if(!isfinite(w->gamma[i]))return false;
    }
    return true;
}

// Restrict a rejected numerical displacement along its actual observed
// residual difference. This is a next-trial model only; the complete physical
// equations must be evaluated again at the proposed point.
BP_ITER_HD double bp_nuclear_backtrack_factor(double base_norm2,double trial_norm2,double cross) {
    if(!isfinite(base_norm2)||!isfinite(trial_norm2)||!isfinite(cross))return .5;
    const double difference=trial_norm2-base_norm2-2.*cross;
    if(!(difference>1e-24))return .5;
    const double factor=-cross/difference;
    if(!isfinite(factor)||factor==0.)return .5;
    return fmax(-.5,fmin(.5,factor));
}

// Leaf-local coordinate-work relaxation. Nuclear/electronic state changes
// between calls, so historical signs cannot bracket the CURRENT functional.
// History may reduce the current residual step, never anchor it to an old root.
// Kinetic/Berry defects remain absent from this numerical proposal.
struct BiprismCoordinateRootFfi {
    double left_alpha,left_residual,right_alpha,right_residual;
    uint32_t flags,reserved;
};
static_assert(sizeof(BiprismCoordinateRootFfi)==40,"Coordinate root history");
BP_ITER_HD double bp_coordinate_root_next(volatile BiprismCoordinateRootFfi* w,
    double alpha,double residual,double dot,bool late) {
    if(!isfinite(alpha)||!isfinite(residual)||!isfinite(dot)||!(dot>1e-24))return alpha;
    double damping=.5;
    if(late&&w->flags){
        const double da=alpha-w->left_alpha,dr=residual-w->left_residual;
        // A secant from changing states is only a damping estimate. In
        // particular, a near-zero dr never divides an unrestricted update.
        if(da!=0.&&dr*da<0.){
            const double estimated=.8*dot*fabs(da/dr);
            if(isfinite(estimated))damping=fmax(.02,fmin(damping,estimated));
        }
    }
    w->left_alpha=alpha;w->left_residual=residual;w->flags=late?1u:0u;
    return alpha+fmin(.25,fmax(-.25,damping*residual/dot));
}

// Propose a represented endpoint component, using the exact scalar quadratic
// K(v+d e_i)-K(v) = linear*d + quadratic*d*d. This chooses representation
// within the EXISTING momentum-reconstruction bound; it never modifies force,
// midpoint, position, or a work ledger. The caller evaluates actual FP32 bits
// and recomputes the complete kinetic energy before any acceptance decision.
BP_ITER_HD double bp_endpoint_rounding_target(double current,double raw,
    double linear,double quadratic,double half,double coordinate_defect) {
    if(!isfinite(current)||!isfinite(raw)||!isfinite(linear)||!isfinite(quadratic)
        ||!(quadratic>0.)||!isfinite(half)||!(half>0.)||!isfinite(coordinate_defect))return NAN;
    const double disc=fma(-4.*quadratic,coordinate_defect,linear*linear);
    double delta=0.;
    if(disc>=0.){
        const double den=linear+copysign(sqrt(disc),linear);
        if(den!=0.)delta=-2.*coordinate_defect/den;
    }else delta=-.5*linear/quadratic;
    // One percent interior margin protects the independent matrix
    // reconstruction (which is evaluated, not replaced by this scalar bound).
    const double radius=.99e-7/half;
    return fmax(raw-radius,fmin(raw+radius,current+delta));
}
#undef BP_ITER_HD
