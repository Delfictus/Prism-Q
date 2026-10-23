#pragma once
#include <cuda_runtime.h>
#include <math.h>
#include "nhs_biprism_coordinate_precision.cuh"

// One radial realization for compensated Phase2, main, and their work owners.
// Constants retain the original represented FP32 values. Geometry remains FP64
// until the declared coefficient/gradient storage boundary.
__host__ __device__ __forceinline__ double bp_geometry_fma(double a,double b,double c) {
#ifdef __CUDA_ARCH__
    return __fma_rn(a,b,c);
#else
    return fma(a,b,c);
#endif
}
__host__ __device__ __forceinline__ double bp_geometry_length_squared(BiprismCoordinate3F64 e) {
    return bp_geometry_fma(e.x,e.x,bp_geometry_fma(e.y,e.y,e.z*e.z))+double(1e-12f);
}
__host__ __device__ __forceinline__ double bp_geometry_live_weight_f64(
    float base,BiprismCoordinate3F64 current,BiprismCoordinate3F64 reference
) {
    const double r=sqrt(bp_geometry_length_squared(current));
    const double r0=sqrt(bp_geometry_length_squared(reference));
    return double(fmaxf(base,1e-4f))*(r0/r);
}
// Explicit compatibility projection for legacy geometry/observation fields.
// Coupled DEC, energy and operator-work consumers use the full result above.
__host__ __device__ __forceinline__ float bp_geometry_live_weight(
    float base,BiprismCoordinate3F64 current,BiprismCoordinate3F64 reference
) { return float(bp_geometry_live_weight_f64(base,current,reference)); }
__host__ __device__ __forceinline__ float3 bp_geometry_live_weight_gradient(
    float live_weight,BiprismCoordinate3F64 current
) {
    const double factor=double(live_weight)/bp_geometry_length_squared(current);
    return {float(factor*current.x),float(factor*current.y),float(factor*current.z)};
}
