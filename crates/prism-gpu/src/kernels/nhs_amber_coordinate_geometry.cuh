#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>
#include <type_traits>
#include "nhs_biprism_coordinate_precision.cuh"

// Only the explicitly selected compensated exports use this fault. The sink
// is the leaf's retained V10 terminal_fault_latch, bound by the graph owner.
constexpr uint32_t BP_NATIVE_COORDINATE_FAULT = 0x00010000u;

__device__ __forceinline__ bool bp_native_coordinate_descriptor_aligned(
    const BiprismCoordinateOwnerFfi* owner
) {
    return owner && (reinterpret_cast<uintptr_t>(owner) &
        (alignof(BiprismCoordinateOwnerFfi)-1u)) == 0u;
}

__device__ __forceinline__ void bp_native_coordinate_fault(uint32_t* fault) {
    if (fault && (reinterpret_cast<uintptr_t>(fault) & (alignof(uint32_t)-1u)) == 0u)
        atomicOr(fault, BP_NATIVE_COORDINATE_FAULT);
    else asm volatile("trap;"); // A missing acceptance owner is never a silent return.
}

__device__ __forceinline__ bool bp_native_coordinate_owner_valid(
    const BiprismCoordinateOwnerFfi* owner, const float* current_hi, int atoms,
    uint32_t required, uint32_t* fault, const float* start_hi = nullptr
) {
    // Leaf/generation are independently bound by the host allocation owner.
    // The native call additionally binds the exact live high-plane pointer.
    const bool ok = fault && (reinterpret_cast<uintptr_t>(fault) & (alignof(uint32_t)-1u)) == 0u &&
        bp_native_coordinate_descriptor_aligned(owner) && atoms > 0 &&
        bp_coord_owner_valid(owner, owner->leaf_id, (uint32_t)atoms,
            owner->generation, 1u, required, current_hi, start_hi);
    if (!ok) bp_native_coordinate_fault(fault);
    return ok && atomicAdd(fault, 0u) == 0u;
}

template<class V> struct BpNativeGeometryVector;
template<> struct BpNativeGeometryVector<float3> { using Real = float; };
template<> struct BpNativeGeometryVector<double3> { using Real = double; };
template<class V> __device__ __forceinline__ V bp_native_vec(
    typename BpNativeGeometryVector<V>::Real x,
    typename BpNativeGeometryVector<V>::Real y,
    typename BpNativeGeometryVector<V>::Real z) { return {x,y,z}; }

// These float overloads retain the original native arithmetic. Only the
// explicitly selected double geometry instantiation uses the double versions.
__device__ __forceinline__ float bp_native_sqrt(float x) { return sqrtf(x); }
__device__ __forceinline__ double bp_native_sqrt(double x) { return sqrt(x); }
__device__ __forceinline__ float bp_native_rsqrt(float x) { return rsqrtf(x); }
__device__ __forceinline__ double bp_native_rsqrt(double x) { return 1.0/sqrt(x); }
__device__ __forceinline__ float bp_native_acos(float x) { return acosf(x); }
__device__ __forceinline__ double bp_native_acos(double x) { return acos(x); }
__device__ __forceinline__ float bp_native_asin(float x) { return asinf(x); }
__device__ __forceinline__ double bp_native_asin(double x) { return asin(x); }
__device__ __forceinline__ float bp_native_sin(float x) { return sinf(x); }
__device__ __forceinline__ double bp_native_sin(double x) { return sin(x); }
__device__ __forceinline__ float bp_native_abs(float x) { return fabsf(x); }
__device__ __forceinline__ double bp_native_abs(double x) { return fabs(x); }
__device__ __forceinline__ float bp_native_nearbyint(float x) { return nearbyintf(x); }
__device__ __forceinline__ double bp_native_nearbyint(double x) { return nearbyint(x); }
template<class A,class B> __device__ __forceinline__ auto bp_native_max(A a,B b) -> decltype(a+b) {
    using R = decltype(a+b);
    if constexpr(std::is_same<R,float>::value) return fmaxf(a,b);
    else return fmax((double)a,(double)b);
}
template<class A,class B> __device__ __forceinline__ auto bp_native_min(A a,B b) -> decltype(a+b) {
    using R = decltype(a+b);
    if constexpr(std::is_same<R,float>::value) return fminf(a,b);
    else return fmin((double)a,(double)b);
}

template<bool Compensated> struct BpNativeGeometry;
template<> struct BpNativeGeometry<false> {
    using Real = float; using Vec = float3;
    __device__ static __forceinline__ Vec load(const float3* x,int i,
        const BiprismCoordinateOwnerFfi*,uint32_t*) { return x[i]; }
    __device__ static __forceinline__ Vec make(Real x,Real y,Real z) { return make_float3(x,y,z); }
    __device__ static __forceinline__ Vec load_plane(const float3* x,int i,
        const BiprismCoordinateOwnerFfi*,BiprismCoordinatePlane,uint32_t*) { return x[i]; }
    __device__ static __forceinline__ Vec difference(const float3* x,int a,int b,
        const BiprismCoordinateOwnerFfi*,BiprismCoordinatePlane,uint32_t*) {
        return {x[a].x-x[b].x,x[a].y-x[b].y,x[a].z-x[b].z};
    }
    __device__ static __forceinline__ bool add_current(float3* x,int i,Vec d,
        const BiprismCoordinateOwnerFfi*,uint32_t*) {
        x[i].x+=d.x;x[i].y+=d.y;x[i].z+=d.z;return true;
    }
    __device__ static __forceinline__ bool snapshot(float3* start,const float3* x,int i,
        const BiprismCoordinateOwnerFfi*,uint32_t*) { start[i]=x[i];return true; }
    __device__ static __forceinline__ bool drift(float3* x,int i,float dt,float3 v,
        const BiprismCoordinateOwnerFfi*,uint32_t*) {
        x[i].x+=dt*v.x;x[i].y+=dt*v.y;x[i].z+=dt*v.z;return true;
    }
};
template<> struct BpNativeGeometry<true> {
    using Real = double; using Vec = double3;
    __device__ static __forceinline__ Vec load(const float3*,int i,
        const BiprismCoordinateOwnerFfi* owner,uint32_t* fault) {
        if (!owner || i < 0 || (uint32_t)i >= owner->atom_count) {
            bp_native_coordinate_fault(fault); return {NAN,NAN,NAN};
        }
        const auto pair = bp_coord_load(*owner,BiprismCoordinatePlane::Current,(uint32_t)i);
        if (!bp_coord_finite(pair)) {
            bp_native_coordinate_fault(fault); return {NAN,NAN,NAN};
        }
        const auto value = bp_coord_value(pair);
        return {value.x,value.y,value.z};
    }
    __device__ static __forceinline__ Vec make(Real x,Real y,Real z) { return make_double3(x,y,z); }
    __device__ static __forceinline__ Vec load_plane(const float3*,int i,
        const BiprismCoordinateOwnerFfi* owner,BiprismCoordinatePlane plane,uint32_t* fault) {
        if(!owner||i<0||(uint32_t)i>=owner->atom_count) {
            bp_native_coordinate_fault(fault);return {NAN,NAN,NAN};
        }
        const auto pair=bp_coord_load(*owner,plane,(uint32_t)i);
        if(!bp_coord_finite(pair)){bp_native_coordinate_fault(fault);return {NAN,NAN,NAN};}
        const auto v=bp_coord_value(pair);return {v.x,v.y,v.z};
    }
    __device__ static __forceinline__ Vec difference(const float3*,int a,int b,
        const BiprismCoordinateOwnerFfi* owner,BiprismCoordinatePlane plane,uint32_t* fault) {
        if(!owner||a<0||b<0||(uint32_t)a>=owner->atom_count||(uint32_t)b>=owner->atom_count) {
            bp_native_coordinate_fault(fault);return {NAN,NAN,NAN};
        }
        const auto pa=bp_coord_load(*owner,plane,(uint32_t)a);
        const auto pb=bp_coord_load(*owner,plane,(uint32_t)b);
        if(!bp_coord_finite(pa)||!bp_coord_finite(pb)) {
            bp_native_coordinate_fault(fault);return {NAN,NAN,NAN};
        }
        const auto d=bp_coord_difference(pa,pb);return {d.x,d.y,d.z};
    }
    __device__ static __forceinline__ bool add_current(float3*,int i,Vec d,
        const BiprismCoordinateOwnerFfi* owner,uint32_t* fault) {
        const auto old=bp_coord_load(*owner,BiprismCoordinatePlane::Current,(uint32_t)i);
        const auto next=bp_coord_add(old,BiprismCoordinate3F64{d.x,d.y,d.z});
        const bool ok=bp_coord_store_current(*owner,(uint32_t)i,next);
        if(!ok)bp_native_coordinate_fault(fault);return ok;
    }
    __device__ static __forceinline__ bool snapshot(float3*,const float3*,int i,
        const BiprismCoordinateOwnerFfi* owner,uint32_t* fault) {
        const auto p=bp_coord_load(*owner,BiprismCoordinatePlane::Current,(uint32_t)i);
        if(!bp_coord_finite(p)){bp_native_coordinate_fault(fault);return false;}
        const size_t k=(size_t)i*3u;
        owner->step_start_hi[k]=p.x.hi;owner->step_start_lo[k]=p.x.lo;
        owner->step_start_hi[k+1u]=p.y.hi;owner->step_start_lo[k+1u]=p.y.lo;
        owner->step_start_hi[k+2u]=p.z.hi;owner->step_start_lo[k+2u]=p.z.lo;
        return true;
    }
    __device__ static __forceinline__ bool drift(float3*,int i,float dt,float3 v,
        const BiprismCoordinateOwnerFfi* owner,uint32_t* fault) {
        const auto p=bp_coord_load(*owner,BiprismCoordinatePlane::Current,(uint32_t)i);
        const BiprismCoordinateExpansion3 next={bp_coord_fma(p.x,(double)dt,(double)v.x),
            bp_coord_fma(p.y,(double)dt,(double)v.y),bp_coord_fma(p.z,(double)dt,(double)v.z)};
        const bool ok=bp_coord_store_current(*owner,(uint32_t)i,next);
        if(!ok)bp_native_coordinate_fault(fault);return ok;
    }
};
