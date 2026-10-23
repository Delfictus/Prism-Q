#pragma once
#include "nhs_biprism_electronic_discrete.cuh"
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// BPS1 is the physical electronic owner. The tile's FP32 spinor is a checked
// compatibility projection only. Neither a potential/operator refresh nor an
// import reconstructs this owner from that projection. Cold initialization
// promotes the original seed values exactly once, without normalization.
using BiprismSpinor64 = BiprismElectronicSpinor<double>;
static constexpr uint32_t BP_SPINOR_MAGIC=0x31535042u;
static constexpr uint32_t BP_SPINOR_VERSION=1u;
static constexpr uint32_t BP_SPINOR_CORE=192u;
struct alignas(128) BiprismSpinorOwnerFfi {
    uint32_t magic,version,leaf,state; // state0=cold allocation,1=accepted main state
    uint64_t allocation_generation,topology_generation;
    uint64_t tile_address,coordinate_owner_address;
    uint32_t accepted_committed_substeps,accepted_step;
    uint64_t reserved[9];
    BiprismSpinor64 accepted[BP_SPINOR_CORE];
    // Exclusive invocation work planes, never physical commit authority.
    // Start advances between admitted inner substeps; candidate is the exact
    // represented C1 endpoint. A rejected outer invocation changes neither
    // accepted[] nor its identity. No plane aliases any numerical scratch.
    BiprismSpinor64 substep_start[BP_SPINOR_CORE];
    BiprismSpinor64 candidate[BP_SPINOR_CORE];
    BiprismElectronicComplex<double> proposed_s0[BP_SPINOR_CORE];
};
static_assert(sizeof(BiprismSpinor64)==32,"native binary64 complex pair");
static_assert(offsetof(BiprismSpinorOwnerFfi,accepted)==128,"BPS1 accepted offset");
static_assert(offsetof(BiprismSpinorOwnerFfi,substep_start)==6272,"BPS1 start offset");
static_assert(offsetof(BiprismSpinorOwnerFfi,candidate)==12416,"BPS1 candidate offset");
static_assert(sizeof(BiprismSpinorOwnerFfi)==21632&&alignof(BiprismSpinorOwnerFfi)==128,"BPS1 owner extent");
#if defined(__CUDACC__)
#define BP_SPINOR_HD __host__ __device__ inline
#else
#define BP_SPINOR_HD inline
#endif
BP_SPINOR_HD double bp_spinor_norm(BiprismSpinor64 p) {
    return bp_ed_norm2(p.s0)+bp_ed_norm2(p.s1);
}
BP_SPINOR_HD bool bp_spinor_finite(BiprismSpinor64 p) {
    return isfinite(p.s0.re)&&isfinite(p.s0.im)&&isfinite(p.s1.re)&&isfinite(p.s1.im);
}
BP_SPINOR_HD bool bp_spinor_identity(const BiprismSpinorOwnerFfi* o,uint32_t leaf,
    const void* tile,const void* coordinate,uint64_t allocation,uint64_t topology) {
    if(!o||(reinterpret_cast<uintptr_t>(o)&127u)||leaf>=8u||!allocation||!topology
        ||o->magic!=BP_SPINOR_MAGIC||o->version!=BP_SPINOR_VERSION||o->leaf!=leaf
        ||o->tile_address!=uint64_t(reinterpret_cast<uintptr_t>(tile))
        ||o->coordinate_owner_address!=uint64_t(reinterpret_cast<uintptr_t>(coordinate))
        ||o->allocation_generation!=allocation||o->topology_generation!=topology)return false;
    for(uint32_t k=0;k<9u;++k)if(o->reserved[k])return false;
    return true;
}
BP_SPINOR_HD bool bp_spinor_owner_valid(const BiprismSpinorOwnerFfi* o,uint32_t leaf,
    const void* tile,const void* coordinate,uint64_t allocation,uint64_t topology,
    uint32_t committed,uint32_t step) {
    return bp_spinor_identity(o,leaf,tile,coordinate,allocation,topology)&&o->state==1u
        &&o->accepted_committed_substeps==committed&&o->accepted_step==step;
}
BP_SPINOR_HD bool bp_spinor_projection_component(double value,float projection) {
    const float rounded=float(value);
    if(!isfinite(value)||!isfinite(rounded))return false;
#if defined(__CUDA_ARCH__)
    return __float_as_uint(rounded)==__float_as_uint(projection);
#else
    uint32_t a,b;memcpy(&a,&rounded,4);memcpy(&b,&projection,4);return a==b;
#endif
}
BP_SPINOR_HD bool bp_spinor_projection(BiprismSpinor64 p,float a,float b,float c,float d) {
    return bp_spinor_projection_component(p.s0.re,a)&&bp_spinor_projection_component(p.s0.im,b)
        &&bp_spinor_projection_component(p.s1.re,c)&&bp_spinor_projection_component(p.s1.im,d);
}
// Data writes and the existing all-eight acceptance/barrier precede this
// metadata publication. This is not a clock increment or new acceptance gate.
BP_SPINOR_HD void bp_spinor_publish(BiprismSpinorOwnerFfi& o,uint32_t committed,uint32_t step) {
    o.accepted_committed_substeps=committed;o.accepted_step=step;o.state=1u;
}
#undef BP_SPINOR_HD
