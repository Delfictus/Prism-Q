#pragma once
#include "nhs_biprism_coordinate_precision.cuh"
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

// Full represented DEC potential is retained separately from the FP32 tile
// compatibility projection. A prepared operator never overwrites the last
// accepted bank. Physical rollback remains outside this descriptor's scope.
static constexpr uint32_t BP_POTENTIAL_MAGIC=0x31505042u; // BPP1
static constexpr uint32_t BP_POTENTIAL_VERSION=1u;
static constexpr uint32_t BP_POTENTIAL_COLD=0u;
static constexpr uint32_t BP_POTENTIAL_ACCEPTED=1u;
static constexpr uint32_t BP_POTENTIAL_PREPARED=2u;
static constexpr uint32_t BP_POTENTIAL_VERTICES=256u;
struct alignas(128) BiprismPotentialOwnerFfi {
    uint32_t abi_magic,version,leaf_id,state;              // 0..15
    uint64_t allocation_generation,topology_generation; // 16,24
    uint64_t tile_address,coordinate_owner_address;      // 32,40
    double* accepted_phi;                               // 48,256 doubles
    double* prepared_phi;                               // 56,256 doubles
    uint32_t accepted_committed_substeps,accepted_step;   // 64,68
    uint32_t prepared_from_committed_substeps,prepared_step; // 72,76
    uint64_t accepted_operator_generation,prepared_operator_generation; // 80,88
    uint32_t projection_state,projection_committed_substeps,projection_step,reserved0; // 96..111
    uint64_t projection_operator_generation,reserved1;    // 112,120
};
static_assert(sizeof(BiprismPotentialOwnerFfi)==128&&alignof(BiprismPotentialOwnerFfi)==128,"potential descriptor extent");
static_assert(offsetof(BiprismPotentialOwnerFfi,allocation_generation)==16
    &&offsetof(BiprismPotentialOwnerFfi,topology_generation)==24
    &&offsetof(BiprismPotentialOwnerFfi,tile_address)==32
    &&offsetof(BiprismPotentialOwnerFfi,coordinate_owner_address)==40
    &&offsetof(BiprismPotentialOwnerFfi,accepted_phi)==48
    &&offsetof(BiprismPotentialOwnerFfi,prepared_phi)==56
    &&offsetof(BiprismPotentialOwnerFfi,accepted_committed_substeps)==64
    &&offsetof(BiprismPotentialOwnerFfi,prepared_from_committed_substeps)==72
    &&offsetof(BiprismPotentialOwnerFfi,accepted_operator_generation)==80
    &&offsetof(BiprismPotentialOwnerFfi,prepared_operator_generation)==88
    &&offsetof(BiprismPotentialOwnerFfi,projection_state)==96
    &&offsetof(BiprismPotentialOwnerFfi,projection_operator_generation)==112
    &&offsetof(BiprismPotentialOwnerFfi,reserved1)==120,"potential descriptor offsets");
#if defined(__CUDACC__)
#define BP_PHI_HD __host__ __device__ inline
#else
#define BP_PHI_HD inline
#endif
BP_PHI_HD bool bp_potential_owner_valid(const BiprismPotentialOwnerFfi* o,uint32_t leaf,
    const void* tile,const BiprismCoordinateOwnerFfi* coordinate,uint64_t topology_generation=0u) {
    if(!o||(reinterpret_cast<uintptr_t>(o)&127u)||!tile||!coordinate
        ||(reinterpret_cast<uintptr_t>(coordinate)&15u)||leaf>=8u)return false;
    if(o->abi_magic!=BP_POTENTIAL_MAGIC||o->version!=BP_POTENTIAL_VERSION||o->leaf_id!=leaf
        ||o->state>BP_POTENTIAL_PREPARED||!o->allocation_generation||!o->topology_generation
        ||(topology_generation&&o->topology_generation!=topology_generation)
        ||o->allocation_generation!=coordinate->generation||coordinate->leaf_id!=leaf
        ||coordinate->abi_magic!=BP_COORDINATE_MAGIC||coordinate->abi_version!=BP_COORDINATE_VERSION
        ||coordinate->unit_frame!=BP_COORD_ATOMIC_BOHR
        ||o->tile_address!=uint64_t(reinterpret_cast<uintptr_t>(tile))
        ||o->coordinate_owner_address!=uint64_t(reinterpret_cast<uintptr_t>(coordinate))
        ||!o->accepted_phi||!o->prepared_phi||o->accepted_phi==o->prepared_phi
        ||(reinterpret_cast<uintptr_t>(o->accepted_phi)&7u)||(reinterpret_cast<uintptr_t>(o->prepared_phi)&7u)
        ||o->reserved0||o->reserved1)return false;
    const uintptr_t a=reinterpret_cast<uintptr_t>(o->accepted_phi),p=reinterpret_cast<uintptr_t>(o->prepared_phi);
    return a<=UINTPTR_MAX-2048u&&p<=UINTPTR_MAX-2048u&&(a+2048u<=p||p+2048u<=a);
}
BP_PHI_HD bool bp_potential_cold_valid(const BiprismPotentialOwnerFfi& o) {
    return o.state==BP_POTENTIAL_COLD&&!o.accepted_committed_substeps&&!o.accepted_step
        &&!o.prepared_from_committed_substeps&&!o.prepared_step&&!o.accepted_operator_generation
        &&!o.prepared_operator_generation&&!o.projection_state&&!o.projection_committed_substeps
        &&!o.projection_step&&!o.projection_operator_generation;
}
BP_PHI_HD bool bp_potential_accepted_valid(const BiprismPotentialOwnerFfi& o,uint32_t committed,uint32_t step) {
    return o.state==BP_POTENTIAL_ACCEPTED&&o.accepted_committed_substeps==committed
        &&o.accepted_step==step&&o.accepted_operator_generation>0u
        &&o.projection_state==BP_POTENTIAL_ACCEPTED&&o.projection_committed_substeps==committed
        &&o.projection_step==step&&o.projection_operator_generation==o.accepted_operator_generation;
}
BP_PHI_HD bool bp_potential_prepare_valid(const BiprismPotentialOwnerFfi& o,uint32_t committed,uint32_t step) {
    return step>0u&&step<UINT32_MAX&&bp_potential_accepted_valid(o,committed,step-1u)
        &&o.accepted_operator_generation<UINT64_MAX;
}
BP_PHI_HD bool bp_potential_prepared_valid(const BiprismPotentialOwnerFfi& o,uint32_t committed,uint32_t step) {
    return o.state==BP_POTENTIAL_PREPARED&&step>0u&&step<UINT32_MAX
        &&o.accepted_committed_substeps==committed&&o.accepted_step==step-1u
        &&o.prepared_from_committed_substeps==committed&&o.prepared_step==step
        &&o.accepted_operator_generation>0u&&o.accepted_operator_generation<UINT64_MAX
        &&o.prepared_operator_generation==o.accepted_operator_generation+1u
        &&o.projection_state==BP_POTENTIAL_PREPARED&&o.projection_committed_substeps==committed
        &&o.projection_step==step&&o.projection_operator_generation==o.prepared_operator_generation;
}
BP_PHI_HD bool bp_potential_input_valid(const BiprismPotentialOwnerFfi& o,uint32_t committed,uint32_t step,bool require_prepared) {
    if(committed>UINT32_MAX-4u||!step||step==UINT32_MAX)return false;
    return bp_potential_prepared_valid(o,committed,step)
        ||(!require_prepared&&bp_potential_accepted_valid(o,committed,step-1u));
}
BP_PHI_HD const double* bp_potential_input_phi(const BiprismPotentialOwnerFfi& o) {
    return o.state==BP_POTENTIAL_PREPARED?o.prepared_phi:o.accepted_phi;
}
BP_PHI_HD bool bp_potential_projection_value_valid(const BiprismPotentialOwnerFfi& o,uint32_t vertex,float projection) {
    if(vertex>=BP_POTENTIAL_VERTICES||(o.state!=BP_POTENTIAL_ACCEPTED&&o.state!=BP_POTENTIAL_PREPARED))return false;
    const double value=bp_potential_input_phi(o)[vertex];const float rounded=float(value);
    if(!isfinite(value)||!isfinite(rounded))return false;
#if defined(__CUDA_ARCH__)
    return __float_as_uint(rounded)==__float_as_uint(projection);
#else
    uint32_t a,b;memcpy(&a,&rounded,4);memcpy(&b,&projection,4);return a==b;
#endif
}
// These metadata publications have no rejecting branch. The caller first
// validates all8 leaves, writes all256 finite values and their FP32 projection,
// then completes its data-write barrier. The final state assignment publishes
// identity; it is not an independent physical acceptance or clock increment.
BP_PHI_HD void bp_potential_publish_cold(BiprismPotentialOwnerFfi& o,uint32_t committed,uint32_t step) {
    o.accepted_committed_substeps=committed;o.accepted_step=step;o.accepted_operator_generation=1u;
    o.projection_state=BP_POTENTIAL_ACCEPTED;o.projection_committed_substeps=committed;
    o.projection_step=step;o.projection_operator_generation=1u;o.state=BP_POTENTIAL_ACCEPTED;
}
BP_PHI_HD void bp_potential_publish_prepared(BiprismPotentialOwnerFfi& o,uint32_t committed,uint32_t step) {
    o.prepared_from_committed_substeps=committed;o.prepared_step=step;
    o.prepared_operator_generation=o.accepted_operator_generation+1u;
    o.projection_state=BP_POTENTIAL_PREPARED;o.projection_committed_substeps=committed;
    o.projection_step=step;o.projection_operator_generation=o.prepared_operator_generation;o.state=BP_POTENTIAL_PREPARED;
}
BP_PHI_HD void bp_potential_publish_main_commit(BiprismPotentialOwnerFfi& o,uint32_t committed,uint32_t step) {
    if(o.state==BP_POTENTIAL_PREPARED)o.accepted_operator_generation=o.prepared_operator_generation;
    o.accepted_committed_substeps=committed;o.accepted_step=step;
    o.projection_state=BP_POTENTIAL_ACCEPTED;o.projection_committed_substeps=committed;
    o.projection_step=step;o.projection_operator_generation=o.accepted_operator_generation;o.state=BP_POTENTIAL_ACCEPTED;
}
#undef BP_PHI_HD
