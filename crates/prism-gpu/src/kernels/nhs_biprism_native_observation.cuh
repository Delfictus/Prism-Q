// Native density and same-emission context for versioned V10S records.
// Included by nhs_amber_fused.cu after its real WarpEntry, ProtocolState and
// exclusion/water helpers. This header does not run the legacy RAF/LIF detector,
// modify coupling, or reinterpret V10S bytes as native SpikeEvent bytes.
// Positions, centroids and grid geometry are native Angstrom quantities.
// Water density retains the native inferred molecules/Angstrom^3 model.
#pragma once
#ifndef NHS_BIPRISM_NATIVE_OBSERVATION_CUH_INCLUDED
#define NHS_BIPRISM_NATIVE_OBSERVATION_CUH_INCLUDED
#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include "nhs_biprism_coordinate_precision.cuh"

struct WarpEntry;
struct ProtocolState;

struct __align__(16) NativeLifObservation {
    uint32_t step;
    uint16_t leaf;
    uint16_t vertex;
    uint32_t atom;
    int32_t cell;
    float water;
    float wd_change;
    int32_t n_excited;
    uint32_t phase_bits;
    // Same-emission producer-coordinate expansion. V10S retains the high
    // components; this independently versioned companion retains all lows.
    float producer_position_lo[3];
    uint32_t coordinate_magic;
    uint64_t coordinate_generation;
    uint32_t coordinate_unit_frame;
    uint32_t coordinate_reserved;
};
static_assert(sizeof(NativeLifObservation)==64,"NOB2 native observation size");
static_assert(alignof(NativeLifObservation)==16,"Native observation alignment");
static_assert(offsetof(NativeLifObservation,leaf)==4 && offsetof(NativeLifObservation,vertex)==6,"Native observation identity");
static_assert(offsetof(NativeLifObservation,atom)==8 && offsetof(NativeLifObservation,cell)==12,"Native observation population");
static_assert(offsetof(NativeLifObservation,water)==16 && offsetof(NativeLifObservation,wd_change)==20,"Native observation density");
static_assert(offsetof(NativeLifObservation,n_excited)==24 && offsetof(NativeLifObservation,phase_bits)==28,"Native observation metadata");
static_assert(offsetof(NativeLifObservation,producer_position_lo)==32 && offsetof(NativeLifObservation,coordinate_generation)==48,"NOB2 coordinate expansion");

// Fifteen actual device pointers followed by the native grid/population shape.
// The host retains all allocations and validates unique active tile coordinates.
// selected_atom_ids points at the current tile's256 packed global_atom_idx words.
struct __align__(16) NativeLifObservationLeafFfi {
    const float3* positions;
    const int32_t* atom_types;
    const float* charges;
    const WarpEntry* warp_matrix;
    const float3* aromatic_centroids;
    const int32_t* is_excited;
    const float* electronic_population;
    const int32_t* atom_to_aromatic;
    const ProtocolState* protocol;
    float* exclusion_field;
    float* water_density;
    float* water_density_prev;
    uint32_t* density_epoch;
    const uint32_t* selected_atom_ids;
    const int32_t* active_tiles;
    uint32_t n_atoms;
    uint32_t n_aromatics;
    uint32_t grid_dim;
    uint32_t n_active_tiles;
    float grid_spacing;
    float origin_x;
    float origin_y;
    float origin_z;
    const BiprismCoordinateOwnerFfi* native_coordinate_owner;
    const BiprismCoordinateOwnerFfi* au_coordinate_owner;
    uint64_t reserved;
};
static_assert(sizeof(NativeLifObservationLeafFfi)==176,"NOB2 native observation leaf ABI");
static_assert(offsetof(NativeLifObservationLeafFfi,n_atoms)==120,"Native observation shape offset");
static_assert(offsetof(NativeLifObservationLeafFfi,grid_spacing)==136,"Native observation grid offset");
static_assert(offsetof(NativeLifObservationLeafFfi,native_coordinate_owner)==152 && offsetof(NativeLifObservationLeafFfi,au_coordinate_owner)==160,"NOB2 coordinate owners");

constexpr uint32_t BIPRISM_NATIVE_OBSERVATION_MAGIC=0x4e4f4232u;
enum NativeLifObservationFault : uint32_t {
    NATIVE_OBSERVATION_BAD_INPUT=1u,
    NATIVE_OBSERVATION_STALE_EPOCH=2u,
    NATIVE_OBSERVATION_CAPACITY=4u,
    NATIVE_OBSERVATION_IDENTITY=8u,
    NATIVE_OBSERVATION_NONFINITE=16u
};
struct __align__(16) NativeLifObservationCaptureStateFfi {
    uint32_t abi_magic;
    uint32_t cursor;
    uint32_t last_step;
    uint32_t fault;
    uint32_t bad_record;
    uint32_t captured_records;
    uint32_t capture_epochs;
    uint32_t reserved;
};
static_assert(sizeof(NativeLifObservationCaptureStateFfi)==32,"Native capture state ABI");

#ifndef BIPRISM_OBSERVATION_ABI_ONLY
// No concurrent protocol/queue/density/retile writer is permitted during these
// launches. Density epochs start UINT32_MAX, and protocol step UINT32_MAX is
// unsupported. Queue count/cursor reset together only at a completed owner epoch.
__device__ __forceinline__ bool biprism_native_observation_leaf_valid(
    const NativeLifObservationLeafFfi& x,uint32_t leaf
) {
    if(!(x.positions && x.atom_types && x.charges && x.warp_matrix &&
        x.atom_to_aromatic && x.protocol && x.exclusion_field &&
        x.water_density && x.water_density_prev && x.density_epoch &&
        x.selected_atom_ids && x.n_atoms>0 && x.n_atoms<=0x01000000u &&
        x.n_aromatics<=0x7fffffffu && x.grid_dim>0 && x.grid_dim<=128u &&
        isfinite(x.grid_spacing) && x.grid_spacing>0.0f &&
        isfinite(x.origin_x) && isfinite(x.origin_y) && isfinite(x.origin_z) &&
        (x.n_aromatics==0 || (x.aromatic_centroids && x.is_excited && x.electronic_population)) &&
        (x.n_active_tiles==0 || x.active_tiles))||x.reserved
        ||!x.native_coordinate_owner||!x.au_coordinate_owner
        ||(reinterpret_cast<uintptr_t>(x.native_coordinate_owner)&15u)
        ||(reinterpret_cast<uintptr_t>(x.au_coordinate_owner)&15u))return false;
    const auto& n=*x.native_coordinate_owner;const auto& a=*x.au_coordinate_owner;
    return bp_coord_owner_valid(&n,leaf,x.n_atoms,a.generation,BP_COORD_NATIVE_ANGSTROM,
        BP_COORD_ALL_INITIALIZED,reinterpret_cast<const float*>(x.positions))
        &&bp_coord_owner_valid(&a,leaf,x.n_atoms,n.generation,BP_COORD_ATOMIC_BOHR,
        BP_COORD_ALL_INITIALIZED,a.current_hi);
}

__device__ __forceinline__ float biprism_native_exclusion_precise(
    BiprismCoordinate3F64 atom,float3 center,int type,float charge) {
    const double dx=double(center.x)-atom.x,dy=double(center.y)-atom.y,dz=double(center.z)-atom.z;
    const float dist_sq=float(dx*dx+dy*dy+dz*dz);
    const float radius=get_exclusion_radius(type),sigma=radius/2.f;
    const float exclusion=expf(-dist_sq/(2.f*sigma*sigma));
    return exclusion*(1.f+.5f*fabsf(charge));
}

// Density-only extraction of nhs_voxel_step_multi_lif Phase1. Native helpers,
// strided atom order,8-lane xor reduction, cached-first64 aromatic modifier and
// voxel-center arithmetic are preserved. Each selected voxel has one writer.
// Inactive sparse voxels receive no new epoch and cannot be projected as fresh.
extern "C" __global__ __launch_bounds__(128,8)
void nhs_biprism_native_density_only_v1(
    const NativeLifObservationLeafFfi* leaves,
    uint32_t leaf_id,
    NativeLifObservationCaptureStateFfi* state
) {
    if (!state) return;
    if (blockDim.x!=128u || blockDim.y!=1u || blockDim.z!=1u || !leaves || leaf_id>=8u ||
        state->abi_magic!=BIPRISM_NATIVE_OBSERVATION_MAGIC) {
        if (threadIdx.x==0) atomicOr(&state->fault,NATIVE_OBSERVATION_BAD_INPUT);
        return;
    }
    __shared__ uint32_t initial_fault;
    if (threadIdx.x==0)initial_fault=atomicAdd(&state->fault,0u);
    __syncthreads();
    if (initial_fault!=0u) return;
    const NativeLifObservationLeafFfi x=leaves[leaf_id];
    if (!biprism_native_observation_leaf_valid(x,leaf_id)) {
        if (threadIdx.x==0) atomicOr(&state->fault,NATIVE_OBSERVATION_BAD_INPUT);
        return;
    }
    if (!prism_protocol_observation_due(x.protocol)) return;
    const uint32_t step=x.protocol->current_step;
    const float temperature=x.protocol->current_temperature;
    if (step==0xffffffffu || !isfinite(temperature) || temperature<=0.0f) {
        if (threadIdx.x==0) atomicOr(&state->fault,NATIVE_OBSERVATION_BAD_INPUT);
        return;
    }
    const uint32_t tiles_x=(x.grid_dim+3u)/4u,tiles_y=(x.grid_dim+1u)/2u,tiles_z=(x.grid_dim+1u)/2u;
    const uint32_t tile_count=x.n_active_tiles?x.n_active_tiles:tiles_x*tiles_y*tiles_z;
    if (x.n_active_tiles>tiles_x*tiles_y*tiles_z || gridDim.x!=tile_count || gridDim.y!=1u || gridDim.z!=1u) {
        if (threadIdx.x==0) atomicOr(&state->fault,NATIVE_OBSERVATION_BAD_INPUT);
        return;
    }
    int bx,by,bz;
    if (x.n_active_tiles) {
        bx=x.active_tiles[blockIdx.x*3u];by=x.active_tiles[blockIdx.x*3u+1u];bz=x.active_tiles[blockIdx.x*3u+2u];
    } else {
        bx=int(blockIdx.x%tiles_x);by=int((blockIdx.x/tiles_x)%tiles_y);bz=int(blockIdx.x/(tiles_x*tiles_y));
    }
    if (bx<0 || by<0 || bz<0 || uint32_t(bx)>=tiles_x || uint32_t(by)>=tiles_y || uint32_t(bz)>=tiles_z) {
        if (threadIdx.x==0) atomicOr(&state->fault,NATIVE_OBSERVATION_BAD_INPUT);
        return;
    }
    __shared__ float aromatic[64*4];
    const int n_arom=int(x.n_aromatics<64u?x.n_aromatics:64u);
    for (int a=int(threadIdx.x);a<n_arom;a+=128) {
        const float3 p=x.aromatic_centroids[a];
        aromatic[a*4]=p.x;aromatic[a*4+1]=p.y;aromatic[a*4+2]=p.z;aromatic[a*4+3]=x.electronic_population[a];
    }
    __syncthreads();
    const int local_voxel=int(threadIdx.x/8u),lane=int(threadIdx.x%8u);
    const int vx=bx*4+local_voxel%4,vy=by*2+(local_voxel/4)%2,vz=bz*2+local_voxel/8;
    const bool valid=vx<int(x.grid_dim)&&vy<int(x.grid_dim)&&vz<int(x.grid_dim);
    if (!valid) return; // Whole8-lane voxel subgroup exits together.
    const int cell=(vz*int(x.grid_dim)+vy)*int(x.grid_dim)+vx;
    const float3 center=make_float3(x.origin_x+(vx+0.5f)*x.grid_spacing,x.origin_y+(vy+0.5f)*x.grid_spacing,x.origin_z+(vz+0.5f)*x.grid_spacing);
    const WarpEntry entry=x.warp_matrix[cell];
    if (entry.n_atoms<0 || entry.n_atoms>16 || x.density_epoch[cell]==step) {
        if (lane==0) atomicOr(&state->fault,entry.n_atoms<0||entry.n_atoms>16?NATIVE_OBSERVATION_BAD_INPUT:NATIVE_OBSERVATION_STALE_EPOCH);
        return;
    }
    float exclusion=0.0f,polar=0.0f;
    bool ok=isfinite(center.x)&&isfinite(center.y)&&isfinite(center.z);
    for (int i=lane;i<entry.n_atoms;i+=8) {
        const int a=entry.atom_indices[i];
        if (a<0 || uint32_t(a)>=x.n_atoms) {ok=false;continue;}
        const auto apos=bp_coord_value(bp_coord_load(*x.native_coordinate_owner,BiprismCoordinatePlane::Current,uint32_t(a)));
        const float charge=x.charges[a];const int type=x.atom_types[a];
        if (!isfinite(apos.x)||!isfinite(apos.y)||!isfinite(apos.z)||!isfinite(charge)||type<0||!isfinite(entry.atom_weights[i])) {ok=false;continue;}
        float contribution=biprism_native_exclusion_precise(apos,center,type,charge);
        if (n_arom>0) {
            float best_mod=1.0f;
            for (int ar=0;ar<n_arom;++ar) {
                if (!x.is_excited[ar]) continue;
                const float ax=aromatic[ar*4],ay=aromatic[ar*4+1],az=aromatic[ar*4+2],epop=aromatic[ar*4+3];
                if (!isfinite(ax)||!isfinite(ay)||!isfinite(az)||!isfinite(epop)) {ok=false;continue;}
                const double dx=apos.x-double(ax),dy=apos.y-double(ay),dz=apos.z-double(az);
                const double dist_sq=dx*dx+dy*dy+dz*dz;
                if (dist_sq<25.0f) {const float mod=1.0f+epop*0.5f;if(mod>best_mod)best_mod=mod;}
            }
            contribution*=best_mod;
        }
        // The old n_arom==0 branch cannot match an aromatic index because
        // n_aromatics is then0. Omitting its dead read changes no contribution.
        exclusion+=contribution*entry.atom_weights[i]*4.0f;
        if (type==1||type==2||type==3)polar+=contribution*0.5f;
    }
    const unsigned mask=0xffu<<((threadIdx.x&31u)/8u*8u);
    for (int off=4;off>0;off>>=1) {
        exclusion+=__shfl_xor_sync(mask,exclusion,off);
        polar+=__shfl_xor_sync(mask,polar,off);
    }
    const bool group_ok=__all_sync(mask,ok);
    if (lane==0) {
        const float old=x.water_density[cell];
        const float total=fminf(1.0f,exclusion);
        const float next=entry.n_atoms?infer_water_density(total,polar,temperature):old;
        if (!group_ok||!isfinite(old)||!isfinite(next)||!isfinite(total)||!isfinite(polar)) {
            atomicOr(&state->fault,NATIVE_OBSERVATION_NONFINITE);return;
        }
        x.water_density_prev[cell]=old;
        if (entry.n_atoms) {x.exclusion_field[cell]=total;x.water_density[cell]=next;}
        x.density_epoch[cell]=step;
    }
}

// Only reads the fixed V10S identity fields. The original96B record is retained
// unchanged. The native observation ABI is checked against the actual producer
// layout in the companion compile test; it is not a native96B event overlay.
__device__ __forceinline__ uint32_t biprism_observation_read_u32(const unsigned char* p) {
    return uint32_t(p[0])|(uint32_t(p[1])<<8u)|(uint32_t(p[2])<<16u)|(uint32_t(p[3])<<24u);
}
__device__ __forceinline__ uint16_t biprism_observation_read_u16(const unsigned char* p) {
    return uint16_t(uint16_t(p[0])|(uint16_t(p[1])<<8u));
}
__device__ __forceinline__ uint32_t biprism_native_observation_make(
    const unsigned char* raw,const NativeLifObservationLeafFfi* leaves,
    uint32_t expected_step,NativeLifObservation& out
) {
    const uint32_t step=biprism_observation_read_u32(raw+12),atom=biprism_observation_read_u32(raw+16);
    const uint32_t magic=biprism_observation_read_u32(raw+72);
    const uint16_t leaf=biprism_observation_read_u16(raw+64),vertex=biprism_observation_read_u16(raw+66);
    const int32_t cell=int32_t(biprism_observation_read_u32(raw+68));
    if (magic!=0x56313053u || leaf>=8u || vertex>=192u || step!=expected_step) return NATIVE_OBSERVATION_IDENTITY;
    const NativeLifObservationLeafFfi x=leaves[leaf];
    if (!biprism_native_observation_leaf_valid(x,leaf)) return NATIVE_OBSERVATION_BAD_INPUT;
    if (atom>=x.n_atoms || (x.selected_atom_ids[vertex]&0x00ffffffu)!=atom || cell<0 || uint32_t(cell)>=x.grid_dim*x.grid_dim*x.grid_dim) return NATIVE_OBSERVATION_IDENTITY;
    if (x.protocol->current_step!=step || !prism_protocol_observation_due(x.protocol) || x.density_epoch[cell]!=step) return NATIVE_OBSERVATION_STALE_EPOCH;
    const float water=x.water_density[cell],previous=x.water_density_prev[cell];
    const float change=fabsf(water-previous);
    if (!isfinite(water)||!isfinite(previous)||!isfinite(change)) return NATIVE_OBSERVATION_NONFINITE;
    const uint32_t phase=x.protocol->current_phase_bits;
    if (phase>1023u) return NATIVE_OBSERVATION_BAD_INPUT;
    const WarpEntry entry=x.warp_matrix[cell];
    if (entry.n_atoms<0||entry.n_atoms>16) return NATIVE_OBSERVATION_BAD_INPUT;
    const int dim=int(x.grid_dim),vx=cell%dim,vy=(cell/dim)%dim,vz=cell/(dim*dim);
    const float3 center=make_float3(x.origin_x+(vx+0.5f)*x.grid_spacing,x.origin_y+(vy+0.5f)*x.grid_spacing,x.origin_z+(vz+0.5f)*x.grid_spacing);
    int nearby=0;
    if (entry.n_atoms>0) for (uint32_t a=0;a<x.n_aromatics && a<64u;++a) {
        if (!x.is_excited[a])continue;
        const float3 p=x.aromatic_centroids[a];
        if (!isfinite(p.x)||!isfinite(p.y)||!isfinite(p.z))return NATIVE_OBSERVATION_NONFINITE;
        const float dx=center.x-p.x,dy=center.y-p.y,dz=center.z-p.z;
        const float d2=dx*dx+dy*dy+dz*dz;
        // Preserve the native sqrtf(dist_sq)<4 branch, not a new radius test.
        if (d2<=400.0f && sqrtf(d2)<4.0f)++nearby;
    }
    const auto position=bp_coord_load(*x.au_coordinate_owner,BiprismCoordinatePlane::Current,atom);
    if(!bp_coord_finite(position))return NATIVE_OBSERVATION_NONFINITE;
    if(biprism_observation_read_u32(raw)!=__float_as_uint(position.x.hi)
        ||biprism_observation_read_u32(raw+4)!=__float_as_uint(position.y.hi)
        ||biprism_observation_read_u32(raw+8)!=__float_as_uint(position.z.hi))return NATIVE_OBSERVATION_IDENTITY;
    out={};out.step=step;out.leaf=leaf;out.vertex=vertex;out.atom=atom;out.cell=cell;
    out.water=water;out.wd_change=change;out.n_excited=nearby;out.phase_bits=phase;
    out.producer_position_lo[0]=position.x.lo;out.producer_position_lo[1]=position.y.lo;out.producer_position_lo[2]=position.z.lo;
    out.coordinate_magic=BIPRISM_NATIVE_OBSERVATION_MAGIC;
    out.coordinate_generation=x.au_coordinate_owner->generation;
    out.coordinate_unit_frame=BP_COORD_ATOMIC_BOHR;
    return 0u;
}

// Launch one CTA of256 after all eight density launches and the completed LIF
// producer, before any Director/aromatic/grid update. Raw queue and companions
// share ordinals. All new records are validated before the first companion write.
// A capacity/identity/stale fault leaves cursor and the new companion interval
// unchanged; raw events remain available for diagnosis. No silent truncation.
extern "C" __global__ __launch_bounds__(256,2)
void nhs_biprism_capture_native_observations_v1(
    const unsigned char* v10s_records,
    const uint32_t* queue_count,
    const NativeLifObservationLeafFfi* leaves,
    NativeLifObservation* observations,
    uint32_t capacity,
    NativeLifObservationCaptureStateFfi* state
) {
    if (!state) return;
    if (gridDim.x!=1u||gridDim.y!=1u||gridDim.z!=1u||blockDim.x!=256u||blockDim.y!=1u||blockDim.z!=1u||
        !v10s_records||!queue_count||!leaves||!observations||capacity>0x7fffffffu||state->abi_magic!=BIPRISM_NATIVE_OBSERVATION_MAGIC) {
        if (threadIdx.x==0)atomicOr(&state->fault,NATIVE_OBSERVATION_BAD_INPUT);return;
    }
    if (atomicAdd(&state->fault,0u)!=0u)return;
    __shared__ uint32_t begin,end,step,fault,bad;
    if (threadIdx.x==0) {
        begin=state->cursor;end=*queue_count;step=0;fault=0;bad=0xffffffffu;
        if (begin>end||end>capacity)fault|=NATIVE_OBSERVATION_CAPACITY;
        for(uint32_t leaf=0;leaf<8u;++leaf) {
            const NativeLifObservationLeafFfi x=leaves[leaf];
            if(!biprism_native_observation_leaf_valid(x,leaf)){fault|=NATIVE_OBSERVATION_BAD_INPUT;continue;}
            if(leaf==0)step=x.protocol->current_step;
            if(x.protocol->current_step!=step||step==0xffffffffu)fault|=NATIVE_OBSERVATION_STALE_EPOCH;
        }
        if(end>begin && (state->capture_epochs==0xffffffffu || end-begin>0xffffffffu-state->captured_records))fault|=NATIVE_OBSERVATION_CAPACITY;
    }
    __syncthreads();
    const bool header_ok=fault==0u;
    __syncthreads();
    if(header_ok)for(uint32_t i=begin+threadIdx.x;i<end;i+=256u) {
        NativeLifObservation row;const uint32_t f=biprism_native_observation_make(v10s_records+size_t(i)*96u,leaves,step,row);
        if(f){atomicOr(&fault,f);atomicMin(&bad,i);}
    }
    __syncthreads();
    if(fault) {if(threadIdx.x==0){atomicOr(&state->fault,fault);state->bad_record=bad;}return;}
    for(uint32_t i=begin+threadIdx.x;i<end;i+=256u) {
        NativeLifObservation row;
        // Inputs are immutable throughout this same-stream launch, so the
        // prevalidated reconstruction has the same identity and values.
        biprism_native_observation_make(v10s_records+size_t(i)*96u,leaves,step,row);
        observations[i]=row;
    }
    __syncthreads();
    if(threadIdx.x==0) {
        state->cursor=end;state->last_step=step;state->bad_record=0xffffffffu;
        if(end>begin){state->captured_records+=end-begin;++state->capture_epochs;}
    }
}
#endif // BIPRISM_OBSERVATION_ABI_ONLY
#endif
