// File: roles/engine/crates/prism-gpu/src/kernels/nhs_biprism_voronoi_lif.cuh
// Candidate: bounded multiplicity ledger, signed grids, 256-endpoint Q39 forces,
// and dedicated epoch-tagged feedback delivered as a Biprism lab-frame kick.
// Exact operator proposals and implementation evidence are retained under
// INTERNAL_QC/v10_delivery_decoder_revision. The versioned96B packet requires
// an explicit decoder; it is not a byte-compatible native GpuSpikeEvent layout.
// Core voltages evolve with fixed halo voltages. Integer pair cancellation does
// not imply exact FP32 momentum/energy conservation after baseline addition/kicks.
// Coupled counts retain this candidate's force definition, not native UV causality.
// Producer refusals are per CTA; delivery prevalidates all8 leaves in one CTA.
// Production graph/RATTLE execution and complete-window acceptance remain separate.
#pragma once
#ifndef NHS_BIPRISM_VORONOI_LIF_CUH_INCLUDED
#define NHS_BIPRISM_VORONOI_LIF_CUH_INCLUDED

#include "nhs_biprism_variational_so3g_v10.cuh"
#include "nhs_biprism_coordinate_geometry.cuh"
#include <stddef.h>

#define SPIKE_EVENT_V10_SCHEMA_MAGIC 0x56313053u

struct __align__(32) GpuSpikeEventCanonical96 {
    float    pos_xyz[3];            // Bytes 0..11:  Deformed Voronoi generator coordinate x_u(\tau)
    uint32_t step_tau;              // Bytes 12..15: Authoritative protocol step
    uint32_t atom_or_site_id;       // Bytes 16..19: Global AMBER atom ID
    float    spike_amplitude;       // Bytes 20..23: Coupled floating-point spike payload p_u
    float    uv_or_holonomy_val;    // Bytes 24..27: Space-Leaf Wilson holonomy deficit (2 - Tr(W_{x\lambda}))
    float    voronoi_dual_vol;      // Bytes 28..31: Warped Voronoi dual volume \star_0(u)
    uint32_t pocket_tile_id;        // Bytes 32..35: Active pocket ID
    int32_t  residue_id;            // Bytes 36..39: Signed primary protein residue ID of firing generator
    uint32_t multiplicity_m;        // Bytes 40..43: Integer spike multiplicity m_u >= 1
    float    v_pre_reset;           // Bytes 44..47: Pre-reset membrane voltage V_{pre,u}
    float    q_discharged;          // Bytes 48..51: High-order discharged charge q_{hi} = __fmul_rn(m_u, \theta_{th})
    float    q_discharged_lo;       // Bytes 52..55: Low-order correction q_{lo} so (q_hi + q_lo) + r_post == v_pre_reset
    float    r_post_reset;          // Bytes 56..59: Retained sub-threshold remainder r_u \in [0, \theta_{th})
    float    lif_threshold_used;    // Bytes 60..63: Threshold \theta_{th}
    uint16_t leaf_id;               // Bytes 64..65: Foliated leaf index (0..7)
    uint16_t vertex_id;             // Bytes 66..67: Tile-local Voronoi vertex index (0..191)
    int32_t  voxel_cell_idx;        // Bytes 68..71: Mapped 3D signal grid cell index (-1 if out of bounds)
    uint32_t event_schema_magic;    // Bytes 72..75: 0x56313053u ("V10S") version tag for host decoder migration
    float    coupled_force_xyz[3];  // Bytes 76..87: Produced force on this generator; delivery is acknowledged separately
    uint32_t q_lo_product_raw_bits; // Bytes 88..91: Raw IEEE-754 bits of __fmaf_rn(m_u, \theta_{th}, -q_hi)
    uint32_t reserved_v5;           // Bytes 92..95: Exact 96-byte closure
};
static_assert(sizeof(GpuSpikeEventCanonical96) == 96, "Versioned candidate event must be96bytes");
static_assert(alignof(GpuSpikeEventCanonical96) == 32, "Versioned event alignment must be32");
static_assert(offsetof(GpuSpikeEventCanonical96, q_discharged_lo) == 52, "q_low ABI offset");
static_assert(offsetof(GpuSpikeEventCanonical96, r_post_reset) == 56, "remainder ABI offset");
static_assert(offsetof(GpuSpikeEventCanonical96, lif_threshold_used) == 60, "threshold ABI offset");
static_assert(offsetof(GpuSpikeEventCanonical96, leaf_id) == 64, "leaf ABI offset");
static_assert(offsetof(GpuSpikeEventCanonical96, vertex_id) == 66, "vertex ABI offset");
static_assert(offsetof(GpuSpikeEventCanonical96, voxel_cell_idx) == 68, "voxel ABI offset");
static_assert(offsetof(GpuSpikeEventCanonical96, event_schema_magic) == 72, "schema ABI offset");
static_assert(offsetof(GpuSpikeEventCanonical96, coupled_force_xyz) == 76, "produced-force ABI offset");
static_assert(offsetof(GpuSpikeEventCanonical96, q_lo_product_raw_bits) == 88, "product-low ABI offset");

struct __align__(16) SignalGridSpecFfi {
    float    origin_xyz[3];
    float    inv_voxel_spacing;
    uint32_t dim_xyz[3];
    uint32_t total_voxels;
};
static_assert(sizeof(SignalGridSpecFfi) == 32, "SignalGridSpecFfi must be 32 bytes");

__device__ __forceinline__ int32_t map_voronoi_vertex_to_signal_cell(
    float3 pos,
    const SignalGridSpecFfi& spec
) {
    const int32_t ix = __float2int_rd((pos.x - spec.origin_xyz[0]) * spec.inv_voxel_spacing);
    const int32_t iy = __float2int_rd((pos.y - spec.origin_xyz[1]) * spec.inv_voxel_spacing);
    const int32_t iz = __float2int_rd((pos.z - spec.origin_xyz[2]) * spec.inv_voxel_spacing);
    if (ix < 0 || iy < 0 || iz < 0 ||
        (uint32_t)ix >= spec.dim_xyz[0] ||
        (uint32_t)iy >= spec.dim_xyz[1] ||
        (uint32_t)iz >= spec.dim_xyz[2]) {
        return -1;
    }
    return (int32_t)(((uint32_t)iz * spec.dim_xyz[1] + (uint32_t)iy) * spec.dim_xyz[0] + (uint32_t)ix);
}

__device__ __forceinline__ int32_t map_voronoi_vertex_to_signal_cell_precise(
    BiprismCoordinate3F64 pos,const SignalGridSpecFfi& spec) {
    const double x=floor((pos.x-double(spec.origin_xyz[0]))*double(spec.inv_voxel_spacing));
    const double y=floor((pos.y-double(spec.origin_xyz[1]))*double(spec.inv_voxel_spacing));
    const double z=floor((pos.z-double(spec.origin_xyz[2]))*double(spec.inv_voxel_spacing));
    if(!isfinite(x)||!isfinite(y)||!isfinite(z)||x<0.||y<0.||z<0.
        ||x>=double(spec.dim_xyz[0])||y>=double(spec.dim_xyz[1])||z>=double(spec.dim_xyz[2]))return -1;
    const uint64_t cell=(uint64_t(z)*spec.dim_xyz[1]+uint64_t(y))*spec.dim_xyz[0]+uint64_t(x);
    return cell<uint64_t(spec.total_voxels)&&cell<=0x7fffffffu?int32_t(cell):-1;
}


// Supported per-launch count domain: at most eight leaves * 192 core emitters,
// each with multiplicity <= 65535. Counts start nonnegative, primary IDs start -1
// (or an existing nonnegative owner), and no other launch writes these grids.
// Every count grid must initially be <= INT_MAX - BIPRISM_LIF_LAUNCH_COUNT_BOUND.
// The host must enforce that bound before launch; per-counter refusal below cannot
// roll back an unsupported concurrent/overflowing epoch. Native owner fault routing
// and complete-window acceptance are unresolved. Emitted-record checks are narrower.
// Event capacity is reserved atomically by CTA; unavailable capacity rejects that CTA.
constexpr uint32_t BIPRISM_LIF_MAX_MULTIPLICITY = 65535u;
constexpr int32_t BIPRISM_LIF_LAUNCH_COUNT_BOUND = 8 * TILE_CORE_VERTICES * 65535;

// Return false without modifying this counter if its supported signed range fails.
// A failed individual update is not a rollback of earlier writes in this launch.
__device__ __forceinline__ bool biprism_add_count_i32_checked(int32_t* addr, uint32_t amount) {
    if (amount > 0x7fffffffu) return false;
    int32_t old = atomicCAS(addr, 0, 0);
    while (old >= 0 && old <= 0x7fffffff - static_cast<int32_t>(amount)) {
        const int32_t seen = atomicCAS(addr, old, old + static_cast<int32_t>(amount));
        if (seen == old) return true;
        old = seen;
    }
    return false;
}

// Reserve every record from one CTA before its physical or grid state commits.
// Insufficient capacity leaves this CTA unchanged except earlier producers' state;
// it is not an eight-leaf transaction and requires an owner fault channel before activation.
__device__ __forceinline__ bool biprism_reserve_event_batch(
    uint32_t* count, uint32_t capacity, uint32_t amount, uint32_t& base
) {
    uint32_t old = atomicCAS(count, 0u, 0u);
    while (old <= capacity && amount <= capacity - old) {
        const uint32_t seen = atomicCAS(count, old, old + amount);
        if (seen == old) { base = old; return true; }
        old = seen;
    }
    return false;
}

// Recompute the frozen-base diagonal DEC operator at post-integrator coordinates.
// This uses the same floor and squared-distance regularization as the v10 main.
__device__ __forceinline__ float biprism_live_diagonal_edge(
    const ContactShellSimplicialStateFfi* tile, const BiprismCoordinateOwnerFfi& coordinates,
    uint32_t e, float3& grad_u
) {
    const uint8_t u = tile->edge_u8[e][0], v = tile->edge_u8[e][1];
    const uint32_t gu = UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[u]);
    const uint32_t gv = UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[v]);
    const auto d=bp_coord_difference(bp_coord_load(coordinates,BiprismCoordinatePlane::Current,gv),
        bp_coord_load(coordinates,BiprismCoordinatePlane::Current,gu));
    const auto ref=bp_coord_difference(bp_coord_load(coordinates,BiprismCoordinatePlane::MeshReference,gv),
        bp_coord_load(coordinates,BiprismCoordinatePlane::MeshReference,gu));
    const float w_live=bp_geometry_live_weight(tile->star1_wt[e],d,ref);
    grad_u=bp_geometry_live_weight_gradient(w_live,d);
    return w_live;
}

// Quantize once and use exact unsigned additive inverses. The conservative edge
// bound keeps every possible signed vertex accumulation below half LLONG_MAX.
__device__ __forceinline__ bool biprism_quantize_edge_q39(float value, long long& q) {
    constexpr long long limit = 0x7fffffffffffffffLL / (2LL * TILE_MAX_EDGES);
    const double scaled = static_cast<double>(value) * static_cast<double>(FP_Q39_SCALE);
    if (!isfinite(scaled) || fabs(scaled) > static_cast<double>(limit)) return false;
    q = __double2ll_rn(scaled);
    return q >= -limit && q <= limit;
}


// Convert the complete zero-sum Q39 vector to a common exactly representable
// binary32 lattice. Largest-remainder rounding minimizes squared error on that
// lattice subject to exact closure. A component moves by less than one common
// quantum, which can exceed one local ULP for a small component. This does not
// establish closure after addition to an arbitrary baseline or a velocity kick.
// All256 threads call this helper after edge accumulation has completed. Input
// Q39 values remain immutable throughout all three calls.
__device__ __forceinline__ float biprism_balanced_q39_axis_f32(
    const unsigned long long (*force_q39)[3], uint32_t axis,
    uint32_t& shared_shift, uint32_t& shared_round_up,
    uint32_t& shared_fault
) {
    const uint32_t tid = threadIdx.x;
    if (tid == 0u) {
        unsigned long long maximum = 0ull, absolute_sum = 0ull;
        long long sum = 0ll;
        bool valid = true;
        for (uint32_t i = 0u; i < TILE_TOTAL_VERTICES; ++i) {
            const long long k = static_cast<long long>(force_q39[i][axis]);
            const unsigned long long magnitude = k < 0ll ? 0ull - static_cast<unsigned long long>(k)
                                                         : static_cast<unsigned long long>(k);
            maximum = magnitude > maximum ? magnitude : maximum;
            // The edge quantization bound proves this absolute-sum limit for
            // every supported input. Check before signed summation, too.
            if (magnitude > 0x7fffffffffffffffull - absolute_sum) valid = false;
            else { absolute_sum += magnitude; if (valid) sum += k; }
        }
        uint32_t shift = 0u;
        if (maximum > 0x1000000ull) {
            shift = static_cast<uint32_t>(64 - __clzll(maximum - 1ull) - 24);
        }
        shared_shift = shift;
        const unsigned long long mask = (1ull << shift) - 1ull;
        long long floor_sum = 0ll;
        for (uint32_t i = 0u; i < TILE_TOTAL_VERTICES; ++i) {
            const long long k = static_cast<long long>(force_q39[i][axis]);
            const unsigned long long magnitude = k < 0ll ? 0ull - static_cast<unsigned long long>(k)
                                                         : static_cast<unsigned long long>(k);
            const long long base = k < 0ll ? -static_cast<long long>((magnitude + mask) >> shift)
                                          : static_cast<long long>(magnitude >> shift);
            floor_sum += base;
        }
        valid = valid && sum == 0ll && floor_sum <= 0ll && floor_sum > -256ll;
        shared_round_up = valid ? static_cast<uint32_t>(-floor_sum) : 0u;
        if (!valid) atomicOr(&shared_fault, 128u);
    }
    __syncthreads();
    const uint32_t shift = shared_shift;
    const unsigned long long mask = (1ull << shift) - 1ull;
    const long long k = static_cast<long long>(force_q39[tid][axis]);
    const unsigned long long magnitude = k < 0ll ? 0ull - static_cast<unsigned long long>(k)
                                                 : static_cast<unsigned long long>(k);
    const long long base = k < 0ll ? -static_cast<long long>((magnitude + mask) >> shift)
                                  : static_cast<long long>(magnitude >> shift);
    const unsigned long long remainder = static_cast<unsigned long long>(k) & mask;
    uint32_t rank = 0u;
    for (uint32_t i = 0u; i < TILE_TOTAL_VERTICES; ++i) {
        const unsigned long long other = force_q39[i][axis] & mask;
        rank += other > remainder || (other == remainder && i < tid);
    }
    const long long rounded = base + (rank < shared_round_up ? 1ll : 0ll);
    // |rounded|<=2^24 and 0<=shift<=39: both conversion and power-of-two
    // scaling are exact and normal in binary32 for the supported Q39 domain.
    const float value = __ll2float_rn(rounded) * __uint_as_float((shift + 88u) << 23u);
    __syncthreads(); // shared metadata may be reused for the next axis only now.
    return value;
}

__device__ __forceinline__ bool biprism_sigma_delta_partition(
    float v_pre, float threshold, uint32_t& m, float& q_high, float& q_low, float& remainder
) {
    m = 0u; q_high = 0.0f; q_low = 0.0f; remainder = v_pre;
    if (!isfinite(v_pre)) return false;
    if (v_pre < threshold) return true; // Preserve finite inhibitory nonfiring states.
    const float ratio = v_pre / threshold;
    // Permit provisional 65536, then correct the represented-input quotient by one.
    if (!isfinite(ratio) || ratio < 1.0f || ratio > 65536.0f) return false;
    m = static_cast<uint32_t>(floorf(ratio));
    remainder = __fmaf_rn(-static_cast<float>(m), threshold, v_pre);
    if (remainder < 0.0f) {
        --m;
        remainder = __fmaf_rn(-static_cast<float>(m), threshold, v_pre);
    } else if (remainder >= threshold) {
        ++m;
        remainder = __fmaf_rn(-static_cast<float>(m), threshold, v_pre);
    }
    if (m == 0u || m > BIPRISM_LIF_MAX_MULTIPLICITY ||
        !(remainder >= 0.0f && remainder < threshold)) return false;
    q_high = static_cast<float>(m) * threshold;
    q_low = __fmaf_rn(static_cast<float>(m), threshold, -q_high);
    return isfinite(q_high) && isfinite(q_low);
}

// Dedicated per-leaf feedback state. Host initializes all bytes to zero, owns the
// protocol clock and this allocation exclusively, and retains it across steps.
// Captured atom IDs prevent a later retile from remapping a pending impulse.
struct __align__(16) BiprismLifCarryoverFfi {
    uint32_t pending;
    uint32_t produced_step;
    uint32_t delivered_step;
    uint32_t fault;
    uint32_t produced_committed_substeps;
    uint32_t reserved[3];
    uint32_t atom_id[TILE_TOTAL_VERTICES];
    float4 force[TILE_TOTAL_VERTICES]; // xyz: produced force; w: reserved zero
};
static_assert(sizeof(BiprismLifCarryoverFfi) == 5152, "Carryover descriptor ABI must be5152bytes");
static_assert(alignof(BiprismLifCarryoverFfi) == 16, "Carryover descriptor alignment must be16");
static_assert(offsetof(BiprismLifCarryoverFfi, produced_committed_substeps) == 16, "Produced success counter ABI offset");
static_assert(offsetof(BiprismLifCarryoverFfi, atom_id) == 32, "Carryover atom IDs start at32");
static_assert(offsetof(BiprismLifCarryoverFfi, force) == 1056, "Carryover forces start at1056");

enum BiprismLifCarryFault : uint32_t {
    BIPRISM_CARRY_BAD_INPUT = 1u,
    BIPRISM_CARRY_PENDING_OVERWRITE = 2u,
    BIPRISM_CARRY_STALE_EPOCH = 4u,
    BIPRISM_CARRY_UPSTREAM_FAULT = 8u,
    BIPRISM_CARRY_NUMERIC_FAULT = 16u,
    BIPRISM_CARRY_MIXED_PENDING = 32u,
    BIPRISM_CARRY_QUEUE_CAPACITY = 64u,
    BIPRISM_CARRY_COUNT_RANGE = 128u
};

// Conservative representation of a physical velocity impulse. Every selected
// value brackets v0+dt*inv_mass*force; no force or time-step projection is used.
__device__ __forceinline__ double biprism_lif_warp_sum_f64(double value) {
    for (uint32_t offset=16u;offset!=0u;offset>>=1u)
        value += __shfl_down_sync(0xffffffffu,value,offset);
    return __shfl_sync(0xffffffffu,value,0);
}

__device__ __forceinline__ bool biprism_lif_velocity_bracket(
    float before,float dt,float inverse,float force,float& lower,float& upper,float& nearest
) {
    // The product of two finite binary32 inputs is exactly representable in
    // binary64. Directed operations enclose the exact third product and sum.
    const double scale=__dmul_rn((double)dt,(double)inverse);
    const double lo=__dadd_rd((double)before,__dmul_rd(scale,(double)force));
    const double hi=__dadd_ru((double)before,__dmul_ru(scale,(double)force));
    lower=__double2float_rd(lo);upper=__double2float_ru(hi);
    nearest=__double2float_rn(__fma_rn(scale,(double)force,(double)before));
    // An interval straddling an exactly represented value is ambiguous here.
    // Refuse instead of silently selecting an exterior rounding candidate.
    return isfinite(lower)&&isfinite(upper)&&isfinite(nearest)&&
        (lower==upper||nextafterf(lower,CUDART_INF_F)==upper)&&
        (nearest==lower||nearest==upper);
}

// One warp owns one complete leaf; original velocities remain read-only until
// all eight leaves have passed. Three axes are independent for momentum closure.
__device__ __forceinline__ bool biprism_lif_round_velocity_leaf(
    const uint32_t lane,const BiprismLifCarryoverFfi& carry,
    const float* __restrict__ inverse,const float* __restrict__ before,
    const float dt,float3* __restrict__ staged
) {
    bool valid=true;
    for(uint32_t axis=0u;axis<3u;++axis){
        for(uint32_t v=lane;v<256u;v+=32u){
            const uint32_t atom=carry.atom_id[v];
            const float force=axis==0u?carry.force[v].x:(axis==1u?carry.force[v].y:carry.force[v].z);
            float lo,hi,nearest;
            valid = biprism_lif_velocity_bracket(before[3u*atom+axis],dt,inverse[atom],force,lo,hi,nearest)&&valid;
            reinterpret_cast<float*>(staged)[3u*v+axis]=nearest;
        }
        valid=__all_sync(0xffffffffu,valid);
        if(!valid)return false;
        __syncwarp();
        bool accepted=false;
        for(uint32_t iteration=0u;iteration<=256u;++iteration){
            double sum=0.0,absolute_sum=0.0;
            for(uint32_t v=lane;v<256u;v+=32u){
                const uint32_t atom=carry.atom_id[v];
                const double dp=((double)reinterpret_cast<const float*>(staged)[3u*v+axis]-(double)before[3u*atom+axis])/(double)inverse[atom];
                sum+=dp;absolute_sum+=fabs(dp);
            }
            sum=biprism_lif_warp_sum_f64(sum);
            absolute_sum=biprism_lif_warp_sum_f64(absolute_sum);
            // 512*epsilon64/2 dominates subtraction/division and any 256-term
            // reduction order, including the rounding of the absolute sum.
            const double bound=__dmul_ru(0x1p-44,absolute_sum);
            if(isfinite(sum)&&__dadd_ru(fabs(sum),bound)<=1.0e-7){accepted=true;break;}
            if(iteration==256u||!isfinite(sum))break;
            double best=fabs(sum);uint32_t best_vertex=0xffffffffu;
            for(uint32_t v=lane;v<256u;v+=32u){
                const uint32_t atom=carry.atom_id[v];
                const float force=axis==0u?carry.force[v].x:(axis==1u?carry.force[v].y:carry.force[v].z);
                float lo,hi,nearest;
                biprism_lif_velocity_bracket(before[3u*atom+axis],dt,inverse[atom],force,lo,hi,nearest);
                const float current=reinterpret_cast<const float*>(staged)[3u*v+axis];
                const float alternative=current==lo?hi:lo;
                const double proposed=fabs(sum+((double)alternative-(double)current)/(double)inverse[atom]);
                if(proposed<best||(proposed==best&&v<best_vertex)){
                    if(proposed<fabs(sum)){best=proposed;best_vertex=v;}
                }
            }
            for(uint32_t offset=16u;offset!=0u;offset>>=1u){
                const double other=__shfl_down_sync(0xffffffffu,best,offset);
                const uint32_t other_vertex=__shfl_down_sync(0xffffffffu,best_vertex,offset);
                if(other<best||(other==best&&other_vertex<best_vertex)){best=other;best_vertex=other_vertex;}
            }
            best_vertex=__shfl_sync(0xffffffffu,best_vertex,0);
            if(best_vertex==0xffffffffu)break;
            if(lane==(best_vertex&31u)){
                const uint32_t atom=carry.atom_id[best_vertex];
                const float force=axis==0u?carry.force[best_vertex].x:(axis==1u?carry.force[best_vertex].y:carry.force[best_vertex].z);
                float lo,hi,nearest;
                biprism_lif_velocity_bracket(before[3u*atom+axis],dt,inverse[atom],force,lo,hi,nearest);
                float& current=reinterpret_cast<float*>(staged)[3u*best_vertex+axis];
                current=current==lo?hi:lo;
            }
            __syncwarp();
        }
        if(!accepted)return false;
    }
    return true;
}

// Full lab-frame feedback kick owned by Biprism: schedule after successful v10
// and before the owner's RATTLE velocity projection. A single CTA prevalidates
// all eight leaves, then applies an adjacent enclosing FP32 representation of
// v0 + dt * true_inv_mass * produced_force once, certified against the unchanged
// 1e-7 physical momentum threshold before any global velocity mutation.
// The owner must provide disjoint velocity buffers and unique captured IDs in
// each leaf, keep all inputs exclusive, and use consistent force/mass/time units.
// This post-step kick is not a claim of Strang order, symplecticity, exact-zero
// momentum, or durable crash replay. delivered_step acknowledges this
// velocity update; it does not attest to downstream RATTLE or graph acceptance.
extern "C" __global__ __launch_bounds__(256, 2) void nhs_deliver_persisted_lif_forces_v1(
    const FoliatedStreamPtrTableFfi* __restrict__ d_stream_ptrs,
    BiprismLifCarryoverFfi* __restrict__ d_lif_carryover,
    const BiprismConvergenceRecordFfi* __restrict__ d_conv_records,
    const uint32_t* __restrict__ d_protocol_current_step,
    const uint32_t atom_count,
    const float arg_dt_feedback_au
) {
    const uint32_t tid = threadIdx.x;
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u ||
        gridDim.x != 1u || gridDim.y != 1u || gridDim.z != 1u ||
        d_stream_ptrs == nullptr || d_lif_carryover == nullptr ||
        d_conv_records == nullptr || d_protocol_current_step == nullptr ||
        atom_count == 0u) {
        if (d_lif_carryover != nullptr && blockIdx.x == 0u && tid == 0u) {
            for (uint32_t r = 0u; r < 8u; ++r) atomicOr(&d_lif_carryover[r].fault, BIPRISM_CARRY_BAD_INPUT);
        }
        return;
    }

    // Production consumes the same selected interval as v10 and the work
    // capture. At this boundary all four main substeps have already committed;
    // the shared accepted-body counter advances only after final admission.
    float dt_feedback_au = arg_dt_feedback_au;
    bool time_valid = false;
    if (d_stream_ptrs->physical_step_mode == 4u) {
        time_valid = d_stream_ptrs->d_shared_step == nullptr &&
            isfinite(dt_feedback_au) && dt_feedback_au > 0.0f;
    } else if (d_stream_ptrs->physical_step_mode == 0u && d_stream_ptrs->d_shared_step != nullptr) {
        const auto& selected = *d_stream_ptrs->d_shared_step;
        time_valid = biprism_shared_selected_valid(selected,
            *d_protocol_current_step, selected.generation);
        if (time_valid) {
            const uint32_t committed = (selected.accepted_bodies + 1u) * BIPRISM_INNER_SUBSTEPS;
            #pragma unroll 1
            for (uint32_t r = 0u; r < 8u; ++r) {
                time_valid = time_valid && d_conv_records[r].abi_magic == BIPRISM_TAIL_ABI_MAGIC &&
                    d_conv_records[r].leaf_id == r && d_conv_records[r].committed_substeps == committed;
            }
            dt_feedback_au = selected.outer_dt_au;
        }
    }
    if (!time_valid) {
        if (tid == 0u) {
            for (uint32_t r = 0u; r < 8u; ++r)
                atomicOr(&d_lif_carryover[r].fault, BIPRISM_CARRY_BAD_INPUT);
        }
        return;
    }

    __shared__ uint32_t s_fault[8];
    __shared__ uint32_t s_pending_count;
    __shared__ uint32_t s_step;
    if (tid == 0u) {
        s_step = *d_protocol_current_step;
        s_pending_count = 0u;
        for (uint32_t r = 0u; r < 8u; ++r) s_pending_count += (d_lif_carryover[r].pending == 1u);
    }
    __syncthreads();
    if (tid < 8u) {
        const BiprismLifCarryoverFfi& carry = d_lif_carryover[tid];
        uint32_t fault = carry.fault;
        if (d_conv_records[tid].abi_magic != BIPRISM_TAIL_ABI_MAGIC ||
            d_conv_records[tid].leaf_id != tid) fault |= BIPRISM_CARRY_BAD_INPUT;
        if (d_conv_records[tid].terminal_fault_latch != 0u ||
            d_conv_records[tid].epoch_leaf_fault != 0u ||
            d_conv_records[tid].epoch_cluster_fault != 0u) fault |= BIPRISM_CARRY_UPSTREAM_FAULT;
        if (carry.pending > 1u || d_stream_ptrs->d_vels_xyz[tid] == nullptr ||
            d_stream_ptrs->d_true_inv_masses[tid] == nullptr) fault |= BIPRISM_CARRY_BAD_INPUT;
        if (s_pending_count != 0u && s_pending_count != 8u) fault |= BIPRISM_CARRY_MIXED_PENDING;
        if (carry.pending == 1u &&
            (carry.produced_step == 0xffffffffu || carry.produced_step + 1u != s_step)) {
            fault |= BIPRISM_CARRY_STALE_EPOCH;
        }
        // Main v10 increments this counter by exactly4 only after collective success.
        // A clean but stale/zero convergence record cannot acknowledge this delivery.
        if (carry.pending == 1u &&
            (carry.produced_committed_substeps < BIPRISM_INNER_SUBSTEPS ||
             carry.produced_committed_substeps > 0xffffffffu - BIPRISM_INNER_SUBSTEPS ||
             d_conv_records[tid].committed_substeps != carry.produced_committed_substeps + BIPRISM_INNER_SUBSTEPS)) {
            fault |= BIPRISM_CARRY_STALE_EPOCH;
        }
        // Leaf buffers must not overlap: all preflight reads precede any write.
        const uintptr_t begin = reinterpret_cast<uintptr_t>(d_stream_ptrs->d_vels_xyz[tid]);
        const uintptr_t bytes = static_cast<uintptr_t>(atom_count) * 3u * sizeof(float);
        const uintptr_t end = begin + bytes;
        if (end < begin) fault |= BIPRISM_CARRY_BAD_INPUT;
        for (uint32_t r = 0u; r < tid; ++r) {
            const uintptr_t other = reinterpret_cast<uintptr_t>(d_stream_ptrs->d_vels_xyz[r]);
            const uintptr_t other_end = other + bytes;
            if (begin < other_end && other < end) fault |= BIPRISM_CARRY_BAD_INPUT;
        }
        s_fault[tid] = fault;
    }
    __syncthreads();

    for (uint32_t r = 0u; r < 8u; ++r) {
        const BiprismLifCarryoverFfi& carry = d_lif_carryover[r];
        if (carry.pending == 1u) {
            const uint32_t atom = carry.atom_id[tid];
            const float4 force = carry.force[tid];
            uint32_t fault = 0u;
            if (atom >= atom_count || force.w != 0.0f ||
                !isfinite(force.x) || !isfinite(force.y) || !isfinite(force.z)) {
                fault |= BIPRISM_CARRY_BAD_INPUT;
            }
            for (uint32_t other = tid + 1u; other < TILE_TOTAL_VERTICES; ++other) {
                if (carry.atom_id[other] == atom) fault |= BIPRISM_CARRY_BAD_INPUT;
            }
            const float* inv_masses = d_stream_ptrs->d_true_inv_masses[r];
            const float* velocities = d_stream_ptrs->d_vels_xyz[r];
            if (atom < atom_count && inv_masses != nullptr && velocities != nullptr) {
                const float inv_mass = inv_masses[atom];
                const float dt_inv_mass = __fmul_rn(dt_feedback_au, inv_mass);
                const float3 dv = {__fmul_rn(dt_inv_mass, force.x), __fmul_rn(dt_inv_mass, force.y), __fmul_rn(dt_inv_mass, force.z)};
                const float3 before = *reinterpret_cast<const float3*>(&velocities[atom * 3u]);
                if (!isfinite(inv_mass) || !(inv_mass > 0.0f) || !isfinite(dt_inv_mass) ||
                    !isfinite(dv.x) || !isfinite(dv.y) || !isfinite(dv.z) ||
                    !isfinite(before.x) || !isfinite(before.y) || !isfinite(before.z) ||
                    !isfinite(__fadd_rn(before.x, dv.x)) || !isfinite(__fadd_rn(before.y, dv.y)) ||
                    !isfinite(__fadd_rn(before.z, dv.z))) fault |= BIPRISM_CARRY_NUMERIC_FAULT;
            }
            if (fault != 0u) atomicOr(&s_fault[r], fault);
        }
    }
    __syncthreads();
    if (tid < 8u && s_fault[tid] != 0u) atomicOr(&d_lif_carryover[tid].fault, s_fault[tid]);
    const bool rejected = (__syncthreads_or(tid < 8u && s_fault[tid] != 0u) != 0);
    if (rejected || s_pending_count == 0u) return;

    __shared__ float3 s_velocity[8][256];
    const uint32_t leaf=tid>>5u;
    const bool rounded=biprism_lif_round_velocity_leaf(tid&31u,d_lif_carryover[leaf],
        d_stream_ptrs->d_true_inv_masses[leaf],d_stream_ptrs->d_vels_xyz[leaf],
        dt_feedback_au,s_velocity[leaf]);
    if(!rounded&&(tid&31u)==0u){
        s_fault[leaf]|=BIPRISM_CARRY_NUMERIC_FAULT;
        atomicOr(&d_lif_carryover[leaf].fault,BIPRISM_CARRY_NUMERIC_FAULT);
    }
    // One failed representation withholds all eight leaves and leaves pending
    // untouched. Rejection telemetry is the only mutation on this path.
    if(__syncthreads_or(!rounded))return;
    for (uint32_t r = 0u; r < 8u; ++r) {
        const uint32_t atom = d_lif_carryover[r].atom_id[tid];
        *reinterpret_cast<float3*>(&d_stream_ptrs->d_vels_xyz[r][atom*3u])=s_velocity[r][tid];
    }
    __threadfence();
    __syncthreads();
    if (tid < 8u) {
        d_lif_carryover[tid].delivered_step = s_step;
        d_lif_carryover[tid].pending = 0u;
    }
}


extern "C" __global__ __launch_bounds__(256, 2) void nhs_voronoi_dual_lif_and_4grid_bypass_v2(
    const FoliatedStreamPtrTableFfi* __restrict__ d_stream_ptrs,
    const ContactShellSimplicialStateFfi* __restrict__ d_tile_state,
    const BiprismConvergenceRecordFfi* __restrict__ d_conv_records,
    const uint32_t* __restrict__ d_protocol_current_step,
    const int32_t* __restrict__ d_atom_residue_ids,
    float* __restrict__ d_voronoi_lif_voltage,
    int32_t* __restrict__ d_voxel_hit_grid,                // Integer Grid 1 (`nhs_rt_full.rs:9439`)
    int32_t* __restrict__ d_coupled_spike_grid,            // Integer Grid 2 (`nhs_rt_full.rs:9439`)
    int32_t* __restrict__ d_primary_residue_id_grid,       // Integer Grid 3 (`nhs_rt_full.rs:9439`)
    int32_t* __restrict__ d_primary_residue_count_grid,    // Integer Grid 4 (`nhs_rt_full.rs:9439`)
    float* __restrict__ d_water_density_grid,               // Float Observer Grid 1 (`fused_engine.rs:21922`)
    float* __restrict__ d_exclusion_field_grid,             // Float Observer Grid 2 (`fused_engine.rs:21922`)
    GpuSpikeEventCanonical96* __restrict__ d_spike_queue,
    uint32_t* __restrict__ d_spike_queue_count,
    const SignalGridSpecFfi grid_spec,
    const uint32_t max_spikes,
    const float lif_decay,
    const float lif_threshold,
    const float spike_force_kick,
    BiprismLifCarryoverFfi* __restrict__ d_lif_carryover,
    const uint32_t deliver_immediate_to_forces
) {
    // 9,216-byte payload plus CTA rejection/reservation metadata; ptxas allocates 9,344 bytes.
    // Rejection here is per CTA; it does not roll back another leaf's writes.
    __shared__ uint32_t s_input_fault;
    __shared__ uint32_t s_round_shift;
    __shared__ uint32_t s_round_up;
    __shared__ uint32_t s_event_base;
    __shared__ uint32_t s_event_offset;
    __shared__ uint32_t s_event_batch_ok;
    __shared__ float              s_v_mem[TILE_TOTAL_VERTICES];             // 1,024 B
    __shared__ float              s_syn_flux[TILE_TOTAL_VERTICES];          // 1,024 B
    __shared__ float              s_spike_fp[TILE_TOTAL_VERTICES];          // 1,024 B
    __shared__ unsigned long long s_coupled_force_q39[TILE_TOTAL_VERTICES][3]; // 6,144 B (Core 0..191 + Halo 192..255)

    const uint32_t tid     = threadIdx.x;
    const uint32_t leaf_id = blockIdx.x & 7u;
    const bool     is_core = (tid < TILE_CORE_VERTICES);

    // All predicates are uniform within the CTA and precede global mutation.
    if (blockDim.x != 256u || blockDim.y != 1u || blockDim.z != 1u ||
        gridDim.x != 8u || gridDim.y != 1u || gridDim.z != 1u ||
        d_stream_ptrs == nullptr || d_tile_state == nullptr || d_conv_records == nullptr ||
        d_protocol_current_step == nullptr || d_atom_residue_ids == nullptr ||
        d_voronoi_lif_voltage == nullptr || d_voxel_hit_grid == nullptr ||
        d_coupled_spike_grid == nullptr || d_primary_residue_id_grid == nullptr ||
        d_primary_residue_count_grid == nullptr || d_spike_queue == nullptr ||
        d_spike_queue_count == nullptr || d_lif_carryover == nullptr ||
        deliver_immediate_to_forces > 1u || !isfinite(lif_threshold) ||
        !(lif_threshold > 0.0f) || !isfinite(lif_decay) || !isfinite(spike_force_kick)) {
        if (d_lif_carryover != nullptr && tid == 0u) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_BAD_INPUT);
        return;
    }

    const uint32_t produced_step = *d_protocol_current_step;
    uint32_t upstream_fault = 0u;
    uint32_t upstream_bad_abi = 0u;
    for (uint32_t r = 0u; r < 8u; ++r) {
        upstream_fault |= d_conv_records[r].terminal_fault_latch | d_conv_records[r].epoch_leaf_fault | d_conv_records[r].epoch_cluster_fault;
        upstream_bad_abi |= (d_conv_records[r].abi_magic != BIPRISM_TAIL_ABI_MAGIC || d_conv_records[r].leaf_id != r) ? 1u : 0u;
        if(!d_stream_ptrs->d_coordinate_owners||(reinterpret_cast<uintptr_t>(d_stream_ptrs->d_coordinate_owners)&15u))upstream_bad_abi|=1u;
        else{const auto& o=d_stream_ptrs->d_coordinate_owners[r];
            if(o.atom_count<256u||o.atom_count>0x01000000u||!bp_coord_owner_valid(&o,r,o.atom_count,
                o.generation,BP_COORD_ATOMIC_BOHR,BP_COORD_ALL_INITIALIZED,d_stream_ptrs->d_coords_xyz[r]))upstream_bad_abi|=1u;}
        if(!d_stream_ptrs->d_potential_owners||
            (reinterpret_cast<uintptr_t>(d_stream_ptrs->d_potential_owners)&127u)||
            !d_stream_ptrs->d_coordinate_owners)upstream_bad_abi|=1u;
        else{const auto& potential=d_stream_ptrs->d_potential_owners[r];
            if(!bp_potential_owner_valid(&potential,r,d_tile_state+r,d_stream_ptrs->d_coordinate_owners+r)
                ||!bp_potential_accepted_valid(potential,d_conv_records[r].committed_substeps,produced_step))upstream_bad_abi|=1u;
            else if(!bp_potential_projection_value_valid(potential,tid,d_tile_state[r].phi_dec[tid]))upstream_bad_abi|=1u;}
    }
    // Every CTA checks all eight immutable owners and all256 projection rows
    // before any observation write. A bad potential cannot admit other leaves.
    if (__syncthreads_or(upstream_bad_abi != 0u)) {
        if (tid == 0u) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_BAD_INPUT);
        return;
    }
    if (upstream_fault != 0u) {
        if (tid == 0u) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_UPSTREAM_FAULT);
        return;
    }
    const uint32_t produced_substeps = d_conv_records[leaf_id].committed_substeps;
    if (produced_substeps < BIPRISM_INNER_SUBSTEPS ||
        produced_substeps > 0xffffffffu - BIPRISM_INNER_SUBSTEPS) {
        if (tid == 0u) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_STALE_EPOCH);
        return;
    }
    if (d_lif_carryover[leaf_id].fault != 0u) return;
    if (d_lif_carryover[leaf_id].pending != 0u) {
        if (tid == 0u) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_PENDING_OVERWRITE);
        return;
    }
    if (produced_step == 0xffffffffu || d_lif_carryover[leaf_id].produced_step > produced_step ||
        d_lif_carryover[leaf_id].delivered_step > produced_step) {
        if (tid == 0u) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_STALE_EPOCH);
        return;
    }

    const ContactShellSimplicialStateFfi* tile = &d_tile_state[leaf_id];
    const auto& coordinates=d_stream_ptrs->d_coordinate_owners[leaf_id];
    const float* __restrict__ d_coords_xyz = d_stream_ptrs->d_coords_xyz[leaf_id];
    float* __restrict__       d_forces_xyz = d_stream_ptrs->d_forces_xyz[leaf_id];

    if (d_coords_xyz == nullptr || d_forces_xyz == nullptr) {
        if (tid == 0u) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_BAD_INPUT);
        return;
    }
    if (tid == 0u) s_input_fault = 0u;
    const uint32_t v_idx = leaf_id * TILE_TOTAL_VERTICES + tid;
    s_v_mem[tid]                 = d_voronoi_lif_voltage[v_idx];
    s_syn_flux[tid]              = 0.0f;
    s_spike_fp[tid]              = 0.0f;
    s_coupled_force_q39[tid][0]  = 0ull;
    s_coupled_force_q39[tid][1]  = 0ull;
    s_coupled_force_q39[tid][2]  = 0ull;
    __syncthreads();

    if (!isfinite(s_v_mem[tid])) atomicOr(&s_input_fault, 32u);
    const uint32_t checked_atom=UNPACK_GLOBAL_ATOM_ID(tile->global_atom_idx[tid]);
    const auto checked_position=bp_coord_load(coordinates,BiprismCoordinatePlane::Current,checked_atom);
    const auto checked_reference=bp_coord_load(coordinates,BiprismCoordinatePlane::MeshReference,checked_atom);
    if(!bp_coord_finite(checked_position)||!bp_coord_finite(checked_reference)
        ||__float_as_uint(checked_reference.x.hi)!=__float_as_uint(tile->ref_pos_q[tid].x)
        ||__float_as_uint(checked_reference.y.hi)!=__float_as_uint(tile->ref_pos_q[tid].y)
        ||__float_as_uint(checked_reference.z.hi)!=__float_as_uint(tile->ref_pos_q[tid].z))atomicOr(&s_input_fault,32u);
    __syncthreads();
    if(s_input_fault){if(tid==0u)atomicOr(&d_lif_carryover[leaf_id].fault,BIPRISM_CARRY_BAD_INPUT);return;}

    // 1. Discrete DEC synaptic diffusion with fixed halo voltages.
    const uint32_t num_edges = min(tile->num_active_edges, (uint32_t)TILE_MAX_EDGES);
    for (uint32_t e = tid; e < num_edges; e += 256) {
        const uint8_t u = tile->edge_u8[e][0];
        const uint8_t v = tile->edge_u8[e][1];
        float3 grad_unused;
        const float w_live = biprism_live_diagonal_edge(tile, coordinates, e, grad_unused);
        const float flux = w_live * (s_v_mem[v] - s_v_mem[u]);
        if (!isfinite(flux)) atomicOr(&s_input_fault, 1u);
        if (u < TILE_CORE_VERTICES) atomicAdd(&s_syn_flux[u],  flux);
        if (v < TILE_CORE_VERTICES) atomicAdd(&s_syn_flux[v], -flux);
    }
    __syncthreads();

    // 2. Evaluate Exact Sigma-Delta (\Sigma\Delta) Multiplicity Ledger: V_{pre,u} = m_u * \theta_{th} + r_u
    const uint32_t packed_id   = tile->global_atom_idx[tid];
    const uint32_t my_g_atom   = UNPACK_GLOBAL_ATOM_ID(packed_id);
    const uint32_t my_elem_cls = UNPACK_ELEM_CLASS(packed_id);

    float    my_v_pre     = 0.0f;
    float    my_q_emit    = 0.0f;
    float    my_q_low     = 0.0f;
    float    my_r_post    = 0.0f;
    uint32_t my_mult_m    = 0u;
    float    my_hol_drive = 0.0f;
    float    my_star0     = 1.0f;
    float    my_phi       = 0.0f;
    int32_t  my_res_id    = -1;
    float3   my_pos       = {0.0f, 0.0f, 0.0f};
    const float th_safe   = lif_threshold; // Preserve the finite positive caller threshold.

    if (is_core) {
        my_res_id    = d_atom_residue_ids[my_g_atom];
        my_pos       = *reinterpret_cast<const float3*>(&d_coords_xyz[my_g_atom * 3u]);
        my_star0     = fmaxf(tile->star0_vol[tid], 1e-4f);
        if (!isfinite(my_pos.x) || !isfinite(my_pos.y) || !isfinite(my_pos.z) ||
            !isfinite(tile->star0_vol[tid])) atomicOr(&s_input_fault, 32u);
        // LIF is an FP32 observer. Its explicit RN32 input projection comes
        // from the accepted FP64 owner, never from a stale Phase2 tile solve.
        my_phi       = __double2float_rn(d_stream_ptrs->d_potential_owners[leaf_id].accepted_phi[tid]);
        my_hol_drive = fmaxf(2.0f - d_conv_records[leaf_id].w_x_lambda_trace, 0.0f);

        const cpx2_f32 psi1 = tile->psi_vertex[tid][1];
        const float pop1    = __fmaf_rn(psi1.re, psi1.re, psi1.im * psi1.im);

        my_v_pre = lif_decay * s_v_mem[tid]
                 + (0.15f / my_star0) * s_syn_flux[tid]
                 + (fabsf(my_phi) + 0.25f * pop1) * (1.0f + my_hol_drive);

        if (!biprism_sigma_delta_partition(my_v_pre, th_safe, my_mult_m,
                                           my_q_emit, my_q_low, my_r_post)) {
            atomicOr(&s_input_fault, 2u);
        } else if (my_mult_m > 0u) {
            s_spike_fp[tid] = my_q_emit * (1.0f + 0.5f * my_hol_drive);
            if (!isfinite(s_spike_fp[tid])) atomicOr(&s_input_fault, 4u);
        }
        const int32_t checked_cell = map_voronoi_vertex_to_signal_cell_precise(
            bp_coord_value(bp_coord_load(coordinates,BiprismCoordinatePlane::Current,my_g_atom)),grid_spec);
        if (checked_cell >= 0 && static_cast<uint32_t>(checked_cell) < grid_spec.total_voxels) {
            const int32_t hits = atomicCAS(&d_voxel_hit_grid[checked_cell], 0, 0);
            const int32_t coupled = atomicCAS(&d_coupled_spike_grid[checked_cell], 0, 0);
            const int32_t owned = atomicCAS(&d_primary_residue_count_grid[checked_cell], 0, 0);
            const int32_t owner = atomicCAS(&d_primary_residue_id_grid[checked_cell], -1, -1);
            // The launch-wide headroom bound is a host precondition. Do not compare
            // an evolving concurrent baseline against that same initial bound here.
            if (hits < 0 || coupled < 0 || owned < 0 || owner < -1) {
                atomicOr(&s_input_fault, 8u);
            }
        }
    }
    __syncthreads();

    // 3. Pairwise Floating-Point Spike Algebra Across ALL Edges (Core-Core, Core-Halo, Halo-Halo)
    //    Accumulates +df into `u` and -df into `v` across all 256 vertices in Q24.39 fixed point
    //    Exact pair cancellation applies to the checked integer accumulator domain;
    //    separate FP32 conversion/writeback is assessed independently.
    for (uint32_t e = tid; e < num_edges; e += 256) {
        const uint8_t u = tile->edge_u8[e][0];
        const uint8_t v = tile->edge_u8[e][1];
        const float pu  = s_spike_fp[u];
        const float pv  = s_spike_fp[v];
        if (pu > 0.0f || pv > 0.0f) {
            const float p_pair = 0.5f * (pu + pv) + 0.25f * (pu * pv);
            float3 gs;
            (void)biprism_live_diagonal_edge(tile, coordinates, e, gs);
            const float3 df    = {
                spike_force_kick * p_pair * gs.x,
                spike_force_kick * p_pair * gs.y,
                spike_force_kick * p_pair * gs.z
            };
            long long kx, ky, kz;
            if (!biprism_quantize_edge_q39(df.x, kx) ||
                !biprism_quantize_edge_q39(df.y, ky) ||
                !biprism_quantize_edge_q39(df.z, kz)) {
                atomicOr(&s_input_fault, 16u);
            } else {
                const unsigned long long ux = static_cast<unsigned long long>(kx);
                const unsigned long long uy = static_cast<unsigned long long>(ky);
                const unsigned long long uz = static_cast<unsigned long long>(kz);
                atomicAdd(&s_coupled_force_q39[u][0], ux);
                atomicAdd(&s_coupled_force_q39[u][1], uy);
                atomicAdd(&s_coupled_force_q39[u][2], uz);
                atomicAdd(&s_coupled_force_q39[v][0], 0ull - ux);
                atomicAdd(&s_coupled_force_q39[v][1], 0ull - uy);
                atomicAdd(&s_coupled_force_q39[v][2], 0ull - uz);
            }
        }
    }
    __syncthreads();

    const float3 f_cpl = {
        biprism_balanced_q39_axis_f32(s_coupled_force_q39, 0u, s_round_shift, s_round_up, s_input_fault),
        biprism_balanced_q39_axis_f32(s_coupled_force_q39, 1u, s_round_shift, s_round_up, s_input_fault),
        biprism_balanced_q39_axis_f32(s_coupled_force_q39, 2u, s_round_shift, s_round_up, s_input_fault)
    };
    const float f_l2 = __fmaf_rn(f_cpl.x, f_cpl.x, __fmaf_rn(f_cpl.y, f_cpl.y, f_cpl.z * f_cpl.z));
    if (deliver_immediate_to_forces != 0u) {
        const float3 before = *reinterpret_cast<const float3*>(&d_forces_xyz[my_g_atom * 3u]);
        if (!isfinite(before.x) || !isfinite(before.y) || !isfinite(before.z) ||
            !isfinite(__fadd_rn(before.x, f_cpl.x)) || !isfinite(__fadd_rn(before.y, f_cpl.y)) ||
            !isfinite(__fadd_rn(before.z, f_cpl.z))) atomicOr(&s_input_fault, 64u);
    }
    __syncthreads();
    // Fault telemetry may change on rejection; physical/grid/carry payloads do not.
    if (s_input_fault != 0u) {
        if (tid == 0u) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_NUMERIC_FAULT);
        return;
    }
    const uint32_t emit_count = static_cast<uint32_t>(__syncthreads_count(is_core && my_mult_m > 0u));
    if (tid == 0u) {
        s_event_offset = 0u;
        s_event_batch_ok = biprism_reserve_event_batch(
            d_spike_queue_count, max_spikes, emit_count, s_event_base) ? 1u : 0u;
    }
    __syncthreads();
    if (s_event_batch_ok == 0u) {
        if (tid == 0u) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_QUEUE_CAPACITY);
        return;
    }

    // 4A. Deposit the balanced represented vector across the unique256-atom tile.
    // The vector sums exactly to zero; adding to a nonzero FP32 baseline can round.
    if (deliver_immediate_to_forces != 0u && f_l2 > 0.0f) {
        atomicAdd(&d_forces_xyz[my_g_atom * 3u + 0u], f_cpl.x);
        atomicAdd(&d_forces_xyz[my_g_atom * 3u + 1u], f_cpl.y);
        atomicAdd(&d_forces_xyz[my_g_atom * 3u + 2u], f_cpl.z);
    }

    // Dedicated carryover stores every endpoint, including exact zero forces.
    d_lif_carryover[leaf_id].atom_id[tid] = my_g_atom;
    d_lif_carryover[leaf_id].force[tid] = make_float4(f_cpl.x, f_cpl.y, f_cpl.z, 0.0f);

    // 4B. Core Voronoi Ledger Commit, 96B Spike Emission, and Multiplicity-Weighted 4-Grid Update
    if (is_core) {
        const int32_t cell_idx = map_voronoi_vertex_to_signal_cell_precise(
            bp_coord_value(bp_coord_load(coordinates,BiprismCoordinatePlane::Current,my_g_atom)),grid_spec);
        const bool valid_cell  = (cell_idx >= 0 && (uint32_t)cell_idx < grid_spec.total_voxels);

        if (valid_cell) {
            if (d_water_density_grid != nullptr) {
                atomicAdd(&d_water_density_grid[cell_idx], my_star0 * 0.125f);
            }
            if (d_exclusion_field_grid != nullptr) {
                const float excl_val = fabsf(c_gfn_xtb_lut[my_elem_cls & 7u].h_s_valence)
                                     + fabsf(my_phi) + 0.5f * s_spike_fp[tid];
                atomicAdd(&d_exclusion_field_grid[cell_idx], excl_val * 0.125f);
            }
        }

        if (my_mult_m > 0u) {
            const uint32_t cur_step = produced_step;

            {
                const uint32_t slot = s_event_base + atomicAdd(&s_event_offset, 1u);
                { // Every slot in this CTA's batch was reserved before state commit.
                    GpuSpikeEventCanonical96 ev = {};
                    ev.pos_xyz[0]         = my_pos.x;
                    ev.pos_xyz[1]         = my_pos.y;
                    ev.pos_xyz[2]         = my_pos.z;
                    ev.step_tau           = cur_step;
                    ev.atom_or_site_id    = my_g_atom;
                    ev.spike_amplitude    = s_spike_fp[tid];
                    ev.uv_or_holonomy_val = my_hol_drive;
                    ev.voronoi_dual_vol   = my_star0;
                    ev.pocket_tile_id     = tile->active_pocket_id;
                    ev.residue_id         = my_res_id;
                    // Explicit Sigma-Delta Ledger Fields: V_{pre} == m_u * \theta_{th} + r_{post}
                    ev.multiplicity_m     = my_mult_m;
                    ev.v_pre_reset        = my_v_pre;
                    ev.q_discharged       = my_q_emit;
                    ev.r_post_reset       = my_r_post;
                    ev.lif_threshold_used = th_safe;
                    ev.q_discharged_lo     = my_q_low;
                    ev.leaf_id             = static_cast<uint16_t>(leaf_id);
                    ev.vertex_id           = static_cast<uint16_t>(tid);
                    ev.voxel_cell_idx      = valid_cell ? cell_idx : -1;
                    ev.event_schema_magic  = SPIKE_EVENT_V10_SCHEMA_MAGIC;
                    ev.coupled_force_xyz[0] = f_cpl.x;
                    ev.coupled_force_xyz[1] = f_cpl.y;
                    ev.coupled_force_xyz[2] = f_cpl.z;
                    ev.q_lo_product_raw_bits = __float_as_uint(my_q_low);
                    d_spike_queue[slot]   = ev;
                }
            }

            // Hits sum multiplicity; coupled uses this candidate's force condition.
            // Primary ownership is a signed -1-only claim; excluded residue IDs never claim.
            const bool is_coupled = (f_l2 > 0.0f);
            const bool hit_ok = !valid_cell || biprism_add_count_i32_checked(&d_voxel_hit_grid[cell_idx], my_mult_m);
            if (!hit_ok) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_COUNT_RANGE);
            if (valid_cell && hit_ok) {
                if (is_coupled) {
                    if (!biprism_add_count_i32_checked(&d_coupled_spike_grid[cell_idx], my_mult_m)) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_COUNT_RANGE);
                }
                if (my_res_id >= 0) {
                    const int32_t prev_res = atomicCAS(&d_primary_residue_id_grid[cell_idx], -1, my_res_id);
                    if (prev_res == -1 || prev_res == my_res_id) {
                        if (!biprism_add_count_i32_checked(&d_primary_residue_count_grid[cell_idx], my_mult_m)) atomicOr(&d_lif_carryover[leaf_id].fault, BIPRISM_CARRY_COUNT_RANGE);
                    }
                }
            }
        }

        // Firing states retain 0 <= r < threshold; finite nonfiring inhibitory states remain unchanged.
        d_voronoi_lif_voltage[v_idx] = my_r_post;
    }
    // The exclusive owner observes pending only after this kernel/stream completes.
    // A nonzero fault preserves the payload for diagnosis and prevents delivery.
    __threadfence();
    __syncthreads();
    if (tid == 0u) {
        d_lif_carryover[leaf_id].produced_step = produced_step;
        d_lif_carryover[leaf_id].produced_committed_substeps = produced_substeps;
        d_lif_carryover[leaf_id].reserved[0] = 0u;
        d_lif_carryover[leaf_id].reserved[1] = 0u;
        d_lif_carryover[leaf_id].reserved[2] = 0u;
        d_lif_carryover[leaf_id].pending = 1u;
    }
}

#endif // NHS_BIPRISM_VORONOI_LIF_CUH_INCLUDED

