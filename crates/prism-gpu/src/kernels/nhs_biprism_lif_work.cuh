#pragma once
#include <stdint.h>
#include <stddef.h>
#include <math.h>

// Measurement of the actual lab-impulse delivery. These records do not alter
// the requested force law, velocity rounding algorithm, or acceptance bounds.
struct alignas(16) BiprismLifWorkBindingFfi {
    const float* native_hmr_mass_dalton;
    double* native_atom_work_kcal;
};
struct alignas(16) BiprismLifWorkRowFfi {
    uint32_t atom_id, leaf_id, local_vertex, state;
    uint32_t scheduled_step, committed_substeps, source_produced_step, pending_before;
    float true_inverse_mass_au, propagation_inverse_mass_au, represented_main_mass_au, native_hmr_mass_dalton;
    float metric_position_au[3], native_position_angstrom[3];
    float accepted_main_velocity[3];
    uint32_t pre_lif_scope; //1 flexible core bridge;2 constrained core includes native projection;0 halo
    float au_velocity_before[3], au_velocity_after[3], native_velocity_before[3], native_velocity_after[3];
    float requested_force_au[3], accepted_metric[6];
    uint32_t reserved;
    double requested_impulse[3], realized_lab_impulse[3], realized_true_covariant_impulse[3];
    // Order is mirrored by the summary record and Rust enum below.
    double requested_lab_work, realized_lab_work, lab_kinetic_delta;
    double requested_true_metric_work, realized_true_metric_work, true_metric_kinetic_delta;
    double main_mass_metric_kinetic_delta, native_hmr_kinetic_delta_kcal;
    double native_true_lab_kinetic_delta, lab_bridge_kinetic_difference;
    double native_true_metric_kinetic_delta, metric_bridge_kinetic_difference;
    double main_vs_true_metric_mass_difference;
    double native_phase_hmr_kinetic_delta_kcal,native_phase_vs_guard_work_difference_kcal;
    double pre_lif_true_metric_delta,pre_lif_main_metric_delta;
    // BLW3 retains the complete position used to realize the accepted metric.
    // Existing fields remain the high planes; lows are not physical work.
    float metric_position_au_lo[3],native_position_angstrom_lo[3];
    uint32_t coordinate_reserved[2];
    double accepted_metric_f64[6];
};
struct alignas(16) BiprismLifWorkLedgerFfi {
    uint32_t abi_magic, leaf_id, state, sticky_fault;
    uint32_t scheduled_step, committed_substeps, produced_step, delivered_step_before;
    uint32_t pending_before, mode, reserved[2];
    uint64_t completed_steps, body_ordinal;
    double dt_feedback_au;
    double totals[17];
    double requested_impulse[3], realized_lab_impulse[3], realized_true_covariant_impulse[3];
    uint64_t reserved_tail;
};
struct alignas(16) BiprismLifWorkWindowFfi {
    uint32_t abi_magic, capacity, first_body, requested_steps;
    uint32_t first_scheduled_step, reserved[3];
};
static_assert(sizeof(BiprismLifWorkRowFfi)==464,"BLW3 LIF work row ABI");
static_assert(offsetof(BiprismLifWorkRowFfi,metric_position_au_lo)==384,"BLW3 position expansion ABI");
static_assert(offsetof(BiprismLifWorkRowFfi,requested_impulse)==176,"LIF work impulse ABI");
static_assert(offsetof(BiprismLifWorkRowFfi,requested_lab_work)==248,"LIF work scalars ABI");
static_assert(offsetof(BiprismLifWorkRowFfi,accepted_metric_f64)==416,"BLW3 FP64 covariance");
static_assert(sizeof(BiprismLifWorkLedgerFfi)==288,"LIF work ledger ABI");
static_assert(offsetof(BiprismLifWorkLedgerFfi,totals)==72,"LIF summary scalars ABI");
static_assert(sizeof(BiprismLifWorkWindowFfi)==32,"LIF work window ABI");
static constexpr uint32_t BIPRISM_LIF_WORK_MAGIC=0x424c5733u;
static constexpr uint32_t BIPRISM_LIF_WORK_CAPTURED=1u, BIPRISM_LIF_WORK_MEASURED=2u, BIPRISM_LIF_WORK_SEALED=3u;
static constexpr uint32_t BIPRISM_LIF_WORK_DOMAIN=1u,BIPRISM_LIF_WORK_EPOCH=2u,BIPRISM_LIF_WORK_NONFINITE=4u;
static constexpr uint32_t BIPRISM_LIF_WORK_TERMINAL=0x08000000u;
#if defined(__CUDACC__)
#define BP_LW_HD __host__ __device__
#else
#define BP_LW_HD
#endif
BP_LW_HD inline bool biprism_lif_work_slot(const BiprismLifWorkWindowFfi& w,
    uint64_t body,uint32_t scheduled,uint32_t* out) {
    if(w.abi_magic!=BIPRISM_LIF_WORK_MAGIC||!w.capacity||w.capacity>500u||!w.requested_steps
        ||w.requested_steps>w.capacity||w.reserved[0]||w.reserved[1]||w.reserved[2]
        ||body<w.first_body||body-w.first_body>=w.requested_steps
        ||uint64_t(w.first_scheduled_step)+(body-w.first_body)>=UINT32_MAX
        ||scheduled!=uint64_t(w.first_scheduled_step)+(body-w.first_body))return false;
    *out=uint32_t(body-w.first_body);return true;
}
BP_LW_HD inline void biprism_lif_work_metric_mul(const double g[6],const double v[3],double out[3]) {
    out[0]=double(g[0])*v[0]+double(g[3])*v[1]+double(g[4])*v[2];
    out[1]=double(g[3])*v[0]+double(g[1])*v[1]+double(g[5])*v[2];
    out[2]=double(g[4])*v[0]+double(g[5])*v[1]+double(g[2])*v[2];
}
BP_LW_HD inline bool biprism_lif_work_metric_spd(const double g[6]) {
    for(int k=0;k<6;++k)if(!isfinite(g[k]))return false;
    const double a=g[0],b=g[1],c=g[2],d=g[3],e=g[4],f=g[5];
    const double minor=a*b-d*d;
    return a>0.0&&minor>0.0&&a*(b*c-f*f)-d*(d*c-e*f)+e*(d*f-b*e)>0.0;
}
BP_LW_HD inline void biprism_lif_work_values(const BiprismLifWorkRowFfi& r,double out[17]) {
    out[0]=r.requested_lab_work;out[1]=r.realized_lab_work;out[2]=r.lab_kinetic_delta;
    out[3]=r.requested_true_metric_work;out[4]=r.realized_true_metric_work;out[5]=r.true_metric_kinetic_delta;
    out[6]=r.main_mass_metric_kinetic_delta;out[7]=r.native_hmr_kinetic_delta_kcal;
    out[8]=r.native_true_lab_kinetic_delta;out[9]=r.lab_bridge_kinetic_difference;
    out[10]=r.native_true_metric_kinetic_delta;out[11]=r.metric_bridge_kinetic_difference;
    out[12]=r.main_vs_true_metric_mass_difference;
    out[13]=r.native_phase_hmr_kinetic_delta_kcal;out[14]=r.native_phase_vs_guard_work_difference_kcal;
    out[15]=r.pre_lif_true_metric_delta;out[16]=r.pre_lif_main_metric_delta;
}
// Pure represented-input calculation, exercised by the actual CPU reference.
// main_mass is the actual FP32 reciprocal/floor result used by C4, not a
// relabelled FP64 reciprocal. The native HMR divisor matches the actual physical governor energy evaluator.
// The different promoted418.4f phase-work convention is retained separately.
BP_LW_HD inline bool biprism_lif_work_calculate(BiprismLifWorkRowFfi& r,double dt,double native_velocity_to_au) {
    if(!isfinite(dt)||dt<=0||!isfinite(native_velocity_to_au)||native_velocity_to_au<=0
        ||!isfinite(r.true_inverse_mass_au)||r.true_inverse_mass_au<=0
        ||!isfinite(r.represented_main_mass_au)||r.represented_main_mass_au<=0
        ||!isfinite(r.native_hmr_mass_dalton)||r.native_hmr_mass_dalton<=0
        ||!biprism_lif_work_metric_spd(r.accepted_metric_f64))return false;
    const double m=1.0/double(r.true_inverse_mass_au);
    double dv[3],avg[3],dn[3],navg[3],g_dv[3],g_req[3],g_dn[3];
    double lab_k=0,metric_k=0,nlab_k=0,nmetric_k=0,hmr=0;
    double pre_dv[3],pre_avg[3],g_pre[3];
    r.requested_lab_work=0;r.realized_lab_work=0;r.requested_true_metric_work=0;r.realized_true_metric_work=0;
    for(int k=0;k<3;++k){
        const double a=r.au_velocity_before[k],b=r.au_velocity_after[k];
        const double na=r.native_velocity_before[k],nb=r.native_velocity_after[k];
        if(!isfinite(a)||!isfinite(b)||!isfinite(na)||!isfinite(nb)||!isfinite(r.requested_force_au[k]))return false;
        dv[k]=b-a;avg[k]=0.5*(b+a);
        pre_dv[k]=a-double(r.accepted_main_velocity[k]);pre_avg[k]=0.5*(a+double(r.accepted_main_velocity[k]));
        dn[k]=(nb-na)*native_velocity_to_au;navg[k]=0.5*(nb+na)*native_velocity_to_au;
        r.requested_impulse[k]=dt*double(r.requested_force_au[k]);
        r.realized_lab_impulse[k]=m*dv[k];
        r.requested_lab_work+=r.requested_impulse[k]*avg[k];
        r.realized_lab_work+=r.realized_lab_impulse[k]*avg[k];
        lab_k+=m*dv[k]*avg[k];nlab_k+=m*dn[k]*navg[k];
        hmr+=0.5*double(r.native_hmr_mass_dalton)*(nb-na)*(nb+na)/418.4;
    }
    biprism_lif_work_metric_mul(r.accepted_metric_f64,dv,g_dv);
    biprism_lif_work_metric_mul(r.accepted_metric_f64,r.requested_impulse,g_req);
    biprism_lif_work_metric_mul(r.accepted_metric_f64,dn,g_dn);
    for(int k=0;k<3;++k){
        r.realized_true_covariant_impulse[k]=m*g_dv[k];
        r.requested_true_metric_work+=g_req[k]*avg[k];
        r.realized_true_metric_work+=r.realized_true_covariant_impulse[k]*avg[k];
        metric_k+=m*g_dv[k]*avg[k];nmetric_k+=m*g_dn[k]*navg[k];
    }
    r.lab_kinetic_delta=lab_k;r.true_metric_kinetic_delta=metric_k;
    r.main_mass_metric_kinetic_delta=r.local_vertex<192u?metric_k*(double(r.represented_main_mass_au)/m):0.0;
    r.native_hmr_kinetic_delta_kcal=hmr;r.native_true_lab_kinetic_delta=nlab_k;
    r.lab_bridge_kinetic_difference=nlab_k-lab_k;r.native_true_metric_kinetic_delta=nmetric_k;
    r.metric_bridge_kinetic_difference=nmetric_k-metric_k;
    r.main_vs_true_metric_mass_difference=r.local_vertex<192u?r.main_mass_metric_kinetic_delta-metric_k:0.0;
    r.native_phase_hmr_kinetic_delta_kcal=hmr*(418.4/double(418.4f));
    r.native_phase_vs_guard_work_difference_kcal=r.native_phase_hmr_kinetic_delta_kcal-hmr;
    biprism_lif_work_metric_mul(r.accepted_metric_f64,pre_dv,g_pre);
    r.pre_lif_true_metric_delta=0;r.pre_lif_main_metric_delta=0;
    if(r.local_vertex<192u)for(int k=0;k<3;++k){r.pre_lif_true_metric_delta+=m*g_pre[k]*pre_avg[k];
        r.pre_lif_main_metric_delta+=double(r.represented_main_mass_au)*g_pre[k]*pre_avg[k];}
    double values[17];biprism_lif_work_values(r,values);
    for(int k=0;k<17;++k)if(!isfinite(values[k]))return false;
    for(int k=0;k<3;++k)if(!isfinite(r.requested_impulse[k])||!isfinite(r.realized_lab_impulse[k])
        ||!isfinite(r.realized_true_covariant_impulse[k]))return false;
    return true;
}
#undef BP_LW_HD
#ifndef BIPRISM_LIF_WORK_ABI_ONLY
#include "nhs_biprism_unit_bridge.cuh"
#include "nhs_biprism_voronoi_lif.cuh"
__device__ inline void biprism_lif_work_refuse(BiprismLifWorkLedgerFfi& l,
    BiprismConvergenceRecordFfi* conv,uint32_t fault) {
    atomicOr(&l.sticky_fault,fault);atomicOr(&conv->epoch_leaf_fault,BIPRISM_LIF_WORK_TERMINAL);
    atomicOr(&conv->terminal_fault_latch,BIPRISM_LIF_WORK_TERMINAL);
}
__device__ inline bool biprism_lif_work_accepted_metric(const FoliatedStreamPtrTableFfi* au,
    const BiprismConvergenceRecordFfi* c,uint32_t leaf) {
    if(!au||!au->d_electronic_exchange[leaf]||!au->d_numerical_scratch[leaf])return false;
    const auto& f=au->d_electronic_exchange[leaf]->footer;
    return !c[leaf].terminal_fault_latch&&!c[leaf].epoch_leaf_fault&&!c[leaf].epoch_cluster_fault
        &&c[leaf].abi_magic==BIPRISM_TAIL_ABI_MAGIC&&c[leaf].leaf_id==leaf&&c[leaf].committed_substeps>=4u
        &&c[leaf].committed_substeps%4u==0u&&f.abi_magic==BIPRISM_FOLIATED_COUPLING_ABI&&f.leaf_id==leaf
        &&f.accepted_substeps_at_entry<=0xfffffffbu&&f.accepted_substeps_at_entry+4u==c[leaf].committed_substeps
        &&f.substep==3u&&f.publication_phase==4u&&f.convergence_vote==1u&&!f.local_fault_mask&&f.iteration;
}
__device__ inline bool biprism_lif_work_coordinate_owners(const FoliatedStreamPtrTableFfi* au,
    const FoliatedStreamPtrTableFfi* native,uint32_t leaf,uint32_t atoms) {
    if(!au||!native||leaf>=8u||!au->d_coordinate_owners||!native->d_coordinate_owners
        ||(reinterpret_cast<uintptr_t>(au->d_coordinate_owners)&15u)
        ||(reinterpret_cast<uintptr_t>(native->d_coordinate_owners)&15u))return false;
    const auto& a=au->d_coordinate_owners[leaf];const auto& n=native->d_coordinate_owners[leaf];
    return bp_coord_owner_valid(&a,leaf,atoms,n.generation,BP_COORD_ATOMIC_BOHR,
        BP_COORD_ALL_INITIALIZED,au->d_coords_xyz[leaf])
        &&bp_coord_owner_valid(&n,leaf,atoms,a.generation,BP_COORD_NATIVE_ANGSTROM,
        BP_COORD_ALL_INITIALIZED,native->d_coords_xyz[leaf]);
}
// Capture AFTER all native velocity refreshes and BEFORE delayed delivery.
// Every pointer is an existing owner or a separately retained measurement plane.
extern "C" __global__ void nhs_biprism_lif_work_capture_v1(
    const BiprismLifWorkBindingFfi* bindings,const ContactShellSimplicialStateFfi* tiles,
    const FoliatedStreamPtrTableFfi* au,const FoliatedStreamPtrTableFfi* native,
    const BiprismLifCarryoverFfi* carry,const BiprismUnitBridgeStatusFfi* bridge,
    BiprismConvergenceRecordFfi* conv,const uint32_t* step,const float* dt_feedback_au,
    uint32_t atoms,uint32_t mode,BiprismLifWorkLedgerFfi* ledgers,BiprismLifWorkRowFfi* rows,
    const BiprismLifWorkWindowFfi* window) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;
    if(leaf>=8u||tid>=256u||!conv||!ledgers)return;
    auto& l=ledgers[leaf];__shared__ uint32_t fault;
    if(tid==0){fault=0u;uint32_t slot=0u;
        if(!bindings||!tiles||!au||!native||!carry||!bridge||!step||!dt_feedback_au||!rows||!window
            ||atoms<256u||!isfinite(*dt_feedback_au)||*dt_feedback_au<=0.0f)fault|=BIPRISM_LIF_WORK_DOMAIN;
        if(!fault){
            for(uint32_t j=0;j<8u;++j)if(!biprism_lif_work_accepted_metric(au,conv,j)
                ||!biprism_lif_work_coordinate_owners(au,native,j,atoms)
                ||conv[j].committed_substeps!=conv[leaf].committed_substeps||carry[j].fault
                ||carry[j].pending!=carry[leaf].pending||carry[j].pending>1u)fault|=BIPRISM_LIF_WORK_EPOCH;
            const uint64_t body=uint64_t(conv[leaf].committed_substeps/4u)-1u;
            if(l.sticky_fault||l.completed_steps!=body||!biprism_lif_work_slot(*window,body,*step,&slot)
                ||(l.abi_magic&&(l.abi_magic!=BIPRISM_LIF_WORK_MAGIC||l.leaf_id!=leaf||l.state!=BIPRISM_LIF_WORK_SEALED))
                ||bridge[leaf].sticky_fault||bridge[leaf].imported_step!=*step||bridge[leaf].imported_fields!=15u
                ||bridge[leaf].last_core_committed!=conv[leaf].committed_substeps
                ||!bindings[leaf].native_hmr_mass_dalton||!bindings[leaf].native_atom_work_kcal
                ||!au->d_true_inv_masses[leaf]||!au->d_inv_masses[leaf]
                ||!au->d_coords_xyz[leaf]||!au->d_vels_xyz[leaf]
                ||!native->d_coords_xyz[leaf]||!native->d_vels_xyz[leaf])fault|=BIPRISM_LIF_WORK_EPOCH;
            if(!fault){l.abi_magic=BIPRISM_LIF_WORK_MAGIC;l.leaf_id=leaf;l.state=BIPRISM_LIF_WORK_CAPTURED;
                l.scheduled_step=*step;l.committed_substeps=conv[leaf].committed_substeps;
                l.produced_step=carry[leaf].produced_step;l.delivered_step_before=carry[leaf].delivered_step;
                l.pending_before=carry[leaf].pending;l.mode=mode;l.body_ordinal=body;l.dt_feedback_au=*dt_feedback_au;
                for(int k=0;k<17;++k)l.totals[k]=0;
                for(int k=0;k<3;++k)l.requested_impulse[k]=l.realized_lab_impulse[k]=l.realized_true_covariant_impulse[k]=0;
            }
        }
    }__syncthreads();
    if(fault){if(tid==0)biprism_lif_work_refuse(l,&conv[leaf],fault);return;}
    auto& r=rows[leaf*256u+tid];r={};
    const uint32_t atom=UNPACK_GLOBAL_ATOM_ID(tiles[leaf].global_atom_idx[tid]);
    if(atom>=atoms||(l.pending_before&&carry[leaf].atom_id[tid]!=atom)){atomicOr(&fault,BIPRISM_LIF_WORK_DOMAIN);}
    else {
        r.atom_id=atom;r.leaf_id=leaf;r.local_vertex=tid;r.state=BIPRISM_LIF_WORK_CAPTURED;
        r.scheduled_step=l.scheduled_step;r.committed_substeps=l.committed_substeps;
        r.source_produced_step=l.produced_step;r.pending_before=l.pending_before;
        r.pre_lif_scope=tid<192u?((mode&1u)?1u:2u):0u;
        r.true_inverse_mass_au=au->d_true_inv_masses[leaf][atom];
        r.propagation_inverse_mass_au=(mode&1u)?r.true_inverse_mass_au:au->d_inv_masses[leaf][atom];
        r.represented_main_mass_au=1.f/fmaxf(r.propagation_inverse_mass_au,1.e-6f);
        r.native_hmr_mass_dalton=bindings[leaf].native_hmr_mass_dalton[atom];
        for(uint32_t j=0;j<tid;++j)if(atom==UNPACK_GLOBAL_ATOM_ID(tiles[leaf].global_atom_idx[j]))atomicOr(&fault,BIPRISM_LIF_WORK_DOMAIN);
        for(int k=0;k<3;++k){r.metric_position_au[k]=au->d_coords_xyz[leaf][3u*atom+k];
            r.native_position_angstrom[k]=native->d_coords_xyz[leaf][3u*atom+k];
            r.metric_position_au_lo[k]=au->d_coordinate_owners[leaf].current_lo[3u*atom+k];
            r.native_position_angstrom_lo[k]=native->d_coordinate_owners[leaf].current_lo[3u*atom+k];
            r.au_velocity_before[k]=au->d_vels_xyz[leaf][3u*atom+k];
            r.native_velocity_before[k]=native->d_vels_xyz[leaf][3u*atom+k];
            r.accepted_main_velocity[k]=tid<192u?au->d_numerical_scratch[leaf]->endpoint_v[k][tid]:r.au_velocity_before[k];
            r.requested_force_au[k]=l.pending_before?reinterpret_cast<const float*>(&carry[leaf].force[tid])[k]:0.0f;
            if(!isfinite(r.metric_position_au[k])||!isfinite(r.native_position_angstrom[k])
                ||!isfinite(r.metric_position_au_lo[k])||!isfinite(r.native_position_angstrom_lo[k])
                ||r.au_velocity_before[k]!=(float)(double(r.native_velocity_before[k])*BIPRISM_NATIVE_VELOCITY_TO_AU))atomicOr(&fault,BIPRISM_LIF_WORK_NONFINITE);
        }
        for(int k=0;k<6;++k){
            r.accepted_metric_f64[k]=tid<192u?au->d_numerical_scratch[leaf]->nuclear_metric[k][tid]:(k<3?1.:0.);
            r.accepted_metric[k]=float(r.accepted_metric_f64[k]);
        }
        if(!biprism_lif_work_metric_spd(r.accepted_metric_f64)||!isfinite(r.true_inverse_mass_au)||r.true_inverse_mass_au<=0
            ||!isfinite(r.propagation_inverse_mass_au)||r.propagation_inverse_mass_au<=0
            ||!isfinite(r.native_hmr_mass_dalton)||r.native_hmr_mass_dalton<=0)atomicOr(&fault,BIPRISM_LIF_WORK_NONFINITE);
    }
    __syncthreads();if(tid==0&&fault)biprism_lif_work_refuse(l,&conv[leaf],fault);
}
// Measure AFTER successful delayed delivery and native delta export, before
// endpoint UV, clamps, CA intervention and RATTLE add their own work.
extern "C" __global__ void nhs_biprism_lif_work_measure_v1(
    const FoliatedStreamPtrTableFfi* au,const FoliatedStreamPtrTableFfi* native,
    const BiprismLifCarryoverFfi* carry,const BiprismUnitBridgeStatusFfi* bridge,
    BiprismConvergenceRecordFfi* conv,const uint32_t* step,
    BiprismLifWorkLedgerFfi* ledgers,BiprismLifWorkRowFfi* rows) {
    const uint32_t leaf=blockIdx.x,tid=threadIdx.x;if(leaf>=8u||tid>=256u||!conv||!ledgers)return;
    auto& l=ledgers[leaf];__shared__ uint32_t fault;
    if(tid==0){fault=0;
        if(!au||!native||!carry||!bridge||!step||!rows||l.state!=BIPRISM_LIF_WORK_CAPTURED||l.sticky_fault
            ||l.scheduled_step!=*step||l.committed_substeps!=conv[leaf].committed_substeps)fault|=BIPRISM_LIF_WORK_EPOCH;
        if(!fault){for(uint32_t j=0;j<8u;++j)if(!native->d_coordinate_owners
            ||(reinterpret_cast<uintptr_t>(native->d_coordinate_owners)&15u)
            ||!biprism_lif_work_coordinate_owners(au,native,j,native->d_coordinate_owners[j].atom_count)
            ||conv[j].terminal_fault_latch||conv[j].epoch_leaf_fault
            ||conv[j].epoch_cluster_fault||carry[j].fault||carry[j].pending
            ||bridge[j].sticky_fault||bridge[j].export_ready_step!=*step||bridge[j].export_ready_kind!=2u
            ||bridge[j].exported_fields!=3u)fault|=BIPRISM_LIF_WORK_EPOCH;
            if(carry[leaf].produced_step!=l.produced_step
                ||carry[leaf].delivered_step!=(l.pending_before?*step:l.delivered_step_before))fault|=BIPRISM_LIF_WORK_EPOCH;
        }
    }__syncthreads();if(fault){if(tid==0)biprism_lif_work_refuse(l,&conv[leaf],fault);return;}
    auto& r=rows[leaf*256u+tid];
    if(r.state!=BIPRISM_LIF_WORK_CAPTURED||r.scheduled_step!=l.scheduled_step
        ||r.committed_substeps!=l.committed_substeps||r.local_vertex!=tid||r.leaf_id!=leaf
        ||r.atom_id>=native->d_coordinate_owners[leaf].atom_count)atomicOr(&fault,BIPRISM_LIF_WORK_EPOCH);
    else {
        for(int k=0;k<3;++k){r.au_velocity_after[k]=au->d_vels_xyz[leaf][3u*r.atom_id+k];
            r.native_velocity_after[k]=native->d_vels_xyz[leaf][3u*r.atom_id+k];
            // Delivery/export is velocity-only. A changed high OR low position
            // invalidates the retained metric/kinetic-work identity.
            if(__float_as_uint(r.metric_position_au[k])!=__float_as_uint(au->d_coords_xyz[leaf][3u*r.atom_id+k])
                ||__float_as_uint(r.metric_position_au_lo[k])!=__float_as_uint(au->d_coordinate_owners[leaf].current_lo[3u*r.atom_id+k])
                ||__float_as_uint(r.native_position_angstrom[k])!=__float_as_uint(native->d_coords_xyz[leaf][3u*r.atom_id+k])
                ||__float_as_uint(r.native_position_angstrom_lo[k])!=__float_as_uint(native->d_coordinate_owners[leaf].current_lo[3u*r.atom_id+k]))atomicOr(&fault,BIPRISM_LIF_WORK_EPOCH);
            if(!l.pending_before&&(r.au_velocity_after[k]!=r.au_velocity_before[k]
                ||r.native_velocity_after[k]!=r.native_velocity_before[k]))atomicOr(&fault,BIPRISM_LIF_WORK_EPOCH);
        }
        if(!biprism_lif_work_calculate(r,l.dt_feedback_au,BIPRISM_NATIVE_VELOCITY_TO_AU))atomicOr(&fault,BIPRISM_LIF_WORK_NONFINITE);
        else r.state=BIPRISM_LIF_WORK_MEASURED;
    }
    __syncthreads();
    if(tid==0){if(fault){biprism_lif_work_refuse(l,&conv[leaf],fault);return;}
        for(uint32_t i=0;i<256u;++i){const auto& p=rows[leaf*256u+i];double v[17];biprism_lif_work_values(p,v);
            for(int k=0;k<17;++k)l.totals[k]+=v[k];
            for(int k=0;k<3;++k){l.requested_impulse[k]+=p.requested_impulse[k];
                l.realized_lab_impulse[k]+=p.realized_lab_impulse[k];l.realized_true_covariant_impulse[k]+=p.realized_true_covariant_impulse[k];}
        }
        for(int k=0;k<17;++k)if(!isfinite(l.totals[k]))fault|=BIPRISM_LIF_WORK_NONFINITE;
        if(fault)biprism_lif_work_refuse(l,&conv[leaf],fault);else l.state=BIPRISM_LIF_WORK_MEASURED;
    }
}
// Collective publication: no atom work or journal is published until all eight
// measured leaves and every target addition are finite. Rejection keeps raw rows.
extern "C" __global__ void nhs_biprism_lif_work_publish_v1(
    const BiprismLifWorkBindingFfi* bindings,BiprismConvergenceRecordFfi* conv,
    BiprismLifWorkLedgerFfi* ledgers,BiprismLifWorkRowFfi* rows,
    const BiprismLifWorkWindowFfi* window,BiprismLifWorkLedgerFfi* journal_ledgers,
    BiprismLifWorkRowFfi* journal_rows) {
    const uint32_t tid=threadIdx.x;if(blockIdx.x||tid>=256u||!conv||!ledgers)return;
    __shared__ uint32_t fault,slot;
    if(tid==0){fault=0;slot=0;
        if(!bindings||!rows||!window||!journal_ledgers||!journal_rows)fault|=BIPRISM_LIF_WORK_DOMAIN;
        if(!fault)for(uint32_t leaf=0;leaf<8u;++leaf){const auto& l=ledgers[leaf];uint32_t s=0;
            if(l.abi_magic!=BIPRISM_LIF_WORK_MAGIC||l.leaf_id!=leaf||l.state!=BIPRISM_LIF_WORK_MEASURED
                ||l.sticky_fault||l.completed_steps!=l.body_ordinal||conv[leaf].terminal_fault_latch
                ||conv[leaf].committed_substeps!=l.committed_substeps||!bindings[leaf].native_atom_work_kcal
                ||!biprism_lif_work_slot(*window,l.body_ordinal,l.scheduled_step,&s)
                ||(leaf&&s!=slot))fault|=BIPRISM_LIF_WORK_EPOCH;
            slot=s;
        }
    }__syncthreads();
    if(!fault)for(uint32_t leaf=0;leaf<8u;++leaf){const auto& r=rows[leaf*256u+tid];
        if(r.state!=BIPRISM_LIF_WORK_MEASURED||!isfinite(bindings[leaf].native_atom_work_kcal[r.atom_id])
            ||!isfinite(bindings[leaf].native_atom_work_kcal[r.atom_id]+r.native_hmr_kinetic_delta_kcal))atomicOr(&fault,BIPRISM_LIF_WORK_NONFINITE);
    }
    __syncthreads();if(fault){if(tid<8u)biprism_lif_work_refuse(ledgers[tid],&conv[tid],fault);return;}
    for(uint32_t leaf=0;leaf<8u;++leaf){auto& r=rows[leaf*256u+tid];
        bindings[leaf].native_atom_work_kcal[r.atom_id]+=r.native_hmr_kinetic_delta_kcal;
        r.state=BIPRISM_LIF_WORK_SEALED;journal_rows[(slot*8u+leaf)*256u+tid]=r;
    }__syncthreads();
    if(tid<8u){auto& l=ledgers[tid];l.state=BIPRISM_LIF_WORK_SEALED;++l.completed_steps;journal_ledgers[slot*8u+tid]=l;}
}
#endif
