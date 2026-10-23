#pragma once
#include <stdint.h>
#include <stddef.h>
#include "../../../prism-nhs/src/cuda/prism_asc_decision.cuh"

// Separate versioned ownership record. The established ASC owner/atom-row ABI
// and ordinary two-half-kick exports remain unchanged. No native core kick
// epoch is manufactured. Core ForceNP1 is composed but explicitly unconsumed.
struct alignas(16) BiprismAscBindingFfi {
    PrismAscDeviceV1* owner;
    const uint8_t* motion_ownership;
};
struct alignas(16) BiprismAscCoreRowFfi {
    uint64_t physical_epoch, decision_epoch;
    uint32_t atom_id, scheduled_step, state, selected;
    float x_start_au[3], x_end_au[3];
    double requested_native_force[3], effective_au_force[3], canonical_impulse_au[3];
    double path_work_hartree, potential_start_kcal_mol, potential_end_kcal_mol;
    double held_force_work_error_kcal_mol;
    float native_total_force[3], native_base_force[3], au_total_force[3], au_base_force[3];
};
struct alignas(16) BiprismAscLedgerFfi {
    uint32_t abi_magic, leaf_id, sticky_fault, state;
    uint64_t generation, physical_epoch, decision_epoch, initial_completed_epoch;
    uint32_t scheduled_step, begin_substeps, end_substeps, selected_core_atoms;
    uint32_t selected_bath_atoms, core_np1_unconsumed_atoms, atom_count, reserved;
    uint64_t completed_steps, completed_epoch;
    double dt_outer_au, path_work_hartree, cumulative_path_work_hartree;
    double held_force_work_error_kcal_mol, cumulative_work_error_kcal_mol;
    double canonical_impulse_au[3], cumulative_impulse_au[3];
    double control_work_kcal_mol;
};
struct alignas(16) BiprismAscWindowFfi {
    uint32_t abi_magic, capacity, first_body, requested_steps;
    uint32_t first_scheduled_step, reserved[3];
};
static_assert(sizeof(BiprismAscWindowFfi)==32,"ASC work window ABI");
static_assert(offsetof(BiprismAscCoreRowFfi,requested_native_force)==56,"ASC row force offset");
static_assert(offsetof(BiprismAscLedgerFfi,control_work_kcal_mol)==184,"ASC control work offset");
static_assert(sizeof(BiprismAscBindingFfi)==16,"ASC mixed binding ABI");
static_assert(sizeof(BiprismAscCoreRowFfi)==208,"ASC core response row ABI");
static_assert(sizeof(BiprismAscLedgerFfi)==192,"ASC core response ledger ABI");
static constexpr uint32_t BIPRISM_ASC_MAGIC=0x42415331u;
static constexpr uint32_t BIPRISM_ASC_DOMAIN=1u, BIPRISM_ASC_EPOCH=2u;
static constexpr uint32_t BIPRISM_ASC_NONFINITE=4u, BIPRISM_ASC_COVERAGE=8u;
static constexpr uint32_t BIPRISM_ASC_TERMINAL=0x10000000u;
static constexpr uint32_t BIPRISM_ASC_PREPARED=1u, BIPRISM_ASC_CORE_ACCEPTED=2u;
static constexpr uint32_t BIPRISM_ASC_SEALED=3u, BIPRISM_ASC_DISABLED=4u;
// Pure index/identity rules shared by the actual kernel and CPU adversarial tests.
#if defined(__CUDACC__)
#define BIPRISM_ASC_HD __host__ __device__
#else
#define BIPRISM_ASC_HD
#endif
BIPRISM_ASC_HD inline bool biprism_asc_window_slot(
    const BiprismAscWindowFfi& w,uint64_t completed_bodies,uint32_t step,uint32_t* slot) {
    if(w.abi_magic!=BIPRISM_ASC_MAGIC || !w.capacity || !w.requested_steps
        ||w.requested_steps>w.capacity ||w.reserved[0]||w.reserved[1]||w.reserved[2]
        ||completed_bodies<w.first_body ||completed_bodies-w.first_body>=w.requested_steps
        ||uint64_t(w.first_scheduled_step)+(completed_bodies-w.first_body)>=UINT32_MAX
        ||step!=uint64_t(w.first_scheduled_step)+(completed_bodies-w.first_body))return false;
    *slot=uint32_t(completed_bodies-w.first_body);return true;
}
BIPRISM_ASC_HD inline bool biprism_asc_work_tail_valid(
    const PrismAscWorkCumulativeV1& w,const PrismAscDeviceV1& d,double work) {
    return w.magic==PRISM_ASC_WORK_MAGIC_V1 && w.completed_steps!=UINT64_MAX
        && w.completed_epoch!=UINT64_MAX && w.completed_epoch+1u==d.physical_epoch
        && w.completed_steps==w.completed_epoch && isfinite(work)
        && isfinite(w.control_work_kcal_mol) && isfinite(w.control_work_kcal_mol+work);
}
#undef BIPRISM_ASC_HD
#ifndef BIPRISM_ASC_ABI_ONLY
#include "nhs_biprism_unit_bridge.cuh"

__device__ inline void biprism_asc_refuse(BiprismAscLedgerFfi& l,
    BiprismConvergenceRecordFfi* c, uint32_t fault) {
    l.sticky_fault|=fault;
    if(c){c->epoch_leaf_fault|=BIPRISM_ASC_TERMINAL;c->terminal_fault_latch|=BIPRISM_ASC_TERMINAL;}
}
__device__ inline bool biprism_asc_owner_valid(const PrismAscDeviceV1* d,uint32_t n) {
    return d && d->abi==1u && d->byte_size==sizeof(*d) && d->n_atoms==n && d->rows
        && d->physical_epoch && d->physical_epoch==d->completed_epoch+1u
        && d->completed_epoch!=UINT64_MAX && !d->coverage_failures
        && (d->status==ASC_OPEN || d->status==ASC_NO_PRIOR_DECISION)
        && ((d->mode==ASC_LEGACY_WHOLE_SYSTEM) ||
            (d->mode==ASC_SELECTED_STRUCTURAL_SUPPORT && d->reserved3 && d->reserved && d->reserved<=n));
}
// Immediately after native ForceN composition and exact bridge import, before
// v10. Requested model force and the effective imported source are both kept.
// Effective F_ASC_AU = double(F_total_AU)-double(convert(base_native)), so the
// partition adds back to the actual imported float force without re-rounding.
extern "C" __global__ void nhs_biprism_asc_capture_force_v1(
    const BiprismAscBindingFfi* bindings,
    const ContactShellSimplicialStateFfi* tiles,
    const FoliatedStreamPtrTableFfi* au,
    const FoliatedStreamPtrTableFfi* native,
    uint32_t atom_count, BiprismConvergenceRecordFfi* convergence,
    const uint32_t* scheduled_step, BiprismAscLedgerFfi* ledgers,
    BiprismAscCoreRowFfi* rows, const float* selected_dt_outer_au, const BiprismAscWindowFfi* window) {
    if(blockIdx.x>=8u || threadIdx.x || !ledgers || !convergence)return;
    const uint32_t leaf=blockIdx.x;auto& l=ledgers[leaf];auto* c=&convergence[leaf];
    const float dt_outer_au=selected_dt_outer_au?*selected_dt_outer_au:0.0f;
    if(!bindings||!tiles||!au||!native||!scheduled_step||!rows||atom_count<256u
        ||!window||!isfinite(dt_outer_au)||dt_outer_au<=0.0f) {biprism_asc_refuse(l,c,BIPRISM_ASC_DOMAIN);return;}
    const auto& b=bindings[leaf];auto* d=b.owner;
    uint32_t slot=0;
    if(!biprism_asc_window_slot(*window,l.completed_steps,*scheduled_step,&slot)) {
        biprism_asc_refuse(l,c,BIPRISM_ASC_EPOCH);return;
    }
    if(!d){
        if(l.sticky_fault || (l.abi_magic && (l.abi_magic!=BIPRISM_ASC_MAGIC
            ||l.state!=BIPRISM_ASC_DISABLED||l.leaf_id!=leaf||l.generation||l.physical_epoch
            ||l.completed_epoch||l.initial_completed_epoch)) ||c->terminal_fault_latch
            ||c->committed_substeps%4u ||l.completed_steps!=c->committed_substeps/4u
            ||c->committed_substeps>0xfffffffbu) {
            biprism_asc_refuse(l,c,BIPRISM_ASC_EPOCH);return;
        }
        l.abi_magic=BIPRISM_ASC_MAGIC;l.leaf_id=leaf;l.state=BIPRISM_ASC_PREPARED;
        l.scheduled_step=*scheduled_step;l.begin_substeps=c->committed_substeps;l.end_substeps=0;
        l.atom_count=atom_count;l.dt_outer_au=dt_outer_au;return;
    }
    if(l.sticky_fault||!biprism_asc_owner_valid(d,atom_count)||!b.motion_ownership
        ||!au->d_coords_xyz[leaf]||!au->d_forces_xyz[leaf]||!native->d_forces_xyz[leaf]
        ||c->terminal_fault_latch||c->committed_substeps>0xfffffffbu) {
        biprism_asc_refuse(l,c,BIPRISM_ASC_DOMAIN);return;}
    if(!l.abi_magic){l.abi_magic=BIPRISM_ASC_MAGIC;l.leaf_id=leaf;
        l.generation=d->generation;l.initial_completed_epoch=d->completed_epoch;l.completed_epoch=d->completed_epoch;}
    if(l.abi_magic!=BIPRISM_ASC_MAGIC||l.leaf_id!=leaf||l.generation!=d->generation
        ||l.completed_epoch!=d->completed_epoch||l.completed_steps!=uint64_t(c->committed_substeps/4u)
        ||c->committed_substeps%4u||l.reserved
        ||(l.state && l.state!=BIPRISM_ASC_SEALED)
        ||(l.state && (*scheduled_step!=l.scheduled_step+1u||l.scheduled_step==UINT32_MAX))) {
        biprism_asc_refuse(l,c,BIPRISM_ASC_EPOCH);return;}
    l.physical_epoch=d->physical_epoch;l.decision_epoch=d->active.producer_epoch;
    l.scheduled_step=*scheduled_step;l.begin_substeps=c->committed_substeps;l.end_substeps=0u;
    l.selected_core_atoms=0;l.selected_bath_atoms=0;l.core_np1_unconsumed_atoms=0;l.atom_count=atom_count;
    l.dt_outer_au=dt_outer_au;l.path_work_hartree=0;l.held_force_work_error_kcal_mol=0;l.control_work_kcal_mol=0;
    for(int k=0;k<3;++k)l.canonical_impulse_au[k]=0.0;
    for(uint32_t i=0;i<192u;++i){
        const uint32_t atom=UNPACK_GLOBAL_ATOM_ID(tiles[leaf].global_atom_idx[i]);
        if(atom>=atom_count || !(b.motion_ownership[atom]&1u)) {biprism_asc_refuse(l,c,BIPRISM_ASC_DOMAIN);return;}
        for(uint32_t j=0;j<i;++j)if(atom==UNPACK_GLOBAL_ATOM_ID(tiles[leaf].global_atom_idx[j])){biprism_asc_refuse(l,c,BIPRISM_ASC_DOMAIN);return;}
        const auto& r=d->rows[atom];auto& out=rows[leaf*192u+i];
        if(r.epoch[0]!=d->physical_epoch||r.decision_epoch[0]!=d->active.producer_epoch
            ||r.epoch[2]==d->physical_epoch||r.epoch[3]==d->physical_epoch) {
            biprism_asc_refuse(l,c,BIPRISM_ASC_EPOCH);return;}
        out={};out.physical_epoch=l.physical_epoch;out.decision_epoch=l.decision_epoch;
        out.atom_id=atom;out.scheduled_step=l.scheduled_step;out.selected=prism_asc_atom_selected(*d,atom)?1u:0u;
        l.selected_core_atoms+=out.selected;out.potential_start_kcal_mol=r.potential[0];
        if(!isfinite(out.potential_start_kcal_mol)){biprism_asc_refuse(l,c,BIPRISM_ASC_NONFINITE);return;}
        for(int k=0;k<3;++k){
            const float total=au->d_forces_xyz[leaf][3u*atom+k];
            const float base=(float)((double)r.base_force[0][k]*BIPRISM_NATIVE_FORCE_TO_AU);
            const float native_total=native->d_forces_xyz[leaf][3u*atom+k];
            const float check_total=(float)((double)native_total*BIPRISM_NATIVE_FORCE_TO_AU);
            out.native_total_force[k]=native_total;out.native_base_force[k]=r.base_force[0][k];
            out.au_total_force[k]=total;out.au_base_force[k]=base;
            out.requested_native_force[k]=r.force[0][k];out.effective_au_force[k]=(double)total-(double)base;
            out.x_start_au[k]=au->d_coords_xyz[leaf][3u*atom+k];
            if(!isfinite(total)||!isfinite(base)||!isfinite(native_total)||!isfinite(out.requested_native_force[k])
                ||!isfinite(out.x_start_au[k])||total!=check_total
                ||(!out.selected && out.effective_au_force[k]!=0.0)){
                biprism_asc_refuse(l,c,BIPRISM_ASC_NONFINITE);return;}
        }
        out.state=BIPRISM_ASC_PREPARED;
    }
    l.state=BIPRISM_ASC_PREPARED;
}
// Runs directly after accepted v10 and before any native core scatter/SHAKE.
// Canonical impulse is a held-source contribution, never an asserted additive
// endpoint velocity response through metric, magnetic or constraint evolution.
extern "C" __global__ void nhs_biprism_asc_finish_core_v1(
    const BiprismAscBindingFfi* bindings,const FoliatedStreamPtrTableFfi* au,
    uint32_t atom_count,BiprismConvergenceRecordFfi* convergence,
    const uint32_t* scheduled_step,BiprismAscLedgerFfi* ledgers,BiprismAscCoreRowFfi* rows) {
    if(blockIdx.x>=8u||threadIdx.x||!ledgers||!convergence)return;
    const uint32_t leaf=blockIdx.x;auto& l=ledgers[leaf];auto* c=&convergence[leaf];
    if(!bindings||!au||!scheduled_step||!rows){biprism_asc_refuse(l,c,BIPRISM_ASC_DOMAIN);return;}
    auto* d=bindings[leaf].owner;
    if(!d){
        if(l.sticky_fault||l.state!=BIPRISM_ASC_PREPARED||l.scheduled_step!=*scheduled_step
            ||c->terminal_fault_latch||c->epoch_leaf_fault||c->epoch_cluster_fault
            ||l.begin_substeps>0xfffffffbu||c->committed_substeps!=l.begin_substeps+4u) {
            biprism_asc_refuse(l,c,BIPRISM_ASC_EPOCH);return;
        }
        l.end_substeps=c->committed_substeps;l.state=BIPRISM_ASC_CORE_ACCEPTED;return;
    }
    if(l.sticky_fault||!biprism_asc_owner_valid(d,atom_count)||l.state!=BIPRISM_ASC_PREPARED
        ||l.scheduled_step!=*scheduled_step||l.physical_epoch!=d->physical_epoch
        ||l.decision_epoch!=d->active.producer_epoch||l.begin_substeps>0xfffffffbu
        ||c->terminal_fault_latch||c->epoch_leaf_fault||c->epoch_cluster_fault
        ||c->committed_substeps!=l.begin_substeps+4u||!au->d_coords_xyz[leaf]) {
        biprism_asc_refuse(l,c,BIPRISM_ASC_EPOCH);return;}
    // Unit identity: (Hartree/bohr)/(kcal/mol/Angstrom) * bohr/Angstrom.
    const double hartree_to_kcal=1.0/(BIPRISM_NATIVE_FORCE_TO_AU*BIPRISM_ANGSTROM_TO_BOHR);
    for(uint32_t i=0;i<192u;++i){auto& r=rows[leaf*192u+i];
        if(r.state!=BIPRISM_ASC_PREPARED||r.atom_id>=atom_count||r.physical_epoch!=l.physical_epoch
            ||r.decision_epoch!=l.decision_epoch||r.scheduled_step!=l.scheduled_step){biprism_asc_refuse(l,c,BIPRISM_ASC_EPOCH);return;}
        float x_native[3];double model_force[3],potential=0.0;
        for(int k=0;k<3;++k){r.x_end_au[k]=au->d_coords_xyz[leaf][3u*r.atom_id+k];
            x_native[k]=(float)((double)r.x_end_au[k]/BIPRISM_ANGSTROM_TO_BOHR);
            r.canonical_impulse_au[k]=l.dt_outer_au*r.effective_au_force[k];
            r.path_work_hartree+=r.effective_au_force[k]*((double)r.x_end_au[k]-(double)r.x_start_au[k]);
            if(!isfinite(r.x_end_au[k])||!isfinite(x_native[k])||!isfinite(r.canonical_impulse_au[k])){biprism_asc_refuse(l,c,BIPRISM_ASC_NONFINITE);return;}
            l.canonical_impulse_au[k]+=r.canonical_impulse_au[k];}
        if(r.selected)prism_asc_law(d->active,x_native,model_force,&potential);
        r.potential_end_kcal_mol=potential;
        r.held_force_work_error_kcal_mol=r.path_work_hartree*hartree_to_kcal+(potential-r.potential_start_kcal_mol);
        if(!isfinite(r.path_work_hartree)||!isfinite(potential)||!isfinite(r.held_force_work_error_kcal_mol)){biprism_asc_refuse(l,c,BIPRISM_ASC_NONFINITE);return;}
        l.path_work_hartree+=r.path_work_hartree;l.held_force_work_error_kcal_mol+=r.held_force_work_error_kcal_mol;r.state=BIPRISM_ASC_CORE_ACCEPTED;
    }
    if(!isfinite(l.path_work_hartree)||!isfinite(l.held_force_work_error_kcal_mol)) {biprism_asc_refuse(l,c,BIPRISM_ASC_NONFINITE);return;}
    for(int k=0;k<3;++k)if(!isfinite(l.canonical_impulse_au[k])){biprism_asc_refuse(l,c,BIPRISM_ASC_NONFINITE);return;}
    l.end_substeps=c->committed_substeps;l.state=BIPRISM_ASC_CORE_ACCEPTED;
}
// New explicit mixed-policy completion. Native phase aggregates retain actual
// all-selected force compositions and bath-only kick consumption. The separate
// ledger supplies core held-source coverage; no core r.epoch[2/3] is stamped.
extern "C" __global__ void nhs_biprism_asc_seal_mixed_v1(
    PrismAscDeviceV1* d,const uint8_t* motion_ownership,
    BiprismAscLedgerFfi* l,const BiprismAscCoreRowFfi* core_rows,uint32_t atom_count,
    const BiprismAscWindowFfi* window,BiprismAscLedgerFfi* journal_ledgers,
    BiprismAscCoreRowFfi* journal_rows,uint32_t leaf,BiprismConvergenceRecordFfi* convergence) {
    if(blockIdx.x||threadIdx.x)return;
    if(!l||!convergence)return;
    auto* c=convergence;
    uint32_t slot=0;
    if(leaf>=8u||!window||!journal_ledgers||!journal_rows||!core_rows||!motion_ownership
        ||!biprism_asc_window_slot(*window,l->completed_steps,l->scheduled_step,&slot)) {
        biprism_asc_refuse(*l,c,BIPRISM_ASC_DOMAIN);return;
    }
    const uint64_t record=uint64_t(slot)*8u+leaf;
    if(!d){
        if(l->sticky_fault||l->state!=BIPRISM_ASC_CORE_ACCEPTED||l->leaf_id!=leaf
            ||c->terminal_fault_latch||c->epoch_leaf_fault||c->epoch_cluster_fault
            ||l->end_substeps!=c->committed_substeps||l->end_substeps!=l->begin_substeps+4u) {
            biprism_asc_refuse(*l,c,BIPRISM_ASC_EPOCH);return;
        }
        ++l->completed_steps;l->state=BIPRISM_ASC_DISABLED;
        journal_ledgers[record]=*l;
        // Disabled leaves have no produced atom rows; their reserved row slots
        // are excluded by the journal's explicit per-leaf enabled mask.
        return;
    }
    uint32_t fault=l->sticky_fault;
    if(c->terminal_fault_latch||c->epoch_leaf_fault||c->epoch_cluster_fault
        ||c->committed_substeps!=l->end_substeps)fault|=BIPRISM_ASC_EPOCH;
    if(!biprism_asc_owner_valid(d,atom_count)||l->abi_magic!=BIPRISM_ASC_MAGIC
        ||l->state!=BIPRISM_ASC_CORE_ACCEPTED||l->generation!=d->generation
        ||l->physical_epoch!=d->physical_epoch||l->decision_epoch!=d->active.producer_epoch
        ||l->completed_epoch==UINT64_MAX||l->completed_epoch+1u!=d->physical_epoch
        ||l->completed_steps==UINT64_MAX||d->composed_steps==UINT64_MAX||d->consumed_steps==UINT64_MAX
        ||l->end_substeps!=l->begin_substeps+4u)fault|=BIPRISM_ASC_EPOCH;
    PrismAscPhaseV1 phase[2]={};double control_work=0.0;uint32_t core_count=0,selected_core=0,bath_count=0;
    for(uint32_t i=0;i<atom_count&&!fault;++i){const auto& r=d->rows[i];const bool core=(motion_ownership[i]&1u)!=0;
        const bool selected=prism_asc_atom_selected(*d,i);
        if(core){++core_count;uint32_t match=192u;
            for(uint32_t k=0;k<192u;++k)if(core_rows[k].atom_id==i){if(match!=192u){fault|=BIPRISM_ASC_COVERAGE;break;}match=k;}
            if(match==192u){fault|=BIPRISM_ASC_COVERAGE;break;}
            const auto& cr=core_rows[match];
            if(cr.state!=BIPRISM_ASC_CORE_ACCEPTED||cr.physical_epoch!=d->physical_epoch
                ||cr.decision_epoch!=d->active.producer_epoch||cr.selected!=(selected?1u:0u)
                ||cr.scheduled_step!=l->scheduled_step||r.epoch[2]==d->physical_epoch||r.epoch[3]==d->physical_epoch)
                fault|=BIPRISM_ASC_COVERAGE;
        }
        if(!selected)continue;
        if(core)++selected_core;else++bath_count;
        for(int p=0;p<2;++p){
            if(r.epoch[p]!=d->physical_epoch||r.decision_epoch[p]!=d->active.producer_epoch
                ||(!core&&(r.epoch[p+2]!=d->physical_epoch||r.decision_epoch[p+2]!=d->active.producer_epoch)))fault|=BIPRISM_ASC_COVERAGE;
            auto& s=phase[p];++s.composed_atoms;bool nf=false,ni=false;
            for(int k=0;k<3;++k){s.force[k]+=r.force[p][k];s.torque[k]+=r.torque[p][k];nf|=r.force[p][k]!=0.0;
                if(!core){s.impulse[k]+=r.impulse[p][k];s.velocity_delta[k]+=r.velocity_delta[p][k];ni|=r.impulse[p][k]!=0.0;}}
            s.nonzero_force_atoms+=nf;s.potential+=r.potential[p];
            if(!core){++s.consumed_atoms;s.nonzero_impulse_atoms+=ni;s.differential_kick_energy+=r.differential_kick_energy[p];}
        }
        control_work+=r.control_work;
    }
    if(core_count!=192u||selected_core!=l->selected_core_atoms
        ||(d->mode==ASC_SELECTED_STRUCTURAL_SUPPORT&&selected_core+bath_count!=d->reserved)
        ||!isfinite(control_work))fault|=BIPRISM_ASC_COVERAGE;
    for(int p=0;p<2;++p){const auto& s=phase[p];
        if(!isfinite(s.potential)||!isfinite(s.differential_kick_energy))fault|=BIPRISM_ASC_NONFINITE;
        for(int k=0;k<3;++k)if(!isfinite(s.force[k])||!isfinite(s.torque[k])||!isfinite(s.impulse[k])||!isfinite(s.velocity_delta[k]))fault|=BIPRISM_ASC_NONFINITE;
    }
    const double cw=l->cumulative_path_work_hartree+l->path_work_hartree;
    const double ce=l->cumulative_work_error_kcal_mol+l->held_force_work_error_kcal_mol;
    if(!isfinite(cw)||!isfinite(ce))fault|=BIPRISM_ASC_NONFINITE;
    for(int k=0;k<3;++k)if(!isfinite(l->cumulative_impulse_au[k]+l->canonical_impulse_au[k]))fault|=BIPRISM_ASC_NONFINITE;
    if(!biprism_asc_work_tail_valid(*prism_asc_work_tail(d),*d,control_work))fault|=BIPRISM_ASC_EPOCH;
    if(fault){biprism_asc_refuse(*l,c,fault);
        if(d->coverage_failures!=UINT64_MAX)++d->coverage_failures;
        d->status=ASC_INCOMPLETE_COVERAGE;return;
    }
    // Native control-parameter work is unchanged. Conservative held-source work
    // and its splitting residual are observations, never added again externally.
    d->phases[0]=phase[0];d->phases[1]=phase[1];d->control_work=control_work;
    prism_asc_commit_control_work(d);
    d->completed_epoch=d->physical_epoch;++d->composed_steps;++d->consumed_steps;
    d->status=d->active.producer_epoch?ASC_COMPLETE:ASC_NO_PRIOR_DECISION;
    l->control_work_kcal_mol=control_work;
    l->selected_bath_atoms=bath_count;l->core_np1_unconsumed_atoms=selected_core;
    l->cumulative_path_work_hartree=cw;l->cumulative_work_error_kcal_mol=ce;
    for(int k=0;k<3;++k)l->cumulative_impulse_au[k]+=l->canonical_impulse_au[k];
    ++l->completed_steps;l->completed_epoch=d->completed_epoch;l->state=BIPRISM_ASC_SEALED;
    for(uint32_t i=0;i<192u;++i)journal_rows[record*192u+i]=core_rows[i];
    journal_ledgers[record]=*l;
}
#endif
