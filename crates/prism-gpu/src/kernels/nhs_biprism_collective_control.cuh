#pragma once
#include "nhs_biprism_nuclear_budget.cuh"
#include "nhs_biprism_unit_bridge.cuh"
#include "nhs_biprism_voronoi_lif.cuh"
#include "nhs_biprism_asc_work.cuh"
#ifndef BIPRISM_SHARED_STEP_ABI_ONLY
#define BIPRISM_SHARED_STEP_ABI_ONLY
#define BIPRISM_COLLECTIVE_UNDEF_SHARED_STEP_ABI_ONLY
#endif
#include "nhs_biprism_shared_step.cuh"
#ifdef BIPRISM_COLLECTIVE_UNDEF_SHARED_STEP_ABI_ONLY
#undef BIPRISM_COLLECTIVE_UNDEF_SHARED_STEP_ABI_ONLY
#undef BIPRISM_SHARED_STEP_ABI_ONLY
#endif
#include "nhs_biprism_lif_work.cuh"
#include "nhs_biprism_phase2_work.cuh"
#include "nhs_biprism_mixed_energy.cuh"
#include "../../../prism-nhs/src/cuda/prism_physical_guard.cuh"
#include "../../../prism-nhs/src/cuda/prism_asc_decision.cuh"
#ifndef BIPRISM_OBSERVATION_ABI_ONLY
#define BIPRISM_OBSERVATION_ABI_ONLY
#define BIPRISM_COLLECTIVE_UNDEF_OBSERVATION_ABI_ONLY
#endif
#include "nhs_biprism_native_observation.cuh"
#ifdef BIPRISM_COLLECTIVE_UNDEF_OBSERVATION_ABI_ONLY
#undef BIPRISM_COLLECTIVE_UNDEF_OBSERVATION_ABI_ONLY
#undef BIPRISM_OBSERVATION_ABI_ONLY
#endif

// This record attests to this ordered composite body, not rollback, durable
// restart, global energy conservation, or an independently validated model.
// The scheduled word is distinct from all native completed-body protocol words.
struct __align__(16) BiprismCollectiveControlFfi {
    uint32_t abi_magic, sticky_fault, last_completed_step, successful_bodies;
    uint32_t last_committed_substeps, failed_scheduled_step;
    uint32_t native_steps[8];
    uint32_t reserved[2];
};
struct __align__(16) BiprismNativeEndpointOwnersFfi {
    const PrismGearGovernorDeviceV2* governor;
    const PrismPhysicalGuardV1* physical_guard;
    const PrismAscDeviceV1* asc;
    uint64_t governor_generation, initial_physical_epoch;
    uint64_t asc_generation, initial_asc_epoch;
    const float* protocol_dt;
    uint32_t source, asc_source, retained_clusters, reserved;
};
static_assert(offsetof(ContactShellSimplicialStateFfi,psi_vertex)==90112,"initial electronic state ABI");
static_assert(sizeof(BiprismNativeEndpointOwnersFfi)==80,"native endpoint owners ABI");
struct BiprismCollectiveLimitsFfi {
    float picard_residual, dec_residual, absolute_energy_drift, unitarity_defect;
};
static_assert(sizeof(BiprismCollectiveControlFfi)==64,"collective control ABI");
static_assert(offsetof(BiprismCollectiveControlFfi,native_steps)==24,"collective native clock ABI");
static_assert(sizeof(BiprismCollectiveLimitsFfi)==16,"collective limits ABI");
constexpr uint32_t BIPRISM_COLLECTIVE_MAGIC=0x42434331u;
constexpr uint32_t BIPRISM_COLLECTIVE_TERMINAL=0x20000000u;
enum BiprismCollectiveFault : uint32_t {
    BIPRISM_COLLECTIVE_INPUT=1u, BIPRISM_COLLECTIVE_CLOCK=2u,
    BIPRISM_COLLECTIVE_CONVERGENCE=4u, BIPRISM_COLLECTIVE_BRIDGE=8u,
    BIPRISM_COLLECTIVE_CARRY=16u, BIPRISM_COLLECTIVE_OBSERVATION=32u,
    BIPRISM_COLLECTIVE_NATIVE_ENDPOINT=64u
};

// This helper owns accepted-record publication after all physical predicates.
// Physical state is not rolled back on refusal; accepted references stay old.
__device__ inline bool biprism_collective_commit_records_v1(
    BiprismCollectiveControlFfi* control,uint32_t upstream_fault,
    BiprismSharedStepFfi* shared,uint64_t generation,const BiprismSharedStepWindowFfi* time_window,
    BiprismSharedStepJournalFfi* time_journal,BiprismMixedEnergyStateFfi* mixed_state,
    BiprismMixedGlobalLedgerFfi* pending,BiprismMixedLeafLedgerFfi* leaves,const BiprismMixedSampleFfi* samples,
    const BiprismMixedWindowFfi* mixed_window,BiprismMixedGlobalLedgerFfi* global_journal,
    BiprismMixedLeafLedgerFfi* leaf_journal,BiprismMixedSampleFfi* sample_journal,
    uint32_t body,uint32_t step,uint32_t committed) {
    uint32_t time_slot=0;
    if(upstream_fault||!control||control->sticky_fault||control->successful_bodies!=body
        ||control->last_completed_step==UINT32_MAX||control->last_completed_step+1u!=step
        ||body>=0x3fffffffu||control->last_committed_substeps!=body*4u||committed!=(body+1u)*4u
        ||!shared||!time_window||!time_journal||shared->accepted_bodies!=body
        ||!biprism_shared_window_slot(*time_window,body,step,time_slot)
        ||!biprism_shared_accept_valid(*shared,step,generation,body+1u,committed,time_journal[time_slot],time_window))return false;
    // The mixed helper completes all remaining checks before its first write.
    if(!biprism_mixed_commit_v1(mixed_state,pending,leaves,samples,mixed_window,
        global_journal,leaf_journal,sample_journal,body,step))return false;
    biprism_publish_accepted_shared_step(*shared,step,body+1u,time_journal[time_slot]);
    control->last_completed_step=step;control->last_committed_substeps=committed;
    control->successful_bodies=body+1u;return true;
}

// Exactly one ordered thread after all eight native endpoints, density/LIF and
// companion capture, immediately before the existing v10 WHILE latch. Any fault
// is broadcast to every convergence record, so that latch cannot advance the
// scheduled word. Earlier physical mutations are retained as failed evidence.
extern "C" __global__ void nhs_biprism_collective_guard_v1(
    BiprismCollectiveControlFfi* control,
    BiprismConvergenceRecordFfi* convergence,
    const BiprismUnitBridgeStatusFfi* bridge,
    const BiprismLifCarryoverFfi* carry,
    const NativeLifObservationCaptureStateFfi* observation,
    const uint32_t* queue_count, uint32_t queue_capacity,
    const uint32_t* scheduled_step,
    const uint32_t* const* native_steps,
    uint32_t atom_count, BiprismCollectiveLimitsFfi limits,
    const BiprismNativeEndpointOwnersFfi* endpoint_owners, BiprismSharedStepFfi* shared,uint64_t shared_generation,
    const BiprismSharedStepWindowFfi* shared_window,BiprismSharedStepJournalFfi* shared_journal,
    const BiprismAscLedgerFfi* asc_ledgers,const BiprismAscWindowFfi* asc_window,
    const BiprismAscLedgerFfi* asc_journal,
    const BiprismLifWorkLedgerFfi* lif_ledgers,const BiprismLifWorkWindowFfi* lif_window,const BiprismLifWorkLedgerFfi* lif_journal,
    const BiprismPhase2WorkLedgerFfi* phase2_ledgers,const BiprismPhase2WorkWindowFfi* phase2_window,const BiprismPhase2WorkLedgerFfi* phase2_journal,
    BiprismMixedEnergyStateFfi* mixed_state,BiprismMixedGlobalLedgerFfi* mixed_pending,BiprismMixedLeafLedgerFfi* mixed_leaves,
    const BiprismMixedSampleFfi* mixed_samples,const BiprismMixedWindowFfi* mixed_window,
    BiprismMixedGlobalLedgerFfi* mixed_global_journal,BiprismMixedLeafLedgerFfi* mixed_leaf_journal,BiprismMixedSampleFfi* mixed_sample_journal
) {
    if (blockIdx.x || threadIdx.x) return;
    if (!control || !convergence) { asm volatile("trap;"); return; }
    uint32_t fault=control->sticky_fault;
    const uint32_t step=scheduled_step?*scheduled_step:0xffffffffu;
    const float selected_native_dt=shared?shared->native_dt_ps:0.f;
    const bool bounds=isfinite(limits.picard_residual)&&limits.picard_residual>=0.0f
        &&isfinite(limits.dec_residual)&&limits.dec_residual>=0.0f
        &&isfinite(limits.absolute_energy_drift)&&limits.absolute_energy_drift>=0.0f
        &&isfinite(limits.unitarity_defect)&&limits.unitarity_defect>=0.0f;
    if (control->abi_magic!=BIPRISM_COLLECTIVE_MAGIC || !bridge || !carry || !observation
        || !queue_count || !scheduled_step || !native_steps || !endpoint_owners || !bounds
        || !asc_ledgers || !asc_window || !asc_journal
        ||!shared||shared->abi_magic!=BIPRISM_SHARED_STEP_MAGIC||shared->state!=1u||shared->sticky_fault
        ||!shared_generation||shared->generation!=shared_generation||shared->scheduled_step!=step
        ||shared->last_accepted_step!=control->last_completed_step||shared->accepted_bodies!=control->successful_bodies
        ||!lif_ledgers||!lif_window||!lif_journal||!isfinite(selected_native_dt)||selected_native_dt<=0.f
        || atom_count<256u || atom_count>0x01000000u || queue_capacity==0u
        || queue_capacity>0x7fffffffu || control->reserved[0] || control->reserved[1])
        fault|=BIPRISM_COLLECTIVE_INPUT;
    if (step==0xffffffffu || control->last_completed_step==0xffffffffu
        || step!=control->last_completed_step+1u
        || control->successful_bodies>=0x3fffffffu
        || control->last_committed_substeps!=control->successful_bodies*4u)
        fault|=BIPRISM_COLLECTIVE_CLOCK;
    const uint32_t committed=control->last_committed_substeps+4u;
    uint32_t asc_slot=0;
    const bool asc_slot_valid=asc_window&&biprism_asc_window_slot(*asc_window,control->successful_bodies,step,&asc_slot);
    if(!asc_slot_valid)fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
    for(uint32_t leaf=0;leaf<8u;++leaf) {
        const auto& c=convergence[leaf];
        const uint32_t actual=(native_steps&&native_steps[leaf])?*native_steps[leaf]:0xffffffffu;
        control->native_steps[leaf]=actual;
        if(actual!=step)fault|=BIPRISM_COLLECTIVE_CLOCK;
        if(c.abi_magic!=BIPRISM_TAIL_ABI_MAGIC || c.leaf_id!=leaf
            || c.epoch_leaf_fault || c.epoch_cluster_fault || c.terminal_fault_latch
            || c.committed_substeps!=committed || c.picard_iters_used==0u
            || c.picard_iters_used>BIPRISM_MAX_NUCLEAR_EVALUATIONS || c.dec_iters_used>BIPRISM_DEC_MAX_ITERS
            || !isfinite(c.last_picard_residual) || c.last_picard_residual<0.0f
            || c.last_picard_residual>limits.picard_residual
            || !isfinite(c.last_dec_residual) || c.last_dec_residual<0.0f
            || c.last_dec_residual>limits.dec_residual
            || !isfinite(c.step_energy_drift) || fabsf(c.step_energy_drift)>limits.absolute_energy_drift
            || !isfinite(c.unitarity_defect) || c.unitarity_defect<0.0f
            || c.unitarity_defect>limits.unitarity_defect
            || !isfinite(c.w_xx_trace) || !isfinite(c.w_x_lambda_trace)
            || !isfinite(c.w_x_tau_trace) || !isfinite(c.pulay_force_norm))
            fault|=BIPRISM_COLLECTIVE_CONVERGENCE;
        if(bridge) {
            const auto& b=bridge[leaf];
            // One cold full import, then full/V/XV imports and two exports/body.
            const uint64_t bodies=uint64_t(control->successful_bodies)+1u;
            if(b.abi_magic!=BIPRISM_BRIDGE_MAGIC || b.leaf_id!=leaf || b.sticky_fault
                || b.atom_count!=atom_count || b.imported_step!=step || b.imported_fields!=31u
                || b.export_ready_step!=step || b.export_ready_kind!=2u || b.exported_fields!=3u
                || b.last_core_committed!=committed || uint64_t(b.import_calls)!=1u+3u*bodies
                || uint64_t(b.export_calls)!=2u*bodies
                || b.reserved[0] || b.reserved[1] || b.reserved[2] || b.reserved[3])
                fault|=BIPRISM_COLLECTIVE_BRIDGE;
        }
        if(endpoint_owners) {
            const auto& n=endpoint_owners[leaf];
            const uint64_t bodies=uint64_t(control->successful_bodies)+1u;
            if(n.reserved || !n.protocol_dt || !isfinite(*n.protocol_dt)
                || *n.protocol_dt!=selected_native_dt)
                fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            if(n.governor) {
                const auto& g=*n.governor;
                const uint64_t epoch=n.initial_physical_epoch+bodies;
                if(epoch<n.initial_physical_epoch || g.abi_version!=2u || g.byte_size!=sizeof(g)
                    || g.source_stream_id!=n.source || g.allocation_generation!=n.governor_generation
                    || !g.bindings.physical_epoch || *g.bindings.physical_epoch!=epoch
                    || g.last_physical_epoch!=epoch || !g.bindings.force_window || !g.bindings.constraint_window
                    || !n.physical_guard) fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
                if(g.bindings.force_window && prism_force_window_status(g.bindings.force_window,n.source,n.governor_generation,epoch)!=PRISM_WINDOW_COMPLETE)
                    fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
                if(g.bindings.constraint_window) {
                    const auto* w=g.bindings.constraint_window;
                    const uint32_t st=prism_constraint_window_status(w,n.source,n.governor_generation,epoch);
                    if(w->n_constraint_clusters!=n.retained_clusters || w->fatal_failure_count
                        || (st!=PRISM_WINDOW_COMPLETE && st!=PRISM_WINDOW_NOT_APPLICABLE)
                        || w->window_epoch!=epoch || w->sealed_epoch!=epoch)
                        fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
                }
                if(n.physical_guard) {
                    const auto& p=*n.physical_guard;const auto& snap=p.snapshot;
                    if(snap.abi!=1u || snap.byte_size!=sizeof(snap) || snap.status!=PRISM_GUARD_COMPLETE
                        || snap.source!=n.source || snap.generation!=n.governor_generation
                        || snap.physical_eval_epoch!=epoch || snap.dt_status!=2u || !snap.energy_valid
                        || snap.transition_pending || p.bindings.governor!=n.governor
                        || p.bindings.dt!=n.protocol_dt || !isfinite(snap.current_pe)
                        || !isfinite(p.current_hamiltonian) || !isfinite(p.current_kinetic)
                        || !isfinite(snap.observed_work)
                        || (p.time_rows && (!p.time_step || *p.time_step!=step || p.step_at_begin!=step
                            || p.time_count!=bodies||p.step_dt_ps!=double(selected_native_dt)
                            ||!shared||p.cumulative_time_ps!=shared->cumulative_accepted_time_ps+double(selected_native_dt))))
                        fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
                }
            } else if(n.physical_guard || n.governor_generation || n.initial_physical_epoch)
                fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            if(n.asc) {
                const auto& a=*n.asc;const uint64_t epoch=n.initial_asc_epoch+bodies;
                if(epoch<n.initial_asc_epoch || a.abi!=1u || a.byte_size!=sizeof(a)
                    || a.source!=n.asc_source || a.generation!=n.asc_generation || a.n_atoms!=atom_count
                    || a.physical_epoch!=epoch || a.completed_epoch!=epoch || a.coverage_failures
                    || (a.status!=ASC_COMPLETE && a.status!=ASC_NO_PRIOR_DECISION)
                    || !isfinite(a.control_work)) fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            } else if(n.asc_generation || n.initial_asc_epoch)
                fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
        }
        if(asc_ledgers && asc_journal && asc_slot_valid && endpoint_owners) {
            const auto& l=asc_ledgers[leaf];const auto& j=asc_journal[uint64_t(asc_slot)*8u+leaf];
            const auto& native=endpoint_owners[leaf];const uint64_t bodies=uint64_t(control->successful_bodies)+1u;
            if(l.abi_magic!=BIPRISM_ASC_MAGIC||l.leaf_id!=leaf||l.sticky_fault||l.reserved
                ||l.scheduled_step!=step||l.completed_steps!=bodies||l.end_substeps!=committed
                ||l.begin_substeps!=committed-4u||l.atom_count!=atom_count
                ||!shared||l.dt_outer_au!=double(shared->outer_dt_au)||!isfinite(l.dt_outer_au)||l.dt_outer_au<=0||!isfinite(l.path_work_hartree)
                ||!isfinite(l.cumulative_path_work_hartree)||!isfinite(l.held_force_work_error_kcal_mol)
                ||!isfinite(l.cumulative_work_error_kcal_mol)||!isfinite(l.control_work_kcal_mol))
                fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            if(native.asc) {
                if(l.state!=BIPRISM_ASC_SEALED||l.generation!=native.asc_generation
                    ||l.initial_completed_epoch!=native.initial_asc_epoch||l.physical_epoch!=native.initial_asc_epoch+bodies
                    ||l.completed_epoch!=l.physical_epoch||l.selected_core_atoms>192u
                    ||l.core_np1_unconsumed_atoms!=l.selected_core_atoms)
                    fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            } else if(l.state!=BIPRISM_ASC_DISABLED||l.generation||l.physical_epoch||l.completed_epoch
                ||l.selected_core_atoms||l.selected_bath_atoms||l.core_np1_unconsumed_atoms)
                fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            for(int k=0;k<3;++k)if(!isfinite(l.canonical_impulse_au[k])||!isfinite(l.cumulative_impulse_au[k]))
                fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            // Unsigned-char access is the defined C++ object representation.
            const unsigned char* a=reinterpret_cast<const unsigned char*>(&l);
            const unsigned char* b=reinterpret_cast<const unsigned char*>(&j);
            for(unsigned k=0;k<sizeof(l);++k)if(a[k]!=b[k])fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
        }
        uint32_t lif_slot=0;
        if(!lif_ledgers||!lif_window||!lif_journal||!biprism_lif_work_slot(*lif_window,control->successful_bodies,step,&lif_slot))
            fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
        else {
            const auto& l=lif_ledgers[leaf];const auto& j=lif_journal[uint64_t(lif_slot)*8u+leaf];
            if(l.abi_magic!=BIPRISM_LIF_WORK_MAGIC||l.leaf_id!=leaf||l.state!=3u||l.sticky_fault
                ||l.scheduled_step!=step||l.committed_substeps!=committed||l.body_ordinal!=control->successful_bodies
                ||l.completed_steps!=uint64_t(control->successful_bodies)+1u||!shared||l.dt_feedback_au!=double(shared->outer_dt_au)
                ||l.pending_before>1u||l.reserved[0]||l.reserved[1]||l.reserved_tail)
                fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            const unsigned char* a=reinterpret_cast<const unsigned char*>(&l);const unsigned char* b=reinterpret_cast<const unsigned char*>(&j);
            for(unsigned k=0;k<sizeof(l);++k)if(a[k]!=b[k])fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
        }
        uint32_t phase2_slot=0;
        if(!phase2_ledgers||!phase2_window||!phase2_journal||!shared
            ||phase2_window->generation!=shared_generation
            ||!biprism_phase2_work_slot(*phase2_window,control->successful_bodies,step,phase2_slot))
            fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
        else {
            const auto& p=phase2_ledgers[leaf];const auto& j=phase2_journal[uint64_t(phase2_slot)*8u+leaf];
            if(p.magic!=BIPRISM_PHASE2_WORK_MAGIC||p.leaf!=leaf||p.state!=3u||p.sticky_fault||p.immutable_mismatches
                ||p.generation!=shared_generation||p.preparation_generation!=phase2_window->preparation_generation
                ||p.scheduled_step!=step||p.body_ordinal!=control->successful_bodies||p.entry_committed!=committed-4u
                ||p.completed_measurements!=uint64_t(control->successful_bodies)+1u||!p.edges||p.edges>1536u||!p.tets||p.tets>1024u
                ||p.reserved||!isfinite(p.entry_refinement_delta)||!isfinite(p.kappa))fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            for(unsigned k=0;k<3;++k)if(!isfinite(p.residual[k])||p.residual[k]<0.f||p.residual[k]>1e-6f||p.iterations[k]>BIPRISM_DEC_MAX_ITERS)
                fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            for(unsigned k=0;k<16;++k)if(!isfinite(p.totals[k]))fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            if(!biprism_phase2_identities_admitted(p.totals))fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
            const unsigned char* a=reinterpret_cast<const unsigned char*>(&p);const unsigned char* b=reinterpret_cast<const unsigned char*>(&j);
            for(unsigned k=0;k<sizeof(p);++k)if(a[k]!=b[k])fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
        }
        if(carry) {
            const auto& k=carry[leaf];
            const uint32_t delivered=control->successful_bodies?step:0u;
            if(k.pending!=1u || k.produced_step!=step || k.delivered_step!=delivered || k.fault
                || k.produced_committed_substeps!=committed || k.reserved[0] || k.reserved[1] || k.reserved[2])
                fault|=BIPRISM_COLLECTIVE_CARRY;
        }
    }
    if(observation && queue_count) {
        const auto& o=*observation;
        if(o.abi_magic!=BIPRISM_NATIVE_OBSERVATION_MAGIC || o.fault || o.reserved
            || o.cursor!=*queue_count || *queue_count>queue_capacity
            || o.last_step!=step || o.bad_record!=0xffffffffu)
            fault|=BIPRISM_COLLECTIVE_OBSERVATION;
    }
    uint32_t mixed_slot=0;
    if(!mixed_state||!mixed_pending||!mixed_leaves||!mixed_samples||!mixed_window||!mixed_global_journal||!mixed_leaf_journal||!mixed_sample_journal
        ||!biprism_mixed_slot(*mixed_window,control->successful_bodies,step,mixed_slot)
        ||mixed_state->sticky_fault||mixed_state->accepted_bodies!=control->successful_bodies
        ||mixed_pending->magic!=BIPRISM_MIXED_MAGIC||mixed_pending->state!=2u||mixed_pending->sticky_fault
        ||mixed_pending->generation!=shared_generation||mixed_pending->scheduled_step!=step||mixed_pending->committed_substeps!=committed
        ||mixed_pending->body_ordinal!=control->successful_bodies||mixed_pending->accepted_bodies!=uint64_t(control->successful_bodies)+1u
        ||!biprism_mixed_energy_admitted(mixed_pending->values[4],mixed_pending->values[7]))fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;
    if(fault) {
        control->sticky_fault=fault;
        if(control->failed_scheduled_step==0u)control->failed_scheduled_step=step;
        for(uint32_t leaf=0;leaf<8u;++leaf)
            convergence[leaf].terminal_fault_latch|=BIPRISM_COLLECTIVE_TERMINAL;
        return;
    }
    if(!biprism_collective_commit_records_v1(control,fault,shared,shared_generation,shared_window,shared_journal,
        mixed_state,mixed_pending,mixed_leaves,mixed_samples,mixed_window,mixed_global_journal,mixed_leaf_journal,mixed_sample_journal,
        control->successful_bodies,step,committed)){
        control->sticky_fault|=BIPRISM_COLLECTIVE_NATIVE_ENDPOINT;control->failed_scheduled_step=step;
        for(uint32_t leaf=0;leaf<8u;++leaf)convergence[leaf].terminal_fault_latch|=BIPRISM_COLLECTIVE_TERMINAL;
        return;
    }
}
