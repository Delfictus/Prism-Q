#pragma once
#include "gearbox.cuh"
#include "../../../prism-gpu/src/kernels/nhs_biprism_shared_step_abi.cuh"
#include <stdint.h>
#include <stddef.h>
#include <cmath>
enum : uint32_t { PRISM_GUARD_UNINITIALIZED=0,PRISM_GUARD_READY=1,PRISM_GUARD_COMPLETE=2,
    PRISM_GUARD_OWNER_MISMATCH=3,PRISM_GUARD_EPOCH_MISMATCH=4,PRISM_GUARD_DT_CONFLICT=5,
    PRISM_GUARD_EMERGENCY=6,PRISM_GUARD_NONFINITE=7,PRISM_GUARD_MEASUREMENT_INCOMPLETE=8 };
enum : uint32_t { PRISM_SCIENCE_UNMEASURED=0,PRISM_SCIENCE_PUBLISHED=1 };
struct alignas(16) PrismPhysicalGuardBindingsV1 {
    uint32_t abi,byte_size,source,n_atoms;
    uint64_t generation;
    PrismGearGovernorDeviceV2* governor;
    const float* forces;
    // Actual native PE plane; ordinary direct-space pairs use the declared
    // ORDINARY_CONTINUOUS_CUTOFF_V1 convention. Exact exceptions and lambda/PME
    // retain their separately declared Hamiltonians. No residual is added to work.
    const double* potential;
    float* velocities;
    const float* masses;
    float* dt;
    const double* step_work;
};
struct alignas(16) PrismPhysicalGuardSnapshotV1 {
    uint32_t abi,byte_size,source,status;
    uint64_t generation,physical_eval_epoch,science_epoch,science_publications;
    uint32_t science_status,science_code,science_request,science_override;
    uint32_t initial_gear,applied_gear,dt_status,energy_valid;
    double previous_pe,current_pe,observed_work,observed_drift;
    double last_transition_work,total_transition_work,pending_interval_work,admitted_dt;
    uint32_t force_status,constraint_status,transition_pending,host_ceiling;
    uint64_t physical_decisions,transition_count,science_age_physical_steps,max_evaluation_gap;
};
// Out-of-band chronology; no ProtocolState or sealed spike packet ABI changes.
struct PrismPhysicalTimeRecordV1 {
    uint64_t step;
    double applied_dt_ps, cumulative_simulated_time_ps;
};
static_assert(sizeof(PrismPhysicalTimeRecordV1)==24,"physical time row ABI");
enum : uint32_t { PRISM_ENERGY_BOOTSTRAP=0, PRISM_ENERGY_ESTABLISHED_VALID=1,
    PRISM_ENERGY_INVALID_REFERENCE=2 };
static constexpr uint32_t PRISM_ENERGY_NORMALIZATION_CONTRACT=1;
// Completed endpoint energies; previous values refer to the previous accepted
// post-integrator/pre-transition endpoint, exactly like Hamiltonian history.
struct PrismPhysicalEnergyBalanceRecordV1 {
    uint32_t abi,byte_size,source,history_status,normalization_contract,reserved;
    uint64_t generation,step,physical_epoch,previous_epoch;
    double previous_potential,current_potential,previous_kinetic,current_kinetic;
    double body_work,carried_transition_work,observed_work,signed_residual;
    double kinetic_reference,normalized_residual,legacy_h_ratio;
    double applied_dt_ps,cumulative_simulated_time_ps;
};
static_assert(sizeof(PrismPhysicalEnergyBalanceRecordV1)==160,"physical energy balance row ABI");
static_assert(offsetof(PrismPhysicalEnergyBalanceRecordV1,previous_potential)==56,"energy row components offset");
struct alignas(16) PrismPhysicalGuardV1 {
    PrismPhysicalGuardSnapshotV1 snapshot;
    PrismPhysicalGuardBindingsV1 bindings;
    double* transition_work_rows;
    uint64_t last_begin_epoch;
    uint32_t transition_from,transition_to,initializing,reserved;
    // Private control history. Public snapshot previous_pe/current_pe remain
    // potential energy; work-corrected drift uses total Hamiltonian energy.
    double previous_hamiltonian,current_hamiltonian;
    double work_at_begin;
    PrismPhysicalTimeRecordV1* time_rows;
    const uint32_t* time_step;
    uint64_t time_capacity,time_count;
    double cumulative_time_ps,step_dt_ps;
    uint64_t step_at_begin;
    double previous_kinetic,current_kinetic;
    PrismPhysicalEnergyBalanceRecordV1* energy_rows;
};
static_assert(sizeof(PrismPhysicalGuardBindingsV1)==80,"guard bindings ABI");
static_assert(sizeof(PrismPhysicalGuardSnapshotV1)==192,"guard snapshot ABI");
static_assert(sizeof(PrismPhysicalGuardV1)==416,"guard state ABI");
static_assert(offsetof(PrismPhysicalGuardSnapshotV1, previous_pe)==80,"guard energy offset");
static_assert(offsetof(PrismPhysicalGuardSnapshotV1, physical_decisions)==160,"guard counters offset");
static_assert(offsetof(PrismPhysicalGuardV1, bindings)==192,"guard binding offset");
static_assert(offsetof(PrismPhysicalGuardV1, previous_hamiltonian)==304,"guard private history offset");
static_assert(offsetof(PrismPhysicalGuardV1, work_at_begin)==320,"guard work baseline offset");
// Cold science is absent, not a fabricated Prune. Hold caps the request at the
// current gear; even a fully permissive physical ceiling cannot create an upshift.
__host__ __device__ inline uint32_t prism_guard_effective_request(uint32_t science_status,uint32_t request,uint32_t current) {
    return science_status==PRISM_SCIENCE_PUBLISHED && request<=PRISM_GEAR_MAX_ACTIVE ? request : current;
}
__host__ __device__ inline uint32_t prism_guard_gear_for_dt(const PrismGearGovernorPolicy& p,float dt) {
    if(!prism_gear_finite_f32(dt)||dt<=0)return PRISM_GEAR_UNSET;
    for(uint32_t g=0;g<=PRISM_GEAR_MAX_ACTIVE;++g){float target=p.gear_dt_fs[g]*0.001f;
        if(prism_gear_finite_f32(target)&&target>0&&fabsf(dt-target)<=target*1e-5f)return g;}
    return PRISM_GEAR_UNSET;
}
__host__ __device__ inline double prism_guard_transition_delta_k(float mass,const float before[3],const float after[3]) {
    double v=0;for(int j=0;j<3;++j)v+=(double)after[j]*after[j]-(double)before[j]*before[j];
    return 0.5*(double)mass*v/418.4;
}
__host__ __device__ inline uint32_t prism_guard_requested_gear(const PrismPhysicalGuardSnapshotV1& s) {
    if(s.science_code==2u || s.science_override==PRISM_GEAR_ABORT || s.host_ceiling==PRISM_GEAR_ABORT)
        return PRISM_GEAR_ABORT;
    uint32_t request=prism_guard_effective_request(s.science_status,s.science_request,s.applied_gear);
    if(s.science_override<=PRISM_GEAR_MAX_ACTIVE)request=prism_gear_min_u32(request,s.science_override);
    if(s.host_ceiling<=PRISM_GEAR_MAX_ACTIVE)request=prism_gear_min_u32(request,s.host_ceiling);
    return request;
}
__host__ __device__ inline bool prism_guard_next_epoch(uint64_t epoch,uint64_t evaluated,uint64_t last_begin) {
    return epoch!=0 && evaluated!=UINT64_MAX && epoch==evaluated+1 && epoch!=last_begin;
}
__host__ __device__ inline double prism_guard_relative_residual(double previous,double current,double work) {
    const double expected=previous+work;
    // A relative residual has no usable denominator in this existing band.
    // Preserve it as nonfinite so the measured-drift emergency gate applies.
    return fabs(expected)>1.0e-12 ? fabs(current-expected)/fabs(expected) : NAN;
}

// Measurement contract v1: all existing numerical bands act on this explicitly
// changed quantity. They are not claimed to retain an independently calibrated
// scientific error meaning. No potential reference or residual is added to work.
__host__ __device__ inline uint32_t prism_guard_energy_history_status(bool established,double previous_kinetic) {
    return !established ? PRISM_ENERGY_BOOTSTRAP :
        (isfinite(previous_kinetic) && previous_kinetic>0.0 ?
            PRISM_ENERGY_ESTABLISHED_VALID : PRISM_ENERGY_INVALID_REFERENCE);
}
__host__ __device__ inline double prism_guard_component_residual(double up,double uc,double kp,double kc,double work) {
    return ((uc-up)+(kc-kp))-work;
}
__host__ __device__ inline double prism_guard_kinetic_normalized_residual(double up,double uc,double kp,double kc,double work) {
    if(prism_guard_energy_history_status(true,kp)!=PRISM_ENERGY_ESTABLISHED_VALID ||
        !isfinite(up)||!isfinite(uc)||!isfinite(kc)||kc<0.0||!isfinite(work))return NAN;
    const double residual=prism_guard_component_residual(up,uc,kp,kc,work);
    return isfinite(residual) ? fabs(residual)/kp : NAN;
}

// Pure contract shared by the native commit node and host tests. Failures leave
// the journal and accumulated time unchanged; capture capacity is checked earlier.
__host__ __device__ inline bool prism_physical_time_commit(PrismPhysicalGuardV1* g,uint64_t step,double live_dt) {
    if(!g||!g->time_rows||step!=g->step_at_begin||step!=g->time_count+1u||
        g->time_count>=g->time_capacity||live_dt!=g->step_dt_ps||
        !isfinite(g->step_dt_ps)||g->step_dt_ps<=0.0) return false;
    const double next=g->cumulative_time_ps+g->step_dt_ps;
    if(!isfinite(next))return false;
    g->time_rows[g->time_count]={step,g->step_dt_ps,next};
    ++g->time_count;g->cumulative_time_ps=next;
    return true;
}

// Explicit body-dt admission profiles. Selected measurement is not a discrete
// gearbox transition: gear 0 is its measurement identity, not a claimed 0.5 fs.
enum : uint32_t { PRISM_GUARD_DT_ORDINARY=1u, PRISM_GUARD_DT_SHARED_SELECTED=2u };

// Pure validation used by the actual native selected-begin kernel and CPU
// negative controls. The owner retains the shared allocation and freezes it
// through all eight endpoint evaluations; acceptance mutates it afterward.
__host__ __device__ inline uint32_t prism_guard_selected_step_status_v1(
    const PrismPhysicalGuardV1& guard, const BiprismSharedStepFfi& shared,
    uint64_t expected_generation, uint32_t leaf_id, uint32_t expected_source) {
    const auto& b=guard.bindings;
    const auto& s=guard.snapshot;
    if (!b.governor || !b.dt || !guard.time_step || !guard.time_rows || !guard.energy_rows ||
        b.source!=expected_source || leaf_id>=8u || expected_generation==0u ||
        shared.abi_magic!=BIPRISM_SHARED_STEP_MAGIC || shared.generation!=expected_generation ||
        shared.state!=BIPRISM_SHARED_STEP_SELECTED || shared.sticky_fault!=0u ||
        (shared.policy!=BIPRISM_SHARED_STEP_COMMON_MIN && shared.policy!=BIPRISM_SHARED_STEP_BOUNDED_COMMON_MIN) || (shared.terminal_frozen_mask&~255u)!=0u)
        return PRISM_GUARD_OWNER_MISMATCH;
    const auto& g=*b.governor;
    if (g.policy.armed || g.policy.chunk_span_steps!=1u || !g.bindings.physical_epoch)
        return PRISM_GUARD_DT_CONFLICT;
    const uint64_t epoch=*g.bindings.physical_epoch;
    if (shared.accepted_bodies==UINT32_MAX || shared.scheduled_step==0u ||
        shared.last_accepted_step==UINT32_MAX ||
        shared.scheduled_step!=shared.last_accepted_step+1u ||
        shared.scheduled_step!=*guard.time_step ||
        guard.time_count!=shared.accepted_bodies || s.physical_eval_epoch!=shared.accepted_bodies ||
        epoch!=static_cast<uint64_t>(shared.accepted_bodies)+1u ||
        shared.scheduled_step!=guard.time_count+1u ||
        !prism_guard_next_epoch(epoch,s.physical_eval_epoch,guard.last_begin_epoch) ||
        guard.time_count>=guard.time_capacity ||
        !isfinite(shared.cumulative_accepted_time_ps) || shared.cumulative_accepted_time_ps<0.0 ||
        shared.cumulative_accepted_time_ps!=guard.cumulative_time_ps)
        return PRISM_GUARD_EPOCH_MISMATCH;
    if (s.status==PRISM_GUARD_UNINITIALIZED) {
        if (shared.accepted_bodies!=0u || guard.last_begin_epoch!=0u || s.dt_status!=0u ||
            s.initial_gear!=PRISM_GEAR_UNSET || s.applied_gear!=PRISM_GEAR_UNSET)
            return PRISM_GUARD_DT_CONFLICT;
    } else if (s.status!=PRISM_GUARD_COMPLETE || s.dt_status!=PRISM_GUARD_DT_SHARED_SELECTED ||
        s.initial_gear!=0u || s.applied_gear!=0u || s.transition_pending!=0u ||
        guard.last_begin_epoch!=s.physical_eval_epoch) {
        return PRISM_GUARD_DT_CONFLICT;
    }
    float minimum=shared.requested_dt_ps[0], maximum=minimum;
    for (uint32_t i=0;i<8u;++i) {
        const float request=shared.requested_dt_ps[i];
        if (!isfinite(request) || request<=0.0f || shared.protocol_control_step[i]==0u ||
            shared.protocol_control_step[i]>shared.scheduled_step)
            return PRISM_GUARD_DT_CONFLICT;
        minimum=request<minimum?request:minimum;
        maximum=request>maximum?request:maximum;
    }
    const float selected=shared.native_dt_ps;
    const float selected_au=static_cast<float>(static_cast<double>(selected)*41341.37333517015);
    if (!isfinite(selected) || selected<=0.0f ||
        (shared.policy==BIPRISM_SHARED_STEP_COMMON_MIN ? selected!=minimum : selected>minimum) ||
        shared.minimum_request_ps!=minimum || shared.maximum_request_ps!=maximum ||
        !isfinite(shared.outer_dt_au) || shared.outer_dt_au<=0.0f || shared.outer_dt_au!=selected_au ||
        *b.dt!=selected)
        return PRISM_GUARD_DT_CONFLICT;
    return PRISM_GUARD_READY;
}
