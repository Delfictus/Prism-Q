#include "prism_physical_guard.cuh"
#include <cassert>
#include <cmath>
#include <cstdio>

int main() {
    static_assert(sizeof(PrismGearGovernorDeviceV2)==384);
    static_assert(sizeof(PrismGearGovernorSnapshotV2)==208);
    static_assert(sizeof(PrismPhysicalGuardBindingsV1)==80);
    static_assert(sizeof(PrismPhysicalGuardSnapshotV1)==192);
    static_assert(sizeof(PrismPhysicalGuardV1)==416);
    PrismGearGovernorPolicy p{};
    p.schema_version=1;p.armed=1;p.chunk_span_steps=1;p.max_uncorrected_window_fs=50;
    p.gear_dt_fs[0]=0.5f;p.gear_dt_fs[1]=2.0f;p.gear_dt_fs[2]=4.0f;
    p.hmr_permission=PRISM_GEAR_PERMISSION_GRANTED;p.constraint_permission=PRISM_GEAR_PERMISSION_GRANTED;
    p.unknown_topology_cap_gear=1;p.drift_abort=0.01f;
    p.drift_downshift_fraction=0.5f;p.drift_hold_fraction=0.25f;p.drift_upshift_fraction=0.1f;
    p.force_clamp_reference=10000;p.force_abort_fraction=1;p.force_downshift_fraction=0.8f;
    p.force_hold_fraction=0.5f;p.force_upshift_fraction=0.2f;
    p.constraint_failure_downshift=1;p.constraint_failure_hold=1;p.require_constraint_guard_for_upshift=1;
    p.max_upshift_step=1;
    p.min_upshift_dwell_steps=2;p.instability_lockout_steps=2;
    assert(prism_guard_gear_for_dt(p,0.0005f)==0);
    assert(prism_guard_gear_for_dt(p,0.002f)==1);
    assert(prism_guard_gear_for_dt(p,0.004f)==2);
    assert(prism_guard_gear_for_dt(p,0.001f)==PRISM_GEAR_UNSET);
    assert(prism_guard_gear_for_dt(p,NAN)==PRISM_GEAR_UNSET);
    assert(prism_guard_gear_for_dt(p,0)==PRISM_GEAR_UNSET);
    assert(prism_gear_chunk_latency_ceiling(&p)==2);
    p.chunk_span_steps=500;assert(prism_gear_chunk_latency_ceiling(&p)==PRISM_GEAR_UNSET);
    p.chunk_span_steps=1;
    PrismPhysicalGuardSnapshotV1 s{};
    s.applied_gear=1;s.science_code=s.science_override=s.host_ceiling=PRISM_GEAR_UNSET;
    s.science_request=2;
    assert(prism_guard_requested_gear(s)==1);
    s.science_status=PRISM_SCIENCE_PUBLISHED;
    assert(prism_guard_requested_gear(s)==2);
    s.host_ceiling=0;assert(prism_guard_requested_gear(s)==0);
    s.host_ceiling=2;s.science_override=1;assert(prism_guard_requested_gear(s)==1);
    s.science_override=PRISM_GEAR_ABORT;assert(prism_guard_requested_gear(s)==PRISM_GEAR_ABORT);
    s.science_override=PRISM_GEAR_UNSET;s.science_code=2;assert(prism_guard_requested_gear(s)==PRISM_GEAR_ABORT);
    s.science_code=0;s.host_ceiling=PRISM_GEAR_ABORT;assert(prism_guard_requested_gear(s)==PRISM_GEAR_ABORT);
    PrismGearGuardObservation o{};
    o.current_gear=1;o.requested_gear=2;o.adjudication_code=PRISM_GEAR_UNSET;
    PrismGearDecision d{};
    prism_gearbox_governor_decide(&p,&o,nullptr,&d);
    assert(d.authorized_gear<=1 && d.downshift_latency_steps==1);
    o.drift_measured=1;o.drift=0;o.force_measured=1;o.force_norm=1;
    o.constraint_measured=1;o.constraint_failures=0;o.stable_dwell_steps=10;o.steps_since_instability=10;
    prism_gearbox_governor_decide(&p,&o,nullptr,&d);assert(d.authorized_gear==2);
    o.force_norm=10000;prism_gearbox_governor_decide(&p,&o,nullptr,&d);assert(d.authorized_gear==PRISM_GEAR_ABORT);
    o.force_norm=1;o.drift=0.008f;prism_gearbox_governor_decide(&p,&o,nullptr,&d);assert(d.authorized_gear==0);
    o.drift=0;o.constraint_failures=1;prism_gearbox_governor_decide(&p,&o,nullptr,&d);assert(d.authorized_gear==0);
    float before[3]={2,-3,4}, after[3]={4,-6,8};
    const double delta=prism_guard_transition_delta_k(12,before,after);
    assert(std::fabs(delta-(0.5*12*87/418.4))<1e-14);
    assert(prism_guard_transition_delta_k(12,after,before)==-delta);
    assert(prism_guard_transition_delta_k(12,before,before)==0);
    assert(prism_guard_next_epoch(1,0,0));
    assert(prism_guard_next_epoch(501,500,500));
    assert(!prism_guard_next_epoch(501,500,501));
    assert(!prism_guard_next_epoch(502,500,500));
    assert(!prism_guard_next_epoch(0,UINT64_MAX,UINT64_MAX));
    assert(std::isnan(prism_guard_relative_residual(0,1,0)));
    assert(std::isnan(prism_guard_relative_residual(1,1,-1)));
    assert(std::isnan(prism_guard_relative_residual(1e-13,1,0)));
    assert(prism_guard_relative_residual(10,12,2)==0);
    static_assert(sizeof(PrismPhysicalEnergyBalanceRecordV1)==160);
    assert(prism_guard_energy_history_status(false,NAN)==PRISM_ENERGY_BOOTSTRAP);
    assert(prism_guard_energy_history_status(true,0)==PRISM_ENERGY_INVALID_REFERENCE);
    assert(prism_guard_energy_history_status(true,-1)==PRISM_ENERGY_INVALID_REFERENCE);
    assert(prism_guard_energy_history_status(true,NAN)==PRISM_ENERGY_INVALID_REFERENCE);
    assert(prism_guard_energy_history_status(true,INFINITY)==PRISM_ENERGY_INVALID_REFERENCE);
    assert(prism_guard_energy_history_status(true,2)==PRISM_ENERGY_ESTABLISHED_VALID);
    const double raw=prism_guard_component_residual(-100,-97,10,12,4);
    assert(raw==1 && prism_guard_kinetic_normalized_residual(-100,-97,10,12,4)==0.1);
    assert(prism_guard_component_residual(900,903,10,12,4)==raw);
    assert(prism_guard_kinetic_normalized_residual(900,903,10,12,4)==0.1);
    // Exactly representable potential offset cancellation, with unchanged KE.
    assert(prism_guard_kinetic_normalized_residual(-10,-12,10,12,0)==0);
    // Body work and carried gear work remain independently measured inputs.
    assert(prism_guard_component_residual(1,2,10,14,2+3)==0);
    assert(std::isnan(prism_guard_kinetic_normalized_residual(1,2,0,2,0)));
    assert(std::isnan(prism_guard_kinetic_normalized_residual(1,2,1,NAN,0)));
    assert(std::isnan(prism_guard_kinetic_normalized_residual(1,2,1,2,INFINITY)));
    o.constraint_failures=0;o.drift=static_cast<float>(prism_guard_kinetic_normalized_residual(0,1,0,1,0));
    prism_gearbox_governor_decide(&p,&o,nullptr,&d);assert(d.authorized_gear==PRISM_GEAR_ABORT);
    std::puts("PASS: physical guard policy/ABI/cold science/override/work/epoch checks; CPU only");
}
