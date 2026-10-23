//! Package 3 acceptance — PHYSICAL STABILITY GOVERNOR gate behaviour.
//!
//! These tests live in an integration target rather than a `#[cfg(test)]`
//! module because pre-existing E0063 breakage in unrelated lib tests
//! (`solvate.rs` / `rt_targets.rs` / `input.rs` constructing
//! `PrismPrepTopology` without `dimer_dyad`) prevents the lib test crate from
//! compiling. The integration target compiles independently and exercises the
//! same public API.
//!
//! Run with:
//! ```text
//! cargo test --release -p prism-nhs --test physical_governor_gates \
//!   --no-default-features --features gpu,ensemble,v2_ignition --locked --offline
//! ```
//!
//! Every test here is pure host computation. No CUDA context is created.

use prism_nhs::physical_governor as pg;

// ═══════════════════════════════════════════════════════════════════════════
// Fixtures
// ═══════════════════════════════════════════════════════════════════════════

/// Base inputs describing a run at the 2.0 fs safety gear whose science is
/// asking for the 4.0 fs campaign gear, with the force guard measured and calm.
fn base_inputs() -> pg::BaseInputs {
    pg::BaseInputs {
        gear_dt_fs: [0.5, 2.0, 4.0, 0.0],
        chunk_span_steps: 1,
        policy_digest: 0xDEAD_BEEF_CAFE_F00D,
        current_gear: pg::GEAR_SAFE,
        requested_gear: pg::GEAR_CAMPAIGN,
        force_measured: 1,
        force_norm: 100.0,
    }
}

/// A base decision that authorized the campaign gear — the most permissive
/// outcome the base governor can produce. Every test that shows a guard
/// blocking an upshift starts from here, so the block is unambiguously the
/// extension layer's doing.
fn permissive_base() -> pg::BaseDecision {
    pg::BaseDecision {
        authorized_gear: pg::GEAR_CAMPAIGN,
        ceiling_gear: pg::GEAR_CAMPAIGN,
        binding_gate: pg::BASE_GATE_NONE,
        gate_verdicts: 0,
        emergency_flags: 0,
        next_stable_dwell_steps: 500,
        next_steps_since_instability: 500,
        downshift_latency_steps: 1,
    }
}

/// Every extension guard MEASURED and inside its OK band, HMR topology-verified.
fn all_clear() -> pg::ExtObservation {
    pg::ExtObservation {
        schema_version: pg::PHYSGOV_SCHEMA_VERSION,
        hmr_applied_verified: pg::PERMISSION_GRANTED,
        force_rate_measured: 1,
        force_norm_prev: 100.0,
        dt_prev_ps: 0.002,
        accel_measured: 1,
        max_accel_a_per_ps2: 100.0,
        thermostat_measured: 1,
        effective_gamma_per_ps: 1.0,
        instantaneous_temp_k: 300.0,
        target_temp_k: 300.0,
        integration_measured: 1,
        max_neighbor_count: 100,
        neighbor_list_capacity: 1024,
        clamped_force_fraction: 0.0,
        integration_faults: 0,
        tsh_memory_measured: 1,
        tsh_instability_memory: 0.0,
        observation_confidence: 1.0,
        ps_since_last_observation: 0.01,
        steps_since_sentinel: 0,
        phase_measured: 1,
        phase_id: 0,
        _reserved: [0; 5],
    }
}

fn armed_policy() -> pg::ExtPolicy {
    let mut p = pg::ExtPolicy::default();
    p.armed = 1;
    p
}

fn decide(obs: &pg::ExtObservation) -> pg::PhysGovDecision {
    pg::compose(&base_inputs(), &armed_policy(), obs, &permissive_base())
}

// ═══════════════════════════════════════════════════════════════════════════
// Baseline: the all-clear case must actually reach the campaign gear, or every
// "this guard blocked the upshift" assertion below would be vacuous.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn all_clear_reaches_the_campaign_gear() {
    let d = decide(&all_clear());
    assert_eq!(
        d.authorized_gear,
        pg::GEAR_CAMPAIGN,
        "all-clear must upshift; otherwise the per-guard blocking tests prove nothing. \
         ceiling={} ext_gate={} ext_verdicts={:#010x}",
        d.ceiling_gear,
        d.ext_binding_gate,
        d.ext_gate_verdicts
    );
    assert_eq!(d.transition, pg::TRANSITION_UPSHIFT);
    assert_eq!(d.ext_binding_gate, pg::GATE_NONE);
    assert!(
        (d.dt_applied_ps - 0.004).abs() < 1e-9,
        "gear 2 is 4.0 fs = 0.004 ps"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 3 — the HARD PHYSICS GATE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hmr_gate_blocks_gear_two_when_hmr_applied_is_false() {
    for permission in [pg::PERMISSION_DENIED, pg::PERMISSION_UNKNOWN] {
        let mut obs = all_clear();
        obs.hmr_applied_verified = permission;
        let d = decide(&obs);
        assert!(
            d.authorized_gear <= pg::GEAR_SAFE,
            "4.0 fs without verified HMR corrupts trajectories silently; \
             permission={permission} authorized={}",
            d.authorized_gear
        );
        assert!(d.ceiling_gear <= pg::GEAR_SAFE);
        assert_eq!(
            d.ext_binding_gate,
            pg::GATE_HMR_HARD,
            "a 4 fs refusal must be attributed to the HMR gate, not to an \
             incidental later gate"
        );
        assert_eq!(d.binding_gate_rank, pg::RANK_HMR_HARD);
    }
}

#[test]
fn hmr_gate_is_not_policy_overridable() {
    // Loosen EVERY switch a policy file can touch, then re-run with HMR
    // unverified. No combination of policy values may buy gear 2.
    let mut p = armed_policy();
    p.require_accel_guard_for_upshift = 0;
    p.require_thermostat_guard_for_upshift = 0;
    p.require_integration_guard_for_upshift = 0;
    p.require_observation_confidence = 0;
    p.max_steps_without_sentinel = 0;
    p.force_rate_hold_per_ps = f32::MAX;
    p.force_rate_downshift_per_ps = f32::MAX;
    p.force_rate_floor_per_ps = f32::MAX;
    p.accel_ok_fraction = f32::MAX;
    p.gamma_dt_ok = f32::MAX;
    p.temp_dev_ok_fraction = f32::MAX;
    p.neighbor_occupancy_ok = f32::MAX;
    p.clamped_fraction_hold = f32::MAX;
    p.tsh_memory_hold = f32::MAX;
    p.min_observation_confidence = 0.0;
    p.max_observation_interval_ps = f32::MAX;
    p.phase_ceiling_gear = [pg::GEAR_CAMPAIGN; 8];

    let mut obs = all_clear();
    obs.hmr_applied_verified = pg::PERMISSION_DENIED;

    let d = pg::compose(&base_inputs(), &p, &obs, &permissive_base());
    assert!(
        d.authorized_gear <= pg::GEAR_SAFE,
        "the HMR gate must survive a fully-permissive policy; authorized={}",
        d.authorized_gear
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 2 — LEXICOGRAPHIC, never a weighted score
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn salience_cannot_override_physical_prohibition() {
    // The strongest scientific outcome expressible in this system is a base
    // decision that already authorized the campaign gear — that is what an
    // arbitrarily large salience buys, because the base treats salience as a
    // ceiling and `min(request, ceiling)` saturates at gear 2. There is no
    // larger number to supply.
    //
    // Under a weighted score there would exist some salience weight large
    // enough to outvote a physical prohibition. Here there is none, because the
    // fold is `min` and the physical gates publish the smaller operand.
    let physical_prohibitions: &[(&str, fn(&mut pg::ExtObservation))] = &[
        ("hmr_denied", |o| {
            o.hmr_applied_verified = pg::PERMISSION_DENIED
        }),
        ("nonfinite_temperature", |o| {
            o.instantaneous_temp_k = f32::NAN
        }),
        ("thermostat_gamma_dt_floor", |o| {
            o.effective_gamma_per_ps = 1.0e6
        }),
        ("neighbor_list_truncating", |o| o.max_neighbor_count = 1024),
        ("force_clipping", |o| o.clamped_force_fraction = 0.5),
        ("force_spike", |o| o.force_norm_prev = 1.0),
        ("acceleration_extreme", |o| o.max_accel_a_per_ps2 = 1.0e6),
    ];

    for (name, degrade) in physical_prohibitions {
        let mut obs = all_clear();
        degrade(&mut obs);
        let d = pg::compose(&base_inputs(), &armed_policy(), &obs, &permissive_base());
        assert!(
            d.authorized_gear < pg::GEAR_CAMPAIGN,
            "{name}: a physical prohibition must survive the maximum possible \
             scientific demand; authorized={} ceiling={} ext_gate={}",
            d.authorized_gear,
            d.ceiling_gear,
            d.ext_binding_gate
        );
    }
}

#[test]
fn authority_ranks_are_strictly_ordered() {
    // The operator order:
    //   (physical_emergency, topology_permission, constraint_stability,
    //    energy_drift, force_stability, tsh_memory, tsh_salience,
    //    phase_requirement, current_gear)
    let chain = [
        pg::RANK_PHYSICAL_EMERGENCY,
        pg::RANK_HMR_HARD,
        pg::RANK_TOPOLOGY_PERMISSION,
        pg::RANK_CONSTRAINT_STABILITY,
        pg::RANK_ENERGY_DRIFT,
        pg::RANK_FORCE_STABILITY,
        pg::RANK_FORCE_SPIKE,
        pg::RANK_ACCELERATION_EXTREME,
        pg::RANK_THERMOSTAT_STABILITY,
        pg::RANK_INTEGRATION_HEALTH,
        pg::RANK_CHUNK_LATENCY,
        pg::RANK_TSH_MEMORY,
        pg::RANK_TSH_SALIENCE,
        pg::RANK_PHASE_REQUIREMENT,
        pg::RANK_DWELL,
        pg::RANK_LOCKOUT,
        pg::RANK_UPSHIFT_RATE,
        pg::RANK_NONE,
    ];
    for w in chain.windows(2) {
        assert!(
            w[0] < w[1],
            "authority ranks must be strictly increasing: {w:?}"
        );
    }
    // Every physical gate outranks every scientific one.
    assert!(pg::RANK_INTEGRATION_HEALTH < pg::RANK_TSH_MEMORY);
    assert!(pg::RANK_INTEGRATION_HEALTH < pg::RANK_TSH_SALIENCE);
    assert!(pg::RANK_HMR_HARD < pg::RANK_TSH_SALIENCE);
}

#[test]
fn more_authoritative_gate_wins_attribution_at_the_same_ceiling() {
    // HMR unverified pins the ceiling to gear 1. A phase requirement that
    // *also* pins gear 1 must not steal the attribution: a 4 fs refusal reading
    // as a "phase" finding would misdirect every downstream reader.
    let mut obs = all_clear();
    obs.hmr_applied_verified = pg::PERMISSION_UNKNOWN;
    obs.phase_id = 1; // ramp_up, ceiling gear 1
    let d = decide(&obs);
    assert_eq!(d.ceiling_gear, pg::GEAR_SAFE);
    assert_eq!(d.ext_binding_gate, pg::GATE_HMR_HARD);
    assert_eq!(d.binding_gate_rank, pg::RANK_HMR_HARD);
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 1/8 — each guard independently blocks upshift
// ═══════════════════════════════════════════════════════════════════════════

/// Degradations that must each, on their own, prevent the upshift the base
/// already licensed. `(name, mutation, expected extension gate)`.
fn independent_degradations() -> Vec<(&'static str, fn(&mut pg::ExtObservation), u32)> {
    vec![
        (
            "hmr_unknown",
            (|o| o.hmr_applied_verified = pg::PERMISSION_UNKNOWN) as fn(&mut pg::ExtObservation),
            pg::GATE_HMR_HARD,
        ),
        // rate = (112-100)/(100*0.002) = 60 /ps -> HOLD band [50,100)
        (
            "force_spike_hold",
            |o| o.force_norm_prev = 100.0,
            pg::GATE_FORCE_SPIKE,
        ),
        // dx at 4 fs = 0.5*3000*(0.004)^2 = 0.024 A >= 0.02 A -> gear 2 refused
        (
            "acceleration_extreme",
            |o| o.max_accel_a_per_ps2 = 3000.0,
            pg::GATE_ACCELERATION_EXTREME,
        ),
        // gamma*dt at 4 fs = 20*0.004 = 0.08 >= 0.05 -> gear 2 refused
        (
            "thermostat_gamma_dt",
            |o| o.effective_gamma_per_ps = 20.0,
            pg::GATE_THERMOSTAT_STABILITY,
        ),
        // |350-300|/300 = 0.167 -> inside the HOLD band [0.10,0.25)
        (
            "thermostat_setpoint",
            |o| o.instantaneous_temp_k = 350.0,
            pg::GATE_THERMOSTAT_STABILITY,
        ),
        // 850/1024 = 0.83 >= 0.80 -> HOLD
        (
            "neighbor_occupancy",
            |o| o.max_neighbor_count = 850,
            pg::GATE_INTEGRATION_HEALTH,
        ),
        // ANY force clipping is non-Hamiltonian
        (
            "force_clamping",
            |o| o.clamped_force_fraction = 1.0e-5,
            pg::GATE_INTEGRATION_HEALTH,
        ),
        // anti-blindness (d)
        (
            "low_observation_confidence",
            |o| o.observation_confidence = 0.2,
            pg::GATE_TSH_MEMORY,
        ),
        // anti-blindness (a)
        (
            "observation_cadence_overdue",
            |o| o.ps_since_last_observation = 2.0,
            pg::GATE_TSH_MEMORY,
        ),
        // anti-blindness (b)
        (
            "sentinel_overdue",
            |o| o.steps_since_sentinel = 20_000,
            pg::GATE_TSH_MEMORY,
        ),
        (
            "tsh_instability_memory",
            |o| o.tsh_instability_memory = 0.3,
            pg::GATE_TSH_MEMORY,
        ),
        // uv_burst phase demands 0.5 fs
        (
            "uv_burst_phase",
            |o| o.phase_id = 5,
            pg::GATE_PHASE_REQUIREMENT,
        ),
        // unmeasured guards block upshift by default
        (
            "accel_unmeasured",
            |o| o.accel_measured = 0,
            pg::GATE_ACCELERATION_EXTREME,
        ),
        (
            "thermostat_unmeasured",
            |o| o.thermostat_measured = 0,
            pg::GATE_THERMOSTAT_STABILITY,
        ),
        (
            "integration_unmeasured",
            |o| o.integration_measured = 0,
            pg::GATE_INTEGRATION_HEALTH,
        ),
    ]
}

#[test]
fn each_guard_independently_blocks_upshift() {
    for (name, degrade, expected_gate) in independent_degradations() {
        let mut obs = all_clear();
        if name == "force_spike_hold" {
            // Force spike needs a delta; move the *current* norm, not the prior.
            let mut inputs = base_inputs();
            inputs.force_norm = 112.0;
            degrade(&mut obs);
            let d = pg::compose(&inputs, &armed_policy(), &obs, &permissive_base());
            assert!(
                d.authorized_gear <= inputs.current_gear,
                "{name} must block the upshift; authorized={}",
                d.authorized_gear
            );
            assert_eq!(d.ext_binding_gate, expected_gate, "{name} attribution");
            continue;
        }
        degrade(&mut obs);
        let d = decide(&obs);
        assert!(
            d.authorized_gear <= pg::GEAR_SAFE,
            "{name} must block the upshift to gear 2; authorized={} ceiling={} \
             ext_gate={} verdicts={:#010x}",
            d.authorized_gear,
            d.ceiling_gear,
            d.ext_binding_gate,
            d.ext_gate_verdicts
        );
        assert_eq!(
            d.ext_binding_gate, expected_gate,
            "{name}: wrong attribution; got gate {} rank {}",
            d.ext_binding_gate, d.binding_gate_rank
        );
        assert_ne!(
            d.ext_verdict(expected_gate),
            pg::VERDICT_OK,
            "{name}: the blocking gate must not record an OK verdict"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 4 — DOWNSHIFT / UPSHIFT ASYMMETRY
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn downshift_is_immediate_and_may_skip_gears() {
    // Running at the campaign gear when the neighbour list starts truncating:
    // gear 2 -> gear 0 in ONE decision is legal and expected.
    let mut inputs = base_inputs();
    inputs.current_gear = pg::GEAR_CAMPAIGN;
    let mut obs = all_clear();
    obs.max_neighbor_count = 1024; // occupancy 1.0 -> FLOOR

    let d = pg::compose(&inputs, &armed_policy(), &obs, &permissive_base());
    assert_eq!(
        d.authorized_gear,
        pg::GEAR_FINE,
        "a FLOOR-class guard must drop straight to the finest gear"
    );
    assert_eq!(d.transition, pg::TRANSITION_DOWNSHIFT);
    assert!(
        d.authorized_gear + 1 < inputs.current_gear,
        "gear skipping must be possible"
    );
}

#[test]
fn downshift_requires_no_dwell_but_upshift_does() {
    // A base decision that has NOT accumulated dwell still downshifts freely
    // under the extension guards, while an upshift under the same conditions is
    // refused. Downshift must be strictly easier and faster than upshift.
    let mut inputs = base_inputs();
    inputs.current_gear = pg::GEAR_CAMPAIGN;

    let mut degrading = all_clear();
    degrading.clamped_force_fraction = 0.5; // FLOOR class

    let no_dwell_base = pg::BaseDecision {
        next_stable_dwell_steps: 0,
        next_steps_since_instability: 0,
        ..permissive_base()
    };
    let down = pg::compose(&inputs, &armed_policy(), &degrading, &no_dwell_base);
    assert_eq!(
        down.authorized_gear,
        pg::GEAR_FINE,
        "downshift needs no dwell"
    );
    assert_eq!(down.transition, pg::TRANSITION_DOWNSHIFT);

    // Same zero-dwell state, guards clean, science asking to go coarser: the
    // extension layer must not manufacture an upshift the guards did not earn.
    let mut up_inputs = base_inputs();
    up_inputs.current_gear = pg::GEAR_FINE;
    let mut one_guard_off = all_clear();
    one_guard_off.observation_confidence = 0.1;
    let up = pg::compose(
        &up_inputs,
        &armed_policy(),
        &one_guard_off,
        &permissive_base(),
    );
    assert_eq!(
        up.authorized_gear,
        pg::GEAR_FINE,
        "upshift requires ALL guards OK; one degraded guard is enough to refuse"
    );
    assert_eq!(up.transition, pg::TRANSITION_HOLD);
}

#[test]
fn extension_guard_degradation_vetoes_a_base_licensed_upshift() {
    // The base governor may license an upshift on the base guards alone. An
    // extension guard that is merely UNMEASURED must still veto it.
    let mut obs = all_clear();
    obs.accel_measured = 0;
    obs.max_accel_a_per_ps2 = f32::NAN;
    let d = decide(&obs);
    assert_eq!(d.authorized_gear, pg::GEAR_SAFE);
    assert_eq!(d.transition, pg::TRANSITION_HOLD);
    assert_eq!(
        d.ext_verdict(pg::GATE_ACCELERATION_EXTREME),
        pg::VERDICT_UNMEASURED
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Structural: the extension layer can only TIGHTEN
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extension_never_loosens_the_base() {
    // Exhaustive over a grid of base decisions x guard degradations.
    let degradations = independent_degradations();
    for base_authorized in [pg::GEAR_FINE, pg::GEAR_SAFE, pg::GEAR_CAMPAIGN] {
        for base_ceiling in [pg::GEAR_FINE, pg::GEAR_SAFE, pg::GEAR_CAMPAIGN] {
            for current in [pg::GEAR_FINE, pg::GEAR_SAFE, pg::GEAR_CAMPAIGN] {
                let base = pg::BaseDecision {
                    authorized_gear: base_authorized,
                    ceiling_gear: base_ceiling,
                    ..permissive_base()
                };
                let mut inputs = base_inputs();
                inputs.current_gear = current;

                // Clean observation first.
                let clean = pg::compose(&inputs, &armed_policy(), &all_clear(), &base);
                assert!(
                    clean.authorized_gear <= base_authorized,
                    "clean: authorized {} > base {}",
                    clean.authorized_gear,
                    base_authorized
                );
                assert!(clean.ceiling_gear <= base_ceiling);

                for (name, degrade, _) in &degradations {
                    let mut obs = all_clear();
                    degrade(&mut obs);
                    let d = pg::compose(&inputs, &armed_policy(), &obs, &base);
                    if d.authorized_gear == pg::GEAR_ABORT {
                        continue; // an abort is a tightening, not a loosening
                    }
                    assert!(
                        d.authorized_gear <= base_authorized,
                        "{name}: extension loosened the base \
                         (authorized {} > base {}) at current={current}",
                        d.authorized_gear,
                        base_authorized
                    );
                    assert!(
                        d.ceiling_gear <= base_ceiling,
                        "{name}: extension loosened the base ceiling \
                         ({} > {}) at current={current}",
                        d.ceiling_gear,
                        base_ceiling
                    );
                }
            }
        }
    }
}

#[test]
fn base_emergency_is_terminal() {
    let base = pg::BaseDecision {
        authorized_gear: pg::GEAR_ABORT,
        ceiling_gear: pg::GEAR_ABORT,
        binding_gate: pg::BASE_GATE_PHYSICAL_EMERGENCY,
        emergency_flags: 0x4, // nonfinite energy
        ..permissive_base()
    };
    let d = pg::compose(&base_inputs(), &armed_policy(), &all_clear(), &base);
    assert_eq!(d.authorized_gear, pg::GEAR_ABORT);
    assert_eq!(d.transition, pg::TRANSITION_ABORT);
    assert_eq!(d.binding_gate_rank, pg::RANK_PHYSICAL_EMERGENCY);
    assert!(d.dt_applied_ps.is_nan(), "gear 3 has no timestep");
    assert_eq!(d.emergency_flags, 0x4, "the base cause bits must survive");
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 5 — GRADUATED response, not abort-only
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn force_spike_response_is_graduated() {
    // Bands are anchored on the relative growth rate of max|F| per simulated ps.
    // prev = 100, dt_prev = 0.002 ps => rate = (now-100)/0.2
    let cases = [
        (100.0_f32, pg::VERDICT_OK, pg::GEAR_CAMPAIGN), // rate 0
        (109.0, pg::VERDICT_OK, pg::GEAR_CAMPAIGN),     // rate 45  < 50
        (112.0, pg::VERDICT_HOLD, pg::GEAR_SAFE),       // rate 60  in [50,100)
        (125.0, pg::VERDICT_DOWNSHIFT, pg::GEAR_FINE),  // rate 125 in [100,250)
        (160.0, pg::VERDICT_FLOOR, pg::GEAR_FINE),      // rate 300 >= 250
    ];
    for (now, expect_verdict, expect_gear) in cases {
        let mut inputs = base_inputs();
        inputs.force_norm = now;
        let d = pg::compose(&inputs, &armed_policy(), &all_clear(), &permissive_base());
        assert_eq!(
            d.ext_verdict(pg::GATE_FORCE_SPIKE),
            expect_verdict,
            "force_norm={now} verdict"
        );
        assert_eq!(d.authorized_gear, expect_gear, "force_norm={now} gear");
    }
    // A COLLAPSING force norm is a relaxation, not an instability, and must not
    // be punished with a finer gear.
    let mut inputs = base_inputs();
    inputs.force_norm = 1.0;
    let d = pg::compose(&inputs, &armed_policy(), &all_clear(), &permissive_base());
    assert_eq!(d.ext_verdict(pg::GATE_FORCE_SPIKE), pg::VERDICT_OK);
}

#[test]
fn acceleration_guard_escalates_to_emergency_only_at_the_finest_gear() {
    // dx at gear 0 = 0.5*a*(0.0005)^2 = 1.25e-7*a. Emergency at dx >= 0.5 A
    // => a >= 4.0e6 A/ps^2.
    let mut obs = all_clear();
    obs.max_accel_a_per_ps2 = 1.0e6; // dx(gear0) = 0.125 A -> not an emergency
    let d = decide(&obs);
    assert_ne!(
        d.authorized_gear,
        pg::GEAR_ABORT,
        "1e6 A/ps^2 must not abort"
    );
    assert_eq!(d.authorized_gear, pg::GEAR_FINE);

    obs.max_accel_a_per_ps2 = 8.0e6; // dx(gear0) = 1.0 A -> half a bond exceeded
    let d = decide(&obs);
    assert_eq!(d.authorized_gear, pg::GEAR_ABORT);
    assert_eq!(
        d.ext_emergency_flags & pg::EMERG_ACCEL_DISPLACEMENT,
        pg::EMERG_ACCEL_DISPLACEMENT
    );

    obs.max_accel_a_per_ps2 = f32::NAN;
    let d = decide(&obs);
    assert_eq!(d.authorized_gear, pg::GEAR_ABORT);
    assert_eq!(
        d.ext_emergency_flags & pg::EMERG_NONFINITE_ACCEL,
        pg::EMERG_NONFINITE_ACCEL
    );
}

#[test]
fn thermostat_guard_never_escalates_past_floor() {
    // A legitimate high-friction equilibration must FLOOR the gear, never abort
    // the run: gamma*dt large is an accuracy defect, not a divergence.
    let mut obs = all_clear();
    obs.effective_gamma_per_ps = 1.0e6;
    let d = decide(&obs);
    assert_eq!(d.authorized_gear, pg::GEAR_FINE);
    assert_ne!(d.authorized_gear, pg::GEAR_ABORT);
    assert_eq!(
        d.ext_verdict(pg::GATE_THERMOSTAT_STABILITY),
        pg::VERDICT_FLOOR
    );
    assert_eq!(d.ext_emergency_flags, 0);
}

#[test]
fn integration_fault_is_terminal() {
    let mut obs = all_clear();
    obs.integration_faults = 3;
    let d = decide(&obs);
    assert_eq!(d.authorized_gear, pg::GEAR_ABORT);
    assert_eq!(d.ext_binding_gate, pg::GATE_INTEGRATION_HEALTH);
    assert_eq!(
        d.ext_emergency_flags & pg::EMERG_INTEGRATION_FAULT,
        pg::EMERG_INTEGRATION_FAULT
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 6 — CHUNK-BOUNDARY CONTRACT
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn chunk_boundary_latency_is_a_named_emitted_quantity() {
    for (span, current, expect_fs) in [
        (1_u32, pg::GEAR_SAFE, 2.0_f32),
        (8, pg::GEAR_SAFE, 16.0),
        (8, pg::GEAR_CAMPAIGN, 32.0),
        (16, pg::GEAR_CAMPAIGN, 64.0),
    ] {
        let mut inputs = base_inputs();
        inputs.chunk_span_steps = span;
        inputs.current_gear = current;
        let d = pg::compose(&inputs, &armed_policy(), &all_clear(), &permissive_base());
        assert_eq!(
            d.downshift_latency_steps, span,
            "downshift latency in steps must equal chunk_span_steps"
        );
        assert!(
            (d.downshift_latency_fs - expect_fs).abs() < 1e-4,
            "span={span} current={current}: expected {expect_fs} fs, got {}",
            d.downshift_latency_fs
        );
    }
    // chunk_span_steps == 0 is not a licence to claim zero latency.
    let mut inputs = base_inputs();
    inputs.chunk_span_steps = 0;
    let d = pg::compose(&inputs, &armed_policy(), &all_clear(), &permissive_base());
    assert_eq!(d.downshift_latency_steps, 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Emission and honesty invariants
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a_decision_is_emitted_even_when_the_gear_does_not_change() {
    let mut inputs = base_inputs();
    inputs.current_gear = pg::GEAR_SAFE;
    inputs.requested_gear = pg::GEAR_SAFE;
    let held_base = pg::BaseDecision {
        authorized_gear: pg::GEAR_SAFE,
        ceiling_gear: pg::GEAR_SAFE,
        ..permissive_base()
    };
    let d = pg::compose(&inputs, &armed_policy(), &all_clear(), &held_base);
    assert_eq!(d.authorized_gear, pg::GEAR_SAFE);
    assert_eq!(d.transition, pg::TRANSITION_HOLD, "HOLD is an emitted fact");
    assert_eq!(
        d.decision_emitted, 1,
        "a gap in the record stream must never be ambiguous between \
         'gear held' and 'decision did not run'"
    );
    assert_eq!(d.schema_version, pg::PHYSGOV_SCHEMA_VERSION);
    assert_eq!(
        d.policy_digest, inputs.policy_digest,
        "provenance must be bound"
    );
}

#[test]
fn the_three_gear_facts_are_never_collapsed() {
    let mut obs = all_clear();
    obs.hmr_applied_verified = pg::PERMISSION_DENIED;
    let d = decide(&obs);
    assert_eq!(
        d.requested_gear,
        pg::GEAR_CAMPAIGN,
        "what the science asked for"
    );
    assert_eq!(d.authorized_gear, pg::GEAR_SAFE, "what physics permitted");
    assert_ne!(
        d.requested_gear, d.authorized_gear,
        "request and authorization must remain distinguishable"
    );
    assert_eq!(
        d.base_authorized_gear,
        pg::GEAR_CAMPAIGN,
        "pre-extension fact preserved"
    );
    // device_measured_gear is deliberately absent: it is measured after the
    // fact and joined by gear epoch, never fabricated here.
}

#[test]
fn unmeasured_is_never_fabricated() {
    let obs = pg::ExtObservation::default();
    assert_eq!(obs.hmr_applied_verified, pg::PERMISSION_UNKNOWN);
    assert!(obs.max_accel_a_per_ps2.is_nan());
    assert!(obs.effective_gamma_per_ps.is_nan());
    assert!(obs.clamped_force_fraction.is_nan());
    assert!(obs.observation_confidence.is_nan());
    assert_eq!(obs.max_neighbor_count, pg::U32_UNMEASURED);
    assert_eq!(obs.steps_since_sentinel, pg::U32_UNMEASURED);
    assert_eq!(obs.phase_id, pg::U32_UNMEASURED);

    // A fully unmeasured observation must NOT deadlock the run: the current
    // gear stays reachable, only the coarser gear is withheld.
    let d = decide(&obs);
    assert_eq!(
        d.authorized_gear,
        pg::GEAR_SAFE,
        "current gear stays reachable"
    );
    assert_ne!(
        d.authorized_gear,
        pg::GEAR_ABORT,
        "unmeasured is not an abort"
    );
    assert!(
        d.authorized_gear < pg::GEAR_CAMPAIGN,
        "but the aggressive gear is withheld"
    );
}

#[test]
fn unknown_phase_is_not_licence_to_run_coarse() {
    let mut obs = all_clear();
    obs.phase_measured = 1;
    obs.phase_id = 99;
    let d = decide(&obs);
    assert!(d.authorized_gear <= pg::GEAR_SAFE);
    assert_eq!(
        d.ext_verdict(pg::GATE_PHASE_REQUIREMENT),
        pg::VERDICT_UNMEASURED
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 7 — POLICY RESOLUTION AND PROVENANCE
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn defaults_resolve_with_recorded_origins() {
    let r = pg::resolve_ext_policy(None, None).expect("defaults must resolve");
    assert_eq!(r.policy.armed, 1);
    assert_eq!(r.policy.schema_version, pg::PHYSGOV_SCHEMA_VERSION);
    assert!(r.policy_path.is_none());
    assert!(r.policy_sha256.is_none());
    assert_ne!(r.digest, 0, "the resolved policy must carry a fingerprint");
    assert_eq!(r.digest, r.policy.ext_policy_digest);
    for (field, origin) in &r.origins {
        assert_eq!(
            *origin,
            pg::ValueOrigin::Default,
            "{field} came from nowhere but the defaults"
        );
    }
    let lines = r.receipt_lines();
    assert!(lines.iter().all(|l| l.contains("origin=")));
    assert!(lines.iter().any(|l| l.starts_with("gamma_dt_ok=")));
}

#[test]
fn policy_file_overrides_record_their_origin() {
    let dir = std::env::temp_dir().join("prism_physgov_policy_test");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let path = dir.join("policy.json");
    std::fs::write(
        &path,
        br#"{"schema_version":1,"gamma_dt_ok":0.02,"phase_ceiling_gear":[2,1,2,1,2,0,2,2]}"#,
    )
    .expect("write policy");
    let p = path.to_str().unwrap();

    let r = pg::resolve_ext_policy(Some(p), None).expect("policy must resolve");
    assert!((r.policy.gamma_dt_ok - 0.02).abs() < 1e-9);
    assert_eq!(
        r.origin_of("gamma_dt_ok"),
        Some(pg::ValueOrigin::PolicyFile)
    );
    assert_eq!(r.origin_of("gamma_dt_hold"), Some(pg::ValueOrigin::Default));
    assert_eq!(
        r.origin_of("phase_ceiling_gear"),
        Some(pg::ValueOrigin::PolicyFile)
    );
    let sha = r.policy_sha256.clone().expect("sha recorded");
    assert_eq!(sha.len(), 64);

    // The digest must change when a value changes.
    let base = pg::resolve_ext_policy(None, None).unwrap();
    assert_ne!(r.digest, base.digest);

    // A pinned digest that does not match must fail loudly.
    let wrong = "0".repeat(64);
    let err = pg::resolve_ext_policy(Some(p), Some(&wrong)).unwrap_err();
    assert!(
        matches!(err, pg::ExtPolicyError::DigestMismatch { .. }),
        "{err}"
    );

    // The correct digest must pass.
    pg::resolve_ext_policy(Some(p), Some(&sha)).expect("pinned digest must match");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn schema_mismatch_and_unknown_keys_fail_loudly() {
    let dir = std::env::temp_dir().join("prism_physgov_policy_test");
    std::fs::create_dir_all(&dir).expect("temp dir");

    let bad_schema = dir.join("bad_schema.json");
    std::fs::write(&bad_schema, br#"{"schema_version":99}"#).unwrap();
    let err = pg::resolve_ext_policy(bad_schema.to_str(), None).unwrap_err();
    assert!(
        matches!(err, pg::ExtPolicyError::SchemaMismatch { .. }),
        "{err}"
    );

    let unknown = dir.join("unknown_key.json");
    std::fs::write(&unknown, br#"{"schema_version":1,"not_a_field":1.0}"#).unwrap();
    let err = pg::resolve_ext_policy(unknown.to_str(), None).unwrap_err();
    assert!(matches!(err, pg::ExtPolicyError::Parse { .. }), "{err}");

    let _ = std::fs::remove_file(&bad_schema);
    let _ = std::fs::remove_file(&unknown);
}

#[test]
fn non_monotone_bands_are_rejected() {
    let mut p = pg::ExtPolicy::default();
    p.gamma_dt_hold = 0.001; // below gamma_dt_ok
    let err = pg::validate_ext_policy(&p).unwrap_err();
    assert!(
        matches!(
            err,
            pg::ExtPolicyError::InvalidValue {
                field: "gamma_dt_hold",
                ..
            }
        ),
        "{err}"
    );

    let mut p = pg::ExtPolicy::default();
    p.accel_displacement_reference_a = 0.0;
    let err = pg::validate_ext_policy(&p).unwrap_err();
    assert!(
        matches!(
            err,
            pg::ExtPolicyError::InvalidValue {
                field: "accel_displacement_reference_a",
                ..
            }
        ),
        "{err}"
    );

    let mut p = pg::ExtPolicy::default();
    p.phase_ceiling_gear[3] = 7;
    let err = pg::validate_ext_policy(&p).unwrap_err();
    assert!(
        matches!(
            err,
            pg::ExtPolicyError::InvalidValue {
                field: "phase_ceiling_gear",
                ..
            }
        ),
        "{err}"
    );

    pg::validate_ext_policy(&pg::ExtPolicy::default()).expect("the defaults must validate");
}

// ═══════════════════════════════════════════════════════════════════════════
// HOST <-> DEVICE PARITY
// ═══════════════════════════════════════════════════════════════════════════

/// Parse `#define PRISM_PHYSGOV_DEFAULT_<NAME> <literal>` out of the CUDA
/// header and assert every literal matches the Rust default.
///
/// This is the mechanical guarantee that the host policy layer and the device
/// evaluator cannot drift on a threshold. If someone retunes a band in the
/// header and forgets the Rust mirror (or the reverse), this test fails.
#[test]
fn defaults_match_cuda_header() {
    let header =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/cuda/physical_governor.cuh");
    let text = std::fs::read_to_string(&header)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", header.display()));

    let mut found = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        let Some(rest) = line.strip_prefix("#define PRISM_PHYSGOV_DEFAULT_") else {
            continue;
        };
        let mut it = rest.split_whitespace();
        let (Some(name), Some(literal)) = (it.next(), it.next()) else {
            continue;
        };
        found.insert(name.to_string(), literal.to_string());
    }
    assert!(
        !found.is_empty(),
        "no PRISM_PHYSGOV_DEFAULT_* macros parsed from {}",
        header.display()
    );

    for (name, expected) in pg::DEFAULT_MACRO_TABLE {
        let literal = found.get(*name).unwrap_or_else(|| {
            panic!(
                "PRISM_PHYSGOV_DEFAULT_{name} is missing from {}",
                header.display()
            )
        });
        match expected {
            pg::DefaultLiteral::U32(v) => {
                let parsed: u32 = literal
                    .trim_end_matches('u')
                    .parse()
                    .unwrap_or_else(|e| panic!("{name}: cannot parse '{literal}': {e}"));
                assert_eq!(
                    parsed, *v,
                    "PRISM_PHYSGOV_DEFAULT_{name}: header says {parsed}, Rust says {v}"
                );
            }
            pg::DefaultLiteral::F32(v) => {
                let parsed: f32 = literal
                    .trim_end_matches('f')
                    .parse()
                    .unwrap_or_else(|e| panic!("{name}: cannot parse '{literal}': {e}"));
                assert!(
                    (parsed - *v).abs() <= f32::EPSILON * v.abs().max(1.0),
                    "PRISM_PHYSGOV_DEFAULT_{name}: header says {parsed}, Rust says {v}"
                );
            }
        }
    }

    // Every macro in the header must appear in the Rust table, or a threshold
    // exists on the device that the host cannot resolve or record.
    let known: std::collections::HashSet<&str> =
        pg::DEFAULT_MACRO_TABLE.iter().map(|(n, _)| *n).collect();
    let mut orphans: Vec<&String> = found
        .keys()
        .filter(|k| !known.contains(k.as_str()))
        .collect();
    orphans.sort();
    assert!(
        orphans.is_empty(),
        "header defaults with no Rust mirror (unresolvable thresholds): {orphans:?}"
    );
}

#[test]
fn parity_vectors_hold() {
    let scenarios: Vec<(&str, pg::BaseInputs, pg::ExtObservation)> = vec![
        ("all_clear_hmr_granted", base_inputs(), all_clear()),
        ("hmr_unknown", base_inputs(), {
            let mut o = all_clear();
            o.hmr_applied_verified = pg::PERMISSION_UNKNOWN;
            o
        }),
        ("hmr_denied", base_inputs(), {
            let mut o = all_clear();
            o.hmr_applied_verified = pg::PERMISSION_DENIED;
            o
        }),
        (
            "force_spike_floor",
            {
                let mut i = base_inputs();
                i.force_norm = 160.0; // rate 300 /ps
                i
            },
            all_clear(),
        ),
        ("accel_extreme_holds_at_fine", base_inputs(), {
            let mut o = all_clear();
            o.max_accel_a_per_ps2 = 1.0e6;
            o
        }),
        ("thermostat_gamma_dt_floor", base_inputs(), {
            let mut o = all_clear();
            o.effective_gamma_per_ps = 1.0e6;
            o
        }),
        ("neighbor_list_saturated", base_inputs(), {
            let mut o = all_clear();
            o.max_neighbor_count = 1024;
            o
        }),
        ("sentinel_overdue", base_inputs(), {
            let mut o = all_clear();
            o.steps_since_sentinel = 20_000;
            o
        }),
        ("uv_burst_phase", base_inputs(), {
            let mut o = all_clear();
            o.phase_id = 5;
            o
        }),
    ];

    assert_eq!(scenarios.len(), pg::PARITY_VECTORS.len());
    for (v, (name, inputs, obs)) in pg::PARITY_VECTORS.iter().zip(scenarios.iter()) {
        assert_eq!(
            v.name, *name,
            "parity vector order must match the scenario table"
        );
        let d = pg::compose(inputs, &armed_policy(), obs, &permissive_base());
        assert_eq!(
            d.ceiling_gear, v.expected_ceiling,
            "{name}: ceiling {} != expected {}",
            d.ceiling_gear, v.expected_ceiling
        );
        assert_eq!(
            d.ext_binding_gate, v.expected_ext_gate,
            "{name}: binding gate {} != expected {}",
            d.ext_binding_gate, v.expected_ext_gate
        );
    }
}

#[test]
fn wire_layouts_match_the_header() {
    // The header static_asserts these exact sizes; the Rust side asserts them at
    // compile time too. This test makes the contract visible in the run log.
    assert_eq!(std::mem::size_of::<pg::ExtPolicy>(), 208);
    assert_eq!(std::mem::align_of::<pg::ExtPolicy>(), 16);
    assert_eq!(std::mem::size_of::<pg::ExtObservation>(), 112);
    assert_eq!(std::mem::align_of::<pg::ExtObservation>(), 16);
    assert_eq!(std::mem::size_of::<pg::PhysGovDecision>(), 128);
    assert_eq!(std::mem::align_of::<pg::PhysGovDecision>(), 16);
    assert_eq!(std::mem::size_of::<pg::BaseDecision>(), 32);
}

#[test]
fn extension_gate_ids_are_disjoint_from_base_gate_ids() {
    for gate in [
        pg::GATE_FORCE_SPIKE,
        pg::GATE_ACCELERATION_EXTREME,
        pg::GATE_THERMOSTAT_STABILITY,
        pg::GATE_INTEGRATION_HEALTH,
        pg::GATE_TSH_MEMORY,
        pg::GATE_PHASE_REQUIREMENT,
        pg::GATE_HMR_HARD,
    ] {
        assert!(
            gate > pg::BASE_GATE_GOVERNOR_UNARMED,
            "gate {gate} collides with the base range"
        );
        assert_ne!(
            pg::gate_rank(gate),
            pg::RANK_NONE,
            "gate {gate} has no rank"
        );
    }
    assert_eq!(pg::gate_rank(pg::GATE_NONE), pg::RANK_NONE);
    assert_eq!(pg::base_gate_rank(9999), pg::RANK_NONE);
}
