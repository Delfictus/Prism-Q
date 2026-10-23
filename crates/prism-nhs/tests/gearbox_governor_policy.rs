//! WP C — Physical Stability Governor acceptance tests.
//!
//! These exercise the PRODUCTION decision function. `governor_decide` calls
//! `prism_gearbox_governor_decide_host`, which is a thin wrapper over the same
//! `prism_gearbox_governor_decide` that `prism_gearbox_sfa_kernel` invokes on
//! the device — there is exactly one implementation of the decision, so a
//! passing test here is a statement about the kernel, not about a host-side
//! re-implementation.
//!
//! No CUDA context is required: `prism_gearbox_governor_default_policy` and
//! `prism_gearbox_governor_decide_host` are pure host computation.
//!
//! Lives in `tests/` rather than in a `#[cfg(test)]` module for the same
//! reason `gearbox_b1_dt_flip.rs` does: pre-existing E0063 breakage in
//! unrelated lib tests (solvate.rs / rt_targets.rs / input.rs constructing
//! `PrismPrepTopology` without `dimer_dyad`) prevents the lib test crate from
//! compiling, while an integration target compiles independently.

#![cfg(feature = "gpu")]

use prism_nhs::gearbox::{
    governor_decide, governor_default_policy, GearGovernorPolicy, GearGuardObservation,
    TshSalienceRequest, GEAR_CAMPAIGN, GEAR_CONSTRAINT_UNMEASURED, GEAR_EMERG_DRIFT_ABORT,
    GEAR_EMERG_NONFINITE_ENERGY, GEAR_FINE, GEAR_GATE_CHUNK_LATENCY, GEAR_GATE_DWELL,
    GEAR_GATE_ENERGY_DRIFT, GEAR_GATE_GOVERNOR_UNARMED, GEAR_GATE_LOCKOUT,
    GEAR_GATE_PHYSICAL_EMERGENCY, GEAR_GATE_TOPOLOGY_PERMISSION, GEAR_GATE_TSH_SALIENCE,
    GEAR_PERMISSION_DENIED, GEAR_PERMISSION_GRANTED, GEAR_PERMISSION_UNKNOWN, GEAR_SAFE,
};

/// A fully-armed policy with every physical permission GRANTED and every
/// guard measured-and-clean. Individual tests degrade exactly one input so a
/// failure names one cause.
fn armed_permissive_policy() -> GearGovernorPolicy {
    let mut p = governor_default_policy().expect("C-side default policy fill");
    p.armed = 1;
    p.hmr_permission = GEAR_PERMISSION_GRANTED;
    p.constraint_permission = GEAR_PERMISSION_GRANTED;
    p.chunk_span_steps = 8; // production --captured-graph-body-unroll
    p
}

/// An observation with every guard MEASURED and comfortably in bounds, and
/// with enough dwell/lockout credit that the asymmetry gates do not bind.
fn clean_observation(current_gear: u32, requested_gear: u32) -> GearGuardObservation {
    GearGuardObservation {
        adjudication_code: 0,
        momentum_violation: 0,
        nonfinite_energy: 0,
        drift_measured: 1,
        drift: 0.0,
        force_measured: 1,
        force_norm: 1.0,
        constraint_measured: 1,
        constraint_failures: 0,
        current_gear,
        requested_gear,
        stable_dwell_steps: 100_000,
        steps_since_instability: 100_000,
        frame: 10_000,
        _reserved: [0; 2],
    }
}

// ── Defaults are exactly the documented literals ───────────────────────────

#[test]
fn c_side_defaults_are_the_documented_literals() {
    let p = governor_default_policy().expect("default policy fill");
    assert_eq!(p.armed, 0, "a fresh policy must be UNARMED");
    assert_eq!(p.hmr_permission, GEAR_PERMISSION_UNKNOWN);
    assert_eq!(p.constraint_permission, GEAR_PERMISSION_UNKNOWN);
    assert_eq!(p.unknown_topology_cap_gear, GEAR_SAFE);
    // Preserved source literals: the cruise threshold and the M1.2.18.5 fuse.
    assert_eq!(p.min_upshift_dwell_steps, 500);
    assert_eq!(p.instability_lockout_steps, 500);
    assert!(
        (p.drift_abort - 0.01).abs() < 1e-9,
        "drift_abort literal moved"
    );
    assert!(
        (p.force_clamp_reference - 1000.0).abs() < 1e-6,
        "force clamp anchor must track MAX_INTEGRATOR_FORCE_KCAL_MOL_A"
    );
    assert_eq!(p.require_force_guard_for_upshift, 1);
    assert_eq!(p.require_constraint_guard_for_upshift, 1);
    // Graduated bands must be strictly ordered or the graduated path collapses.
    assert!(p.drift_upshift_fraction < p.drift_hold_fraction);
    assert!(p.drift_hold_fraction < p.drift_downshift_fraction);
    assert!(p.drift_downshift_fraction <= 1.0);
    assert!(p.force_upshift_fraction < p.force_hold_fraction);
    assert!(p.force_hold_fraction < p.force_downshift_fraction);
    assert!(p.force_downshift_fraction <= p.force_abort_fraction);
}

// ── The core defect: 4 fs must be unreachable without verified HMR ─────────

#[test]
fn unarmed_governor_cannot_reach_the_campaign_gear() {
    // This is the exact scenario the old `counter >= 500` predicate allowed:
    // a long quiet KL run requesting 4 fs on a topology nobody verified.
    let p = governor_default_policy().expect("default policy fill");
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    let d = governor_decide(&p, &obs, None).expect("decide");
    assert_eq!(
        d.authorized_gear, GEAR_SAFE,
        "an unarmed governor must never authorize 4 fs"
    );
    assert_eq!(d.binding_gate, GEAR_GATE_GOVERNOR_UNARMED);
}

#[test]
fn unknown_hmr_cannot_reach_the_campaign_gear() {
    let mut p = armed_permissive_policy();
    p.hmr_permission = GEAR_PERMISSION_UNKNOWN;
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    let d = governor_decide(&p, &obs, None).expect("decide");
    assert_eq!(d.authorized_gear, GEAR_SAFE);
    assert_eq!(d.binding_gate, GEAR_GATE_TOPOLOGY_PERMISSION);
}

#[test]
fn denied_hmr_cannot_reach_the_campaign_gear() {
    let mut p = armed_permissive_policy();
    p.hmr_permission = GEAR_PERMISSION_DENIED;
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    let d = governor_decide(&p, &obs, None).expect("decide");
    assert_eq!(d.authorized_gear, GEAR_SAFE);
    assert_eq!(d.binding_gate, GEAR_GATE_TOPOLOGY_PERMISSION);
}

#[test]
fn denied_hydrogen_constraints_floor_at_the_finest_gear() {
    let mut p = armed_permissive_policy();
    p.constraint_permission = GEAR_PERMISSION_DENIED;
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    let d = governor_decide(&p, &obs, None).expect("decide");
    assert_eq!(
        d.authorized_gear, GEAR_FINE,
        "2 fs is not integrable without X-H constraints"
    );
}

#[test]
fn granted_permissions_and_clean_guards_do_authorize_the_campaign_gear() {
    // The governor must not be merely restrictive: with every physical
    // precondition genuinely satisfied the upshift is granted.
    let p = armed_permissive_policy();
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    let d = governor_decide(&p, &obs, None).expect("decide");
    assert_eq!(d.authorized_gear, GEAR_CAMPAIGN);
}

// ── Unmeasured guards withhold upshift without deadlocking ────────────────

#[test]
fn unmeasured_force_guard_withholds_upshift_but_keeps_running() {
    let p = armed_permissive_policy();
    let mut obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    obs.force_measured = 0;
    obs.force_norm = f32::NAN;
    let d = governor_decide(&p, &obs, None).expect("decide");
    assert_eq!(
        d.authorized_gear, GEAR_SAFE,
        "no upshift on unmeasured force"
    );
    assert_ne!(
        d.authorized_gear, GEAR_ABORT_SENTINEL,
        "an unmeasured guard must not abort the run"
    );
}

const GEAR_ABORT_SENTINEL: u32 = 3;

#[test]
fn unmeasured_drift_withholds_upshift() {
    let p = armed_permissive_policy();
    let mut obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    obs.drift_measured = 0;
    obs.drift = f32::NAN;
    let d = governor_decide(&p, &obs, None).expect("decide");
    assert_eq!(d.authorized_gear, GEAR_SAFE);
    assert_eq!(d.binding_gate, GEAR_GATE_ENERGY_DRIFT);
}

#[test]
fn unmeasured_constraint_counter_withholds_upshift() {
    let p = armed_permissive_policy();
    let mut obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    obs.constraint_measured = 0;
    obs.constraint_failures = GEAR_CONSTRAINT_UNMEASURED;
    let d = governor_decide(&p, &obs, None).expect("decide");
    assert_eq!(d.authorized_gear, GEAR_SAFE);
}

// ── The graduated Hamiltonian fuse ────────────────────────────────────────

#[test]
fn drift_bands_are_graduated_before_they_abort() {
    let p = armed_permissive_policy();
    let abort = p.drift_abort;

    // Band 1: quiet — upshift permitted.
    let mut obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    obs.drift = abort * 0.05;
    assert_eq!(
        governor_decide(&p, &obs, None).unwrap().authorized_gear,
        GEAR_CAMPAIGN
    );

    // Band 2: mild — HOLD, no upshift, no downshift.
    obs.drift = abort * 0.15;
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(d.authorized_gear, GEAR_SAFE, "hold band must not upshift");
    assert_eq!(d.binding_gate, GEAR_GATE_ENERGY_DRIFT);

    // Band 3: degrading — DOWNSHIFT one gear, still far below the abort bound.
    let mut obs2 = clean_observation(GEAR_CAMPAIGN, GEAR_CAMPAIGN);
    obs2.drift = abort * 0.35;
    let d = governor_decide(&p, &obs2, None).unwrap();
    assert_eq!(
        d.authorized_gear, GEAR_SAFE,
        "degrading drift must downshift BEFORE it aborts"
    );

    // Band 4: severe — floor to the finest gear, still not an abort.
    obs2.drift = abort * 0.75;
    let d = governor_decide(&p, &obs2, None).unwrap();
    assert_eq!(d.authorized_gear, GEAR_FINE);
    assert_eq!(d.emergency_flags, 0, "band 4 is a downshift, not an abort");

    // Band 5: past the preserved 1% literal — abort, exactly as before WP C.
    obs2.drift = abort * 1.5;
    let d = governor_decide(&p, &obs2, None).unwrap();
    assert_eq!(d.authorized_gear, GEAR_ABORT_SENTINEL);
    assert_eq!(d.binding_gate, GEAR_GATE_PHYSICAL_EMERGENCY);
    assert_ne!(d.emergency_flags & GEAR_EMERG_DRIFT_ABORT, 0);
}

#[test]
fn nonfinite_energy_still_aborts() {
    // The pre-WP-C fuse tripped on NaN V_t. That must be preserved verbatim.
    let p = armed_permissive_policy();
    let mut obs = clean_observation(GEAR_CAMPAIGN, GEAR_CAMPAIGN);
    obs.nonfinite_energy = 1;
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(d.authorized_gear, GEAR_ABORT_SENTINEL);
    assert_ne!(d.emergency_flags & GEAR_EMERG_NONFINITE_ENERGY, 0);
}

#[test]
fn an_unarmed_governor_still_fires_the_emergency_fuse() {
    // Unarmed must be SAFER than the old inline fuse, never weaker.
    let p = governor_default_policy().expect("default policy fill");
    let mut obs = clean_observation(GEAR_SAFE, GEAR_SAFE);
    obs.drift = p.drift_abort * 2.0;
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(d.authorized_gear, GEAR_ABORT_SENTINEL);
}

// ── Asymmetry: downshift immediate, upshift earned ────────────────────────

#[test]
fn downshift_is_immediate_and_ignores_dwell_and_lockout() {
    let p = armed_permissive_policy();
    let mut obs = clean_observation(GEAR_CAMPAIGN, GEAR_CAMPAIGN);
    obs.stable_dwell_steps = 0;
    obs.steps_since_instability = 0;
    obs.force_norm = p.force_clamp_reference * 1.5; // clipping => floor
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(
        d.authorized_gear, GEAR_FINE,
        "a downshift must never be dwell-gated or rate-limited"
    );
}

#[test]
fn upshift_requires_minimum_dwell() {
    let p = armed_permissive_policy();
    let mut obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    obs.stable_dwell_steps = p.min_upshift_dwell_steps - 1;
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(d.authorized_gear, GEAR_SAFE);
    assert_eq!(d.binding_gate, GEAR_GATE_DWELL);
}

#[test]
fn upshift_requires_instability_lockout_to_elapse() {
    let p = armed_permissive_policy();
    let mut obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    obs.steps_since_instability = p.instability_lockout_steps - 1;
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(d.authorized_gear, GEAR_SAFE);
    assert_eq!(d.binding_gate, GEAR_GATE_LOCKOUT);
}

#[test]
fn upshift_crosses_at_most_max_upshift_step_gears() {
    let p = armed_permissive_policy();
    assert_eq!(p.max_upshift_step, 1);
    let obs = clean_observation(GEAR_FINE, GEAR_CAMPAIGN);
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(
        d.authorized_gear, GEAR_SAFE,
        "0 -> 2 must be staged through 1"
    );
}

#[test]
fn a_gear_change_restarts_the_physical_dwell() {
    // Evidence that gear A was stable is not evidence that gear B is.
    let p = armed_permissive_policy();
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(d.authorized_gear, GEAR_CAMPAIGN);
    assert_eq!(d.next_stable_dwell_steps, 0);
}

#[test]
fn recovery_from_abort_restarts_at_the_finest_gear() {
    let p = armed_permissive_policy();
    // current_gear == 3 means the previous decision aborted.
    let obs = clean_observation(GEAR_ABORT_SENTINEL, GEAR_CAMPAIGN);
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(
        d.authorized_gear, GEAR_SAFE,
        "recovery must stage up from the finest gear, not resume where it blew up"
    );
}

// ── Chunk-boundary reality ────────────────────────────────────────────────

#[test]
fn chunk_latency_gate_withdraws_gears_whose_uncorrected_window_is_too_long() {
    let mut p = armed_permissive_policy();
    // 8 x 4 fs = 32 fs <= 50 fs default bound -> campaign gear survives.
    p.chunk_span_steps = 8;
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    assert_eq!(
        governor_decide(&p, &obs, None).unwrap().authorized_gear,
        GEAR_CAMPAIGN
    );

    // 16 x 4 fs = 64 fs > 50 fs -> the campaign gear is withdrawn; 16 x 2 = 32
    // fs still fits, so the run continues at the safe gear.
    p.chunk_span_steps = 16;
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(d.authorized_gear, GEAR_SAFE);
    assert_eq!(d.binding_gate, GEAR_GATE_CHUNK_LATENCY);
}

#[test]
fn downshift_latency_is_emitted_not_assumed() {
    let mut p = armed_permissive_policy();
    p.chunk_span_steps = 8;
    let obs = clean_observation(GEAR_SAFE, GEAR_SAFE);
    let d = governor_decide(&p, &obs, None).unwrap();
    assert_eq!(
        d.downshift_latency_steps, 8,
        "the bounded downshift latency must be reported on every decision"
    );
}

// ── TSH salience is a ceiling, never a lift ───────────────────────────────

#[test]
fn salience_can_only_lower_the_authorized_gear() {
    let p = armed_permissive_policy();
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    let req = TshSalienceRequest::request(GEAR_FINE, 1.0, obs.frame).expect("valid request");
    let d = governor_decide(&p, &obs, Some(&req)).unwrap();
    assert_eq!(d.authorized_gear, GEAR_FINE);
    assert_eq!(d.binding_gate, GEAR_GATE_TSH_SALIENCE);
}

#[test]
fn maximum_salience_cannot_outvote_a_physical_prohibition() {
    // The whole reason the gates are lexicographic and not a weighted score.
    let mut p = armed_permissive_policy();
    p.hmr_permission = GEAR_PERMISSION_DENIED;
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    let req = TshSalienceRequest::request(GEAR_CAMPAIGN, 1.0, obs.frame).expect("valid request");
    let d = governor_decide(&p, &obs, Some(&req)).unwrap();
    assert_eq!(
        d.authorized_gear, GEAR_SAFE,
        "salience must never obtain a gear the physical gates refused"
    );
    assert_eq!(d.binding_gate, GEAR_GATE_TOPOLOGY_PERMISSION);
}

#[test]
fn a_stale_salience_request_is_ignored_not_obeyed_and_not_waited_on() {
    let p = armed_permissive_policy();
    let obs = clean_observation(GEAR_SAFE, GEAR_CAMPAIGN);
    // Stamped far enough in the past to exceed salience_max_age_frames.
    let stale_frame = obs.frame - (p.salience_max_age_frames + 1);
    let req = TshSalienceRequest::request(GEAR_FINE, 1.0, stale_frame).expect("valid request");
    let d = governor_decide(&p, &obs, Some(&req)).unwrap();
    assert_eq!(
        d.authorized_gear, GEAR_CAMPAIGN,
        "a stale request must be dropped, not obeyed and not blocked on"
    );
}

#[test]
fn salience_request_refuses_to_encode_an_abort() {
    assert!(
        TshSalienceRequest::request(GEAR_ABORT_SENTINEL, 1.0, 0).is_none(),
        "the salience layer must not be able to request an abort"
    );
}

#[test]
fn an_unset_salience_request_carries_nan_not_zero() {
    let unset = TshSalienceRequest::unset();
    assert!(
        unset.salience.is_nan(),
        "an unset demand must not read as 'demanded nothing'"
    );
}
