//! Package 3 acceptance — composition against the REAL base governor.
//!
//! Everything here drives `prism_gearbox_governor_decide` through
//! `crate::gearbox::governor_decide` — the same C function the SFA kernel calls
//! — and then composes the physical-governor extension guards on top. That
//! makes these tests a check of the production decision path rather than of a
//! host reimplementation of it.
//!
//! The base decision function is pure host computation, so no CUDA context is
//! created and no GPU is required; only the `gpu` feature (which links the
//! compiled `gearbox` archive) is.
//!
//! Run with:
//! ```text
//! cargo test --release -p prism-nhs --test physical_governor_base_composition \
//!   --no-default-features --features gpu,ensemble,v2_ignition --locked --offline
//! ```

#![cfg(feature = "gpu")]

use prism_nhs::gearbox as gb;
use prism_nhs::physical_governor as pg;

/// A base policy that is ARMED with both topology permissions granted — the
/// most permissive base configuration, so anything the composed governor
/// refuses below is refused on physical grounds and not on configuration.
fn armed_base_policy() -> gb::GearGovernorPolicy {
    let mut p = gb::governor_default_policy().expect("C defaults must fill");
    p.armed = 1;
    p.hmr_permission = gb::GEAR_PERMISSION_GRANTED;
    p.constraint_permission = gb::GEAR_PERMISSION_GRANTED;
    p.chunk_span_steps = 8;
    p
}

/// A calm, fully measured base observation at the 2.0 fs gear whose science is
/// asking for the 4.0 fs campaign gear, with dwell and lockout satisfied.
fn calm_base_observation(dwell: u32, lockout: u32) -> gb::GearGuardObservation {
    let mut o = gb::GearGuardObservation::unmeasured(gb::GEAR_SAFE, gb::GEAR_CAMPAIGN, 10_000);
    o.drift_measured = 1;
    o.drift = 0.0;
    o.force_measured = 1;
    o.force_norm = 100.0;
    o.constraint_measured = 1;
    o.constraint_failures = 0;
    o.stable_dwell_steps = dwell;
    o.steps_since_instability = lockout;
    o
}

fn all_clear_ext() -> pg::ExtObservation {
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

fn armed_ext_policy() -> pg::ExtPolicy {
    let mut p = pg::ExtPolicy::default();
    p.armed = 1;
    p
}

// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn a_fully_clean_state_upshifts_through_the_real_base_governor() {
    let d = pg::decide_with_base_governor(
        &armed_base_policy(),
        &calm_base_observation(500, 500),
        None,
        &armed_ext_policy(),
        &all_clear_ext(),
    )
    .expect("base governor must accept these inputs");

    assert_eq!(
        d.authorized_gear,
        pg::GEAR_CAMPAIGN,
        "clean state must reach 4.0 fs; base_authorized={} ceiling={} ext_gate={}",
        d.base_authorized_gear,
        d.ceiling_gear,
        d.ext_binding_gate
    );
    assert_eq!(d.transition, pg::TRANSITION_UPSHIFT);
    assert_eq!(
        d.downshift_latency_steps, 8,
        "chunk_span_steps must be carried through"
    );
    assert!(
        (d.downshift_latency_fs - 16.0).abs() < 1e-4,
        "8 steps at the 2.0 fs gear is a 16 fs uncorrected window, got {}",
        d.downshift_latency_fs
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 8 — DWELL ENFORCEMENT (through the production gates)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn dwell_is_enforced_before_any_upshift() {
    for dwell in [0_u32, 1, 100, 499] {
        let d = pg::decide_with_base_governor(
            &armed_base_policy(),
            &calm_base_observation(dwell, 500),
            None,
            &armed_ext_policy(),
            &all_clear_ext(),
        )
        .expect("base governor must accept these inputs");
        assert_eq!(
            d.authorized_gear,
            gb::GEAR_SAFE,
            "dwell={dwell} is below the 500-step minimum; upshift must be refused"
        );
        assert_eq!(d.base_binding_gate, gb::GEAR_GATE_DWELL);
        assert_eq!(d.transition, pg::TRANSITION_HOLD);
    }

    let d = pg::decide_with_base_governor(
        &armed_base_policy(),
        &calm_base_observation(500, 500),
        None,
        &armed_ext_policy(),
        &all_clear_ext(),
    )
    .unwrap();
    assert_eq!(
        d.authorized_gear,
        gb::GEAR_CAMPAIGN,
        "dwell satisfied => upshift"
    );
}

#[test]
fn instability_lockout_is_enforced_before_any_upshift() {
    let d = pg::decide_with_base_governor(
        &armed_base_policy(),
        &calm_base_observation(500, 10),
        None,
        &armed_ext_policy(),
        &all_clear_ext(),
    )
    .expect("base governor must accept these inputs");
    assert_eq!(d.authorized_gear, gb::GEAR_SAFE);
    assert_eq!(d.base_binding_gate, gb::GEAR_GATE_LOCKOUT);
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 2 — SALIENCE CANNOT OVERRIDE A PHYSICAL PROHIBITION
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn enormous_salience_cannot_purchase_a_physically_refused_gear() {
    // Every salience magnitude the wire format can carry, including values far
    // outside the documented [0,1] range. Under a weighted score one of these
    // would eventually outvote the physical gate; under lexicographic ordering
    // none of them can, because salience is a CEILING operand of a `min` and
    // never a term in a sum.
    let saliences = [0.0_f32, 0.5, 1.0, 1.0e6, f32::MAX, f32::INFINITY];

    for salience in saliences {
        let mut request = gb::TshSalienceRequest::request(gb::GEAR_CAMPAIGN, salience, 10_000)
            .expect("gear 2 is a legal salience request");
        request.priority = u32::MAX;

        // Physical prohibition: the topology has no verified HMR.
        let mut ext = all_clear_ext();
        ext.hmr_applied_verified = pg::PERMISSION_DENIED;

        let d = pg::decide_with_base_governor(
            &armed_base_policy(),
            &calm_base_observation(500, 500),
            Some(&request),
            &armed_ext_policy(),
            &ext,
        )
        .expect("base governor must accept these inputs");

        assert!(
            d.authorized_gear <= gb::GEAR_SAFE,
            "salience={salience} obtained gear {}; a 4 fs timestep without \
             hydrogen-mass repartitioning must be unreachable at ANY salience",
            d.authorized_gear
        );
        assert_eq!(d.ext_binding_gate, pg::GATE_HMR_HARD);
        assert_eq!(d.binding_gate_rank, pg::RANK_HMR_HARD);
    }
}

#[test]
fn salience_may_only_make_the_gear_finer() {
    // A salience request for the FINE gear must be honoured (finer resolution
    // may be triggered by demand alone), while a request for a coarser gear
    // than the guards permit must not be.
    let request = gb::TshSalienceRequest::request(gb::GEAR_FINE, 1.0, 10_000)
        .expect("gear 0 is a legal salience request");

    let d = pg::decide_with_base_governor(
        &armed_base_policy(),
        &calm_base_observation(500, 500),
        Some(&request),
        &armed_ext_policy(),
        &all_clear_ext(),
    )
    .expect("base governor must accept these inputs");
    assert_eq!(
        d.authorized_gear,
        gb::GEAR_FINE,
        "finer resolution may be triggered by salience alone"
    );
    assert_eq!(d.transition, pg::TRANSITION_DOWNSHIFT);
}

// ═══════════════════════════════════════════════════════════════════════════
// Operator requirement 5 — GRADUATED DRIFT RESPONSE, not abort-only
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hamiltonian_drift_downshifts_before_it_aborts() {
    // drift_abort = 0.01 (the unchanged M1.2.18.5 fuse literal). Bands are
    // fractions of it: 0.10 / 0.25 / 0.50 / 1.00.
    let cases = [
        (0.0000_f32, gb::GEAR_CAMPAIGN), // OK, upshift permitted
        (0.0005, gb::GEAR_CAMPAIGN),     // <= 0.10*abort, still OK
        (0.0020, gb::GEAR_SAFE),         // HOLD band
        (0.0040, gb::GEAR_FINE),         // DOWNSHIFT band (from gear 1 -> 0)
        (0.0080, gb::GEAR_FINE),         // FLOOR band
        (0.0200, gb::GEAR_ABORT),        // past the abort bound
    ];
    for (drift, expect) in cases {
        let mut o = calm_base_observation(500, 500);
        o.drift = drift;
        let d = pg::decide_with_base_governor(
            &armed_base_policy(),
            &o,
            None,
            &armed_ext_policy(),
            &all_clear_ext(),
        )
        .expect("base governor must accept these inputs");
        assert_eq!(
            d.authorized_gear, expect,
            "drift={drift}: expected gear {expect}, got {} (base_gate={})",
            d.authorized_gear, d.base_binding_gate
        );
    }
    // The old fuse could only abort. Prove the graduated path really is
    // reachable: there exists a drift that downshifts without aborting.
    let mut o = calm_base_observation(500, 500);
    o.drift = 0.0040;
    let d = pg::decide_with_base_governor(
        &armed_base_policy(),
        &o,
        None,
        &armed_ext_policy(),
        &all_clear_ext(),
    )
    .unwrap();
    assert_ne!(d.authorized_gear, gb::GEAR_ABORT);
    assert_eq!(d.transition, pg::TRANSITION_DOWNSHIFT);
}

// ═══════════════════════════════════════════════════════════════════════════
// Structural: the extension layer only ever tightens the production base
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn extension_never_loosens_the_production_base_decision() {
    let policy = armed_base_policy();
    let ext_policy = armed_ext_policy();

    for dwell in [0_u32, 500] {
        for lockout in [0_u32, 500] {
            for drift in [0.0_f32, 0.002, 0.006] {
                for current in [gb::GEAR_FINE, gb::GEAR_SAFE, gb::GEAR_CAMPAIGN] {
                    let mut o = calm_base_observation(dwell, lockout);
                    o.drift = drift;
                    o.current_gear = current;

                    let base = gb::governor_decide(&policy, &o, None).expect("base decide");

                    for degrade in degradations() {
                        let mut ext = all_clear_ext();
                        degrade(&mut ext);
                        let d = pg::decide_with_base_governor(&policy, &o, None, &ext_policy, &ext)
                            .expect("compose");
                        if d.authorized_gear == pg::GEAR_ABORT {
                            continue; // an abort is a tightening
                        }
                        assert!(
                            d.authorized_gear <= base.authorized_gear,
                            "extension loosened the base: {} > {} \
                             (dwell={dwell} lockout={lockout} drift={drift} current={current})",
                            d.authorized_gear,
                            base.authorized_gear
                        );
                    }
                }
            }
        }
    }
}

fn degradations() -> Vec<fn(&mut pg::ExtObservation)> {
    vec![
        |o| o.hmr_applied_verified = pg::PERMISSION_UNKNOWN,
        |o| o.hmr_applied_verified = pg::PERMISSION_DENIED,
        |o| o.max_accel_a_per_ps2 = 3000.0,
        |o| o.effective_gamma_per_ps = 20.0,
        |o| o.instantaneous_temp_k = 350.0,
        |o| o.max_neighbor_count = 1024,
        |o| o.clamped_force_fraction = 1.0e-5,
        |o| o.observation_confidence = 0.1,
        |o| o.ps_since_last_observation = 5.0,
        |o| o.steps_since_sentinel = 20_000,
        |o| o.tsh_instability_memory = 0.9,
        |o| o.phase_id = 5,
        |o| o.accel_measured = 0,
        |o| o.thermostat_measured = 0,
        |o| o.integration_measured = 0,
    ]
}

// ═══════════════════════════════════════════════════════════════════════════
// Provenance
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn every_decision_binds_both_policy_digests() {
    let mut policy = armed_base_policy();
    policy.policy_digest = 0x1122_3344_5566_7788;
    let resolution = pg::resolve_ext_policy(None, None).expect("ext policy");

    let d = pg::decide_with_base_governor(
        &policy,
        &calm_base_observation(500, 500),
        None,
        &resolution.policy,
        &all_clear_ext(),
    )
    .expect("compose");

    assert_eq!(
        d.policy_digest, 0x1122_3344_5566_7788,
        "base policy binding"
    );
    assert_eq!(
        d.ext_policy_digest, resolution.digest,
        "extension policy binding"
    );
    assert_eq!(d.ext_armed, 1);
    assert_eq!(d.decision_emitted, 1);
}
