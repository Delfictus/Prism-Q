//! PHYSICAL STABILITY GOVERNOR — host-side policy, types and evaluation.
//!
//! Package 3 of the coupled temporal / gearbox / tensor-family architecture
//! (`OPERATOR_CONTROL/CAPTURED_GRAPH_TWIN_RUNTIME_REPAIR_20260731/OPERATOR_CONTROL/
//! COUPLED_TEMPORAL_GEARBOX_TENSOR_ARCHITECTURE.md`, operator directive
//! 2026-08-02).
//!
//! # What this module is
//!
//! The governor is the **sole authority over what timestep is SAFE**. It never
//! asks for a finer or coarser gear on scientific grounds. It publishes a
//! **CEILING**. The TSH arm publishes a **REQUEST**. The gearbox applies
//! `min(request, ceiling)`.
//!
//! # Consumer contract
//!
//! ```text
//! normal decision:  gear = min(requested_gear, ceiling_gear)
//! abort  decision:  transition == TRANSITION_ABORT
//!                   && authorized_gear == GEAR_ABORT
//!                   && dt_applied_ps.is_nan()      -> TRAP, do not integrate
//! ```
//!
//! On an abort the CEILING is pinned to [`GEAR_FINE`], never to [`GEAR_ABORT`].
//! [`GEAR_ABORT`] is the largest value the field can hold, and `min` is the one
//! arithmetic that erases it: a consumer computing `min(requested=1, ceiling=3)`
//! would get 1 and keep integrating at 2.0 fs straight through a physical
//! emergency. The abort is carried by `authorized_gear`, `transition` and the
//! NaN `dt_applied_ps`; the ceiling stays a genuine ceiling, and a consumer that
//! only knows how to `min` is driven to the finest timestep rather than the
//! coarsest.
//!
//! ```text
//!   requested_gear   (TSH / adjudication)  — what the science asked for
//!   authorized_gear  (THIS MODULE)         — what physics permitted
//!   device_measured_gear                   — what the device actually ran
//! ```
//!
//! Those are three separate facts and this module never collapses them. In
//! particular it does **not** synthesise a `device_measured_gear`: that is read
//! back after the fact by [`crate::gearbox::snapshot_governor`] and joined by
//! gear epoch. Fabricating it here would reproduce exactly the `gear_id: 0`
//! defect the architecture document identifies, where a build-time literal was
//! indistinguishable from a measurement and silently asserted 4x finer
//! resolution than the run had.
//!
//! # Relationship to [`crate::gearbox`]
//!
//! `crates/prism-nhs/src/cuda/gearbox.cuh` already carries a governor core
//! (gates 1..10) with a pinned wire format. This module **does not restate any
//! of it**. [`compose`] takes the base decision as an input and applies the
//! extension guards on top, so there is exactly one implementation of the base
//! gates — the C function `prism_gearbox_governor_decide`, shared by the SFA
//! kernel, the base host FFI, and (through
//! [`decide_with_base_governor`]) this module.
//!
//! The device-side extension evaluator lives in
//! `crates/prism-nhs/src/cuda/physical_governor.cuh` and is compiled by
//! `crates/prism-nhs/src/cuda/physical_governor.cu`, which `build.rs` builds
//! into `libphysical_governor.a` on every `gpu`-feature build. This file is its
//! host mirror. **Four** mechanical anti-drift controls bind them:
//!
//! 1. every struct here is `#[repr(C)]` with the same `const` size/offset
//!    assertions the header `static_assert`s — and because the header is now
//!    inside a real translation unit, the C-side half of those assertions is
//!    evaluated by nvcc rather than merely written down;
//! 2. [`DEFAULT_MACRO_TABLE`] names the exact `PRISM_PHYSGOV_DEFAULT_*` macro
//!    behind every default, and
//!    `tests/physical_governor_gates.rs::defaults_match_cuda_header` **parses
//!    the header at test time** and asserts each literal matches;
//! 3. [`PARITY_VECTORS`] is a fixed case table of scenario -> (ceiling,
//!    binding gate). `tests/physical_governor_gates.rs::parity_vectors_hold`
//!    asserts the host side, and [`decide_on_compiled_evaluator`] replays the
//!    identical inputs through the compiled
//!    `prism_physical_governor_decide` — the exact function
//!    `prism_physical_governor_decide_kernel` calls on the device — so the
//!    transliterated half is checked against the compiled one, not asserted to
//!    match it;
//! 4. [`verify_compiled_evaluator_abi`] reads `sizeof` and `offsetof` back out
//!    of the archive the build just produced and compares them with this
//!    module's own pins. A hand-transcribed size in a comment is not evidence;
//!    a number returned by the compiler that built the code is.
//!
//! # The nine guards
//!
//! | # | guard | observable | class | owner |
//! |---|---|---|---|---|
//! | 1 | topology HMR permission | `PrismPrepTopology::hmr_applied` | HARD | ext gate 38 + base gate 2 |
//! | 2 | constraint stability | SHAKE/RATTLE non-convergence count | graduated | base gate 3 |
//! | 3 | Hamiltonian drift | relative energy drift | graduated | base gate 4 (+ base gate 1 abort) |
//! | 4 | force norm | `max\|F\|` vs the integrator clamp | graduated | base gate 5 |
//! | 5 | force spike | `d\|F\|/dt` per simulated ps | graduated | ext gate 32 |
//! | 6 | acceleration extremes | `0.5*a*dt^2` vs the X–H bond | per-gear | ext gate 33 |
//! | 7 | thermostat stability | `gamma*dt` + setpoint deviation | per-gear | ext gate 34 |
//! | 8 | minimum stable dwell / instability memory | dwell + lockout steps | upshift-only | base gates 8, 9 |
//! | 9 | neighbour-list / integration health | occupancy, clamped fraction, faults | graduated | ext gate 35 |
//!
//! plus TSH memory with anti-blindness (ext gate 36) and protocol-phase
//! requirement (ext gate 37), both ceiling-only.
//!
//! # Non-negotiable properties
//!
//! * **Lexicographic, never a weighted score.** There is no coefficient in this
//!   file that multiplies a salience against a physical quantity. See
//!   [`gate_rank`] and the proof in
//!   `tests/physical_governor_gates.rs::salience_cannot_override_physical_prohibition`.
//! * **Tighten-only.** [`compose`] ends in a single `min` against the base
//!   decision, so an extension defect can only refuse a gear that was
//!   available — never authorise one that was not.
//! * **HMR hard gate.** Gear 2 (4.0 fs) is unreachable unless
//!   [`ExtObservation::hmr_applied_verified`] is `PERMISSION_GRANTED`. That
//!   check is re-asserted unconditionally after all policy arithmetic and there
//!   is no policy field that disables it.
//! * **Asymmetry.** Downshift on any guard degradation at the next boundary,
//!   with no dwell requirement, and it may skip gears. Upshift requires every
//!   required guard measured and OK, plus dwell, plus lockout, plus the
//!   base's `max_upshift_step`.
//! * **No fabricated values.** Every guard is tri-valued; unmeasured is an
//!   explicit flag with a NaN / sentinel payload, never a plausible default.
//! * **No hardcoded thresholds.** Every threshold is policy-resolved; the
//!   source literal appears once, as a documented default, and the resolved
//!   value plus its origin is recorded in [`ExtPolicyResolution::origins`].
//!
//! # Chunk-boundary contract
//!
//! The governor runs once per captured-graph replay. With
//! `--captured-graph-body-unroll N > 1` the Director advances `N` protocol
//! steps per replay, so the gear authorized at replay `k` is in force for all
//! `N` steps of replay `k` and a degradation seen during replay `k` cannot be
//! corrected before replay `k+1`. "Immediate downshift" therefore means
//! "immediate at the next chunk boundary", bounded by two **named, emitted**
//! quantities:
//!
//! ```text
//! downshift_latency_steps = chunk_span_steps
//! downshift_latency_fs    = chunk_span_steps * gear_dt_fs[current_gear]
//! ```
//!
//! Both are written into every [`PhysGovDecision`]. The governor selects a gear
//! that is safe for the **whole chunk on entry**; mid-chunk correction is not
//! available and a within-chunk emergency remains the abort trap's job
//! (gear 3).

use std::mem::{align_of, size_of};

// ═══════════════════════════════════════════════════════════════════════════
// Constants — mirrors of `physical_governor.cuh` / `gearbox.cuh`
// ═══════════════════════════════════════════════════════════════════════════

/// Mirrors `PRISM_PHYSGOV_SCHEMA_VERSION`.
pub const PHYSGOV_SCHEMA_VERSION: u32 = 1;

/// 0.5 fs — burst / high-resolution gear. Mirrors `PRISM_GEAR_FINE`.
pub const GEAR_FINE: u32 = 0;
/// 2.0 fs — constrained-hydrogen safety gear. Mirrors `PRISM_GEAR_SAFE`.
pub const GEAR_SAFE: u32 = 1;
/// 4.0 fs — HMR-ONLY campaign gear. Mirrors `PRISM_GEAR_CAMPAIGN`.
pub const GEAR_CAMPAIGN: u32 = 2;
/// Abort sentinel. Mirrors `PRISM_GEAR_ABORT`.
pub const GEAR_ABORT: u32 = 3;
/// Coarsest gear that is a real timestep. Mirrors `PRISM_GEAR_MAX_ACTIVE`.
pub const GEAR_MAX_ACTIVE: u32 = GEAR_CAMPAIGN;

/// Tri-state permission: not established. Mirrors `PRISM_GEAR_PERMISSION_UNKNOWN`.
pub const PERMISSION_UNKNOWN: u32 = 0;
/// Tri-state permission: established false. Mirrors `PRISM_GEAR_PERMISSION_DENIED`.
pub const PERMISSION_DENIED: u32 = 1;
/// Tri-state permission: established true. Mirrors `PRISM_GEAR_PERMISSION_GRANTED`.
pub const PERMISSION_GRANTED: u32 = 2;

/// Per-gate verdicts. Mirror `PRISM_GEAR_VERDICT_*`.
pub const VERDICT_UNMEASURED: u32 = 0;
/// Guard measured and inside its OK band.
pub const VERDICT_OK: u32 = 1;
/// No upshift; hold the current gear.
pub const VERDICT_HOLD: u32 = 2;
/// One gear finer.
pub const VERDICT_DOWNSHIFT: u32 = 3;
/// Hard drop to the finest gear.
pub const VERDICT_FLOOR: u32 = 4;
/// Gear 3 trap.
pub const VERDICT_EMERGENCY: u32 = 5;

/// Extension gate ids. Mirror `PRISM_PHYSGOV_GATE_*`; disjoint from the base
/// `PRISM_GEAR_GATE_*` range (0..11) so a binding-gate id is unambiguous.
pub const GATE_NONE: u32 = 0;
/// `d|F|/dt` per simulated ps.
pub const GATE_FORCE_SPIKE: u32 = 32;
/// `0.5*a*dt^2` vs the X–H bond length.
pub const GATE_ACCELERATION_EXTREME: u32 = 33;
/// Langevin `gamma*dt` and thermostat setpoint deviation.
pub const GATE_THERMOSTAT_STABILITY: u32 = 34;
/// Neighbour-list occupancy, clamped-force fraction, integration faults.
pub const GATE_INTEGRATION_HEALTH: u32 = 35;
/// TSH integrated instability memory plus the anti-blindness controls.
pub const GATE_TSH_MEMORY: u32 = 36;
/// Protocol-phase resolution requirement.
pub const GATE_PHASE_REQUIREMENT: u32 = 37;
/// The non-policy-overridable hydrogen-mass-repartitioning gate.
pub const GATE_HMR_HARD: u32 = 38;
/// The extension analogue of base gate 11 ([`BASE_GATE_GOVERNOR_UNARMED`]).
///
/// Reported when the extension layer could not BIND a governed policy or
/// observation — an unarmed [`ExtPolicy`], or a schema version this build does
/// not understand. It exists so that "nobody armed the extension" and "this
/// policy is from a different wire version" are never reported as physics
/// findings. See [`PhysGovDecision::ext_binding_refusal`] for the named cause.
pub const GATE_EXT_BINDING: u32 = 39;
/// First extension gate id.
pub const GATE_EXT_FIRST: u32 = GATE_FORCE_SPIKE;
/// Last extension gate id.
pub const GATE_EXT_LAST: u32 = GATE_EXT_BINDING;
/// Number of extension gate ids — also the number of 4-bit nibbles in
/// [`PhysGovDecision::ext_gate_verdicts`] (8 * 4 == 32 bits, exactly full).
pub const GATE_EXT_COUNT: usize = 8;
/// Number of extension gates that contribute an operand to the CEILING fold.
/// [`GATE_EXT_BINDING`] is not one of them: it is an early terminal branch that
/// replaces the fold rather than joining it.
pub const CEILING_GATE_COUNT: usize = 7;

/// [`PhysGovDecision::ext_binding_refusal`]: the extension layer bound a
/// governed policy and observation. Mirrors `PRISM_PHYSGOV_REFUSAL_NONE`.
pub const REFUSAL_NONE: u32 = 0;
/// No host ever armed the extension policy. Mirrors `PRISM_PHYSGOV_REFUSAL_UNARMED`.
pub const REFUSAL_UNARMED: u32 = 1;
/// The extension policy declares a wire version this build cannot parse.
/// Mirrors `PRISM_PHYSGOV_REFUSAL_POLICY_SCHEMA`.
pub const REFUSAL_POLICY_SCHEMA: u32 = 2;
/// The extension observation declares a wire version this build cannot parse.
/// Mirrors `PRISM_PHYSGOV_REFUSAL_OBSERVATION_SCHEMA`.
pub const REFUSAL_OBSERVATION_SCHEMA: u32 = 3;

/// Base gate ids consulted for attribution. Mirror `PRISM_GEAR_GATE_*`.
pub const BASE_GATE_NONE: u32 = 0;
/// Base gate 1.
pub const BASE_GATE_PHYSICAL_EMERGENCY: u32 = 1;
/// Base gate 2.
pub const BASE_GATE_TOPOLOGY_PERMISSION: u32 = 2;
/// Base gate 3.
pub const BASE_GATE_CONSTRAINT_STABILITY: u32 = 3;
/// Base gate 4.
pub const BASE_GATE_ENERGY_DRIFT: u32 = 4;
/// Base gate 5.
pub const BASE_GATE_FORCE_STABILITY: u32 = 5;
/// Base gate 6.
pub const BASE_GATE_CHUNK_LATENCY: u32 = 6;
/// Base gate 7.
pub const BASE_GATE_TSH_SALIENCE: u32 = 7;
/// Base gate 8.
pub const BASE_GATE_DWELL: u32 = 8;
/// Base gate 9.
pub const BASE_GATE_LOCKOUT: u32 = 9;
/// Base gate 10.
pub const BASE_GATE_UPSHIFT_RATE: u32 = 10;
/// Base gate 11.
pub const BASE_GATE_GOVERNOR_UNARMED: u32 = 11;

// ─── LEXICOGRAPHIC AUTHORITY RANKS — mirror `PRISM_PHYSGOV_RANK_*` ─────────
//
// Lower rank = more authoritative. Rank decides ATTRIBUTION when two gates
// land on the same ceiling; it is never a weight and never multiplies a value.

/// Rank of the terminal physical-emergency class.
pub const RANK_PHYSICAL_EMERGENCY: u32 = 10;
/// Rank of the HMR hard gate — above the policy topology gate by design.
pub const RANK_HMR_HARD: u32 = 15;
/// Rank of the policy topology-permission gate.
pub const RANK_TOPOLOGY_PERMISSION: u32 = 20;
/// Rank of the unarmed-governor attribution alias.
pub const RANK_GOVERNOR_UNARMED: u32 = 25;
/// Rank of the SHAKE/RATTLE constraint gate.
pub const RANK_CONSTRAINT_STABILITY: u32 = 30;
/// Rank of the graduated Hamiltonian-drift gate.
pub const RANK_ENERGY_DRIFT: u32 = 40;
/// Rank of the graduated force-norm gate.
pub const RANK_FORCE_STABILITY: u32 = 50;
/// Rank of the force-spike refinement.
pub const RANK_FORCE_SPIKE: u32 = 52;
/// Rank of the acceleration-extreme gate.
pub const RANK_ACCELERATION_EXTREME: u32 = 54;
/// Rank of the thermostat-stability gate.
pub const RANK_THERMOSTAT_STABILITY: u32 = 56;
/// Rank of the neighbour-list / integration-health gate.
pub const RANK_INTEGRATION_HEALTH: u32 = 58;
/// Rank of the chunk-latency scheduling bound.
pub const RANK_CHUNK_LATENCY: u32 = 60;
/// Rank of the TSH memory / anti-blindness gate.
pub const RANK_TSH_MEMORY: u32 = 70;
/// Rank of the TSH salience request.
pub const RANK_TSH_SALIENCE: u32 = 80;
/// Rank of the protocol-phase requirement.
pub const RANK_PHASE_REQUIREMENT: u32 = 90;
/// Rank of the upshift dwell requirement.
pub const RANK_DWELL: u32 = 100;
/// Rank of the instability lockout.
pub const RANK_LOCKOUT: u32 = 101;
/// Rank of the per-decision upshift-rate limit.
pub const RANK_UPSHIFT_RATE: u32 = 102;
/// Sorts last; an unknown gate can never claim authority it was not granted.
pub const RANK_NONE: u32 = 255;

/// Extension emergency cause bits. Mirror `PRISM_PHYSGOV_EMERG_*`; disjoint
/// from the base `PRISM_GEAR_EMERG_*` bits (0x1..0x10).
pub const EMERG_NONFINITE_ACCEL: u32 = 0x100;
/// Per-step displacement exceeds the abort fraction at the finest gear.
pub const EMERG_ACCEL_DISPLACEMENT: u32 = 0x200;
/// The integrator reported a hard fault.
pub const EMERG_INTEGRATION_FAULT: u32 = 0x400;
/// `|T_inst - T_target| / T_target >= temp_dev_abort_fraction`. At the default
/// 2.00 that is a threefold temperature runaway — not a thermostat that is
/// merely fighting, but a system whose kinetic energy is diverging. No timestep
/// repairs it. Mirrors `PRISM_PHYSGOV_EMERG_TEMPERATURE_RUNAWAY`.
pub const EMERG_TEMPERATURE_RUNAWAY: u32 = 0x800;

/// Transition classification. Mirrors `PRISM_PHYSGOV_TRANSITION_*`.
/// HOLD is an EMITTED FACT, not an absence — a gap in the record stream must
/// never be ambiguous between "gear held" and "the decision did not run".
pub const TRANSITION_HOLD: u32 = 0;
/// The authorized gear is finer than the current gear.
pub const TRANSITION_DOWNSHIFT: u32 = 1;
/// The authorized gear is coarser than the current gear.
pub const TRANSITION_UPSHIFT: u32 = 2;
/// Gear 3 trap.
pub const TRANSITION_ABORT: u32 = 3;

/// Explicit "no value" for a `u32` observable. Mirrors
/// `PRISM_PHYSGOV_U32_UNMEASURED`.
pub const U32_UNMEASURED: u32 = 0xFFFF_FFFF;

// ═══════════════════════════════════════════════════════════════════════════
// Documented defaults
//
// These are the ONLY default literals on the Rust side and each names the
// `PRISM_PHYSGOV_DEFAULT_*` macro it mirrors. `DEFAULT_MACRO_TABLE` binds the
// two, and the header-parsing test asserts they agree.
// ═══════════════════════════════════════════════════════════════════════════

/// `PRISM_PHYSGOV_DEFAULT_FORCE_RATE_HOLD_PER_PS` — relative growth of
/// `max|F|` per simulated ps above which no upshift is permitted. At gear 2 one
/// step is 0.004 ps, so 50/ps is a 20 %-per-step jump.
pub const DEFAULT_FORCE_RATE_HOLD_PER_PS: f32 = 50.0;
/// `PRISM_PHYSGOV_DEFAULT_FORCE_RATE_DOWNSHIFT_PER_PS` — 40 %/step at 4 fs.
pub const DEFAULT_FORCE_RATE_DOWNSHIFT_PER_PS: f32 = 100.0;
/// `PRISM_PHYSGOV_DEFAULT_FORCE_RATE_FLOOR_PER_PS` — `max|F|` at least
/// doubling within one 4 fs step.
pub const DEFAULT_FORCE_RATE_FLOOR_PER_PS: f32 = 250.0;
/// `PRISM_PHYSGOV_DEFAULT_FORCE_RATE_NORM_FLOOR` — denominator floor in
/// kcal/mol/A, three orders of magnitude below `MAX_INTEGRATOR_FORCE_KCAL_MOL_A`.
pub const DEFAULT_FORCE_RATE_NORM_FLOOR: f32 = 1.0;

/// `PRISM_PHYSGOV_DEFAULT_ACCEL_DISPLACEMENT_REFERENCE_A` — the X–H bond
/// length SHAKE must restore, in Angstrom.
pub const DEFAULT_ACCEL_DISPLACEMENT_REFERENCE_A: f32 = 1.0;
/// `PRISM_PHYSGOV_DEFAULT_ACCEL_OK_FRACTION` — 2 % of a bond per step.
pub const DEFAULT_ACCEL_OK_FRACTION: f32 = 0.02;
/// `PRISM_PHYSGOV_DEFAULT_ACCEL_HOLD_FRACTION` — 5 %.
pub const DEFAULT_ACCEL_HOLD_FRACTION: f32 = 0.05;
/// `PRISM_PHYSGOV_DEFAULT_ACCEL_DOWNSHIFT_FRACTION` — 10 %.
pub const DEFAULT_ACCEL_DOWNSHIFT_FRACTION: f32 = 0.10;
/// `PRISM_PHYSGOV_DEFAULT_ACCEL_ABORT_FRACTION` — half a bond length in one
/// step at the finest gear; SHAKE cannot recover from that.
pub const DEFAULT_ACCEL_ABORT_FRACTION: f32 = 0.50;

/// `PRISM_PHYSGOV_DEFAULT_GAMMA_DT_OK` — standard Langevin sampling practice.
pub const DEFAULT_GAMMA_DT_OK: f32 = 0.05;
/// `PRISM_PHYSGOV_DEFAULT_GAMMA_DT_HOLD`.
pub const DEFAULT_GAMMA_DT_HOLD: f32 = 0.10;
/// `PRISM_PHYSGOV_DEFAULT_GAMMA_DT_DOWNSHIFT`.
pub const DEFAULT_GAMMA_DT_DOWNSHIFT: f32 = 0.25;
/// `PRISM_PHYSGOV_DEFAULT_GAMMA_DT_ABORT` — the point at which the Langevin
/// update stops being inertial: `c1 = exp(-gamma*dt) <= 0.37`, so the velocity
/// is essentially fully resampled every step.
///
/// Despite the name this is NOT an abort — the thermostat guard never escalates
/// past FLOOR, because a legitimate high-friction equilibration must not kill
/// the run. It is not decorative either: crossing it raises the INSTABILITY
/// flag, which zeroes the dwell counter and restarts the lockout window, so it
/// governs an emitted state transition.
pub const DEFAULT_GAMMA_DT_ABORT: f32 = 1.00;
/// `PRISM_PHYSGOV_DEFAULT_TEMP_DEV_OK_FRACTION`.
pub const DEFAULT_TEMP_DEV_OK_FRACTION: f32 = 0.10;
/// `PRISM_PHYSGOV_DEFAULT_TEMP_DEV_HOLD_FRACTION`.
pub const DEFAULT_TEMP_DEV_HOLD_FRACTION: f32 = 0.25;
/// `PRISM_PHYSGOV_DEFAULT_TEMP_DEV_DOWNSHIFT_FRACTION`.
pub const DEFAULT_TEMP_DEV_DOWNSHIFT_FRACTION: f32 = 0.50;
/// `PRISM_PHYSGOV_DEFAULT_TEMP_DEV_ABORT_FRACTION`.
pub const DEFAULT_TEMP_DEV_ABORT_FRACTION: f32 = 2.00;

/// `PRISM_PHYSGOV_DEFAULT_NEIGHBOR_OCCUPANCY_OK` — fraction of
/// `NEIGHBOR_LIST_SIZE` below which the pair list is comfortably sized.
pub const DEFAULT_NEIGHBOR_OCCUPANCY_OK: f32 = 0.80;
/// `PRISM_PHYSGOV_DEFAULT_NEIGHBOR_OCCUPANCY_HOLD`.
pub const DEFAULT_NEIGHBOR_OCCUPANCY_HOLD: f32 = 0.90;
/// `PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_HOLD` — ANY force clipping is
/// non-Hamiltonian by construction, so the hold threshold is exactly zero.
pub const DEFAULT_CLAMPED_FRACTION_HOLD: f32 = 0.0;
/// `PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_DOWNSHIFT`.
pub const DEFAULT_CLAMPED_FRACTION_DOWNSHIFT: f32 = 0.0001;
/// `PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_FLOOR`.
pub const DEFAULT_CLAMPED_FRACTION_FLOOR: f32 = 0.001;

/// `PRISM_PHYSGOV_DEFAULT_TSH_MEMORY_HOLD`.
pub const DEFAULT_TSH_MEMORY_HOLD: f32 = 0.25;
/// `PRISM_PHYSGOV_DEFAULT_TSH_MEMORY_FLOOR`.
pub const DEFAULT_TSH_MEMORY_FLOOR: f32 = 0.75;
/// `PRISM_PHYSGOV_DEFAULT_MIN_OBSERVATION_CONFIDENCE` — anti-blindness control
/// (d): no coarsening while observation confidence is below this.
pub const DEFAULT_MIN_OBSERVATION_CONFIDENCE: f32 = 0.50;
/// `PRISM_PHYSGOV_DEFAULT_MAX_OBSERVATION_INTERVAL_PS` — anti-blindness control
/// (a): the minimum physical-time observation cadence.
pub const DEFAULT_MAX_OBSERVATION_INTERVAL_PS: f32 = 1.0;
/// `PRISM_PHYSGOV_DEFAULT_MAX_STEPS_WITHOUT_SENTINEL` — anti-blindness control
/// (b): the maximum interval without a forced high-resolution sample.
pub const DEFAULT_MAX_STEPS_WITHOUT_SENTINEL: u32 = 10000;

/// `PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_0` — cold_hold, unconstrained.
pub const DEFAULT_PHASE_CEILING_0: u32 = 2;
/// `PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_1` — ramp_up, 2.0 fs ceiling.
pub const DEFAULT_PHASE_CEILING_1: u32 = 1;
/// `PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_2` — warm_hold, unconstrained.
pub const DEFAULT_PHASE_CEILING_2: u32 = 2;
/// `PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_3` — ramp_down, 2.0 fs ceiling.
pub const DEFAULT_PHASE_CEILING_3: u32 = 1;
/// `PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_4` — cold_return, unconstrained.
pub const DEFAULT_PHASE_CEILING_4: u32 = 2;
/// `PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_5` — uv_burst, 0.5 fs.
pub const DEFAULT_PHASE_CEILING_5: u32 = 0;
/// `PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_6` — reserved.
pub const DEFAULT_PHASE_CEILING_6: u32 = 2;
/// `PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_7` — reserved.
pub const DEFAULT_PHASE_CEILING_7: u32 = 2;
/// `PRISM_PHYSGOV_DEFAULT_N_PHASE_SLOTS`.
pub const DEFAULT_N_PHASE_SLOTS: u32 = 8;

/// `PRISM_PHYSGOV_DEFAULT_REQUIRE_ACCEL_GUARD` — an unread sensor is not
/// verification, so an unmeasured acceleration guard blocks upshift by default.
pub const DEFAULT_REQUIRE_ACCEL_GUARD: u32 = 1;
/// `PRISM_PHYSGOV_DEFAULT_REQUIRE_THERMOSTAT_GUARD`.
pub const DEFAULT_REQUIRE_THERMOSTAT_GUARD: u32 = 1;
/// `PRISM_PHYSGOV_DEFAULT_REQUIRE_INTEGRATION_GUARD`.
pub const DEFAULT_REQUIRE_INTEGRATION_GUARD: u32 = 1;
/// `PRISM_PHYSGOV_DEFAULT_REQUIRE_OBSERVATION_CONFIDENCE`.
pub const DEFAULT_REQUIRE_OBSERVATION_CONFIDENCE: u32 = 1;

/// `PRISM_PHYSGOV_DEFAULT_UNARMED_CAP_GEAR` — the gear ceiling applied when the
/// extension layer cannot bind a governed policy or observation.
///
/// Deliberately NOT policy-overridable and deliberately not read from
/// [`ExtPolicy`]: the whole point of the refusal is that the policy could not be
/// trusted, so taking the cap from it would be circular. It is a compile-time
/// constant, mirrored in the header and pinned by
/// `physical_governor_gates.rs::defaults_match_cuda_header`.
///
/// FAIL CLOSED. A zeroed [`ExtPolicy`] has `force_rate_floor_per_ps == 0`,
/// `accel_ok_fraction == 0` and `clamped_fraction_floor == 0`; running the
/// guards against it manufactures a FLOOR on every gate and the emitted record
/// becomes indistinguishable from genuinely violent physics. The refusal branch
/// pins the same finest gear that zeroed policy would have produced, but
/// attributes it honestly — and never deadlocks, because gear 0 is always
/// reachable.
pub const DEFAULT_UNARMED_CAP_GEAR: u32 = 0;

/// A default literal as it appears in the CUDA header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DefaultLiteral {
    /// An unsigned literal, written `<n>u` in the header.
    U32(u32),
    /// A float literal, written `<x>f` in the header.
    F32(f32),
}

/// Every documented default, paired with the `PRISM_PHYSGOV_DEFAULT_*` macro
/// it mirrors.
///
/// `tests/physical_governor_gates.rs::defaults_match_cuda_header` reads
/// `src/cuda/physical_governor.cuh`, extracts each `#define`, and asserts the
/// literal equals the value here. That is the mechanical guarantee that the
/// host policy and the device evaluator cannot drift on a threshold.
pub const DEFAULT_MACRO_TABLE: &[(&str, DefaultLiteral)] = &[
    (
        "FORCE_RATE_HOLD_PER_PS",
        DefaultLiteral::F32(DEFAULT_FORCE_RATE_HOLD_PER_PS),
    ),
    (
        "FORCE_RATE_DOWNSHIFT_PER_PS",
        DefaultLiteral::F32(DEFAULT_FORCE_RATE_DOWNSHIFT_PER_PS),
    ),
    (
        "FORCE_RATE_FLOOR_PER_PS",
        DefaultLiteral::F32(DEFAULT_FORCE_RATE_FLOOR_PER_PS),
    ),
    (
        "FORCE_RATE_NORM_FLOOR",
        DefaultLiteral::F32(DEFAULT_FORCE_RATE_NORM_FLOOR),
    ),
    (
        "ACCEL_DISPLACEMENT_REFERENCE_A",
        DefaultLiteral::F32(DEFAULT_ACCEL_DISPLACEMENT_REFERENCE_A),
    ),
    (
        "ACCEL_OK_FRACTION",
        DefaultLiteral::F32(DEFAULT_ACCEL_OK_FRACTION),
    ),
    (
        "ACCEL_HOLD_FRACTION",
        DefaultLiteral::F32(DEFAULT_ACCEL_HOLD_FRACTION),
    ),
    (
        "ACCEL_DOWNSHIFT_FRACTION",
        DefaultLiteral::F32(DEFAULT_ACCEL_DOWNSHIFT_FRACTION),
    ),
    (
        "ACCEL_ABORT_FRACTION",
        DefaultLiteral::F32(DEFAULT_ACCEL_ABORT_FRACTION),
    ),
    ("GAMMA_DT_OK", DefaultLiteral::F32(DEFAULT_GAMMA_DT_OK)),
    ("GAMMA_DT_HOLD", DefaultLiteral::F32(DEFAULT_GAMMA_DT_HOLD)),
    (
        "GAMMA_DT_DOWNSHIFT",
        DefaultLiteral::F32(DEFAULT_GAMMA_DT_DOWNSHIFT),
    ),
    (
        "GAMMA_DT_ABORT",
        DefaultLiteral::F32(DEFAULT_GAMMA_DT_ABORT),
    ),
    (
        "TEMP_DEV_OK_FRACTION",
        DefaultLiteral::F32(DEFAULT_TEMP_DEV_OK_FRACTION),
    ),
    (
        "TEMP_DEV_HOLD_FRACTION",
        DefaultLiteral::F32(DEFAULT_TEMP_DEV_HOLD_FRACTION),
    ),
    (
        "TEMP_DEV_DOWNSHIFT_FRACTION",
        DefaultLiteral::F32(DEFAULT_TEMP_DEV_DOWNSHIFT_FRACTION),
    ),
    (
        "TEMP_DEV_ABORT_FRACTION",
        DefaultLiteral::F32(DEFAULT_TEMP_DEV_ABORT_FRACTION),
    ),
    (
        "NEIGHBOR_OCCUPANCY_OK",
        DefaultLiteral::F32(DEFAULT_NEIGHBOR_OCCUPANCY_OK),
    ),
    (
        "NEIGHBOR_OCCUPANCY_HOLD",
        DefaultLiteral::F32(DEFAULT_NEIGHBOR_OCCUPANCY_HOLD),
    ),
    (
        "CLAMPED_FRACTION_HOLD",
        DefaultLiteral::F32(DEFAULT_CLAMPED_FRACTION_HOLD),
    ),
    (
        "CLAMPED_FRACTION_DOWNSHIFT",
        DefaultLiteral::F32(DEFAULT_CLAMPED_FRACTION_DOWNSHIFT),
    ),
    (
        "CLAMPED_FRACTION_FLOOR",
        DefaultLiteral::F32(DEFAULT_CLAMPED_FRACTION_FLOOR),
    ),
    (
        "TSH_MEMORY_HOLD",
        DefaultLiteral::F32(DEFAULT_TSH_MEMORY_HOLD),
    ),
    (
        "TSH_MEMORY_FLOOR",
        DefaultLiteral::F32(DEFAULT_TSH_MEMORY_FLOOR),
    ),
    (
        "MIN_OBSERVATION_CONFIDENCE",
        DefaultLiteral::F32(DEFAULT_MIN_OBSERVATION_CONFIDENCE),
    ),
    (
        "MAX_OBSERVATION_INTERVAL_PS",
        DefaultLiteral::F32(DEFAULT_MAX_OBSERVATION_INTERVAL_PS),
    ),
    (
        "MAX_STEPS_WITHOUT_SENTINEL",
        DefaultLiteral::U32(DEFAULT_MAX_STEPS_WITHOUT_SENTINEL),
    ),
    (
        "PHASE_CEILING_0",
        DefaultLiteral::U32(DEFAULT_PHASE_CEILING_0),
    ),
    (
        "PHASE_CEILING_1",
        DefaultLiteral::U32(DEFAULT_PHASE_CEILING_1),
    ),
    (
        "PHASE_CEILING_2",
        DefaultLiteral::U32(DEFAULT_PHASE_CEILING_2),
    ),
    (
        "PHASE_CEILING_3",
        DefaultLiteral::U32(DEFAULT_PHASE_CEILING_3),
    ),
    (
        "PHASE_CEILING_4",
        DefaultLiteral::U32(DEFAULT_PHASE_CEILING_4),
    ),
    (
        "PHASE_CEILING_5",
        DefaultLiteral::U32(DEFAULT_PHASE_CEILING_5),
    ),
    (
        "PHASE_CEILING_6",
        DefaultLiteral::U32(DEFAULT_PHASE_CEILING_6),
    ),
    (
        "PHASE_CEILING_7",
        DefaultLiteral::U32(DEFAULT_PHASE_CEILING_7),
    ),
    ("N_PHASE_SLOTS", DefaultLiteral::U32(DEFAULT_N_PHASE_SLOTS)),
    (
        "REQUIRE_ACCEL_GUARD",
        DefaultLiteral::U32(DEFAULT_REQUIRE_ACCEL_GUARD),
    ),
    (
        "REQUIRE_THERMOSTAT_GUARD",
        DefaultLiteral::U32(DEFAULT_REQUIRE_THERMOSTAT_GUARD),
    ),
    (
        "REQUIRE_INTEGRATION_GUARD",
        DefaultLiteral::U32(DEFAULT_REQUIRE_INTEGRATION_GUARD),
    ),
    (
        "REQUIRE_OBSERVATION_CONFIDENCE",
        DefaultLiteral::U32(DEFAULT_REQUIRE_OBSERVATION_CONFIDENCE),
    ),
    (
        "UNARMED_CAP_GEAR",
        DefaultLiteral::U32(DEFAULT_UNARMED_CAP_GEAR),
    ),
];

// ═══════════════════════════════════════════════════════════════════════════
// Wire structures
// ═══════════════════════════════════════════════════════════════════════════

/// Resolved extension policy. Mirrors `PrismPhysGovExtPolicy` (208 bytes).
///
/// Immutable for the life of a run. `ext_policy_digest` is the host-computed
/// fingerprint of the resolved values, their per-field origins and the policy
/// file's sha256; it is copied verbatim into every decision so an offline
/// reader can bind a gear choice to the exact policy that authorized it.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtPolicy {
    /// [`PHYSGOV_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// 0 until the host resolves and arms a policy.
    pub armed: u32,
    /// Whether an unmeasured acceleration guard blocks upshift.
    pub require_accel_guard_for_upshift: u32,
    /// Whether an unmeasured thermostat guard blocks upshift.
    pub require_thermostat_guard_for_upshift: u32,
    /// Whether an unmeasured integration-health guard blocks upshift.
    pub require_integration_guard_for_upshift: u32,
    /// Whether observation confidence is required before coarsening.
    pub require_observation_confidence: u32,
    /// Anti-blindness control (b): forced high-resolution sentinel interval.
    pub max_steps_without_sentinel: u32,
    /// Number of populated entries in [`ExtPolicy::phase_ceiling_gear`].
    pub n_phase_slots: u32,

    /// Force-spike HOLD band, per simulated ps.
    pub force_rate_hold_per_ps: f32,
    /// Force-spike DOWNSHIFT band, per simulated ps.
    pub force_rate_downshift_per_ps: f32,
    /// Force-spike FLOOR band, per simulated ps.
    pub force_rate_floor_per_ps: f32,
    /// Denominator floor for the relative force rate, kcal/mol/A.
    pub force_rate_norm_floor: f32,

    /// Reference displacement (the X–H bond length), Angstrom.
    pub accel_displacement_reference_a: f32,
    /// Acceleration OK band, as a fraction of the reference.
    pub accel_ok_fraction: f32,
    /// Acceleration HOLD band.
    pub accel_hold_fraction: f32,
    /// Acceleration DOWNSHIFT band.
    pub accel_downshift_fraction: f32,
    /// Acceleration EMERGENCY band, evaluated at the finest gear.
    pub accel_abort_fraction: f32,

    /// `gamma*dt` OK band.
    pub gamma_dt_ok: f32,
    /// `gamma*dt` HOLD band.
    pub gamma_dt_hold: f32,
    /// `gamma*dt` DOWNSHIFT band.
    pub gamma_dt_downshift: f32,
    /// `gamma*dt` diagnostic abort band (recorded; never escalates past FLOOR).
    pub gamma_dt_abort: f32,
    /// Thermostat setpoint-deviation OK band.
    pub temp_dev_ok_fraction: f32,
    /// Thermostat setpoint-deviation HOLD band.
    pub temp_dev_hold_fraction: f32,
    /// Thermostat setpoint-deviation DOWNSHIFT band.
    pub temp_dev_downshift_fraction: f32,
    /// Thermostat setpoint-deviation abort band.
    pub temp_dev_abort_fraction: f32,

    /// Neighbour-list occupancy OK band.
    pub neighbor_occupancy_ok: f32,
    /// Neighbour-list occupancy HOLD band.
    pub neighbor_occupancy_hold: f32,
    /// Clamped-force-fraction HOLD band.
    pub clamped_fraction_hold: f32,
    /// Clamped-force-fraction DOWNSHIFT band.
    pub clamped_fraction_downshift: f32,
    /// Clamped-force-fraction FLOOR band.
    pub clamped_fraction_floor: f32,

    /// TSH instability-memory HOLD band.
    pub tsh_memory_hold: f32,
    /// TSH instability-memory FLOOR band.
    pub tsh_memory_floor: f32,
    /// Anti-blindness control (d) threshold.
    pub min_observation_confidence: f32,
    /// Anti-blindness control (a) cadence, in simulated ps.
    pub max_observation_interval_ps: f32,

    /// Per-phase gear ceilings, indexed by `phase_id`.
    pub phase_ceiling_gear: [u32; 8],
    /// Fingerprint of the resolved policy, echoed by every decision.
    pub ext_policy_digest: u64,
    /// Reserved; zeroed.
    pub _reserved: [u32; 8],
}

const _: () = assert!(size_of::<ExtPolicy>() == 208);
const _: () = assert!(align_of::<ExtPolicy>() == 16);

impl Default for ExtPolicy {
    /// The documented defaults. Mirrors
    /// `prism_physgov_fill_default_ext_policy`. `armed` is 0: the host must
    /// resolve a policy and arm it explicitly, so an unarmed governor is never
    /// mistaken for a governed one.
    fn default() -> Self {
        Self {
            schema_version: PHYSGOV_SCHEMA_VERSION,
            armed: 0,
            require_accel_guard_for_upshift: DEFAULT_REQUIRE_ACCEL_GUARD,
            require_thermostat_guard_for_upshift: DEFAULT_REQUIRE_THERMOSTAT_GUARD,
            require_integration_guard_for_upshift: DEFAULT_REQUIRE_INTEGRATION_GUARD,
            require_observation_confidence: DEFAULT_REQUIRE_OBSERVATION_CONFIDENCE,
            max_steps_without_sentinel: DEFAULT_MAX_STEPS_WITHOUT_SENTINEL,
            n_phase_slots: DEFAULT_N_PHASE_SLOTS,
            force_rate_hold_per_ps: DEFAULT_FORCE_RATE_HOLD_PER_PS,
            force_rate_downshift_per_ps: DEFAULT_FORCE_RATE_DOWNSHIFT_PER_PS,
            force_rate_floor_per_ps: DEFAULT_FORCE_RATE_FLOOR_PER_PS,
            force_rate_norm_floor: DEFAULT_FORCE_RATE_NORM_FLOOR,
            accel_displacement_reference_a: DEFAULT_ACCEL_DISPLACEMENT_REFERENCE_A,
            accel_ok_fraction: DEFAULT_ACCEL_OK_FRACTION,
            accel_hold_fraction: DEFAULT_ACCEL_HOLD_FRACTION,
            accel_downshift_fraction: DEFAULT_ACCEL_DOWNSHIFT_FRACTION,
            accel_abort_fraction: DEFAULT_ACCEL_ABORT_FRACTION,
            gamma_dt_ok: DEFAULT_GAMMA_DT_OK,
            gamma_dt_hold: DEFAULT_GAMMA_DT_HOLD,
            gamma_dt_downshift: DEFAULT_GAMMA_DT_DOWNSHIFT,
            gamma_dt_abort: DEFAULT_GAMMA_DT_ABORT,
            temp_dev_ok_fraction: DEFAULT_TEMP_DEV_OK_FRACTION,
            temp_dev_hold_fraction: DEFAULT_TEMP_DEV_HOLD_FRACTION,
            temp_dev_downshift_fraction: DEFAULT_TEMP_DEV_DOWNSHIFT_FRACTION,
            temp_dev_abort_fraction: DEFAULT_TEMP_DEV_ABORT_FRACTION,
            neighbor_occupancy_ok: DEFAULT_NEIGHBOR_OCCUPANCY_OK,
            neighbor_occupancy_hold: DEFAULT_NEIGHBOR_OCCUPANCY_HOLD,
            clamped_fraction_hold: DEFAULT_CLAMPED_FRACTION_HOLD,
            clamped_fraction_downshift: DEFAULT_CLAMPED_FRACTION_DOWNSHIFT,
            clamped_fraction_floor: DEFAULT_CLAMPED_FRACTION_FLOOR,
            tsh_memory_hold: DEFAULT_TSH_MEMORY_HOLD,
            tsh_memory_floor: DEFAULT_TSH_MEMORY_FLOOR,
            min_observation_confidence: DEFAULT_MIN_OBSERVATION_CONFIDENCE,
            max_observation_interval_ps: DEFAULT_MAX_OBSERVATION_INTERVAL_PS,
            phase_ceiling_gear: [
                DEFAULT_PHASE_CEILING_0,
                DEFAULT_PHASE_CEILING_1,
                DEFAULT_PHASE_CEILING_2,
                DEFAULT_PHASE_CEILING_3,
                DEFAULT_PHASE_CEILING_4,
                DEFAULT_PHASE_CEILING_5,
                DEFAULT_PHASE_CEILING_6,
                DEFAULT_PHASE_CEILING_7,
            ],
            ext_policy_digest: 0,
            _reserved: [0; 8],
        }
    }
}

/// One decision's extension guard observations. Mirrors
/// `PrismPhysGovExtObservation` (112 bytes).
///
/// Every `*_measured` field is the tri-state discriminator: 0 = UNMEASURED,
/// and the paired value carries NaN or [`U32_UNMEASURED`]. Nothing in this
/// struct is ever back-filled with a plausible default.
///
/// **Producer contract (determinism).** `max_accel_a_per_ps2`,
/// `force_norm_prev` and `max_neighbor_count` are MAXIMA; `integration_faults`
/// and `steps_since_sentinel` are integer COUNTS. Those are order-invariant
/// reductions and may be produced with `atomicMax` or integer `atomicAdd`
/// without making two identical runs disagree.
///
/// `clamped_force_fraction` is **not** in that set, despite being a ratio of
/// two counts: it is an `f32` that arrives pre-divided, so a producer reading
/// the sentence above as licence would reach for a FLOAT `atomicAdd` — exactly
/// the non-associative reduction this contract forbids — and the low mantissa
/// bits would flip a gear at the `clamped_fraction_downshift = 1e-4` boundary
/// between two identical runs. Its numerator and denominator must be reduced as
/// integers and divided once, or the ratio must come from
/// [`crate::deterministic_reduce`] (Package 2).
///
/// Do not add a float SUM observable here without specifying a fixed-order
/// reduction: `zstr_force_stage_f4_kernel`'s `atomicAdd` sum of `|F|^2` is
/// exactly the pattern that must not be reused for a decision input, because
/// float addition is not associative and the low bits can flip a threshold
/// comparison.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtObservation {
    /// [`PHYSGOV_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// TOPOLOGY-DERIVED `PrismPrepTopology::hmr_applied`, tri-state
    /// `PERMISSION_*`. The HMR hard gate consults this, never a policy field.
    pub hmr_applied_verified: u32,
    /// 0 = the force-rate pair was not measured.
    pub force_rate_measured: u32,
    /// `max|F|` at the previous decision, kcal/mol/A.
    pub force_norm_prev: f32,
    /// Simulated picoseconds the force delta spans.
    pub dt_prev_ps: f32,
    /// 0 = the acceleration extreme was not measured.
    pub accel_measured: u32,
    /// `max |F|/m` over all atoms, A/ps^2.
    pub max_accel_a_per_ps2: f32,
    /// 0 = the thermostat state was not measured.
    pub thermostat_measured: u32,
    /// `ProtocolState.effective_gamma`, ps^-1.
    pub effective_gamma_per_ps: f32,
    /// Instantaneous temperature, K.
    pub instantaneous_temp_k: f32,
    /// `ProtocolState.current_temperature`, K.
    pub target_temp_k: f32,
    /// 0 = integration health was not measured.
    pub integration_measured: u32,
    /// Largest per-atom neighbour count this window.
    pub max_neighbor_count: u32,
    /// `NEIGHBOR_LIST_SIZE`.
    pub neighbor_list_capacity: u32,
    /// Fraction of atoms whose force was clipped by the integrator clamp.
    pub clamped_force_fraction: f32,
    /// Hard faults reported by the integrator.
    pub integration_faults: u32,
    /// 0 = no TSH arm bound.
    pub tsh_memory_measured: u32,
    /// TSH integrated instability recollection in [0,1], normalised per
    /// simulated ps — never per record, per chunk or per invocation.
    pub tsh_instability_memory: f32,
    /// Observation confidence in [0,1] (anti-blindness control (c)/(d)).
    pub observation_confidence: f32,
    /// Simulated ps since the last tensor observation (anti-blindness (a)).
    pub ps_since_last_observation: f32,
    /// Steps since the last forced high-resolution sentinel (anti-blindness (b)).
    pub steps_since_sentinel: u32,
    /// 0 = the protocol phase was not measured.
    pub phase_measured: u32,
    /// Protocol phase index into [`ExtPolicy::phase_ceiling_gear`].
    pub phase_id: u32,
    /// Reserved; zeroed.
    pub _reserved: [u32; 5],
}

const _: () = assert!(size_of::<ExtObservation>() == 112);
const _: () = assert!(align_of::<ExtObservation>() == 16);

impl Default for ExtObservation {
    /// Everything UNMEASURED, with explicit NaN / sentinel payloads. This is
    /// the honest zero state: it asserts nothing about the physics.
    fn default() -> Self {
        Self {
            schema_version: PHYSGOV_SCHEMA_VERSION,
            hmr_applied_verified: PERMISSION_UNKNOWN,
            force_rate_measured: 0,
            force_norm_prev: f32::NAN,
            dt_prev_ps: f32::NAN,
            accel_measured: 0,
            max_accel_a_per_ps2: f32::NAN,
            thermostat_measured: 0,
            effective_gamma_per_ps: f32::NAN,
            instantaneous_temp_k: f32::NAN,
            target_temp_k: f32::NAN,
            integration_measured: 0,
            max_neighbor_count: U32_UNMEASURED,
            neighbor_list_capacity: U32_UNMEASURED,
            clamped_force_fraction: f32::NAN,
            integration_faults: 0,
            tsh_memory_measured: 0,
            tsh_instability_memory: f32::NAN,
            observation_confidence: f32::NAN,
            ps_since_last_observation: f32::NAN,
            steps_since_sentinel: U32_UNMEASURED,
            phase_measured: 0,
            phase_id: U32_UNMEASURED,
            _reserved: [0; 5],
        }
    }
}

/// The base governor's decision. Mirrors `PrismGearDecision` (32 bytes),
/// which is produced by `prism_gearbox_governor_decide` — the single
/// implementation of base gates 1..10, shared by the SFA kernel and the host.
///
/// [`compose`] takes this as an INPUT rather than recomputing it, so the base
/// gate logic exists exactly once in the tree.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BaseDecision {
    /// Gear the base governor authorized.
    pub authorized_gear: u32,
    /// Ceiling the base gates published.
    pub ceiling_gear: u32,
    /// Base gate that bound the ceiling, `PRISM_GEAR_GATE_*`.
    pub binding_gate: u32,
    /// Packed 4-bit base gate verdicts.
    pub gate_verdicts: u32,
    /// `PRISM_GEAR_EMERG_*` cause bits.
    pub emergency_flags: u32,
    /// Dwell counter for the next decision.
    pub next_stable_dwell_steps: u32,
    /// Lockout counter for the next decision.
    pub next_steps_since_instability: u32,
    /// `chunk_span_steps`.
    pub downshift_latency_steps: u32,
}

const _: () = assert!(size_of::<BaseDecision>() == 32);

/// The base-policy and base-observation fields the extension layer needs.
///
/// Deliberately NOT a second mirror of the 144-byte `PrismGearGovernorPolicy`:
/// re-declaring that ABI here would create the exact duplicate-source-of-truth
/// hazard this module exists to avoid. Only the fields the extension gates
/// actually read appear, and [`BaseInputs::from_gearbox`] builds it from the
/// canonical structs when the `gpu` feature is on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaseInputs {
    /// `policy.gear_dt_fs[0..4]` — canonical gear timesteps in fs.
    pub gear_dt_fs: [f32; 4],
    /// `policy.chunk_span_steps` — the captured-graph `body_unroll`.
    pub chunk_span_steps: u32,
    /// `policy.policy_digest`.
    pub policy_digest: u64,
    /// `observation.current_gear`.
    pub current_gear: u32,
    /// `observation.requested_gear` — the scientific request.
    pub requested_gear: u32,
    /// `observation.force_measured`.
    pub force_measured: u32,
    /// `observation.force_norm`, kcal/mol/A.
    pub force_norm: f32,
}

impl Default for BaseInputs {
    /// The HONEST ZERO: nothing measured, nothing resolved, nothing assumed.
    ///
    /// This deliberately does NOT transcribe `d_gearbox_table`
    /// (`{0.5, 2.0, 4.0, NaN}` fs). A plausible-looking gear table and a
    /// plausible-looking `current_gear` in a `Default` are indistinguishable
    /// from a resolved policy once they are three call frames away from here,
    /// and this module's stated rule is that no fabricated value may pass as a
    /// measurement. The canonical table has exactly one source — the base
    /// `PrismGearGovernorPolicy` — and [`BaseInputs::from_gearbox`] is how it
    /// gets here.
    ///
    /// A NaN-filled `gear_dt_fs` makes [`BaseInputs::gear_dt_ps`] return `None`
    /// for every gear, so any decision built on this default publishes a NaN
    /// `dt_applied_ps` rather than a believable timestep, and `current_gear`
    /// starts at the finest (always-reachable, fail-closed) gear.
    fn default() -> Self {
        Self {
            gear_dt_fs: [f32::NAN; 4],
            chunk_span_steps: 1,
            policy_digest: 0,
            current_gear: GEAR_FINE,
            requested_gear: GEAR_FINE,
            force_measured: 0,
            force_norm: f32::NAN,
        }
    }
}

impl BaseInputs {
    /// `dt` in picoseconds for a gear, or `None` when the slot has no timestep
    /// (gear 3, or an unresolvable policy). Never substitutes a default.
    #[must_use]
    pub fn gear_dt_ps(&self, gear: u32) -> Option<f32> {
        if gear as usize >= self.gear_dt_fs.len() {
            return None;
        }
        let fs = self.gear_dt_fs[gear as usize];
        if !fs.is_finite() || fs <= 0.0 {
            return None;
        }
        Some(fs * 0.001)
    }

    /// Build from the canonical gearbox structs.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn from_gearbox(
        policy: &crate::gearbox::GearGovernorPolicy,
        observation: &crate::gearbox::GearGuardObservation,
    ) -> Self {
        Self {
            gear_dt_fs: policy.gear_dt_fs,
            chunk_span_steps: policy.chunk_span_steps,
            policy_digest: policy.policy_digest,
            current_gear: observation.current_gear,
            requested_gear: observation.requested_gear,
            force_measured: observation.force_measured,
            force_norm: observation.force_norm,
        }
    }
}

/// The composed governor decision. Mirrors `PrismPhysGovDecision` (128 bytes).
///
/// This is the COMPACT NATIVE RECORD written to the device ring. It is POD with
/// no pointers, and producing it involves no JSON, hashing, compression or
/// filesystem work — the hot path stays compact by construction.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PhysGovDecision {
    /// [`PHYSGOV_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// `min(base authorized, extension ceiling)`, then the HMR hard gate.
    pub authorized_gear: u32,
    /// `min(base ceiling, extension ceiling)`, then the HMR hard gate.
    pub ceiling_gear: u32,
    /// What the base governor authorized before this layer.
    pub base_authorized_gear: u32,
    /// What the base governor's ceiling was.
    pub base_ceiling_gear: u32,
    /// Base binding gate, `PRISM_GEAR_GATE_*`.
    pub base_binding_gate: u32,
    /// Extension binding gate, `GATE_*`, or [`GATE_NONE`].
    pub ext_binding_gate: u32,
    /// Authority rank of whichever gate bound the ceiling.
    pub binding_gate_rank: u32,
    /// Packed base gate verdicts.
    pub base_gate_verdicts: u32,
    /// Packed extension gate verdicts.
    pub ext_gate_verdicts: u32,
    /// `PRISM_GEAR_EMERG_*`.
    pub emergency_flags: u32,
    /// `EMERG_*`.
    pub ext_emergency_flags: u32,
    /// What the science asked for. Never collapsed into `authorized_gear`.
    pub requested_gear: u32,
    /// The gear in force when this decision ran.
    pub current_gear: u32,
    /// Dwell counter for the next decision.
    pub next_stable_dwell_steps: u32,
    /// Lockout counter for the next decision.
    pub next_steps_since_instability: u32,
    /// The named chunk-boundary bound, in protocol steps.
    pub downshift_latency_steps: u32,
    /// The named chunk-boundary bound, in fs.
    pub downshift_latency_fs: f32,
    /// [`TRANSITION_HOLD`] / `DOWNSHIFT` / `UPSHIFT` / `ABORT`.
    pub transition: u32,
    /// ALWAYS 1 — emit on no-change too.
    pub decision_emitted: u32,
    /// `gear_dt_fs[authorized]/1000`, or NaN at gear 3.
    pub dt_applied_ps: f32,
    /// Echo of [`ExtPolicy::armed`].
    pub ext_armed: u32,
    /// Base policy fingerprint.
    pub policy_digest: u64,
    /// Extension policy fingerprint.
    pub ext_policy_digest: u64,
    /// Named cause when [`Self::ext_binding_gate`] is [`GATE_EXT_BINDING`]:
    /// [`REFUSAL_NONE`] / [`REFUSAL_UNARMED`] / [`REFUSAL_POLICY_SCHEMA`] /
    /// [`REFUSAL_OBSERVATION_SCHEMA`]. Every refusal path in this module names
    /// its cause here, so no refusal is ever unexplained.
    pub ext_binding_refusal: u32,
    /// Reserved; zeroed.
    pub _reserved: [u32; 5],
}

const _: () = assert!(size_of::<PhysGovDecision>() == 128);
const _: () = assert!(align_of::<PhysGovDecision>() == 16);

impl PhysGovDecision {
    /// Extract an extension gate's 4-bit verdict.
    #[must_use]
    pub fn ext_verdict(&self, gate: u32) -> u32 {
        if !(GATE_EXT_FIRST..=GATE_EXT_LAST).contains(&gate) {
            return VERDICT_UNMEASURED;
        }
        (self.ext_gate_verdicts >> ((gate - GATE_EXT_FIRST) * 4)) & 0xF
    }

    /// True when this decision authorized a coarser gear than was in force.
    #[must_use]
    pub fn is_upshift(&self) -> bool {
        self.transition == TRANSITION_UPSHIFT
    }

    /// True when this decision authorized a finer gear than was in force.
    #[must_use]
    pub fn is_downshift(&self) -> bool {
        self.transition == TRANSITION_DOWNSHIFT
    }
}

fn set_ext_verdict(verdicts: &mut u32, gate: u32, verdict: u32) {
    if !(GATE_EXT_FIRST..=GATE_EXT_LAST).contains(&gate) {
        return;
    }
    let shift = (gate - GATE_EXT_FIRST) * 4;
    *verdicts = (*verdicts & !(0xF << shift)) | ((verdict & 0xF) << shift);
}

/// Authority rank of a base gate id. Unknown ids sort last, so an unexpected
/// gate can never claim authority it was not granted.
#[must_use]
pub fn base_gate_rank(gate: u32) -> u32 {
    match gate {
        BASE_GATE_PHYSICAL_EMERGENCY => RANK_PHYSICAL_EMERGENCY,
        BASE_GATE_TOPOLOGY_PERMISSION => RANK_TOPOLOGY_PERMISSION,
        BASE_GATE_GOVERNOR_UNARMED => RANK_GOVERNOR_UNARMED,
        BASE_GATE_CONSTRAINT_STABILITY => RANK_CONSTRAINT_STABILITY,
        BASE_GATE_ENERGY_DRIFT => RANK_ENERGY_DRIFT,
        BASE_GATE_FORCE_STABILITY => RANK_FORCE_STABILITY,
        BASE_GATE_CHUNK_LATENCY => RANK_CHUNK_LATENCY,
        BASE_GATE_TSH_SALIENCE => RANK_TSH_SALIENCE,
        BASE_GATE_DWELL => RANK_DWELL,
        BASE_GATE_LOCKOUT => RANK_LOCKOUT,
        BASE_GATE_UPSHIFT_RATE => RANK_UPSHIFT_RATE,
        _ => RANK_NONE,
    }
}

/// Authority rank of an extension gate id.
#[must_use]
pub fn gate_rank(gate: u32) -> u32 {
    match gate {
        GATE_HMR_HARD => RANK_HMR_HARD,
        GATE_FORCE_SPIKE => RANK_FORCE_SPIKE,
        GATE_ACCELERATION_EXTREME => RANK_ACCELERATION_EXTREME,
        GATE_THERMOSTAT_STABILITY => RANK_THERMOSTAT_STABILITY,
        GATE_INTEGRATION_HEALTH => RANK_INTEGRATION_HEALTH,
        GATE_TSH_MEMORY => RANK_TSH_MEMORY,
        GATE_PHASE_REQUIREMENT => RANK_PHASE_REQUIREMENT,
        GATE_EXT_BINDING => RANK_GOVERNOR_UNARMED,
        _ => RANK_NONE,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-guard evaluation
//
// Each returns (ceiling, verdict). A ceiling of GEAR_MAX_ACTIVE means "this
// guard constrains nothing". Transliteration of the same-named functions in
// `physical_governor.cuh`.
// ═══════════════════════════════════════════════════════════════════════════

/// Mutable accumulators threaded through the guards, mirroring the `bool*`
/// out-parameters in the header.
///
/// `Default` is NOT derived. A derived `Default` yields
/// `all_guards_ok: false` — the opposite of the value [`compose`] constructs —
/// and a guard accumulator that starts life claiming the guards already failed
/// is a trap waiting for the first caller who reaches for `..Default::default()`.
/// [`GuardAccum::clean`] is the only way to make one, and it names its
/// starting state.
#[derive(Debug, Clone, Copy)]
struct GuardAccum {
    instability: bool,
    all_guards_ok: bool,
    ext_emergency: u32,
}

impl GuardAccum {
    /// The starting state of a decision: nothing unstable, no guard has yet
    /// reported degradation, no emergency raised.
    const fn clean() -> Self {
        Self {
            instability: false,
            all_guards_ok: true,
            ext_emergency: 0,
        }
    }
}

fn one_finer(gear: u32) -> u32 {
    if gear > GEAR_FINE {
        gear - 1
    } else {
        GEAR_FINE
    }
}

/// Band a per-candidate-gear observable landed in, expressed as
/// `(coarsest gear that band permits, the band's own verdict name)`.
///
/// This is the single place the "HOLD pins to the current gear, DOWNSHIFT pins
/// one finer" rule is written, shared by the acceleration and thermostat guards
/// so the two cannot drift apart. Mirrors `prism_physgov_band_cap`.
///
/// A non-finite observable at a candidate gear is UNMEASURED and permits only
/// the finest gear: an unresolvable number is not licence to run coarse.
fn band_cap(value: f32, ok: f32, hold: f32, downshift: f32, cur: u32) -> (u32, u32) {
    if !value.is_finite() {
        return (GEAR_FINE, VERDICT_UNMEASURED);
    }
    if value < ok {
        (GEAR_MAX_ACTIVE, VERDICT_OK)
    } else if value < hold {
        (cur, VERDICT_HOLD)
    } else if value < downshift {
        (one_finer(cur), VERDICT_DOWNSHIFT)
    } else {
        (GEAR_FINE, VERDICT_FLOOR)
    }
}

/// Guard 5 — FORCE SPIKE. Relative growth of `max|F|` per simulated ps.
fn gate_force_spike(
    p: &ExtPolicy,
    b: &BaseInputs,
    e: &ExtObservation,
    cur: u32,
    acc: &mut GuardAccum,
) -> (u32, u32) {
    if e.force_rate_measured == 0 || b.force_measured == 0 {
        // A refinement of the force guard — but it must not delegate its own
        // upshift veto to `require_force_guard_for_upshift`, a BASE policy
        // switch this layer cannot read. With that switch at 0 the base leaves
        // `all_guards_ok` true, so abstaining here would let the composed
        // governor authorise 4.0 fs with no force evidence at all. An upshift
        // is a claim that the physics is verified safe; an unread force pair is
        // not verification. Hold the current gear.
        acc.all_guards_ok = false;
        return (cur, VERDICT_UNMEASURED);
    }
    if !e.dt_prev_ps.is_finite()
        || e.dt_prev_ps <= 0.0
        || !e.force_norm_prev.is_finite()
        || !b.force_norm.is_finite()
    {
        // Present but unusable: recorded as UNMEASURED, never guessed at.
        acc.all_guards_ok = false;
        return (cur, VERDICT_UNMEASURED);
    }

    let denom_floor = if p.force_rate_norm_floor.is_finite() && p.force_rate_norm_floor > 0.0 {
        p.force_rate_norm_floor
    } else {
        1.0
    };
    let prev = if e.force_norm_prev > denom_floor {
        e.force_norm_prev
    } else {
        denom_floor
    };
    // `force_rate_norm_floor` is a DENOMINATOR floor only. Clamping the
    // numerator as well under-reports real growth: with prev=0.1, floor=1.0 and
    // now=2.0 the true relative growth is 19/dt but a clamped numerator yields
    // 1/dt — a 19x under-report that errs toward a COARSER gear, the one
    // direction this module is not allowed to err in.
    let rate = (b.force_norm - e.force_norm_prev) / (prev * e.dt_prev_ps);

    // Only GROWTH matters. A collapsing force norm is a relaxation, not an
    // instability, and must not be punished with a finer gear.
    if !rate.is_finite() || rate <= 0.0 {
        return (GEAR_MAX_ACTIVE, VERDICT_OK);
    }
    if rate >= p.force_rate_floor_per_ps {
        acc.instability = true;
        acc.all_guards_ok = false;
        return (GEAR_FINE, VERDICT_FLOOR);
    }
    if rate >= p.force_rate_downshift_per_ps {
        acc.instability = true;
        acc.all_guards_ok = false;
        return (one_finer(cur), VERDICT_DOWNSHIFT);
    }
    if rate >= p.force_rate_hold_per_ps {
        acc.all_guards_ok = false;
        return (cur, VERDICT_HOLD);
    }
    (GEAR_MAX_ACTIVE, VERDICT_OK)
}

/// Guard 6 — ACCELERATION EXTREMES. Per-candidate-gear: `dx = 0.5*a*dt^2`.
fn gate_acceleration(
    p: &ExtPolicy,
    b: &BaseInputs,
    e: &ExtObservation,
    cur: u32,
    acc: &mut GuardAccum,
) -> (u32, u32) {
    if e.accel_measured == 0 {
        if p.require_accel_guard_for_upshift != 0 {
            acc.all_guards_ok = false;
            return (cur, VERDICT_UNMEASURED);
        }
        return (GEAR_MAX_ACTIVE, VERDICT_UNMEASURED);
    }
    if !e.max_accel_a_per_ps2.is_finite() {
        acc.ext_emergency |= EMERG_NONFINITE_ACCEL;
        acc.instability = true;
        acc.all_guards_ok = false;
        return (GEAR_FINE, VERDICT_EMERGENCY);
    }

    let reference =
        if p.accel_displacement_reference_a.is_finite() && p.accel_displacement_reference_a > 0.0 {
            p.accel_displacement_reference_a
        } else {
            1.0
        };
    let a = e.max_accel_a_per_ps2.max(0.0);

    // EMERGENCY at the FINEST gear: if even 0.5 fs cannot keep the displacement
    // below half a bond length, no gear can and SHAKE will not reconverge.
    if let Some(dt0) = b.gear_dt_ps(GEAR_FINE) {
        if 0.5 * a * dt0 * dt0 >= p.accel_abort_fraction * reference {
            acc.ext_emergency |= EMERG_ACCEL_DISPLACEMENT;
            acc.instability = true;
            acc.all_guards_ok = false;
            return (GEAR_FINE, VERDICT_EMERGENCY);
        }
    }

    // PER-CANDIDATE-GEAR: the coarsest gear its OWN band still permits.
    // `accel_hold_fraction` and `accel_downshift_fraction` GOVERN here; they do
    // not merely relabel a verdict. Scanning only for `dx < ok` and flooring
    // when nothing qualifies over-restricts by the whole width of the HOLD
    // band — at cur = 1 with dx(1) = 0.035 A the documented answer is HOLD AT
    // GEAR 1, not a forced downshift to gear 0.
    let ok_edge = p.accel_ok_fraction * reference;
    let hold_edge = p.accel_hold_fraction * reference;
    let down_edge = p.accel_downshift_fraction * reference;

    let mut ceiling = GEAR_FINE;
    for g in 0..=GEAR_MAX_ACTIVE {
        // No dt for this gear means no claim about it, in either direction.
        let Some(dt) = b.gear_dt_ps(g) else { continue };
        let (cap, _) = band_cap(0.5 * a * dt * dt, ok_edge, hold_edge, down_edge, cur);
        if g <= cap && g > ceiling {
            ceiling = g;
        }
    }

    // The verdict names the band the CURRENT gear sits in — the timestep
    // actually in force — falling back to the ceiling's relationship to `cur`
    // when the current gear is itself inside the OK band.
    let dxc = b.gear_dt_ps(cur).map_or(f32::NAN, |dt| 0.5 * a * dt * dt);
    let (_, cur_band) = band_cap(dxc, ok_edge, hold_edge, down_edge, cur);
    let verdict = if cur_band == VERDICT_FLOOR || cur_band == VERDICT_DOWNSHIFT {
        acc.instability = true;
        acc.all_guards_ok = false;
        cur_band
    } else if cur_band != VERDICT_OK {
        acc.all_guards_ok = false;
        cur_band
    } else if ceiling < cur {
        acc.instability = true;
        acc.all_guards_ok = false;
        VERDICT_DOWNSHIFT
    } else if ceiling < GEAR_MAX_ACTIVE {
        acc.all_guards_ok = false;
        VERDICT_HOLD
    } else {
        VERDICT_OK
    };
    (ceiling, verdict)
}

/// Guard 7 — THERMOSTAT STABILITY. Per-candidate-gear on `gamma*dt`, plus a
/// setpoint-deviation band.
///
/// `gamma*dt` never escalates past FLOOR: a large `gamma*dt` is an accuracy
/// defect, not a divergence, and a legitimate high-friction equilibration must
/// not abort the run. [`ExtPolicy::gamma_dt_abort`] is not decorative either —
/// at `gamma*dt >= 1.0` the OU factor `c1 = exp(-gamma*dt) <= 0.37`, the
/// velocity is essentially fully resampled every step and the trajectory has
/// stopped being inertial, so crossing it raises INSTABILITY (zeroing dwell and
/// restarting the lockout window) on top of the FLOOR.
///
/// The SETPOINT band does escalate: at
/// `dev >= temp_dev_abort_fraction` the temperature has run away threefold and
/// no timestep repairs a diverging kinetic energy, so that is
/// [`EMERG_TEMPERATURE_RUNAWAY`] and the run traps. A NON-FINITE temperature is
/// FLOOR plus instability, NOT an emergency: NaN is an absent measurement, and
/// claiming a threefold runaway from it would fabricate a finding the
/// instrument never made.
fn gate_thermostat(
    p: &ExtPolicy,
    b: &BaseInputs,
    e: &ExtObservation,
    cur: u32,
    acc: &mut GuardAccum,
) -> (u32, u32) {
    if e.thermostat_measured == 0 {
        if p.require_thermostat_guard_for_upshift != 0 {
            acc.all_guards_ok = false;
            return (cur, VERDICT_UNMEASURED);
        }
        return (GEAR_MAX_ACTIVE, VERDICT_UNMEASURED);
    }

    let mut ceiling = GEAR_MAX_ACTIVE;
    let mut verdict = VERDICT_OK;

    if e.effective_gamma_per_ps.is_finite() && e.effective_gamma_per_ps > 0.0 {
        // PER-CANDIDATE-GEAR: the coarsest gear its OWN band still permits.
        let mut g_ceiling = GEAR_FINE;
        for g in 0..=GEAR_MAX_ACTIVE {
            // No dt for this gear means no claim about it, in either direction.
            let Some(dt) = b.gear_dt_ps(g) else { continue };
            let (cap, _) = band_cap(
                e.effective_gamma_per_ps * dt,
                p.gamma_dt_ok,
                p.gamma_dt_hold,
                p.gamma_dt_downshift,
                cur,
            );
            if g <= cap && g > g_ceiling {
                g_ceiling = g;
            }
        }

        let xc = b
            .gear_dt_ps(cur)
            .map_or(f32::NAN, |dt| e.effective_gamma_per_ps * dt);
        let (_, cur_band) = band_cap(
            xc,
            p.gamma_dt_ok,
            p.gamma_dt_hold,
            p.gamma_dt_downshift,
            cur,
        );

        // `gamma_dt_abort` governs an emitted state transition: it raises
        // INSTABILITY, which zeroes the dwell counter and restarts the lockout
        // window for the next decision.
        if xc.is_finite() && xc >= p.gamma_dt_abort {
            acc.instability = true;
        }

        if cur_band != VERDICT_OK {
            verdict = cur_band;
            acc.all_guards_ok = false;
        } else if g_ceiling < cur {
            verdict = VERDICT_DOWNSHIFT;
            acc.all_guards_ok = false;
        } else if g_ceiling < GEAR_MAX_ACTIVE {
            verdict = VERDICT_HOLD;
            acc.all_guards_ok = false;
        }
        ceiling = ceiling.min(g_ceiling);
    }

    // ── SETPOINT AUTHORITY ─────────────────────────────────────────────────
    //
    // The deviation guard below divides by the SETPOINT. This code used to
    // substitute a literal `1.0` when `target_temp_k` was absent or
    // non-physical, and that substituted constant then became the DENOMINATOR
    // of `dev = |T - target| / target`.
    //
    // That is not a weaker measurement of the same quantity, it is a
    // measurement of a DIFFERENT quantity — the deviation from a 1 K setpoint
    // the run never had. At any real MD temperature it makes `dev` two to three
    // orders of magnitude larger than any band, so the branch below reported
    // `EMERG_TEMPERATURE_RUNAWAY` and trapped the run while naming a runaway
    // the instrument never observed. A fabricated denominator is a fabricated
    // finding; an absent setpoint is UNMEASURED.
    //
    // `> 1.0` (not `> 0.0`) is retained as the physicality floor: a
    // sub-kelvin MD setpoint is outside every gear band this governor tunes,
    // so it is treated as absent rather than as an authority.
    let setpoint = if e.target_temp_k.is_finite() && e.target_temp_k > 1.0 {
        Some(e.target_temp_k)
    } else {
        None
    };
    if !e.instantaneous_temp_k.is_finite() {
        // A non-finite instantaneous temperature is a genuine finding about the
        // integrator, independent of whether a setpoint exists. It keeps its
        // documented FLOOR-plus-instability treatment.
        verdict = VERDICT_FLOOR;
        ceiling = GEAR_FINE;
        acc.instability = true;
        acc.all_guards_ok = false;
    } else if let Some(target) = setpoint {
        let dev = (e.instantaneous_temp_k - target).abs() / target;
        if dev >= p.temp_dev_abort_fraction {
            verdict = VERDICT_EMERGENCY;
            acc.ext_emergency |= EMERG_TEMPERATURE_RUNAWAY;
            ceiling = GEAR_FINE;
            acc.instability = true;
            acc.all_guards_ok = false;
        } else if dev >= p.temp_dev_downshift_fraction {
            verdict = VERDICT_FLOOR;
            ceiling = GEAR_FINE;
            acc.instability = true;
            acc.all_guards_ok = false;
        } else if dev >= p.temp_dev_hold_fraction {
            // Do not overwrite a stronger gamma-derived FLOOR: the record must
            // name the most authoritative cause that actually held, not an
            // incidental weaker one.
            if verdict != VERDICT_FLOOR {
                verdict = VERDICT_DOWNSHIFT;
            }
            ceiling = ceiling.min(one_finer(cur));
            acc.instability = true;
            acc.all_guards_ok = false;
        } else if dev >= p.temp_dev_ok_fraction {
            if verdict == VERDICT_OK {
                verdict = VERDICT_HOLD;
            }
            ceiling = ceiling.min(cur);
            acc.all_guards_ok = false;
        }
    } else {
        // MEASURED instantaneous temperature, NO setpoint authority. The
        // deviation guard has no denominator, so it makes no finding in either
        // direction — it neither licenses a coarser gear nor claims a runaway.
        //
        // It still blocks an UPSHIFT, mirroring the `thermostat_measured == 0`
        // handling at the top of this gate: an unmeasured required guard is
        // never a licence. `VERDICT_UNMEASURED` is written only when the
        // gamma half made no finding of its own, so a real gamma verdict is
        // never demoted to "unmeasured".
        if verdict == VERDICT_OK {
            verdict = VERDICT_UNMEASURED;
        }
        if p.require_thermostat_guard_for_upshift != 0 {
            acc.all_guards_ok = false;
            ceiling = ceiling.min(cur);
        }
    }

    (ceiling, verdict)
}

/// Guard 9 — NEIGHBOUR-LIST / INTEGRATION HEALTH.
///
/// A finer timestep does not repair a truncated pair list, so occupancy never
/// escalates past FLOOR and never claims a coarser gear is warranted. It still
/// publishes a ceiling, because a wrong force integrated over a longer
/// uncorrected window compounds faster.
fn gate_integration_health(
    p: &ExtPolicy,
    e: &ExtObservation,
    cur: u32,
    acc: &mut GuardAccum,
) -> (u32, u32) {
    if e.integration_measured == 0 {
        if p.require_integration_guard_for_upshift != 0 {
            acc.all_guards_ok = false;
            return (cur, VERDICT_UNMEASURED);
        }
        return (GEAR_MAX_ACTIVE, VERDICT_UNMEASURED);
    }

    // `integration_measured` above is ALREADY the discriminator, so this test
    // does not also treat `U32_UNMEASURED` as "not measured". An in-band
    // sentinel on a COUNT means a saturating fault counter that reaches
    // 0xFFFFFFFF — the most violent state representable — would read as
    // "unmeasured" and produce no emergency at all. A sentinel must be
    // out-of-band.
    if e.integration_faults != 0 {
        acc.ext_emergency |= EMERG_INTEGRATION_FAULT;
        acc.instability = true;
        acc.all_guards_ok = false;
        return (GEAR_FINE, VERDICT_EMERGENCY);
    }

    let mut ceiling = GEAR_MAX_ACTIVE;
    let mut verdict = VERDICT_OK;

    if e.neighbor_list_capacity != 0
        && e.neighbor_list_capacity != U32_UNMEASURED
        && e.max_neighbor_count != U32_UNMEASURED
    {
        let occ = e.max_neighbor_count as f32 / e.neighbor_list_capacity as f32;
        if occ >= 1.0 {
            verdict = VERDICT_FLOOR;
            ceiling = GEAR_FINE;
            acc.all_guards_ok = false;
        } else if occ >= p.neighbor_occupancy_hold {
            verdict = VERDICT_DOWNSHIFT;
            ceiling = ceiling.min(one_finer(cur));
            acc.all_guards_ok = false;
        } else if occ >= p.neighbor_occupancy_ok {
            verdict = VERDICT_HOLD;
            ceiling = ceiling.min(cur);
            acc.all_guards_ok = false;
        }
    }

    if e.clamped_force_fraction.is_finite() {
        let f = e.clamped_force_fraction;
        if f > p.clamped_fraction_floor {
            verdict = VERDICT_FLOOR;
            ceiling = GEAR_FINE;
            acc.instability = true;
            acc.all_guards_ok = false;
        } else if f > p.clamped_fraction_downshift {
            if verdict != VERDICT_FLOOR {
                verdict = VERDICT_DOWNSHIFT;
            }
            ceiling = ceiling.min(one_finer(cur));
            acc.instability = true;
            acc.all_guards_ok = false;
        } else if f > p.clamped_fraction_hold {
            if verdict == VERDICT_OK {
                verdict = VERDICT_HOLD;
            }
            ceiling = ceiling.min(cur);
            acc.all_guards_ok = false;
        }
    } else {
        if verdict == VERDICT_OK {
            verdict = VERDICT_UNMEASURED;
        }
        if p.require_integration_guard_for_upshift != 0 {
            ceiling = ceiling.min(cur);
            acc.all_guards_ok = false;
        }
    }

    (ceiling, verdict)
}

/// TSH MEMORY + ANTI-BLINDNESS. Ceiling-only and structurally incapable of
/// raising a gear.
///
/// Enforces all four required anti-blindness controls:
/// (a) minimum physical-time observation cadence,
/// (b) maximum interval without a forced high-resolution sentinel sample,
/// (c) uncertainty growth when evidence is sparse (carried in
///     `observation_confidence`, which the TSH arm decays with elapsed
///     simulated ps rather than with record count),
/// (d) no coarsening while observation confidence is below threshold.
fn gate_tsh_memory(
    p: &ExtPolicy,
    e: &ExtObservation,
    cur: u32,
    acc: &mut GuardAccum,
) -> (u32, u32) {
    if e.tsh_memory_measured == 0 {
        // No TSH arm bound is a legitimate configuration (the governor runs
        // standalone), so it is not an upshift blocker on its own — the
        // physical guards remain in full force.
        return (GEAR_MAX_ACTIVE, VERDICT_UNMEASURED);
    }

    // (b) forced high-resolution sentinel — the strongest anti-blindness rule.
    if p.max_steps_without_sentinel != 0
        && e.steps_since_sentinel != U32_UNMEASURED
        && e.steps_since_sentinel > p.max_steps_without_sentinel
    {
        acc.all_guards_ok = false;
        return (GEAR_FINE, VERDICT_FLOOR);
    }

    let mut ceiling = GEAR_MAX_ACTIVE;
    let mut verdict = VERDICT_OK;

    // (d) no coarsening below the confidence threshold.
    if p.require_observation_confidence != 0
        && (!e.observation_confidence.is_finite()
            || e.observation_confidence < p.min_observation_confidence)
    {
        verdict = VERDICT_HOLD;
        ceiling = ceiling.min(cur);
        acc.all_guards_ok = false;
    }

    // (a) minimum physical-time observation cadence.
    if e.ps_since_last_observation.is_finite()
        && p.max_observation_interval_ps.is_finite()
        && p.max_observation_interval_ps > 0.0
        && e.ps_since_last_observation > p.max_observation_interval_ps
    {
        if verdict == VERDICT_OK {
            verdict = VERDICT_HOLD;
        }
        ceiling = ceiling.min(cur);
        acc.all_guards_ok = false;
    }

    if e.tsh_instability_memory.is_finite() {
        if e.tsh_instability_memory >= p.tsh_memory_floor {
            verdict = VERDICT_FLOOR;
            ceiling = GEAR_FINE;
            acc.all_guards_ok = false;
        } else if e.tsh_instability_memory >= p.tsh_memory_hold {
            if verdict != VERDICT_FLOOR {
                verdict = VERDICT_HOLD;
            }
            ceiling = ceiling.min(cur);
            acc.all_guards_ok = false;
        }
    }

    (ceiling, verdict)
}

/// PROTOCOL-PHASE REQUIREMENT. Ceiling-only.
fn gate_phase(p: &ExtPolicy, e: &ExtObservation, cur: u32) -> (u32, u32) {
    if e.phase_measured == 0 {
        return (GEAR_MAX_ACTIVE, VERDICT_UNMEASURED);
    }
    let slots = p.n_phase_slots.min(8);
    if slots == 0 || e.phase_id >= slots {
        // An unknown phase is not licence to run coarse.
        return (cur, VERDICT_UNMEASURED);
    }
    let ceiling = p.phase_ceiling_gear[e.phase_id as usize].min(GEAR_MAX_ACTIVE);
    let verdict = if ceiling < cur {
        VERDICT_DOWNSHIFT
    } else if ceiling < GEAR_MAX_ACTIVE {
        VERDICT_HOLD
    } else {
        VERDICT_OK
    };
    (ceiling, verdict)
}

/// HMR HARD GATE — operator requirement 3, NOT POLICY-OVERRIDABLE.
///
/// Gear 2 is 4.0 fs. A 4 fs timestep without hydrogen-mass repartitioning does
/// not crash; it silently produces plausible numbers with the wrong dynamics,
/// which is worse than a crash because the output looks usable.
///
/// The base governor's topology gate reads `policy.hmr_permission`, a POLICY
/// field a wrong policy file could assert GRANTED. This gate reads
/// [`ExtObservation::hmr_applied_verified`], which is TOPOLOGY-DERIVED. Gear 2
/// requires BOTH. There is no field in [`ExtPolicy`] that disables this
/// function, and [`compose`] re-asserts it unconditionally after all policy
/// arithmetic.
fn gate_hmr_hard(e: &ExtObservation) -> (u32, u32) {
    if e.hmr_applied_verified == PERMISSION_GRANTED {
        return (GEAR_MAX_ACTIVE, VERDICT_OK);
    }
    let verdict = if e.hmr_applied_verified == PERMISSION_DENIED {
        VERDICT_DOWNSHIFT
    } else {
        // UNKNOWN. Absence of evidence is not evidence of HMR.
        VERDICT_UNMEASURED
    };
    (GEAR_SAFE, verdict)
}

// ═══════════════════════════════════════════════════════════════════════════
// The composed governor
// ═══════════════════════════════════════════════════════════════════════════

/// Compose the extension guards on top of a base governor decision.
///
/// Pure function: no I/O, no global state, no randomness, no allocation. Host
/// mirror of `prism_physical_governor_decide`.
///
/// # Tighten-only
///
/// The result satisfies `authorized_gear <= base.authorized_gear` for every
/// possible input, because the fold is a single `min` against the base result
/// followed only by further `min`s. An extension defect can refuse a gear that
/// was available; it can never authorise one that was not.
#[must_use]
pub fn compose(
    base_inputs: &BaseInputs,
    ext_policy: &ExtPolicy,
    ext_obs: &ExtObservation,
    base: &BaseDecision,
) -> PhysGovDecision {
    let span = if base_inputs.chunk_span_steps == 0 {
        1
    } else {
        base_inputs.chunk_span_steps
    };
    let cur = if base_inputs.current_gear <= GEAR_MAX_ACTIVE {
        base_inputs.current_gear
    } else {
        GEAR_FINE
    };

    // Fixed deterministic preamble. EMIT ON NO-CHANGE TOO: every field is
    // written on every invocation, so a gap in the record stream unambiguously
    // means "the decision did not run".
    let mut out = PhysGovDecision {
        schema_version: PHYSGOV_SCHEMA_VERSION,
        base_authorized_gear: base.authorized_gear,
        base_ceiling_gear: base.ceiling_gear,
        base_binding_gate: base.binding_gate,
        base_gate_verdicts: base.gate_verdicts,
        emergency_flags: base.emergency_flags,
        requested_gear: base_inputs.requested_gear,
        current_gear: base_inputs.current_gear,
        next_stable_dwell_steps: base.next_stable_dwell_steps,
        next_steps_since_instability: base.next_steps_since_instability,
        downshift_latency_steps: span,
        decision_emitted: 1,
        ext_armed: ext_policy.armed,
        policy_digest: base_inputs.policy_digest,
        ext_policy_digest: ext_policy.ext_policy_digest,
        ..PhysGovDecision::default()
    };

    // CHUNK-BOUNDARY CONTRACT: the named, emitted latency bound.
    out.downshift_latency_fs = if (cur as usize) < base_inputs.gear_dt_fs.len()
        && base_inputs.gear_dt_fs[cur as usize].is_finite()
    {
        span as f32 * base_inputs.gear_dt_fs[cur as usize]
    } else {
        f32::NAN
    };

    // Base emergency is terminal. Nothing below can lift it.
    //
    // ABORT SENTINEL PLACEMENT. `authorized_gear` carries the abort; the
    // ceiling does NOT. [`PhysGovDecision::ceiling_gear`] is documented as the
    // operand of `min(request, ceiling)`, and [`GEAR_ABORT`] is the LARGEST
    // value in that field — the one arithmetic that erases it
    // (`min(1, 3) == 1` would keep integrating at 2.0 fs straight through a
    // physical emergency). The ceiling is therefore pinned to the FINEST gear
    // on an abort: a consumer that honours the trap traps, and a consumer that
    // only knows how to `min` is driven to the most conservative timestep
    // instead of the least.
    if base.authorized_gear == GEAR_ABORT || base.binding_gate == BASE_GATE_PHYSICAL_EMERGENCY {
        out.authorized_gear = GEAR_ABORT;
        out.ceiling_gear = GEAR_FINE;
        out.ext_binding_gate = GATE_NONE;
        out.binding_gate_rank = RANK_PHYSICAL_EMERGENCY;
        out.transition = TRANSITION_ABORT;
        out.dt_applied_ps = f32::NAN;
        return out;
    }

    // EXTENSION BINDING GATE — the extension analogue of base gate 11.
    //
    // Runs BEFORE any guard. A zeroed or wrong-version policy makes every
    // threshold below meaningless: `force_rate_floor_per_ps == 0` returns FLOOR
    // for any positive force rate, `accel_ok_fraction == 0` makes no gear OK,
    // `clamped_fraction_floor == 0` FLOORs on any clipping — the emitted record
    // would be indistinguishable from genuinely violent physics, and the only
    // tell (`ext_armed == 0`) was carried but never acted on. Refusing here is
    // what stops "nobody armed the extension" from being reported as a physics
    // finding.
    //
    // Fail CLOSED: the cap is [`DEFAULT_UNARMED_CAP_GEAR`], a compile-time
    // constant, not a value read out of the policy that could not be trusted.
    // It never deadlocks — gear 0 is always reachable — and the cause is named
    // in [`PhysGovDecision::ext_binding_refusal`].
    let refusal = if ext_policy.armed == 0 {
        REFUSAL_UNARMED
    } else if ext_policy.schema_version != PHYSGOV_SCHEMA_VERSION {
        REFUSAL_POLICY_SCHEMA
    } else if ext_obs.schema_version != PHYSGOV_SCHEMA_VERSION {
        REFUSAL_OBSERVATION_SCHEMA
    } else {
        REFUSAL_NONE
    };
    if refusal != REFUSAL_NONE {
        for gate in GATE_EXT_FIRST..=GATE_EXT_LAST {
            set_ext_verdict(&mut out.ext_gate_verdicts, gate, VERDICT_UNMEASURED);
        }
        out.ext_binding_refusal = refusal;
        out.ceiling_gear = base.ceiling_gear.min(DEFAULT_UNARMED_CAP_GEAR);
        out.authorized_gear = base.authorized_gear.min(DEFAULT_UNARMED_CAP_GEAR);
        out.ext_binding_gate = GATE_EXT_BINDING;
        out.binding_gate_rank = RANK_GOVERNOR_UNARMED;
        out.transition = if out.authorized_gear < cur {
            TRANSITION_DOWNSHIFT
        } else if out.authorized_gear > cur {
            TRANSITION_UPSHIFT
        } else {
            TRANSITION_HOLD
        };
        // No guard was evaluated, so no dwell was earned.
        out.next_stable_dwell_steps = 0;
        out.dt_applied_ps = base_inputs
            .gear_dt_ps(out.authorized_gear)
            .unwrap_or(f32::NAN);
        return out;
    }

    let mut acc = GuardAccum::clean();

    // Extension gates, evaluated in authority order.
    let mut ext: [(u32, u32); CEILING_GATE_COUNT] =
        [(GATE_NONE, GEAR_MAX_ACTIVE); CEILING_GATE_COUNT];

    let (c, v) = gate_hmr_hard(ext_obs);
    ext[0] = (GATE_HMR_HARD, c);
    set_ext_verdict(&mut out.ext_gate_verdicts, GATE_HMR_HARD, v);

    let (c, v) = gate_force_spike(ext_policy, base_inputs, ext_obs, cur, &mut acc);
    ext[1] = (GATE_FORCE_SPIKE, c);
    set_ext_verdict(&mut out.ext_gate_verdicts, GATE_FORCE_SPIKE, v);

    let (c, v) = gate_acceleration(ext_policy, base_inputs, ext_obs, cur, &mut acc);
    ext[2] = (GATE_ACCELERATION_EXTREME, c);
    set_ext_verdict(&mut out.ext_gate_verdicts, GATE_ACCELERATION_EXTREME, v);

    let (c, v) = gate_thermostat(ext_policy, base_inputs, ext_obs, cur, &mut acc);
    ext[3] = (GATE_THERMOSTAT_STABILITY, c);
    set_ext_verdict(&mut out.ext_gate_verdicts, GATE_THERMOSTAT_STABILITY, v);

    let (c, v) = gate_integration_health(ext_policy, ext_obs, cur, &mut acc);
    ext[4] = (GATE_INTEGRATION_HEALTH, c);
    set_ext_verdict(&mut out.ext_gate_verdicts, GATE_INTEGRATION_HEALTH, v);

    let (c, v) = gate_tsh_memory(ext_policy, ext_obs, cur, &mut acc);
    ext[5] = (GATE_TSH_MEMORY, c);
    set_ext_verdict(&mut out.ext_gate_verdicts, GATE_TSH_MEMORY, v);

    let (c, v) = gate_phase(ext_policy, ext_obs, cur);
    ext[6] = (GATE_PHASE_REQUIREMENT, c);
    set_ext_verdict(&mut out.ext_gate_verdicts, GATE_PHASE_REQUIREMENT, v);

    out.ext_emergency_flags = acc.ext_emergency;

    // Extension emergency: terminal, same class as base gate 1.
    //
    // Same abort-sentinel placement as the base emergency above: the ceiling is
    // pinned FINE, never GEAR_ABORT, so `min(request, ceiling)` cannot erase
    // the trap.
    if acc.ext_emergency != 0 {
        out.authorized_gear = GEAR_ABORT;
        out.ceiling_gear = GEAR_FINE;
        out.ext_binding_gate = if acc.ext_emergency & EMERG_INTEGRATION_FAULT != 0 {
            GATE_INTEGRATION_HEALTH
        } else if acc.ext_emergency & EMERG_TEMPERATURE_RUNAWAY != 0 {
            GATE_THERMOSTAT_STABILITY
        } else {
            GATE_ACCELERATION_EXTREME
        };
        out.binding_gate_rank = RANK_PHYSICAL_EMERGENCY;
        out.transition = TRANSITION_ABORT;
        out.dt_applied_ps = f32::NAN;
        out.next_stable_dwell_steps = 0;
        out.next_steps_since_instability = 0;
        return out;
    }

    let ext_min = ext.iter().fold(GEAR_MAX_ACTIVE, |m, (_, c)| m.min(*c));

    // THE SINGLE TIGHTEN-ONLY FOLD.
    out.ceiling_gear = base.ceiling_gear.min(ext_min);
    let mut authorized = base.authorized_gear.min(ext_min);

    // HARD PHYSICS GATE, re-asserted unconditionally (requirement 3). Applied
    // last and outside every policy switch: even if every band above were
    // mis-tuned, gear 2 is unreachable without verified HMR.
    if ext_obs.hmr_applied_verified != PERMISSION_GRANTED {
        authorized = authorized.min(GEAR_SAFE);
        out.ceiling_gear = out.ceiling_gear.min(GEAR_SAFE);
    }

    // ASYMMETRY (requirement 4). The base already applied dwell / lockout /
    // upshift-rate to its own authorized gear; because this layer only
    // tightens, an upshift that survives here is one the base licensed, so the
    // asymmetry holds transitively. The one addition: extension guard
    // degradation observed in THIS decision also vetoes an upshift the base
    // licensed on the base guards alone.
    if authorized > cur && !acc.all_guards_ok {
        authorized = cur;
    }
    out.authorized_gear = authorized;

    // Attribution: the most authoritative gate at the binding ceiling.
    let mut best_rank = RANK_NONE;
    let mut best_ext = GATE_NONE;
    for (gate, ceiling) in ext.iter() {
        if *ceiling != out.ceiling_gear || *ceiling >= GEAR_MAX_ACTIVE {
            continue;
        }
        let r = gate_rank(*gate);
        if r < best_rank {
            best_rank = r;
            best_ext = *gate;
        }
    }
    let base_rank = if base.ceiling_gear == out.ceiling_gear && base.ceiling_gear < GEAR_MAX_ACTIVE
    {
        base_gate_rank(base.binding_gate)
    } else {
        RANK_NONE
    };
    if base_rank <= best_rank {
        out.ext_binding_gate = GATE_NONE;
        out.binding_gate_rank = base_rank;
    } else {
        out.ext_binding_gate = best_ext;
        out.binding_gate_rank = best_rank;
    }
    // A 4 fs refusal must never read as a thermostat or phase finding.
    if ext_obs.hmr_applied_verified != PERMISSION_GRANTED
        && out.ceiling_gear == GEAR_SAFE
        && out.binding_gate_rank > RANK_HMR_HARD
    {
        out.ext_binding_gate = GATE_HMR_HARD;
        out.binding_gate_rank = RANK_HMR_HARD;
    }

    // Transition classification, emitted even when nothing changed.
    out.transition = if out.authorized_gear < cur {
        TRANSITION_DOWNSHIFT
    } else if out.authorized_gear > cur {
        TRANSITION_UPSHIFT
    } else {
        TRANSITION_HOLD
    };

    // Dwell / instability bookkeeping for the next decision.
    if acc.instability {
        out.next_stable_dwell_steps = 0;
        out.next_steps_since_instability = 0;
    } else if !acc.all_guards_ok {
        out.next_stable_dwell_steps = 0;
    }
    // A gear change itself restarts the physical dwell — evidence that gear A
    // was stable is not evidence that gear B is. The base applies this to its
    // OWN authorized gear (`gearbox.cuh:1050`); it must also apply to the
    // COMPOSED gear, or an extension-driven downshift carries forward dwell
    // earned at the coarser gear and licenses an upshift with zero stable steps
    // actually spent at the finer one — an upshift on evidence from a gear the
    // run is no longer in.
    if out.authorized_gear != cur {
        out.next_stable_dwell_steps = 0;
    }

    out.dt_applied_ps = base_inputs
        .gear_dt_ps(out.authorized_gear)
        .unwrap_or(f32::NAN);

    out
}

/// Run the canonical base governor (`prism_gearbox_governor_decide`, the same
/// C function the SFA kernel calls) and compose the extension guards on top.
///
/// This is the production host entry point. It performs no CUDA work — the base
/// decision function is pure host computation — so it is usable in dry-run
/// policy validation as well as in the live path.
///
/// # Errors
///
/// Returns the base FFI's non-zero return code when the base decision function
/// refuses the inputs (null pointer or schema mismatch).
#[cfg(feature = "gpu")]
pub fn decide_with_base_governor(
    base_policy: &crate::gearbox::GearGovernorPolicy,
    base_obs: &crate::gearbox::GearGuardObservation,
    salience: Option<&crate::gearbox::TshSalienceRequest>,
    ext_policy: &ExtPolicy,
    ext_obs: &ExtObservation,
) -> Result<PhysGovDecision, i32> {
    let base = crate::gearbox::governor_decide(base_policy, base_obs, salience)?;
    let inputs = BaseInputs::from_gearbox(base_policy, base_obs);
    // SAFETY-FREE: `GearDecision` is a plain `#[repr(C)]` POD; the field-by-field
    // copy below avoids a transmute so a future layout change fails to compile
    // instead of silently misreading.
    let base_decision = BaseDecision {
        authorized_gear: base.authorized_gear,
        ceiling_gear: base.ceiling_gear,
        binding_gate: base.binding_gate,
        gate_verdicts: base.gate_verdicts,
        emergency_flags: base.emergency_flags,
        next_stable_dwell_steps: base.next_stable_dwell_steps,
        next_steps_since_instability: base.next_steps_since_instability,
        downshift_latency_steps: base.downshift_latency_steps,
    };
    Ok(compose(&inputs, ext_policy, ext_obs, &base_decision))
}

// ═══════════════════════════════════════════════════════════════════════════
// Policy resolution and provenance
// ═══════════════════════════════════════════════════════════════════════════

/// Where a resolved threshold came from.
///
/// The numbers alone are not auditable: `gamma_dt_ok = 0.05` does not say
/// whether that came from the documented source literal or from an operator
/// override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueOrigin {
    /// The documented source default: `physical_governor.cuh` for the
    /// device-wire [`ExtPolicy`] rows, and this module's own `DEFAULT_*`
    /// constant for host-only rows such as [`StreamTopologyPolicy`].
    Default,
    /// A versioned policy file.
    PolicyFile,
    /// A process-environment override.
    ///
    /// [`resolve_ext_policy`] never produces this variant: the device-wire
    /// policy has no environment layer, so [`ExtPolicy::ext_policy_digest`]
    /// is unchanged by its existence. It is produced only by
    /// [`resolve_stream_topology`], where an operator raising stream capacity
    /// for one run must be distinguishable from one who edited the policy
    /// file for every run.
    Environment,
    /// A command-line override.
    Cli,
}

impl ValueOrigin {
    /// Stable string for receipts.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::PolicyFile => "policy",
            Self::Environment => "env",
            Self::Cli => "cli",
        }
    }
}

/// Failure modes of extension-policy resolution.
#[derive(Debug, thiserror::Error)]
pub enum ExtPolicyError {
    /// The policy file could not be read.
    #[error("physical governor policy file {path} could not be read: {source}")]
    Read {
        /// The path that failed.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The policy file is not a valid document.
    #[error("physical governor policy file {path} is not a valid policy document: {message}")]
    Parse {
        /// The path that failed.
        path: String,
        /// The parser's message.
        message: String,
    },
    /// The policy file's sha256 does not match the pinned digest.
    #[error(
        "physical governor policy file {path} sha256 {actual} does not match pinned {expected}"
    )]
    DigestMismatch {
        /// The path that failed.
        path: String,
        /// The digest that was computed.
        actual: String,
        /// The digest that was pinned.
        expected: String,
    },
    /// The document declares an unknown schema version.
    #[error(
        "physical governor policy file {path} declares schema_version {found}, expected {expected}"
    )]
    SchemaMismatch {
        /// The path that failed.
        path: String,
        /// What the document said.
        found: u32,
        /// What this build understands.
        expected: u32,
    },
    /// A resolved field is out of range or breaks band monotonicity.
    #[error("physical governor policy field '{field}' is invalid: {reason}")]
    InvalidValue {
        /// Which field.
        field: &'static str,
        /// Why it was rejected.
        reason: String,
    },
}

/// The on-disk policy document.
///
/// `deny_unknown_fields` makes a document written for a different governor
/// version fail loudly instead of having its extra keys silently ignored.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExtPolicyDocument {
    /// Must equal [`PHYSGOV_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Override for [`ExtPolicy::require_accel_guard_for_upshift`].
    #[serde(default)]
    pub require_accel_guard_for_upshift: Option<bool>,
    /// Override for [`ExtPolicy::require_thermostat_guard_for_upshift`].
    #[serde(default)]
    pub require_thermostat_guard_for_upshift: Option<bool>,
    /// Override for [`ExtPolicy::require_integration_guard_for_upshift`].
    #[serde(default)]
    pub require_integration_guard_for_upshift: Option<bool>,
    /// Override for [`ExtPolicy::require_observation_confidence`].
    #[serde(default)]
    pub require_observation_confidence: Option<bool>,
    /// Override for [`ExtPolicy::max_steps_without_sentinel`].
    #[serde(default)]
    pub max_steps_without_sentinel: Option<u32>,
    /// Override for [`ExtPolicy::force_rate_hold_per_ps`].
    #[serde(default)]
    pub force_rate_hold_per_ps: Option<f32>,
    /// Override for [`ExtPolicy::force_rate_downshift_per_ps`].
    #[serde(default)]
    pub force_rate_downshift_per_ps: Option<f32>,
    /// Override for [`ExtPolicy::force_rate_floor_per_ps`].
    #[serde(default)]
    pub force_rate_floor_per_ps: Option<f32>,
    /// Override for [`ExtPolicy::force_rate_norm_floor`].
    #[serde(default)]
    pub force_rate_norm_floor: Option<f32>,
    /// Override for [`ExtPolicy::accel_displacement_reference_a`].
    #[serde(default)]
    pub accel_displacement_reference_a: Option<f32>,
    /// Override for [`ExtPolicy::accel_ok_fraction`].
    #[serde(default)]
    pub accel_ok_fraction: Option<f32>,
    /// Override for [`ExtPolicy::accel_hold_fraction`].
    #[serde(default)]
    pub accel_hold_fraction: Option<f32>,
    /// Override for [`ExtPolicy::accel_downshift_fraction`].
    #[serde(default)]
    pub accel_downshift_fraction: Option<f32>,
    /// Override for [`ExtPolicy::accel_abort_fraction`].
    #[serde(default)]
    pub accel_abort_fraction: Option<f32>,
    /// Override for [`ExtPolicy::gamma_dt_ok`].
    #[serde(default)]
    pub gamma_dt_ok: Option<f32>,
    /// Override for [`ExtPolicy::gamma_dt_hold`].
    #[serde(default)]
    pub gamma_dt_hold: Option<f32>,
    /// Override for [`ExtPolicy::gamma_dt_downshift`].
    #[serde(default)]
    pub gamma_dt_downshift: Option<f32>,
    /// Override for [`ExtPolicy::gamma_dt_abort`].
    #[serde(default)]
    pub gamma_dt_abort: Option<f32>,
    /// Override for [`ExtPolicy::temp_dev_ok_fraction`].
    #[serde(default)]
    pub temp_dev_ok_fraction: Option<f32>,
    /// Override for [`ExtPolicy::temp_dev_hold_fraction`].
    #[serde(default)]
    pub temp_dev_hold_fraction: Option<f32>,
    /// Override for [`ExtPolicy::temp_dev_downshift_fraction`].
    #[serde(default)]
    pub temp_dev_downshift_fraction: Option<f32>,
    /// Override for [`ExtPolicy::temp_dev_abort_fraction`].
    #[serde(default)]
    pub temp_dev_abort_fraction: Option<f32>,
    /// Override for [`ExtPolicy::neighbor_occupancy_ok`].
    #[serde(default)]
    pub neighbor_occupancy_ok: Option<f32>,
    /// Override for [`ExtPolicy::neighbor_occupancy_hold`].
    #[serde(default)]
    pub neighbor_occupancy_hold: Option<f32>,
    /// Override for [`ExtPolicy::clamped_fraction_hold`].
    #[serde(default)]
    pub clamped_fraction_hold: Option<f32>,
    /// Override for [`ExtPolicy::clamped_fraction_downshift`].
    #[serde(default)]
    pub clamped_fraction_downshift: Option<f32>,
    /// Override for [`ExtPolicy::clamped_fraction_floor`].
    #[serde(default)]
    pub clamped_fraction_floor: Option<f32>,
    /// Override for [`ExtPolicy::tsh_memory_hold`].
    #[serde(default)]
    pub tsh_memory_hold: Option<f32>,
    /// Override for [`ExtPolicy::tsh_memory_floor`].
    #[serde(default)]
    pub tsh_memory_floor: Option<f32>,
    /// Override for [`ExtPolicy::min_observation_confidence`].
    #[serde(default)]
    pub min_observation_confidence: Option<f32>,
    /// Override for [`ExtPolicy::max_observation_interval_ps`].
    #[serde(default)]
    pub max_observation_interval_ps: Option<f32>,
    /// Override for [`ExtPolicy::phase_ceiling_gear`].
    #[serde(default)]
    pub phase_ceiling_gear: Option<Vec<u32>>,
}

/// A resolved extension policy together with the provenance of every value.
#[derive(Debug, Clone)]
pub struct ExtPolicyResolution {
    /// The resolved policy, ready to upload. `armed` is set to 1.
    pub policy: ExtPolicy,
    /// `(field_name, origin)` for every governed field, in declaration order.
    pub origins: Vec<(&'static str, ValueOrigin)>,
    /// Path of the policy file, when one was used.
    pub policy_path: Option<String>,
    /// sha256 of the policy file, when one was used.
    pub policy_sha256: Option<String>,
    /// Fingerprint of the resolved policy; echoed by every decision.
    pub digest: u64,
}

impl ExtPolicyResolution {
    /// Origin of a single field, for targeted evidence lines.
    #[must_use]
    pub fn origin_of(&self, field: &str) -> Option<ValueOrigin> {
        self.origins
            .iter()
            .find(|(name, _)| *name == field)
            .map(|(_, origin)| *origin)
    }

    /// A receipt line per governed field: `field=value origin=...`.
    ///
    /// This is what makes "no hardcoded scientific thresholds" auditable rather
    /// than merely asserted.
    #[must_use]
    pub fn receipt_lines(&self) -> Vec<String> {
        let p = &self.policy;
        let f = |name: &'static str, value: String| -> String {
            format!(
                "{name}={value} origin={}",
                self.origin_of(name).map_or("unknown", ValueOrigin::as_str)
            )
        };
        vec![
            f(
                "force_rate_hold_per_ps",
                p.force_rate_hold_per_ps.to_string(),
            ),
            f(
                "force_rate_downshift_per_ps",
                p.force_rate_downshift_per_ps.to_string(),
            ),
            f(
                "force_rate_floor_per_ps",
                p.force_rate_floor_per_ps.to_string(),
            ),
            f("force_rate_norm_floor", p.force_rate_norm_floor.to_string()),
            f(
                "accel_displacement_reference_a",
                p.accel_displacement_reference_a.to_string(),
            ),
            f("accel_ok_fraction", p.accel_ok_fraction.to_string()),
            f("accel_hold_fraction", p.accel_hold_fraction.to_string()),
            f(
                "accel_downshift_fraction",
                p.accel_downshift_fraction.to_string(),
            ),
            f("accel_abort_fraction", p.accel_abort_fraction.to_string()),
            f("gamma_dt_ok", p.gamma_dt_ok.to_string()),
            f("gamma_dt_hold", p.gamma_dt_hold.to_string()),
            f("gamma_dt_downshift", p.gamma_dt_downshift.to_string()),
            f("gamma_dt_abort", p.gamma_dt_abort.to_string()),
            f("temp_dev_ok_fraction", p.temp_dev_ok_fraction.to_string()),
            f(
                "temp_dev_hold_fraction",
                p.temp_dev_hold_fraction.to_string(),
            ),
            f(
                "temp_dev_downshift_fraction",
                p.temp_dev_downshift_fraction.to_string(),
            ),
            f(
                "temp_dev_abort_fraction",
                p.temp_dev_abort_fraction.to_string(),
            ),
            f("neighbor_occupancy_ok", p.neighbor_occupancy_ok.to_string()),
            f(
                "neighbor_occupancy_hold",
                p.neighbor_occupancy_hold.to_string(),
            ),
            f("clamped_fraction_hold", p.clamped_fraction_hold.to_string()),
            f(
                "clamped_fraction_downshift",
                p.clamped_fraction_downshift.to_string(),
            ),
            f(
                "clamped_fraction_floor",
                p.clamped_fraction_floor.to_string(),
            ),
            f("tsh_memory_hold", p.tsh_memory_hold.to_string()),
            f("tsh_memory_floor", p.tsh_memory_floor.to_string()),
            f(
                "min_observation_confidence",
                p.min_observation_confidence.to_string(),
            ),
            f(
                "max_observation_interval_ps",
                p.max_observation_interval_ps.to_string(),
            ),
            f(
                "max_steps_without_sentinel",
                p.max_steps_without_sentinel.to_string(),
            ),
            f(
                "require_accel_guard_for_upshift",
                p.require_accel_guard_for_upshift.to_string(),
            ),
            f(
                "require_thermostat_guard_for_upshift",
                p.require_thermostat_guard_for_upshift.to_string(),
            ),
            f(
                "require_integration_guard_for_upshift",
                p.require_integration_guard_for_upshift.to_string(),
            ),
            f(
                "require_observation_confidence",
                p.require_observation_confidence.to_string(),
            ),
            f("phase_ceiling_gear", format!("{:?}", p.phase_ceiling_gear)),
            f("n_phase_slots", p.n_phase_slots.to_string()),
        ]
    }

    /// Every field named in [`Self::receipt_lines`], for the mechanical check
    /// that no governed threshold is missing a receipt and no receipt names a
    /// field that is not governed.
    ///
    /// A receipt line for a value that no comparison ever reads is a FALSE
    /// RECEIPT: it hands the operator provenance evidence for a number that
    /// governs nothing, which is the same fabrication class as a hardcoded
    /// gear id. Everything listed here is read by
    /// [`validate_ext_policy`] and by at least one gate.
    #[must_use]
    pub fn receipted_fields() -> &'static [&'static str] {
        &[
            "force_rate_hold_per_ps",
            "force_rate_downshift_per_ps",
            "force_rate_floor_per_ps",
            "force_rate_norm_floor",
            "accel_displacement_reference_a",
            "accel_ok_fraction",
            "accel_hold_fraction",
            "accel_downshift_fraction",
            "accel_abort_fraction",
            "gamma_dt_ok",
            "gamma_dt_hold",
            "gamma_dt_downshift",
            "gamma_dt_abort",
            "temp_dev_ok_fraction",
            "temp_dev_hold_fraction",
            "temp_dev_downshift_fraction",
            "temp_dev_abort_fraction",
            "neighbor_occupancy_ok",
            "neighbor_occupancy_hold",
            "clamped_fraction_hold",
            "clamped_fraction_downshift",
            "clamped_fraction_floor",
            "tsh_memory_hold",
            "tsh_memory_floor",
            "min_observation_confidence",
            "max_observation_interval_ps",
            "max_steps_without_sentinel",
            "require_accel_guard_for_upshift",
            "require_thermostat_guard_for_upshift",
            "require_integration_guard_for_upshift",
            "require_observation_confidence",
            "phase_ceiling_gear",
            "n_phase_slots",
        ]
    }
}

/// Resolve an extension policy: documented defaults, overlaid by an optional
/// versioned policy file, with every field's origin recorded.
///
/// `expected_sha256`, when supplied, pins the policy file — a policy that
/// silently changed under the run is exactly the provenance failure this whole
/// package exists to prevent.
///
/// # Errors
///
/// [`ExtPolicyError`] on unreadable / unparseable / mis-versioned / mis-digested
/// files, or when a resolved value breaks band monotonicity.
pub fn resolve_ext_policy(
    policy_path: Option<&str>,
    expected_sha256: Option<&str>,
) -> Result<ExtPolicyResolution, ExtPolicyError> {
    let mut policy = ExtPolicy::default();
    let mut origins: Vec<(&'static str, ValueOrigin)> = Vec::new();
    let mut sha_hex: Option<String> = None;

    let document: Option<ExtPolicyDocument> = match policy_path {
        None => None,
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|source| ExtPolicyError::Read {
                path: path.to_string(),
                source,
            })?;
            let digest = {
                use sha2::Digest as _;
                let mut h = sha2::Sha256::new();
                h.update(&bytes);
                h.finalize()
            };
            let hex = digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            if let Some(expected) = expected_sha256 {
                if !expected.eq_ignore_ascii_case(&hex) {
                    return Err(ExtPolicyError::DigestMismatch {
                        path: path.to_string(),
                        actual: hex,
                        expected: expected.to_string(),
                    });
                }
            }
            sha_hex = Some(hex);
            let doc: ExtPolicyDocument =
                serde_json::from_slice(&bytes).map_err(|e| ExtPolicyError::Parse {
                    path: path.to_string(),
                    message: e.to_string(),
                })?;
            if doc.schema_version != PHYSGOV_SCHEMA_VERSION {
                return Err(ExtPolicyError::SchemaMismatch {
                    path: path.to_string(),
                    found: doc.schema_version,
                    expected: PHYSGOV_SCHEMA_VERSION,
                });
            }
            Some(doc)
        }
    };

    macro_rules! overlay_f32 {
        ($field:ident) => {{
            let origin = match document.as_ref().and_then(|d| d.$field) {
                Some(v) => {
                    policy.$field = v;
                    ValueOrigin::PolicyFile
                }
                None => ValueOrigin::Default,
            };
            origins.push((stringify!($field), origin));
        }};
    }
    macro_rules! overlay_u32 {
        ($field:ident) => {{
            let origin = match document.as_ref().and_then(|d| d.$field) {
                Some(v) => {
                    policy.$field = v;
                    ValueOrigin::PolicyFile
                }
                None => ValueOrigin::Default,
            };
            origins.push((stringify!($field), origin));
        }};
    }
    macro_rules! overlay_bool {
        ($field:ident) => {{
            let origin = match document.as_ref().and_then(|d| d.$field) {
                Some(v) => {
                    policy.$field = u32::from(v);
                    ValueOrigin::PolicyFile
                }
                None => ValueOrigin::Default,
            };
            origins.push((stringify!($field), origin));
        }};
    }

    overlay_f32!(force_rate_hold_per_ps);
    overlay_f32!(force_rate_downshift_per_ps);
    overlay_f32!(force_rate_floor_per_ps);
    overlay_f32!(force_rate_norm_floor);
    overlay_f32!(accel_displacement_reference_a);
    overlay_f32!(accel_ok_fraction);
    overlay_f32!(accel_hold_fraction);
    overlay_f32!(accel_downshift_fraction);
    overlay_f32!(accel_abort_fraction);
    overlay_f32!(gamma_dt_ok);
    overlay_f32!(gamma_dt_hold);
    overlay_f32!(gamma_dt_downshift);
    overlay_f32!(gamma_dt_abort);
    overlay_f32!(temp_dev_ok_fraction);
    overlay_f32!(temp_dev_hold_fraction);
    overlay_f32!(temp_dev_downshift_fraction);
    overlay_f32!(temp_dev_abort_fraction);
    overlay_f32!(neighbor_occupancy_ok);
    overlay_f32!(neighbor_occupancy_hold);
    overlay_f32!(clamped_fraction_hold);
    overlay_f32!(clamped_fraction_downshift);
    overlay_f32!(clamped_fraction_floor);
    overlay_f32!(tsh_memory_hold);
    overlay_f32!(tsh_memory_floor);
    overlay_f32!(min_observation_confidence);
    overlay_f32!(max_observation_interval_ps);
    overlay_u32!(max_steps_without_sentinel);
    overlay_bool!(require_accel_guard_for_upshift);
    overlay_bool!(require_thermostat_guard_for_upshift);
    overlay_bool!(require_integration_guard_for_upshift);
    overlay_bool!(require_observation_confidence);

    {
        let origin = match document
            .as_ref()
            .and_then(|d| d.phase_ceiling_gear.as_ref())
        {
            Some(v) => {
                if v.len() > 8 {
                    return Err(ExtPolicyError::InvalidValue {
                        field: "phase_ceiling_gear",
                        reason: format!("at most 8 slots, got {}", v.len()),
                    });
                }
                if v.is_empty() {
                    // An empty vector sets `n_phase_slots = 0`, and `gate_phase`
                    // then returns UNMEASURED for EVERY measured phase and pins
                    // the ceiling to the current gear forever — a permanent
                    // upshift lock with no named terminal error. Refuse it at
                    // resolution time, where the operator can still see why.
                    return Err(ExtPolicyError::InvalidValue {
                        field: "phase_ceiling_gear",
                        reason: "must declare at least one phase slot; an empty table \
                                 makes every measured phase unknown and locks the gear \
                                 at the current one for the life of the run"
                            .to_string(),
                    });
                }
                for (i, g) in v.iter().enumerate() {
                    if *g > GEAR_MAX_ACTIVE {
                        return Err(ExtPolicyError::InvalidValue {
                            field: "phase_ceiling_gear",
                            reason: format!("slot {i} is gear {g}; must be <= {GEAR_MAX_ACTIVE}"),
                        });
                    }
                    policy.phase_ceiling_gear[i] = *g;
                }
                policy.n_phase_slots = v.len() as u32;
                ValueOrigin::PolicyFile
            }
            None => ValueOrigin::Default,
        };
        origins.push(("phase_ceiling_gear", origin));
        // `n_phase_slots` is derived from the same document key, so it carries
        // the same origin. It contributes to the digest and it decides whether
        // `gate_phase` can resolve a phase at all, so it needs a recorded
        // provenance of its own rather than being an unnamed side effect.
        origins.push(("n_phase_slots", origin));
    }

    validate_ext_policy(&policy)?;

    policy.armed = 1;
    policy.ext_policy_digest = ext_policy_digest(&policy, &origins, sha_hex.as_deref());

    Ok(ExtPolicyResolution {
        policy,
        origins,
        policy_path: policy_path.map(str::to_string),
        policy_sha256: sha_hex,
        digest: policy.ext_policy_digest,
    })
}

/// Reject a policy whose bands are non-monotone or out of range.
///
/// A HOLD threshold above a DOWNSHIFT threshold makes a band unreachable in the
/// order it is tested — the guard would silently stop firing.
///
/// # Errors
///
/// [`ExtPolicyError::InvalidValue`] naming the offending field.
pub fn validate_ext_policy(p: &ExtPolicy) -> Result<(), ExtPolicyError> {
    let finite = |name: &'static str, v: f32| -> Result<(), ExtPolicyError> {
        if v.is_finite() {
            Ok(())
        } else {
            Err(ExtPolicyError::InvalidValue {
                field: name,
                reason: format!("must be finite, got {v}"),
            })
        }
    };
    let monotone = |name: &'static str,
                    lo: f32,
                    hi: f32,
                    lo_n: &str,
                    hi_n: &str|
     -> Result<(), ExtPolicyError> {
        if lo < hi {
            Ok(())
        } else {
            Err(ExtPolicyError::InvalidValue {
                field: name,
                reason: format!("{lo_n} ({lo}) must be strictly below {hi_n} ({hi})"),
            })
        }
    };

    finite("force_rate_hold_per_ps", p.force_rate_hold_per_ps)?;
    finite("force_rate_downshift_per_ps", p.force_rate_downshift_per_ps)?;
    finite("force_rate_floor_per_ps", p.force_rate_floor_per_ps)?;
    monotone(
        "force_rate_downshift_per_ps",
        p.force_rate_hold_per_ps,
        p.force_rate_downshift_per_ps,
        "hold",
        "downshift",
    )?;
    monotone(
        "force_rate_floor_per_ps",
        p.force_rate_downshift_per_ps,
        p.force_rate_floor_per_ps,
        "downshift",
        "floor",
    )?;
    if !(p.force_rate_norm_floor.is_finite() && p.force_rate_norm_floor > 0.0) {
        return Err(ExtPolicyError::InvalidValue {
            field: "force_rate_norm_floor",
            reason: format!("must be finite and > 0, got {}", p.force_rate_norm_floor),
        });
    }

    if !(p.accel_displacement_reference_a.is_finite() && p.accel_displacement_reference_a > 0.0) {
        return Err(ExtPolicyError::InvalidValue {
            field: "accel_displacement_reference_a",
            reason: format!(
                "must be finite and > 0 Angstrom, got {}",
                p.accel_displacement_reference_a
            ),
        });
    }
    monotone(
        "accel_hold_fraction",
        p.accel_ok_fraction,
        p.accel_hold_fraction,
        "ok",
        "hold",
    )?;
    monotone(
        "accel_downshift_fraction",
        p.accel_hold_fraction,
        p.accel_downshift_fraction,
        "hold",
        "downshift",
    )?;
    monotone(
        "accel_abort_fraction",
        p.accel_downshift_fraction,
        p.accel_abort_fraction,
        "downshift",
        "abort",
    )?;

    monotone(
        "gamma_dt_hold",
        p.gamma_dt_ok,
        p.gamma_dt_hold,
        "ok",
        "hold",
    )?;
    monotone(
        "gamma_dt_downshift",
        p.gamma_dt_hold,
        p.gamma_dt_downshift,
        "hold",
        "downshift",
    )?;
    monotone(
        "gamma_dt_abort",
        p.gamma_dt_downshift,
        p.gamma_dt_abort,
        "downshift",
        "abort",
    )?;
    monotone(
        "temp_dev_hold_fraction",
        p.temp_dev_ok_fraction,
        p.temp_dev_hold_fraction,
        "ok",
        "hold",
    )?;
    monotone(
        "temp_dev_downshift_fraction",
        p.temp_dev_hold_fraction,
        p.temp_dev_downshift_fraction,
        "hold",
        "downshift",
    )?;
    monotone(
        "temp_dev_abort_fraction",
        p.temp_dev_downshift_fraction,
        p.temp_dev_abort_fraction,
        "downshift",
        "abort",
    )?;
    monotone(
        "neighbor_occupancy_hold",
        p.neighbor_occupancy_ok,
        p.neighbor_occupancy_hold,
        "ok",
        "hold",
    )?;
    monotone(
        "clamped_fraction_downshift",
        p.clamped_fraction_hold,
        p.clamped_fraction_downshift,
        "hold",
        "downshift",
    )?;
    monotone(
        "clamped_fraction_floor",
        p.clamped_fraction_downshift,
        p.clamped_fraction_floor,
        "downshift",
        "floor",
    )?;
    monotone(
        "tsh_memory_floor",
        p.tsh_memory_hold,
        p.tsh_memory_floor,
        "hold",
        "floor",
    )?;

    if !(0.0..=1.0).contains(&p.min_observation_confidence)
        || !p.min_observation_confidence.is_finite()
    {
        return Err(ExtPolicyError::InvalidValue {
            field: "min_observation_confidence",
            reason: format!("must be in [0,1], got {}", p.min_observation_confidence),
        });
    }
    if !(p.max_observation_interval_ps.is_finite() && p.max_observation_interval_ps > 0.0) {
        return Err(ExtPolicyError::InvalidValue {
            field: "max_observation_interval_ps",
            reason: format!(
                "must be finite and > 0 ps, got {}",
                p.max_observation_interval_ps
            ),
        });
    }
    // Anti-blindness control (b) must not be silently disabled. `0` reads
    // naturally as "force a sentinel every step" — the strongest possible
    // setting — while the evaluator's `!= 0` guard treats it as "the rule is
    // off". A receipt line `max_steps_without_sentinel=0 origin=policy` would
    // therefore document a tightening that is in fact a total disable. There is
    // no "disabled" encoding, on purpose: the rule is not optional.
    if p.max_steps_without_sentinel == 0 {
        return Err(ExtPolicyError::InvalidValue {
            field: "max_steps_without_sentinel",
            reason: "must be >= 1; 0 would disable the forced-sentinel \
                     anti-blindness rule while reading as the strictest setting"
                .to_string(),
        });
    }
    // A zero-slot phase table makes `gate_phase` return UNMEASURED for every
    // measured phase and pins the ceiling to the current gear for the life of
    // the run — a permanent upshift lock with no named terminal error.
    if p.n_phase_slots == 0 || p.n_phase_slots > 8 {
        return Err(ExtPolicyError::InvalidValue {
            field: "n_phase_slots",
            reason: format!(
                "must be in 1..=8, got {}; 0 locks the gear at the current one \
                 for every measured phase",
                p.n_phase_slots
            ),
        });
    }
    for (i, g) in p.phase_ceiling_gear.iter().enumerate() {
        if *g > GEAR_MAX_ACTIVE {
            return Err(ExtPolicyError::InvalidValue {
                field: "phase_ceiling_gear",
                reason: format!("slot {i} is gear {g}; must be <= {GEAR_MAX_ACTIVE}"),
            });
        }
    }
    Ok(())
}

/// Fingerprint of a resolved policy, its per-field ORIGINS, and the policy
/// file's sha256.
///
/// Field order is fixed and every governed value contributes, so two policies
/// that differ anywhere produce different digests. The origins are hashed too:
/// "0.05 because the header documents it" and "0.05 because an operator wrote
/// it into a policy file" are different provenance facts about the same number,
/// and a fingerprint that claims to bind provenance has to be able to tell them
/// apart.
fn ext_policy_digest(
    p: &ExtPolicy,
    origins: &[(&'static str, ValueOrigin)],
    file_sha256: Option<&str>,
) -> u64 {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(PHYSGOV_SCHEMA_VERSION.to_be_bytes());
    for v in [
        p.require_accel_guard_for_upshift,
        p.require_thermostat_guard_for_upshift,
        p.require_integration_guard_for_upshift,
        p.require_observation_confidence,
        p.max_steps_without_sentinel,
        p.n_phase_slots,
    ] {
        hasher.update(v.to_be_bytes());
    }
    for v in [
        p.force_rate_hold_per_ps,
        p.force_rate_downshift_per_ps,
        p.force_rate_floor_per_ps,
        p.force_rate_norm_floor,
        p.accel_displacement_reference_a,
        p.accel_ok_fraction,
        p.accel_hold_fraction,
        p.accel_downshift_fraction,
        p.accel_abort_fraction,
        p.gamma_dt_ok,
        p.gamma_dt_hold,
        p.gamma_dt_downshift,
        p.gamma_dt_abort,
        p.temp_dev_ok_fraction,
        p.temp_dev_hold_fraction,
        p.temp_dev_downshift_fraction,
        p.temp_dev_abort_fraction,
        p.neighbor_occupancy_ok,
        p.neighbor_occupancy_hold,
        p.clamped_fraction_hold,
        p.clamped_fraction_downshift,
        p.clamped_fraction_floor,
        p.tsh_memory_hold,
        p.tsh_memory_floor,
        p.min_observation_confidence,
        p.max_observation_interval_ps,
    ] {
        hasher.update(v.to_be_bytes());
    }
    for g in p.phase_ceiling_gear {
        hasher.update(g.to_be_bytes());
    }
    for (field, origin) in origins {
        hasher.update(field.as_bytes());
        hasher.update([0u8]);
        hasher.update(origin.as_str().as_bytes());
        hasher.update([0u8]);
    }
    hasher.update(file_sha256.unwrap_or("no_policy_file").as_bytes());
    let out = hasher.finalize();
    let mut lead = [0u8; 8];
    lead.copy_from_slice(&out[..8]);
    u64::from_be_bytes(lead)
}

// ═══════════════════════════════════════════════════════════════════════════
// Parity vectors
// ═══════════════════════════════════════════════════════════════════════════

/// One parity case: a named scenario, its extension-only ceiling, and the
/// binding extension gate.
///
/// The device evaluator in `physical_governor.cuh` must produce the same
/// ceiling and binding gate for the same inputs. Replaying this table on both
/// sides is what turns "the two implementations agree" from an assertion into a
/// check.
#[derive(Debug, Clone, Copy)]
pub struct ParityVector {
    /// Human-readable scenario name.
    pub name: &'static str,
    /// Expected composed ceiling gear.
    pub expected_ceiling: u32,
    /// Expected binding extension gate, or [`GATE_NONE`].
    pub expected_ext_gate: u32,
}

/// The canonical parity table. Each entry is constructed by
/// `tests/physical_governor_gates.rs::parity_vectors_hold`.
pub const PARITY_VECTORS: &[ParityVector] = &[
    ParityVector {
        name: "all_clear_hmr_granted",
        expected_ceiling: GEAR_CAMPAIGN,
        expected_ext_gate: GATE_NONE,
    },
    ParityVector {
        name: "hmr_unknown",
        expected_ceiling: GEAR_SAFE,
        expected_ext_gate: GATE_HMR_HARD,
    },
    ParityVector {
        name: "hmr_denied",
        expected_ceiling: GEAR_SAFE,
        expected_ext_gate: GATE_HMR_HARD,
    },
    ParityVector {
        name: "force_spike_floor",
        expected_ceiling: GEAR_FINE,
        expected_ext_gate: GATE_FORCE_SPIKE,
    },
    ParityVector {
        name: "accel_extreme_holds_at_fine",
        expected_ceiling: GEAR_FINE,
        expected_ext_gate: GATE_ACCELERATION_EXTREME,
    },
    ParityVector {
        name: "thermostat_gamma_dt_floor",
        expected_ceiling: GEAR_FINE,
        expected_ext_gate: GATE_THERMOSTAT_STABILITY,
    },
    ParityVector {
        name: "neighbor_list_saturated",
        expected_ceiling: GEAR_FINE,
        expected_ext_gate: GATE_INTEGRATION_HEALTH,
    },
    ParityVector {
        name: "sentinel_overdue",
        expected_ceiling: GEAR_FINE,
        expected_ext_gate: GATE_TSH_MEMORY,
    },
    ParityVector {
        name: "uv_burst_phase",
        expected_ceiling: GEAR_FINE,
        expected_ext_gate: GATE_PHASE_REQUIREMENT,
    },
];

// ═══════════════════════════════════════════════════════════════════════════
// Compile-time invariants
// ═══════════════════════════════════════════════════════════════════════════

// Lexicographic authority chain. If anybody renumbers a rank so a scientific
// gate outranks a physical one, this fails to compile rather than silently
// permitting a salience to outvote a prohibition.
const _: () = assert!(RANK_PHYSICAL_EMERGENCY < RANK_HMR_HARD);
const _: () = assert!(RANK_HMR_HARD < RANK_TOPOLOGY_PERMISSION);
const _: () = assert!(RANK_TOPOLOGY_PERMISSION < RANK_GOVERNOR_UNARMED);
const _: () = assert!(RANK_GOVERNOR_UNARMED < RANK_CONSTRAINT_STABILITY);
const _: () = assert!(RANK_CONSTRAINT_STABILITY < RANK_ENERGY_DRIFT);
const _: () = assert!(RANK_ENERGY_DRIFT < RANK_FORCE_STABILITY);
const _: () = assert!(RANK_FORCE_STABILITY < RANK_FORCE_SPIKE);
const _: () = assert!(RANK_FORCE_SPIKE < RANK_ACCELERATION_EXTREME);
const _: () = assert!(RANK_ACCELERATION_EXTREME < RANK_THERMOSTAT_STABILITY);
const _: () = assert!(RANK_THERMOSTAT_STABILITY < RANK_INTEGRATION_HEALTH);
const _: () = assert!(RANK_INTEGRATION_HEALTH < RANK_CHUNK_LATENCY);
const _: () = assert!(RANK_CHUNK_LATENCY < RANK_TSH_MEMORY);
const _: () = assert!(RANK_TSH_MEMORY < RANK_TSH_SALIENCE);
const _: () = assert!(RANK_TSH_SALIENCE < RANK_PHASE_REQUIREMENT);
const _: () = assert!(RANK_PHASE_REQUIREMENT < RANK_DWELL);
const _: () = assert!(RANK_DWELL < RANK_LOCKOUT);
const _: () = assert!(RANK_LOCKOUT < RANK_UPSHIFT_RATE);
const _: () = assert!(RANK_UPSHIFT_RATE < RANK_NONE);

// Extension gate ids must not collide with the base range.
const _: () = assert!(GATE_EXT_FIRST > BASE_GATE_GOVERNOR_UNARMED);
const _: () = assert!((GATE_EXT_LAST - GATE_EXT_FIRST + 1) as usize == GATE_EXT_COUNT);
// The verdict nibbles must fit the 32-bit packed field, or one gate's verdict
// would silently overwrite another's.
const _: () = assert!(GATE_EXT_COUNT * 4 <= 32);
// Exactly one extension gate stands outside the ceiling fold: GATE_EXT_BINDING,
// which replaces the fold rather than contributing an operand to it.
const _: () = assert!(CEILING_GATE_COUNT + 1 == GATE_EXT_COUNT);
// The unarmed cap must be a real, always-reachable gear, or a binding refusal
// would deadlock the run instead of slowing it.
const _: () = assert!(DEFAULT_UNARMED_CAP_GEAR <= GEAR_MAX_ACTIVE);

// The HMR hard gate is only meaningful while the safe gear is below campaign.
const _: () = assert!(GEAR_SAFE < GEAR_CAMPAIGN);

// ═══════════════════════════════════════════════════════════════════════════
// THE COMPILED DEVICE EVALUATOR — WHAT IT IS AND WHAT IT IS NOT
//
// `crates/prism-nhs/src/cuda/physical_governor.cu` is a REAL translation unit,
// registered in `crates/prism-nhs/build.rs` and compiled by nvcc on every
// `gpu`-feature build. That is what makes the header's `static_assert` wall —
// three `sizeof` pins, ten `offsetof` pins, the lexicographic authority chain
// and the band-monotonicity chain — actually fire; before it existed, no
// compiler had ever evaluated a single one of them and
// `prism_physical_governor_decide` was present in no binary.
//
// COMPILED IS NOT LINKED IS NOT CALLED. For a period after that registration
// landed, every reference to this archive lived under `#[cfg(test)]`. The
// `--release` binary therefore contained none of it — `nm -a
// target/release/nhs_rt_full | grep -c physgov` returned 0 — while this
// module's own documentation read as though a device evaluator were running.
// It was not. The three facts are now pinned separately:
//
//   * REFERENCE IMPLEMENTATION: [`compose`], in this file, in Rust. This is
//     what [`decide_with_base_governor`] evaluates and what the emitted
//     `prismos_physical_governor_decision` rows describe. The gear the run
//     integrates at comes from here.
//
//   * CONFORMANCE AUTHORITY: the compiled archive, reached from the RUN path
//     (not merely from tests) at two points in
//     `bin/nhs_rt_full.rs::PhysicalGovernorBridge`:
//       1. `arm()` calls [`verify_compiled_evaluator_abi`] once per stream
//          before any CUDA work and refuses to arm on drift;
//       2. `decide_at_chunk_boundary()` calls
//          [`decide_on_compiled_evaluator`] once per governed decision with
//          the identical inputs and requires bit-equality on every ACTUATING
//          field. Divergence is a named terminal error, because a host mirror
//          and a compiled evaluator that disagree about a timestep leave no
//          authoritative answer to integrate on.
//     Both halves are compiled IEEE-strict specifically so that comparison is
//     exact rather than approximate.
//
//   * DEVICE CODE GENERATION ONLY: `prism_physical_governor_decide_kernel`.
//     Nothing in the run path LAUNCHES it. The governor takes one decision per
//     chunk from host-reduced observables, so there is no device-resident
//     input for a kernel to read. It exists so the `__device__` half is driven
//     through ptxas rather than merely parsed. Do not read its existence as
//     evidence that a governor kernel runs on the GPU — it does not.
//
// `abi_witness` regenerates the C-side sizes and offsets from the build that
// just happened, instead of trusting a hand-transcribed comment.
// ═══════════════════════════════════════════════════════════════════════════

/// Number of `offsetof` pins `prism_physgov_abi_offsets` reports.
pub const ABI_OFFSET_COUNT: usize = 10;

#[cfg(feature = "gpu")]
mod ffi {
    use super::{ExtObservation, ExtPolicy, PhysGovDecision};

    extern "C" {
        /// Link probe for `libphysical_governor.a`. Returns `0x5047_5631`.
        pub fn prism_physgov_link_probe() -> u32;

        /// C-side `sizeof` for the three wire structs plus the schema version.
        pub fn prism_physgov_abi_sizes(
            ext_policy_bytes: *mut u32,
            ext_observation_bytes: *mut u32,
            decision_bytes: *mut u32,
            schema_version: *mut u32,
        );

        /// C-side `offsetof` for the fields the header pins.
        pub fn prism_physgov_abi_offsets(out: *mut u32, capacity: u32) -> u32;

        /// Fill the documented extension-policy defaults from the C side.
        pub fn prism_physgov_default_ext_policy(out: *mut ExtPolicy) -> i32;

        /// The composed evaluator — the same `prism_physical_governor_decide`
        /// the device kernel calls.
        pub fn prism_physical_governor_decide_host(
            base_policy: *const crate::gearbox::GearGovernorPolicy,
            ext_policy: *const ExtPolicy,
            base_observation: *const crate::gearbox::GearGuardObservation,
            ext_observation: *const ExtObservation,
            tsh: *const crate::gearbox::TshSalienceRequest,
            out: *mut PhysGovDecision,
        ) -> i32;
    }
}

/// What the C compiler that built `libphysical_governor.a` measured.
///
/// Not a transcription: every number here comes back from the archive produced
/// by the build that is running these bindings.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiWitness {
    /// `sizeof(PrismPhysGovExtPolicy)`.
    pub ext_policy_bytes: u32,
    /// `sizeof(PrismPhysGovExtObservation)`.
    pub ext_observation_bytes: u32,
    /// `sizeof(PrismPhysGovDecision)`.
    pub decision_bytes: u32,
    /// `PRISM_PHYSGOV_SCHEMA_VERSION`.
    pub schema_version: u32,
    /// The ten pinned `offsetof` values, in the order documented on
    /// `prism_physgov_abi_offsets`.
    pub offsets: [u32; ABI_OFFSET_COUNT],
    /// The link probe sentinel.
    pub link_probe: u32,
}

/// Read the ABI the compiled device evaluator was built with.
///
/// This is the reachability proof in executable form: if
/// `src/cuda/physical_governor.cu` were not compiled and linked, this function
/// would not resolve at link time and the crate would not build.
#[cfg(feature = "gpu")]
#[must_use]
pub fn abi_witness() -> AbiWitness {
    let mut ext_policy_bytes = 0u32;
    let mut ext_observation_bytes = 0u32;
    let mut decision_bytes = 0u32;
    let mut schema_version = 0u32;
    let mut offsets = [0u32; ABI_OFFSET_COUNT];
    // SAFETY: all pointers are to live, correctly-sized local storage; the C
    // side writes at most `ABI_OFFSET_COUNT` u32s because it is told the
    // capacity, and it treats every pointer as optional.
    let (link_probe, written) = unsafe {
        ffi::prism_physgov_abi_sizes(
            &mut ext_policy_bytes,
            &mut ext_observation_bytes,
            &mut decision_bytes,
            &mut schema_version,
        );
        let written = ffi::prism_physgov_abi_offsets(offsets.as_mut_ptr(), ABI_OFFSET_COUNT as u32);
        (ffi::prism_physgov_link_probe(), written)
    };
    debug_assert_eq!(written as usize, ABI_OFFSET_COUNT);
    AbiWitness {
        ext_policy_bytes,
        ext_observation_bytes,
        decision_bytes,
        schema_version,
        offsets,
        link_probe,
    }
}

/// Failure modes of the compiled-evaluator ABI cross-check.
#[cfg(feature = "gpu")]
#[derive(Debug, thiserror::Error)]
pub enum AbiError {
    /// The linked archive is not the expected one.
    #[error("physical governor link probe returned {actual:#010x}, expected {expected:#010x}")]
    LinkProbe {
        /// What the archive returned.
        actual: u32,
        /// What this build expects.
        expected: u32,
    },
    /// A wire struct has a different size on the two sides.
    #[error(
        "physical governor ABI drift: {what} is {c_side} bytes in CUDA and {rust_side} in Rust"
    )]
    Size {
        /// Which struct.
        what: &'static str,
        /// `sizeof` as the C compiler measured it.
        c_side: u32,
        /// `size_of` as rustc measured it.
        rust_side: u32,
    },
    /// A pinned field offset differs.
    #[error("physical governor ABI drift: pinned offset {index} is {c_side} in CUDA and {rust_side} in Rust")]
    Offset {
        /// Index into the documented pin order.
        index: usize,
        /// `offsetof` as the C compiler measured it.
        c_side: u32,
        /// The offset this build pins.
        rust_side: u32,
    },
    /// The two halves disagree on the wire version.
    #[error("physical governor schema drift: CUDA says {c_side}, Rust says {rust_side}")]
    Schema {
        /// `PRISM_PHYSGOV_SCHEMA_VERSION` as compiled.
        c_side: u32,
        /// [`PHYSGOV_SCHEMA_VERSION`].
        rust_side: u32,
    },
}

/// The Rust-side values the C `offsetof` pins must equal, in the order
/// `prism_physgov_abi_offsets` documents.
#[cfg(feature = "gpu")]
const PINNED_OFFSETS: [u32; ABI_OFFSET_COUNT] = [
    32,  // ExtPolicy.force_rate_hold_per_ps
    136, // ExtPolicy.phase_ceiling_gear
    168, // ExtPolicy.ext_policy_digest
    4,   // ExtObservation.hmr_applied_verified
    24,  // ExtObservation.max_accel_a_per_ps2
    68,  // ExtObservation.tsh_instability_memory
    24,  // Decision.ext_binding_gate
    64,  // Decision.downshift_latency_steps
    88,  // Decision.policy_digest
    104, // Decision.ext_binding_refusal
];

/// Assert that the compiled device evaluator and this host mirror share one
/// ABI.
///
/// The header `static_assert`s pin the C half and the `const _: () = assert!`s
/// above pin the Rust half; this closes the loop between them at run time,
/// against the archive that was actually linked.
///
/// # Errors
///
/// [`AbiError`] naming the first field that drifted.
#[cfg(feature = "gpu")]
pub fn verify_compiled_evaluator_abi() -> Result<AbiWitness, AbiError> {
    const EXPECTED_PROBE: u32 = 0x5047_5631;
    let w = abi_witness();
    if w.link_probe != EXPECTED_PROBE {
        return Err(AbiError::LinkProbe {
            actual: w.link_probe,
            expected: EXPECTED_PROBE,
        });
    }
    if w.schema_version != PHYSGOV_SCHEMA_VERSION {
        return Err(AbiError::Schema {
            c_side: w.schema_version,
            rust_side: PHYSGOV_SCHEMA_VERSION,
        });
    }
    for (what, c_side, rust_side) in [
        (
            "PrismPhysGovExtPolicy",
            w.ext_policy_bytes,
            size_of::<ExtPolicy>() as u32,
        ),
        (
            "PrismPhysGovExtObservation",
            w.ext_observation_bytes,
            size_of::<ExtObservation>() as u32,
        ),
        (
            "PrismPhysGovDecision",
            w.decision_bytes,
            size_of::<PhysGovDecision>() as u32,
        ),
    ] {
        if c_side != rust_side {
            return Err(AbiError::Size {
                what,
                c_side,
                rust_side,
            });
        }
    }
    for (index, (c_side, rust_side)) in w.offsets.iter().zip(PINNED_OFFSETS.iter()).enumerate() {
        if c_side != rust_side {
            return Err(AbiError::Offset {
                index,
                c_side: *c_side,
                rust_side: *rust_side,
            });
        }
    }
    Ok(w)
}

/// The documented extension-policy defaults, read back from the COMPILED
/// device evaluator rather than from the Rust transcription of them.
///
/// `armed` comes back 0: the C side deliberately does not arm a policy it
/// merely filled with defaults.
///
/// # Errors
///
/// The C return code when the call is refused.
#[cfg(feature = "gpu")]
pub fn device_default_ext_policy() -> Result<ExtPolicy, i32> {
    let mut policy = ExtPolicy::default();
    // SAFETY: `policy` is a live, correctly-aligned `#[repr(C, align(16))]`
    // value whose layout the ABI check above verifies against the C half.
    let rc = unsafe { ffi::prism_physgov_default_ext_policy(&mut policy) };
    if rc != 0 {
        return Err(rc);
    }
    Ok(policy)
}

/// Evaluate the composed governor through the COMPILED device evaluator.
///
/// [`compose`] is the host mirror and the reference implementation; this is the
/// same decision taken by `prism_physical_governor_decide` in
/// `src/cuda/physical_governor.cu` — the exact function
/// `prism_physical_governor_decide_kernel` calls on the device. Running both
/// over the same inputs is what turns "the two implementations agree" from a
/// claim into a check; see [`PARITY_VECTORS`].
///
/// Performs no CUDA work: the evaluator is pure arithmetic, so this is usable
/// in dry-run policy validation as well as in the live path.
///
/// # Errors
///
/// The C return code when the evaluator refuses the inputs (null pointer). An
/// unarmed or wrong-schema extension policy is NOT an error — it is a governed
/// outcome, emitted as a full record with
/// [`PhysGovDecision::ext_binding_refusal`] set.
#[cfg(feature = "gpu")]
pub fn decide_on_compiled_evaluator(
    base_policy: &crate::gearbox::GearGovernorPolicy,
    base_obs: &crate::gearbox::GearGuardObservation,
    salience: Option<&crate::gearbox::TshSalienceRequest>,
    ext_policy: &ExtPolicy,
    ext_obs: &ExtObservation,
) -> Result<PhysGovDecision, i32> {
    let mut out = PhysGovDecision::default();
    let tsh = salience.map_or(std::ptr::null(), |s| s as *const _);
    // SAFETY: every reference is live for the duration of the call, `out` is a
    // correctly-aligned `#[repr(C, align(16))]` value, and a null `tsh` is the
    // documented "no scientific opinion" encoding the C side accepts.
    let rc = unsafe {
        ffi::prism_physical_governor_decide_host(
            base_policy,
            ext_policy,
            base_obs,
            ext_obs,
            tsh,
            &mut out,
        )
    };
    if rc != 0 {
        return Err(rc);
    }
    Ok(out)
}

// ═══════════════════════════════════════════════════════════════════════════
// Stream-topology admission policy (HOST-ONLY; never crosses the device wire)
// ═══════════════════════════════════════════════════════════════════════════

/// Wire version of the stream-topology policy document.
///
/// Deliberately independent of [`PHYSGOV_SCHEMA_VERSION`]: [`ExtPolicy`] is a
/// 208-byte DEVICE struct whose layout is `static_assert`ed against
/// `physical_governor.cuh` and read back out of the compiled archive by
/// [`verify_compiled_evaluator_abi`]. Stream topology is a HOST admission fact
/// decided before `cuInit` and never uploaded, so it must not be carried in
/// that struct: adding a field there would change `sizeof`, break both halves
/// of the ABI pin, and force a device rebuild for a host-only edit. The two
/// schema versions therefore move independently.
pub const STREAM_TOPOLOGY_SCHEMA_VERSION: u32 = 1;

/// Documented default for [`StreamTopologyPolicy::stream_floor`].
///
/// Eight is the stream count every captured-graph twin run to date has used,
/// so this default reproduces the historical admission decision exactly. It is
/// a FLOOR, not an equality: how many streams a run may open is a capacity
/// question, and capacity is measured against real VRAM after the context
/// exists — never asserted by a comparison written into the source.
pub const DEFAULT_STREAM_FLOOR: u32 = 8;

/// Documented default for [`StreamTopologyPolicy::bridge_group_count`].
///
/// Four is the bridge-group count the multi-differential emitter already
/// reports for every admitted run, so the divisibility rule below restates the
/// topology the engine already builds rather than adding a constraint on it.
pub const DEFAULT_BRIDGE_GROUP_COUNT: u32 = 4;

/// Environment variable naming the stream-topology policy file. Mirrors the
/// `PRISM_GHOST_TOPOLOGY_POLICY_PATH` convention the build already uses.
pub const ENV_STREAM_TOPOLOGY_POLICY_PATH: &str = "PRISM_STREAM_TOPOLOGY_POLICY_PATH";

/// Environment variable pinning the sha256 of that file.
pub const ENV_STREAM_TOPOLOGY_POLICY_SHA256: &str = "PRISM_STREAM_TOPOLOGY_POLICY_SHA256";

/// Environment override for [`StreamTopologyPolicy::stream_floor`].
pub const ENV_STREAM_FLOOR: &str = "PRISM_STREAM_FLOOR";

/// Environment override for [`StreamTopologyPolicy::bridge_group_count`].
///
/// Accepted for provenance and receipt symmetry with every other topology row,
/// but the only value that resolves successfully is
/// [`BRIDGE_GROUP_STRUCTURAL_WIDTH`] — see [`validate_stream_topology`].
pub const ENV_BRIDGE_GROUP_COUNT: &str = "PRISM_BRIDGE_GROUP_COUNT";

/// The bridge-group count the multi-differential emitter is COMPILED against.
///
/// This is not a default and not a preference: it is the width of compiled
/// data structures. `CryoUvProtocol::twin_differential_set()` returns
/// `[Self; 4]`, and the per-group accumulators (`group_residue_counts`,
/// `group_residue_phasors`, the `[f64; 4]` phase/lag arrays) are all allocated
/// at that same fixed width. The emitter then indexes them with
/// `group_idx = stream_id / epg_val` where `epg_val = (n_streams / 4).max(1)`.
///
/// A stream count that is congruent to some OTHER divisor therefore clears the
/// admission gate and then indexes out of bounds. With a divisor of 2, ten
/// streams satisfy `10 % 2 == 0`, but `epg_val` is still `2`, so stream 9
/// yields `group_idx = 4` against four-wide accumulators and the run panics
/// after the device lease has already been paid.
///
/// Widening the topology is therefore an emitter change first and a policy
/// change second, never policy alone. Keeping the row policy-resolved (with a
/// real [`ValueOrigin`] and receipt line) while pinning its admissible value
/// preserves provenance without making an out-of-bounds index reachable.
pub const BRIDGE_GROUP_STRUCTURAL_WIDTH: u32 = 4;

/// Stable id of the admission rule these rows govern.
pub const STREAM_TOPOLOGY_RULE_ID: &str = "CAPTURED_GRAPH_TWIN_STREAM_TOPOLOGY";

/// [`StreamTopologyVerdict::reason`]: the stream set satisfies the topology.
pub const STREAM_TOPOLOGY_ADMITTED: &str = "ADMITTED";

/// [`StreamTopologyVerdict::reason`]: fewer streams than the resolved floor.
pub const STREAM_TOPOLOGY_BELOW_FLOOR: &str = "CAPTURED_GRAPH_TWIN_STREAM_TOPOLOGY_BELOW_FLOOR";

/// [`StreamTopologyVerdict::reason`]: the stream set does not divide evenly
/// into the resolved bridge groups.
pub const STREAM_TOPOLOGY_UNEVEN_BRIDGE_GROUPS: &str =
    "CAPTURED_GRAPH_TWIN_STREAM_TOPOLOGY_UNEVEN_BRIDGE_GROUPS";

/// Failure modes of stream-topology resolution.
#[derive(Debug, thiserror::Error)]
pub enum StreamTopologyError {
    /// The policy file could not be read.
    #[error("stream topology policy file {path} could not be read: {source}")]
    Read {
        /// The path that failed.
        path: String,
        /// The underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// The policy file is not a valid document.
    #[error("stream topology policy file {path} is not a valid policy document: {message}")]
    Parse {
        /// The path that failed.
        path: String,
        /// The parser's message.
        message: String,
    },
    /// The policy file's sha256 does not match the pinned digest.
    #[error("stream topology policy file {path} sha256 {actual} does not match pinned {expected}")]
    DigestMismatch {
        /// The path that failed.
        path: String,
        /// The digest that was computed.
        actual: String,
        /// The digest that was pinned.
        expected: String,
    },
    /// The document declares an unknown schema version.
    #[error(
        "stream topology policy file {path} declares schema_version {found}, expected {expected}"
    )]
    SchemaMismatch {
        /// The path that failed.
        path: String,
        /// What the document said.
        found: u32,
        /// What this build understands.
        expected: u32,
    },
    /// An environment override is present but unusable.
    ///
    /// A malformed override is an ERROR, never a silent fall-through to the
    /// default: an operator who exported `PRISM_STREAM_FLOOR=sixteen` must not
    /// be told the run was governed by a floor they did not choose.
    #[error("stream topology environment override {var}={value} is unusable: {message}")]
    EnvParse {
        /// Which variable.
        var: &'static str,
        /// What it held.
        value: String,
        /// Why it was rejected.
        message: String,
    },
    /// A resolved field is structurally incoherent.
    #[error("stream topology policy field '{field}' is invalid: {reason}")]
    InvalidValue {
        /// Which field.
        field: &'static str,
        /// Why it was rejected.
        reason: String,
    },
}

/// The on-disk stream-topology policy document.
///
/// `deny_unknown_fields` makes a document written for a different topology
/// version fail loudly instead of having its extra keys silently ignored.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamTopologyDocument {
    /// Must equal [`STREAM_TOPOLOGY_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Override for [`StreamTopologyPolicy::stream_floor`].
    #[serde(default)]
    pub stream_floor: Option<u32>,
    /// Override for [`StreamTopologyPolicy::bridge_group_count`].
    #[serde(default)]
    pub bridge_group_count: Option<u32>,
}

/// Overrides supplied by one layer of the resolution stack.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamTopologyOverrides {
    /// Override for [`StreamTopologyPolicy::stream_floor`], when the layer set one.
    pub stream_floor: Option<u32>,
    /// Override for [`StreamTopologyPolicy::bridge_group_count`], when the layer set one.
    pub bridge_group_count: Option<u32>,
}

/// The resolved stream topology an admission decision is taken against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamTopologyPolicy {
    /// [`STREAM_TOPOLOGY_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Smallest admissible stream count.
    pub stream_floor: u32,
    /// Number of bridge groups the stream set must divide into evenly.
    pub bridge_group_count: u32,
}

impl Default for StreamTopologyPolicy {
    /// The documented defaults. Resolving with no policy file, no environment
    /// and no command line reproduces the historical eight-stream / four-group
    /// topology exactly, so an unchanged operator invocation is unchanged.
    fn default() -> Self {
        Self {
            schema_version: STREAM_TOPOLOGY_SCHEMA_VERSION,
            stream_floor: DEFAULT_STREAM_FLOOR,
            bridge_group_count: DEFAULT_BRIDGE_GROUP_COUNT,
        }
    }
}

/// One stream-topology admission decision, with every operand it was taken on.
///
/// A refusal carries the observed and required numbers, not a bare code: a
/// gate that refuses without publishing its arithmetic cannot be audited and
/// cannot be repaired by the operator it refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamTopologyVerdict {
    /// [`STREAM_TOPOLOGY_RULE_ID`].
    pub rule_id: &'static str,
    /// Whether the stream set is admitted.
    pub admitted: bool,
    /// The stream count that was adjudicated.
    pub observed_streams: u32,
    /// The floor that was in force.
    pub stream_floor: u32,
    /// The bridge-group count that was in force.
    pub bridge_group_count: u32,
    /// `observed_streams / bridge_group_count` when admitted; 0 otherwise.
    pub streams_per_bridge_group: u32,
    /// Named outcome code.
    pub reason: &'static str,
    /// The arithmetic behind `reason`, in observed-vs-required form.
    pub detail: String,
}

/// A resolved stream topology together with the provenance of every value.
#[derive(Debug, Clone)]
pub struct StreamTopologyResolution {
    /// The resolved policy.
    pub policy: StreamTopologyPolicy,
    /// `(field_name, origin)` for every governed field, in declaration order.
    pub origins: Vec<(&'static str, ValueOrigin)>,
    /// Path of the policy file, when one was used.
    pub policy_path: Option<String>,
    /// sha256 of the policy file, when one was used.
    pub policy_sha256: Option<String>,
    /// Fingerprint of the resolved topology, its origins, and the file sha256.
    pub digest: u64,
}

impl StreamTopologyResolution {
    /// Origin of a single field, for targeted evidence lines.
    #[must_use]
    pub fn origin_of(&self, field: &str) -> Option<ValueOrigin> {
        self.origins
            .iter()
            .find(|(name, _)| *name == field)
            .map(|(_, origin)| *origin)
    }

    /// A receipt line per governed field: `field=value origin=...`.
    ///
    /// Same discipline as [`ExtPolicyResolution::receipt_lines`]: this is what
    /// makes "no hardcoded stream gate" auditable rather than merely asserted.
    #[must_use]
    pub fn receipt_lines(&self) -> Vec<String> {
        let p = &self.policy;
        let f = |name: &'static str, value: String| -> String {
            format!(
                "{name}={value} origin={}",
                self.origin_of(name).map_or("unknown", ValueOrigin::as_str)
            )
        };
        vec![
            f("stream_floor", p.stream_floor.to_string()),
            f("bridge_group_count", p.bridge_group_count.to_string()),
        ]
    }

    /// Every field named in [`Self::receipt_lines`], for the mechanical check
    /// that no governed value is missing a receipt and no receipt names a
    /// value that governs nothing. Both rows below are read by
    /// [`validate_stream_topology`] and by [`Self::admits`].
    #[must_use]
    pub fn receipted_fields() -> &'static [&'static str] {
        &["stream_floor", "bridge_group_count"]
    }

    /// The receipt object for the engine-dispatch record.
    ///
    /// Every number is published next to the origin that authorised it, so a
    /// reader can bind an admitted stream count to the exact authority that
    /// admitted it rather than to a source literal.
    #[must_use]
    pub fn receipt_json(&self) -> serde_json::Value {
        serde_json::json!({
            "rule_id": STREAM_TOPOLOGY_RULE_ID,
            "schema_version": self.policy.schema_version,
            "stream_floor": self.policy.stream_floor,
            "stream_floor_origin": self
                .origin_of("stream_floor")
                .map_or("unknown", ValueOrigin::as_str),
            "bridge_group_count": self.policy.bridge_group_count,
            "bridge_group_count_origin": self
                .origin_of("bridge_group_count")
                .map_or("unknown", ValueOrigin::as_str),
            "policy_path": self.policy_path,
            "policy_sha256": self.policy_sha256,
            "stream_topology_digest": format!("{:016x}", self.digest),
            "receipt_lines": self.receipt_lines(),
        })
    }

    /// Adjudicate a stream count against the resolved topology.
    ///
    /// This is the policy-resolved replacement for an equality test against a
    /// source literal. It is a floor plus a divisibility requirement, so a
    /// larger, structurally legal stream set is admitted without a source
    /// edit, while a set that cannot be split across the bridge groups the
    /// emitter builds is still refused — before any device work is paid for.
    ///
    /// This decides ADMISSIBILITY only. Whether the device can actually carry
    /// the admitted stream count is a capacity question that must be measured
    /// against real free VRAM after the context exists; relaxing this rule
    /// without that second stage converts a clean pre-CUDA refusal into an
    /// out-of-memory failure after the lease is paid.
    #[must_use]
    pub fn admits(&self, multi_stream: u32) -> StreamTopologyVerdict {
        let p = &self.policy;
        let mut verdict = StreamTopologyVerdict {
            rule_id: STREAM_TOPOLOGY_RULE_ID,
            admitted: false,
            observed_streams: multi_stream,
            stream_floor: p.stream_floor,
            bridge_group_count: p.bridge_group_count,
            streams_per_bridge_group: 0,
            reason: STREAM_TOPOLOGY_ADMITTED,
            detail: String::new(),
        };
        if multi_stream < p.stream_floor {
            verdict.reason = STREAM_TOPOLOGY_BELOW_FLOOR;
            verdict.detail = format!(
                "observed {} streams, resolved floor is {} (origin={})",
                multi_stream,
                p.stream_floor,
                self.origin_of("stream_floor")
                    .map_or("unknown", ValueOrigin::as_str)
            );
            return verdict;
        }
        if p.bridge_group_count == 0 || !multi_stream.is_multiple_of(p.bridge_group_count) {
            verdict.reason = STREAM_TOPOLOGY_UNEVEN_BRIDGE_GROUPS;
            verdict.detail = format!(
                "observed {} streams does not divide into {} bridge groups (origin={})",
                multi_stream,
                p.bridge_group_count,
                self.origin_of("bridge_group_count")
                    .map_or("unknown", ValueOrigin::as_str)
            );
            return verdict;
        }
        verdict.admitted = true;
        verdict.streams_per_bridge_group = multi_stream / p.bridge_group_count;
        verdict.detail = format!(
            "{} streams >= floor {} and divides into {} bridge groups of {}",
            multi_stream, p.stream_floor, p.bridge_group_count, verdict.streams_per_bridge_group
        );
        verdict
    }
}

/// Reject a stream topology that cannot describe a legal stream set.
///
/// Every refusal here happens at RESOLUTION time, where the operator can still
/// see and repair the cause — never as an unreachable gate discovered after a
/// device lease is paid.
///
/// # Errors
///
/// [`StreamTopologyError::InvalidValue`] naming the offending field.
pub fn validate_stream_topology(p: &StreamTopologyPolicy) -> Result<(), StreamTopologyError> {
    if p.schema_version != STREAM_TOPOLOGY_SCHEMA_VERSION {
        return Err(StreamTopologyError::InvalidValue {
            field: "schema_version",
            reason: format!(
                "resolved schema {} is not the {} this build understands",
                p.schema_version, STREAM_TOPOLOGY_SCHEMA_VERSION
            ),
        });
    }
    if p.bridge_group_count == 0 {
        return Err(StreamTopologyError::InvalidValue {
            field: "bridge_group_count",
            reason: "must be at least 1; zero groups makes every stream count \
                     indivisible and refuses the run with no reachable repair"
                .to_string(),
        });
    }
    if p.bridge_group_count != BRIDGE_GROUP_STRUCTURAL_WIDTH {
        return Err(StreamTopologyError::InvalidValue {
            field: "bridge_group_count",
            reason: format!(
                "resolved {} but the multi-differential emitter is compiled \
                 against exactly {} bridge groups: `CryoUvProtocol::\
                 twin_differential_set()` returns `[Self; {}]` and the \
                 per-group accumulators are allocated at that fixed width. \
                 Admitting a stream count congruent to {} would let \
                 `group_idx = stream_id / epg_val` exceed the accumulator \
                 bound and panic AFTER the device lease is paid. This divisor \
                 is a structural property of the compiled emitter, not a \
                 tunable policy: widening it requires widening the emitter \
                 first",
                p.bridge_group_count,
                BRIDGE_GROUP_STRUCTURAL_WIDTH,
                BRIDGE_GROUP_STRUCTURAL_WIDTH,
                p.bridge_group_count
            ),
        });
    }
    if p.stream_floor == 0 {
        return Err(StreamTopologyError::InvalidValue {
            field: "stream_floor",
            reason: "must be at least 1; a floor of zero admits a run with no \
                     streams, which has no producer for any tensor family"
                .to_string(),
        });
    }
    if !p.stream_floor.is_multiple_of(p.bridge_group_count) {
        return Err(StreamTopologyError::InvalidValue {
            field: "stream_floor",
            reason: format!(
                "floor {} does not divide into {} bridge groups, so the floor \
                 itself would be refused and no stream count at all could be \
                 admitted below {}",
                p.stream_floor,
                p.bridge_group_count,
                p.stream_floor + (p.bridge_group_count - p.stream_floor % p.bridge_group_count)
            ),
        });
    }
    Ok(())
}

/// Read the environment layer of the stream topology.
///
/// # Errors
///
/// [`StreamTopologyError::EnvParse`] when a variable is present but is not a
/// decimal `u32`. An unset or empty variable is "this layer said nothing".
pub fn stream_topology_overrides_from_env() -> Result<StreamTopologyOverrides, StreamTopologyError>
{
    Ok(StreamTopologyOverrides {
        stream_floor: env_u32(ENV_STREAM_FLOOR)?,
        bridge_group_count: env_u32(ENV_BRIDGE_GROUP_COUNT)?,
    })
}

fn env_u32(var: &'static str) -> Result<Option<u32>, StreamTopologyError> {
    let Some(raw) = std::env::var_os(var) else {
        return Ok(None);
    };
    let Some(text) = raw.to_str() else {
        return Err(StreamTopologyError::EnvParse {
            var,
            value: raw.to_string_lossy().into_owned(),
            message: "not valid UTF-8".to_string(),
        });
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }
    trimmed
        .parse::<u32>()
        .map(Some)
        .map_err(|e| StreamTopologyError::EnvParse {
            var,
            value: trimmed.to_string(),
            message: e.to_string(),
        })
}

fn env_text(var: &'static str) -> Option<String> {
    let raw = std::env::var_os(var)?;
    let text = raw.to_str()?.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Resolve the stream topology from every layer, with each value's origin
/// recorded: documented defaults, overlaid by an optional versioned policy
/// file, overlaid by the environment, overlaid by the command line.
///
/// `policy_path` / `expected_sha256` fall back to
/// [`ENV_STREAM_TOPOLOGY_POLICY_PATH`] / [`ENV_STREAM_TOPOLOGY_POLICY_SHA256`]
/// when the caller passes `None`. Pinning the digest is what prevents a policy
/// that silently changed under the run — the same provenance failure
/// [`resolve_ext_policy`] exists to prevent.
///
/// # Errors
///
/// [`StreamTopologyError`] on unreadable / unparseable / mis-versioned /
/// mis-digested files, on an unusable environment override, or when the
/// resolved rows cannot describe a legal stream set.
pub fn resolve_stream_topology(
    policy_path: Option<&str>,
    expected_sha256: Option<&str>,
    cli: StreamTopologyOverrides,
) -> Result<StreamTopologyResolution, StreamTopologyError> {
    let path = policy_path
        .map(str::to_string)
        .or_else(|| env_text(ENV_STREAM_TOPOLOGY_POLICY_PATH));
    let sha = expected_sha256
        .map(str::to_string)
        .or_else(|| env_text(ENV_STREAM_TOPOLOGY_POLICY_SHA256));
    let env = stream_topology_overrides_from_env()?;
    resolve_stream_topology_layered(path.as_deref(), sha.as_deref(), env, cli)
}

/// The pure core of [`resolve_stream_topology`], with the environment layer
/// passed in rather than read.
///
/// Keeping the process environment out of the resolver is what lets the layer
/// precedence be tested deterministically: mutating real environment variables
/// from a test would race every other test in the same process.
///
/// # Errors
///
/// [`StreamTopologyError`], as [`resolve_stream_topology`].
pub fn resolve_stream_topology_layered(
    policy_path: Option<&str>,
    expected_sha256: Option<&str>,
    env: StreamTopologyOverrides,
    cli: StreamTopologyOverrides,
) -> Result<StreamTopologyResolution, StreamTopologyError> {
    let mut policy = StreamTopologyPolicy::default();
    let mut origins: Vec<(&'static str, ValueOrigin)> = Vec::new();
    let mut sha_hex: Option<String> = None;

    let document: Option<StreamTopologyDocument> = match policy_path {
        None => None,
        Some(path) => {
            let bytes = std::fs::read(path).map_err(|source| StreamTopologyError::Read {
                path: path.to_string(),
                source,
            })?;
            let digest = {
                use sha2::Digest as _;
                let mut h = sha2::Sha256::new();
                h.update(&bytes);
                h.finalize()
            };
            let hex = digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            if let Some(expected) = expected_sha256 {
                if !expected.eq_ignore_ascii_case(&hex) {
                    return Err(StreamTopologyError::DigestMismatch {
                        path: path.to_string(),
                        actual: hex,
                        expected: expected.to_string(),
                    });
                }
            }
            sha_hex = Some(hex);
            let doc: StreamTopologyDocument =
                serde_json::from_slice(&bytes).map_err(|e| StreamTopologyError::Parse {
                    path: path.to_string(),
                    message: e.to_string(),
                })?;
            if doc.schema_version != STREAM_TOPOLOGY_SCHEMA_VERSION {
                return Err(StreamTopologyError::SchemaMismatch {
                    path: path.to_string(),
                    found: doc.schema_version,
                    expected: STREAM_TOPOLOGY_SCHEMA_VERSION,
                });
            }
            Some(doc)
        }
    };

    let mut overlay = |field: &'static str,
                       slot: &mut u32,
                       from_file: Option<u32>,
                       from_env: Option<u32>,
                       from_cli: Option<u32>| {
        let (value, origin) = match (from_cli, from_env, from_file) {
            (Some(v), _, _) => (Some(v), ValueOrigin::Cli),
            (None, Some(v), _) => (Some(v), ValueOrigin::Environment),
            (None, None, Some(v)) => (Some(v), ValueOrigin::PolicyFile),
            (None, None, None) => (None, ValueOrigin::Default),
        };
        if let Some(v) = value {
            *slot = v;
        }
        origins.push((field, origin));
    };

    overlay(
        "stream_floor",
        &mut policy.stream_floor,
        document.as_ref().and_then(|d| d.stream_floor),
        env.stream_floor,
        cli.stream_floor,
    );
    overlay(
        "bridge_group_count",
        &mut policy.bridge_group_count,
        document.as_ref().and_then(|d| d.bridge_group_count),
        env.bridge_group_count,
        cli.bridge_group_count,
    );

    validate_stream_topology(&policy)?;

    let digest = stream_topology_digest(&policy, &origins, sha_hex.as_deref());

    Ok(StreamTopologyResolution {
        policy,
        origins,
        policy_path: policy_path.map(str::to_string),
        policy_sha256: sha_hex,
        digest,
    })
}

/// Fingerprint of a resolved topology, its per-field ORIGINS, and the policy
/// file's sha256.
///
/// Same rule as [`ext_policy_digest`]: the origins are hashed too, because
/// "8 because the module documents it" and "8 because an operator wrote it"
/// are different provenance facts about the same number.
fn stream_topology_digest(
    p: &StreamTopologyPolicy,
    origins: &[(&'static str, ValueOrigin)],
    file_sha256: Option<&str>,
) -> u64 {
    use sha2::Digest as _;
    let mut hasher = sha2::Sha256::new();
    hasher.update(STREAM_TOPOLOGY_RULE_ID.as_bytes());
    hasher.update([0u8]);
    for v in [p.schema_version, p.stream_floor, p.bridge_group_count] {
        hasher.update(v.to_be_bytes());
    }
    for (field, origin) in origins {
        hasher.update(field.as_bytes());
        hasher.update([0u8]);
        hasher.update(origin.as_str().as_bytes());
        hasher.update([0u8]);
    }
    hasher.update(file_sha256.unwrap_or("no_policy_file").as_bytes());
    let out = hasher.finalize();
    let mut lead = [0u8; 8];
    lead.copy_from_slice(&out[..8]);
    u64::from_be_bytes(lead)
}

// ═══════════════════════════════════════════════════════════════════════════
// REACHABILITY AND PARITY TESTS
//
// These run in the `--lib` target, so they execute on every `cargo test -p
// prism-nhs --lib` without depending on any binary target. Their point is not
// coverage: it is that the compiled device evaluator is genuinely REACHED from
// this crate. `abi_and_link_probe_bind_the_compiled_translation_unit` cannot
// pass — cannot even link — unless `src/cuda/physical_governor.cu` was
// compiled by `build.rs` and archived into `libphysical_governor.a`.
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn all_clear() -> ExtObservation {
        ExtObservation {
            schema_version: PHYSGOV_SCHEMA_VERSION,
            hmr_applied_verified: PERMISSION_GRANTED,
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

    fn inputs() -> BaseInputs {
        BaseInputs {
            gear_dt_fs: [0.5, 2.0, 4.0, 0.0],
            chunk_span_steps: 1,
            policy_digest: 0xDEAD_BEEF_CAFE_F00D,
            current_gear: GEAR_SAFE,
            requested_gear: GEAR_CAMPAIGN,
            force_measured: 1,
            force_norm: 100.0,
        }
    }

    fn permissive_base() -> BaseDecision {
        BaseDecision {
            authorized_gear: GEAR_CAMPAIGN,
            ceiling_gear: GEAR_CAMPAIGN,
            binding_gate: BASE_GATE_NONE,
            gate_verdicts: 0,
            emergency_flags: 0,
            next_stable_dwell_steps: 500,
            next_steps_since_instability: 500,
            downshift_latency_steps: 1,
        }
    }

    fn armed() -> ExtPolicy {
        let mut p = ExtPolicy::default();
        p.armed = 1;
        p
    }

    // ─── Defect 1: the compiled translation unit is REACHED ────────────────

    /// The proof that `src/cuda/physical_governor.cu` is compiled AND linked.
    ///
    /// Every symbol this touches lives in `libphysical_governor.a`; if the
    /// `build.rs` registration were removed, this test would fail to LINK, not
    /// merely fail to pass. The sizes and offsets come back from the compiler
    /// that built the archive, so the header's `static_assert` wall is checked
    /// against this module's `const` pins on real data rather than by
    /// transcription.
    #[cfg(feature = "gpu")]
    #[test]
    fn abi_and_link_probe_bind_the_compiled_translation_unit() {
        let witness = verify_compiled_evaluator_abi()
            .expect("the compiled device evaluator must share this module's ABI");
        assert_eq!(witness.link_probe, 0x5047_5631);
        assert_eq!(witness.ext_policy_bytes, 208);
        assert_eq!(witness.ext_observation_bytes, 112);
        assert_eq!(witness.decision_bytes, 128);
        assert_eq!(witness.schema_version, PHYSGOV_SCHEMA_VERSION);
        assert_eq!(witness.offsets, PINNED_OFFSETS);
    }

    /// The documented defaults, read back out of the COMPILED evaluator.
    ///
    /// `defaults_match_cuda_header` parses the header as TEXT; this compares
    /// against the values the device would actually use, which is the stronger
    /// statement.
    #[cfg(feature = "gpu")]
    #[test]
    fn compiled_defaults_match_the_rust_mirror() {
        let c = device_default_ext_policy().expect("C default fill must succeed");
        let mut rust = ExtPolicy::default();
        // The C side deliberately leaves `armed` at 0 and the digest at 0.
        rust.armed = 0;
        rust.ext_policy_digest = 0;
        assert_eq!(
            c, rust,
            "the compiled evaluator's documented defaults must equal the Rust mirror"
        );
        assert_eq!(c.armed, 0, "a default-filled policy is never pre-armed");
    }

    /// Host mirror vs compiled evaluator, over the production base governor,
    /// on every scenario the parity table names plus the refusal paths.
    #[cfg(feature = "gpu")]
    #[test]
    fn host_mirror_and_compiled_evaluator_agree() {
        let mut policy = crate::gearbox::governor_default_policy().expect("C defaults");
        policy.armed = 1;
        policy.hmr_permission = crate::gearbox::GEAR_PERMISSION_GRANTED;
        policy.constraint_permission = crate::gearbox::GEAR_PERMISSION_GRANTED;

        let mut base_obs = crate::gearbox::GearGuardObservation::unmeasured(
            crate::gearbox::GEAR_SAFE,
            crate::gearbox::GEAR_CAMPAIGN,
            10_000,
        );
        base_obs.drift_measured = 1;
        base_obs.drift = 0.0;
        base_obs.force_measured = 1;
        base_obs.force_norm = 100.0;
        base_obs.constraint_measured = 1;
        base_obs.constraint_failures = 0;
        base_obs.stable_dwell_steps = 500;
        base_obs.steps_since_instability = 500;

        let mutations: &[(&str, fn(&mut ExtObservation))] = &[
            ("all_clear", |_| {}),
            ("hmr_unknown", |o| {
                o.hmr_applied_verified = PERMISSION_UNKNOWN
            }),
            ("hmr_denied", |o| o.hmr_applied_verified = PERMISSION_DENIED),
            ("accel_extreme", |o| o.max_accel_a_per_ps2 = 3000.0),
            ("accel_no_ok_gear", |o| o.max_accel_a_per_ps2 = 1.0e6),
            ("accel_nonfinite", |o| o.max_accel_a_per_ps2 = f32::NAN),
            ("gamma_dt_hold_band", |o| o.effective_gamma_per_ps = 40.0),
            ("gamma_dt_floor_band", |o| o.effective_gamma_per_ps = 1.0e6),
            ("temp_setpoint_hold", |o| o.instantaneous_temp_k = 350.0),
            ("temp_runaway", |o| o.instantaneous_temp_k = 1200.0),
            ("temp_nonfinite", |o| o.instantaneous_temp_k = f32::NAN),
            ("neighbor_saturated", |o| o.max_neighbor_count = 1024),
            ("integration_fault", |o| o.integration_faults = 3),
            ("integration_fault_saturated", |o| {
                o.integration_faults = U32_UNMEASURED
            }),
            ("clamped_force", |o| o.clamped_force_fraction = 1.0e-5),
            ("sentinel_overdue", |o| o.steps_since_sentinel = 20_000),
            ("low_confidence", |o| o.observation_confidence = 0.2),
            ("uv_burst_phase", |o| o.phase_id = 5),
            ("unknown_phase", |o| o.phase_id = 99),
            ("force_unmeasured", |o| o.force_rate_measured = 0),
            ("observation_schema_drift", |o| o.schema_version = 99),
        ];

        for policy_state in ["armed", "unarmed", "policy_schema_drift"] {
            let mut ext_policy = armed();
            match policy_state {
                "unarmed" => ext_policy.armed = 0,
                "policy_schema_drift" => ext_policy.schema_version = 99,
                _ => {}
            }
            for (name, mutate) in mutations {
                let mut obs = all_clear();
                mutate(&mut obs);

                let host = decide_with_base_governor(&policy, &base_obs, None, &ext_policy, &obs)
                    .expect("host compose");
                let compiled =
                    decide_on_compiled_evaluator(&policy, &base_obs, None, &ext_policy, &obs)
                        .expect("compiled evaluator");

                let label = format!("{policy_state}/{name}");
                assert_eq!(
                    host.authorized_gear, compiled.authorized_gear,
                    "{label} authorized"
                );
                assert_eq!(host.ceiling_gear, compiled.ceiling_gear, "{label} ceiling");
                assert_eq!(
                    host.ext_binding_gate, compiled.ext_binding_gate,
                    "{label} gate"
                );
                assert_eq!(
                    host.binding_gate_rank, compiled.binding_gate_rank,
                    "{label} rank"
                );
                assert_eq!(
                    host.ext_gate_verdicts, compiled.ext_gate_verdicts,
                    "{label} verdicts"
                );
                assert_eq!(
                    host.ext_emergency_flags, compiled.ext_emergency_flags,
                    "{label} ext emergency"
                );
                assert_eq!(host.transition, compiled.transition, "{label} transition");
                assert_eq!(
                    host.ext_binding_refusal, compiled.ext_binding_refusal,
                    "{label} refusal cause"
                );
                assert_eq!(
                    host.next_stable_dwell_steps, compiled.next_stable_dwell_steps,
                    "{label} dwell"
                );
                assert_eq!(
                    host.dt_applied_ps.is_nan(),
                    compiled.dt_applied_ps.is_nan(),
                    "{label} dt nan-ness"
                );
                if host.dt_applied_ps.is_finite() {
                    assert!(
                        (host.dt_applied_ps - compiled.dt_applied_ps).abs() < 1e-12,
                        "{label} dt {} vs {}",
                        host.dt_applied_ps,
                        compiled.dt_applied_ps
                    );
                }
                assert_eq!(host.decision_emitted, 1, "{label} must emit");
                assert_eq!(compiled.decision_emitted, 1, "{label} must emit");
            }
        }
    }

    // ─── Defect 2: dwell resets on an EXTENSION-driven gear change ─────────

    #[test]
    fn extension_driven_gear_change_resets_the_dwell_counter() {
        // Running at the campaign gear, every guard clean, but the protocol
        // phase pins gear 1. The base authorized gear 2 and carried 500 steps
        // of dwell earned AT GEAR 2. After the extension downshift those 500
        // steps are evidence about a gear the run is no longer in.
        let mut i = inputs();
        i.current_gear = GEAR_CAMPAIGN;
        let mut obs = all_clear();
        obs.phase_id = 1; // ramp_up -> ceiling gear 1

        let d = compose(&i, &armed(), &obs, &permissive_base());
        assert_eq!(d.authorized_gear, GEAR_SAFE, "the phase ceiling must bind");
        assert_eq!(d.transition, TRANSITION_DOWNSHIFT);
        assert_eq!(
            d.next_stable_dwell_steps, 0,
            "dwell earned at gear 2 must not license an upshift out of gear 1"
        );

        // A HOLD at the same gear keeps the base's dwell: no gear changed, so
        // the evidence is still about the gear in force.
        let mut held = inputs();
        held.current_gear = GEAR_SAFE;
        held.requested_gear = GEAR_SAFE;
        let held_base = BaseDecision {
            authorized_gear: GEAR_SAFE,
            ceiling_gear: GEAR_SAFE,
            ..permissive_base()
        };
        let h = compose(&held, &armed(), &all_clear(), &held_base);
        assert_eq!(h.transition, TRANSITION_HOLD);
        assert_eq!(
            h.next_stable_dwell_steps, 500,
            "a hold accrues dwell normally"
        );
    }

    // ─── Defect 3: the documented bands GOVERN a ceiling ───────────────────

    #[test]
    fn gamma_dt_hold_and_downshift_bands_move_the_ceiling() {
        // Gear table 0.5 / 2.0 / 4.0 fs, current gear 1 (0.002 ps).
        // gamma chosen so gamma*dt(gear0) is already past the OK band, which is
        // where the old code FLOORed unconditionally.
        //
        //   gamma =    40 -> gamma*dt(cur) = 0.080 in [hold 0.10)? no: 0.08 in
        //                     [ok 0.05, hold 0.10) -> HOLD  -> ceiling = cur
        //   gamma =   100 -> 0.200 in [hold 0.10, downshift 0.25) -> DOWNSHIFT
        //   gamma = 1_000 -> 2.000 >= downshift -> FLOOR
        let mut i = inputs();
        i.current_gear = GEAR_SAFE;
        for (gamma, expect_ceiling, expect_verdict) in [
            (40.0_f32, GEAR_SAFE, VERDICT_HOLD),
            (100.0, GEAR_FINE, VERDICT_DOWNSHIFT),
            (1000.0, GEAR_FINE, VERDICT_FLOOR),
        ] {
            let mut obs = all_clear();
            obs.effective_gamma_per_ps = gamma;
            let d = compose(&i, &armed(), &obs, &permissive_base());
            assert_eq!(
                d.ceiling_gear, expect_ceiling,
                "gamma={gamma}: ceiling {} != {expect_ceiling}",
                d.ceiling_gear
            );
            assert_eq!(
                d.ext_verdict(GATE_THERMOSTAT_STABILITY),
                expect_verdict,
                "gamma={gamma} verdict"
            );
            assert_ne!(d.authorized_gear, GEAR_ABORT, "gamma*dt never aborts");
        }
    }

    #[test]
    fn accel_hold_and_downshift_bands_move_the_ceiling() {
        // dx(g) = 0.5 * a * dt(g)^2, reference 1.0 A, ok 0.02 / hold 0.05 /
        // downshift 0.10. Gear table 0.5 / 2.0 / 4.0 fs, so
        // dx(0) = 1.25e-7*a, dx(1) = 2.0e-6*a, dx(2) = 8.0e-6*a.
        //
        // THE LOAD-BEARING CASE. At the 2.0 fs gear with a = 1.75e4:
        //   dx(0) = 2.19e-3  -> OK        gear 0 unconstrained
        //   dx(1) = 0.035    -> HOLD      gear 1 permitted up to cur == 1
        //   dx(2) = 0.14     -> FLOOR     gear 2 refused
        // The documented answer is HOLD AT GEAR 1. An OK-band-only scan
        // publishes gear 0 and forces a downshift the physics did not ask for —
        // that is `accel_hold_fraction` governing a ceiling instead of
        // decorating a receipt.
        let mut at_safe = inputs();
        at_safe.current_gear = GEAR_SAFE;
        let mut obs = all_clear();
        obs.max_accel_a_per_ps2 = 1.75e4;
        let d = compose(&at_safe, &armed(), &obs, &permissive_base());
        assert_eq!(
            d.ceiling_gear, GEAR_SAFE,
            "the HOLD band permits the gear in force; an OK-only scan would say {}",
            GEAR_FINE
        );
        assert_eq!(d.ext_verdict(GATE_ACCELERATION_EXTREME), VERDICT_HOLD);
        assert_eq!(d.transition, TRANSITION_HOLD, "HOLD is not a downshift");

        // And the DOWNSHIFT band one gear further out: a = 3.0e4 puts
        // dx(1) = 0.06 into [hold 0.05, downshift 0.10), so gear 1 is permitted
        // only BELOW cur — the ceiling drops to gear 0.
        let mut obs = all_clear();
        obs.max_accel_a_per_ps2 = 3.0e4;
        let d = compose(&at_safe, &armed(), &obs, &permissive_base());
        assert_eq!(
            d.ceiling_gear, GEAR_FINE,
            "the DOWNSHIFT band pins one finer"
        );
        assert_eq!(d.ext_verdict(GATE_ACCELERATION_EXTREME), VERDICT_DOWNSHIFT);
        assert_eq!(d.transition, TRANSITION_DOWNSHIFT);

        // At the finest gear every band still resolves, and none of them aborts
        // below the emergency fraction.
        let mut at_fine = inputs();
        at_fine.current_gear = GEAR_FINE;
        for (a, expect_verdict) in [
            (2.0e5_f32, VERDICT_HOLD),  // dx(0) = 0.025 in [0.02, 0.05)
            (5.0e5, VERDICT_DOWNSHIFT), // dx(0) = 0.0625 in [0.05, 0.10)
            (1.0e6, VERDICT_FLOOR),     // dx(0) = 0.125 >= 0.10
        ] {
            let mut obs = all_clear();
            obs.max_accel_a_per_ps2 = a;
            let d = compose(&at_fine, &armed(), &obs, &permissive_base());
            assert_eq!(d.ceiling_gear, GEAR_FINE, "a={a} ceiling");
            assert_eq!(
                d.ext_verdict(GATE_ACCELERATION_EXTREME),
                expect_verdict,
                "a={a} verdict"
            );
            assert_ne!(
                d.authorized_gear, GEAR_ABORT,
                "a={a} is below the abort band"
            );
        }
    }

    // ─── Defect 4: the EMERGENCY class the header documents is implemented ─

    #[test]
    fn threefold_temperature_runaway_traps() {
        let mut obs = all_clear();
        obs.target_temp_k = 300.0;
        obs.instantaneous_temp_k = 1200.0; // dev = 3.0 >= 2.0
        let d = compose(&inputs(), &armed(), &obs, &permissive_base());
        assert_eq!(d.authorized_gear, GEAR_ABORT, "a 3x runaway must trap");
        assert_eq!(d.transition, TRANSITION_ABORT);
        assert_eq!(
            d.ext_emergency_flags & EMERG_TEMPERATURE_RUNAWAY,
            EMERG_TEMPERATURE_RUNAWAY
        );
        assert_eq!(d.ext_binding_gate, GATE_THERMOSTAT_STABILITY);
        assert!(d.dt_applied_ps.is_nan());

        // A NON-FINITE temperature is an absent measurement, not a proven
        // runaway: FLOOR, never a fabricated emergency.
        let mut nan_obs = all_clear();
        nan_obs.instantaneous_temp_k = f32::NAN;
        let n = compose(&inputs(), &armed(), &nan_obs, &permissive_base());
        assert_ne!(
            n.authorized_gear, GEAR_ABORT,
            "NaN is not evidence of a runaway"
        );
        assert_eq!(n.authorized_gear, GEAR_FINE);
        assert_eq!(n.ext_emergency_flags & EMERG_TEMPERATURE_RUNAWAY, 0);
    }

    #[test]
    fn absent_setpoint_is_unmeasured_never_a_fabricated_denominator() {
        // REGRESSION. `gate_thermostat` used to substitute a literal 1.0 K for
        // an absent or non-physical `target_temp_k` and then use that
        // substituted constant as the DENOMINATOR of
        // `dev = |T - target| / target`. At any real MD temperature that makes
        // `dev` ~299, which clears every band including
        // `temp_dev_abort_fraction`, so the run TRAPPED on
        // EMERG_TEMPERATURE_RUNAWAY while the instrument had observed no
        // runaway at all — it had observed no setpoint.
        for absent in [f32::NAN, f32::INFINITY, 0.0f32, -300.0f32, 1.0f32] {
            let mut obs = all_clear();
            obs.target_temp_k = absent;
            obs.instantaneous_temp_k = 300.0; // perfectly healthy
            let d = compose(&inputs(), &armed(), &obs, &permissive_base());
            assert_ne!(
                d.authorized_gear, GEAR_ABORT,
                "target_temp_k={absent:?}: an absent setpoint must not trap the run"
            );
            assert_eq!(
                d.ext_emergency_flags & EMERG_TEMPERATURE_RUNAWAY,
                0,
                "target_temp_k={absent:?}: no runaway may be claimed without a setpoint"
            );
            assert_eq!(
                d.ext_verdict(GATE_THERMOSTAT_STABILITY),
                VERDICT_UNMEASURED,
                "target_temp_k={absent:?}: the deviation guard has no denominator"
            );
        }

        // ...and it is still not a LICENCE: with the guard required for
        // upshift, an unmeasured setpoint cannot authorise a coarser gear.
        let mut policy = armed();
        policy.require_thermostat_guard_for_upshift = 1;
        let mut at_fine = inputs();
        at_fine.current_gear = GEAR_FINE;
        let mut obs = all_clear();
        obs.target_temp_k = f32::NAN;
        obs.instantaneous_temp_k = 300.0;
        let d = compose(&at_fine, &policy, &obs, &permissive_base());
        assert_eq!(
            d.authorized_gear, GEAR_FINE,
            "an unmeasured required guard must not license an upshift"
        );

        // A REAL setpoint still produces the documented findings, so the fix
        // did not disable the guard.
        let mut healthy = all_clear();
        healthy.target_temp_k = 300.0;
        healthy.instantaneous_temp_k = 1200.0;
        let t = compose(&inputs(), &armed(), &healthy, &permissive_base());
        assert_eq!(
            t.ext_emergency_flags & EMERG_TEMPERATURE_RUNAWAY,
            EMERG_TEMPERATURE_RUNAWAY,
            "a measured 3x runaway must still trap"
        );
    }

    #[test]
    fn gamma_dt_abort_governs_the_instability_flag() {
        // gamma*dt(cur) >= gamma_dt_abort (1.0) must reset the lockout window,
        // which is a governed state transition — not a decorative receipt.
        let mut i = inputs();
        i.current_gear = GEAR_SAFE;
        let mut obs = all_clear();
        obs.effective_gamma_per_ps = 1000.0; // gamma*dt(cur) = 2.0 >= 1.0
        let d = compose(&i, &armed(), &obs, &permissive_base());
        assert_ne!(d.authorized_gear, GEAR_ABORT, "FLOOR, never EMERGENCY");
        assert_eq!(
            d.next_steps_since_instability, 0,
            "a fully-resampled velocity is an instability, and the lockout must restart"
        );

        // Just below the abort band the lockout window keeps running.
        let mut mild = all_clear();
        mild.effective_gamma_per_ps = 100.0; // gamma*dt(cur) = 0.2 < 1.0
        let m = compose(&i, &armed(), &mild, &permissive_base());
        assert_eq!(m.next_steps_since_instability, 500);
    }

    #[test]
    fn the_abort_thresholds_are_policy_resolved_and_receipted() {
        let r = resolve_ext_policy(None, None).expect("defaults resolve");
        let lines = r.receipt_lines();
        for field in ["gamma_dt_abort", "temp_dev_abort_fraction", "n_phase_slots"] {
            assert!(
                lines.iter().any(|l| l.starts_with(&format!("{field}="))),
                "{field} governs a decision and must carry a receipt"
            );
            assert!(
                r.origin_of(field).is_some(),
                "{field} must have a recorded origin"
            );
        }
        // Every receipted field is a field with a recorded origin, and every
        // origin is receipted: no false receipts, no silent thresholds.
        for field in ExtPolicyResolution::receipted_fields() {
            assert!(
                r.origin_of(field).is_some(),
                "{field} is receipted but has no origin"
            );
        }
        assert_eq!(
            r.origins.len(),
            ExtPolicyResolution::receipted_fields().len(),
            "every resolved field must appear in exactly one receipt line"
        );
    }

    #[test]
    fn abort_bands_must_be_monotone_and_the_sentinel_rule_cannot_be_disabled() {
        let mut p = ExtPolicy::default();
        p.gamma_dt_abort = 0.1; // below gamma_dt_downshift
        assert!(matches!(
            validate_ext_policy(&p),
            Err(ExtPolicyError::InvalidValue {
                field: "gamma_dt_abort",
                ..
            })
        ));

        let mut p = ExtPolicy::default();
        p.temp_dev_abort_fraction = 0.2; // below temp_dev_downshift_fraction
        assert!(matches!(
            validate_ext_policy(&p),
            Err(ExtPolicyError::InvalidValue {
                field: "temp_dev_abort_fraction",
                ..
            })
        ));

        let mut p = ExtPolicy::default();
        p.max_steps_without_sentinel = 0;
        assert!(matches!(
            validate_ext_policy(&p),
            Err(ExtPolicyError::InvalidValue {
                field: "max_steps_without_sentinel",
                ..
            })
        ));

        let mut p = ExtPolicy::default();
        p.n_phase_slots = 0;
        assert!(matches!(
            validate_ext_policy(&p),
            Err(ExtPolicyError::InvalidValue {
                field: "n_phase_slots",
                ..
            })
        ));

        validate_ext_policy(&ExtPolicy::default()).expect("the defaults must validate");
    }

    // ─── Defect 5: a zeroed policy FAILS CLOSED with an honest attribution ──

    #[test]
    fn a_zeroed_policy_fails_closed_and_is_never_reported_as_physics() {
        // The exact scenario the audit named: a caller memsets the policy and
        // forgets to upload the resolved one. Every threshold is then 0.0.
        let zeroed = ExtPolicy {
            schema_version: 0,
            armed: 0,
            require_accel_guard_for_upshift: 0,
            require_thermostat_guard_for_upshift: 0,
            require_integration_guard_for_upshift: 0,
            require_observation_confidence: 0,
            max_steps_without_sentinel: 0,
            n_phase_slots: 0,
            force_rate_hold_per_ps: 0.0,
            force_rate_downshift_per_ps: 0.0,
            force_rate_floor_per_ps: 0.0,
            force_rate_norm_floor: 0.0,
            accel_displacement_reference_a: 0.0,
            accel_ok_fraction: 0.0,
            accel_hold_fraction: 0.0,
            accel_downshift_fraction: 0.0,
            accel_abort_fraction: 0.0,
            gamma_dt_ok: 0.0,
            gamma_dt_hold: 0.0,
            gamma_dt_downshift: 0.0,
            gamma_dt_abort: 0.0,
            temp_dev_ok_fraction: 0.0,
            temp_dev_hold_fraction: 0.0,
            temp_dev_downshift_fraction: 0.0,
            temp_dev_abort_fraction: 0.0,
            neighbor_occupancy_ok: 0.0,
            neighbor_occupancy_hold: 0.0,
            clamped_fraction_hold: 0.0,
            clamped_fraction_downshift: 0.0,
            clamped_fraction_floor: 0.0,
            tsh_memory_hold: 0.0,
            tsh_memory_floor: 0.0,
            min_observation_confidence: 0.0,
            max_observation_interval_ps: 0.0,
            phase_ceiling_gear: [0; 8],
            ext_policy_digest: 0,
            _reserved: [0; 8],
        };
        let d = compose(&inputs(), &zeroed, &all_clear(), &permissive_base());

        assert_eq!(
            d.authorized_gear, DEFAULT_UNARMED_CAP_GEAR,
            "an unarmed extension must fail closed, not silently permit"
        );
        assert_eq!(d.ext_binding_gate, GATE_EXT_BINDING);
        assert_eq!(d.binding_gate_rank, RANK_GOVERNOR_UNARMED);
        assert_eq!(d.ext_binding_refusal, REFUSAL_UNARMED);
        assert_eq!(d.ext_armed, 0);
        assert_eq!(d.decision_emitted, 1, "a refusal is still an emitted fact");
        // The crucial property: NO physics gate may claim this outcome.
        for gate in [
            GATE_FORCE_SPIKE,
            GATE_ACCELERATION_EXTREME,
            GATE_THERMOSTAT_STABILITY,
            GATE_INTEGRATION_HEALTH,
            GATE_TSH_MEMORY,
            GATE_PHASE_REQUIREMENT,
            GATE_HMR_HARD,
        ] {
            assert_eq!(
                d.ext_verdict(gate),
                VERDICT_UNMEASURED,
                "gate {gate} must not manufacture a finding from an unarmed policy"
            );
        }
        assert_eq!(
            d.ext_emergency_flags, 0,
            "an unarmed policy is not an emergency"
        );
        // And it must not deadlock: the finest gear stays reachable.
        assert!(
            d.dt_applied_ps.is_finite(),
            "the capped gear must have a timestep"
        );
    }

    #[test]
    fn a_stale_schema_is_refused_with_its_own_named_cause() {
        let mut p = armed();
        p.schema_version = 0; // a v0 policy under a v1 build
        let d = compose(&inputs(), &p, &all_clear(), &permissive_base());
        assert_eq!(d.ext_binding_refusal, REFUSAL_POLICY_SCHEMA);
        assert_eq!(d.ext_binding_gate, GATE_EXT_BINDING);
        assert_eq!(d.authorized_gear, DEFAULT_UNARMED_CAP_GEAR);

        let mut o = all_clear();
        o.schema_version = 0;
        let d = compose(&inputs(), &armed(), &o, &permissive_base());
        assert_eq!(d.ext_binding_refusal, REFUSAL_OBSERVATION_SCHEMA);
        assert_eq!(d.ext_binding_gate, GATE_EXT_BINDING);

        // An armed, current-schema pair binds and records no refusal.
        let d = compose(&inputs(), &armed(), &all_clear(), &permissive_base());
        assert_eq!(d.ext_binding_refusal, REFUSAL_NONE);
        assert_ne!(d.ext_binding_gate, GATE_EXT_BINDING);
    }

    // ─── Defect 6: the governor publishes a CEILING, and the abort sentinel
    //     cannot be erased by the documented `min` ─────────────────────────

    #[test]
    fn the_abort_sentinel_survives_the_consumer_min() {
        for base in [
            BaseDecision {
                authorized_gear: GEAR_ABORT,
                ceiling_gear: GEAR_ABORT,
                binding_gate: BASE_GATE_PHYSICAL_EMERGENCY,
                emergency_flags: 0x4,
                ..permissive_base()
            },
            permissive_base(),
        ] {
            let mut obs = all_clear();
            let aborting = base.binding_gate == BASE_GATE_PHYSICAL_EMERGENCY;
            if !aborting {
                obs.integration_faults = 3; // extension emergency instead
            }
            let d = compose(&inputs(), &armed(), &obs, &base);
            assert_eq!(d.authorized_gear, GEAR_ABORT);
            assert_eq!(d.transition, TRANSITION_ABORT);
            assert!(d.dt_applied_ps.is_nan());
            // The documented consumer contract is `min(request, ceiling)`.
            // GEAR_ABORT in the ceiling would be ERASED by that min.
            assert!(
                d.ceiling_gear < GEAR_ABORT,
                "an abort ceiling of {} would be erased by min(request, ceiling)",
                d.ceiling_gear
            );
            let consumer_gear = d.requested_gear.min(d.ceiling_gear);
            assert!(
                consumer_gear <= GEAR_FINE,
                "a min-only consumer must be driven to the finest gear on an abort, got {consumer_gear}"
            );
        }
    }

    #[test]
    fn the_ceiling_is_a_ceiling_and_never_a_chosen_gear() {
        // Across a grid of requests, the published ceiling must depend on the
        // GUARDS and not on what the science asked for: a ceiling that moved
        // with the request would be a chosen gear wearing a ceiling's name.
        let mut reference: Option<u32> = None;
        for requested in [GEAR_FINE, GEAR_SAFE, GEAR_CAMPAIGN, GEAR_ABORT, 99] {
            let mut i = inputs();
            i.requested_gear = requested;
            let d = compose(&i, &armed(), &all_clear(), &permissive_base());
            assert_eq!(
                d.requested_gear, requested,
                "the request must survive verbatim"
            );
            match reference {
                None => reference = Some(d.ceiling_gear),
                Some(r) => assert_eq!(
                    d.ceiling_gear, r,
                    "the ceiling moved with the request (requested={requested})"
                ),
            }
            assert!(
                d.authorized_gear <= d.ceiling_gear,
                "authorized {} exceeded the ceiling {}",
                d.authorized_gear,
                d.ceiling_gear
            );
        }
    }

    // ─── The force-spike guard must not delegate its veto to the base ──────

    #[test]
    fn an_unmeasured_force_pair_blocks_the_upshift_on_its_own() {
        let mut obs = all_clear();
        obs.force_rate_measured = 0;
        let d = compose(&inputs(), &armed(), &obs, &permissive_base());
        assert!(
            d.authorized_gear <= inputs().current_gear,
            "4.0 fs with no force evidence at all is not a verified-safe claim"
        );
        assert_eq!(d.ext_verdict(GATE_FORCE_SPIKE), VERDICT_UNMEASURED);
    }

    #[test]
    fn a_saturated_fault_counter_is_still_a_fault() {
        let mut obs = all_clear();
        obs.integration_faults = U32_UNMEASURED; // a saturating counter at max
        let d = compose(&inputs(), &armed(), &obs, &permissive_base());
        assert_eq!(
            d.authorized_gear, GEAR_ABORT,
            "0xFFFFFFFF faults is the most violent state representable, \
             not an absence of measurement"
        );
        assert_eq!(
            d.ext_emergency_flags & EMERG_INTEGRATION_FAULT,
            EMERG_INTEGRATION_FAULT
        );
    }

    #[test]
    fn the_force_rate_numerator_is_not_clamped_by_the_denominator_floor() {
        // prev = 0.1, floor = 1.0, now = 2.0. True relative growth is
        // (2.0-0.1)/(1.0*dt) = 1.9/dt; a clamped numerator would report
        // (2.0-1.0)/(1.0*dt) = 1.0/dt and under-report by ~2x — toward a
        // COARSER gear.
        let mut i = inputs();
        i.force_norm = 2.0;
        let mut obs = all_clear();
        obs.force_norm_prev = 0.1;
        obs.dt_prev_ps = 0.002;
        // rate = 1.9 / (1.0 * 0.002) = 950 /ps -> past the FLOOR band (250).
        let d = compose(&i, &armed(), &obs, &permissive_base());
        assert_eq!(d.ext_verdict(GATE_FORCE_SPIKE), VERDICT_FLOOR);
        assert_eq!(d.authorized_gear, GEAR_FINE);
    }

    #[test]
    fn base_inputs_default_fabricates_no_gear_table() {
        let d = BaseInputs::default();
        for gear in 0..=GEAR_MAX_ACTIVE {
            assert!(
                d.gear_dt_ps(gear).is_none(),
                "gear {gear} must have no timestep until a real policy resolves one"
            );
        }
        assert_eq!(d.force_measured, 0);
        assert!(d.force_norm.is_nan());
    }

    // ─── Stream topology: policy-resolved rows, no source-literal gate ──────

    fn no_overrides() -> StreamTopologyOverrides {
        StreamTopologyOverrides::default()
    }

    fn scratch_policy(name: &str, body: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("prism_stream_topology_test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join(format!("{}_{}.json", name, std::process::id()));
        std::fs::write(&path, body).expect("write policy");
        path
    }

    #[test]
    fn stream_topology_defaults_resolve_and_every_row_is_receipted() {
        let r = resolve_stream_topology_layered(None, None, no_overrides(), no_overrides())
            .expect("defaults resolve");
        assert_eq!(r.policy.stream_floor, DEFAULT_STREAM_FLOOR);
        assert_eq!(r.policy.bridge_group_count, DEFAULT_BRIDGE_GROUP_COUNT);
        assert_ne!(
            r.digest, 0,
            "the resolved topology must carry a fingerprint"
        );
        let lines = r.receipt_lines();
        for field in StreamTopologyResolution::receipted_fields() {
            assert_eq!(r.origin_of(field), Some(ValueOrigin::Default), "{field}");
            assert!(
                lines.iter().any(|l| l.starts_with(&format!("{field}="))),
                "{field} governs an admission and must carry a receipt"
            );
        }
        assert_eq!(
            r.origins.len(),
            StreamTopologyResolution::receipted_fields().len(),
            "every resolved row must appear in exactly one receipt line"
        );
        assert!(lines.iter().all(|l| l.contains("origin=")));
    }

    #[test]
    fn stream_topology_cli_beats_env_beats_file_and_each_layer_is_named() {
        let path = scratch_policy(
            "layered",
            r#"{"schema_version":1,"stream_floor":16,"bridge_group_count":4}"#,
        );
        let p = path.to_str().expect("utf-8 path");

        let file_only =
            resolve_stream_topology_layered(Some(p), None, no_overrides(), no_overrides())
                .expect("file resolves");
        assert_eq!(file_only.policy.stream_floor, 16);
        assert_eq!(
            file_only.origin_of("stream_floor"),
            Some(ValueOrigin::PolicyFile)
        );
        assert_eq!(file_only.policy_sha256.as_ref().map(String::len), Some(64));

        let env = StreamTopologyOverrides {
            stream_floor: Some(32),
            bridge_group_count: None,
        };
        let with_env = resolve_stream_topology_layered(Some(p), None, env, no_overrides())
            .expect("env resolves");
        assert_eq!(with_env.policy.stream_floor, 32);
        assert_eq!(
            with_env.origin_of("stream_floor"),
            Some(ValueOrigin::Environment)
        );
        assert_eq!(
            with_env.origin_of("bridge_group_count"),
            Some(ValueOrigin::PolicyFile),
            "a layer that said nothing must not claim authorship"
        );

        // Precedence is demonstrated on `stream_floor`, the row that is
        // genuinely tunable. `bridge_group_count` is restated at its structural
        // width rather than varied: it is pinned to the compiled emitter arity
        // (see `a_non_structural_divisor_is_refused_before_any_lease`), so
        // varying it here would be testing precedence with a value the
        // resolver must refuse.
        let cli = StreamTopologyOverrides {
            stream_floor: Some(64),
            bridge_group_count: Some(BRIDGE_GROUP_STRUCTURAL_WIDTH),
        };
        let with_cli =
            resolve_stream_topology_layered(Some(p), None, env, cli).expect("cli resolves");
        assert_eq!(with_cli.policy.stream_floor, 64);
        assert_eq!(
            with_cli.policy.bridge_group_count,
            BRIDGE_GROUP_STRUCTURAL_WIDTH
        );
        assert_eq!(with_cli.origin_of("stream_floor"), Some(ValueOrigin::Cli));
        assert_eq!(
            with_cli.origin_of("bridge_group_count"),
            Some(ValueOrigin::Cli),
            "a pinned row still records which layer authored the value"
        );
        assert_ne!(
            with_cli.digest, with_env.digest,
            "a different resolved topology must fingerprint differently"
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn the_eight_stream_equality_is_no_longer_the_rule() {
        let r = resolve_stream_topology_layered(None, None, no_overrides(), no_overrides())
            .expect("defaults resolve");

        let eight = r.admits(8);
        assert!(
            eight.admitted,
            "the historical topology must still be admitted"
        );
        assert_eq!(eight.streams_per_bridge_group, 2);
        assert_eq!(eight.reason, STREAM_TOPOLOGY_ADMITTED);

        let sixteen = r.admits(16);
        assert!(
            sixteen.admitted,
            "a 16-stream topology must be admissible without a source edit"
        );
        assert_eq!(sixteen.streams_per_bridge_group, 4);

        let six = r.admits(6);
        assert!(!six.admitted);
        assert_eq!(six.reason, STREAM_TOPOLOGY_BELOW_FLOOR);
        assert!(
            six.detail.contains("origin=default"),
            "a refusal must name the authority that refused"
        );

        let ten = r.admits(10);
        assert!(!ten.admitted, "10 streams cannot be split across 4 groups");
        assert_eq!(ten.reason, STREAM_TOPOLOGY_UNEVEN_BRIDGE_GROUPS);
        assert_eq!(ten.streams_per_bridge_group, 0);
    }

    #[test]
    /// The structural divisor must equal the width the emitter is compiled
    /// against. If `twin_differential_set()` ever returns a different arity,
    /// this fails here rather than as an out-of-bounds index mid-run.
    #[test]
    #[cfg(feature = "gpu")]
    fn the_bridge_group_width_matches_the_compiled_protocol_set() {
        assert_eq!(
            crate::fused_engine::CryoUvProtocol::twin_differential_set().len(),
            BRIDGE_GROUP_STRUCTURAL_WIDTH as usize,
            "the multi-differential emitter allocates per-group accumulators at \
             the arity of twin_differential_set(); BRIDGE_GROUP_STRUCTURAL_WIDTH \
             must track it or the admission gate will admit a stream count that \
             indexes out of bounds after the device lease is paid"
        );
    }

    /// A divisor other than the compiled width is refused at RESOLUTION, which
    /// is the whole point: `PRISM_BRIDGE_GROUP_COUNT=2` with `--multi-stream 10`
    /// satisfies `10 % 2 == 0` and would otherwise reach the emitter, where
    /// `epg_val = (10/4).max(1) = 2` makes stream 9 index group 4 against
    /// four-wide accumulators.
    #[test]
    fn a_non_structural_divisor_is_refused_before_any_lease() {
        for divisor in [1u32, 2, 3, 5, 8, 16] {
            let cli = StreamTopologyOverrides {
                stream_floor: None,
                bridge_group_count: Some(divisor),
            };
            assert!(
                matches!(
                    resolve_stream_topology_layered(None, None, no_overrides(), cli),
                    Err(StreamTopologyError::InvalidValue {
                        field: "bridge_group_count",
                        ..
                    })
                ),
                "divisor {divisor} is not the compiled emitter width and must be \
                 refused at resolution, not discovered as a panic mid-run"
            );
        }

        let cli = StreamTopologyOverrides {
            stream_floor: None,
            bridge_group_count: Some(BRIDGE_GROUP_STRUCTURAL_WIDTH),
        };
        assert!(
            resolve_stream_topology_layered(None, None, no_overrides(), cli).is_ok(),
            "the structural width itself must still resolve, with origin=Cli"
        );
    }

    #[test]
    fn an_incoherent_topology_is_refused_at_resolution_not_after_the_lease() {
        let cli = StreamTopologyOverrides {
            stream_floor: Some(6),
            bridge_group_count: Some(4),
        };
        assert!(
            matches!(
                resolve_stream_topology_layered(None, None, no_overrides(), cli),
                Err(StreamTopologyError::InvalidValue {
                    field: "stream_floor",
                    ..
                })
            ),
            "a floor that itself fails the divisibility rule admits nothing"
        );

        let cli = StreamTopologyOverrides {
            stream_floor: None,
            bridge_group_count: Some(0),
        };
        assert!(matches!(
            resolve_stream_topology_layered(None, None, no_overrides(), cli),
            Err(StreamTopologyError::InvalidValue {
                field: "bridge_group_count",
                ..
            })
        ));

        validate_stream_topology(&StreamTopologyPolicy::default())
            .expect("the defaults must validate");
    }

    #[test]
    fn a_stream_topology_that_changed_under_the_run_is_refused() {
        let path = scratch_policy("pinned", r#"{"schema_version":1,"stream_floor":16}"#);
        let p = path.to_str().expect("utf-8 path");
        let wrong = "0".repeat(64);
        assert!(matches!(
            resolve_stream_topology_layered(Some(p), Some(&wrong), no_overrides(), no_overrides()),
            Err(StreamTopologyError::DigestMismatch { .. })
        ));

        let stale = scratch_policy("stale_schema", r#"{"schema_version":99,"stream_floor":16}"#);
        let s = stale.to_str().expect("utf-8 path");
        assert!(matches!(
            resolve_stream_topology_layered(Some(s), None, no_overrides(), no_overrides()),
            Err(StreamTopologyError::SchemaMismatch { found: 99, .. })
        ));

        let unknown = scratch_policy("unknown_key", r#"{"schema_version":1,"max_streams":64}"#);
        let u = unknown.to_str().expect("utf-8 path");
        assert!(
            matches!(
                resolve_stream_topology_layered(Some(u), None, no_overrides(), no_overrides()),
                Err(StreamTopologyError::Parse { .. })
            ),
            "a key this build does not govern must fail loudly, not be ignored"
        );

        for f in [&path, &stale, &unknown] {
            let _ = std::fs::remove_file(f);
        }
    }

    #[test]
    fn the_stream_topology_receipt_carries_provenance_not_just_numbers() {
        let cli = StreamTopologyOverrides {
            stream_floor: Some(16),
            bridge_group_count: None,
        };
        let r =
            resolve_stream_topology_layered(None, None, no_overrides(), cli).expect("cli resolves");
        let receipt = r.receipt_json();
        assert_eq!(receipt["rule_id"], STREAM_TOPOLOGY_RULE_ID);
        assert_eq!(receipt["stream_floor"], 16);
        assert_eq!(receipt["stream_floor_origin"], "cli");
        assert_eq!(receipt["bridge_group_count_origin"], "default");
        assert_eq!(
            receipt["stream_topology_digest"]
                .as_str()
                .expect("digest hex")
                .len(),
            16
        );
    }
}
