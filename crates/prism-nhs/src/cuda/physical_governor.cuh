// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D — PHYSICAL STABILITY GOVERNOR (Package 3)
// Device-side inline evaluation.  Sole authority over what timestep is SAFE.
// ═══════════════════════════════════════════════════════════════════════════
//
// Operator directive 2026-08-02, `COUPLED_TEMPORAL_GEARBOX_TENSOR_ARCHITECTURE.md`
// §4 ("Role division — non-negotiable") and §5 ("Chunk-boundary reality").
//
// ─── Relationship to `gearbox.cuh` ─────────────────────────────────────────
//
// `gearbox.cuh` already carries a governor core (gates 1..10) with a PINNED
// wire format: `PrismGearGovernorPolicy` (144 B), `PrismGearGuardObservation`
// (64 B), `PrismGearDecision` (32 B), `PrismTshSalienceRequest` (32 B).
//
// This header does NOT restate any of that.  It `#include`s it and adds the
// guards the core does not have — force SPIKE, acceleration extremes,
// thermostat stability, neighbour-list / integration health, TSH memory with
// anti-blindness, protocol-phase requirement — plus the NON-POLICY-OVERRIDABLE
// HMR hard gate.  There is exactly ONE wire format for the shared fields and
// exactly ONE base decision function; duplicating either would create two
// sources of truth for a physics threshold, which is the failure mode this
// whole package exists to prevent.
//
// STRUCTURAL TIGHTEN-ONLY GUARANTEE.  The composed evaluator computes
//
//     authorized = min(base_decision.authorized_gear, extension_ceiling)
//
// with a single `min`.  The extension layer therefore CANNOT loosen the base
// governor's decision under any input, any policy, or any bug in the band
// arithmetic below: the worst an extension defect can do is refuse a gear that
// was physically available.  That asymmetry is deliberate — the failure
// direction of this file is "too conservative", never "authorized an unsafe
// timestep".
//
// ─── The nine guards (operator requirement 1) ──────────────────────────────
//
//  #  guard                    observable                              class    owner
//  1  topology HMR permission  PrismPrepTopology::hmr_applied          HARD     ext (gate 38) + base gate 2
//  2  constraint stability     SHAKE/RATTLE non-convergence count      GRADUATED base gate 3
//  3  Hamiltonian drift        |V_t-(V_{t-1}+W_ext)|/|V_{t-1}+W_ext|   GRADUATED base gate 4 (+ base gate 1 abort)
//  4  force norm               max|F| vs the integrator clamp          GRADUATED base gate 5
//  5  force spike              d|F|/dt normalised per simulated ps     GRADUATED ext gate 32
//  6  acceleration extremes    0.5*a_max*dt^2 vs the X–H bond length   PER-GEAR   ext gate 33
//  7  thermostat stability     gamma*dt (Langevin OU) + |dT|/T         PER-GEAR   ext gate 34
//  8  minimum stable dwell     continuous all-guards-OK steps          UPSHIFT-ONLY base gate 8
//     recent-instability mem.  steps since the last instability        UPSHIFT-ONLY base gate 9
//  9  neighbour-list /         max neighbour occupancy vs capacity,    GRADUATED ext gate 35
//     integration health       clamped-force fraction, fault count
//
//  plus:  TSH memory + anti-blindness   ext gate 36   (ceiling-only)
//         protocol-phase requirement    ext gate 37   (ceiling-only)
//         extension binding refusal     ext gate 39   (terminal, pre-guard)
//
// Gate 39 is the extension analogue of base gate 11.  It runs BEFORE any guard
// and replaces the whole ceiling fold: an unarmed or wrong-schema policy makes
// every threshold below meaningless, and evaluating the guards against it would
// manufacture a FLOOR on every gate and emit a record indistinguishable from
// genuinely violent physics.
//
// ─── Determinism (operator invariant) ──────────────────────────────────────
//
// Every observable this header consumes must be an ORDER-INVARIANT reduction:
// `fmaxf` maxima and integer counts.  Both are bit-exact regardless of the
// order in which partial results combine, so an `atomicMax` / `atomicAdd`-on-
// integers producer cannot make two identical runs take different gears.  The
// one field that is an f32 ratio rather than a maximum or a count —
// `clamped_force_fraction` — carries its own producer note on the observation
// struct; it must NOT be produced with a float `atomicAdd`.
// This is a deliberate design constraint, not an accident of the choice of
// statistics: the one quantity in the loop that genuinely needs a SUM — the
// Hamiltonian drift's potential energy — stays with the base gate's f64
// fixed-order path and is NOT recomputed here.  Do NOT add a float-SUM
// observable to `PrismPhysGovExtObservation` without also specifying a
// fixed-order reduction for it; `zstr_force_stage_f4_kernel`'s `atomicAdd`
// sum|F|^2 is exactly the pattern that must not be reused for a decision
// input, because float addition is not associative and the low bits can flip
// a threshold comparison between runs.  If a future guard genuinely needs a
// float SUM, take it from `crates/prism-nhs/src/cuda/deterministic_reduce.cuh`
// (Package 2), which provides fixed-order pairwise reductions whose summation
// order is independent of thread, block and atomic arrival order — do not add
// an `atomicAdd` accumulator to `PrismPhysGovExtObservation`.
//
// ─── No fabricated measurements ────────────────────────────────────────────
//
// Every extension guard is tri-valued: MEASURED-OK / MEASURED-DEGRADED /
// UNMEASURED.  UNMEASURED is an explicit `*_measured == 0` flag and the
// corresponding observed value is left at its NaN / sentinel encoding — never
// replaced by a plausible default.  An UNMEASURED guard that is required for
// upshift pins the ceiling to the current gear.  It never deadlocks the run:
// the current gear stays reachable, only the coarser gear is withheld.
//
// ─── No hardcoded thresholds ───────────────────────────────────────────────
//
// Every threshold lives in `PrismPhysGovExtPolicy`.  The `PRISM_PHYSGOV_DEFAULT_*`
// macros below are the DOCUMENTED DEFAULTS and the single source of default
// truth.  `crates/prism-nhs/src/physical_governor.rs` mirrors them, and
// `crates/prism-nhs/tests/physical_governor_gates.rs::defaults_match_cuda_header`
// PARSES THIS FILE at test time and asserts every literal matches the Rust
// default, so the two halves cannot drift silently.
//
// ─── CHUNK-BOUNDARY CONTRACT (operator requirement 6) ──────────────────────
//
// The governor runs ONCE PER CAPTURED-GRAPH REPLAY.  With
// `--captured-graph-body-unroll N > 1` the Director advances N protocol steps
// per replay, so:
//
//   * the gear authorized at replay k is in force for ALL N steps of replay k;
//   * a degradation observed during replay k cannot be corrected until the
//     entry of replay k+1;
//   * therefore the governor selects a gear that must be SAFE FOR THE WHOLE
//     CHUNK, evaluated on entry, not a gear that is safe for the next step.
//
// The bound is a NAMED, EMITTED quantity, never an assumption:
//
//     downshift_latency_steps = policy.chunk_span_steps                (= N)
//     downshift_latency_fs    = chunk_span_steps * gear_dt_fs[current]
//
// Both are written into every `PrismPhysGovDecision`.  Mid-chunk correction is
// NOT available; a within-chunk emergency remains the abort trap's job
// (Gear 3), which is what the trap is for.  The base CHUNK_LATENCY gate
// (gate 6) refuses any gear whose uncorrected window exceeds
// `max_uncorrected_window_fs`; this header does not second-guess it.
//
// ─── COMPILATION (this is a COMPILED contract, not a claim) ────────────────
//
// Every entry point below is `__host__ __device__ __forceinline__`, so this
// file declares no storage of its own.  That is NOT the same as being exempt
// from compilation: a `static_assert` in a header that no translation unit
// `#include`s is never evaluated, and every size / offset / rank-order pin in
// this file would then be decorative.
//
// The translation unit that makes them binding is
// `crates/prism-nhs/src/cuda/physical_governor.cu`.  It `#include`s this file,
// exposes `prism_physical_governor_decide_host` plus the ABI probe, and
// instantiates `prism_physical_governor_decide_kernel` so the __device__ half
// is driven through ptxas rather than merely parsed.  `crates/prism-nhs/
// build.rs` compiles it with `compile_to_static_archive_ieee754` (the same
// IEEE-strict floating-point contract `deterministic_reduce.cu` uses, because
// this file's output is a CONTROL DECISION and must be re-derivable off-device)
// and emits `cargo:rerun-if-changed` for both files.
//
// It therefore compiles under `nvcc -std=c++17` on every build of this crate
// with the `gpu` feature, and under a host C++17 compiler with the
// `gearbox.cuh` CUDA shims.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include "gearbox.cuh"

// ═══════════════════════════════════════════════════════════════════════════
// Schema
// ═══════════════════════════════════════════════════════════════════════════

/// Extension wire-format version.  Bump whenever `PrismPhysGovExtPolicy`,
/// `PrismPhysGovExtObservation` or `PrismPhysGovDecision` changes shape.  The
/// Rust mirror asserts on this value and every emitted decision carries it, so
/// a reader that sees an unknown version fails loudly instead of misparsing a
/// field into a timestep.
#define PRISM_PHYSGOV_SCHEMA_VERSION 1u

// ═══════════════════════════════════════════════════════════════════════════
// Extension gate identifiers
//
// Disjoint from `PRISM_GEAR_GATE_*` (which occupy 0..11) so a binding-gate id
// is unambiguous across the composed decision.  Ids start at 32.
// ═══════════════════════════════════════════════════════════════════════════

#define PRISM_PHYSGOV_GATE_NONE                  0u
#define PRISM_PHYSGOV_GATE_FORCE_SPIKE          32u
#define PRISM_PHYSGOV_GATE_ACCELERATION_EXTREME 33u
#define PRISM_PHYSGOV_GATE_THERMOSTAT_STABILITY 34u
#define PRISM_PHYSGOV_GATE_INTEGRATION_HEALTH   35u
#define PRISM_PHYSGOV_GATE_TSH_MEMORY           36u
#define PRISM_PHYSGOV_GATE_PHASE_REQUIREMENT    37u
#define PRISM_PHYSGOV_GATE_HMR_HARD             38u
/// The extension analogue of base gate 11 (`PRISM_GEAR_GATE_GOVERNOR_UNARMED`).
/// Reported when the extension layer could NOT bind a governed policy or
/// observation — an unarmed `PrismPhysGovExtPolicy`, or a schema version this
/// build does not understand.  It exists so "nobody armed the extension" and
/// "this policy is from a different wire version" are never reported as physics
/// findings.  See `PRISM_PHYSGOV_REFUSAL_*` for the specific cause.
#define PRISM_PHYSGOV_GATE_EXT_BINDING          39u
#define PRISM_PHYSGOV_GATE_EXT_FIRST            PRISM_PHYSGOV_GATE_FORCE_SPIKE
#define PRISM_PHYSGOV_GATE_EXT_LAST             PRISM_PHYSGOV_GATE_EXT_BINDING
/// Number of extension gate ids, == the number of 4-bit nibbles in
/// `PrismPhysGovDecision.ext_gate_verdicts` (8 * 4 == 32 bits, exactly full).
#define PRISM_PHYSGOV_GATE_EXT_COUNT            8u
/// Number of extension gates that participate in the CEILING fold.  The
/// binding-refusal gate is not one of them: it is an early, terminal branch
/// that replaces the fold rather than contributing an operand to it.
#define PRISM_PHYSGOV_CEILING_GATE_COUNT        7u

// ─── Extension binding-refusal causes ──────────────────────────────────────
//
// Emitted in `PrismPhysGovDecision.ext_binding_refusal`.  A NAMED TERMINAL
// cause for every refusal path, so no wait or refusal is ever unexplained.
#define PRISM_PHYSGOV_REFUSAL_NONE                0u
#define PRISM_PHYSGOV_REFUSAL_UNARMED             1u
#define PRISM_PHYSGOV_REFUSAL_POLICY_SCHEMA       2u
#define PRISM_PHYSGOV_REFUSAL_OBSERVATION_SCHEMA  3u

// ═══════════════════════════════════════════════════════════════════════════
// LEXICOGRAPHIC AUTHORITY RANKS
//
// The operator order is
//
//   (physical_emergency, topology_permission, constraint_stability,
//    energy_drift, force_stability, tsh_memory, tsh_salience,
//    phase_requirement, current_gear)
//
// LOWER RANK = MORE AUTHORITATIVE.  A gate can only ever TIGHTEN the ceiling;
// rank decides ATTRIBUTION when two gates land on the same ceiling, so the
// record names the most authoritative reason rather than an incidental one.
//
// This is emphatically NOT a weighted score.  There is no coefficient anywhere
// in this file that multiplies a salience against a physical quantity, so
// there is no salience value — 1.0, 10^9, or NaN — that can purchase a gear a
// physical gate refused.  The `static_assert` rank chain at the foot of this
// file and
// `physical_governor_base_composition.rs::enormous_salience_cannot_purchase_a_physically_refused_gear`
// prove it mechanically, the latter by driving every representable salience
// magnitude through the production base governor.
//
// Rank spacing is 10 with room to interleave; the intermediate ranks below the
// force-stability band (52/54/56/58) are the extension physical guards, which
// sit BELOW `force_stability` in authority (they refine it) and ABOVE
// `chunk_latency` (which is a scheduling bound, not a physics finding).
// ═══════════════════════════════════════════════════════════════════════════

#define PRISM_PHYSGOV_RANK_PHYSICAL_EMERGENCY     10u
#define PRISM_PHYSGOV_RANK_HMR_HARD               15u
#define PRISM_PHYSGOV_RANK_TOPOLOGY_PERMISSION    20u
#define PRISM_PHYSGOV_RANK_GOVERNOR_UNARMED       25u
#define PRISM_PHYSGOV_RANK_CONSTRAINT_STABILITY   30u
#define PRISM_PHYSGOV_RANK_ENERGY_DRIFT           40u
#define PRISM_PHYSGOV_RANK_FORCE_STABILITY        50u
#define PRISM_PHYSGOV_RANK_FORCE_SPIKE            52u
#define PRISM_PHYSGOV_RANK_ACCELERATION_EXTREME   54u
#define PRISM_PHYSGOV_RANK_THERMOSTAT_STABILITY   56u
#define PRISM_PHYSGOV_RANK_INTEGRATION_HEALTH     58u
#define PRISM_PHYSGOV_RANK_CHUNK_LATENCY          60u
#define PRISM_PHYSGOV_RANK_TSH_MEMORY             70u
#define PRISM_PHYSGOV_RANK_TSH_SALIENCE           80u
#define PRISM_PHYSGOV_RANK_PHASE_REQUIREMENT      90u
#define PRISM_PHYSGOV_RANK_DWELL                 100u
#define PRISM_PHYSGOV_RANK_LOCKOUT               101u
#define PRISM_PHYSGOV_RANK_UPSHIFT_RATE          102u
#define PRISM_PHYSGOV_RANK_NONE                  255u

// ─── Extension emergency cause bits ────────────────────────────────────────
//
// Disjoint bit positions from `PRISM_GEAR_EMERG_*` (0x1..0x10) so a composed
// record can carry both sets without either being reinterpreted.
#define PRISM_PHYSGOV_EMERG_NONFINITE_ACCEL     0x100u
#define PRISM_PHYSGOV_EMERG_ACCEL_DISPLACEMENT  0x200u
#define PRISM_PHYSGOV_EMERG_INTEGRATION_FAULT   0x400u
/// `|T_inst - T_target| / T_target >= temp_dev_abort_fraction`.  At the default
/// 2.00 that is a threefold temperature runaway: not a thermostat that is
/// merely fighting, but a system whose kinetic energy is diverging.  No
/// timestep repairs it, so the trap is the honest outcome.
#define PRISM_PHYSGOV_EMERG_TEMPERATURE_RUNAWAY 0x800u

// ─── Transition classification (emitted on EVERY decision) ─────────────────
//
// A gap in the record stream must never be ambiguous between "gear held" and
// "the decision did not run".  `PrismPhysGovDecision.decision_emitted` is
// unconditionally 1 and `transition` is unconditionally one of these — HOLD is
// an emitted fact, not an absence.
#define PRISM_PHYSGOV_TRANSITION_HOLD      0u
#define PRISM_PHYSGOV_TRANSITION_DOWNSHIFT 1u
#define PRISM_PHYSGOV_TRANSITION_UPSHIFT   2u
#define PRISM_PHYSGOV_TRANSITION_ABORT     3u

// ─── Tri-state "not measured" sentinels ────────────────────────────────────
//
// Explicit sentinels; nothing is ever back-filled with a plausible value.
#define PRISM_PHYSGOV_U32_UNMEASURED 0xFFFFFFFFu

// ═══════════════════════════════════════════════════════════════════════════
// DOCUMENTED DEFAULTS
//
// Parsed verbatim by `physical_governor_gates.rs::defaults_match_cuda_header`.
// Format is load-bearing: `#define PRISM_PHYSGOV_DEFAULT_<NAME> <literal>`,
// one per line, literal is an integer with a `u` suffix or a float with an `f`
// suffix.
// ═══════════════════════════════════════════════════════════════════════════

// ─── Guard 5 — FORCE SPIKE ─────────────────────────────────────────────────
//
// Observable: the RELATIVE growth rate of the maximum force norm, normalised
// per SIMULATED PICOSECOND so the guard does not alias when the gear changes:
//
//     force_rate_per_ps = (|F|_t - |F|_{t-1})
//                       / (max(|F|_{t-1}, force_rate_norm_floor) * dt_prev_ps)
//
// Anchoring the bands (POLICY DEFAULTS, stated arithmetic, not measurements):
// at Gear 2 one step is 0.004 ps, so a 25 % jump in max|F| within a single 4 fs
// step is 0.25/0.004 = 62.5 /ps, and a 100 % jump is 250 /ps.  A well-behaved
// condensed-phase force norm moves a few percent per step; a >25 %/step jump is
// a close-contact onset that a coarse timestep converts into a blow-up before
// the next decision boundary.
//
//   rate <  50 /ps  -> OK          (< 20 %/step at 4 fs)
//   rate < 100 /ps  -> HOLD        (< 40 %/step at 4 fs)
//   rate < 250 /ps  -> DOWNSHIFT   (< 100 %/step at 4 fs)
//   rate >= 250 /ps -> FLOOR       (max|F| at least doubling within one 4 fs step)
//
// FLOOR, never EMERGENCY: a spike is a reason to integrate finer, and the
// absolute magnitude that justifies aborting is already the base gate-1
// `force_abort_fraction` test on |F| itself.  Aborting on a RATE would kill
// runs whose forces spike from a small base.
#define PRISM_PHYSGOV_DEFAULT_FORCE_RATE_HOLD_PER_PS      50.0f
#define PRISM_PHYSGOV_DEFAULT_FORCE_RATE_DOWNSHIFT_PER_PS 100.0f
#define PRISM_PHYSGOV_DEFAULT_FORCE_RATE_FLOOR_PER_PS     250.0f

/// Denominator floor for the relative rate, in kcal/mol/A.  Without it a
/// near-zero previous force norm makes the ratio explode and the guard fires on
/// numerical noise rather than on physics.  1.0 kcal/mol/A is three orders of
/// magnitude below the integrator clamp (`MAX_INTEGRATOR_FORCE_KCAL_MOL_A` =
/// 1000.0, `nhs_amber_fused.cu:59`), so it never suppresses a real spike.
#define PRISM_PHYSGOV_DEFAULT_FORCE_RATE_NORM_FLOOR       1.0f

// ─── Guard 6 — ACCELERATION EXTREMES ───────────────────────────────────────
//
// The classical timestep-safety criterion.  The per-step displacement produced
// by the largest acceleration in the system,
//
//     dx(g) = 0.5 * a_max * dt_ps(g)^2         [A, with a_max in A/ps^2]
//
// must stay far below the shortest constrained distance, the X–H bond
// (~1.0 A), because SHAKE (`shake_cluster_positions`, relative_tolerance =
// 1.0e-4, `nhs_amber_fused.cu:1619`) reconverges from a perturbed geometry
// only while the perturbation is small compared to the bond it is restoring.
//
// This guard is PER-CANDIDATE-GEAR: `dx` is quadratic in dt, so the same
// acceleration is safe at 0.5 fs and unsafe at 4 fs.  The evaluator scans the
// gears and publishes the highest one whose `dx` is inside the OK band — that
// is a CEILING derived from physics, not a downshift decision.
//
// The band a candidate gear lands in decides HOW HIGH THAT GEAR MAY SIT, and
// the ceiling is the coarsest gear its own band still permits.  Every threshold
// below therefore governs a ceiling; none of them merely relabels a verdict:
//
//   dx < 0.02 A  (2 % of a bond)  -> OK        gear g unconstrained
//   dx < 0.05 A  (5 %)            -> HOLD      gear g permitted only up to cur
//   dx < 0.10 A  (10 %)           -> DOWNSHIFT gear g permitted only below cur
//   dx >= 0.10 A                  -> FLOOR     gear g refused above the finest
//   dx >= 0.50 A at the FINEST gear -> EMERGENCY (SHAKE cannot recover half a
//                                     bond length in one step)
//
// Reading the bands this way is what makes "PER-CANDIDATE-GEAR" true. Scanning
// only for `dx < ok` and flooring when nothing qualifies over-restricts by the
// whole width of the HOLD band: at cur = 1 with dx(1) = 0.035 A the documented
// answer is HOLD AT GEAR 1, while an OK-only scan publishes gear 0 and forces a
// downshift the physics did not ask for.
//
// The ceiling can never exceed `PRISM_GEAR_MAX_ACTIVE`, and a gear whose dt is
// unresolvable is skipped rather than guessed at.
#define PRISM_PHYSGOV_DEFAULT_ACCEL_DISPLACEMENT_REFERENCE_A 1.0f
#define PRISM_PHYSGOV_DEFAULT_ACCEL_OK_FRACTION              0.02f
#define PRISM_PHYSGOV_DEFAULT_ACCEL_HOLD_FRACTION            0.05f
#define PRISM_PHYSGOV_DEFAULT_ACCEL_DOWNSHIFT_FRACTION       0.10f
#define PRISM_PHYSGOV_DEFAULT_ACCEL_ABORT_FRACTION           0.50f

// ─── Guard 7 — THERMOSTAT STABILITY ────────────────────────────────────────
//
// `langevin_thermostat` (`nhs_amber_fused.cu:1523`) is an Ornstein–Uhlenbeck
// update with `c1 = expf(-gamma*dt)`.  Its discretisation is faithful only
// while `gamma*dt` is small; as `gamma*dt` approaches 1 the velocity is
// essentially fully resampled every step and the trajectory stops being
// inertial — the dynamics silently become Brownian.  `effective_gamma` is
// Director-computed per step (`protocol_state.cuh:62`) and rises sharply during
// the cryo / equilibration blend, which is exactly when a coarse gear must be
// withheld.
//
// PER-CANDIDATE-GEAR, same structure as the acceleration guard: the band a
// candidate gear lands in decides how high THAT gear may sit, and the ceiling
// is the coarsest gear its own band still permits.
//
//   gamma*dt < 0.05  -> OK        gear g unconstrained
//   gamma*dt < 0.10  -> HOLD      gear g permitted only up to cur
//   gamma*dt < 0.25  -> DOWNSHIFT gear g permitted only below cur
//   gamma*dt >= 0.25 -> FLOOR     gear g refused above the finest
//
// FLOOR, never EMERGENCY.  A large `gamma*dt` is an ACCURACY defect, not a
// divergence; a legitimate high-friction equilibration setting must not abort
// the run.
//
// `gamma_dt_abort` is NOT decorative and is NOT an abort: at `gamma*dt >= 1.0`
// the OU factor `c1 = exp(-gamma*dt) <= 0.37` — the velocity is essentially
// fully resampled every step and the trajectory has stopped being inertial.
// That is a positively-established loss of integrator character, so crossing it
// raises the INSTABILITY flag (which zeroes the dwell counter and restarts the
// lockout window, blocking any upshift for `instability_lockout_steps`) on top
// of the FLOOR the DOWNSHIFT band already produced.  It governs an emitted
// state transition; it does not merely decorate a receipt.
#define PRISM_PHYSGOV_DEFAULT_GAMMA_DT_OK        0.05f
#define PRISM_PHYSGOV_DEFAULT_GAMMA_DT_HOLD      0.10f
#define PRISM_PHYSGOV_DEFAULT_GAMMA_DT_DOWNSHIFT 0.25f
#define PRISM_PHYSGOV_DEFAULT_GAMMA_DT_ABORT     1.00f

// Instantaneous-temperature deviation from the Director's target,
// `|T_inst - T_target| / max(T_target, 1 K)`.  A thermostat that cannot hold
// its setpoint is either fighting an instability or being fed bad kinetic
// energy; either way a coarser timestep is not defensible.
//
//   dev < 0.10 -> OK        dev < 0.25 -> HOLD
//   dev < 0.50 -> DOWNSHIFT dev >= 0.50 -> FLOOR
//   dev >= 2.00 -> EMERGENCY (`PRISM_PHYSGOV_EMERG_TEMPERATURE_RUNAWAY`): T has
//                  run away by 3x, no timestep repairs it, the run traps.
//
// A NON-FINITE instantaneous temperature is FLOOR + instability, NOT emergency.
// NaN is an absent measurement, and absence of evidence is not evidence of a
// threefold runaway; claiming EMERGENCY from it would fabricate a finding the
// instrument never made.  The finest gear plus a reset lockout window is the
// honest response to a thermostat we cannot read.
#define PRISM_PHYSGOV_DEFAULT_TEMP_DEV_OK_FRACTION        0.10f
#define PRISM_PHYSGOV_DEFAULT_TEMP_DEV_HOLD_FRACTION      0.25f
#define PRISM_PHYSGOV_DEFAULT_TEMP_DEV_DOWNSHIFT_FRACTION 0.50f
#define PRISM_PHYSGOV_DEFAULT_TEMP_DEV_ABORT_FRACTION     2.00f

// ─── Guard 9 — NEIGHBOUR-LIST / INTEGRATION HEALTH ─────────────────────────
//
// The fused integrator's O(N) path indexes a fixed per-atom neighbour list of
// `NEIGHBOR_LIST_SIZE = 1024` (`nhs_amber_fused.cu:141`).  When an atom's true
// neighbour count reaches capacity the list TRUNCATES: pair interactions are
// silently dropped and the force is wrong without any NaN to trip a trap.
//
// A finer timestep does not repair a truncated pair list — so this guard never
// escalates past FLOOR on occupancy, and never claims a coarser gear is
// warranted.  It still publishes a ceiling, because a wrong force integrated
// over a longer uncorrected window compounds faster.
//
//   occupancy < 0.80 -> OK      occupancy < 0.90 -> HOLD
//   occupancy < 1.00 -> DOWNSHIFT   occupancy >= 1.00 -> FLOOR (truncating now)
#define PRISM_PHYSGOV_DEFAULT_NEIGHBOR_OCCUPANCY_OK   0.80f
#define PRISM_PHYSGOV_DEFAULT_NEIGHBOR_OCCUPANCY_HOLD 0.90f

// Fraction of atoms whose force was clipped by
// `MAX_INTEGRATOR_FORCE_KCAL_MOL_A` in `apply_force_half_kick_with_intervention_work`
// (`nhs_amber_fused.cu:1575`).  Clipping is non-Hamiltonian BY CONSTRUCTION,
// so ANY clipping forbids an upshift; sustained clipping floors the gear.
//
//   frac <= 0        -> OK
//   frac <= 1e-4     -> HOLD
//   frac <= 1e-3     -> DOWNSHIFT
//   frac >  1e-3     -> FLOOR
#define PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_HOLD      0.0f
#define PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_DOWNSHIFT 0.0001f
#define PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_FLOOR     0.001f

// ─── TSH memory + ANTI-BLINDNESS ───────────────────────────────────────────
//
// The architecture's positive-feedback failure mode:
//
//     low salience -> coarser gear -> fewer snapshots -> less evidence
//                  -> lower measured salience -> coarser still -> ...
//
// stable in the WRONG equilibrium.  The four required anti-blindness controls
// are enforced here, all ceiling-only:
//
//   (a) minimum physical-time observation cadence
//         `ps_since_last_observation > max_observation_interval_ps` -> HOLD
//   (b) maximum interval without a forced high-resolution sentinel sample
//         `steps_since_sentinel > max_steps_without_sentinel` -> FLOOR to gear 0
//   (c) uncertainty growth when evidence is sparse
//         carried by the producer in `observation_confidence`, which the TSH arm
//         decays with elapsed simulated ps rather than with record count
//   (d) NO coarsening while observation confidence is below threshold
//         `observation_confidence < min_observation_confidence` -> ceiling = current gear
//
// `tsh_instability_memory` in [0,1] is the TSH arm's integrated recollection of
// recent instability, normalised per simulated ps (NEVER per record, per chunk
// or per invocation).
#define PRISM_PHYSGOV_DEFAULT_TSH_MEMORY_HOLD               0.25f
#define PRISM_PHYSGOV_DEFAULT_TSH_MEMORY_FLOOR              0.75f
#define PRISM_PHYSGOV_DEFAULT_MIN_OBSERVATION_CONFIDENCE    0.50f
#define PRISM_PHYSGOV_DEFAULT_MAX_OBSERVATION_INTERVAL_PS   1.0f
#define PRISM_PHYSGOV_DEFAULT_MAX_STEPS_WITHOUT_SENTINEL 10000u

// ─── Protocol-phase requirement ────────────────────────────────────────────
//
// Some protocol phases demand fine resolution regardless of how quiet the
// physics looks — a UV burst (`ProtocolState.uv_burst_active`) injects energy
// on a femtosecond scale, and a temperature ramp is a moving setpoint.
// `phase_ceiling_gear[phase_id]` is a policy-resolved CEILING, so a phase can
// only ever make the gear finer.  Default slots mirror the CCNS 5-phase
// protocol boundaries in `protocol_state.cuh:24-29` plus the UV burst:
//
//   0 cold_hold   -> unconstrained (gear 2)
//   1 ramp_up     -> gear 1
//   2 warm_hold   -> unconstrained (gear 2)
//   3 ramp_down   -> gear 1
//   4 cold_return -> unconstrained (gear 2)
//   5 uv_burst    -> gear 0
//   6,7 reserved  -> unconstrained (gear 2)
#define PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_0 2u
#define PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_1 1u
#define PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_2 2u
#define PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_3 1u
#define PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_4 2u
#define PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_5 0u
#define PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_6 2u
#define PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_7 2u
#define PRISM_PHYSGOV_DEFAULT_N_PHASE_SLOTS   8u

// ─── Upshift-requirement switches ──────────────────────────────────────────
//
// Whether an UNMEASURED extension guard blocks upshift.  Default 1 (blocks):
// an upshift is a claim that the physics is verified safe, and an unread
// sensor is not verification.  Setting any of these to 0 in a policy file is an
// explicit, recorded operator decision to upshift on unmeasured physics.
#define PRISM_PHYSGOV_DEFAULT_REQUIRE_ACCEL_GUARD        1u
#define PRISM_PHYSGOV_DEFAULT_REQUIRE_THERMOSTAT_GUARD   1u
#define PRISM_PHYSGOV_DEFAULT_REQUIRE_INTEGRATION_GUARD  1u
#define PRISM_PHYSGOV_DEFAULT_REQUIRE_OBSERVATION_CONFIDENCE 1u

// ─── Unarmed / unbindable cap ──────────────────────────────────────────────
//
// The gear ceiling applied when the extension layer cannot bind a governed
// policy or observation (see `PRISM_PHYSGOV_GATE_EXT_BINDING`).
//
// This is deliberately NOT policy-overridable and deliberately NOT read from
// `PrismPhysGovExtPolicy`: the whole point is that the policy could not be
// trusted, so taking the cap from it would be circular.  It is a compile-time
// constant, mirrored on the Rust side and pinned by
// `physical_governor_gates.rs::defaults_match_cuda_header`.
//
// FAIL CLOSED, not fail quiet.  A `memset`-to-zero `PrismPhysGovExtPolicy` has
// `force_rate_floor_per_ps == 0`, `accel_ok_fraction == 0` and
// `clamped_fraction_floor == 0`; running the guards against it manufactures a
// FLOOR verdict on every gate and the emitted record becomes indistinguishable
// from genuinely violent physics.  The refusal branch pins the same finest gear
// the zeroed policy would have produced, but attributes it honestly and never
// deadlocks: Gear 0 is always reachable.
#define PRISM_PHYSGOV_DEFAULT_UNARMED_CAP_GEAR 0u

// ═══════════════════════════════════════════════════════════════════════════
// Extension policy
// ═══════════════════════════════════════════════════════════════════════════

/// Resolved extension policy.  Immutable for the life of a run.
///
/// `ext_policy_digest` is the host-computed fingerprint of the resolved values
/// AND their per-field origins AND the policy file's sha256.  It is copied
/// verbatim into every decision record, so an offline reader can bind a gear
/// choice to the exact policy that authorised it without trusting a side
/// channel.
struct __align__(16) PrismPhysGovExtPolicy {
    uint32_t schema_version;                        // @   0
    uint32_t armed;                                 // @   4
    uint32_t require_accel_guard_for_upshift;       // @   8
    uint32_t require_thermostat_guard_for_upshift;  // @  12
    uint32_t require_integration_guard_for_upshift; // @  16
    uint32_t require_observation_confidence;        // @  20
    uint32_t max_steps_without_sentinel;            // @  24
    uint32_t n_phase_slots;                         // @  28

    float    force_rate_hold_per_ps;                // @  32
    float    force_rate_downshift_per_ps;           // @  36
    float    force_rate_floor_per_ps;               // @  40
    float    force_rate_norm_floor;                 // @  44

    float    accel_displacement_reference_a;        // @  48
    float    accel_ok_fraction;                     // @  52
    float    accel_hold_fraction;                   // @  56
    float    accel_downshift_fraction;              // @  60
    float    accel_abort_fraction;                  // @  64

    float    gamma_dt_ok;                           // @  68
    float    gamma_dt_hold;                         // @  72
    float    gamma_dt_downshift;                    // @  76
    float    gamma_dt_abort;                        // @  80
    float    temp_dev_ok_fraction;                  // @  84
    float    temp_dev_hold_fraction;                // @  88
    float    temp_dev_downshift_fraction;           // @  92
    float    temp_dev_abort_fraction;               // @  96

    float    neighbor_occupancy_ok;                 // @ 100
    float    neighbor_occupancy_hold;               // @ 104
    float    clamped_fraction_hold;                 // @ 108
    float    clamped_fraction_downshift;            // @ 112
    float    clamped_fraction_floor;                // @ 116

    float    tsh_memory_hold;                       // @ 120
    float    tsh_memory_floor;                      // @ 124
    float    min_observation_confidence;            // @ 128
    float    max_observation_interval_ps;           // @ 132

    uint32_t phase_ceiling_gear[8];                 // @ 136..168
    uint64_t ext_policy_digest;                     // @ 168
    uint32_t _reserved[8];                          // @ 176..208
};

static_assert(sizeof(PrismPhysGovExtPolicy) == 208,
              "PrismPhysGovExtPolicy MUST be 208 bytes (Rust mirror pins the same).");
static_assert(alignof(PrismPhysGovExtPolicy) == 16,
              "PrismPhysGovExtPolicy MUST be 16-byte aligned.");
static_assert(offsetof(PrismPhysGovExtPolicy, force_rate_hold_per_ps) == 32,
              "force_rate_hold_per_ps offset drift: must be 32.");
static_assert(offsetof(PrismPhysGovExtPolicy, phase_ceiling_gear) == 136,
              "phase_ceiling_gear offset drift: must be 136.");
static_assert(offsetof(PrismPhysGovExtPolicy, ext_policy_digest) == 168,
              "ext_policy_digest offset drift: must be 168.");

// ═══════════════════════════════════════════════════════════════════════════
// Extension observation
// ═══════════════════════════════════════════════════════════════════════════

/// One decision's extension guard observations.  Assembled by the caller so
/// the decision function performs NO I/O and is identically testable on host
/// and device.
///
/// Every `*_measured` field is the tri-state discriminator: 0 = UNMEASURED,
/// and the paired value carries NaN / `PRISM_PHYSGOV_U32_UNMEASURED`.  Nothing
/// in this struct is ever back-filled with a plausible default.
///
/// PRODUCER CONTRACT (determinism).  `max_accel_a_per_ps2`, `force_norm_prev`
/// and `max_neighbor_count` are MAXIMA; `integration_faults` and
/// `steps_since_sentinel` are integer COUNTS.  Those are order-invariant
/// reductions and may be produced with `atomicMax` / integer `atomicAdd`
/// without making two identical runs disagree.
///
/// `clamped_force_fraction` is NOT in that set, despite being a ratio of two
/// counts: it is an `f32` that arrives pre-divided, so a producer that read the
/// sentence above as licence would reach for a FLOAT `atomicAdd` — exactly the
/// non-associative reduction this contract forbids — and the low mantissa bits
/// would flip a gear at the `clamped_fraction_downshift = 1e-4` boundary
/// between two identical runs.  Its numerator and denominator must be reduced
/// as integers and divided once, or the ratio must come from
/// `deterministic_reduce.cuh` (Package 2), whose summation order is independent
/// of thread, block and atomic arrival order.
///
/// A float SUM must NOT be added here without a fixed-order reduction.
struct __align__(16) PrismPhysGovExtObservation {
    uint32_t schema_version;             // @  0  PRISM_PHYSGOV_SCHEMA_VERSION

    /// TOPOLOGY-DERIVED, not policy-derived: `PrismPrepTopology::hmr_applied`
    /// as read from the parsed topology (`input.rs:554/1893`).  Tri-state
    /// `PRISM_GEAR_PERMISSION_*`.  The HMR hard gate consults THIS, not
    /// `policy.hmr_permission`, so a mis-set policy file cannot buy Gear 2.
    uint32_t hmr_applied_verified;       // @  4

    uint32_t force_rate_measured;        // @  8
    float    force_norm_prev;            // @ 12  |F| at the previous decision
    float    dt_prev_ps;                 // @ 16  simulated ps the delta spans

    uint32_t accel_measured;             // @ 20
    float    max_accel_a_per_ps2;        // @ 24  max |F|/m over all atoms

    uint32_t thermostat_measured;        // @ 28
    float    effective_gamma_per_ps;     // @ 32  ProtocolState.effective_gamma
    float    instantaneous_temp_k;       // @ 36
    float    target_temp_k;              // @ 40  ProtocolState.current_temperature

    uint32_t integration_measured;       // @ 44
    uint32_t max_neighbor_count;         // @ 48
    uint32_t neighbor_list_capacity;     // @ 52  NEIGHBOR_LIST_SIZE
    float    clamped_force_fraction;     // @ 56
    uint32_t integration_faults;         // @ 60  kernel-reported hard faults

    uint32_t tsh_memory_measured;        // @ 64
    float    tsh_instability_memory;     // @ 68  [0,1], per simulated ps
    float    observation_confidence;     // @ 72  [0,1]
    float    ps_since_last_observation;  // @ 76
    uint32_t steps_since_sentinel;       // @ 80

    uint32_t phase_measured;             // @ 84
    uint32_t phase_id;                   // @ 88

    uint32_t _reserved[5];               // @ 92..112
};

static_assert(sizeof(PrismPhysGovExtObservation) == 112,
              "PrismPhysGovExtObservation MUST be 112 bytes (Rust mirror pins the same).");
static_assert(alignof(PrismPhysGovExtObservation) == 16,
              "PrismPhysGovExtObservation MUST be 16-byte aligned.");
static_assert(offsetof(PrismPhysGovExtObservation, hmr_applied_verified) == 4,
              "hmr_applied_verified offset drift: must be 4.");
static_assert(offsetof(PrismPhysGovExtObservation, max_accel_a_per_ps2) == 24,
              "max_accel_a_per_ps2 offset drift: must be 24.");
static_assert(offsetof(PrismPhysGovExtObservation, tsh_instability_memory) == 68,
              "tsh_instability_memory offset drift: must be 68.");

// ═══════════════════════════════════════════════════════════════════════════
// Composed decision record
// ═══════════════════════════════════════════════════════════════════════════

/// THE COMPACT NATIVE RECORD.  128 bytes, POD, no pointers.  Written to a
/// device ring by the caller.  There is no JSON, hashing, compression or
/// filesystem work anywhere in this file — the hot path stays compact by
/// construction.
///
/// THREE SEPARATE GEAR FACTS, never collapsed:
///   `requested_gear`         — what the science asked for (TSH / adjudication)
///   `authorized_gear`        — what this governor permitted
///   `device_measured_gear`   — NOT in this struct: it is measured after the
///                              fact by `prism_gearbox_snapshot_state` and
///                              joined by `gear_epoch_id`.  Fabricating it here
///                              would be exactly the `gear_id: 0` failure the
///                              architecture document names.
struct __align__(16) PrismPhysGovDecision {
    uint32_t schema_version;               // @   0
    uint32_t authorized_gear;              // @   4  min(base authorized, ext ceiling)
    uint32_t ceiling_gear;                 // @   8  min(base ceiling, ext ceiling)
    uint32_t base_authorized_gear;         // @  12
    uint32_t base_ceiling_gear;            // @  16
    uint32_t base_binding_gate;            // @  20  PRISM_GEAR_GATE_*
    uint32_t ext_binding_gate;             // @  24  PRISM_PHYSGOV_GATE_*
    uint32_t binding_gate_rank;            // @  28  PRISM_PHYSGOV_RANK_*
    uint32_t base_gate_verdicts;           // @  32  packed 4-bit, base gates 1..8
    uint32_t ext_gate_verdicts;            // @  36  packed 4-bit, ext gates 32..38
    uint32_t emergency_flags;              // @  40  PRISM_GEAR_EMERG_*
    uint32_t ext_emergency_flags;          // @  44  PRISM_PHYSGOV_EMERG_*
    uint32_t requested_gear;               // @  48
    uint32_t current_gear;                 // @  52
    uint32_t next_stable_dwell_steps;      // @  56
    uint32_t next_steps_since_instability; // @  60
    uint32_t downshift_latency_steps;      // @  64  == chunk_span_steps
    float    downshift_latency_fs;         // @  68  chunk_span * dt_fs[current]
    uint32_t transition;                   // @  72  PRISM_PHYSGOV_TRANSITION_*
    uint32_t decision_emitted;             // @  76  ALWAYS 1
    float    dt_applied_ps;                // @  80  gear_dt_fs[authorized]/1000
    uint32_t ext_armed;                    // @  84
    uint64_t policy_digest;                // @  88  base policy fingerprint
    uint64_t ext_policy_digest;            // @  96  extension policy fingerprint
    uint32_t ext_binding_refusal;          // @ 104  PRISM_PHYSGOV_REFUSAL_*
    uint32_t _reserved[5];                 // @ 108..128
};

static_assert(sizeof(PrismPhysGovDecision) == 128,
              "PrismPhysGovDecision MUST be 128 bytes (Rust mirror pins the same).");
static_assert(alignof(PrismPhysGovDecision) == 16,
              "PrismPhysGovDecision MUST be 16-byte aligned.");
static_assert(offsetof(PrismPhysGovDecision, ext_binding_gate) == 24,
              "ext_binding_gate offset drift: must be 24.");
static_assert(offsetof(PrismPhysGovDecision, downshift_latency_steps) == 64,
              "downshift_latency_steps offset drift: must be 64.");
static_assert(offsetof(PrismPhysGovDecision, policy_digest) == 88,
              "policy_digest offset drift: must be 88.");
static_assert(offsetof(PrismPhysGovDecision, ext_binding_refusal) == 104,
              "ext_binding_refusal offset drift: must be 104.");

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Pack an extension verdict.  Extension gates 32..38 map to nibbles 0..6.
__host__ __device__ __forceinline__ void prism_physgov_set_verdict(
    uint32_t* verdicts, uint32_t ext_gate_id, uint32_t verdict
) {
    if (ext_gate_id < PRISM_PHYSGOV_GATE_EXT_FIRST ||
        ext_gate_id > PRISM_PHYSGOV_GATE_EXT_LAST) {
        return;
    }
    const uint32_t shift = (ext_gate_id - PRISM_PHYSGOV_GATE_EXT_FIRST) * 4u;
    *verdicts = (*verdicts & ~(0xfu << shift)) | ((verdict & 0xfu) << shift);
}

__host__ __device__ __forceinline__ uint32_t prism_physgov_get_verdict(
    uint32_t verdicts, uint32_t ext_gate_id
) {
    if (ext_gate_id < PRISM_PHYSGOV_GATE_EXT_FIRST ||
        ext_gate_id > PRISM_PHYSGOV_GATE_EXT_LAST) {
        return PRISM_GEAR_VERDICT_UNMEASURED;
    }
    return (verdicts >> ((ext_gate_id - PRISM_PHYSGOV_GATE_EXT_FIRST) * 4u)) & 0xfu;
}

/// Authority rank of a BASE gate id.  Unknown ids sort last so an unexpected
/// gate can never claim authority it was not granted.
__host__ __device__ __forceinline__ uint32_t prism_physgov_base_gate_rank(uint32_t gate) {
    switch (gate) {
        case PRISM_GEAR_GATE_PHYSICAL_EMERGENCY:   return PRISM_PHYSGOV_RANK_PHYSICAL_EMERGENCY;
        case PRISM_GEAR_GATE_TOPOLOGY_PERMISSION:  return PRISM_PHYSGOV_RANK_TOPOLOGY_PERMISSION;
        case PRISM_GEAR_GATE_GOVERNOR_UNARMED:     return PRISM_PHYSGOV_RANK_GOVERNOR_UNARMED;
        case PRISM_GEAR_GATE_CONSTRAINT_STABILITY: return PRISM_PHYSGOV_RANK_CONSTRAINT_STABILITY;
        case PRISM_GEAR_GATE_ENERGY_DRIFT:         return PRISM_PHYSGOV_RANK_ENERGY_DRIFT;
        case PRISM_GEAR_GATE_FORCE_STABILITY:      return PRISM_PHYSGOV_RANK_FORCE_STABILITY;
        case PRISM_GEAR_GATE_CHUNK_LATENCY:        return PRISM_PHYSGOV_RANK_CHUNK_LATENCY;
        case PRISM_GEAR_GATE_TSH_SALIENCE:         return PRISM_PHYSGOV_RANK_TSH_SALIENCE;
        case PRISM_GEAR_GATE_DWELL:                return PRISM_PHYSGOV_RANK_DWELL;
        case PRISM_GEAR_GATE_LOCKOUT:              return PRISM_PHYSGOV_RANK_LOCKOUT;
        case PRISM_GEAR_GATE_UPSHIFT_RATE:         return PRISM_PHYSGOV_RANK_UPSHIFT_RATE;
        default:                                   return PRISM_PHYSGOV_RANK_NONE;
    }
}

/// Authority rank of an EXTENSION gate id.
__host__ __device__ __forceinline__ uint32_t prism_physgov_ext_gate_rank(uint32_t gate) {
    switch (gate) {
        case PRISM_PHYSGOV_GATE_HMR_HARD:               return PRISM_PHYSGOV_RANK_HMR_HARD;
        case PRISM_PHYSGOV_GATE_FORCE_SPIKE:            return PRISM_PHYSGOV_RANK_FORCE_SPIKE;
        case PRISM_PHYSGOV_GATE_ACCELERATION_EXTREME:   return PRISM_PHYSGOV_RANK_ACCELERATION_EXTREME;
        case PRISM_PHYSGOV_GATE_THERMOSTAT_STABILITY:   return PRISM_PHYSGOV_RANK_THERMOSTAT_STABILITY;
        case PRISM_PHYSGOV_GATE_INTEGRATION_HEALTH:     return PRISM_PHYSGOV_RANK_INTEGRATION_HEALTH;
        case PRISM_PHYSGOV_GATE_TSH_MEMORY:             return PRISM_PHYSGOV_RANK_TSH_MEMORY;
        case PRISM_PHYSGOV_GATE_PHASE_REQUIREMENT:      return PRISM_PHYSGOV_RANK_PHASE_REQUIREMENT;
        case PRISM_PHYSGOV_GATE_EXT_BINDING:            return PRISM_PHYSGOV_RANK_GOVERNOR_UNARMED;
        default:                                        return PRISM_PHYSGOV_RANK_NONE;
    }
}

/// One gear finer, saturating at the finest gear.
__host__ __device__ __forceinline__ uint32_t prism_physgov_one_finer(uint32_t gear) {
    return (gear > PRISM_GEAR_FINE) ? (gear - 1u) : PRISM_GEAR_FINE;
}

/// `dt` in picoseconds for a gear, or a non-finite value when the slot has no
/// timestep (Gear 3 / an unresolvable policy).  Never substitutes a default.
__host__ __device__ __forceinline__ float prism_physgov_gear_dt_ps(
    const PrismGearGovernorPolicy* p, uint32_t gear
) {
    if (gear > PRISM_GEAR_MAX_ACTIVE) return 0.0f / 0.0f;   /* NaN */
    const float fs = p->gear_dt_fs[gear];
    if (!prism_gear_finite_f32(fs) || fs <= 0.0f) return 0.0f / 0.0f;
    return fs * 0.001f;
}

/// Fill `out` with the documented defaults.  The ONLY place the extension
/// default values appear as literals is the `PRISM_PHYSGOV_DEFAULT_*` macro
/// block above; this function copies them and does not invent any.
__host__ __device__ __forceinline__ void prism_physgov_fill_default_ext_policy(
    PrismPhysGovExtPolicy* out
) {
    out->schema_version                        = PRISM_PHYSGOV_SCHEMA_VERSION;
    out->armed                                 = 0u;   /* host must arm it */
    out->require_accel_guard_for_upshift       = PRISM_PHYSGOV_DEFAULT_REQUIRE_ACCEL_GUARD;
    out->require_thermostat_guard_for_upshift  = PRISM_PHYSGOV_DEFAULT_REQUIRE_THERMOSTAT_GUARD;
    out->require_integration_guard_for_upshift = PRISM_PHYSGOV_DEFAULT_REQUIRE_INTEGRATION_GUARD;
    out->require_observation_confidence        = PRISM_PHYSGOV_DEFAULT_REQUIRE_OBSERVATION_CONFIDENCE;
    out->max_steps_without_sentinel            = PRISM_PHYSGOV_DEFAULT_MAX_STEPS_WITHOUT_SENTINEL;
    out->n_phase_slots                         = PRISM_PHYSGOV_DEFAULT_N_PHASE_SLOTS;

    out->force_rate_hold_per_ps                = PRISM_PHYSGOV_DEFAULT_FORCE_RATE_HOLD_PER_PS;
    out->force_rate_downshift_per_ps           = PRISM_PHYSGOV_DEFAULT_FORCE_RATE_DOWNSHIFT_PER_PS;
    out->force_rate_floor_per_ps               = PRISM_PHYSGOV_DEFAULT_FORCE_RATE_FLOOR_PER_PS;
    out->force_rate_norm_floor                 = PRISM_PHYSGOV_DEFAULT_FORCE_RATE_NORM_FLOOR;

    out->accel_displacement_reference_a        = PRISM_PHYSGOV_DEFAULT_ACCEL_DISPLACEMENT_REFERENCE_A;
    out->accel_ok_fraction                     = PRISM_PHYSGOV_DEFAULT_ACCEL_OK_FRACTION;
    out->accel_hold_fraction                   = PRISM_PHYSGOV_DEFAULT_ACCEL_HOLD_FRACTION;
    out->accel_downshift_fraction              = PRISM_PHYSGOV_DEFAULT_ACCEL_DOWNSHIFT_FRACTION;
    out->accel_abort_fraction                  = PRISM_PHYSGOV_DEFAULT_ACCEL_ABORT_FRACTION;

    out->gamma_dt_ok                           = PRISM_PHYSGOV_DEFAULT_GAMMA_DT_OK;
    out->gamma_dt_hold                         = PRISM_PHYSGOV_DEFAULT_GAMMA_DT_HOLD;
    out->gamma_dt_downshift                    = PRISM_PHYSGOV_DEFAULT_GAMMA_DT_DOWNSHIFT;
    out->gamma_dt_abort                        = PRISM_PHYSGOV_DEFAULT_GAMMA_DT_ABORT;
    out->temp_dev_ok_fraction                  = PRISM_PHYSGOV_DEFAULT_TEMP_DEV_OK_FRACTION;
    out->temp_dev_hold_fraction                = PRISM_PHYSGOV_DEFAULT_TEMP_DEV_HOLD_FRACTION;
    out->temp_dev_downshift_fraction           = PRISM_PHYSGOV_DEFAULT_TEMP_DEV_DOWNSHIFT_FRACTION;
    out->temp_dev_abort_fraction               = PRISM_PHYSGOV_DEFAULT_TEMP_DEV_ABORT_FRACTION;

    out->neighbor_occupancy_ok                 = PRISM_PHYSGOV_DEFAULT_NEIGHBOR_OCCUPANCY_OK;
    out->neighbor_occupancy_hold               = PRISM_PHYSGOV_DEFAULT_NEIGHBOR_OCCUPANCY_HOLD;
    out->clamped_fraction_hold                 = PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_HOLD;
    out->clamped_fraction_downshift            = PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_DOWNSHIFT;
    out->clamped_fraction_floor                = PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_FLOOR;

    out->tsh_memory_hold                       = PRISM_PHYSGOV_DEFAULT_TSH_MEMORY_HOLD;
    out->tsh_memory_floor                      = PRISM_PHYSGOV_DEFAULT_TSH_MEMORY_FLOOR;
    out->min_observation_confidence            = PRISM_PHYSGOV_DEFAULT_MIN_OBSERVATION_CONFIDENCE;
    out->max_observation_interval_ps           = PRISM_PHYSGOV_DEFAULT_MAX_OBSERVATION_INTERVAL_PS;

    out->phase_ceiling_gear[0]                 = PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_0;
    out->phase_ceiling_gear[1]                 = PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_1;
    out->phase_ceiling_gear[2]                 = PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_2;
    out->phase_ceiling_gear[3]                 = PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_3;
    out->phase_ceiling_gear[4]                 = PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_4;
    out->phase_ceiling_gear[5]                 = PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_5;
    out->phase_ceiling_gear[6]                 = PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_6;
    out->phase_ceiling_gear[7]                 = PRISM_PHYSGOV_DEFAULT_PHASE_CEILING_7;

    out->ext_policy_digest                     = 0ull;
    for (uint32_t i = 0u; i < 8u; ++i) out->_reserved[i] = 0u;
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-guard evaluation
//
// Each returns a CEILING and writes a verdict.  A ceiling of
// PRISM_GEAR_MAX_ACTIVE means "this guard constrains nothing".
// ═══════════════════════════════════════════════════════════════════════════

/// Guard 5 — FORCE SPIKE.  Relative growth of max|F| per simulated ps.
__host__ __device__ __forceinline__ uint32_t prism_physgov_gate_force_spike(
    const PrismPhysGovExtPolicy*      p,
    const PrismGearGuardObservation*  base,
    const PrismPhysGovExtObservation* e,
    uint32_t                          cur,
    uint32_t*                         verdict,
    bool*                             instability,
    bool*                             all_guards_ok
) {
    if (e->force_rate_measured == 0u || base->force_measured == 0u) {
        // The spike guard is a REFINEMENT of the force guard, but it must not
        // delegate its own upshift veto to `require_force_guard_for_upshift`,
        // a BASE policy switch this evaluator cannot read.  With that switch at
        // 0 the base leaves `all_guards_ok` true, so abstaining here would let
        // the composed governor authorise 4.0 fs with NO force evidence at all.
        // An upshift is a claim that the physics is verified safe; an unread
        // force pair is not verification.  Hold the current gear.
        *verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        *all_guards_ok = false;
        return cur;
    }
    if (!prism_gear_finite_f32(e->dt_prev_ps) || e->dt_prev_ps <= 0.0f ||
        !prism_gear_finite_f32(e->force_norm_prev) ||
        !prism_gear_finite_f32(base->force_norm)) {
        // Present but unusable: recorded as UNMEASURED, never guessed at.
        *verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        *all_guards_ok = false;
        return cur;
    }

    const float denom_floor = (prism_gear_finite_f32(p->force_rate_norm_floor) &&
                               p->force_rate_norm_floor > 0.0f)
                            ? p->force_rate_norm_floor : 1.0f;
    // `force_rate_norm_floor` is a DENOMINATOR floor only.  Clamping the
    // numerator as well under-reports real growth: with prev=0.1, floor=1.0 and
    // now=2.0 the true relative growth is 19/dt but a clamped numerator yields
    // 1/dt — a 19x under-report that errs toward a COARSER gear, which is the
    // one direction this file is not allowed to err in.
    const float prev  = (e->force_norm_prev > denom_floor) ? e->force_norm_prev : denom_floor;
    const float rate  = (base->force_norm - e->force_norm_prev) / (prev * e->dt_prev_ps);

    // Only GROWTH matters.  A collapsing force norm is a relaxation, not an
    // instability, and must not be punished with a finer gear.
    if (!prism_gear_finite_f32(rate) || rate <= 0.0f) {
        *verdict = PRISM_GEAR_VERDICT_OK;
        return PRISM_GEAR_MAX_ACTIVE;
    }
    if (rate >= p->force_rate_floor_per_ps) {
        *verdict = PRISM_GEAR_VERDICT_FLOOR;
        *instability = true;
        *all_guards_ok = false;
        return PRISM_GEAR_FINE;
    }
    if (rate >= p->force_rate_downshift_per_ps) {
        *verdict = PRISM_GEAR_VERDICT_DOWNSHIFT;
        *instability = true;
        *all_guards_ok = false;
        return (cur > PRISM_GEAR_FINE) ? (cur - 1u) : PRISM_GEAR_FINE;
    }
    if (rate >= p->force_rate_hold_per_ps) {
        *verdict = PRISM_GEAR_VERDICT_HOLD;
        *all_guards_ok = false;
        return cur;
    }
    *verdict = PRISM_GEAR_VERDICT_OK;
    return PRISM_GEAR_MAX_ACTIVE;
}

/// Band a per-candidate-gear observable landed in, expressed as the COARSEST
/// gear that band still permits.
///
/// This is the single place the "HOLD pins to the current gear, DOWNSHIFT pins
/// one finer" rule is written, shared by the acceleration and thermostat
/// guards so the two cannot drift apart.  `ok` / `hold` / `downshift` are the
/// policy-resolved band edges; `value` is the observable AT THE CANDIDATE GEAR.
///
/// `verdict_out` receives the band's own name, so the emitted record can say
/// which band produced the ceiling rather than inferring it.
__host__ __device__ __forceinline__ uint32_t prism_physgov_band_cap(
    float value, float ok, float hold, float downshift, uint32_t cur,
    uint32_t* verdict_out
) {
    if (!prism_gear_finite_f32(value)) {
        // An unresolvable observable at this gear is not licence to run it.
        *verdict_out = PRISM_GEAR_VERDICT_UNMEASURED;
        return PRISM_GEAR_FINE;
    }
    if (value < ok)        { *verdict_out = PRISM_GEAR_VERDICT_OK;        return PRISM_GEAR_MAX_ACTIVE; }
    if (value < hold)      { *verdict_out = PRISM_GEAR_VERDICT_HOLD;      return cur; }
    if (value < downshift) { *verdict_out = PRISM_GEAR_VERDICT_DOWNSHIFT; return prism_physgov_one_finer(cur); }
    *verdict_out = PRISM_GEAR_VERDICT_FLOOR;
    return PRISM_GEAR_FINE;
}

/// Guard 6 — ACCELERATION EXTREMES.  PER-CANDIDATE-GEAR: dx = 0.5*a*dt^2.
/// `emerg_out` receives the extension emergency bits.
__host__ __device__ __forceinline__ uint32_t prism_physgov_gate_acceleration(
    const PrismGearGovernorPolicy*    bp,
    const PrismPhysGovExtPolicy*      p,
    const PrismPhysGovExtObservation* e,
    uint32_t                          cur,
    uint32_t*                         verdict,
    uint32_t*                         emerg_out,
    bool*                             instability,
    bool*                             all_guards_ok
) {
    if (e->accel_measured == 0u) {
        *verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        if (p->require_accel_guard_for_upshift != 0u) {
            *all_guards_ok = false;
            return cur;
        }
        return PRISM_GEAR_MAX_ACTIVE;
    }
    if (!prism_gear_finite_f32(e->max_accel_a_per_ps2)) {
        *verdict = PRISM_GEAR_VERDICT_EMERGENCY;
        *emerg_out |= PRISM_PHYSGOV_EMERG_NONFINITE_ACCEL;
        *instability = true;
        *all_guards_ok = false;
        return PRISM_GEAR_FINE;
    }

    const float ref = (prism_gear_finite_f32(p->accel_displacement_reference_a) &&
                       p->accel_displacement_reference_a > 0.0f)
                    ? p->accel_displacement_reference_a : 1.0f;
    const float a   = (e->max_accel_a_per_ps2 > 0.0f) ? e->max_accel_a_per_ps2 : 0.0f;

    // EMERGENCY test at the FINEST gear: if even 0.5 fs cannot keep the
    // displacement below half a bond length, no gear can and SHAKE will not
    // reconverge.
    {
        const float dt0 = prism_physgov_gear_dt_ps(bp, PRISM_GEAR_FINE);
        if (prism_gear_finite_f32(dt0)) {
            const float dx0 = 0.5f * a * dt0 * dt0;
            if (dx0 >= p->accel_abort_fraction * ref) {
                *verdict = PRISM_GEAR_VERDICT_EMERGENCY;
                *emerg_out |= PRISM_PHYSGOV_EMERG_ACCEL_DISPLACEMENT;
                *instability = true;
                *all_guards_ok = false;
                return PRISM_GEAR_FINE;
            }
        }
    }

    // PER-CANDIDATE-GEAR: the coarsest gear its OWN band still permits.
    const float ok_edge   = p->accel_ok_fraction        * ref;
    const float hold_edge = p->accel_hold_fraction      * ref;
    const float down_edge = p->accel_downshift_fraction * ref;

    uint32_t ceiling = PRISM_GEAR_FINE;
    for (uint32_t g = 0u; g <= PRISM_GEAR_MAX_ACTIVE; ++g) {
        const float dt = prism_physgov_gear_dt_ps(bp, g);
        if (!prism_gear_finite_f32(dt)) continue;   /* no dt, no claim */
        uint32_t band_v = PRISM_GEAR_VERDICT_UNMEASURED;
        const uint32_t cap = prism_physgov_band_cap(
            0.5f * a * dt * dt, ok_edge, hold_edge, down_edge, cur, &band_v);
        if (g <= cap && g > ceiling) ceiling = g;
    }

    // The verdict names the band the CURRENT gear sits in — the timestep
    // actually in force — falling back to the ceiling's relationship to `cur`
    // when the current gear is itself inside the OK band.
    {
        const float dtc = prism_physgov_gear_dt_ps(bp, cur);
        uint32_t cur_band = PRISM_GEAR_VERDICT_UNMEASURED;
        (void)prism_physgov_band_cap(
            prism_gear_finite_f32(dtc) ? (0.5f * a * dtc * dtc) : (0.0f / 0.0f),
            ok_edge, hold_edge, down_edge, cur, &cur_band);
        if (cur_band == PRISM_GEAR_VERDICT_FLOOR ||
            cur_band == PRISM_GEAR_VERDICT_DOWNSHIFT) {
            *verdict = cur_band;
            *instability = true;
            *all_guards_ok = false;
        } else if (cur_band != PRISM_GEAR_VERDICT_OK) {
            *verdict = cur_band;               /* HOLD or UNMEASURED */
            *all_guards_ok = false;
        } else if (ceiling < cur) {
            *verdict = PRISM_GEAR_VERDICT_DOWNSHIFT;
            *instability = true;
            *all_guards_ok = false;
        } else if (ceiling < PRISM_GEAR_MAX_ACTIVE) {
            *verdict = PRISM_GEAR_VERDICT_HOLD;
            *all_guards_ok = false;
        } else {
            *verdict = PRISM_GEAR_VERDICT_OK;
        }
    }
    return ceiling;
}

/// Guard 7 — THERMOSTAT STABILITY.  PER-CANDIDATE-GEAR on gamma*dt, plus a
/// setpoint-deviation band.
__host__ __device__ __forceinline__ uint32_t prism_physgov_gate_thermostat(
    const PrismGearGovernorPolicy*    bp,
    const PrismPhysGovExtPolicy*      p,
    const PrismPhysGovExtObservation* e,
    uint32_t                          cur,
    uint32_t*                         verdict,
    uint32_t*                         emerg_out,
    bool*                             instability,
    bool*                             all_guards_ok
) {
    if (e->thermostat_measured == 0u) {
        *verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        if (p->require_thermostat_guard_for_upshift != 0u) {
            *all_guards_ok = false;
            return cur;
        }
        return PRISM_GEAR_MAX_ACTIVE;
    }

    uint32_t ceiling = PRISM_GEAR_MAX_ACTIVE;
    uint32_t v       = PRISM_GEAR_VERDICT_OK;

    // ── gamma*dt: PER-CANDIDATE-GEAR band evaluation ──
    if (prism_gear_finite_f32(e->effective_gamma_per_ps) &&
        e->effective_gamma_per_ps > 0.0f) {
        uint32_t g_ceiling = PRISM_GEAR_FINE;
        for (uint32_t g = 0u; g <= PRISM_GEAR_MAX_ACTIVE; ++g) {
            const float dt = prism_physgov_gear_dt_ps(bp, g);
            if (!prism_gear_finite_f32(dt)) continue;   /* no dt, no claim */
            uint32_t band_v = PRISM_GEAR_VERDICT_UNMEASURED;
            const uint32_t cap = prism_physgov_band_cap(
                e->effective_gamma_per_ps * dt,
                p->gamma_dt_ok, p->gamma_dt_hold, p->gamma_dt_downshift,
                cur, &band_v);
            if (g <= cap && g > g_ceiling) g_ceiling = g;
        }

        const float dtc = prism_physgov_gear_dt_ps(bp, cur);
        const float xc  = prism_gear_finite_f32(dtc)
                        ? (e->effective_gamma_per_ps * dtc) : (0.0f / 0.0f);
        uint32_t cur_band = PRISM_GEAR_VERDICT_UNMEASURED;
        (void)prism_physgov_band_cap(
            xc, p->gamma_dt_ok, p->gamma_dt_hold, p->gamma_dt_downshift,
            cur, &cur_band);

        // `gamma_dt_abort`: FLOOR, never EMERGENCY, but not decorative either.
        // c1 = exp(-gamma*dt) <= 0.37 means the velocity is essentially fully
        // resampled every step and the trajectory has stopped being inertial.
        // That is a positively-established loss of integrator character, so it
        // raises INSTABILITY — which zeroes the dwell counter and restarts the
        // lockout window for the next decision.
        if (prism_gear_finite_f32(xc) && xc >= p->gamma_dt_abort) {
            *instability = true;
        }

        if (cur_band != PRISM_GEAR_VERDICT_OK) {
            v = cur_band;
            *all_guards_ok = false;
        } else if (g_ceiling < cur) {
            v = PRISM_GEAR_VERDICT_DOWNSHIFT;
            *all_guards_ok = false;
        } else if (g_ceiling < PRISM_GEAR_MAX_ACTIVE) {
            v = PRISM_GEAR_VERDICT_HOLD;
            *all_guards_ok = false;
        }
        ceiling = prism_gear_min_u32(ceiling, g_ceiling);
    }

    // ── setpoint deviation ──
    //
    // SETPOINT AUTHORITY.  `dev` divides by the setpoint.  This block used to
    // substitute a literal 1.0f when `target_temp_k` was absent or
    // non-physical, and that substituted constant then became the DENOMINATOR.
    // At any real MD temperature that makes `dev` two to three orders of
    // magnitude larger than any band, so the EMERGENCY branch fired and the run
    // trapped on a runaway the instrument never observed.  A fabricated
    // denominator is a fabricated finding; an absent setpoint is UNMEASURED.
    //
    // Kept bit-identical to the Rust mirror in `physical_governor.rs`
    // (`gate_thermostat`), which `PARITY_VECTORS` /
    // `decide_on_compiled_evaluator` compare against this function.
    {
        const bool have_setpoint =
            prism_gear_finite_f32(e->target_temp_k) && e->target_temp_k > 1.0f;
        const float target = e->target_temp_k;
        if (!prism_gear_finite_f32(e->instantaneous_temp_k)) {
            // NaN is an ABSENT measurement, not a proven runaway.  FLOOR plus
            // instability; claiming EMERGENCY here would fabricate a finding
            // the instrument never made.
            v = PRISM_GEAR_VERDICT_FLOOR;
            ceiling = PRISM_GEAR_FINE;
            *instability = true;
            *all_guards_ok = false;
        } else if (have_setpoint) {
            const float dev = (e->instantaneous_temp_k > target)
                            ? (e->instantaneous_temp_k - target) / target
                            : (target - e->instantaneous_temp_k) / target;
            if (dev >= p->temp_dev_abort_fraction) {
                // Threefold temperature runaway at the default 2.00.  No
                // timestep repairs a diverging kinetic energy — trap.
                v = PRISM_GEAR_VERDICT_EMERGENCY;
                *emerg_out |= PRISM_PHYSGOV_EMERG_TEMPERATURE_RUNAWAY;
                ceiling = PRISM_GEAR_FINE;
                *instability = true;
                *all_guards_ok = false;
            } else if (dev >= p->temp_dev_downshift_fraction) {
                v = PRISM_GEAR_VERDICT_FLOOR;
                ceiling = PRISM_GEAR_FINE;
                *instability = true;
                *all_guards_ok = false;
            } else if (dev >= p->temp_dev_hold_fraction) {
                // Do not overwrite a stronger gamma-derived FLOOR: the record
                // must name the most authoritative cause that actually held,
                // not an incidental weaker one.
                if (v != PRISM_GEAR_VERDICT_FLOOR) v = PRISM_GEAR_VERDICT_DOWNSHIFT;
                ceiling = prism_gear_min_u32(ceiling, prism_physgov_one_finer(cur));
                *instability = true;
                *all_guards_ok = false;
            } else if (dev >= p->temp_dev_ok_fraction) {
                if (v == PRISM_GEAR_VERDICT_OK) v = PRISM_GEAR_VERDICT_HOLD;
                ceiling = prism_gear_min_u32(ceiling, cur);
                *all_guards_ok = false;
            }
        } else {
            // MEASURED instantaneous temperature, NO setpoint authority.  The
            // deviation guard has no denominator, so it makes no finding in
            // either direction — it neither licenses a coarser gear nor claims
            // a runaway.  It still blocks an UPSHIFT, mirroring the
            // `thermostat_measured == 0` handling at the top of this gate.
            // UNMEASURED is written only when the gamma half made no finding of
            // its own, so a real gamma verdict is never demoted.
            if (v == PRISM_GEAR_VERDICT_OK) v = PRISM_GEAR_VERDICT_UNMEASURED;
            if (p->require_thermostat_guard_for_upshift != 0u) {
                *all_guards_ok = false;
                ceiling = prism_gear_min_u32(ceiling, cur);
            }
        }
    }

    *verdict = v;
    return ceiling;
}

/// Guard 9 — NEIGHBOUR-LIST / INTEGRATION HEALTH.
__host__ __device__ __forceinline__ uint32_t prism_physgov_gate_integration_health(
    const PrismPhysGovExtPolicy*      p,
    const PrismPhysGovExtObservation* e,
    uint32_t                          cur,
    uint32_t*                         verdict,
    uint32_t*                         emerg_out,
    bool*                             instability,
    bool*                             all_guards_ok
) {
    if (e->integration_measured == 0u) {
        *verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        if (p->require_integration_guard_for_upshift != 0u) {
            *all_guards_ok = false;
            return cur;
        }
        return PRISM_GEAR_MAX_ACTIVE;
    }

    // A hard integration fault reported by the kernel is terminal.
    //
    // `integration_measured` above is ALREADY the discriminator, so this test
    // does not also treat 0xFFFFFFFF as "not measured".  An in-band sentinel on
    // a COUNT means a saturating fault counter that reaches 0xFFFFFFFF — the
    // most violent state representable — would read as "unmeasured" and produce
    // no emergency at all.  Invariant: a sentinel must be out-of-band.
    if (e->integration_faults != 0u) {
        *verdict = PRISM_GEAR_VERDICT_EMERGENCY;
        *emerg_out |= PRISM_PHYSGOV_EMERG_INTEGRATION_FAULT;
        *instability = true;
        *all_guards_ok = false;
        return PRISM_GEAR_FINE;
    }

    uint32_t ceiling = PRISM_GEAR_MAX_ACTIVE;
    uint32_t v       = PRISM_GEAR_VERDICT_OK;

    // ── neighbour-list occupancy ──
    if (e->neighbor_list_capacity != 0u &&
        e->neighbor_list_capacity != PRISM_PHYSGOV_U32_UNMEASURED &&
        e->max_neighbor_count != PRISM_PHYSGOV_U32_UNMEASURED) {
        const float occ = (float)e->max_neighbor_count / (float)e->neighbor_list_capacity;
        if (occ >= 1.0f) {
            // Truncating NOW: pair interactions are being dropped silently.
            v = PRISM_GEAR_VERDICT_FLOOR;
            ceiling = PRISM_GEAR_FINE;
            *all_guards_ok = false;
        } else if (occ >= p->neighbor_occupancy_hold) {
            v = PRISM_GEAR_VERDICT_DOWNSHIFT;
            ceiling = prism_gear_min_u32(
                ceiling, (cur > PRISM_GEAR_FINE) ? (cur - 1u) : PRISM_GEAR_FINE);
            *all_guards_ok = false;
        } else if (occ >= p->neighbor_occupancy_ok) {
            v = PRISM_GEAR_VERDICT_HOLD;
            ceiling = prism_gear_min_u32(ceiling, cur);
            *all_guards_ok = false;
        }
    }

    // ── clamped-force fraction ──
    if (prism_gear_finite_f32(e->clamped_force_fraction)) {
        const float f = e->clamped_force_fraction;
        if (f > p->clamped_fraction_floor) {
            v = PRISM_GEAR_VERDICT_FLOOR;
            ceiling = PRISM_GEAR_FINE;
            *instability = true;
            *all_guards_ok = false;
        } else if (f > p->clamped_fraction_downshift) {
            if (v != PRISM_GEAR_VERDICT_FLOOR) v = PRISM_GEAR_VERDICT_DOWNSHIFT;
            ceiling = prism_gear_min_u32(
                ceiling, (cur > PRISM_GEAR_FINE) ? (cur - 1u) : PRISM_GEAR_FINE);
            *instability = true;
            *all_guards_ok = false;
        } else if (f > p->clamped_fraction_hold) {
            if (v == PRISM_GEAR_VERDICT_OK) v = PRISM_GEAR_VERDICT_HOLD;
            ceiling = prism_gear_min_u32(ceiling, cur);
            *all_guards_ok = false;
        }
    } else {
        if (v == PRISM_GEAR_VERDICT_OK) v = PRISM_GEAR_VERDICT_UNMEASURED;
        if (p->require_integration_guard_for_upshift != 0u) {
            ceiling = prism_gear_min_u32(ceiling, cur);
            *all_guards_ok = false;
        }
    }

    *verdict = v;
    return ceiling;
}

/// TSH MEMORY + ANTI-BLINDNESS.  Ceiling-only, and structurally incapable of
/// raising a gear: every branch either returns PRISM_GEAR_MAX_ACTIVE (no
/// constraint) or something no larger.
__host__ __device__ __forceinline__ uint32_t prism_physgov_gate_tsh_memory(
    const PrismPhysGovExtPolicy*      p,
    const PrismPhysGovExtObservation* e,
    uint32_t                          cur,
    uint32_t*                         verdict,
    bool*                             all_guards_ok
) {
    if (e->tsh_memory_measured == 0u) {
        // No TSH arm bound.  That is a legitimate configuration (the governor
        // runs standalone), so it is NOT an upshift blocker on its own — the
        // physical guards remain in full force.
        *verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        return PRISM_GEAR_MAX_ACTIVE;
    }

    uint32_t ceiling = PRISM_GEAR_MAX_ACTIVE;
    uint32_t v       = PRISM_GEAR_VERDICT_OK;

    // (b) forced high-resolution sentinel — the strongest anti-blindness rule.
    if (p->max_steps_without_sentinel != 0u &&
        e->steps_since_sentinel != PRISM_PHYSGOV_U32_UNMEASURED &&
        e->steps_since_sentinel > p->max_steps_without_sentinel) {
        *verdict = PRISM_GEAR_VERDICT_FLOOR;
        *all_guards_ok = false;
        return PRISM_GEAR_FINE;
    }

    // (d) NO coarsening while observation confidence is below threshold.
    if (p->require_observation_confidence != 0u) {
        if (!prism_gear_finite_f32(e->observation_confidence) ||
            e->observation_confidence < p->min_observation_confidence) {
            v = PRISM_GEAR_VERDICT_HOLD;
            ceiling = prism_gear_min_u32(ceiling, cur);
            *all_guards_ok = false;
        }
    }

    // (a) minimum physical-time observation cadence.
    if (prism_gear_finite_f32(e->ps_since_last_observation) &&
        prism_gear_finite_f32(p->max_observation_interval_ps) &&
        p->max_observation_interval_ps > 0.0f &&
        e->ps_since_last_observation > p->max_observation_interval_ps) {
        if (v == PRISM_GEAR_VERDICT_OK) v = PRISM_GEAR_VERDICT_HOLD;
        ceiling = prism_gear_min_u32(ceiling, cur);
        *all_guards_ok = false;
    }

    // Integrated instability recollection.
    if (prism_gear_finite_f32(e->tsh_instability_memory)) {
        if (e->tsh_instability_memory >= p->tsh_memory_floor) {
            v = PRISM_GEAR_VERDICT_FLOOR;
            ceiling = PRISM_GEAR_FINE;
            *all_guards_ok = false;
        } else if (e->tsh_instability_memory >= p->tsh_memory_hold) {
            if (v != PRISM_GEAR_VERDICT_FLOOR) v = PRISM_GEAR_VERDICT_HOLD;
            ceiling = prism_gear_min_u32(ceiling, cur);
            *all_guards_ok = false;
        }
    }

    *verdict = v;
    return ceiling;
}

/// PROTOCOL-PHASE REQUIREMENT.  Ceiling-only.
__host__ __device__ __forceinline__ uint32_t prism_physgov_gate_phase(
    const PrismPhysGovExtPolicy*      p,
    const PrismPhysGovExtObservation* e,
    uint32_t                          cur,
    uint32_t*                         verdict
) {
    if (e->phase_measured == 0u) {
        *verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        return PRISM_GEAR_MAX_ACTIVE;
    }
    const uint32_t slots = (p->n_phase_slots > 8u) ? 8u : p->n_phase_slots;
    if (slots == 0u || e->phase_id >= slots) {
        // An unknown phase is not licence to run coarse.
        *verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        return cur;
    }
    uint32_t ceiling = p->phase_ceiling_gear[e->phase_id];
    if (ceiling > PRISM_GEAR_MAX_ACTIVE) ceiling = PRISM_GEAR_MAX_ACTIVE;
    if (ceiling < cur)                        *verdict = PRISM_GEAR_VERDICT_DOWNSHIFT;
    else if (ceiling < PRISM_GEAR_MAX_ACTIVE) *verdict = PRISM_GEAR_VERDICT_HOLD;
    else                                      *verdict = PRISM_GEAR_VERDICT_OK;
    return ceiling;
}

/// HMR HARD GATE — operator requirement 3, NOT POLICY-OVERRIDABLE.
///
/// Gear 2 is 4.0 fs.  A 4 fs timestep without hydrogen-mass repartitioning does
/// not crash; it silently produces plausible numbers with the wrong dynamics,
/// which is strictly worse than a crash because the output looks usable.
///
/// The base governor's topology gate reads `policy.hmr_permission`, a POLICY
/// field — a wrong policy file could assert GRANTED.  This gate reads
/// `observation.hmr_applied_verified`, which is TOPOLOGY-DERIVED
/// (`PrismPrepTopology::hmr_applied`).  Gear 2 requires BOTH.  There is no
/// policy field anywhere in `PrismPhysGovExtPolicy` that can disable this
/// function, and it is applied unconditionally AFTER every other computation in
/// `prism_physical_governor_decide`.
__host__ __device__ __forceinline__ uint32_t prism_physgov_gate_hmr_hard(
    const PrismPhysGovExtObservation* e,
    uint32_t*                         verdict
) {
    if (e->hmr_applied_verified == PRISM_GEAR_PERMISSION_GRANTED) {
        *verdict = PRISM_GEAR_VERDICT_OK;
        return PRISM_GEAR_MAX_ACTIVE;
    }
    if (e->hmr_applied_verified == PRISM_GEAR_PERMISSION_DENIED) {
        *verdict = PRISM_GEAR_VERDICT_DOWNSHIFT;
    } else {
        // UNKNOWN.  Absence of evidence is not evidence of HMR.
        *verdict = PRISM_GEAR_VERDICT_UNMEASURED;
    }
    return PRISM_GEAR_SAFE;   /* 2.0 fs ceiling; Gear 2 unreachable */
}

// ═══════════════════════════════════════════════════════════════════════════
// THE COMPOSED GOVERNOR
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluate the full physical stability governor.
///
/// Pure function of (base policy, extension policy, base observation,
/// extension observation, salience request).  No I/O, no global state, no
/// randomness, no atomics, no allocation — so the host FFI entry point, a host
/// unit test and the device kernel provably agree.
///
/// EXECUTION ORDER inside the fused decision path (architecture §1):
///   1 read force/constraint/phase/probe/tensor inputs   <- caller
///   2 form QNANL scientific state                       <- caller
///   3 update TSH using physical-time-normalised evidence<- caller (TSH arm)
///   4 generate resolution request                       <- caller (TSH arm)
///   5 evaluate HMR/constraint/drift/force/instability   <- THIS FUNCTION
///   6 select the conservative authorized gear           <- THIS FUNCTION
///   7 apply or stage the new dt at a validated boundary <- caller
///   8 increment cumulative physical time with the APPLIED dt <- caller
///   9 emit a provenance-bound state + gear record       <- caller, from `out`
///
/// TIGHTEN-ONLY: `out->authorized_gear <= base_decision.authorized_gear` for
/// every possible input.  Asserted by
/// `physical_governor_gates.rs::extension_never_loosens_the_base`.
__host__ __device__ __forceinline__ void prism_physical_governor_decide(
    const PrismGearGovernorPolicy*    bp,
    const PrismPhysGovExtPolicy*      ep,
    const PrismGearGuardObservation*  bo,
    const PrismPhysGovExtObservation* eo,
    const PrismTshSalienceRequest*    tsh,   /* nullable */
    PrismPhysGovDecision*             out
) {
    const uint32_t span = (bp->chunk_span_steps == 0u) ? 1u : bp->chunk_span_steps;
    const uint32_t cur =
        (bo->current_gear <= PRISM_GEAR_MAX_ACTIVE) ? bo->current_gear : PRISM_GEAR_FINE;

    // ── Base governor: gates 1..10, already lexicographic internally ───────
    PrismGearDecision base;
    prism_gearbox_governor_decide(bp, bo, tsh, &base);

    // ── Fixed, deterministic preamble.  EMIT ON NO-CHANGE TOO: every field
    //    below is written on every invocation, so a record always exists and a
    //    gap in the stream unambiguously means "the decision did not run". ──
    out->schema_version               = PRISM_PHYSGOV_SCHEMA_VERSION;
    out->base_authorized_gear         = base.authorized_gear;
    out->base_ceiling_gear            = base.ceiling_gear;
    out->base_binding_gate            = base.binding_gate;
    out->base_gate_verdicts           = base.gate_verdicts;
    out->ext_gate_verdicts            = 0u;
    out->emergency_flags              = base.emergency_flags;
    out->ext_emergency_flags          = 0u;
    out->requested_gear               = bo->requested_gear;
    out->current_gear                 = bo->current_gear;
    out->next_stable_dwell_steps      = base.next_stable_dwell_steps;
    out->next_steps_since_instability = base.next_steps_since_instability;
    out->downshift_latency_steps      = span;
    out->decision_emitted             = 1u;
    out->ext_armed                    = ep->armed;
    out->policy_digest                = bp->policy_digest;
    out->ext_policy_digest            = ep->ext_policy_digest;
    out->ext_binding_refusal          = PRISM_PHYSGOV_REFUSAL_NONE;
    for (uint32_t i = 0u; i < 5u; ++i) out->_reserved[i] = 0u;

    // CHUNK-BOUNDARY CONTRACT: the named, emitted latency bound.
    {
        const float dtc_fs = (cur <= PRISM_GEAR_MAX_ACTIVE) ? bp->gear_dt_fs[cur] : 0.0f;
        out->downshift_latency_fs =
            prism_gear_finite_f32(dtc_fs) ? ((float)span * dtc_fs) : (0.0f / 0.0f);
    }

    // ── Base emergency is terminal.  Nothing below can lift it. ───────────
    //
    // ABORT SENTINEL PLACEMENT.  `authorized_gear` carries the abort; the
    // ceiling does NOT.  `ceiling_gear` is documented as the operand of
    // `min(request, ceiling)`, and `PRISM_GEAR_ABORT` is the LARGEST value in
    // that field — the one arithmetic that erases it (`min(1, 3) == 1` would
    // keep integrating at 2.0 fs straight through a physical emergency).  The
    // ceiling is therefore pinned to the FINEST gear on an abort: a consumer
    // that honours the trap traps, and a consumer that only knows how to `min`
    // is driven to the most conservative timestep instead of the least.
    if (base.authorized_gear == PRISM_GEAR_ABORT ||
        base.binding_gate == PRISM_GEAR_GATE_PHYSICAL_EMERGENCY) {
        out->authorized_gear   = PRISM_GEAR_ABORT;
        out->ceiling_gear      = PRISM_GEAR_FINE;
        out->ext_binding_gate  = PRISM_PHYSGOV_GATE_NONE;
        out->binding_gate_rank = PRISM_PHYSGOV_RANK_PHYSICAL_EMERGENCY;
        out->transition        = PRISM_PHYSGOV_TRANSITION_ABORT;
        out->dt_applied_ps     = 0.0f / 0.0f;   /* Gear 3 has no timestep */
        return;
    }

    // ── EXTENSION BINDING GATE (the extension analogue of base gate 11) ───
    //
    // Runs BEFORE any guard.  A zeroed or wrong-version policy makes every
    // threshold below meaningless: `force_rate_floor_per_ps == 0` returns FLOOR
    // for any positive force rate, `accel_ok_fraction == 0` makes no gear OK,
    // `clamped_fraction_floor == 0` FLOORs on any clipping — the record would
    // be indistinguishable from genuinely violent physics.  Refusing here means
    // "nobody armed the extension" is never reported as a physics finding.
    //
    // Fail CLOSED: the cap is the documented `PRISM_PHYSGOV_DEFAULT_UNARMED_CAP_GEAR`,
    // not a value read out of the policy that could not be trusted.  It never
    // deadlocks — Gear 0 is always reachable — and the cause is named.
    {
        uint32_t refusal = PRISM_PHYSGOV_REFUSAL_NONE;
        if (ep->armed == 0u) {
            refusal = PRISM_PHYSGOV_REFUSAL_UNARMED;
        } else if (ep->schema_version != PRISM_PHYSGOV_SCHEMA_VERSION) {
            refusal = PRISM_PHYSGOV_REFUSAL_POLICY_SCHEMA;
        } else if (eo->schema_version != PRISM_PHYSGOV_SCHEMA_VERSION) {
            refusal = PRISM_PHYSGOV_REFUSAL_OBSERVATION_SCHEMA;
        }
        if (refusal != PRISM_PHYSGOV_REFUSAL_NONE) {
            const uint32_t cap = PRISM_PHYSGOV_DEFAULT_UNARMED_CAP_GEAR;
            for (uint32_t g = PRISM_PHYSGOV_GATE_EXT_FIRST;
                 g <= PRISM_PHYSGOV_GATE_EXT_LAST; ++g) {
                prism_physgov_set_verdict(&out->ext_gate_verdicts, g,
                                          PRISM_GEAR_VERDICT_UNMEASURED);
            }
            out->ext_binding_refusal = refusal;
            out->ceiling_gear    = prism_gear_min_u32(base.ceiling_gear, cap);
            out->authorized_gear = prism_gear_min_u32(base.authorized_gear, cap);
            out->ext_binding_gate  = PRISM_PHYSGOV_GATE_EXT_BINDING;
            out->binding_gate_rank = PRISM_PHYSGOV_RANK_GOVERNOR_UNARMED;
            if      (out->authorized_gear < cur) out->transition = PRISM_PHYSGOV_TRANSITION_DOWNSHIFT;
            else if (out->authorized_gear > cur) out->transition = PRISM_PHYSGOV_TRANSITION_UPSHIFT;
            else                                 out->transition = PRISM_PHYSGOV_TRANSITION_HOLD;
            // No guard was evaluated, so no dwell was earned.
            out->next_stable_dwell_steps = 0u;
            out->dt_applied_ps = prism_physgov_gear_dt_ps(bp, out->authorized_gear);
            return;
        }
    }

    // ── Extension gates ───────────────────────────────────────────────────
    uint32_t ext_emerg     = 0u;
    bool     instability   = false;
    bool     all_guards_ok = true;

    uint32_t ext_ceiling[PRISM_PHYSGOV_CEILING_GATE_COUNT];
    uint32_t ext_gate_id[PRISM_PHYSGOV_CEILING_GATE_COUNT];
    for (uint32_t i = 0u; i < PRISM_PHYSGOV_CEILING_GATE_COUNT; ++i) {
        ext_ceiling[i] = PRISM_GEAR_MAX_ACTIVE;
        ext_gate_id[i] = PRISM_PHYSGOV_GATE_NONE;
    }

    uint32_t v = PRISM_GEAR_VERDICT_UNMEASURED;

    // HMR HARD (rank 15) — evaluated first among the extension gates so its
    // ceiling is present before anything else can be attributed.
    ext_gate_id[0] = PRISM_PHYSGOV_GATE_HMR_HARD;
    ext_ceiling[0] = prism_physgov_gate_hmr_hard(eo, &v);
    prism_physgov_set_verdict(&out->ext_gate_verdicts, PRISM_PHYSGOV_GATE_HMR_HARD, v);
    // Deliberately does NOT touch `all_guards_ok`: an unverified HMR topology
    // is a permanent ceiling, not a transient degradation, and folding it into
    // the dwell accumulator would reset the dwell every decision and make the
    // 2.0 fs gear unreachable on a non-HMR topology.

    // FORCE SPIKE (rank 52)
    ext_gate_id[1] = PRISM_PHYSGOV_GATE_FORCE_SPIKE;
    ext_ceiling[1] = prism_physgov_gate_force_spike(
        ep, bo, eo, cur, &v, &instability, &all_guards_ok);
    prism_physgov_set_verdict(&out->ext_gate_verdicts, PRISM_PHYSGOV_GATE_FORCE_SPIKE, v);

    // ACCELERATION EXTREMES (rank 54)
    ext_gate_id[2] = PRISM_PHYSGOV_GATE_ACCELERATION_EXTREME;
    ext_ceiling[2] = prism_physgov_gate_acceleration(
        bp, ep, eo, cur, &v, &ext_emerg, &instability, &all_guards_ok);
    prism_physgov_set_verdict(&out->ext_gate_verdicts,
                              PRISM_PHYSGOV_GATE_ACCELERATION_EXTREME, v);

    // THERMOSTAT STABILITY (rank 56)
    ext_gate_id[3] = PRISM_PHYSGOV_GATE_THERMOSTAT_STABILITY;
    ext_ceiling[3] = prism_physgov_gate_thermostat(
        bp, ep, eo, cur, &v, &ext_emerg, &instability, &all_guards_ok);
    prism_physgov_set_verdict(&out->ext_gate_verdicts,
                              PRISM_PHYSGOV_GATE_THERMOSTAT_STABILITY, v);

    // INTEGRATION HEALTH (rank 58)
    ext_gate_id[4] = PRISM_PHYSGOV_GATE_INTEGRATION_HEALTH;
    ext_ceiling[4] = prism_physgov_gate_integration_health(
        ep, eo, cur, &v, &ext_emerg, &instability, &all_guards_ok);
    prism_physgov_set_verdict(&out->ext_gate_verdicts,
                              PRISM_PHYSGOV_GATE_INTEGRATION_HEALTH, v);

    // TSH MEMORY + ANTI-BLINDNESS (rank 70)
    ext_gate_id[5] = PRISM_PHYSGOV_GATE_TSH_MEMORY;
    ext_ceiling[5] = prism_physgov_gate_tsh_memory(ep, eo, cur, &v, &all_guards_ok);
    prism_physgov_set_verdict(&out->ext_gate_verdicts, PRISM_PHYSGOV_GATE_TSH_MEMORY, v);

    // PHASE REQUIREMENT (rank 90)
    ext_gate_id[6] = PRISM_PHYSGOV_GATE_PHASE_REQUIREMENT;
    ext_ceiling[6] = prism_physgov_gate_phase(ep, eo, cur, &v);
    prism_physgov_set_verdict(&out->ext_gate_verdicts,
                              PRISM_PHYSGOV_GATE_PHASE_REQUIREMENT, v);

    out->ext_emergency_flags = ext_emerg;

    // ── Extension emergency: terminal, same class as base gate 1 ──────────
    //
    // Same abort-sentinel placement as the base emergency above: the ceiling is
    // pinned FINE, never GEAR_ABORT, so `min(request, ceiling)` cannot erase
    // the trap.
    if (ext_emerg != 0u) {
        out->authorized_gear   = PRISM_GEAR_ABORT;
        out->ceiling_gear      = PRISM_GEAR_FINE;
        if (ext_emerg & PRISM_PHYSGOV_EMERG_INTEGRATION_FAULT) {
            out->ext_binding_gate = PRISM_PHYSGOV_GATE_INTEGRATION_HEALTH;
        } else if (ext_emerg & PRISM_PHYSGOV_EMERG_TEMPERATURE_RUNAWAY) {
            out->ext_binding_gate = PRISM_PHYSGOV_GATE_THERMOSTAT_STABILITY;
        } else {
            out->ext_binding_gate = PRISM_PHYSGOV_GATE_ACCELERATION_EXTREME;
        }
        out->binding_gate_rank = PRISM_PHYSGOV_RANK_PHYSICAL_EMERGENCY;
        out->transition        = PRISM_PHYSGOV_TRANSITION_ABORT;
        out->dt_applied_ps     = 0.0f / 0.0f;
        out->next_stable_dwell_steps      = 0u;
        out->next_steps_since_instability = 0u;
        return;
    }

    // ── Fold the extension ceilings ───────────────────────────────────────
    uint32_t ext_min = PRISM_GEAR_MAX_ACTIVE;
    for (uint32_t i = 0u; i < PRISM_PHYSGOV_CEILING_GATE_COUNT; ++i) {
        ext_min = prism_gear_min_u32(ext_min, ext_ceiling[i]);
    }

    // ── THE SINGLE TIGHTEN-ONLY FOLD ──────────────────────────────────────
    //
    // Structural guarantee: whatever the extension gates computed, the result
    // is a `min` against the base decision.  An extension defect can only
    // refuse a gear that was available; it can never authorise one that was
    // not.
    out->ceiling_gear    = prism_gear_min_u32(base.ceiling_gear, ext_min);
    uint32_t authorized  = prism_gear_min_u32(base.authorized_gear, ext_min);

    // ── HARD PHYSICS GATE, re-asserted unconditionally (requirement 3) ────
    //
    // Applied last and outside every policy switch.  Even if every band above
    // were mis-tuned, Gear 2 is unreachable without verified HMR.
    if (eo->hmr_applied_verified != PRISM_GEAR_PERMISSION_GRANTED) {
        if (authorized > PRISM_GEAR_SAFE)          authorized = PRISM_GEAR_SAFE;
        if (out->ceiling_gear > PRISM_GEAR_SAFE)   out->ceiling_gear = PRISM_GEAR_SAFE;
    }

    // ── ASYMMETRY (requirement 4) ─────────────────────────────────────────
    //
    // The base already applied dwell / lockout / upshift-rate to its own
    // authorized gear.  Because this layer only ever tightens, an upshift that
    // survives here is one the base already licensed, so the asymmetry holds
    // transitively.  The one addition: extension guard degradation observed in
    // THIS decision must also veto an upshift the base licensed on the base
    // guards alone.
    if (authorized > cur && !all_guards_ok) {
        authorized = cur;
    }

    out->authorized_gear = authorized;

    // ── Attribution: most authoritative gate at the binding ceiling ───────
    {
        uint32_t best_rank = PRISM_PHYSGOV_RANK_NONE;
        uint32_t best_ext  = PRISM_PHYSGOV_GATE_NONE;
        for (uint32_t i = 0u; i < PRISM_PHYSGOV_CEILING_GATE_COUNT; ++i) {
            if (ext_ceiling[i] != out->ceiling_gear) continue;
            if (ext_ceiling[i] >= PRISM_GEAR_MAX_ACTIVE) continue;
            const uint32_t r = prism_physgov_ext_gate_rank(ext_gate_id[i]);
            if (r < best_rank) { best_rank = r; best_ext = ext_gate_id[i]; }
        }
        const uint32_t base_rank =
            (base.ceiling_gear == out->ceiling_gear && base.ceiling_gear < PRISM_GEAR_MAX_ACTIVE)
                ? prism_physgov_base_gate_rank(base.binding_gate)
                : PRISM_PHYSGOV_RANK_NONE;

        if (base_rank <= best_rank) {
            out->ext_binding_gate  = PRISM_PHYSGOV_GATE_NONE;
            out->binding_gate_rank = base_rank;
        } else {
            out->ext_binding_gate  = best_ext;
            out->binding_gate_rank = best_rank;
        }
        // The HMR hard gate always owns the attribution when it is what pinned
        // the ceiling to Gear 1 — a 4 fs refusal must never read as a
        // thermostat or phase finding.
        if (eo->hmr_applied_verified != PRISM_GEAR_PERMISSION_GRANTED &&
            out->ceiling_gear == PRISM_GEAR_SAFE &&
            out->binding_gate_rank > PRISM_PHYSGOV_RANK_HMR_HARD) {
            out->ext_binding_gate  = PRISM_PHYSGOV_GATE_HMR_HARD;
            out->binding_gate_rank = PRISM_PHYSGOV_RANK_HMR_HARD;
        }
    }

    // ── Transition classification (emitted even when nothing changed) ─────
    if      (out->authorized_gear <  cur) out->transition = PRISM_PHYSGOV_TRANSITION_DOWNSHIFT;
    else if (out->authorized_gear >  cur) out->transition = PRISM_PHYSGOV_TRANSITION_UPSHIFT;
    else                                  out->transition = PRISM_PHYSGOV_TRANSITION_HOLD;

    // ── Dwell / instability bookkeeping for the NEXT decision ─────────────
    if (instability) {
        out->next_stable_dwell_steps      = 0u;
        out->next_steps_since_instability = 0u;
    } else if (!all_guards_ok) {
        // Guards not all OK but nothing unstable: dwell does not accumulate,
        // the lockout window keeps running.
        out->next_stable_dwell_steps = 0u;
    }
    // A gear change itself restarts the physical dwell — evidence that gear A
    // was stable is not evidence that gear B is.  The base applies this to its
    // OWN authorized gear (`gearbox.cuh:1050`); it must also be applied to the
    // COMPOSED gear, or an extension-driven downshift carries forward dwell
    // earned at the coarser gear and licenses an upshift with zero stable steps
    // actually spent at the finer one.
    if (out->authorized_gear != cur) {
        out->next_stable_dwell_steps = 0u;
    }

    // ── The applied timestep ──────────────────────────────────────────────
    out->dt_applied_ps = prism_physgov_gear_dt_ps(bp, out->authorized_gear);
}

// ═══════════════════════════════════════════════════════════════════════════
// Structural self-checks
// ═══════════════════════════════════════════════════════════════════════════

/// The lexicographic authority order, asserted as a compile-time chain.  If
/// anybody renumbers a rank so a scientific gate outranks a physical one, this
/// fails to compile rather than silently permitting a salience to outvote a
/// prohibition.
static_assert(PRISM_PHYSGOV_RANK_PHYSICAL_EMERGENCY   < PRISM_PHYSGOV_RANK_HMR_HARD,             "rank order");
static_assert(PRISM_PHYSGOV_RANK_HMR_HARD             < PRISM_PHYSGOV_RANK_TOPOLOGY_PERMISSION,  "rank order");
static_assert(PRISM_PHYSGOV_RANK_TOPOLOGY_PERMISSION  < PRISM_PHYSGOV_RANK_GOVERNOR_UNARMED,     "rank order");
static_assert(PRISM_PHYSGOV_RANK_GOVERNOR_UNARMED     < PRISM_PHYSGOV_RANK_CONSTRAINT_STABILITY, "rank order");
static_assert(PRISM_PHYSGOV_RANK_CONSTRAINT_STABILITY < PRISM_PHYSGOV_RANK_ENERGY_DRIFT,         "rank order");
static_assert(PRISM_PHYSGOV_RANK_ENERGY_DRIFT         < PRISM_PHYSGOV_RANK_FORCE_STABILITY,      "rank order");
static_assert(PRISM_PHYSGOV_RANK_FORCE_STABILITY      < PRISM_PHYSGOV_RANK_FORCE_SPIKE,          "rank order");
static_assert(PRISM_PHYSGOV_RANK_FORCE_SPIKE          < PRISM_PHYSGOV_RANK_ACCELERATION_EXTREME, "rank order");
static_assert(PRISM_PHYSGOV_RANK_ACCELERATION_EXTREME < PRISM_PHYSGOV_RANK_THERMOSTAT_STABILITY, "rank order");
static_assert(PRISM_PHYSGOV_RANK_THERMOSTAT_STABILITY < PRISM_PHYSGOV_RANK_INTEGRATION_HEALTH,   "rank order");
static_assert(PRISM_PHYSGOV_RANK_INTEGRATION_HEALTH   < PRISM_PHYSGOV_RANK_CHUNK_LATENCY,        "rank order");
static_assert(PRISM_PHYSGOV_RANK_CHUNK_LATENCY        < PRISM_PHYSGOV_RANK_TSH_MEMORY,           "rank order");
static_assert(PRISM_PHYSGOV_RANK_TSH_MEMORY           < PRISM_PHYSGOV_RANK_TSH_SALIENCE,         "rank order");
static_assert(PRISM_PHYSGOV_RANK_TSH_SALIENCE         < PRISM_PHYSGOV_RANK_PHASE_REQUIREMENT,    "rank order");
static_assert(PRISM_PHYSGOV_RANK_PHASE_REQUIREMENT    < PRISM_PHYSGOV_RANK_DWELL,                "rank order");
static_assert(PRISM_PHYSGOV_RANK_DWELL                < PRISM_PHYSGOV_RANK_LOCKOUT,              "rank order");
static_assert(PRISM_PHYSGOV_RANK_LOCKOUT              < PRISM_PHYSGOV_RANK_UPSHIFT_RATE,         "rank order");
static_assert(PRISM_PHYSGOV_RANK_UPSHIFT_RATE         < PRISM_PHYSGOV_RANK_NONE,                 "rank order");

/// Extension gate ids must not collide with base gate ids.
static_assert(PRISM_PHYSGOV_GATE_EXT_FIRST > PRISM_GEAR_GATE_COUNT,
              "extension gate ids must be disjoint from PRISM_GEAR_GATE_*");
static_assert(PRISM_PHYSGOV_GATE_EXT_LAST - PRISM_PHYSGOV_GATE_EXT_FIRST + 1u
                  == PRISM_PHYSGOV_GATE_EXT_COUNT,
              "PRISM_PHYSGOV_GATE_EXT_COUNT must span FIRST..LAST inclusive");
/// The verdict nibbles must fit the 32-bit packed field, or a gate's verdict
/// would silently overwrite another gate's.
static_assert(PRISM_PHYSGOV_GATE_EXT_COUNT * 4u <= 32u,
              "extension verdict nibbles must fit ext_gate_verdicts");
/// The ceiling-fold gate count is the extension gate count minus the
/// binding-refusal gate, which replaces the fold rather than joining it.
static_assert(PRISM_PHYSGOV_CEILING_GATE_COUNT + 1u == PRISM_PHYSGOV_GATE_EXT_COUNT,
              "exactly one extension gate stands outside the ceiling fold");
/// The unarmed cap must be a real, always-reachable gear, or a refusal would
/// deadlock the run instead of slowing it.
static_assert(PRISM_PHYSGOV_DEFAULT_UNARMED_CAP_GEAR <= PRISM_GEAR_MAX_ACTIVE,
              "the unarmed cap must be an active gear");

/// Band monotonicity: a HOLD threshold that sits above a DOWNSHIFT threshold
/// would make the bands unreachable in the order they are tested.
static_assert(PRISM_PHYSGOV_DEFAULT_FORCE_RATE_HOLD_PER_PS
                  < PRISM_PHYSGOV_DEFAULT_FORCE_RATE_DOWNSHIFT_PER_PS,
              "force-rate bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_FORCE_RATE_DOWNSHIFT_PER_PS
                  < PRISM_PHYSGOV_DEFAULT_FORCE_RATE_FLOOR_PER_PS,
              "force-rate bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_ACCEL_OK_FRACTION
                  < PRISM_PHYSGOV_DEFAULT_ACCEL_HOLD_FRACTION,
              "acceleration bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_ACCEL_HOLD_FRACTION
                  < PRISM_PHYSGOV_DEFAULT_ACCEL_DOWNSHIFT_FRACTION,
              "acceleration bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_ACCEL_DOWNSHIFT_FRACTION
                  < PRISM_PHYSGOV_DEFAULT_ACCEL_ABORT_FRACTION,
              "acceleration bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_GAMMA_DT_OK < PRISM_PHYSGOV_DEFAULT_GAMMA_DT_HOLD,
              "gamma*dt bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_GAMMA_DT_HOLD < PRISM_PHYSGOV_DEFAULT_GAMMA_DT_DOWNSHIFT,
              "gamma*dt bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_GAMMA_DT_DOWNSHIFT < PRISM_PHYSGOV_DEFAULT_GAMMA_DT_ABORT,
              "gamma*dt bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_TEMP_DEV_OK_FRACTION
                  < PRISM_PHYSGOV_DEFAULT_TEMP_DEV_HOLD_FRACTION,
              "temperature-deviation bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_TEMP_DEV_HOLD_FRACTION
                  < PRISM_PHYSGOV_DEFAULT_TEMP_DEV_DOWNSHIFT_FRACTION,
              "temperature-deviation bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_TEMP_DEV_DOWNSHIFT_FRACTION
                  < PRISM_PHYSGOV_DEFAULT_TEMP_DEV_ABORT_FRACTION,
              "temperature-deviation bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_NEIGHBOR_OCCUPANCY_OK
                  < PRISM_PHYSGOV_DEFAULT_NEIGHBOR_OCCUPANCY_HOLD,
              "neighbour-occupancy bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_HOLD
                  < PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_DOWNSHIFT,
              "clamped-fraction bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_DOWNSHIFT
                  < PRISM_PHYSGOV_DEFAULT_CLAMPED_FRACTION_FLOOR,
              "clamped-fraction bands must be monotone");
static_assert(PRISM_PHYSGOV_DEFAULT_TSH_MEMORY_HOLD < PRISM_PHYSGOV_DEFAULT_TSH_MEMORY_FLOOR,
              "tsh-memory bands must be monotone");

/// The HMR hard gate's ceiling must be strictly below the campaign gear, or the
/// gate is decorative.
static_assert(PRISM_GEAR_SAFE < PRISM_GEAR_CAMPAIGN,
              "HMR hard gate requires GEAR_SAFE < GEAR_CAMPAIGN");
