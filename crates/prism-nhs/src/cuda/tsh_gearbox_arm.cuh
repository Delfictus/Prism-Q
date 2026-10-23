// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D — GEARBOX-FACING TSH ARM (device-side salience update)
// Package 4 of the Coupled Temporal / Gearbox / Tensor-Family architecture.
// ═══════════════════════════════════════════════════════════════════════════
//
// AUTHORITY BOUNDARY — read this before touching anything below.
//
//   This arm determines scientific-resolution DEMAND.  It has NO authority to
//   authorize a timestep.  It writes a `PrismTshSalienceRequest` (the typed
//   insertion point defined in `gearbox.cuh`) and nothing else.  The PHYSICAL
//   STABILITY GOVERNOR is the sole authority over what is SAFE, and the
//   gearbox applies `min(demanded, permitted)`.  A salience of 1.0 cannot
//   obtain a gear the physical gates refused — that is the entire point of the
//   lexicographic ordering in `prism_gearbox_governor_decide`.
//
//   Consequently this header NEVER writes PRISM_GEAR_ABORT.  Aborting is a
//   physical decision the governor owns; the request type refuses to encode
//   one and `prism_tsh_arm_decide` clamps to PRISM_GEAR_MAX_ACTIVE.
//
// ───────────────────────────────────────────────────────────────────────────
// THE CRITICAL CONSTRAINT — salience is normalised per SIMULATED PICOSECOND
// ───────────────────────────────────────────────────────────────────────────
//
// Never per record, never per chunk, never per kernel invocation.  If salience
// were measured per record the loop has positive feedback in the wrong
// direction:
//
//     low salience -> coarser gear -> fewer snapshots -> less evidence
//                  -> lower measured salience -> coarser still -> ...
//
// which is stable in the WRONG equilibrium: coarsening is rewarded by its own
// consequence, so it does not self-correct.  The controller talks itself into
// blindness during exactly the quiet interval that precedes a transition.
//
// Three mechanisms make that structurally impossible here.
//
//   1. Every branch integrates its evidence with the exact solution of
//
//          dA/dt = -A/tau + r(t),      r = activity per simulated picosecond
//
//      over the SIMULATED-TIME span the evidence covers:
//
//          A <- A * exp(-s/tau) + r * tau * (1 - exp(-s/tau))
//
//      This composes exactly under subdivision of the interval: integrating a
//      constant-rate window of span s in one step, or in two steps of
//      s1 + s2 = s, yields the identical A.  `sample_count` is carried for
//      provenance only and is NEVER a divisor anywhere in this header.
//
//   2. THE ACTIVITY INTEGRATOR IS FROZEN OVER UNOBSERVED SIMULATED TIME.
//      Decaying it there would integrate r = 0 — i.e. it would record ABSENCE
//      OF EVIDENCE as MEASURED CALM, which is exactly the positive feedback
//      above.  Measured on constant physics whose true salience is 0.5000, the
//      decaying form read 0.5000 at full coverage, 0.3045 at a 50% duty cycle,
//      0.1395 at 20% (below the upshift threshold: COARSENING ELIGIBLE) and
//      0.0376 at 5%.  Same physics, four different answers, chosen by the
//      observer.  With the integrator frozen the steady state is A -> r*tau
//      regardless of duty cycle, so the salience of constant physics is one
//      number.  What sparse observation costs is CONFIDENCE, not activity.
//
//      A branch that has genuinely gone quiet says so by REPORTING calm
//      (accumulated_activity = 0 over a real span), which drives the
//      integrator down exactly as before.  SILENCE IS NOT CALM.
//
//   3. STAGED EVIDENCE IS NEVER TRUNCATED.  When a staged window is longer
//      than the decision interval the surplus is CARRIED FORWARD at the same
//      rate rather than discarded, so an observation cadence coarser than the
//      decision cadence integrates the identical total evidence.
//
// Therefore changing the SAMPLING RATE, the DUTY CYCLE, or the DECISION RATE
// at constant physics leaves the salience unchanged — up to float rounding.
//
// ───────────────────────────────────────────────────────────────────────────
// ANTI-BLINDNESS — all four are mandatory, none is optional, all four bite
// ───────────────────────────────────────────────────────────────────────────
//
//   (a) MINIMUM PHYSICAL-TIME OBSERVATION CADENCE.  A branch whose newest
//       evidence is older than `min_observation_cadence_ps` starts losing
//       confidence.  The cadence is in picoseconds of SIMULATED time, so it
//       cannot be satisfied by emitting more records at a coarser gear.  Past
//       `max_sentinel_interval_ps` of staleness the confidence is hard ZERO,
//       not an exponential tail: with the integrator frozen, an arbitrarily
//       old activity would otherwise keep voting forever.
//
//   (b) MAXIMUM INTERVAL WITHOUT A FORCED HIGH-RESOLUTION SENTINEL SAMPLE.
//       Independent of salience, independent of memory state, independent of
//       confidence: once `max_sentinel_interval_ps` of simulated time have
//       elapsed since the last sample actually taken at PRISM_GEAR_FINE, the
//       arm requests PRISM_GEAR_FINE.  The sentinel clock is reset ONLY by an
//       authorized gear that was MEASURED to be fine — never by the request
//       itself.  If the governor keeps refusing, the sentinel stays due and
//       the arm keeps asking; it never silently gives up.
//
//   (c) UNCERTAINTY GROWS WHEN EVIDENCE IS SPARSE — MEASURED, NOT ASSERTED.
//       Branch confidence is `recency * coverage`, where COVERAGE is the
//       leaky-integrated fraction of recent simulated time that staged
//       evidence actually covered.  A branch that reports punctually but
//       covers 20% of the simulated time reports 0.2, not 1.0.  Recency alone
//       — the first revision — returned 1.000 in every sparse-coverage regime,
//       so gate (d) waved through exactly what it exists to refuse.
//
//       A branch that has NEVER reported has confidence 0.0 and contributes
//       NOTHING to the integrated salience — it does NOT contribute a zero.  A
//       branch reporting calm and a branch reporting nothing are different
//       facts and are represented differently.  When no branch has admissible
//       evidence the integrated salience is NaN (unmeasured), not 0.0.
//
//   (d) NO COARSENING WHILE OBSERVATION CONFIDENCE IS BELOW THRESHOLD.  The
//       coarsening branch of the decision is gated on
//       `confidence >= min_confidence_for_coarsening`.  Finer resolution may
//       be triggered by salience alone; coarser requires low salience AND
//       sufficient confidence AND dwell AND a permissive memory state — and
//       then still only reaches the governor as a request.
//
// ───────────────────────────────────────────────────────────────────────────
// AGGREGATION IS LEXICOGRAPHIC, NEVER A SCORE
// ───────────────────────────────────────────────────────────────────────────
//
// Every resolution gate is taken on the PEAK admissible branch salience, never
// on the confidence-weighted mean.  A mean lets calm branches outvote an
// excursion: with six equal weights, one branch at 0.999 and five calm reads
// 0.1665 — below the upshift threshold, i.e. COARSENING ELIGIBLE — so a
// maximal single-family demand produced a request for a COARSER gear, and
// wiring up more evidence families made the arm blinder to each one.  The mean
// survives as `PrismTshArmRecord.integrated_salience`, a REPORTED STATISTIC
// that gates nothing; `peak_branch_salience` is the gate input and travels on
// every record so the verdict is auditable.
//
// ───────────────────────────────────────────────────────────────────────────
// ASC IS THE FORCE BUFFER — BRANCH 4 IS SIGNAL-GRID ONLY
// ───────────────────────────────────────────────────────────────────────────
//
// `{stem}_streamNN_asc_vectors.bin` is written from the `host_forces` buffer.
// What ships under the name "ASC vectors" IS the force observation under
// another filename, not an independent state.  A branch reading it alongside
// PRISM_TSH_ARM_BRANCH_FORCE_ENERGY does not add a second observation of the
// system — it counts one observation twice, giving force 2/6 of the aggregate
// weight against 1/6 for every genuinely distinct family, with nothing on the
// record to disclose it.
//
// PRISM_TSH_ARM_BRANCH_SIGNAL_GRID_WARP is therefore SIGNAL-GRID ONLY, its
// default weight is 0.0, and `prism_tsh_arm_observe` REFUSES an observation on
// it unless the resolved policy sets `signal_grid_warp_independent`.  Evidence
// that genuinely derives from the force buffer belongs on
// PRISM_TSH_ARM_BRANCH_FORCE_ENERGY, where it is counted exactly once.
//
// ───────────────────────────────────────────────────────────────────────────
// DETERMINISM
// ───────────────────────────────────────────────────────────────────────────
//
// Every reduction in this header is a fixed-order serial loop over branch
// indices 0..PRISM_TSH_ARM_N_BRANCHES-1 executed by ONE thread.  There is no
// atomicAdd anywhere.  Float addition is not associative, so an atomicAdd-based
// reduction varies run to run in the low bits and can flip a threshold
// comparison — two identical runs would then take different gears.  The
// existing `zstr_force_stage_f4_kernel` uses atomicAdd for sum|F|^2; that
// pattern is deliberately NOT reused for any decision input.
//
// EVERY POLICY THRESHOLD IS `double` ON BOTH SIDES.  The first revision stored
// them as `float` here while the host used `f64`, so `0.60` meant
// 0.60000002384185791015625 on this side and 0.59999999999999997779 on the
// other: the two implementations compared against literally different numbers,
// and a float `branch_tau_ps` perturbed the integrator by ~1e-7 relative PER
// STEP, compounding over a trajectory.  That can flip a gear decision at a
// threshold boundary — the failure mode the determinism rule exists to prevent,
// arriving by a route the doc did not admit.  Matching widths removes it.
//
// Host and device `exp()` may still differ in the last ULP.  That does not
// weaken run-to-run determinism (same binary + same device is exact); it does
// mean the host mirror in `crates/prism-nhs/src/tsh_gearbox_arm.rs` is the
// REFERENCE ORACLE for policy and tests, not a bit-exact replica of a device
// run.  Whichever side actually ran is the side whose record is emitted.
//
// ───────────────────────────────────────────────────────────────────────────
// HOT PATH
// ───────────────────────────────────────────────────────────────────────────
//
// `prism_tsh_arm_decide` performs O(6) flops and one 80-byte structure store.
// No JSON, no hashing, no compression, no filesystem, no allocation, no
// printf.  The record goes to a device ring (`PrismTshArmRing`) which is a
// SINGLE-PRODUCER ring written by the same single thread that decides — hence
// no atomics on `head` either.  `record_sequence` increments on EVERY decision
// including a no-change decision, so a gap in the emitted stream is
// unambiguously "the decision did not run" and never "the gear held".
//
// ───────────────────────────────────────────────────────────────────────────
// CHUNK REALITY
// ───────────────────────────────────────────────────────────────────────────
//
// Gear flips apply end-of-chunk when body_unroll > 1.  This arm therefore
// produces a request for the WHOLE NEXT CHUNK; "immediate downshift" is
// bounded by chunk latency and is stated as such.  Mid-chunk correction is not
// available; within-chunk emergencies remain the abort trap's job (gear 3),
// which this header never requests.
//
// ───────────────────────────────────────────────────────────────────────────
// COMPILATION — this header IS compiled
// ───────────────────────────────────────────────────────────────────────────
//
// `__host__ __device__ __forceinline__` throughout, exactly like
// `prism_gearbox_governor_decide`, so one implementation serves both the
// captured-graph kernel and any host-side evaluation.
//
// `crates/prism-nhs/src/cuda/tsh_gearbox_arm.cu` is a real translation unit
// that includes this header, is registered in `crates/prism-nhs/build.rs`, and
// exports host-callable `extern "C"` entry points.  That matters for two
// reasons that are not stylistic:
//
//   * every `static_assert` below EXECUTES on every build.  An uncompiled
//     header's assertions are vacuous — they look like enforcement and enforce
//     nothing, which is the built-but-never-activated defect this package
//     exists to repair.
//   * the Rust parity test calls THIS implementation and compares it against
//     the host oracle.  The previous parity test compared the oracle against
//     numeric literals transcribed into a comment, which proved only that the
//     comment and the oracle agreed.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>

#include "gearbox.cuh"

// ═══════════════════════════════════════════════════════════════════════════
// Schema + cardinality
// ═══════════════════════════════════════════════════════════════════════════

/// Wire-format version of every struct in this header.  A reader that sees any
/// other value MUST fail loudly rather than misparse.
#define PRISM_TSH_ARM_SCHEMA_VERSION 1u

/// Six dendritic branches.  Fixed order; the order IS the reduction order.
#define PRISM_TSH_ARM_N_BRANCHES 6u

// ─── Branch identities ──────────────────────────────────────────────────────
//
// Each branch accumulates independently and they are integrated only at the
// end.  A branch is a distinct evidence FAMILY, not a distinct sensor: the
// families are the ones named by the operator directive.

/// Force / energy stability: force norms, Hamiltonian drift proxies.
#define PRISM_TSH_ARM_BRANCH_FORCE_ENERGY 0u
/// KCC and coupling change.
#define PRISM_TSH_ARM_BRANCH_KCC_COUPLING 1u
/// Probe interaction.
#define PRISM_TSH_ARM_BRANCH_PROBE_INTERACTION 2u
/// Geometry / contact transition (aromatic proximity, contact make/break).
#define PRISM_TSH_ARM_BRANCH_GEOMETRY_CONTACT 3u
/// Signal grid / warp activity — NOT ASC.  See the ASC/force block at the top:
/// the engine's only "ASC" product is the force buffer under another filename,
/// so this branch is bound only when the policy declares an independent
/// signal-grid observable, and its default weight is 0.0.
#define PRISM_TSH_ARM_BRANCH_SIGNAL_GRID_WARP 4u
/// Deprecated spelling of PRISM_TSH_ARM_BRANCH_SIGNAL_GRID_WARP.  Kept
/// reachable so existing references still resolve; the name asserted an
/// independence from the force representation that the evidence does not have.
#define PRISM_TSH_ARM_BRANCH_SIGNAL_WARP_ASC PRISM_TSH_ARM_BRANCH_SIGNAL_GRID_WARP
/// Prior gearbox instability (downshifts, floors, emergencies already seen).
#define PRISM_TSH_ARM_BRANCH_PRIOR_GEARBOX_INSTABILITY 5u

// ─── Branch mask layout inside `PrismTshArmRecord.active_branch_mask` ───────
//
// Three 6-bit fields in one u32.  Kept in one word because the record is on
// the hot path and 64 bytes is one sector.

/// Bits 0..5 — the branch delivered FRESH evidence in this decision window.
#define PRISM_TSH_ARM_MASK_FRESH_SHIFT 0u
/// Bits 8..13 — the branch is SALIENT (its own salience >= downshift bound).
#define PRISM_TSH_ARM_MASK_SALIENT_SHIFT 8u
/// Bits 16..21 — the branch is STALE (ever observed, but confidence < 1.0).
#define PRISM_TSH_ARM_MASK_STALE_SHIFT 16u
/// Bits 24..29 — the branch has NEVER been observed (structurally unbound).
#define PRISM_TSH_ARM_MASK_UNBOUND_SHIFT 24u

// ─── Memory states (operator's table) ──────────────────────────────────────
//
//   baseline / low      -> may permit coarser IF physically safe
//   accumulating        -> hold, or begin downshift
//   threshold approach  -> downshift
//   active memory       -> maintain fine
//   reinforced          -> keep fine
//   input withdrawal    -> hold fine temporarily
//   decay               -> gradual upshift eligibility
//   rechallenge         -> immediate downshift

#define PRISM_TSH_MEMORY_BASELINE           0u
#define PRISM_TSH_MEMORY_LOW                1u
#define PRISM_TSH_MEMORY_ACCUMULATING       2u
#define PRISM_TSH_MEMORY_THRESHOLD_APPROACH 3u
#define PRISM_TSH_MEMORY_ACTIVE             4u
#define PRISM_TSH_MEMORY_REINFORCED         5u
#define PRISM_TSH_MEMORY_INPUT_WITHDRAWAL   6u
#define PRISM_TSH_MEMORY_DECAY              7u
#define PRISM_TSH_MEMORY_RECHALLENGE        8u

// ─── Request reasons ────────────────────────────────────────────────────────

/// No decision has been taken yet.
#define PRISM_TSH_REASON_UNSET 0u
/// Gear held; nothing in the evidence justified a change.
#define PRISM_TSH_REASON_HOLD 1u
/// Salience crossed the downshift bound: request finer.
#define PRISM_TSH_REASON_SALIENCE_DOWNSHIFT 2u
/// Memory state mandates fine resolution.
#define PRISM_TSH_REASON_MEMORY_HOLD_FINE 3u
/// Rechallenge inside the rechallenge window: immediate finest.
#define PRISM_TSH_REASON_RECHALLENGE 4u
/// Anti-blindness (b): forced high-resolution sentinel sample.
#define PRISM_TSH_REASON_SENTINEL 5u
/// Anti-blindness (c): no branch has evidence; bias fine.
#define PRISM_TSH_REASON_EVIDENCE_ABSENT 6u
/// Low salience, sufficient confidence, dwell satisfied: request coarser.
#define PRISM_TSH_REASON_UPSHIFT_ELIGIBLE 7u
/// Coarsening was otherwise eligible but the confidence gate refused it.
#define PRISM_TSH_REASON_CONFIDENCE_GATE 8u
/// Coarsening was otherwise eligible but minimum dwell had not elapsed.
#define PRISM_TSH_REASON_DWELL_GATE 9u
/// No policy was ever resolved; the arm declines to have an opinion.
#define PRISM_TSH_REASON_POLICY_UNARMED 10u
/// Memory reached the threshold-approach state: downshift one gear.
#define PRISM_TSH_REASON_MEMORY_THRESHOLD_DOWNSHIFT 11u

// ─── Record flags ──────────────────────────────────────────────────────────

#define PRISM_TSH_FLAG_SENTINEL_FORCED     0x01u
#define PRISM_TSH_FLAG_EVIDENCE_ABSENT     0x02u
#define PRISM_TSH_FLAG_CONFIDENCE_BLOCKED  0x04u
#define PRISM_TSH_FLAG_DWELL_BLOCKED       0x08u
#define PRISM_TSH_FLAG_RECHALLENGE         0x10u
#define PRISM_TSH_FLAG_RING_DROPPED        0x20u
#define PRISM_TSH_FLAG_POLICY_UNARMED      0x40u
#define PRISM_TSH_FLAG_AUTHORIZED_UNMEASURED 0x80u
#define PRISM_TSH_FLAG_INVALID_REFERENCE   0x100u
/// ANTI-BLINDNESS (c): weighted observation coverage was below
/// `min_confidence_for_coarsening` on this decision.  Raised whenever it is
/// true, not only inside the coarsening branch, so a reader can see the arm
/// was under-observed even while it was demanding FINER resolution.
#define PRISM_TSH_FLAG_OBSERVATION_SPARSE  0x200u
/// A branch's newest evidence was older than `max_sentinel_interval_ps`; its
/// frozen activity was refused as evidence about the present.
#define PRISM_TSH_FLAG_STALE_HORIZON       0x400u
/// A signal-grid observation was refused because the policy does not declare an
/// independent signal-grid observable.  A producer is wired to the wrong branch.
#define PRISM_TSH_FLAG_SIGNAL_GRID_UNBOUND 0x800u
/// A branch's carried-forward evidence backlog exceeded the staleness horizon
/// and its oldest surplus was dropped.  The producer is staging simulated time
/// faster than the arm is deciding.  Bounded and reported, never silent.
#define PRISM_TSH_FLAG_BACKLOG_TRUNCATED   0x1000u

/// Explicit "not measured" for a u32 slot in this header.  Shares the encoding
/// of PRISM_GEAR_UNSET so a gear slot never has to guess whether `0` means
/// "fine" or "unset".
#define PRISM_TSH_ARM_UNMEASURED_U32 0xFFFFFFFFu

// ─── Codec-registered thermodynamic action opcodes ─────────────────────────
//
// These mirror `ThermodynamicActionOpcode` in
// `crates/prism-tokenizer/src/types.rs` (artifact family ThermodynamicAction,
// slot 74, field `tsh_state`).  The registered opcode had ZERO producers; this
// arm is its producer, and the value travels on every emitted record.
//
// 0 is deliberately NOT an opcode, so a zero-filled slot cannot decode as one.

#define PRISM_TSH_ACTION_ABSENT        0u
#define PRISM_TSH_ACTION_STABLE        1u
#define PRISM_TSH_ACTION_SALIENCE_RISE 2u
#define PRISM_TSH_ACTION_SALIENCE_FALL 3u
#define PRISM_TSH_ACTION_HYSTERESIS    4u
#define PRISM_TSH_ACTION_BOUNDARY      5u
#define PRISM_TSH_ACTION_HANDOFF       6u

// ═══════════════════════════════════════════════════════════════════════════
// DOCUMENTED DEFAULTS — the ONLY place a TSH-arm threshold is a literal
// ═══════════════════════════════════════════════════════════════════════════
//
// Every one of these is a POLICY DEFAULT, not a measurement.  The host
// resolver (`crate::tsh_gearbox_arm::resolve_arm_policy`) starts from these,
// applies operator policy, and records BOTH the resolved value AND its origin
// on every emitted record.  Nothing downstream may treat a default as evidence.

/// Leaky-integrator time constant for every branch, in SIMULATED picoseconds.
///
/// Rationale — reuse, not invention: the governor's documented cruise window
/// is PRISM_GEARBOX_CRUISE_THRESHOLD = 500 steps, and the campaign gear
/// (Gear 2) is 4.0 fs.  500 x 4 fs = 2000 fs = 2.0 ps.  So one integrator time
/// constant is exactly one cruise window of campaign-gear physics: long enough
/// to integrate a regime change, short enough that 2 ps-old evidence no longer
/// dominates the present demand.
#define PRISM_TSH_ARM_DEFAULT_BRANCH_TAU_PS 2.0

/// ANTI-BLINDNESS (a).  Minimum physical-time observation cadence, in
/// SIMULATED picoseconds.  Evidence newer than this keeps a branch at full
/// confidence.  Default = tau/4: at least four observations per integrator
/// time constant, which is the Nyquist-with-margin condition for the
/// integrator to TRACK a regime change instead of aliasing it.
#define PRISM_TSH_ARM_DEFAULT_MIN_OBSERVATION_CADENCE_PS 0.5

/// ANTI-BLINDNESS (c).  Confidence e-folding time once a branch is past
/// cadence, in SIMULATED picoseconds.  Default = the cadence itself: one full
/// missed cadence interval costs a factor of e of confidence.  One anchor,
/// two uses.
#define PRISM_TSH_ARM_DEFAULT_CONFIDENCE_DECAY_PS 0.5

/// ANTI-BLINDNESS (d).  Observation confidence required before COARSENING may
/// even be considered.  Default 0.75: three quarters of the weighted evidence
/// the arm expects must actually be present and fresh.  Below it the arm holds
/// or asks finer, never coarser.
#define PRISM_TSH_ARM_DEFAULT_MIN_CONFIDENCE_FOR_COARSENING 0.75

/// ANTI-BLINDNESS (b).  Maximum SIMULATED time between samples actually taken
/// at PRISM_GEAR_FINE, in picoseconds.  Default = 5 tau: no more than five
/// e-foldings of the slowest structure the arm can represent may pass without
/// a high-resolution look, no matter how quiet the evidence claims to be.
#define PRISM_TSH_ARM_DEFAULT_MAX_SENTINEL_INTERVAL_PS 10.0

/// How long a forced sentinel sample is held at the fine gear, in SIMULATED
/// picoseconds.  Default 0.05 ps = 50 fs, matching the governor's own
/// documented `PRISM_GEAR_DEFAULT_MAX_UNCORRECTED_WINDOW_FS`: one period of
/// the fastest motion the constrained system resolves.  A sentinel shorter
/// than one period of the fastest resolved motion is not a sample of it.
#define PRISM_TSH_ARM_DEFAULT_SENTINEL_HOLD_PS 0.05

/// HYSTERESIS, downshift side.  Integrated salience at or above which the arm
/// requests FINER resolution.  The reference rate is the half-saturation point
/// (salience 0.5), so 0.60 means "the observed activity rate is about 1.5x the
/// policy reference".  Downshift is fast and easy by design.
#define PRISM_TSH_ARM_DEFAULT_SALIENCE_DOWNSHIFT 0.60

/// HYSTERESIS, upshift side.  Integrated salience STRICTLY BELOW which
/// coarsening becomes eligible.  0.25 means "the observed activity rate has
/// fallen to about a third of the policy reference".  The gap to 0.60 is the
/// hysteresis band that prevents gear chatter under a noisy input.  Upshift is
/// slow and gated by construction.
#define PRISM_TSH_ARM_DEFAULT_SALIENCE_UPSHIFT 0.25

/// Integrated salience at or above which an active memory becomes REINFORCED.
/// 0.85 corresponds to roughly 5.7x the policy reference rate.
#define PRISM_TSH_ARM_DEFAULT_SALIENCE_REINFORCE 0.85

/// Minimum SIMULATED time the arm must hold its current requested gear before
/// requesting a COARSER one.  Default = tau: coarsening at most once per
/// integrator time constant.  Requests for FINER resolution are exempt.
#define PRISM_TSH_ARM_DEFAULT_MIN_GEAR_DWELL_PS 2.0

/// SIMULATED time a decaying memory must persist before each single step of
/// upshift eligibility.  Default = tau.  "Gradual" is the operator's word and
/// this is what makes it gradual.
#define PRISM_TSH_ARM_DEFAULT_DECAY_PERIOD_PS 2.0

/// SIMULATED time fine resolution is HELD after input withdrawal, before the
/// memory is allowed to enter decay.  Default = tau.
#define PRISM_TSH_ARM_DEFAULT_WITHDRAWAL_HOLD_PS 2.0

/// SIMULATED time after leaving active memory during which a renewed salience
/// excursion counts as a RECHALLENGE and triggers an immediate downshift.
/// Default = 5 tau, the same anchor as the sentinel interval.
#define PRISM_TSH_ARM_DEFAULT_RECHALLENGE_WINDOW_PS 10.0

/// SIMULATED time salience must stay at or above the reinforce bound before
/// active memory is promoted to reinforced.  Default = tau/2.
#define PRISM_TSH_ARM_DEFAULT_REINFORCE_DWELL_PS 1.0

/// Default per-branch integration weight for every branch that is a distinct
/// evidence family.  Equal: the arm has no evidence that one family predicts
/// resolution demand better than another, and inventing an unequal prior would
/// be a fabricated measurement.
#define PRISM_TSH_ARM_DEFAULT_BRANCH_WEIGHT 1.0

/// Default weight for PRISM_TSH_ARM_BRANCH_SIGNAL_GRID_WARP — ZERO.
///
/// Not a prior about how informative signal-grid activity is.  The engine has
/// no independent signal-grid observable today: the file that ships as
/// `{stem}_streamNN_asc_vectors.bin` is the `host_forces` buffer.  Weighting
/// this branch alongside the force branch would give force two sixths of the
/// aggregate against one sixth for every other family, silently.  Zero is the
/// honest default; the operator raises it at the same time as declaring
/// `signal_grid_warp_independent`.
#define PRISM_TSH_ARM_DEFAULT_SIGNAL_GRID_WARP_WEIGHT 0.0

/// Whether the resolved policy declares an independent signal-grid observable.
/// 0 = not independent, which is what the current emitters justify.
#define PRISM_TSH_ARM_DEFAULT_SIGNAL_GRID_INDEPENDENT 0u

/// Default per-branch reference activity RATE, in activity-units per SIMULATED
/// picosecond.  This is the half-saturation point of the branch response
/// `s = r / (r + ref)`.
///
/// 1.0 is a UNIT reference: it asserts that the producer delivers an activity
/// measure already normalised so that one unit per simulated picosecond is
/// "ordinary".  A family whose natural units differ MUST set this from policy.
/// Leaving it at the default is a recorded operator choice, not a measurement,
/// and the emitted origin says so.
///
/// # THE STAGED QUANTITY MUST BE INTENSIVE
///
/// `s = r / (r + ref)` is only a measurement when `r` is a RATE OF CHANGE or a
/// NORMALISED DEVIATION.  Stage an EXTENSIVE magnitude instead — an absolute
/// force norm multiplied by the elapsed span, an absolute count — and the
/// division by the span hands back the magnitude itself; against `ref = 1.0`
/// the ratio pins to ~0.999 on EVERY decision and the branch records a constant
/// that a reader cannot distinguish from a measurement.  The host reference
/// oracle states the per-family units in
/// `SalienceBranch::activity_unit` (`tsh_gearbox_arm.rs`); both sides must mean
/// the same thing by this number:
///
///   branch 0 force / energy      : d|F|max / max|F| per simulated ps  [ps^-1]
///   branch 3 geometry / contact  : angstrom RMS aromatic-centroid drift per
///                                  simulated ps                      [A/ps]
///
/// Half-saturation at 1.0 is therefore "the extreme force turns over completely
/// in one picosecond" and "an aromatic ring traverses one pi-stacking
/// contact-formation length in one picosecond" respectively — physically
/// meaningful excursions, not arbitrary scale factors.
#define PRISM_TSH_ARM_DEFAULT_BRANCH_REFERENCE_RATE 1.0

/// Canonical gear timesteps in fs, mirroring `d_gearbox_table` slot 0
/// (0.0005 / 0.0020 / 0.0040 ps).  Gear 3 is the abort sentinel and has no
/// timestep; this arm never requests it.
#define PRISM_TSH_ARM_DEFAULT_DT_FS_G0 0.5
#define PRISM_TSH_ARM_DEFAULT_DT_FS_G1 2.0
#define PRISM_TSH_ARM_DEFAULT_DT_FS_G2 4.0
#define PRISM_TSH_ARM_DEFAULT_DT_FS_G3 0.0

// ═══════════════════════════════════════════════════════════════════════════
// Structures
// ═══════════════════════════════════════════════════════════════════════════

/// Per-branch persistent state.  64 bytes.
///
/// `pending_activity` / `pending_span_ps` are the CURRENT decision window's
/// staged evidence.  They are accumulated by `prism_tsh_arm_observe` and
/// consumed by `prism_tsh_arm_decide`, which carries any surplus FORWARD
/// rather than discarding it.  Staging them as a (sum activity, sum span) pair
/// rather than a running mean is what makes the per-simulated-picosecond
/// normalisation exact under any split of the window into samples: the sample
/// count cancels identically.
struct __align__(16) PrismTshArmBranchState {
    /// Leaky-integrated activity, units: activity.  Rate = activity / tau.
    /// FROZEN over simulated time no evidence covered — see mechanism 2.
    double   activity;            // @  0
    /// Simulated-time coordinate (ps) of the newest evidence ever staged.
    double   last_evidence_ps;    // @  8
    /// Sum of staged activity for the current decision window.
    double   pending_activity;    // @ 16
    /// Sum of staged SIMULATED span (ps) for the current decision window.
    double   pending_span_ps;     // @ 24
    /// Leaky-integrated observation COVERAGE.  `coverage_activity / tau` is the
    /// fraction of recent simulated time that staged evidence actually covered,
    /// in [0,1].  Driven by 1 over covered spans and 0 over uncovered ones, and
    /// — unlike `activity` — NOT frozen over uncovered time: the absence itself
    /// is the thing this integrator measures.
    double   coverage_activity;   // @ 32
    /// 1 once this branch has ever delivered evidence.  0 means UNBOUND, which
    /// is a different fact from "reported calm" and is treated differently.
    uint32_t ever_observed;       // @ 40
    /// Samples staged in the current window.  PROVENANCE ONLY.  This value is
    /// never a divisor anywhere in this header — dividing by it is exactly the
    /// per-record normalisation the architecture forbids.
    uint32_t pending_samples;     // @ 44
    /// Total samples ever staged.  Provenance only.
    uint64_t total_samples;       // @ 48
    uint64_t _reserved;           // @ 56
};

static_assert(sizeof(PrismTshArmBranchState) == 64,
              "PrismTshArmBranchState MUST be 64 bytes (Rust mirror pins the same).");
static_assert(alignof(PrismTshArmBranchState) == 16,
              "PrismTshArmBranchState MUST be 16-byte aligned.");
static_assert(offsetof(PrismTshArmBranchState, coverage_activity) == 32,
              "coverage_activity offset drift: must be 32.");
static_assert(offsetof(PrismTshArmBranchState, ever_observed) == 40,
              "ever_observed offset drift: must be 40.");

/// Resolved TSH-arm policy.  Immutable for the life of a run.  Every field is
/// resolved on the host with a recorded ORIGIN; nothing here is a measurement.
/// 288 bytes.
///
/// EVERY THRESHOLD IS `double`.  See the determinism block at the top: as
/// `float` these compared against different numbers than the host oracle did.
/// The field ORDER of the fourteen scalars matches
/// `TshArmPolicy::scalar_fields()` exactly, because the host encoder writes
/// them from that array.
struct __align__(16) PrismTshArmPolicy {
    /// Must equal PRISM_TSH_ARM_SCHEMA_VERSION.
    uint32_t schema_version;                  // @   0
    /// 1 once a host has resolved and uploaded a policy.  While 0 the arm
    /// declines to have an opinion (`PRISM_TSH_SALIENCE_UNSET`) rather than
    /// guessing — an unarmed policy layer leaves the governor in sole control.
    uint32_t armed;                           // @   4
    /// 1 when PRISM_TSH_ARM_BRANCH_SIGNAL_GRID_WARP is bound to an observable
    /// INDEPENDENT of the force representation.  While 0, observations on that
    /// branch are refused rather than double-counted against force.
    uint32_t signal_grid_warp_independent;    // @   8
    uint32_t _pad0;                           // @  12
    /// Per-branch integration weight, fixed order.
    double   branch_weight[PRISM_TSH_ARM_N_BRANCHES];          // @  16
    /// Per-branch half-saturation reference rate (activity per simulated ps).
    double   branch_reference_rate[PRISM_TSH_ARM_N_BRANCHES];  // @  64
    /// Leaky-integrator time constant, simulated ps.  Must be > 0.
    double   branch_tau_ps;                   // @ 112
    /// ANTI-BLINDNESS (a).
    double   min_observation_cadence_ps;      // @ 120
    /// ANTI-BLINDNESS (c).
    double   confidence_decay_ps;             // @ 128
    /// ANTI-BLINDNESS (d).
    double   min_confidence_for_coarsening;   // @ 136
    /// ANTI-BLINDNESS (b).  Also the staleness HORIZON past which a branch
    /// contributes nothing at all.
    double   max_sentinel_interval_ps;        // @ 144
    double   sentinel_hold_ps;                // @ 152
    /// HYSTERESIS.
    double   salience_downshift_threshold;    // @ 160
    double   salience_upshift_threshold;      // @ 168
    double   salience_reinforce_threshold;    // @ 176
    double   min_gear_dwell_ps;               // @ 184
    double   decay_period_ps;                 // @ 192
    double   withdrawal_hold_ps;              // @ 200
    double   rechallenge_window_ps;           // @ 208
    double   reinforce_dwell_ps;              // @ 216
    /// Per-gear timestep in fs, derived from the live gear table.
    double   gear_dt_fs[4];                   // @ 224
    /// Host-computed fingerprint of (values + per-field origins + policy-file
    /// sha256).  Copied verbatim into the checkpoint so a restore under a
    /// different policy is detectable rather than silent.
    uint64_t policy_digest;                   // @ 256
    uint64_t _reserved[3];                    // @ 264
};

static_assert(sizeof(PrismTshArmPolicy) == 288,
              "PrismTshArmPolicy MUST be 288 bytes (Rust mirror pins the same).");
static_assert(alignof(PrismTshArmPolicy) == 16,
              "PrismTshArmPolicy MUST be 16-byte aligned.");
static_assert(offsetof(PrismTshArmPolicy, branch_weight) == 16,
              "branch_weight offset drift: must be 16.");
static_assert(offsetof(PrismTshArmPolicy, branch_tau_ps) == 112,
              "branch_tau_ps offset drift: must be 112 (first of the 14 scalars).");
static_assert(offsetof(PrismTshArmPolicy, gear_dt_fs) == 224,
              "gear_dt_fs offset drift: must be 224.");
static_assert(offsetof(PrismTshArmPolicy, policy_digest) == 256,
              "policy_digest offset drift: must be 256.");
// The fourteen scalars must be contiguous and in the host's `scalar_fields()`
// order, because the host encoder writes them as one run.
static_assert(offsetof(PrismTshArmPolicy, reinforce_dwell_ps) ==
                  offsetof(PrismTshArmPolicy, branch_tau_ps) + 13 * sizeof(double),
              "the 14 policy scalars must be contiguous doubles in host order.");

/// Persistent arm state.  PATH-DEPENDENT: reproducing a run requires
/// reproducing the salience trajectory, so the seed alone is insufficient and
/// this whole struct is checkpointed.  512 bytes.
///
/// `TshArmState::to_device_bytes` / `from_device_bytes` on the host are the
/// declared bridge to and from this layout, mapping `Option<f64>` <-> the
/// ±INFINITY sentinels below and NEVER `None` -> 0.0.  Without that bridge the
/// captured-graph path — the path this header exists to serve — could not be
/// checkpointed at all.
///
/// `record_sequence` and `decisions` are 64-bit.  As `uint32_t` they wrapped
/// silently after 2^32 decisions, and the mitigation the host comment offered
/// ("reconstruct the high word from the host record") does not exist on a path
/// whose whole point is that there is no host record.
struct __align__(16) PrismTshArmState {
    /// Must equal PRISM_TSH_ARM_SCHEMA_VERSION.
    uint32_t schema_version;        // @   0
    /// 1 while a rechallenge window is open.
    uint32_t rechallenge_armed;     // @   4
    /// PRISM_TSH_MEMORY_*.
    uint32_t memory_state;          // @   8
    /// The gear this arm last REQUESTED.  Not what ran.
    uint32_t last_requested_gear;   // @  12
    /// Monotonic.  Increments on EVERY decision, including no-change ones, so
    /// a gap in the record stream is never ambiguous between "gear held" and
    /// "decision did not run".
    uint64_t record_sequence;       // @  16
    /// Decisions taken since reset.  Provenance only.
    uint64_t decisions;             // @  24
    /// Simulated-time coordinate (ps) of the last decision.
    double   now_ps;                // @  32
    /// Simulated time (ps) at which the current memory state was entered.
    double   memory_entered_ps;     // @  40
    /// Simulated time (ps) of the last MEASURED fine-gear sample.  Reset only
    /// by an authorized gear measured as PRISM_GEAR_FINE.
    double   last_sentinel_ps;      // @  48
    /// Simulated time (ps) at which the requested gear last changed.
    double   request_changed_ps;    // @  56
    /// Simulated time (ps) at which active memory was last left.  -INFINITY
    /// until it has been; never 0.0, which is a legitimate coordinate.
    double   active_memory_exit_ps; // @  64
    /// Simulated time (ps) since which salience has been at/above reinforce.
    /// +INFINITY when it is not.
    double   reinforce_since_ps;    // @  72
    /// Simulated time (ps) at which the current sentinel hold ends.
    /// -INFINITY when no hold is active.
    double   sentinel_hold_until_ps;// @  80
    /// Last PEAK salience (the gate input), or NaN when unmeasured.
    double   last_salience;         // @  88
    /// Last observation confidence, or NaN when unmeasured.
    double   last_confidence;       // @  96
    /// Last weighted observation coverage, or NaN when unmeasured.
    double   last_coverage;         // @ 104
    /// Fixed-order branch states.  The order IS the reduction order.
    PrismTshArmBranchState branch[PRISM_TSH_ARM_N_BRANCHES]; // @ 112 .. 496
    /// Records the ring refused.  Never silently zero.
    uint64_t dropped_records;       // @ 496
    /// Observations refused by the ASC / force exclusivity rule.  Non-zero
    /// means a producer is wired to the wrong branch.
    uint64_t signal_grid_refusals;  // @ 504
};

static_assert(sizeof(PrismTshArmState) == 512,
              "PrismTshArmState MUST be 512 bytes (Rust mirror pins the same).");
static_assert(alignof(PrismTshArmState) == 16,
              "PrismTshArmState MUST be 16-byte aligned.");
static_assert(offsetof(PrismTshArmState, record_sequence) == 16,
              "record_sequence offset drift: must be 16.");
static_assert(offsetof(PrismTshArmState, branch) == 112,
              "branch array offset drift: must be 112.");
static_assert(offsetof(PrismTshArmState, dropped_records) == 496,
              "dropped_records offset drift: must be 496.");

/// The compact emitted record.  80 bytes, one store.
///
/// This is the ENTIRE hot-path emission.  Anything richer (JSON, hashes,
/// provenance strings) is built on the host from this record, never inside the
/// decision path.
///
/// The field set is chosen so the record is SELF-SUFFICIENT for the Coupled
/// Chronology Record's TSH-demand block (`PRISM_CCR_OFF_TSH_*` in
/// `coupled_chronology.cuh`): the per-simulated-picosecond rate, the memory
/// state, both confidences, the active-branch mask, and the requested gear all
/// travel together with the step and the cumulative simulated time that
/// contextualise them.  A post-hoc host-side join cannot recover information a
/// record never carried, which is precisely how the prior campaign's evidence
/// became uninterpretable.
struct __align__(16) PrismTshArmRecord {
    /// PRISM_TSH_ARM_SCHEMA_VERSION.
    uint32_t schema_version;        // @  0
    /// Echo of `PrismTshArmState.record_sequence`.  Gap => decision missing.
    uint32_t record_sequence;       // @  4
    /// Global MD step this decision belongs to.
    uint64_t global_md_step;        // @  8
    /// Cumulative SIMULATED time in ps.  Mandatory once the gear varies: step
    /// index stops being proportional to elapsed physics the moment dt changes.
    double   cumulative_ps;         // @ 16
    /// Integrated salience in [0,1], or NaN when NO branch had evidence.
    /// NaN is the honest value; 0.0 would read as "measured calm".
    float    integrated_salience;   // @ 24
    /// Confidence in THIS REQUEST, in [0,1], or NaN when unmeasured.
    ///
    /// Distinct from `observation_confidence`: a request forced by the
    /// sentinel or by a rechallenge is structurally mandated and therefore
    /// fully confident even when the evidence is thin, while a request that
    /// only exists because there was NO evidence has confidence 0.0 — it is a
    /// fail-safe default, not an inference.
    float    request_confidence;    // @ 28
    /// The gear this arm REQUESTS.  Never PRISM_GEAR_ABORT.  This is a DEMAND,
    /// not an authorization and not a measurement.
    uint32_t requested_gear;        // @ 32
    /// dt in fs implied by `requested_gear` under the resolved policy.
    float    requested_dt_fs;       // @ 36
    /// PRISM_TSH_REASON_*.
    uint32_t request_reason;        // @ 40
    /// Packed 6-bit fields; see PRISM_TSH_ARM_MASK_*_SHIFT.
    uint32_t active_branch_mask;    // @ 44
    /// PRISM_TSH_MEMORY_*.
    uint32_t memory_state;          // @ 48
    /// SIMULATED ps spent in the current memory state.
    float    memory_age_ps;         // @ 52
    /// The authorized gear the arm was told about, or
    /// PRISM_TSH_ARM_UNMEASURED_U32.  Echoed so an offline reader can bind
    /// requested / authorized / device-measured without a host-side join.
    uint32_t authorized_gear_echo;  // @ 56
    /// PRISM_TSH_FLAG_*.
    uint32_t flags;                 // @ 60
    /// ANTI-BLINDNESS (d) gate input: weighted freshness of the evidence, in
    /// [0,1], or NaN when no branch carries weight.  This is the quantity the
    /// coarsening gate compares against `min_confidence_for_coarsening`, so it
    /// must travel with the record or the gate's verdict is unauditable.
    float    observation_confidence; // @ 64
    /// Integrated salience as a RATE, per SIMULATED PICOSECOND — the
    /// canonical quantity for `PRISM_CCR_OFF_TSH_SALIENCE_PER_PS`.  NaN when
    /// unmeasured.  Never per record, per chunk, or per invocation.
    float    integrated_rate_per_ps; // @ 68
    /// Fingerprint of the resolved policy that produced this request.  Binds a
    /// gear demand to the exact thresholds and origins behind it.
    uint64_t policy_digest;          // @ 72
    /// PEAK admissible branch salience — THE RESOLUTION GATE INPUT — or NaN
    /// when unmeasured.  Distinct from `integrated_salience`, which is the
    /// confidence-weighted mean and gates nothing.  A gate input that does not
    /// travel makes the verdict unauditable.
    float    peak_branch_salience;   // @ 80
    /// Weighted observation COVERAGE in [0,1], or NaN.  ANTI-BLINDNESS (c) as
    /// a measured quantity: the fraction of recent simulated time that staged
    /// evidence actually covered.
    float    observation_coverage;   // @ 84
    /// PRISM_TSH_ACTION_* — the codec-registered thermodynamic action.
    /// PRISM_TSH_ACTION_ABSENT (0) is not a registered opcode, so it doubles as
    /// "this slot was never written".
    uint32_t thermodynamic_action;   // @ 88
    /// Index of the branch that produced `peak_branch_salience`, or
    /// PRISM_TSH_ARM_UNMEASURED_U32.
    uint32_t peak_branch_index;      // @ 92
};

static_assert(sizeof(PrismTshArmRecord) == 96,
              "PrismTshArmRecord MUST be 96 bytes (Rust mirror pins the same).");
static_assert(alignof(PrismTshArmRecord) == 16,
              "PrismTshArmRecord MUST be 16-byte aligned.");
static_assert(offsetof(PrismTshArmRecord, cumulative_ps) == 16,
              "cumulative_ps offset drift: must be 16.");
static_assert(offsetof(PrismTshArmRecord, flags) == 60,
              "flags offset drift: must be 60.");
static_assert(offsetof(PrismTshArmRecord, policy_digest) == 72,
              "policy_digest offset drift: must be 72.");
static_assert(offsetof(PrismTshArmRecord, peak_branch_salience) == 80,
              "peak_branch_salience offset drift: must be 80.");
static_assert(offsetof(PrismTshArmRecord, thermodynamic_action) == 88,
              "thermodynamic_action offset drift: must be 88.");

/// The value an UNWRITTEN ring slot must hold.
///
/// A zero-filled slot decodes as a plausible CALM record: `memory_state` 0 is
/// PRISM_TSH_MEMORY_BASELINE, `requested_gear` 0 is PRISM_GEAR_FINE,
/// `integrated_salience` 0.0f reads as measured calm, `request_reason` 0 is
/// PRISM_TSH_REASON_UNSET.  Only `schema_version` distinguishes it, which makes
/// a drained ring indistinguishable from a run of fine-gear decisions — the
/// exact sentinel-collision class this architecture forbids, since gear 0 and
/// memory state 0 are both LEGITIMATE values.  Pre-fill with this instead.
__host__ __device__ __forceinline__ void
prism_tsh_arm_record_unwritten(PrismTshArmRecord* r) {
    if (r == nullptr) {
        return;
    }
    r->schema_version       = 0u;
    r->record_sequence      = 0u;
    r->global_md_step       = ~0ull;
    r->cumulative_ps        = NAN;
    r->integrated_salience  = NAN;
    r->request_confidence   = NAN;
    r->requested_gear       = PRISM_GEAR_UNSET;
    r->requested_dt_fs      = NAN;
    r->request_reason       = PRISM_TSH_ARM_UNMEASURED_U32;
    r->active_branch_mask   = 0u;
    r->memory_state         = PRISM_TSH_ARM_UNMEASURED_U32;
    r->memory_age_ps        = NAN;
    r->authorized_gear_echo = PRISM_TSH_ARM_UNMEASURED_U32;
    r->flags                = 0u;
    r->observation_confidence = NAN;
    r->integrated_rate_per_ps = NAN;
    r->policy_digest        = 0ull;
    r->peak_branch_salience = NAN;
    r->observation_coverage = NAN;
    r->thermodynamic_action = PRISM_TSH_ACTION_ABSENT;
    r->peak_branch_index    = PRISM_TSH_ARM_UNMEASURED_U32;
}

/// Whether a ring slot holds a record an emitter actually wrote.
///
/// The host mirror is `TshArmRecordAbi::is_valid`.  Without this a reader
/// cannot tell a gear-0 decision from an empty slot.
__host__ __device__ __forceinline__ bool
prism_tsh_arm_record_is_valid(const PrismTshArmRecord* r) {
    return r != nullptr &&
           r->schema_version == PRISM_TSH_ARM_SCHEMA_VERSION &&
           r->request_reason != PRISM_TSH_ARM_UNMEASURED_U32 &&
           r->memory_state   != PRISM_TSH_ARM_UNMEASURED_U32;
}

/// Map a PRISM_TSH_REASON_* onto the codec-registered PRISM_TSH_ACTION_*.
///
/// Total by construction: the default arm is HANDOFF, so a reason this build
/// does not know publishes "the arm declines", never a plausible STABLE.
__host__ __device__ __forceinline__ uint32_t
prism_tsh_arm_action_for_reason(uint32_t reason) {
    switch (reason) {
    case PRISM_TSH_REASON_HOLD:
        return PRISM_TSH_ACTION_STABLE;
    case PRISM_TSH_REASON_SALIENCE_DOWNSHIFT:
    case PRISM_TSH_REASON_MEMORY_THRESHOLD_DOWNSHIFT:
    case PRISM_TSH_REASON_MEMORY_HOLD_FINE:
        return PRISM_TSH_ACTION_SALIENCE_RISE;
    case PRISM_TSH_REASON_UPSHIFT_ELIGIBLE:
        return PRISM_TSH_ACTION_SALIENCE_FALL;
    case PRISM_TSH_REASON_CONFIDENCE_GATE:
    case PRISM_TSH_REASON_DWELL_GATE:
        return PRISM_TSH_ACTION_HYSTERESIS;
    case PRISM_TSH_REASON_SENTINEL:
    case PRISM_TSH_REASON_RECHALLENGE:
        return PRISM_TSH_ACTION_BOUNDARY;
    default:
        return PRISM_TSH_ACTION_HANDOFF;
    }
}

/// SINGLE-PRODUCER device ring.  Written by the one thread that decides, so
/// neither index needs an atomic.
///
/// A full ring DROPS and COUNTS the drop.  It never blocks the decision, and
/// it never overwrites a record the consumer has not taken: `head == tail` is
/// empty and `(head + 1) % capacity == tail` is full.  One slot is therefore
/// always unused, which is the price of distinguishing the two states without
/// a separate count — and distinguishing them is what makes `dropped` an exact
/// number rather than an estimate.
///
/// Silently overwriting would be worse than dropping: the record stream's
/// whole purpose is that a gap is unambiguous, and an unaccounted overwrite
/// manufactures a gap that looks like a decision that never ran.
struct __align__(16) PrismTshArmRing {
    PrismTshArmRecord* records;   // @  0
    uint32_t           capacity;  // @  8
    /// Producer index.  Next slot to write.
    uint32_t           head;      // @ 12
    /// Consumer index.  Advanced by the host after it drains.
    uint32_t           tail;      // @ 16
    uint32_t           _pad;      // @ 20
    uint64_t           emitted;   // @ 24
    uint64_t           dropped;   // @ 32
    uint64_t           _reserved; // @ 40
};

static_assert(sizeof(PrismTshArmRing) == 48,
              "PrismTshArmRing MUST be 48 bytes.");
static_assert(offsetof(PrismTshArmRing, dropped) == 32,
              "dropped offset drift: must be 32.");

// ═══════════════════════════════════════════════════════════════════════════
// Implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Finiteness test that does not depend on whether `<cmath>` left `isfinite`
/// as a macro, in `std::`, or in the global namespace — the three are not
/// portable across host toolchains and device compilation.  NaN fails the
/// self-comparison; the infinities fail the bounds.
__host__ __device__ __forceinline__ bool prism_tsh_is_finite(double x) {
    return (x == x) && (x < INFINITY) && (x > -INFINITY);
}

/// Fill `out` with the documented defaults, UNARMED.
///
/// This is the single point of truth the Rust resolver starts from; it does
/// not decide anything and an unarmed policy makes the arm silent, not
/// opinionated.
__host__ __device__ __forceinline__ void
prism_tsh_arm_default_policy(PrismTshArmPolicy* out) {
    if (out == nullptr) {
        return;
    }
    out->schema_version = PRISM_TSH_ARM_SCHEMA_VERSION;
    out->armed          = 0u;
    out->signal_grid_warp_independent = PRISM_TSH_ARM_DEFAULT_SIGNAL_GRID_INDEPENDENT;
    out->_pad0          = 0u;
    for (uint32_t b = 0u; b < PRISM_TSH_ARM_N_BRANCHES; ++b) {
        out->branch_weight[b]         =
            (b == PRISM_TSH_ARM_BRANCH_SIGNAL_GRID_WARP)
                ? PRISM_TSH_ARM_DEFAULT_SIGNAL_GRID_WARP_WEIGHT
                : PRISM_TSH_ARM_DEFAULT_BRANCH_WEIGHT;
        out->branch_reference_rate[b] = PRISM_TSH_ARM_DEFAULT_BRANCH_REFERENCE_RATE;
    }
    out->branch_tau_ps                = PRISM_TSH_ARM_DEFAULT_BRANCH_TAU_PS;
    out->min_observation_cadence_ps   = PRISM_TSH_ARM_DEFAULT_MIN_OBSERVATION_CADENCE_PS;
    out->confidence_decay_ps          = PRISM_TSH_ARM_DEFAULT_CONFIDENCE_DECAY_PS;
    out->min_confidence_for_coarsening= PRISM_TSH_ARM_DEFAULT_MIN_CONFIDENCE_FOR_COARSENING;
    out->max_sentinel_interval_ps     = PRISM_TSH_ARM_DEFAULT_MAX_SENTINEL_INTERVAL_PS;
    out->sentinel_hold_ps             = PRISM_TSH_ARM_DEFAULT_SENTINEL_HOLD_PS;
    out->salience_downshift_threshold = PRISM_TSH_ARM_DEFAULT_SALIENCE_DOWNSHIFT;
    out->salience_upshift_threshold   = PRISM_TSH_ARM_DEFAULT_SALIENCE_UPSHIFT;
    out->salience_reinforce_threshold = PRISM_TSH_ARM_DEFAULT_SALIENCE_REINFORCE;
    out->min_gear_dwell_ps            = PRISM_TSH_ARM_DEFAULT_MIN_GEAR_DWELL_PS;
    out->decay_period_ps              = PRISM_TSH_ARM_DEFAULT_DECAY_PERIOD_PS;
    out->withdrawal_hold_ps           = PRISM_TSH_ARM_DEFAULT_WITHDRAWAL_HOLD_PS;
    out->rechallenge_window_ps        = PRISM_TSH_ARM_DEFAULT_RECHALLENGE_WINDOW_PS;
    out->reinforce_dwell_ps           = PRISM_TSH_ARM_DEFAULT_REINFORCE_DWELL_PS;
    out->gear_dt_fs[0]                = PRISM_TSH_ARM_DEFAULT_DT_FS_G0;
    out->gear_dt_fs[1]                = PRISM_TSH_ARM_DEFAULT_DT_FS_G1;
    out->gear_dt_fs[2]                = PRISM_TSH_ARM_DEFAULT_DT_FS_G2;
    out->gear_dt_fs[3]                = PRISM_TSH_ARM_DEFAULT_DT_FS_G3;
    out->policy_digest                = 0ull;
    for (uint32_t i = 0u; i < 3u; ++i) {
        out->_reserved[i] = 0ull;
    }
}

/// Reset the arm to its pre-decision baseline at simulated time `t0_ps`.
///
/// Every branch starts UNBOUND (`ever_observed = 0`), not calm.  The sentinel
/// clock starts at `t0_ps`, so the first sentinel is due one full interval in
/// — not immediately, and not never.
__host__ __device__ __forceinline__ void
prism_tsh_arm_reset(PrismTshArmState* s, double t0_ps) {
    if (s == nullptr) {
        return;
    }
    s->schema_version         = PRISM_TSH_ARM_SCHEMA_VERSION;
    s->record_sequence        = 0u;
    s->memory_state           = PRISM_TSH_MEMORY_BASELINE;
    s->last_requested_gear    = PRISM_GEAR_UNSET;
    s->now_ps                 = t0_ps;
    s->memory_entered_ps      = t0_ps;
    s->last_sentinel_ps       = t0_ps;
    s->request_changed_ps     = t0_ps;
    s->active_memory_exit_ps  = -INFINITY;
    s->reinforce_since_ps     = INFINITY;
    s->sentinel_hold_until_ps = -INFINITY;
    s->last_salience          = NAN;
    s->last_confidence        = NAN;
    s->last_coverage          = NAN;
    s->rechallenge_armed      = 0u;
    s->decisions              = 0ull;
    for (uint32_t b = 0u; b < PRISM_TSH_ARM_N_BRANCHES; ++b) {
        s->branch[b].activity          = 0.0;
        s->branch[b].last_evidence_ps  = -INFINITY;
        s->branch[b].pending_activity  = 0.0;
        s->branch[b].pending_span_ps   = 0.0;
        s->branch[b].coverage_activity = 0.0;
        s->branch[b].ever_observed     = 0u;
        s->branch[b].pending_samples   = 0u;
        s->branch[b].total_samples     = 0ull;
        s->branch[b]._reserved         = 0ull;
    }
    s->dropped_records      = 0ull;
    s->signal_grid_refusals = 0ull;
}

/// Stage one branch observation.
///
/// `accumulated_activity` is the branch's own non-negative activity measure
/// summed over `span_ps` picoseconds of SIMULATED time ending at
/// `window_end_ps`.  `sample_count` is recorded and NEVER divided by.
///
/// A non-positive or non-finite span is REFUSED (the branch keeps its previous
/// state and its staleness keeps growing).  Refusing is the honest behaviour:
/// a zero-span observation carries no physical-time normalisation, and
/// inventing one would be a fabricated measurement.
///
/// A signal-grid observation is REFUSED, and the refusal COUNTED, unless the
/// policy declares an independent signal-grid observable — see the ASC/force
/// block at the top of this header.  Refusing is the only honest option: the
/// alternative is to accept it and silently give the force representation two
/// sixths of the aggregate weight.
///
/// Returns 1 if the observation was staged, 0 if it was refused.
__host__ __device__ __forceinline__ uint32_t
prism_tsh_arm_observe(const PrismTshArmPolicy* p,
                      PrismTshArmState* s,
                      uint32_t          branch,
                      double            accumulated_activity,
                      double            span_ps,
                      double            window_end_ps,
                      uint32_t          sample_count) {
    if (s == nullptr || branch >= PRISM_TSH_ARM_N_BRANCHES) {
        return 0u;
    }
    if (!(span_ps > 0.0) || !prism_tsh_is_finite(span_ps)) {
        return 0u;
    }
    if (!prism_tsh_is_finite(accumulated_activity) || accumulated_activity < 0.0) {
        return 0u;
    }
    if (branch == PRISM_TSH_ARM_BRANCH_SIGNAL_GRID_WARP &&
        (p == nullptr || p->signal_grid_warp_independent == 0u)) {
        s->signal_grid_refusals += 1ull;
        return 0u;
    }
    PrismTshArmBranchState* b = &s->branch[branch];
    b->pending_activity += accumulated_activity;
    b->pending_span_ps  += span_ps;
    b->pending_samples  += sample_count;
    b->total_samples    += (uint64_t)sample_count;
    b->ever_observed     = 1u;
    if (!(window_end_ps <= b->last_evidence_ps)) {
        b->last_evidence_ps = window_end_ps;
    }
    return 1u;
}

/// Per-branch observation COVERAGE in [0,1]: the fraction of recent simulated
/// time this branch's staged evidence actually covered.
///
/// Exactly the duty cycle in steady state — a branch staging 0.2 ps of evidence
/// per 1.0 ps of simulated time converges to 0.2.  0.0 for a branch that never
/// reported.
__host__ __device__ __forceinline__ double
prism_tsh_arm_branch_coverage(const PrismTshArmPolicy*      p,
                              const PrismTshArmBranchState* b) {
    if (b->ever_observed == 0u) {
        return 0.0;
    }
    const double tau = p->branch_tau_ps;
    if (!(tau > 0.0)) {
        return 0.0;
    }
    double c = b->coverage_activity / tau;
    if (!prism_tsh_is_finite(c) || c < 0.0) {
        return 0.0;
    }
    return (c > 1.0) ? 1.0 : c;
}

/// Per-branch confidence in [0,1]: RECENCY * COVERAGE.
///
/// Both factors are mandatory and they answer different questions.  RECENCY
/// asks when the branch last said anything; COVERAGE asks how much of the
/// simulated time it was supposed to be watching it actually watched.  The
/// first revision had only recency, so a branch reporting punctually but
/// thinly returned 1.000 and gate (d) waved through exactly the sparse regimes
/// it exists to refuse.
///
/// A branch that has NEVER reported returns 0.0 — it contributes nothing to
/// the integration rather than contributing a zero salience.  That distinction
/// is anti-blindness (c) and is the difference between "calm" and "blind".
///
/// Past `max_sentinel_interval_ps` of staleness the result is hard ZERO: with
/// the activity integrator frozen, an arbitrarily old value would otherwise
/// keep voting forever.
__host__ __device__ __forceinline__ double
prism_tsh_arm_branch_confidence(const PrismTshArmPolicy*      p,
                                const PrismTshArmBranchState* b,
                                double                        now_ps) {
    if (b->ever_observed == 0u) {
        return 0.0;
    }
    const double coverage = prism_tsh_arm_branch_coverage(p, b);
    if (!(coverage > 0.0)) {
        return 0.0;
    }
    const double cadence = p->min_observation_cadence_ps;
    const double decay   = p->confidence_decay_ps;
    const double horizon = p->max_sentinel_interval_ps;
    double stale = now_ps - b->last_evidence_ps;
    if (!(stale > 0.0)) {
        return coverage;
    }
    if (prism_tsh_is_finite(horizon) && horizon > 0.0 && stale > horizon) {
        return 0.0;
    }
    if (!(cadence > 0.0) || !(decay > 0.0)) {
        // A policy that cannot express cadence cannot express freshness.
        // Report NO confidence rather than full confidence: the failure
        // direction must be "blocked from coarsening", never "free to coarsen".
        return 0.0;
    }
    const double recency = (stale <= cadence) ? 1.0 : exp(-(stale - cadence) / decay);
    return recency * coverage;
}

/// Consume `covered` picoseconds of a staged window, CARRYING FORWARD the
/// remainder at the same rate, up to `backlog_limit_ps`.
///
/// Discarding `pending_span_ps - covered` — the first revision — threw away
/// real evidence every time the observation cadence was coarser than the
/// decision cadence, and the discarded evidence read as quiet.
///
/// The carry is BOUNDED because an unbounded one is its own hazard: a producer
/// staging spans far longer than the decision interval would grow a backlog
/// without limit and the arm would integrate the distant past as the present,
/// falling arbitrarily far behind the run with nothing saying so.  The bound is
/// the staleness horizon — evidence older than the arm may CONSIDER is not
/// evidence it should be queueing.  Truncation drops the OLDEST surplus (span
/// shrinks, rate is preserved) and raises PRISM_TSH_FLAG_BACKLOG_TRUNCATED, so
/// the loss is bounded, named, and never silent.
///
/// Returns 1 when the backlog was truncated.
__host__ __device__ __forceinline__ uint32_t
prism_tsh_arm_consume_pending(PrismTshArmBranchState* b,
                              double                  covered,
                              double                  backlog_limit_ps) {
    if (!(b->pending_span_ps > 0.0)) {
        b->pending_activity = 0.0;
        b->pending_span_ps  = 0.0;
        b->pending_samples  = 0u;
        return 0u;
    }
    const double rate = b->pending_activity / b->pending_span_ps;
    const double remaining = b->pending_span_ps - covered;
    if (remaining > 0.0 && prism_tsh_is_finite(remaining)) {
        double keep = remaining;
        uint32_t truncated = 0u;
        if (prism_tsh_is_finite(backlog_limit_ps) && backlog_limit_ps > 0.0 &&
            keep > backlog_limit_ps) {
            keep = backlog_limit_ps;
            truncated = 1u;
        }
        b->pending_span_ps  = keep;
        b->pending_activity = rate * keep;
        // `pending_samples` is provenance for the staged window and belongs to
        // the carried remainder.  It is never a divisor, so retaining it cannot
        // bias a rate.
        return truncated;
    }
    b->pending_activity = 0.0;
    b->pending_span_ps  = 0.0;
    b->pending_samples  = 0u;
    return 0u;
}

/// Advance every branch to `now_ps`, folding in the staged window.
///
/// EXACT solution of `dA/dt = -A/tau + r` with `r = pending_activity /
/// pending_span_ps` held over the span the evidence COVERS, and the activity
/// integrator FROZEN over the uncovered remainder.  Surplus staged span is
/// carried forward, never discarded.  See mechanisms 2 and 3 at the top: a
/// pure decay over uncovered time integrates absence of evidence as measured
/// calm, which is the positive-feedback loop this module claims immunity to.
///
/// COVERAGE, by contrast, IS decayed over the uncovered remainder — measuring
/// the absence is exactly its job.
///
/// Fixed-order serial loop, one thread, no atomics.
///
/// Returns 1 when any branch's carried backlog had to be truncated.
__host__ __device__ __forceinline__ uint32_t
prism_tsh_arm_advance(const PrismTshArmPolicy* p,
                      PrismTshArmState*        s,
                      double                   now_ps) {
    const double tau = p->branch_tau_ps;
    const double backlog_limit = p->max_sentinel_interval_ps;
    uint32_t truncated = 0u;
    double dt = now_ps - s->now_ps;
    if (!(dt > 0.0) || !prism_tsh_is_finite(dt)) {
        dt = 0.0;
    }
    for (uint32_t i = 0u; i < PRISM_TSH_ARM_N_BRANCHES; ++i) {
        PrismTshArmBranchState* b = &s->branch[i];
        if (!(tau > 0.0)) {
            // Unusable policy: refuse to evolve rather than divide by zero.
            (void)prism_tsh_arm_consume_pending(b, INFINITY, backlog_limit);
            continue;
        }
        double covered = b->pending_span_ps;
        if (covered > dt) {
            covered = dt;
        }
        if (covered > 0.0 && b->pending_span_ps > 0.0) {
            const double r  = b->pending_activity / b->pending_span_ps;
            const double e  = exp(-covered / tau);
            b->activity          = b->activity * e + r * tau * (1.0 - e);
            b->coverage_activity = b->coverage_activity * e + tau * (1.0 - e);
        }
        const double rest = dt - covered;
        if (rest > 0.0) {
            // ACTIVITY IS FROZEN here.  COVERAGE decays.
            b->coverage_activity *= exp(-rest / tau);
        }
        truncated |= prism_tsh_arm_consume_pending(b, covered, backlog_limit);
    }
    return truncated;
}

/// Branch salience in [0,1): `s = r / (r + ref)` with `r = activity / tau`.
///
/// NaN when the branch has NEVER reported — an unbound branch has activity 0,
/// and returning 0.0 for it would make "no sensor" arithmetically identical to
/// "sensor reporting perfect calm".  NaN also when the policy reference is
/// unusable.  Never a plausible zero.
__host__ __device__ __forceinline__ double
prism_tsh_arm_branch_salience(const PrismTshArmPolicy*      p,
                              const PrismTshArmBranchState* b,
                              uint32_t                      index) {
    if (b->ever_observed == 0u) {
        return NAN;
    }
    const double tau = p->branch_tau_ps;
    const double ref = p->branch_reference_rate[index];
    if (!(tau > 0.0) || !(ref > 0.0)) {
        return NAN;
    }
    double r = b->activity / tau;
    if (!prism_tsh_is_finite(r) || r < 0.0) {
        return NAN;
    }
    return r / (r + ref);
}

/// Push one record into the single-producer ring.
///
/// Returns 1 when the record was stored, 0 when the ring was full and the
/// record was DROPPED.  A drop increments both `ring->dropped` and, when
/// `state` is non-null, `state->dropped_records`, so the next decision can
/// raise PRISM_TSH_FLAG_RING_DROPPED and the loss is never silent.
__host__ __device__ __forceinline__ uint32_t
prism_tsh_arm_ring_push(PrismTshArmRing*         ring,
                        PrismTshArmState*        state,
                        const PrismTshArmRecord* rec) {
    if (ring == nullptr || ring->records == nullptr || ring->capacity == 0u ||
        rec == nullptr) {
        return 0u;
    }
    const uint32_t next = (ring->head + 1u) % ring->capacity;
    if (next == ring->tail) {
        // Full.  Drop the NEW record rather than the oldest unread one: the
        // consumer's view stays contiguous and the loss is a counted tail, not
        // a hole in the middle of the sequence.
        ring->dropped += 1ull;
        if (state != nullptr) {
            state->dropped_records += 1ull;
        }
        return 0u;
    }
    ring->records[ring->head] = *rec;
    ring->head = next;
    ring->emitted += 1ull;
    return 1u;
}

/// THE ARM.
///
/// Pure function of (policy, state, clock, authorized-gear echo): no I/O, no
/// randomness, no global state beyond `s`.  Emits ALWAYS — a no-change
/// decision still produces a record, because a gap in the stream must never be
/// ambiguous between "gear held" and "decision did not run".
///
/// `authorized_gear_valid == 0` means the host has NO measurement of what the
/// governor authorized.  In that case the sentinel clock is NOT reset (an
/// unmeasured gear is not evidence that a fine sample was taken) and the
/// AUTHORIZED_UNMEASURED flag is raised.
///
/// `out_req` receives the request the governor will read.  `out_rec` receives
/// the compact record for the ring.  Either may be null.
__host__ __device__ __forceinline__ void
prism_tsh_arm_decide(const PrismTshArmPolicy* p,
                     PrismTshArmState*        s,
                     double                   now_ps,
                     uint64_t                 global_md_step,
                     uint32_t                 frame,
                     uint32_t                 authorized_gear,
                     uint32_t                 authorized_gear_valid,
                     PrismTshSalienceRequest* out_req,
                     PrismTshArmRecord*       out_rec) {
    if (p == nullptr || s == nullptr) {
        return;
    }

    uint32_t flags = 0u;
    if (s->dropped_records != 0ull) {
        // A previous record never reached the ring.  Say so on every record
        // until the host clears the counter, so a reader cannot mistake the
        // truncated stream for a complete one.
        flags |= PRISM_TSH_FLAG_RING_DROPPED;
    }

    // ── Sentinel clock: only a MEASURED fine gear counts as a sample. ──────
    if (authorized_gear_valid != 0u && authorized_gear == PRISM_GEAR_FINE) {
        s->last_sentinel_ps = now_ps;
    } else if (authorized_gear_valid == 0u) {
        flags |= PRISM_TSH_FLAG_AUTHORIZED_UNMEASURED;
    }

    // ── Evolve the branches over the simulated interval. ───────────────────
    uint32_t fresh_mask = 0u;
    for (uint32_t i = 0u; i < PRISM_TSH_ARM_N_BRANCHES; ++i) {
        if (s->branch[i].pending_span_ps > 0.0) {
            fresh_mask |= (1u << i);
        }
    }
    if (prism_tsh_arm_advance(p, s, now_ps) != 0u) {
        flags |= PRISM_TSH_FLAG_BACKLOG_TRUNCATED;
    }
    s->now_ps = now_ps;
    s->decisions += 1ull;
    s->record_sequence += 1ull;

    // ── Fixed-order integration.  No atomics; index order IS reduce order. ─
    //
    // Two aggregates, two jobs.  `num/den` is the confidence-weighted MEAN and
    // is REPORTED ONLY.  `peak` is the maximum over admissible branches and is
    // what every gate below is taken on: a demand is not a vote, and no number
    // of calm branches may dilute one branch's excursion.
    double num = 0.0;
    double den = 0.0;
    double rate_num = 0.0;
    double weight_total = 0.0;
    double conf_weighted = 0.0;
    double cov_weighted = 0.0;
    double peak = NAN;
    uint32_t peak_index = PRISM_TSH_ARM_UNMEASURED_U32;
    uint32_t salient_mask = 0u;
    uint32_t stale_mask = 0u;
    uint32_t unbound_mask = 0u;
    const double horizon = p->max_sentinel_interval_ps;
    for (uint32_t i = 0u; i < PRISM_TSH_ARM_N_BRANCHES; ++i) {
        const double w = p->branch_weight[i];
        const double c = prism_tsh_arm_branch_confidence(p, &s->branch[i], now_ps);
        const double sb = prism_tsh_arm_branch_salience(p, &s->branch[i], i);
        if (w > 0.0) {
            weight_total += w;
            conf_weighted += w * c;
            cov_weighted += w * prism_tsh_arm_branch_coverage(p, &s->branch[i]);
        }
        if (s->branch[i].ever_observed == 0u) {
            unbound_mask |= (1u << i);
        } else {
            if (c < 1.0) {
                stale_mask |= (1u << i);
            }
            if (prism_tsh_is_finite(horizon) && horizon > 0.0 &&
                (now_ps - s->branch[i].last_evidence_ps) > horizon) {
                flags |= PRISM_TSH_FLAG_STALE_HORIZON;
            }
        }
        if (!prism_tsh_is_finite(sb)) {
            // A never-reported branch is UNBOUND, not misconfigured; only a
            // policy that cannot express a salience raises INVALID_REFERENCE.
            if (s->branch[i].ever_observed != 0u) {
                flags |= PRISM_TSH_FLAG_INVALID_REFERENCE;
            }
            continue;
        }
        if (w > 0.0 && c > 0.0) {
            num += w * c * sb;
            den += w * c;
            // The RATE, per simulated picosecond.  Carried alongside the
            // bounded salience because the chronology record's canonical TSH
            // quantity is the rate, not the saturating response.
            rate_num += w * c * (s->branch[i].activity / p->branch_tau_ps);
            // Admissible: carries weight, has reported, still credible.  Ties
            // keep the LOWEST index so the peak is a function of the state and
            // not of comparison order.
            if (!prism_tsh_is_finite(peak) || sb > peak) {
                peak = sb;
                peak_index = i;
            }
        }
        if (sb >= p->salience_downshift_threshold) {
            salient_mask |= (1u << i);
        }
    }

    // The REPORTED statistic.
    const double mean_salience = (den > 0.0) ? (num / den) : NAN;
    double rate_per_ps = (den > 0.0) ? (rate_num / den) : NAN;
    double confidence = (weight_total > 0.0) ? (conf_weighted / weight_total) : NAN;
    double coverage   = (weight_total > 0.0) ? (cov_weighted / weight_total) : NAN;
    if (!(den > 0.0)) {
        flags |= PRISM_TSH_FLAG_EVIDENCE_ABSENT;
    }
    // Raised whenever true, not only inside the coarsening branch, so a reader
    // can see the arm was under-observed even while demanding FINER.
    if (prism_tsh_is_finite(coverage) && coverage < p->min_confidence_for_coarsening) {
        flags |= PRISM_TSH_FLAG_OBSERVATION_SPARSE;
    }
    if (s->signal_grid_refusals != 0ull) {
        flags |= PRISM_TSH_FLAG_SIGNAL_GRID_UNBOUND;
    }
    // THE GATE INPUT.  Every `salience` below is the peak.
    const double salience = peak;
    s->last_salience   = salience;
    s->last_confidence = confidence;
    s->last_coverage   = coverage;

    // ── Memory state machine (operator's table). ──────────────────────────
    //
    // Driven by the PEAK for the same reason the gears are: the memory is
    // supposed to remember an excursion, and a mean lets calm branches erase
    // one branch's excursion before the memory ever sees it.
    const double down      = p->salience_downshift_threshold;
    const double up        = p->salience_upshift_threshold;
    const double reinforce = p->salience_reinforce_threshold;
    const double memory_age = now_ps - s->memory_entered_ps;

    const uint32_t prev_state = s->memory_state;
    uint32_t next_state = prev_state;
    const uint32_t have_salience = prism_tsh_is_finite(salience) ? 1u : 0u;

    // Track how long salience has been at/above the reinforce bound.
    if (have_salience && salience >= reinforce) {
        if (!prism_tsh_is_finite(s->reinforce_since_ps) || s->reinforce_since_ps > now_ps) {
            s->reinforce_since_ps = now_ps;
        }
    } else {
        s->reinforce_since_ps = INFINITY;
    }

    // Rechallenge takes precedence over every other transition: a renewed
    // excursion inside the rechallenge window is the operator's "immediate
    // downshift" case and must not be filtered by dwell or hysteresis.
    const double since_active_exit = now_ps - s->active_memory_exit_ps;
    const uint32_t in_rechallenge_window =
        (s->rechallenge_armed != 0u &&
         prism_tsh_is_finite(since_active_exit) &&
         since_active_exit <= p->rechallenge_window_ps) ? 1u : 0u;

    if (have_salience && salience >= down && in_rechallenge_window != 0u &&
        (prev_state == PRISM_TSH_MEMORY_DECAY ||
         prev_state == PRISM_TSH_MEMORY_INPUT_WITHDRAWAL ||
         prev_state == PRISM_TSH_MEMORY_LOW ||
         prev_state == PRISM_TSH_MEMORY_BASELINE)) {
        next_state = PRISM_TSH_MEMORY_RECHALLENGE;
        flags |= PRISM_TSH_FLAG_RECHALLENGE;
    } else {
        switch (prev_state) {
        case PRISM_TSH_MEMORY_BASELINE:
        case PRISM_TSH_MEMORY_LOW:
            if (have_salience && salience >= down) {
                next_state = PRISM_TSH_MEMORY_THRESHOLD_APPROACH;
            } else if (have_salience && salience >= up) {
                next_state = PRISM_TSH_MEMORY_ACCUMULATING;
            } else if (have_salience) {
                next_state = PRISM_TSH_MEMORY_LOW;
            }
            break;
        case PRISM_TSH_MEMORY_ACCUMULATING:
            if (have_salience && salience >= down) {
                next_state = PRISM_TSH_MEMORY_THRESHOLD_APPROACH;
            } else if (have_salience && salience < up &&
                       memory_age >= p->decay_period_ps) {
                next_state = PRISM_TSH_MEMORY_DECAY;
            }
            break;
        case PRISM_TSH_MEMORY_THRESHOLD_APPROACH:
            if (have_salience && salience >= reinforce) {
                next_state = PRISM_TSH_MEMORY_ACTIVE;
            } else if (have_salience && salience < up) {
                next_state = PRISM_TSH_MEMORY_INPUT_WITHDRAWAL;
            }
            break;
        case PRISM_TSH_MEMORY_ACTIVE:
            if (have_salience && salience >= reinforce &&
                prism_tsh_is_finite(s->reinforce_since_ps) &&
                (now_ps - s->reinforce_since_ps) >= p->reinforce_dwell_ps) {
                next_state = PRISM_TSH_MEMORY_REINFORCED;
            } else if (have_salience && salience < up) {
                next_state = PRISM_TSH_MEMORY_INPUT_WITHDRAWAL;
            }
            break;
        case PRISM_TSH_MEMORY_REINFORCED:
            if (have_salience && salience < up) {
                next_state = PRISM_TSH_MEMORY_INPUT_WITHDRAWAL;
            }
            break;
        case PRISM_TSH_MEMORY_INPUT_WITHDRAWAL:
            if (have_salience && salience >= down) {
                next_state = PRISM_TSH_MEMORY_THRESHOLD_APPROACH;
            } else if (memory_age >= p->withdrawal_hold_ps) {
                next_state = PRISM_TSH_MEMORY_DECAY;
            }
            break;
        case PRISM_TSH_MEMORY_DECAY:
            if (have_salience && salience >= down) {
                next_state = PRISM_TSH_MEMORY_THRESHOLD_APPROACH;
            } else if (memory_age >= p->decay_period_ps) {
                next_state = PRISM_TSH_MEMORY_BASELINE;
            }
            break;
        case PRISM_TSH_MEMORY_RECHALLENGE:
            next_state = PRISM_TSH_MEMORY_ACTIVE;
            break;
        default:
            next_state = PRISM_TSH_MEMORY_BASELINE;
            break;
        }
    }

    if (next_state != prev_state) {
        // Leaving active memory opens the rechallenge window.
        if (prev_state == PRISM_TSH_MEMORY_ACTIVE ||
            prev_state == PRISM_TSH_MEMORY_REINFORCED) {
            s->active_memory_exit_ps = now_ps;
            s->rechallenge_armed = 1u;
        }
        s->memory_state      = next_state;
        s->memory_entered_ps = now_ps;
    }
    // The rechallenge window expires by ELAPSED SIMULATED TIME, not by the
    // memory happening to pass through baseline.  Clearing it on the baseline
    // transition would close the window after one decay period regardless of
    // `rechallenge_window_ps`, making the knob inert and losing the very
    // re-excursion it exists to catch.
    if (s->rechallenge_armed != 0u &&
        s->memory_state != PRISM_TSH_MEMORY_RECHALLENGE &&
        prism_tsh_is_finite(s->active_memory_exit_ps) &&
        (now_ps - s->active_memory_exit_ps) > p->rechallenge_window_ps) {
        s->rechallenge_armed = 0u;
    }

    // ── Gear demand. ──────────────────────────────────────────────────────
    //
    // Starting point is the gear the arm last REQUESTED, or the authorized
    // gear when the arm has no prior request.  The arm tracks reality, not its
    // own history: if the governor refused, the arm asks again from where the
    // run actually is.
    uint32_t base_gear = s->last_requested_gear;
    if (base_gear > PRISM_GEAR_MAX_ACTIVE) {
        base_gear = (authorized_gear_valid != 0u && authorized_gear <= PRISM_GEAR_MAX_ACTIVE)
                        ? authorized_gear
                        : PRISM_GEAR_SAFE;
    }

    // The demand is combined by MINIMUM (finest wins), never by a weighted
    // score.  A score would admit one demand large enough to outvote another,
    // and the only safe aggregation direction for a DEMAND is "the finest
    // resolution anything asked for".  The lexicographic ordering that must
    // never be a score lives in the GOVERNOR; here the min IS the conservative
    // choice.  `reason` reports the MOST AUTHORITATIVE demand that produced
    // the final gear, in the fixed order below.
    uint32_t requested = base_gear;
    uint32_t reason    = PRISM_TSH_REASON_HOLD;

    const double since_sentinel = now_ps - s->last_sentinel_ps;
    const uint32_t sentinel_due =
        (prism_tsh_is_finite(since_sentinel) &&
         since_sentinel >= p->max_sentinel_interval_ps) ? 1u : 0u;
    const uint32_t sentinel_holding =
        (prism_tsh_is_finite(s->sentinel_hold_until_ps) && now_ps < s->sentinel_hold_until_ps) ? 1u : 0u;

    const uint32_t one_finer =
        (base_gear > PRISM_GEAR_FINE) ? (base_gear - 1u) : PRISM_GEAR_FINE;

    if (p->armed == 0u) {
        // Unarmed: no opinion at all.  The governor stays in sole control.
        // Nothing below runs; an unresolved policy must not steer anything.
        flags    |= PRISM_TSH_FLAG_POLICY_UNARMED;
        reason    = PRISM_TSH_REASON_POLICY_UNARMED;
        requested = PRISM_GEAR_UNSET;
    } else {
        // Collect every FINE-ward demand, least authoritative first, so a more
        // authoritative one overwrites `reason` only when it also binds.

        // (6) salience excursion -> one gear finer.
        if (have_salience != 0u && salience >= down && one_finer <= requested) {
            requested = one_finer;
            reason    = PRISM_TSH_REASON_SALIENCE_DOWNSHIFT;
        }
        // (5) memory: threshold approach -> downshift one gear.
        if (s->memory_state == PRISM_TSH_MEMORY_THRESHOLD_APPROACH &&
            one_finer <= requested) {
            requested = one_finer;
            reason    = PRISM_TSH_REASON_MEMORY_THRESHOLD_DOWNSHIFT;
        }
        // (4) memory: active / reinforced / input withdrawal -> maintain fine.
        // (PRISM_GEAR_FINE is 0, so it is unconditionally <= `requested`; the
        // min is written out only where the comparison is not vacuous.)
        if (s->memory_state == PRISM_TSH_MEMORY_ACTIVE ||
            s->memory_state == PRISM_TSH_MEMORY_REINFORCED ||
            s->memory_state == PRISM_TSH_MEMORY_INPUT_WITHDRAWAL) {
            requested = PRISM_GEAR_FINE;
            reason    = PRISM_TSH_REASON_MEMORY_HOLD_FINE;
        }
        // (3) ANTI-BLINDNESS (c): unmeasured is not calm.  Bias fine.
        if (have_salience == 0u) {
            requested = PRISM_GEAR_FINE;
            reason    = PRISM_TSH_REASON_EVIDENCE_ABSENT;
        }
        // (2) ANTI-BLINDNESS (b): unconditional; salience plays no part.
        if (sentinel_due != 0u || sentinel_holding != 0u) {
            requested = PRISM_GEAR_FINE;
            reason    = PRISM_TSH_REASON_SENTINEL;
            flags    |= PRISM_TSH_FLAG_SENTINEL_FORCED;
            if (sentinel_due != 0u && sentinel_holding == 0u) {
                s->sentinel_hold_until_ps = now_ps + p->sentinel_hold_ps;
            }
        }
        // (1) rechallenge -> immediate finest, ahead of everything.
        if (s->memory_state == PRISM_TSH_MEMORY_RECHALLENGE) {
            requested = PRISM_GEAR_FINE;
            reason    = PRISM_TSH_REASON_RECHALLENGE;
        }

        // COARSENING is considered ONLY when nothing above asked for finer.
        if (requested == base_gear &&
            reason == PRISM_TSH_REASON_HOLD &&
            have_salience != 0u &&
            salience < up &&
            (s->memory_state == PRISM_TSH_MEMORY_BASELINE ||
             s->memory_state == PRISM_TSH_MEMORY_LOW ||
             s->memory_state == PRISM_TSH_MEMORY_DECAY)) {
            // Every gate must pass; any failure HOLDS.  Coarser requires low
            // salience AND verified observability, never salience alone.
            const uint32_t confident =
                (prism_tsh_is_finite(confidence) &&
                 confidence >= p->min_confidence_for_coarsening) ? 1u : 0u;
            const double dwell = now_ps - s->request_changed_ps;
            const uint32_t dwelled =
                (prism_tsh_is_finite(dwell) && dwell >= p->min_gear_dwell_ps) ? 1u : 0u;
            // Age of the state the arm is in NOW, after this decision's
            // transition — not the age of the state it just left.
            const double memory_age_post = now_ps - s->memory_entered_ps;
            const uint32_t decayed_long_enough =
                (s->memory_state != PRISM_TSH_MEMORY_DECAY ||
                 memory_age_post >= p->decay_period_ps) ? 1u : 0u;

            if (confident == 0u) {
                // ANTI-BLINDNESS (d).
                flags |= PRISM_TSH_FLAG_CONFIDENCE_BLOCKED;
                reason = PRISM_TSH_REASON_CONFIDENCE_GATE;
            } else if (dwelled == 0u || decayed_long_enough == 0u) {
                flags |= PRISM_TSH_FLAG_DWELL_BLOCKED;
                reason = PRISM_TSH_REASON_DWELL_GATE;
            } else {
                requested = (base_gear < PRISM_GEAR_MAX_ACTIVE) ? (base_gear + 1u)
                                                                : PRISM_GEAR_MAX_ACTIVE;
                reason    = PRISM_TSH_REASON_UPSHIFT_ELIGIBLE;
            }
        }
    }

    // ── AUTHORITY BOUNDARY, enforced at the exit. ─────────────────────────
    //
    // The arm REQUESTS ONLY.  Nothing above authorizes a gear, writes a
    // timestep, or touches the gearbox activation switch — it computes a
    // number and hands it to the governor, which applies
    // min(demanded, permitted).  This clamp is the last statement about the
    // gear before it leaves, so the abort gear can never be named even by an
    // arithmetic accident: aborting is a physical decision the governor owns.
    if (requested != PRISM_GEAR_UNSET && requested > PRISM_GEAR_MAX_ACTIVE) {
        requested = PRISM_GEAR_MAX_ACTIVE;
    }

    if (requested != s->last_requested_gear) {
        s->request_changed_ps = now_ps;
    }
    s->last_requested_gear = requested;

    // ── Publish the request the governor reads. ───────────────────────────
    //
    // Two states, not three.  While `armed`, `requested` derives from
    // `base_gear`, which is always clamped to <= PRISM_GEAR_MAX_ACTIVE, so it
    // can never be PRISM_GEAR_UNSET — the old PRISM_TSH_SALIENCE_HELD branch
    // was unreachable, and the host mirror had no HELD concept at all.  The two
    // sides therefore disagreed about which `state` a bound-but-silent arm
    // publishes, which matters because the governor reads that field to decide
    // whether a policy layer is bound.  An unarmed arm publishes UNSET; an
    // armed one publishes VALID.
    //
    // The published salience is the PEAK — the quantity this arm's own gate was
    // taken on.  Publishing the mean would show the governor a number that
    // never decided anything.
    if (out_req != nullptr) {
        out_req->schema_version = PRISM_TSH_SALIENCE_SCHEMA_VERSION;
        out_req->requested_gear = requested;
        out_req->request_frame  = frame;
        out_req->priority       = reason;
        out_req->_reserved[0]   = 0u;
        out_req->_reserved[1]   = 0u;
        if (p->armed == 0u) {
            out_req->state    = PRISM_TSH_SALIENCE_UNSET;
            out_req->salience = NAN;
        } else {
            out_req->state    = PRISM_TSH_SALIENCE_VALID;
            out_req->salience = prism_tsh_is_finite(salience) ? (float)salience : NAN;
        }
    }

    // ── Confidence in THIS REQUEST, distinct from observation confidence. ─
    //
    //   * a structurally mandated request (sentinel, rechallenge) is fully
    //     confident: policy, not inference, produced it;
    //   * a request that exists only because there was NO evidence has
    //     confidence 0.0 — it is a fail-safe default and must never be read as
    //     an inference;
    //   * otherwise it is the observation confidence attenuated by how close
    //     the salience sits to the threshold that would have decided
    //     differently.  A decision taken on the knife edge is less trustworthy
    //     than the same decision taken far from it, and saying so is cheaper
    //     than pretending otherwise.
    double request_confidence;
    if (p->armed == 0u) {
        request_confidence = NAN;
    } else if (reason == PRISM_TSH_REASON_SENTINEL ||
               reason == PRISM_TSH_REASON_RECHALLENGE) {
        request_confidence = 1.0;
    } else if (have_salience == 0u) {
        request_confidence = 0.0;
    } else if (!prism_tsh_is_finite(confidence)) {
        request_confidence = NAN;
    } else {
        // Distance to the nearer threshold, saturating at the full band
        // OUTSIDE it.  Taking |min(s-up, down-s)| — the first revision — turned
        // a large negative distance into a small positive one, so a maximally
        // calm branch set at s = 0.0 reported 0.714 instead of 1.0.  A decision
        // taken far outside the band is not a knife-edge decision.
        const double band = p->salience_downshift_threshold -
                            p->salience_upshift_threshold;
        double margin = 1.0;
        if (band > 0.0) {
            if (salience > p->salience_upshift_threshold &&
                salience < p->salience_downshift_threshold) {
                const double d_up = salience - p->salience_upshift_threshold;
                const double d_dn = p->salience_downshift_threshold - salience;
                const double nearest = (d_up < d_dn) ? d_up : d_dn;
                margin = nearest / band;
                if (margin > 1.0) {
                    margin = 1.0;
                } else if (margin < 0.0) {
                    margin = 0.0;
                }
            }
        }
        request_confidence = confidence * margin;
    }

    // ── Emit ALWAYS, including on no-change. ──────────────────────────────
    if (out_rec != nullptr) {
        out_rec->schema_version   = PRISM_TSH_ARM_SCHEMA_VERSION;
        // Low 32 bits of the 64-bit host counter; the full width lives in the
        // state, which is what a checkpoint carries.
        out_rec->record_sequence  = (uint32_t)(s->record_sequence & 0xFFFFFFFFull);
        out_rec->global_md_step   = global_md_step;
        out_rec->cumulative_ps    = now_ps;
        out_rec->integrated_salience =
            prism_tsh_is_finite(mean_salience) ? (float)mean_salience : NAN;
        out_rec->request_confidence  =
            prism_tsh_is_finite(request_confidence) ? (float)request_confidence : NAN;
        out_rec->requested_gear   = requested;
        out_rec->requested_dt_fs  = (requested <= 3u) ? (float)p->gear_dt_fs[requested] : NAN;
        out_rec->request_reason   = reason;
        out_rec->active_branch_mask =
            ((fresh_mask   & 0x3Fu) << PRISM_TSH_ARM_MASK_FRESH_SHIFT)   |
            ((salient_mask & 0x3Fu) << PRISM_TSH_ARM_MASK_SALIENT_SHIFT) |
            ((stale_mask   & 0x3Fu) << PRISM_TSH_ARM_MASK_STALE_SHIFT)   |
            ((unbound_mask & 0x3Fu) << PRISM_TSH_ARM_MASK_UNBOUND_SHIFT);
        out_rec->memory_state     = s->memory_state;
        out_rec->memory_age_ps    = (float)(now_ps - s->memory_entered_ps);
        out_rec->authorized_gear_echo =
            (authorized_gear_valid != 0u) ? authorized_gear : PRISM_TSH_ARM_UNMEASURED_U32;
        out_rec->flags            = flags;
        out_rec->observation_confidence =
            prism_tsh_is_finite(confidence) ? (float)confidence : NAN;
        out_rec->integrated_rate_per_ps =
            prism_tsh_is_finite(rate_per_ps) ? (float)rate_per_ps : NAN;
        out_rec->policy_digest    = p->policy_digest;
        out_rec->peak_branch_salience =
            prism_tsh_is_finite(salience) ? (float)salience : NAN;
        out_rec->observation_coverage =
            prism_tsh_is_finite(coverage) ? (float)coverage : NAN;
        out_rec->thermodynamic_action = prism_tsh_arm_action_for_reason(reason);
        out_rec->peak_branch_index    = peak_index;
    }
}
