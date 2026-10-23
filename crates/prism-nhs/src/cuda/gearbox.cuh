// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / G26 Chronometric Gearbox — Wave B.1 (Foundation)
// ═══════════════════════════════════════════════════════════════════════════
//
// "The Gearbox is the sole master of the chronometric plane." — operator
// directive 2026-05-02 (RULING: Gearbox > Adaptive-DT).
//
// Wave B.1 deliverables (this header / TU):
//
//   1. `__constant__ float d_gearbox_table[16]`
//      Layout: 4 gears × 4 floats each; first slot per gear is dt (ps),
//      remaining three reserved for B.2 (VelocityRescale ratios + Berendsen
//      coupling tau).  Gear 3 is the abort sentinel — all four slots = NaN
//      so any kernel that accidentally reads from this gear's row produces
//      a non-finite result and trips the existing FP-exception traps.
//
//        Gear 0 (0.5 fs): {0.0005, 0, 0, 0}
//        Gear 1 (2.0 fs): {0.0020, 0, 0, 0}
//        Gear 2 (4.0 fs): {0.0040, 0, 0, 0}
//        Gear 3 (Abort) : {NaN,    NaN, NaN, NaN}
//
//   2. `prism_gearbox_init_table_async(host_table[16], stream)` host helper:
//      cudaMemcpyToSymbolAsync of the 64-byte table into __constant__
//      memory.  Called ONCE per pipeline at build time, BEFORE the capture
//      window opens (host-resident write — captured-graph-incompatible).
//
//   3. `prism_gearbox_pointer_swap_kernel`:
//      Single-thread <<<1,1>>> kernel.  State machine:
//        adj->adjudication_code == 2 (Violation) ⇒ Gear 3 (abort)
//        adj->adjudication_code == 1 (Burst)     ⇒ Gear 0; reset cruise.counter,
//                                                   stamp last_burst_frame
//        adj->adjudication_code == 0 (Prune)     ⇒ cruise.counter++, then:
//          counter <  500  ⇒ Gear 1 (2.0 fs safety)
//          counter >= 500  ⇒ Gear 2 (4.0 fs campaign)
//      Kernel writes d_gearbox_table[gear*4+0] into *(adj->d_dt) — the
//      ProtocolState.dt address that all three integrator kernels
//      (nhs_amber_fused_step, nhs_voxel_step, nhs_voxel_step_multi_lif)
//      read on their next launch.
//
//   4. `ChronometricStateTensor` — 16-byte F2-pool device buffer.
//      Persistent state across captured-graph launches; never touched
//      by host once the pipeline is built (Zero-CPU mandate).
//
// Wave B.2 will add VelocityRescale + Berendsen kernels (gear-transition
// momentum conservation) and the predicate bridge that wires this kernel
// into the G26 SWITCH body sub-graphs.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ─── ChronometricStateTensor — 16-byte persistent device state ──────────────
//
// Accessed via raw pointer (not __constant__) so the kernel can both READ
// and WRITE it — __constant__ is read-only from device code.  Layout
// pinned by the Rust-side `#[repr(C, align(16))]` mirror in gearbox.rs.

struct __align__(16) ChronometricStateTensor {
    uint32_t counter;            // frames since last burst (0..∞)        @ 0
    uint32_t last_burst_frame;   // global frame index of most-recent Burst @ 4
    uint32_t current_gear;       // 0/1/2/3 — current PointerSwap choice  @ 8
    uint32_t previous_gear;      // B.2 — gear active BEFORE the latest    @ 12
                                 //       PointerSwap launch (symplectic
                                 //       ratio kernel reads it for
                                 //       λ = dt_new / dt_old).
    // M1.2.17 — Hamiltonian Stability Fuse state.  V_{t-1} for the
    // rolling-window drift check.  Initialised to 0.0 (sentinel for
    // "first frame; skip drift check"); SFA writes V_t here at the
    // end of each launch.
    double   v_prev;             // V at frame t-1 (kcal/mol, f64)        @ 16
    // Pad to 32 bytes for L2-cache-line-aligned access (24 → 32 via
    // align(16)).  Operator-locked layout with explicit padding.
    uint64_t _pad_v_prev;        //                                       @ 24
};

static_assert(sizeof(ChronometricStateTensor) == 32,
              "ChronometricStateTensor MUST be 32 bytes (M1.2.17 v_prev extension).");
static_assert(alignof(ChronometricStateTensor) == 16,
              "ChronometricStateTensor MUST be 16-byte aligned.");
static_assert(offsetof(ChronometricStateTensor, v_prev) == 16,
              "v_prev offset drift: must be 16.");

// ─── Gear-table cardinality ─────────────────────────────────────────────────
//
// 4 gears × 4 floats = 16 floats = 64 bytes.  Pinned to 64 because
// the Blackwell sm_120 L1 constant cache fetches in 64-byte / 128-byte
// lines — the entire table fits in one cache line, broadcasting in a
// single LDC across all 32 lanes of any consuming warp.

#define PRISM_GEARBOX_N_GEARS         4u
#define PRISM_GEARBOX_FLOATS_PER_GEAR 4u
#define PRISM_GEARBOX_TABLE_LEN       (PRISM_GEARBOX_N_GEARS * PRISM_GEARBOX_FLOATS_PER_GEAR)

// Cruise hysteresis threshold — frames of consecutive code=0 (Equilibrium)
// before the gearbox upshifts from Gear 1 (safety) to Gear 2 (campaign).
#define PRISM_GEARBOX_CRUISE_THRESHOLD 500u

// ─── GEAR-PROVENANCE-002 (2026-08-02) — host-readable gear snapshot ────────
//
// Until this surface existed, `cruise->current_gear` NEVER left the device:
// every host-side `gear_id` in the tree was a build-time literal, and the
// only in-band encoding (0..3) has no way to say "unknown" because 0 is a
// legitimate gear.  These constants + `PrismGearboxHostSnapshot` give the
// host a MEASURED gear and the dt that was actually applied, so provenance
// can distinguish "policy asked for gear 0" from "device integrated at
// gear 0".
//
// Sentinel for "no gear measurement exists".  Mirrors
// `PRISM_ZSTR_GEAR_ID_UNMEASURED` in zstr_kernels.cuh and
// `crate::zstr::ZSTR_GEAR_ID_UNMEASURED` on the Rust side.
#define PRISM_GEARBOX_GEAR_UNMEASURED 0xFFFFFFFFu

// Snapshot record schema.  Bump when a field is added or its meaning
// changes; the Rust mirror in gearbox.rs asserts on this value so an old
// reader fails loudly instead of misparsing.
#define PRISM_GEARBOX_SNAPSHOT_SCHEMA_VERSION 1u

// `PrismGearboxHostSnapshot.status` codes.  Every non-OK code names an
// explicit terminal condition; none of them is ever silently reinterpreted
// as a measurement.
#define PRISM_GEARBOX_SNAPSHOT_OK             0u
#define PRISM_GEARBOX_SNAPSHOT_NULL_ARG       1u
#define PRISM_GEARBOX_SNAPSHOT_CAPTURE_ACTIVE 2u
#define PRISM_GEARBOX_SNAPSHOT_EVENT_FAILED   3u
#define PRISM_GEARBOX_SNAPSHOT_TIMEOUT        4u
#define PRISM_GEARBOX_SNAPSHOT_COPY_FAILED    5u

// Return code used by `prism_gearbox_snapshot_state` when the bounded drain
// wait expired.  cudaError_t values are all >= 0, so -1 cannot collide.
#define PRISM_GEARBOX_RC_TIMEOUT (-1)

// Documented default for the bounded drain wait, in milliseconds.  This is
// the SOURCE LITERAL default only; the Rust caller resolves the effective
// budget from policy (`PRISM_GEARBOX_SNAPSHOT_TIMEOUT_MS`) and records both
// the resolved value and its origin.  Passing 0 selects this default.
#define PRISM_GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS 2000u

// Poll granularity of the bounded drain wait, in microseconds.  The wait is
// a nanosleep-throttled `cudaEventQuery` loop, never a spin.
#define PRISM_GEARBOX_SNAPSHOT_POLL_US 100u

/// Host-side gearbox measurement record.  48 bytes, layout pinned by the
/// static_assert below and by the `#[repr(C, align(16))]` mirror in
/// `gearbox.rs`.  Produced ONLY by `prism_gearbox_snapshot_state`.
///
/// `current_gear` / `previous_gear` carry `PRISM_GEARBOX_GEAR_UNMEASURED`
/// whenever `status != PRISM_GEARBOX_SNAPSHOT_OK`; they are never left at a
/// plausible-looking 0.
struct __align__(16) PrismGearboxHostSnapshot {
    uint32_t schema_version;    // @  0  PRISM_GEARBOX_SNAPSHOT_SCHEMA_VERSION
    uint32_t status;            // @  4  PRISM_GEARBOX_SNAPSHOT_*
    uint32_t counter;           // @  8  cruise.counter (0 when unmeasured)
    uint32_t last_burst_frame;  // @ 12  cruise.last_burst_frame
    uint32_t current_gear;      // @ 16  cruise.current_gear or UNMEASURED
    uint32_t previous_gear;     // @ 20  cruise.previous_gear or UNMEASURED
    float    dt_ps_applied;     // @ 24  *(adj->d_dt); valid iff dt_valid==1
    uint32_t dt_valid;          // @ 28  1 = dt_ps_applied is a measurement
    double   v_prev;            // @ 32  cruise.v_prev (Hamiltonian fuse V_{t-1})
    uint64_t waited_us;         // @ 40  MEASURED CLOCK_MONOTONIC microseconds
                                //       spent in the bounded wait (not a count
                                //       of nominal sleep quanta)
};

static_assert(sizeof(PrismGearboxHostSnapshot) == 48,
              "PrismGearboxHostSnapshot MUST be 48 bytes (Rust mirror pins the same).");
static_assert(alignof(PrismGearboxHostSnapshot) == 16,
              "PrismGearboxHostSnapshot MUST be 16-byte aligned.");
static_assert(offsetof(PrismGearboxHostSnapshot, current_gear) == 16,
              "current_gear offset drift: must be 16.");
static_assert(offsetof(PrismGearboxHostSnapshot, dt_ps_applied) == 24,
              "dt_ps_applied offset drift: must be 24.");
static_assert(offsetof(PrismGearboxHostSnapshot, v_prev) == 32,
              "v_prev offset drift: must be 32.");

// ═══════════════════════════════════════════════════════════════════════════
// WP C (2026-08-02) — PHYSICAL STABILITY GOVERNOR
// ═══════════════════════════════════════════════════════════════════════════
//
// ─── Why this exists ───────────────────────────────────────────────────────
//
// The Wave-B.1 automaton (`prism_gearbox_pointer_swap_kernel`, and the
// B.3.2 `prism_gearbox_sfa_kernel` that inherits its transition table)
// upshifts to Gear 2 (4.0 fs) on a single predicate:
//
//     cruise.counter >= PRISM_GEARBOX_CRUISE_THRESHOLD
//
// `cruise.counter` counts consecutive `adjudication_code == 0` frames.
// `adjudication_code` is produced by the interferometric Adjudicator from a
// KL-divergence burst test against a thermal noise floor.  That is an
// INFORMATION-THEORETIC statistic about how surprising the SO(3) manifold
// looks — it says nothing about whether the integrator is numerically stable
// at 4 fs.  Letting it choose the timestep is a category error: "nothing
// interesting happened recently" is not "the Hamiltonian is conserved and
// the fastest bond motion is constrained".
//
// A 4 fs timestep is integrable only when hydrogen-mass repartitioning has
// actually been applied to the topology AND X–H stretches are constrained.
// Running 4 fs without HMR does not crash — it silently produces
// plausible-looking numbers with the wrong dynamics.
//
// ─── The architecture ──────────────────────────────────────────────────────
//
//     scientific-resolution REQUEST   (what detail is valuable now)
//               |                      = the KL/adjudication automaton today,
//               |                        + the TSH salience layer later
//     PHYSICAL STABILITY GOVERNOR     (what timestep is SAFE now)
//               |                      = THIS BLOCK
//     authorized gear = the MORE CONSERVATIVE of the two
//
// The governor never asks for a coarser or finer gear on scientific grounds.
// It publishes a CEILING.  The requester publishes a REQUEST.  The authorized
// gear is `min(request, ceiling)`.  Consequently no amount of scientific
// salience can numerically outvote a physical prohibition — which is why the
// gates below are LEXICOGRAPHIC and not a weighted score.
//
// ─── Gate order (strict; earlier gates cannot be outvoted by later ones) ───
//
//   1. PHYSICAL_EMERGENCY    — non-finite energy, momentum violation,
//                              Adjudicator Violation, drift past the abort
//                              bound, force past the abort bound.  Terminal:
//                              Gear 3 (trap).  No later gate can lift it.
//   2. TOPOLOGY_PERMISSION   — HMR / constraint permission (tri-state).
//   3. CONSTRAINT_STABILITY  — SHAKE/RATTLE failure count in the window.
//   4. ENERGY_DRIFT          — graduated Hamiltonian drift bands.
//   5. FORCE_STABILITY       — graduated force-norm bands vs the integrator
//                              clamp.
//   6. CHUNK_LATENCY         — the uncorrected-window bound (see below).
//   7. TSH_SALIENCE          — the scientific-resolution request.  TYPED
//                              INSERTION POINT; can only LOWER the ceiling.
//   8/9/10. DWELL / LOCKOUT / UPSHIFT_RATE — asymmetry enforcement.  These
//                              restrict UPSHIFT ONLY and never block a
//                              downshift.
//
// ─── Asymmetry contract ────────────────────────────────────────────────────
//
//   DOWNSHIFT: taken on ANY guard degradation or salience request, at the
//              next decision boundary, with no dwell requirement, and may
//              skip gears (2 -> 0 in one decision is legal and expected).
//   UPSHIFT:   requires ALL of — every required guard MEASURED and OK,
//              topology permission GRANTED, `min_upshift_dwell_steps` of
//              continuous all-guards-OK dwell, `instability_lockout_steps`
//              since the last instability, and at most `max_upshift_step`
//              gears per decision (default 1).
//
// ─── CHUNK-BOUNDARY REALITY (operator requirement 5) ───────────────────────
//
// The governor runs once per captured-graph replay.  With
// `--captured-graph-body-unroll N > 1` the Director advances N protocol
// steps per replay, so the gear chosen at replay k is in force for ALL N
// steps of replay k, and a downshift decided during replay k cannot take
// effect until replay k+1.  "Immediate downshift" therefore means "immediate
// at the next chunk boundary", with a bounded, named latency:
//
//     downshift_latency_steps = policy.chunk_span_steps
//     downshift_latency_fs    = chunk_span_steps * gear_dt_fs[current_gear]
//
// The governor does NOT pretend it can correct mid-chunk.  It instead
// chooses a gear that is safe for the WHOLE chunk on entry: the
// CHUNK_LATENCY gate refuses any gear whose uncorrected window
// (`chunk_span_steps * gear_dt_fs[g]`) exceeds `max_uncorrected_window_fs`.
// `downshift_latency_steps` is written into the governor state on every
// decision, so the latency is an emitted quantity and not an assumption.
//
// ─── No fabricated measurements (operator rule 1) ──────────────────────────
//
// Every guard is tri-valued: MEASURED-OK / MEASURED-DEGRADED / UNMEASURED.
// An unbound measurement pointer yields UNMEASURED, which is recorded as such
// (`observed_* = NaN` or an explicit sentinel) and — when that guard is
// required for upshift — pins the ceiling to the current gear.  The governor
// never substitutes a plausible default for a quantity it did not read.
// UNMEASURED is not a deadlock: the safe gear stays reachable and the run
// proceeds; only the aggressive gear is withheld.
//
// ─── No hardcoded thresholds (operator rule 2) ─────────────────────────────
//
// Every threshold lives in `PrismGearGovernorPolicy`.  The literals below are
// the DOCUMENTED DEFAULTS and are the single source of default truth:
// `prism_gearbox_governor_default_policy()` fills them, and the Rust resolver
// in `gearbox.rs` starts from that C-filled struct before overlaying a
// versioned policy file, so the two sides cannot drift.  The resolved values
// and their per-field origin are recorded host-side and bound to every device
// decision through `policy_digest`.
// ═══════════════════════════════════════════════════════════════════════════

/// Governor wire-format version.  Bump whenever `PrismGearGovernorPolicy`,
/// `PrismGearGovernorState`, `PrismGearGuardObservation`, `PrismGearDecision`
/// or `PrismTshSalienceRequest` changes shape.  The Rust mirror asserts on
/// this value and every persisted snapshot carries it, so a reader that sees
/// an unknown version fails loudly instead of misparsing.
#define PRISM_GEAR_GOVERNOR_SCHEMA_VERSION 1u

// ─── Gear identifiers ──────────────────────────────────────────────────────
#define PRISM_GEAR_FINE       0u  /* 0.5 fs — burst / high-resolution        */
#define PRISM_GEAR_SAFE       1u  /* 2.0 fs — constrained-H safety gear      */
#define PRISM_GEAR_CAMPAIGN   2u  /* 4.0 fs — HMR-ONLY campaign sprint       */
#define PRISM_GEAR_ABORT      3u  /* trap sentinel                            */
#define PRISM_GEAR_MAX_ACTIVE PRISM_GEAR_CAMPAIGN
/// Explicit "no value".  Shares the encoding of PRISM_GEARBOX_GEAR_UNMEASURED
/// so a gear slot never has to guess whether 0 means "fine" or "unset".
#define PRISM_GEAR_UNSET      PRISM_GEARBOX_GEAR_UNMEASURED

// ─── Tri-state permission.  Absence of evidence is not evidence of absence,
//     but it still forbids the aggressive gear. ───────────────────────────
#define PRISM_GEAR_PERMISSION_UNKNOWN 0u
#define PRISM_GEAR_PERMISSION_DENIED  1u
#define PRISM_GEAR_PERMISSION_GRANTED 2u

// ─── Per-gate verdicts (4 bits each in PrismGearDecision.gate_verdicts) ────
#define PRISM_GEAR_VERDICT_UNMEASURED 0u
#define PRISM_GEAR_VERDICT_OK         1u
#define PRISM_GEAR_VERDICT_HOLD       2u  /* no upshift; hold current gear   */
#define PRISM_GEAR_VERDICT_DOWNSHIFT  3u  /* one gear finer                  */
#define PRISM_GEAR_VERDICT_FLOOR      4u  /* hard drop to the finest gear    */
#define PRISM_GEAR_VERDICT_EMERGENCY  5u  /* Gear 3 trap                     */
#define PRISM_GEAR_VERDICT_STALE      6u  /* request present but out of date */

// ─── Gate identifiers, in strict authority order ───────────────────────────
#define PRISM_GEAR_GATE_NONE                  0u
#define PRISM_GEAR_GATE_PHYSICAL_EMERGENCY    1u
#define PRISM_GEAR_GATE_TOPOLOGY_PERMISSION   2u
#define PRISM_GEAR_GATE_CONSTRAINT_STABILITY  3u
#define PRISM_GEAR_GATE_ENERGY_DRIFT          4u
#define PRISM_GEAR_GATE_FORCE_STABILITY       5u
#define PRISM_GEAR_GATE_CHUNK_LATENCY         6u
#define PRISM_GEAR_GATE_TSH_SALIENCE          7u
#define PRISM_GEAR_GATE_DWELL                 8u
#define PRISM_GEAR_GATE_LOCKOUT               9u
#define PRISM_GEAR_GATE_UPSHIFT_RATE         10u
#define PRISM_GEAR_GATE_GOVERNOR_UNARMED     11u
#define PRISM_GEAR_GATE_COUNT                12u

// Ceiling-bearing gates occupy ids 1..7.
#define PRISM_GEAR_CEILING_GATE_FIRST PRISM_GEAR_GATE_PHYSICAL_EMERGENCY
#define PRISM_GEAR_CEILING_GATE_LAST  PRISM_GEAR_GATE_TSH_SALIENCE

// ─── Emergency cause bits (PrismGearDecision.emergency_flags) ──────────────
#define PRISM_GEAR_EMERG_ADJUDICATION_VIOLATION 0x1u
#define PRISM_GEAR_EMERG_MOMENTUM_VIOLATION     0x2u
#define PRISM_GEAR_EMERG_NONFINITE_ENERGY       0x4u
#define PRISM_GEAR_EMERG_DRIFT_ABORT            0x8u
#define PRISM_GEAR_EMERG_FORCE_ABORT            0x10u

// ─── TSH salience request states ───────────────────────────────────────────
#define PRISM_TSH_SALIENCE_UNSET 0u  /* no TSH policy layer bound            */
#define PRISM_TSH_SALIENCE_VALID 1u  /* a request is present                 */
#define PRISM_TSH_SALIENCE_HELD  2u  /* layer bound but deliberately silent  */
#define PRISM_TSH_SALIENCE_SCHEMA_VERSION 1u

// ═══════════════════════════════════════════════════════════════════════════
// DOCUMENTED DEFAULTS — the ONLY place a governor threshold is a literal
// ═══════════════════════════════════════════════════════════════════════════

/// Ceiling applied when topology permission is UNKNOWN, or when the governor
/// was never armed.  Default = Gear 1 (2.0 fs) — the gear the captured
/// pipeline already seeds (`ChronometricStateTensor::initial()`, and
/// `prism_gearbox_create_handle_ffi(default_value = 1)`).  Degrading to the
/// engine's own documented safe default beats inventing a new one.
#define PRISM_GEAR_DEFAULT_UNKNOWN_TOPOLOGY_CAP PRISM_GEAR_SAFE

/// Maximum gears an upshift may cross in one decision.  Default 1: a governor
/// that jumps 0 -> 2 has skipped the dwell evidence for gear 1.
#define PRISM_GEAR_DEFAULT_MAX_UPSHIFT_STEP 1u

/// Continuous all-guards-OK dwell required before ANY upshift, in protocol
/// steps.  Default = PRISM_GEARBOX_CRUISE_THRESHOLD (500) — reusing the
/// existing documented cruise literal rather than inventing a second one.
/// This is a PHYSICAL dwell and is independent of the automaton's own KL
/// cruise counter, which keeps counting for its own (scientific) purpose.
#define PRISM_GEAR_DEFAULT_MIN_UPSHIFT_DWELL_STEPS PRISM_GEARBOX_CRUISE_THRESHOLD

/// Steps that must elapse after any instability before an upshift is legal.
/// Default = PRISM_GEARBOX_CRUISE_THRESHOLD: one full cruise window clear of
/// instability.  Same literal, same rationale.
#define PRISM_GEAR_DEFAULT_INSTABILITY_LOCKOUT_STEPS PRISM_GEARBOX_CRUISE_THRESHOLD

/// A TSH salience request older than this many frames is treated as UNSET
/// (verdict STALE).  Bounded staleness: the governor never waits on the
/// policy layer and never blocks on a request that stopped arriving.
#define PRISM_GEAR_DEFAULT_SALIENCE_MAX_AGE_FRAMES PRISM_GEARBOX_CRUISE_THRESHOLD

/// Whether an UNMEASURED force / constraint guard blocks upshift.  Default 1
/// (blocks), per the operator requirement that upshift additionally require
/// physical guards.  Setting either to 0 in a policy file is an explicit,
/// recorded operator decision to upshift on unmeasured physics.
#define PRISM_GEAR_DEFAULT_REQUIRE_FORCE_GUARD      1u
#define PRISM_GEAR_DEFAULT_REQUIRE_CONSTRAINT_GUARD 1u

/// Canonical gear timesteps in fs, mirroring `d_gearbox_table` slot 0
/// (0.0005 / 0.0020 / 0.0040 ps).  Gear 3 is the abort sentinel and has no
/// timestep; 0.0 keeps the chunk-latency arithmetic finite.
#define PRISM_GEAR_DEFAULT_DT_FS_G0 0.5f
#define PRISM_GEAR_DEFAULT_DT_FS_G1 2.0f
#define PRISM_GEAR_DEFAULT_DT_FS_G2 4.0f
#define PRISM_GEAR_DEFAULT_DT_FS_G3 0.0f

/// Longest window, in fs, the governor will leave uncorrected between
/// decision boundaries (`chunk_span_steps * gear_dt_fs[g]`).
///
/// Rationale — POLICY DEFAULT, not a measurement: with X–H stretches
/// constrained by SHAKE the fastest remaining motion is heavy-atom bond
/// stretch/bend, period roughly 20–50 fs.  A 50 fs bound keeps the
/// uncorrected window at or below one period of the fastest resolved motion,
/// so an instability cannot complete a full oscillation before the governor
/// looks again.  At the production unroll of 8 this admits Gear 2 (8x4 = 32
/// fs) and Gear 1 (8x2 = 16 fs); an unroll of 16 pushes Gear 2 to 64 fs and
/// the gate withdraws it.
#define PRISM_GEAR_DEFAULT_MAX_UNCORRECTED_WINDOW_FS 50.0f

/// Drift threshold for the explicitly identified measurement contract. The
/// ordinary physical guard uses previous-endpoint kinetic normalization v1;
/// legacy graph observers retain their declared historical measurement. All
/// numerical graduated bands are unchanged, but this does NOT establish equal
/// calibration across different measurement contracts.
/// Historical relative Hamiltonian drift literal. UNCHANGED
/// source literal from the M1.2.18.5 fuse (the `drift > 0.01` test in
/// `prism_gearbox_sfa_kernel`).  The graduated bands below sit strictly BELOW
/// it within a given measurement contract. Comparisons across normalization
/// contracts require separate calibration; equal literals do not prove it.
#define PRISM_GEAR_DEFAULT_DRIFT_ABORT 0.01f

/// Graduated drift bands as fractions of `drift_abort`:
///   drift <= 0.10 * abort (0.1%)  -> OK, upshift permitted
///   drift <= 0.25 * abort (0.25%) -> HOLD, no upshift
///   drift <= 0.50 * abort (0.5%)  -> DOWNSHIFT one gear
///   drift <= 1.00 * abort (1%)    -> FLOOR, drop to the finest gear
///   drift  > 1.00 * abort         -> EMERGENCY, Gear 3 trap
/// Expressing them as fractions of the single documented abort literal means
/// retuning the fuse moves one anchor and the bands follow.
#define PRISM_GEAR_DEFAULT_DRIFT_UPSHIFT_FRACTION   0.10f
#define PRISM_GEAR_DEFAULT_DRIFT_HOLD_FRACTION      0.25f
#define PRISM_GEAR_DEFAULT_DRIFT_DOWNSHIFT_FRACTION 0.50f

/// Force-guard anchor: the integrator's own force clamp,
/// `MAX_INTEGRATOR_FORCE_KCAL_MOL_A` (nhs_amber_fused.cu).  At or above this
/// magnitude the integrator is clipping forces, which is non-Hamiltonian by
/// construction, so it is the natural physical anchor for the bands.
#define PRISM_GEAR_DEFAULT_FORCE_CLAMP_REFERENCE 1000.0f

/// Graduated force bands as fractions of the clamp reference:
///   |F| <  0.25 * clamp -> OK
///   |F| <  0.50 * clamp -> HOLD
///   |F| <  1.00 * clamp -> DOWNSHIFT one gear
///   |F| >= 1.00 * clamp -> FLOOR (the integrator is clipping)
///   |F| >= 4.00 * clamp -> EMERGENCY
#define PRISM_GEAR_DEFAULT_FORCE_UPSHIFT_FRACTION   0.25f
#define PRISM_GEAR_DEFAULT_FORCE_HOLD_FRACTION      0.50f
#define PRISM_GEAR_DEFAULT_FORCE_DOWNSHIFT_FRACTION 1.00f
#define PRISM_GEAR_DEFAULT_FORCE_ABORT_FRACTION     4.00f

/// SHAKE/RATTLE non-convergence counts within the decision window.  A single
/// non-converged cluster means the constraint algebra is already straining at
/// the current dt, so it blocks upshift; a persistent cluster of failures
/// forces the finest gear.
#define PRISM_GEAR_DEFAULT_CONSTRAINT_FAILURE_HOLD      1u
#define PRISM_GEAR_DEFAULT_CONSTRAINT_FAILURE_DOWNSHIFT 8u

/// Sentinel written into `observed_constraint_failures` when the constraint
/// counter is not bound.  Distinguishable from every real count.
#define PRISM_GEAR_CONSTRAINT_UNMEASURED 0xFFFFFFFFu

/// Consecutive quiet (`adjudication_code == 0`) frames the KL automaton
/// requires before it REQUESTS the campaign gear.  Default =
/// PRISM_GEARBOX_CRUISE_THRESHOLD (500), the existing source literal.
///
/// This governs the REQUESTER, not the governor: it decides when the
/// information-theoretic side asks for 4 fs, and the governor still has to
/// authorize it.  It is policy-resolvable here so that no threshold on either
/// side of the split remains a hardcoded literal.  A resolved value of 0 is
/// read as "use the documented default" rather than "no hysteresis at all" —
/// same convention as `PRISM_GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS`.
#define PRISM_GEAR_DEFAULT_CRUISE_THRESHOLD_STEPS PRISM_GEARBOX_CRUISE_THRESHOLD

// ═══════════════════════════════════════════════════════════════════════════
// Governor structures
// ═══════════════════════════════════════════════════════════════════════════

/// TYPED INSERTION POINT for the TSH (trajectory-surface-hopping) policy
/// layer.  WP C implements the GOVERNOR half only; this struct is the
/// contract the salience layer writes and the governor reads, so the policy
/// layer lands without re-surgery on the governor.
///
/// Contract:
///   * The salience layer writes `schema_version`, `state`, `requested_gear`
///     (0..2, or PRISM_GEAR_UNSET for "no opinion"), `request_frame` and
///     `salience` in [0,1].  It NEVER writes PRISM_GEAR_ABORT: aborting is a
///     physical decision the governor owns.
///   * The governor treats the request as a CEILING ONLY — it can make the
///     authorized gear finer, never coarser.  A salience of 1.0 cannot obtain
///     a gear the physical gates refused.  That is the whole point of the
///     lexicographic ordering.
///   * `request_frame` bounds staleness.  Older than
///     `policy.salience_max_age_frames` -> verdict STALE, treated as UNSET.
///     No wait, no retry, no terminal deadlock: a silent policy layer simply
///     leaves the governor in sole control.
struct __align__(8) PrismTshSalienceRequest {
    uint32_t schema_version;   // @  0  PRISM_TSH_SALIENCE_SCHEMA_VERSION
    uint32_t state;            // @  4  PRISM_TSH_SALIENCE_{UNSET,VALID,HELD}
    uint32_t requested_gear;   // @  8  0..2, or PRISM_GEAR_UNSET
    uint32_t request_frame;    // @ 12  frame the request was written
    float    salience;         // @ 16  [0,1] resolution demand; NaN if UNSET
    uint32_t priority;         // @ 20  reserved for the policy layer's tiering
    uint32_t _reserved[2];     // @ 24
};

static_assert(sizeof(PrismTshSalienceRequest) == 32,
              "PrismTshSalienceRequest MUST be 32 bytes (Rust mirror pins the same).");

/// Resolved governor policy.  Immutable for the life of a run; lives in
/// `__constant__` memory so every decision reads it with one broadcast LDC.
///
/// `policy_digest` is the host-computed fingerprint of the resolved policy
/// (values + per-field origins + policy-file sha256).  It is copied verbatim
/// into every decision record, so an offline reader can bind a gear choice to
/// the exact policy that authorized it without trusting a side channel.
struct __align__(16) PrismGearGovernorPolicy {
    uint32_t schema_version;                       // @   0
    uint32_t armed;                                // @   4
    uint32_t unknown_topology_cap_gear;            // @   8
    uint32_t max_upshift_step;                     // @  12
    uint32_t min_upshift_dwell_steps;              // @  16
    uint32_t instability_lockout_steps;            // @  20
    uint32_t require_force_guard_for_upshift;      // @  24
    uint32_t require_constraint_guard_for_upshift; // @  28
    uint32_t salience_max_age_frames;              // @  32
    uint32_t chunk_span_steps;                     // @  36  body_unroll
    uint32_t hmr_permission;                       // @  40  tri-state
    uint32_t constraint_permission;                // @  44  tri-state
    float    gear_dt_fs[PRISM_GEARBOX_N_GEARS];    // @  48  0.5/2.0/4.0/0
    float    max_uncorrected_window_fs;            // @  64
    float    drift_abort;                          // @  68
    float    drift_upshift_fraction;               // @  72
    float    drift_hold_fraction;                  // @  76
    float    drift_downshift_fraction;             // @  80
    float    force_clamp_reference;                // @  84
    float    force_upshift_fraction;               // @  88
    float    force_hold_fraction;                  // @  92
    float    force_downshift_fraction;             // @  96
    float    force_abort_fraction;                 // @ 100
    uint32_t constraint_failure_hold;              // @ 104
    uint32_t constraint_failure_downshift;         // @ 108
    uint64_t policy_digest;                        // @ 112
    /// Requester-side hysteresis: quiet frames before the KL automaton asks
    /// for the campaign gear.  0 => use PRISM_GEAR_DEFAULT_CRUISE_THRESHOLD_STEPS.
    uint32_t cruise_threshold_steps;               // @ 120
    uint32_t _reserved[5];                         // @ 124..144
};

static_assert(sizeof(PrismGearGovernorPolicy) == 144,
              "PrismGearGovernorPolicy MUST be 144 bytes (Rust mirror pins the same).");
static_assert(offsetof(PrismGearGovernorPolicy, gear_dt_fs) == 48,
              "gear_dt_fs offset drift: must be 48.");
static_assert(offsetof(PrismGearGovernorPolicy, policy_digest) == 112,
              "policy_digest offset drift: must be 112.");

/// Device pointers to the LIVE physical measurements the governor consults.
/// Every pointer is nullable, and null means UNMEASURED — never a substituted
/// default.  Bound once, pre-capture, by
/// `prism_gearbox_governor_bind_measurements_async`.
struct __align__(16) PrismGearGovernorBindings {
    /// Device-visible f32 holding the current force norm in kcal/mol/A,
    /// comparable against `policy.force_clamp_reference`.  Null => the force
    /// guard reports UNMEASURED.
    const float*    d_force_norm;               // @  0
    /// Device-visible u32 holding SHAKE/RATTLE non-convergence events in the
    /// current window.  Null => the constraint guard reports UNMEASURED.
    const uint32_t* d_constraint_failure_count; // @  8
    /// Host-stamped epoch of the last binding update, carried into the
    /// decision record so a stale binding is visible rather than silent.
    uint64_t        binding_epoch;              // @ 16
    uint64_t        _reserved;                  // @ 24
};

static_assert(sizeof(PrismGearGovernorBindings) == 32,
              "PrismGearGovernorBindings MUST be 32 bytes (Rust mirror pins the same).");

/// Persistent governor state plus the emitted decision record.  Device-side
/// module global; the host reads it back with
/// `prism_gearbox_governor_snapshot`.  This is what lets the AUTHORIZED gear
/// and the policy that authorized it leave the device together, so provenance
/// records a measured decision instead of a build-time literal.
struct __align__(16) PrismGearGovernorState {
    uint32_t schema_version;               // @   0
    uint32_t decision_epoch;               // @   4  increments per decision
    uint32_t authorized_gear;              // @   8  PRISM_GEAR_UNSET before
                                           //        the first decision
    uint32_t ceiling_gear;                 // @  12
    uint32_t requested_gear;               // @  16  pre-governor request
    uint32_t binding_gate;                 // @  20  PRISM_GEAR_GATE_*
    uint32_t gate_verdicts;                // @  24  packed 4-bit verdicts
    uint32_t stable_dwell_steps;           // @  28
    uint32_t steps_since_instability;      // @  32
    uint32_t last_downshift_epoch;         // @  36
    uint32_t last_upshift_epoch;           // @  40
    uint32_t downshift_latency_steps;      // @  44  == chunk_span_steps
    float    observed_drift;               // @  48  NaN when UNMEASURED
    float    observed_force_norm;          // @  52  NaN when UNMEASURED
    uint32_t observed_constraint_failures; // @  56  UNMEASURED sentinel
    uint32_t emergency_flags;              // @  60  PRISM_GEAR_EMERG_*
    uint32_t governor_armed;               // @  64  echo of policy.armed
    uint32_t policy_schema_version;        // @  68
    uint64_t policy_digest;                // @  72  binds gear -> policy
    PrismTshSalienceRequest salience;      // @  80..112  insertion point
    uint32_t _reserved[4];                 // @ 112..128
};

static_assert(sizeof(PrismGearGovernorState) == 128,
              "PrismGearGovernorState MUST be 128 bytes (Rust mirror pins the same).");
static_assert(offsetof(PrismGearGovernorState, policy_digest) == 72,
              "policy_digest offset drift: must be 72.");
static_assert(offsetof(PrismGearGovernorState, salience) == 80,
              "salience offset drift: must be 80.");

// R04: one context per physical stream, retained by every capturing parent.
#include "prism_governor_measurement.cuh"
struct alignas(16) PrismGearGovernorBindingsV2 {
    PrismForceWindowV1* force_window;
    PrismConstraintWindowV1* constraint_window;
    uint64_t* physical_epoch;
    uint64_t reserved;
};
struct alignas(16) PrismGearGovernorDeviceV2 {
    uint32_t abi_version, byte_size, source_stream_id, lifecycle;
    uint64_t allocation_generation, binding_epoch;
    PrismGearGovernorPolicy policy;
    PrismGearGovernorBindingsV2 bindings;
    PrismGearGovernorState state;
    uint64_t decision_epoch, observed_force_epoch, observed_constraint_epoch, last_physical_epoch;
    uint32_t force_status, constraint_status, reset_reason, reserved;
};
struct alignas(16) PrismGearGovernorSnapshotV2 {
    uint32_t abi_version, byte_size, source_stream_id, lifecycle;
    uint64_t allocation_generation, binding_epoch;
    PrismGearGovernorState state;
    uint64_t decision_epoch, observed_force_epoch, observed_constraint_epoch, last_physical_epoch;
    uint32_t force_status, constraint_status, reset_reason, reserved;
};
static_assert(sizeof(PrismGearGovernorBindingsV2) == 32, "governor bindings v2 ABI");
static_assert(sizeof(PrismGearGovernorDeviceV2) == 384, "governor device v2 ABI");
static_assert(offsetof(PrismGearGovernorDeviceV2, bindings) == 176, "governor bindings offset");
static_assert(offsetof(PrismGearGovernorDeviceV2, state) == 208, "governor state offset");
static_assert(sizeof(PrismGearGovernorSnapshotV2) == 208, "governor snapshot v2 ABI");

/// One decision's worth of guard observations.  Assembled by the caller (the
/// SFA kernel in production, a host test otherwise) so the decision function
/// performs no I/O and is identically testable on host and device.
struct __align__(16) PrismGearGuardObservation {
    uint32_t adjudication_code;       // @  0  0/1/2 from the Adjudicator
    uint32_t momentum_violation;      // @  4  adj.momentum_violation_flag
    uint32_t nonfinite_energy;        // @  8  V_t is NaN/Inf
    uint32_t drift_measured;          // @ 12  0 => UNMEASURED
    float    drift;                   // @ 16  relative Hamiltonian drift
    uint32_t force_measured;          // @ 20
    float    force_norm;              // @ 24  kcal/mol/A
    uint32_t constraint_measured;     // @ 28
    uint32_t constraint_failures;     // @ 32
    uint32_t current_gear;            // @ 36  gear in force right now
    uint32_t requested_gear;          // @ 40  scientific / KL request
    uint32_t stable_dwell_steps;      // @ 44  carried from prior decision
    uint32_t steps_since_instability; // @ 48  carried from prior decision
    uint32_t frame;                   // @ 52  current frame index
    uint32_t _reserved[2];            // @ 56..64
};

static_assert(sizeof(PrismGearGuardObservation) == 64,
              "PrismGearGuardObservation MUST be 64 bytes (Rust mirror pins the same).");

/// The governor's output for one decision.
struct __align__(16) PrismGearDecision {
    uint32_t authorized_gear;              // @  0
    uint32_t ceiling_gear;                 // @  4
    uint32_t binding_gate;                 // @  8
    uint32_t gate_verdicts;                // @ 12
    uint32_t emergency_flags;              // @ 16
    uint32_t next_stable_dwell_steps;      // @ 20
    uint32_t next_steps_since_instability; // @ 24
    uint32_t downshift_latency_steps;      // @ 28
};

static_assert(sizeof(PrismGearDecision) == 32,
              "PrismGearDecision MUST be 32 bytes (Rust mirror pins the same).");

// ═══════════════════════════════════════════════════════════════════════════
// The decision function — ONE implementation, host and device
// ═══════════════════════════════════════════════════════════════════════════

__host__ __device__ __forceinline__ uint32_t prism_gear_min_u32(uint32_t a, uint32_t b) {
    return (a < b) ? a : b;
}

/// Saturating add — a long quiet run must never wrap a dwell counter back
/// through zero and re-arm an upshift it did not earn.
__host__ __device__ __forceinline__ uint32_t prism_gear_sat_add_u32(uint32_t a, uint32_t b) {
    const uint32_t s = a + b;
    return (s < a) ? 0xffffffffu : s;
}

/// Finite test that behaves identically on host and device without pulling in
/// intrinsics whose availability differs between compilation modes.
__host__ __device__ __forceinline__ bool prism_gear_finite_f32(float v) {
    return (v == v) && (v < 3.0e38f) && (v > -3.0e38f);
}

__host__ __device__ __forceinline__ void prism_gear_set_verdict(
    uint32_t* verdicts, uint32_t gate_id, uint32_t verdict
) {
    // Gates 1..8 occupy nibbles 0..7.
    if (gate_id < 1u || gate_id > 8u) return;
    const uint32_t shift = (gate_id - 1u) * 4u;
    *verdicts = (*verdicts & ~(0xfu << shift)) | ((verdict & 0xfu) << shift);
}

__host__ __device__ __forceinline__ uint32_t prism_gear_get_verdict(
    uint32_t verdicts, uint32_t gate_id
) {
    if (gate_id < 1u || gate_id > 8u) return PRISM_GEAR_VERDICT_UNMEASURED;
    return (verdicts >> ((gate_id - 1u) * 4u)) & 0xfu;
}

/// Highest gear whose uncorrected window fits inside the policy bound.
/// Returns PRISM_GEAR_FINE when even the finest gear overruns: the run must
/// still proceed, so this is a recorded ceiling and not an abort.
__host__ __device__ __forceinline__ uint32_t prism_gear_chunk_latency_ceiling(
    const PrismGearGovernorPolicy* p
) {
    const uint32_t span  = (p->chunk_span_steps == 0u) ? 1u : p->chunk_span_steps;
    const float    bound = p->max_uncorrected_window_fs;
    if (!prism_gear_finite_f32(bound) || bound <= 0.0f) {
        // Bound not resolvable: refuse to let an unresolvable policy silently
        // authorize anything, and fall back to the documented cap.
        return PRISM_GEAR_UNSET;
    }
    uint32_t best = PRISM_GEAR_UNSET;
    for (uint32_t g = 0u; g <= PRISM_GEAR_MAX_ACTIVE; ++g) {
        const float dt = p->gear_dt_fs[g];
        if (!prism_gear_finite_f32(dt) || dt <= 0.0f) continue;
        if ((float)span * dt <= bound) best = g;
    }
    return best;
}

/// THE GOVERNOR.
///
/// Pure function of (policy, observation, salience request): no I/O, no
/// global state, no randomness — so the host FFI entry point and the device
/// kernel provably agree.
///
/// Each ceiling-bearing gate publishes its own ceiling into `gate_ceiling[]`.
/// The final ceiling is the minimum, and the reported `binding_gate` is the
/// MOST AUTHORITATIVE gate whose ceiling equals that minimum.  That is what
/// makes the ordering lexicographic rather than a score: a later gate can
/// only ever tighten the result, never loosen or outvote an earlier one.
__host__ __device__ __forceinline__ void prism_gearbox_governor_decide(
    const PrismGearGovernorPolicy*   p,
    const PrismGearGuardObservation* o,
    const PrismTshSalienceRequest*   tsh,   /* nullable */
    PrismGearDecision*               out
) {
    const uint32_t span = (p->chunk_span_steps == 0u) ? 1u : p->chunk_span_steps;

    out->authorized_gear              = PRISM_GEAR_SAFE;
    out->ceiling_gear                 = PRISM_GEAR_SAFE;
    out->binding_gate                 = PRISM_GEAR_GATE_NONE;
    out->gate_verdicts                = 0u;
    out->emergency_flags              = 0u;
    out->next_stable_dwell_steps      = 0u;
    out->next_steps_since_instability = 0u;
    out->downshift_latency_steps      = span;

    // A current gear of 3 means the previous decision aborted.  Treat the
    // effective current gear as the finest one: recovery restarts from 0.5 fs,
    // never from the gear that blew up.
    const uint32_t cur =
        (o->current_gear <= PRISM_GEAR_MAX_ACTIVE) ? o->current_gear : PRISM_GEAR_FINE;

    // ── GATE 1 — PHYSICAL EMERGENCY (terminal; nothing outranks it) ───────
    uint32_t emerg = 0u;
    if (o->adjudication_code == 2u)  emerg |= PRISM_GEAR_EMERG_ADJUDICATION_VIOLATION;
    if (o->momentum_violation != 0u) emerg |= PRISM_GEAR_EMERG_MOMENTUM_VIOLATION;
    if (o->nonfinite_energy != 0u)   emerg |= PRISM_GEAR_EMERG_NONFINITE_ENERGY;
    if (o->drift_measured != 0u) {
        if (!prism_gear_finite_f32(o->drift) || o->drift > p->drift_abort) {
            emerg |= PRISM_GEAR_EMERG_DRIFT_ABORT;
        }
    }
    if (o->force_measured != 0u) {
        const float abort_force = p->force_clamp_reference * p->force_abort_fraction;
        if (!prism_gear_finite_f32(o->force_norm) || o->force_norm >= abort_force) {
            emerg |= PRISM_GEAR_EMERG_FORCE_ABORT;
        }
    }
    if (emerg != 0u) {
        out->authorized_gear = PRISM_GEAR_ABORT;
        out->ceiling_gear    = PRISM_GEAR_ABORT;
        out->binding_gate    = PRISM_GEAR_GATE_PHYSICAL_EMERGENCY;
        out->emergency_flags = emerg;
        prism_gear_set_verdict(&out->gate_verdicts,
                               PRISM_GEAR_GATE_PHYSICAL_EMERGENCY,
                               PRISM_GEAR_VERDICT_EMERGENCY);
        out->next_stable_dwell_steps      = 0u;
        out->next_steps_since_instability = 0u;
        return;
    }
    prism_gear_set_verdict(&out->gate_verdicts,
                           PRISM_GEAR_GATE_PHYSICAL_EMERGENCY,
                           PRISM_GEAR_VERDICT_OK);

    // Per-gate ceilings, all starting unconstrained at the fastest active
    // gear.  Indexed by gate id; slots 0 and 8.. stay unconstrained.
    uint32_t gate_ceiling[PRISM_GEAR_GATE_COUNT];
    for (uint32_t i = 0u; i < PRISM_GEAR_GATE_COUNT; ++i) {
        gate_ceiling[i] = PRISM_GEAR_MAX_ACTIVE;
    }

    bool instability   = false;  // resets the lockout window
    bool all_guards_ok = true;   // every REQUIRED guard MEASURED and OK

    // ── GATE 2 — TOPOLOGY PERMISSION ─────────────────────────────────────
    // A 4 fs gear is integrable only with hydrogen-mass repartitioning
    // actually applied; a 2 fs gear only with X–H constrained.  UNKNOWN is
    // not GRANTED.
    {
        uint32_t ceil_top = PRISM_GEAR_MAX_ACTIVE;
        uint32_t verdict  = PRISM_GEAR_VERDICT_OK;

        if (p->hmr_permission == PRISM_GEAR_PERMISSION_GRANTED) {
            // 4 fs permitted by topology.
        } else if (p->hmr_permission == PRISM_GEAR_PERMISSION_DENIED) {
            ceil_top = prism_gear_min_u32(ceil_top, PRISM_GEAR_SAFE);
            verdict  = PRISM_GEAR_VERDICT_DOWNSHIFT;
        } else {
            ceil_top = prism_gear_min_u32(ceil_top, p->unknown_topology_cap_gear);
            verdict  = PRISM_GEAR_VERDICT_UNMEASURED;
        }

        if (p->constraint_permission == PRISM_GEAR_PERMISSION_DENIED) {
            // Without H constraints even 2 fs is not integrable.
            ceil_top = PRISM_GEAR_FINE;
            verdict  = PRISM_GEAR_VERDICT_FLOOR;
        } else if (p->constraint_permission != PRISM_GEAR_PERMISSION_GRANTED) {
            ceil_top = prism_gear_min_u32(ceil_top, p->unknown_topology_cap_gear);
            if (verdict == PRISM_GEAR_VERDICT_OK) verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        }

        if (p->armed == 0u) {
            // Unarmed governor: no host ever resolved a policy, so nothing is
            // permitted beyond the documented safe default.  Recorded
            // explicitly below (binding_gate becomes GOVERNOR_UNARMED) so
            // "unarmed" is never mistaken for "governed".
            ceil_top = prism_gear_min_u32(ceil_top, p->unknown_topology_cap_gear);
            // Only downgrade an OK verdict.  A DENIED permission is a stronger,
            // positively-established finding than "nobody armed the governor";
            // overwriting it would report a weaker cause than the one that
            // actually held, which is a fabricated attribution.
            if (verdict == PRISM_GEAR_VERDICT_OK) {
                verdict = PRISM_GEAR_VERDICT_UNMEASURED;
            }
        }

        gate_ceiling[PRISM_GEAR_GATE_TOPOLOGY_PERMISSION] = ceil_top;
        prism_gear_set_verdict(&out->gate_verdicts,
                               PRISM_GEAR_GATE_TOPOLOGY_PERMISSION, verdict);
    }

    // ── GATE 3 — CONSTRAINT STABILITY ────────────────────────────────────
    {
        uint32_t ceil_c = PRISM_GEAR_MAX_ACTIVE;
        uint32_t verdict;
        if (o->constraint_measured == 0u) {
            verdict = PRISM_GEAR_VERDICT_UNMEASURED;
            if (p->require_constraint_guard_for_upshift != 0u) {
                ceil_c = prism_gear_min_u32(ceil_c, cur);   // hold, no upshift
                all_guards_ok = false;
            }
        } else if (o->constraint_failures >= p->constraint_failure_downshift) {
            verdict       = PRISM_GEAR_VERDICT_FLOOR;
            ceil_c        = PRISM_GEAR_FINE;
            instability   = true;
            all_guards_ok = false;
        } else if (o->constraint_failures >= p->constraint_failure_hold) {
            verdict       = PRISM_GEAR_VERDICT_HOLD;
            ceil_c        = prism_gear_min_u32(ceil_c, cur);
            all_guards_ok = false;
        } else {
            verdict = PRISM_GEAR_VERDICT_OK;
        }
        gate_ceiling[PRISM_GEAR_GATE_CONSTRAINT_STABILITY] = ceil_c;
        prism_gear_set_verdict(&out->gate_verdicts,
                               PRISM_GEAR_GATE_CONSTRAINT_STABILITY, verdict);
    }

    // ── GATE 4 — ENERGY DRIFT (graduated Hamiltonian fuse) ───────────────
    // The abort band was handled in gate 1.  Everything here is the GRADUATED
    // path the old fuse lacked: degrading drift walks the gearbox down one
    // step at a time, long before it reaches the trap.
    {
        uint32_t ceil_d = PRISM_GEAR_MAX_ACTIVE;
        uint32_t verdict;
        if (o->drift_measured == 0u) {
            // Drift is the primary stability witness.  Without it no upshift
            // is defensible, regardless of the other policy switches.
            verdict       = PRISM_GEAR_VERDICT_UNMEASURED;
            ceil_d        = prism_gear_min_u32(ceil_d, cur);
            all_guards_ok = false;
        } else {
            const float abort_d     = p->drift_abort;
            const float downshift_d = abort_d * p->drift_downshift_fraction;
            const float hold_d      = abort_d * p->drift_hold_fraction;
            const float upshift_d   = abort_d * p->drift_upshift_fraction;
            if (o->drift > downshift_d) {
                // 0.5%..1% — hard drop to the finest gear, one step from the trap.
                verdict       = PRISM_GEAR_VERDICT_FLOOR;
                ceil_d        = PRISM_GEAR_FINE;
                instability   = true;
                all_guards_ok = false;
            } else if (o->drift > hold_d) {
                // 0.25%..0.5% — one gear finer, immediately.
                verdict       = PRISM_GEAR_VERDICT_DOWNSHIFT;
                ceil_d        = (cur > PRISM_GEAR_FINE) ? (cur - 1u) : PRISM_GEAR_FINE;
                instability   = true;
                all_guards_ok = false;
            } else if (o->drift > upshift_d) {
                // 0.1%..0.25% — hold: not degrading enough to downshift, not
                // quiet enough to earn a coarser gear.
                verdict       = PRISM_GEAR_VERDICT_HOLD;
                ceil_d        = prism_gear_min_u32(ceil_d, cur);
                all_guards_ok = false;
            } else {
                verdict = PRISM_GEAR_VERDICT_OK;
            }
        }
        gate_ceiling[PRISM_GEAR_GATE_ENERGY_DRIFT] = ceil_d;
        prism_gear_set_verdict(&out->gate_verdicts,
                               PRISM_GEAR_GATE_ENERGY_DRIFT, verdict);
    }

    // ── GATE 5 — FORCE STABILITY (graduated) ─────────────────────────────
    {
        uint32_t ceil_f = PRISM_GEAR_MAX_ACTIVE;
        uint32_t verdict;
        if (o->force_measured == 0u) {
            verdict = PRISM_GEAR_VERDICT_UNMEASURED;
            if (p->require_force_guard_for_upshift != 0u) {
                ceil_f = prism_gear_min_u32(ceil_f, cur);
                all_guards_ok = false;
            }
        } else {
            const float ref         = p->force_clamp_reference;
            const float downshift_f = ref * p->force_downshift_fraction;
            const float hold_f      = ref * p->force_hold_fraction;
            const float upshift_f   = ref * p->force_upshift_fraction;
            if (o->force_norm >= downshift_f) {
                // At or above the integrator clamp the forces are being
                // clipped, which is non-Hamiltonian by construction.
                verdict       = PRISM_GEAR_VERDICT_FLOOR;
                ceil_f        = PRISM_GEAR_FINE;
                instability   = true;
                all_guards_ok = false;
            } else if (o->force_norm >= hold_f) {
                verdict       = PRISM_GEAR_VERDICT_DOWNSHIFT;
                ceil_f        = (cur > PRISM_GEAR_FINE) ? (cur - 1u) : PRISM_GEAR_FINE;
                instability   = true;
                all_guards_ok = false;
            } else if (o->force_norm >= upshift_f) {
                verdict       = PRISM_GEAR_VERDICT_HOLD;
                ceil_f        = prism_gear_min_u32(ceil_f, cur);
                all_guards_ok = false;
            } else {
                verdict = PRISM_GEAR_VERDICT_OK;
            }
        }
        gate_ceiling[PRISM_GEAR_GATE_FORCE_STABILITY] = ceil_f;
        prism_gear_set_verdict(&out->gate_verdicts,
                               PRISM_GEAR_GATE_FORCE_STABILITY, verdict);
    }

    // ── GATE 6 — CHUNK LATENCY (the uncorrected-window bound) ────────────
    {
        const uint32_t ceil_l = prism_gear_chunk_latency_ceiling(p);
        if (ceil_l == PRISM_GEAR_UNSET) {
            // Even the finest enabled gear exceeds the governing latency.
            // Do not mislabel its minimum as an eligible physical timestep.
            out->authorized_gear=out->ceiling_gear=PRISM_GEAR_ABORT;
            out->binding_gate=PRISM_GEAR_GATE_CHUNK_LATENCY;
            prism_gear_set_verdict(&out->gate_verdicts,PRISM_GEAR_GATE_CHUNK_LATENCY,PRISM_GEAR_VERDICT_EMERGENCY);
            return;
        }
        gate_ceiling[PRISM_GEAR_GATE_CHUNK_LATENCY] = ceil_l;
        prism_gear_set_verdict(
            &out->gate_verdicts, PRISM_GEAR_GATE_CHUNK_LATENCY,
            (ceil_l >= PRISM_GEAR_MAX_ACTIVE) ? PRISM_GEAR_VERDICT_OK
                                              : PRISM_GEAR_VERDICT_DOWNSHIFT);
    }

    // ── GATE 7 — TSH SALIENCE REQUEST (ceiling only; never a lift) ───────
    {
        uint32_t ceil_s  = PRISM_GEAR_MAX_ACTIVE;
        uint32_t verdict = PRISM_GEAR_VERDICT_UNMEASURED;
        if (tsh != nullptr && tsh->state == PRISM_TSH_SALIENCE_VALID) {
            const uint32_t age = (o->frame >= tsh->request_frame)
                               ? (o->frame - tsh->request_frame)
                               : 0u;   // frame counter reset => treat as fresh
            if (tsh->schema_version != PRISM_TSH_SALIENCE_SCHEMA_VERSION) {
                // Unknown wire format: refuse to interpret it rather than
                // misparse some field into a timestep.
                verdict = PRISM_GEAR_VERDICT_STALE;
            } else if (age > p->salience_max_age_frames) {
                verdict = PRISM_GEAR_VERDICT_STALE;
            } else if (tsh->requested_gear <= PRISM_GEAR_MAX_ACTIVE) {
                ceil_s  = tsh->requested_gear;
                verdict = (ceil_s >= PRISM_GEAR_MAX_ACTIVE)
                        ? PRISM_GEAR_VERDICT_OK : PRISM_GEAR_VERDICT_DOWNSHIFT;
            } else {
                // PRISM_GEAR_UNSET or out of range => no opinion.
                verdict = PRISM_GEAR_VERDICT_OK;
            }
        } else if (tsh != nullptr && tsh->state == PRISM_TSH_SALIENCE_HELD) {
            verdict = PRISM_GEAR_VERDICT_OK;
        }
        gate_ceiling[PRISM_GEAR_GATE_TSH_SALIENCE] = ceil_s;
        prism_gear_set_verdict(&out->gate_verdicts,
                               PRISM_GEAR_GATE_TSH_SALIENCE, verdict);
    }

    // ── Fold the ceilings, then attribute the binding gate ───────────────
    uint32_t ceiling = PRISM_GEAR_MAX_ACTIVE;
    for (uint32_t g = PRISM_GEAR_CEILING_GATE_FIRST;
         g <= PRISM_GEAR_CEILING_GATE_LAST; ++g) {
        ceiling = prism_gear_min_u32(ceiling, gate_ceiling[g]);
    }
    uint32_t binding = PRISM_GEAR_GATE_NONE;
    if (ceiling < PRISM_GEAR_MAX_ACTIVE) {
        for (uint32_t g = PRISM_GEAR_CEILING_GATE_FIRST;
             g <= PRISM_GEAR_CEILING_GATE_LAST; ++g) {
            if (gate_ceiling[g] == ceiling) { binding = g; break; }
        }
    }
    // An unarmed governor is reported as such even though its ceiling is
    // carried by the topology gate — otherwise "unarmed" reads as a topology
    // finding, which would be a fabricated attribution.
    if (p->armed == 0u && binding == PRISM_GEAR_GATE_TOPOLOGY_PERMISSION) {
        binding = PRISM_GEAR_GATE_GOVERNOR_UNARMED;
    }

    // The request.  PRISM_GEAR_ABORT or PRISM_GEAR_UNSET means "no scientific
    // opinion"; aborting is gate 1's business only.
    const uint32_t req =
        (o->requested_gear <= PRISM_GEAR_MAX_ACTIVE) ? o->requested_gear : ceiling;

    // authorized = the MORE CONSERVATIVE of request and ceiling
    uint32_t authorized = prism_gear_min_u32(req, ceiling);

    // ── GATES 8/9/10 — ASYMMETRY: restrict UPSHIFT ONLY ──────────────────
    if (authorized > cur) {
        if (!all_guards_ok) {
            // Belt and braces: a required-but-unmeasured guard already pinned
            // a ceiling, but never let an upshift through on incomplete
            // evidence even if a policy switch loosened that ceiling.
            authorized = cur;
            if (binding == PRISM_GEAR_GATE_NONE) binding = PRISM_GEAR_GATE_DWELL;
        }
        if (authorized > cur && o->stable_dwell_steps < p->min_upshift_dwell_steps) {
            authorized = cur;
            binding    = PRISM_GEAR_GATE_DWELL;
        }
        if (authorized > cur &&
            o->steps_since_instability < p->instability_lockout_steps) {
            authorized = cur;
            binding    = PRISM_GEAR_GATE_LOCKOUT;
        }
        if (authorized > cur) {
            const uint32_t step = (p->max_upshift_step == 0u) ? 1u : p->max_upshift_step;
            if (authorized - cur > step) {
                authorized = cur + step;
                binding    = PRISM_GEAR_GATE_UPSHIFT_RATE;
            }
        }
    }
    // Downshifts are never rate-limited, never dwell-gated, and may skip
    // gears.  That asymmetry IS the safety property.

    // ── Dwell / lockout bookkeeping, in protocol steps ───────────────────
    out->next_stable_dwell_steps =
        all_guards_ok ? prism_gear_sat_add_u32(o->stable_dwell_steps, span) : 0u;
    out->next_steps_since_instability =
        instability ? 0u : prism_gear_sat_add_u32(o->steps_since_instability, span);
    // A gear change itself restarts the physical dwell: evidence that gear A
    // was stable is not evidence that gear B is.
    if (authorized != cur) out->next_stable_dwell_steps = 0u;

    out->authorized_gear         = authorized;
    out->ceiling_gear            = ceiling;
    out->binding_gate            = binding;
    out->downshift_latency_steps = span;
}

// Forward-declaration of the FFI struct from adjudicator.cuh.  We do NOT
// #include "adjudicator.cuh" here because the gearbox kernel reads only
// two fields (adjudication_code and d_dt) and including the full header
// would drag the SO(3) tile definitions into this TU.  Layout drift is
// caught by the static_assert in adjudicator.cuh which pins both fields.
struct InterferometricAdjudicatorFfi;

#ifdef __cplusplus
extern "C" {
#endif

/// Host helper: writes the 16-float gear table into `__constant__` memory.
/// Stream-ordered (cudaMemcpyToSymbolAsync). Call ONCE per pipeline build,
/// BEFORE the captured-graph window opens.  `host_table` is a 16-element
/// f32 array following the layout described at the top of this header.
int prism_gearbox_init_table_async(
    const float* host_table,   /* 16 floats */
    void*        stream);

/// **DEPRECATED (WP C) — UNGOVERNED Wave-A path.  Not for production.**
///
/// This kernel runs the raw information-theoretic automaton with NO physical
/// stability governor: it will upshift to Gear 2 (4.0 fs) purely because
/// `cruise.counter >= PRISM_GEARBOX_CRUISE_THRESHOLD`, with no check that
/// hydrogen-mass repartitioning was applied, no Hamiltonian drift fuse and no
/// force guard.  On a non-HMR topology that silently corrupts the trajectory.
///
/// It is retained deliberately, unchanged and reachable, because it is the
/// Wave-B.1 acceptance fixture (`tests/gearbox_b1_dt_flip.rs` asserts the
/// 2.0 fs -> 4.0 fs flip without host intervention).  Deprecated is not
/// deleted.
///
/// The GOVERNED production path is `prism_gearbox_launch_sfa`.  Anything
/// wiring the gearbox into a real run must use that one.
///
/// Records the PointerSwap kernel onto `stream` as a single graph node.
/// The kernel is single-thread; reads adj->adjudication_code +
/// adj->d_dt, mutates the cruise tensor, writes the active gear's dt
/// through *(adj->d_dt).  Returns cudaError_t cast to int.
///
/// `current_frame` is the global frame index used to stamp
/// `cruise->last_burst_frame` on a Burst transition.  In Wave A the
/// caller passes `0` outside the capture window; B.2's predicate bridge
/// passes a device-resident frame counter.
int prism_gearbox_launch_pointer_swap(
    const InterferometricAdjudicatorFfi* adj,
    ChronometricStateTensor*             cruise,
    uint32_t                             current_frame,
    void*                                stream);

/// B.3.2 — SFA-only kernel (Logic/Action Bifurcation).
/// **This is the GOVERNED production path.**
///
/// Decoupled from the dt-write path: this kernel runs the Stateful
/// Finite Automaton that maps `adj->adjudication_code` → `target_gear`
/// and updates the cruise tensor (counter, current_gear, previous_gear,
/// last_burst_frame).  Does NOT write `*(adj->d_dt)` — that
/// side-effect is owned by the SWITCH body's apply-fixed-dt kernel
/// after the SWITCH has routed the bodies.
///
/// **WP C contract change.** The automaton's `target_gear` is no longer the
/// gear that is applied — it is the scientific-resolution REQUEST.  The
/// kernel then evaluates `prism_gearbox_governor_decide` against the
/// `__constant__` governor policy and the bound live measurements, and writes
/// `cruise->current_gear = min(request, governor ceiling)`.  Consequences the
/// caller must know:
///
///   * The governor is consulted UNCONDITIONALLY on this path.  When it has
///     not been armed (`prism_gearbox_governor_arm_async` never called) the
///     ceiling is `PRISM_GEAR_DEFAULT_UNKNOWN_TOPOLOGY_CAP` (Gear 1 / 2.0 fs)
///     and the decision records `PRISM_GEAR_GATE_GOVERNOR_UNARMED`.  Reaching
///     Gear 2 REQUIRES an armed policy with `hmr_permission == GRANTED` plus
///     measured, in-bounds drift / force / constraint guards.
///   * This legacy SFA observer retains its historical Hamiltonian metric.
///     The separate per-step physical guard declares previous-endpoint kinetic
///     normalization v1. Both retain the numerical `policy.drift_abort` and
///     graduated-band constants; equal scientific calibration across these
///     measurement contracts is not asserted.
///   * The kernel additionally writes the decision record into the device
///     governor state; read it back with `prism_gearbox_governor_snapshot`.
///
/// Wired into the captured pipeline AFTER the Adjudicator step and
/// BEFORE the predicate-bridge kernel.  The cruise.current_gear it
/// writes is the predicate-bridge's input.
int prism_gearbox_launch_sfa(
    const InterferometricAdjudicatorFfi* adj,
    ChronometricStateTensor*             cruise,
    uint32_t                             current_frame,
    void*                                stream);

// ─── B.2 — Velocity Rescale + Berendsen Guard + Predicate Bridge ───────────
//
// Three new kernels per operator directive 2026-05-02 §3.1–§3.3.

/// T12.2 — Symplectic Velocity Rescale.
///
/// Pure momentum-scale kernel: v_i ← v_i · ratio for every f32 in the
/// `n_floats = n_atoms · 3` velocity buffer.  Does NOT touch positions
/// or forces.  Vectorised float4 main loop with a 1-thread tail handler
/// for the `n_floats % 4` trailing scalars; the f32 entry point keeps
/// the FFI surface pointer-aligned-agnostic (the kernel re-interprets
/// to float4 internally and asserts 16-byte alignment).
///
/// `ratio` should be `dt_new / dt_old` (computed by the symplectic
/// ratio kernel — B.3) or a Berendsen λ (computed by the Berendsen
/// guard — same call signature).
int prism_gearbox_launch_velocity_rescale(
    float*    d_velocities,        /* n_atoms × 3 contiguous f32 (AoS) */
    uint32_t  n_floats,
    float     ratio,
    void*     stream);

/// T12.3 — Berendsen weak-coupling guard (Gear 0 entry only).
///
/// Reads `*d_current_temp` (= d_protocol->current_temperature) and
/// `*d_dt` (= d_protocol->dt, the gear-0 timestep just written by
/// PointerSwap), computes
///     λ = sqrt(max(1 + (dt/τ_T)·(T₀/T − 1), ε))
/// and applies v_i ← v_i · λ across the whole velocity buffer.  Single
/// captured node: each thread re-derives λ from the broadcast loads of
/// d_current_temp + d_dt (perfect L1 const-cache hit after the first
/// warp; ~10 extra ALU ops per thread is negligible vs. avoiding a
/// separate scratch-buffer + compute kernel pair).
int prism_gearbox_launch_berendsen_guard(
    float*       d_velocities,
    uint32_t     n_floats,
    const float* d_current_temp,    /* device ptr to d_protocol->current_temperature */
    const float* d_dt,              /* device ptr to d_protocol->dt                 */
    float        target_temp_K,
    float        tau_ps,            /* Berendsen τ_T (operator default 0.5 ps)      */
    void*        stream);

/// T12.4 — 4-way Predicate Bridge.
///
/// Captured AFTER PointerSwap.  Reads cruise->current_gear (already
/// computed by PointerSwap's stateful finite automaton) and forwards
/// the 0/1/2/3 value into the G26 SWITCH conditional handle via
/// `cudaGraphSetConditional(handle, gear)`.  Trivial 1-thread kernel —
/// the heavy lifting is in PointerSwap.  This kernel is the FFI
/// handshake that completes the autonomous gear-shift loop.
///
/// `handle_v` is `cudaGraphConditionalHandle` cast to u64 (driver
/// typedef).  Created via `cudaGraphConditionalHandleCreate` at
/// pipeline build time and bound to the captured graph.
///
/// B.3.2 — additionally consults `adj->gear_override` (u32 at offset 100;
/// Emergency Rectification 2026-05-02 reverted from the M1.2.18.5 attempt
/// to relocate to 136).  When the host writes 0..3 via cuMemcpyHtoDAsync,
/// the bridge short-circuits the SFA's calculated gear with the override.
/// Sentinel 0xFF = Auto.  Pass null `adj` for tests that don't have
/// an InterferometricAdjudicatorFfi fixture (override is simply not
/// consulted).
int prism_gearbox_launch_predicate_bridge(
    uint64_t                                handle_v,
    const InterferometricAdjudicatorFfi*    adj,
    const ChronometricStateTensor*          cruise,
    void*                                   stream);

/// Forward-declaration of the diagnostic trace block defined in
/// adjudicator.cuh; gearbox.cu only takes the device pointer as opaque
/// `uint64_t branch_trace_dev` and does not need the field layout in
/// this header (the kernel reinterprets the pointer).
struct PrismBranchTrace;

/// G26 predicate-bridge launch shim — TRACED variant.
/// Equivalent to `prism_gearbox_launch_predicate_bridge` plus an
/// additional `branch_trace_dev` device pointer aliasing a
/// PrismBranchTrace struct (declared in adjudicator.cuh). Pass 0 to
/// disable trace writes — null-trace path adds zero overhead.
int prism_gearbox_launch_predicate_bridge_traced(
    uint64_t                                handle_v,
    const InterferometricAdjudicatorFfi*    adj,
    const ChronometricStateTensor*          cruise,
    void*                                   stream,
    uint64_t                                branch_trace_dev);

// ─── B.3-narrow — SWITCH body kernels + ratio matrix ───────────────────────

/// B.3 — Constant-memory transition-ratio matrix init.
///
/// 3×3 matrix of dt-ratio factors `[prev_gear * 3 + target_gear]` for
/// the three active gears (0=0.5fs / 1=2.0fs / 2=4.0fs).  Gear 3 (abort)
/// is OUT-OF-BAND — the body 3 sub-graph fires the trap kernel directly
/// without consulting this matrix.  Pre-baked at host build time:
///
///     d_rescale_ratios[0..3]  = { 1.0,    4.0,   8.0  }    // prev=0 → 0/1/2
///     d_rescale_ratios[3..6]  = { 0.25,   1.0,   2.0  }    // prev=1
///     d_rescale_ratios[6..9]  = { 0.125,  0.5,   1.0  }    // prev=2
///
/// Operator constraint (2026-05-02 §2): zero divisions in the body
/// kernels; the rescale kernel performs a single LDC + multiply.
int prism_gearbox_init_rescale_ratios_async(void* stream);

/// B.3 — Symplectic velocity rescale, ratio-matrix variant.
///
/// Reads `cruise->previous_gear` at runtime, indexes the constant
/// ratio matrix, multiplies every f32 in `d_velocities` by that ratio.
/// Single LDC for the ratio (broadcast across the warp); single LDG +
/// MUL + STG per thread.  Block size 256 ⇒ LDG.E.128 / STG.E.128 at
/// the warp level (sm_120 ptxas behaviour).
///
/// `target_gear` is the body's hardcoded 0/1/2.  Body 3 (abort) does
/// NOT call this kernel — the trap kernel runs instead.
int prism_gearbox_launch_rescale(
    float*                              d_velocities,
    uint32_t                            n_floats,
    const ChronometricStateTensor*      cruise,
    uint32_t                            target_gear,
    void*                               stream);

/// B.3 — Apply-fixed-dt kernel.  Single-thread <<<1,1>>> writes
/// `d_gearbox_table[target_gear * 4]` into `*(adj->d_dt)`.  Used inside
/// each body sub-graph (Gears 0/1/2) to commit the gear's timestep
/// after the velocity rescale has completed.  Idempotent: if the
/// upstream PointerSwap already wrote the same value, the second write
/// is a no-op observable.
int prism_gearbox_launch_apply_fixed_dt(
    const InterferometricAdjudicatorFfi* adj,
    uint32_t                             target_gear,
    void*                                stream);

/// B.3 — Hardware trap kernel.  Single-thread <<<1,1>>> issues
/// `asm volatile("trap;");`.  Used inside body 3 (Gear 3 / Abort).
/// Halts the GPU stream and surfaces a CUDA error to the host.
int prism_gearbox_launch_trap(void* stream);

/// B.3 — Populate the four phGraph_out body sub-graphs returned by
/// `prism_wire_g26_gearbox_ffi`.  For each active gear N ∈ {0, 1, 2},
/// adds rescale → [Berendsen if N==0] → apply_fixed_dt as kernel
/// nodes with intra-body dependency edges.  For Gear 3, adds the trap
/// kernel as a single node.
///
/// Returns cudaSuccess on full population.  If any phGraph_out[i] is
/// null (CUDA driver bug — the Wave A B.1 risk register flag),
/// returns `cudaErrorInvalidValue` early as the operator-mandated
/// "Smoking Gun" signal for the kernel-conditional fallback path.
///
/// Berendsen knobs (target_temp_K, tau_ps) are passed through to the
/// Gear 0 body kernel.  Pass null d_current_temp / d_dt to skip the
/// Berendsen guard (for tests that don't have a ProtocolState).
int prism_gearbox_populate_switch_bodies_ffi(
    cudaGraph_t*                          body_subgraphs,        /* [4] */
    const InterferometricAdjudicatorFfi*  adj,
    float*                                d_velocities,
    uint32_t                              n_floats,
    const ChronometricStateTensor*        cruise,
    const float*                          d_current_temp,        /* nullable */
    const float*                          d_dt,                  /* nullable */
    float                                 target_temp_K,
    float                                 tau_ps);

// ─── GEAR-PROVENANCE-002 — device → host gear measurement ─────────────────

/// Copy the live gearbox state off the device so the host can record a
/// MEASURED gear instead of a build-time literal.
///
/// Sequence:
///   1. Refuse if `stream` is mid-capture (a host-resident D2H copy is
///      capture-incompatible).  Status `CAPTURE_ACTIVE`; nothing is read.
///   2. BOUNDED drain wait on `stream`: record an event, then poll
///      `cudaEventQuery` with a `nanosleep(PRISM_GEARBOX_SNAPSHOT_POLL_US)`
///      throttle until the event completes or `timeout_ms` expires.  This is
///      never a spin and never unbounded; expiry is the named terminal
///      condition `PRISM_GEARBOX_RC_TIMEOUT` / status `TIMEOUT`.  No copy is
///      issued before the wait resolves, so a timeout can never leave an
///      asynchronous write aimed at the caller's buffer.
///   3. Synchronous D2H of the 32-byte `ChronometricStateTensor` and, when
///      `d_dt` is non-null, the 4-byte applied timestep.  The stream is
///      already drained, so these do not wait on device work.
///
/// `timeout_ms == 0` selects `PRISM_GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS`.
/// `out` is always fully written, including on every failure path, with
/// `PRISM_GEARBOX_GEAR_UNMEASURED` in both gear slots and `dt_valid = 0`
/// whenever no measurement was obtained.
///
/// Returns `cudaSuccess` on a complete measurement, `PRISM_GEARBOX_RC_TIMEOUT`
/// on bounded-wait expiry, otherwise the failing `cudaError_t` as int.
int prism_gearbox_snapshot_state(
    const ChronometricStateTensor*  cruise,
    const float*                    d_dt,        /* nullable */
    void*                           stream,      /* nullable = default stream */
    uint32_t                        timeout_ms,  /* 0 = documented default */
    struct PrismGearboxHostSnapshot* out);

// ─── WP C — Physical Stability Governor host surface ──────────────────────

/// Fill `out` with the documented default policy (every
/// PRISM_GEAR_DEFAULT_* literal in this header), `armed = 0`,
/// permissions UNKNOWN, `chunk_span_steps = 1`, `policy_digest = 0`.
///
/// This is the SINGLE SOURCE OF DEFAULT TRUTH.  The Rust resolver starts from
/// this struct before overlaying a versioned policy file, so the C defaults
/// and the Rust defaults cannot drift apart.  Pure host computation — safe to
/// call with no CUDA context and no device present.
///
/// Returns `cudaSuccess`, or `cudaErrorInvalidValue` when `out` is null.
int prism_gearbox_governor_default_policy(struct PrismGearGovernorPolicy* out);

/// Evaluate the governor on the host.  Thin wrapper over the same
/// `prism_gearbox_governor_decide` the SFA kernel calls, so a host test
/// exercises the production decision path rather than a re-implementation.
/// `tsh` is nullable (no salience layer bound).  Pure computation; no CUDA
/// context required.
///
/// Returns `cudaSuccess`, or `cudaErrorInvalidValue` on a null argument.
int prism_gearbox_governor_decide_host(
    const struct PrismGearGovernorPolicy*   policy,
    const struct PrismGearGuardObservation* observation,
    const struct PrismTshSalienceRequest*   tsh,     /* nullable */
    struct PrismGearDecision*               out);

/// Arm the governor: copy the resolved policy into `__constant__` memory and
/// reset the device governor state to its pre-decision baseline
/// (`authorized_gear = PRISM_GEAR_UNSET`, zero dwell, zero lockout credit).
///
/// Stream-ordered (cudaMemcpyToSymbolAsync).  MUST be called BEFORE the
/// capture window opens — it is a host-resident write and is
/// captured-graph-incompatible, exactly like
/// `prism_gearbox_init_table_async`.
///
/// Rejects a policy whose `schema_version` is not
/// PRISM_GEAR_GOVERNOR_SCHEMA_VERSION with `cudaErrorInvalidValue`, so a
/// stale writer fails loudly instead of having its bytes reinterpreted.
///
/// Until this is called the governor is UNARMED: the SFA still consults it,
/// the emergency fuse still fires, and the ceiling is pinned to
/// `PRISM_GEAR_DEFAULT_UNKNOWN_TOPOLOGY_CAP` with
/// `binding_gate = PRISM_GEAR_GATE_GOVERNOR_UNARMED`.  Unarmed therefore
/// means "cannot reach 4 fs", never "unrestricted".
int prism_gearbox_governor_arm_async(
    const struct PrismGearGovernorPolicy* policy,
    void*                                 stream);

/// Bind the live physical-measurement pointers the governor consults.
/// Stream-ordered; pre-capture, same discipline as arming.  Passing a
/// `bindings` whose pointers are null (or never calling this at all) leaves
/// the corresponding guards UNMEASURED — which is recorded as such and
/// withholds upshift.  Nothing is ever substituted for an unread quantity.
int prism_gearbox_governor_bind_measurements_async(
    const struct PrismGearGovernorBindings* bindings,
    void*                                   stream);

/// Publish a TSH salience request into the device governor state.
/// Stream-ordered; pre-capture.  This is the host-side half of the typed
/// insertion point — the TSH policy layer calls it, the governor reads the
/// result, and no governor surgery is required when that layer lands.
///
/// Rejects `requested_gear == PRISM_GEAR_ABORT` with `cudaErrorInvalidValue`:
/// the salience layer may request finer resolution, never an abort.
int prism_gearbox_governor_publish_salience_async(
    const struct PrismTshSalienceRequest* request,
    void*                                 stream);

/// Copy the device governor state (last decision + running counters) back to
/// the host.  Synchronous read on `stream`; refuses while `stream` is
/// mid-capture.  `out` is always fully written, including on failure paths,
/// with `authorized_gear = PRISM_GEAR_UNSET` when no measurement was
/// obtained.
///
/// This is the surface that lets an AUTHORIZED gear — together with the
/// `policy_digest` that authorized it and the `binding_gate` that limited it
/// — leave the device, so a provenance record can carry a measured decision
/// instead of a build-time literal.
int prism_gearbox_governor_snapshot(
    void*                            stream,   /* nullable = default stream */
    struct PrismGearGovernorState*   out);

#ifdef __cplusplus
}  // extern "C"
#endif
