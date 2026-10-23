//! GEARBOX-FACING TSH ARM — scientific-resolution DEMAND.
//!
//! Package 4 of the Coupled Temporal / Gearbox / Tensor-Family architecture
//! (`OPERATOR_CONTROL/CAPTURED_GRAPH_TWIN_RUNTIME_REPAIR_20260731/
//! OPERATOR_CONTROL/COUPLED_TEMPORAL_GEARBOX_TENSOR_ARCHITECTURE.md`).
//!
//! # Authority boundary — this module has none
//!
//! This arm determines scientific-resolution **demand**. It has **no authority
//! to authorize a timestep**. Its entire output is a
//! [`crate::gearbox::TshSalienceRequest`] — the typed insertion point the
//! PHYSICAL STABILITY GOVERNOR already reads — plus a record describing how it
//! got there. The governor is the sole authority over what is *safe*, and the
//! gearbox applies `min(demanded, permitted)`.
//!
//! Concretely: this module can never make the run coarser than the governor
//! permits, and it never emits `GEAR_ABORT`. Aborting is a physical decision
//! the governor owns; the request type refuses to encode one.
//!
//! Three gear facts must never collapse, and this module produces exactly one
//! of them:
//!
//! | fact | produced by |
//! |---|---|
//! | `requested_gear` | **this module** |
//! | `authorized_gear` | the governor |
//! | `device_measured_gear` | `gearbox::snapshot_gear_state` |
//!
//! # The critical constraint — per SIMULATED PICOSECOND
//!
//! Salience is normalised per unit **simulated time**: never per record, never
//! per chunk, never per kernel invocation. If it were measured per record the
//! loop has positive feedback in the wrong direction —
//!
//! ```text
//! low salience -> coarser gear -> fewer snapshots -> less evidence
//!              -> lower measured salience -> coarser still -> ...
//! ```
//!
//! — which is stable in the *wrong* equilibrium, because coarsening is
//! rewarded by its own consequence. The controller would talk itself into
//! blindness during exactly the quiet interval preceding a transition, then
//! lack the resolution to see the transition begin.
//!
//! Three mechanisms make that structurally impossible here:
//!
//! 1. **The type system.** A rate can only be built by
//!    [`PerSimulatedPs::from_accumulated`], whose divisor is a
//!    [`SimulatedPsSpan`]. There is no constructor anywhere in this module
//!    that divides by a sample count, a record count, or an invocation count,
//!    and [`SimulatedPsSpan`] cannot be built from a count.
//!    [`BranchObservation::sample_count`] exists for provenance and is never a
//!    divisor.
//!
//! 2. **The integrator advances only over OBSERVED simulated time.** Each
//!    branch integrates the exact solution of `dA/dt = -A/tau + r` over the
//!    simulated span its evidence actually covers, and is **frozen** — not
//!    decayed — over simulated time no evidence covers. Decaying over
//!    unobserved time is the bug that made this property false in the first
//!    revision of this module: it integrates *absence of evidence* as
//!    *measured calm*, so a branch reporting the same physics at a 20% duty
//!    cycle read `0.1395` where full coverage read `0.5000`, crossing from
//!    "hold" into "quiet, coarsening eligible" on identical physics.
//!
//! 3. **Staged evidence is never truncated.** When a staged window is longer
//!    than the decision interval, the surplus is CARRIED FORWARD at the same
//!    rate instead of being discarded, so an observation cadence coarser than
//!    the decision cadence integrates the identical total evidence. Discarding
//!    it was the second half of the same defect.
//!
//! Together these make the salience a function of the physics and of the
//! simulated time it was observed over — not of how often the observer chose
//! to look. See `salience_is_invariant_to_observation_duty_cycle`,
//! `salience_is_invariant_to_observation_cadence`,
//! `normalisation_is_per_simulated_ps_not_per_sample` and
//! `normalisation_is_invariant_to_decision_rate`.
//!
//! # Every branch input must be INTENSIVE, and its unit is stated in-source
//!
//! Normalising per simulated picosecond is necessary but not sufficient. The
//! branch salience is `r / (r + reference)`, so if a producer stages an
//! EXTENSIVE magnitude — an absolute force norm multiplied by the elapsed
//! span, an absolute count — the division by the span returns the magnitude
//! itself, and against any physically plausible reference the ratio pins to
//! `~0.999` on every single decision. The branch then records a CONSTANT that a
//! reader cannot tell from a measurement, and it carries no information about
//! the physics at all. The force branch did exactly this
//! (`|F|_max * span / span = |F|_max ≈ 10²–10³` against a reference of `1.0`).
//!
//! Every producer is therefore contracted to stage a RATE OF CHANGE or a
//! NORMALISED DEVIATION, in the unit [`SalienceBranch::activity_unit`] declares
//! for that family, and those units are chosen so the reference rate's
//! half-saturation point is a physically meaningful excursion:
//!
//! | branch | unit | half-saturation at 1.0 means |
//! |---|---|---|
//! | force / energy | `Δ|F|max / max|F|` per ps | the extreme force turns over completely in 1 ps |
//! | geometry / contact | Å RMS aromatic-centroid drift per ps | a ring traverses one π-stacking contact length in 1 ps |
//!
//! [`SalienceBranch::run_path_producer`] names the code that stages each
//! family, or `None` when this build has none;
//! [`TshGearboxArm::branch_binding_report`] emits the whole table so a
//! permanently absent family is auditable from the artifact rather than from
//! source archaeology. That closes the last hole in "absent is not calm":
//! `ever_observed = false` alone cannot distinguish "no producer exists" from
//! "the producer has not reported yet".
//!
//! # Anti-blindness — all four are mandatory, and all four bite
//!
//! * **(a) minimum physical-time observation cadence** —
//!   [`TshArmPolicy::min_observation_cadence_ps`]. Measured in simulated ps, so
//!   it cannot be satisfied by emitting more records at a coarser gear. Past
//!   the cadence a branch loses confidence; past
//!   [`TshArmPolicy::max_sentinel_interval_ps`] of staleness its confidence is
//!   hard **zero** and it stops contributing at all — an arbitrarily old frozen
//!   activity is not admissible evidence about the present.
//! * **(b) maximum interval without a forced high-resolution sentinel sample**
//!   — [`TshArmPolicy::max_sentinel_interval_ps`]. Fires regardless of
//!   salience, memory state or confidence. The sentinel clock is reset **only**
//!   by an authorized gear *measured* to be fine, never by the request itself.
//! * **(c) uncertainty grows when evidence is sparse** — measured, not
//!   asserted. Branch confidence is `recency x coverage`, where coverage is the
//!   leaky-integrated FRACTION of recent simulated time that staged evidence
//!   actually covered. A branch that reports on time but covers 20% of the
//!   simulated time reports confidence `0.2`, not `1.0`. A branch that never
//!   reported has confidence `0.0` and contributes **nothing**; it does not
//!   contribute a zero salience. With no branch reporting, integrated salience
//!   is `None` (unmeasured), not `Some(0.0)`.
//! * **(d) no coarsening while observation confidence is below threshold** —
//!   [`TshArmPolicy::min_confidence_for_coarsening`]. Because (c) now folds
//!   coverage in, this gate refuses a thin duty cycle instead of waving it
//!   through at confidence `1.000`.
//!
//! # Aggregation is LEXICOGRAPHIC, never a score
//!
//! The fine-ward demand is gated on the **peak** admissible branch salience,
//! never on the confidence-weighted mean. A mean lets five calm branches
//! outvote one branch at maximum salience: with six equal weights, one branch
//! at `0.999` reads `0.1665` — below the upshift threshold, i.e. *coarsening
//! eligible* — and wiring up more evidence families makes the arm blinder to
//! each one. The mean survives as
//! [`TshGearRequest::integrated_salience`], a REPORTED STATISTIC that gates
//! nothing; [`TshGearRequest::peak_branch_salience`] is the gate input and
//! travels on every record so the verdict is auditable.
//!
//! Finer resolution may be triggered by one branch's salience alone. Coarser
//! requires EVERY admissible branch quiet **and** verified observability **and**
//! dwell **and** a permissive memory state — and even then only reaches the
//! governor as a request.
//!
//! # State is path-dependent
//!
//! Reproducing a run requires reproducing the salience trajectory; a seed is
//! not sufficient. [`TshGearboxArm::checkpoint`] / [`TshGearboxArm::restore`]
//! carry the whole trajectory state and refuse a restore under a different
//! policy.
//!
//! # Relationship to the device header
//!
//! `crates/prism-nhs/src/cuda/tsh_gearbox_arm.cuh` implements the identical
//! algorithm for the zero-CPU captured-graph path, with the same struct
//! layouts (asserted on both sides). That header is COMPILED — by
//! `crates/prism-nhs/src/cuda/tsh_gearbox_arm.cu`, registered in `build.rs` —
//! so its `static_assert`s actually execute, and the parity test below calls
//! the compiled device implementation through its `extern "C"` host entry
//! points rather than comparing against literals transcribed into a comment.
//! An uncompiled header's assertions are vacuous, which is the same
//! built-but-never-activated defect this package exists to repair.
//!
//! Every threshold is `double` on **both** sides. The first revision stored the
//! device policy as `float`, so `0.60` meant `0.60000002384185791015625` there
//! and `0.59999999999999997779...` here: the two sides compared against
//! literally different numbers and a threshold-boundary decision could diverge.
//! This module is the REFERENCE ORACLE for policy resolution; with matching
//! widths the only residual difference is the last ULP of `exp()`. Run-to-run
//! determinism on a given side is exact: every reduction here is a fixed-order
//! serial loop over branch indices `0..6`, never an `atomicAdd`-style reduction
//! whose non-associativity could flip a threshold between two identical runs.
//!
//! # Checkpointing covers the side that actually runs
//!
//! [`TshGearboxArm::checkpoint`] carries the host trajectory;
//! [`TshArmCheckpoint::to_device_state_bytes`] /
//! [`TshArmCheckpoint::from_device_state_bytes`] are the declared bridge to and
//! from the device POD, mapping `None` <-> the device's `±INFINITY` and never
//! `None` -> `0.0`. Without it the captured-graph path — the path the `.cuh`
//! exists to serve — could not be checkpointed at all.
//!
//! # Non-interference is PER ARM
//!
//! This arm declares `dt_mutation = true` **for itself**
//! ([`TshGearboxArm::non_interference_claim`]) and touches no global ledger
//! field: `rg TshRuntimeNonInterferenceLedger` finds nothing in this file, so
//! the Stage-II observer arm's own `dt_mutation = false` claim stays intact and
//! independently assertable. While this arm is active, physics-signature
//! equivalence is reportable only as a scoped
//! [`PhysicsSignatureEquivalence::NotApplicableGearboxArmActive`] — never a
//! bare `true`/`false`, because a run whose timestep the arm influenced has no
//! baseline to be equivalent to.

use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ═══════════════════════════════════════════════════════════════════════════
// Schema, cardinality, and the gear encoding
// ═══════════════════════════════════════════════════════════════════════════

/// Wire-format version shared with `PRISM_TSH_ARM_SCHEMA_VERSION`.
pub const TSH_ARM_SCHEMA_VERSION: u32 = 1;

/// Six dendritic branches. Fixed order; the order *is* the reduction order.
pub const TSH_ARM_N_BRANCHES: usize = 6;

/// 0.5 fs — high-resolution capture. Mirrors `PRISM_GEAR_FINE`.
pub const GEAR_FINE: u32 = 0;
/// 2.0 fs — the constrained-hydrogen safety gear. Mirrors `PRISM_GEAR_SAFE`.
pub const GEAR_SAFE: u32 = 1;
/// 4.0 fs — reachable only with verified HMR. Mirrors `PRISM_GEAR_CAMPAIGN`.
pub const GEAR_CAMPAIGN: u32 = 2;
/// Highest gear that actually integrates. Gear 3 is the abort sentinel and
/// this module never requests it.
pub const GEAR_MAX_ACTIVE: u32 = GEAR_CAMPAIGN;
/// Explicit "no value". Shares `PRISM_GEAR_UNSET` / `GEAR_UNMEASURED`.
pub const GEAR_UNSET: u32 = 0xFFFF_FFFF;

// The gear encoding is shared with the governor, so a divergence would be a
// silent 8x rescaling of every downstream time axis.
//
// Two pins, because one of them is unavailable on one of the lanes.
// `crate::gearbox` is `#[cfg(feature = "gpu")]` (lib.rs), so under
// `--no-default-features --features cpu-analysis` the cross-module equality
// cannot be stated. The literal pin below therefore runs on EVERY feature
// combination and fixes this module's encoding to the values in
// `cuda/gearbox.cuh`; the cross-module pin runs whenever the governor is
// present and fixes the two to each other. Neither alone is sufficient: the
// literal pin cannot see a governor change, and the cross-module pin does not
// exist on the analysis lane.
const _: () = {
    assert!(GEAR_FINE == 0);
    assert!(GEAR_SAFE == 1);
    assert!(GEAR_CAMPAIGN == 2);
    assert!(GEAR_MAX_ACTIVE == 2);
    // Gear 3 is the abort sentinel. This module must never be able to name it
    // as a request, so it is not a constant here at all — only the ceiling is.
    assert!(GEAR_UNSET == 0xFFFF_FFFF);
    assert!(TSH_ARM_SCHEMA_VERSION == 1);
};

#[cfg(feature = "gpu")]
const _: () = {
    assert!(GEAR_FINE == crate::gearbox::GEAR_FINE);
    assert!(GEAR_SAFE == crate::gearbox::GEAR_SAFE);
    assert!(GEAR_CAMPAIGN == crate::gearbox::GEAR_CAMPAIGN);
    assert!(GEAR_MAX_ACTIVE == crate::gearbox::GEAR_MAX_ACTIVE);
    assert!(GEAR_UNSET == crate::gearbox::GEAR_UNSET);
    assert!(TSH_ARM_SCHEMA_VERSION == crate::gearbox::TSH_SALIENCE_SCHEMA_VERSION);
};

// ─── Record flags. Mirror `PRISM_TSH_FLAG_*`. ──────────────────────────────

/// Anti-blindness (b) fired: a high-resolution sentinel sample was forced.
pub const FLAG_SENTINEL_FORCED: u32 = 0x01;
/// Anti-blindness (c): no branch had usable evidence this decision.
pub const FLAG_EVIDENCE_ABSENT: u32 = 0x02;
/// Anti-blindness (d): coarsening was otherwise eligible and was refused.
pub const FLAG_CONFIDENCE_BLOCKED: u32 = 0x04;
/// Coarsening was otherwise eligible and minimum dwell had not elapsed.
pub const FLAG_DWELL_BLOCKED: u32 = 0x08;
/// A rechallenge fired.
pub const FLAG_RECHALLENGE: u32 = 0x10;
/// The device ring refused a record.
pub const FLAG_RING_DROPPED: u32 = 0x20;
/// No policy was ever resolved; the arm declined to have an opinion.
pub const FLAG_POLICY_UNARMED: u32 = 0x40;
/// The caller had no measurement of the authorized gear.
pub const FLAG_AUTHORIZED_UNMEASURED: u32 = 0x80;
/// A branch reference rate was unusable, so that branch produced no salience.
pub const FLAG_INVALID_REFERENCE: u32 = 0x100;
/// Anti-blindness (c): the weighted observation coverage this decision was
/// below [`TshArmPolicy::min_confidence_for_coarsening`]. Raised on EVERY
/// decision it is true of, not only inside the coarsening branch, so a reader
/// can see the arm was under-observed even while it was demanding finer.
pub const FLAG_OBSERVATION_SPARSE: u32 = 0x200;
/// A branch's newest evidence was older than
/// [`TshArmPolicy::max_sentinel_interval_ps`]; its frozen activity was refused
/// as evidence about the present.
pub const FLAG_STALE_HORIZON_EXCEEDED: u32 = 0x400;
/// [`SalienceBranch::SignalGridWarp`] evidence was refused because the resolved
/// policy does not declare an independent signal-grid observable. See the
/// ASC / force provenance note on that variant.
pub const FLAG_SIGNAL_GRID_UNBOUND: u32 = 0x800;
/// A branch's carried-forward evidence backlog exceeded the staleness horizon
/// and its oldest surplus was dropped. The producer is staging simulated time
/// faster than the arm is deciding. Bounded and reported, never silent.
pub const FLAG_EVIDENCE_BACKLOG_TRUNCATED: u32 = 0x1000;

/// Explicit "not measured" for a `u32` slot. Mirrors
/// `PRISM_TSH_ARM_UNMEASURED_U32`.
pub const UNMEASURED_U32: u32 = 0xFFFF_FFFF;

// ─── Device allocation sizes ───────────────────────────────────────────────
//
// The device structures live in `cuda/tsh_gearbox_arm.cuh` and their layouts
// are pinned there by `static_assert`. These constants exist so the pipeline
// that allocates them does not have to guess, and so a layout change breaks
// loudly at the allocation site rather than quietly at the first decode.
//
// These are NOT transcribed from a comment. `cuda/tsh_gearbox_arm.cu` is a
// compiled translation unit (registered in `build.rs`) that includes the
// header — so every `static_assert` in it executes on every build — and
// exports `prism_tsh_arm_abi_sizes`, which the test
// `device_abi_sizes_match_the_compiled_header` calls to compare these
// constants against the sizes the compiler actually produced.

/// `sizeof(PrismTshArmBranchState)`.
pub const DEVICE_BRANCH_STATE_BYTES: usize = 64;
/// `sizeof(PrismTshArmPolicy)` — upload once, pre-capture.
pub const DEVICE_POLICY_BYTES: usize = 288;
/// `sizeof(PrismTshArmState)` — persistent across captured-graph launches.
pub const DEVICE_STATE_BYTES: usize = 512;
/// `sizeof(PrismTshArmRecord)` — one ring slot.
pub const DEVICE_RECORD_BYTES: usize = 96;
/// `sizeof(PrismTshArmRing)` — the ring header, excluding its slot array.
pub const DEVICE_RING_HEADER_BYTES: usize = 48;

/// Byte offset of the branch array inside `PrismTshArmState`.
pub const DEVICE_STATE_BRANCH_OFFSET: usize = 112;

const _: () = {
    // The host record mirror and the device record must agree, or every field
    // past the first divergence is misparsed as science.
    assert!(DEVICE_RECORD_BYTES == std::mem::size_of::<TshArmRecordAbi>());
    // Six branch states are embedded in the device state at offset 112, and
    // the trailing 16 bytes are `dropped_records` plus one reserved word.
    assert!(
        DEVICE_STATE_BYTES
            == DEVICE_STATE_BRANCH_OFFSET + TSH_ARM_N_BRANCHES * DEVICE_BRANCH_STATE_BYTES + 16
    );
};

// ═══════════════════════════════════════════════════════════════════════════
// Simulated-time types — the normalisation the architecture depends on
// ═══════════════════════════════════════════════════════════════════════════

/// An absolute coordinate on the **simulated** time axis, in picoseconds.
///
/// This is `cumulative_simulated_time_ps`: the quantity that becomes mandatory
/// the moment the gear varies, because step index stops being proportional to
/// elapsed physics. Two records at the same step can represent different
/// physical moments; two records at the same `SimulatedPs` cannot.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SimulatedPs(f64);

impl SimulatedPs {
    /// Build a simulated-time coordinate. Rejects non-finite input rather than
    /// admitting a NaN clock.
    pub fn new(ps: f64) -> Option<Self> {
        if ps.is_finite() {
            Some(Self(ps))
        } else {
            None
        }
    }

    /// The coordinate in picoseconds.
    pub fn ps(self) -> f64 {
        self.0
    }

    /// Advance by the timestep that was **actually applied**.
    ///
    /// Named for the invariant it serves: cumulative physical time is
    /// incremented with the applied `dt`, never with the requested one.
    pub fn advanced_by_applied_dt(self, applied_dt_ps: f64, steps: u64) -> Option<Self> {
        if !applied_dt_ps.is_finite() || applied_dt_ps <= 0.0 {
            return None;
        }
        Self::new(self.0 + applied_dt_ps * steps as f64)
    }
}

/// A strictly positive elapsed interval of **simulated** time, in picoseconds.
///
/// # Why there is no constructor taking a count
///
/// This type is the only legal divisor for [`PerSimulatedPs::from_accumulated`].
/// If a `SimulatedPsSpan` could be built from a sample count, a record count or
/// an invocation count, the whole per-simulated-picosecond guarantee would be
/// one careless call away from collapsing into the per-record normalisation
/// that drives the controller into blindness. It cannot: every constructor
/// below takes a physical duration, or a step count *multiplied by a timestep*,
/// which is a duration.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct SimulatedPsSpan(f64);

impl SimulatedPsSpan {
    /// A span of `ps` picoseconds of simulated time. `None` unless strictly
    /// positive and finite — a zero span carries no normalisation and
    /// inventing one would be a fabricated measurement.
    pub fn new(ps: f64) -> Option<Self> {
        if ps.is_finite() && ps > 0.0 {
            Some(Self(ps))
        } else {
            None
        }
    }

    /// `steps` integration steps at `dt_ps` each. This is a duration, not a
    /// count: the count is multiplied by physical time, never divided into it.
    pub fn from_steps(steps: u64, dt_ps: f64) -> Option<Self> {
        Self::new(dt_ps * steps as f64)
    }

    /// The interval between two simulated-time coordinates, `later - earlier`.
    pub fn between(earlier: SimulatedPs, later: SimulatedPs) -> Option<Self> {
        Self::new(later.ps() - earlier.ps())
    }

    /// The span in picoseconds.
    pub fn ps(self) -> f64 {
        self.0
    }
}

/// A quantity normalised per unit **simulated picosecond**.
///
/// The newtype exists so a future caller cannot accidentally divide by a
/// sample count: [`PerSimulatedPs::from_accumulated`] is the only constructor,
/// and its divisor is a [`SimulatedPsSpan`]. There is deliberately no
/// `From<f64>`, no `new(f64)`, and no arithmetic that would let a rate be
/// rebuilt from a bare float.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct PerSimulatedPs<T> {
    value: T,
}

impl PerSimulatedPs<f64> {
    /// Divide an accumulated quantity by the **simulated** span it covers.
    ///
    /// This is the only way a rate enters this module.
    pub fn from_accumulated(accumulated: f64, span: SimulatedPsSpan) -> Self {
        Self {
            value: accumulated / span.ps(),
        }
    }

    /// The rate, in accumulated-units per simulated picosecond.
    pub fn get(self) -> f64 {
        self.value
    }
}

impl fmt::Display for PerSimulatedPs<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.9}/ps_simulated", self.value)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Branches
// ═══════════════════════════════════════════════════════════════════════════

/// The six dendritic branches. Each accumulates independently; they are
/// integrated only at the end of a decision.
///
/// A branch is an evidence **family**, not a sensor. Several of these families
/// are not yet chronology-bound in this engine — each branch therefore
/// degrades to "no evidence", which *raises uncertainty*, and never to a
/// silent zero that would read as calm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum SalienceBranch {
    /// Force / energy stability: force norms, Hamiltonian drift proxies.
    ForceEnergyStability = 0,
    /// KCC and coupling change.
    KccCoupling = 1,
    /// Probe interaction.
    ProbeInteraction = 2,
    /// Geometry / contact transition (aromatic proximity, contact make/break).
    GeometryContact = 3,
    /// Signal grid / warp activity — **not** ASC.
    ///
    /// # Why "not ASC", and why this branch is unbound by default
    ///
    /// `{stem}_streamNN_asc_vectors.bin` is written from the `host_forces`
    /// buffer (`bin/nhs_rt_full.rs`, the ASC vector emitter): what ships under
    /// the name "ASC vectors" IS the force buffer under another filename, not
    /// an independent state. A branch reading it alongside
    /// [`Self::ForceEnergyStability`] does not add a second observation of the
    /// system — it counts the force observation twice, giving force `2/6` of
    /// the aggregate weight while every other family gets `1/6`, with nothing
    /// in the record saying so.
    ///
    /// This branch is therefore **signal-grid / warp only**, and its default
    /// weight is [`DEFAULT_SIGNAL_GRID_WARP_WEIGHT`] = `0.0`: unbound until a
    /// genuinely independent signal-grid observable exists and the operator
    /// binds it by policy. [`TshArmPolicy::signal_grid_warp_independent`] is
    /// the opt-in; without it [`TshGearboxArm::observe`] REFUSES an observation
    /// on this branch with
    /// [`ObservationRejection::SignalGridWarpNotIndependentOfForce`] rather
    /// than quietly double-weighting force.
    ///
    /// Evidence that genuinely derives from the force buffer has a supported
    /// path: [`TshGearboxArm::observe_force_derived`], which stages it into
    /// [`Self::ForceEnergyStability`] — the representation it actually belongs
    /// to — so the same physics is counted exactly once.
    SignalGridWarp = 4,
    /// Prior gearbox instability: downshifts, floors, emergencies already seen.
    PriorGearboxInstability = 5,
}

impl SalienceBranch {
    /// Deprecated name for [`Self::SignalGridWarp`].
    ///
    /// Kept reachable so existing references keep resolving, and deprecated
    /// because the name asserted an independence the evidence does not have:
    /// the ASC vector file is the force buffer. Use
    /// [`TshGearboxArm::observe_force_derived`] for force-derived evidence.
    #[deprecated(
        since = "0.1.0",
        note = "ASC vectors are the force buffer under another name; this branch is signal-grid/warp only. \
                Use SalienceBranch::SignalGridWarp for an independent grid observable, or \
                TshGearboxArm::observe_force_derived for force-derived evidence."
    )]
    pub const SignalWarpAsc: SalienceBranch = SalienceBranch::SignalGridWarp;

    /// Fixed iteration order. This order is the reduction order and must not
    /// be permuted: a different order is a different low-bit sum.
    pub const ALL: [SalienceBranch; TSH_ARM_N_BRANCHES] = [
        SalienceBranch::ForceEnergyStability,
        SalienceBranch::KccCoupling,
        SalienceBranch::ProbeInteraction,
        SalienceBranch::GeometryContact,
        SalienceBranch::SignalGridWarp,
        SalienceBranch::PriorGearboxInstability,
    ];

    /// Index into the fixed-order arrays.
    pub fn index(self) -> usize {
        self as u32 as usize
    }

    /// Stable machine-readable label for evidence records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ForceEnergyStability => "force_energy_stability",
            Self::KccCoupling => "kcc_coupling_change",
            Self::ProbeInteraction => "probe_interaction",
            Self::GeometryContact => "geometry_contact_transition",
            Self::SignalGridWarp => "signal_grid_warp_activity",
            Self::PriorGearboxInstability => "prior_gearbox_instability",
        }
    }

    /// Whether this branch's evidence is derived from the force representation.
    ///
    /// Exactly one branch may be so derived, and it is
    /// [`Self::ForceEnergyStability`]. This predicate exists so the exclusivity
    /// against the signal-grid branch is a stated property rather than a
    /// comment someone has to remember.
    pub fn is_force_derived(self) -> bool {
        matches!(self, Self::ForceEnergyStability)
    }

    /// The UNIT of this branch's `accumulated_activity / span` rate.
    ///
    /// # Why a branch must declare a unit
    ///
    /// `TshGearboxArm::branch_salience` evaluates `r / (r + reference)`, which
    /// is only a measurement when `r` and `reference` share a unit AND that unit
    /// is INTENSIVE — a rate of change or a normalised deviation. Stage an
    /// extensive magnitude instead (an absolute force norm, an absolute count)
    /// and the ratio pins to `~0.999` on every decision at any physically
    /// plausible reference: the branch then reports a CONSTANT that a reader
    /// cannot distinguish from a measurement, and the whole family stops
    /// carrying information about the physics.
    ///
    /// The units below are the contract every producer is held to. They travel
    /// on the emitted governor record so a reader can check that the number
    /// staged is the number the normaliser was scaled for.
    ///
    /// `per_simulated_ps` appears in every one of them because normalisation is
    /// per SIMULATED PICOSECOND — never per record, chunk or invocation.
    pub fn activity_unit(self) -> &'static str {
        match self {
            // |ΔF_max| / max(F_max) per simulated ps: the FRACTIONAL rate of
            // change of the extreme force. Dimensionless numerator, so the unit
            // is a pure inverse time and the reference rate `1.0` means "the
            // extreme force turns over completely in one picosecond".
            Self::ForceEnergyStability => "fractional_change_of_max_force_norm_per_simulated_ps",
            Self::KccCoupling => "fractional_change_of_kcc_coupling_per_simulated_ps",
            Self::ProbeInteraction => "fractional_change_of_probe_contact_per_simulated_ps",
            // RMS aromatic ring-centroid displacement in ångström per simulated
            // ps. A LENGTH rate, not a length: the reference rate `1.0 Å/ps` is
            // one π-stacking contact-formation length traversed per picosecond,
            // which is the rate at which a contact can physically make or break.
            Self::GeometryContact => "angstrom_rms_aromatic_centroid_displacement_per_simulated_ps",
            Self::SignalGridWarp => "unit_undeclared_branch_unbound",
            Self::PriorGearboxInstability => "gearbox_instability_events_per_simulated_ps",
        }
    }

    /// The name of the RUN-PATH PRODUCER that stages this branch in this build,
    /// or `None` when the branch has no producer at all.
    ///
    /// # A branch with no input must never look like a branch reporting calm
    ///
    /// A family nobody stages degrades to `ever_observed = false`, which
    /// `TshGearboxArm::branch_salience` answers with `None` and the record
    /// reports through [`BranchMask::unbound`]. That keeps the two facts
    /// arithmetically distinct — but only a reader who already knows which
    /// families this build wired can tell "unbound because the producer has not
    /// run YET" from "unbound because no producer exists AT ALL". This method
    /// is that missing statement, and it travels on the governor closeout, so
    /// a permanently absent family is auditable from the artifact instead of
    /// from source archaeology.
    pub fn run_path_producer(self) -> Option<&'static str> {
        match self {
            Self::ForceEnergyStability => Some(
                "nhs_rt_full PhysicalGovernorBridge: d_forces DtoH -> reduce_force_buffer \
                 -> fractional max|F| change rate between consecutive decisions",
            ),
            Self::GeometryContact => Some(
                "nhs_rt_full PhysicalGovernorBridge: d_aromatic_centroids DtoH -> \
                 reduce_aromatic_centroid_motion -> RMS centroid displacement rate",
            ),
            // NO PRODUCER IN THIS BUILD. Stated, not implied by silence.
            Self::KccCoupling | Self::ProbeInteraction | Self::PriorGearboxInstability => None,
            // Structurally refused until an independent grid observable exists;
            // see the variant doc and `ObservationRejection`.
            Self::SignalGridWarp => None,
        }
    }
}

// Exactly one branch carries the force representation. If a second one ever
// claims it, integrated salience silently double-weights force.
const _: () = {
    let mut i = 0usize;
    let mut n = 0usize;
    while i < TSH_ARM_N_BRANCHES {
        if SalienceBranch::ALL[i].is_force_derived_const() {
            n += 1;
        }
        i += 1;
    }
    assert!(
        n == 1,
        "exactly one branch may carry the force representation"
    );
};

impl SalienceBranch {
    // `const fn` twin of `is_force_derived` so the exclusivity can be asserted
    // at compile time. Kept private; the public predicate is the one to call.
    const fn is_force_derived_const(self) -> bool {
        matches!(self, Self::ForceEnergyStability)
    }
}

/// One branch observation, staged into the current decision window.
///
/// `accumulated_activity` is the branch's own non-negative activity measure,
/// summed over `span` picoseconds of simulated time ending at `window_end`.
///
/// `sample_count` is recorded for provenance and is **never** a divisor. That
/// is not a stylistic preference: dividing by it is precisely the per-record
/// normalisation that makes the control loop reward its own blindness.
#[derive(Debug, Clone, Copy)]
pub struct BranchObservation {
    /// Which evidence family this belongs to.
    pub branch: SalienceBranch,
    /// Non-negative accumulated activity over `span`.
    pub accumulated_activity: f64,
    /// The simulated-time span the accumulation covers.
    pub span: SimulatedPsSpan,
    /// Simulated-time coordinate at which the span ends.
    pub window_end: SimulatedPs,
    /// Underlying device samples. PROVENANCE ONLY.
    pub sample_count: u32,
}

impl BranchObservation {
    /// The observed rate, per simulated picosecond.
    ///
    /// Goes through [`PerSimulatedPs::from_accumulated`], so the divisor is
    /// structurally guaranteed to be a simulated-time span.
    pub fn rate(&self) -> PerSimulatedPs<f64> {
        PerSimulatedPs::from_accumulated(self.accumulated_activity, self.span)
    }
}

/// Why an observation was refused. Refusal is recorded, never silently
/// swallowed and never replaced by a substituted value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservationRejection {
    /// `accumulated_activity` was NaN, infinite, or negative.
    NonFiniteOrNegativeActivity,
    /// An observation was staged on [`SalienceBranch::SignalGridWarp`] while
    /// the resolved policy does not declare an independent signal-grid
    /// observable.
    ///
    /// Accepting it would double-count the force representation, because the
    /// only "ASC" product in this engine is the force buffer under another
    /// filename. Route force-derived evidence through
    /// [`TshGearboxArm::observe_force_derived`] instead, or set
    /// `signal_grid_warp_independent` in the policy document once a genuinely
    /// independent grid observable exists.
    SignalGridWarpNotIndependentOfForce,
}

impl fmt::Display for ObservationRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteOrNegativeActivity => {
                write!(f, "TSH_ARM_OBSERVATION_NONFINITE_OR_NEGATIVE_ACTIVITY")
            }
            Self::SignalGridWarpNotIndependentOfForce => {
                write!(
                    f,
                    "TSH_ARM_OBSERVATION_SIGNAL_GRID_NOT_INDEPENDENT_OF_FORCE"
                )
            }
        }
    }
}

impl std::error::Error for ObservationRejection {}

/// Packed branch masks. Mirrors the four 6-bit fields of
/// `PrismTshArmRecord.active_branch_mask`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BranchMask {
    /// The branch had evidence covering this decision interval — either newly
    /// staged, or carried forward from a window longer than the interval.
    /// Carried-forward evidence counts because it does cover this interval;
    /// that is the point of carrying it rather than discarding it.
    pub fresh: u8,
    /// The branch's own salience reached the downshift bound. This is the
    /// per-branch view of the quantity the fine-ward gate acts on.
    pub salient: u8,
    /// Ever observed, but its confidence is below 1.0 — from staleness, from
    /// thin coverage, or from both. Confidence is `recency x coverage`, so a
    /// branch reporting punctually at a 20% duty cycle is stale in this sense
    /// even though its newest evidence is current.
    pub stale: u8,
    /// Never observed. Structurally unbound, not calm.
    pub unbound: u8,
}

impl BranchMask {
    /// Pack into the on-wire `u32`, matching `PRISM_TSH_ARM_MASK_*_SHIFT`.
    pub fn to_packed_u32(self) -> u32 {
        ((self.fresh as u32) & 0x3F)
            | (((self.salient as u32) & 0x3F) << 8)
            | (((self.stale as u32) & 0x3F) << 16)
            | (((self.unbound as u32) & 0x3F) << 24)
    }

    /// Unpack from the on-wire `u32`.
    pub fn from_packed_u32(v: u32) -> Self {
        Self {
            fresh: (v & 0x3F) as u8,
            salient: ((v >> 8) & 0x3F) as u8,
            stale: ((v >> 16) & 0x3F) as u8,
            unbound: ((v >> 24) & 0x3F) as u8,
        }
    }

    /// Whether `branch` is set in `field`.
    pub fn has(field: u8, branch: SalienceBranch) -> bool {
        field & (1u8 << branch.index()) != 0
    }
}

/// One row of [`TshGearboxArm::branch_binding_report`].
///
/// Serialize-only: it is a VIEW of resolved policy plus live trajectory state,
/// never something a caller can construct off the wire and feed back in.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BranchBindingRow {
    /// The family.
    pub branch: SalienceBranch,
    /// [`SalienceBranch::as_str`].
    pub label: &'static str,
    /// The unit this branch's producer is contracted to stage — see
    /// [`SalienceBranch::activity_unit`].
    pub activity_unit: &'static str,
    /// The run-path producer, or `None` when this build has none.
    pub run_path_producer: Option<&'static str>,
    /// Resolved integration weight.
    pub weight: f64,
    /// Resolved half-saturation reference rate, in `activity_unit`.
    pub reference_rate: f64,
    /// Whether the branch has EVER staged evidence in this run.
    pub ever_observed: bool,
    /// `run_path_producer.is_none()`. TRUE means the family cannot report in
    /// this build at all — a different fact from "has not reported yet", and a
    /// very different fact from "reported calm".
    pub structurally_absent: bool,
    /// Leaky-integrated observation coverage in `[0,1]`.
    pub coverage: f64,
    /// The branch's own salience, or `None` when it is not admissible evidence.
    pub salience: Option<f64>,
    /// Samples ever staged. PROVENANCE ONLY, never a divisor.
    pub total_samples: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// Memory semantics — the operator's table
// ═══════════════════════════════════════════════════════════════════════════

/// TSH memory state.
///
/// | state | resolution consequence |
/// |---|---|
/// | [`Baseline`](Self::Baseline) / [`Low`](Self::Low) | may permit coarser **if physically safe** |
/// | [`Accumulating`](Self::Accumulating) | hold, or begin downshift |
/// | [`ThresholdApproach`](Self::ThresholdApproach) | downshift |
/// | [`ActiveMemory`](Self::ActiveMemory) | maintain fine |
/// | [`Reinforced`](Self::Reinforced) | keep fine |
/// | [`InputWithdrawal`](Self::InputWithdrawal) | hold fine temporarily |
/// | [`Decay`](Self::Decay) | gradual upshift eligibility |
/// | [`Rechallenge`](Self::Rechallenge) | immediate downshift |
///
/// "May permit coarser" is the strongest statement any state here can make.
/// Permission is the governor's; this column describes demand only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum TshMemoryState {
    /// No accumulated demand.
    Baseline = 0,
    /// Measured, and low.
    Low = 1,
    /// Salience is building inside the hysteresis band.
    Accumulating = 2,
    /// Salience crossed the downshift bound.
    ThresholdApproach = 3,
    /// A transition-grade excursion is under way.
    ActiveMemory = 4,
    /// The excursion persisted past the reinforcement dwell.
    Reinforced = 5,
    /// The driving input stopped; fine resolution is held anyway.
    InputWithdrawal = 6,
    /// Decaying; upshift eligibility returns gradually.
    Decay = 7,
    /// A renewed excursion inside the rechallenge window.
    Rechallenge = 8,
}

impl TshMemoryState {
    /// Stable machine-readable label for evidence records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Low => "low",
            Self::Accumulating => "accumulating",
            Self::ThresholdApproach => "threshold_approach",
            Self::ActiveMemory => "active_memory",
            Self::Reinforced => "reinforced",
            Self::InputWithdrawal => "input_withdrawal",
            Self::Decay => "decay",
            Self::Rechallenge => "rechallenge",
        }
    }

    /// The operator table's resolution consequence, verbatim.
    pub fn resolution_consequence(self) -> &'static str {
        match self {
            Self::Baseline | Self::Low => "may permit coarser if physically safe",
            Self::Accumulating => "hold or begin downshift",
            Self::ThresholdApproach => "downshift",
            Self::ActiveMemory => "maintain fine",
            Self::Reinforced => "keep fine",
            Self::InputWithdrawal => "hold fine temporarily",
            Self::Decay => "gradual upshift eligibility",
            Self::Rechallenge => "immediate downshift",
        }
    }

    /// Whether this state may *ever* be the basis of a coarsening request.
    /// Everything else holds or asks finer.
    pub fn permits_coarsening_request(self) -> bool {
        matches!(self, Self::Baseline | Self::Low | Self::Decay)
    }

    fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Baseline,
            1 => Self::Low,
            2 => Self::Accumulating,
            3 => Self::ThresholdApproach,
            4 => Self::ActiveMemory,
            5 => Self::Reinforced,
            6 => Self::InputWithdrawal,
            7 => Self::Decay,
            8 => Self::Rechallenge,
            _ => return None,
        })
    }
}

/// Why the arm requested what it requested. Mirrors `PRISM_TSH_REASON_*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum RequestReason {
    /// No decision has been taken yet.
    Unset = 0,
    /// Gear held; nothing in the evidence justified a change.
    Hold = 1,
    /// Salience crossed the downshift bound: request finer.
    SalienceDownshift = 2,
    /// Memory state mandates fine resolution.
    MemoryHoldFine = 3,
    /// Rechallenge inside the rechallenge window: immediate finest.
    Rechallenge = 4,
    /// Anti-blindness (b): forced high-resolution sentinel sample.
    Sentinel = 5,
    /// Anti-blindness (c): no branch has evidence; bias fine.
    EvidenceAbsent = 6,
    /// Low salience, sufficient confidence, dwell satisfied: request coarser.
    UpshiftEligible = 7,
    /// Anti-blindness (d): coarsening refused by the confidence gate.
    ConfidenceGate = 8,
    /// Coarsening refused because minimum dwell had not elapsed.
    DwellGate = 9,
    /// No policy was ever resolved; the arm declines to have an opinion.
    PolicyUnarmed = 10,
    /// Memory reached threshold approach: downshift one gear.
    MemoryThresholdDownshift = 11,
}

impl RequestReason {
    /// Stable machine-readable label for evidence records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unset => "unset",
            Self::Hold => "hold",
            Self::SalienceDownshift => "salience_downshift",
            Self::MemoryHoldFine => "memory_hold_fine",
            Self::Rechallenge => "rechallenge_immediate_downshift",
            Self::Sentinel => "anti_blindness_sentinel_sample",
            Self::EvidenceAbsent => "anti_blindness_evidence_absent",
            Self::UpshiftEligible => "upshift_eligible",
            Self::ConfidenceGate => "anti_blindness_confidence_gate",
            Self::DwellGate => "hysteresis_dwell_gate",
            Self::PolicyUnarmed => "policy_unarmed",
            Self::MemoryThresholdDownshift => "memory_threshold_downshift",
        }
    }

    /// The **codec-registered** action this reason publishes.
    ///
    /// [`RequestReason`] is the fine-grained sub-reason; the codec-visible
    /// action is `prism_tokenizer`'s [`ThermodynamicActionOpcode`] (artifact
    /// family `ThermodynamicAction`, slot 74, field `tsh_state`). That opcode
    /// is registered in the token schema and had ZERO producers — a vocabulary
    /// entry nothing could ever emit is indistinguishable from a vocabulary
    /// entry that does not exist. This mapping, and
    /// [`TshGearRequest::thermodynamic_action_tokens`], are its producer.
    ///
    /// The mapping is stated as a total function so a new reason cannot be
    /// added without deciding what it means thermodynamically:
    ///
    /// | reason | opcode | why |
    /// |---|---|---|
    /// | `Hold` | `Stable` | nothing in the evidence moved the gear |
    /// | `SalienceDownshift`, `MemoryThresholdDownshift`, `MemoryHoldFine` | `SalienceRise` | demand for finer resolution *is* a salience rise |
    /// | `UpshiftEligible` | `SalienceFall` | the only reason that asks for coarser |
    /// | `ConfidenceGate`, `DwellGate` | `Hysteresis` | a change was otherwise eligible and a hysteresis gate refused it |
    /// | `Sentinel`, `Rechallenge` | `Boundary` | structurally mandated by a clock, not inferred from evidence |
    /// | `PolicyUnarmed`, `EvidenceAbsent`, `Unset` | `Handoff` | the arm declines; the governor is in sole control |
    pub fn thermodynamic_action(self) -> prism_tokenizer::types::ThermodynamicActionOpcode {
        use prism_tokenizer::types::ThermodynamicActionOpcode as Op;
        match self {
            Self::Hold => Op::Stable,
            Self::SalienceDownshift | Self::MemoryThresholdDownshift | Self::MemoryHoldFine => {
                Op::SalienceRise
            }
            Self::UpshiftEligible => Op::SalienceFall,
            Self::ConfidenceGate | Self::DwellGate => Op::Hysteresis,
            Self::Sentinel | Self::Rechallenge => Op::Boundary,
            Self::PolicyUnarmed | Self::EvidenceAbsent | Self::Unset => Op::Handoff,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DOCUMENTED DEFAULTS — the only place a threshold is a source literal
// ═══════════════════════════════════════════════════════════════════════════
//
// Each of these mirrors a `PRISM_TSH_ARM_DEFAULT_*` in
// `cuda/tsh_gearbox_arm.cuh`, where the rationale is documented in full. They
// are POLICY DEFAULTS, not measurements. Every resolved value carries its
// origin onto every emitted record.

/// Leaky-integrator time constant, simulated ps. 500 cruise steps x 4.0 fs.
pub const DEFAULT_BRANCH_TAU_PS: f64 = 2.0;
/// Anti-blindness (a). tau/4: four observations per integrator time constant.
pub const DEFAULT_MIN_OBSERVATION_CADENCE_PS: f64 = 0.5;
/// Anti-blindness (c). One missed cadence interval costs a factor of e.
pub const DEFAULT_CONFIDENCE_DECAY_PS: f64 = 0.5;
/// Anti-blindness (d). Three quarters of expected evidence must be present.
pub const DEFAULT_MIN_CONFIDENCE_FOR_COARSENING: f64 = 0.75;
/// Anti-blindness (b). 5 tau between forced high-resolution samples.
pub const DEFAULT_MAX_SENTINEL_INTERVAL_PS: f64 = 10.0;
/// 50 fs — one period of the fastest resolved motion, per the governor's own
/// `PRISM_GEAR_DEFAULT_MAX_UNCORRECTED_WINDOW_FS`.
pub const DEFAULT_SENTINEL_HOLD_PS: f64 = 0.05;
/// Hysteresis, downshift side. ~1.5x the policy reference rate.
pub const DEFAULT_SALIENCE_DOWNSHIFT: f64 = 0.60;
/// Hysteresis, upshift side. ~1/3 of the policy reference rate.
pub const DEFAULT_SALIENCE_UPSHIFT: f64 = 0.25;
/// Reinforcement bound. ~5.7x the policy reference rate.
pub const DEFAULT_SALIENCE_REINFORCE: f64 = 0.85;
/// Minimum simulated time at a gear before requesting a coarser one.
pub const DEFAULT_MIN_GEAR_DWELL_PS: f64 = 2.0;
/// Simulated time a decaying memory must persist per step of eligibility.
pub const DEFAULT_DECAY_PERIOD_PS: f64 = 2.0;
/// Simulated time fine resolution is held after input withdrawal.
pub const DEFAULT_WITHDRAWAL_HOLD_PS: f64 = 2.0;
/// Simulated time after leaving active memory during which a renewed
/// excursion counts as a rechallenge.
pub const DEFAULT_RECHALLENGE_WINDOW_PS: f64 = 10.0;
/// Simulated time at/above the reinforce bound before active becomes
/// reinforced.
pub const DEFAULT_REINFORCE_DWELL_PS: f64 = 1.0;
/// Default per-branch integration weight for every branch that is a distinct
/// evidence family. Equal: an unequal prior would be a fabricated measurement.
pub const DEFAULT_BRANCH_WEIGHT: f64 = 1.0;
/// Default weight for [`SalienceBranch::SignalGridWarp`] — **zero**.
///
/// Not a prior about how informative signal-grid activity is. The engine has no
/// independent signal-grid observable today: the file that ships as
/// `{stem}_streamNN_asc_vectors.bin` is the `host_forces` buffer. Weighting
/// this branch at 1.0 alongside `ForceEnergyStability` would give force two
/// sixths of the aggregate against one sixth for every other family, silently.
/// Zero is the honest default; the operator raises it by policy at the same
/// time as declaring `signal_grid_warp_independent`.
pub const DEFAULT_SIGNAL_GRID_WARP_WEIGHT: f64 = 0.0;
/// The default weight vector, in fixed branch order.
pub const DEFAULT_BRANCH_WEIGHTS: [f64; TSH_ARM_N_BRANCHES] = [
    DEFAULT_BRANCH_WEIGHT,           // 0 force / energy stability
    DEFAULT_BRANCH_WEIGHT,           // 1 KCC coupling
    DEFAULT_BRANCH_WEIGHT,           // 2 probe interaction
    DEFAULT_BRANCH_WEIGHT,           // 3 geometry / contact
    DEFAULT_SIGNAL_GRID_WARP_WEIGHT, // 4 signal grid / warp — unbound
    DEFAULT_BRANCH_WEIGHT,           // 5 prior gearbox instability
];
/// Whether the resolved policy declares an independent signal-grid observable.
/// False by default: see [`SalienceBranch::SignalGridWarp`].
pub const DEFAULT_SIGNAL_GRID_WARP_INDEPENDENT: bool = false;
/// Default per-branch half-saturation reference rate, activity per simulated
/// ps. A unit reference; a family whose natural units differ MUST set this
/// from policy, and leaving it is a recorded choice, not a measurement.
///
/// # The reference rate is only meaningful with the branch's UNIT
///
/// `branch_salience = r / (r + reference)`, so `reference` has the SAME units
/// as the branch's rate and the number `1.0` means nothing until that unit is
/// stated. A producer that stages an EXTENSIVE magnitude — an absolute force
/// norm, an absolute count — against a reference of `1.0` saturates the ratio
/// to `~0.999` on every decision and the branch stops carrying information: a
/// constant is then recorded as a measurement. Every producer wired to this
/// module must therefore stage a quantity in the unit declared by
/// [`SalienceBranch::activity_unit`], and those units are chosen so the
/// half-saturation point of `1.0` is a physically meaningful excursion.
pub const DEFAULT_BRANCH_REFERENCE_RATE: f64 = 1.0;
/// The default reference-rate vector, in fixed branch order.
///
/// Numerically uniform at [`DEFAULT_BRANCH_REFERENCE_RATE`]. It is spelled out
/// per branch because each entry carries a DIFFERENT unit — see
/// [`SalienceBranch::activity_unit`] — and the shared numeral is a coincidence
/// of how the units were chosen, not evidence that the branches are
/// interchangeable.
pub const DEFAULT_BRANCH_REFERENCE_RATES: [f64; TSH_ARM_N_BRANCHES] = [
    // 0 force / energy stability — 1.0 ps⁻¹: the extreme force changes by
    //   100% of itself within one simulated picosecond.
    DEFAULT_BRANCH_REFERENCE_RATE,
    // 1 KCC coupling — 1.0 ps⁻¹ (fractional coupling change per ps).
    DEFAULT_BRANCH_REFERENCE_RATE,
    // 2 probe interaction — 1.0 ps⁻¹ (fractional probe-contact change per ps).
    DEFAULT_BRANCH_REFERENCE_RATE,
    // 3 geometry / contact — 1.0 Å/ps: an aromatic ring centroid traverses one
    //   full π-stacking contact-formation length in one simulated picosecond.
    DEFAULT_BRANCH_REFERENCE_RATE,
    // 4 signal grid / warp — unit undeclared; the branch is unbound and its
    //   weight is zero, so this entry governs nothing until a producer exists.
    DEFAULT_BRANCH_REFERENCE_RATE,
    // 5 prior gearbox instability — 1.0 events/ps.
    DEFAULT_BRANCH_REFERENCE_RATE,
];
/// Canonical gear timesteps in fs, mirroring `d_gearbox_table` slot 0
/// (0.0005 / 0.0020 / 0.0040 ps). Gear 3 is the abort sentinel.
pub const DEFAULT_GEAR_DT_FS: [f64; 4] = [0.5, 2.0, 4.0, 0.0];
/// Upper bound of the `ThermodynamicActionField::DeltaTimeFs` quantizer range.
///
/// The coarsest gear this arm can ever request is
/// [`GEAR_CAMPAIGN`] at 4.0 fs; the bound is that value, so the token's range
/// is the range the controller actually has rather than an arbitrary ceiling.
/// A policy that widens `gear_dt_fs` beyond it is refused by
/// [`TshArmPolicy::validate`].
pub const TSH_ARM_MAX_DT_FS_BOUND: f64 = 4.0;

// ─── Policy keys ───────────────────────────────────────────────────────────

/// Optional versioned policy document. Absent means "documented defaults
/// only", which is a legal, recorded configuration and not an error.
pub const POLICY_PATH_ENV: &str = "PRISM_TSH_ARM_POLICY_PATH";
/// Optional sha256 pin for the policy document. A mismatch is a hard failure.
pub const POLICY_SHA256_ENV: &str = "PRISM_TSH_ARM_POLICY_SHA256";

const ENV_BRANCH_TAU_PS: &str = "PRISM_TSH_ARM_BRANCH_TAU_PS";
const ENV_MIN_OBSERVATION_CADENCE_PS: &str = "PRISM_TSH_ARM_MIN_OBSERVATION_CADENCE_PS";
const ENV_CONFIDENCE_DECAY_PS: &str = "PRISM_TSH_ARM_CONFIDENCE_DECAY_PS";
const ENV_MIN_CONFIDENCE_FOR_COARSENING: &str = "PRISM_TSH_ARM_MIN_CONFIDENCE_FOR_COARSENING";
const ENV_MAX_SENTINEL_INTERVAL_PS: &str = "PRISM_TSH_ARM_MAX_SENTINEL_INTERVAL_PS";
const ENV_SENTINEL_HOLD_PS: &str = "PRISM_TSH_ARM_SENTINEL_HOLD_PS";
const ENV_SALIENCE_DOWNSHIFT: &str = "PRISM_TSH_ARM_SALIENCE_DOWNSHIFT";
const ENV_SALIENCE_UPSHIFT: &str = "PRISM_TSH_ARM_SALIENCE_UPSHIFT";
const ENV_SALIENCE_REINFORCE: &str = "PRISM_TSH_ARM_SALIENCE_REINFORCE";
const ENV_MIN_GEAR_DWELL_PS: &str = "PRISM_TSH_ARM_MIN_GEAR_DWELL_PS";
const ENV_DECAY_PERIOD_PS: &str = "PRISM_TSH_ARM_DECAY_PERIOD_PS";
const ENV_WITHDRAWAL_HOLD_PS: &str = "PRISM_TSH_ARM_WITHDRAWAL_HOLD_PS";
const ENV_RECHALLENGE_WINDOW_PS: &str = "PRISM_TSH_ARM_RECHALLENGE_WINDOW_PS";
const ENV_REINFORCE_DWELL_PS: &str = "PRISM_TSH_ARM_REINFORCE_DWELL_PS";

/// Operator opt-in declaring that [`SalienceBranch::SignalGridWarp`] is bound
/// to a signal-grid observable that is independent of the force
/// representation. `1`/`true` enables. Absent means NOT independent, which is
/// what the engine's current emitters justify.
pub const SIGNAL_GRID_INDEPENDENT_ENV: &str = "PRISM_TSH_ARM_SIGNAL_GRID_WARP_INDEPENDENT";

// ═══════════════════════════════════════════════════════════════════════════
// Policy resolution — value plus provenance, never a bare number
// ═══════════════════════════════════════════════════════════════════════════

/// Where a resolved value came from. Mirrors the governor's `PolicyOrigin`
/// discipline: a record that states a threshold without stating where the
/// threshold came from is not reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueOrigin {
    /// The documented source literal in this module and in the `.cuh`.
    SourceLiteralDefault,
    /// Read from the versioned policy document.
    PolicyFile,
    /// Read from the per-key environment override.
    PolicyEnv,
    /// The override was present but unusable; the default was used and the
    /// rejection was logged. Never silently honoured.
    PolicyEnvRejected,
}

impl ValueOrigin {
    /// Stable machine-readable label for evidence records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SourceLiteralDefault => "source_literal_default",
            Self::PolicyFile => "policy_file",
            Self::PolicyEnv => "policy_env",
            Self::PolicyEnvRejected => "policy_env_rejected_fell_back_to_source_literal",
        }
    }
}

/// A resolved scalar plus the provenance of the value that was used.
///
/// `Serialize` only: `key` is a `&'static str` naming a policy key that exists
/// in this build. A resolved policy is produced by
/// [`resolve_arm_policy`], never deserialised — accepting a policy struct off
/// the wire would let a value arrive without its resolution ever having run,
/// which is exactly the provenance gap this type exists to close. The on-disk
/// form is [`TshArmPolicyDocument`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ResolvedValue {
    /// The effective value.
    pub value: f64,
    /// Where `value` came from.
    pub origin: ValueOrigin,
    /// The policy key this value answers to.
    pub key: &'static str,
}

impl ResolvedValue {
    fn source_literal(key: &'static str, value: f64) -> Self {
        Self {
            value,
            origin: ValueOrigin::SourceLiteralDefault,
            key,
        }
    }
}

/// The on-disk policy document. `deny_unknown_fields` so a stale or misspelled
/// key fails loudly instead of being silently ignored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TshArmPolicyDocument {
    /// Must equal [`TSH_ARM_SCHEMA_VERSION`] when present.
    #[serde(default)]
    pub schema_version: Option<u32>,
    /// Leaky-integrator time constant, simulated ps.
    #[serde(default)]
    pub branch_tau_ps: Option<f64>,
    /// Anti-blindness (a).
    #[serde(default)]
    pub min_observation_cadence_ps: Option<f64>,
    /// Anti-blindness (c).
    #[serde(default)]
    pub confidence_decay_ps: Option<f64>,
    /// Anti-blindness (d).
    #[serde(default)]
    pub min_confidence_for_coarsening: Option<f64>,
    /// Anti-blindness (b).
    #[serde(default)]
    pub max_sentinel_interval_ps: Option<f64>,
    /// How long a forced sentinel sample is held.
    #[serde(default)]
    pub sentinel_hold_ps: Option<f64>,
    /// Hysteresis, downshift side.
    #[serde(default)]
    pub salience_downshift_threshold: Option<f64>,
    /// Hysteresis, upshift side.
    #[serde(default)]
    pub salience_upshift_threshold: Option<f64>,
    /// Reinforcement bound.
    #[serde(default)]
    pub salience_reinforce_threshold: Option<f64>,
    /// Minimum dwell before a coarsening request.
    #[serde(default)]
    pub min_gear_dwell_ps: Option<f64>,
    /// Decay period per step of upshift eligibility.
    #[serde(default)]
    pub decay_period_ps: Option<f64>,
    /// Hold period after input withdrawal.
    #[serde(default)]
    pub withdrawal_hold_ps: Option<f64>,
    /// Rechallenge window.
    #[serde(default)]
    pub rechallenge_window_ps: Option<f64>,
    /// Reinforcement dwell.
    #[serde(default)]
    pub reinforce_dwell_ps: Option<f64>,
    /// Per-branch integration weights, fixed order.
    #[serde(default)]
    pub branch_weight: Option<[f64; TSH_ARM_N_BRANCHES]>,
    /// Per-branch half-saturation reference rates, fixed order.
    #[serde(default)]
    pub branch_reference_rate: Option<[f64; TSH_ARM_N_BRANCHES]>,
    /// Per-gear timestep in fs, derived from the live gear table.
    #[serde(default)]
    pub gear_dt_fs: Option<[f64; 4]>,
    /// Declare that [`SalienceBranch::SignalGridWarp`] is bound to a signal
    /// grid observable INDEPENDENT of the force representation.
    ///
    /// Absent or `false` means the arm refuses observations on that branch —
    /// see [`SalienceBranch::SignalGridWarp`] for why the default is refusal.
    #[serde(default)]
    pub signal_grid_warp_independent: Option<bool>,
}

/// Named terminal failures of policy resolution.
///
/// None of these means "carried on with a guessed value": a policy that cannot
/// be resolved is not silently replaced by one that can.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TshArmPolicyError {
    /// The policy document could not be read.
    FileUnreadable { path: PathBuf, message: String },
    /// The policy document was not valid JSON, or carried an unknown field.
    FileMalformed { path: PathBuf, message: String },
    /// The sha256 pin did not match.
    Sha256Mismatch {
        path: PathBuf,
        expected: String,
        actual: String,
    },
    /// The document declares a schema this build cannot interpret.
    SchemaMismatch { found: u32, expected: u32 },
    /// A resolved value is not physically usable.
    InvalidValue { field: &'static str, reason: String },
}

impl fmt::Display for TshArmPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileUnreadable { path, message } => write!(
                f,
                "TSH_ARM_POLICY_FILE_UNREADABLE path={} message={message}",
                path.display()
            ),
            Self::FileMalformed { path, message } => write!(
                f,
                "TSH_ARM_POLICY_FILE_MALFORMED path={} message={message}",
                path.display()
            ),
            Self::Sha256Mismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "TSH_ARM_POLICY_SHA256_MISMATCH path={} expected={expected} actual={actual}",
                path.display()
            ),
            Self::SchemaMismatch { found, expected } => write!(
                f,
                "TSH_ARM_POLICY_SCHEMA_MISMATCH found={found} expected={expected}"
            ),
            Self::InvalidValue { field, reason } => {
                write!(
                    f,
                    "TSH_ARM_POLICY_INVALID_VALUE field={field} reason={reason}"
                )
            }
        }
    }
}

impl std::error::Error for TshArmPolicyError {}

/// A fully resolved policy. Immutable for the life of a run.
///
/// Every field carries its own origin, and [`TshArmPolicy::digest`] binds a
/// gear request to this exact policy so an offline reader can tell a request
/// taken under operator policy from one taken under the documented defaults.
///
/// `Serialize` only; see [`ResolvedValue`]. Read a policy with
/// [`resolve_arm_policy`] or [`TshArmPolicy::from_document`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TshArmPolicy {
    /// [`TSH_ARM_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// True once a host has resolved a policy. An UNARMED arm has no opinion
    /// at all and leaves the governor in sole control.
    pub armed: bool,
    /// Leaky-integrator time constant, simulated ps.
    pub branch_tau_ps: ResolvedValue,
    /// Anti-blindness (a).
    pub min_observation_cadence_ps: ResolvedValue,
    /// Anti-blindness (c).
    pub confidence_decay_ps: ResolvedValue,
    /// Anti-blindness (d).
    pub min_confidence_for_coarsening: ResolvedValue,
    /// Anti-blindness (b).
    pub max_sentinel_interval_ps: ResolvedValue,
    /// Sentinel hold.
    pub sentinel_hold_ps: ResolvedValue,
    /// Hysteresis, downshift side.
    pub salience_downshift_threshold: ResolvedValue,
    /// Hysteresis, upshift side.
    pub salience_upshift_threshold: ResolvedValue,
    /// Reinforcement bound.
    pub salience_reinforce_threshold: ResolvedValue,
    /// Minimum dwell before a coarsening request.
    pub min_gear_dwell_ps: ResolvedValue,
    /// Decay period.
    pub decay_period_ps: ResolvedValue,
    /// Withdrawal hold.
    pub withdrawal_hold_ps: ResolvedValue,
    /// Rechallenge window.
    pub rechallenge_window_ps: ResolvedValue,
    /// Reinforcement dwell.
    pub reinforce_dwell_ps: ResolvedValue,
    /// Per-branch weights, fixed order.
    pub branch_weight: [f64; TSH_ARM_N_BRANCHES],
    /// Per-branch reference rates, fixed order.
    pub branch_reference_rate: [f64; TSH_ARM_N_BRANCHES],
    /// Origin of the branch arrays.
    pub branch_arrays_origin: ValueOrigin,
    /// Whether [`SalienceBranch::SignalGridWarp`] is bound to an observable
    /// independent of the force representation. See that variant.
    pub signal_grid_warp_independent: bool,
    /// Origin of [`Self::signal_grid_warp_independent`].
    pub signal_grid_warp_origin: ValueOrigin,
    /// Per-gear timestep in fs.
    pub gear_dt_fs: [f64; 4],
    /// Origin of `gear_dt_fs`.
    pub gear_dt_origin: ValueOrigin,
    /// sha256 of the policy document, when one was read.
    pub policy_file_sha256: Option<String>,
    /// Fingerprint of (values + origins + document sha256).
    pub policy_digest: u64,
}

impl TshArmPolicy {
    /// The documented defaults, UNARMED.
    ///
    /// An unarmed policy is the honest starting state: it produces
    /// `TSH_SALIENCE_UNSET` requests, which the governor treats as "no policy
    /// layer bound" and ignores. Nothing is steered by a policy nobody
    /// resolved.
    pub fn unarmed_defaults() -> Self {
        let mut p = Self {
            schema_version: TSH_ARM_SCHEMA_VERSION,
            armed: false,
            branch_tau_ps: ResolvedValue::source_literal(ENV_BRANCH_TAU_PS, DEFAULT_BRANCH_TAU_PS),
            min_observation_cadence_ps: ResolvedValue::source_literal(
                ENV_MIN_OBSERVATION_CADENCE_PS,
                DEFAULT_MIN_OBSERVATION_CADENCE_PS,
            ),
            confidence_decay_ps: ResolvedValue::source_literal(
                ENV_CONFIDENCE_DECAY_PS,
                DEFAULT_CONFIDENCE_DECAY_PS,
            ),
            min_confidence_for_coarsening: ResolvedValue::source_literal(
                ENV_MIN_CONFIDENCE_FOR_COARSENING,
                DEFAULT_MIN_CONFIDENCE_FOR_COARSENING,
            ),
            max_sentinel_interval_ps: ResolvedValue::source_literal(
                ENV_MAX_SENTINEL_INTERVAL_PS,
                DEFAULT_MAX_SENTINEL_INTERVAL_PS,
            ),
            sentinel_hold_ps: ResolvedValue::source_literal(
                ENV_SENTINEL_HOLD_PS,
                DEFAULT_SENTINEL_HOLD_PS,
            ),
            salience_downshift_threshold: ResolvedValue::source_literal(
                ENV_SALIENCE_DOWNSHIFT,
                DEFAULT_SALIENCE_DOWNSHIFT,
            ),
            salience_upshift_threshold: ResolvedValue::source_literal(
                ENV_SALIENCE_UPSHIFT,
                DEFAULT_SALIENCE_UPSHIFT,
            ),
            salience_reinforce_threshold: ResolvedValue::source_literal(
                ENV_SALIENCE_REINFORCE,
                DEFAULT_SALIENCE_REINFORCE,
            ),
            min_gear_dwell_ps: ResolvedValue::source_literal(
                ENV_MIN_GEAR_DWELL_PS,
                DEFAULT_MIN_GEAR_DWELL_PS,
            ),
            decay_period_ps: ResolvedValue::source_literal(
                ENV_DECAY_PERIOD_PS,
                DEFAULT_DECAY_PERIOD_PS,
            ),
            withdrawal_hold_ps: ResolvedValue::source_literal(
                ENV_WITHDRAWAL_HOLD_PS,
                DEFAULT_WITHDRAWAL_HOLD_PS,
            ),
            rechallenge_window_ps: ResolvedValue::source_literal(
                ENV_RECHALLENGE_WINDOW_PS,
                DEFAULT_RECHALLENGE_WINDOW_PS,
            ),
            reinforce_dwell_ps: ResolvedValue::source_literal(
                ENV_REINFORCE_DWELL_PS,
                DEFAULT_REINFORCE_DWELL_PS,
            ),
            branch_weight: DEFAULT_BRANCH_WEIGHTS,
            branch_reference_rate: DEFAULT_BRANCH_REFERENCE_RATES,
            branch_arrays_origin: ValueOrigin::SourceLiteralDefault,
            signal_grid_warp_independent: DEFAULT_SIGNAL_GRID_WARP_INDEPENDENT,
            signal_grid_warp_origin: ValueOrigin::SourceLiteralDefault,
            gear_dt_fs: DEFAULT_GEAR_DT_FS,
            gear_dt_origin: ValueOrigin::SourceLiteralDefault,
            policy_file_sha256: None,
            policy_digest: 0,
        };
        p.policy_digest = p.compute_digest();
        p
    }

    /// Resolve a policy from a document, ARMED, validating every invariant.
    ///
    /// This is the same path the policy file takes; tests use it so they
    /// exercise production resolution rather than a re-implementation.
    pub fn from_document(
        doc: &TshArmPolicyDocument,
        origin: ValueOrigin,
        policy_file_sha256: Option<String>,
    ) -> Result<Self, TshArmPolicyError> {
        if let Some(v) = doc.schema_version {
            if v != TSH_ARM_SCHEMA_VERSION {
                return Err(TshArmPolicyError::SchemaMismatch {
                    found: v,
                    expected: TSH_ARM_SCHEMA_VERSION,
                });
            }
        }
        let mut p = Self::unarmed_defaults();
        p.armed = true;
        p.policy_file_sha256 = policy_file_sha256;

        fn apply(slot: &mut ResolvedValue, v: Option<f64>, origin: ValueOrigin) {
            if let Some(v) = v {
                slot.value = v;
                slot.origin = origin;
            }
        }
        apply(&mut p.branch_tau_ps, doc.branch_tau_ps, origin);
        apply(
            &mut p.min_observation_cadence_ps,
            doc.min_observation_cadence_ps,
            origin,
        );
        apply(&mut p.confidence_decay_ps, doc.confidence_decay_ps, origin);
        apply(
            &mut p.min_confidence_for_coarsening,
            doc.min_confidence_for_coarsening,
            origin,
        );
        apply(
            &mut p.max_sentinel_interval_ps,
            doc.max_sentinel_interval_ps,
            origin,
        );
        apply(&mut p.sentinel_hold_ps, doc.sentinel_hold_ps, origin);
        apply(
            &mut p.salience_downshift_threshold,
            doc.salience_downshift_threshold,
            origin,
        );
        apply(
            &mut p.salience_upshift_threshold,
            doc.salience_upshift_threshold,
            origin,
        );
        apply(
            &mut p.salience_reinforce_threshold,
            doc.salience_reinforce_threshold,
            origin,
        );
        apply(&mut p.min_gear_dwell_ps, doc.min_gear_dwell_ps, origin);
        apply(&mut p.decay_period_ps, doc.decay_period_ps, origin);
        apply(&mut p.withdrawal_hold_ps, doc.withdrawal_hold_ps, origin);
        apply(
            &mut p.rechallenge_window_ps,
            doc.rechallenge_window_ps,
            origin,
        );
        apply(&mut p.reinforce_dwell_ps, doc.reinforce_dwell_ps, origin);
        if let Some(w) = doc.branch_weight {
            p.branch_weight = w;
            p.branch_arrays_origin = origin;
        }
        if let Some(r) = doc.branch_reference_rate {
            p.branch_reference_rate = r;
            p.branch_arrays_origin = origin;
        }
        if let Some(g) = doc.gear_dt_fs {
            p.gear_dt_fs = g;
            p.gear_dt_origin = origin;
        }
        if let Some(v) = doc.signal_grid_warp_independent {
            p.signal_grid_warp_independent = v;
            p.signal_grid_warp_origin = origin;
        }

        p.validate()?;
        p.policy_digest = p.compute_digest();
        Ok(p)
    }

    /// Reject anything not physically usable. Never clamps: a bad value is a
    /// named terminal error, not a silently repaired one.
    fn validate(&self) -> Result<(), TshArmPolicyError> {
        fn positive(field: &'static str, v: f64) -> Result<(), TshArmPolicyError> {
            if v.is_finite() && v > 0.0 {
                Ok(())
            } else {
                Err(TshArmPolicyError::InvalidValue {
                    field,
                    reason: format!("must be finite and strictly positive, got {v}"),
                })
            }
        }
        fn non_negative(field: &'static str, v: f64) -> Result<(), TshArmPolicyError> {
            if v.is_finite() && v >= 0.0 {
                Ok(())
            } else {
                Err(TshArmPolicyError::InvalidValue {
                    field,
                    reason: format!("must be finite and non-negative, got {v}"),
                })
            }
        }
        positive("branch_tau_ps", self.branch_tau_ps.value)?;
        positive(
            "min_observation_cadence_ps",
            self.min_observation_cadence_ps.value,
        )?;
        positive("confidence_decay_ps", self.confidence_decay_ps.value)?;
        positive(
            "max_sentinel_interval_ps",
            self.max_sentinel_interval_ps.value,
        )?;
        non_negative("sentinel_hold_ps", self.sentinel_hold_ps.value)?;
        non_negative("min_gear_dwell_ps", self.min_gear_dwell_ps.value)?;
        non_negative("decay_period_ps", self.decay_period_ps.value)?;
        non_negative("withdrawal_hold_ps", self.withdrawal_hold_ps.value)?;
        non_negative("rechallenge_window_ps", self.rechallenge_window_ps.value)?;
        non_negative("reinforce_dwell_ps", self.reinforce_dwell_ps.value)?;

        let conf = self.min_confidence_for_coarsening.value;
        if !(conf.is_finite() && (0.0..=1.0).contains(&conf)) {
            return Err(TshArmPolicyError::InvalidValue {
                field: "min_confidence_for_coarsening",
                reason: format!("must lie in [0,1], got {conf}"),
            });
        }

        let up = self.salience_upshift_threshold.value;
        let down = self.salience_downshift_threshold.value;
        let reinforce = self.salience_reinforce_threshold.value;
        for (field, v) in [
            ("salience_upshift_threshold", up),
            ("salience_downshift_threshold", down),
            ("salience_reinforce_threshold", reinforce),
        ] {
            if !(v.is_finite() && (0.0..=1.0).contains(&v)) {
                return Err(TshArmPolicyError::InvalidValue {
                    field,
                    reason: format!("must lie in [0,1], got {v}"),
                });
            }
        }
        // The hysteresis band must be non-empty and correctly ordered, or the
        // arm chatters: with up >= down there is no interval in which the arm
        // holds, and every noise excursion flips a gear.
        if !(up < down) {
            return Err(TshArmPolicyError::InvalidValue {
                field: "salience_upshift_threshold",
                reason: format!(
                    "hysteresis band is empty: upshift {up} must be strictly below downshift {down}"
                ),
            });
        }
        if !(down <= reinforce) {
            return Err(TshArmPolicyError::InvalidValue {
                field: "salience_reinforce_threshold",
                reason: format!("reinforce {reinforce} must be at or above downshift {down}"),
            });
        }

        for (i, w) in self.branch_weight.iter().enumerate() {
            if !(w.is_finite() && *w >= 0.0) {
                return Err(TshArmPolicyError::InvalidValue {
                    field: "branch_weight",
                    reason: format!("branch {i} weight must be finite and non-negative, got {w}"),
                });
            }
        }
        if self.branch_weight.iter().sum::<f64>() <= 0.0 {
            return Err(TshArmPolicyError::InvalidValue {
                field: "branch_weight",
                reason: "at least one branch weight must be positive".to_string(),
            });
        }
        for (i, r) in self.branch_reference_rate.iter().enumerate() {
            if !(r.is_finite() && *r > 0.0) {
                return Err(TshArmPolicyError::InvalidValue {
                    field: "branch_reference_rate",
                    reason: format!("branch {i} reference rate must be strictly positive, got {r}"),
                });
            }
        }
        // ASC / FORCE EXCLUSIVITY, enforced rather than documented.
        //
        // `{stem}_streamNN_asc_vectors.bin` is the `host_forces` buffer under
        // another filename. Giving the signal-grid branch a positive weight
        // while the force branch also carries weight, WITHOUT declaring an
        // independent grid observable, would count the force observation twice
        // and bias integrated salience toward whatever the force buffer says —
        // with nothing in the record disclosing it. A weight is only a weight
        // if the thing it weighs is a distinct observation.
        let grid = SalienceBranch::SignalGridWarp.index();
        let force = SalienceBranch::ForceEnergyStability.index();
        if self.branch_weight[grid] > 0.0
            && !self.signal_grid_warp_independent
            && self.branch_weight[force] > 0.0
        {
            return Err(TshArmPolicyError::InvalidValue {
                field: "branch_weight",
                reason: format!(
                    "branch {grid} (signal_grid_warp) has weight {} alongside force branch {force} \
                     (weight {}) without signal_grid_warp_independent: the engine's only 'ASC' \
                     product is the force buffer under another filename, so this double-counts \
                     force. Set signal_grid_warp_independent once a genuinely independent grid \
                     observable exists, or leave the weight at 0.0.",
                    self.branch_weight[grid], self.branch_weight[force]
                ),
            });
        }
        // Gear 3 is the abort sentinel and carries no timestep; gears 0..2 must
        // be positive and strictly increasing, or "one gear coarser" is not a
        // coarser timestep.
        for g in 0..3usize {
            if !(self.gear_dt_fs[g].is_finite() && self.gear_dt_fs[g] > 0.0) {
                return Err(TshArmPolicyError::InvalidValue {
                    field: "gear_dt_fs",
                    reason: format!("gear {g} timestep must be strictly positive"),
                });
            }
            if g > 0 && !(self.gear_dt_fs[g] > self.gear_dt_fs[g - 1]) {
                return Err(TshArmPolicyError::InvalidValue {
                    field: "gear_dt_fs",
                    reason: format!("gear {g} timestep must exceed gear {}", g - 1),
                });
            }
            // The emitted `ThermodynamicActionField::DeltaTimeFs` token
            // quantizes against a fixed bound. A gear timestep above it would
            // saturate to the escape code, i.e. the run's dt would be
            // unrecoverable from the token stream.
            if self.gear_dt_fs[g] > TSH_ARM_MAX_DT_FS_BOUND {
                return Err(TshArmPolicyError::InvalidValue {
                    field: "gear_dt_fs",
                    reason: format!(
                        "gear {g} timestep {} fs exceeds the codec dt bound {TSH_ARM_MAX_DT_FS_BOUND} fs",
                        self.gear_dt_fs[g]
                    ),
                });
            }
        }
        Ok(())
    }

    fn compute_digest(&self) -> u64 {
        let mut h = Sha256::new();
        h.update(b"PRISM_TSH_GEARBOX_ARM_POLICY_V1");
        h.update(TSH_ARM_SCHEMA_VERSION.to_le_bytes());
        h.update([u8::from(self.armed)]);
        for r in self.scalar_fields() {
            h.update(r.key.as_bytes());
            h.update(r.value.to_le_bytes());
            h.update(r.origin.as_str().as_bytes());
        }
        for v in self.branch_weight.iter() {
            h.update(v.to_le_bytes());
        }
        for v in self.branch_reference_rate.iter() {
            h.update(v.to_le_bytes());
        }
        h.update(self.branch_arrays_origin.as_str().as_bytes());
        h.update([u8::from(self.signal_grid_warp_independent)]);
        h.update(self.signal_grid_warp_origin.as_str().as_bytes());
        for v in self.gear_dt_fs.iter() {
            h.update(v.to_le_bytes());
        }
        h.update(self.gear_dt_origin.as_str().as_bytes());
        match &self.policy_file_sha256 {
            Some(s) => h.update(s.as_bytes()),
            None => h.update(b"no_policy_file"),
        }
        let d = h.finalize();
        u64::from_le_bytes(d[0..8].try_into().expect("sha256 yields 32 bytes"))
    }

    /// Every resolved scalar, in fixed order. Fixed order matters: the digest
    /// is order-sensitive by construction.
    pub fn scalar_fields(&self) -> [ResolvedValue; 14] {
        [
            self.branch_tau_ps,
            self.min_observation_cadence_ps,
            self.confidence_decay_ps,
            self.min_confidence_for_coarsening,
            self.max_sentinel_interval_ps,
            self.sentinel_hold_ps,
            self.salience_downshift_threshold,
            self.salience_upshift_threshold,
            self.salience_reinforce_threshold,
            self.min_gear_dwell_ps,
            self.decay_period_ps,
            self.withdrawal_hold_ps,
            self.rechallenge_window_ps,
            self.reinforce_dwell_ps,
        ]
    }

    /// The fingerprint binding a request to this exact policy.
    pub fn digest(&self) -> u64 {
        self.policy_digest
    }
}

/// Resolve the arm policy from the environment.
///
/// Order, per key: the policy document if `PRISM_TSH_ARM_POLICY_PATH` is set,
/// then the per-key environment override, else the documented source literal.
/// A present-but-unusable environment override is REJECTED and logged, never
/// honoured.
///
/// Absence of every key resolves to the documented defaults, ARMED — that is a
/// legal recorded configuration. To leave the arm without an opinion at all,
/// use [`TshArmPolicy::unarmed_defaults`] instead.
pub fn resolve_arm_policy() -> Result<TshArmPolicy, TshArmPolicyError> {
    let (doc, file_origin, sha) = match std::env::var(POLICY_PATH_ENV) {
        Ok(raw) if !raw.trim().is_empty() => {
            let path = PathBuf::from(raw.trim());
            let (doc, sha) = read_policy_document(&path)?;
            (doc, ValueOrigin::PolicyFile, Some(sha))
        }
        _ => (
            TshArmPolicyDocument::default(),
            ValueOrigin::PolicyFile,
            None,
        ),
    };

    let mut policy = TshArmPolicy::from_document(&doc, file_origin, sha)?;

    // Per-key environment overrides sit above the document.
    let overrides: [(&'static str, &mut ResolvedValue); 14] = [
        (ENV_BRANCH_TAU_PS, &mut policy.branch_tau_ps),
        (
            ENV_MIN_OBSERVATION_CADENCE_PS,
            &mut policy.min_observation_cadence_ps,
        ),
        (ENV_CONFIDENCE_DECAY_PS, &mut policy.confidence_decay_ps),
        (
            ENV_MIN_CONFIDENCE_FOR_COARSENING,
            &mut policy.min_confidence_for_coarsening,
        ),
        (
            ENV_MAX_SENTINEL_INTERVAL_PS,
            &mut policy.max_sentinel_interval_ps,
        ),
        (ENV_SENTINEL_HOLD_PS, &mut policy.sentinel_hold_ps),
        (
            ENV_SALIENCE_DOWNSHIFT,
            &mut policy.salience_downshift_threshold,
        ),
        (ENV_SALIENCE_UPSHIFT, &mut policy.salience_upshift_threshold),
        (
            ENV_SALIENCE_REINFORCE,
            &mut policy.salience_reinforce_threshold,
        ),
        (ENV_MIN_GEAR_DWELL_PS, &mut policy.min_gear_dwell_ps),
        (ENV_DECAY_PERIOD_PS, &mut policy.decay_period_ps),
        (ENV_WITHDRAWAL_HOLD_PS, &mut policy.withdrawal_hold_ps),
        (ENV_RECHALLENGE_WINDOW_PS, &mut policy.rechallenge_window_ps),
        (ENV_REINFORCE_DWELL_PS, &mut policy.reinforce_dwell_ps),
    ];
    for (key, slot) in overrides {
        if let Ok(raw) = std::env::var(key) {
            match raw.trim().parse::<f64>() {
                Ok(v) if v.is_finite() => {
                    slot.value = v;
                    slot.origin = ValueOrigin::PolicyEnv;
                }
                _ => {
                    log::warn!(
                        "[TSH-ARM] {key}='{raw}' is not a usable value; using {} from {}",
                        slot.value,
                        slot.origin.as_str()
                    );
                    slot.origin = ValueOrigin::PolicyEnvRejected;
                }
            }
        }
    }

    // The signal-grid independence declaration is a boolean, not a scalar, so
    // it does not travel through the numeric override table. A present but
    // unparseable value is REJECTED and recorded, never read as "true": the
    // failure direction has to be "keep refusing ASC-as-independent".
    if let Ok(raw) = std::env::var(SIGNAL_GRID_INDEPENDENT_ENV) {
        match raw.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => {
                policy.signal_grid_warp_independent = true;
                policy.signal_grid_warp_origin = ValueOrigin::PolicyEnv;
            }
            "0" | "false" | "no" | "off" => {
                policy.signal_grid_warp_independent = false;
                policy.signal_grid_warp_origin = ValueOrigin::PolicyEnv;
            }
            _ => {
                log::warn!(
                    "[TSH-ARM] {SIGNAL_GRID_INDEPENDENT_ENV}='{raw}' is not a usable boolean; \
                     keeping {} from {}",
                    policy.signal_grid_warp_independent,
                    policy.signal_grid_warp_origin.as_str()
                );
                policy.signal_grid_warp_origin = ValueOrigin::PolicyEnvRejected;
            }
        }
    }

    policy.validate()?;
    policy.policy_digest = policy.compute_digest();
    Ok(policy)
}

fn read_policy_document(path: &Path) -> Result<(TshArmPolicyDocument, String), TshArmPolicyError> {
    let bytes = std::fs::read(path).map_err(|e| TshArmPolicyError::FileUnreadable {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if let Ok(expected) = std::env::var(POLICY_SHA256_ENV) {
        let expected = expected.trim().to_ascii_lowercase();
        if !expected.is_empty() && expected != actual {
            return Err(TshArmPolicyError::Sha256Mismatch {
                path: path.to_path_buf(),
                expected,
                actual,
            });
        }
    }
    let doc: TshArmPolicyDocument =
        serde_json::from_slice(&bytes).map_err(|e| TshArmPolicyError::FileMalformed {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
    Ok((doc, actual))
}

// ═══════════════════════════════════════════════════════════════════════════
// Arm state
// ═══════════════════════════════════════════════════════════════════════════

/// Per-branch persistent state. Mirrors `PrismTshArmBranchState`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BranchState {
    /// Leaky-integrated activity. Rate = `activity / tau`.
    pub activity: f64,
    /// Simulated-time coordinate of the newest evidence ever staged.
    /// `None` while the branch has never reported.
    ///
    /// An explicit `None` rather than a `-inf` sentinel: infinities do not
    /// survive a JSON checkpoint round trip (`serde_json` writes them as
    /// `null` and then refuses to read them back as `f64`), and a checkpoint
    /// that silently loses the evidence clock cannot reproduce the salience
    /// trajectory it exists to reproduce.
    pub last_evidence_ps: Option<f64>,
    /// Staged activity for the current decision window.
    pub pending_activity: f64,
    /// Staged simulated span for the current decision window.
    pub pending_span_ps: f64,
    /// True once the branch has ever delivered evidence. False means UNBOUND,
    /// which is a different fact from "reported calm".
    pub ever_observed: bool,
    /// Samples staged in the current window. PROVENANCE ONLY.
    pub pending_samples: u32,
    /// Samples ever staged. PROVENANCE ONLY.
    pub total_samples: u64,
    /// Leaky-integrated observation COVERAGE, in the same units as
    /// [`Self::activity`] with a unit source. `coverage_activity / tau` is the
    /// fraction of recent simulated time that staged evidence actually
    /// covered, in `[0,1]`.
    ///
    /// This is the quantity that makes anti-blindness (c) a measurement rather
    /// than a claim. The integrator is driven by `1` while evidence covers the
    /// interval and by `0` while it does not, over the FULL decision interval —
    /// unlike [`Self::activity`], which is frozen over uncovered time because
    /// there is no evidence about what the physics did there. Coverage is not
    /// frozen precisely because the absence itself is the thing being measured.
    pub coverage_activity: f64,
}

impl BranchState {
    fn unbound() -> Self {
        Self {
            activity: 0.0,
            last_evidence_ps: None,
            pending_activity: 0.0,
            pending_span_ps: 0.0,
            ever_observed: false,
            pending_samples: 0,
            total_samples: 0,
            coverage_activity: 0.0,
        }
    }

    /// Consume `covered` picoseconds of the staged window, CARRYING FORWARD the
    /// remainder at the same rate, up to `backlog_limit_ps`.
    ///
    /// The first revision discarded the surplus, which meant an observation
    /// cadence coarser than the decision cadence threw away
    /// `pending_span_ps - covered` picoseconds of real evidence every decision
    /// and read as quiet. Carrying the remainder forward at the same rate `r`
    /// makes the integrated evidence identical to the same physics reported at
    /// the decision cadence.
    ///
    /// # Why the carry is BOUNDED
    ///
    /// An unbounded carry is its own hazard: a producer staging spans far
    /// longer than the decision interval would grow a backlog without limit and
    /// the arm would integrate the distant past as if it were the present,
    /// falling arbitrarily far behind the run with nothing saying so. The bound
    /// is the staleness horizon — evidence older than the arm is allowed to
    /// CONSIDER is not evidence it should be queueing. Truncation drops the
    /// OLDEST surplus (span shrinks, rate is preserved) and is reported through
    /// [`FLAG_EVIDENCE_BACKLOG_TRUNCATED`], so the loss is bounded, named, and
    /// never silent.
    ///
    /// Returns whether the backlog was truncated.
    fn consume_pending(&mut self, covered: f64, backlog_limit_ps: f64) -> bool {
        if !(self.pending_span_ps > 0.0) {
            self.pending_activity = 0.0;
            self.pending_span_ps = 0.0;
            self.pending_samples = 0;
            return false;
        }
        let rate = self.pending_activity / self.pending_span_ps;
        let remaining = self.pending_span_ps - covered;
        if remaining > 0.0 && remaining.is_finite() {
            let mut keep = remaining;
            let mut truncated = false;
            if backlog_limit_ps.is_finite() && backlog_limit_ps > 0.0 && keep > backlog_limit_ps {
                keep = backlog_limit_ps;
                truncated = true;
            }
            self.pending_span_ps = keep;
            self.pending_activity = rate * keep;
            // Samples are provenance for the window that was *staged*; once a
            // window is partly consumed the count belongs to the carried
            // remainder, so it is retained rather than zeroed. It is never a
            // divisor, so retention cannot bias a rate.
            truncated
        } else {
            self.pending_activity = 0.0;
            self.pending_span_ps = 0.0;
            self.pending_samples = 0;
            false
        }
    }
}

/// The path-dependent trajectory state.
///
/// Semantically equivalent to `PrismTshArmState` in
/// `cuda/tsh_gearbox_arm.cuh`, with one deliberate representational
/// difference: the device struct encodes "has not happened yet" as `±INFINITY`
/// in a `double`, while this struct uses `Option<f64>`. The device needs a
/// fixed-size POD with no tag; the host needs a form that survives a JSON
/// checkpoint, and `serde_json` writes an infinity as `null` and then refuses
/// to read it back as `f64`. A checkpoint that silently lost those clocks
/// could not reproduce the trajectory it exists to reproduce, so the two
/// representations differ on purpose and a bridge must map `None` <-> the
/// device's infinity, never `None` -> `0.0`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TshArmState {
    /// Monotonic. Increments on EVERY decision, including no-change ones.
    pub record_sequence: u64,
    /// Current memory state.
    pub memory_state: TshMemoryState,
    /// The gear the arm last REQUESTED. Not what ran.
    pub last_requested_gear: u32,
    /// Simulated-time coordinate of the last decision.
    pub now_ps: f64,
    /// Simulated time at which the current memory state was entered.
    pub memory_entered_ps: f64,
    /// Simulated time of the last MEASURED fine-gear sample.
    pub last_sentinel_ps: f64,
    /// Simulated time at which the requested gear last changed.
    pub request_changed_ps: f64,
    /// Simulated time at which active memory was last left. `None` until it
    /// has been.
    pub active_memory_exit_ps: Option<f64>,
    /// Simulated time since which salience has been at/above the reinforce
    /// bound. `None` when it is not.
    pub reinforce_since_ps: Option<f64>,
    /// Simulated time at which the current sentinel hold ends. `None` when no
    /// hold is active.
    pub sentinel_hold_until_ps: Option<f64>,
    /// True while a rechallenge window is open.
    pub rechallenge_armed: bool,
    /// Decisions taken since reset. Provenance only.
    pub decisions: u64,
    /// Fixed-order branch states.
    pub branch: [BranchState; TSH_ARM_N_BRANCHES],
    /// Records a downstream ring refused. Mirrors
    /// `PrismTshArmState.dropped_records`. While non-zero every emitted record
    /// carries [`FLAG_RING_DROPPED`], so a truncated stream cannot be mistaken
    /// for a complete one.
    pub dropped_records: u64,
    /// Observations refused by the ASC / force exclusivity rule. Mirrors
    /// `PrismTshArmState.signal_grid_refusals`.
    #[serde(default)]
    pub signal_grid_refusals: u64,
}

impl TshArmState {
    fn reset(t0_ps: f64) -> Self {
        Self {
            record_sequence: 0,
            memory_state: TshMemoryState::Baseline,
            last_requested_gear: GEAR_UNSET,
            now_ps: t0_ps,
            memory_entered_ps: t0_ps,
            last_sentinel_ps: t0_ps,
            request_changed_ps: t0_ps,
            active_memory_exit_ps: None,
            reinforce_since_ps: None,
            sentinel_hold_until_ps: None,
            rechallenge_armed: false,
            decisions: 0,
            branch: [BranchState::unbound(); TSH_ARM_N_BRANCHES],
            dropped_records: 0,
            signal_grid_refusals: 0,
        }
    }
}

/// The clock and reality inputs for one decision.
#[derive(Debug, Clone, Copy)]
pub struct DecisionClock {
    /// `cumulative_simulated_time_ps` — the time base once the gear varies.
    pub cumulative_simulated_time: SimulatedPs,
    /// Global MD step this decision belongs to.
    pub global_md_step: u64,
    /// Frame index, stamped into the request so the governor can bound
    /// staleness.
    pub frame: u32,
    /// What the governor last AUTHORIZED, if the caller has a measurement.
    /// `None` is UNMEASURED: it does not reset the sentinel clock and it is
    /// never collapsed into a gear value.
    pub authorized_gear: Option<u32>,
}

// ═══════════════════════════════════════════════════════════════════════════
// The emitted request
// ═══════════════════════════════════════════════════════════════════════════

/// serde bridge for the codec-registered opcode.
///
/// `prism_tokenizer`'s [`ThermodynamicActionOpcode`] is `#[repr(u64)]` and
/// deliberately carries no serde derives — its wire identity is the token
/// field value, not a Rust enum name. This bridge writes exactly that value and
/// refuses an unregistered one on the way back in, so a checkpoint or an
/// evidence file cannot reintroduce an opcode this build does not know.
///
/// [`ThermodynamicActionOpcode`]: prism_tokenizer::types::ThermodynamicActionOpcode
pub mod thermodynamic_action_serde {
    use prism_tokenizer::types::ThermodynamicActionOpcode as Op;
    use serde::{Deserialize, Deserializer, Serializer};

    /// The registered opcode for a wire value, or `None` when unregistered.
    pub fn from_u64(v: u64) -> Option<Op> {
        Some(match v {
            1 => Op::Stable,
            2 => Op::SalienceRise,
            3 => Op::SalienceFall,
            4 => Op::Hysteresis,
            5 => Op::Boundary,
            6 => Op::Handoff,
            _ => return None,
        })
    }

    /// Stable machine-readable label for evidence records.
    pub fn as_str(op: Op) -> &'static str {
        match op {
            Op::Stable => "stable",
            Op::SalienceRise => "salience_rise",
            Op::SalienceFall => "salience_fall",
            Op::Hysteresis => "hysteresis",
            Op::Boundary => "boundary",
            Op::Handoff => "handoff",
        }
    }

    pub(crate) fn serialize<S: Serializer>(op: &Op, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(*op as u64)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Op, D::Error> {
        let v = u64::deserialize(d)?;
        from_u64(v).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "TSH_ARM_UNREGISTERED_THERMODYNAMIC_ACTION_OPCODE value={v}"
            ))
        })
    }
}

// Every registered opcode must round-trip through the wire bridge, or a value
// this build emits could be one a reader of the same build refuses.
const _: () = {
    use prism_tokenizer::types::ThermodynamicActionOpcode as Op;
    assert!(Op::Stable as u64 == 1);
    assert!(Op::SalienceRise as u64 == 2);
    assert!(Op::SalienceFall as u64 == 3);
    assert!(Op::Hysteresis as u64 == 4);
    assert!(Op::Boundary as u64 == 5);
    assert!(Op::Handoff as u64 == 6);
};

/// One decision's emitted request. Produced on EVERY decision, including
/// no-change ones: a gap in this stream must never be ambiguous between "gear
/// held" and "decision did not run", and `record_sequence` is what makes the
/// difference detectable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TshGearRequest {
    /// [`TSH_ARM_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Monotonic decision counter. A gap means a decision did not run.
    pub record_sequence: u64,
    /// Global MD step.
    pub global_md_step: u64,
    /// Cumulative simulated time, ps.
    pub cumulative_simulated_time_ps: f64,
    /// The gear DEMANDED. `None` when the arm is unarmed and has no opinion.
    /// Never `GEAR_ABORT`.
    pub requested_gear: Option<u32>,
    /// dt in fs implied by `requested_gear` under the resolved policy.
    pub requested_dt_fs: Option<f64>,
    /// Why.
    pub request_reason: RequestReason,
    /// Confidence in **this request**, in `[0,1]`.
    ///
    /// Distinct from [`Self::observation_confidence`]. A request forced by the
    /// sentinel or by a rechallenge is structurally mandated and therefore
    /// fully confident even when the evidence is thin; a request that exists
    /// only because there was no evidence has confidence `0.0` — it is a
    /// fail-safe default, not an inference. Otherwise it is the observation
    /// confidence attenuated by how close the salience sits to the threshold
    /// that would have decided differently.
    pub request_confidence: Option<f64>,
    /// Anti-blindness (d) gate input: weighted freshness of the evidence, in
    /// `[0,1]`. This is the quantity compared against
    /// [`TshArmPolicy::min_confidence_for_coarsening`], so it travels with the
    /// record or the gate's verdict would be unauditable.
    pub observation_confidence: Option<f64>,
    /// Confidence-weighted mean branch salience in `[0,1]`. `None` means
    /// UNMEASURED — no branch had admissible evidence. It is never `Some(0.0)`
    /// as a stand-in for that.
    ///
    /// **This is a REPORTED STATISTIC and gates nothing.** It used to be the
    /// sole input to the fine-ward demand, which made the arm a weighted score
    /// in violation of the lexicographic rule: one branch at salience `0.999`
    /// with five calm branches averages to `0.1665`, below the upshift
    /// threshold, so a maximal single-family demand produced a request for a
    /// COARSER gear — and wiring up more families made the arm blinder to each
    /// one. [`Self::peak_branch_salience`] is the gate input.
    pub integrated_salience: Option<f64>,
    /// The **maximum** salience over admissible branches — the quantity every
    /// resolution gate is actually taken on.
    ///
    /// A branch is admissible when it carries positive weight, has reported at
    /// least once, and has non-zero confidence. `None` when no branch is
    /// admissible, which is UNMEASURED and drives the arm finest.
    ///
    /// Travels on every record because a gate input that does not travel makes
    /// the verdict unauditable.
    pub peak_branch_salience: Option<f64>,
    /// Which branch produced [`Self::peak_branch_salience`]. `None` when that
    /// is `None`.
    pub peak_branch: Option<SalienceBranch>,
    /// Weighted observation COVERAGE in `[0,1]`: the fraction of recent
    /// simulated time that staged evidence actually covered, weighted the same
    /// way as [`Self::observation_confidence`]. `None` when no branch carries
    /// weight.
    ///
    /// This is the anti-blindness (c) measurement. It is reported separately
    /// from the confidence it feeds so a reader can tell a branch that reported
    /// late from one that reported punctually but thinly.
    pub observation_coverage: Option<f64>,
    /// The codec-registered thermodynamic action this decision publishes.
    ///
    /// See [`RequestReason::thermodynamic_action`]. Emitted on EVERY decision,
    /// which is what gives the registered opcode a producer.
    #[serde(with = "thermodynamic_action_serde")]
    pub thermodynamic_action: prism_tokenizer::types::ThermodynamicActionOpcode,
    /// The confidence-weighted mean branch RATE, per simulated picosecond.
    /// Typed so it cannot be re-derived by dividing by a sample count.
    pub integrated_rate: Option<PerSimulatedPs<f64>>,
    /// Fresh / salient / stale / unbound branch masks.
    pub active_branch_mask: BranchMask,
    /// Memory state after this decision.
    pub memory_state: TshMemoryState,
    /// Simulated ps spent in the current memory state.
    pub memory_age_ps: f64,
    /// Echo of the authorized gear the arm was told about. `None` is
    /// UNMEASURED, never collapsed into a gear.
    pub authorized_gear_echo: Option<u32>,
    /// `FLAG_*` bits.
    pub flags: u32,
    /// Binds this request to the exact policy that produced it.
    pub policy_digest: u64,
}

impl TshGearRequest {
    /// Whether anti-blindness (b) forced this request.
    pub fn sentinel_forced(&self) -> bool {
        self.flags & FLAG_SENTINEL_FORCED != 0
    }

    /// Whether anti-blindness (d) refused a coarsening this decision.
    pub fn coarsening_blocked_by_confidence(&self) -> bool {
        self.flags & FLAG_CONFIDENCE_BLOCKED != 0
    }

    /// The typed insertion point the governor reads.
    ///
    /// The governor treats this as a CEILING ONLY: it can make the authorized
    /// gear finer, never coarser. Nothing here can obtain a gear the physical
    /// gates refused.
    #[cfg(feature = "gpu")]
    pub fn to_salience_request(&self, frame: u32) -> crate::gearbox::TshSalienceRequest {
        use crate::gearbox as gb;
        match self.requested_gear {
            None => gb::TshSalienceRequest::unset(),
            Some(gear) => {
                // The PEAK, not the mean: the governor is being handed the
                // quantity this arm's own gate was taken on. Publishing the
                // mean here would show the governor a number that never
                // decided anything.
                let salience = self.peak_branch_salience.map_or(f32::NAN, |s| s as f32);
                // `request` refuses anything above GEAR_MAX_ACTIVE, so an
                // abort can never be encoded. If it ever refused, publish
                // UNSET rather than a coerced gear.
                let mut req = match gb::TshSalienceRequest::request(gear, salience, frame) {
                    Some(r) => r,
                    None => gb::TshSalienceRequest::unset(),
                };
                req.priority = self.request_reason as u32;
                req
            }
        }
    }

    /// The compact 64-byte device-ring record.
    pub fn to_abi(&self) -> TshArmRecordAbi {
        TshArmRecordAbi {
            schema_version: self.schema_version,
            // The ABI slot is u32 to match the device struct; the host counter
            // is u64. A reader that spans more than 2^32 decisions must
            // reconstruct the high word from the host record.
            record_sequence: self.record_sequence as u32,
            global_md_step: self.global_md_step,
            cumulative_ps: self.cumulative_simulated_time_ps,
            integrated_salience: self.integrated_salience.map_or(f32::NAN, |v| v as f32),
            request_confidence: self.request_confidence.map_or(f32::NAN, |v| v as f32),
            requested_gear: self.requested_gear.unwrap_or(GEAR_UNSET),
            requested_dt_fs: self.requested_dt_fs.map_or(f32::NAN, |v| v as f32),
            request_reason: self.request_reason as u32,
            active_branch_mask: self.active_branch_mask.to_packed_u32(),
            memory_state: self.memory_state as u32,
            memory_age_ps: self.memory_age_ps as f32,
            authorized_gear_echo: self.authorized_gear_echo.unwrap_or(UNMEASURED_U32),
            flags: self.flags,
            observation_confidence: self.observation_confidence.map_or(f32::NAN, |v| v as f32),
            integrated_rate_per_ps: self.integrated_rate.map_or(f32::NAN, |r| r.get() as f32),
            policy_digest: self.policy_digest,
            peak_branch_salience: self.peak_branch_salience.map_or(f32::NAN, |v| v as f32),
            observation_coverage: self.observation_coverage.map_or(f32::NAN, |v| v as f32),
            thermodynamic_action: self.thermodynamic_action as u64 as u32,
            peak_branch_index: self
                .peak_branch
                .map_or(UNMEASURED_U32, |b| b.index() as u32),
        }
    }

    /// The codec-registered tokens this decision publishes.
    ///
    /// Returns the `ThermodynamicAction` family's two field tokens: the TSH
    /// state opcode (`ThermodynamicActionField::TshState`) and the requested
    /// timestep (`ThermodynamicActionField::DeltaTimeFs`). `None` for the dt
    /// token when the arm has no opinion — there is no token for a timestep
    /// nobody asked for.
    ///
    /// This method is the producer the registered opcode never had. Every
    /// decision, including a no-change one, yields one.
    ///
    /// `verified` is true only when the decision rested on MEASURED evidence:
    /// a request produced by [`RequestReason::EvidenceAbsent`] or
    /// [`RequestReason::PolicyUnarmed`] is a fail-safe default, not an
    /// inference, and marking it verified would promote a default to evidence.
    /// `provenance` is always true — every request carries
    /// [`Self::policy_digest`], which binds it to the exact resolved thresholds
    /// and their origins.
    pub fn thermodynamic_action_tokens(
        &self,
    ) -> (prism_tokenizer::Token, Option<prism_tokenizer::Token>) {
        use prism_tokenizer::types::{
            native_thermodynamic_action_opcode, native_thermodynamic_dt_token, ArtifactFamily,
        };
        let tier = prism_tokenizer::leverage_tier(ArtifactFamily::ThermodynamicAction);
        let verified = self.integrated_salience.is_some()
            && !matches!(
                self.request_reason,
                RequestReason::EvidenceAbsent | RequestReason::PolicyUnarmed | RequestReason::Unset
            );
        let opcode =
            native_thermodynamic_action_opcode(self.thermodynamic_action, tier, verified, true);
        // The dt bound is the coarsest gear's timestep in the resolved policy,
        // carried on the request itself rather than re-derived, so the
        // quantizer's range is the range the run actually had.
        let dt = self.requested_dt_fs.map(|dt_fs| {
            native_thermodynamic_dt_token(dt_fs, TSH_ARM_MAX_DT_FS_BOUND, tier, verified, true)
        });
        (opcode, dt)
    }

    /// Stamp this request's TSH-demand block into a Coupled Chronology Record.
    ///
    /// The producer stamps its own provenance inside the fence that publishes
    /// it. A later host-side join cannot recover information the record never
    /// carried — that is exactly what failed in the prior campaign, where a
    /// hardcoded `gear_id: 0` was indistinguishable from a measured gear 0 and
    /// silently asserted 4x finer resolution than the run actually had.
    ///
    /// Only the TSH-demand fields are written. `authorized_gear` and
    /// `device_measured_gear` belong to the governor and the device readback
    /// respectively, and this method deliberately does not touch them: the
    /// three gear facts must never collapse.
    ///
    /// Fields this arm did not measure are left at the record's UNMEASURED
    /// sentinels rather than being filled with a plausible zero.
    ///
    /// `tsh_integrated_salience_per_ps` receives the confidence-weighted mean
    /// RATE, which is what that field is defined to carry. The chronology
    /// record has no slot for [`Self::peak_branch_salience`] — the quantity the
    /// gates were actually taken on — so a reader reconstructing WHY a gear was
    /// requested must join against the arm record
    /// ([`TshArmRecordAbi::peak_branch_salience`]), which carries it. Recording
    /// the mean in a slot named for the demand would be the collapse this
    /// module exists to prevent, so the two are kept distinguishable rather
    /// than one being written into the other's slot.
    pub fn stamp_chronology_demand(
        &self,
        record: &mut crate::coupled_chronology::CoupledChronologyRecord,
    ) {
        use crate::coupled_chronology as ccr;
        record.global_md_step = self.global_md_step;
        record.cumulative_simulated_time_ps = self.cumulative_simulated_time_ps;
        record.tsh_integrated_salience_per_ps = self
            .integrated_rate
            .map_or(ccr::CCR_F64_UNMEASURED, |r| r.get());
        record.tsh_memory_state = self.memory_state as u32 as f64;
        record.tsh_request_confidence = self.request_confidence.unwrap_or(ccr::CCR_F64_UNMEASURED);
        record.observation_confidence = self
            .observation_confidence
            .unwrap_or(ccr::CCR_F64_UNMEASURED);
        record.requested_gear = self.requested_gear.unwrap_or(ccr::CCR_GEAR_UNMEASURED);
        record.tsh_active_branches = self.active_branch_mask.to_packed_u32();
    }
}

/// The compact device-ring record. Mirrors `PrismTshArmRecord` exactly.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TshArmRecordAbi {
    /// [`TSH_ARM_SCHEMA_VERSION`].                                    @ 0
    pub schema_version: u32,
    /// Low 32 bits of the monotonic decision counter.                 @ 4
    pub record_sequence: u32,
    /// Global MD step.                                                @ 8
    pub global_md_step: u64,
    /// Cumulative simulated time, ps.                                 @ 16
    pub cumulative_ps: f64,
    /// Integrated salience, or NaN when UNMEASURED.                   @ 24
    pub integrated_salience: f32,
    /// Observation confidence, or NaN when UNMEASURED.                @ 28
    pub request_confidence: f32,
    /// Requested gear, or [`GEAR_UNSET`].                             @ 32
    pub requested_gear: u32,
    /// dt in fs, or NaN.                                              @ 36
    pub requested_dt_fs: f32,
    /// `RequestReason as u32`.                                        @ 40
    pub request_reason: u32,
    /// Packed branch masks.                                           @ 44
    pub active_branch_mask: u32,
    /// `TshMemoryState as u32`.                                       @ 48
    pub memory_state: u32,
    /// Simulated ps in the current memory state.                      @ 52
    pub memory_age_ps: f32,
    /// Authorized gear echo, or [`UNMEASURED_U32`].                   @ 56
    pub authorized_gear_echo: u32,
    /// `FLAG_*` bits.                                                 @ 60
    pub flags: u32,
    /// Anti-blindness (d) gate input, or NaN.                         @ 64
    pub observation_confidence: f32,
    /// Integrated salience as a rate per simulated ps, or NaN.        @ 68
    pub integrated_rate_per_ps: f32,
    /// Fingerprint of the policy that produced this request.          @ 72
    pub policy_digest: u64,
    /// PEAK admissible branch salience — the resolution gate input —  @ 80
    /// or NaN when UNMEASURED. Distinct from `integrated_salience`,
    /// which is the confidence-weighted mean and gates nothing.
    pub peak_branch_salience: f32,
    /// Weighted observation COVERAGE in `[0,1]`, or NaN.              @ 84
    /// Anti-blindness (c) as a measured quantity.
    pub observation_coverage: f32,
    /// Codec-registered `ThermodynamicActionOpcode` value.            @ 88
    /// `0` is not a registered opcode, so it doubles as "absent".
    pub thermodynamic_action: u32,
    /// Index of the branch that produced `peak_branch_salience`, or   @ 92
    /// [`UNMEASURED_U32`].
    pub peak_branch_index: u32,
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};
    assert!(size_of::<TshArmRecordAbi>() == 96);
    assert!(align_of::<TshArmRecordAbi>() == 16);
    assert!(offset_of!(TshArmRecordAbi, cumulative_ps) == 16);
    assert!(offset_of!(TshArmRecordAbi, flags) == 60);
    assert!(offset_of!(TshArmRecordAbi, policy_digest) == 72);
    assert!(offset_of!(TshArmRecordAbi, peak_branch_salience) == 80);
    assert!(offset_of!(TshArmRecordAbi, thermodynamic_action) == 88);
};

impl TshArmRecordAbi {
    /// The value an UNWRITTEN ring slot must hold.
    ///
    /// A zero-filled slot decodes as a plausible calm record — `memory_state`
    /// 0 is `Baseline`, `requested_gear` 0 is `GEAR_FINE`, `integrated_salience`
    /// 0.0 reads as measured calm, `request_reason` 0 is `Unset` — and only
    /// `schema_version` distinguishes it. That is the invariant-1 collision
    /// class, and the fix is to never let a slot be zero-filled: the allocator
    /// pre-fills with this, whose every measurement slot is NaN or an
    /// out-of-band sentinel.
    pub fn unwritten() -> Self {
        Self {
            schema_version: 0,
            record_sequence: 0,
            global_md_step: u64::MAX,
            cumulative_ps: f64::NAN,
            integrated_salience: f32::NAN,
            request_confidence: f32::NAN,
            requested_gear: GEAR_UNSET,
            requested_dt_fs: f32::NAN,
            request_reason: UNMEASURED_U32,
            active_branch_mask: 0,
            memory_state: UNMEASURED_U32,
            memory_age_ps: f32::NAN,
            authorized_gear_echo: UNMEASURED_U32,
            flags: 0,
            observation_confidence: f32::NAN,
            integrated_rate_per_ps: f32::NAN,
            policy_digest: 0,
            peak_branch_salience: f32::NAN,
            observation_coverage: f32::NAN,
            thermodynamic_action: 0,
            peak_branch_index: UNMEASURED_U32,
        }
    }

    /// Whether this slot holds a record an emitter actually wrote.
    ///
    /// Mirrors `prism_tsh_arm_record_is_valid` in the header. A drained or
    /// never-written slot fails this; a real record passes. Without it a reader
    /// cannot tell a gear-0 decision from an empty slot, and `gear 0` is a
    /// LEGITIMATE gear.
    pub fn is_valid(&self) -> bool {
        self.schema_version == TSH_ARM_SCHEMA_VERSION
            && self.request_reason != UNMEASURED_U32
            && self.memory_state != UNMEASURED_U32
    }

    /// The registered opcode this record carries, or `None` when absent.
    pub fn thermodynamic_action_opcode(
        &self,
    ) -> Option<prism_tokenizer::types::ThermodynamicActionOpcode> {
        thermodynamic_action_serde::from_u64(self.thermodynamic_action as u64)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Checkpoint
// ═══════════════════════════════════════════════════════════════════════════

/// A restorable snapshot of the salience trajectory.
///
/// Reproducing a run requires reproducing this: the arm's state is
/// path-dependent, so a seed alone is insufficient. A restore under a
/// different policy is refused rather than silently accepted, because the same
/// trajectory under different thresholds is a different controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TshArmCheckpoint {
    /// [`TSH_ARM_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Digest of the policy that produced this trajectory.
    pub policy_digest: u64,
    /// The trajectory state.
    pub state: TshArmState,
}

/// Why a restore was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointError {
    /// The checkpoint declares a schema this build cannot interpret.
    SchemaMismatch { found: u32, expected: u32 },
    /// The checkpoint was taken under a different policy.
    PolicyDigestMismatch { found: u64, expected: u64 },
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaMismatch { found, expected } => write!(
                f,
                "TSH_ARM_CHECKPOINT_SCHEMA_MISMATCH found={found} expected={expected}"
            ),
            Self::PolicyDigestMismatch { found, expected } => write!(
                f,
                "TSH_ARM_CHECKPOINT_POLICY_DIGEST_MISMATCH found={found:#018x} expected={expected:#018x}"
            ),
        }
    }
}

impl std::error::Error for CheckpointError {}

// ═══════════════════════════════════════════════════════════════════════════
// Host <-> device state bridge
// ═══════════════════════════════════════════════════════════════════════════
//
// The device struct is a fixed-size POD with no tag, so "has not happened yet"
// is `±INFINITY` in a `double`. The host struct uses `Option<f64>`, because
// `serde_json` writes an infinity as `null` and then refuses to read it back as
// an `f64` — a checkpoint that silently lost those clocks could not reproduce
// the trajectory it exists to reproduce.
//
// The two representations therefore differ ON PURPOSE, and this is the bridge
// the module doc promised. Before it existed, the captured-graph path — the one
// the `.cuh` exists to serve — could not be checkpointed or restored at all,
// so checkpointing covered only the reference oracle and not the side that runs.

/// `+INFINITY` on the device: a clock whose event lies in the unreachable
/// future, i.e. "not yet armed".
const DEVICE_FUTURE_SENTINEL: f64 = f64::INFINITY;
/// `-INFINITY` on the device: a clock whose event lies in the unreachable
/// past, i.e. "has never happened".
const DEVICE_PAST_SENTINEL: f64 = f64::NEG_INFINITY;

#[inline]
fn to_device_past(v: Option<f64>) -> f64 {
    v.unwrap_or(DEVICE_PAST_SENTINEL)
}

#[inline]
fn to_device_future(v: Option<f64>) -> f64 {
    v.unwrap_or(DEVICE_FUTURE_SENTINEL)
}

#[inline]
fn from_device_clock(v: f64) -> Option<f64> {
    // NEVER `0.0`: zero is a legitimate simulated-time coordinate and mapping a
    // sentinel onto it would silently claim an event happened at t=0.
    v.is_finite().then_some(v)
}

/// Why a device-state decode was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceStateError {
    /// The buffer is not [`DEVICE_STATE_BYTES`] long.
    WrongLength { found: usize, expected: usize },
    /// The buffer declares a schema this build cannot interpret.
    SchemaMismatch { found: u32, expected: u32 },
    /// A memory-state word is not a registered [`TshMemoryState`].
    UnknownMemoryState { found: u32 },
}

impl fmt::Display for DeviceStateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongLength { found, expected } => write!(
                f,
                "TSH_ARM_DEVICE_STATE_WRONG_LENGTH found={found} expected={expected}"
            ),
            Self::SchemaMismatch { found, expected } => write!(
                f,
                "TSH_ARM_DEVICE_STATE_SCHEMA_MISMATCH found={found} expected={expected}"
            ),
            Self::UnknownMemoryState { found } => {
                write!(f, "TSH_ARM_DEVICE_STATE_UNKNOWN_MEMORY_STATE found={found}")
            }
        }
    }
}

impl std::error::Error for DeviceStateError {}

struct LeCursor<'a> {
    bytes: &'a mut [u8],
    at: usize,
}

impl<'a> LeCursor<'a> {
    fn new(bytes: &'a mut [u8]) -> Self {
        Self { bytes, at: 0 }
    }
    fn seek(&mut self, at: usize) {
        self.at = at;
    }
    fn u32(&mut self, v: u32) {
        self.bytes[self.at..self.at + 4].copy_from_slice(&v.to_le_bytes());
        self.at += 4;
    }
    fn u64(&mut self, v: u64) {
        self.bytes[self.at..self.at + 8].copy_from_slice(&v.to_le_bytes());
        self.at += 8;
    }
    fn f64(&mut self, v: f64) {
        self.bytes[self.at..self.at + 8].copy_from_slice(&v.to_le_bytes());
        self.at += 8;
    }
}

struct LeReader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> LeReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }
    fn seek(&mut self, at: usize) {
        self.at = at;
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(
            self.bytes[self.at..self.at + 4]
                .try_into()
                .expect("4 bytes"),
        );
        self.at += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(
            self.bytes[self.at..self.at + 8]
                .try_into()
                .expect("8 bytes"),
        );
        self.at += 8;
        v
    }
    fn f64(&mut self) -> f64 {
        f64::from_bits(self.u64())
    }
}

impl TshArmState {
    /// Encode into the `PrismTshArmState` POD layout.
    ///
    /// Field order and offsets mirror `cuda/tsh_gearbox_arm.cuh` exactly; the
    /// parity test round-trips a populated state through the compiled device
    /// struct, so a drift on either side fails a test rather than misparsing
    /// science.
    pub fn to_device_bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; DEVICE_STATE_BYTES];
        {
            let mut w = LeCursor::new(&mut out);
            w.u32(TSH_ARM_SCHEMA_VERSION);
            w.u32(u32::from(self.rechallenge_armed));
            w.u32(self.memory_state as u32);
            w.u32(self.last_requested_gear);
            w.u64(self.record_sequence);
            w.u64(self.decisions);
            w.f64(self.now_ps);
            w.f64(self.memory_entered_ps);
            w.f64(self.last_sentinel_ps);
            w.f64(self.request_changed_ps);
            // `None` maps to the device's infinity, NEVER to 0.0.
            w.f64(to_device_past(self.active_memory_exit_ps));
            w.f64(to_device_future(self.reinforce_since_ps));
            w.f64(to_device_past(self.sentinel_hold_until_ps));
            w.f64(f64::NAN); // last_salience — recomputed on the next decision
            w.f64(f64::NAN); // last_confidence — likewise
            for i in 0..TSH_ARM_N_BRANCHES {
                let b = &self.branch[i];
                w.seek(DEVICE_STATE_BRANCH_OFFSET + i * DEVICE_BRANCH_STATE_BYTES);
                w.f64(b.activity);
                w.f64(to_device_past(b.last_evidence_ps));
                w.f64(b.pending_activity);
                w.f64(b.pending_span_ps);
                w.f64(b.coverage_activity);
                w.u32(u32::from(b.ever_observed));
                w.u32(b.pending_samples);
                w.u64(b.total_samples);
                w.u64(0); // _reserved
            }
            w.seek(DEVICE_STATE_BRANCH_OFFSET + TSH_ARM_N_BRANCHES * DEVICE_BRANCH_STATE_BYTES);
            w.u64(self.dropped_records);
            w.u64(self.signal_grid_refusals);
        }
        out
    }

    /// Decode from the `PrismTshArmState` POD layout.
    pub fn from_device_bytes(bytes: &[u8]) -> Result<Self, DeviceStateError> {
        if bytes.len() != DEVICE_STATE_BYTES {
            return Err(DeviceStateError::WrongLength {
                found: bytes.len(),
                expected: DEVICE_STATE_BYTES,
            });
        }
        let mut r = LeReader::new(bytes);
        let schema = r.u32();
        if schema != TSH_ARM_SCHEMA_VERSION {
            return Err(DeviceStateError::SchemaMismatch {
                found: schema,
                expected: TSH_ARM_SCHEMA_VERSION,
            });
        }
        let rechallenge_armed = r.u32() != 0;
        let memory_raw = r.u32();
        let memory_state = TshMemoryState::from_u32(memory_raw)
            .ok_or(DeviceStateError::UnknownMemoryState { found: memory_raw })?;
        let last_requested_gear = r.u32();
        let record_sequence = r.u64();
        let decisions = r.u64();
        let now_ps = r.f64();
        let memory_entered_ps = r.f64();
        let last_sentinel_ps = r.f64();
        let request_changed_ps = r.f64();
        let active_memory_exit_ps = from_device_clock(r.f64());
        let reinforce_since_ps = from_device_clock(r.f64());
        let sentinel_hold_until_ps = from_device_clock(r.f64());
        let _last_salience = r.f64();
        let _last_confidence = r.f64();

        let mut branch = [BranchState::unbound(); TSH_ARM_N_BRANCHES];
        for (i, slot) in branch.iter_mut().enumerate() {
            r.seek(DEVICE_STATE_BRANCH_OFFSET + i * DEVICE_BRANCH_STATE_BYTES);
            slot.activity = r.f64();
            slot.last_evidence_ps = from_device_clock(r.f64());
            slot.pending_activity = r.f64();
            slot.pending_span_ps = r.f64();
            slot.coverage_activity = r.f64();
            slot.ever_observed = r.u32() != 0;
            slot.pending_samples = r.u32();
            slot.total_samples = r.u64();
        }
        r.seek(DEVICE_STATE_BRANCH_OFFSET + TSH_ARM_N_BRANCHES * DEVICE_BRANCH_STATE_BYTES);
        let dropped_records = r.u64();
        let signal_grid_refusals = r.u64();

        Ok(Self {
            record_sequence,
            memory_state,
            last_requested_gear,
            now_ps,
            memory_entered_ps,
            last_sentinel_ps,
            request_changed_ps,
            active_memory_exit_ps,
            reinforce_since_ps,
            sentinel_hold_until_ps,
            rechallenge_armed,
            decisions,
            branch,
            dropped_records,
            signal_grid_refusals,
        })
    }
}

impl TshArmPolicy {
    /// Encode into the `PrismTshArmPolicy` POD layout.
    ///
    /// Every threshold is a `double` on both sides. Storing them as `float` on
    /// the device — the first revision — meant `0.60` was
    /// `0.60000002384185791015625` there and `0.5999999999999999778` here, so
    /// the two implementations compared against different numbers and a
    /// decision taken at a threshold boundary could legitimately differ. That
    /// is precisely the divergence class the determinism rule exists to
    /// prevent, arriving by a route the doc did not admit.
    pub fn to_device_bytes(&self) -> Vec<u8> {
        let mut out = vec![0u8; DEVICE_POLICY_BYTES];
        {
            let mut w = LeCursor::new(&mut out);
            w.u32(TSH_ARM_SCHEMA_VERSION);
            w.u32(u32::from(self.armed));
            w.u32(u32::from(self.signal_grid_warp_independent));
            w.u32(0); // _pad0
            for v in self.branch_weight.iter() {
                w.f64(*v);
            }
            for v in self.branch_reference_rate.iter() {
                w.f64(*v);
            }
            for v in self.scalar_fields().iter() {
                w.f64(v.value);
            }
            for v in self.gear_dt_fs.iter() {
                w.f64(*v);
            }
            w.u64(self.policy_digest);
        }
        out
    }
}

impl TshArmCheckpoint {
    /// The device POD this checkpoint restores into.
    pub fn to_device_state_bytes(&self) -> Vec<u8> {
        self.state.to_device_bytes()
    }

    /// Rebuild a checkpoint from a device POD read back off the GPU.
    ///
    /// `policy_digest` is supplied by the caller because the device state does
    /// not carry one — the digest lives on the policy, which is uploaded once.
    /// Binding it here is what makes [`TshGearboxArm::restore`] able to refuse a
    /// trajectory captured under different thresholds.
    pub fn from_device_state_bytes(
        bytes: &[u8],
        policy_digest: u64,
    ) -> Result<Self, DeviceStateError> {
        Ok(Self {
            schema_version: TSH_ARM_SCHEMA_VERSION,
            policy_digest,
            state: TshArmState::from_device_bytes(bytes)?,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Per-arm non-interference
// ═══════════════════════════════════════════════════════════════════════════

/// Whether a run's physics signature can be compared against a baseline.
///
/// The Stage-II observer arm claims `dt_mutation = false` and backs it with a
/// baseline-vs-enabled physics-signature equivalence. A gearbox-facing arm
/// cannot make that claim about the same run and must not be allowed to answer
/// the question with a bare boolean: a run whose timestep this arm influenced
/// has no baseline it is equivalent to, and reporting `true` would assert an
/// equivalence that was never tested while reporting `false` would assert a
/// divergence that was never measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhysicsSignatureEquivalence {
    /// Measured: the signature matched the baseline.
    Equivalent,
    /// Measured: the signature diverged from the baseline.
    Divergent,
    /// SCOPED NOT-APPLICABLE. The gearbox-facing arm is active, so the run has
    /// no fixed-dt baseline to be equivalent to. Not a failure and not a pass.
    NotApplicableGearboxArmActive,
}

impl PhysicsSignatureEquivalence {
    /// Stable machine-readable label for evidence records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Equivalent => "equivalent",
            Self::Divergent => "divergent",
            Self::NotApplicableGearboxArmActive => "not_applicable_gearbox_arm_active",
        }
    }
}

/// This arm's OWN non-interference declaration.
///
/// Per-arm, deliberately. `TshRuntimeNonInterferenceLedger` in
/// `tsh_stage_ii.rs` is one flat struct with no arm dimension, and this module
/// contains ZERO references to it — verified by
/// `arm_declares_non_interference_per_arm_without_touching_the_global_ledger`.
/// The gearbox-facing arm therefore declares `dt_mutation = true` for ITSELF
/// here, and the observer arm's `dt_mutation = false` claim about ITSELF stays
/// intact and independently assertable. Flipping one global flag would have
/// made the observer's claim false without any test noticing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TshArmNonInterferenceClaim {
    /// Stable identity of the arm making the claim.
    pub arm: &'static str,
    /// This arm influences the timestep. TRUE — that is its entire purpose.
    pub dt_mutation: bool,
    /// This arm never writes a force.
    pub force_mutation: bool,
    /// This arm never changes what is sampled, only how finely time advances.
    pub sampling_bias: bool,
    /// Scoped physics-signature equivalence for a run this arm is active in.
    pub physics_signature_equivalent: PhysicsSignatureEquivalence,
}

/// Stable identity of the gearbox-facing arm in a non-interference ledger.
pub const TSH_ARM_IDENTITY: &str = "tsh_gearbox_arm";

// ═══════════════════════════════════════════════════════════════════════════
// The arm
// ═══════════════════════════════════════════════════════════════════════════

/// The gearbox-facing TSH arm.
///
/// Determines scientific-resolution DEMAND. Has no authority to authorize a
/// timestep; it only requests.
#[derive(Debug, Clone)]
pub struct TshGearboxArm {
    policy: TshArmPolicy,
    state: TshArmState,
}

impl TshGearboxArm {
    /// Start an arm under `policy` at simulated time `t0`.
    pub fn new(policy: TshArmPolicy, t0: SimulatedPs) -> Self {
        Self {
            state: TshArmState::reset(t0.ps()),
            policy,
        }
    }

    /// The resolved policy in force.
    pub fn policy(&self) -> &TshArmPolicy {
        &self.policy
    }

    /// The current trajectory state.
    pub fn state(&self) -> &TshArmState {
        &self.state
    }

    /// This arm's OWN non-interference declaration. See
    /// [`TshArmNonInterferenceClaim`].
    ///
    /// `dt_mutation` is TRUE and scoped to this arm; the observer arm's
    /// `dt_mutation = false` claim about itself is untouched. Physics-signature
    /// equivalence is reported as a scoped NOT-APPLICABLE while this arm is
    /// active, never as a bare boolean.
    pub fn non_interference_claim(&self) -> TshArmNonInterferenceClaim {
        TshArmNonInterferenceClaim {
            arm: TSH_ARM_IDENTITY,
            // An UNARMED arm has no opinion and never moves a gear, so it does
            // not mutate dt. Claiming otherwise would overstate its reach.
            dt_mutation: self.policy.armed,
            force_mutation: false,
            sampling_bias: false,
            physics_signature_equivalent: if self.policy.armed {
                PhysicsSignatureEquivalence::NotApplicableGearboxArmActive
            } else {
                // Unarmed: it emitted no request, so the run's signature is
                // whatever it would have been. Still not this arm's measurement
                // to make — it reports the scoped value only when it is active.
                PhysicsSignatureEquivalence::Equivalent
            },
        }
    }

    /// Stage one branch observation into the current decision window.
    ///
    /// Staging is a `(sum activity, sum span)` pair, never a running mean.
    /// That is what makes the per-simulated-picosecond normalisation exact
    /// under any split of the window into samples: the sample count cancels
    /// identically, so a change in sampling rate at constant physics cannot
    /// move the salience.
    pub fn observe(&mut self, obs: BranchObservation) -> Result<(), ObservationRejection> {
        if !obs.accumulated_activity.is_finite() || obs.accumulated_activity < 0.0 {
            return Err(ObservationRejection::NonFiniteOrNegativeActivity);
        }
        // ASC / FORCE EXCLUSIVITY at the staging point.
        //
        // Refused, not silently dropped and not silently accepted. The only
        // "ASC" product this engine writes is the force buffer under another
        // filename, so admitting it here as a second family would double-weight
        // force in the aggregate with nothing on the record to show for it.
        if matches!(obs.branch, SalienceBranch::SignalGridWarp)
            && !self.policy.signal_grid_warp_independent
        {
            self.state.signal_grid_refusals = self.state.signal_grid_refusals.saturating_add(1);
            return Err(ObservationRejection::SignalGridWarpNotIndependentOfForce);
        }
        let b = &mut self.state.branch[obs.branch.index()];
        b.pending_activity += obs.accumulated_activity;
        b.pending_span_ps += obs.span.ps();
        b.pending_samples = b.pending_samples.saturating_add(obs.sample_count);
        b.total_samples = b.total_samples.saturating_add(obs.sample_count as u64);
        b.ever_observed = true;
        b.last_evidence_ps = Some(match b.last_evidence_ps {
            Some(prev) if prev >= obs.window_end.ps() => prev,
            _ => obs.window_end.ps(),
        });
        Ok(())
    }

    /// Stage evidence that is DERIVED FROM THE FORCE REPRESENTATION.
    ///
    /// The engine's ASC vector product — `{stem}_streamNN_asc_vectors.bin` — is
    /// written straight out of the `host_forces` buffer. It is therefore the
    /// force observation under a different filename, not an independent state,
    /// and this method is the supported way to feed it in: it stages into
    /// [`SalienceBranch::ForceEnergyStability`], the representation it actually
    /// belongs to, so the same physics is integrated exactly once no matter
    /// which filename the caller happened to read it from.
    ///
    /// The alternative — a second branch — is what double-counts. Two branches
    /// backed by one buffer give that buffer two sixths of the aggregate weight
    /// against one sixth for every genuinely distinct family, and the emitted
    /// record cannot show that it happened.
    pub fn observe_force_derived(
        &mut self,
        accumulated_activity: f64,
        span: SimulatedPsSpan,
        window_end: SimulatedPs,
        sample_count: u32,
    ) -> Result<(), ObservationRejection> {
        self.observe(BranchObservation {
            branch: SalienceBranch::ForceEnergyStability,
            accumulated_activity,
            span,
            window_end,
            sample_count,
        })
    }

    /// How many observations were refused by the ASC / force exclusivity rule.
    /// Non-zero means a producer is wired to the wrong branch.
    pub fn signal_grid_refusals(&self) -> u64 {
        self.state.signal_grid_refusals
    }

    /// The per-branch BINDING table: for every family, the unit it is contracted
    /// to stage, the run-path producer that stages it (or `None` when no
    /// producer exists in this build), the resolved weight and reference rate,
    /// and whether it has in fact ever reported.
    ///
    /// This is the artifact-side statement that closes the
    /// "absent is not calm" gap. `ever_observed = false` alone cannot say
    /// WHETHER a producer exists, so a permanently unbound family and a family
    /// whose first observation has not landed yet look identical in the record.
    /// `run_path_producer = None` with `ever_observed = false` is a
    /// STRUCTURALLY ABSENT branch; `Some(..)` with `ever_observed = false` is a
    /// wired branch that has not yet reported. Emitting this makes the two
    /// distinguishable offline.
    pub fn branch_binding_report(&self) -> Vec<BranchBindingRow> {
        SalienceBranch::ALL
            .iter()
            .map(|&b| {
                let i = b.index();
                BranchBindingRow {
                    branch: b,
                    label: b.as_str(),
                    activity_unit: b.activity_unit(),
                    run_path_producer: b.run_path_producer(),
                    weight: self.policy.branch_weight[i],
                    reference_rate: self.policy.branch_reference_rate[i],
                    ever_observed: self.state.branch[i].ever_observed,
                    structurally_absent: b.run_path_producer().is_none(),
                    coverage: self.branch_coverage(i),
                    salience: self.branch_salience(i),
                    total_samples: self.state.branch[i].total_samples,
                }
            })
            .collect()
    }

    /// The branch's integrated rate, per simulated picosecond. `None` while
    /// the branch has never reported — unbound is not calm.
    pub fn branch_rate(&self, branch: SalienceBranch) -> Option<PerSimulatedPs<f64>> {
        let b = &self.state.branch[branch.index()];
        if !b.ever_observed {
            return None;
        }
        let tau = self.policy.branch_tau_ps.value;
        SimulatedPsSpan::new(tau).map(|span| PerSimulatedPs::from_accumulated(b.activity, span))
    }

    /// Advance every branch to `now`, folding in the staged window.
    ///
    /// Exact solution of `dA/dt = -A/tau + r` with `r` held over the span the
    /// evidence COVERS, and the activity integrator **frozen** over the
    /// uncovered remainder. Surplus staged span is carried forward, never
    /// discarded. Fixed-order serial loop; no atomics.
    ///
    /// # Why freeze rather than decay
    ///
    /// Decaying `activity` over simulated time no observation covered
    /// integrates `r = 0` there — it records *absence of evidence* as
    /// *measured calm*. That is the exact positive-feedback loop the module
    /// header says is structurally impossible: sparser evidence lowers the
    /// salience, a lower salience earns a coarser gear, a coarser gear yields
    /// sparser evidence. With the decay in place, the same constant physics
    /// whose true salience is `0.5000` read `0.5000` at full coverage, `0.3045`
    /// at 50%, `0.1395` at 20% (below the upshift threshold: coarsening
    /// eligible) and `0.0376` at 5% — with observation confidence `1.000`
    /// throughout, because confidence measured recency only.
    ///
    /// Freezing makes the steady state `A -> r * tau` regardless of duty cycle,
    /// so the salience of constant physics is the same number no matter how
    /// often the observer looked. What sparse observation costs is CONFIDENCE
    /// (see [`Self::branch_confidence`]), which is the honest place for it: the
    /// arm knows less, it does not observe less activity.
    ///
    /// A branch that has genuinely gone quiet says so by REPORTING calm —
    /// `accumulated_activity = 0` over a real span — which drives the
    /// integrator down exactly as before. Silence is not calm.
    ///
    /// Returns whether any branch's carried backlog had to be truncated.
    fn advance(&mut self, now: f64) -> bool {
        let tau = self.policy.branch_tau_ps.value;
        let backlog_limit = self.policy.max_sentinel_interval_ps.value;
        let mut truncated = false;
        let mut dt = now - self.state.now_ps;
        if !(dt > 0.0) || !dt.is_finite() {
            dt = 0.0;
        }
        for i in 0..TSH_ARM_N_BRANCHES {
            let b = &mut self.state.branch[i];
            if !(tau > 0.0) {
                // Unusable policy: refuse to evolve rather than divide by zero.
                b.consume_pending(f64::INFINITY, backlog_limit);
                continue;
            }
            let covered = if b.pending_span_ps > dt {
                dt
            } else {
                b.pending_span_ps
            };
            if covered > 0.0 && b.pending_span_ps > 0.0 {
                let r = b.pending_activity / b.pending_span_ps;
                let e = (-covered / tau).exp();
                b.activity = b.activity * e + r * tau * (1.0 - e);
                // Coverage is driven by 1 over the covered span.
                b.coverage_activity = b.coverage_activity * e + tau * (1.0 - e);
            }
            let rest = dt - covered;
            if rest > 0.0 {
                // ACTIVITY IS FROZEN over uncovered simulated time.
                // COVERAGE decays over it — that is what makes the absence
                // measurable instead of invisible.
                b.coverage_activity *= (-rest / tau).exp();
            }
            truncated |= b.consume_pending(covered, backlog_limit);
        }
        truncated
    }

    /// The fraction of recent simulated time this branch's evidence covered,
    /// in `[0,1]`. `0.0` for a branch that never reported.
    ///
    /// This is anti-blindness (c) as an arithmetic quantity. It is exactly the
    /// duty cycle in steady state: a branch staging `0.2 ps` of evidence per
    /// `1.0 ps` of simulated time converges to `0.2`, not `1.0`.
    pub fn branch_coverage(&self, index: usize) -> f64 {
        let b = &self.state.branch[index];
        if !b.ever_observed {
            return 0.0;
        }
        let tau = self.policy.branch_tau_ps.value;
        if !(tau > 0.0) {
            return 0.0;
        }
        let c = b.coverage_activity / tau;
        if !c.is_finite() || c < 0.0 {
            return 0.0;
        }
        if c > 1.0 {
            1.0
        } else {
            c
        }
    }

    /// Per-branch confidence in `[0,1]`: `recency x coverage`.
    ///
    /// Both factors are mandatory and they answer different questions.
    /// RECENCY asks "when did this branch last say anything"; COVERAGE asks
    /// "how much of the simulated time it was supposed to be watching did it
    /// actually watch". The first revision had only recency, so a branch
    /// reporting punctually but thinly returned `1.000` and the coarsening gate
    /// waved through exactly the sparse-observation regimes it exists to
    /// refuse.
    ///
    /// Past [`TshArmPolicy::max_sentinel_interval_ps`] of staleness the result
    /// is hard **zero**, not an exponential tail: with the integrator frozen,
    /// an arbitrarily old activity would otherwise keep voting forever. Beyond
    /// the horizon the branch stops contributing at all, which — if every
    /// branch is beyond it — makes the integrated salience UNMEASURED and
    /// drives the arm to the finest gear.
    fn branch_confidence(&self, index: usize, now: f64) -> f64 {
        let b = &self.state.branch[index];
        let last_evidence = match b.last_evidence_ps {
            Some(t) if b.ever_observed => t,
            _ => return 0.0,
        };
        let coverage = self.branch_coverage(index);
        if !(coverage > 0.0) {
            return 0.0;
        }
        let cadence = self.policy.min_observation_cadence_ps.value;
        let decay = self.policy.confidence_decay_ps.value;
        let horizon = self.policy.max_sentinel_interval_ps.value;
        let stale = now - last_evidence;
        if !(stale > 0.0) {
            return coverage;
        }
        if horizon.is_finite() && horizon > 0.0 && stale > horizon {
            return 0.0;
        }
        if !(cadence > 0.0) || !(decay > 0.0) {
            // A policy that cannot express cadence cannot express freshness.
            // Fail towards "blocked from coarsening", never "free to coarsen".
            return 0.0;
        }
        let recency = if stale <= cadence {
            1.0
        } else {
            (-(stale - cadence) / decay).exp()
        };
        recency * coverage
    }

    /// This branch's own salience in `[0,1)`, or `None` when it is not
    /// admissible evidence.
    ///
    /// `None` for a branch that has NEVER reported: an unbound branch has
    /// `activity = 0`, and returning `Some(0.0)` for it would make "no sensor"
    /// arithmetically identical to "sensor reporting perfect calm". That is the
    /// collision this whole module exists to avoid.
    fn branch_salience(&self, index: usize) -> Option<f64> {
        if !self.state.branch[index].ever_observed {
            return None;
        }
        let tau = self.policy.branch_tau_ps.value;
        let reference = self.policy.branch_reference_rate[index];
        if !(tau > 0.0) || !(reference > 0.0) {
            return None;
        }
        let r = self.state.branch[index].activity / tau;
        if !r.is_finite() || r < 0.0 {
            return None;
        }
        Some(r / (r + reference))
    }

    /// Take one decision and emit a request.
    ///
    /// Emits ALWAYS, including on no-change: a gap in the record stream must
    /// never be ambiguous between "gear held" and "decision did not run".
    pub fn decide(&mut self, clock: DecisionClock) -> TshGearRequest {
        let now = clock.cumulative_simulated_time.ps();
        let mut flags = 0u32;
        if self.state.dropped_records != 0 {
            flags |= FLAG_RING_DROPPED;
        }

        // ── Sentinel clock: only a MEASURED fine gear counts as a sample. ──
        match clock.authorized_gear {
            Some(g) if g == GEAR_FINE => self.state.last_sentinel_ps = now,
            Some(_) => {}
            None => flags |= FLAG_AUTHORIZED_UNMEASURED,
        }

        // ── Which branches carried fresh evidence this window. ────────────
        let mut fresh: u8 = 0;
        for i in 0..TSH_ARM_N_BRANCHES {
            if self.state.branch[i].pending_span_ps > 0.0 {
                fresh |= 1u8 << i;
            }
        }

        if self.advance(now) {
            flags |= FLAG_EVIDENCE_BACKLOG_TRUNCATED;
        }
        self.state.now_ps = now;
        self.state.decisions = self.state.decisions.saturating_add(1);
        self.state.record_sequence = self.state.record_sequence.saturating_add(1);

        // ── Fixed-order integration. Index order IS reduce order. ─────────
        //
        // Two aggregates come out of this loop and they have different jobs:
        //
        //   * `numerator/denominator` — the confidence-weighted MEAN. Reported,
        //     never gated on. It is a summary of what the whole evidence set
        //     looks like.
        //   * `peak` — the MAXIMUM over admissible branches. Every resolution
        //     gate below is taken on this. A demand is not a vote: the finest
        //     resolution anything credible asked for wins, and no number of
        //     calm branches may dilute one branch's excursion.
        let mut numerator = 0.0f64;
        let mut denominator = 0.0f64;
        let mut weight_total = 0.0f64;
        let mut confidence_weighted = 0.0f64;
        let mut coverage_weighted = 0.0f64;
        let mut salient: u8 = 0;
        let mut stale: u8 = 0;
        let mut unbound: u8 = 0;
        let mut activity_numerator = 0.0f64;
        let mut peak: Option<(f64, SalienceBranch)> = None;
        let down = self.policy.salience_downshift_threshold.value;
        let horizon = self.policy.max_sentinel_interval_ps.value;

        for i in 0..TSH_ARM_N_BRANCHES {
            let w = self.policy.branch_weight[i];
            let c = self.branch_confidence(i, now);
            let sb = self.branch_salience(i);
            if w > 0.0 {
                weight_total += w;
                confidence_weighted += w * c;
                coverage_weighted += w * self.branch_coverage(i);
            }
            if !self.state.branch[i].ever_observed {
                unbound |= 1u8 << i;
            } else {
                if c < 1.0 {
                    stale |= 1u8 << i;
                }
                if let Some(last) = self.state.branch[i].last_evidence_ps {
                    if horizon.is_finite() && horizon > 0.0 && (now - last) > horizon {
                        flags |= FLAG_STALE_HORIZON_EXCEEDED;
                    }
                }
            }
            let sb = match sb {
                Some(v) => v,
                None => {
                    // A never-reported branch is UNBOUND, not misconfigured;
                    // only a policy that cannot express a salience at all
                    // raises the invalid-reference flag.
                    if self.state.branch[i].ever_observed {
                        flags |= FLAG_INVALID_REFERENCE;
                    }
                    continue;
                }
            };
            if w > 0.0 && c > 0.0 {
                numerator += w * c * sb;
                denominator += w * c;
                activity_numerator += w * c * self.state.branch[i].activity;
                // Admissible: carries weight, has reported, is still credible.
                // Ties keep the LOWEST branch index so the peak is a fixed
                // function of the state and not of comparison order.
                let better = match peak {
                    None => true,
                    Some((best, _)) => sb > best,
                };
                if better {
                    peak = Some((sb, SalienceBranch::ALL[i]));
                }
            }
            if sb >= down {
                salient |= 1u8 << i;
            }
        }

        // The REPORTED statistic.
        let mean_salience = if denominator > 0.0 {
            Some(numerator / denominator)
        } else {
            flags |= FLAG_EVIDENCE_ABSENT;
            None
        };
        let peak_branch = peak.map(|(_, b)| b);
        // The GATE INPUT. Every `salience` below is this one.
        let salience = peak.map(|(v, _)| v);
        // The weighted-mean ACTIVITY divided by the integrator time constant.
        // Built through `from_accumulated` with a genuine simulated-time span,
        // so the rate cannot be laundered into existence from a bare float.
        let integrated_rate = if denominator > 0.0 {
            SimulatedPsSpan::new(self.policy.branch_tau_ps.value).map(|tau_span| {
                PerSimulatedPs::from_accumulated(activity_numerator / denominator, tau_span)
            })
        } else {
            None
        };
        let confidence = if weight_total > 0.0 {
            Some(confidence_weighted / weight_total)
        } else {
            None
        };
        let coverage = if weight_total > 0.0 {
            Some(coverage_weighted / weight_total)
        } else {
            None
        };
        // Raised on every decision it is true of, not only inside the
        // coarsening branch, so a reader can see the arm was under-observed
        // even on the decisions where it was demanding finer resolution.
        if coverage.is_some_and(|c| c < self.policy.min_confidence_for_coarsening.value) {
            flags |= FLAG_OBSERVATION_SPARSE;
        }
        if self.state.signal_grid_refusals != 0 {
            flags |= FLAG_SIGNAL_GRID_UNBOUND;
        }

        // ── Memory state machine (operator's table). ──────────────────────
        //
        // Driven by the PEAK, for the same reason the gears are: the memory is
        // supposed to remember an excursion, and a mean lets five quiet
        // branches erase one branch's excursion before the memory ever sees it.
        let up = self.policy.salience_upshift_threshold.value;
        let reinforce = self.policy.salience_reinforce_threshold.value;
        let memory_age = now - self.state.memory_entered_ps;
        let previous_state = self.state.memory_state;

        match salience {
            Some(s) if s >= reinforce => {
                if self.state.reinforce_since_ps.map_or(true, |t| t > now) {
                    self.state.reinforce_since_ps = Some(now);
                }
            }
            _ => self.state.reinforce_since_ps = None,
        }

        let in_rechallenge_window = self.state.rechallenge_armed
            && self
                .state
                .active_memory_exit_ps
                .is_some_and(|exit| (now - exit) <= self.policy.rechallenge_window_ps.value);

        let mut next_state = previous_state;
        let excursion = matches!(salience, Some(s) if s >= down);
        let quiet = matches!(salience, Some(s) if s < up);

        if excursion
            && in_rechallenge_window
            && matches!(
                previous_state,
                TshMemoryState::Decay
                    | TshMemoryState::InputWithdrawal
                    | TshMemoryState::Low
                    | TshMemoryState::Baseline
            )
        {
            next_state = TshMemoryState::Rechallenge;
            flags |= FLAG_RECHALLENGE;
        } else {
            match previous_state {
                TshMemoryState::Baseline | TshMemoryState::Low => {
                    if excursion {
                        next_state = TshMemoryState::ThresholdApproach;
                    } else if salience.is_some() && !quiet {
                        next_state = TshMemoryState::Accumulating;
                    } else if salience.is_some() {
                        next_state = TshMemoryState::Low;
                    }
                }
                TshMemoryState::Accumulating => {
                    if excursion {
                        next_state = TshMemoryState::ThresholdApproach;
                    } else if quiet && memory_age >= self.policy.decay_period_ps.value {
                        next_state = TshMemoryState::Decay;
                    }
                }
                TshMemoryState::ThresholdApproach => {
                    if matches!(salience, Some(s) if s >= reinforce) {
                        next_state = TshMemoryState::ActiveMemory;
                    } else if quiet {
                        next_state = TshMemoryState::InputWithdrawal;
                    }
                }
                TshMemoryState::ActiveMemory => {
                    let held = self
                        .state
                        .reinforce_since_ps
                        .is_some_and(|t| (now - t) >= self.policy.reinforce_dwell_ps.value);
                    if matches!(salience, Some(s) if s >= reinforce) && held {
                        next_state = TshMemoryState::Reinforced;
                    } else if quiet {
                        next_state = TshMemoryState::InputWithdrawal;
                    }
                }
                TshMemoryState::Reinforced => {
                    if quiet {
                        next_state = TshMemoryState::InputWithdrawal;
                    }
                }
                TshMemoryState::InputWithdrawal => {
                    if excursion {
                        next_state = TshMemoryState::ThresholdApproach;
                    } else if memory_age >= self.policy.withdrawal_hold_ps.value {
                        next_state = TshMemoryState::Decay;
                    }
                }
                TshMemoryState::Decay => {
                    if excursion {
                        next_state = TshMemoryState::ThresholdApproach;
                    } else if memory_age >= self.policy.decay_period_ps.value {
                        next_state = TshMemoryState::Baseline;
                    }
                }
                TshMemoryState::Rechallenge => {
                    next_state = TshMemoryState::ActiveMemory;
                }
            }
        }

        if next_state != previous_state {
            if matches!(
                previous_state,
                TshMemoryState::ActiveMemory | TshMemoryState::Reinforced
            ) {
                self.state.active_memory_exit_ps = Some(now);
                self.state.rechallenge_armed = true;
            }
            self.state.memory_state = next_state;
            self.state.memory_entered_ps = now;
        }
        // The rechallenge window expires by ELAPSED SIMULATED TIME, not by the
        // memory happening to pass through baseline. Clearing it on the
        // baseline transition would have closed the window after one decay
        // period regardless of `rechallenge_window_ps`, making the knob inert
        // and losing the very re-excursion it exists to catch.
        if self.state.rechallenge_armed
            && self.state.memory_state != TshMemoryState::Rechallenge
            && self
                .state
                .active_memory_exit_ps
                .is_some_and(|exit| (now - exit) > self.policy.rechallenge_window_ps.value)
        {
            self.state.rechallenge_armed = false;
        }

        // ── Gear demand. Combined by MINIMUM (finest wins), never a score. ─
        //
        // A weighted score would admit one demand large enough to outvote
        // another. The only safe aggregation for a DEMAND is "the finest
        // resolution anything asked for". The lexicographic ordering that must
        // never be a score lives in the GOVERNOR; here the min IS the
        // conservative choice. `reason` reports the most authoritative demand
        // that produced the final gear.
        let mut base_gear = self.state.last_requested_gear;
        if base_gear > GEAR_MAX_ACTIVE {
            base_gear = match clock.authorized_gear {
                Some(g) if g <= GEAR_MAX_ACTIVE => g,
                _ => GEAR_SAFE,
            };
        }
        let one_finer = base_gear.saturating_sub(1);

        let since_sentinel = now - self.state.last_sentinel_ps;
        let sentinel_due = since_sentinel.is_finite()
            && since_sentinel >= self.policy.max_sentinel_interval_ps.value;
        let sentinel_holding = self
            .state
            .sentinel_hold_until_ps
            .is_some_and(|until| now < until);

        let mut requested = base_gear;
        let mut reason = RequestReason::Hold;

        if !self.policy.armed {
            flags |= FLAG_POLICY_UNARMED;
            reason = RequestReason::PolicyUnarmed;
            requested = GEAR_UNSET;
        } else {
            // (6) salience excursion -> one gear finer.
            if excursion && one_finer <= requested {
                requested = one_finer;
                reason = RequestReason::SalienceDownshift;
            }
            // (5) memory: threshold approach -> downshift one gear.
            if self.state.memory_state == TshMemoryState::ThresholdApproach
                && one_finer <= requested
            {
                requested = one_finer;
                reason = RequestReason::MemoryThresholdDownshift;
            }
            // (4) memory: active / reinforced / withdrawal -> maintain fine.
            if matches!(
                self.state.memory_state,
                TshMemoryState::ActiveMemory
                    | TshMemoryState::Reinforced
                    | TshMemoryState::InputWithdrawal
            ) {
                requested = GEAR_FINE;
                reason = RequestReason::MemoryHoldFine;
            }
            // (3) anti-blindness (c): unmeasured is not calm. Bias fine.
            if salience.is_none() {
                requested = GEAR_FINE;
                reason = RequestReason::EvidenceAbsent;
            }
            // (2) anti-blindness (b): unconditional; salience plays no part.
            if sentinel_due || sentinel_holding {
                requested = GEAR_FINE;
                reason = RequestReason::Sentinel;
                flags |= FLAG_SENTINEL_FORCED;
                if sentinel_due && !sentinel_holding {
                    self.state.sentinel_hold_until_ps =
                        Some(now + self.policy.sentinel_hold_ps.value);
                }
            }
            // (1) rechallenge -> immediate finest, ahead of everything.
            if self.state.memory_state == TshMemoryState::Rechallenge {
                requested = GEAR_FINE;
                reason = RequestReason::Rechallenge;
            }

            // COARSENING is considered ONLY when nothing above asked for finer.
            if requested == base_gear
                && reason == RequestReason::Hold
                && quiet
                && self.state.memory_state.permits_coarsening_request()
            {
                let confident = confidence
                    .map(|c| c >= self.policy.min_confidence_for_coarsening.value)
                    .unwrap_or(false);
                let dwell = now - self.state.request_changed_ps;
                let dwelled = dwell.is_finite() && dwell >= self.policy.min_gear_dwell_ps.value;
                // Age of the state the arm is in NOW, after this decision's
                // transition — not the age of the state it just left.
                let memory_age_post = now - self.state.memory_entered_ps;
                let decayed_long_enough = self.state.memory_state != TshMemoryState::Decay
                    || memory_age_post >= self.policy.decay_period_ps.value;

                if !confident {
                    // ANTI-BLINDNESS (d).
                    flags |= FLAG_CONFIDENCE_BLOCKED;
                    reason = RequestReason::ConfidenceGate;
                } else if !dwelled || !decayed_long_enough {
                    flags |= FLAG_DWELL_BLOCKED;
                    reason = RequestReason::DwellGate;
                } else {
                    requested = (base_gear + 1).min(GEAR_MAX_ACTIVE);
                    reason = RequestReason::UpshiftEligible;
                }
            }
        }

        // ── AUTHORITY BOUNDARY, enforced at the exit. ─────────────────────
        //
        // The arm REQUESTS ONLY. There is no branch above that authorizes a
        // gear, writes a timestep, or touches device state: it computes a
        // number and hands it to the governor, which applies
        // `min(demanded, permitted)`. This clamp is the last statement about
        // the gear before it leaves, and it exists so the abort gear can never
        // be named even by an arithmetic accident — aborting is a physical
        // decision the governor owns.
        //
        // `arm_source_contains_no_authorization_path` holds this mechanically:
        // it reads this file and the device header and fails if either ever
        // acquires a write to an authorized gear, a timestep, or the gearbox
        // activation switch.
        if requested != GEAR_UNSET && requested > GEAR_MAX_ACTIVE {
            requested = GEAR_MAX_ACTIVE;
        }
        debug_assert!(
            requested == GEAR_UNSET || requested <= GEAR_MAX_ACTIVE,
            "TSH arm produced a gear outside the requestable range: {requested}"
        );
        if requested != self.state.last_requested_gear {
            self.state.request_changed_ps = now;
        }
        self.state.last_requested_gear = requested;

        let requested_gear = if requested == GEAR_UNSET {
            None
        } else {
            Some(requested)
        };
        let requested_dt_fs =
            requested_gear.and_then(|g| self.policy.gear_dt_fs.get(g as usize).copied());

        // Confidence in THIS REQUEST, distinct from observation confidence.
        //
        //   * a structurally mandated request (sentinel, rechallenge) is fully
        //     confident: policy, not inference, produced it;
        //   * a request that exists only because there was NO evidence has
        //     confidence 0.0 — it is a fail-safe default and must never be
        //     read as an inference;
        //   * otherwise it is the observation confidence attenuated by how
        //     close the salience sits to the threshold that would have decided
        //     differently. A decision taken on the knife edge is less
        //     trustworthy than the same decision taken far from it, and saying
        //     so is cheaper than pretending otherwise.
        let request_confidence = if !self.policy.armed {
            None
        } else if matches!(reason, RequestReason::Sentinel | RequestReason::Rechallenge) {
            Some(1.0)
        } else {
            match (salience, confidence) {
                (None, _) => Some(0.0),
                (Some(_), None) => None,
                (Some(s), Some(c)) => {
                    // Distance to the nearer threshold, saturating at the full
                    // band OUTSIDE it.
                    //
                    // The first revision took `|min(s-up, down-s)|`, which at
                    // s = 0.0 — maximally calm, as far from the downshift bound
                    // as it is possible to be — yielded 0.25/0.35 = 0.714
                    // instead of 1.0, because the absolute value turned a large
                    // negative distance into a small positive one. A decision
                    // taken far outside the band is not a knife-edge decision.
                    let band = down - up;
                    let margin = if band > 0.0 {
                        if s <= up || s >= down {
                            1.0
                        } else {
                            let nearest = (s - up).min(down - s);
                            (nearest / band).clamp(0.0, 1.0)
                        }
                    } else {
                        1.0
                    };
                    Some(c * margin)
                }
            }
        };

        TshGearRequest {
            schema_version: TSH_ARM_SCHEMA_VERSION,
            record_sequence: self.state.record_sequence,
            global_md_step: clock.global_md_step,
            cumulative_simulated_time_ps: now,
            requested_gear,
            requested_dt_fs,
            request_reason: reason,
            request_confidence,
            observation_confidence: confidence,
            integrated_salience: mean_salience,
            peak_branch_salience: salience,
            peak_branch,
            observation_coverage: coverage,
            thermodynamic_action: reason.thermodynamic_action(),
            integrated_rate,
            active_branch_mask: BranchMask {
                fresh,
                salient,
                stale,
                unbound,
            },
            memory_state: self.state.memory_state,
            memory_age_ps: now - self.state.memory_entered_ps,
            authorized_gear_echo: clock.authorized_gear,
            flags,
            policy_digest: self.policy.policy_digest,
        }
    }

    /// Record that a downstream ring refused `count` emitted records.
    ///
    /// Every subsequent request carries [`FLAG_RING_DROPPED`] until
    /// [`Self::clear_dropped_records`] is called, because a truncated record
    /// stream that does not say it is truncated is indistinguishable from a
    /// complete one — and the entire value of the sequence number is that a
    /// gap is unambiguous.
    pub fn note_dropped_records(&mut self, count: u64) {
        self.state.dropped_records = self.state.dropped_records.saturating_add(count);
    }

    /// Acknowledge and clear the drop counter, returning what it held.
    pub fn clear_dropped_records(&mut self) -> u64 {
        std::mem::replace(&mut self.state.dropped_records, 0)
    }

    /// Snapshot the salience trajectory.
    pub fn checkpoint(&self) -> TshArmCheckpoint {
        TshArmCheckpoint {
            schema_version: TSH_ARM_SCHEMA_VERSION,
            policy_digest: self.policy.policy_digest,
            state: self.state.clone(),
        }
    }

    /// Restore a salience trajectory.
    ///
    /// Refuses a checkpoint from a different schema or a different policy: the
    /// same trajectory under different thresholds is a different controller,
    /// and accepting it would produce a run that cannot be reproduced from
    /// either side.
    pub fn restore(&mut self, checkpoint: &TshArmCheckpoint) -> Result<(), CheckpointError> {
        if checkpoint.schema_version != TSH_ARM_SCHEMA_VERSION {
            return Err(CheckpointError::SchemaMismatch {
                found: checkpoint.schema_version,
                expected: TSH_ARM_SCHEMA_VERSION,
            });
        }
        if checkpoint.policy_digest != self.policy.policy_digest {
            return Err(CheckpointError::PolicyDigestMismatch {
                found: checkpoint.policy_digest,
                expected: self.policy.policy_digest,
            });
        }
        self.state = checkpoint.state.clone();
        Ok(())
    }
}

impl TshArmState {
    /// Decode a memory state from its wire value, for readers of the device
    /// mirror. `None` for an unrecognised value — never silently mapped to
    /// `Baseline`.
    pub fn memory_state_from_wire(v: u32) -> Option<TshMemoryState> {
        TshMemoryState::from_u32(v)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// The compiled device implementation
// ═══════════════════════════════════════════════════════════════════════════

/// Host-callable entry points exported by `cuda/tsh_gearbox_arm.cu`.
///
/// Every function in `cuda/tsh_gearbox_arm.cuh` is `__host__ __device__`, so
/// these run the SAME code the captured graph runs, without a GPU. That is what
/// makes [`device`] a parity harness rather than a mock: the test compares two
/// implementations, where the previous test compared this implementation
/// against numeric literals someone had typed into a comment.
///
/// Available only under the `gpu` feature, which is the feature that compiles
/// the translation unit.
#[cfg(feature = "gpu")]
pub mod device {
    use super::*;

    /// Slot order of [`abi_sizes`].
    pub const ABI_SLOT_BRANCH: usize = 0;
    /// `sizeof(PrismTshArmPolicy)`.
    pub const ABI_SLOT_POLICY: usize = 1;
    /// `sizeof(PrismTshArmState)`.
    pub const ABI_SLOT_STATE: usize = 2;
    /// `sizeof(PrismTshArmRecord)`.
    pub const ABI_SLOT_RECORD: usize = 3;
    /// `sizeof(PrismTshArmRing)`.
    pub const ABI_SLOT_RING: usize = 4;
    /// `offsetof(PrismTshArmState, branch)`.
    pub const ABI_SLOT_STATE_BRANCH_OFFSET: usize = 5;
    /// `offsetof(PrismTshArmPolicy, branch_tau_ps)`.
    pub const ABI_SLOT_POLICY_SCALAR_OFFSET: usize = 6;
    /// Number of slots [`abi_sizes`] fills.
    pub const ABI_SLOT_COUNT: usize = 7;

    extern "C" {
        fn prism_tsh_arm_abi_sizes(out: *mut u64, capacity: u32) -> u32;
        fn prism_tsh_arm_schema_version() -> u32;
        fn prism_tsh_arm_host_default_policy(policy: *mut std::ffi::c_void);
        fn prism_tsh_arm_host_reset(state: *mut std::ffi::c_void, t0_ps: f64);
        fn prism_tsh_arm_host_observe(
            policy: *const std::ffi::c_void,
            state: *mut std::ffi::c_void,
            branch: u32,
            accumulated_activity: f64,
            span_ps: f64,
            window_end_ps: f64,
            sample_count: u32,
        ) -> u32;
        fn prism_tsh_arm_host_decide(
            policy: *const std::ffi::c_void,
            state: *mut std::ffi::c_void,
            now_ps: f64,
            global_md_step: u64,
            frame: u32,
            authorized_gear: u32,
            authorized_gear_valid: u32,
            out_req: *mut std::ffi::c_void,
            out_rec: *mut std::ffi::c_void,
        );
        fn prism_tsh_arm_host_record_unwritten(record: *mut std::ffi::c_void);
        fn prism_tsh_arm_host_record_is_valid(record: *const std::ffi::c_void) -> u32;
        fn prism_tsh_arm_host_branch_coverage(
            policy: *const std::ffi::c_void,
            state: *const std::ffi::c_void,
            branch: u32,
        ) -> f64;
    }

    /// The struct sizes and offsets the CUDA compiler actually produced.
    ///
    /// Not constants re-declared on the C++ side — `sizeof` / `offsetof` as
    /// evaluated during the build. Comparing these against `DEVICE_*_BYTES` is
    /// what makes those Rust constants a checked fact rather than a comment.
    pub fn abi_sizes() -> [u64; ABI_SLOT_COUNT] {
        let mut out = [0u64; ABI_SLOT_COUNT];
        let n = unsafe { prism_tsh_arm_abi_sizes(out.as_mut_ptr(), ABI_SLOT_COUNT as u32) };
        assert_eq!(
            n as usize, ABI_SLOT_COUNT,
            "device ABI probe filled {n} of {ABI_SLOT_COUNT} slots"
        );
        out
    }

    /// The schema version the compiled header speaks.
    pub fn schema_version() -> u32 {
        unsafe { prism_tsh_arm_schema_version() }
    }

    /// An owned device-layout buffer for one of the arm's POD structs.
    ///
    /// 16-byte aligned, because every one of those structs is `__align__(16)`
    /// and a misaligned buffer would be undefined behaviour on the C++ side.
    pub struct DeviceBuffer {
        raw: Vec<u128>,
        len: usize,
    }

    impl DeviceBuffer {
        /// A zeroed buffer of `len` bytes, aligned to 16.
        pub fn zeroed(len: usize) -> Self {
            Self {
                raw: vec![0u128; len.div_ceil(16)],
                len,
            }
        }

        /// A buffer initialised from `bytes`.
        pub fn from_bytes(bytes: &[u8]) -> Self {
            let mut b = Self::zeroed(bytes.len());
            b.as_mut_slice().copy_from_slice(bytes);
            b
        }

        /// The buffer's bytes.
        pub fn as_slice(&self) -> &[u8] {
            unsafe { std::slice::from_raw_parts(self.raw.as_ptr().cast::<u8>(), self.len) }
        }

        /// The buffer's bytes, mutably.
        pub fn as_mut_slice(&mut self) -> &mut [u8] {
            unsafe { std::slice::from_raw_parts_mut(self.raw.as_mut_ptr().cast::<u8>(), self.len) }
        }

        fn ptr(&self) -> *const std::ffi::c_void {
            self.raw.as_ptr().cast()
        }

        fn ptr_mut(&mut self) -> *mut std::ffi::c_void {
            self.raw.as_mut_ptr().cast()
        }
    }

    /// Fill `policy` with the device's documented defaults, UNARMED.
    pub fn default_policy(policy: &mut DeviceBuffer) {
        assert_eq!(policy.len, DEVICE_POLICY_BYTES);
        unsafe { prism_tsh_arm_host_default_policy(policy.ptr_mut()) }
    }

    /// Reset `state` to its pre-decision baseline at simulated time `t0_ps`.
    pub fn reset(state: &mut DeviceBuffer, t0_ps: f64) {
        assert_eq!(state.len, DEVICE_STATE_BYTES);
        unsafe { prism_tsh_arm_host_reset(state.ptr_mut(), t0_ps) }
    }

    /// Stage one branch observation. `true` when staged, `false` when refused.
    pub fn observe(
        policy: &DeviceBuffer,
        state: &mut DeviceBuffer,
        branch: SalienceBranch,
        accumulated_activity: f64,
        span_ps: f64,
        window_end_ps: f64,
        sample_count: u32,
    ) -> bool {
        let p = policy.ptr();
        unsafe {
            prism_tsh_arm_host_observe(
                p,
                state.ptr_mut(),
                branch.index() as u32,
                accumulated_activity,
                span_ps,
                window_end_ps,
                sample_count,
            ) != 0
        }
    }

    /// Take one decision, returning the emitted record.
    pub fn decide(
        policy: &DeviceBuffer,
        state: &mut DeviceBuffer,
        now_ps: f64,
        global_md_step: u64,
        frame: u32,
        authorized_gear: Option<u32>,
    ) -> TshArmRecordAbi {
        let mut rec = TshArmRecordAbi::unwritten();
        let p = policy.ptr();
        unsafe {
            prism_tsh_arm_host_decide(
                p,
                state.ptr_mut(),
                now_ps,
                global_md_step,
                frame,
                authorized_gear.unwrap_or(0),
                u32::from(authorized_gear.is_some()),
                std::ptr::null_mut(),
                std::ptr::from_mut(&mut rec).cast(),
            );
        }
        rec
    }

    /// The device's per-branch observation coverage in `[0,1]`.
    pub fn branch_coverage(
        policy: &DeviceBuffer,
        state: &DeviceBuffer,
        branch: SalienceBranch,
    ) -> f64 {
        unsafe {
            prism_tsh_arm_host_branch_coverage(policy.ptr(), state.ptr(), branch.index() as u32)
        }
    }

    /// The device's UNWRITTEN ring-slot pattern.
    pub fn record_unwritten() -> TshArmRecordAbi {
        let mut rec = TshArmRecordAbi::unwritten();
        unsafe { prism_tsh_arm_host_record_unwritten(std::ptr::from_mut(&mut rec).cast()) };
        rec
    }

    /// The device's validity predicate for a ring slot.
    pub fn record_is_valid(rec: &TshArmRecordAbi) -> bool {
        unsafe { prism_tsh_arm_host_record_is_valid(std::ptr::from_ref(rec).cast()) != 0 }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ps(v: f64) -> SimulatedPs {
        SimulatedPs::new(v).expect("finite")
    }

    fn span(v: f64) -> SimulatedPsSpan {
        SimulatedPsSpan::new(v).expect("positive finite")
    }

    /// A policy with anti-blindness knobs pushed out of the way, so a test can
    /// isolate one property at a time. Resolution goes through the production
    /// document path, not a bypass.
    fn policy_with(doc: TshArmPolicyDocument) -> TshArmPolicy {
        TshArmPolicy::from_document(&doc, ValueOrigin::PolicyFile, None).expect("valid policy")
    }

    fn quiet_sentinel_doc() -> TshArmPolicyDocument {
        TshArmPolicyDocument {
            // Far enough out that the sentinel does not confound the property
            // under test. The sentinel itself is tested separately.
            max_sentinel_interval_ps: Some(1.0e6),
            ..Default::default()
        }
    }

    fn clock(t: f64, step: u64, authorized: Option<u32>) -> DecisionClock {
        DecisionClock {
            cumulative_simulated_time: ps(t),
            global_md_step: step,
            frame: step as u32,
            authorized_gear: authorized,
        }
    }

    /// The branches a DEFAULT policy admits.
    ///
    /// Five, not six: [`SalienceBranch::SignalGridWarp`] is unbound by default
    /// and `observe` refuses it, because the engine's only "ASC" product is the
    /// force buffer under another filename. A test that iterated all six and
    /// unwrapped the result would be asserting the double-count is accepted.
    const BOUND_BRANCHES: [SalienceBranch; 5] = [
        SalienceBranch::ForceEnergyStability,
        SalienceBranch::KccCoupling,
        SalienceBranch::ProbeInteraction,
        SalienceBranch::GeometryContact,
        SalienceBranch::PriorGearboxInstability,
    ];

    const _: () = {
        assert!(BOUND_BRANCHES.len() + 1 == TSH_ARM_N_BRANCHES);
    };

    /// The module's own source, with this test module removed.
    ///
    /// The source-scanning tests below assert that the ARM contains no
    /// authorization path. The tests themselves legitimately name the things
    /// the arm must not do — that is what makes them tests — so scanning the
    /// whole file would make them fail on their own assertions.
    fn arm_source_without_tests() -> &'static str {
        const SOURCE: &str = include_str!("tsh_gearbox_arm.rs");
        const MARKER: &str = "#[cfg(test)]\nmod tests {";
        let end = SOURCE
            .find(MARKER)
            .expect("the test module marker must be findable, or the scan is vacuous");
        &SOURCE[..end]
    }

    /// Whether `line` is a comment. The authority boundary is DOCUMENTED by
    /// naming what it excludes, so a comment mentioning a forbidden symbol is
    /// the point, not a breach.
    fn is_comment(line: &str) -> bool {
        let t = line.trim_start();
        t.starts_with("//") || t.starts_with("/*") || t.starts_with('*')
    }

    // ───────────────────────────────────────────────────────────────────────
    // THE anti-blindness property: salience is a property of the physics, not
    // of the sampling rate.
    //
    // The two tests immediately below vary the SAMPLE COUNT and the DECISION
    // RATE, which the first revision already handled — they hold
    // `span == dt` in every case, so they never exercised truncation or the
    // uncovered remainder. The two after them are the ones that failed: they
    // vary the DUTY CYCLE and the OBSERVATION CADENCE at a FIXED decision
    // cadence, which is where "absence of evidence integrated as measured
    // calm" actually shows up.
    // ───────────────────────────────────────────────────────────────────────

    /// Steady-state salience for a constant activity rate `r` against a unit
    /// reference: `r/(r+1)`. The number the physics implies, independent of any
    /// observer.
    fn steady_state_salience(rate: f64) -> f64 {
        rate / (rate + DEFAULT_BRANCH_REFERENCE_RATE)
    }

    /// Drive ONE branch at a constant activity rate for `total_ps` of simulated
    /// time, taking a decision every `decision_ps` and covering `duty` of each
    /// decision interval with evidence. Returns the last request.
    ///
    /// `duty < 1.0` is the adversarial case: the physics is identical, the
    /// observer simply looked at less of it.
    fn drive_constant_physics(
        policy: TshArmPolicy,
        rate: f64,
        decision_ps: f64,
        duty: f64,
        total_ps: f64,
    ) -> (TshGearboxArm, TshGearRequest) {
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let mut t = 0.0f64;
        let mut step = 0u64;
        let mut last = None;
        while t + decision_ps <= total_ps + 1e-12 {
            let covered = decision_ps * duty;
            if covered > 0.0 {
                arm.observe(BranchObservation {
                    branch: SalienceBranch::ForceEnergyStability,
                    accumulated_activity: rate * covered,
                    span: span(covered),
                    window_end: ps(t + covered),
                    sample_count: 1,
                })
                .expect("staged");
            }
            t += decision_ps;
            step += 1;
            last = Some(arm.decide(clock(t, step, Some(GEAR_SAFE))));
        }
        (arm, last.expect("at least one decision"))
    }

    #[test]
    fn salience_is_invariant_to_observation_duty_cycle() {
        // THE property that stops the controller coarsening itself into
        // blindness. Same physics, same decision cadence, four observers with
        // wildly different appetites for looking.
        //
        // With the integrator decayed over uncovered time — the first revision
        // — these read 0.5000 / 0.3045 / 0.1395 / 0.0376, and the 20% observer
        // crossed below the 0.25 upshift threshold into COARSENING ELIGIBLE on
        // physics that had not changed at all.
        let policy = policy_with(quiet_sentinel_doc());
        let truth = steady_state_salience(1.0);
        assert!((truth - 0.5).abs() < 1e-15);

        // Stated precisely: at constant physics the salience is a function of
        // the SIMULATED TIME OBSERVED, and of nothing else. So each observer is
        // run until it has covered the same 40 ps of evidence — 20 time
        // constants at the default tau — and all four must land on the same
        // number. Under the decaying integrator they landed on 0.5000, 0.3045,
        // 0.1395 and 0.0376 instead.
        let mut observed = Vec::new();
        for duty in [1.0, 0.5, 0.2, 0.05] {
            let covered_target_ps = 40.0;
            let (_, r) =
                drive_constant_physics(policy.clone(), 1.0, 0.25, duty, covered_target_ps / duty);
            let s = r.peak_branch_salience.expect("measured");
            observed.push((duty, s));
            assert!(
                (s - truth).abs() < 1e-9,
                "duty cycle {duty} moved the salience of CONSTANT physics at equal \
                 observed time: {s} vs {truth}"
            );
        }
        // All four agree with each other, not merely with the closed form.
        for w in observed.windows(2) {
            assert!(
                (w[0].1 - w[1].1).abs() < 1e-9,
                "duty {} and duty {} disagree: {} vs {}",
                w[0].0,
                w[1].0,
                w[0].1,
                w[1].1
            );
        }

        // And the harm the defect actually caused: at EQUAL WALL TIME, a thin
        // observer must not read as QUIET on physics that is not quiet. This is
        // the assertion that fails the moment uncovered time is integrated as
        // measured calm — the 20% observer used to report 0.1395, below the
        // 0.25 upshift threshold, i.e. eligible to coarsen itself blinder.
        for duty in [1.0, 0.5, 0.2, 0.05] {
            let (_, r) = drive_constant_physics(policy.clone(), 1.0, 0.25, duty, 40.0);
            let s = r.peak_branch_salience.expect("measured");
            assert!(
                s >= DEFAULT_SALIENCE_UPSHIFT,
                "duty cycle {duty} pushed identical physics below the upshift \
                 threshold ({s} < {DEFAULT_SALIENCE_UPSHIFT})"
            );
        }

        // And the arm knows the difference: what a thin duty cycle costs is
        // COVERAGE — i.e. CONFIDENCE — which is the honest place for it. The
        // salience says what the physics is doing; the coverage says how much
        // of it was actually watched. Conflating the two is the defect.
        let (full_arm, full) = drive_constant_physics(policy.clone(), 1.0, 0.25, 1.0, 40.0);
        let (thin_arm, thin) = drive_constant_physics(policy, 1.0, 0.25, 0.2, 40.0);
        let b = SalienceBranch::ForceEnergyStability.index();
        let full_cov = full_arm.branch_coverage(b);
        let thin_cov = thin_arm.branch_coverage(b);
        assert!(
            (full_cov - 1.0).abs() < 1e-6,
            "continuous observation must reach full coverage: {full_cov}"
        );
        // Coverage tracks the duty cycle, sampled at the TROUGH of its ripple.
        //
        // The integrator is driven by a square wave — 1 while evidence covers
        // the interval, 0 while it does not — and a decision lands at the end
        // of the uncovered part, which is the low point of the cycle. For
        // c = 0.05 ps covered out of T = 0.25 ps at tau = 2.0 ps the fixed
        // point is tau*(1-e^{-c/tau})*e^{-(T-c)/tau} / (1-e^{-T/tau}) / tau
        // = 0.19013, against a duty cycle of 0.200. Sampling the trough is the
        // conservative direction: the arm under-reports how well it is
        // observing, never over-reports.
        assert!(
            thin_cov <= 0.2 + 1e-9 && (thin_cov - 0.2).abs() < 0.02,
            "coverage must track the duty cycle from below: {thin_cov} vs 0.2"
        );
        // Salience unchanged, coverage collapsed — exactly the separation the
        // architecture requires.
        let full_s = full.peak_branch_salience.expect("measured");
        let thin_s = thin.peak_branch_salience.expect("measured");
        // At EQUAL WALL TIME the thin observer has integrated one fifth as much
        // evidence, so it is one fifth of the way along the same exponential
        // approach — 8 ps of covered evidence is 4 tau, leaving e^-4 = 1.8% to
        // go. That residual is convergence, not duty-cycle dependence: the
        // exact-equality assertion at equal OBSERVED time is above. What
        // matters is the magnitude — the decaying integrator put this at
        // 0.1395, a 72% error in the direction of "coarsen".
        assert!(
            (full_s - thin_s).abs() < 0.02,
            "salience tracked the duty cycle: {full_s} vs {thin_s}"
        );
        assert!(
            (thin_s - truth).abs() < 0.02,
            "a 20% observer must still measure the physics: {thin_s} vs {truth}"
        );
        assert!(
            thin.flags & FLAG_OBSERVATION_SPARSE != 0,
            "a 20% duty cycle must be recorded as sparse observation"
        );
        assert_eq!(observed.len(), 4);
    }

    #[test]
    fn salience_is_invariant_to_observation_cadence() {
        // The other half of the same defect: an observation window LONGER than
        // the decision interval had its surplus discarded, so evidence was
        // thrown away every decision and the loss read as quiet.
        //
        // Here the decision cadence is fixed at 0.25 ps and the OBSERVATION
        // cadence varies from 0.25 ps up to 4.0 ps. Every observation covers
        // its full interval, so the physics seen is identical in all four runs.
        let policy = policy_with(quiet_sentinel_doc());
        let truth = steady_state_salience(1.0);

        let run = |obs_ps: f64| -> TshGearRequest {
            let mut arm = TshGearboxArm::new(policy.clone(), ps(0.0));
            let decision_ps = 0.25f64;
            let total_ps = 60.0f64;
            let mut t = 0.0f64;
            let mut step = 0u64;
            let mut next_obs = obs_ps;
            let mut last = None;
            while t + decision_ps <= total_ps + 1e-12 {
                t += decision_ps;
                step += 1;
                if t >= next_obs - 1e-12 {
                    arm.observe(BranchObservation {
                        branch: SalienceBranch::ForceEnergyStability,
                        accumulated_activity: 1.0 * obs_ps,
                        span: span(obs_ps),
                        window_end: ps(t),
                        sample_count: 1,
                    })
                    .expect("staged");
                    next_obs += obs_ps;
                }
                last = Some(arm.decide(clock(t, step, Some(GEAR_SAFE))));
            }
            last.expect("at least one decision")
        };

        for obs_ps in [0.25f64, 1.0, 2.0, 4.0] {
            let r = run(obs_ps);
            let s = r.peak_branch_salience.expect("measured");
            assert!(
                (s - truth).abs() < 5e-3,
                "observation cadence {obs_ps} ps moved the salience of CONSTANT physics: \
                 {s} vs {truth}"
            );
            assert!(
                s >= DEFAULT_SALIENCE_UPSHIFT,
                "cadence {obs_ps} ps pushed identical physics below the upshift threshold \
                 ({s} < {DEFAULT_SALIENCE_UPSHIFT}) — the controller is coarsening itself blind"
            );
        }
    }

    #[test]
    fn a_silent_branch_is_not_a_calm_branch() {
        // Two branches, identical history, then one REPORTS calm and the other
        // says nothing. They must not converge to the same answer: one is a
        // measurement, the other is an absence.
        let policy = policy_with(quiet_sentinel_doc());

        let (mut reporting, _) = drive_constant_physics(policy.clone(), 4.0, 0.25, 1.0, 20.0);
        let mut silent = reporting.clone();

        // The reporting branch now says "zero activity over real simulated
        // time" — that is evidence of calm and must drive the integrator down.
        let mut t = 20.0f64;
        for step in 100..140u64 {
            reporting
                .observe(BranchObservation {
                    branch: SalienceBranch::ForceEnergyStability,
                    accumulated_activity: 0.0,
                    span: span(0.25),
                    window_end: ps(t + 0.25),
                    sample_count: 1,
                })
                .expect("staged");
            t += 0.25;
            reporting.decide(clock(t, step, Some(GEAR_SAFE)));
        }

        // The silent branch says nothing at all over the same span.
        let mut t2 = 20.0f64;
        let mut silent_last = None;
        for step in 100..140u64 {
            t2 += 0.25;
            silent_last = Some(silent.decide(clock(t2, step, Some(GEAR_SAFE))));
        }
        let silent_last = silent_last.expect("decided");

        let reported_activity = reporting.state().branch[0].activity;
        let silent_activity = silent.state().branch[0].activity;
        assert!(
            reported_activity < silent_activity * 0.5,
            "a branch REPORTING calm must decay; got reported={reported_activity} \
             silent={silent_activity}"
        );
        // And the silent one is not permitted to look confident about it.
        assert!(
            silent_last.observation_confidence.expect("weights") < 0.5,
            "silence must cost confidence"
        );
    }

    #[test]
    fn normalisation_is_per_simulated_ps_not_per_sample() {
        let policy = policy_with(quiet_sentinel_doc());

        // Identical physics — 2.0 activity units over 1.0 ps of simulated time
        // — delivered as 1 sample, then as 8, then as 64.
        let mut activities = Vec::new();
        for n in [1usize, 8, 64] {
            let mut arm = TshGearboxArm::new(policy.clone(), ps(0.0));
            let chunk = 1.0 / n as f64;
            for i in 0..n {
                arm.observe(BranchObservation {
                    branch: SalienceBranch::ForceEnergyStability,
                    accumulated_activity: 2.0 / n as f64,
                    span: span(chunk),
                    window_end: ps((i + 1) as f64 * chunk),
                    sample_count: 1,
                })
                .expect("staged");
            }
            let request = arm.decide(clock(1.0, 100, Some(GEAR_SAFE)));
            activities.push((
                arm.state().branch[0].activity,
                request.integrated_salience.expect("measured"),
            ));
        }

        // Exact, not approximate: the leaky integrator's solution composes
        // under subdivision, so more samples at constant physics changes
        // nothing at all. If this ever becomes approximate, the normalisation
        // has silently acquired a per-record term.
        assert_eq!(activities[0].0, activities[1].0);
        assert_eq!(activities[0].0, activities[2].0);
        assert_eq!(activities[0].1, activities[1].1);
        assert_eq!(activities[0].1, activities[2].1);

        // And the sample_count field genuinely does not participate.
        let mut arm_a = TshGearboxArm::new(policy.clone(), ps(0.0));
        arm_a
            .observe(BranchObservation {
                branch: SalienceBranch::ForceEnergyStability,
                accumulated_activity: 2.0,
                span: span(1.0),
                window_end: ps(1.0),
                sample_count: 1,
            })
            .expect("staged");
        let mut arm_b = TshGearboxArm::new(policy, ps(0.0));
        arm_b
            .observe(BranchObservation {
                branch: SalienceBranch::ForceEnergyStability,
                accumulated_activity: 2.0,
                span: span(1.0),
                window_end: ps(1.0),
                sample_count: 4096,
            })
            .expect("staged");
        let ra = arm_a.decide(clock(1.0, 100, Some(GEAR_SAFE)));
        let rb = arm_b.decide(clock(1.0, 100, Some(GEAR_SAFE)));
        assert_eq!(ra.integrated_salience, rb.integrated_salience);
    }

    #[test]
    fn normalisation_is_invariant_to_decision_rate() {
        let policy = policy_with(quiet_sentinel_doc());

        // The same constant-rate physics over 4.0 ps, decided once vs decided
        // 16 times. A per-record normalisation would show a difference here;
        // a per-simulated-ps one cannot.
        let mut arm_coarse = TshGearboxArm::new(policy.clone(), ps(0.0));
        arm_coarse
            .observe(BranchObservation {
                branch: SalienceBranch::KccCoupling,
                accumulated_activity: 3.0 * 4.0,
                span: span(4.0),
                window_end: ps(4.0),
                sample_count: 1,
            })
            .expect("staged");
        let coarse = arm_coarse.decide(clock(4.0, 400, Some(GEAR_SAFE)));

        let mut arm_fine = TshGearboxArm::new(policy, ps(0.0));
        for i in 0..16 {
            let t0 = i as f64 * 0.25;
            let t1 = t0 + 0.25;
            arm_fine
                .observe(BranchObservation {
                    branch: SalienceBranch::KccCoupling,
                    accumulated_activity: 3.0 * 0.25,
                    span: span(0.25),
                    window_end: ps(t1),
                    sample_count: 1,
                })
                .expect("staged");
            arm_fine.decide(clock(t1, (i as u64 + 1) * 25, Some(GEAR_SAFE)));
        }
        let fine_activity = arm_fine.state().branch[SalienceBranch::KccCoupling.index()].activity;
        let coarse_activity =
            arm_coarse.state().branch[SalienceBranch::KccCoupling.index()].activity;

        // Float rounding across 16 compositions vs 1 is the only admissible
        // difference.
        assert!(
            (fine_activity - coarse_activity).abs() <= 1e-12 * coarse_activity.abs().max(1.0),
            "decision-rate dependence: {fine_activity} vs {coarse_activity}"
        );
        let fine_salience = arm_fine
            .state()
            .branch
            .get(SalienceBranch::KccCoupling.index())
            .map(|b| {
                let r = b.activity / DEFAULT_BRANCH_TAU_PS;
                r / (r + DEFAULT_BRANCH_REFERENCE_RATE)
            })
            .expect("branch exists");
        assert!(
            (fine_salience - coarse.integrated_salience.expect("measured")).abs() < 1e-12,
            "salience drifted with decision rate"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Anti-blindness (b): the sentinel fires on schedule.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn sentinel_sampling_fires_on_schedule_regardless_of_salience() {
        // Sentinel every 3.0 ps; keep every branch perfectly calm and fully
        // observed so nothing else could possibly demand fine resolution.
        let policy = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(3.0),
            sentinel_hold_ps: Some(0.05),
            ..Default::default()
        });
        let mut arm = TshGearboxArm::new(policy, ps(0.0));

        let mut sentinel_times = Vec::new();
        let mut t = 0.0f64;
        for step in 1..=60u64 {
            t += 0.25;
            for branch in BOUND_BRANCHES {
                arm.observe(BranchObservation {
                    branch,
                    accumulated_activity: 0.0,
                    span: span(0.25),
                    window_end: ps(t),
                    sample_count: 1,
                })
                .expect("staged");
            }
            // The governor never grants fine, so the sentinel clock is never
            // reset by a measured sample — the arm must keep asking.
            let r = arm.decide(clock(t, step, Some(GEAR_CAMPAIGN)));
            if r.sentinel_forced() {
                sentinel_times.push(t);
                assert_eq!(r.requested_gear, Some(GEAR_FINE));
                assert_eq!(r.request_reason, RequestReason::Sentinel);
            }
        }
        assert!(
            !sentinel_times.is_empty(),
            "anti-blindness (b) never fired in 15 ps with a 3 ps interval"
        );
        // First one lands at the first decision at or past the interval.
        assert!(
            (sentinel_times[0] - 3.0).abs() < 1e-9,
            "first sentinel at {} ps, expected 3.0",
            sentinel_times[0]
        );

        // And it resets on a MEASURED fine gear, not on the request.
        let policy2 = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(3.0),
            ..Default::default()
        });
        let mut arm2 = TshGearboxArm::new(policy2, ps(0.0));
        for branch in BOUND_BRANCHES {
            arm2.observe(BranchObservation {
                branch,
                accumulated_activity: 0.0,
                span: span(3.5),
                window_end: ps(3.5),
                sample_count: 1,
            })
            .expect("staged");
        }
        let granted = arm2.decide(clock(3.5, 1, Some(GEAR_FINE)));
        assert!(
            !granted.sentinel_forced(),
            "a measured fine gear must reset the sentinel clock"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Anti-blindness (d): the confidence gate blocks coarsening.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn confidence_gate_blocks_coarsening() {
        let policy = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(1.0e6),
            min_gear_dwell_ps: Some(0.0),
            decay_period_ps: Some(0.0),
            ..Default::default()
        });

        // ONE branch reports calm; the other four weighted branches have never
        // been bound. Confidence is at most 1/5, far below the 0.75 gate. A
        // mean that treated "unbound" as "calm" would coarsen here — that is
        // exactly the blindness this gate exists to prevent.
        let mut arm = TshGearboxArm::new(policy.clone(), ps(0.0));
        let mut t = 0.0;
        let mut coarsened = false;
        for step in 1..=20u64 {
            t += 0.25;
            arm.observe(BranchObservation {
                branch: SalienceBranch::ForceEnergyStability,
                accumulated_activity: 0.0,
                span: span(0.25),
                window_end: ps(t),
                sample_count: 1,
            })
            .expect("staged");
            let r = arm.decide(clock(t, step, Some(GEAR_SAFE)));
            assert!(
                r.observation_confidence.expect("weights present") < 0.75,
                "four unbound branches must not read as confident"
            );
            if r.request_reason == RequestReason::UpshiftEligible {
                coarsened = true;
            }
            if r.coarsening_blocked_by_confidence() {
                assert_eq!(r.request_reason, RequestReason::ConfidenceGate);
                assert_eq!(r.requested_gear, Some(GEAR_SAFE), "gear must hold");
            }
        }
        assert!(!coarsened, "coarsened below the confidence threshold");

        // With every weighted branch reporting calm AND enough simulated time
        // observed for the coverage integrator to charge, the same arm is
        // permitted to ask for coarser. This proves the gate is the binding
        // constraint above and not some other refusal.
        //
        // The warm-up is not incidental: coverage is `1 - exp(-t/tau)` under
        // full observation, so reaching the 0.75 gate takes 1.386 tau = 2.77 ps
        // of continuously observed simulated time. An arm that has just started
        // has not yet verified its own observability and must not coarsen —
        // that is anti-blindness (d) doing its job at t=0, not a bug.
        let mut arm_full = TshGearboxArm::new(policy, ps(0.0));
        let mut t = 0.0;
        let mut ever_coarsened = false;
        let mut peak_confidence = 0.0f64;
        for step in 1..=200u64 {
            t += 0.25;
            for branch in BOUND_BRANCHES {
                arm_full
                    .observe(BranchObservation {
                        branch,
                        accumulated_activity: 0.0,
                        span: span(0.25),
                        window_end: ps(t),
                        sample_count: 1,
                    })
                    .expect("staged");
            }
            let r = arm_full.decide(clock(t, step, Some(GEAR_SAFE)));
            let c = r.observation_confidence.expect("weights present");
            assert!((0.0..=1.0).contains(&c), "confidence out of range: {c}");
            peak_confidence = peak_confidence.max(c);
            if r.request_reason == RequestReason::UpshiftEligible {
                ever_coarsened = true;
                assert!(
                    r.observation_coverage.expect("weights present")
                        >= DEFAULT_MIN_CONFIDENCE_FOR_COARSENING - 1e-12,
                    "coarsened while under-observed"
                );
            }
        }
        assert!(
            peak_confidence > 0.99,
            "continuous full-coverage observation must approach confidence 1.0, got {peak_confidence}"
        );
        assert!(
            ever_coarsened,
            "fully observed calm must eventually be allowed to REQUEST coarser"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Memory semantics: rechallenge is immediate.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn rechallenge_triggers_immediate_downshift() {
        let policy = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(1.0e6),
            reinforce_dwell_ps: Some(0.0),
            withdrawal_hold_ps: Some(0.5),
            decay_period_ps: Some(0.5),
            rechallenge_window_ps: Some(50.0),
            ..Default::default()
        });
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let mut t = 0.0f64;

        let drive = |arm: &mut TshGearboxArm, t: &mut f64, rate: f64, steps: usize| {
            let mut last = None;
            for _ in 0..steps {
                let t0 = *t;
                *t += 0.25;
                for branch in BOUND_BRANCHES {
                    arm.observe(BranchObservation {
                        branch,
                        accumulated_activity: rate * (*t - t0),
                        span: span(*t - t0),
                        window_end: ps(*t),
                        sample_count: 1,
                    })
                    .expect("staged");
                }
                last = Some(arm.decide(clock(*t, (*t * 100.0) as u64, Some(GEAR_SAFE))));
            }
            last.expect("at least one decision")
        };

        // Drive hard enough to reach active memory. Steady-state salience for
        // a constant input rate r is r/(r+1); r = 20 gives ~0.952 > 0.85.
        let r = drive(&mut arm, &mut t, 20.0, 40);
        assert!(
            matches!(
                r.memory_state,
                TshMemoryState::ActiveMemory | TshMemoryState::Reinforced
            ),
            "expected active/reinforced, got {:?}",
            r.memory_state
        );

        // Withdraw the input and let it decay out of the fine-holding states.
        let mut decayed = None;
        for _ in 0..200 {
            let r = drive(&mut arm, &mut t, 0.0, 1);
            if r.memory_state == TshMemoryState::Decay {
                decayed = Some(r);
                break;
            }
        }
        let decayed = decayed.expect("memory must reach decay after withdrawal");
        assert!(arm.state().rechallenge_armed, "rechallenge window must arm");
        assert!(
            !matches!(
                decayed.request_reason,
                RequestReason::MemoryHoldFine | RequestReason::Rechallenge
            ),
            "decay must not still be demanding fine: {:?}",
            decayed.request_reason
        );

        // Let the decayed memory actually earn a coarser gear, so the
        // rechallenge below has somewhere to fall FROM. Otherwise the test
        // would pass trivially with the arm already pinned fine.
        for _ in 0..400 {
            if arm.state().last_requested_gear > GEAR_FINE {
                break;
            }
            drive(&mut arm, &mut t, 0.0, 1);
        }
        assert!(
            arm.state().last_requested_gear > GEAR_FINE,
            "quiet, fully observed, decayed memory never became coarsening-eligible"
        );

        // A single renewed excursion must be answered IMMEDIATELY with the
        // finest gear — not a one-step downshift, and not after a dwell.
        let before = arm.state().last_requested_gear;
        let re = drive(&mut arm, &mut t, 20.0, 1);
        assert_eq!(re.memory_state, TshMemoryState::Rechallenge);
        assert_eq!(re.request_reason, RequestReason::Rechallenge);
        assert_eq!(re.requested_gear, Some(GEAR_FINE));
        assert!(re.flags & FLAG_RECHALLENGE != 0);
        assert!(
            before > GEAR_FINE,
            "the rechallenge must have crossed more than zero gears"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Hysteresis: no chatter under a noisy input.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn hysteresis_prevents_chatter_under_noisy_input() {
        let policy = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(1.0e6),
            ..Default::default()
        });
        let mut arm = TshGearboxArm::new(policy, ps(0.0));

        // A deterministic input oscillating right across the hysteresis band:
        // the steady-state salience for rate r is r/(r+1), so r swinging
        // between 0.30 and 0.90 straddles both 0.25 and 0.60.
        let mut t = 0.0f64;
        let mut changes = 0usize;
        let mut previous: Option<u32> = None;
        let mut reasons_seen = Vec::new();
        for step in 1..=400u64 {
            let t0 = t;
            t += 0.05;
            let phase = (step % 20) as f64 / 20.0;
            let rate = if phase < 0.5 { 0.30 } else { 0.90 };
            for branch in BOUND_BRANCHES {
                arm.observe(BranchObservation {
                    branch,
                    accumulated_activity: rate * (t - t0),
                    span: span(t - t0),
                    window_end: ps(t),
                    sample_count: 1,
                })
                .expect("staged");
            }
            let r = arm.decide(clock(t, step, Some(GEAR_SAFE)));
            let gear = r.requested_gear.expect("armed");
            if let Some(p) = previous {
                if p != gear {
                    changes += 1;
                }
            }
            previous = Some(gear);
            reasons_seen.push(r.request_reason);
        }

        // 400 decisions across 20 full oscillations of the driving input. An
        // unhysteretic controller would flip on every crossing (>= 40 changes).
        assert!(
            changes <= 4,
            "gear chattered {changes} times under a noisy input; hysteresis is not binding"
        );
        // And the dwell gate must have been the thing that refused at least
        // once, otherwise this test passed for an unrelated reason.
        assert!(
            reasons_seen.iter().any(|r| matches!(
                r,
                RequestReason::DwellGate
                    | RequestReason::MemoryHoldFine
                    | RequestReason::MemoryThresholdDownshift
                    | RequestReason::Hold
            )),
            "no holding reason ever appeared"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Structural invariants.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn unbound_branch_is_not_calm_and_raises_uncertainty() {
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));

        // Nothing has ever reported.
        let r = arm.decide(clock(1.0, 10, Some(GEAR_CAMPAIGN)));
        assert_eq!(
            r.integrated_salience, None,
            "no evidence must be UNMEASURED, never Some(0.0)"
        );
        assert_eq!(r.observation_confidence, Some(0.0));
        assert_eq!(
            r.request_confidence,
            Some(0.0),
            "a fail-safe default is not an inference"
        );
        assert_eq!(r.request_reason, RequestReason::EvidenceAbsent);
        assert_eq!(r.requested_gear, Some(GEAR_FINE), "bias fine when blind");
        assert!(r.flags & FLAG_EVIDENCE_ABSENT != 0);
        assert_eq!(r.active_branch_mask.unbound, 0b0011_1111);
        assert_eq!(r.active_branch_mask.fresh, 0);
    }

    #[test]
    fn uncertainty_grows_when_evidence_is_sparse() {
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        for branch in BOUND_BRANCHES {
            arm.observe(BranchObservation {
                branch,
                accumulated_activity: 0.5,
                span: span(0.25),
                window_end: ps(0.25),
                sample_count: 1,
            })
            .expect("staged");
        }
        let fresh = arm.decide(clock(0.25, 1, Some(GEAR_SAFE)));
        // Not 1.0: one decision covering 0.25 ps has charged the coverage
        // integrator to `1 - exp(-0.25/2.0)` = 0.1175. Full confidence is
        // something a run EARNS by observing, not something it starts with.
        let fresh_c = fresh.observation_confidence.expect("weights present");
        assert!(
            fresh_c > 0.0 && fresh_c < 1.0,
            "a single short window must be neither blind nor fully confident: {fresh_c}"
        );

        // Let simulated time pass with no evidence at all. Confidence must
        // fall monotonically — the arm must know it is going blind.
        let mut last = fresh_c;
        let mut t = 0.25;
        for step in 2..=20u64 {
            t += 0.5;
            let c = arm
                .decide(clock(t, step, Some(GEAR_SAFE)))
                .observation_confidence
                .expect("weights present");
            assert!(c <= last + 1e-15, "confidence rose without new evidence");
            last = c;
        }
        assert!(last < 0.1, "confidence did not decay: {last}");
    }

    #[test]
    fn arm_never_requests_the_abort_gear() {
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let mut t = 0.0;
        for step in 1..=200u64 {
            let t0 = t;
            t += 0.1;
            for branch in BOUND_BRANCHES {
                arm.observe(BranchObservation {
                    branch,
                    accumulated_activity: (step as f64 % 7.0) * (t - t0),
                    span: span(t - t0),
                    window_end: ps(t),
                    sample_count: step as u32,
                })
                .expect("staged");
            }
            // Feed back an abort gear as the "authorized" value; the arm must
            // still never request one.
            let r = arm.decide(clock(t, step, Some(3)));
            let gear = r.requested_gear.expect("armed");
            assert!(gear <= GEAR_MAX_ACTIVE, "requested gear {gear}");
            assert_ne!(gear, 3, "the arm requested the abort gear");
        }
    }

    #[test]
    fn emits_on_no_change_with_a_gapless_sequence() {
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let mut expected = 0u64;
        let mut t = 0.0;
        for step in 1..=25u64 {
            t += 0.25;
            for branch in BOUND_BRANCHES {
                arm.observe(BranchObservation {
                    branch,
                    accumulated_activity: 0.25,
                    span: span(0.25),
                    window_end: ps(t),
                    sample_count: 1,
                })
                .expect("staged");
            }
            let r = arm.decide(clock(t, step, Some(GEAR_SAFE)));
            expected += 1;
            assert_eq!(
                r.record_sequence, expected,
                "a decision that changed nothing must still emit"
            );
        }
    }

    #[test]
    fn checkpoint_restore_reproduces_the_salience_trajectory() {
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy.clone(), ps(0.0));
        let mut t = 0.0;
        for step in 1..=12u64 {
            t += 0.25;
            for branch in BOUND_BRANCHES {
                arm.observe(BranchObservation {
                    branch,
                    accumulated_activity: (step as f64).sin().abs() * 0.25,
                    span: span(0.25),
                    window_end: ps(t),
                    sample_count: 1,
                })
                .expect("staged");
            }
            arm.decide(clock(t, step, Some(GEAR_SAFE)));
        }
        let checkpoint = arm.checkpoint();
        let json = serde_json::to_string(&checkpoint).expect("checkpoint serialises");
        let decoded: TshArmCheckpoint = serde_json::from_str(&json).expect("round trip");

        // Continue the original.
        let mut continued = Vec::new();
        let mut t_a = t;
        let mut arm_a = arm.clone();
        for step in 13..=20u64 {
            t_a += 0.25;
            for branch in BOUND_BRANCHES {
                arm_a
                    .observe(BranchObservation {
                        branch,
                        accumulated_activity: (step as f64).sin().abs() * 0.25,
                        span: span(0.25),
                        window_end: ps(t_a),
                        sample_count: 1,
                    })
                    .expect("staged");
            }
            continued.push(arm_a.decide(clock(t_a, step, Some(GEAR_SAFE))));
        }

        // Restore into a fresh arm and continue identically.
        let mut arm_b = TshGearboxArm::new(policy, ps(0.0));
        arm_b.restore(&decoded).expect("restore accepted");
        let mut replayed = Vec::new();
        let mut t_b = t;
        for step in 13..=20u64 {
            t_b += 0.25;
            for branch in BOUND_BRANCHES {
                arm_b
                    .observe(BranchObservation {
                        branch,
                        accumulated_activity: (step as f64).sin().abs() * 0.25,
                        span: span(0.25),
                        window_end: ps(t_b),
                        sample_count: 1,
                    })
                    .expect("staged");
            }
            replayed.push(arm_b.decide(clock(t_b, step, Some(GEAR_SAFE))));
        }
        assert_eq!(continued, replayed, "restored trajectory diverged");
    }

    #[test]
    fn restore_refuses_a_foreign_policy() {
        let arm_policy = policy_with(quiet_sentinel_doc());
        let other_policy = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(1.0e6),
            salience_downshift_threshold: Some(0.55),
            ..Default::default()
        });
        assert_ne!(arm_policy.digest(), other_policy.digest());

        let source = TshGearboxArm::new(other_policy, ps(0.0));
        let checkpoint = source.checkpoint();
        let mut target = TshGearboxArm::new(arm_policy, ps(0.0));
        let err = target.restore(&checkpoint).expect_err("must refuse");
        assert!(matches!(err, CheckpointError::PolicyDigestMismatch { .. }));
    }

    #[test]
    fn unarmed_policy_has_no_opinion() {
        let mut arm = TshGearboxArm::new(TshArmPolicy::unarmed_defaults(), ps(0.0));
        for branch in BOUND_BRANCHES {
            arm.observe(BranchObservation {
                branch,
                accumulated_activity: 100.0,
                span: span(0.25),
                window_end: ps(0.25),
                sample_count: 1,
            })
            .expect("staged");
        }
        let r = arm.decide(clock(0.25, 1, Some(GEAR_CAMPAIGN)));
        assert_eq!(r.requested_gear, None);
        assert_eq!(r.request_reason, RequestReason::PolicyUnarmed);
        assert!(r.flags & FLAG_POLICY_UNARMED != 0);
    }

    #[test]
    fn policy_rejects_an_empty_hysteresis_band() {
        let err = TshArmPolicy::from_document(
            &TshArmPolicyDocument {
                salience_upshift_threshold: Some(0.7),
                salience_downshift_threshold: Some(0.6),
                ..Default::default()
            },
            ValueOrigin::PolicyFile,
            None,
        )
        .expect_err("an empty band must be refused");
        assert!(matches!(
            err,
            TshArmPolicyError::InvalidValue {
                field: "salience_upshift_threshold",
                ..
            }
        ));
    }

    #[test]
    fn policy_rejects_a_non_positive_reference_rate() {
        let err = TshArmPolicy::from_document(
            &TshArmPolicyDocument {
                branch_reference_rate: Some([1.0, 1.0, 0.0, 1.0, 1.0, 1.0]),
                ..Default::default()
            },
            ValueOrigin::PolicyFile,
            None,
        )
        .expect_err("a zero reference rate must be refused");
        assert!(matches!(
            err,
            TshArmPolicyError::InvalidValue {
                field: "branch_reference_rate",
                ..
            }
        ));
    }

    #[test]
    fn defaults_carry_their_origin_and_a_stable_digest() {
        let p = TshArmPolicy::unarmed_defaults();
        for field in p.scalar_fields() {
            assert_eq!(field.origin, ValueOrigin::SourceLiteralDefault);
        }
        assert_eq!(p.gear_dt_fs, DEFAULT_GEAR_DT_FS);
        // Verified ground truth: gear 0/1/2 are 0.5 / 2.0 / 4.0 fs.
        assert_eq!(p.gear_dt_fs[0], 0.5);
        assert_eq!(p.gear_dt_fs[1], 2.0);
        assert_eq!(p.gear_dt_fs[2], 4.0);
        assert_eq!(p.digest(), TshArmPolicy::unarmed_defaults().digest());

        let armed = policy_with(TshArmPolicyDocument {
            branch_tau_ps: Some(3.0),
            ..Default::default()
        });
        assert_eq!(armed.branch_tau_ps.origin, ValueOrigin::PolicyFile);
        assert_ne!(armed.digest(), p.digest(), "digest must track the values");
    }

    #[test]
    fn simulated_span_refuses_degenerate_intervals() {
        assert!(SimulatedPsSpan::new(0.0).is_none());
        assert!(SimulatedPsSpan::new(-1.0).is_none());
        assert!(SimulatedPsSpan::new(f64::NAN).is_none());
        assert!(SimulatedPsSpan::new(f64::INFINITY).is_none());
        assert!(SimulatedPsSpan::from_steps(0, 0.002).is_none());
        assert_eq!(
            SimulatedPsSpan::from_steps(500, 0.004)
                .expect("positive")
                .ps(),
            2.0
        );
        // A rate is only ever built by dividing by a simulated span.
        let r = PerSimulatedPs::from_accumulated(4.0, span(2.0));
        assert_eq!(r.get(), 2.0);
    }

    #[test]
    fn observation_refuses_a_fabricated_activity() {
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let err = arm
            .observe(BranchObservation {
                branch: SalienceBranch::ProbeInteraction,
                accumulated_activity: f64::NAN,
                span: span(0.1),
                window_end: ps(0.1),
                sample_count: 1,
            })
            .expect_err("NaN activity must be refused");
        assert_eq!(err, ObservationRejection::NonFiniteOrNegativeActivity);
        assert!(!arm.state().branch[SalienceBranch::ProbeInteraction.index()].ever_observed);
    }

    #[test]
    fn record_abi_round_trips_the_masks_and_sentinels() {
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let r = arm.decide(clock(0.5, 7, None));
        let abi = r.to_abi();
        assert_eq!(abi.schema_version, TSH_ARM_SCHEMA_VERSION);
        assert!(abi.integrated_salience.is_nan(), "unmeasured stays NaN");
        assert_eq!(abi.authorized_gear_echo, UNMEASURED_U32);
        assert!(r.flags & FLAG_AUTHORIZED_UNMEASURED != 0);
        let mask = BranchMask::from_packed_u32(abi.active_branch_mask);
        assert_eq!(mask, r.active_branch_mask);
        assert!(BranchMask::has(
            mask.unbound,
            SalienceBranch::SignalGridWarp
        ));
        // The gate inputs travel too, or the verdict is unauditable.
        assert!(abi.peak_branch_salience.is_nan(), "unmeasured stays NaN");
        assert_eq!(abi.peak_branch_index, UNMEASURED_U32);
        assert_eq!(
            abi.thermodynamic_action,
            prism_tokenizer::types::ThermodynamicActionOpcode::Handoff as u64 as u32,
            "an evidence-absent decision hands off to the governor"
        );
        assert!(
            abi.is_valid(),
            "a real record must pass the validity predicate"
        );

        // A zero-filled slot decodes as a plausible calm record at gear 0, and
        // gear 0 is a LEGITIMATE gear — so an unwritten slot must be filled
        // with out-of-band sentinels, not zeros.
        let zeroed: TshArmRecordAbi = unsafe { std::mem::zeroed() };
        assert!(
            !zeroed.is_valid(),
            "a zero-filled ring slot must not decode as a record"
        );
        let unwritten = TshArmRecordAbi::unwritten();
        assert!(!unwritten.is_valid());
        assert!(unwritten.integrated_salience.is_nan());
        assert_eq!(unwritten.requested_gear, GEAR_UNSET);
        assert_eq!(unwritten.thermodynamic_action, 0);
        assert!(
            unwritten.thermodynamic_action_opcode().is_none(),
            "0 must not be a registered opcode"
        );
    }

    /// The reference vector that used to be transcribed into a comment.
    ///
    /// Kept as a HOST-ONLY self-consistency check with the derivation stated,
    /// so the analysis lane (which does not compile CUDA) still pins the
    /// arithmetic. `host_matches_the_compiled_device_implementation` below is
    /// the one that proves host/device agreement, by calling the device.
    #[test]
    fn host_arithmetic_matches_the_closed_form() {
        // Defaults, ARMED, one branch observing 2.0 activity units over 1.0 ps.
        let policy = policy_with(TshArmPolicyDocument::default());
        assert_eq!(policy.branch_tau_ps.value, DEFAULT_BRANCH_TAU_PS);
        assert_eq!(
            policy.max_sentinel_interval_ps.value,
            DEFAULT_MAX_SENTINEL_INTERVAL_PS
        );

        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        arm.observe(BranchObservation {
            branch: SalienceBranch::ForceEnergyStability,
            accumulated_activity: 2.0,
            span: span(1.0),
            window_end: ps(1.0),
            sample_count: 1,
        })
        .expect("staged");
        let r = arm.decide(clock(1.0, 100, Some(GEAR_SAFE)));

        // Closed form: A(1.0) = r*tau*(1 - exp(-s/tau)) with r = 2.0/ps,
        // tau = 2.0 ps, s = 1.0 ps  =>  4*(1 - exp(-0.5)) = 1.5738773611494664.
        let expected_activity = 4.0 * (1.0 - (-0.5f64).exp());
        assert!(
            (arm.state().branch[0].activity - expected_activity).abs() < 1e-15,
            "leaky integrator diverged from its own closed form: {} vs {expected_activity}",
            arm.state().branch[0].activity
        );
        // Branch salience: rate = A/tau, s = rate/(rate + ref).
        let rate = expected_activity / DEFAULT_BRANCH_TAU_PS;
        let expected_salience = rate / (rate + DEFAULT_BRANCH_REFERENCE_RATE);
        assert!(
            (r.peak_branch_salience.expect("measured") - expected_salience).abs() < 1e-15,
            "peak salience diverged: {:?}",
            r.peak_branch_salience
        );
        // Coverage: 1 - exp(-1.0/2.0) over the one covered picosecond, and it
        // is the only weighted branch that has ever reported, so the weighted
        // observation confidence is that coverage divided by the five weighted
        // branches.
        let expected_coverage = 1.0 - (-0.5f64).exp();
        let expected_conf = expected_coverage / 5.0;
        assert!(
            (r.observation_confidence.expect("measured") - expected_conf).abs() < 1e-15,
            "observation confidence is not (recency x coverage) weighted: {:?}",
            r.observation_confidence
        );
        assert_eq!(r.requested_gear, Some(GEAR_SAFE));
        assert_eq!(r.request_reason, RequestReason::Hold);
        assert_eq!(RequestReason::Hold as u32, 1);
        assert_eq!(r.requested_dt_fs, Some(2.0));
        assert_eq!(
            r.thermodynamic_action,
            prism_tokenizer::types::ThermodynamicActionOpcode::Stable
        );

        // Nothing ever observed at all.
        let mut blind = TshGearboxArm::new(policy_with(TshArmPolicyDocument::default()), ps(0.0));
        let d = blind.decide(clock(1.0, 100, Some(GEAR_CAMPAIGN)));
        assert_eq!(d.integrated_salience, None);
        assert_eq!(d.peak_branch_salience, None);
        assert_eq!(d.requested_gear, Some(GEAR_FINE));
        assert_eq!(d.request_reason, RequestReason::EvidenceAbsent);
        assert_eq!(RequestReason::EvidenceAbsent as u32, 6);
        assert!(d.flags & FLAG_EVIDENCE_ABSENT != 0);
        assert_eq!(
            d.thermodynamic_action,
            prism_tokenizer::types::ThermodynamicActionOpcode::Handoff
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Host <-> COMPILED DEVICE parity.
    //
    // These call `cuda/tsh_gearbox_arm.cu`, which includes the header and is
    // registered in build.rs. Two consequences that the previous
    // comment-transcribed "reference vector" could not deliver:
    //
    //   * the header's static_asserts execute, so a layout drift is a compile
    //     error rather than a misparsed science record;
    //   * the comparison is between two IMPLEMENTATIONS. A test that compares
    //     an implementation against literals someone typed into a comment
    //     proves only that the comment and the implementation agree.
    // ───────────────────────────────────────────────────────────────────────

    #[cfg(feature = "gpu")]
    #[test]
    fn device_abi_sizes_match_the_compiled_header() {
        let sizes = device::abi_sizes();
        assert_eq!(
            sizes[device::ABI_SLOT_BRANCH] as usize,
            DEVICE_BRANCH_STATE_BYTES
        );
        assert_eq!(sizes[device::ABI_SLOT_POLICY] as usize, DEVICE_POLICY_BYTES);
        assert_eq!(sizes[device::ABI_SLOT_STATE] as usize, DEVICE_STATE_BYTES);
        assert_eq!(sizes[device::ABI_SLOT_RECORD] as usize, DEVICE_RECORD_BYTES);
        assert_eq!(
            sizes[device::ABI_SLOT_RING] as usize,
            DEVICE_RING_HEADER_BYTES
        );
        assert_eq!(
            sizes[device::ABI_SLOT_STATE_BRANCH_OFFSET] as usize,
            DEVICE_STATE_BRANCH_OFFSET
        );
        // The host policy encoder writes the fourteen scalars as one
        // contiguous run starting here.
        assert_eq!(
            sizes[device::ABI_SLOT_POLICY_SCALAR_OFFSET] as usize,
            16 + 2 * TSH_ARM_N_BRANCHES * 8
        );
        assert_eq!(device::schema_version(), TSH_ARM_SCHEMA_VERSION);
        // And the Rust record mirror is the same size the compiler produced.
        assert_eq!(
            std::mem::size_of::<TshArmRecordAbi>(),
            sizes[device::ABI_SLOT_RECORD] as usize
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn host_matches_the_compiled_device_implementation() {
        // Same policy, same observations, same clock, on both sides.
        let policy = policy_with(TshArmPolicyDocument {
            // Deliberately NOT exactly representable in binary32: as `float`
            // the device compared against different numbers than the host, and
            // the only parity test in the first revision used values that were
            // exactly representable and so could not detect it.
            branch_tau_ps: Some(0.3),
            salience_downshift_threshold: Some(0.61),
            salience_upshift_threshold: Some(0.23),
            salience_reinforce_threshold: Some(0.87),
            min_observation_cadence_ps: Some(0.07),
            confidence_decay_ps: Some(0.13),
            ..Default::default()
        });

        let mut host = TshGearboxArm::new(policy.clone(), ps(0.0));
        let mut dev_policy = device::DeviceBuffer::from_bytes(&policy.to_device_bytes());
        let mut dev_state = device::DeviceBuffer::zeroed(DEVICE_STATE_BYTES);
        device::reset(&mut dev_state, 0.0);

        let mut t = 0.0f64;
        for step in 1..=64u64 {
            // A deterministic, non-trivial input that crosses the thresholds.
            let rate = 0.4 + 1.6 * ((step as f64) * 0.37).sin().abs();
            let dt = 0.11f64;
            let covered = if step % 3 == 0 { dt } else { dt * 0.4 };
            for branch in BOUND_BRANCHES.iter().take(3) {
                host.observe(BranchObservation {
                    branch: *branch,
                    accumulated_activity: rate * covered,
                    span: span(covered),
                    window_end: ps(t + covered),
                    sample_count: step as u32,
                })
                .expect("staged");
                assert!(
                    device::observe(
                        &dev_policy,
                        &mut dev_state,
                        *branch,
                        rate * covered,
                        covered,
                        t + covered,
                        step as u32,
                    ),
                    "device refused an observation the host accepted"
                );
            }
            t += dt;
            let authorized = if step % 7 == 0 {
                Some(GEAR_FINE)
            } else {
                Some(GEAR_SAFE)
            };
            let h = host.decide(clock(t, step, authorized)).to_abi();
            let d = device::decide(
                &dev_policy,
                &mut dev_state,
                t,
                step,
                step as u32,
                authorized,
            );

            assert_eq!(h.schema_version, d.schema_version, "step {step}");
            assert_eq!(h.record_sequence, d.record_sequence, "step {step}");
            assert_eq!(
                h.requested_gear, d.requested_gear,
                "gear diverged at {step}"
            );
            assert_eq!(
                h.request_reason, d.request_reason,
                "reason diverged at {step}"
            );
            assert_eq!(h.memory_state, d.memory_state, "memory diverged at {step}");
            assert_eq!(h.flags, d.flags, "flags diverged at {step}");
            assert_eq!(
                h.active_branch_mask, d.active_branch_mask,
                "branch mask diverged at {step}"
            );
            assert_eq!(
                h.thermodynamic_action, d.thermodynamic_action,
                "opcode diverged at {step}"
            );
            assert_eq!(
                h.peak_branch_index, d.peak_branch_index,
                "peak branch diverged at {step}"
            );
            // f32 record fields: the two libm `exp()` may differ in the last
            // ULP, and that difference compounds through the integrator. What
            // must NOT differ is any DECISION, which the equalities above pin.
            let close = |a: f32, b: f32, what: &str| {
                if a.is_nan() && b.is_nan() {
                    return;
                }
                assert!(
                    (a - b).abs() <= 1e-5 * a.abs().max(1.0),
                    "{what} diverged at step {step}: host={a} device={b}"
                );
            };
            close(
                h.integrated_salience,
                d.integrated_salience,
                "mean salience",
            );
            close(
                h.peak_branch_salience,
                d.peak_branch_salience,
                "peak salience",
            );
            close(
                h.observation_confidence,
                d.observation_confidence,
                "observation confidence",
            );
            close(h.observation_coverage, d.observation_coverage, "coverage");
            close(
                h.request_confidence,
                d.request_confidence,
                "request confidence",
            );
        }

        // The coverage integrator agrees branch-by-branch, not just in the mean.
        for branch in BOUND_BRANCHES.iter().take(3) {
            let hc = host.branch_coverage(branch.index());
            let dc = device::branch_coverage(&dev_policy, &dev_state, *branch);
            assert!(
                (hc - dc).abs() < 1e-9,
                "branch {} coverage diverged: host={hc} device={dc}",
                branch.as_str()
            );
        }

        // Silence the unused-mut lint on a buffer the FFI only reads.
        let _ = dev_policy.as_mut_slice();
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn device_refuses_the_asc_branch_exactly_as_the_host_does() {
        let policy = policy_with(TshArmPolicyDocument::default());
        let dev_policy = device::DeviceBuffer::from_bytes(&policy.to_device_bytes());
        let mut dev_state = device::DeviceBuffer::zeroed(DEVICE_STATE_BYTES);
        device::reset(&mut dev_state, 0.0);
        assert!(
            !device::observe(
                &dev_policy,
                &mut dev_state,
                SalienceBranch::SignalGridWarp,
                1.0,
                0.5,
                0.5,
                1
            ),
            "the device must refuse ASC-as-independent under the default policy"
        );
        assert!(
            device::observe(
                &dev_policy,
                &mut dev_state,
                SalienceBranch::ForceEnergyStability,
                1.0,
                0.5,
                0.5,
                1
            ),
            "the force branch must still accept"
        );
        let rec = device::decide(&dev_policy, &mut dev_state, 0.5, 1, 1, Some(GEAR_SAFE));
        assert!(
            rec.flags & FLAG_SIGNAL_GRID_UNBOUND != 0,
            "the refusal must be recorded, not silent"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn device_state_bridge_round_trips_the_sentinels() {
        // The bridge the module doc declared and the first revision never
        // implemented. Without it the captured-graph path — the side the .cuh
        // exists to serve — cannot be checkpointed at all.
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy.clone(), ps(0.0));

        // Leave the three optional clocks in their "has not happened" state.
        assert_eq!(arm.state().active_memory_exit_ps, None);
        assert_eq!(arm.state().reinforce_since_ps, None);
        assert_eq!(arm.state().sentinel_hold_until_ps, None);

        let bytes = arm.checkpoint().to_device_state_bytes();
        assert_eq!(bytes.len(), DEVICE_STATE_BYTES);

        // The device must read this as a legal reset-equivalent state and keep
        // deciding from it.
        let dev_policy = device::DeviceBuffer::from_bytes(&policy.to_device_bytes());
        let mut dev_state = device::DeviceBuffer::from_bytes(&bytes);
        let rec = device::decide(&dev_policy, &mut dev_state, 0.25, 1, 1, Some(GEAR_SAFE));
        assert!(rec.is_valid(), "the device refused a bridged state");
        assert!(device::record_is_valid(&rec));

        // Round trip: host -> device layout -> host.
        let back =
            TshArmCheckpoint::from_device_state_bytes(&bytes, policy.digest()).expect("decodes");
        assert_eq!(
            back.state.active_memory_exit_ps, None,
            "None must not become 0.0"
        );
        assert_eq!(back.state.reinforce_since_ps, None);
        assert_eq!(back.state.sentinel_hold_until_ps, None);
        for b in back.state.branch.iter() {
            assert_eq!(b.last_evidence_ps, None, "an unbound clock must stay None");
        }

        // And with the clocks POPULATED.
        let mut t = 0.0f64;
        for step in 1..=40u64 {
            t += 0.25;
            for branch in BOUND_BRANCHES {
                arm.observe(BranchObservation {
                    branch,
                    accumulated_activity: 20.0 * 0.25,
                    span: span(0.25),
                    window_end: ps(t),
                    sample_count: 1,
                })
                .expect("staged");
            }
            arm.decide(clock(t, step, Some(GEAR_SAFE)));
        }
        let live = arm.checkpoint();
        let bytes = live.to_device_state_bytes();
        let back =
            TshArmCheckpoint::from_device_state_bytes(&bytes, policy.digest()).expect("decodes");
        assert_eq!(
            back.state.record_sequence, live.state.record_sequence,
            "the 64-bit sequence must survive the device layout"
        );
        assert_eq!(back.state.decisions, live.state.decisions);
        assert_eq!(back.state.memory_state, live.state.memory_state);
        for (a, b) in back.state.branch.iter().zip(live.state.branch.iter()) {
            assert_eq!(a.activity, b.activity);
            assert_eq!(a.coverage_activity, b.coverage_activity);
            assert_eq!(a.last_evidence_ps, b.last_evidence_ps);
            assert_eq!(a.ever_observed, b.ever_observed);
        }
        // A restore under a different policy is still refused.
        let other = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(1.0e6),
            salience_downshift_threshold: Some(0.55),
            ..Default::default()
        });
        let mut target = TshGearboxArm::new(other, ps(0.0));
        assert!(matches!(
            target.restore(&back).expect_err("must refuse"),
            CheckpointError::PolicyDigestMismatch { .. }
        ));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn device_unwritten_slot_matches_the_host_pattern() {
        let d = device::record_unwritten();
        let h = TshArmRecordAbi::unwritten();
        assert_eq!(d.schema_version, h.schema_version);
        assert_eq!(d.requested_gear, h.requested_gear);
        assert_eq!(d.request_reason, h.request_reason);
        assert_eq!(d.memory_state, h.memory_state);
        assert_eq!(d.thermodynamic_action, h.thermodynamic_action);
        assert_eq!(d.peak_branch_index, h.peak_branch_index);
        assert!(d.integrated_salience.is_nan() && h.integrated_salience.is_nan());
        assert!(!device::record_is_valid(&d));
        assert!(!d.is_valid());
    }

    #[test]
    fn a_refused_emission_marks_every_later_record() {
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let clean = arm.decide(clock(0.25, 1, Some(GEAR_SAFE)));
        assert_eq!(clean.flags & FLAG_RING_DROPPED, 0);

        arm.note_dropped_records(3);
        let after = arm.decide(clock(0.5, 2, Some(GEAR_SAFE)));
        assert!(
            after.flags & FLAG_RING_DROPPED != 0,
            "a truncated stream must say it is truncated"
        );
        // The sequence itself stays gapless: the arm decided, the ring lost it.
        assert_eq!(after.record_sequence, clean.record_sequence + 1);
        assert_eq!(arm.clear_dropped_records(), 3);
        let cleared = arm.decide(clock(0.75, 3, Some(GEAR_SAFE)));
        assert_eq!(cleared.flags & FLAG_RING_DROPPED, 0);
    }

    #[test]
    fn the_carried_evidence_backlog_is_bounded_and_reported() {
        // Carrying the surplus forward is what makes the salience independent
        // of the observation cadence — but an UNBOUNDED carry is its own
        // hazard. A producer staging simulated time faster than the arm decides
        // would grow a queue without limit, and the arm would integrate the
        // distant past as if it were the present, arbitrarily far behind the
        // run, with nothing on the record saying so.
        let horizon = 2.0f64;
        let policy = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(horizon),
            ..Default::default()
        });
        let mut arm = TshGearboxArm::new(policy, ps(0.0));

        // Stage 5 ps of evidence per 0.1 ps decision: 50x faster than the arm
        // can consume it.
        let mut t = 0.0f64;
        let mut saw_truncation = false;
        for step in 1..=60u64 {
            arm.observe(BranchObservation {
                branch: SalienceBranch::ForceEnergyStability,
                accumulated_activity: 5.0,
                span: span(5.0),
                window_end: ps(t + 5.0),
                sample_count: 1,
            })
            .expect("staged");
            t += 0.1;
            let r = arm.decide(clock(t, step, Some(GEAR_SAFE)));
            if r.flags & FLAG_EVIDENCE_BACKLOG_TRUNCATED != 0 {
                saw_truncation = true;
            }
            // The bound holds on EVERY decision, not just eventually.
            let pending = arm.state().branch[0].pending_span_ps;
            assert!(
                pending <= horizon + 1e-12,
                "backlog {pending} ps exceeded the {horizon} ps staleness horizon at step {step}"
            );
        }
        assert!(
            saw_truncation,
            "a bounded drop must be REPORTED; a silent one is the same defect \
             class as the discard it replaced"
        );

        // Truncation preserves the RATE — the arm keeps measuring the same
        // physics, it just stops queueing arbitrarily old evidence.
        let b = &arm.state().branch[0];
        let rate = b.pending_activity / b.pending_span_ps;
        assert!(
            (rate - 1.0).abs() < 1e-12,
            "truncation must drop span, not distort the rate: {rate}"
        );

        // And a producer staying within the horizon is never truncated.
        let policy = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(horizon),
            ..Default::default()
        });
        let mut calm = TshGearboxArm::new(policy, ps(0.0));
        let mut t = 0.0f64;
        for step in 1..=60u64 {
            calm.observe(BranchObservation {
                branch: SalienceBranch::ForceEnergyStability,
                accumulated_activity: 0.25,
                span: span(0.25),
                window_end: ps(t + 0.25),
                sample_count: 1,
            })
            .expect("staged");
            t += 0.25;
            let r = calm.decide(clock(t, step, Some(GEAR_FINE)));
            assert_eq!(
                r.flags & FLAG_EVIDENCE_BACKLOG_TRUNCATED,
                0,
                "a matched producer must never be truncated"
            );
        }
    }

    #[test]
    fn the_deprecated_asc_branch_name_stays_reachable() {
        // Deprecated capabilities are preserved AS DEPRECATED AND REACHABLE.
        // The old name asserted an independence from the force representation
        // that the evidence does not support, so it is deprecated — but code
        // that still names it must keep resolving, to the branch it always
        // meant.
        #[allow(deprecated)]
        let alias = SalienceBranch::SignalWarpAsc;
        assert_eq!(alias, SalienceBranch::SignalGridWarp);
        assert_eq!(alias.index(), 4);
        assert_eq!(alias.as_str(), "signal_grid_warp_activity");
        assert!(!alias.is_force_derived());
        assert!(SalienceBranch::ForceEnergyStability.is_force_derived());
    }

    #[test]
    fn memory_states_expose_the_operator_table() {
        assert_eq!(
            TshMemoryState::Baseline.resolution_consequence(),
            "may permit coarser if physically safe"
        );
        assert_eq!(
            TshMemoryState::Rechallenge.resolution_consequence(),
            "immediate downshift"
        );
        assert!(TshMemoryState::Decay.permits_coarsening_request());
        assert!(!TshMemoryState::ActiveMemory.permits_coarsening_request());
        assert!(!TshMemoryState::Reinforced.permits_coarsening_request());
        assert!(!TshMemoryState::InputWithdrawal.permits_coarsening_request());
        for v in 0..=8u32 {
            assert!(TshArmState::memory_state_from_wire(v).is_some());
        }
        assert!(TshArmState::memory_state_from_wire(9).is_none());
    }

    // ───────────────────────────────────────────────────────────────────────
    // ASC IS THE FORCE BUFFER.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn asc_evidence_cannot_be_counted_as_a_second_family() {
        let policy = policy_with(quiet_sentinel_doc());
        assert!(
            !policy.signal_grid_warp_independent,
            "the default must not claim an independent signal grid"
        );
        assert_eq!(
            policy.branch_weight[SalienceBranch::SignalGridWarp.index()],
            0.0,
            "the signal-grid branch must be unweighted by default"
        );

        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let err = arm
            .observe(BranchObservation {
                branch: SalienceBranch::SignalGridWarp,
                accumulated_activity: 1.0,
                span: span(0.5),
                window_end: ps(0.5),
                sample_count: 1,
            })
            .expect_err("ASC-as-independent must be refused, not silently accepted");
        assert_eq!(
            err,
            ObservationRejection::SignalGridWarpNotIndependentOfForce
        );
        assert!(
            !arm.state().branch[SalienceBranch::SignalGridWarp.index()].ever_observed,
            "a refused observation must not bind the branch"
        );
        assert_eq!(arm.signal_grid_refusals(), 1);
        let r = arm.decide(clock(0.5, 1, Some(GEAR_SAFE)));
        assert!(
            r.flags & FLAG_SIGNAL_GRID_UNBOUND != 0,
            "the refusal must reach the record; a silent drop is the same defect"
        );

        // The supported route for force-derived evidence: it lands on the FORCE
        // branch, so the same physics is integrated exactly once.
        let mut a = TshGearboxArm::new(policy_with(quiet_sentinel_doc()), ps(0.0));
        a.observe_force_derived(2.0, span(1.0), ps(1.0), 1)
            .expect("force-derived evidence is admissible");
        assert!(a.state().branch[SalienceBranch::ForceEnergyStability.index()].ever_observed);
        assert!(!a.state().branch[SalienceBranch::SignalGridWarp.index()].ever_observed);

        // A policy that weights BOTH without declaring independence is refused
        // at resolution, before a single observation is staged.
        let err = TshArmPolicy::from_document(
            &TshArmPolicyDocument {
                branch_weight: Some([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                ..Default::default()
            },
            ValueOrigin::PolicyFile,
            None,
        )
        .expect_err("weighting ASC alongside force must be refused");
        assert!(matches!(
            err,
            TshArmPolicyError::InvalidValue {
                field: "branch_weight",
                ..
            }
        ));

        // Declaring independence makes it legal — and the declaration is
        // recorded in the digest, so a run under it is distinguishable.
        let independent = TshArmPolicy::from_document(
            &TshArmPolicyDocument {
                branch_weight: Some([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                signal_grid_warp_independent: Some(true),
                ..Default::default()
            },
            ValueOrigin::PolicyFile,
            None,
        )
        .expect("an operator-declared independent grid is admissible");
        assert!(independent.signal_grid_warp_independent);
        assert_ne!(
            independent.digest(),
            policy_with(TshArmPolicyDocument::default()).digest(),
            "the independence declaration must change the policy digest"
        );
        let mut arm2 = TshGearboxArm::new(independent, ps(0.0));
        arm2.observe(BranchObservation {
            branch: SalienceBranch::SignalGridWarp,
            accumulated_activity: 1.0,
            span: span(0.5),
            window_end: ps(0.5),
            sample_count: 1,
        })
        .expect("declared-independent grid evidence is admissible");
    }

    // ───────────────────────────────────────────────────────────────────────
    // LEXICOGRAPHIC, never a score.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn one_maximal_branch_is_not_outvoted_by_calm_branches() {
        // THE adversarial case. One branch at maximum salience, every other
        // weighted branch reporting perfect calm.
        //
        // Under the confidence-weighted mean the first revision gated on, this
        // reads 0.999/5 = 0.1998 — below the 0.25 upshift threshold, i.e.
        // QUIET and COARSENING ELIGIBLE. An arbitrarily large single-family
        // demand produced a request for a COARSER gear, and wiring up more
        // families made the arm blinder to each one.
        let policy = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(1.0e6),
            ..Default::default()
        });
        let mut arm = TshGearboxArm::new(policy, ps(0.0));

        let mut t = 0.0f64;
        let mut last = None;
        for step in 1..=80u64 {
            t += 0.25;
            for branch in BOUND_BRANCHES {
                // Rate 2000/ps on branch 0 saturates its salience to ~0.999;
                // everything else reports measured calm.
                let rate = if branch == SalienceBranch::ForceEnergyStability {
                    2000.0
                } else {
                    0.0
                };
                arm.observe(BranchObservation {
                    branch,
                    accumulated_activity: rate * 0.25,
                    span: span(0.25),
                    window_end: ps(t),
                    sample_count: 1,
                })
                .expect("staged");
            }
            last = Some(arm.decide(clock(t, step, Some(GEAR_SAFE))));
        }
        let r = last.expect("decided");

        let peak = r.peak_branch_salience.expect("measured");
        let mean = r.integrated_salience.expect("measured");
        assert!(
            peak > 0.99,
            "the excursion must be visible in the peak: {peak}"
        );
        assert!(
            mean < DEFAULT_SALIENCE_UPSHIFT,
            "the mean must still be diluted — otherwise this test proves nothing: {mean}"
        );
        assert_eq!(
            r.peak_branch,
            Some(SalienceBranch::ForceEnergyStability),
            "the record must name the branch that produced the demand"
        );
        // The gate followed the PEAK, not the mean.
        assert_eq!(
            r.requested_gear,
            Some(GEAR_FINE),
            "a maximal single-branch excursion must demand the finest gear, \
             not be averaged into a coarsening request"
        );
        assert_ne!(r.request_reason, RequestReason::UpshiftEligible);
        assert!(
            BranchMask::has(
                r.active_branch_mask.salient,
                SalienceBranch::ForceEnergyStability
            ),
            "the salient mask must mark the branch the gate acted on"
        );
        // And adding MORE calm evidence families must not weaken the demand.
        assert!(
            r.memory_state != TshMemoryState::Baseline,
            "the memory must have seen the excursion, not the average"
        );
    }

    #[test]
    fn coarsening_requires_every_admissible_branch_quiet() {
        // The mirror of the previous test: the peak governs the QUIET side too,
        // so one noisy branch is enough to refuse a coarsening even when the
        // mean is comfortably below the threshold.
        let policy = policy_with(TshArmPolicyDocument {
            max_sentinel_interval_ps: Some(1.0e6),
            min_gear_dwell_ps: Some(0.0),
            decay_period_ps: Some(0.0),
            ..Default::default()
        });
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let mut t = 0.0f64;
        let mut coarsened = false;
        for step in 1..=200u64 {
            t += 0.25;
            for branch in BOUND_BRANCHES {
                // One branch parked between the thresholds; the rest calm.
                let rate = if branch == SalienceBranch::KccCoupling {
                    0.6 // steady-state salience 0.375, inside the dead band
                } else {
                    0.0
                };
                arm.observe(BranchObservation {
                    branch,
                    accumulated_activity: rate * 0.25,
                    span: span(0.25),
                    window_end: ps(t),
                    sample_count: 1,
                })
                .expect("staged");
            }
            let r = arm.decide(clock(t, step, Some(GEAR_SAFE)));
            if r.request_reason == RequestReason::UpshiftEligible {
                coarsened = true;
            }
        }
        assert!(
            !coarsened,
            "a branch inside the hysteresis band must block coarsening even \
             when the mean over branches is quiet"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // The codec-registered opcode has a producer.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn every_decision_emits_a_registered_thermodynamic_action() {
        use prism_tokenizer::types::{
            unpack_namespace, ArtifactFamily, ThermodynamicActionField, ThermodynamicActionOpcode,
        };
        use std::collections::BTreeSet;

        // The mapping is total: every reason publishes a registered opcode.
        let reasons = [
            RequestReason::Unset,
            RequestReason::Hold,
            RequestReason::SalienceDownshift,
            RequestReason::MemoryHoldFine,
            RequestReason::Rechallenge,
            RequestReason::Sentinel,
            RequestReason::EvidenceAbsent,
            RequestReason::UpshiftEligible,
            RequestReason::ConfidenceGate,
            RequestReason::DwellGate,
            RequestReason::PolicyUnarmed,
            RequestReason::MemoryThresholdDownshift,
        ];
        let mut seen = BTreeSet::new();
        for r in reasons {
            let op = r.thermodynamic_action();
            assert!(
                thermodynamic_action_serde::from_u64(op as u64).is_some(),
                "{} mapped to an unregistered opcode",
                r.as_str()
            );
            seen.insert(op as u64);
        }
        // All six registered opcodes are reachable — none is decorative.
        assert_eq!(
            seen.len(),
            6,
            "some registered opcode has no reason mapping"
        );

        // And the tokens are real tokens in the registered namespace.
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        arm.observe(BranchObservation {
            branch: SalienceBranch::ForceEnergyStability,
            accumulated_activity: 2.0,
            span: span(1.0),
            window_end: ps(1.0),
            sample_count: 1,
        })
        .expect("staged");
        let req = arm.decide(clock(1.0, 100, Some(GEAR_SAFE)));
        let (opcode_token, dt_token) = req.thermodynamic_action_tokens();

        let (family_slot, field_slot) = unpack_namespace(opcode_token.namespace());
        assert_eq!(
            family_slot,
            ArtifactFamily::ThermodynamicAction.family_slot()
        );
        assert_eq!(field_slot, ThermodynamicActionField::TshState as u8);
        assert_eq!(
            opcode_token.field_token(),
            ThermodynamicActionOpcode::Stable as u64,
            "a held gear is thermodynamically Stable"
        );
        assert_eq!(
            opcode_token.leverage_tier(),
            prism_tokenizer::leverage_tier(ArtifactFamily::ThermodynamicAction)
        );

        let dt_token = dt_token.expect("an armed arm requests a timestep");
        let (dt_family, dt_field) = unpack_namespace(dt_token.namespace());
        assert_eq!(dt_family, ArtifactFamily::ThermodynamicAction.family_slot());
        assert_eq!(dt_field, ThermodynamicActionField::DeltaTimeFs as u8);
        assert_ne!(
            dt_token.field_token(),
            prism_tokenizer::Token::ESCAPE_CODE,
            "the gear timestep must quantize inside the declared bound"
        );

        // A decision taken with NO evidence is a fail-safe default, and the
        // token must not claim it was verified evidence.
        let mut blind = TshGearboxArm::new(policy_with(quiet_sentinel_doc()), ps(0.0));
        let d = blind.decide(clock(1.0, 1, Some(GEAR_SAFE)));
        let (blind_token, _) = d.thermodynamic_action_tokens();
        assert!(
            !blind_token.verified(),
            "an evidence-absent default must not be marked verified"
        );
        assert!(
            blind_token.provenance(),
            "every request carries a policy digest, so provenance holds"
        );
        assert!(
            opcode_token.verified(),
            "a decision taken on measured evidence is verified"
        );

        // The opcode round-trips through the wire form.
        let json = serde_json::to_string(&req).expect("serialises");
        let back: TshGearRequest = serde_json::from_str(&json).expect("round trip");
        assert_eq!(back.thermodynamic_action, req.thermodynamic_action);
        assert_eq!(back, req);
        // An unregistered opcode is refused rather than silently accepted.
        let corrupted = json.replace("\"thermodynamic_action\":1", "\"thermodynamic_action\":99");
        assert!(
            serde_json::from_str::<TshGearRequest>(&corrupted).is_err(),
            "an unregistered opcode must not decode"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // The arm REQUESTS ONLY, and says so per-arm.
    // ───────────────────────────────────────────────────────────────────────

    #[test]
    fn arm_source_contains_no_authorization_path() {
        // Mechanical, not aspirational. The claim "this module has no authority
        // to authorize a timestep" is only worth the enforcement behind it, and
        // a doc comment is not enforcement. This reads the two files that make
        // up the arm and fails if either acquires a write to the authorized
        // gear, the timestep, or the gearbox activation switch.
        let host = arm_source_without_tests();
        let device_header = include_str!("cuda/tsh_gearbox_arm.cuh");
        let device_tu = include_str!("cuda/tsh_gearbox_arm.cu");

        // Each entry is (needle, why it would be an authority breach).
        let forbidden: &[(&str, &str)] = &[
            ("set_gearbox_active", "flips the gearbox activation switch"),
            ("apply_fixed_dt", "actuates a timestep"),
            ("d_protocol->dt", "writes the integrator's timestep"),
            ("adj->d_dt", "writes the timestep actuation pointer"),
            ("authorized_gear =", "assigns an authorized gear"),
            ("GEAR_ABORT", "names the abort gear"),
            ("PRISM_GEAR_VERDICT_", "issues a governor verdict"),
            ("prism_gearbox_governor_decide", "invokes the governor"),
        ];
        // `x = ` is an assignment; `x == ` is a comparison, and the arm READS
        // the authorized gear on every decision — that is its input, not its
        // output. Distinguishing them is the whole point of the needle.
        let breaches = |line: &str, needle: &str| -> bool {
            let mut from = 0usize;
            while let Some(i) = line[from..].find(needle) {
                let at = from + i;
                let rest = &line[at + needle.len()..];
                if needle.ends_with('=') && rest.starts_with('=') {
                    from = at + needle.len();
                    continue;
                }
                return true;
            }
            false
        };

        for (source, name) in [
            (host, "tsh_gearbox_arm.rs"),
            (device_header, "tsh_gearbox_arm.cuh"),
            (device_tu, "tsh_gearbox_arm.cu"),
        ] {
            for (needle, why) in forbidden {
                for (n, line) in source.lines().enumerate() {
                    if is_comment(line) {
                        continue;
                    }
                    assert!(
                        !breaches(line, needle),
                        "{name}:{} {needle} — {why}; the arm REQUESTS ONLY",
                        n + 1
                    );
                }
            }
        }
        // The scan itself must be non-vacuous: a needle that matches nothing
        // anywhere would let a real breach through unnoticed.
        assert!(
            device_header.contains("authorized_gear"),
            "the scan is looking at the wrong file"
        );

        // And the emitted gear is always inside the requestable range, under
        // every input the decision path can see.
        let policy = policy_with(quiet_sentinel_doc());
        for authorized in [
            None,
            Some(0u32),
            Some(1),
            Some(2),
            Some(3),
            Some(GEAR_UNSET),
        ] {
            let mut arm = TshGearboxArm::new(policy.clone(), ps(0.0));
            let mut t = 0.0;
            for step in 1..=30u64 {
                t += 0.2;
                for branch in BOUND_BRANCHES {
                    arm.observe(BranchObservation {
                        branch,
                        accumulated_activity: (step as f64 % 5.0) * 0.2,
                        span: span(0.2),
                        window_end: ps(t),
                        sample_count: 1,
                    })
                    .expect("staged");
                }
                let r = arm.decide(clock(t, step, authorized));
                if let Some(g) = r.requested_gear {
                    assert!(g <= GEAR_MAX_ACTIVE, "requested gear {g} out of range");
                }
            }
        }
    }

    #[test]
    fn arm_declares_non_interference_per_arm_without_touching_the_global_ledger() {
        // The Stage-II observer arm claims dt_mutation = false about ITSELF and
        // backs it with a physics-signature equivalence. A gearbox-facing arm
        // in the same crate must be able to claim dt_mutation = true about
        // ITSELF without voiding that: the ledger has no arm dimension, so
        // flipping one global flag would make the observer's claim false with
        // no test noticing.
        let source = arm_source_without_tests();
        for needle in ["TshRuntimeNonInterferenceLedger", "tsh_stage_ii"] {
            for (n, line) in source.lines().enumerate() {
                if is_comment(line) {
                    continue;
                }
                assert!(
                    !line.contains(needle),
                    "tsh_gearbox_arm.rs:{} touches {needle}; the observer arm's claim \
                     must stay independently assertable",
                    n + 1
                );
            }
        }

        let armed = TshGearboxArm::new(policy_with(quiet_sentinel_doc()), ps(0.0));
        let claim = armed.non_interference_claim();
        assert_eq!(claim.arm, TSH_ARM_IDENTITY);
        assert!(claim.dt_mutation, "the arm's whole purpose is to move dt");
        assert!(!claim.force_mutation);
        assert!(!claim.sampling_bias);
        assert_eq!(
            claim.physics_signature_equivalent,
            PhysicsSignatureEquivalence::NotApplicableGearboxArmActive,
            "a run this arm steered has no fixed-dt baseline to be equivalent to; \
             a bare true or false would be an unmeasured claim either way"
        );
        assert_eq!(
            claim.physics_signature_equivalent.as_str(),
            "not_applicable_gearbox_arm_active"
        );

        // An UNARMED arm never moved anything and must not overstate its reach.
        let unarmed = TshGearboxArm::new(TshArmPolicy::unarmed_defaults(), ps(0.0));
        let claim = unarmed.non_interference_claim();
        assert!(!claim.dt_mutation);
        assert_eq!(
            claim.physics_signature_equivalent,
            PhysicsSignatureEquivalence::Equivalent
        );
    }

    #[test]
    fn request_confidence_saturates_outside_the_hysteresis_band() {
        // At maximum calm — as far from the downshift threshold as it is
        // possible to be — the margin must be 1.0. Taking the absolute value of
        // the nearer distance made s = 0.0 report 0.714, which described a
        // knife-edge decision that was not one.
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let mut t = 0.0f64;
        let mut last = None;
        for step in 1..=80u64 {
            t += 0.25;
            for branch in BOUND_BRANCHES {
                arm.observe(BranchObservation {
                    branch,
                    accumulated_activity: 0.0,
                    span: span(0.25),
                    window_end: ps(t),
                    sample_count: 1,
                })
                .expect("staged");
            }
            last = Some(arm.decide(clock(t, step, Some(GEAR_SAFE))));
        }
        let r = last.expect("decided");
        assert_eq!(r.peak_branch_salience, Some(0.0), "measured perfect calm");
        let conf = r.observation_confidence.expect("weights present");
        let req = r.request_confidence.expect("measured");
        assert!(
            (req - conf).abs() < 1e-12,
            "outside the band the request confidence IS the observation \
             confidence: req={req} conf={conf}"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn salience_request_never_encodes_an_abort() {
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        let r = arm.decide(clock(0.5, 1, Some(GEAR_CAMPAIGN)));
        let req = r.to_salience_request(1);
        assert_eq!(
            req.schema_version,
            crate::gearbox::TSH_SALIENCE_SCHEMA_VERSION
        );
        assert!(req.requested_gear <= crate::gearbox::GEAR_MAX_ACTIVE);
        assert_ne!(req.requested_gear, crate::gearbox::GEAR_ABORT);
        assert_eq!(req.state, crate::gearbox::TSH_SALIENCE_VALID);

        let unarmed = TshGearboxArm::new(TshArmPolicy::unarmed_defaults(), ps(0.0)).decide(clock(
            0.5,
            1,
            Some(GEAR_SAFE),
        ));
        let unset = unarmed.to_salience_request(1);
        assert_eq!(unset.state, crate::gearbox::TSH_SALIENCE_UNSET);
        assert!(unset.salience.is_nan());
    }

    #[test]
    fn every_branch_declares_an_intensive_activity_unit_normalised_per_simulated_ps() {
        // A branch whose unit is a bare magnitude saturates `r/(r+reference)`
        // and turns a measurement into a constant. The unit string is the
        // contract, so it must exist, be distinct per family, and name the
        // simulated-time normalisation — except the one branch that is
        // explicitly unbound and says so.
        let mut seen: Vec<&'static str> = Vec::new();
        for b in SalienceBranch::ALL {
            let u = b.activity_unit();
            assert!(!u.is_empty(), "{} declares no activity unit", b.as_str());
            assert!(
                !seen.contains(&u),
                "{} reuses another family's unit '{u}'",
                b.as_str()
            );
            seen.push(u);
            if matches!(b, SalienceBranch::SignalGridWarp) {
                assert_eq!(u, "unit_undeclared_branch_unbound");
                continue;
            }
            assert!(
                u.ends_with("per_simulated_ps"),
                "{} must normalise per SIMULATED ps, got '{u}'",
                b.as_str()
            );
            assert!(
                u.contains("change") || u.contains("displacement") || u.contains("events"),
                "{} must stage a RATE OF CHANGE or a normalised deviation, not a \
                 magnitude; got '{u}'",
                b.as_str()
            );
        }
    }

    #[test]
    fn a_branch_with_no_producer_is_reported_as_structurally_absent_not_calm() {
        // The invariant: a branch with no input must never be indistinguishable
        // from a branch reporting calm. Two facts have to be separable in the
        // artifact — "no producer exists" and "the producer has not reported".
        let policy = policy_with(quiet_sentinel_doc());
        let mut arm = TshGearboxArm::new(policy, ps(0.0));
        arm.observe(BranchObservation {
            branch: SalienceBranch::ForceEnergyStability,
            accumulated_activity: 0.0,
            span: span(0.25),
            window_end: ps(0.25),
            sample_count: 4,
        })
        .expect("staged");
        let _ = arm.decide(clock(0.25, 1, Some(GEAR_SAFE)));

        let rows = arm.branch_binding_report();
        assert_eq!(rows.len(), TSH_ARM_N_BRANCHES);

        let force = rows[SalienceBranch::ForceEnergyStability.index()];
        assert!(force.run_path_producer.is_some(), "force has a producer");
        assert!(!force.structurally_absent);
        assert!(force.ever_observed);
        // Reported calm: an ADMISSIBLE salience of exactly zero.
        assert_eq!(force.salience, Some(0.0));

        let geometry = rows[SalienceBranch::GeometryContact.index()];
        assert!(
            geometry.run_path_producer.is_some(),
            "the geometry branch must have a wired producer or be declared absent"
        );

        let kcc = rows[SalienceBranch::KccCoupling.index()];
        assert!(kcc.structurally_absent, "no producer exists in this build");
        assert!(!kcc.ever_observed);
        // Structurally absent: salience is UNMEASURED, never Some(0.0).
        assert_eq!(kcc.salience, None);
        assert_ne!(
            kcc.salience, force.salience,
            "an absent branch and a calm branch must not read the same"
        );
    }

    #[test]
    fn branch_reference_rate_defaults_are_spelled_out_per_branch() {
        // The vector is numerically uniform but semantically per-branch: each
        // entry carries its own unit. Collapsing it back to a single scalar is
        // how a unit mismatch gets introduced without anything noticing.
        assert_eq!(DEFAULT_BRANCH_REFERENCE_RATES.len(), TSH_ARM_N_BRANCHES);
        let p = TshArmPolicy::unarmed_defaults();
        assert_eq!(p.branch_reference_rate, DEFAULT_BRANCH_REFERENCE_RATES);
        for (i, r) in DEFAULT_BRANCH_REFERENCE_RATES.iter().enumerate() {
            assert!(
                r.is_finite() && *r > 0.0,
                "branch {i} reference rate must be finite and positive"
            );
        }
    }
}
