//! PRISM-TWIN v3.0 — GPU-resident Protocol State Machine
//!
//! ProtocolState lives in VRAM. The Director kernel updates it each step
//! BEFORE physics kernels read from it. This makes the entire simulation
//! loop capturable as a CUDA Graph (Gate 2).
//!
//! Gate 0: ProtocolState struct + Director kernel (standalone validation)
//! Gate 1: Physics kernels read from ProtocolState (replaces scalar params)

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::fused_engine::CryoUvProtocol;
use crate::twin_kernels::find_twin_ptx;

/// Stage 2: Closed-loop steering focus residue with weight.
///
/// One entry in `ProtocolState::steering_focus_residues`. The ASC controller
/// fills this array each chunk with the top-K residues by GC-PID
/// synergy_fraction (Stage 1B-3 — see `crates/prism-nhs/src/gcpid.rs`).
/// The `ring_buffer_read_and_adapt` kernel reads the array on its next
/// launch and applies a multiplicative threshold-reduction boost to spikes
/// whose primary residue id matches an active entry.
///
/// `residue_id = -1` and/or `weight = 0.0` mark the entry as inactive.
///
/// Layout MUST match the CUDA `SteerEntry` struct nested in
/// `ProtocolState` in `protocol_state.cuh`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SteerEntry {
    pub residue_id: i32,
    pub weight: f32,
}

/// Maximum number of focus residues the steering buffer can hold.
/// MUST match the CUDA `steering_focus_residues[64]` array dimension.
pub const STEERING_FOCUS_MAX: usize = 64;
/// Maximum protocol UV scan wavelengths retained in GPU ProtocolState.
/// MUST match `scan_wavelengths[8]` in protocol_state.cuh.
pub const MAX_SCAN_WAVELENGTHS: usize = 8;

// ─────────────────────────────────────────────────────────────────────────────
// ProtocolState — must match protocol_state.cuh struct exactly
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-resident protocol state. Layout must be identical to the CUDA struct
/// in protocol_state.cuh (716 bytes, naturally aligned).
///
/// Fields are partitioned into:
/// - **Immutable** (set once at init, safe to bake as scalar args)
/// - **Dynamic** (updated by Director kernel each step, must live in VRAM)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProtocolState {
    // ── Step tracking ──
    pub current_step: u32, // DYNAMIC: incremented by Director
    pub total_steps: u32,  // Immutable

    // ── Temperature (CCNS 5-phase) ──
    pub current_temperature: f32, // DYNAMIC: computed by Director
    pub start_temp: f32,          // Immutable
    pub end_temp: f32,            // Immutable

    // ── Phase boundaries (set once at init) ──
    pub cold_hold_end: i32, // Immutable
    pub ramp_end: i32,      // Immutable
    pub warm_hold_end: i32, // Immutable
    pub ramp_down_end: i32, // Immutable

    // ── UV burst state ──
    pub uv_burst_active: i32,   // DYNAMIC: 0 or 1
    pub uv_burst_energy: f32,   // Immutable (base energy)
    pub uv_wavelength_nm: f32,  // DYNAMIC: current wavelength
    pub uv_burst_interval: i32, // Immutable
    pub uv_burst_duration: i32, // Immutable
    pub uv_target_idx: i32,     // DYNAMIC: current aromatic target

    // ── Wavelength scan table ──
    pub scan_wavelengths: [f32; MAX_SCAN_WAVELENGTHS], // Immutable
    pub n_wavelengths: i32,                            // Immutable
    pub wavelength_dwell_steps: i32,                   // Immutable

    // ── Langevin parameters ──
    pub dt: f32,     // DYNAMIC if adaptive_dt enabled
    pub gamma: f32,  // Immutable (base value, kept for compat)
    pub cutoff: f32, // Immutable

    // ── Fused step tracking ──
    pub fused_inner_steps: i32,  // Immutable
    pub fused_step_counter: i32, // DYNAMIC

    // ════════════════════════════════════════════════════════════════════════
    // Gate 1 expansion: dynamic thermodynamic state (+32 bytes)
    // ════════════════════════════════════════════════════════════════════════

    // ── Friction computation ──
    pub gamma_base: f32,          // Immutable: base friction coefficient
    pub cryo_enabled: i32,        // Immutable: 0 or 1
    pub equilibration_steps: i32, // Immutable: steps for high-friction warmup
    pub equilibration_gamma: f32, // Immutable: extreme friction value (1000.0)
    pub effective_gamma: f32,     // DYNAMIC: computed by Director each step

    // ── Adaptive timestep ──
    pub adaptive_dt_enabled: i32, // Immutable policy:0 disabled,1 bounded holdsbase/rampshalf,2 explicit historicalMay holds1.5base/rampsbase
    pub base_dt: f32,             // Immutable: timestep before adaptive scaling

    // ── Multi-LIF coupling double-buffer phase ──
    pub coupling_phase: i32, // DYNAMIC: 0 or 1, toggled by Director

    // ════════════════════════════════════════════════════════════════════════
    // Gate 2 expansion: autonomous housekeeping + heartbeat (+12 bytes)
    // ════════════════════════════════════════════════════════════════════════
    pub nl_rebuild_interval: i32, // Immutable: steps between NL rebuilds (default 20)
    pub com_removal_interval: i32, // Immutable: steps between COM removal (default 100)
    pub status_code: i32,         // DYNAMIC: 0=OK, 1=NaN, 2=diverged (written by heartbeat)

    // ════════════════════════════════════════════════════════════════════════
    // ASC Fusion Controller hooks (+16 bytes)
    // ════════════════════════════════════════════════════════════════════════
    /// Multiplicative UV energy adjustment (1.0 = neutral). Written by coupling kernel.
    pub steering_uv_boost: f32,
    /// Additive temperature offset in K (0.0 = neutral). Written by coupling kernel.
    pub steering_temp_bias: f32,
    /// Residue ID to focus UV on (-1 = no focus). Written by coupling kernel.
    pub steering_focus_residue: i32,
    /// Bit flags: 0x1=phase_lock, 0x2=cryo_stabilize, 0x4=adaptive_slow
    pub steering_flags: i32,

    // ════════════════════════════════════════════════════════════════════════
    // Phase-coherence metrology (4 bytes)
    // ════════════════════════════════════════════════════════════════════════
    /// 10-bit CCNS phase angle (0-1023), updated by Director each step
    pub current_phase_bits: u32,

    // ════════════════════════════════════════════════════════════════════════
    // Stage 2: Closed-loop ASC steering — focus residues (516 bytes)
    // ════════════════════════════════════════════════════════════════════════
    //
    // The ASC controller writes the top-K residues by GC-PID
    // synergy_fraction into this fixed-size array each chunk via
    // cuMemcpyHtoDAsync on the engine stream. The
    // `ring_buffer_read_and_adapt` kernel reads it on its next launch and
    // applies a multiplicative threshold-reduction boost to spikes whose
    // primary residue id matches.
    //
    // The autonomous CUDA Graph captures a POINTER to this struct (not a
    // value), so contents change between graph replays without requiring
    // re-capture — the canonical "captured pointer with mutable contents"
    // pattern. See graph_capture.rs and persistent_engine.rs::kcc_step_once
    // for the same pattern at work elsewhere.
    //
    // We steer toward synergy_fraction-ranked residues, NOT pocket
    // detection residues. The Stage 1B-3 validation on 4LPK showed the
    // SII signature pocket residues have HIGH total MI but LOW
    // synergy_fraction (their information is mostly redundant across
    // groups), while OTHER residues (152, 45, 65, 105, ...) have LOWER
    // total MI but HIGHER synergy_fraction (cross-group coordination
    // points). Steering toward coordination points produces real
    // cross-group divergence; steering toward already-agreed-upon
    // pocket residues does nothing.
    /// Number of active entries in `steering_focus_residues` (0..64).
    pub steering_focus_count: i32,

    /// Top-K focus residues by GC-PID synergy_fraction. Inactive entries
    /// have `residue_id = -1` and `weight = 0.0`.
    pub steering_focus_residues: [SteerEntry; STEERING_FOCUS_MAX],

    /// Diagnostic: incremented (atomicAdd) inside `ring_buffer_read_and_adapt`
    /// every time a spike's primary residue id matches an active focus entry.
    /// Lets the host verify end-to-end that the steering writeback is
    /// actually being read by the kernel — distinguishes "loop dead" from
    /// "loop alive but boost saturated by threshold floor". Reset to 0 by
    /// the Director's init kernel; never decremented.
    pub focus_match_count: u32,

    /// Diagnostic: incremented for every spike the kernel actually processes
    /// in its inner loop. Lets us distinguish "kernel never runs" (counter=0)
    /// from "kernel runs but matches fail" (counter>0 but focus_match_count=0).
    pub processed_spike_count: u32,

    /// Diagnostic: the kernel writes `steering_focus_residues[0].residue_id`
    /// here on every launch. Lets the host verify the kernel sees the same
    /// data the upload wrote (catches struct layout mismatch).
    pub last_seen_focus_id: i32,

    /// Diagnostic: the kernel writes the LAST spike's `primary_residue_id &
    /// 0xFFFF` here. Lets the host see what residue ID space the spikes
    /// actually live in (catches a residue id encoding mismatch).
    pub last_seen_spike_residue: i32,
}

const _: () = {
    // Compile-time size check: must match CUDA struct.
    //
    // Layout breakdown:
    //   148 (Gates 0-2)
    // +  16 (legacy single-residue ASC hooks)
    // +   4 (current_phase_bits)
    // +   4 (steering_focus_count)         ← Stage 2
    // +  16 (expanded scan_wavelengths[8])
    // + 512 (steering_focus_residues[64])  ← Stage 2 (64 × 8 bytes)
    // +   4 (focus_match_count)            ← Stage 2 calibration
    // +   4 (processed_spike_count)        ← Stage 2 calibration
    // +   4 (last_seen_focus_id)           ← Stage 2 calibration
    // +   4 (last_seen_spike_residue)      ← Stage 2 calibration
    // = 716 bytes total
    assert!(std::mem::size_of::<ProtocolState>() == 716);
    assert!(std::mem::size_of::<SteerEntry>() == 8);
};

// ══════════════════════════════════════════════════════════════════════════
// P2 — UV CONTROL INSTRUMENT (run-level declaration)
// ══════════════════════════════════════════════════════════════════════════
//
// WHY THIS IS RUN-LEVEL AND NOT A PROTOCOL FIELD.
// UV suppression is an operator DECLARATION about the whole run ("this is a
// thermal-baseline control"), not a per-protocol parameter. A per-protocol
// field cannot be made total: several production lanes build CryoUvProtocol
// from constructor literals and upload them without passing through any
// override path — `run_multi_differential_pipeline` via
// `CryoUvProtocol::twin_differential_set()`, `run_coupled_twin_pipeline`, and
// `BatchProcessor::process_batch`, which re-uploads a fresh literal per
// structure. A missed lane there would produce a FULLY PUMPED run that
// self-certifies as a control, which is strictly worse than no control at all.
//
// WHY THE ARMING IS TOTAL BY CONSTRUCTION.
// `ProtocolState::from_cryo_uv` is the sole non-test constructor of the
// device-side ProtocolState, and it has exactly ONE non-test caller:
// `FusedNhsEngine::set_cryo_uv_protocol` (fused_engine.rs), which is the only
// route to the `memcpy_htod` that publishes ProtocolState to VRAM.
// `PersistentNhsEngine::set_cryo_uv_protocol` delegates to it. Therefore every
// upload — on every lane, present or future — reads this state. A new caller
// cannot bypass it without also bypassing the only upload path.
//
// WHY NO NEW ProtocolState FIELD.
// ProtocolState is size-asserted at 716 bytes on the host and mirrored in
// protocol_state.cuh; adding a field breaks the FFI contract on both sides.
// Suppression is instead expressed through the EXISTING `uv_burst_interval`
// field, which the Director already treats as a hard off-switch:
//   protocol_director.cu:140-154 and :298-300
//     if (state->uv_burst_interval > 0) { ... } else { state->uv_burst_active = 0; }
// With the interval at 0 the Director takes the else branch every step, so
// `uv_burst_active` is permanently 0, and the burst trigger in
// `nhs_uv_pump_probe_step` (nhs_amber_fused.cu) never fires. That trigger
// dominates the single write of `is_excited[i] = 1`
// (nhs_excited_state.cuh, inside `excite_aromatic_wavelength`), so ALL THREE
// consequences of excitation are suppressed together:
//   (a) excited-state partial-charge rescale inside the Coulomb loop
//   (b) vibrational energy deposit into the 5 A shell
//   (c) reduced local hydrophobicity collapsing the water-density field
// No kernel change is required, and therefore no PTX regeneration.
//
// WHY NOT `--uv-burst-energy 0.0`.
// `0.0` is the KERNEL'S declared sentinel for "no override supplied"
// (nhs_excited_state.cuh documents `<= 0.0f` as preserving the built-in
// cross-section baseline). Writing an operator-meant "off" into that slot is
// what produced the original defect: the documented control condition was the
// treatment condition. The kernel contract is correct; the host was wrong to
// overload it. A negative value is equally unusable — it takes the identical
// branch, and it is arithmetically live upstream.

/// Per-protocol UV schedule governs. Default; byte-for-byte pre-P2 behaviour.
pub const UV_BURST_MODE_PROTOCOL: u32 = 0;
/// Operator supplied an explicit burst magnitude via `--uv-burst-energy N`.
pub const UV_BURST_MODE_OVERRIDE: u32 = 1;
/// Operator DECLARED a UV-off control run via `--uv-burst-suppress`.
///
/// Deliberately out-of-band: the dangerous misparse here is a stale or zeroed
/// read certifying a pumped run AS a control, so `0` is bound to PROTOCOL
/// (i.e. treatment) and suppression is reachable by no accidental value.
pub const UV_BURST_MODE_SUPPRESSED: u32 = 0xFFFF_FFFF;

/// `ProtocolState.steering_flags` bit carrying the P2 UV-off declaration to the
/// device and back.
///
/// Bits 0x1 / 0x2 / 0x4 are the pre-existing phase_lock / cryo_stabilize /
/// adaptive_slow allocation (protocol_state.cuh:89). Bits 0x8 upward are free:
/// the only tree-wide writers of this word are `ring_buffer.cu` (`|= 0x1`)
/// and `protocol_director.cu` (`&= ~0x1`), both non-clobbering, and
/// `fused_engine.rs`, which read-modify-writes with `| 0x1` from a device
/// readback and therefore preserves this bit.
///
/// Using a free bit rather than a new field keeps `size_of::<ProtocolState>()`
/// at 716 and moves no FFI offset. It rides the ProtocolState VRAM readback in
/// the per-stream teardown, so the declaration is DEVICE-ATTESTED, not merely
/// host-asserted — the `focus_match_count` precedent (see above).
pub const UV_BURST_SUPPRESSED_FLAG: i32 = 0x8;

/// Preserve the warm tail when neither ramp-down nor cold return was requested.
/// The unused flag keeps the device ABI unchanged and is preserved by ASC writes.
pub const WARM_TAIL_WITHOUT_COOLING_FLAG: i32 = 0x10;

static UV_BURST_MODE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(UV_BURST_MODE_PROTOCOL);

/// Declare the run's UV burst mode. Called once from argv before any engine is
/// constructed. Idempotent; later identical declarations are accepted.
pub fn declare_uv_burst_mode(mode: u32) {
    UV_BURST_MODE.store(mode, std::sync::atomic::Ordering::SeqCst);
}

/// The run's declared UV burst mode. Read at every ProtocolState upload.
pub fn uv_burst_mode() -> u32 {
    UV_BURST_MODE.load(std::sync::atomic::Ordering::SeqCst)
}

/// True when the operator declared a UV-off control run.
pub fn uv_burst_suppressed() -> bool {
    uv_burst_mode() == UV_BURST_MODE_SUPPRESSED
}

/// Stable token for receipts and logs, so a control run is identifiable from
/// evidence alone rather than by re-reading argv.
pub fn uv_burst_mode_token(mode: u32) -> &'static str {
    match mode {
        UV_BURST_MODE_PROTOCOL => "protocol_schedule",
        UV_BURST_MODE_OVERRIDE => "operator_magnitude_override",
        UV_BURST_MODE_SUPPRESSED => "suppressed_declared_operator_uv_off_control",
        _ => "unrecognized_uv_burst_mode",
    }
}

// ════════════════════════════════════════════════════════════════════
// P7a — ASC (Adjudicated Steering Control) RUN-LEVEL DECLARATION
// ════════════════════════════════════════════════════════════════════
//
// Mirrors the P2 UV-burst declaration above, for the same reason: a run's
// receipts must state what the run DID, and must not leave a reader to infer
// it from argv or from the presence of a file.
//
// The ASC surface has two distinct implementations and they are not
// interchangeable:
//
//   * `prism_asc_apply_kernel` (cuda/adjudicator.cu) — captured as Node D of
//     the V2 pipeline when `PipelineConfig::asc` is `Some`. Non-atomic
//     `d_forces[i*3+k] += scale * d{x,y,z}`, self-gated on
//     `adjudication_code == PRISM_ADJ_CONSTRUCT`.
//
//   * `asc_inject_repulsion_v4_kernel` (cuda/asc_steering.cu) — vector
//     atomics, `atom.global.add.v4.f32` on sm_120+. Compiled by build.rs into
//     the `asc_steering` static archive and has ZERO launchers tree-wide. It
//     is preserved as a reachable deprecated capability, not deleted, and it
//     is NOT what a run exercises. A receipt that says "ASC" without saying
//     WHICH is not evidence.
//
// Engagement is not the same as effect. Node D accumulates into the force
// buffer that the fused engine's FORCE_N / FORCE_NP1 phases overwrite with
// `forces[tid] = local_force` — an assignment. Whether a contribution
// survives to an integrator kick is a replay-ordering property, which is why
// the kernel now also reports measured application counts (see
// `AscConfig::d_asc_applied_atoms`). `asc_mode` states what was CONFIGURED;
// the device counters state what was APPLIED. Both belong in the receipt.

/// Node D was not captured. `PipelineConfig::asc` was `None`; no ASC kernel
/// exists in the replayed graph. Default, and the pre-P7a behaviour of every
/// construction site but one.
pub const ASC_MODE_DISENGAGED: u32 = 0;

/// Node D captured with a DIFFERENTIATED per-atom cluster mask — some atoms
/// in, some out. This is the designed configuration: ASC steers a pocket.
pub const ASC_MODE_ENGAGED_DIFFERENTIATED: u32 = 1;

/// Node D captured with an UNDIFFERENTIATED (all-ones) cluster mask.
///
/// Every atom reads as in-cluster, so the steering term is applied to the
/// whole system rather than to a pocket: F_i = α·Δ·(x_i − X_c) for all i,
/// a global outward dilation about the tile centroid. That is a different
/// experiment from pocket steering, not a degenerate case of it, and it must
/// never be reported under the same token.
///
/// Deliberately out-of-band from DIFFERENTIATED so a stale or zeroed read
/// cannot certify an undifferentiated run as a pocket-steered one.
pub const ASC_MODE_ENGAGED_UNDIFFERENTIATED: u32 = 0xFFFF_FFFE;

static ASC_MODE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(ASC_MODE_DISENGAGED);

/// Declare the run's ASC engagement mode. Called from the V2 pipeline build
/// once the mask's character is known. Idempotent.
pub fn declare_asc_mode(mode: u32) {
    ASC_MODE.store(mode, std::sync::atomic::Ordering::SeqCst);
}

/// The run's declared ASC engagement mode.
pub fn asc_mode() -> u32 {
    ASC_MODE.load(std::sync::atomic::Ordering::SeqCst)
}

/// True when no ASC kernel was captured into the replayed graph.
pub fn asc_disengaged() -> bool {
    asc_mode() == ASC_MODE_DISENGAGED
}

/// Stable token for receipts and logs.
pub fn asc_mode_token(mode: u32) -> &'static str {
    match mode {
        ASC_MODE_DISENGAGED => "disengaged_node_d_not_captured",
        ASC_MODE_ENGAGED_DIFFERENTIATED => "engaged_differentiated_cluster_mask",
        ASC_MODE_ENGAGED_UNDIFFERENTIATED => "engaged_undifferentiated_all_atom_mask",
        _ => "unrecognized_asc_mode",
    }
}

/// Which ASC implementation a run can exercise. There is exactly one, and it
/// is not the vectorized kernel the header documentation is written around.
pub const ASC_ACTIVE_IMPLEMENTATION: &str = "prism_asc_compose_v1@cuda/adjudicator.cu+prism_asc_measure_kick@nhs_amber_fused.cu";

/// The vectorized ASC kernel, compiled into the `asc_steering` static archive
/// with no launcher. Named here so a receipt can state that it exists, is
/// reachable, and was NOT exercised — rather than omitting it and leaving a
/// reader to assume it was.
pub const ASC_UNLAUNCHED_IMPLEMENTATION: &str =
    "asc_inject_repulsion_v4_kernel@cuda/asc_steering.cu";

// ════════════════════════════════════════════════════════════════════
// P8 — G26 DEVICE GOVERNOR ARMING DECLARATION
// ════════════════════════════════════════════════════════════════════
//
// There are TWO governors in a run and they had different armed states.
//
//   * The HOST bridge (`PhysicalGovernorBridge` in nhs_rt_full) resolves a
//     policy with `armed = 1`, composes all gates, and actuates by writing
//     `adj.gear_override@100`.
//
//   * The DEVICE governor inside `prism_gearbox_sfa_kernel` reads the
//     `__constant__ d_gear_governor_policy` and assigns its verdict straight
//     into `cruise->current_gear` (gearbox.cu:635) — the gear the SWITCH body
//     uses to select dt.
//
// `prism_gearbox_governor_arm_async` is the ONLY writer of that __constant__,
// and before P8 its Rust wrapper `gearbox::arm_governor` had zero callers
// tree-wide — as did `bind_governor_measurements`, `publish_governor_salience`
// and `snapshot_governor`. The device policy therefore kept its static
// initialiser `armed = 0` forever, pinning the ceiling to
// PRISM_GEAR_DEFAULT_UNKNOWN_TOPOLOGY_CAP (Gear 1 / 2.0 fs) and attributing it
// to GEAR_GATE_GOVERNOR_UNARMED — while the host logged "ARMED".
//
// This declaration makes which regime a run executed in a fact the receipt
// states, rather than one a reader must infer from a log line that was wrong.

/// The device `__constant__` policy was never uploaded. Pre-P8 behaviour.
pub const GOVERNOR_DEVICE_UNARMED: u32 = 0;
/// The host-resolved policy was uploaded to the device.
pub const GOVERNOR_DEVICE_ARMED: u32 = 1;
/// `arm_governor` was called and refused or failed. The device governor is
/// unarmed, and — unlike the pre-P8 silence — the run says so.
pub const GOVERNOR_DEVICE_ARM_FAILED: u32 = 0xFFFF_FFFD;
/// Armed, but `bind_governor_measurements` failed, so the force guard reads
/// UNMEASURED and gate 5 pins the ceiling to the current gear.
pub const GOVERNOR_DEVICE_ARM_BIND_FAILED: u32 = 0xFFFF_FFFC;

static GOVERNOR_DEVICE_ARM: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(GOVERNOR_DEVICE_UNARMED);

/// Declare the device governor's arming outcome for this run.
pub fn declare_governor_device_arm(state: u32) {
    GOVERNOR_DEVICE_ARM.store(state, std::sync::atomic::Ordering::SeqCst);
}

/// The run's device governor arming outcome.
pub fn governor_device_arm() -> u32 {
    GOVERNOR_DEVICE_ARM.load(std::sync::atomic::Ordering::SeqCst)
}

/// Stable token for receipts and logs.
pub fn governor_device_arm_token(state: u32) -> &'static str {
    match state {
        GOVERNOR_DEVICE_UNARMED => "device_unarmed_ceiling_pinned_to_unknown_topology_cap",
        GOVERNOR_DEVICE_ARMED => "device_armed_policy_uploaded",
        GOVERNOR_DEVICE_ARM_FAILED => "device_arm_attempted_and_failed",
        GOVERNOR_DEVICE_ARM_BIND_FAILED => "device_armed_but_measurement_bind_failed",
        _ => "unrecognized_governor_device_arm_state",
    }
}

/// Why an armed device governor still cannot upshift.
///
/// Gate 6 (constraint stability) pins the ceiling to the CURRENT gear whenever
/// `constraint_measured == 0` and `require_constraint_guard_for_upshift != 0`,
/// and that default is 1. No device-side SHAKE/RATTLE failure counter exists
/// in `nhs_amber_fused.cu` — the solvers return a local `bool converged` and
/// accumulate nothing — so the guard cannot be bound. P8 therefore arms the
/// governor and binds the FORCE guard (enabling downshift on an excursion,
/// the safety direction) while upshift stays structurally blocked. Unblocking
/// it changes the timestep and is what requires the re-baseline.
pub const GOVERNOR_UPSHIFT_BLOCKER: &str =
    "constraint_guard_unbound_no_device_shake_rattle_failure_counter";

/// Sentinel: the device governor was never snapshotted this run.
pub const GOVERNOR_SNAPSHOT_NOT_TAKEN: u32 = 0xFFFF_FFFF;

static GOV_SNAP_ARMED_ECHO: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(GOVERNOR_SNAPSHOT_NOT_TAKEN);
static GOV_SNAP_AUTHORIZED_GEAR: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(GOVERNOR_SNAPSHOT_NOT_TAKEN);
static GOV_SNAP_BINDING_GATE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(GOVERNOR_SNAPSHOT_NOT_TAKEN);
static GOV_SNAP_FORCE_NORM_BITS: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(GOVERNOR_SNAPSHOT_NOT_TAKEN);

/// Record a device governor snapshot read back at teardown.
///
/// Last writer wins: the snapshot is the governor's FINAL state, and every
/// stream shares one device `__constant__` policy and one state object.
pub fn record_governor_snapshot(
    armed_echo: u32,
    authorized_gear: u32,
    binding_gate: u32,
    observed_force_norm: f32,
) {
    use std::sync::atomic::Ordering::SeqCst;
    GOV_SNAP_ARMED_ECHO.store(armed_echo, SeqCst);
    GOV_SNAP_AUTHORIZED_GEAR.store(authorized_gear, SeqCst);
    GOV_SNAP_BINDING_GATE.store(binding_gate, SeqCst);
    GOV_SNAP_FORCE_NORM_BITS.store(observed_force_norm.to_bits(), SeqCst);
}

/// The device governor's final snapshot, or `None` if never taken.
///
/// Returns `(armed_echo, authorized_gear, binding_gate, observed_force_norm)`.
/// `observed_force_norm` is NaN when the force guard read UNMEASURED, which is
/// deliberately distinct from a measured zero.
pub fn governor_snapshot() -> Option<(u32, u32, u32, f32)> {
    use std::sync::atomic::Ordering::SeqCst;
    if GOV_SNAP_ARMED_ECHO.load(SeqCst) == GOVERNOR_SNAPSHOT_NOT_TAKEN {
        return None;
    }
    Some((
        GOV_SNAP_ARMED_ECHO.load(SeqCst),
        GOV_SNAP_AUTHORIZED_GEAR.load(SeqCst),
        GOV_SNAP_BINDING_GATE.load(SeqCst),
        f32::from_bits(GOV_SNAP_FORCE_NORM_BITS.load(SeqCst)),
    ))
}

/// Sentinel: no ASC application measurement was taken this run.
///
/// Distinct from a measured zero. "The counters were never read" and "the
/// kernel applied steering to zero atoms" are different facts and a receipt
/// that renders both as `0` is not evidence.
pub const ASC_APPLIED_NOT_MEASURED: u64 = u64::MAX;

static ASC_APPLIED_ATOMS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(ASC_APPLIED_NOT_MEASURED);
static ASC_APPLIED_SUM_F_BITS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(ASC_APPLIED_NOT_MEASURED);

/// Record a device-side ASC application measurement read back at teardown.
///
/// Accumulates across streams: each V2 stream owns its own counter pair, and
/// the run-level figure is their sum.
pub fn record_asc_application(applied_atoms: u32, applied_sum_f: f32) {
    use std::sync::atomic::Ordering::SeqCst;
    let _ = ASC_APPLIED_ATOMS.fetch_update(SeqCst, SeqCst, |cur| {
        Some(if cur == ASC_APPLIED_NOT_MEASURED {
            applied_atoms as u64
        } else {
            cur.saturating_add(applied_atoms as u64)
        })
    });
    let _ = ASC_APPLIED_SUM_F_BITS.fetch_update(SeqCst, SeqCst, |cur| {
        let prev = if cur == ASC_APPLIED_NOT_MEASURED {
            0.0f64
        } else {
            f64::from_bits(cur)
        };
        Some((prev + applied_sum_f as f64).to_bits())
    });
}

/// Atoms that measurably received a non-zero ASC steering force, summed over
/// streams. `None` when no measurement was taken.
pub fn asc_applied_atoms() -> Option<u64> {
    match ASC_APPLIED_ATOMS.load(std::sync::atomic::Ordering::SeqCst) {
        ASC_APPLIED_NOT_MEASURED => None,
        v => Some(v),
    }
}

/// Σ|F_ASC| over the counted atoms, summed over streams. `None` when no
/// measurement was taken.
pub fn asc_applied_sum_f() -> Option<f64> {
    match ASC_APPLIED_SUM_F_BITS.load(std::sync::atomic::Ordering::SeqCst) {
        ASC_APPLIED_NOT_MEASURED => None,
        bits => Some(f64::from_bits(bits)),
    }
}

impl ProtocolState {
    /// Build from CryoUvProtocol + simulation parameters.
    ///
    /// Phase boundaries are precomputed as cumulative step counts:
    ///   cold_hold_end = cold_hold_steps
    ///   ramp_end      = cold_hold_end + ramp_steps
    ///   warm_hold_end = ramp_end + warm_hold_steps
    ///   ramp_down_end = warm_hold_end + ramp_down_steps
    pub fn from_cryo_uv(
        protocol: &CryoUvProtocol,
        total_steps: u32,
        dt: f32,
        gamma: f32,
        cutoff: f32,
        fused_inner_steps: i32,
        cryo_enabled: bool,
        adaptive_dt_enabled: bool,
    ) -> Self {
        // P2 — read the run-level declaration. Total by construction: this is
        // the only non-test constructor of the device ProtocolState, and its
        // only non-test caller is the sole path to the VRAM upload.
        let uv_suppressed = uv_burst_suppressed();

        let cold_hold_end = protocol.cold_hold_steps;
        let ramp_end = cold_hold_end + protocol.ramp_steps;
        let warm_hold_end = ramp_end + protocol.warm_hold_steps;
        let ramp_down_end = warm_hold_end + protocol.ramp_down_steps;

        // Pack all production protocol wavelengths without truncating HIS/BNZ channels.
        let mut scan_wl = [0.0f32; MAX_SCAN_WAVELENGTHS];
        let n_wl = protocol.scan_wavelengths.len().min(MAX_SCAN_WAVELENGTHS);
        for i in 0..n_wl {
            scan_wl[i] = protocol.scan_wavelengths[i];
        }

        Self {
            current_step: 0,
            total_steps,
            current_temperature: protocol.start_temp,
            start_temp: protocol.start_temp,
            end_temp: protocol.end_temp,
            cold_hold_end,
            ramp_end,
            warm_hold_end,
            ramp_down_end,
            uv_burst_active: 0,
            uv_burst_energy: protocol.uv_burst_energy,
            uv_wavelength_nm: scan_wl[0],
            // P2 — the burst SCHEDULE is preserved exactly. Suppression removes
            // the pump kernel LAUNCH (fused_engine.rs,
            // `launch_post_fused_phase_split`); it does not silence the Director.
            //
            // Zeroing `uv_burst_interval` would drive protocol_director.cu:152-154
            // (and :310) and pin `uv_burst_active` to 0 for the whole run. That is
            // UV_ABSENT, not UV_SUPPRESSED, and it changes the control arm in two
            // ways that have nothing to do with excitation:
            //   * nhs_amber_fused.cu:3411 and :4661 gate the STANDARD water-density
            //     LIF update on `!uv_burst_active`, so the detector would run at a
            //     different duty cycle than in the treatment arm;
            //   * protocol_director.cu:144-151 sits inside `if (uv_burst_interval > 0)`,
            //     so the wavelength scan would freeze at `scan_wavelengths[0]` and
            //     `uv_target_idx` would pin at 0;
            //   * uv_lif_coupling.cuh:470 reads it as a 1.5x gain term.
            // Keeping the schedule makes this a MATCHED-SCHEDULE control whose only
            // difference from the treatment arm is that no aromatic is ever excited.
            //
            // `uv_burst_energy` is likewise left exactly as the protocol set it: the
            // pump kernel that would read it is never launched.
            uv_burst_interval: protocol.uv_burst_interval,
            uv_burst_duration: protocol.uv_burst_duration,
            uv_target_idx: 0,
            scan_wavelengths: scan_wl,
            n_wavelengths: n_wl as i32,
            wavelength_dwell_steps: protocol.wavelength_dwell_steps,
            dt,
            gamma,
            cutoff,
            fused_inner_steps,
            fused_step_counter: 0,
            // Gate 1 expansion
            gamma_base: gamma,
            cryo_enabled: cryo_enabled as i32,
            equilibration_steps: 10000,
            equilibration_gamma: 1000.0,
            effective_gamma: gamma,
            adaptive_dt_enabled: adaptive_dt_enabled as i32,
            base_dt: dt,
            coupling_phase: 0,
            // Gate 2
            nl_rebuild_interval: 20,
            com_removal_interval: 100,
            status_code: 0,
            // ASC hooks (neutral defaults)
            steering_uv_boost: 1.0,
            steering_temp_bias: 0.0,
            steering_focus_residue: -1,
            // P2 — device-attested UV-off declaration. Uploaded with the rest of
            // ProtocolState, preserved by the only two kernel writers of this word
            // (ring_buffer.cu:227 `|= 0x1`, protocol_director.cu:201 `&= ~0x1`),
            // read back out of VRAM at the per-stream teardown and emitted into
            // `*_protocol_state.json`. A reader holding only that file can compute
            // `steering_flags & 0x8` and know the device was armed.
            steering_flags: (if uv_suppressed {
                UV_BURST_SUPPRESSED_FLAG
            } else {
                0
            }) | if protocol.ramp_down_steps == 0 && protocol.cold_return_steps == 0 {
                WARM_TAIL_WITHOUT_COOLING_FLAG
            } else {
                0
            },
            // Phase metrology
            current_phase_bits: 0,
            // Stage 2 closed-loop steering — initialized empty.
            // The ASC controller will populate this each chunk via
            // cuMemcpyHtoDAsync once --closed-loop-steering is enabled.
            steering_focus_count: 0,
            steering_focus_residues: [SteerEntry {
                residue_id: -1,
                weight: 0.0,
            }; STEERING_FOCUS_MAX],
            focus_match_count: 0,
            processed_spike_count: 0,
            last_seen_focus_id: -2,
            last_seen_spike_residue: -2,
        }
    }

    /// Compute expected temperature at a given step (CPU reference implementation).
    /// Used for test validation against the GPU Director kernel.
    pub fn expected_temperature_at_step(&self, step: u32) -> f32 {
        let s = step as i32;
        if s < self.cold_hold_end {
            self.start_temp
        } else if s < self.ramp_end {
            let ramp_len = self.ramp_end - self.cold_hold_end;
            let progress = (s - self.cold_hold_end) as f32 / ramp_len as f32;
            self.start_temp + progress * (self.end_temp - self.start_temp)
        } else if s < self.warm_hold_end {
            self.end_temp
        } else if s < self.ramp_down_end {
            let ramp_len = self.ramp_down_end - self.warm_hold_end;
            let progress = (s - self.warm_hold_end) as f32 / ramp_len as f32;
            self.end_temp - progress * (self.end_temp - self.start_temp)
        } else if self.steering_flags & WARM_TAIL_WITHOUT_COOLING_FLAG != 0 {
            self.end_temp
        } else {
            self.start_temp
        }
    }

    /// Compute expected effective_gamma at a given step (CPU reference).
    /// Matches the Director kernel's gamma computation exactly.
    pub fn expected_gamma_at_step(&self, step: u32) -> f32 {
        let s = step as i32;
        let base_gamma = if s < self.equilibration_steps {
            let progress = s as f32 / self.equilibration_steps as f32;
            let decay = (-3.0 * progress).exp();
            self.equilibration_gamma * decay + self.gamma_base * (1.0 - decay)
        } else {
            self.gamma_base
        };

        let temp = self.expected_temperature_at_step(step);
        if self.cryo_enabled != 0 && temp < 300.0 {
            let t_clamped = temp.max(10.0);
            let scale = 300.0 / t_clamped;
            base_gamma * scale.sqrt()
        } else {
            base_gamma
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProtocolDirector — GPU kernel wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Manages the GPU-resident ProtocolState and Director kernel launches.
pub struct ProtocolDirector {
    /// Device allocation for ProtocolState.
    /// Passed as `const ProtocolState*` to physics kernels.
    pub d_state: CudaSlice<u8>,
    /// Cached host-side copy for diagnostics
    host_state: ProtocolState,
    // Kernel functions
    director_fn: CudaFunction,
    init_fn: CudaFunction,
    read_fn: CudaFunction,
}

impl ProtocolDirector {
    /// Load PTX and allocate device state.
    pub fn new(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<(Self, Arc<CudaModule>)> {
        let ptx_path = find_twin_ptx("protocol_director.ptx")?;
        let module = context
            .load_module(Ptx::from_file(&ptx_path))
            .with_context(|| format!("Failed to load protocol_director.ptx from {}", ptx_path))?;

        let director_fn = module
            .load_function("protocol_director")
            .context("protocol_director function not found in PTX")?;
        let init_fn = module
            .load_function("init_protocol_state")
            .context("init_protocol_state function not found in PTX")?;
        let read_fn = module
            .load_function("read_protocol_state")
            .context("read_protocol_state function not found in PTX")?;

        let state_size = std::mem::size_of::<ProtocolState>();
        let d_state = stream.alloc_zeros::<u8>(state_size)?;

        let host_state = unsafe { std::mem::zeroed::<ProtocolState>() };

        Ok((
            Self {
                d_state,
                host_state,
                director_fn,
                init_fn,
                read_fn,
            },
            module,
        ))
    }

    /// Initialize ProtocolState on GPU via direct memcpy.
    /// The struct is #[repr(C)] so byte layout matches the CUDA struct exactly.
    pub fn initialize(&mut self, stream: &Arc<CudaStream>, state: &ProtocolState) -> Result<()> {
        self.host_state = *state;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                state as *const ProtocolState as *const u8,
                std::mem::size_of::<ProtocolState>(),
            )
        };
        stream.memcpy_htod(bytes, &mut self.d_state)?;

        log::info!(
            "ProtocolDirector initialized: {}K -> {}K, {} total steps",
            state.start_temp,
            state.end_temp,
            state.total_steps
        );
        log::info!(
            "  Phase boundaries: cold_hold={}, ramp={}, warm_hold={}, ramp_down={}",
            state.cold_hold_end,
            state.ramp_end,
            state.warm_hold_end,
            state.ramp_down_end
        );
        log::info!(
            "  Gamma: base={:.1}, cryo={}, equilibration={} steps @ {:.0}",
            state.gamma_base,
            state.cryo_enabled != 0,
            state.equilibration_steps,
            state.equilibration_gamma
        );
        log::info!(
            "  Adaptive dt: {} (base={:.4} ps)",
            state.adaptive_dt_enabled != 0,
            state.base_dt
        );

        Ok(())
    }

    /// Run one Director step on the given stream.
    ///
    /// CRITICAL: Must be launched on the SAME stream as the physics kernels.
    /// CUDA stream ordering guarantees Director writes are visible to subsequent
    /// kernel launches on the same stream — no cudaDeviceSynchronize needed.
    pub fn step(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.director_fn)
                .arg(&mut self.d_state)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Read current state from GPU (synchronizes stream).
    /// Returns (current_step, current_temperature, uv_burst_active, uv_wavelength_nm).
    pub fn read_state(&self, stream: &Arc<CudaStream>) -> Result<(u32, f32, i32, f32)> {
        let mut out_step = stream.alloc_zeros::<u32>(1)?;
        let mut out_temp = stream.alloc_zeros::<f32>(1)?;
        let mut out_uv = stream.alloc_zeros::<i32>(1)?;
        let mut out_wl = stream.alloc_zeros::<f32>(1)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            stream
                .launch_builder(&self.read_fn)
                .arg(&self.d_state)
                .arg(&mut out_step)
                .arg(&mut out_temp)
                .arg(&mut out_uv)
                .arg(&mut out_wl)
                .launch(cfg)?;
        }

        let mut step = [0u32; 1];
        let mut temp = [0.0f32; 1];
        let mut uv = [0i32; 1];
        let mut wl = [0.0f32; 1];
        stream.memcpy_dtoh(&out_step, &mut step)?;
        stream.memcpy_dtoh(&out_temp, &mut temp)?;
        stream.memcpy_dtoh(&out_uv, &mut uv)?;
        stream.memcpy_dtoh(&out_wl, &mut wl)?;

        Ok((step[0], temp[0], uv[0], wl[0]))
    }

    /// Download entire ProtocolState from GPU (for full diagnostics).
    pub fn download_full_state(&self, stream: &Arc<CudaStream>) -> Result<ProtocolState> {
        let mut buf = vec![0u8; std::mem::size_of::<ProtocolState>()];
        stream.memcpy_dtoh(&self.d_state, &mut buf)?;
        let state: ProtocolState = unsafe { std::ptr::read(buf.as_ptr() as *const ProtocolState) };
        Ok(state)
    }

    /// Get cached host-side state (may be stale — use read_state() for live values).
    pub fn host_state(&self) -> &ProtocolState {
        &self.host_state
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Serializes every test that touches the process-global UV burst mode.
    /// `UV_BURST_MODE` is a run-level `AtomicU32`; cargo runs tests in threads
    /// inside ONE process, so a test that declares a mode must hold this lock or
    /// it will race with any other test that reads mode-dependent state.
    static UV_MODE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// P2 — the suppression mechanism, proven at the choke point.
    ///
    /// Suppression removes the pump kernel LAUNCH
    /// (fused_engine.rs `launch_post_fused_phase_split`); it does NOT zero the
    /// burst schedule. Zeroing `uv_burst_interval` would drive the Director's
    /// else branch (protocol_director.cu:152-154) and pin `uv_burst_active` to 0
    /// for the whole run — that is UV_ABSENT, not UV_SUPPRESSED, and it changes
    /// the LIF detector duty cycle (nhs_amber_fused.cu:3411, :4661) and freezes
    /// the wavelength scan (protocol_director.cu:144-151), neither of which has
    /// anything to do with excitation. The schedule is therefore preserved so the
    /// control is MATCHED-SCHEDULE, and the device-attested declaration rides
    /// `steering_flags` bit 0x8 instead.
    #[test]
    fn test_p2_uv_suppression_preserves_the_burst_schedule() {
        let _guard = UV_MODE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let protocol = CryoUvProtocol::standard();
        assert!(
            protocol.uv_burst_interval > 0,
            "precondition: the standard protocol must schedule bursts"
        );

        // Default (PROTOCOL): schedule preserved byte-for-byte, no declaration bit.
        declare_uv_burst_mode(UV_BURST_MODE_PROTOCOL);
        let unsuppressed =
            ProtocolState::from_cryo_uv(&protocol, 45000, 0.002, 1.0, 9.0, 4, true, true);
        assert_eq!(unsuppressed.uv_burst_interval, protocol.uv_burst_interval);
        assert_eq!(unsuppressed.uv_burst_duration, protocol.uv_burst_duration);
        assert_eq!(unsuppressed.uv_burst_energy, protocol.uv_burst_energy);
        assert_eq!(unsuppressed.steering_flags, 0);

        // OVERRIDE is a magnitude declaration, not a suppression: schedule intact.
        declare_uv_burst_mode(UV_BURST_MODE_OVERRIDE);
        let overridden =
            ProtocolState::from_cryo_uv(&protocol, 45000, 0.002, 1.0, 9.0, 4, true, true);
        assert_eq!(overridden.uv_burst_interval, protocol.uv_burst_interval);
        assert_eq!(overridden.uv_burst_duration, protocol.uv_burst_duration);
        assert_eq!(overridden.steering_flags, 0);

        // SUPPRESSED: schedule PRESERVED (matched-schedule control), declaration
        // bit armed so the UV-off claim survives the VRAM round trip.
        declare_uv_burst_mode(UV_BURST_MODE_SUPPRESSED);
        assert!(uv_burst_suppressed());
        let suppressed =
            ProtocolState::from_cryo_uv(&protocol, 45000, 0.002, 1.0, 9.0, 4, true, true);
        assert_eq!(
            suppressed.uv_burst_interval, protocol.uv_burst_interval,
            "the control must keep the treatment arm's burst schedule"
        );
        assert_eq!(suppressed.uv_burst_duration, protocol.uv_burst_duration);
        // Energy is deliberately untouched: the pump kernel that would read it is
        // never launched, and preserving it keeps the host consumers that
        // square-root and accumulate uv_burst_energy bit-identical.
        assert_eq!(suppressed.uv_burst_energy, protocol.uv_burst_energy);
        assert_eq!(suppressed.uv_burst_active, 0);
        assert_eq!(
            suppressed.steering_flags, UV_BURST_SUPPRESSED_FLAG,
            "the UV-off declaration must be device-attested via steering_flags 0x8"
        );

        declare_uv_burst_mode(UV_BURST_MODE_PROTOCOL);
    }

    /// The declaration bit must never collide with the pre-existing
    /// phase_lock / cryo_stabilize / adaptive_slow allocation (protocol_state.cuh:89),
    /// and `ProtocolState` must stay 716 bytes so no FFI offset moves.
    #[test]
    fn test_p2_uv_suppressed_flag_does_not_collide_with_live_bits() {
        assert_eq!(UV_BURST_SUPPRESSED_FLAG & 0x7, 0);
        assert_eq!(UV_BURST_SUPPRESSED_FLAG, 0x8);
        assert_eq!(std::mem::size_of::<ProtocolState>(), 716);
    }

    /// The mode tokens are stable, and 0 is bound to PROTOCOL so that a stale or
    /// zeroed read degrades to treatment, never to a false control certificate.
    #[test]
    fn test_p2_uv_mode_tokens_and_fail_direction() {
        assert_eq!(UV_BURST_MODE_PROTOCOL, 0);
        assert_eq!(UV_BURST_MODE_SUPPRESSED, 0xFFFF_FFFF);
        assert_ne!(UV_BURST_MODE_SUPPRESSED, 0);
        assert_eq!(uv_burst_mode_token(UV_BURST_MODE_PROTOCOL), "protocol_schedule");
        assert_eq!(
            uv_burst_mode_token(UV_BURST_MODE_SUPPRESSED),
            "suppressed_declared_operator_uv_off_control"
        );
        assert_eq!(uv_burst_mode_token(7), "unrecognized_uv_burst_mode");
    }

    fn test_protocol_state_size() {
        // 700-byte Stage 2 layout + 16 bytes for scan_wavelengths[8].
        assert_eq!(std::mem::size_of::<ProtocolState>(), 716);
        assert_eq!(std::mem::size_of::<SteerEntry>(), 8);
    }

    #[test]
    fn test_steering_defaults_are_neutral() {
        let _guard = UV_MODE_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // `steering_flags` now carries the P2 UV-off declaration, so this test
        // must declare the mode it is asserting rather than inherit whatever
        // another test left in the process-global atomic.
        declare_uv_burst_mode(UV_BURST_MODE_PROTOCOL);
        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(&protocol, 45000, 0.002, 1.0, 9.0, 4, true, true);
        // Default-initialized steering should not affect the kernel's behavior.
        assert_eq!(state.steering_focus_count, 0);
        for entry in state.steering_focus_residues.iter() {
            assert_eq!(entry.residue_id, -1);
            assert_eq!(entry.weight, 0.0);
        }
        // Legacy steering hooks should also be at neutral defaults
        assert_eq!(state.steering_uv_boost, 1.0);
        assert_eq!(state.steering_temp_bias, 0.0);
        assert_eq!(state.steering_focus_residue, -1);
        assert_eq!(state.steering_flags, 0);

        // Paired suppressed case: the ONLY bit that may differ is 0x8, and every
        // other steering default stays neutral.
        declare_uv_burst_mode(UV_BURST_MODE_SUPPRESSED);
        let suppressed =
            ProtocolState::from_cryo_uv(&protocol, 45000, 0.002, 1.0, 9.0, 4, true, true);
        assert_eq!(suppressed.steering_flags, UV_BURST_SUPPRESSED_FLAG);
        assert_eq!(suppressed.steering_uv_boost, 1.0);
        assert_eq!(suppressed.steering_temp_bias, 0.0);
        assert_eq!(suppressed.steering_focus_residue, -1);
        assert_eq!(suppressed.steering_focus_count, 0);
        declare_uv_burst_mode(UV_BURST_MODE_PROTOCOL);
    }

    #[test]
    fn test_from_cryo_uv_fast_35k() {
        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(&protocol, 45000, 0.002, 1.0, 9.0, 4, true, true);

        assert_eq!(state.current_step, 0);
        assert_eq!(state.total_steps, 45000);
        assert_eq!(state.start_temp, 50.0);
        assert_eq!(state.end_temp, 300.0);
        assert_eq!(state.cold_hold_end, 14000);
        assert_eq!(state.ramp_end, 20000);
        assert_eq!(state.warm_hold_end, 35000);
        assert_eq!(state.ramp_down_end, 41000);
        assert_eq!(state.uv_burst_energy, 42.0);
        assert_eq!(state.uv_burst_interval, 250);
        assert_eq!(state.n_wavelengths, 5);
        assert_eq!(
            &state.scan_wavelengths[..state.n_wavelengths as usize],
            &[280.0, 274.0, 258.0, 254.0, 211.0]
        );
        assert_eq!(state.fused_inner_steps, 4);
        // Gate 1 fields
        assert_eq!(state.gamma_base, 1.0);
        assert_eq!(state.cryo_enabled, 1);
        assert_eq!(state.equilibration_steps, 10000);
        assert_eq!(state.equilibration_gamma, 1000.0);
        assert_eq!(state.adaptive_dt_enabled, 1);
        assert_eq!(state.base_dt, 0.002);
        assert_eq!(state.coupling_phase, 0);
    }

    #[test]
    fn test_from_cryo_uv_preserves_up_to_eight_wavelengths() {
        let mut protocol = CryoUvProtocol::fast_35k();
        protocol.scan_wavelengths = vec![
            280.0, 274.0, 258.0, 254.0, 211.0, 250.0, 265.0, 290.0, 300.0,
        ];
        let state = ProtocolState::from_cryo_uv(&protocol, 45000, 0.002, 1.0, 9.0, 4, true, true);

        assert_eq!(state.n_wavelengths, MAX_SCAN_WAVELENGTHS as i32);
        assert_eq!(
            &state.scan_wavelengths,
            &[280.0, 274.0, 258.0, 254.0, 211.0, 250.0, 265.0, 290.0]
        );
    }

    #[test]
    fn test_expected_temperature_phases() {
        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(&protocol, 45000, 0.002, 1.0, 9.0, 4, true, false);

        // Phase 1: Cold hold → 50K
        assert_eq!(state.expected_temperature_at_step(0), 50.0);
        assert_eq!(state.expected_temperature_at_step(13999), 50.0);
        // Phase 2: Ramp up → 50K to 300K
        let mid_ramp = state.expected_temperature_at_step(17000);
        assert!((mid_ramp - 175.0).abs() < 0.1);
        // Phase 3: Warm hold → 300K
        assert_eq!(state.expected_temperature_at_step(27000), 300.0);
        // Phase 4: Ramp down → 300K to 50K
        let mid_down = state.expected_temperature_at_step(38000);
        assert!((mid_down - 175.0).abs() < 0.1);
        // Phase 5: Cold return → 50K
        assert_eq!(state.expected_temperature_at_step(44999), 50.0);
    }

    #[test]
    fn test_non_cooling_tail_preserves_requested_temperature_and_boundaries() {
        let protocol = CryoUvProtocol {
            start_temp: 83.0,
            end_temp: 317.0,
            cold_hold_steps: 10,
            ramp_steps: 20,
            warm_hold_steps: 30,
            ramp_down_steps: 0,
            cold_return_steps: 0,
            ..CryoUvProtocol::standard()
        };
        let state = ProtocolState::from_cryo_uv(&protocol, 1007, 0.002, 1.0, 9.0, 4, true, false);
        assert_eq!(std::mem::size_of::<ProtocolState>(), 716);
        assert_eq!(WARM_TAIL_WITHOUT_COOLING_FLAG & 0xf, 0);
        assert_eq!(state.total_steps, 1007);
        assert_eq!((state.cold_hold_end, state.ramp_end, state.warm_hold_end, state.ramp_down_end), (10, 30, 60, 60));
        assert_ne!(state.steering_flags & WARM_TAIL_WITHOUT_COOLING_FLAG, 0);
        assert_eq!(state.expected_temperature_at_step(9), 83.0);
        assert_eq!(state.expected_temperature_at_step(20), 200.0);
        for step in [30, 59, 60, 61, 1007] {
            assert_eq!(state.expected_temperature_at_step(step), 317.0);
        }
    }

    #[test]
    fn test_explicit_cold_return_without_ramp_remains_a_quench() {
        let protocol = CryoUvProtocol {
            cold_hold_steps: 10,
            ramp_steps: 20,
            warm_hold_steps: 30,
            ramp_down_steps: 0,
            cold_return_steps: 7,
            ..CryoUvProtocol::standard()
        };
        let state = ProtocolState::from_cryo_uv(&protocol, 1007, 0.002, 1.0, 9.0, 4, true, false);
        assert_eq!(state.steering_flags & WARM_TAIL_WITHOUT_COOLING_FLAG, 0);
        assert_eq!(state.expected_temperature_at_step(59), protocol.end_temp);
        assert_eq!(state.expected_temperature_at_step(60), protocol.start_temp);
        assert_eq!(state.expected_temperature_at_step(1007), protocol.start_temp);
    }

    #[test]
    fn test_expected_gamma_equilibration() {
        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(&protocol, 45000, 0.002, 1.0, 9.0, 4, true, false);

        // Step 0: full equilibration gamma (1000) with cryo scaling at 50K
        // base = 1000 * exp(0) + 1.0 * (1-1) = 1000
        // cryo: 1000 * sqrt(300/50) = 1000 * 2.449 = 2449
        let g0 = state.expected_gamma_at_step(0);
        assert!(g0 > 2000.0, "Step 0 gamma should be very high: {}", g0);

        // Step 10000: past equilibration, pure cryo scaling
        // base = 1.0, cryo: 1.0 * sqrt(300/50) = 2.449
        let g10k = state.expected_gamma_at_step(10000);
        assert!((g10k - 2.449).abs() < 0.1, "Step 10000 gamma: {}", g10k);

        // Step 27000: warm hold at 300K, no cryo scaling
        // base = 1.0, temp = 300K >= T_REF → no scaling
        let g_warm = state.expected_gamma_at_step(27000);
        assert!((g_warm - 1.0).abs() < 0.01, "Warm hold gamma: {}", g_warm);
    }

    /// GPU integration test: Director kernel temperature ramp + gamma + dt.
    #[test]
    #[ignore] // Requires GPU
    fn test_director_kernel_temperature_ramp() {
        env_logger::builder().is_test(true).try_init().ok();

        let context = CudaContext::new(0).expect("CUDA not available");
        let stream = context.default_stream();

        let (mut director, _module) =
            ProtocolDirector::new(&context, &stream).expect("Failed to create ProtocolDirector");

        let protocol = CryoUvProtocol::fast_35k();
        let total_steps = 45000u32;
        let state =
            ProtocolState::from_cryo_uv(&protocol, total_steps, 0.002, 1.0, 9.0, 4, true, true);
        director
            .initialize(&stream, &state)
            .expect("Failed to initialize");

        // Phase boundary checkpoints
        let checkpoints: Vec<(u32, u32, f32, f32, &str)> = vec![
            (1, 1, 50.0, 0.01, "step 1 cold hold"),
            (13999, 14000, 50.0, 0.01, "cold hold end"),
            (3000, 17000, 175.0, 0.5, "mid ramp up"),
            (3000, 20000, 300.0, 0.5, "ramp end"),
            (7500, 27500, 300.0, 0.01, "mid warm hold"),
            (7500, 35000, 300.0, 0.01, "warm hold end"),
            (3000, 38000, 175.0, 0.5, "mid ramp down"),
            (3000, 41000, 50.0, 0.5, "ramp down end"),
            (2000, 43000, 50.0, 0.01, "cold return"),
            (2000, 45000, 50.0, 0.01, "final step"),
        ];

        for (run_steps, expected_step, expected_temp, tol, label) in &checkpoints {
            for _ in 0..*run_steps {
                director.step(&stream).expect("director step failed");
            }

            let (step, temp, _uv, _wl) = director.read_state(&stream).expect("read failed");
            assert_eq!(step, *expected_step, "Step mismatch at '{}'", label);
            assert!(
                (temp - expected_temp).abs() < *tol,
                "Temp mismatch at '{}' (step {}): GPU={:.2}K, expected={:.2}K",
                label,
                step,
                temp,
                expected_temp
            );
        }

        // Verify gamma + dt via full state download at final step
        let final_state = director
            .download_full_state(&stream)
            .expect("download failed");
        // At step 45000 (cold return at 50K), past equilibration:
        // effective_gamma = gamma_base * sqrt(300/50) = 1.0 * 2.449
        let expected_gamma = state.expected_gamma_at_step(45000);
        assert!(
            (final_state.effective_gamma - expected_gamma).abs() < 0.1,
            "Gamma mismatch: GPU={:.3}, CPU={:.3}",
            final_state.effective_gamma,
            expected_gamma
        );
        // Cold return is a hold phase → adaptive dt returns to base.
        assert!(
            (final_state.dt - 0.002).abs() < 0.0001,
            "Adaptive dt should be 0.002 in hold phase, got {}",
            final_state.dt
        );
        // Coupling phase should have toggled 45000 times → 45000 % 2 = 0
        assert_eq!(
            final_state.coupling_phase, 0,
            "Coupling phase should be 0 after even steps"
        );

        println!("Director kernel Gate 1: ALL CHECKPOINTS PASSED (temp + gamma + dt + coupling)");
    }

    /// GPU integration test: verify UV burst scheduling.
    #[test]
    #[ignore] // Requires GPU
    fn test_director_kernel_uv_scheduling() {
        env_logger::builder().is_test(true).try_init().ok();

        let context = CudaContext::new(0).expect("CUDA not available");
        let stream = context.default_stream();

        let (mut director, _module) =
            ProtocolDirector::new(&context, &stream).expect("Failed to create ProtocolDirector");

        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(&protocol, 1000, 0.002, 1.0, 9.0, 4, true, false);
        director.initialize(&stream, &state).expect("init failed");

        let mut burst_count = 0;
        let mut seen_wavelengths = std::collections::HashSet::new();

        for i in 0..500 {
            director.step(&stream).expect("step failed");
            if i % 50 == 0 {
                let (_step, _temp, uv_active, wl) =
                    director.read_state(&stream).expect("read failed");
                if uv_active != 0 {
                    burst_count += 1;
                    seen_wavelengths.insert((wl * 10.0) as i32);
                }
            }
        }

        assert!(
            burst_count >= 1,
            "Expected at least 1 UV burst in 500 steps, got {}",
            burst_count
        );
        println!(
            "UV scheduling: {} bursts detected, {} distinct wavelengths",
            burst_count,
            seen_wavelengths.len()
        );
    }
}

#[cfg(test)]
mod asc_declaration_tests {
    use super::*;

    #[test]
    fn asc_mode_constants_are_mutually_out_of_band() {
        // The dangerous misparse is a stale or zeroed read certifying an
        // undifferentiated (whole-system) steering run AS a pocket-steered
        // one, so 0 is bound to DISENGAGED and the hazardous mode is
        // reachable by no accidental value. Same discipline as
        // UV_BURST_MODE_SUPPRESSED above.
        assert_eq!(ASC_MODE_DISENGAGED, 0);
        assert_ne!(ASC_MODE_ENGAGED_DIFFERENTIATED, ASC_MODE_DISENGAGED);
        assert_ne!(ASC_MODE_ENGAGED_UNDIFFERENTIATED, ASC_MODE_DISENGAGED);
        assert_ne!(
            ASC_MODE_ENGAGED_UNDIFFERENTIATED,
            ASC_MODE_ENGAGED_DIFFERENTIATED
        );
        assert!(ASC_MODE_ENGAGED_UNDIFFERENTIATED > 0xFFFF_0000);
        // A run must never be able to report the two engaged modes under one
        // token — they are different experiments.
        assert_ne!(
            asc_mode_token(ASC_MODE_ENGAGED_DIFFERENTIATED),
            asc_mode_token(ASC_MODE_ENGAGED_UNDIFFERENTIATED)
        );
        assert_eq!(asc_mode_token(12345), "unrecognized_asc_mode");
    }

    #[test]
    fn asc_application_not_measured_is_distinct_from_measured_zero() {
        // `None` (counters never read) and `Some(0)` (kernel ran and applied
        // nothing) are different facts. A receipt that renders both as 0
        // would let an unmeasured run read as a proven-inert one.
        assert_eq!(ASC_APPLIED_NOT_MEASURED, u64::MAX);
        let fresh = std::sync::atomic::AtomicU64::new(ASC_APPLIED_NOT_MEASURED);
        assert_eq!(
            fresh.load(std::sync::atomic::Ordering::SeqCst),
            ASC_APPLIED_NOT_MEASURED
        );
        // record_asc_application mutates process-global state shared with
        // other tests, so the sentinel semantics are asserted on the
        // accumulate rule rather than by mutating the global here.
        let accumulate = |cur: u64, add: u32| -> u64 {
            if cur == ASC_APPLIED_NOT_MEASURED {
                add as u64
            } else {
                cur.saturating_add(add as u64)
            }
        };
        assert_eq!(accumulate(ASC_APPLIED_NOT_MEASURED, 0), 0);
        assert_eq!(accumulate(ASC_APPLIED_NOT_MEASURED, 7), 7);
        assert_eq!(accumulate(7, 5), 12);
    }

    #[test]
    fn asc_implementations_are_named_separately() {
        // Two ASC kernels exist with different numerics. The vectorized one
        // has no launcher tree-wide; a receipt saying only "ASC" would let a
        // reader assume it ran.
        assert!(ASC_ACTIVE_IMPLEMENTATION.contains("prism_asc_compose_v1"));
        assert!(ASC_ACTIVE_IMPLEMENTATION.contains("adjudicator.cu"));
        assert!(ASC_UNLAUNCHED_IMPLEMENTATION.contains("asc_inject_repulsion_v4_kernel"));
        assert!(ASC_UNLAUNCHED_IMPLEMENTATION.contains("asc_steering.cu"));
        assert_ne!(ASC_ACTIVE_IMPLEMENTATION, ASC_UNLAUNCHED_IMPLEMENTATION);
    }

    /// The evidence reduction added in P7a introduces `__syncthreads()` into a
    /// kernel that previously used early returns. A `return` taken by only
    /// part of a block desynchronises the barrier, which is undefined
    /// behaviour — and the final block is partially populated whenever
    /// `n_atoms` is not a multiple of `blockDim`, so that is the normal case.
    ///
    /// This asserts the structural property against the device source: the
    /// per-atom work is predicated, not early-returned.
    #[test]
    fn asc_kernel_barrier_is_not_divergent() {
        const SRC: &str = include_str!("cuda/adjudicator.cu");
        let start = SRC
            .find("__global__ void prism_asc_apply_kernel(")
            .expect("prism_asc_apply_kernel not found");
        let body = &SRC[start..];
        let end = body.find("\n}\n").expect("kernel body end not found");
        let body = &body[..end];

        assert!(
            body.contains("__syncthreads();"),
            "the evidence reduction's barrier is gone"
        );
        assert!(
            body.contains("if (active) {"),
            "per-atom work must be predicated on `active`, not early-returned, \
             or the barrier desynchronises for the partially-populated final block"
        );
        // No bare `return` may sit BETWEEN the two barriers.
        //
        // The dangerous region is not "before the first barrier" — the first
        // barrier is in the counter-init block and precedes all per-atom work.
        // It is the span between the init barrier and the flush barrier: a
        // thread returning there has passed the first barrier and will never
        // reach the second, so the block hangs or the reduction is read before
        // every contributor has written.
        let first = body
            .find("__syncthreads();")
            .expect("init barrier missing");
        let last = body
            .rfind("__syncthreads();")
            .expect("flush barrier missing");
        assert!(
            last > first,
            "prism_asc_apply_kernel must have two distinct barriers (init, flush); \
             found one, so either the init or the flush reduction was removed"
        );
        let between = &body[first + "__syncthreads();".len()..last];
        assert!(
            !between.contains("return;"),
            "a `return` was reintroduced between the init and flush barriers of \
             prism_asc_apply_kernel. Threads taking it pass __syncthreads() once \
             and never reach the second, which is undefined behaviour — and the \
             final block is partially populated whenever n_atoms is not a \
             multiple of blockDim, so this is the normal case, not an edge case. \
             Predicate the work on `active` instead."
        );
        // The reduction must flush once per block, not once per thread.
        assert!(
            body.contains("if (threadIdx.x == 0) {"),
            "evidence flush is no longer restricted to one thread per block"
        );
        assert!(
            body.contains("atomicAdd(&s_applied_n, 1u);"),
            "per-atom accumulation must land in shared memory, not global"
        );
    }

    /// Masked-out atoms must not be counted as steered. `scale` carries the
    /// cluster mask, so a masked-out atom yields exactly 0.0f force and the
    /// magnitude guard excludes it.
    #[test]
    fn asc_evidence_excludes_masked_out_atoms() {
        const SRC: &str = include_str!("cuda/adjudicator.cu");
        assert!(
            SRC.contains("const float mask  = static_cast<float>(d_atom_in_cluster[i]);"),
            "cluster mask read changed"
        );
        assert!(
            SRC.contains("if (fmag > 0.0f) {"),
            "the evidence counter must exclude zero-magnitude (masked-out) atoms, \
             otherwise an all-zero mask would still report every atom as steered"
        );
    }
}

#[cfg(test)]
mod governor_arming_tests {
    use super::*;

    #[test]
    fn governor_device_arm_states_are_distinguishable() {
        assert_eq!(GOVERNOR_DEVICE_UNARMED, 0);
        for (a, b) in [
            (GOVERNOR_DEVICE_UNARMED, GOVERNOR_DEVICE_ARMED),
            (GOVERNOR_DEVICE_ARMED, GOVERNOR_DEVICE_ARM_FAILED),
            (GOVERNOR_DEVICE_ARM_FAILED, GOVERNOR_DEVICE_ARM_BIND_FAILED),
            (GOVERNOR_DEVICE_UNARMED, GOVERNOR_DEVICE_ARM_BIND_FAILED),
        ] {
            assert_ne!(a, b);
            assert_ne!(governor_device_arm_token(a), governor_device_arm_token(b));
        }
        // "nobody armed it" and "arming was attempted and failed" are different
        // facts; collapsing them would hide a real runtime failure behind the
        // pre-P8 default.
        assert_ne!(
            governor_device_arm_token(GOVERNOR_DEVICE_UNARMED),
            governor_device_arm_token(GOVERNOR_DEVICE_ARM_FAILED)
        );
        assert_eq!(governor_device_arm_token(7777), "unrecognized_governor_device_arm_state");
    }

    /// The P8 defect was not a wrong value — it was that four FFI wrappers
    /// existed, were documented as the mechanism, and had ZERO callers
    /// tree-wide. `gearbox::arm_governor` is the only writer of the
    /// `__constant__ d_gear_governor_policy`, so with no caller the device
    /// governor ran its entire gate composition against a static initialiser
    /// carrying `armed = 0` — while the host logged "ARMED".
    ///
    /// A value test cannot catch that class. This asserts the call sites exist.
    #[test]
    fn device_governor_wrappers_are_actually_called() {
        const RT: &str = include_str!("bin/nhs_rt_full.rs");
        // Strip comments first. An earlier version of this test matched the
        // bare symbol name and was satisfied by a DOC COMMENT mentioning
        // `gearbox::arm_governor` — it passed with the real call site renamed
        // away. Match a call, in code.
        let code: String = RT
            .lines()
            .map(|l| {
                let t = l.trim_start();
                if t.starts_with("//") || t.starts_with("///") || t.starts_with("*") {
                    ""
                } else {
                    l
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        for (f, why) in [
            (
                "gearbox::arm_governor(",
                "the ONLY writer of __constant__ d_gear_governor_policy; without it \
                 the device governor pins the gear ceiling to the unknown-topology \
                 cap and attributes it to GEAR_GATE_GOVERNOR_UNARMED forever",
            ),
            (
                "gearbox::bind_governor_measurements(",
                "without it the force guard reads UNMEASURED and gate 5 pins the \
                 ceiling to the current gear, making the guard inert in BOTH \
                 directions — no upshift authorisation and no excursion downshift",
            ),
            (
                "gearbox::snapshot_governor(",
                "the documented join for gear provenance (physical_governor.rs:41); \
                 without it the device governor that selects dt every chunk leaves \
                 no trace in any artifact",
            ),
        ] {
            assert!(
                code.contains(f),
                "{f} has no call site in nhs_rt_full.rs (comments excluded) — {why}"
            );
        }
    }

    #[test]
    fn governor_snapshot_sentinel_distinguishes_absent_from_zero() {
        // A snapshot that was never taken must not render as a governor that
        // reported gear 0 / gate 0 / armed 0 — that is exactly the unarmed
        // state, so the two would be indistinguishable in evidence.
        assert_eq!(GOVERNOR_SNAPSHOT_NOT_TAKEN, u32::MAX);
        assert_ne!(GOVERNOR_SNAPSHOT_NOT_TAKEN, GOVERNOR_DEVICE_UNARMED);
        assert_ne!(GOVERNOR_SNAPSHOT_NOT_TAKEN, 0);
    }

    /// The mirror must be seeded NaN, not zero. A zero force norm is precisely
    /// what a frozen or unstaged system reports, and the governor would read it
    /// as a quiescent measurement and authorise an upshift on data that was
    /// never produced.
    #[test]
    fn force_norm_mirror_is_seeded_not_measured() {
        const RT: &str = include_str!("bin/nhs_rt_full.rs");
        assert!(
            RT.contains("cuMemsetD32_v2(d_fnorm_mirror, f32::NAN.to_bits(), 1)"),
            "the force-norm mirror must be seeded with NaN (NOT MEASURED), never 0.0"
        );
        const ZK: &str = include_str!("cuda/zstr_kernels.cu");
        assert!(
            ZK.contains("if (d_force_norm_mirror != nullptr) {"),
            "the ZSTR sqrt pass no longer publishes to the stable mirror"
        );
        // The no-slot path must leave the mirror alone rather than zero it.
        let k = ZK
            .find("zstr_force_norm_sqrt_kernel(")
            .expect("kernel not found");
        let body = &ZK[k..k + 2600];
        assert!(
            !body.contains("*d_force_norm_mirror = 0.0f"),
            "the mirror must never be zeroed: a zero force norm is indistinguishable \
             from a real quiescent measurement"
        );
    }

    #[test]
    fn upshift_blocker_is_named() {
        // An armed governor that still cannot upshift must say WHY, or the
        // receipt reads as though arming achieved the re-baseline it did not.
        assert!(GOVERNOR_UPSHIFT_BLOCKER.contains("constraint_guard_unbound"));
        assert!(!GOVERNOR_UPSHIFT_BLOCKER.is_empty());
    }
}

/// Immutable protocol declarations admitted by the actual native constructors.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolExecutionIdentityV1 {
    pub uv_burst_mode: u32,
    pub asc_mode: u32,
    pub governor_device_arm: u32,
}
impl ProtocolExecutionIdentityV1 {
    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(matches!(self.uv_burst_mode,
            UV_BURST_MODE_PROTOCOL | UV_BURST_MODE_OVERRIDE | UV_BURST_MODE_SUPPRESSED),
            "unrecognized native protocol UV execution mode");
        anyhow::ensure!(matches!(self.asc_mode,
            ASC_MODE_DISENGAGED | ASC_MODE_ENGAGED_DIFFERENTIATED | ASC_MODE_ENGAGED_UNDIFFERENTIATED),
            "unrecognized native protocol ASC execution mode");
        anyhow::ensure!(matches!(self.governor_device_arm,
            GOVERNOR_DEVICE_UNARMED | GOVERNOR_DEVICE_ARMED | GOVERNOR_DEVICE_ARM_FAILED | GOVERNOR_DEVICE_ARM_BIND_FAILED),
            "unrecognized native protocol governor execution mode");
        Ok(())
    }
}
/// All nine production process-global atomic words in this module. Device
/// ProtocolState allocations are separately captured by the native engine.
/// Floating-point measurements remain raw integer bits, including NaN payloads,
/// signed zero, and the original not-measured sentinel.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProtocolGlobalsContinuationV1 {
    schema_version: u32,
    asc_active_implementation: String,
    pub execution_identity: ProtocolExecutionIdentityV1,
    governor_snapshot_words: [u32; 4],
    asc_applied_words: [u64; 2],
}
impl ProtocolGlobalsContinuationV1 {
    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(self.schema_version == 1
            && self.asc_active_implementation == ASC_ACTIVE_IMPLEMENTATION,
            "native protocol global continuation schema/implementation differs");
        self.execution_identity.validate()
    }
}
/// Read current declarations without substituting checkpoint values for actual
/// reconstructed native admission. Call after every relevant constructor has
/// completed and while all workers remain held.
pub fn protocol_execution_identity() -> ProtocolExecutionIdentityV1 {
    ProtocolExecutionIdentityV1 {
        uv_burst_mode: uv_burst_mode(), asc_mode: asc_mode(),
        governor_device_arm: governor_device_arm(),
    }
}
/// Validate the argv declaration before any ProtocolState device upload. ASC
/// and governor identities are validated later, after their actual native
/// constructors have re-established the admitted configuration.
pub fn validate_protocol_argv_for_continuation(saved: &ProtocolGlobalsContinuationV1) -> anyhow::Result<()> {
    saved.validate()?;
    anyhow::ensure!(uv_burst_mode() == saved.execution_identity.uv_burst_mode,
        "resumed argv UV mode differs from checkpoint");
    Ok(())
}
/// Capture exact process-global scientific output state.
///
/// # Safety
/// Every worker, output producer and native configuration constructor must be
/// quiescent through capture and capsule commit. SeqCst orders individual
/// words; the caller's all-stream rendezvous supplies tuple atomicity.
pub unsafe fn capture_protocol_globals_quiescent() -> anyhow::Result<ProtocolGlobalsContinuationV1> {
    use std::sync::atomic::Ordering::SeqCst;
    let state = ProtocolGlobalsContinuationV1 {
        schema_version: 1,
        asc_active_implementation: ASC_ACTIVE_IMPLEMENTATION.to_owned(),
        execution_identity: protocol_execution_identity(),
        governor_snapshot_words: [GOV_SNAP_ARMED_ECHO.load(SeqCst),
            GOV_SNAP_AUTHORIZED_GEAR.load(SeqCst), GOV_SNAP_BINDING_GATE.load(SeqCst),
            GOV_SNAP_FORCE_NORM_BITS.load(SeqCst)],
        asc_applied_words: [ASC_APPLIED_ATOMS.load(SeqCst), ASC_APPLIED_SUM_F_BITS.load(SeqCst)],
    };
    state.validate()?;
    Ok(state)
}
/// Restore accumulated measurements only after the fresh native constructors
/// have declared the identical execution mode. A mismatch refuses before any
/// atomic output word is modified; this function never promotes a default or
/// failed fresh mode by copying the checkpoint's mode over it.
///
/// # Safety
/// The exclusive run owner must hold all workers and producers. Perform once
/// per process after all reconstruction and before any replay or publication.
pub unsafe fn restore_protocol_globals_quiescent(saved: &ProtocolGlobalsContinuationV1) -> anyhow::Result<()> {
    use std::sync::atomic::Ordering::SeqCst;
    saved.validate()?;
    let fresh = protocol_execution_identity();
    fresh.validate()?;
    anyhow::ensure!(fresh == saved.execution_identity,
        "fresh native protocol execution modes differ from checkpoint: fresh={:?} saved={:?}",
        fresh, saved.execution_identity);
    GOV_SNAP_ARMED_ECHO.store(saved.governor_snapshot_words[0], SeqCst);
    GOV_SNAP_AUTHORIZED_GEAR.store(saved.governor_snapshot_words[1], SeqCst);
    GOV_SNAP_BINDING_GATE.store(saved.governor_snapshot_words[2], SeqCst);
    GOV_SNAP_FORCE_NORM_BITS.store(saved.governor_snapshot_words[3], SeqCst);
    ASC_APPLIED_ATOMS.store(saved.asc_applied_words[0], SeqCst);
    ASC_APPLIED_SUM_F_BITS.store(saved.asc_applied_words[1], SeqCst);
    Ok(())
}

#[cfg(test)]
mod continuation_tests {
    use super::*;
    use std::sync::atomic::Ordering::SeqCst;
    /// Isolated real processes avoid racing the existing process-global tests.
    #[test]
    fn process_global_continuation_preserves_bits_and_refuses_mode_change() {
        for case in ["bits", "identity"] {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "protocol_state::continuation_tests::protocol_continuation_child", "--ignored", "--nocapture"])
                .env("PRISM_PROTOCOL_CONTINUATION_TEST_CHILD", case).status().unwrap();
            assert!(status.success(), "protocol continuation child failed: {case}");
        }
    }
    #[test]
    #[ignore = "entry point executed in isolated real subprocess by parent test"]
    fn protocol_continuation_child() {
        let case = std::env::var("PRISM_PROTOCOL_CONTINUATION_TEST_CHILD").expect("parent case required");
        declare_uv_burst_mode(UV_BURST_MODE_SUPPRESSED);
        declare_asc_mode(ASC_MODE_ENGAGED_DIFFERENTIATED);
        declare_governor_device_arm(GOVERNOR_DEVICE_ARMED);
        GOV_SNAP_ARMED_ECHO.store(1, SeqCst);
        GOV_SNAP_AUTHORIZED_GEAR.store(2, SeqCst);
        GOV_SNAP_BINDING_GATE.store(7, SeqCst);
        GOV_SNAP_FORCE_NORM_BITS.store(0x7fc0_12ab, SeqCst);
        ASC_APPLIED_ATOMS.store(0x1234_5678_9abc_def0, SeqCst);
        ASC_APPLIED_SUM_F_BITS.store(0xfff8_0000_0123_4567, SeqCst);
        let saved = unsafe { capture_protocol_globals_quiescent() }.unwrap();
        let encoded = crate::checkpoint_codec::to_bytes(&saved).unwrap();
        let decoded: ProtocolGlobalsContinuationV1 = crate::checkpoint_codec::from_bytes(&encoded).unwrap();
        assert_eq!(saved, decoded);
        if case == "bits" {
            GOV_SNAP_FORCE_NORM_BITS.store(0, SeqCst);
            ASC_APPLIED_ATOMS.store(0, SeqCst);
            ASC_APPLIED_SUM_F_BITS.store(0, SeqCst);
            unsafe { restore_protocol_globals_quiescent(&decoded) }.unwrap();
            assert_eq!(unsafe { capture_protocol_globals_quiescent() }.unwrap(), saved);
            for bits in [0u64, 1u64 << 63, ASC_APPLIED_NOT_MEASURED] {
                ASC_APPLIED_SUM_F_BITS.store(bits, SeqCst);
                let exact = unsafe { capture_protocol_globals_quiescent() }.unwrap();
                ASC_APPLIED_SUM_F_BITS.store(123, SeqCst);
                unsafe { restore_protocol_globals_quiescent(&exact) }.unwrap();
                assert_eq!(ASC_APPLIED_SUM_F_BITS.load(SeqCst), bits);
            }
        } else {
            declare_uv_burst_mode(UV_BURST_MODE_PROTOCOL);
            assert!(validate_protocol_argv_for_continuation(&saved).is_err());
            declare_uv_burst_mode(UV_BURST_MODE_SUPPRESSED);
            validate_protocol_argv_for_continuation(&saved).unwrap();
            declare_asc_mode(ASC_MODE_DISENGAGED);
            ASC_APPLIED_ATOMS.store(17, SeqCst);
            let before = unsafe { capture_protocol_globals_quiescent() }.unwrap();
            assert!(unsafe { restore_protocol_globals_quiescent(&saved) }.is_err());
            assert_eq!(unsafe { capture_protocol_globals_quiescent() }.unwrap(), before);
            declare_asc_mode(ASC_MODE_ENGAGED_DIFFERENTIATED);
            declare_governor_device_arm(GOVERNOR_DEVICE_ARM_FAILED);
            let before = unsafe { capture_protocol_globals_quiescent() }.unwrap();
            assert!(unsafe { restore_protocol_globals_quiescent(&saved) }.is_err());
            assert_eq!(unsafe { capture_protocol_globals_quiescent() }.unwrap(), before);
        }
    }
}
