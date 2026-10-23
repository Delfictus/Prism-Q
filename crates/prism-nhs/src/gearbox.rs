//! G26 Chronometric Gearbox — Wave B.1 (Foundation)
//!
//! Operator-mandated three-commit sequence (B.1 → B.2 → B.3) ratified
//! 2026-05-02.  This module lands the **B.1 foundation only**:
//!
//!   * `__constant__ float d_gearbox_table[16]` — 4 gears × 4 floats,
//!     dt slot per gear; Gear 3 = NaN sentinel (double-redundant abort).
//!   * `ChronometricStateTensor` — 16-byte F2-pool persistent state,
//!     `{counter, last_burst_frame, current_gear, _pad}`.
//!   * `prism_gearbox_pointer_swap_kernel` — Stateful Finite Automaton
//!     that reads `adj->adjudication_code` + cruise counter and writes
//!     the active gear's dt into `*(adj->d_dt)` (the address the
//!     T12 Pre-Flight commit wired to `&d_protocol->dt`).
//!
//! B.2 will add VelocityRescale + Berendsen kernels and the predicate
//! bridge that wires this kernel into the G26 SWITCH body sub-graphs.
//! B.3 lands the PTX-trap kernel for Gear 3 + the 1k-step Gear 2→0→2
//! transition test.
//!
//! # Hierarchy ruling (operator 2026-05-02)
//!
//! When the V2 captured pipeline is built, the legacy `--adaptive-dt`
//! host-side write to `engine.dt` is bypassed; the gearbox is the sole
//! master of the chronometric plane.  See `fused_engine.rs` for the
//! `gearbox_active` gate.

#![cfg(feature = "gpu")]

use std::ffi::c_void;

// ============================================================================
// ChronometricStateTensor — 16-byte persistent device-resident state
// ============================================================================
//
// Layout pinned by both the Rust `#[repr(C, align(16))]` and the C-side
// `__align__(16)` + `static_assert(sizeof == 16)` in `gearbox.cuh`.

/// Persistent gearbox state — lives in the F2 pool, mutated only by the
/// PointerSwap kernel (Zero-CPU mandate).  Allocated once per pipeline,
/// pointer-stable for the campaign.
///
/// **M1.2.17**: extended to 32 bytes to hold `v_prev: f64` for the
/// Hamiltonian Stability Fuse.  Layout pinned by const_assert and
/// the C-side static_assert in `gearbox.cuh`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct ChronometricStateTensor {
    /// Frames since the most recent Burst (`adjudication_code == 1`).  @ 0
    /// Drives the cruise hysteresis: counter < 500 → Gear 1 (2.0 fs
    /// safety); counter ≥ 500 → Gear 2 (4.0 fs campaign).
    pub counter: u32,
    /// Global frame index of the most-recent Burst.                    @ 4
    pub last_burst_frame: u32,
    /// Most-recent gear chosen by the kernel.                           @ 8
    pub current_gear: u32,
    /// Gear active BEFORE the most-recent PointerSwap launch.           @ 12
    /// Used by the symplectic-ratio kernel for λ = dt_new / dt_old.
    pub previous_gear: u32,
    /// **M1.2.17 — V_{t-1}** for the rolling-window stability fuse.    @ 16
    /// Initialised to 0.0 (first-frame sentinel; the SFA skips the
    /// drift check when v_prev == 0).  SFA writes `adj.d_potential_energy`
    /// here at the end of each launch so the next launch sees the
    /// correct V_{t-1}.
    pub v_prev: f64,
    /// Padding to 32-byte boundary for L2 cache-line alignment.        @ 24
    pub _pad_v_prev: u64,
}

impl ChronometricStateTensor {
    /// Default cruise state: counter = 0, no burst yet, current_gear = 1
    /// (the safety gear — 2.0 fs).  v_prev = 0.0 sentinel triggers
    /// first-frame-skip in the SFA stability fuse.
    pub const fn initial() -> Self {
        Self {
            counter: 0,
            last_burst_frame: 0,
            current_gear: 1,
            previous_gear: 1,
            v_prev: 0.0,
            _pad_v_prev: 0,
        }
    }
}

const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<ChronometricStateTensor>() == 32);
    assert!(align_of::<ChronometricStateTensor>() == 16);
};

// ============================================================================
// Gearbox dt table — 4 gears × 4 floats = 64 bytes (one L1 const-cache line)
// ============================================================================

/// Number of gears in the gearbox (3 active + 1 abort sentinel).
pub const PRISM_GEARBOX_N_GEARS: u32 = 4;
/// Floats per gear (slot 0 = dt; 1..3 reserved for B.2 VelocityRescale +
/// Berendsen tau).
pub const PRISM_GEARBOX_FLOATS_PER_GEAR: u32 = 4;
/// Total table length, in floats.  16 floats × 4 bytes = 64 bytes.
pub const PRISM_GEARBOX_TABLE_LEN: usize = 16;

/// Cruise hysteresis threshold.  Frames of consecutive
/// `adjudication_code == 0` (Equilibrium) before the gearbox upshifts
/// from Gear 1 (safety) to Gear 2 (campaign velocity).
pub const PRISM_GEARBOX_CRUISE_THRESHOLD: u32 = 500;

/// Default gear-table values in picoseconds.  Slot 0 of each row is dt;
/// the remaining three slots are zero in B.1 and will hold VelocityRescale
/// ratios + Berendsen tau in B.2.  Gear 3 is NaN-everywhere — the
/// double-redundant abort sentinel.
pub fn default_gearbox_table() -> [f32; PRISM_GEARBOX_TABLE_LEN] {
    let nan = f32::NAN;
    [
        // Gear 0 — 0.5 fs (high-resolution capture).
        0.0005, 0.0, 0.0, 0.0,
        // Gear 1 — 2.0 fs (default monitoring / cruise safety).
        0.0020, 0.0, 0.0, 0.0, // Gear 2 — 4.0 fs (HMR-stabilised campaign sprint).
        0.0040, 0.0, 0.0, 0.0, // Gear 3 — abort sentinel.
        nan, nan, nan, nan,
    ]
}

// ============================================================================
// GEAR-PROVENANCE-002 — device → host gear measurement
// ============================================================================
//
// Before this surface existed `cruise.current_gear` never left the GPU, so
// every host-side `gear_id` in the tree was a build-time literal. Under the
// documented encoding (0 = 0.5 fs, 1 = 2.0 fs, 2 = 4.0 fs HMR, 3 = abort)
// there is no in-band "unknown": 0 is a legitimate gear, so a placeholder 0
// is indistinguishable offline from a genuine high-resolution frame and
// silently rescales every downstream time axis by up to 8x.
//
// This block provides the read path plus a typed result in which "not
// measured" is representable (`Option<u8>` / an explicit failure reason).

/// Sentinel meaning "no gear measurement exists". Mirrors
/// `PRISM_GEARBOX_GEAR_UNMEASURED` in `gearbox.cuh` and
/// [`crate::zstr::ZSTR_GEAR_ID_UNMEASURED`].
pub const GEAR_UNMEASURED: u32 = 0xFFFF_FFFF;

/// Schema version of [`GearboxHostSnapshot`]. Mirrors
/// `PRISM_GEARBOX_SNAPSHOT_SCHEMA_VERSION`. A mismatch is reported as
/// [`GearSnapshotError::SchemaMismatch`] rather than being reinterpreted.
pub const GEARBOX_SNAPSHOT_SCHEMA_VERSION: u32 = 1;

/// `GearboxHostSnapshot::status` codes; mirror of `PRISM_GEARBOX_SNAPSHOT_*`.
pub const GEARBOX_SNAPSHOT_OK: u32 = 0;
pub const GEARBOX_SNAPSHOT_NULL_ARG: u32 = 1;
pub const GEARBOX_SNAPSHOT_CAPTURE_ACTIVE: u32 = 2;
pub const GEARBOX_SNAPSHOT_EVENT_FAILED: u32 = 3;
pub const GEARBOX_SNAPSHOT_TIMEOUT: u32 = 4;
pub const GEARBOX_SNAPSHOT_COPY_FAILED: u32 = 5;

/// Return code from `prism_gearbox_snapshot_state` when the bounded drain
/// wait expired. Mirrors `PRISM_GEARBOX_RC_TIMEOUT`.
pub const GEARBOX_RC_TIMEOUT: i32 = -1;

/// **Source literal default** for the bounded gear-readback wait, in
/// milliseconds. Mirrors `PRISM_GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS`.
///
/// This is the documented default only. The effective budget is resolved by
/// [`resolve_snapshot_timeout`], and both the resolved value and its origin
/// are recorded on every emitted control record.
pub const GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS: u32 = 2000;

/// Policy key that overrides the bounded gear-readback wait.
pub const GEARBOX_SNAPSHOT_TIMEOUT_ENV: &str = "PRISM_GEARBOX_SNAPSHOT_TIMEOUT_MS";

/// A resolved bound plus the provenance of the value that was used.
///
/// Carrying the origin alongside the number is what keeps the bound
/// auditable: a record that says `2000 ms` without saying whether that came
/// from the source literal or from operator policy is not reproducible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedTimeout {
    /// Effective bound in milliseconds. Never zero.
    pub millis: u32,
    /// Where `millis` came from.
    pub origin: TimeoutOrigin,
}

/// Provenance of a [`ResolvedTimeout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeoutOrigin {
    /// `GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS`, the documented source literal.
    SourceLiteralDefault,
    /// Parsed from `PRISM_GEARBOX_SNAPSHOT_TIMEOUT_MS`.
    PolicyEnv,
    /// The policy key was present but unusable; the source literal was used
    /// and the failure was logged.
    PolicyEnvRejected,
}

impl TimeoutOrigin {
    /// Stable machine-readable label for evidence records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SourceLiteralDefault => "source_literal_default",
            Self::PolicyEnv => "policy_env_prism_gearbox_snapshot_timeout_ms",
            Self::PolicyEnvRejected => "policy_env_rejected_fell_back_to_source_literal",
        }
    }
}

/// Resolve the bounded gear-readback wait.
///
/// Order: `PRISM_GEARBOX_SNAPSHOT_TIMEOUT_MS` if it parses to a non-zero
/// `u32`, else the documented source literal. A zero or unparsable override
/// is REJECTED rather than honoured — an unbounded (0 = "wait forever")
/// reading of this knob would reintroduce exactly the deadlock class the
/// bound exists to prevent.
pub fn resolve_snapshot_timeout() -> ResolvedTimeout {
    match std::env::var(GEARBOX_SNAPSHOT_TIMEOUT_ENV) {
        Ok(raw) => match raw.trim().parse::<u32>() {
            Ok(0) => {
                log::warn!(
                    "[G26 GEAR-PROVENANCE] {}=0 is not a legal bound (0 would mean \
                     'wait forever'); using the documented default {} ms.",
                    GEARBOX_SNAPSHOT_TIMEOUT_ENV,
                    GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS
                );
                ResolvedTimeout {
                    millis: GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS,
                    origin: TimeoutOrigin::PolicyEnvRejected,
                }
            }
            Ok(v) => ResolvedTimeout {
                millis: v,
                origin: TimeoutOrigin::PolicyEnv,
            },
            Err(e) => {
                log::warn!(
                    "[G26 GEAR-PROVENANCE] {}='{}' is unparsable ({}); using the \
                     documented default {} ms.",
                    GEARBOX_SNAPSHOT_TIMEOUT_ENV,
                    raw,
                    e,
                    GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS
                );
                ResolvedTimeout {
                    millis: GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS,
                    origin: TimeoutOrigin::PolicyEnvRejected,
                }
            }
        },
        Err(_) => ResolvedTimeout {
            millis: GEARBOX_SNAPSHOT_DEFAULT_TIMEOUT_MS,
            origin: TimeoutOrigin::SourceLiteralDefault,
        },
    }
}

/// Raw 48-byte host snapshot produced by `prism_gearbox_snapshot_state`.
/// Layout pinned by the const_assert below and the C-side static_assert in
/// `gearbox.cuh`.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GearboxHostSnapshot {
    /// `GEARBOX_SNAPSHOT_SCHEMA_VERSION`.                              @ 0
    pub schema_version: u32,
    /// `GEARBOX_SNAPSHOT_*` status code.                                @ 4
    pub status: u32,
    /// `cruise.counter` — frames since the last Burst.                  @ 8
    pub counter: u32,
    /// `cruise.last_burst_frame`.                                       @ 12
    pub last_burst_frame: u32,
    /// `cruise.current_gear`, or `GEAR_UNMEASURED`.                     @ 16
    pub current_gear: u32,
    /// `cruise.previous_gear`, or `GEAR_UNMEASURED`.                    @ 20
    pub previous_gear: u32,
    /// `*(adj->d_dt)` in picoseconds; meaningful only if `dt_valid == 1`. @ 24
    pub dt_ps_applied: f32,
    /// 1 when `dt_ps_applied` is a measurement, 0 when it is absent.    @ 28
    pub dt_valid: u32,
    /// `cruise.v_prev` — V_{t-1} of the Hamiltonian stability fuse.     @ 32
    pub v_prev: f64,
    /// MEASURED `CLOCK_MONOTONIC` microseconds spent inside the bounded  @ 40
    /// drain wait. This is a clock reading, not a count of nominal sleep
    /// quanta, so it stays truthful when `nanosleep` oversleeps under load.
    pub waited_us: u64,
}

const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<GearboxHostSnapshot>() == 48);
    assert!(align_of::<GearboxHostSnapshot>() == 16);
};

impl GearboxHostSnapshot {
    /// The explicit no-measurement value. Both gears carry the sentinel; no
    /// field is left at a plausible zero.
    pub const fn unmeasured(status: u32) -> Self {
        Self {
            schema_version: GEARBOX_SNAPSHOT_SCHEMA_VERSION,
            status,
            counter: 0,
            last_burst_frame: 0,
            current_gear: GEAR_UNMEASURED,
            previous_gear: GEAR_UNMEASURED,
            dt_ps_applied: 0.0,
            dt_valid: 0,
            v_prev: 0.0,
            waited_us: 0,
        }
    }
}

/// Named terminal conditions for a gear readback. Every variant is a reason
/// the host has NO measurement; none of them is ever collapsed into a gear
/// value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GearSnapshotError {
    /// No `ChronometricStateTensor` was allocated for this pipeline.
    NullCruiseTensor,
    /// The stream is mid-capture; a host-resident copy would corrupt the
    /// graph, so no read was attempted.
    CaptureActive { rc: i32 },
    /// The CUDA event used for the bounded wait could not be created or
    /// recorded.
    EventFailed { rc: i32 },
    /// The bounded drain wait expired. This is the named terminal error for
    /// the wait; it is NOT retried in a loop by the caller.
    Timeout { budget_ms: u32, waited_us: u64 },
    /// The device-to-host copy failed.
    CopyFailed { rc: i32 },
    /// The device reported a snapshot schema this build cannot read. Failing
    /// loudly here is deliberate: silently accepting an unknown layout would
    /// publish misparsed gears as measurements.
    SchemaMismatch { expected: u32, observed: u32 },
    /// A CUDA error that maps to no other variant.
    Cuda { rc: i32 },
}

impl std::fmt::Display for GearSnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NullCruiseTensor => {
                write!(f, "G26_GEAR_READBACK_NO_CRUISE_TENSOR")
            }
            Self::CaptureActive { rc } => {
                write!(f, "G26_GEAR_READBACK_CAPTURE_ACTIVE rc={rc}")
            }
            Self::EventFailed { rc } => {
                write!(f, "G26_GEAR_READBACK_EVENT_FAILED rc={rc}")
            }
            Self::Timeout {
                budget_ms,
                waited_us,
            } => write!(
                f,
                "G26_GEAR_READBACK_BOUNDED_WAIT_EXPIRED budget_ms={budget_ms} waited_us={waited_us}"
            ),
            Self::CopyFailed { rc } => {
                write!(f, "G26_GEAR_READBACK_COPY_FAILED rc={rc}")
            }
            Self::SchemaMismatch { expected, observed } => write!(
                f,
                "G26_GEAR_READBACK_SCHEMA_MISMATCH expected={expected} observed={observed}"
            ),
            Self::Cuda { rc } => write!(f, "G26_GEAR_READBACK_CUDA rc={rc}"),
        }
    }
}

impl std::error::Error for GearSnapshotError {}

impl GearSnapshotError {
    /// Stable machine-readable label for evidence records.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::NullCruiseTensor => "no_cruise_tensor",
            Self::CaptureActive { .. } => "capture_active",
            Self::EventFailed { .. } => "event_failed",
            Self::Timeout { .. } => "bounded_wait_expired",
            Self::CopyFailed { .. } => "copy_failed",
            Self::SchemaMismatch { .. } => "schema_mismatch",
            Self::Cuda { .. } => "cuda_error",
        }
    }
}

/// A gearbox measurement in which absence is representable.
///
/// `current_gear`/`previous_gear` are `None` when the device did not supply
/// a value; they are never defaulted to a legal gear.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasuredGear {
    /// Gear the device state tensor held after the decision.
    pub current_gear: Option<u8>,
    /// Gear the device state tensor held before the decision.
    pub previous_gear: Option<u8>,
    /// Timestep in picoseconds actually bound to `ProtocolState.dt`.
    pub dt_ps_applied: Option<f64>,
    /// Frames since the last Burst, as counted on the device.
    pub counter: u32,
    /// Global frame index of the most recent Burst, as stamped on device.
    pub last_burst_frame: u32,
    /// `V_{t-1}` held by the Hamiltonian stability fuse.
    pub v_prev: f64,
    /// Microseconds spent inside the bounded drain wait.
    pub waited_us: u64,
    /// Bound that governed the wait, with its provenance.
    pub timeout: ResolvedTimeout,
}

/// Convert a raw device gear word into an `Option<u8>`.
///
/// Anything outside the documented 0..=3 encoding — including the
/// `GEAR_UNMEASURED` sentinel — becomes `None`. An out-of-range gear is a
/// broken measurement, not a new gear, and must not be published as one.
fn decode_gear(raw: u32) -> Option<u8> {
    if raw <= 3 {
        Some(raw as u8)
    } else {
        None
    }
}

/// Read the live gear + applied timestep off the device.
///
/// `cruise_dev` is the `ChronometricStateTensor` address (see
/// `CapturedAdjudicationPipeline::cruise_state_dev_ptr`); `d_dt_dev` is the
/// `ProtocolState.dt` address wired into `adj->d_dt`, or 0 when unwired.
/// `stream` is the stream whose work must drain before the read is coherent.
///
/// The wait is bounded by [`resolve_snapshot_timeout`] and expiry surfaces as
/// [`GearSnapshotError::Timeout`]. This function must NOT be called inside a
/// capture window; it refuses with [`GearSnapshotError::CaptureActive`] if it
/// detects one.
///
/// # Safety
/// `cruise_dev` and `d_dt_dev` must be valid device addresses in the current
/// context (or 0), and `stream` a valid stream or null.
pub unsafe fn snapshot_measured_gear(
    cruise_dev: u64,
    d_dt_dev: u64,
    stream: *mut c_void,
    timeout: ResolvedTimeout,
) -> Result<MeasuredGear, GearSnapshotError> {
    if cruise_dev == 0 {
        return Err(GearSnapshotError::NullCruiseTensor);
    }
    let mut raw = GearboxHostSnapshot::unmeasured(GEARBOX_SNAPSHOT_NULL_ARG);
    let rc = ffi::prism_gearbox_snapshot_state(
        cruise_dev as *const ChronometricStateTensor,
        if d_dt_dev == 0 {
            std::ptr::null()
        } else {
            d_dt_dev as *const f32
        },
        stream,
        timeout.millis,
        &mut raw as *mut GearboxHostSnapshot,
    );

    if raw.schema_version != GEARBOX_SNAPSHOT_SCHEMA_VERSION {
        return Err(GearSnapshotError::SchemaMismatch {
            expected: GEARBOX_SNAPSHOT_SCHEMA_VERSION,
            observed: raw.schema_version,
        });
    }
    if rc == GEARBOX_RC_TIMEOUT || raw.status == GEARBOX_SNAPSHOT_TIMEOUT {
        return Err(GearSnapshotError::Timeout {
            budget_ms: timeout.millis,
            waited_us: raw.waited_us,
        });
    }
    match raw.status {
        GEARBOX_SNAPSHOT_OK if rc == 0 => {}
        GEARBOX_SNAPSHOT_NULL_ARG => return Err(GearSnapshotError::NullCruiseTensor),
        GEARBOX_SNAPSHOT_CAPTURE_ACTIVE => return Err(GearSnapshotError::CaptureActive { rc }),
        GEARBOX_SNAPSHOT_EVENT_FAILED => return Err(GearSnapshotError::EventFailed { rc }),
        GEARBOX_SNAPSHOT_COPY_FAILED => return Err(GearSnapshotError::CopyFailed { rc }),
        _ => return Err(GearSnapshotError::Cuda { rc }),
    }

    Ok(MeasuredGear {
        current_gear: decode_gear(raw.current_gear),
        previous_gear: decode_gear(raw.previous_gear),
        dt_ps_applied: if raw.dt_valid == 1 && raw.dt_ps_applied.is_finite() {
            Some(raw.dt_ps_applied as f64)
        } else {
            None
        },
        counter: raw.counter,
        last_burst_frame: raw.last_burst_frame,
        v_prev: raw.v_prev,
        waited_us: raw.waited_us,
        timeout,
    })
}

// ============================================================================
// FFI — extern "C" forward declarations from gearbox.cu
// ============================================================================

pub mod ffi {
    use super::{ChronometricStateTensor, GearboxHostSnapshot};
    use crate::interferometric_adjudicator::InterferometricAdjudicatorFfi;
    use std::ffi::c_void;

    extern "C" {
        /// GEAR-PROVENANCE-002 — copy the live gearbox state to the host.
        ///
        /// Refuses inside a capture window, performs a BOUNDED drain wait
        /// (nanosleep-throttled `cudaEventQuery`, never a spin), then a
        /// synchronous D2H of the 32-byte cruise tensor plus the 4-byte
        /// applied dt. `out` is fully written on every path, with
        /// `GEAR_UNMEASURED` in both gear slots when no measurement exists.
        ///
        /// Returns `GEARBOX_RC_TIMEOUT` on bounded-wait expiry.
        pub fn prism_gearbox_snapshot_state(
            cruise: *const ChronometricStateTensor,
            d_dt: *const f32,
            stream: *mut c_void,
            timeout_ms: u32,
            out: *mut GearboxHostSnapshot,
        ) -> i32;

        /// Host-side cudaMemcpyToSymbolAsync of the 16-float gear table
        /// into `__constant__ d_gearbox_table`.  Stream-ordered.  Call
        /// ONCE per pipeline build, BEFORE the captured-graph window
        /// opens.
        pub fn prism_gearbox_init_table_async(
            host_table: *const f32, // 16 floats
            stream: *mut c_void,
        ) -> i32;

        /// Records the PointerSwap kernel onto `stream` as one graph
        /// node.  Single-thread <<<1,1>>>; reads adj->adjudication_code
        /// + adj->d_dt, mutates the cruise tensor, writes the active
        /// gear's dt through *(adj->d_dt).
        pub fn prism_gearbox_launch_pointer_swap(
            adj: *const InterferometricAdjudicatorFfi,
            cruise: *mut ChronometricStateTensor,
            current_frame: u32,
            stream: *mut c_void,
        ) -> i32;

        // ── Wave B.2 — Velocity Rescale + Berendsen + Predicate Bridge ──

        /// T12.2 — Symplectic Velocity Rescale.  Records one bulk kernel
        /// node onto `stream`: each thread multiplies one f32 of the
        /// AoS velocity buffer by `ratio`.  Block size 256 ⇒ each warp
        /// reads 32 contiguous f32 = 128 bytes ⇒ ptxas emits LDG.E.128 +
        /// STG.E.128 (operator-mandated vectorised path).
        pub fn prism_gearbox_launch_velocity_rescale(
            d_velocities: *mut f32,
            n_floats: u32,
            ratio: f32,
            stream: *mut c_void,
        ) -> i32;

        /// T12.3 — Berendsen weak-coupling guard.  Reads
        /// `*d_current_temp` (= d_protocol->current_temperature) and
        /// `*d_dt` (= d_protocol->dt for the current gear), computes
        /// λ = sqrt(max(1 + (dt/τ_T)·(T₀/T − 1), ε)), applies
        /// v_i ← v_i · λ across the whole velocity buffer.  Single
        /// captured node; no scratch buffer.
        pub fn prism_gearbox_launch_berendsen_guard(
            d_velocities: *mut f32,
            n_floats: u32,
            d_current_temp: *const f32,
            d_dt: *const f32,
            target_temp_K: f32,
            tau_ps: f32,
            stream: *mut c_void,
        ) -> i32;

        /// T12.4 — 4-way Predicate Bridge.  Single-thread kernel reads
        /// `cruise->current_gear` and forwards the value to the G26
        /// SWITCH conditional handle via cudaGraphSetConditional.
        ///
        /// B.3.2 — also consults `adj->gear_override` (u32 at offset 100;
        /// Emergency Rectification 2026-05-02 reverted from M1.2.18.5
        /// attempt to relocate to 136).  When non-0xFF, the override
        /// short-circuits the SFA's calculated gear.  Pass null `adj`
        /// to skip override consult.
        pub fn prism_gearbox_launch_predicate_bridge(
            handle_v: u64,
            adj: *const InterferometricAdjudicatorFfi,
            cruise: *const ChronometricStateTensor,
            stream: *mut c_void,
        ) -> i32;

        /// CHUNK13_DIAG — TRACED variant. Equivalent to
        /// `prism_gearbox_launch_predicate_bridge` plus an additional
        /// `branch_trace_dev` device pointer (pinned-mapped
        /// `PrismBranchTrace`). Pass 0 to disable trace writes.
        pub fn prism_gearbox_launch_predicate_bridge_traced(
            handle_v: u64,
            adj: *const InterferometricAdjudicatorFfi,
            cruise: *const ChronometricStateTensor,
            stream: *mut c_void,
            branch_trace_dev: u64,
        ) -> i32;

        /// B.3.2 — SFA-only kernel (Logic/Action Bifurcation).
        /// Runs the same KL-driven state machine as
        /// prism_gearbox_pointer_swap_kernel but does NOT write
        /// *(adj->d_dt) — the dt-write side-effect is owned by the
        /// SWITCH body's apply-fixed-dt kernel.  Decoupling lets the
        /// Blackwell scheduler hide the SFA's logic-only cycles
        /// behind the integrator's global-memory writes.
        pub fn prism_gearbox_launch_sfa(
            adj: *const InterferometricAdjudicatorFfi,
            cruise: *mut ChronometricStateTensor,
            current_frame: u32,
            stream: *mut c_void,
        ) -> i32;

        // ── B.3-narrow — SWITCH body kernels + populator ──

        /// B.3 — No-op on the canonical 0.5/2.0/4.0 fs gearbox; the
        /// 3×3 ratio matrix is statically initialised in __constant__
        /// memory.  Exists for ABI symmetry with future runtime
        /// re-tuning.  Returns cudaSuccess.
        pub fn prism_gearbox_init_rescale_ratios_async(stream: *mut c_void) -> i32;

        /// B.3 — Symplectic velocity rescale (ratio-matrix variant).
        /// Reads cruise->previous_gear, indexes the constant ratio
        /// matrix at [prev*3 + target_gear], multiplies every f32 in
        /// d_velocities by that ratio.  Single LDC + LDG + MUL + STG
        /// per thread; block 256 ⇒ LDG.E.128 / STG.E.128 warp-coalesced.
        pub fn prism_gearbox_launch_rescale(
            d_velocities: *mut f32,
            n_floats: u32,
            cruise: *const ChronometricStateTensor,
            target_gear: u32,
            stream: *mut c_void,
        ) -> i32;

        /// B.3 — Apply-fixed-dt kernel.  Single-thread <<<1,1>>> writes
        /// d_gearbox_table[target_gear*4] into *(adj->d_dt).  Used
        /// inside SWITCH body sub-graphs (Gears 0/1/2).
        pub fn prism_gearbox_launch_apply_fixed_dt(
            adj: *const InterferometricAdjudicatorFfi,
            target_gear: u32,
            stream: *mut c_void,
        ) -> i32;

        /// B.3 — Hardware trap kernel.  Single-thread asm("trap;").
        /// Used inside SWITCH body 3 (Gear 3 / Abort).
        pub fn prism_gearbox_launch_trap(stream: *mut c_void) -> i32;

        /// B.3 — Populate the four phGraph_out body sub-graphs returned
        /// by prism_wire_g26_gearbox_ffi with the gear-specific kernel
        /// node sequences.  Returns cudaErrorInvalidValue if any
        /// body_subgraphs[i] is null — the operator-mandated "Smoking
        /// Gun" signal that the caller must pivot to the kernel-
        /// conditional fallback pattern.
        ///
        /// `d_current_temp` and `d_dt` are nullable — pass null to skip
        /// the Berendsen guard in body 0 (e.g., for tests without a
        /// ProtocolState fixture).
        // Reconciled with captured_pipeline.rs:865 to silence
        // `clashing_extern_declarations`. The CUgraph alias in cudarc resolves
        // to `*mut CUgraph_st`, so `*mut CUgraph` is `*mut *mut CUgraph_st`,
        // ABI-identical to `*mut *mut c_void` but type-distinguishable to the
        // Rust compiler. Using cudarc's CUgraph here matches the canonical
        // captured-pipeline declaration verbatim. Same applies to the cruise
        // pointer: kept `*const c_void` to match captured_pipeline.rs (the
        // C++ producer takes `void*` and the Rust caller casts on the way in).
        pub fn prism_gearbox_populate_switch_bodies_ffi(
            body_subgraphs: *mut cudarc::driver::sys::CUgraph, // [4]
            adj: *const InterferometricAdjudicatorFfi,
            d_velocities: *mut f32,
            n_floats: u32,
            cruise: *const std::ffi::c_void, // ChronometricStateTensor*
            d_current_temp: *const f32,
            d_dt: *const f32,
            target_temp_K: f32,
            tau_ps: f32,
        ) -> i32;

        // ── WP C — Physical Stability Governor ──

        /// Fill `out` with the C-side documented defaults (every
        /// `PRISM_GEAR_DEFAULT_*` literal in `gearbox.cuh`). Pure host
        /// computation; no CUDA context required. This is the single source
        /// of default truth — the Rust resolver starts from it rather than
        /// re-declaring the literals, so the two sides cannot drift.
        pub fn prism_gearbox_governor_default_policy(out: *mut super::GearGovernorPolicy) -> i32;

        /// Evaluate the governor on the host through the SAME
        /// `prism_gearbox_governor_decide` the SFA kernel calls. `tsh` is
        /// nullable. Pure computation; no CUDA context required.
        pub fn prism_gearbox_governor_decide_host(
            policy: *const super::GearGovernorPolicy,
            observation: *const super::GearGuardObservation,
            tsh: *const super::TshSalienceRequest,
            out: *mut super::GearDecision,
        ) -> i32;

        /// Upload the resolved policy into `__constant__` memory and reset the
        /// device governor state. Stream-ordered; MUST be called before the
        /// capture window opens. Refuses a capturing stream and refuses a
        /// policy whose `schema_version` is unrecognised.
        pub fn prism_gearbox_governor_arm_async(
            policy: *const super::GearGovernorPolicy,
            stream: *mut c_void,
        ) -> i32;

        /// Bind the live physical-measurement device pointers. A null pointer
        /// leaves that guard UNMEASURED — recorded as such, never substituted.
        pub fn prism_gearbox_governor_bind_measurements_async(
            bindings: *const super::GearGovernorBindings,
            stream: *mut c_void,
        ) -> i32;

        /// Publish a TSH salience request into the device governor state.
        /// Rejects `requested_gear == GEAR_ABORT`: the salience layer may ask
        /// for finer resolution, never for an abort.
        pub fn prism_gearbox_governor_publish_salience_async(
            request: *const super::TshSalienceRequest,
            stream: *mut c_void,
        ) -> i32;

        /// Read the device governor state (last decision + running counters)
        /// back to the host. Bounded wait; refuses a capturing stream; always
        /// writes `out`, with `GEAR_UNSET` sentinels on every failure path.
        pub fn prism_gearbox_governor_snapshot(
            stream: *mut c_void,
            out: *mut super::GearGovernorState,
        ) -> i32;
    }
}

// ============================================================================
// WP C — PHYSICAL STABILITY GOVERNOR (host mirror + policy resolution)
// ============================================================================
//
// The device half lives in `cuda/gearbox.cuh` + `cuda/gearbox.cu`; the
// architecture rationale is documented there in full. This module is the host
// half: the layout mirrors, the versioned policy resolver, and the typed
// arming / binding / readback wrappers.
//
// Division of labour, restated because it is the whole point:
//
//     scientific-resolution REQUEST  (KL automaton today, TSH layer later)
//               |
//     PHYSICAL STABILITY GOVERNOR    (what timestep is SAFE now)
//               |
//     authorized gear = the MORE CONSERVATIVE of the two
//
// Nothing here decides anything. `resolve_governor_policy` produces the
// numbers and their provenance; `prism_gearbox_governor_decide` (C, shared by
// host and device) is the only implementation of the decision itself.

/// Governor wire-format version. Mirrors
/// `PRISM_GEAR_GOVERNOR_SCHEMA_VERSION`. A reader that sees any other value
/// MUST fail loudly rather than misparse — the const_asserts below and
/// `GovernorPolicyError::SchemaMismatch` are that failure.
pub const GEAR_GOVERNOR_SCHEMA_VERSION: u32 = 1;

/// Gear ids. Mirror `PRISM_GEAR_*` in `gearbox.cuh`.
pub const GEAR_FINE: u32 = 0;
/// 2.0 fs — the constrained-hydrogen safety gear.
pub const GEAR_SAFE: u32 = 1;
/// 4.0 fs — reachable ONLY with verified hydrogen-mass repartitioning.
pub const GEAR_CAMPAIGN: u32 = 2;
/// Trap sentinel.
pub const GEAR_ABORT: u32 = 3;
/// Highest gear that actually integrates.
pub const GEAR_MAX_ACTIVE: u32 = GEAR_CAMPAIGN;
/// Explicit "no value". Shares the encoding of [`GEAR_UNMEASURED`] so a gear
/// slot never has to guess whether `0` means "fine" or "unset".
pub const GEAR_UNSET: u32 = GEAR_UNMEASURED;

/// Tri-state permission. Mirrors `PRISM_GEAR_PERMISSION_*`.
pub const GEAR_PERMISSION_UNKNOWN: u32 = 0;
/// Topology positively lacks the property.
pub const GEAR_PERMISSION_DENIED: u32 = 1;
/// Topology positively has the property.
pub const GEAR_PERMISSION_GRANTED: u32 = 2;

/// Per-gate verdicts. Mirror `PRISM_GEAR_VERDICT_*`.
pub const GEAR_VERDICT_UNMEASURED: u32 = 0;
/// Guard measured and within bounds.
pub const GEAR_VERDICT_OK: u32 = 1;
/// Hold the current gear; no upshift.
pub const GEAR_VERDICT_HOLD: u32 = 2;
/// Drop one gear.
pub const GEAR_VERDICT_DOWNSHIFT: u32 = 3;
/// Drop to the finest gear.
pub const GEAR_VERDICT_FLOOR: u32 = 4;
/// Gear 3 trap.
pub const GEAR_VERDICT_EMERGENCY: u32 = 5;
/// A request was present but out of date.
pub const GEAR_VERDICT_STALE: u32 = 6;

/// Gate identifiers, in strict authority order. Mirror `PRISM_GEAR_GATE_*`.
pub const GEAR_GATE_NONE: u32 = 0;
/// Non-finite energy / momentum violation / adjudicator violation / drift or
/// force past the abort bound. Terminal.
pub const GEAR_GATE_PHYSICAL_EMERGENCY: u32 = 1;
/// HMR and constraint permission.
pub const GEAR_GATE_TOPOLOGY_PERMISSION: u32 = 2;
/// SHAKE/RATTLE failure count.
pub const GEAR_GATE_CONSTRAINT_STABILITY: u32 = 3;
/// Graduated Hamiltonian drift bands.
pub const GEAR_GATE_ENERGY_DRIFT: u32 = 4;
/// Graduated force-norm bands.
pub const GEAR_GATE_FORCE_STABILITY: u32 = 5;
/// The uncorrected-window bound imposed by chunked replay.
pub const GEAR_GATE_CHUNK_LATENCY: u32 = 6;
/// The TSH scientific-resolution request.
pub const GEAR_GATE_TSH_SALIENCE: u32 = 7;
/// Insufficient continuous all-guards-OK dwell for an upshift.
pub const GEAR_GATE_DWELL: u32 = 8;
/// Too soon after an instability for an upshift.
pub const GEAR_GATE_LOCKOUT: u32 = 9;
/// Upshift would have crossed more gears than policy allows.
pub const GEAR_GATE_UPSHIFT_RATE: u32 = 10;
/// No host ever resolved a policy; the ceiling is the documented safe default.
pub const GEAR_GATE_GOVERNOR_UNARMED: u32 = 11;

/// Emergency cause bits. Mirror `PRISM_GEAR_EMERG_*`.
pub const GEAR_EMERG_ADJUDICATION_VIOLATION: u32 = 0x1;
/// Center-of-mass translation guard tripped.
pub const GEAR_EMERG_MOMENTUM_VIOLATION: u32 = 0x2;
/// Potential energy is NaN or Inf.
pub const GEAR_EMERG_NONFINITE_ENERGY: u32 = 0x4;
/// Relative Hamiltonian drift exceeded the abort bound.
pub const GEAR_EMERG_DRIFT_ABORT: u32 = 0x8;
/// Force norm exceeded the abort bound.
pub const GEAR_EMERG_FORCE_ABORT: u32 = 0x10;

/// TSH salience request states. Mirror `PRISM_TSH_SALIENCE_*`.
pub const TSH_SALIENCE_UNSET: u32 = 0;
/// A request is present.
pub const TSH_SALIENCE_VALID: u32 = 1;
/// Layer bound but deliberately silent.
pub const TSH_SALIENCE_HELD: u32 = 2;
/// Salience wire-format version.
pub const TSH_SALIENCE_SCHEMA_VERSION: u32 = 1;

/// Sentinel written when the constraint-failure counter is not bound.
pub const GEAR_CONSTRAINT_UNMEASURED: u32 = 0xFFFF_FFFF;

/// Policy file path key. Optional: absent means "documented defaults only",
/// which is a legal, recorded configuration and not an error.
pub const GEAR_GOVERNOR_POLICY_PATH_ENV: &str = "PRISM_GEARBOX_GOVERNOR_POLICY_PATH";
/// Optional sha256 pin for the policy file. When set, a mismatch is a hard
/// failure — a governor policy that silently changed under the run is exactly
/// the drift this pin exists to catch.
pub const GEAR_GOVERNOR_POLICY_SHA256_ENV: &str = "PRISM_GEARBOX_GOVERNOR_POLICY_SHA256";

// ─── Layout mirrors ─────────────────────────────────────────────────────────

/// Typed insertion point for the TSH policy layer. Mirrors
/// `PrismTshSalienceRequest` (32 bytes).
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TshSalienceRequest {
    /// Must equal [`TSH_SALIENCE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// `TSH_SALIENCE_{UNSET,VALID,HELD}`.
    pub state: u32,
    /// `0..=2`, or [`GEAR_UNSET`] for "no opinion". Never [`GEAR_ABORT`].
    pub requested_gear: u32,
    /// Frame at which the request was written; bounds staleness.
    pub request_frame: u32,
    /// Resolution demand in `[0,1]`. NaN when unset.
    pub salience: f32,
    /// Reserved for the policy layer's own tiering.
    pub priority: u32,
    /// Reserved.
    pub _reserved: [u32; 2],
}

impl TshSalienceRequest {
    /// The "no TSH layer bound" state. `salience` is NaN, not 0.0 — an unset
    /// demand must not read as "demanded nothing".
    pub const fn unset() -> Self {
        Self {
            schema_version: TSH_SALIENCE_SCHEMA_VERSION,
            state: TSH_SALIENCE_UNSET,
            requested_gear: GEAR_UNSET,
            request_frame: 0,
            salience: f32::NAN,
            priority: 0,
            _reserved: [0; 2],
        }
    }

    /// A live request for at most `requested_gear`, stamped at `frame`.
    ///
    /// Returns `None` for [`GEAR_ABORT`] or any gear above
    /// [`GEAR_MAX_ACTIVE`]: the salience layer requests resolution, never an
    /// abort, and the type refuses to encode one.
    pub fn request(requested_gear: u32, salience: f32, frame: u32) -> Option<Self> {
        if requested_gear > GEAR_MAX_ACTIVE {
            return None;
        }
        Some(Self {
            schema_version: TSH_SALIENCE_SCHEMA_VERSION,
            state: TSH_SALIENCE_VALID,
            requested_gear,
            request_frame: frame,
            salience,
            priority: 0,
            _reserved: [0; 2],
        })
    }
}

/// Resolved governor policy. Mirrors `PrismGearGovernorPolicy` (144 bytes).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GearGovernorPolicy {
    /// Must equal [`GEAR_GOVERNOR_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// 1 once a host has resolved and uploaded a policy.
    pub armed: u32,
    /// Ceiling applied when topology permission is UNKNOWN or unarmed.
    pub unknown_topology_cap_gear: u32,
    /// Maximum gears an upshift may cross in one decision.
    pub max_upshift_step: u32,
    /// Continuous all-guards-OK dwell required before any upshift, in steps.
    pub min_upshift_dwell_steps: u32,
    /// Steps that must elapse after an instability before an upshift is legal.
    pub instability_lockout_steps: u32,
    /// Non-zero => an unmeasured force guard blocks upshift.
    pub require_force_guard_for_upshift: u32,
    /// Non-zero => an unmeasured constraint guard blocks upshift.
    pub require_constraint_guard_for_upshift: u32,
    /// Salience requests older than this are treated as unset.
    pub salience_max_age_frames: u32,
    /// Protocol steps per decision boundary (`--captured-graph-body-unroll`).
    pub chunk_span_steps: u32,
    /// Tri-state HMR permission.
    pub hmr_permission: u32,
    /// Tri-state hydrogen-constraint permission.
    pub constraint_permission: u32,
    /// Per-gear timestep in fs, derived from the live gear table.
    pub gear_dt_fs: [f32; 4],
    /// Longest window left uncorrected between decision boundaries, in fs.
    pub max_uncorrected_window_fs: f32,
    /// Normalized energy-balance residual at which the run aborts to Gear 3.
    /// The measurement producer defines the reference: the native physical
    /// guard uses previous accepted endpoint kinetic energy and records its
    /// versioned contract with this policy. This is a stability limit, not an
    /// independently calibrated scientific-accuracy tolerance.
    pub drift_abort: f32,
    /// Fraction of `drift_abort` below which upshift is permitted.
    pub drift_upshift_fraction: f32,
    /// Fraction of `drift_abort` above which the gear is held.
    pub drift_hold_fraction: f32,
    /// Fraction of `drift_abort` above which the gear drops one step.
    pub drift_downshift_fraction: f32,
    /// Integrator force-clamp anchor in kcal/mol/A.
    pub force_clamp_reference: f32,
    /// Fraction of the clamp below which upshift is permitted.
    pub force_upshift_fraction: f32,
    /// Fraction of the clamp above which the gear is held.
    pub force_hold_fraction: f32,
    /// Fraction of the clamp above which the gear drops to the floor.
    pub force_downshift_fraction: f32,
    /// Fraction of the clamp above which the run aborts.
    pub force_abort_fraction: f32,
    /// Constraint failures in the window that block upshift.
    pub constraint_failure_hold: u32,
    /// Constraint failures in the window that force the finest gear.
    pub constraint_failure_downshift: u32,
    /// Host-computed fingerprint binding a gear choice to this exact policy.
    pub policy_digest: u64,
    /// Requester-side hysteresis: consecutive quiet frames before the KL
    /// automaton REQUESTS the campaign gear. Governs the requester, not the
    /// governor — the governor still has to authorize the result. `0` reads as
    /// "use the documented default", never as "no hysteresis".
    pub cruise_threshold_steps: u32,
    /// Reserved.
    pub _reserved: [u32; 5],
}

/// Live measurement bindings. Mirrors `PrismGearGovernorBindings` (32 bytes).
///
/// Every pointer is nullable and null means UNMEASURED. Nothing is ever
/// substituted for a quantity that was not read.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GearGovernorBindings {
    /// Device-visible `f32` force norm in kcal/mol/A, or null.
    pub d_force_norm: *const f32,
    /// Device-visible `u32` SHAKE/RATTLE failure count, or null.
    pub d_constraint_failure_count: *const u32,
    /// Host-stamped epoch of this binding update.
    pub binding_epoch: u64,
    /// Reserved.
    pub _reserved: u64,
}

impl GearGovernorBindings {
    /// All guards UNMEASURED. The honest starting state.
    pub const fn unbound() -> Self {
        Self {
            d_force_norm: std::ptr::null(),
            d_constraint_failure_count: std::ptr::null(),
            binding_epoch: 0,
            _reserved: 0,
        }
    }
}

/// Governor state + last decision record. Mirrors `PrismGearGovernorState`
/// (128 bytes). This is the record that lets an AUTHORIZED gear leave the
/// device together with the policy digest that authorized it.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GearGovernorState {
    /// Must equal [`GEAR_GOVERNOR_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Increments once per governor decision.
    pub decision_epoch: u32,
    /// The gear the governor authorized, or [`GEAR_UNSET`] before the first
    /// decision.
    pub authorized_gear: u32,
    /// The ceiling the gates produced.
    pub ceiling_gear: u32,
    /// The pre-governor scientific request.
    pub requested_gear: u32,
    /// `GEAR_GATE_*` — which gate actually bound the decision.
    pub binding_gate: u32,
    /// Packed 4-bit per-gate verdicts; decode with [`gear_gate_verdict`].
    pub gate_verdicts: u32,
    /// Continuous all-guards-OK dwell, in steps.
    pub stable_dwell_steps: u32,
    /// Steps since the last instability, in steps.
    pub steps_since_instability: u32,
    /// Epoch of the last downshift.
    pub last_downshift_epoch: u32,
    /// Epoch of the last upshift.
    pub last_upshift_epoch: u32,
    /// Bounded downshift latency in steps (== `chunk_span_steps`).
    pub downshift_latency_steps: u32,
    /// Measured relative drift, or NaN when unmeasured.
    pub observed_drift: f32,
    /// Measured force norm, or NaN when unmeasured.
    pub observed_force_norm: f32,
    /// Measured constraint failures, or [`GEAR_CONSTRAINT_UNMEASURED`].
    pub observed_constraint_failures: u32,
    /// `GEAR_EMERG_*` bits.
    pub emergency_flags: u32,
    /// Echo of `policy.armed`.
    pub governor_armed: u32,
    /// Echo of `policy.schema_version`.
    pub policy_schema_version: u32,
    /// Echo of `policy.policy_digest`.
    pub policy_digest: u64,
    /// The salience request in force at the last decision.
    pub salience: TshSalienceRequest,
    /// Reserved.
    pub _reserved: [u32; 4],
}

/// One decision's guard observations. Mirrors `PrismGearGuardObservation`
/// (64 bytes).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GearGuardObservation {
    /// Adjudicator code (0 prune / 1 burst / 2 violation).
    pub adjudication_code: u32,
    /// `adj.momentum_violation_flag`.
    pub momentum_violation: u32,
    /// 1 when the potential energy is NaN or Inf.
    pub nonfinite_energy: u32,
    /// 0 => drift UNMEASURED.
    pub drift_measured: u32,
    /// Relative Hamiltonian drift.
    pub drift: f32,
    /// 0 => force norm UNMEASURED.
    pub force_measured: u32,
    /// Force norm in kcal/mol/A.
    pub force_norm: f32,
    /// 0 => constraint failures UNMEASURED.
    pub constraint_measured: u32,
    /// SHAKE/RATTLE failures in the window.
    pub constraint_failures: u32,
    /// Gear currently in force.
    pub current_gear: u32,
    /// Scientific / KL request.
    pub requested_gear: u32,
    /// Dwell carried from the prior decision.
    pub stable_dwell_steps: u32,
    /// Lockout credit carried from the prior decision.
    pub steps_since_instability: u32,
    /// Current frame index.
    pub frame: u32,
    /// Reserved.
    pub _reserved: [u32; 2],
}

impl GearGuardObservation {
    /// An observation with EVERY guard UNMEASURED. Callers then fill in only
    /// what they actually measured, so forgetting a field yields "unmeasured"
    /// rather than a fabricated zero.
    pub const fn unmeasured(current_gear: u32, requested_gear: u32, frame: u32) -> Self {
        Self {
            adjudication_code: 0,
            momentum_violation: 0,
            nonfinite_energy: 0,
            drift_measured: 0,
            drift: f32::NAN,
            force_measured: 0,
            force_norm: f32::NAN,
            constraint_measured: 0,
            constraint_failures: GEAR_CONSTRAINT_UNMEASURED,
            current_gear,
            requested_gear,
            stable_dwell_steps: 0,
            steps_since_instability: 0,
            frame,
            _reserved: [0; 2],
        }
    }
}

/// A governor decision. Mirrors `PrismGearDecision` (32 bytes).
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GearDecision {
    /// The authorized gear.
    pub authorized_gear: u32,
    /// The ceiling the gates produced.
    pub ceiling_gear: u32,
    /// `GEAR_GATE_*` — which gate bound the decision.
    pub binding_gate: u32,
    /// Packed 4-bit per-gate verdicts.
    pub gate_verdicts: u32,
    /// `GEAR_EMERG_*` bits.
    pub emergency_flags: u32,
    /// Dwell to carry into the next decision.
    pub next_stable_dwell_steps: u32,
    /// Lockout credit to carry into the next decision.
    pub next_steps_since_instability: u32,
    /// Bounded downshift latency in steps.
    pub downshift_latency_steps: u32,
}

const _: () = {
    use std::mem::{align_of, size_of};
    assert!(size_of::<TshSalienceRequest>() == 32);
    assert!(size_of::<GearGovernorPolicy>() == 144);
    assert!(align_of::<GearGovernorPolicy>() == 16);
    assert!(size_of::<GearGovernorBindings>() == 32);
    assert!(size_of::<GearGovernorState>() == 128);
    assert!(align_of::<GearGovernorState>() == 16);
    assert!(size_of::<GearGuardObservation>() == 64);
    assert!(size_of::<GearDecision>() == 32);
};

/// Decode one gate's 4-bit verdict out of a packed `gate_verdicts` word.
/// Gates 1..=8 are packed; anything else reads as UNMEASURED.
pub fn gear_gate_verdict(gate_verdicts: u32, gate_id: u32) -> u32 {
    if !(1..=8).contains(&gate_id) {
        return GEAR_VERDICT_UNMEASURED;
    }
    (gate_verdicts >> ((gate_id - 1) * 4)) & 0xF
}

/// Stable machine-readable label for a `GEAR_GATE_*` id, for evidence records.
pub fn gear_gate_label(gate_id: u32) -> &'static str {
    match gate_id {
        GEAR_GATE_NONE => "none",
        GEAR_GATE_PHYSICAL_EMERGENCY => "physical_emergency",
        GEAR_GATE_TOPOLOGY_PERMISSION => "topology_permission",
        GEAR_GATE_CONSTRAINT_STABILITY => "constraint_stability",
        GEAR_GATE_ENERGY_DRIFT => "energy_drift",
        GEAR_GATE_FORCE_STABILITY => "force_stability",
        GEAR_GATE_CHUNK_LATENCY => "chunk_latency",
        GEAR_GATE_TSH_SALIENCE => "tsh_salience",
        GEAR_GATE_DWELL => "dwell",
        GEAR_GATE_LOCKOUT => "instability_lockout",
        GEAR_GATE_UPSHIFT_RATE => "upshift_rate",
        GEAR_GATE_GOVERNOR_UNARMED => "governor_unarmed",
        _ => "unknown_gate",
    }
}

// ─── Policy resolution ──────────────────────────────────────────────────────

/// Where a resolved policy value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovernorValueOrigin {
    /// A `PRISM_GEAR_DEFAULT_*` literal in `gearbox.cuh`, obtained through
    /// `prism_gearbox_governor_default_policy`.
    SourceLiteralDefault,
    /// Overridden by the versioned policy file.
    PolicyFile,
    /// Derived from host-verified topology facts (`topology.hmr_applied`,
    /// hydrogen-constraint presence).
    HostMeasuredTopology,
    /// Derived from live runtime geometry (body unroll, the gear table).
    HostRuntimeGeometry,
}

impl GovernorValueOrigin {
    /// Stable machine-readable label for evidence records.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SourceLiteralDefault => "source_literal_default",
            Self::PolicyFile => "policy_file",
            Self::HostMeasuredTopology => "host_measured_topology",
            Self::HostRuntimeGeometry => "host_runtime_geometry",
        }
    }
}

/// Named terminal failures of policy resolution.
///
/// Every variant is a hard stop with a specific cause. There is no variant
/// meaning "carried on with a guessed value": a policy that cannot be
/// resolved is not silently replaced by one that can.
#[derive(Debug, thiserror::Error)]
pub enum GovernorPolicyError {
    /// `prism_gearbox_governor_default_policy` refused.
    #[error("C-side default policy fill failed (rc={0})")]
    DefaultFillFailed(i32),
    /// The policy file could not be read.
    #[error("governor policy file {path} could not be read: {source}")]
    PolicyFileUnreadable {
        /// The path that was attempted.
        path: String,
        /// The underlying I/O failure.
        source: std::io::Error,
    },
    /// The policy file was not valid JSON, or carried an unknown field.
    #[error("governor policy file {path} is not a valid policy document: {message}")]
    PolicyFileMalformed {
        /// The path that was attempted.
        path: String,
        /// The parse failure.
        message: String,
    },
    /// `PRISM_GEARBOX_GOVERNOR_POLICY_SHA256` did not match the file.
    #[error("governor policy file {path} sha256 {actual} does not match the pinned {expected}")]
    PolicyFileDigestMismatch {
        /// The path that was attempted.
        path: String,
        /// The digest the file actually has.
        actual: String,
        /// The digest that was pinned.
        expected: String,
    },
    /// The policy document declares a version this build cannot interpret.
    #[error("governor policy file {path} declares schema_version {found}, expected {expected}")]
    SchemaMismatch {
        /// The path that was attempted.
        path: String,
        /// The version in the document.
        found: u32,
        /// The version this build understands.
        expected: u32,
    },
    /// A resolved value is not physically usable.
    #[error("governor policy field '{field}' is invalid: {reason}")]
    InvalidValue {
        /// Which field.
        field: &'static str,
        /// Why it was rejected.
        reason: String,
    },
}

/// Host-verified topology facts that gate the aggressive gears.
///
/// `None` means "the host did not establish this", which resolves to
/// [`GEAR_PERMISSION_UNKNOWN`]. Unknown is NOT granted: the governor caps at
/// the documented safe gear. That is the whole reason the field is an
/// `Option<bool>` and not a `bool`.
#[derive(Debug, Clone, Copy, Default)]
pub struct TopologyPermissions {
    /// `PrismPrepTopology::hmr_applied`. Required for ANY 4 fs gear —
    /// a 4 fs timestep without verified hydrogen-mass repartitioning
    /// corrupts trajectories silently.
    pub hmr_applied: Option<bool>,
    /// Whether X–H bonds are constrained (SHAKE/RATTLE active). Required for
    /// the 2.0 fs gear; without it the governor floors at 0.5 fs.
    pub h_constraints_enabled: Option<bool>,
}

impl TopologyPermissions {
    fn permission(flag: Option<bool>) -> u32 {
        match flag {
            Some(true) => GEAR_PERMISSION_GRANTED,
            Some(false) => GEAR_PERMISSION_DENIED,
            None => GEAR_PERMISSION_UNKNOWN,
        }
    }
}

/// A resolved policy together with the provenance of every value in it.
///
/// The numbers alone are not auditable — `min_upshift_dwell_steps = 500` does
/// not say whether that came from the source literal or from an operator
/// override. `origins` closes that gap, and `digest` binds the whole thing to
/// every device decision made under it.
#[derive(Debug, Clone)]
pub struct GovernorPolicyResolution {
    /// The resolved policy, ready to upload.
    pub policy: GearGovernorPolicy,
    /// `(field_name, origin)` for every governed field, in declaration order.
    pub origins: Vec<(&'static str, GovernorValueOrigin)>,
    /// Path of the policy file, when one was used.
    pub policy_path: Option<String>,
    /// sha256 of the policy file, when one was used.
    pub policy_sha256: Option<String>,
    /// Fingerprint of the resolved policy; echoed by every device decision.
    pub digest: u64,
}

impl GovernorPolicyResolution {
    /// Origin of a single field, for targeted evidence lines.
    pub fn origin_of(&self, field: &str) -> Option<GovernorValueOrigin> {
        self.origins
            .iter()
            .find(|(name, _)| *name == field)
            .map(|(_, origin)| *origin)
    }
}

/// The on-disk policy document. `deny_unknown_fields` makes a document
/// written for a different governor version fail loudly instead of having its
/// extra keys silently ignored.
#[derive(Debug, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GovernorPolicyDocument {
    schema_version: u32,
    #[serde(default)]
    unknown_topology_cap_gear: Option<u32>,
    #[serde(default)]
    max_upshift_step: Option<u32>,
    #[serde(default)]
    min_upshift_dwell_steps: Option<u32>,
    #[serde(default)]
    instability_lockout_steps: Option<u32>,
    #[serde(default)]
    require_force_guard_for_upshift: Option<bool>,
    #[serde(default)]
    require_constraint_guard_for_upshift: Option<bool>,
    #[serde(default)]
    salience_max_age_frames: Option<u32>,
    #[serde(default)]
    max_uncorrected_window_fs: Option<f32>,
    #[serde(default)]
    drift_abort: Option<f32>,
    #[serde(default)]
    drift_upshift_fraction: Option<f32>,
    #[serde(default)]
    drift_hold_fraction: Option<f32>,
    #[serde(default)]
    drift_downshift_fraction: Option<f32>,
    #[serde(default)]
    force_clamp_reference: Option<f32>,
    #[serde(default)]
    force_upshift_fraction: Option<f32>,
    #[serde(default)]
    force_hold_fraction: Option<f32>,
    #[serde(default)]
    force_downshift_fraction: Option<f32>,
    #[serde(default)]
    force_abort_fraction: Option<f32>,
    #[serde(default)]
    constraint_failure_hold: Option<u32>,
    #[serde(default)]
    constraint_failure_downshift: Option<u32>,
    #[serde(default)]
    cruise_threshold_steps: Option<u32>,
}

/// Fetch the C-side documented defaults. This is the only place the default
/// values enter the Rust side, so the two halves cannot drift.
pub fn governor_default_policy() -> Result<GearGovernorPolicy, GovernorPolicyError> {
    let mut policy = unsafe { std::mem::zeroed::<GearGovernorPolicy>() };
    let rc = unsafe { ffi::prism_gearbox_governor_default_policy(&mut policy) };
    if rc != 0 {
        return Err(GovernorPolicyError::DefaultFillFailed(rc));
    }
    Ok(policy)
}

/// Resolve the governor policy for a run.
///
/// Layering, in order, with each layer recorded:
///   1. the C-side documented defaults (`source_literal_default`);
///   2. the versioned policy file at `PRISM_GEARBOX_GOVERNOR_POLICY_PATH`,
///      optionally sha256-pinned (`policy_file`);
///   3. host-verified topology facts (`host_measured_topology`);
///   4. live runtime geometry — body unroll and the actual gear table
///      (`host_runtime_geometry`).
///
/// `gear_table_ps` is the live 16-float gearbox table; slot `g*4` is gear
/// `g`'s timestep in ps. Non-finite entries (the Gear 3 NaN sentinel) become
/// `0.0` fs, which the chunk-latency gate skips.
///
/// The returned policy has `armed = 1`: a successfully resolved policy IS the
/// arming condition. If this function returns an error, nothing is uploaded
/// and the governor stays unarmed — which caps the run at the documented safe
/// gear rather than letting it proceed unrestricted.
pub fn resolve_governor_policy(
    permissions: TopologyPermissions,
    chunk_span_steps: u32,
    gear_table_ps: &[f32; PRISM_GEARBOX_TABLE_LEN],
) -> Result<GovernorPolicyResolution, GovernorPolicyError> {
    let mut policy = governor_default_policy()?;

    let mut origins: Vec<(&'static str, GovernorValueOrigin)> = vec![
        (
            "unknown_topology_cap_gear",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "max_upshift_step",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "min_upshift_dwell_steps",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "instability_lockout_steps",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "require_force_guard_for_upshift",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "require_constraint_guard_for_upshift",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "salience_max_age_frames",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "max_uncorrected_window_fs",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        ("drift_abort", GovernorValueOrigin::SourceLiteralDefault),
        (
            "drift_upshift_fraction",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "drift_hold_fraction",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "drift_downshift_fraction",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "force_clamp_reference",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "force_upshift_fraction",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "force_hold_fraction",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "force_downshift_fraction",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "force_abort_fraction",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "constraint_failure_hold",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "constraint_failure_downshift",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        (
            "cruise_threshold_steps",
            GovernorValueOrigin::SourceLiteralDefault,
        ),
        ("hmr_permission", GovernorValueOrigin::HostMeasuredTopology),
        (
            "constraint_permission",
            GovernorValueOrigin::HostMeasuredTopology,
        ),
        ("chunk_span_steps", GovernorValueOrigin::HostRuntimeGeometry),
        ("gear_dt_fs", GovernorValueOrigin::HostRuntimeGeometry),
    ];

    let mut mark = |field: &'static str| {
        if let Some(slot) = origins.iter_mut().find(|(name, _)| *name == field) {
            slot.1 = GovernorValueOrigin::PolicyFile;
        }
    };

    // ── Layer 2 — versioned policy file (optional) ──
    let mut policy_path: Option<String> = None;
    let mut policy_sha256: Option<String> = None;
    if let Ok(raw_path) = std::env::var(GEAR_GOVERNOR_POLICY_PATH_ENV) {
        let path = raw_path.trim().to_string();
        if !path.is_empty() {
            let bytes = std::fs::read(&path).map_err(|source| {
                GovernorPolicyError::PolicyFileUnreadable {
                    path: path.clone(),
                    source,
                }
            })?;

            let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
            sha2::Digest::update(&mut hasher, &bytes);
            let actual = format!("{:x}", sha2::Digest::finalize(hasher));

            if let Ok(pinned) = std::env::var(GEAR_GOVERNOR_POLICY_SHA256_ENV) {
                let expected = pinned.trim().to_ascii_lowercase();
                if !expected.is_empty() && expected != actual {
                    return Err(GovernorPolicyError::PolicyFileDigestMismatch {
                        path,
                        actual,
                        expected,
                    });
                }
            }

            let doc: GovernorPolicyDocument = serde_json::from_slice(&bytes).map_err(|e| {
                GovernorPolicyError::PolicyFileMalformed {
                    path: path.clone(),
                    message: e.to_string(),
                }
            })?;
            if doc.schema_version != GEAR_GOVERNOR_SCHEMA_VERSION {
                return Err(GovernorPolicyError::SchemaMismatch {
                    path,
                    found: doc.schema_version,
                    expected: GEAR_GOVERNOR_SCHEMA_VERSION,
                });
            }

            if let Some(v) = doc.unknown_topology_cap_gear {
                policy.unknown_topology_cap_gear = v;
                mark("unknown_topology_cap_gear");
            }
            if let Some(v) = doc.max_upshift_step {
                policy.max_upshift_step = v;
                mark("max_upshift_step");
            }
            if let Some(v) = doc.min_upshift_dwell_steps {
                policy.min_upshift_dwell_steps = v;
                mark("min_upshift_dwell_steps");
            }
            if let Some(v) = doc.instability_lockout_steps {
                policy.instability_lockout_steps = v;
                mark("instability_lockout_steps");
            }
            if let Some(v) = doc.require_force_guard_for_upshift {
                policy.require_force_guard_for_upshift = u32::from(v);
                mark("require_force_guard_for_upshift");
            }
            if let Some(v) = doc.require_constraint_guard_for_upshift {
                policy.require_constraint_guard_for_upshift = u32::from(v);
                mark("require_constraint_guard_for_upshift");
            }
            if let Some(v) = doc.salience_max_age_frames {
                policy.salience_max_age_frames = v;
                mark("salience_max_age_frames");
            }
            if let Some(v) = doc.max_uncorrected_window_fs {
                policy.max_uncorrected_window_fs = v;
                mark("max_uncorrected_window_fs");
            }
            if let Some(v) = doc.drift_abort {
                policy.drift_abort = v;
                mark("drift_abort");
            }
            if let Some(v) = doc.drift_upshift_fraction {
                policy.drift_upshift_fraction = v;
                mark("drift_upshift_fraction");
            }
            if let Some(v) = doc.drift_hold_fraction {
                policy.drift_hold_fraction = v;
                mark("drift_hold_fraction");
            }
            if let Some(v) = doc.drift_downshift_fraction {
                policy.drift_downshift_fraction = v;
                mark("drift_downshift_fraction");
            }
            if let Some(v) = doc.force_clamp_reference {
                policy.force_clamp_reference = v;
                mark("force_clamp_reference");
            }
            if let Some(v) = doc.force_upshift_fraction {
                policy.force_upshift_fraction = v;
                mark("force_upshift_fraction");
            }
            if let Some(v) = doc.force_hold_fraction {
                policy.force_hold_fraction = v;
                mark("force_hold_fraction");
            }
            if let Some(v) = doc.force_downshift_fraction {
                policy.force_downshift_fraction = v;
                mark("force_downshift_fraction");
            }
            if let Some(v) = doc.force_abort_fraction {
                policy.force_abort_fraction = v;
                mark("force_abort_fraction");
            }
            if let Some(v) = doc.constraint_failure_hold {
                policy.constraint_failure_hold = v;
                mark("constraint_failure_hold");
            }
            if let Some(v) = doc.constraint_failure_downshift {
                policy.constraint_failure_downshift = v;
                mark("constraint_failure_downshift");
            }
            if let Some(v) = doc.cruise_threshold_steps {
                policy.cruise_threshold_steps = v;
                mark("cruise_threshold_steps");
            }

            policy_path = Some(path);
            policy_sha256 = Some(actual);
        }
    }

    // ── Layer 3 — host-verified topology facts ──
    policy.hmr_permission = TopologyPermissions::permission(permissions.hmr_applied);
    policy.constraint_permission =
        TopologyPermissions::permission(permissions.h_constraints_enabled);

    // ── Layer 4 — live runtime geometry ──
    // chunk_span_steps of 0 is meaningless; 1 (every step is a boundary) is
    // the conservative reading and is what the C default uses.
    policy.chunk_span_steps = chunk_span_steps.max(1);
    for gear in 0..PRISM_GEARBOX_N_GEARS as usize {
        let dt_ps = gear_table_ps[gear * PRISM_GEARBOX_FLOATS_PER_GEAR as usize];
        policy.gear_dt_fs[gear] = if dt_ps.is_finite() && dt_ps > 0.0 {
            dt_ps * 1000.0
        } else {
            // Gear 3's NaN sentinel, or a retuned table with a disabled gear.
            0.0
        };
    }

    validate_governor_policy(&policy)?;

    policy.armed = 1;
    policy.policy_digest = governor_policy_digest(&policy, policy_sha256.as_deref());

    Ok(GovernorPolicyResolution {
        policy,
        origins,
        policy_path,
        policy_sha256,
        digest: policy.policy_digest,
    })
}

/// Reject a policy that is not physically usable.
///
/// Every check names the field and the reason. Nothing is clamped into range:
/// silently repairing a threshold would hide the very misconfiguration the
/// operator needs to see.
fn validate_governor_policy(p: &GearGovernorPolicy) -> Result<(), GovernorPolicyError> {
    let invalid =
        |field: &'static str, reason: String| GovernorPolicyError::InvalidValue { field, reason };

    if p.unknown_topology_cap_gear > GEAR_MAX_ACTIVE {
        return Err(invalid(
            "unknown_topology_cap_gear",
            format!(
                "{} is not an active gear (0..={GEAR_MAX_ACTIVE}); an unknown \
                 topology must never map to the abort gear",
                p.unknown_topology_cap_gear
            ),
        ));
    }
    if p.max_upshift_step == 0 || p.max_upshift_step > GEAR_MAX_ACTIVE {
        return Err(invalid(
            "max_upshift_step",
            format!("{} must be in 1..={GEAR_MAX_ACTIVE}", p.max_upshift_step),
        ));
    }
    if !(p.drift_abort.is_finite() && p.drift_abort > 0.0) {
        return Err(invalid(
            "drift_abort",
            format!("{} must be finite and > 0", p.drift_abort),
        ));
    }
    if !(p.force_clamp_reference.is_finite() && p.force_clamp_reference > 0.0) {
        return Err(invalid(
            "force_clamp_reference",
            format!("{} must be finite and > 0", p.force_clamp_reference),
        ));
    }
    if !(p.max_uncorrected_window_fs.is_finite() && p.max_uncorrected_window_fs > 0.0) {
        return Err(invalid(
            "max_uncorrected_window_fs",
            format!("{} must be finite and > 0", p.max_uncorrected_window_fs),
        ));
    }

    // Band ordering. If these are not strictly increasing the graduated path
    // collapses — e.g. a hold band above the downshift band would mean drift
    // could trigger a downshift without ever having triggered a hold.
    let drift_bands = [
        ("drift_upshift_fraction", p.drift_upshift_fraction),
        ("drift_hold_fraction", p.drift_hold_fraction),
        ("drift_downshift_fraction", p.drift_downshift_fraction),
    ];
    for (name, v) in drift_bands {
        if !(v.is_finite() && v > 0.0 && v <= 1.0) {
            return Err(invalid(name, format!("{v} must be finite and in (0,1]")));
        }
    }
    if !(p.drift_upshift_fraction < p.drift_hold_fraction
        && p.drift_hold_fraction < p.drift_downshift_fraction
        && p.drift_downshift_fraction <= 1.0)
    {
        return Err(invalid(
            "drift_downshift_fraction",
            format!(
                "drift bands must satisfy upshift ({}) < hold ({}) < downshift ({}) <= 1.0",
                p.drift_upshift_fraction, p.drift_hold_fraction, p.drift_downshift_fraction
            ),
        ));
    }

    let force_bands = [
        ("force_upshift_fraction", p.force_upshift_fraction),
        ("force_hold_fraction", p.force_hold_fraction),
        ("force_downshift_fraction", p.force_downshift_fraction),
        ("force_abort_fraction", p.force_abort_fraction),
    ];
    for (name, v) in force_bands {
        if !(v.is_finite() && v > 0.0) {
            return Err(invalid(name, format!("{v} must be finite and > 0")));
        }
    }
    if !(p.force_upshift_fraction < p.force_hold_fraction
        && p.force_hold_fraction < p.force_downshift_fraction
        && p.force_downshift_fraction <= p.force_abort_fraction)
    {
        return Err(invalid(
            "force_abort_fraction",
            format!(
                "force bands must satisfy upshift ({}) < hold ({}) < downshift ({}) \
                 <= abort ({})",
                p.force_upshift_fraction,
                p.force_hold_fraction,
                p.force_downshift_fraction,
                p.force_abort_fraction
            ),
        ));
    }

    if p.constraint_failure_hold > p.constraint_failure_downshift {
        return Err(invalid(
            "constraint_failure_hold",
            format!(
                "hold ({}) must not exceed downshift ({})",
                p.constraint_failure_hold, p.constraint_failure_downshift
            ),
        ));
    }
    if p.chunk_span_steps == 0 {
        return Err(invalid(
            "chunk_span_steps",
            "0 steps per decision boundary is not a bound".to_string(),
        ));
    }
    if !p.gear_dt_fs[..=2].iter().any(|dt| dt.is_finite() && *dt>0.0 && (p.chunk_span_steps as f32)*dt<=p.max_uncorrected_window_fs) {
        return Err(invalid("chunk_span_steps",format!("{} physical steps between observations admits no active gear under {} fs; cadence must be repaired without widening this bound",p.chunk_span_steps,p.max_uncorrected_window_fs)));
    }
    // cruise_threshold_steps == 0 is the documented "use the default"
    // encoding, so it is legal; anything else is accepted as-is because it
    // only shapes the REQUEST, which the governor still has to authorize.
    Ok(())
}

/// Fingerprint of a resolved policy: sha256 over the canonical byte encoding
/// of every governed field plus the policy-file digest, truncated to the
/// leading 8 bytes. Echoed by every device decision so an offline reader can
/// bind a gear choice to the exact policy that authorized it.
fn governor_policy_digest(p: &GearGovernorPolicy, file_sha256: Option<&str>) -> u64 {
    let mut hasher = <sha2::Sha256 as sha2::Digest>::new();
    let u = |h: &mut sha2::Sha256, v: u32| sha2::Digest::update(h, v.to_le_bytes());
    let f = |h: &mut sha2::Sha256, v: f32| sha2::Digest::update(h, v.to_bits().to_le_bytes());

    sha2::Digest::update(&mut hasher, b"PRISM_GEAR_GOVERNOR_POLICY_V1");
    u(&mut hasher, p.schema_version);
    u(&mut hasher, p.unknown_topology_cap_gear);
    u(&mut hasher, p.max_upshift_step);
    u(&mut hasher, p.min_upshift_dwell_steps);
    u(&mut hasher, p.instability_lockout_steps);
    u(&mut hasher, p.require_force_guard_for_upshift);
    u(&mut hasher, p.require_constraint_guard_for_upshift);
    u(&mut hasher, p.salience_max_age_frames);
    u(&mut hasher, p.chunk_span_steps);
    u(&mut hasher, p.hmr_permission);
    u(&mut hasher, p.constraint_permission);
    for dt in p.gear_dt_fs {
        f(&mut hasher, dt);
    }
    f(&mut hasher, p.max_uncorrected_window_fs);
    f(&mut hasher, p.drift_abort);
    f(&mut hasher, p.drift_upshift_fraction);
    f(&mut hasher, p.drift_hold_fraction);
    f(&mut hasher, p.drift_downshift_fraction);
    f(&mut hasher, p.force_clamp_reference);
    f(&mut hasher, p.force_upshift_fraction);
    f(&mut hasher, p.force_hold_fraction);
    f(&mut hasher, p.force_downshift_fraction);
    f(&mut hasher, p.force_abort_fraction);
    u(&mut hasher, p.constraint_failure_hold);
    u(&mut hasher, p.constraint_failure_downshift);
    u(&mut hasher, p.cruise_threshold_steps);
    sha2::Digest::update(
        &mut hasher,
        file_sha256.unwrap_or("no_policy_file").as_bytes(),
    );

    let out = sha2::Digest::finalize(hasher);
    let mut lead = [0u8; 8];
    lead.copy_from_slice(&out[..8]);
    u64::from_be_bytes(lead)
}

/// Evaluate the governor on the host through the production decision
/// function. Pure computation — no CUDA context is required, so this is
/// usable in tests and in dry-run policy validation.
pub fn governor_decide(
    policy: &GearGovernorPolicy,
    observation: &GearGuardObservation,
    salience: Option<&TshSalienceRequest>,
) -> Result<GearDecision, i32> {
    let mut out = unsafe { std::mem::zeroed::<GearDecision>() };
    let tsh = salience
        .map(|s| s as *const TshSalienceRequest)
        .unwrap_or(std::ptr::null());
    let rc = unsafe { ffi::prism_gearbox_governor_decide_host(policy, observation, tsh, &mut out) };
    if rc != 0 {
        return Err(rc);
    }
    Ok(out)
}

/// Upload a resolved policy and reset the device governor state.
///
/// # Safety
/// `stream` must be a valid `CUstream` (or null for the default stream) and
/// must NOT be capturing — the call refuses a capturing stream rather than
/// corrupting the graph, but the pointer itself must still be valid.
pub unsafe fn arm_governor(
    resolution: &GovernorPolicyResolution,
    stream: *mut c_void,
) -> Result<(), i32> {
    let rc = ffi::prism_gearbox_governor_arm_async(&resolution.policy, stream);
    if rc != 0 {
        return Err(rc);
    }
    log::info!(
        "[G26 GOVERNOR] armed: digest={:#018x} hmr={} constraints={} chunk_span={} \
         dwell={} lockout={} drift_abort={} policy_file={}",
        resolution.digest,
        resolution.policy.hmr_permission,
        resolution.policy.constraint_permission,
        resolution.policy.chunk_span_steps,
        resolution.policy.min_upshift_dwell_steps,
        resolution.policy.instability_lockout_steps,
        resolution.policy.drift_abort,
        resolution
            .policy_path
            .as_deref()
            .unwrap_or("<defaults only>")
    );
    Ok(())
}

/// Bind the live physical-measurement device pointers.
///
/// # Safety
/// Every non-null pointer in `bindings` must remain valid and device-readable
/// for the life of the captured graph, and `stream` must be a valid,
/// non-capturing `CUstream` or null.
pub unsafe fn bind_governor_measurements(
    bindings: &GearGovernorBindings,
    stream: *mut c_void,
) -> Result<(), i32> {
    let rc = ffi::prism_gearbox_governor_bind_measurements_async(bindings, stream);
    if rc != 0 {
        return Err(rc);
    }
    Ok(())
}

/// Publish a TSH salience request to the device governor.
///
/// # Safety
/// `stream` must be a valid, non-capturing `CUstream` or null.
pub unsafe fn publish_governor_salience(
    request: &TshSalienceRequest,
    stream: *mut c_void,
) -> Result<(), i32> {
    let rc = ffi::prism_gearbox_governor_publish_salience_async(request, stream);
    if rc != 0 {
        return Err(rc);
    }
    Ok(())
}

/// Read the device governor state back to the host.
///
/// # Safety
/// `stream` must be a valid `CUstream` or null. The call refuses a capturing
/// stream and its wait is bounded.
pub unsafe fn snapshot_governor(stream: *mut c_void) -> Result<GearGovernorState, i32> {
    let mut out = std::mem::zeroed::<GearGovernorState>();
    let rc = ffi::prism_gearbox_governor_snapshot(stream, &mut out);
    if rc != 0 {
        return Err(rc);
    }
    Ok(out)
}

// ============================================================================
// Tests
// ============================================================================
//
// The Wave B.1 acceptance gate ("dt 2.0 fs → 4.0 fs without host
// intervention") lives in `crates/prism-nhs/tests/gearbox_b1_dt_flip.rs`
// — moved out of the lib `#[cfg(test)]` module because pre-existing
// E0063 breakage in unrelated lib tests (solvate.rs / rt_targets.rs /
// input.rs constructing `PrismPrepTopology` without `dimer_dyad`)
// prevents the lib test crate from compiling. The integration target
// compiles independently and exercises the same kernel via the same
// FFI surface.
