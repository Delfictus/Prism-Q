//! Coupled Chronology Record (CCR) — the unified provenance record for the
//! fused Temporal / Gearbox / Tensor-Family decision path.
//!
//! One record, stamped by the producer inside the fence that publishes it,
//! carrying the whole provenance tuple. A post-hoc host-side join cannot
//! recover information a record never carried — that is precisely what failed
//! in the prior campaign.
//!
//! # Why every field here exists
//!
//! | field | why it is not derivable later |
//! |---|---|
//! | `global_md_step` | the bridge between spike-local and frame-global clocks. It lives HERE and nowhere else: the sealed 96 B spike record has no field for it, and widening that record to carry one invalidates the QNANL force-kernel packet. Spike chunks are placed on the frame axis by merging against this stream on `stream_id` + `record_seq`, which is exact because a chunk emits at most one coordinate frame. |
//! | `cumulative_simulated_time_ps` | **mandatory once the gear varies.** Step index stops being proportional to elapsed physics the moment `dt` changes. Two records at the same step can represent different physical moments. |
//! | `authorized_gear` + `dt_applied_ps` | what the integrator was permitted to use, and what it actually used |
//! | `requested_gear` | what the science (TSH) asked for |
//! | `device_measured_gear` | what the device actually ran |
//! | `gear_epoch_id` | monotonic; increments on every authorized transition |
//! | `stream_id`, `replica_id` | file-level today; lost the instant streams merge in memory, and replica timesteps collide |
//! | `emitter_tag` | UV/LIF/EFP and LADD carry *different clocks* in the same file today |
//! | `tensor_snapshot_id` | binds a tensor to its coordinate epoch |
//! | `record_seq` | the COMPLETENESS term. A dropped record at a held gear satisfies the physical-time continuity identity exactly, so cumulative time alone cannot detect it. |
//!
//! # Three separate gear facts
//!
//! `requested_gear` (TSH demand), `authorized_gear` (governor verdict) and
//! `device_measured_gear` (what ran) are three separate facts and MUST NEVER
//! be collapsed. The reason the prior evidence is uninterpretable is that a
//! hardcoded `gear_id: 0` was indistinguishable from a measured gear 0 — and
//! `0` means 0.5 fs, so it silently asserted 4x-to-8x finer temporal
//! resolution than the run actually had.
//!
//! # Sentinel discipline
//!
//! Gear 0 is a LEGITIMATE gear. `0` can therefore NEVER mean "no
//! measurement". Every field has an out-of-band sentinel that no valid
//! measurement can take:
//!
//! * unsigned scalars → [`CCR_U32_UNMEASURED`] / [`CCR_U64_UNMEASURED`]
//! * gears → [`CCR_GEAR_UNMEASURED`] (mirrors `PRISM_GEARBOX_GEAR_UNMEASURED`
//!   in `cuda/gearbox.cuh` and `ZSTR_GEAR_ID_UNMEASURED` in [`crate::zstr`])
//! * `f64` → [`CCR_F64_UNMEASURED`], a quiet NaN with the distinguished
//!   payload `0x5052534D` ("PRSM"). A *different* NaN means "measured as
//!   non-finite", which is a physical trap condition, not an absence. The two
//!   are never conflated.
//! * `transition_reason` / `emitter_tag` reserve `0` as "not recorded" and
//!   number every real value from 1, so a zero-initialised buffer can never
//!   masquerade as a recorded decision.
//!
//! # Layout discipline
//!
//! `#[repr(C, align(64))]`, 192 bytes, no implicit padding, frozen byte
//! offsets — the same discipline [`crate::zstr::ZstrFrameHeader`] uses, and
//! for the same reason: the record is written by a device kernel into a ring
//! and parsed off-line, so a layout drift is a silently corrupt science
//! record rather than a compile error.
//!
//! The offsets are a THREE-way contract, mechanically enforced:
//!
//! 1. `src/cuda/coupled_chronology.cuh` `PRISM_CCR_OFF_*` macros — the device
//!    writer's view.
//! 2. The `CCR_OFF_*` constants in this module, which are **parsed out of
//!    that header at compile time** by [`ccr_cuh_define`] via `include_str!`.
//! 3. The `#[repr(C)]` struct, pinned to (2) by `offset_of!` assertions in a
//!    `const` block.
//!
//! Drift in any one of the three is a COMPILE ERROR. There is no path by
//! which the header and the struct can disagree at runtime.
//!
//! # Hot-path constraint
//!
//! The decision path emits a compact native record into a device ring. NO
//! JSON, hashing, compression or filesystem work happens there. Everything in
//! this module past [`CoupledChronologyRecord`] itself is host-side decode,
//! validation and conservation accounting, which run off the hot path.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
// §1  Compile-time binding to the C header
// ═══════════════════════════════════════════════════════════════════════════

/// The device-side header text, embedded at compile time so its `#define`
/// values can be parsed and asserted against this module's constants.
const CCR_CUH_SOURCE: &str = include_str!("cuda/coupled_chronology.cuh");

const fn ccr_bytes_eq_at(hay: &[u8], at: usize, needle: &[u8]) -> bool {
    if at + needle.len() > hay.len() {
        return false;
    }
    let mut i = 0usize;
    while i < needle.len() {
        if hay[at + i] != needle[i] {
            return false;
        }
        i += 1;
    }
    true
}

const fn ccr_hex_digit(b: u8) -> u64 {
    if b >= b'0' && b <= b'9' {
        (b - b'0') as u64
    } else if b >= b'a' && b <= b'f' {
        (b - b'a') as u64 + 10
    } else if b >= b'A' && b <= b'F' {
        (b - b'A') as u64 + 10
    } else {
        // Unreachable: the caller checks `ccr_is_hex_digit` first.
        0
    }
}

const fn ccr_is_hex_digit(b: u8) -> bool {
    (b >= b'0' && b <= b'9') || (b >= b'a' && b <= b'f') || (b >= b'A' && b <= b'F')
}

/// Parse the integer value of `#define <name> <value>` out of the embedded
/// device header, in a `const` context.
///
/// Handles plain decimal and `0x` hex literals with the usual `u` / `U` /
/// `l` / `L` suffixes. Parenthesised expressions such as `(1u << 3)` are NOT
/// handled here — the bit-flag macros are tied to their Rust mirrors by the
/// `cuh_bit_flag_macros_match_rust` unit test instead.
///
/// Panics at compile time if the macro is absent or unparseable, which is the
/// desired failure mode: a missing offset must never silently default.
pub const fn ccr_cuh_define(name: &str) -> u64 {
    let hay = CCR_CUH_SOURCE.as_bytes();
    let pre = b"#define ";
    let nm = name.as_bytes();
    let mut i = 0usize;
    while i < hay.len() {
        if ccr_bytes_eq_at(hay, i, pre) {
            let after = i + pre.len();
            if ccr_bytes_eq_at(hay, after, nm) {
                let mut j = after + nm.len();
                // A delimiter is REQUIRED so `PRISM_CCR_OFF_RESERVED` cannot
                // match the earlier `PRISM_CCR_OFF_RESERVED_U32` line.
                if j < hay.len() && (hay[j] == b' ' || hay[j] == b'\t') {
                    while j < hay.len() && (hay[j] == b' ' || hay[j] == b'\t') {
                        j += 1;
                    }
                    if j + 1 < hay.len()
                        && hay[j] == b'0'
                        && (hay[j + 1] == b'x' || hay[j + 1] == b'X')
                    {
                        j += 2;
                        let mut v: u64 = 0;
                        let mut any = false;
                        while j < hay.len() && ccr_is_hex_digit(hay[j]) {
                            v = v * 16 + ccr_hex_digit(hay[j]);
                            any = true;
                            j += 1;
                        }
                        if any {
                            return v;
                        }
                    } else {
                        let mut v: u64 = 0;
                        let mut any = false;
                        while j < hay.len() && hay[j] >= b'0' && hay[j] <= b'9' {
                            v = v * 10 + (hay[j] - b'0') as u64;
                            any = true;
                            j += 1;
                        }
                        if any {
                            return v;
                        }
                    }
                }
            }
        }
        i += 1;
    }
    panic!("coupled_chronology.cuh: required #define is missing or unparseable");
}

// ═══════════════════════════════════════════════════════════════════════════
// §2  Schema constants — every value parsed out of the device header
// ═══════════════════════════════════════════════════════════════════════════

/// Current CCR schema version. Bump only when a field is added or its meaning
/// changes; offsets 0..152 are frozen.
pub const CCR_SCHEMA_VERSION: u32 = ccr_cuh_define("PRISM_CCR_SCHEMA_VERSION") as u32;

/// Total on-wire record size in bytes.
pub const CCR_RECORD_BYTES: usize = ccr_cuh_define("PRISM_CCR_RECORD_BYTES") as usize;

/// Required record alignment in bytes.
pub const CCR_RECORD_ALIGN: usize = ccr_cuh_define("PRISM_CCR_RECORD_ALIGN") as usize;

/// Number of reserved `u32` slots in the record tail.
pub const CCR_RESERVED_U32_COUNT: usize = ccr_cuh_define("PRISM_CCR_RESERVED_U32_COUNT") as usize;

// ── Frozen byte offsets ────────────────────────────────────────────────────

pub const CCR_OFF_GLOBAL_MD_STEP: usize = ccr_cuh_define("PRISM_CCR_OFF_GLOBAL_MD_STEP") as usize;
pub const CCR_OFF_CUM_SIM_TIME_PS: usize = ccr_cuh_define("PRISM_CCR_OFF_CUM_SIM_TIME_PS") as usize;
pub const CCR_OFF_DT_APPLIED_PS: usize = ccr_cuh_define("PRISM_CCR_OFF_DT_APPLIED_PS") as usize;
pub const CCR_OFF_GEAR_EPOCH_ID: usize = ccr_cuh_define("PRISM_CCR_OFF_GEAR_EPOCH_ID") as usize;
pub const CCR_OFF_TENSOR_SNAPSHOT_ID: usize =
    ccr_cuh_define("PRISM_CCR_OFF_TENSOR_SNAPSHOT_ID") as usize;
pub const CCR_OFF_TSH_SALIENCE_PER_PS: usize =
    ccr_cuh_define("PRISM_CCR_OFF_TSH_SALIENCE_PER_PS") as usize;
pub const CCR_OFF_TSH_MEMORY_STATE: usize =
    ccr_cuh_define("PRISM_CCR_OFF_TSH_MEMORY_STATE") as usize;
pub const CCR_OFF_TSH_REQ_CONFIDENCE: usize =
    ccr_cuh_define("PRISM_CCR_OFF_TSH_REQ_CONFIDENCE") as usize;
pub const CCR_OFF_OBSERVATION_CONF: usize =
    ccr_cuh_define("PRISM_CCR_OFF_OBSERVATION_CONF") as usize;
pub const CCR_OFF_CONSTRAINT_RESIDUAL: usize =
    ccr_cuh_define("PRISM_CCR_OFF_CONSTRAINT_RESIDUAL") as usize;
pub const CCR_OFF_HAMILTONIAN_DRIFT: usize =
    ccr_cuh_define("PRISM_CCR_OFF_HAMILTONIAN_DRIFT") as usize;
pub const CCR_OFF_FORCE_NORM: usize = ccr_cuh_define("PRISM_CCR_OFF_FORCE_NORM") as usize;
pub const CCR_OFF_SCHEMA_VERSION: usize = ccr_cuh_define("PRISM_CCR_OFF_SCHEMA_VERSION") as usize;
pub const CCR_OFF_RECORD_FLAGS: usize = ccr_cuh_define("PRISM_CCR_OFF_RECORD_FLAGS") as usize;
pub const CCR_OFF_REQUESTED_GEAR: usize = ccr_cuh_define("PRISM_CCR_OFF_REQUESTED_GEAR") as usize;
pub const CCR_OFF_AUTHORIZED_GEAR: usize = ccr_cuh_define("PRISM_CCR_OFF_AUTHORIZED_GEAR") as usize;
pub const CCR_OFF_DEVICE_MEASURED_GEAR: usize =
    ccr_cuh_define("PRISM_CCR_OFF_DEVICE_MEASURED_GEAR") as usize;
pub const CCR_OFF_TSH_ACTIVE_BRANCHES: usize =
    ccr_cuh_define("PRISM_CCR_OFF_TSH_ACTIVE_BRANCHES") as usize;
pub const CCR_OFF_PHYSICAL_GUARD_STATUS: usize =
    ccr_cuh_define("PRISM_CCR_OFF_PHYSICAL_GUARD_STATUS") as usize;
pub const CCR_OFF_TRANSITION_REASON: usize =
    ccr_cuh_define("PRISM_CCR_OFF_TRANSITION_REASON") as usize;
pub const CCR_OFF_HMR_VERIFIED: usize = ccr_cuh_define("PRISM_CCR_OFF_HMR_VERIFIED") as usize;
pub const CCR_OFF_STREAM_ID: usize = ccr_cuh_define("PRISM_CCR_OFF_STREAM_ID") as usize;
pub const CCR_OFF_REPLICA_ID: usize = ccr_cuh_define("PRISM_CCR_OFF_REPLICA_ID") as usize;
pub const CCR_OFF_EMITTER_TAG: usize = ccr_cuh_define("PRISM_CCR_OFF_EMITTER_TAG") as usize;
pub const CCR_OFF_PUBLICATION_FENCE: usize =
    ccr_cuh_define("PRISM_CCR_OFF_PUBLICATION_FENCE") as usize;
pub const CCR_OFF_RESERVED_U32: usize = ccr_cuh_define("PRISM_CCR_OFF_RESERVED_U32") as usize;
pub const CCR_OFF_RECORD_SEQ: usize = ccr_cuh_define("PRISM_CCR_OFF_RECORD_SEQ") as usize;
pub const CCR_OFF_RESERVED: usize = ccr_cuh_define("PRISM_CCR_OFF_RESERVED") as usize;

// ── Sentinels ──────────────────────────────────────────────────────────────

/// "Not measured" for a `u32` field whose `0` is a legitimate value.
pub const CCR_U32_UNMEASURED: u32 = ccr_cuh_define("PRISM_CCR_U32_UNMEASURED") as u32;

/// "Not measured" for a `u64` field whose `0` is a legitimate value.
pub const CCR_U64_UNMEASURED: u64 = ccr_cuh_define("PRISM_CCR_U64_UNMEASURED");

/// "No gear measurement exists."
///
/// Gear `0` means 0.5 fs and is a LEGITIMATE gear, so absence must be out of
/// band. Mirrors `PRISM_GEARBOX_GEAR_UNMEASURED` (`cuda/gearbox.cuh`) and
/// [`crate::zstr::ZSTR_GEAR_ID_UNMEASURED`].
pub const CCR_GEAR_UNMEASURED: u32 = ccr_cuh_define("PRISM_CCR_GEAR_UNMEASURED") as u32;

/// Bit pattern of the `f64` "not measured" sentinel: a quiet NaN carrying the
/// payload `0x5052534D` ("PRSM").
pub const CCR_F64_UNMEASURED_BITS: u64 = ccr_cuh_define("PRISM_CCR_F64_UNMEASURED_BITS");

/// The `f64` "not measured" sentinel.
///
/// It is a NaN, so it can never be read as a physical quantity, and its
/// payload is distinctive, so it stays separable from an ordinary NaN
/// produced by a real numerical trap (a G29 non-finite force norm is a
/// MEASUREMENT of a non-physical state, not an absence).
pub const CCR_F64_UNMEASURED: f64 = f64::from_bits(CCR_F64_UNMEASURED_BITS);

/// True iff `v` carries the exact "not measured" payload.
#[inline]
pub fn ccr_f64_is_unmeasured(v: f64) -> bool {
    v.to_bits() == CCR_F64_UNMEASURED_BITS
}

// ── Gear encoding ──────────────────────────────────────────────────────────

pub const CCR_GEAR_FINE_0P5FS: u32 = ccr_cuh_define("PRISM_CCR_GEAR_FINE_0P5FS") as u32;
pub const CCR_GEAR_DEFAULT_2FS: u32 = ccr_cuh_define("PRISM_CCR_GEAR_DEFAULT_2FS") as u32;
pub const CCR_GEAR_HMR_4FS: u32 = ccr_cuh_define("PRISM_CCR_GEAR_HMR_4FS") as u32;
pub const CCR_GEAR_ABORT_TRAP: u32 = ccr_cuh_define("PRISM_CCR_GEAR_ABORT_TRAP") as u32;
pub const CCR_GEAR_MAX: u32 = ccr_cuh_define("PRISM_CCR_GEAR_MAX") as u32;

/// Source-literal mirror of `__constant__ float d_gearbox_table`
/// (`crates/prism-nhs/src/cuda/gearbox.cu:28-33`), in PICOSECONDS.
///
/// Gear 3 is the abort trap: the device table holds NaN there and the host
/// canonical table (`gearbox.rs:112-122`) overwrites gear 3 with
/// `f32::NAN x4`. It has NO timestep, which is why it is `None` here rather
/// than a number.
///
/// This is used ONLY to cross-check a record's `dt_applied_ps` against its
/// `device_measured_gear`. It never supplies a value into a record — a
/// derived dt would be a fabricated measurement.
pub const CCR_GEAR_DT_PS_SOURCE_LITERAL: [Option<f64>; 4] =
    [Some(0.0005), Some(0.0020), Some(0.0040), None];

// ── HMR tri-state ──────────────────────────────────────────────────────────

pub const CCR_HMR_NOT_APPLIED: u32 = ccr_cuh_define("PRISM_CCR_HMR_NOT_APPLIED") as u32;
pub const CCR_HMR_APPLIED: u32 = ccr_cuh_define("PRISM_CCR_HMR_APPLIED") as u32;
pub const CCR_HMR_UNVERIFIED: u32 = ccr_cuh_define("PRISM_CCR_HMR_UNVERIFIED") as u32;

// ── Publication fence ──────────────────────────────────────────────────────

pub const CCR_FENCE_DIRTY: u32 = ccr_cuh_define("PRISM_CCR_FENCE_DIRTY") as u32;
pub const CCR_FENCE_PUBLISHED: u32 = ccr_cuh_define("PRISM_CCR_FENCE_PUBLISHED") as u32;

// ── record_flags bits (mirrors PRISM_CCR_FLAG_*) ───────────────────────────

/// Set on EVERY record the decision path produces.
///
/// **Emit on no-change too.** A gap in the record stream must never be
/// ambiguous between "the gear was held" and "the decision did not run".
pub const CCR_FLAG_DECISION_EXECUTED: u32 = 1 << 0;
/// The authorized gear is unchanged from the previous record.
pub const CCR_FLAG_GEAR_HELD: u32 = 1 << 1;
/// The authorized gear differs from the previous record.
pub const CCR_FLAG_GEAR_CHANGED: u32 = 1 << 2;
/// Decision staged, not yet applied: gear flips land end-of-chunk when
/// `body_unroll > 1`, so the authorized gear was not live at this step.
pub const CCR_FLAG_STAGED_PENDING_BOUNDARY: u32 = 1 << 3;
/// The governor selected a conservative gear for the WHOLE CHUNK on entry.
pub const CCR_FLAG_CHUNK_ENTRY_CONSERVATIVE: u32 = 1 << 4;
/// Anti-blindness: forced high-resolution sentinel sample.
pub const CCR_FLAG_SENTINEL_SAMPLE: u32 = 1 << 5;
/// Anti-blindness: observation confidence below threshold — coarsening
/// prohibited regardless of salience.
pub const CCR_FLAG_OBSERVATION_STARVED: u32 = 1 << 6;
/// The chronology clock did NOT advance because `dt_applied_ps` was
/// unmeasured. Advancing by the authorized dt instead would fabricate
/// elapsed physics.
pub const CCR_FLAG_CLOCK_STALLED: u32 = 1 << 7;
/// Uncertainty was grown for this interval because evidence was sparse.
pub const CCR_FLAG_UNCERTAINTY_GROWN: u32 = 1 << 8;
/// The abort trap (gear 3) fired.
pub const CCR_FLAG_ABORT_TRAP: u32 = 1 << 9;

// ── physical_guard_status bits (mirrors PRISM_CCR_GUARD_*) ─────────────────
//
// A SET bit means the guard is ASSERTING. Bit order is the LEXICOGRAPHIC
// decision order; lower bit index == higher priority. This is deliberately
// NOT a weighted score: a weighted score always admits a salience value
// large enough to outvote a physical prohibition.

pub const CCR_GUARD_PHYSICAL_EMERGENCY: u32 = 1 << 0;
pub const CCR_GUARD_TOPOLOGY_FORBIDS: u32 = 1 << 1;
pub const CCR_GUARD_CONSTRAINT_UNSTABLE: u32 = 1 << 2;
pub const CCR_GUARD_ENERGY_DRIFT: u32 = 1 << 3;
pub const CCR_GUARD_FORCE_UNSTABLE: u32 = 1 << 4;
pub const CCR_GUARD_TSH_MEMORY: u32 = 1 << 5;
pub const CCR_GUARD_TSH_SALIENCE: u32 = 1 << 6;
pub const CCR_GUARD_PHASE_REQUIREMENT: u32 = 1 << 7;
pub const CCR_GUARD_DWELL_UNSATISFIED: u32 = 1 << 8;
pub const CCR_GUARD_OBSERVATION_CONF: u32 = 1 << 9;
pub const CCR_GUARD_INSTABILITY_MEMORY: u32 = 1 << 10;

/// Number of defined guard bits.
pub const CCR_GUARD_BIT_COUNT: u32 = ccr_cuh_define("PRISM_CCR_GUARD_BIT_COUNT") as u32;

/// Guard bits in lexicographic priority order, highest priority first.
pub const CCR_GUARD_PRIORITY_ORDER: [(u32, &str); 11] = [
    (CCR_GUARD_PHYSICAL_EMERGENCY, "physical_emergency"),
    (CCR_GUARD_TOPOLOGY_FORBIDS, "topology_permission"),
    (CCR_GUARD_CONSTRAINT_UNSTABLE, "constraint_stability"),
    (CCR_GUARD_ENERGY_DRIFT, "energy_drift"),
    (CCR_GUARD_FORCE_UNSTABLE, "force_stability"),
    (CCR_GUARD_TSH_MEMORY, "tsh_memory"),
    (CCR_GUARD_TSH_SALIENCE, "tsh_salience"),
    (CCR_GUARD_PHASE_REQUIREMENT, "phase_requirement"),
    (CCR_GUARD_DWELL_UNSATISFIED, "current_gear_dwell"),
    (CCR_GUARD_OBSERVATION_CONF, "observation_confidence"),
    (CCR_GUARD_INSTABILITY_MEMORY, "instability_memory"),
];

// ── transition_reason values ───────────────────────────────────────────────

pub const CCR_REASON_NOT_RECORDED: u32 = ccr_cuh_define("PRISM_CCR_REASON_NOT_RECORDED") as u32;
pub const CCR_REASON_INITIALIZATION: u32 = ccr_cuh_define("PRISM_CCR_REASON_INITIALIZATION") as u32;
pub const CCR_REASON_HELD_NO_CHANGE: u32 = ccr_cuh_define("PRISM_CCR_REASON_HELD_NO_CHANGE") as u32;
pub const CCR_REASON_DOWNSHIFT_SALIENCE: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_DOWNSHIFT_SALIENCE") as u32;
pub const CCR_REASON_DOWNSHIFT_PHYSICAL_GUARD: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_DOWNSHIFT_PHYSICAL_GUARD") as u32;
pub const CCR_REASON_UPSHIFT_GRANTED: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_UPSHIFT_GRANTED") as u32;
pub const CCR_REASON_UPSHIFT_DENIED_GUARD: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_UPSHIFT_DENIED_GUARD") as u32;
pub const CCR_REASON_UPSHIFT_DENIED_OBSERVATION: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_UPSHIFT_DENIED_OBSERVATION") as u32;
pub const CCR_REASON_UPSHIFT_DENIED_DWELL: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_UPSHIFT_DENIED_DWELL") as u32;
pub const CCR_REASON_TOPOLOGY_CEILING: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_TOPOLOGY_CEILING") as u32;
pub const CCR_REASON_CHUNK_ENTRY_CONSERVATIVE: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_CHUNK_ENTRY_CONSERVATIVE") as u32;
pub const CCR_REASON_SENTINEL_SAMPLE_FORCED: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_SENTINEL_SAMPLE_FORCED") as u32;
pub const CCR_REASON_ABORT_TRAP: u32 = ccr_cuh_define("PRISM_CCR_REASON_ABORT_TRAP") as u32;
pub const CCR_REASON_STAGED_PENDING_BOUNDARY: u32 =
    ccr_cuh_define("PRISM_CCR_REASON_STAGED_PENDING_BOUNDARY") as u32;
pub const CCR_REASON_MAX: u32 = ccr_cuh_define("PRISM_CCR_REASON_MAX") as u32;

/// Stable label for a `transition_reason` value; `None` when unrecognised.
pub fn ccr_reason_name(reason: u32) -> Option<&'static str> {
    Some(match reason {
        r if r == CCR_REASON_NOT_RECORDED => "not_recorded",
        r if r == CCR_REASON_INITIALIZATION => "initialization",
        r if r == CCR_REASON_HELD_NO_CHANGE => "held_no_change",
        r if r == CCR_REASON_DOWNSHIFT_SALIENCE => "downshift_salience",
        r if r == CCR_REASON_DOWNSHIFT_PHYSICAL_GUARD => "downshift_physical_guard",
        r if r == CCR_REASON_UPSHIFT_GRANTED => "upshift_granted",
        r if r == CCR_REASON_UPSHIFT_DENIED_GUARD => "upshift_denied_guard",
        r if r == CCR_REASON_UPSHIFT_DENIED_OBSERVATION => "upshift_denied_observation",
        r if r == CCR_REASON_UPSHIFT_DENIED_DWELL => "upshift_denied_dwell",
        r if r == CCR_REASON_TOPOLOGY_CEILING => "topology_ceiling",
        r if r == CCR_REASON_CHUNK_ENTRY_CONSERVATIVE => "chunk_entry_conservative",
        r if r == CCR_REASON_SENTINEL_SAMPLE_FORCED => "sentinel_sample_forced",
        r if r == CCR_REASON_ABORT_TRAP => "abort_trap",
        r if r == CCR_REASON_STAGED_PENDING_BOUNDARY => "staged_pending_boundary",
        _ => return None,
    })
}

// ── emitter_tag values ─────────────────────────────────────────────────────

pub const CCR_EMITTER_UNSPECIFIED: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_UNSPECIFIED") as u32;
pub const CCR_EMITTER_GEARBOX_DECISION: u32 =
    ccr_cuh_define("PRISM_CCR_EMITTER_GEARBOX_DECISION") as u32;
pub const CCR_EMITTER_TENSOR_SNAPSHOT: u32 =
    ccr_cuh_define("PRISM_CCR_EMITTER_TENSOR_SNAPSHOT") as u32;
pub const CCR_EMITTER_UV: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_UV") as u32;
pub const CCR_EMITTER_LIF: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_LIF") as u32;
pub const CCR_EMITTER_EFP: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_EFP") as u32;
pub const CCR_EMITTER_LADD: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_LADD") as u32;
pub const CCR_EMITTER_COFIRE: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_COFIRE") as u32;
pub const CCR_EMITTER_ZSTR_FRAME: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_ZSTR_FRAME") as u32;
pub const CCR_EMITTER_GHOST_SEGMENT: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_GHOST_SEGMENT") as u32;
pub const CCR_EMITTER_SPIKE_ARROW: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_SPIKE_ARROW") as u32;
pub const CCR_EMITTER_MAX: u32 = ccr_cuh_define("PRISM_CCR_EMITTER_MAX") as u32;

// ═══════════════════════════════════════════════════════════════════════════
// §3  The record
// ═══════════════════════════════════════════════════════════════════════════

/// The unified coupled-chronology record.
///
/// `#[repr(C, align(64))]`, exactly [`CCR_RECORD_BYTES`] bytes, no implicit
/// padding — every byte from 0 to 192 is a named field. A device kernel
/// writes this struct directly into a ring slot through
/// `PrismCoupledChronologyRecord` in `cuda/coupled_chronology.cuh`.
///
/// # Reading a record
///
/// Never read the public fields directly for scientific purposes: they carry
/// sentinels. Use the `*_opt()` accessors, which return `None` for
/// not-measured. The raw fields are public only because the struct is a wire
/// format shared with device code.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CoupledChronologyRecord {
    // ── chronology core ──
    /// Global MD step index this record describes. [`CCR_U64_UNMEASURED`]
    /// when not established.
    pub global_md_step: u64, // @0
    /// Cumulative simulated time in picoseconds at `global_md_step`.
    ///
    /// **Mandatory once the gear varies.** Step index stops being
    /// proportional to elapsed physics the moment `dt` changes.
    /// [`CCR_F64_UNMEASURED`] when not established.
    pub cumulative_simulated_time_ps: f64, // @8
    /// The timestep in picoseconds that was ACTUALLY integrated over the
    /// interval ending at `global_md_step`. Not the requested dt, not the
    /// authorized dt. [`CCR_F64_UNMEASURED`] when not established.
    pub dt_applied_ps: f64, // @16
    /// Monotonic gear epoch; increments on every authorized transition.
    pub gear_epoch_id: u64, // @24
    /// Identifier of the tensor snapshot bound to this coordinate epoch.
    pub tensor_snapshot_id: u64, // @32

    // ── TSH demand summary (scientific DEMAND — never authority) ──
    /// Integrated salience NORMALISED PER SIMULATED PICOSECOND.
    ///
    /// Never per record, per chunk, or per invocation. Per-record
    /// normalisation creates positive feedback into blindness: low salience →
    /// coarser gear → fewer snapshots → less evidence → lower measured
    /// salience → coarser still, which is stable in the WRONG equilibrium.
    pub tsh_integrated_salience_per_ps: f64, // @40
    /// TSH memory state scalar.
    pub tsh_memory_state: f64, // @48
    /// TSH confidence in its own resolution request.
    pub tsh_request_confidence: f64, // @56
    /// Anti-blindness observation confidence. Coarsening is prohibited while
    /// this is below the policy threshold, regardless of salience.
    pub observation_confidence: f64, // @64

    // ── physical governor summary (SOLE safety authority) ──
    /// Constraint residual observed by the governor.
    pub constraint_residual: f64, // @72
    /// Hamiltonian drift `|V_t - (V_{t-1} + W_ext)| / |V_{t-1} + W_ext|`.
    pub hamiltonian_drift: f64, // @80
    /// `||F||_2` of the integrator force buffer.
    pub force_norm: f64, // @88

    // ── 4-byte block ──
    /// [`CCR_SCHEMA_VERSION`].
    pub schema_version: u32, // @96
    /// `CCR_FLAG_*` bitmask.
    pub record_flags: u32, // @100
    /// Gear the TSH arm DEMANDED. Not a measurement of what ran.
    pub requested_gear: u32, // @104
    /// Gear the governor AUTHORIZED = `min(demanded, permitted)`.
    pub authorized_gear: u32, // @108
    /// Gear the DEVICE actually ran. The only field that states the
    /// resolution the integrator held.
    pub device_measured_gear: u32, // @112
    /// TSH active-branch bitmask. `0` is legitimate (no branches active);
    /// absence is [`CCR_U32_UNMEASURED`].
    pub tsh_active_branches: u32, // @116
    /// `CCR_GUARD_*` bitmask. `0` is legitimate (all guards clear); absence
    /// is [`CCR_U32_UNMEASURED`].
    pub physical_guard_status: u32, // @120
    /// `CCR_REASON_*`. `0` is [`CCR_REASON_NOT_RECORDED`].
    pub transition_reason: u32, // @124
    /// HMR tri-state: applied / not-applied / unverified.
    pub hmr_verified: u32, // @128
    /// Producing stream. `0` is a legitimate stream id.
    pub stream_id: u32, // @132
    /// Producing replica. `0` is a legitimate replica id.
    pub replica_id: u32, // @136
    /// `CCR_EMITTER_*`. `0` is [`CCR_EMITTER_UNSPECIFIED`].
    pub emitter_tag: u32, // @140
    /// `CCR_FENCE_DIRTY` / `CCR_FENCE_PUBLISHED`. Raised LAST by the
    /// producer, after a system-scope threadfence.
    pub publication_fence: u32, // @144
    /// Reserved; producers write 0.
    pub _reserved_u32: u32, // @148
    /// Per-emitter monotonic record sequence, incremented by exactly 1 on
    /// every emitted record. [`CCR_U64_UNMEASURED`] when not stamped.
    ///
    /// **Why a sequence is required even though the record carries
    /// cumulative simulated time.** The physical-time continuity identity
    /// `cum[i] - cum[i-1] == dt_applied[i] * (step[i] - step[i-1])` detects
    /// a lost record ONLY when the gear changed across the loss. While the
    /// gear is HELD — the common case — a dropped record leaves the identity
    /// exactly satisfied, because the survivors' larger step gap multiplied
    /// by the same constant `dt` still reproduces the elapsed time. The
    /// continuity identity is a PHYSICS check; the sequence is the
    /// COMPLETENESS check. They are separate conservation terms and neither
    /// substitutes for the other.
    pub record_seq: u64, // @152
    /// Reserved tail; producers write 0.
    pub _reserved: [u32; 8], // @160..192
}

// Compile-time pin: the struct layout MUST match the offsets the device
// kernel stamps through and the offline parser reads. Drift here is a
// silently corrupt science record, so it is a compile error instead.
const _: () = {
    use std::mem::{align_of, offset_of, size_of};

    assert!(size_of::<CoupledChronologyRecord>() == CCR_RECORD_BYTES);
    assert!(align_of::<CoupledChronologyRecord>() == CCR_RECORD_ALIGN);
    assert!(CCR_RECORD_BYTES == 192);
    assert!(CCR_RECORD_ALIGN == 64);
    assert!(CCR_RESERVED_U32_COUNT == 8);

    // No implicit padding: the named fields tile the whole record.
    //   3 x u64  global_md_step, gear_epoch_id, tensor_snapshot_id      24 B
    //   9 x f64  cum_time, dt_applied, 4 TSH, 3 governor                72 B
    //  13 x u32  schema_version .. publication_fence                    52 B
    //   1 x u32  _reserved_u32                                           4 B
    //   1 x u64  record_seq                                              8 B
    //   8 x u32  _reserved                                              32 B
    //  ─────────────────────────────────────────────────────────────────────
    //                                                                  192 B
    assert!(
        size_of::<u64>() * 3
            + size_of::<f64>() * 9
            + size_of::<u32>() * 13
            + size_of::<u32>()
            + size_of::<u64>()
            + size_of::<u32>() * CCR_RESERVED_U32_COUNT
            == CCR_RECORD_BYTES
    );

    // Rust struct <-> Rust offset constants.
    assert!(offset_of!(CoupledChronologyRecord, global_md_step) == CCR_OFF_GLOBAL_MD_STEP);
    assert!(
        offset_of!(CoupledChronologyRecord, cumulative_simulated_time_ps)
            == CCR_OFF_CUM_SIM_TIME_PS
    );
    assert!(offset_of!(CoupledChronologyRecord, dt_applied_ps) == CCR_OFF_DT_APPLIED_PS);
    assert!(offset_of!(CoupledChronologyRecord, gear_epoch_id) == CCR_OFF_GEAR_EPOCH_ID);
    assert!(offset_of!(CoupledChronologyRecord, tensor_snapshot_id) == CCR_OFF_TENSOR_SNAPSHOT_ID);
    assert!(
        offset_of!(CoupledChronologyRecord, tsh_integrated_salience_per_ps)
            == CCR_OFF_TSH_SALIENCE_PER_PS
    );
    assert!(offset_of!(CoupledChronologyRecord, tsh_memory_state) == CCR_OFF_TSH_MEMORY_STATE);
    assert!(
        offset_of!(CoupledChronologyRecord, tsh_request_confidence) == CCR_OFF_TSH_REQ_CONFIDENCE
    );
    assert!(
        offset_of!(CoupledChronologyRecord, observation_confidence) == CCR_OFF_OBSERVATION_CONF
    );
    assert!(
        offset_of!(CoupledChronologyRecord, constraint_residual) == CCR_OFF_CONSTRAINT_RESIDUAL
    );
    assert!(offset_of!(CoupledChronologyRecord, hamiltonian_drift) == CCR_OFF_HAMILTONIAN_DRIFT);
    assert!(offset_of!(CoupledChronologyRecord, force_norm) == CCR_OFF_FORCE_NORM);
    assert!(offset_of!(CoupledChronologyRecord, schema_version) == CCR_OFF_SCHEMA_VERSION);
    assert!(offset_of!(CoupledChronologyRecord, record_flags) == CCR_OFF_RECORD_FLAGS);
    assert!(offset_of!(CoupledChronologyRecord, requested_gear) == CCR_OFF_REQUESTED_GEAR);
    assert!(offset_of!(CoupledChronologyRecord, authorized_gear) == CCR_OFF_AUTHORIZED_GEAR);
    assert!(
        offset_of!(CoupledChronologyRecord, device_measured_gear) == CCR_OFF_DEVICE_MEASURED_GEAR
    );
    assert!(
        offset_of!(CoupledChronologyRecord, tsh_active_branches) == CCR_OFF_TSH_ACTIVE_BRANCHES
    );
    assert!(
        offset_of!(CoupledChronologyRecord, physical_guard_status) == CCR_OFF_PHYSICAL_GUARD_STATUS
    );
    assert!(offset_of!(CoupledChronologyRecord, transition_reason) == CCR_OFF_TRANSITION_REASON);
    assert!(offset_of!(CoupledChronologyRecord, hmr_verified) == CCR_OFF_HMR_VERIFIED);
    assert!(offset_of!(CoupledChronologyRecord, stream_id) == CCR_OFF_STREAM_ID);
    assert!(offset_of!(CoupledChronologyRecord, replica_id) == CCR_OFF_REPLICA_ID);
    assert!(offset_of!(CoupledChronologyRecord, emitter_tag) == CCR_OFF_EMITTER_TAG);
    assert!(offset_of!(CoupledChronologyRecord, publication_fence) == CCR_OFF_PUBLICATION_FENCE);
    assert!(offset_of!(CoupledChronologyRecord, _reserved_u32) == CCR_OFF_RESERVED_U32);
    assert!(offset_of!(CoupledChronologyRecord, record_seq) == CCR_OFF_RECORD_SEQ);
    assert!(offset_of!(CoupledChronologyRecord, _reserved) == CCR_OFF_RESERVED);

    // Sentinel sanity: gear 0 is a LEGITIMATE gear and must never collide
    // with "unknown".
    assert!(CCR_GEAR_FINE_0P5FS == 0);
    assert!(CCR_GEAR_UNMEASURED != CCR_GEAR_FINE_0P5FS);
    assert!(CCR_GEAR_UNMEASURED > CCR_GEAR_MAX);
    assert!(CCR_U32_UNMEASURED == u32::MAX);
    assert!(CCR_U64_UNMEASURED == u64::MAX);
    assert!(CCR_HMR_UNVERIFIED != CCR_HMR_APPLIED);
    assert!(CCR_HMR_UNVERIFIED != CCR_HMR_NOT_APPLIED);
    assert!(CCR_REASON_NOT_RECORDED == 0);
    assert!(CCR_EMITTER_UNSPECIFIED == 0);
    // The f64 sentinel must be a quiet NaN (exponent all ones, mantissa MSB
    // set) so it can never be read as a physical quantity.
    assert!(CCR_F64_UNMEASURED_BITS & 0x7FF8_0000_0000_0000 == 0x7FF8_0000_0000_0000);
    assert!(CCR_F64_UNMEASURED_BITS != 0x7FF8_0000_0000_0000); // distinguishable payload
    assert!(CCR_GUARD_BIT_COUNT == 11);
};

/// Byte-exact equality. `f64` fields are compared BITWISE, not numerically,
/// because a NaN sentinel must round-trip as equal to itself — `NaN == NaN`
/// is false and would make every unmeasured record compare unequal.
impl PartialEq for CoupledChronologyRecord {
    fn eq(&self, other: &Self) -> bool {
        self.to_bytes() == other.to_bytes()
    }
}
impl Eq for CoupledChronologyRecord {}

/// HMR topology-permission tri-state.
///
/// The governor must be able to say "I have not established topology
/// permission", which is a different fact from "HMR is off". Collapsing the
/// two would let an unestablished permission read as a verified prohibition
/// (or worse, as a verified grant).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HmrVerification {
    /// Verified: `topology.hmr_applied == true`.
    Applied,
    /// Verified: `topology.hmr_applied == false`.
    NotApplied,
    /// NOT established. No topology permission has been proven either way.
    Unverified,
}

impl HmrVerification {
    /// Wire encoding.
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Applied => CCR_HMR_APPLIED,
            Self::NotApplied => CCR_HMR_NOT_APPLIED,
            Self::Unverified => CCR_HMR_UNVERIFIED,
        }
    }

    /// Decode. Any unrecognised encoding decodes to [`Self::Unverified`] —
    /// an unreadable permission is an unestablished permission, never a
    /// grant.
    pub fn from_u32(v: u32) -> Self {
        if v == CCR_HMR_APPLIED {
            Self::Applied
        } else if v == CCR_HMR_NOT_APPLIED {
            Self::NotApplied
        } else {
            Self::Unverified
        }
    }

    /// True only when topology permission was actually established.
    pub fn is_established(self) -> bool {
        !matches!(self, Self::Unverified)
    }
}

impl Default for CoupledChronologyRecord {
    /// A fully UNMEASURED record.
    ///
    /// Every field carries its explicit sentinel; nothing reads as a
    /// measurement. This is the exact host mirror of
    /// `prism_ccr_reset_unmeasured` in the device header, so a record built
    /// here and a record built on device start from the same state.
    fn default() -> Self {
        Self::unmeasured()
    }
}

/// Map a `GpuSpikeEvent.spike_source` onto a `CCR_EMITTER_*` tag.
///
/// The spike record's `spike_source` names the PHYSICS CHANNEL (1=UV, 2=LIF,
/// 3=EFP, 4=LADD, 5=COFIRE). An unrecognised channel maps to
/// [`CCR_EMITTER_UNSPECIFIED`] rather than to a plausible neighbour: mislabeling
/// an emitter is worse than declining to label it, because UV/LIF/EFP and LADD
/// are stamped from different clocks.
pub fn spike_source_to_emitter_tag(spike_source: i32) -> u32 {
    match spike_source {
        1 => CCR_EMITTER_UV,
        2 => CCR_EMITTER_LIF,
        3 => CCR_EMITTER_EFP,
        4 => CCR_EMITTER_LADD,
        5 => CCR_EMITTER_COFIRE,
        _ => CCR_EMITTER_UNSPECIFIED,
    }
}

impl CoupledChronologyRecord {
    /// A fully UNMEASURED record with the schema version stamped.
    ///
    /// Producers MUST start here and then overwrite only the fields they
    /// actually measured, so an unwritten field reads as explicit absence
    /// rather than as a zero that looks like a measurement.
    pub fn unmeasured() -> Self {
        Self {
            global_md_step: CCR_U64_UNMEASURED,
            cumulative_simulated_time_ps: CCR_F64_UNMEASURED,
            dt_applied_ps: CCR_F64_UNMEASURED,
            gear_epoch_id: CCR_U64_UNMEASURED,
            tensor_snapshot_id: CCR_U64_UNMEASURED,
            tsh_integrated_salience_per_ps: CCR_F64_UNMEASURED,
            tsh_memory_state: CCR_F64_UNMEASURED,
            tsh_request_confidence: CCR_F64_UNMEASURED,
            observation_confidence: CCR_F64_UNMEASURED,
            constraint_residual: CCR_F64_UNMEASURED,
            hamiltonian_drift: CCR_F64_UNMEASURED,
            force_norm: CCR_F64_UNMEASURED,
            schema_version: CCR_SCHEMA_VERSION,
            record_flags: 0,
            requested_gear: CCR_GEAR_UNMEASURED,
            authorized_gear: CCR_GEAR_UNMEASURED,
            device_measured_gear: CCR_GEAR_UNMEASURED,
            tsh_active_branches: CCR_U32_UNMEASURED,
            physical_guard_status: CCR_U32_UNMEASURED,
            transition_reason: CCR_REASON_NOT_RECORDED,
            hmr_verified: CCR_HMR_UNVERIFIED,
            stream_id: CCR_U32_UNMEASURED,
            replica_id: CCR_U32_UNMEASURED,
            emitter_tag: CCR_EMITTER_UNSPECIFIED,
            publication_fence: CCR_FENCE_DIRTY,
            _reserved_u32: 0,
            record_seq: CCR_U64_UNMEASURED,
            _reserved: [0; 8],
        }
    }

    /// Bind a spike record's EMITTER identity, plus the host-side chunk
    /// context, into a chronology record.
    ///
    /// PACKAGE C. This constructor used to copy `global_md_step` /
    /// `cumulative_simulated_time_ps` / `stream_id` / `chunk_id` out of a
    /// widened 128 B `GpuSpikeEvent`. That widening invalidated the sealed
    /// QNANL force-kernel packet and has been reverted: the spike record is
    /// back to 96 B and carries NO clock provenance at all. The clock bridge is
    /// this sidecar, sourced from the governor, not from the spike.
    ///
    /// What this constructor can still state honestly:
    ///
    /// * `emitter_tag` — read off the record's own `spike_source`.
    /// * `stream_id` / `record_seq` — supplied by the CALLER from the host-side
    ///   chunk loop that owns the drain. These are host facts about which
    ///   stream and which drain epoch produced the chunk; they are not
    ///   re-derivations of anything in the record.
    ///
    /// What it must NOT state: `global_md_step` and
    /// `cumulative_simulated_time_ps` stay at the CCR's own out-of-band
    /// sentinels. Nothing in a 96 B spike, and nothing available at the chunk
    /// boundary, establishes either one — so they are left UNMEASURED rather
    /// than filled with a plausible value. Gear, TSH and governor fields stay
    /// UNMEASURED for the same reason. A caller that needs the global clock on
    /// these records must merge them against the per-step governor stream
    /// stamped by [`GearboxClock`], which is the only measured source.
    ///
    /// `CCR_FLAG_CLOCK_STALLED` is deliberately NOT raised here. It means "the
    /// clock did not advance between two records" — an observation the governor
    /// path makes. Raising it from an *absent* clock would stamp a stall
    /// finding on 100 % of spike-bound records and destroy the flag's meaning.
    ///
    /// `stream_id` / `record_seq` arguments equal to [`CCR_U32_UNMEASURED`] /
    /// [`CCR_U64_UNMEASURED`] are passed through as unmeasured, so a caller
    /// without a chunk context can say so.
    #[cfg(feature = "gpu")]
    pub fn from_spike_event(
        spike: &crate::fused_engine::GpuSpikeEvent,
        stream_id: u32,
        record_seq: u64,
    ) -> Self {
        let mut r = Self::unmeasured();
        r.stream_id = stream_id;
        r.record_seq = record_seq;
        // A spike record carries one identity for its emitting engine: the
        // physics channel that produced it. The replica axis stays UNMEASURED
        // rather than being duplicated from the stream, because a duplicated
        // value would read as two independent witnesses agreeing.
        r.emitter_tag = spike_source_to_emitter_tag(spike.spike_source);
        r.publication_fence = CCR_FENCE_PUBLISHED;
        r
    }

    // ── typed accessors: `None` means NOT MEASURED ─────────────────────────

    /// Global MD step, or `None` when not established.
    pub fn global_md_step_opt(&self) -> Option<u64> {
        (self.global_md_step != CCR_U64_UNMEASURED).then_some(self.global_md_step)
    }

    /// Cumulative simulated time in ps, or `None` when not established.
    pub fn cumulative_simulated_time_ps_opt(&self) -> Option<f64> {
        (!ccr_f64_is_unmeasured(self.cumulative_simulated_time_ps))
            .then_some(self.cumulative_simulated_time_ps)
    }

    /// The `dt` ACTUALLY integrated, or `None` when not established.
    pub fn dt_applied_ps_opt(&self) -> Option<f64> {
        (!ccr_f64_is_unmeasured(self.dt_applied_ps)).then_some(self.dt_applied_ps)
    }

    /// Gear epoch, or `None` when not established.
    pub fn gear_epoch_id_opt(&self) -> Option<u64> {
        (self.gear_epoch_id != CCR_U64_UNMEASURED).then_some(self.gear_epoch_id)
    }

    /// Per-emitter record sequence, or `None` when not stamped.
    pub fn record_seq_opt(&self) -> Option<u64> {
        (self.record_seq != CCR_U64_UNMEASURED).then_some(self.record_seq)
    }

    /// Bound tensor snapshot id, or `None` when not established.
    pub fn tensor_snapshot_id_opt(&self) -> Option<u64> {
        (self.tensor_snapshot_id != CCR_U64_UNMEASURED).then_some(self.tensor_snapshot_id)
    }

    /// Gear the TSH arm demanded. `None` means no demand was recorded —
    /// NEVER gear 0.
    pub fn requested_gear_opt(&self) -> Option<u32> {
        (self.requested_gear != CCR_GEAR_UNMEASURED).then_some(self.requested_gear)
    }

    /// Gear the governor authorized. `None` means no authorization was
    /// recorded — NEVER gear 0.
    pub fn authorized_gear_opt(&self) -> Option<u32> {
        (self.authorized_gear != CCR_GEAR_UNMEASURED).then_some(self.authorized_gear)
    }

    /// Gear the DEVICE actually ran. `None` means no measurement reached the
    /// host — NEVER gear 0. This is the only gear field that states what the
    /// integrator held.
    pub fn device_measured_gear_opt(&self) -> Option<u32> {
        (self.device_measured_gear != CCR_GEAR_UNMEASURED).then_some(self.device_measured_gear)
    }

    /// TSH active-branch bitmask, or `None` when not measured. `Some(0)` is a
    /// legitimate measurement meaning "no branches active".
    pub fn tsh_active_branches_opt(&self) -> Option<u32> {
        (self.tsh_active_branches != CCR_U32_UNMEASURED).then_some(self.tsh_active_branches)
    }

    /// Guard bitmask, or `None` when not measured. `Some(0)` is a legitimate
    /// measurement meaning "every guard clear".
    pub fn physical_guard_status_opt(&self) -> Option<u32> {
        (self.physical_guard_status != CCR_U32_UNMEASURED).then_some(self.physical_guard_status)
    }

    /// Producing stream, or `None` when not attributed.
    pub fn stream_id_opt(&self) -> Option<u32> {
        (self.stream_id != CCR_U32_UNMEASURED).then_some(self.stream_id)
    }

    /// Producing replica, or `None` when not attributed.
    pub fn replica_id_opt(&self) -> Option<u32> {
        (self.replica_id != CCR_U32_UNMEASURED).then_some(self.replica_id)
    }

    /// Emitter tag, or `None` when unspecified.
    pub fn emitter_tag_opt(&self) -> Option<u32> {
        (self.emitter_tag != CCR_EMITTER_UNSPECIFIED).then_some(self.emitter_tag)
    }

    /// Transition reason, or `None` when not recorded.
    pub fn transition_reason_opt(&self) -> Option<u32> {
        (self.transition_reason != CCR_REASON_NOT_RECORDED).then_some(self.transition_reason)
    }

    /// TSH salience per SIMULATED picosecond, or `None` when not measured.
    pub fn tsh_integrated_salience_per_ps_opt(&self) -> Option<f64> {
        (!ccr_f64_is_unmeasured(self.tsh_integrated_salience_per_ps))
            .then_some(self.tsh_integrated_salience_per_ps)
    }

    /// TSH memory state, or `None` when not measured.
    pub fn tsh_memory_state_opt(&self) -> Option<f64> {
        (!ccr_f64_is_unmeasured(self.tsh_memory_state)).then_some(self.tsh_memory_state)
    }

    /// TSH request confidence, or `None` when not measured.
    pub fn tsh_request_confidence_opt(&self) -> Option<f64> {
        (!ccr_f64_is_unmeasured(self.tsh_request_confidence)).then_some(self.tsh_request_confidence)
    }

    /// Anti-blindness observation confidence, or `None` when not measured.
    pub fn observation_confidence_opt(&self) -> Option<f64> {
        (!ccr_f64_is_unmeasured(self.observation_confidence)).then_some(self.observation_confidence)
    }

    /// Constraint residual, or `None` when not measured.
    pub fn constraint_residual_opt(&self) -> Option<f64> {
        (!ccr_f64_is_unmeasured(self.constraint_residual)).then_some(self.constraint_residual)
    }

    /// Hamiltonian drift, or `None` when not measured.
    pub fn hamiltonian_drift_opt(&self) -> Option<f64> {
        (!ccr_f64_is_unmeasured(self.hamiltonian_drift)).then_some(self.hamiltonian_drift)
    }

    /// Force norm, or `None` when not measured. A `Some` value that is
    /// non-finite is a MEASURED non-physical state (the G29 trap class), not
    /// an absence.
    pub fn force_norm_opt(&self) -> Option<f64> {
        (!ccr_f64_is_unmeasured(self.force_norm)).then_some(self.force_norm)
    }

    /// HMR topology-permission tri-state.
    pub fn hmr_verification(&self) -> HmrVerification {
        HmrVerification::from_u32(self.hmr_verified)
    }

    /// True iff the producer raised the publication fence.
    pub fn is_published(&self) -> bool {
        self.publication_fence == CCR_FENCE_PUBLISHED
    }

    /// True iff `flag` is set in `record_flags`.
    pub fn has_flag(&self, flag: u32) -> bool {
        self.record_flags & flag != 0
    }

    /// Highest-priority asserting guard, in LEXICOGRAPHIC order, or `None`
    /// when every guard is clear or the mask was not measured.
    ///
    /// The order is fixed and priority-ordered on purpose: a weighted score
    /// always admits a salience value large enough to outvote a physical
    /// prohibition, so the governor's verdict is read lexicographically.
    pub fn highest_priority_guard(&self) -> Option<&'static str> {
        let mask = self.physical_guard_status_opt()?;
        CCR_GUARD_PRIORITY_ORDER
            .iter()
            .find(|(bit, _)| mask & bit != 0)
            .map(|(_, name)| *name)
    }

    /// Every field for which no measurement exists, as stable labels. Empty
    /// means the record is fully populated.
    pub fn unmeasured_fields(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.global_md_step_opt().is_none() {
            out.push("global_md_step");
        }
        if self.cumulative_simulated_time_ps_opt().is_none() {
            out.push("cumulative_simulated_time_ps");
        }
        if self.dt_applied_ps_opt().is_none() {
            out.push("dt_applied_ps");
        }
        if self.gear_epoch_id_opt().is_none() {
            out.push("gear_epoch_id");
        }
        if self.tensor_snapshot_id_opt().is_none() {
            out.push("tensor_snapshot_id");
        }
        if self.record_seq_opt().is_none() {
            out.push("record_seq");
        }
        if self.tsh_integrated_salience_per_ps_opt().is_none() {
            out.push("tsh_integrated_salience_per_ps");
        }
        if self.tsh_memory_state_opt().is_none() {
            out.push("tsh_memory_state");
        }
        if self.tsh_request_confidence_opt().is_none() {
            out.push("tsh_request_confidence");
        }
        if self.observation_confidence_opt().is_none() {
            out.push("observation_confidence");
        }
        if self.constraint_residual_opt().is_none() {
            out.push("constraint_residual");
        }
        if self.hamiltonian_drift_opt().is_none() {
            out.push("hamiltonian_drift");
        }
        if self.force_norm_opt().is_none() {
            out.push("force_norm");
        }
        if self.requested_gear_opt().is_none() {
            out.push("requested_gear");
        }
        if self.authorized_gear_opt().is_none() {
            out.push("authorized_gear");
        }
        if self.device_measured_gear_opt().is_none() {
            out.push("device_measured_gear");
        }
        if self.tsh_active_branches_opt().is_none() {
            out.push("tsh_active_branches");
        }
        if self.physical_guard_status_opt().is_none() {
            out.push("physical_guard_status");
        }
        if self.transition_reason_opt().is_none() {
            out.push("transition_reason");
        }
        if !self.hmr_verification().is_established() {
            out.push("hmr_verified");
        }
        if self.stream_id_opt().is_none() {
            out.push("stream_id");
        }
        if self.replica_id_opt().is_none() {
            out.push("replica_id");
        }
        if self.emitter_tag_opt().is_none() {
            out.push("emitter_tag");
        }
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §4  Host-side decode / encode
// ═══════════════════════════════════════════════════════════════════════════

/// Why a byte buffer could not be decoded into a record.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CcrDecodeError {
    /// Buffer shorter than [`CCR_RECORD_BYTES`].
    ShortBuffer { have: usize, need: usize },
    /// Buffer length is not a whole multiple of [`CCR_RECORD_BYTES`].
    RaggedBuffer { have: usize, stride: usize },
    /// `schema_version` is not [`CCR_SCHEMA_VERSION`].
    ///
    /// This is a HARD failure on purpose: a reader that guesses at an unknown
    /// schema misparses silently, which is the exact failure class this
    /// record exists to eliminate.
    SchemaVersionMismatch { found: u32, expected: u32 },
}

impl std::fmt::Display for CcrDecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ShortBuffer { have, need } => {
                write!(f, "CCR short buffer: have {have} bytes, need {need}")
            }
            Self::RaggedBuffer { have, stride } => {
                write!(
                    f,
                    "CCR ragged buffer: {have} bytes is not a multiple of {stride}"
                )
            }
            Self::SchemaVersionMismatch { found, expected } => {
                write!(f, "CCR schema_version {found} != expected {expected}")
            }
        }
    }
}

impl std::error::Error for CcrDecodeError {}

#[inline]
fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

#[inline]
fn rd_u64(b: &[u8], off: usize) -> u64 {
    let mut t = [0u8; 8];
    t.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(t)
}

#[inline]
fn wr_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

#[inline]
fn wr_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

impl CoupledChronologyRecord {
    /// Encode to the frozen 192-byte little-endian wire layout.
    ///
    /// `f64` fields are written as raw IEEE-754 bits (`to_bits`), so the
    /// unmeasured NaN payload survives the round trip intact. A
    /// float-formatting round trip would not preserve it, and losing the
    /// payload would collapse "not measured" into "measured as NaN".
    pub fn to_bytes(&self) -> [u8; CCR_RECORD_BYTES] {
        let mut b = [0u8; CCR_RECORD_BYTES];
        wr_u64(&mut b, CCR_OFF_GLOBAL_MD_STEP, self.global_md_step);
        wr_u64(
            &mut b,
            CCR_OFF_CUM_SIM_TIME_PS,
            self.cumulative_simulated_time_ps.to_bits(),
        );
        wr_u64(&mut b, CCR_OFF_DT_APPLIED_PS, self.dt_applied_ps.to_bits());
        wr_u64(&mut b, CCR_OFF_GEAR_EPOCH_ID, self.gear_epoch_id);
        wr_u64(&mut b, CCR_OFF_TENSOR_SNAPSHOT_ID, self.tensor_snapshot_id);
        wr_u64(
            &mut b,
            CCR_OFF_TSH_SALIENCE_PER_PS,
            self.tsh_integrated_salience_per_ps.to_bits(),
        );
        wr_u64(
            &mut b,
            CCR_OFF_TSH_MEMORY_STATE,
            self.tsh_memory_state.to_bits(),
        );
        wr_u64(
            &mut b,
            CCR_OFF_TSH_REQ_CONFIDENCE,
            self.tsh_request_confidence.to_bits(),
        );
        wr_u64(
            &mut b,
            CCR_OFF_OBSERVATION_CONF,
            self.observation_confidence.to_bits(),
        );
        wr_u64(
            &mut b,
            CCR_OFF_CONSTRAINT_RESIDUAL,
            self.constraint_residual.to_bits(),
        );
        wr_u64(
            &mut b,
            CCR_OFF_HAMILTONIAN_DRIFT,
            self.hamiltonian_drift.to_bits(),
        );
        wr_u64(&mut b, CCR_OFF_FORCE_NORM, self.force_norm.to_bits());
        wr_u32(&mut b, CCR_OFF_SCHEMA_VERSION, self.schema_version);
        wr_u32(&mut b, CCR_OFF_RECORD_FLAGS, self.record_flags);
        wr_u32(&mut b, CCR_OFF_REQUESTED_GEAR, self.requested_gear);
        wr_u32(&mut b, CCR_OFF_AUTHORIZED_GEAR, self.authorized_gear);
        wr_u32(
            &mut b,
            CCR_OFF_DEVICE_MEASURED_GEAR,
            self.device_measured_gear,
        );
        wr_u32(
            &mut b,
            CCR_OFF_TSH_ACTIVE_BRANCHES,
            self.tsh_active_branches,
        );
        wr_u32(
            &mut b,
            CCR_OFF_PHYSICAL_GUARD_STATUS,
            self.physical_guard_status,
        );
        wr_u32(&mut b, CCR_OFF_TRANSITION_REASON, self.transition_reason);
        wr_u32(&mut b, CCR_OFF_HMR_VERIFIED, self.hmr_verified);
        wr_u32(&mut b, CCR_OFF_STREAM_ID, self.stream_id);
        wr_u32(&mut b, CCR_OFF_REPLICA_ID, self.replica_id);
        wr_u32(&mut b, CCR_OFF_EMITTER_TAG, self.emitter_tag);
        wr_u32(&mut b, CCR_OFF_PUBLICATION_FENCE, self.publication_fence);
        wr_u32(&mut b, CCR_OFF_RESERVED_U32, self._reserved_u32);
        wr_u64(&mut b, CCR_OFF_RECORD_SEQ, self.record_seq);
        for (i, v) in self._reserved.iter().enumerate() {
            wr_u32(&mut b, CCR_OFF_RESERVED + i * 4, *v);
        }
        b
    }

    /// Decode one record from the frozen wire layout.
    ///
    /// Rejects an unknown `schema_version` rather than guessing.
    pub fn from_bytes(b: &[u8]) -> Result<Self, CcrDecodeError> {
        if b.len() < CCR_RECORD_BYTES {
            return Err(CcrDecodeError::ShortBuffer {
                have: b.len(),
                need: CCR_RECORD_BYTES,
            });
        }
        let schema_version = rd_u32(b, CCR_OFF_SCHEMA_VERSION);
        if schema_version != CCR_SCHEMA_VERSION {
            return Err(CcrDecodeError::SchemaVersionMismatch {
                found: schema_version,
                expected: CCR_SCHEMA_VERSION,
            });
        }
        let mut reserved = [0u32; 8];
        for (i, slot) in reserved.iter_mut().enumerate() {
            *slot = rd_u32(b, CCR_OFF_RESERVED + i * 4);
        }
        Ok(Self {
            global_md_step: rd_u64(b, CCR_OFF_GLOBAL_MD_STEP),
            cumulative_simulated_time_ps: f64::from_bits(rd_u64(b, CCR_OFF_CUM_SIM_TIME_PS)),
            dt_applied_ps: f64::from_bits(rd_u64(b, CCR_OFF_DT_APPLIED_PS)),
            gear_epoch_id: rd_u64(b, CCR_OFF_GEAR_EPOCH_ID),
            tensor_snapshot_id: rd_u64(b, CCR_OFF_TENSOR_SNAPSHOT_ID),
            tsh_integrated_salience_per_ps: f64::from_bits(rd_u64(b, CCR_OFF_TSH_SALIENCE_PER_PS)),
            tsh_memory_state: f64::from_bits(rd_u64(b, CCR_OFF_TSH_MEMORY_STATE)),
            tsh_request_confidence: f64::from_bits(rd_u64(b, CCR_OFF_TSH_REQ_CONFIDENCE)),
            observation_confidence: f64::from_bits(rd_u64(b, CCR_OFF_OBSERVATION_CONF)),
            constraint_residual: f64::from_bits(rd_u64(b, CCR_OFF_CONSTRAINT_RESIDUAL)),
            hamiltonian_drift: f64::from_bits(rd_u64(b, CCR_OFF_HAMILTONIAN_DRIFT)),
            force_norm: f64::from_bits(rd_u64(b, CCR_OFF_FORCE_NORM)),
            schema_version,
            record_flags: rd_u32(b, CCR_OFF_RECORD_FLAGS),
            requested_gear: rd_u32(b, CCR_OFF_REQUESTED_GEAR),
            authorized_gear: rd_u32(b, CCR_OFF_AUTHORIZED_GEAR),
            device_measured_gear: rd_u32(b, CCR_OFF_DEVICE_MEASURED_GEAR),
            tsh_active_branches: rd_u32(b, CCR_OFF_TSH_ACTIVE_BRANCHES),
            physical_guard_status: rd_u32(b, CCR_OFF_PHYSICAL_GUARD_STATUS),
            transition_reason: rd_u32(b, CCR_OFF_TRANSITION_REASON),
            hmr_verified: rd_u32(b, CCR_OFF_HMR_VERIFIED),
            stream_id: rd_u32(b, CCR_OFF_STREAM_ID),
            replica_id: rd_u32(b, CCR_OFF_REPLICA_ID),
            emitter_tag: rd_u32(b, CCR_OFF_EMITTER_TAG),
            publication_fence: rd_u32(b, CCR_OFF_PUBLICATION_FENCE),
            _reserved_u32: rd_u32(b, CCR_OFF_RESERVED_U32),
            record_seq: rd_u64(b, CCR_OFF_RECORD_SEQ),
            _reserved: reserved,
        })
    }

    /// Decode a packed stream of records (a drained device-ring region or an
    /// on-disk segment).
    ///
    /// A ragged tail is an ERROR, not a truncation to ignore: a partial
    /// record means the stream is not a complete account of the run.
    pub fn decode_stream(b: &[u8]) -> Result<Vec<Self>, CcrDecodeError> {
        if b.len() % CCR_RECORD_BYTES != 0 {
            return Err(CcrDecodeError::RaggedBuffer {
                have: b.len(),
                stride: CCR_RECORD_BYTES,
            });
        }
        let mut out = Vec::with_capacity(b.len() / CCR_RECORD_BYTES);
        for chunk in b.chunks_exact(CCR_RECORD_BYTES) {
            out.push(Self::from_bytes(chunk)?);
        }
        Ok(out)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §5  Policy-resolved thresholds
// ═══════════════════════════════════════════════════════════════════════════

/// A threshold together with the value it resolved to and where that value
/// came from.
///
/// NO hardcoded scientific thresholds. Every threshold is policy-resolved,
/// the source literal is kept as the documented default, and BOTH the
/// resolved value and its origin are recorded on the report that used it —
/// so a later reader can reproduce the decision without reading source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResolvedThreshold {
    /// Stable name of the threshold.
    pub name: String,
    /// Effective value used.
    pub value: f64,
    /// The in-source default, retained verbatim even when overridden.
    pub source_literal: f64,
    /// Environment key consulted for an override.
    pub env_key: String,
    /// Stable label naming where `value` came from.
    pub origin: String,
}

impl ResolvedThreshold {
    /// Resolve from an already-read raw override string.
    ///
    /// Split out from [`Self::resolve`] so the resolution rule is testable
    /// without mutating process environment (which races under a parallel
    /// test harness).
    pub fn from_raw(
        name: &'static str,
        env_key: &'static str,
        source_literal: f64,
        raw: Option<&str>,
    ) -> Self {
        match raw.map(str::trim) {
            None => Self {
                name: name.to_string(),
                value: source_literal,
                source_literal,
                env_key: env_key.to_string(),
                origin: "source_literal_default".to_string(),
            },
            Some(text) => match text.parse::<f64>() {
                Ok(v) if v.is_finite() && v > 0.0 => Self {
                    name: name.to_string(),
                    value: v,
                    source_literal,
                    env_key: env_key.to_string(),
                    origin: format!("env:{env_key}"),
                },
                Ok(v) => Self {
                    name: name.to_string(),
                    value: source_literal,
                    source_literal,
                    env_key: env_key.to_string(),
                    origin: format!("env:{env_key}=rejected_non_positive_or_non_finite({v})"),
                },
                Err(_) => Self {
                    name: name.to_string(),
                    value: source_literal,
                    source_literal,
                    env_key: env_key.to_string(),
                    origin: format!("env:{env_key}=unparseable_fell_back_to_source_literal"),
                },
            },
        }
    }

    /// Resolve, consulting the process environment.
    pub fn resolve(name: &'static str, env_key: &'static str, source_literal: f64) -> Self {
        let raw = std::env::var(env_key).ok();
        Self::from_raw(name, env_key, source_literal, raw.as_deref())
    }

    /// True when the effective value is the in-source default.
    pub fn is_source_literal(&self) -> bool {
        self.origin == "source_literal_default"
    }
}

/// Environment key for the dt/gear consistency tolerance.
pub const CCR_ENV_DT_GEAR_REL_TOL: &str = "PRISM_CCR_DT_GEAR_REL_TOL";
/// Environment key for the physical-time continuity tolerance.
pub const CCR_ENV_TIME_CONTINUITY_REL_TOL: &str = "PRISM_CCR_TIME_CONTINUITY_REL_TOL";

/// Source literal: relative tolerance when checking a record's
/// `dt_applied_ps` against the gear table entry for its
/// `device_measured_gear`.
///
/// The device gear table is `float` (`__constant__ float d_gearbox_table`),
/// so a value that round-trips through `f32` can differ from the exact `f64`
/// decimal by ~1e-7 relative. `1e-6` clears that representation gap without
/// admitting an actual gear/dt disagreement, whose smallest possible
/// magnitude is the 2x ratio between adjacent gears.
pub const CCR_DT_GEAR_REL_TOL_SOURCE_LITERAL: f64 = 1e-6;

/// Source literal: relative tolerance for the physical-time continuity
/// identity `cum[i] - cum[i-1] == dt_applied[i] * (step[i] - step[i-1])`.
///
/// The identity is accumulated in `f64`, so the error budget is dominated by
/// summation rounding over the record count. `1e-9` relative is ~6 orders of
/// magnitude above `f64` epsilon while still far below any real lost-record
/// discontinuity, whose smallest magnitude is one applied timestep.
pub const CCR_TIME_CONTINUITY_REL_TOL_SOURCE_LITERAL: f64 = 1e-9;

/// The resolved threshold set used by a validation or conservation pass.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoupledChronologyPolicy {
    /// Tolerance for `dt_applied_ps` vs the gear table.
    pub dt_gear_rel_tol: ResolvedThreshold,
    /// Tolerance for the physical-time continuity identity.
    pub time_continuity_rel_tol: ResolvedThreshold,
}

impl CoupledChronologyPolicy {
    /// Resolve every threshold from policy, recording each value and origin.
    pub fn resolve() -> Self {
        Self {
            dt_gear_rel_tol: ResolvedThreshold::resolve(
                "dt_gear_rel_tol",
                CCR_ENV_DT_GEAR_REL_TOL,
                CCR_DT_GEAR_REL_TOL_SOURCE_LITERAL,
            ),
            time_continuity_rel_tol: ResolvedThreshold::resolve(
                "time_continuity_rel_tol",
                CCR_ENV_TIME_CONTINUITY_REL_TOL,
                CCR_TIME_CONTINUITY_REL_TOL_SOURCE_LITERAL,
            ),
        }
    }

    /// Resolve using source literals only, consulting no environment.
    pub fn source_literals() -> Self {
        Self {
            dt_gear_rel_tol: ResolvedThreshold::from_raw(
                "dt_gear_rel_tol",
                CCR_ENV_DT_GEAR_REL_TOL,
                CCR_DT_GEAR_REL_TOL_SOURCE_LITERAL,
                None,
            ),
            time_continuity_rel_tol: ResolvedThreshold::from_raw(
                "time_continuity_rel_tol",
                CCR_ENV_TIME_CONTINUITY_REL_TOL,
                CCR_TIME_CONTINUITY_REL_TOL_SOURCE_LITERAL,
                None,
            ),
        }
    }
}

impl Default for CoupledChronologyPolicy {
    fn default() -> Self {
        Self::source_literals()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §6  Validation
// ═══════════════════════════════════════════════════════════════════════════

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CcrSeverity {
    /// The record or stream is not usable as scientific evidence as-is.
    Error,
    /// Usable, but the condition must be visible in the receipt.
    Warning,
}

/// One validation finding, anchored to the record that produced it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CcrFinding {
    /// Stable machine-readable code.
    pub code: String,
    /// Severity.
    pub severity: CcrSeverity,
    /// Index of the offending record within the validated stream.
    pub record_index: usize,
    /// `global_md_step` of the offending record, when established.
    pub global_md_step: Option<u64>,
    /// Human-readable detail. Never the sole carrier of a fact — the code
    /// and the indices above are what a consumer branches on.
    pub detail: String,
}

impl CcrFinding {
    fn new(
        code: &'static str,
        severity: CcrSeverity,
        record_index: usize,
        rec: &CoupledChronologyRecord,
        detail: String,
    ) -> Self {
        Self {
            code: code.to_string(),
            severity,
            record_index,
            global_md_step: rec.global_md_step_opt(),
            detail,
        }
    }
}

/// Result of validating a record or a record stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoupledChronologyValidation {
    /// Thresholds actually used, with their resolved values and origins.
    pub policy: CoupledChronologyPolicy,
    /// Records examined.
    pub records_examined: u64,
    /// Findings, in discovery order.
    pub findings: Vec<CcrFinding>,
}

impl CoupledChronologyValidation {
    /// True iff no `Error`-severity finding was raised.
    pub fn is_pass(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.severity == CcrSeverity::Error)
    }

    /// Count of `Error`-severity findings.
    pub fn error_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == CcrSeverity::Error)
            .count()
    }

    /// Count of `Warning`-severity findings.
    pub fn warning_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == CcrSeverity::Warning)
            .count()
    }

    /// True iff a finding with `code` is present.
    pub fn has(&self, code: &str) -> bool {
        self.findings.iter().any(|f| f.code == code)
    }
}

fn rel_diff(a: f64, b: f64) -> f64 {
    let scale = a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
    (a - b).abs() / scale
}

/// Validate a single record against the record-local invariants.
///
/// Record-local means: everything checkable without a neighbour. Sequence
/// invariants (monotonicity, gear-epoch increments, physical-time
/// continuity) live in [`validate_stream`].
pub fn validate_record(
    rec: &CoupledChronologyRecord,
    index: usize,
    policy: &CoupledChronologyPolicy,
    out: &mut Vec<CcrFinding>,
) {
    use CcrSeverity::{Error, Warning};

    if rec.schema_version != CCR_SCHEMA_VERSION {
        out.push(CcrFinding::new(
            "CCR_SCHEMA_VERSION_MISMATCH",
            Error,
            index,
            rec,
            format!(
                "schema_version {} != expected {}",
                rec.schema_version, CCR_SCHEMA_VERSION
            ),
        ));
    }

    if !rec.is_published() {
        out.push(CcrFinding::new(
            "CCR_NOT_PUBLISHED",
            Error,
            index,
            rec,
            "publication_fence is not PUBLISHED — the producer had not \
             finished writing this slot"
                .to_string(),
        ));
    }

    // ── provenance completeness ────────────────────────────────────────────
    if rec.global_md_step_opt().is_none() {
        out.push(CcrFinding::new(
            "CCR_GLOBAL_STEP_UNMEASURED",
            Error,
            index,
            rec,
            "global_md_step is unmeasured — the record cannot be placed on \
             any clock"
                .to_string(),
        ));
    }
    if rec.record_seq_opt().is_none() {
        out.push(CcrFinding::new(
            "CCR_RECORD_SEQ_UNSTAMPED",
            Error,
            index,
            rec,
            "record_seq is unstamped — record loss is undetectable while the \
             gear is held, because the physical-time continuity identity \
             stays satisfied across a drop at constant dt"
                .to_string(),
        ));
    }
    if rec.stream_id_opt().is_none() {
        out.push(CcrFinding::new(
            "CCR_STREAM_UNATTRIBUTED",
            Error,
            index,
            rec,
            "stream_id is unmeasured — attribution is lost the instant \
             streams merge in memory"
                .to_string(),
        ));
    }
    if rec.replica_id_opt().is_none() {
        out.push(CcrFinding::new(
            "CCR_REPLICA_UNATTRIBUTED",
            Error,
            index,
            rec,
            "replica_id is unmeasured — replica timesteps collide without it".to_string(),
        ));
    }
    if rec.emitter_tag_opt().is_none() {
        out.push(CcrFinding::new(
            "CCR_EMITTER_UNSPECIFIED",
            Error,
            index,
            rec,
            "emitter_tag is unspecified — UV/LIF/EFP and LADD carry different \
             clocks in the same file"
                .to_string(),
        ));
    } else if rec.emitter_tag > CCR_EMITTER_MAX {
        out.push(CcrFinding::new(
            "CCR_EMITTER_OUT_OF_RANGE",
            Error,
            index,
            rec,
            format!("emitter_tag {} > max {}", rec.emitter_tag, CCR_EMITTER_MAX),
        ));
    }

    // ── gear range ─────────────────────────────────────────────────────────
    for (label, raw, opt) in [
        (
            "requested_gear",
            rec.requested_gear,
            rec.requested_gear_opt(),
        ),
        (
            "authorized_gear",
            rec.authorized_gear,
            rec.authorized_gear_opt(),
        ),
        (
            "device_measured_gear",
            rec.device_measured_gear,
            rec.device_measured_gear_opt(),
        ),
    ] {
        if let Some(g) = opt {
            if g > CCR_GEAR_MAX {
                out.push(CcrFinding::new(
                    "CCR_GEAR_OUT_OF_RANGE",
                    Error,
                    index,
                    rec,
                    format!("{label} = {raw} is neither a legal gear (0..={CCR_GEAR_MAX}) nor the unmeasured sentinel"),
                ));
            }
        }
    }
    if rec.authorized_gear_opt().is_none() {
        out.push(CcrFinding::new(
            "CCR_AUTHORIZED_GEAR_UNMEASURED",
            Error,
            index,
            rec,
            "authorized_gear is unmeasured — the governor's verdict is the \
             record's reason for existing"
                .to_string(),
        ));
    }

    // ── role separation: authorized = min(demanded, permitted) ─────────────
    //
    // Lower gear index == finer resolution == safer. The governor may only
    // make the gear FINER than or equal to what TSH demanded; it can never
    // grant coarser than demanded. Gear 3 is the abort trap and is exempt:
    // it is a physical emergency, not a resolution choice.
    if let (Some(req), Some(auth)) = (rec.requested_gear_opt(), rec.authorized_gear_opt()) {
        if auth != CCR_GEAR_ABORT_TRAP && req != CCR_GEAR_ABORT_TRAP && auth > req {
            out.push(CcrFinding::new(
                "CCR_AUTHORIZED_COARSER_THAN_REQUESTED",
                Error,
                index,
                rec,
                format!(
                    "authorized_gear {auth} is COARSER than requested_gear {req}; \
                     the gearbox must apply min(demanded, permitted)"
                ),
            ));
        }
    }

    // ── HMR topology permission gates the 4 fs gear ────────────────────────
    if rec.authorized_gear_opt() == Some(CCR_GEAR_HMR_4FS)
        && rec.hmr_verification() != HmrVerification::Applied
    {
        out.push(CcrFinding::new(
            "CCR_HMR_GEAR_WITHOUT_VERIFIED_PERMISSION",
            Error,
            index,
            rec,
            format!(
                "authorized_gear = {CCR_GEAR_HMR_4FS} (4 fs, HMR) but \
                 hmr_verified = {:?}; the 4 fs gear requires verified \
                 topology.hmr_applied",
                rec.hmr_verification()
            ),
        ));
    }

    // ── decision-record integrity: EMIT ON NO-CHANGE TOO ───────────────────
    if !rec.has_flag(CCR_FLAG_DECISION_EXECUTED) {
        out.push(CcrFinding::new(
            "CCR_DECISION_FLAG_ABSENT",
            Error,
            index,
            rec,
            "CCR_FLAG_DECISION_EXECUTED is not set — a record must never be \
             ambiguous between 'gear held' and 'decision did not run'"
                .to_string(),
        ));
    }
    let held = rec.has_flag(CCR_FLAG_GEAR_HELD);
    let changed = rec.has_flag(CCR_FLAG_GEAR_CHANGED);
    if held == changed {
        out.push(CcrFinding::new(
            "CCR_GEAR_CHANGE_FLAG_AMBIGUOUS",
            Error,
            index,
            rec,
            format!(
                "exactly one of GEAR_HELD / GEAR_CHANGED must be set \
                 (held={held}, changed={changed})"
            ),
        ));
    }
    if rec.transition_reason_opt().is_none() {
        out.push(CcrFinding::new(
            "CCR_TRANSITION_REASON_NOT_RECORDED",
            Error,
            index,
            rec,
            "transition_reason is NOT_RECORDED".to_string(),
        ));
    } else if rec.transition_reason > CCR_REASON_MAX {
        out.push(CcrFinding::new(
            "CCR_TRANSITION_REASON_OUT_OF_RANGE",
            Error,
            index,
            rec,
            format!(
                "transition_reason {} > max {}",
                rec.transition_reason, CCR_REASON_MAX
            ),
        ));
    }

    // ── governor authority: coarsening requires a clear verdict ────────────
    if rec.transition_reason == CCR_REASON_UPSHIFT_GRANTED {
        match rec.physical_guard_status_opt() {
            Some(0) => {}
            Some(mask) => out.push(CcrFinding::new(
                "CCR_UPSHIFT_WITH_GUARD_ASSERTED",
                Error,
                index,
                rec,
                format!(
                    "coarser resolution granted while guard mask 0x{mask:08X} is \
                     asserting (highest priority: {}); coarsening requires low \
                     salience AND verified physical safety",
                    rec.highest_priority_guard().unwrap_or("unknown")
                ),
            )),
            None => out.push(CcrFinding::new(
                "CCR_UPSHIFT_WITH_UNMEASURED_GUARDS",
                Error,
                index,
                rec,
                "coarser resolution granted while physical_guard_status is \
                 unmeasured; safety must be VERIFIED, not merely unobserved"
                    .to_string(),
            )),
        }
        if rec.has_flag(CCR_FLAG_OBSERVATION_STARVED) {
            out.push(CcrFinding::new(
                "CCR_UPSHIFT_WHILE_OBSERVATION_STARVED",
                Error,
                index,
                rec,
                "coarser resolution granted while OBSERVATION_STARVED is set; \
                 anti-blindness forbids coarsening below the observation \
                 confidence floor"
                    .to_string(),
            ));
        }
    }

    // ── abort trap consistency ─────────────────────────────────────────────
    let abort_gear = rec.authorized_gear_opt() == Some(CCR_GEAR_ABORT_TRAP);
    let abort_flag = rec.has_flag(CCR_FLAG_ABORT_TRAP);
    if abort_gear != abort_flag {
        out.push(CcrFinding::new(
            "CCR_ABORT_TRAP_INCONSISTENT",
            Error,
            index,
            rec,
            format!(
                "authorized_gear==abort is {abort_gear} but ABORT_TRAP flag is \
                 {abort_flag}"
            ),
        ));
    }

    // ── timestep sanity ────────────────────────────────────────────────────
    match rec.dt_applied_ps_opt() {
        Some(dt) if !dt.is_finite() || dt <= 0.0 => out.push(CcrFinding::new(
            "CCR_DT_APPLIED_NON_PHYSICAL",
            Error,
            index,
            rec,
            format!("dt_applied_ps = {dt} is not a positive finite timestep"),
        )),
        Some(dt) => {
            // Cross-check against the gear that actually ran, when both are
            // measured. A disagreement means the record's own units are
            // internally inconsistent.
            if let Some(g) = rec.device_measured_gear_opt() {
                if let Some(Some(table_dt)) = CCR_GEAR_DT_PS_SOURCE_LITERAL.get(g as usize).copied()
                {
                    let d = rel_diff(dt, table_dt);
                    if d > policy.dt_gear_rel_tol.value {
                        out.push(CcrFinding::new(
                            "CCR_DT_GEAR_INCONSISTENT",
                            Error,
                            index,
                            rec,
                            format!(
                                "dt_applied_ps {dt} disagrees with gear {g} table dt \
                                 {table_dt} (rel diff {d:.3e} > tol {:.3e}, origin {})",
                                policy.dt_gear_rel_tol.value, policy.dt_gear_rel_tol.origin
                            ),
                        ));
                    }
                } else if g == CCR_GEAR_ABORT_TRAP {
                    out.push(CcrFinding::new(
                        "CCR_DT_APPLIED_UNDER_ABORT_GEAR",
                        Error,
                        index,
                        rec,
                        format!(
                            "device_measured_gear is the abort trap ({CCR_GEAR_ABORT_TRAP}), \
                             which has NO timestep, yet dt_applied_ps = {dt}"
                        ),
                    ));
                }
            }
        }
        None => {
            // Not measured. The clock MUST have stalled: advancing physical
            // time by an unmeasured dt would fabricate elapsed physics.
            if !rec.has_flag(CCR_FLAG_CLOCK_STALLED) {
                out.push(CcrFinding::new(
                    "CCR_CLOCK_ADVANCED_ON_UNMEASURED_DT",
                    Error,
                    index,
                    rec,
                    "dt_applied_ps is unmeasured but CLOCK_STALLED is not set — \
                     cumulative physical time cannot advance on an unmeasured \
                     timestep"
                        .to_string(),
                ));
            }
        }
    }

    if let Some(t) = rec.cumulative_simulated_time_ps_opt() {
        if !t.is_finite() || t < 0.0 {
            out.push(CcrFinding::new(
                "CCR_CUMULATIVE_TIME_NON_PHYSICAL",
                Error,
                index,
                rec,
                format!("cumulative_simulated_time_ps = {t} is not finite and non-negative"),
            ));
        }
    } else {
        out.push(CcrFinding::new(
            "CCR_CUMULATIVE_TIME_UNMEASURED",
            Error,
            index,
            rec,
            "cumulative_simulated_time_ps is unmeasured — mandatory once the \
             gear varies"
                .to_string(),
        ));
    }

    // ── salience normalisation cannot be checked from one record, but its
    //    absence can. ────────────────────────────────────────────────────────
    if rec.tsh_integrated_salience_per_ps_opt().is_none()
        && rec.transition_reason_opt().is_some()
        && matches!(
            rec.transition_reason,
            r if r == CCR_REASON_DOWNSHIFT_SALIENCE || r == CCR_REASON_UPSHIFT_GRANTED
        )
    {
        out.push(CcrFinding::new(
            "CCR_SALIENCE_DRIVEN_DECISION_WITHOUT_SALIENCE",
            Error,
            index,
            rec,
            "the transition reason cites salience but \
             tsh_integrated_salience_per_ps is unmeasured"
                .to_string(),
        ));
    }

    // ── reserved tail ──────────────────────────────────────────────────────
    if rec._reserved_u32 != 0 || rec._reserved.iter().any(|v| *v != 0) {
        out.push(CcrFinding::new(
            "CCR_RESERVED_TAIL_NONZERO",
            Warning,
            index,
            rec,
            "reserved tail is non-zero; a future schema may already be in use".to_string(),
        ));
    }
}

/// Validate a whole record stream: record-local invariants plus the sequence
/// invariants that only exist between neighbours.
///
/// The stream is assumed to be in producer order for one (`stream_id`,
/// `replica_id`, `emitter_tag`) triple. Mixing emitters in one call is a
/// caller error and is reported, not silently tolerated, because UV/LIF/EFP
/// and LADD carry different clocks.
pub fn validate_stream(
    records: &[CoupledChronologyRecord],
    policy: &CoupledChronologyPolicy,
) -> CoupledChronologyValidation {
    use CcrSeverity::Error;
    let mut findings: Vec<CcrFinding> = Vec::new();

    for (i, rec) in records.iter().enumerate() {
        validate_record(rec, i, policy, &mut findings);
    }

    // Homogeneity: one clock per validated stream.
    if let Some(first) = records.first() {
        for (i, rec) in records.iter().enumerate().skip(1) {
            if rec.stream_id != first.stream_id
                || rec.replica_id != first.replica_id
                || rec.emitter_tag != first.emitter_tag
            {
                findings.push(CcrFinding::new(
                    "CCR_STREAM_NOT_HOMOGENEOUS",
                    Error,
                    i,
                    rec,
                    format!(
                        "record carries (stream {}, replica {}, emitter {}) but the \
                         stream opened with (stream {}, replica {}, emitter {}); \
                         different emitters carry different clocks",
                        rec.stream_id,
                        rec.replica_id,
                        rec.emitter_tag,
                        first.stream_id,
                        first.replica_id,
                        first.emitter_tag
                    ),
                ));
                break;
            }
        }
    }

    for i in 1..records.len() {
        let prev = &records[i - 1];
        let cur = &records[i];

        // ── step monotonicity ──────────────────────────────────────────────
        let steps = match (prev.global_md_step_opt(), cur.global_md_step_opt()) {
            (Some(a), Some(b)) => {
                if b <= a {
                    findings.push(CcrFinding::new(
                        "CCR_STEP_NOT_MONOTONIC",
                        Error,
                        i,
                        cur,
                        format!("global_md_step {b} does not advance past {a}"),
                    ));
                    None
                } else {
                    Some(b - a)
                }
            }
            _ => None,
        };

        // ── record sequence completeness ───────────────────────────────────
        //
        // Separate from physical-time continuity on purpose: continuity is a
        // physics check that a constant-dt drop passes silently.
        match (prev.record_seq_opt(), cur.record_seq_opt()) {
            (Some(a), Some(b)) if b == a + 1 => {}
            (Some(a), Some(b)) if b <= a => findings.push(CcrFinding::new(
                "CCR_RECORD_SEQ_NOT_MONOTONIC",
                Error,
                i,
                cur,
                format!("record_seq {b} does not advance past {a}"),
            )),
            (Some(a), Some(b)) => findings.push(CcrFinding::new(
                "CCR_RECORD_SEQ_GAP",
                Error,
                i,
                cur,
                format!(
                    "record_seq jumped {a} -> {b}: {} record(s) between them \
                     never reached this stream",
                    b - a - 1
                ),
            )),
            _ => {}
        }

        // ── gear epoch monotonicity + transition binding ────────────────────
        match (prev.gear_epoch_id_opt(), cur.gear_epoch_id_opt()) {
            (Some(a), Some(b)) => {
                if b < a {
                    findings.push(CcrFinding::new(
                        "CCR_GEAR_EPOCH_NOT_MONOTONIC",
                        Error,
                        i,
                        cur,
                        format!("gear_epoch_id {b} regressed from {a}"),
                    ));
                }
                let gear_changed = match (prev.authorized_gear_opt(), cur.authorized_gear_opt()) {
                    (Some(pg), Some(cg)) => Some(pg != cg),
                    _ => None,
                };
                match gear_changed {
                    Some(true) if b <= a => findings.push(CcrFinding::new(
                        "CCR_GEAR_EPOCH_STALLED_ACROSS_TRANSITION",
                        Error,
                        i,
                        cur,
                        format!(
                            "authorized_gear changed {:?} -> {:?} but gear_epoch_id \
                             did not increment ({a} -> {b})",
                            prev.authorized_gear_opt(),
                            cur.authorized_gear_opt()
                        ),
                    )),
                    Some(false) if b > a => findings.push(CcrFinding::new(
                        "CCR_GEAR_EPOCH_INCREMENT_WITHOUT_TRANSITION",
                        Error,
                        i,
                        cur,
                        format!(
                            "gear_epoch_id advanced {a} -> {b} with no authorized \
                             transition (gear held at {:?})",
                            cur.authorized_gear_opt()
                        ),
                    )),
                    _ => {}
                }
                // The flag must agree with the fact.
                if let Some(ch) = gear_changed {
                    if ch != cur.has_flag(CCR_FLAG_GEAR_CHANGED) {
                        findings.push(CcrFinding::new(
                            "CCR_GEAR_CHANGE_FLAG_DISAGREES",
                            Error,
                            i,
                            cur,
                            format!(
                                "GEAR_CHANGED flag is {} but authorized_gear went \
                                 {:?} -> {:?}",
                                cur.has_flag(CCR_FLAG_GEAR_CHANGED),
                                prev.authorized_gear_opt(),
                                cur.authorized_gear_opt()
                            ),
                        ));
                    }
                }
            }
            _ => findings.push(CcrFinding::new(
                "CCR_GEAR_EPOCH_UNMEASURED",
                Error,
                i,
                cur,
                "gear_epoch_id is unmeasured on one side of this pair".to_string(),
            )),
        }

        // ── physical-time continuity: advance by the APPLIED dt ────────────
        //
        //   cum[i] - cum[i-1] == dt_applied[i] * (step[i] - step[i-1])
        //
        // `dt_applied[i]` is the timestep integrated over the interval that
        // ENDS at record i, which is exactly the chunk-boundary semantics:
        // a gear flip lands end-of-chunk, so the dt is constant across the
        // interval a single record covers.
        match (
            prev.cumulative_simulated_time_ps_opt(),
            cur.cumulative_simulated_time_ps_opt(),
            cur.dt_applied_ps_opt(),
            steps,
        ) {
            (Some(t0), Some(t1), Some(dt), Some(n)) => {
                let expected = dt * n as f64;
                let observed = t1 - t0;
                if rel_diff(observed, expected) > policy.time_continuity_rel_tol.value {
                    findings.push(CcrFinding::new(
                        "CCR_TIME_NOT_ADVANCED_BY_APPLIED_DT",
                        Error,
                        i,
                        cur,
                        format!(
                            "cumulative_simulated_time_ps advanced {observed} ps over \
                             {n} steps, but the APPLIED dt {dt} ps requires \
                             {expected} ps (tol {:.3e}, origin {})",
                            policy.time_continuity_rel_tol.value,
                            policy.time_continuity_rel_tol.origin
                        ),
                    ));
                }
            }
            (Some(t0), Some(t1), None, _) => {
                // dt unmeasured: the clock must not have moved.
                if t1 != t0 {
                    findings.push(CcrFinding::new(
                        "CCR_TIME_ADVANCED_WITHOUT_APPLIED_DT",
                        Error,
                        i,
                        cur,
                        format!(
                            "cumulative_simulated_time_ps moved {t0} -> {t1} while \
                             dt_applied_ps is unmeasured"
                        ),
                    ));
                }
            }
            _ => {}
        }
    }

    CoupledChronologyValidation {
        policy: policy.clone(),
        records_examined: records.len() as u64,
        findings,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §7  Chronology clock — the host-side record producer
// ═══════════════════════════════════════════════════════════════════════════

/// TSH decision summary supplied to [`CoupledChronologyClock::commit`].
///
/// `None` in any field means NOT MEASURED and is stamped as the explicit
/// sentinel. Nothing here is ever defaulted to a plausible number.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TshSummary {
    /// Integrated salience per SIMULATED picosecond. Never per record, per
    /// chunk, or per invocation.
    pub integrated_salience_per_ps: Option<f64>,
    /// TSH memory state scalar.
    pub memory_state: Option<f64>,
    /// TSH confidence in its own request.
    pub request_confidence: Option<f64>,
    /// Anti-blindness observation confidence.
    pub observation_confidence: Option<f64>,
    /// Active branch bitmask. `Some(0)` legitimately means "none active".
    pub active_branches: Option<u32>,
}

/// Physical-governor summary supplied to [`CoupledChronologyClock::commit`].
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GovernorSummary {
    /// Constraint residual.
    pub constraint_residual: Option<f64>,
    /// Hamiltonian drift.
    pub hamiltonian_drift: Option<f64>,
    /// Force norm.
    pub force_norm: Option<f64>,
    /// `CCR_GUARD_*` bitmask. `Some(0)` legitimately means "all clear".
    pub guard_status: Option<u32>,
}

/// One fused decision, ready to be committed to the chronology.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GearDecision {
    /// MD steps this record covers. With `body_unroll > 1` this is the chunk
    /// length, because the gear is constant across a chunk.
    pub steps_advanced: u64,
    /// The `dt` ACTUALLY integrated over those steps, in ps.
    ///
    /// `None` when no measurement exists. The clock then STALLS rather than
    /// substituting the authorized dt — substituting would fabricate elapsed
    /// physics, which is the failure this record exists to prevent.
    pub dt_applied_ps: Option<f64>,
    /// Gear the TSH arm demanded.
    pub requested_gear: Option<u32>,
    /// Gear the governor authorized.
    pub authorized_gear: Option<u32>,
    /// Gear read back off the device.
    pub device_measured_gear: Option<u32>,
    /// `CCR_REASON_*`.
    pub transition_reason: u32,
    /// HMR topology permission.
    pub hmr: HmrVerification,
    /// TSH demand summary.
    pub tsh: TshSummary,
    /// Governor summary.
    pub governor: GovernorSummary,
    /// Producing stream.
    pub stream_id: Option<u32>,
    /// Producing replica.
    pub replica_id: Option<u32>,
    /// `CCR_EMITTER_*`.
    pub emitter_tag: u32,
    /// Bound tensor snapshot.
    pub tensor_snapshot_id: Option<u64>,
    /// Extra `CCR_FLAG_*` bits the caller wants stamped. `DECISION_EXECUTED`,
    /// `GEAR_HELD` / `GEAR_CHANGED`, `CLOCK_STALLED` and `ABORT_TRAP` are
    /// derived by the clock and must not be passed here.
    pub extra_flags: u32,
}

impl Default for GearDecision {
    fn default() -> Self {
        Self {
            steps_advanced: 0,
            dt_applied_ps: None,
            requested_gear: None,
            authorized_gear: None,
            device_measured_gear: None,
            transition_reason: CCR_REASON_NOT_RECORDED,
            hmr: HmrVerification::Unverified,
            tsh: TshSummary::default(),
            governor: GovernorSummary::default(),
            stream_id: None,
            replica_id: None,
            emitter_tag: CCR_EMITTER_UNSPECIFIED,
            tensor_snapshot_id: None,
            extra_flags: 0,
        }
    }
}

/// The authoritative host-side chronology: global step, cumulative simulated
/// time, and gear epoch.
///
/// The three facts are advanced in ONE place so they cannot drift apart. In
/// particular, cumulative simulated time is advanced ONLY by the APPLIED
/// timestep — never the requested one, never the authorized one.
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledChronologyClock {
    step: u64,
    cumulative_ps: f64,
    gear_epoch_id: u64,
    last_authorized_gear: Option<u32>,
    stalled_intervals: u64,
    records_emitted: u64,
    next_record_seq: u64,
}

impl Default for CoupledChronologyClock {
    fn default() -> Self {
        Self::new()
    }
}

impl CoupledChronologyClock {
    /// A clock at step 0, 0 ps elapsed, gear epoch 0, with no gear yet
    /// established.
    pub fn new() -> Self {
        Self {
            step: 0,
            cumulative_ps: 0.0,
            gear_epoch_id: 0,
            last_authorized_gear: None,
            stalled_intervals: 0,
            records_emitted: 0,
            next_record_seq: 0,
        }
    }

    /// Resume from a checkpoint.
    ///
    /// The decision path is STATEFUL across invocations (the Hamiltonian fuse
    /// needs `V(t-1)` and accumulated `W_ext`), so the chronology must be
    /// checkpointable for reproducibility.
    pub fn resume(
        step: u64,
        cumulative_ps: f64,
        gear_epoch_id: u64,
        last_authorized_gear: Option<u32>,
        next_record_seq: u64,
    ) -> Self {
        Self {
            step,
            cumulative_ps,
            gear_epoch_id,
            last_authorized_gear,
            stalled_intervals: 0,
            records_emitted: 0,
            next_record_seq,
        }
    }

    /// The sequence number the next emitted record will carry. Checkpoint
    /// this alongside step / cumulative time / gear epoch: a resumed run that
    /// restarts the sequence would make its own records look duplicated.
    pub fn next_record_seq(&self) -> u64 {
        self.next_record_seq
    }

    /// Current global MD step.
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Cumulative simulated time in picoseconds.
    pub fn cumulative_simulated_time_ps(&self) -> f64 {
        self.cumulative_ps
    }

    /// Current gear epoch.
    pub fn gear_epoch_id(&self) -> u64 {
        self.gear_epoch_id
    }

    /// The authorized gear in force, or `None` when none has been
    /// established.
    pub fn last_authorized_gear(&self) -> Option<u32> {
        self.last_authorized_gear
    }

    /// Intervals over which the clock refused to advance because
    /// `dt_applied_ps` was unmeasured.
    pub fn stalled_intervals(&self) -> u64 {
        self.stalled_intervals
    }

    /// Records emitted by this clock.
    pub fn records_emitted(&self) -> u64 {
        self.records_emitted
    }

    /// Advance the chronology by one decision and produce the record.
    ///
    /// A record is produced on EVERY call, including when the gear is held —
    /// a gap in the stream must never be ambiguous between "gear held" and
    /// "decision did not run".
    ///
    /// # Advancement rules
    ///
    /// * `step` advances by `steps_advanced` unconditionally: it counts MD
    ///   steps and does not depend on `dt`.
    /// * `cumulative_simulated_time_ps` advances by
    ///   `dt_applied_ps * steps_advanced` — the APPLIED timestep only. When
    ///   `dt_applied_ps` is unmeasured, the clock STALLS and
    ///   `CCR_FLAG_CLOCK_STALLED` is stamped.
    /// * `gear_epoch_id` increments exactly when the authorized gear differs
    ///   from the previously established one.
    pub fn commit(&mut self, d: &GearDecision) -> CoupledChronologyRecord {
        let mut rec = CoupledChronologyRecord::unmeasured();
        let mut flags = d.extra_flags
            & !(CCR_FLAG_DECISION_EXECUTED
                | CCR_FLAG_GEAR_HELD
                | CCR_FLAG_GEAR_CHANGED
                | CCR_FLAG_CLOCK_STALLED
                | CCR_FLAG_ABORT_TRAP);
        flags |= CCR_FLAG_DECISION_EXECUTED;

        // ── gear epoch ─────────────────────────────────────────────────────
        let changed = match (self.last_authorized_gear, d.authorized_gear) {
            (Some(prev), Some(now)) => {
                if prev != now {
                    self.gear_epoch_id = self.gear_epoch_id.saturating_add(1);
                    true
                } else {
                    false
                }
            }
            // First established gear opens epoch 0; it is a transition from
            // "undefined", so it is reported as CHANGED without incrementing.
            (None, Some(_)) => true,
            // No authorization available: the epoch cannot move.
            (_, None) => false,
        };
        if d.authorized_gear.is_some() {
            self.last_authorized_gear = d.authorized_gear;
        }
        flags |= if changed {
            CCR_FLAG_GEAR_CHANGED
        } else {
            CCR_FLAG_GEAR_HELD
        };
        if d.authorized_gear == Some(CCR_GEAR_ABORT_TRAP) {
            flags |= CCR_FLAG_ABORT_TRAP;
        }

        // ── clock ──────────────────────────────────────────────────────────
        self.step = self.step.saturating_add(d.steps_advanced);
        match d.dt_applied_ps {
            Some(dt) if dt.is_finite() && dt > 0.0 => {
                self.cumulative_ps += dt * d.steps_advanced as f64;
                rec.dt_applied_ps = dt;
            }
            _ => {
                // Unmeasured, or measured as non-physical. Either way the
                // physical clock cannot advance.
                self.stalled_intervals += 1;
                flags |= CCR_FLAG_CLOCK_STALLED;
                rec.dt_applied_ps = match d.dt_applied_ps {
                    Some(dt) => dt, // preserve a measured non-physical value
                    None => CCR_F64_UNMEASURED,
                };
            }
        }

        rec.global_md_step = self.step;
        rec.cumulative_simulated_time_ps = self.cumulative_ps;
        rec.gear_epoch_id = self.gear_epoch_id;
        rec.tensor_snapshot_id = d.tensor_snapshot_id.unwrap_or(CCR_U64_UNMEASURED);

        rec.tsh_integrated_salience_per_ps = d
            .tsh
            .integrated_salience_per_ps
            .unwrap_or(CCR_F64_UNMEASURED);
        rec.tsh_memory_state = d.tsh.memory_state.unwrap_or(CCR_F64_UNMEASURED);
        rec.tsh_request_confidence = d.tsh.request_confidence.unwrap_or(CCR_F64_UNMEASURED);
        rec.observation_confidence = d.tsh.observation_confidence.unwrap_or(CCR_F64_UNMEASURED);
        rec.tsh_active_branches = d.tsh.active_branches.unwrap_or(CCR_U32_UNMEASURED);

        rec.constraint_residual = d.governor.constraint_residual.unwrap_or(CCR_F64_UNMEASURED);
        rec.hamiltonian_drift = d.governor.hamiltonian_drift.unwrap_or(CCR_F64_UNMEASURED);
        rec.force_norm = d.governor.force_norm.unwrap_or(CCR_F64_UNMEASURED);
        rec.physical_guard_status = d.governor.guard_status.unwrap_or(CCR_U32_UNMEASURED);

        rec.requested_gear = d.requested_gear.unwrap_or(CCR_GEAR_UNMEASURED);
        rec.authorized_gear = d.authorized_gear.unwrap_or(CCR_GEAR_UNMEASURED);
        rec.device_measured_gear = d.device_measured_gear.unwrap_or(CCR_GEAR_UNMEASURED);
        rec.transition_reason = d.transition_reason;
        rec.hmr_verified = d.hmr.to_u32();
        rec.stream_id = d.stream_id.unwrap_or(CCR_U32_UNMEASURED);
        rec.replica_id = d.replica_id.unwrap_or(CCR_U32_UNMEASURED);
        rec.emitter_tag = d.emitter_tag;
        rec.record_flags = flags;
        rec.record_seq = self.next_record_seq;
        rec.publication_fence = CCR_FENCE_PUBLISHED;

        self.next_record_seq += 1;
        self.records_emitted += 1;
        rec
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §8  Conservation receipt
// ═══════════════════════════════════════════════════════════════════════════

/// Conservation receipt for a coupled-chronology record stream.
///
/// Shaped to match the WP4 ZSTR receipt ([`crate::zstr::ZstrStats`]) so a
/// campaign can check both with one rule:
///
/// ```text
///   expected == produced + backpressure_events
///   produced == persisted                        (lossless persistence)
///   duplicates == 0 && gaps == 0
///   continuity_unverifiable == 0
///   terminal_error.is_none()
/// ```
///
/// # Why `gaps` and `continuity_breaks` are SEPARATE terms
///
/// ZSTR counts gaps as missing device steps, which works only because its
/// stride is 1. Under a varying gear the step delta between records is not
/// constant, so a step-count gap is meaningless.
///
/// Two independent checks replace it, and neither subsumes the other:
///
/// * `gaps` — missing `record_seq` values. This is the COMPLETENESS check.
/// * `continuity_breaks` — violations of the physical-time identity
///   `cum[i] - cum[i-1] == dt_applied[i] * (step[i] - step[i-1])`. This is
///   the PHYSICS check: elapsed simulated time that no applied timestep
///   accounts for.
///
/// A dropped record while the gear is HELD satisfies the continuity identity
/// exactly — the survivors' larger step gap times the same constant `dt`
/// reproduces the elapsed time — so continuity alone would report the stream
/// as conserved. Conversely a forged or mis-stamped cumulative time can leave
/// the sequence perfectly dense. Both terms are required.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoupledChronologyConservation {
    /// Decision-path invocations the producer executed.
    pub expected: u64,
    /// Invocations that staged a record and raised the publication fence.
    pub produced: u64,
    /// Records the consumer durably persisted.
    pub persisted: u64,
    /// Non-advancing `record_seq` or `global_md_step` between consecutive
    /// persisted records — a slot re-read without a fresh producer stamp.
    /// Both are counted, so a stream that repeats one and not the other is
    /// still reported.
    pub duplicates: u64,
    /// Missing `record_seq` values between consecutive persisted records —
    /// the COMPLETENESS term. Records the producer emitted that never
    /// reached this stream.
    pub gaps: u64,
    /// Violations of the physical-time continuity identity — the PHYSICS
    /// term. Elapsed simulated time that no applied timestep accounts for.
    pub continuity_breaks: u64,
    /// Consecutive pairs whose continuity could NOT be checked because a
    /// required term was unmeasured. Never silently a pass: an unverifiable
    /// chronology is not a conserved chronology.
    pub continuity_unverifiable: u64,
    /// Claims refused by producer-side backpressure.
    pub backpressure_events: u64,
    /// First persisted `global_md_step`.
    pub first_global_md_step: Option<u64>,
    /// Last persisted `global_md_step`.
    pub last_global_md_step: Option<u64>,
    /// First persisted `gear_epoch_id`.
    pub first_gear_epoch_id: Option<u64>,
    /// Last persisted `gear_epoch_id`.
    pub last_gear_epoch_id: Option<u64>,
    /// Total simulated picoseconds spanned by the persisted stream, computed
    /// from the endpoints' cumulative time. `None` when either endpoint is
    /// unmeasured.
    pub simulated_time_span_ps: Option<f64>,
    /// Records whose clock stalled on an unmeasured `dt_applied_ps`.
    pub clock_stalled_records: u64,
    /// Records carrying no device-measured gear.
    pub device_gear_unmeasured_records: u64,
    /// Distinct authorized gears observed, ascending.
    pub authorized_gears_observed: Vec<u32>,
    /// Schema version observed in the last persisted record.
    pub observed_schema_version: u32,
    /// Validation verdict for the persisted stream.
    pub validation_pass: bool,
    /// Error-severity validation findings.
    pub validation_error_count: u64,
    /// First fatal fault. Any value means the stream is NOT a complete
    /// account of the run.
    pub terminal_error: Option<String>,
}

impl CoupledChronologyConservation {
    /// A receipt for a producer that never emitted.
    pub fn aborted(reason: impl Into<String>) -> Self {
        Self {
            expected: 0,
            produced: 0,
            persisted: 0,
            duplicates: 0,
            gaps: 0,
            continuity_breaks: 0,
            continuity_unverifiable: 0,
            backpressure_events: 0,
            first_global_md_step: None,
            last_global_md_step: None,
            first_gear_epoch_id: None,
            last_gear_epoch_id: None,
            simulated_time_span_ps: None,
            clock_stalled_records: 0,
            device_gear_unmeasured_records: 0,
            authorized_gears_observed: Vec::new(),
            observed_schema_version: 0,
            validation_pass: false,
            validation_error_count: 0,
            terminal_error: Some(reason.into()),
        }
    }

    /// Build a receipt from the persisted stream plus the producer's own
    /// counters.
    ///
    /// `expected` and `backpressure_events` come from the producer control
    /// block; they cannot be derived from the persisted records, which is
    /// exactly why loss is detectable.
    pub fn from_stream(
        expected: u64,
        produced: u64,
        backpressure_events: u64,
        records: &[CoupledChronologyRecord],
        policy: &CoupledChronologyPolicy,
    ) -> Self {
        let validation = validate_stream(records, policy);

        let mut duplicates = 0u64;
        let mut gaps = 0u64;
        let mut continuity_breaks = 0u64;
        let mut continuity_unverifiable = 0u64;
        let mut clock_stalled = 0u64;
        let mut gear_unmeasured = 0u64;
        let mut gears: Vec<u32> = Vec::new();

        for rec in records {
            if rec.has_flag(CCR_FLAG_CLOCK_STALLED) {
                clock_stalled += 1;
            }
            if rec.device_measured_gear_opt().is_none() {
                gear_unmeasured += 1;
            }
            if let Some(g) = rec.authorized_gear_opt() {
                if !gears.contains(&g) {
                    gears.push(g);
                }
            }
        }
        gears.sort_unstable();

        for i in 1..records.len() {
            let prev = &records[i - 1];
            let cur = &records[i];
            // Completeness: missing sequence numbers.
            match (prev.record_seq_opt(), cur.record_seq_opt()) {
                (Some(a), Some(b)) if b > a => gaps += b - a - 1,
                (Some(_), Some(_)) => duplicates += 1,
                _ => {}
            }
            // Duplicate detection also covers a non-advancing step, which is
            // what a slot re-read without a fresh stamp looks like.
            let steps = match (prev.global_md_step_opt(), cur.global_md_step_opt()) {
                (Some(a), Some(b)) if b > a => Some(b - a),
                (Some(_), Some(_)) => {
                    duplicates += 1;
                    None
                }
                _ => None,
            };
            // Physics: elapsed time must equal the APPLIED dt times the steps.
            match (
                prev.cumulative_simulated_time_ps_opt(),
                cur.cumulative_simulated_time_ps_opt(),
                cur.dt_applied_ps_opt(),
                steps,
            ) {
                (Some(t0), Some(t1), Some(dt), Some(n)) => {
                    let expected_delta = dt * n as f64;
                    if rel_diff(t1 - t0, expected_delta) > policy.time_continuity_rel_tol.value {
                        continuity_breaks += 1;
                    }
                }
                _ => continuity_unverifiable += 1,
            }
        }

        let first = records.first();
        let last = records.last();
        let span = match (
            first.and_then(|r| r.cumulative_simulated_time_ps_opt()),
            last.and_then(|r| r.cumulative_simulated_time_ps_opt()),
        ) {
            (Some(a), Some(b)) => Some(b - a),
            _ => None,
        };

        Self {
            expected,
            produced,
            persisted: records.len() as u64,
            duplicates,
            gaps,
            continuity_breaks,
            continuity_unverifiable,
            backpressure_events,
            first_global_md_step: first.and_then(|r| r.global_md_step_opt()),
            last_global_md_step: last.and_then(|r| r.global_md_step_opt()),
            first_gear_epoch_id: first.and_then(|r| r.gear_epoch_id_opt()),
            last_gear_epoch_id: last.and_then(|r| r.gear_epoch_id_opt()),
            simulated_time_span_ps: span,
            clock_stalled_records: clock_stalled,
            device_gear_unmeasured_records: gear_unmeasured,
            authorized_gears_observed: gears,
            observed_schema_version: last.map(|r| r.schema_version).unwrap_or(0),
            validation_pass: validation.is_pass(),
            validation_error_count: validation.error_count() as u64,
            terminal_error: None,
        }
    }

    /// True iff every conservation term is satisfied.
    pub fn conservation_pass(&self) -> bool {
        self.terminal_error.is_none()
            && self.produced == self.persisted
            && self.expected == self.produced + self.backpressure_events
            && self.duplicates == 0
            && self.gaps == 0
            && self.continuity_breaks == 0
            && self.continuity_unverifiable == 0
            && self.observed_schema_version == CCR_SCHEMA_VERSION
            && self.validation_pass
    }

    /// Stable machine-readable list of the failing conservation terms. Empty
    /// iff [`Self::conservation_pass`] is true.
    pub fn failing_terms(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.terminal_error.is_some() {
            out.push("terminal_error");
        }
        if self.produced != self.persisted {
            out.push("produced_ne_persisted");
        }
        if self.expected != self.produced + self.backpressure_events {
            out.push("expected_ne_produced_plus_backpressure");
        }
        if self.duplicates != 0 {
            out.push("duplicates");
        }
        if self.gaps != 0 {
            out.push("gaps");
        }
        if self.continuity_breaks != 0 {
            out.push("continuity_breaks");
        }
        if self.continuity_unverifiable != 0 {
            out.push("continuity_unverifiable");
        }
        if self.observed_schema_version != CCR_SCHEMA_VERSION {
            out.push("schema_version");
        }
        if !self.validation_pass {
            out.push("validation");
        }
        out
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §9  Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a `#define NAME (1u << N)` bit-flag macro out of the embedded
    /// header at runtime. The const parser deliberately does not handle
    /// parenthesised expressions, so this closes the tie for the flag and
    /// guard masks.
    fn cuh_shift_define(name: &str) -> u32 {
        let needle = format!("#define {name} ");
        for line in CCR_CUH_SOURCE.lines() {
            // Strip any trailing `//` comment before parsing the value.
            let line = line.split("//").next().unwrap_or("").trim_end();
            if let Some(rest) = line.strip_prefix(&needle) {
                let rest = rest.trim();
                let inner = rest
                    .trim_start_matches('(')
                    .trim_end_matches(')')
                    .trim()
                    .to_string();
                let (lhs, rhs) = inner
                    .split_once("<<")
                    .unwrap_or_else(|| panic!("{name}: not a shift expression: {rest}"));
                assert_eq!(lhs.trim().trim_end_matches(['u', 'U']), "1", "{name}: lhs");
                let shift: u32 = rhs
                    .trim()
                    .trim_end_matches(['u', 'U'])
                    .parse()
                    .unwrap_or_else(|e| panic!("{name}: shift parse: {e}"));
                return 1u32 << shift;
            }
        }
        panic!("{name}: #define not found in coupled_chronology.cuh");
    }

    /// Accumulated f64 sums differ from a single multiply in the last ULP,
    /// so tests compare with the same relative tolerance the validator uses
    /// rather than asserting bit equality on derived physics.
    fn close(a: f64, b: f64) {
        assert!(
            rel_diff(a, b) <= CCR_TIME_CONTINUITY_REL_TOL_SOURCE_LITERAL,
            "{a} != {b} within {CCR_TIME_CONTINUITY_REL_TOL_SOURCE_LITERAL:e}"
        );
    }

    fn published(mut rec: CoupledChronologyRecord) -> CoupledChronologyRecord {
        rec.publication_fence = CCR_FENCE_PUBLISHED;
        rec
    }

    /// A fully-populated, internally consistent decision at gear 1.
    fn healthy_decision() -> GearDecision {
        GearDecision {
            steps_advanced: 100,
            dt_applied_ps: Some(0.0020),
            requested_gear: Some(CCR_GEAR_DEFAULT_2FS),
            authorized_gear: Some(CCR_GEAR_DEFAULT_2FS),
            device_measured_gear: Some(CCR_GEAR_DEFAULT_2FS),
            transition_reason: CCR_REASON_HELD_NO_CHANGE,
            hmr: HmrVerification::NotApplied,
            tsh: TshSummary {
                integrated_salience_per_ps: Some(0.25),
                memory_state: Some(0.1),
                request_confidence: Some(0.9),
                observation_confidence: Some(0.8),
                active_branches: Some(0b0011),
            },
            governor: GovernorSummary {
                constraint_residual: Some(1e-8),
                hamiltonian_drift: Some(2e-6),
                force_norm: Some(1234.5),
                guard_status: Some(0),
            },
            stream_id: Some(0),
            replica_id: Some(0),
            emitter_tag: CCR_EMITTER_GEARBOX_DECISION,
            tensor_snapshot_id: Some(0),
            extra_flags: 0,
        }
    }

    // ── layout ─────────────────────────────────────────────────────────────

    #[test]
    fn record_layout_is_frozen() {
        assert_eq!(std::mem::size_of::<CoupledChronologyRecord>(), 192);
        assert_eq!(std::mem::align_of::<CoupledChronologyRecord>(), 64);
        assert_eq!(CCR_RECORD_BYTES, 192);
        assert_eq!(CCR_RECORD_ALIGN, 64);

        // Frozen offsets, spelled out so a reorder is caught here as well as
        // in the const block.
        assert_eq!(CCR_OFF_GLOBAL_MD_STEP, 0);
        assert_eq!(CCR_OFF_CUM_SIM_TIME_PS, 8);
        assert_eq!(CCR_OFF_DT_APPLIED_PS, 16);
        assert_eq!(CCR_OFF_GEAR_EPOCH_ID, 24);
        assert_eq!(CCR_OFF_TENSOR_SNAPSHOT_ID, 32);
        assert_eq!(CCR_OFF_TSH_SALIENCE_PER_PS, 40);
        assert_eq!(CCR_OFF_TSH_MEMORY_STATE, 48);
        assert_eq!(CCR_OFF_TSH_REQ_CONFIDENCE, 56);
        assert_eq!(CCR_OFF_OBSERVATION_CONF, 64);
        assert_eq!(CCR_OFF_CONSTRAINT_RESIDUAL, 72);
        assert_eq!(CCR_OFF_HAMILTONIAN_DRIFT, 80);
        assert_eq!(CCR_OFF_FORCE_NORM, 88);
        assert_eq!(CCR_OFF_SCHEMA_VERSION, 96);
        assert_eq!(CCR_OFF_RECORD_FLAGS, 100);
        assert_eq!(CCR_OFF_REQUESTED_GEAR, 104);
        assert_eq!(CCR_OFF_AUTHORIZED_GEAR, 108);
        assert_eq!(CCR_OFF_DEVICE_MEASURED_GEAR, 112);
        assert_eq!(CCR_OFF_TSH_ACTIVE_BRANCHES, 116);
        assert_eq!(CCR_OFF_PHYSICAL_GUARD_STATUS, 120);
        assert_eq!(CCR_OFF_TRANSITION_REASON, 124);
        assert_eq!(CCR_OFF_HMR_VERIFIED, 128);
        assert_eq!(CCR_OFF_STREAM_ID, 132);
        assert_eq!(CCR_OFF_REPLICA_ID, 136);
        assert_eq!(CCR_OFF_EMITTER_TAG, 140);
        assert_eq!(CCR_OFF_PUBLICATION_FENCE, 144);
        assert_eq!(CCR_OFF_RESERVED_U32, 148);
        assert_eq!(CCR_OFF_RECORD_SEQ, 152);
        assert_eq!(CCR_OFF_RESERVED, 160);
        assert_eq!(
            CCR_OFF_RESERVED + CCR_RESERVED_U32_COUNT * 4,
            CCR_RECORD_BYTES
        );
    }

    #[test]
    fn wire_offsets_place_fields_where_the_header_says() {
        // Write a distinct value into every field, then read the raw bytes
        // back at the header's offsets. This proves the encoder honours the
        // frozen layout, not merely that the struct does.
        let mut rec = CoupledChronologyRecord::unmeasured();
        rec.global_md_step = 0x1122_3344_5566_7788;
        rec.gear_epoch_id = 0x0102_0304_0506_0708;
        rec.tensor_snapshot_id = 0x00FF_00FF_00FF_00FF;
        rec.cumulative_simulated_time_ps = 12.5;
        rec.dt_applied_ps = 0.0040;
        rec.requested_gear = 2;
        rec.authorized_gear = 1;
        rec.device_measured_gear = 0;
        rec.stream_id = 7;
        rec.replica_id = 3;
        rec.emitter_tag = CCR_EMITTER_LADD;
        rec.record_seq = 0xAABB_CCDD_EEFF_0011;

        let b = rec.to_bytes();
        assert_eq!(rd_u64(&b, CCR_OFF_GLOBAL_MD_STEP), 0x1122_3344_5566_7788);
        assert_eq!(rd_u64(&b, CCR_OFF_GEAR_EPOCH_ID), 0x0102_0304_0506_0708);
        assert_eq!(
            rd_u64(&b, CCR_OFF_TENSOR_SNAPSHOT_ID),
            0x00FF_00FF_00FF_00FF
        );
        assert_eq!(f64::from_bits(rd_u64(&b, CCR_OFF_CUM_SIM_TIME_PS)), 12.5);
        assert_eq!(f64::from_bits(rd_u64(&b, CCR_OFF_DT_APPLIED_PS)), 0.0040);
        assert_eq!(rd_u32(&b, CCR_OFF_REQUESTED_GEAR), 2);
        assert_eq!(rd_u32(&b, CCR_OFF_AUTHORIZED_GEAR), 1);
        assert_eq!(rd_u32(&b, CCR_OFF_DEVICE_MEASURED_GEAR), 0);
        assert_eq!(rd_u32(&b, CCR_OFF_STREAM_ID), 7);
        assert_eq!(rd_u32(&b, CCR_OFF_REPLICA_ID), 3);
        assert_eq!(rd_u32(&b, CCR_OFF_EMITTER_TAG), CCR_EMITTER_LADD);
        assert_eq!(rd_u32(&b, CCR_OFF_SCHEMA_VERSION), CCR_SCHEMA_VERSION);
        assert_eq!(rd_u64(&b, CCR_OFF_RECORD_SEQ), 0xAABB_CCDD_EEFF_0011);
    }

    #[test]
    fn cuh_bit_flag_macros_match_rust() {
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_DECISION_EXECUTED"),
            CCR_FLAG_DECISION_EXECUTED
        );
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_GEAR_HELD"),
            CCR_FLAG_GEAR_HELD
        );
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_GEAR_CHANGED"),
            CCR_FLAG_GEAR_CHANGED
        );
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_STAGED_PENDING_BOUNDARY"),
            CCR_FLAG_STAGED_PENDING_BOUNDARY
        );
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_CHUNK_ENTRY_CONSERVATIVE"),
            CCR_FLAG_CHUNK_ENTRY_CONSERVATIVE
        );
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_SENTINEL_SAMPLE"),
            CCR_FLAG_SENTINEL_SAMPLE
        );
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_OBSERVATION_STARVED"),
            CCR_FLAG_OBSERVATION_STARVED
        );
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_CLOCK_STALLED"),
            CCR_FLAG_CLOCK_STALLED
        );
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_UNCERTAINTY_GROWN"),
            CCR_FLAG_UNCERTAINTY_GROWN
        );
        assert_eq!(
            cuh_shift_define("PRISM_CCR_FLAG_ABORT_TRAP"),
            CCR_FLAG_ABORT_TRAP
        );

        let guards = [
            (
                "PRISM_CCR_GUARD_PHYSICAL_EMERGENCY",
                CCR_GUARD_PHYSICAL_EMERGENCY,
            ),
            (
                "PRISM_CCR_GUARD_TOPOLOGY_FORBIDS",
                CCR_GUARD_TOPOLOGY_FORBIDS,
            ),
            (
                "PRISM_CCR_GUARD_CONSTRAINT_UNSTABLE",
                CCR_GUARD_CONSTRAINT_UNSTABLE,
            ),
            ("PRISM_CCR_GUARD_ENERGY_DRIFT", CCR_GUARD_ENERGY_DRIFT),
            ("PRISM_CCR_GUARD_FORCE_UNSTABLE", CCR_GUARD_FORCE_UNSTABLE),
            ("PRISM_CCR_GUARD_TSH_MEMORY", CCR_GUARD_TSH_MEMORY),
            ("PRISM_CCR_GUARD_TSH_SALIENCE", CCR_GUARD_TSH_SALIENCE),
            (
                "PRISM_CCR_GUARD_PHASE_REQUIREMENT",
                CCR_GUARD_PHASE_REQUIREMENT,
            ),
            (
                "PRISM_CCR_GUARD_DWELL_UNSATISFIED",
                CCR_GUARD_DWELL_UNSATISFIED,
            ),
            (
                "PRISM_CCR_GUARD_OBSERVATION_CONF",
                CCR_GUARD_OBSERVATION_CONF,
            ),
            (
                "PRISM_CCR_GUARD_INSTABILITY_MEMORY",
                CCR_GUARD_INSTABILITY_MEMORY,
            ),
        ];
        for (name, rust) in guards {
            assert_eq!(cuh_shift_define(name), rust, "{name}");
        }
        assert_eq!(guards.len() as u32, CCR_GUARD_BIT_COUNT);
        assert_eq!(CCR_GUARD_PRIORITY_ORDER.len() as u32, CCR_GUARD_BIT_COUNT);
        // Priority order is strictly ascending in bit index — lower bit
        // index means higher priority, which is what makes the read
        // lexicographic rather than weighted.
        for w in CCR_GUARD_PRIORITY_ORDER.windows(2) {
            assert!(w[0].0 < w[1].0, "guard priority order must be ascending");
        }
    }

    // ── sentinels ──────────────────────────────────────────────────────────

    #[test]
    fn unmeasured_record_reports_every_field_absent() {
        let rec = CoupledChronologyRecord::unmeasured();
        assert!(rec.global_md_step_opt().is_none());
        assert!(rec.cumulative_simulated_time_ps_opt().is_none());
        assert!(rec.dt_applied_ps_opt().is_none());
        assert!(rec.gear_epoch_id_opt().is_none());
        assert!(rec.tensor_snapshot_id_opt().is_none());
        assert!(rec.requested_gear_opt().is_none());
        assert!(rec.authorized_gear_opt().is_none());
        assert!(rec.device_measured_gear_opt().is_none());
        assert!(rec.tsh_active_branches_opt().is_none());
        assert!(rec.physical_guard_status_opt().is_none());
        assert!(rec.transition_reason_opt().is_none());
        assert!(rec.stream_id_opt().is_none());
        assert!(rec.replica_id_opt().is_none());
        assert!(rec.emitter_tag_opt().is_none());
        assert_eq!(rec.hmr_verification(), HmrVerification::Unverified);
        assert!(!rec.is_published());
        assert_eq!(rec.unmeasured_fields().len(), 23);
    }

    #[test]
    fn gear_zero_is_a_legitimate_gear_and_never_means_unknown() {
        // This is the whole reason the record exists: a hardcoded gear_id 0
        // was indistinguishable from a measured gear 0, and 0 means 0.5 fs.
        assert_ne!(CCR_GEAR_UNMEASURED, CCR_GEAR_FINE_0P5FS);
        assert!(CCR_GEAR_UNMEASURED > CCR_GEAR_MAX);

        let mut measured = CoupledChronologyRecord::unmeasured();
        measured.device_measured_gear = CCR_GEAR_FINE_0P5FS;
        assert_eq!(measured.device_measured_gear_opt(), Some(0));

        let absent = CoupledChronologyRecord::unmeasured();
        assert_eq!(absent.device_measured_gear_opt(), None);

        // And they are different bytes on the wire.
        assert_ne!(
            rd_u32(&measured.to_bytes(), CCR_OFF_DEVICE_MEASURED_GEAR),
            rd_u32(&absent.to_bytes(), CCR_OFF_DEVICE_MEASURED_GEAR)
        );
    }

    #[test]
    fn f64_unmeasured_is_distinguishable_from_a_measured_nan() {
        assert!(CCR_F64_UNMEASURED.is_nan());
        assert!(ccr_f64_is_unmeasured(CCR_F64_UNMEASURED));

        // An ordinary NaN — a MEASURED non-physical state, e.g. the G29
        // force-norm trap — must NOT read as absence.
        assert!(!ccr_f64_is_unmeasured(f64::NAN));
        assert!(!ccr_f64_is_unmeasured(0.0_f64 / 0.0));
        assert!(!ccr_f64_is_unmeasured(f64::INFINITY));
        assert!(!ccr_f64_is_unmeasured(0.0));

        let mut rec = CoupledChronologyRecord::unmeasured();
        rec.force_norm = f64::NAN;
        assert!(
            rec.force_norm_opt().is_some(),
            "measured NaN is a measurement"
        );
        assert!(rec.force_norm_opt().unwrap().is_nan());
    }

    #[test]
    fn sentinels_round_trip_through_the_wire_format() {
        let rec = published(CoupledChronologyRecord::unmeasured());
        let bytes = rec.to_bytes();
        let back = CoupledChronologyRecord::from_bytes(&bytes).expect("decode");
        assert_eq!(rec, back);
        assert_eq!(back.to_bytes(), bytes);
        // The NaN payload survived — otherwise "not measured" would have
        // collapsed into "measured as NaN".
        assert!(ccr_f64_is_unmeasured(back.dt_applied_ps));
        assert!(back.dt_applied_ps_opt().is_none());
        assert_eq!(back.unmeasured_fields().len(), 23);
    }

    #[test]
    fn populated_record_round_trips_byte_exactly() {
        let mut clock = CoupledChronologyClock::new();
        let rec = clock.commit(&healthy_decision());
        let back = CoupledChronologyRecord::from_bytes(&rec.to_bytes()).expect("decode");
        assert_eq!(rec, back);
        assert_eq!(back.dt_applied_ps_opt(), Some(0.0020));
        assert_eq!(back.authorized_gear_opt(), Some(CCR_GEAR_DEFAULT_2FS));
        assert_eq!(back.tsh_active_branches_opt(), Some(0b0011));
        assert_eq!(back.physical_guard_status_opt(), Some(0));
    }

    #[test]
    fn decode_rejects_unknown_schema_and_ragged_buffers() {
        let mut b = published(CoupledChronologyRecord::unmeasured()).to_bytes();
        wr_u32(&mut b, CCR_OFF_SCHEMA_VERSION, CCR_SCHEMA_VERSION + 1);
        assert!(matches!(
            CoupledChronologyRecord::from_bytes(&b),
            Err(CcrDecodeError::SchemaVersionMismatch { .. })
        ));

        assert!(matches!(
            CoupledChronologyRecord::from_bytes(&b[..10]),
            Err(CcrDecodeError::ShortBuffer { .. })
        ));

        let good = published(CoupledChronologyRecord::unmeasured()).to_bytes();
        let mut stream = good.to_vec();
        stream.extend_from_slice(&good[..64]);
        assert!(matches!(
            CoupledChronologyRecord::decode_stream(&stream),
            Err(CcrDecodeError::RaggedBuffer { .. })
        ));
        assert_eq!(
            CoupledChronologyRecord::decode_stream(&good)
                .expect("decode")
                .len(),
            1
        );
    }

    // ── chronology invariants ──────────────────────────────────────────────

    #[test]
    fn cumulative_time_advances_by_the_applied_dt_not_requested_or_authorized() {
        // Requested gear 0 (0.5 fs), authorized gear 1 (2 fs), but the device
        // actually integrated at gear 2 (4 fs). Only the APPLIED dt may move
        // the clock; using either of the other two would misstate elapsed
        // physics by 8x or 2x.
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.steps_advanced = 500;
        d.requested_gear = Some(CCR_GEAR_FINE_0P5FS);
        d.authorized_gear = Some(CCR_GEAR_DEFAULT_2FS);
        d.device_measured_gear = Some(CCR_GEAR_HMR_4FS);
        d.dt_applied_ps = Some(0.0040);
        d.transition_reason = CCR_REASON_INITIALIZATION;

        let rec = clock.commit(&d);

        let applied = 0.0040 * 500.0;
        let requested_would_be = 0.0005 * 500.0;
        let authorized_would_be = 0.0020 * 500.0;

        assert_eq!(clock.cumulative_simulated_time_ps(), applied);
        assert_eq!(rec.cumulative_simulated_time_ps_opt(), Some(applied));
        assert_ne!(
            rec.cumulative_simulated_time_ps_opt(),
            Some(requested_would_be)
        );
        assert_ne!(
            rec.cumulative_simulated_time_ps_opt(),
            Some(authorized_would_be)
        );
        assert_eq!(rec.global_md_step_opt(), Some(500));

        // All three gear facts survive independently.
        assert_eq!(rec.requested_gear_opt(), Some(CCR_GEAR_FINE_0P5FS));
        assert_eq!(rec.authorized_gear_opt(), Some(CCR_GEAR_DEFAULT_2FS));
        assert_eq!(rec.device_measured_gear_opt(), Some(CCR_GEAR_HMR_4FS));
    }

    #[test]
    fn clock_stalls_rather_than_fabricating_time_on_an_unmeasured_dt() {
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        let first = clock.commit(&d);
        let t_after_first = first.cumulative_simulated_time_ps_opt().unwrap();

        let mut d2 = healthy_decision();
        d2.dt_applied_ps = None;
        let second = clock.commit(&d2);

        assert!(second.has_flag(CCR_FLAG_CLOCK_STALLED));
        assert!(second.dt_applied_ps_opt().is_none());
        assert_eq!(
            second.cumulative_simulated_time_ps_opt(),
            Some(t_after_first)
        );
        assert_eq!(clock.stalled_intervals(), 1);
        // The step counter still advances: it counts MD steps, not physics.
        assert_eq!(second.global_md_step_opt(), Some(200));
    }

    #[test]
    fn gear_epoch_id_is_monotonic_and_increments_only_on_authorized_transitions() {
        let mut clock = CoupledChronologyClock::new();

        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        let r0 = clock.commit(&d);
        assert_eq!(r0.gear_epoch_id_opt(), Some(0));
        assert!(r0.has_flag(CCR_FLAG_GEAR_CHANGED));

        // Held.
        let r1 = clock.commit(&healthy_decision());
        assert_eq!(r1.gear_epoch_id_opt(), Some(0));
        assert!(r1.has_flag(CCR_FLAG_GEAR_HELD));
        assert!(!r1.has_flag(CCR_FLAG_GEAR_CHANGED));

        // Downshift to gear 0 — a transition.
        let mut d2 = healthy_decision();
        d2.requested_gear = Some(CCR_GEAR_FINE_0P5FS);
        d2.authorized_gear = Some(CCR_GEAR_FINE_0P5FS);
        d2.device_measured_gear = Some(CCR_GEAR_FINE_0P5FS);
        d2.dt_applied_ps = Some(0.0005);
        d2.transition_reason = CCR_REASON_DOWNSHIFT_SALIENCE;
        let r2 = clock.commit(&d2);
        assert_eq!(r2.gear_epoch_id_opt(), Some(1));
        assert!(r2.has_flag(CCR_FLAG_GEAR_CHANGED));

        // Held again at the new gear.
        let r3 = clock.commit(&d2);
        assert_eq!(r3.gear_epoch_id_opt(), Some(1));
        assert!(r3.has_flag(CCR_FLAG_GEAR_HELD));

        let epochs: Vec<u64> = [r0, r1, r2, r3]
            .iter()
            .map(|r| r.gear_epoch_id_opt().unwrap())
            .collect();
        assert_eq!(epochs, vec![0, 0, 1, 1]);
        for w in epochs.windows(2) {
            assert!(w[1] >= w[0], "gear_epoch_id must be monotonic");
        }
    }

    #[test]
    fn a_record_is_emitted_even_when_the_gear_is_held() {
        // EMIT ON NO-CHANGE: a gap must never be ambiguous between "gear
        // held" and "decision did not run".
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        let _ = clock.commit(&d);
        for _ in 0..5 {
            let r = clock.commit(&healthy_decision());
            assert!(r.has_flag(CCR_FLAG_DECISION_EXECUTED));
            assert!(r.has_flag(CCR_FLAG_GEAR_HELD));
        }
        assert_eq!(clock.records_emitted(), 6);
    }

    #[test]
    fn clock_resume_preserves_the_chronology() {
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        let _ = clock.commit(&d);
        let _ = clock.commit(&healthy_decision());

        let mut resumed = CoupledChronologyClock::resume(
            clock.step(),
            clock.cumulative_simulated_time_ps(),
            clock.gear_epoch_id(),
            clock.last_authorized_gear(),
            clock.next_record_seq(),
        );
        let r = resumed.commit(&healthy_decision());
        assert_eq!(r.global_md_step_opt(), Some(300));
        close(
            r.cumulative_simulated_time_ps_opt().unwrap(),
            0.0020 * 300.0,
        );
        assert!(r.has_flag(CCR_FLAG_GEAR_HELD));
        // The sequence must continue, not restart: a restarted sequence would
        // make a resumed run's own records look duplicated.
        assert_eq!(r.record_seq_opt(), Some(2));
    }

    // ── validation ─────────────────────────────────────────────────────────

    #[test]
    fn a_healthy_stream_validates_clean() {
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut records = Vec::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        records.push(clock.commit(&d));
        for _ in 0..4 {
            records.push(clock.commit(&healthy_decision()));
        }
        let v = validate_stream(&records, &policy);
        assert!(v.is_pass(), "unexpected findings: {:#?}", v.findings);
        assert_eq!(v.records_examined, 5);
        assert_eq!(v.warning_count(), 0);
    }

    #[test]
    fn validation_rejects_authorized_coarser_than_requested() {
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        // TSH demanded 0.5 fs; the governor "authorized" 4 fs. The gearbox
        // must apply min(demanded, permitted), so this is impossible.
        d.requested_gear = Some(CCR_GEAR_FINE_0P5FS);
        d.authorized_gear = Some(CCR_GEAR_HMR_4FS);
        d.device_measured_gear = Some(CCR_GEAR_HMR_4FS);
        d.dt_applied_ps = Some(0.0040);
        d.hmr = HmrVerification::Applied;
        let rec = clock.commit(&d);
        let v = validate_stream(&[rec], &policy);
        assert!(v.has("CCR_AUTHORIZED_COARSER_THAN_REQUESTED"));
        assert!(!v.is_pass());
    }

    #[test]
    fn validation_rejects_the_hmr_gear_without_verified_topology_permission() {
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        d.requested_gear = Some(CCR_GEAR_HMR_4FS);
        d.authorized_gear = Some(CCR_GEAR_HMR_4FS);
        d.device_measured_gear = Some(CCR_GEAR_HMR_4FS);
        d.dt_applied_ps = Some(0.0040);
        d.hmr = HmrVerification::Unverified;
        let v = validate_stream(&[clock.commit(&d)], &policy);
        assert!(v.has("CCR_HMR_GEAR_WITHOUT_VERIFIED_PERMISSION"));

        // With verified permission the same record is clean.
        let mut clock2 = CoupledChronologyClock::new();
        d.hmr = HmrVerification::Applied;
        let v2 = validate_stream(&[clock2.commit(&d)], &policy);
        assert!(!v2.has("CCR_HMR_GEAR_WITHOUT_VERIFIED_PERMISSION"));
        assert!(v2.is_pass(), "unexpected findings: {:#?}", v2.findings);
    }

    #[test]
    fn validation_rejects_coarsening_while_a_guard_asserts_or_observation_is_starved() {
        let policy = CoupledChronologyPolicy::source_literals();

        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_UPSHIFT_GRANTED;
        d.governor.guard_status = Some(CCR_GUARD_ENERGY_DRIFT);
        let rec = clock.commit(&d);
        assert_eq!(rec.highest_priority_guard(), Some("energy_drift"));
        let v = validate_stream(&[rec], &policy);
        assert!(v.has("CCR_UPSHIFT_WITH_GUARD_ASSERTED"));

        let mut clock2 = CoupledChronologyClock::new();
        let mut d2 = healthy_decision();
        d2.transition_reason = CCR_REASON_UPSHIFT_GRANTED;
        d2.governor.guard_status = Some(0);
        d2.extra_flags = CCR_FLAG_OBSERVATION_STARVED;
        let v2 = validate_stream(&[clock2.commit(&d2)], &policy);
        assert!(v2.has("CCR_UPSHIFT_WHILE_OBSERVATION_STARVED"));

        let mut clock3 = CoupledChronologyClock::new();
        let mut d3 = healthy_decision();
        d3.transition_reason = CCR_REASON_UPSHIFT_GRANTED;
        d3.governor.guard_status = None;
        let v3 = validate_stream(&[clock3.commit(&d3)], &policy);
        assert!(
            v3.has("CCR_UPSHIFT_WITH_UNMEASURED_GUARDS"),
            "safety must be VERIFIED, not merely unobserved"
        );
    }

    #[test]
    fn validation_rejects_a_dt_that_disagrees_with_the_measured_gear() {
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        d.device_measured_gear = Some(CCR_GEAR_HMR_4FS); // 4 fs
        d.dt_applied_ps = Some(0.0020); // but 2 fs was applied
        let v = validate_stream(&[clock.commit(&d)], &policy);
        assert!(v.has("CCR_DT_GEAR_INCONSISTENT"));
    }

    #[test]
    fn validation_detects_a_time_advance_that_the_applied_dt_cannot_explain() {
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        let r0 = clock.commit(&d);
        let mut r1 = clock.commit(&healthy_decision());
        // Forge elapsed physics that no applied timestep accounts for — the
        // signature of a lost record.
        r1.cumulative_simulated_time_ps += 0.5;
        let v = validate_stream(&[r0, r1], &policy);
        assert!(v.has("CCR_TIME_NOT_ADVANCED_BY_APPLIED_DT"));
    }

    #[test]
    fn validation_requires_an_unambiguous_gear_change_flag() {
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        let mut rec = clock.commit(&d);
        rec.record_flags |= CCR_FLAG_GEAR_HELD; // now both bits set
        let v = validate_stream(&[rec], &policy);
        assert!(v.has("CCR_GEAR_CHANGE_FLAG_AMBIGUOUS"));

        let mut rec2 = CoupledChronologyRecord::unmeasured();
        rec2.record_flags = 0; // neither bit, and no DECISION_EXECUTED
        let v2 = validate_stream(&[published(rec2)], &policy);
        assert!(v2.has("CCR_GEAR_CHANGE_FLAG_AMBIGUOUS"));
        assert!(v2.has("CCR_DECISION_FLAG_ABSENT"));
    }

    #[test]
    fn validation_rejects_an_unattributed_or_unpublished_record() {
        let policy = CoupledChronologyPolicy::source_literals();
        let rec = CoupledChronologyRecord::unmeasured();
        let v = validate_stream(&[rec], &policy);
        assert!(v.has("CCR_NOT_PUBLISHED"));
        assert!(v.has("CCR_STREAM_UNATTRIBUTED"));
        assert!(v.has("CCR_REPLICA_UNATTRIBUTED"));
        assert!(v.has("CCR_EMITTER_UNSPECIFIED"));
        assert!(v.has("CCR_GLOBAL_STEP_UNMEASURED"));
        assert!(v.has("CCR_AUTHORIZED_GEAR_UNMEASURED"));
        assert!(v.has("CCR_TRANSITION_REASON_NOT_RECORDED"));
        assert!(v.has("CCR_RECORD_SEQ_UNSTAMPED"));
        assert!(!v.is_pass());
    }

    #[test]
    fn validation_rejects_a_stream_that_mixes_emitters() {
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        let r0 = clock.commit(&d);
        let mut d1 = healthy_decision();
        d1.emitter_tag = CCR_EMITTER_LADD; // a different clock
        let r1 = clock.commit(&d1);
        let v = validate_stream(&[r0, r1], &policy);
        assert!(v.has("CCR_STREAM_NOT_HOMOGENEOUS"));
    }

    // ── policy ─────────────────────────────────────────────────────────────

    #[test]
    fn thresholds_record_their_resolved_value_and_origin() {
        let d = ResolvedThreshold::from_raw("t", "PRISM_CCR_TEST", 1e-6, None);
        assert_eq!(d.value, 1e-6);
        assert!(d.is_source_literal());
        assert_eq!(d.origin, "source_literal_default");

        let o = ResolvedThreshold::from_raw("t", "PRISM_CCR_TEST", 1e-6, Some(" 2.5e-4 "));
        assert_eq!(o.value, 2.5e-4);
        assert_eq!(o.source_literal, 1e-6);
        assert_eq!(o.origin, "env:PRISM_CCR_TEST");
        assert!(!o.is_source_literal());

        let bad = ResolvedThreshold::from_raw("t", "PRISM_CCR_TEST", 1e-6, Some("banana"));
        assert_eq!(bad.value, 1e-6);
        assert!(bad.origin.contains("unparseable"));

        let neg = ResolvedThreshold::from_raw("t", "PRISM_CCR_TEST", 1e-6, Some("-3"));
        assert_eq!(neg.value, 1e-6);
        assert!(neg.origin.contains("rejected_non_positive"));

        // The report carries the resolved values, so a decision is
        // reproducible without reading source.
        let policy = CoupledChronologyPolicy::source_literals();
        let v = validate_stream(&[], &policy);
        assert_eq!(
            v.policy.dt_gear_rel_tol.value,
            CCR_DT_GEAR_REL_TOL_SOURCE_LITERAL
        );
        assert_eq!(
            v.policy.time_continuity_rel_tol.value,
            CCR_TIME_CONTINUITY_REL_TOL_SOURCE_LITERAL
        );
        assert!(v.policy.dt_gear_rel_tol.is_source_literal());
    }

    // ── conservation ───────────────────────────────────────────────────────

    #[test]
    fn conservation_passes_on_a_lossless_stream() {
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut records = Vec::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        records.push(clock.commit(&d));
        for _ in 0..7 {
            records.push(clock.commit(&healthy_decision()));
        }
        let c = CoupledChronologyConservation::from_stream(8, 8, 0, &records, &policy);
        assert!(c.conservation_pass(), "failing: {:?}", c.failing_terms());
        assert_eq!(c.persisted, 8);
        assert_eq!(c.gaps, 0);
        assert_eq!(c.duplicates, 0);
        assert_eq!(c.continuity_unverifiable, 0);
        assert_eq!(c.first_global_md_step, Some(100));
        assert_eq!(c.last_global_md_step, Some(800));
        close(c.simulated_time_span_ps.unwrap(), 0.0020 * 700.0);
        assert_eq!(c.authorized_gears_observed, vec![CCR_GEAR_DEFAULT_2FS]);
        assert_eq!(c.observed_schema_version, CCR_SCHEMA_VERSION);
    }

    #[test]
    fn a_dropped_record_at_constant_gear_is_caught_by_the_sequence_not_continuity() {
        // This is the case that motivates carrying `record_seq` at all.
        //
        // With the gear HELD, dropping a record leaves the physical-time
        // continuity identity exactly satisfied: the survivors' step gap
        // doubles and the same constant dt reproduces the elapsed time. A
        // receipt that relied on continuity alone would call this stream
        // conserved.
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut records = Vec::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        records.push(clock.commit(&d));
        for _ in 0..3 {
            records.push(clock.commit(&healthy_decision()));
        }
        records.remove(2);

        let c = CoupledChronologyConservation::from_stream(4, 4, 0, &records, &policy);
        assert_eq!(c.gaps, 1, "the missing record_seq must be counted");
        assert_eq!(
            c.continuity_breaks, 0,
            "continuity is silent here — that is exactly why the sequence \
             term is required and not redundant"
        );
        assert!(!c.conservation_pass());
        assert!(c.failing_terms().contains(&"gaps"));
        assert!(c.failing_terms().contains(&"produced_ne_persisted"));

        // The stream validator names the same loss.
        let v = validate_stream(&records, &policy);
        assert!(v.has("CCR_RECORD_SEQ_GAP"));
        assert!(!v.has("CCR_TIME_NOT_ADVANCED_BY_APPLIED_DT"));
    }

    #[test]
    fn forged_elapsed_time_is_caught_by_continuity_even_with_a_dense_sequence() {
        // The converse case: the sequence is perfectly dense, so the
        // completeness term is silent, but the physics does not add up.
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        let r0 = clock.commit(&d);
        let mut r1 = clock.commit(&healthy_decision());
        r1.cumulative_simulated_time_ps += 0.5;

        let c = CoupledChronologyConservation::from_stream(2, 2, 0, &[r0, r1], &policy);
        assert_eq!(c.gaps, 0, "sequence is dense");
        assert_eq!(c.continuity_breaks, 1);
        assert!(!c.conservation_pass());
        assert!(c.failing_terms().contains(&"continuity_breaks"));
    }

    #[test]
    fn conservation_detects_duplicates_backpressure_and_unverifiable_continuity() {
        let policy = CoupledChronologyPolicy::source_literals();
        let mut clock = CoupledChronologyClock::new();
        let mut d = healthy_decision();
        d.transition_reason = CCR_REASON_INITIALIZATION;
        let r0 = clock.commit(&d);
        let r1 = r0; // a re-read of the same slot without a fresh stamp
        let c = CoupledChronologyConservation::from_stream(2, 2, 0, &[r0, r1], &policy);
        assert_eq!(c.duplicates, 2, "both the sequence and the step repeated");
        assert!(!c.conservation_pass());
        let v = validate_stream(&[r0, r1], &policy);
        assert!(v.has("CCR_RECORD_SEQ_NOT_MONOTONIC"));
        assert!(v.has("CCR_STEP_NOT_MONOTONIC"));

        // Backpressure is idleness, not loss: expected == produced + events.
        let mut clock2 = CoupledChronologyClock::new();
        let mut records = Vec::new();
        let mut d2 = healthy_decision();
        d2.transition_reason = CCR_REASON_INITIALIZATION;
        records.push(clock2.commit(&d2));
        records.push(clock2.commit(&healthy_decision()));
        let ok = CoupledChronologyConservation::from_stream(5, 2, 3, &records, &policy);
        assert!(ok.conservation_pass(), "failing: {:?}", ok.failing_terms());

        // A stalled clock makes continuity unverifiable, and an unverifiable
        // chronology is not a conserved chronology.
        let mut clock3 = CoupledChronologyClock::new();
        let mut stalled = Vec::new();
        let mut d3 = healthy_decision();
        d3.transition_reason = CCR_REASON_INITIALIZATION;
        stalled.push(clock3.commit(&d3));
        let mut d4 = healthy_decision();
        d4.dt_applied_ps = None;
        stalled.push(clock3.commit(&d4));
        let s = CoupledChronologyConservation::from_stream(2, 2, 0, &stalled, &policy);
        assert_eq!(s.clock_stalled_records, 1);
        assert_eq!(s.continuity_unverifiable, 1);
        assert!(!s.conservation_pass());
        assert!(s.failing_terms().contains(&"continuity_unverifiable"));
    }

    #[test]
    fn aborted_conservation_receipt_never_passes() {
        let c = CoupledChronologyConservation::aborted("producer never started");
        assert!(!c.conservation_pass());
        assert!(c.failing_terms().contains(&"terminal_error"));
    }
}

#[cfg(all(test, feature = "gpu"))]
mod spike_clock_binding_tests {
    use super::*;
    use crate::fused_engine::GpuSpikeEvent;

    /// The host-supplied chunk context transfers verbatim, and the record's own
    /// `spike_source` becomes the emitter tag. Nothing else may be stated.
    #[test]
    fn host_chunk_context_and_emitter_transfer_into_the_chronology_record() {
        let spike = GpuSpikeEvent {
            spike_source: 2, // LIF
            ..GpuSpikeEvent::default()
        };
        let r = CoupledChronologyRecord::from_spike_event(&spike, 3, 11);
        assert_eq!(r.stream_id_opt(), Some(3));
        assert_eq!(r.record_seq_opt(), Some(11));
        assert_eq!(r.emitter_tag, CCR_EMITTER_LIF);
        // A spike record carries no gear, TSH or governor evidence, so those
        // MUST stay unmeasured rather than acquire a plausible default.
        assert_eq!(r.authorized_gear_opt(), None);
        assert_eq!(r.device_measured_gear_opt(), None);
        assert_eq!(r.requested_gear_opt(), None);
        assert_eq!(r.dt_applied_ps_opt(), None);
        assert_eq!(r.force_norm_opt(), None);
        assert_eq!(r.replica_id_opt(), None);
    }

    /// The 96 B spike record carries NO global clock, so a spike-bound CCR must
    /// leave both clock fields at the CCR's own out-of-band sentinels. A
    /// plausible 0 here would be indistinguishable from a measured step 0.
    #[test]
    fn spike_bound_record_never_asserts_a_global_clock() {
        let r = CoupledChronologyRecord::from_spike_event(&GpuSpikeEvent::default(), 0, 0);
        assert_eq!(r.global_md_step_opt(), None);
        assert_eq!(r.global_md_step, CCR_U64_UNMEASURED);
        assert_ne!(r.global_md_step, 0);
        assert_eq!(r.cumulative_simulated_time_ps_opt(), None);
        // CLOCK_STALLED means "did not advance", which the governor observes.
        // An absent clock is not a stall, and stamping it on every spike-bound
        // record would make the flag meaningless.
        assert!(!r.has_flag(CCR_FLAG_CLOCK_STALLED));
    }

    /// Stream 0 / chunk 0 are LEGITIMATE host facts and must survive as
    /// measured, while a caller with no chunk context can say so by passing the
    /// CCR sentinels.
    #[test]
    fn zero_valued_chunk_context_survives_as_measurement() {
        let spike = GpuSpikeEvent {
            spike_source: 1,
            ..GpuSpikeEvent::default()
        };
        let r = CoupledChronologyRecord::from_spike_event(&spike, 0, 0);
        assert_eq!(r.stream_id_opt(), Some(0));
        assert_eq!(r.record_seq_opt(), Some(0));
        assert_ne!(r.stream_id, CCR_U32_UNMEASURED);

        let unbound = CoupledChronologyRecord::from_spike_event(
            &spike,
            CCR_U32_UNMEASURED,
            CCR_U64_UNMEASURED,
        );
        assert_eq!(unbound.stream_id_opt(), None);
        assert_eq!(unbound.record_seq_opt(), None);
    }

    /// Emitter tags must be exact. UV/LIF/EFP and LADD are stamped from
    /// different clocks, so guessing a neighbouring tag is worse than
    /// declining to tag.
    #[test]
    fn emitter_tag_mapping_is_exact_and_refuses_to_guess() {
        assert_eq!(spike_source_to_emitter_tag(1), CCR_EMITTER_UV);
        assert_eq!(spike_source_to_emitter_tag(2), CCR_EMITTER_LIF);
        assert_eq!(spike_source_to_emitter_tag(3), CCR_EMITTER_EFP);
        assert_eq!(spike_source_to_emitter_tag(4), CCR_EMITTER_LADD);
        assert_eq!(spike_source_to_emitter_tag(5), CCR_EMITTER_COFIRE);
        for unknown in [-1, 0, 6, 99] {
            assert_eq!(
                spike_source_to_emitter_tag(unknown),
                CCR_EMITTER_UNSPECIFIED,
                "unknown spike_source {unknown} must not be given a plausible tag"
            );
        }
    }

    /// The chronology sidecar is a wire format: it must survive the byte
    /// round trip the offline reader performs.
    #[test]
    fn spike_bound_record_round_trips_through_the_wire_format() {
        let spike = GpuSpikeEvent {
            spike_source: 4, // LADD
            ..GpuSpikeEvent::default()
        };
        let r = CoupledChronologyRecord::from_spike_event(&spike, 1, 4);
        let bytes = r.to_bytes();
        assert_eq!(bytes.len(), CCR_RECORD_BYTES);
        let back = CoupledChronologyRecord::from_bytes(&bytes).expect("CCR decode");
        assert_eq!(back.global_md_step_opt(), r.global_md_step_opt());
        assert_eq!(
            back.cumulative_simulated_time_ps_opt(),
            r.cumulative_simulated_time_ps_opt()
        );
        assert_eq!(back.stream_id_opt(), Some(1));
        assert_eq!(back.emitter_tag, CCR_EMITTER_LADD);
        assert_eq!(back.record_seq_opt(), Some(4));
    }
}
