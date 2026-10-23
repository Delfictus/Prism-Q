//! Gate G3 — `phase_bits` (u32) layout + decoder.
//!
//! `phase_bits` is the per-spike `u32` field emitted into the
//! `spike_events.arrow` schema (see [`crate::spike_arrow_writer`]'s
//! `phase_bits` column at the schema's column 16) and carried on
//! [`crate::fused_engine::GpuSpikeEvent::phase_bits`].
//!
//! # Authoritative layout (current implementation)
//!
//! Bits  0–9  : **CCNS phase index** (`u10`, range 0–1023). Quantized
//!              representation of the per-spike CCNS protocol phase
//!              angle, written by the engine's protocol-state Director
//!              each step (see `crate::protocol_state::ProtocolState
//!              ::current_phase_bits` at line 138). The continuous
//!              angle in radians is recovered by
//!              `phi_radians = (phase_index as f64 / 1024.0) * 2*PI`.
//!              The denominator is 1024 (not 1023) because the angle
//!              wraps cyclically: index 1024 ~ index 0 ~ phase 0.
//!              Pinned by `crate::bin::nhs_rt_full.rs:4559`
//!              (`let phi = (spike.phase_bits as f64 / 1024.0) * two_pi;`).
//!
//! Bits 10–31 : **Reserved / unused** in the current implementation.
//!              Read-as-zero. Future extensions that pack additional
//!              fields into these bits (e.g., SDST hysteresis-phase
//!              flags, protocol-stage indicators, alternative phase
//!              axis encodings) MUST update both this module's
//!              documented layout and the consumer-facing schema doc
//!              at `docs/phase_bits_schema.md` in the same commit.
//!
//! # What is NOT in `phase_bits` (frequently confused)
//!
//! * **CCNS 4-class phase** (cold_hold / ramp / warm_hold / cooling):
//!   carried in the SEPARATE `ccns_phase: u8` Arrow column (column 24),
//!   NOT packed into `phase_bits`. Both fields describe CCNS phase but
//!   at different resolutions: `phase_bits` is the 10-bit continuous
//!   angle quantization, `ccns_phase` is the 4-class enum.
//!
//! * **SDST hysteresis-phase fractions** (`sdst_cold_fraction`,
//!   `sdst_hot_fraction`, `sdst_delta`): carried per-site in the
//!   `binding_sites.json sites[].phase{}` sub-dict (emitted at
//!   `crate::bin::nhs_rt_full.rs:10846–10868`), NOT per-spike, and NOT
//!   packed into `phase_bits`.
//!
//! * **Protocol stage flags**: tracked in
//!   `crate::protocol_state::ProtocolState`, exposed through audit
//!   sidecars (`<prefix>.topology.asc_events.bin`,
//!   `<prefix>.topology.acl_contrast.bin`), NOT packed into
//!   `phase_bits`.
//!
//! # Compile-time enforcement of layout
//!
//! The constants below pin the bit positions and masks. Any future
//! extension that uses bits 10+ MUST update `RESERVED_MASK` and
//! introduce its own named mask alongside, AND surface the change in
//! the schema doc. The unit tests in this module pin the current
//! layout via concrete decode results.

use serde::{Deserialize, Serialize};

/// Bit width of the CCNS phase index field.
pub const CCNS_PHASE_INDEX_BITS: u32 = 10;
/// Maximum value of the phase index (`2^10 - 1 = 1023`).
pub const CCNS_PHASE_INDEX_MAX: u32 = (1 << CCNS_PHASE_INDEX_BITS) - 1;
/// Cyclic period used to recover the continuous phase angle. `1024`
/// (not `1023`) because index `1024 ≡ index 0 (mod 2*PI)`.
pub const CCNS_PHASE_PERIOD: u32 = 1 << CCNS_PHASE_INDEX_BITS;
/// Bit-mask for the CCNS phase index field (`0x000003FF`).
pub const CCNS_PHASE_INDEX_MASK: u32 = CCNS_PHASE_INDEX_MAX;
/// Bit-mask for the reserved bits (`0xFFFFFC00`). MUST be zero in any
/// `phase_bits` value emitted by the current producer; verified by
/// [`PhaseBitsDecoded::reserved`] at decode time.
pub const RESERVED_MASK: u32 = !CCNS_PHASE_INDEX_MASK;

/// Decoded representation of a `phase_bits: u32` value.
///
/// Returned by [`decode_phase_bits`]. Carries both the integer index
/// form (for histogramming / bincount) and the continuous angle form
/// (for trig-based analyses), plus the reserved-bits payload so a
/// future extension that packs additional fields into bits 10+ does
/// not silently lose the data.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseBitsDecoded {
    /// CCNS phase index in `[0, 1023]`. Quantized representation of
    /// the protocol phase angle.
    pub phase_index: u16,
    /// Continuous phase angle in radians, in `[0, 2*PI)`. Computed as
    /// `(phase_index / 1024.0) * 2*PI`.
    pub phase_radians: f64,
    /// Reserved bits 10–31, right-shifted so the value occupies the
    /// low 22 bits of the returned `u32`. In the current
    /// implementation this is always `0`. Future schema versions that
    /// pack additional fields here will document the sub-layout in
    /// `docs/phase_bits_schema.md` and add named accessors to this
    /// struct.
    pub reserved: u32,
}

impl PhaseBitsDecoded {
    /// True iff the value's reserved bits are zero (the only valid
    /// state under the current schema). False values flag either a
    /// schema upgrade in flight or a corrupted payload — both should
    /// be surfaced explicitly to the caller rather than silently
    /// ignored.
    #[inline]
    pub fn reserved_is_clean(&self) -> bool {
        self.reserved == 0
    }
}

/// Decode a raw `phase_bits: u32` into named fields.
///
/// Pure function; no I/O, no allocations. The phase-radians
/// computation uses `f64` to preserve precision when the index is
/// small (e.g., index `1` → `0.00613...` rad would lose digits in
/// `f32`).
#[inline]
pub fn decode_phase_bits(raw: u32) -> PhaseBitsDecoded {
    let phase_index = (raw & CCNS_PHASE_INDEX_MASK) as u16;
    let reserved = (raw & RESERVED_MASK) >> CCNS_PHASE_INDEX_BITS;
    let phase_radians = (phase_index as f64 / CCNS_PHASE_PERIOD as f64) * std::f64::consts::TAU;
    PhaseBitsDecoded {
        phase_index,
        phase_radians,
        reserved,
    }
}

/// Inverse of [`decode_phase_bits`]: pack a phase index (and
/// optionally a reserved-bits payload) into a `u32`. Useful for
/// constructing test fixtures and for any future writer that needs
/// to emit `phase_bits` values from named fields rather than raw
/// integers. Asserts the phase index is in range to catch caller
/// bugs at construction site rather than at decode time.
#[inline]
pub fn encode_phase_bits(phase_index: u16, reserved_payload: u32) -> u32 {
    debug_assert!(
        (phase_index as u32) <= CCNS_PHASE_INDEX_MAX,
        "phase_index {} exceeds CCNS_PHASE_INDEX_MAX {}",
        phase_index,
        CCNS_PHASE_INDEX_MAX
    );
    let reserved_payload_max = (1u32 << (32 - CCNS_PHASE_INDEX_BITS)) - 1;
    debug_assert!(
        reserved_payload <= reserved_payload_max,
        "reserved_payload {} exceeds 22-bit max {}",
        reserved_payload,
        reserved_payload_max
    );
    (phase_index as u32) | (reserved_payload << CCNS_PHASE_INDEX_BITS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_zero_yields_phase_zero() {
        let d = decode_phase_bits(0);
        assert_eq!(d.phase_index, 0);
        assert_eq!(d.phase_radians, 0.0);
        assert_eq!(d.reserved, 0);
        assert!(d.reserved_is_clean());
    }

    #[test]
    fn decode_max_index_is_one_step_below_two_pi() {
        // index 1023 → angle = 1023/1024 * 2π ≈ 6.2769466... rad
        let d = decode_phase_bits(1023);
        assert_eq!(d.phase_index, 1023);
        assert_eq!(d.reserved, 0);
        let expected = (1023.0 / 1024.0) * std::f64::consts::TAU;
        assert!((d.phase_radians - expected).abs() < 1e-12);
    }

    #[test]
    fn decode_quarter_phase() {
        // index 256 = 1024/4 → angle = π/2
        let d = decode_phase_bits(256);
        assert_eq!(d.phase_index, 256);
        assert!((d.phase_radians - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn decode_half_phase() {
        // index 512 = 1024/2 → angle = π
        let d = decode_phase_bits(512);
        assert_eq!(d.phase_index, 512);
        assert!((d.phase_radians - std::f64::consts::PI).abs() < 1e-12);
    }

    #[test]
    fn decode_pinned_value_from_canonical_run() {
        // Pin a concrete value from the m1_readiness_verify run: a
        // hypothetical spike with phase_bits = 410 should decode to
        // index 410 and angle 410/1024 * 2π = 2.5158... rad.
        let d = decode_phase_bits(410);
        assert_eq!(d.phase_index, 410);
        assert_eq!(d.reserved, 0);
        let expected = (410.0 / 1024.0) * std::f64::consts::TAU;
        assert!((d.phase_radians - expected).abs() < 1e-12);
    }

    #[test]
    fn decode_with_reserved_bits_set_recovers_payload() {
        // Future-proofing: if a future schema uses bits 10+, the
        // current decoder still recovers the phase index correctly
        // and exposes the reserved payload to the caller rather than
        // silently dropping it.
        //
        // Construct via encode_phase_bits to guarantee no overlap
        // between phase index and reserved payload bits.
        let phase_index: u16 = 0x1AB; // = 427
        let reserved_payload: u32 = 0x12345; // arbitrary non-zero
        let raw = encode_phase_bits(phase_index, reserved_payload);
        let d = decode_phase_bits(raw);
        assert_eq!(d.phase_index, phase_index);
        assert_eq!(d.reserved, reserved_payload);
        assert!(!d.reserved_is_clean());
    }

    #[test]
    fn encode_decode_roundtrip_clean() {
        for idx in [0u16, 1, 100, 256, 511, 512, 1023] {
            let raw = encode_phase_bits(idx, 0);
            let d = decode_phase_bits(raw);
            assert_eq!(d.phase_index, idx);
            assert_eq!(d.reserved, 0);
        }
    }

    #[test]
    fn mask_constants_pin_layout() {
        // Pin the layout constants. Any change to these without a
        // corresponding update to docs/phase_bits_schema.md is a
        // schema-drift bug.
        assert_eq!(CCNS_PHASE_INDEX_BITS, 10);
        assert_eq!(CCNS_PHASE_INDEX_MAX, 1023);
        assert_eq!(CCNS_PHASE_PERIOD, 1024);
        assert_eq!(CCNS_PHASE_INDEX_MASK, 0x000003FF);
        assert_eq!(RESERVED_MASK, 0xFFFFFC00);
        // Disjoint coverage of the full u32:
        assert_eq!(CCNS_PHASE_INDEX_MASK | RESERVED_MASK, u32::MAX);
        assert_eq!(CCNS_PHASE_INDEX_MASK & RESERVED_MASK, 0);
    }
}
