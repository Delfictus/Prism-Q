//! Typed record layouts (§5). The `GpuSpikeEvent` is the spike-packet payload record.
//!
//! Layout is the engine's `#[repr(C, align(32))]` struct from
//! `crates/prism-nhs/src/fused_engine.rs` (verbatim offsets), parsed little-endian:
//! 24 four-byte fields = exactly 96 B, no tail padding.
//!
//! # One generation
//!
//! The record carries ONE temporal field, `timestep`, stamped from the emitting
//! kernel's PROTOCOL-LOCAL counter. It is NOT a frame index and the record makes
//! no claim about which global clock it corresponds to.
//!
//! Clock provenance (global MD step, cumulative simulated time, stream/chunk
//! identity, clock domain) is deliberately NOT in the record. An in-record
//! global step only ever resolved to REPLAY granularity anyway — the spike is
//! emitted inside a body iteration while the global counter advances once per
//! replay — and there is at most one coordinate frame per chunk, so
//! chunk-resolution provenance already IS frame-resolution provenance. That
//! binding therefore lives in the per-chunk governor sidecar on the host side,
//! joined to these records by the envelope, not carried per record.
//!
//! A decoder MUST still dispatch on the envelope's declared `byte_stride`: an
//! unknown stride is a hard error, never a best-effort parse, because bytes
//! 0..95 of a misaligned record still look like a plausible spike.

/// Size of one serialized `GpuSpikeEvent` record.
pub const GPU_SPIKE_EVENT_BYTES: usize = 96;

/// True for a record stride this decoder actually understands.
pub fn known_record_stride(stride: usize) -> bool {
    stride == GPU_SPIKE_EVENT_BYTES
}

/// A decoded spike telemetry record (A1 SPIKE_EVENT source form).
///
/// `PartialEq` is BYTEWISE, not IEEE. The record carries float observables
/// (`intensity`, `water_density`, `wd_change`, …) that a device emission may
/// legitimately leave as NaN, and IEEE NaN compares unequal to itself — so a
/// derived `PartialEq` would report that a record does not equal its own
/// byte-exact round trip. For a byte-lossless evidence record the correct
/// identity is "the same bytes", which is what this implements.
#[derive(Clone, Copy, Debug)]
pub struct GpuSpikeEvent {
    pub timestep: i32,             // @0  PROTOCOL-LOCAL clock (not a frame index)
    pub voxel_idx: i32,            // @4
    pub position: [f32; 3],        // @8  (x,y,z engine-space Å)
    pub intensity: f32,            // @20 emission a.u.
    pub nearby_residues: [i32; 8], // @24
    pub n_residues: i32,           // @56
    pub spike_source: i32,         // @60 1=UV 2=LIF 3=EFP 4=LADD 5=COFIRE 6=VORONOI_LIF
    pub wavelength_nm: f32,        // @64 280 = Trp/Tyr line
    pub aromatic_type: i32,        // @68 0=TRP 1=TYR 2=PHE 3=SS -1=none
    pub aromatic_residue_id: i32,  // @72 -1 = unresolved sentinel
    pub water_density: f32,        // @76
    pub vibrational_energy: f32,   // @80
    pub n_nearby_excited: i32,     // @84
    pub wd_change: f32,            // @88
    pub phase_bits: u32,           // @92 10-bit CCNS phase (0..1023 → 0..2π)
}

impl PartialEq for GpuSpikeEvent {
    fn eq(&self, other: &Self) -> bool {
        self.to_le_bytes() == other.to_le_bytes()
    }
}
impl Eq for GpuSpikeEvent {}

#[inline]
fn i32_at(b: &[u8], o: usize) -> i32 {
    i32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}
#[inline]
fn f32_at(b: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

impl GpuSpikeEvent {
    /// Parse one record of `stride` bytes from the head of `b` (little-endian).
    ///
    /// Returns `None` when `stride` is not a stride this decoder understands, or
    /// when `b` is shorter than one record. Callers MUST treat `None` as a hard
    /// failure — a partial parse of an unknown layout produces plausible
    /// garbage, which is exactly the failure mode this stride check exists to
    /// prevent.
    pub fn from_le_bytes_with_stride(b: &[u8], stride: usize) -> Option<GpuSpikeEvent> {
        if !known_record_stride(stride) || b.len() < stride {
            return None;
        }
        let mut nearby_residues = [0i32; 8];
        for (k, slot) in nearby_residues.iter_mut().enumerate() {
            *slot = i32_at(b, 24 + 4 * k);
        }
        Some(GpuSpikeEvent {
            timestep: i32_at(b, 0),
            voxel_idx: i32_at(b, 4),
            position: [f32_at(b, 8), f32_at(b, 12), f32_at(b, 16)],
            intensity: f32_at(b, 20),
            nearby_residues,
            n_residues: i32_at(b, 56),
            spike_source: i32_at(b, 60),
            wavelength_nm: f32_at(b, 64),
            aromatic_type: i32_at(b, 68),
            aromatic_residue_id: i32_at(b, 72),
            water_density: f32_at(b, 76),
            vibrational_energy: f32_at(b, 80),
            n_nearby_excited: i32_at(b, 84),
            wd_change: f32_at(b, 88),
            phase_bits: u32_at(b, 92),
        })
    }

    /// Parse one 96 B record. Equivalent to
    /// [`GpuSpikeEvent::from_le_bytes_with_stride`] at [`GPU_SPIKE_EVENT_BYTES`];
    /// use that form when an envelope declares a stride, so a foreign layout is
    /// refused rather than silently reinterpreted.
    pub fn from_le_bytes(b: &[u8]) -> Option<GpuSpikeEvent> {
        Self::from_le_bytes_with_stride(b, GPU_SPIKE_EVENT_BYTES)
    }

    /// A record with every field zeroed. Test/round-trip support only — an
    /// all-zero record is a legitimate measurement, not a "missing" marker.
    pub fn zeroed() -> GpuSpikeEvent {
        GpuSpikeEvent {
            timestep: 0,
            voxel_idx: 0,
            position: [0.0; 3],
            intensity: 0.0,
            nearby_residues: [0; 8],
            n_residues: 0,
            spike_source: 0,
            wavelength_nm: 0.0,
            aromatic_type: 0,
            aromatic_residue_id: 0,
            water_density: 0.0,
            vibrational_energy: 0.0,
            n_nearby_excited: 0,
            wd_change: 0.0,
            phase_bits: 0,
        }
    }

    /// Serialize back to 96 little-endian bytes (round-trip / test support).
    pub fn to_le_bytes(&self) -> [u8; GPU_SPIKE_EVENT_BYTES] {
        let mut b = [0u8; GPU_SPIKE_EVENT_BYTES];
        b[0..4].copy_from_slice(&self.timestep.to_le_bytes());
        b[4..8].copy_from_slice(&self.voxel_idx.to_le_bytes());
        for k in 0..3 {
            b[8 + 4 * k..12 + 4 * k].copy_from_slice(&self.position[k].to_le_bytes());
        }
        b[20..24].copy_from_slice(&self.intensity.to_le_bytes());
        for k in 0..8 {
            b[24 + 4 * k..28 + 4 * k].copy_from_slice(&self.nearby_residues[k].to_le_bytes());
        }
        b[56..60].copy_from_slice(&self.n_residues.to_le_bytes());
        b[60..64].copy_from_slice(&self.spike_source.to_le_bytes());
        b[64..68].copy_from_slice(&self.wavelength_nm.to_le_bytes());
        b[68..72].copy_from_slice(&self.aromatic_type.to_le_bytes());
        b[72..76].copy_from_slice(&self.aromatic_residue_id.to_le_bytes());
        b[76..80].copy_from_slice(&self.water_density.to_le_bytes());
        b[80..84].copy_from_slice(&self.vibrational_energy.to_le_bytes());
        b[84..88].copy_from_slice(&self.n_nearby_excited.to_le_bytes());
        b[88..92].copy_from_slice(&self.wd_change.to_le_bytes());
        b[92..96].copy_from_slice(&self.phase_bits.to_le_bytes());
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GpuSpikeEvent {
        GpuSpikeEvent {
            timestep: 1234,
            voxel_idx: -7,
            position: [1.5, -2.25, 3.75],
            intensity: 42.0,
            nearby_residues: [10, 11, 12, 13, 14, 15, 16, 17],
            n_residues: 8,
            spike_source: 1,
            wavelength_nm: 280.0,
            aromatic_type: 0,
            aromatic_residue_id: -1, // unresolved sentinel
            water_density: 0.91,
            vibrational_energy: 5.0,
            n_nearby_excited: 3,
            wd_change: 0.02,
            phase_bits: 1023,
        }
    }

    #[test]
    fn record_is_96_bytes_and_round_trips() {
        let ev = sample();
        let bytes = ev.to_le_bytes();
        assert_eq!(bytes.len(), 96);
        assert_eq!(GPU_SPIKE_EVENT_BYTES, 96);
        assert_eq!(GpuSpikeEvent::from_le_bytes(&bytes), Some(ev));
    }

    #[test]
    fn known_offsets() {
        // spike_source at 60, wavelength_nm at 64, phase_bits at 92 — the last
        // four bytes of the record, so the struct ends exactly at 96 with no pad.
        let mut b = [0u8; 96];
        b[60..64].copy_from_slice(&5i32.to_le_bytes());
        b[64..68].copy_from_slice(&280.0f32.to_le_bytes());
        b[92..96].copy_from_slice(&1023u32.to_le_bytes());
        let ev = GpuSpikeEvent::from_le_bytes(&b).unwrap();
        assert_eq!(ev.spike_source, 5);
        assert_eq!(ev.wavelength_nm, 280.0);
        assert_eq!(ev.phase_bits, 1023);
    }

    /// An unknown stride must FAIL, not best-effort parse. Bytes 0..95 of a
    /// misaligned record still look like a plausible spike, so a permissive
    /// decoder would emit fabricated physics with no diagnostic.
    #[test]
    fn unknown_stride_is_refused() {
        let b = [0u8; 256];
        assert!(GpuSpikeEvent::from_le_bytes_with_stride(&b, 112).is_none());
        assert!(GpuSpikeEvent::from_le_bytes_with_stride(&b, 128).is_none());
        assert!(GpuSpikeEvent::from_le_bytes_with_stride(&b, 64).is_none());
        assert!(GpuSpikeEvent::from_le_bytes_with_stride(&b, 0).is_none());
        assert!(!known_record_stride(112));
        assert!(!known_record_stride(128));
        assert!(known_record_stride(96));
    }

    #[test]
    fn short_slice_is_none() {
        assert_eq!(GpuSpikeEvent::from_le_bytes(&[0u8; 95]), None);
        assert!(
            GpuSpikeEvent::from_le_bytes_with_stride(&[0u8; 95], GPU_SPIKE_EVENT_BYTES).is_none()
        );
    }
}
