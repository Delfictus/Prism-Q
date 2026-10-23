//! RichSpike — 64-byte cache-line-aligned event record.
//!
//! Per the PRISM-4D Continuous Learning Architecture mandate §1
//! (operator directive 2026-04-29). Replaces the legacy 16-byte
//! spike with a four-plane 64-byte schema that captures the full
//! state of the neuromorphic event generator at the moment of
//! emission.
//!
//! # Why 64 bytes
//!
//! Aligns to the Blackwell sm_120 cache line. A single
//! `LDG.E.128` instruction pulls 16 bytes (one plane) into
//! registers in one transaction; four such loads cover the full
//! spike with zero partial-line writes.
//!
//! # Layout (4 × 16 B planes)
//!
//! | Plane | Field | Size | Purpose |
//! |---|---|---|---|
//! | 1 | `pos[3]` (x, y, z) | 12 B | Spatial coordinates (Å) |
//! | 1 | `t_frame` | 4 B | Frame index within CCNS phase |
//! | 2 | `water_density` | 4 B | Local ρ_w |
//! | 2 | `wd_change` | 4 B | d(ρ_w)/dt — solvation flux |
//! | 2 | `vib_energy` | 4 B | Local vibrational stress |
//! | 2 | `intensity_packed` | 4 B | 8-bit pct + 24-bit intensity |
//! | 3 | `residue_id` | 4 B | Authoritative anchor (Mandate §M2) |
//! | 3 | `cluster_id` | 4 B | LBVH-assigned cluster |
//! | 3 | `causal_lag` | 4 B | Cross-correlation lag |
//! | 3 | `n_excited` | 4 B | Multi-neuron excitation count |
//! | 4 | `origin_phase` | 4 B | CCNS phase (Cold/Heat/Warm/Cool) |
//! | 4 | `spike_source` | 4 B | LIF / UV / EFP / LADD / COFIRE |
//! | 4 | `chem_flags` | 4 B | Bit-packed pharmacophore |
//! | 4 | `kinetic_delta` | 4 B | ΔE excursion value |
//!
//! # Composite sort key (forward declaration)
//!
//! Per Mandate §1.3 the 128-bit Karras-tree sort key for LBVH-2 is
//! `[30-bit Morton | 32-bit residue_id | 66-bit feature_hash]`. The
//! 64-bit feature-hash component is computed by
//! [`cpu_rich_spike_feature_hash`] (CPU reference) /
//! `prism_rich_spike_feature_hash` (GPU `__host__ __device__`). The
//! upper 2 bits of the 66-bit conceptual hash come from a
//! chain_id + atom_index discriminator (handled at sort-key
//! assembly time in CLA-1b).
//!
//! # V3-style verification posture
//!
//! Every helper here is a literal Rust port of the corresponding
//! `__host__ __device__` function in `rich_spike.cuh`. The
//! `cpu_gpu_*` parity tests pin the CPU/GPU output bit-for-bit.

use serde::{Deserialize, Serialize};

// ============================================================================
// RichSpike (#[repr(C, align(64))])
// ============================================================================

/// 64-byte cache-line-aligned event record. Layout-pinned by the
/// `rich_spike_layout_is_64_bytes_64_aligned` test below.
///
/// Field order matches the C-side `struct RichSpike` byte-for-byte.
/// Changing any field order or type without the corresponding C-side
/// change is an FFI break; the layout pin will catch it at test time.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RichSpike {
    // Plane 1 — Spatiotemporal Coordinates (16 B)
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub t_frame: u32,

    // Plane 2 — Thermodynamic Gradients (16 B)
    pub water_density: f32,
    pub wd_change: f32,
    pub vib_energy: f32,
    pub intensity_packed: u32,

    // Plane 3 — Causal & Neuromorphic Metadata (16 B)
    pub residue_id: i32,
    pub cluster_id: i32,
    pub causal_lag: f32,
    pub n_excited: u32,

    // Plane 4 — Provenance & Chemical Tags (16 B)
    pub origin_phase: u32,
    pub spike_source: u32,
    pub chem_flags: u32,
    pub kinetic_delta: f32,
}

impl RichSpike {
    /// Sentinel for `cluster_id` when the LBVH has not yet assigned
    /// the spike. Matches `RICH_SPIKE_UNCLUSTERED_ID` in the .cuh.
    pub const UNCLUSTERED_ID: i32 = -1;

    /// Sentinel for `residue_id` when the spike's anchor residue is
    /// unresolved (e.g., the spike came from a solvent voxel).
    pub const UNRESOLVED_RESIDUE: i32 = -1;

    /// Construct a zeroed RichSpike — all fields at their type's
    /// zero. Convenient for tests and as a default base for builders.
    pub const fn zero() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            t_frame: 0,
            water_density: 0.0,
            wd_change: 0.0,
            vib_energy: 0.0,
            intensity_packed: 0,
            residue_id: 0,
            cluster_id: 0,
            causal_lag: 0.0,
            n_excited: 0,
            origin_phase: 0,
            spike_source: 0,
            chem_flags: 0,
            kinetic_delta: 0.0,
        }
    }
}

impl Default for RichSpike {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// Intensity-pack helpers (CPU reference; bit-equivalent to .cuh)
// ============================================================================

/// Pack an 8-bit percentile rank (0..255) and a 24-bit intensity
/// value (0..16777215) into a single u32. Layout: bits 31..24 =
/// percentile, bits 23..0 = intensity. Bit-equivalent to
/// `prism_rich_spike_pack_intensity` in `rich_spike.cuh`.
#[inline]
pub fn cpu_pack_intensity(percentile_8: u32, intensity_24: u32) -> u32 {
    let p = percentile_8 & 0xFF;
    let i = intensity_24 & 0x00FF_FFFF;
    (p << 24) | i
}

/// Unpack the 8-bit percentile rank from a packed u32.
#[inline]
pub fn cpu_unpack_percentile(packed: u32) -> u32 {
    (packed >> 24) & 0xFF
}

/// Unpack the 24-bit intensity value from a packed u32.
#[inline]
pub fn cpu_unpack_intensity(packed: u32) -> u32 {
    packed & 0x00FF_FFFF
}

// ============================================================================
// Feature hash (CPU reference; bit-equivalent to .cuh)
// ============================================================================

/// SplitMix64 mixer. Bit-equivalent to `prism_splitmix64` in
/// `rich_spike.cuh`. Used by the feature-hash chain.
#[inline]
pub fn cpu_splitmix64(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// 64-bit feature hash for a RichSpike. Mixes every non-spatial
/// field via SplitMix64. Spatial (x, y, z) is excluded — the
/// Morton-code component of the composite sort key already
/// discriminates spatially. Bit-equivalent to
/// `prism_rich_spike_feature_hash` in `rich_spike.cuh`.
pub fn cpu_rich_spike_feature_hash(s: &RichSpike) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325;
    h ^= cpu_splitmix64(s.t_frame as u64);
    // Bitcast floats to u32 so hash is bit-exact regardless of FPU
    // rounding mode.
    h ^= cpu_splitmix64(s.water_density.to_bits() as u64);
    h ^= cpu_splitmix64(s.wd_change.to_bits() as u64);
    h ^= cpu_splitmix64(s.vib_energy.to_bits() as u64);
    h ^= cpu_splitmix64(s.intensity_packed as u64);
    h ^= cpu_splitmix64(s.residue_id as u32 as u64);
    h ^= cpu_splitmix64(s.cluster_id as u32 as u64);
    h ^= cpu_splitmix64(s.causal_lag.to_bits() as u64);
    h ^= cpu_splitmix64(s.n_excited as u64);
    h ^= cpu_splitmix64(s.origin_phase as u64);
    h ^= cpu_splitmix64(s.spike_source as u64);
    h ^= cpu_splitmix64(s.chem_flags as u64);
    h ^= cpu_splitmix64(s.kinetic_delta.to_bits() as u64);
    cpu_splitmix64(h)
}

// ============================================================================
// FFI surface — link probe only at this stage
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)]
mod ffi {
    extern "C" {
        pub fn prism_rich_spike_link_probe() -> u32;
    }
}

#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    unsafe { ffi::prism_rich_spike_link_probe() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rich_spike_layout_is_64_bytes_64_aligned() {
        // Layout pin: must match the C-side static_assert.
        // sizeof == 64, alignof == 64. Failure here means the
        // FFI layout has drifted between C and Rust — a
        // hardware-trap vector under the strict alignment policy
        // (Rectification §2: unaligned access traps on Blackwell).
        assert_eq!(std::mem::size_of::<RichSpike>(), 64);
        assert_eq!(std::mem::align_of::<RichSpike>(), 64);
    }

    #[test]
    fn rich_spike_field_offsets() {
        // Each plane is 16 bytes contiguous. Verify field offsets
        // match the documented layout. We use unsafe pointer math
        // on a stack-allocated zero spike.
        let s = RichSpike::zero();
        let base = &s as *const RichSpike as usize;
        macro_rules! ofs {
            ($field:ident) => {
                (&s.$field as *const _ as usize) - base
            };
        }
        // Plane 1
        assert_eq!(ofs!(x), 0);
        assert_eq!(ofs!(y), 4);
        assert_eq!(ofs!(z), 8);
        assert_eq!(ofs!(t_frame), 12);
        // Plane 2
        assert_eq!(ofs!(water_density), 16);
        assert_eq!(ofs!(wd_change), 20);
        assert_eq!(ofs!(vib_energy), 24);
        assert_eq!(ofs!(intensity_packed), 28);
        // Plane 3
        assert_eq!(ofs!(residue_id), 32);
        assert_eq!(ofs!(cluster_id), 36);
        assert_eq!(ofs!(causal_lag), 40);
        assert_eq!(ofs!(n_excited), 44);
        // Plane 4
        assert_eq!(ofs!(origin_phase), 48);
        assert_eq!(ofs!(spike_source), 52);
        assert_eq!(ofs!(chem_flags), 56);
        assert_eq!(ofs!(kinetic_delta), 60);
    }

    #[test]
    fn intensity_pack_round_trip() {
        // Round-trip every (percentile, intensity) at boundaries.
        for &(p, i) in &[
            (0u32, 0u32),
            (0u32, 0x00FF_FFFF), // max intensity
            (0xFF, 0u32),        // max percentile
            (0xFF, 0x00FF_FFFF), // both max
            (42, 1234567),       // arbitrary
        ] {
            let packed = cpu_pack_intensity(p, i);
            assert_eq!(
                cpu_unpack_percentile(packed),
                p,
                "percentile round-trip failed: input ({}, {})",
                p,
                i
            );
            assert_eq!(
                cpu_unpack_intensity(packed),
                i,
                "intensity round-trip failed: input ({}, {})",
                p,
                i
            );
        }
    }

    #[test]
    fn intensity_pack_truncates_overflow() {
        // Inputs above the field width are silently truncated.
        assert_eq!(
            cpu_unpack_percentile(cpu_pack_intensity(0xFFFF_FF00, 0)),
            0,
            "percentile high bits should be masked"
        );
        assert_eq!(
            cpu_unpack_intensity(cpu_pack_intensity(0, 0xFFFF_FFFF)),
            0x00FF_FFFF,
            "intensity high bits should be masked"
        );
    }

    #[test]
    fn feature_hash_distinguishes_changed_fields() {
        // Mutating any non-spatial field MUST change the hash. This
        // pins the contract that every field contributes to the
        // composite sort key's discrimination.
        let s0 = RichSpike::zero();
        let h0 = cpu_rich_spike_feature_hash(&s0);

        let mut s1 = s0;
        s1.t_frame = 42;
        assert_ne!(
            cpu_rich_spike_feature_hash(&s1),
            h0,
            "t_frame must affect hash"
        );

        let mut s2 = s0;
        s2.water_density = 1.0;
        assert_ne!(
            cpu_rich_spike_feature_hash(&s2),
            h0,
            "water_density must affect hash"
        );

        let mut s3 = s0;
        s3.residue_id = 7;
        assert_ne!(
            cpu_rich_spike_feature_hash(&s3),
            h0,
            "residue_id must affect hash"
        );

        let mut s4 = s0;
        s4.kinetic_delta = std::f32::consts::PI;
        assert_ne!(
            cpu_rich_spike_feature_hash(&s4),
            h0,
            "kinetic_delta must affect hash"
        );
    }

    #[test]
    fn feature_hash_excludes_spatial_fields() {
        // Per the Mandate §1.3 design: spatial (x, y, z) is NOT in
        // the feature hash — Morton-code discrimination handles
        // spatial alignment in the composite key. Two spikes
        // identical in every non-spatial field but at different
        // positions must hash to the same feature value.
        let mut s0 = RichSpike::zero();
        s0.t_frame = 42;
        s0.residue_id = 7;
        let h0 = cpu_rich_spike_feature_hash(&s0);

        let mut s1 = s0;
        s1.x = 100.0;
        s1.y = -50.0;
        s1.z = 25.0;
        let h1 = cpu_rich_spike_feature_hash(&s1);

        assert_eq!(h0, h1, "spatial fields must not affect feature hash");
    }

    #[test]
    fn feature_hash_zero_spike_is_deterministic() {
        // Same input → same output. Cheap regression check that
        // catches any non-determinism in the mixer chain.
        let s = RichSpike::zero();
        let h_a = cpu_rich_spike_feature_hash(&s);
        let h_b = cpu_rich_spike_feature_hash(&s);
        assert_eq!(h_a, h_b);
    }

    #[test]
    fn splitmix64_zero_is_zero() {
        // SplitMix64(0) == 0 by construction (every shift on 0 is 0,
        // every multiplication of 0 is 0). This is a property of the
        // algorithm, not a bug. Documented here so future readers
        // don't add a "fix" for an apparent zero collision.
        assert_eq!(cpu_splitmix64(0), 0);
    }

    #[test]
    fn splitmix64_nonzero_input_diffuses() {
        // Distinct small inputs produce distinct outputs. A weak
        // mixer would collide on small adjacent values.
        let outs: std::collections::HashSet<u64> = (0u64..1024).map(cpu_splitmix64).collect();
        assert_eq!(
            outs.len(),
            1024,
            "SplitMix64 collided on small adjacent inputs (mixer broken)"
        );
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        // Sentinel 0x6164 ("ad" — for "rich [a]d[ata]"). Confirms
        // rich_spike.cu linked into the static archive and FFI ABI
        // is round-tripping.
        assert_eq!(super::link_probe(), 0x0000_6164);
    }
}

/// GPU consumption view only. The original RichSpike archive remains authoritative.
/// Constants are bit-exact per-population values, never inferred from absence.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeV2SeedLayout {
    pub format: u32,
    pub causal_lag_bits: u32,
    pub chem_flags: u32,
    pub reserved: u32,
}
impl NativeV2SeedLayout {
    pub fn validate(self) -> bool { self.format <= 1 && self.reserved == 0 }
    pub fn stride_bytes(self) -> usize { if self.format == 1 { 32 } else { 64 } }
}
/// Every non-constant field read by the existing V2 kernels, without quantization.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct NativeV2Seed32 {
    pub x: f32, pub y: f32, pub z: f32,
    pub water_density: f32, pub wd_change: f32, pub vib_energy: f32,
    pub intensity_packed: u32,
    pub residue_id: i16,
    /// Exact10-bit native phase plus6-bit native source; source5 is present in real records.
    pub phase_source: u16,
}
const _: () = {
    assert!(std::mem::size_of::<NativeV2SeedLayout>() == 16);
    assert!(std::mem::size_of::<NativeV2Seed32>() == 32);
    assert!(std::mem::offset_of!(NativeV2Seed32, intensity_packed) == 24);
    assert!(std::mem::offset_of!(NativeV2Seed32, residue_id) == 28);
    assert!(std::mem::offset_of!(NativeV2Seed32, phase_source) == 30);
};
#[derive(Debug)]
pub struct NativeV2SeedPopulation {
    pub layout: NativeV2SeedLayout,
    pub compact: Vec<NativeV2Seed32>,
    pub full_layout_reason: Option<&'static str>,
}
impl NativeV2SeedPopulation {
    /// Nonuniform constants or out-of-range integers select the unchanged full
    /// representation. No record is dropped and no numeric value is rounded.
    pub fn prepare(raw: &[RichSpike]) -> Self {
        let full = |reason| Self { layout:NativeV2SeedLayout::default(),
            compact:Vec::new(),full_layout_reason:Some(reason) };
        let Some(first)=raw.first() else { return full("empty_population"); };
        for s in raw {
            if s.causal_lag.to_bits()!=first.causal_lag.to_bits() || s.chem_flags!=first.chem_flags {
                return full("nonuniform_population_constants");
            }
            if i16::try_from(s.residue_id).is_err() || s.origin_phase>1023 || s.spike_source>63 {
                return full("integer_range_requires_full_layout");
            }
        }
        let compact=raw.iter().map(|s|NativeV2Seed32 {
            x:s.x,y:s.y,z:s.z,water_density:s.water_density,wd_change:s.wd_change,
            vib_energy:s.vib_energy,intensity_packed:s.intensity_packed,
            residue_id:s.residue_id as i16,
            phase_source:(s.origin_phase | (s.spike_source<<10)) as u16,
        }).collect();
        Self { layout:NativeV2SeedLayout {format:1,causal_lag_bits:first.causal_lag.to_bits(),
            chem_flags:first.chem_flags,reserved:0},compact,full_layout_reason:None }
    }
    pub fn bytes<'a>(&'a self, raw: &'a [RichSpike]) -> &'a [u8] {
        if self.layout.format==1 {
            assert_eq!(self.compact.len(),raw.len());
            unsafe { std::slice::from_raw_parts(self.compact.as_ptr().cast(),self.compact.len()*32) }
        } else {
            unsafe { std::slice::from_raw_parts(raw.as_ptr().cast(),raw.len()*64) }
        }
    }
}

#[cfg(test)]
mod native_v2_seed_view_tests {
    use super::*;
    #[test]
    fn consumed_fields_preserve_bits_and_full_archive_stays_unchanged() {
        let mut a=RichSpike::zero();
        a.x=-0.0; a.y=f32::from_bits(1); a.z=3.25; a.water_density=0.13;
        a.wd_change=-4.5; a.vib_energy=f32::from_bits(0x7fc12345);
        a.intensity_packed=u32::MAX; a.residue_id=i16::MIN as i32;
        a.origin_phase=1023; a.spike_source=63; a.causal_lag=-0.0; a.chem_flags=0xfedcba98;
        a.t_frame=1234; a.cluster_id=57; a.n_excited=99; a.kinetic_delta=0.17;
        let mut b=a;b.residue_id=i16::MAX as i32;b.t_frame=999;b.spike_source=5;
        let raw=[a,b]; let before=unsafe {std::slice::from_raw_parts(raw.as_ptr().cast::<u8>(),128)}.to_vec();
        let p=NativeV2SeedPopulation::prepare(&raw);assert_eq!(p.layout.format,1);
        assert_eq!(p.bytes(&raw).len(),64);
        for (s,c) in raw.iter().zip(&p.compact) {
            for (x,y) in [s.x,s.y,s.z,s.water_density,s.wd_change,s.vib_energy].into_iter()
                .zip([c.x,c.y,c.z,c.water_density,c.wd_change,c.vib_energy]) {assert_eq!(x.to_bits(),y.to_bits());}
            assert_eq!(s.intensity_packed,c.intensity_packed);assert_eq!(s.residue_id,i32::from(c.residue_id));
            assert_eq!(s.origin_phase,u32::from(c.phase_source&1023));
            assert_eq!(s.spike_source,u32::from(c.phase_source>>10));
            assert_eq!(s.causal_lag.to_bits(),p.layout.causal_lag_bits);assert_eq!(s.chem_flags,p.layout.chem_flags);
        }
        assert_eq!(before,unsafe{std::slice::from_raw_parts(raw.as_ptr().cast::<u8>(),128)});
    }
    #[test]
    fn nonuniform_or_unrepresentable_population_keeps_all_original_bytes() {
        let a=RichSpike::zero();
        for k in 0..5 {let mut b=a;match k {0=>b.residue_id=32768,1=>b.origin_phase=1024,
            2=>b.spike_source=64,3=>b.causal_lag=-0.0,_=>b.chem_flags=1};
            let raw=[a,b];let p=NativeV2SeedPopulation::prepare(&raw);assert_eq!(p.layout.format,0);
            assert!(p.compact.is_empty());assert_eq!(p.bytes(&raw).len(),128);
            assert_eq!(p.bytes(&raw).as_ptr(),raw.as_ptr().cast::<u8>());
        }
    }
}
