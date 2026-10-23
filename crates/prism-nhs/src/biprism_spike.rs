//! Versioned Voronoi LIF packets and their complete Sigma-Delta ledger.
//!
//! A queue owner must select this schema explicitly. The magic at byte 72 is
//! also a valid bit pattern in a native packet's aromatic residue field, so it
//! cannot by itself choose between two 96-byte layouts. Decode bytes without
//! creating aligned references into an arbitrary readback buffer.

pub const SPIKE_EVENT_V10_SCHEMA_MAGIC: u32 = 0x5631_3053;
pub const BIPRISM_SPIKE_RECORD_BYTES: usize = 96;
pub const BIPRISM_MAX_MULTIPLICITY: u32 = 65_535;
/// Distinct from the existing thermal/dewetting LIF source (2).
pub const VORONOI_LIF_SPIKE_SOURCE: i32 = 6;
pub const BIPRISM_ANGSTROM_TO_BOHR: f64 = 1.8897261259077824;
pub const NATIVE_LIF_OBSERVATION_BYTES: usize = 64;
pub const NATIVE_LIF_OBSERVATION_MAGIC: u32 = 0x4e4f_4232;

/// The queue owner's declared physical frame. There is no arbitrary scale or
/// automatic unit inference from event values or a schema-magic bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BiprismEventFrame {
    #[default]
    NativeAngstrom,
    AtomicBohr,
}
impl BiprismEventFrame {
    pub fn name(self) -> &'static str {
        match self {
            Self::NativeAngstrom => "native_angstrom",
            Self::AtomicBohr => "atomic_bohr",
        }
    }
    pub fn position_unit(self) -> &'static str {
        match self {
            Self::NativeAngstrom => "angstrom",
            Self::AtomicBohr => "bohr",
        }
    }
    pub fn force_unit(self) -> &'static str {
        match self {
            // Historical diagnostic callers did not qualify their force model.
            Self::NativeAngstrom => "producer_unbound",
            Self::AtomicBohr => "hartree/bohr",
        }
    }
    pub fn dual_volume_unit(self) -> &'static str {
        match self {
            Self::NativeAngstrom => "angstrom^3",
            Self::AtomicBohr => "bohr^3",
        }
    }
    fn length_scale(self) -> f64 {
        match self {
            Self::NativeAngstrom => 1.0,
            Self::AtomicBohr => BIPRISM_ANGSTROM_TO_BOHR,
        }
    }
}

/// Rounded values used by the actual CUDA signal-grid mapper. The native grid
/// remains the source of cell identity and the eventual Å voxel-center output.
#[cfg(feature = "gpu")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BiprismProducerGrid {
    pub origin_xyz: [f32; 3],
    pub spacing: f32,
    pub inv_voxel_spacing: f32,
    pub dimension: u32,
    pub total_voxels: u32,
}
#[cfg(feature = "gpu")]
impl BiprismProducerGrid {
    pub fn from_native_geometry(
        geometry: crate::sdst_bridge::SdstGridGeometry,
        frame: BiprismEventFrame,
    ) -> Result<Self, String> {
        geometry.validate().map_err(|e| e.to_string())?;
        let scale = frame.length_scale();
        let spacing = (f64::from(geometry.spacing_a) * scale) as f32;
        let origin_xyz = geometry.origin_a.map(|x| (f64::from(x) * scale) as f32);
        let inv_voxel_spacing = 1.0f32 / spacing;
        if origin_xyz.iter().any(|x| !x.is_finite())
            || !spacing.is_finite()
            || spacing <= 0.0
            || !inv_voxel_spacing.is_finite()
            || inv_voxel_spacing <= 0.0
        {
            return Err(
                "producer-frame grid conversion is outside finite binary32 geometry".into(),
            );
        }
        Ok(Self {
            origin_xyz,
            spacing,
            inv_voxel_spacing,
            dimension: geometry.dimension,
            total_voxels: geometry.dimension.pow(3),
        })
    }

    /// Mirrors (pos-origin)*inverse_spacing followed by CUDA floor-to-int and
    /// bounds checks. Never divide an AU event back to Å before this decision.
    /// NOB2 keeps the exact producer AU split; do not first round hi+lo to f32.
    pub fn recorded_cell_compensated(self, high: [f32; 3], low: [f32; 3]) -> Result<i32, String> {
        if high.iter().chain(&low).any(|x| !x.is_finite()) {
            return Err("nonfinite compensated producer position".into());
        }
        let mut xyz = [0u32; 3];
        for axis in 0..3 {
            let x = f64::from(high[axis]) + f64::from(low[axis]);
            let coordinate = ((x - f64::from(self.origin_xyz[axis]))
                * f64::from(self.inv_voxel_spacing)).floor();
            if !coordinate.is_finite() || coordinate < 0.0 || coordinate >= f64::from(self.dimension) { return Ok(-1); }
            xyz[axis] = coordinate as u32;
        }
        Ok(((xyz[2] * self.dimension + xyz[1]) * self.dimension + xyz[0]) as i32)
    }

    pub fn recorded_cell(self, position: [f32; 3]) -> Result<i32, String> {
        if position.iter().any(|x| !x.is_finite()) {
            return Err("nonfinite producer position".into());
        }
        let mut xyz = [0u32; 3];
        for axis in 0..3 {
            let coordinate =
                ((position[axis] - self.origin_xyz[axis]) * self.inv_voxel_spacing).floor();
            if !coordinate.is_finite() || coordinate < 0.0 || coordinate >= self.dimension as f32 {
                return Ok(-1);
            }
            xyz[axis] = coordinate as u32;
        }
        Ok(((xyz[2] * self.dimension + xyz[1]) * self.dimension + xyz[0]) as i32)
    }
}

#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GpuSpikeEventCanonical96Ffi {
    pub pos_xyz: [f32; 3],
    pub step_tau: u32,
    pub atom_or_site_id: u32,
    pub spike_amplitude: f32,
    pub uv_or_holonomy_val: f32,
    pub voronoi_dual_vol: f32,
    pub pocket_tile_id: u32,
    pub residue_id: i32,
    pub multiplicity_m: u32,
    pub v_pre_reset: f32,
    pub q_discharged: f32,
    pub q_discharged_lo: f32,
    pub r_post_reset: f32,
    pub lif_threshold_used: f32,
    pub leaf_id: u16,
    pub vertex_id: u16,
    pub voxel_cell_idx: i32,
    pub event_schema_magic: u32,
    /// Force produced at this endpoint. This is not a consumed impulse/work receipt.
    pub coupled_force_xyz: [f32; 3],
    pub q_lo_product_raw_bits: u32,
    pub reserved_v5: u32,
}

const _: () = {
    assert!(core::mem::size_of::<GpuSpikeEventCanonical96Ffi>() == 96);
    assert!(core::mem::align_of::<GpuSpikeEventCanonical96Ffi>() == 32);
    assert!(core::mem::offset_of!(GpuSpikeEventCanonical96Ffi, q_discharged_lo) == 52);
    assert!(core::mem::offset_of!(GpuSpikeEventCanonical96Ffi, leaf_id) == 64);
    assert!(core::mem::offset_of!(GpuSpikeEventCanonical96Ffi, event_schema_magic) == 72);
    assert!(core::mem::offset_of!(GpuSpikeEventCanonical96Ffi, coupled_force_xyz) == 76);
    assert!(core::mem::offset_of!(GpuSpikeEventCanonical96Ffi, reserved_v5) == 92);
};

#[derive(Clone, Copy, Debug)]
pub struct BiprismDecodeLimits {
    pub num_atoms: u32,
    pub num_residues: u32,
    pub total_voxels: u32,
    /// Set for a single-step drain; a multi-step queue validates chronology separately.
    pub expected_step: Option<u32>,
}

impl GpuSpikeEventCanonical96Ffi {
    pub fn from_le_bytes(bytes: &[u8], limits: BiprismDecodeLimits) -> Result<Self, String> {
        if bytes.len() != BIPRISM_SPIKE_RECORD_BYTES {
            return Err(format!(
                "V10S record requires 96 bytes, got {}",
                bytes.len()
            ));
        }
        let u32_at = |i| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        let f32_at = |i| f32::from_bits(u32_at(i));
        let u16_at = |i| u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap());
        let event = Self {
            pos_xyz: [f32_at(0), f32_at(4), f32_at(8)],
            step_tau: u32_at(12),
            atom_or_site_id: u32_at(16),
            spike_amplitude: f32_at(20),
            uv_or_holonomy_val: f32_at(24),
            voronoi_dual_vol: f32_at(28),
            pocket_tile_id: u32_at(32),
            residue_id: u32_at(36) as i32,
            multiplicity_m: u32_at(40),
            v_pre_reset: f32_at(44),
            q_discharged: f32_at(48),
            q_discharged_lo: f32_at(52),
            r_post_reset: f32_at(56),
            lif_threshold_used: f32_at(60),
            leaf_id: u16_at(64),
            vertex_id: u16_at(66),
            voxel_cell_idx: u32_at(68) as i32,
            event_schema_magic: u32_at(72),
            coupled_force_xyz: [f32_at(76), f32_at(80), f32_at(84)],
            q_lo_product_raw_bits: u32_at(88),
            reserved_v5: u32_at(92),
        };
        event.validate(limits)?;
        Ok(event)
    }

    pub fn validate(&self, limits: BiprismDecodeLimits) -> Result<(), String> {
        if self.event_schema_magic != SPIKE_EVENT_V10_SCHEMA_MAGIC || self.reserved_v5 != 0 {
            return Err("unsupported V10S schema discriminator or reserved bits".into());
        }
        let scalar = [
            self.spike_amplitude,
            self.uv_or_holonomy_val,
            self.voronoi_dual_vol,
            self.v_pre_reset,
            self.q_discharged,
            self.q_discharged_lo,
            self.r_post_reset,
            self.lif_threshold_used,
            f32::from_bits(self.q_lo_product_raw_bits),
        ];
        if scalar
            .iter()
            .chain(self.pos_xyz.iter())
            .chain(self.coupled_force_xyz.iter())
            .any(|x| !x.is_finite())
        {
            return Err("nonfinite V10S field".into());
        }
        if self.leaf_id >= 8 || self.vertex_id >= 192 || self.atom_or_site_id >= limits.num_atoms {
            return Err("V10S leaf, core vertex, or atom outside its population".into());
        }
        if self.residue_id < -1
            || (self.residue_id >= 0 && self.residue_id as u32 >= limits.num_residues)
        {
            return Err("V10S residue is neither excluded (-1) nor a valid mapped residue".into());
        }
        if self.voxel_cell_idx < -1
            || (self.voxel_cell_idx >= 0 && self.voxel_cell_idx as u32 >= limits.total_voxels)
        {
            return Err("V10S cell outside its grid".into());
        }
        if limits
            .expected_step
            .is_some_and(|step| step != self.step_tau)
        {
            return Err("V10S step does not match the queue's authoritative clock".into());
        }
        if !(1..=BIPRISM_MAX_MULTIPLICITY).contains(&self.multiplicity_m)
            || self.lif_threshold_used <= 0.0
            || self.r_post_reset < 0.0
            || self.r_post_reset >= self.lif_threshold_used
            || self.spike_amplitude < 0.0
            || self.uv_or_holonomy_val < 0.0
            || self.voronoi_dual_vol <= 0.0
        {
            return Err(
                "V10S multiplicity, threshold, remainder, or physical field outside its domain"
                    .into(),
            );
        }
        // The bounded integer has <=16 significant bits; multiplying by a
        // binary32 threshold is exact in f64. Verify the canonical rounded high
        // part, both low-part witnesses, and the complete partition independently.
        let product = f64::from(self.multiplicity_m) * f64::from(self.lif_threshold_used);
        let high = product as f32;
        let low = (product - f64::from(high)) as f32;
        if self.q_discharged != high
            || self.q_discharged_lo != low
            || f32::from_bits(self.q_lo_product_raw_bits) != low
            || f64::from(high) + f64::from(low) != product
        {
            return Err("V10S discharged quantum is not the canonical exact TwoProd split".into());
        }
        let canonical_remainder =
            (-(self.multiplicity_m as f32)).mul_add(self.lif_threshold_used, self.v_pre_reset);
        if self.r_post_reset != canonical_remainder
            || product + f64::from(self.r_post_reset) != f64::from(self.v_pre_reset)
        {
            return Err(
                "V10S voltage partition does not close exactly in the supported domain".into(),
            );
        }
        Ok(())
    }

    pub fn to_le_bytes(&self) -> [u8; BIPRISM_SPIKE_RECORD_BYTES] {
        let mut b = [0u8; BIPRISM_SPIKE_RECORD_BYTES];
        let words = [
            self.pos_xyz[0].to_bits(),
            self.pos_xyz[1].to_bits(),
            self.pos_xyz[2].to_bits(),
            self.step_tau,
            self.atom_or_site_id,
            self.spike_amplitude.to_bits(),
            self.uv_or_holonomy_val.to_bits(),
            self.voronoi_dual_vol.to_bits(),
            self.pocket_tile_id,
            self.residue_id as u32,
            self.multiplicity_m,
            self.v_pre_reset.to_bits(),
            self.q_discharged.to_bits(),
            self.q_discharged_lo.to_bits(),
            self.r_post_reset.to_bits(),
            self.lif_threshold_used.to_bits(),
            u32::from(self.leaf_id) | (u32::from(self.vertex_id) << 16),
            self.voxel_cell_idx as u32,
            self.event_schema_magic,
            self.coupled_force_xyz[0].to_bits(),
            self.coupled_force_xyz[1].to_bits(),
            self.coupled_force_xyz[2].to_bits(),
            self.q_lo_product_raw_bits,
            self.reserved_v5,
        ];
        for (i, word) in words.iter().enumerate() {
            b[4 * i..4 * i + 4].copy_from_slice(&word.to_le_bytes());
        }
        b
    }
}

/// Full records remain attached to their row identity. No ledger bits are stored
/// in residue slots, UV metadata, or the native packet's phase field.
#[derive(Clone, Debug, PartialEq)]
pub struct VoronoiSpikeLedgerRecord {
    pub queue_row: u64,
    pub event: GpuSpikeEventCanonical96Ffi,
}

/// Decode the explicitly selected V10S producer queue after stream completion.
/// Event rows count emission groups; sum(multiplicity_m) counts discharged quanta.
pub fn decode_v10_spike_queue(
    raw: &[u8],
    count: u32,
    capacity: u32,
    limits: BiprismDecodeLimits,
) -> Result<Vec<VoronoiSpikeLedgerRecord>, String> {
    if count > capacity {
        return Err("V10S queue count exceeds allocation capacity".into());
    }
    let required = (count as usize)
        .checked_mul(BIPRISM_SPIKE_RECORD_BYTES)
        .ok_or("V10S queue byte count overflow")?;
    if raw.len() < required {
        return Err("V10S queue readback is truncated".into());
    }
    raw[..required]
        .chunks_exact(BIPRISM_SPIKE_RECORD_BYTES)
        .enumerate()
        .map(|(i, b)| {
            GpuSpikeEventCanonical96Ffi::from_le_bytes(b, limits)
                .map(|event| VoronoiSpikeLedgerRecord {
                    queue_row: i as u64,
                    event,
                })
                .map_err(|e| format!("V10S queue row {i}: {e}"))
        })
        .collect()
}

/// Native observer quantities that are absent from the V10S packet. They must
/// come from the same emission epoch; dual volume is not water density, and
/// Wilson holonomy is not the native UV wavelength or CCNS phase.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NativeLifObservation {
    pub step_tau: u32,
    pub leaf_id: u16,
    pub vertex_id: u16,
    pub atom_or_site_id: u32,
    pub voxel_cell_idx: i32,
    pub water_density: f32,
    pub wd_change: f32,
    pub n_nearby_excited: i32,
    pub phase_bits: u32,
    pub producer_position_lo: [f32; 3],
    pub coordinate_magic: u32,
    pub coordinate_generation: u64,
    pub coordinate_unit_frame: u32,
    pub coordinate_reserved: u32,
}
const _: () = {
    assert!(core::mem::size_of::<NativeLifObservation>() == NATIVE_LIF_OBSERVATION_BYTES);
    assert!(core::mem::align_of::<NativeLifObservation>() == 16);
    assert!(core::mem::offset_of!(NativeLifObservation, leaf_id) == 4);
    assert!(core::mem::offset_of!(NativeLifObservation, vertex_id) == 6);
    assert!(core::mem::offset_of!(NativeLifObservation, atom_or_site_id) == 8);
    assert!(core::mem::offset_of!(NativeLifObservation, voxel_cell_idx) == 12);
    assert!(core::mem::offset_of!(NativeLifObservation, water_density) == 16);
    assert!(core::mem::offset_of!(NativeLifObservation, phase_bits) == 28);
    assert!(core::mem::offset_of!(NativeLifObservation, producer_position_lo) == 32);
    assert!(core::mem::offset_of!(NativeLifObservation, coordinate_magic) == 44);
    assert!(core::mem::offset_of!(NativeLifObservation, coordinate_generation) == 48);
    assert!(core::mem::offset_of!(NativeLifObservation, coordinate_unit_frame) == 56);
};
impl NativeLifObservation {
    pub fn from_le_bytes(bytes: &[u8], limits: BiprismDecodeLimits) -> Result<Self, String> {
        if bytes.len() != NATIVE_LIF_OBSERVATION_BYTES {
            return Err("NOB2 native LIF observation requires exactly 64 bytes".into());
        }
        let word = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let half = |offset| u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
        let value = Self {
            step_tau: word(0),
            leaf_id: half(4),
            vertex_id: half(6),
            atom_or_site_id: word(8),
            voxel_cell_idx: word(12) as i32,
            water_density: f32::from_bits(word(16)),
            wd_change: f32::from_bits(word(20)),
            n_nearby_excited: word(24) as i32,
            phase_bits: word(28),
            producer_position_lo: [f32::from_bits(word(32)), f32::from_bits(word(36)), f32::from_bits(word(40))],
            coordinate_magic: word(44),
            coordinate_generation: u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            coordinate_unit_frame: word(56),
            coordinate_reserved: word(60),
        };
        value.validate(limits)?;
        Ok(value)
    }
    pub fn to_le_bytes(self) -> [u8; NATIVE_LIF_OBSERVATION_BYTES] {
        let mut out = [0u8; NATIVE_LIF_OBSERVATION_BYTES];
        out[0..4].copy_from_slice(&self.step_tau.to_le_bytes());
        out[4..6].copy_from_slice(&self.leaf_id.to_le_bytes());
        out[6..8].copy_from_slice(&self.vertex_id.to_le_bytes());
        for (offset, value) in [(8,self.atom_or_site_id),(12,self.voxel_cell_idx as u32),
            (16,self.water_density.to_bits()),(20,self.wd_change.to_bits()),(24,self.n_nearby_excited as u32),
            (28,self.phase_bits),(32,self.producer_position_lo[0].to_bits()),(36,self.producer_position_lo[1].to_bits()),
            (40,self.producer_position_lo[2].to_bits()),(44,self.coordinate_magic),(56,self.coordinate_unit_frame),(60,self.coordinate_reserved)] {
            out[offset..offset+4].copy_from_slice(&value.to_le_bytes());
        }
        out[48..56].copy_from_slice(&self.coordinate_generation.to_le_bytes());
        out
    }
    pub fn validate(self, limits: BiprismDecodeLimits) -> Result<(), String> {
        if self.coordinate_magic != NATIVE_LIF_OBSERVATION_MAGIC || self.coordinate_generation == 0
            || self.coordinate_unit_frame != 2 || self.coordinate_reserved != 0
            || self.producer_position_lo.iter().any(|x| !x.is_finite()) {
            return Err("invalid NOB2 coordinate authority or producer low plane".into());
        }
        if self.leaf_id >= 8
            || self.vertex_id >= 192
            || self.atom_or_site_id >= limits.num_atoms
            || self.voxel_cell_idx < -1
            || (self.voxel_cell_idx >= 0 && self.voxel_cell_idx as u32 >= limits.total_voxels)
            || limits
                .expected_step
                .is_some_and(|step| step != self.step_tau)
        {
            return Err(
                "native LIF observation is outside its emission population or clock".into(),
            );
        }
        if !self.water_density.is_finite()
            || self.water_density < 0.0
            || !self.wd_change.is_finite()
            || self.wd_change < 0.0
            || !(0..=255).contains(&self.n_nearby_excited)
            || self.phase_bits > 1023
        {
            return Err("invalid native observer quantities or CCNS phase bits".into());
        }
        Ok(())
    }
}

pub fn decode_native_lif_observation_queue(
    raw: &[u8],
    count: u32,
    capacity: u32,
    limits: BiprismDecodeLimits,
) -> Result<Vec<NativeLifObservation>, String> {
    if count > capacity {
        return Err("native observer count exceeds its allocation".into());
    }
    let required = (count as usize)
        .checked_mul(NATIVE_LIF_OBSERVATION_BYTES)
        .ok_or("native observer extent overflow")?;
    if raw.len() < required {
        return Err("native observer readback is truncated".into());
    }
    raw[..required]
        .chunks_exact(NATIVE_LIF_OBSERVATION_BYTES)
        .enumerate()
        .map(|(row, bytes)| {
            NativeLifObservation::from_le_bytes(bytes, limits)
                .map_err(|e| format!("native observer row{row}: {e}"))
        })
        .collect()
}

#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct DecodedVoronoiSpikeQueue {
    pub event_frame: BiprismEventFrame,
    /// Existing Arrow V5 / PathB row type; never a replacement wire layout.
    pub native_events: Vec<crate::fused_engine::GpuSpikeEvent>,
    /// One complete record for each native row, with the same ordinal.
    pub ledger: Vec<VoronoiSpikeLedgerRecord>,
}

#[cfg(feature = "gpu")]
pub fn project_v10_to_native(
    ev: &GpuSpikeEventCanonical96Ffi,
    observation: NativeLifObservation,
    limits: BiprismDecodeLimits,
    geometry: crate::sdst_bridge::SdstGridGeometry,
) -> Result<crate::fused_engine::GpuSpikeEvent, String> {
    project_v10_to_native_in_frame(
        ev,
        observation,
        limits,
        geometry,
        BiprismEventFrame::AtomicBohr,
    )
}

#[cfg(feature = "gpu")]
pub fn project_v10_to_native_in_frame(
    ev: &GpuSpikeEventCanonical96Ffi,
    observation: NativeLifObservation,
    limits: BiprismDecodeLimits,
    geometry: crate::sdst_bridge::SdstGridGeometry,
    frame: BiprismEventFrame,
) -> Result<crate::fused_engine::GpuSpikeEvent, String> {
    ev.validate(limits)?;
    observation.validate(limits)?;
    if frame != BiprismEventFrame::AtomicBohr { return Err("NOB2 producer coordinates require the explicit atomic-bohr frame".into()); }
    if (
        observation.step_tau,
        observation.leaf_id,
        observation.vertex_id,
        observation.atom_or_site_id,
        observation.voxel_cell_idx,
    ) != (
        ev.step_tau,
        ev.leaf_id,
        ev.vertex_id,
        ev.atom_or_site_id,
        ev.voxel_cell_idx,
    ) {
        return Err("native observer context belongs to another V10S emission".into());
    }
    let timestep = i32::try_from(ev.step_tau)
        .map_err(|_| "V10S protocol step exceeds the native signed clock domain")?;
    geometry.validate().map_err(|e| e.to_string())?;
    let dim = geometry.dimension;
    if dim * dim * dim != limits.total_voxels || ev.voxel_cell_idx < 0 {
        return Err(
            "native PathB projection requires an in-grid event and matching cubic geometry".into(),
        );
    }
    let index = ev.voxel_cell_idx as u32;
    let producer_grid = BiprismProducerGrid::from_native_geometry(geometry, frame)?;
    if producer_grid.recorded_cell_compensated(ev.pos_xyz, observation.producer_position_lo)? != ev.voxel_cell_idx {
        return Err(
            "V10S atom coordinate and recorded voxel disagree with producer-frame geometry".into(),
        );
    }
    let xyz = [index % dim, (index / dim) % dim, index / (dim * dim)];
    let mut position = [0.0; 3];
    for axis in 0..3 {
        position[axis] =
            (xyz[axis] as f32 + 0.5).mul_add(geometry.spacing_a, geometry.origin_a[axis]);
        if !position[axis].is_finite() {
            return Err("native PathB voxel center overflows binary32".into());
        }
    }
    let mut nearby_residues = [-1; 8];
    if ev.residue_id >= 0 {
        nearby_residues[0] = ev.residue_id;
    }
    let native = crate::fused_engine::GpuSpikeEvent {
        // Native PathB records voxel centers; the original atom coordinate is
        // preserved without alteration in the complete companion record.
        timestep,
        voxel_idx: ev.voxel_cell_idx,
        position,
        intensity: ev.spike_amplitude,
        nearby_residues,
        n_residues: i32::from(ev.residue_id >= 0),
        // This producer is LIF. Native UV-specific fields are inapplicable;
        // quantum/holonomy measurements remain in the typed companion ledger.
        spike_source: VORONOI_LIF_SPIKE_SOURCE,
        wavelength_nm: 0.0,
        aromatic_type: -1,
        aromatic_residue_id: -1,
        water_density: observation.water_density,
        vibrational_energy: 0.0,
        n_nearby_excited: observation.n_nearby_excited,
        wd_change: observation.wd_change,
        phase_bits: observation.phase_bits,
    };
    geometry
        .validate_recorded_center(&native)
        .map_err(|e| e.to_string())?;
    Ok(native)
}

/// Explicit V10S readback route for the existing Arrow and PathB consumers.
/// The owner retains the returned ledger alongside the native projections.
/// A queue selected as native continues to use GpuSpikeEvent's native decoder.
#[cfg(feature = "gpu")]
pub fn decode_spike_queue_for_arrow_and_pathb(
    raw: &[u8],
    count: u32,
    capacity: u32,
    limits: BiprismDecodeLimits,
    observations: &[NativeLifObservation],
    geometry: crate::sdst_bridge::SdstGridGeometry,
) -> Result<DecodedVoronoiSpikeQueue, String> {
    decode_spike_queue_for_arrow_and_pathb_in_frame(
        raw,
        count,
        capacity,
        limits,
        observations,
        geometry,
        BiprismEventFrame::AtomicBohr,
    )
}

#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn decode_spike_queue_for_arrow_and_pathb_in_frame(
    raw: &[u8],
    count: u32,
    capacity: u32,
    limits: BiprismDecodeLimits,
    observations: &[NativeLifObservation],
    geometry: crate::sdst_bridge::SdstGridGeometry,
    frame: BiprismEventFrame,
) -> Result<DecodedVoronoiSpikeQueue, String> {
    if observations.len() != count as usize {
        return Err("each V10S row requires its matched native observer context".into());
    }
    let ledger = decode_v10_spike_queue(raw, count, capacity, limits)?;
    let native_events = ledger
        .iter()
        .zip(observations)
        .enumerate()
        .map(|(i, (r, o))| {
            project_v10_to_native_in_frame(&r.event, *o, limits, geometry, frame)
                .map_err(|e| format!("V10S projection row {i}: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DecodedVoronoiSpikeQueue {
        event_frame: frame,
        native_events,
        ledger,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn limits() -> BiprismDecodeLimits {
        BiprismDecodeLimits {
            num_atoms: 2685,
            num_residues: 169,
            total_voxels: 4096,
            expected_step: Some(17),
        }
    }
    fn event() -> GpuSpikeEventCanonical96Ffi {
        let m = 65_535u32;
        let th = 1.1f32;
        let v = 72_088.507_812_5f32;
        let hi = (m as f32) * th;
        let lo = (m as f32).mul_add(th, -hi);
        let r = (-(m as f32)).mul_add(th, v);
        GpuSpikeEventCanonical96Ffi {
            pos_xyz: [1.0, 2.0, 3.0],
            step_tau: 17,
            atom_or_site_id: 20,
            spike_amplitude: hi,
            uv_or_holonomy_val: 0.0,
            voronoi_dual_vol: 1.0,
            pocket_tile_id: 0,
            residue_id: 0,
            multiplicity_m: m,
            v_pre_reset: v,
            q_discharged: hi,
            q_discharged_lo: lo,
            r_post_reset: r,
            lif_threshold_used: th,
            leaf_id: 7,
            vertex_id: 191,
            voxel_cell_idx: 123,
            event_schema_magic: SPIKE_EVENT_V10_SCHEMA_MAGIC,
            coupled_force_xyz: [1.0, -1.0, 0.0],
            q_lo_product_raw_bits: lo.to_bits(),
            reserved_v5: 0,
        }
    }
    #[test]
    fn unaligned_expanded_ledger_round_trip() {
        let e = event();
        assert_ne!(
            f64::from(e.q_discharged) + f64::from(e.r_post_reset),
            f64::from(e.v_pre_reset)
        );
        let mut unaligned = vec![0u8];
        unaligned.extend(e.to_le_bytes());
        assert_eq!(
            GpuSpikeEventCanonical96Ffi::from_le_bytes(&unaligned[1..], limits()).unwrap(),
            e
        );
    }
    #[test]
    fn malformed_ledger_and_identity_are_refused() {
        let controls: &[fn(&mut GpuSpikeEventCanonical96Ffi)] = &[
            |e| e.v_pre_reset = f32::NAN,
            |e| e.q_discharged_lo = f32::INFINITY,
            |e| e.multiplicity_m = 0,
            |e| e.multiplicity_m = 65_536,
            |e| e.r_post_reset = e.lif_threshold_used,
            |e| e.r_post_reset = -0.1,
            |e| e.lif_threshold_used = 0.0,
            |e| e.leaf_id = 8,
            |e| e.vertex_id = 192,
            |e| e.atom_or_site_id = 2685,
            |e| e.residue_id = 169,
            |e| e.residue_id = -2,
            |e| e.step_tau = 18,
            |e| e.voxel_cell_idx = 4096,
            |e| e.voxel_cell_idx = -2,
            |e| e.coupled_force_xyz[1] = f32::NAN,
            |e| e.event_schema_magic = 0,
            |e| e.reserved_v5 = 1,
            |e| e.q_lo_product_raw_bits = 0,
            |e| {
                e.q_discharged += 1.0;
                e.q_discharged_lo -= 1.0;
            },
        ];
        for (i, mutate) in controls.iter().enumerate() {
            let mut e = event();
            mutate(&mut e);
            assert!(e.validate(limits()).is_err(), "malformed control {i}");
        }
    }
    #[test]
    fn excluded_residue_and_outside_grid_retain_provenance() {
        let mut e = event();
        e.residue_id = -1;
        e.voxel_cell_idx = -1;
        assert_eq!(
            GpuSpikeEventCanonical96Ffi::from_le_bytes(&e.to_le_bytes(), limits()).unwrap(),
            e
        );
    }
    #[test]
    fn queue_bounds_and_row_ids() {
        let bytes = event().to_le_bytes();
        assert!(decode_v10_spike_queue(&bytes, 2, 1, limits()).is_err());
        assert!(decode_v10_spike_queue(&bytes[..95], 1, 1, limits()).is_err());
        let raw = [bytes, bytes].concat();
        let rows = decode_v10_spike_queue(&raw, 2, 2, limits()).unwrap();
        assert_eq!(rows.iter().map(|r| r.queue_row).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(
            rows.iter()
                .map(|r| u64::from(r.event.multiplicity_m))
                .sum::<u64>(),
            131070
        );
        assert!(decode_v10_spike_queue(&[], 0, 0, limits())
            .unwrap()
            .is_empty());
    }
    #[test]
    fn sub_f64_ulp_remainder_is_not_lost_by_validation() {
        let mut e = event();
        e.multiplicity_m = 1;
        e.lif_threshold_used = 1.0;
        e.v_pre_reset = 1.0;
        e.q_discharged = 1.0;
        e.q_discharged_lo = 0.0;
        e.q_lo_product_raw_bits = 0;
        e.r_post_reset = 1e-30;
        assert_eq!(1.0f64 + f64::from(e.r_post_reset), 1.0);
        assert!(e.validate(limits()).is_err());
    }
    #[test]
    fn native_companion_decodes_unaligned_exact_64_byte_records() {
        let words = [
            17u32,
            7 | (191 << 16),
            20,
            123,
            0.75f32.to_bits(),
            0.125f32.to_bits(),
            2,
            777,
            0, 0, 0, NATIVE_LIF_OBSERVATION_MAGIC, 1, 0, 2, 0,
        ];
        let mut bytes = vec![0xff];
        for word in words {
            bytes.extend(word.to_le_bytes());
        }
        let observations =
            decode_native_lif_observation_queue(&bytes[1..], 1, 1, limits()).unwrap();
        let o = observations[0];
        assert_eq!(
            (
                o.step_tau,
                o.leaf_id,
                o.vertex_id,
                o.atom_or_site_id,
                o.voxel_cell_idx
            ),
            (17, 7, 191, 20, 123)
        );
        assert_eq!(
            (
                o.water_density,
                o.wd_change,
                o.n_nearby_excited,
                o.phase_bits
            ),
            (0.75, 0.125, 2, 777)
        );
        assert!(decode_native_lif_observation_queue(&bytes[1..32], 1, 1, limits()).is_err());
        assert!(decode_native_lif_observation_queue(&bytes[1..], 2, 1, limits()).is_err());
        let mut invalid = bytes[1..].to_vec();
        invalid[28..32].copy_from_slice(&1024u32.to_le_bytes());
        assert!(NativeLifObservation::from_le_bytes(&invalid, limits()).is_err());
        invalid[28..32].copy_from_slice(&777u32.to_le_bytes());
        invalid[16..20].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(NativeLifObservation::from_le_bytes(&invalid, limits()).is_err());
        assert_eq!(o.to_le_bytes().as_slice(), &bytes[1..]);
        for offset in [44, 48, 56] { let mut bad=o.to_le_bytes(); bad[offset..offset+4].fill(0); assert!(NativeLifObservation::from_le_bytes(&bad,limits()).is_err()); }
    }
}
