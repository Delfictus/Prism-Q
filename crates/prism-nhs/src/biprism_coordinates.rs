//! Persistent two-float coordinate authority for the explicit Biprism profile.
//! High projections remain available to legacy-shaped output consumers. The
//! physical coordinate is the FP64 sum of both represented components.
use cudarc::driver::DeviceRepr;
use serde::{Deserialize, Serialize};

pub const COORDINATE_MAGIC: u32 = 0x3150_4342;
pub const COORDINATE_VERSION: u32 = 1;
pub const NATIVE_ANGSTROM: u32 = 1;
pub const ATOMIC_BOHR: u32 = 2;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BiprismCoordinateOwnerFfi {
    pub magic: u32,
    pub version: u32,
    pub leaf_id: u32,
    pub atom_count: u32,
    pub generation: u64,
    pub unit_frame: u32,
    pub initialized_mask: u32,
    pub current_hi: u64,
    pub current_lo: u64,
    pub step_start_hi: u64,
    pub step_start_lo: u64,
    pub reference_hi: u64,
    pub reference_lo: u64,
    pub reserved: [u64; 2],
}
unsafe impl DeviceRepr for BiprismCoordinateOwnerFfi {}
const _: () = {
    assert!(core::mem::size_of::<BiprismCoordinateOwnerFfi>() == 96);
    assert!(core::mem::align_of::<BiprismCoordinateOwnerFfi>() == 16);
    assert!(core::mem::offset_of!(BiprismCoordinateOwnerFfi, current_hi) == 32);
    assert!(core::mem::offset_of!(BiprismCoordinateOwnerFfi, step_start_hi) == 48);
    assert!(core::mem::offset_of!(BiprismCoordinateOwnerFfi, reference_hi) == 64);
    assert!(core::mem::offset_of!(BiprismCoordinateOwnerFfi, reserved) == 80);
};
impl BiprismCoordinateOwnerFfi {
    pub fn validate(&self, leaf: u32, atoms: u32, frame: u32, high: u64) -> Result<(), String> {
        if self.magic != COORDINATE_MAGIC || self.version != COORDINATE_VERSION
            || self.leaf_id != leaf || leaf >= 8 || self.atom_count != atoms || atoms < 256
            || self.generation == 0 || self.unit_frame != frame || !matches!(frame, 1 | 2)
            || self.initialized_mask != 7 || self.current_hi != high || self.reserved != [0; 2]
        { return Err("coordinate owner identity/units/initialization mismatch".into()); }
        let pointers = [self.current_hi, self.current_lo, self.step_start_hi,
            self.step_start_lo, self.reference_hi, self.reference_lo];
        let bytes = u64::from(atoms).checked_mul(12).ok_or("coordinate extent overflow")?;
        for (i, &p) in pointers.iter().enumerate() {
            let end = p.checked_add(bytes).ok_or("coordinate address overflow")?;
            if p == 0 || p % 4 != 0 || pointers[..i].iter().any(|&q| p < q + bytes && q < end) {
                return Err("coordinate planes are missing, unaligned or overlap".into());
            }
        }
        Ok(())
    }
}

/// Pointer-free checkpoint identity. Device addresses are rebuilt from the
/// destination owner's retained allocations; they are never restored as data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BiprismCoordinateIdentity {
    pub version: u32,
    pub leaf_id: u32,
    pub atom_count: u32,
    pub generation: u64,
    pub unit_frame: u32,
}
impl From<BiprismCoordinateOwnerFfi> for BiprismCoordinateIdentity {
    fn from(o: BiprismCoordinateOwnerFfi) -> Self {
        Self { version: o.version, leaf_id: o.leaf_id, atom_count: o.atom_count,
            generation: o.generation, unit_frame: o.unit_frame }
    }
}

/// Complete retained coordinate state. This is a coordinate-owner image, not
/// a claim that other electronic/RNG/graph state forms a restart checkpoint.
#[derive(Clone, Debug)]
pub struct BiprismCoordinateSnapshot {
    pub identity:BiprismCoordinateIdentity,
    pub current_high:Vec<f32>,pub current_low:Vec<f32>,
    pub step_start_high:Vec<f32>,pub step_start_low:Vec<f32>,
    pub reference_high:Vec<f32>,pub reference_low:Vec<f32>,
}
impl BiprismCoordinateSnapshot {
    pub fn planes(&self)->[(&'static str,&[f32]);6]{[
        ("current_high",&self.current_high),("current_low",&self.current_low),
        ("step_start_high",&self.step_start_high),("step_start_low",&self.step_start_low),
        ("mesh_reference_high",&self.reference_high),("mesh_reference_low",&self.reference_low)]}
    pub fn validate(&self)->Result<(),String>{
        if self.identity.version!=COORDINATE_VERSION||self.identity.generation==0||self.identity.leaf_id>=8
            ||!matches!(self.identity.unit_frame,1|2)||self.identity.atom_count<256 {return Err("coordinate snapshot identity".into());}
        let n=self.identity.atom_count as usize*3;
        if self.planes().iter().any(|(_,p)|p.len()!=n||p.iter().any(|v|!v.is_finite())) {return Err("coordinate snapshot population/nonfinite".into());}
        Ok(())
    }
}

pub fn represented_positions(hi: &[f32], lo: &[f32]) -> Result<Vec<f64>, String> {
    if hi.len() != lo.len() || hi.len() % 3 != 0 { return Err("coordinate split extent mismatch".into()); }
    hi.iter().zip(lo).map(|(&h, &l)| {
        let x = f64::from(h) + f64::from(l);
        if h.is_finite() && l.is_finite() && x.is_finite() { Ok(x) }
        else { Err("coordinate split contains a nonfinite value".into()) }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn low_survives_fp32_high_projection() {
        let h = 64.0f32; let l = 1e-7f32;
        assert_eq!(h + l, h);
        assert!(represented_positions(&[h; 3], &[l; 3]).unwrap()[0] > f64::from(h));
        assert!(represented_positions(&[h; 3], &[f32::NAN; 3]).is_err());
    }
    #[test]
    fn owner_refuses_alias_and_wrong_high() {
        let mut o = BiprismCoordinateOwnerFfi { magic: COORDINATE_MAGIC, version: 1,
            leaf_id: 0, atom_count: 256, generation: 1, unit_frame: 1, initialized_mask: 7,
            current_hi: 0x1000, current_lo: 0x2000, step_start_hi: 0x3000,
            step_start_lo: 0x4000, reference_hi: 0x5000, reference_lo: 0x6000, reserved: [0; 2] };
        assert!(o.validate(0, 256, 1, 0x1000).is_ok());
        assert!(o.validate(0, 256, 1, 0x2000).is_err());
        o.reference_lo = o.current_lo;
        assert!(o.validate(0, 256, 1, 0x1000).is_err());
    }
}
