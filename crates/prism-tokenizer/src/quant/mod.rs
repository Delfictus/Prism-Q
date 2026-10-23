//! §7 — per-field mixture-of-quantizers. One quantizer per ontological field, geometry
//! matched to the field's true distribution.
//!
//! Every quantizer implements [`crate::types::Quantizer`]. `fit` accumulates a code-usage
//! histogram so `perplexity` (§13 codebook-health monitor) is exact: `2^H(usage)`.

pub mod circular;
pub mod critical;
pub mod enum_cat;
pub mod lattice_vq3d;
pub mod linear;
pub mod logmag;
pub mod scalar;
pub mod sentinel;
pub mod smiles;
pub mod spectral;

pub use circular::CircularQuantizer;
pub use critical::CriticalRegionQuantizer;
pub use enum_cat::EnumQuantizer;
pub use lattice_vq3d::{Bcc3dQuantizer, Vq3dEma};
pub use linear::LinearQuantizer;
pub use logmag::LogMagnitudeQuantizer;
pub use scalar::{BooleanQuantizer, JoinKeyQuantizer};
pub use sentinel::SentinelQuantizer;
pub use smiles::{tokenize as smiles_tokenize, tokenize_deep as smiles_tokenize_deep, SmilesVocab};
pub use spectral::SpectralLineQuantizer;

use crate::types::{DecisionSpec, DistortionBound, FieldGeometry, FieldSpec, Quantizer};

/// Construct the scalar quantizer for a field geometry (the dispatch that ties the §6 registry
/// to the §7 quantizers). Returns `None` for non-scalar geometries (LatticeVq3d is a 3-vector;
/// SmilesGrammar / OpcodeIsa are sequences; Opaque is a sidecar escape) — those are handled by
/// their own (non-`f64`-scalar) encoders, not this dispatch.
pub fn build_scalar_quantizer(g: &FieldGeometry) -> Option<Box<dyn Quantizer>> {
    use FieldGeometry as G;
    Some(match g {
        G::Enum { cardinality } => Box::new(EnumQuantizer::new(*cardinality)),
        G::SpectralLine {
            lines,
            residual_bits,
        } => Box::new(SpectralLineQuantizer::new(lines, *residual_bits)),
        G::Circular { bits, period } => Box::new(CircularQuantizer::new(*bits, period.period())),
        G::LogMagnitude { bits, floor, ceil } => Box::new(LogMagnitudeQuantizer::new(
            *bits,
            *floor as f64,
            *ceil as f64,
        )),
        G::LinearBounded { bits, lo, hi } => {
            Box::new(LinearQuantizer::new(*bits, *lo as f64, *hi as f64))
        }
        G::Sentinel { value, inner } => {
            let inner_q = build_scalar_quantizer(inner)?;
            Box::new(SentinelQuantizer::new(*value as f64, inner_q))
        }
        G::Boolean => Box::new(BooleanQuantizer::new()),
        G::JoinKey => Box::new(JoinKeyQuantizer::new()),
        // non-scalar — handled outside this dispatch.
        G::LatticeVq3d { .. } | G::SmilesGrammar | G::OpcodeIsa | G::Opaque => return None,
    })
}

/// Build the quantizer for a full field spec: a decision-aware critical-region (companded)
/// quantizer for `Threshold` fields (INV-8 §3), else the geometry's scalar quantizer. Returns
/// None for non-scalar geometries (handled separately).
pub fn build_field_quantizer(spec: &FieldSpec) -> Option<Box<dyn Quantizer>> {
    if let DecisionSpec::Threshold { points, region } = spec.decision_relevance {
        let (lo, hi, tail_log) = match &spec.geometry {
            FieldGeometry::LinearBounded { lo, hi, .. } => (*lo as f64, *hi as f64, false),
            FieldGeometry::LogMagnitude { floor, ceil, .. } => (*floor as f64, *ceil as f64, true),
            other => return build_scalar_quantizer(other),
        };
        let pts: Vec<f64> = points.iter().map(|&x| x as f64).collect();
        let region = (region.0 as f64, region.1 as f64);
        let fine_step = critical_fine_step(spec.max_distortion, region);
        return Some(Box::new(CriticalRegionQuantizer::new(
            lo, hi, region, &pts, fine_step, tail_log,
        )));
    }
    build_scalar_quantizer(&spec.geometry)
}

/// Fine-bin width (= 2·max_distortion in the field's units) for a critical-region quantizer.
fn critical_fine_step(bound: DistortionBound, region: (f64, f64)) -> f64 {
    match bound {
        DistortionBound::AbsoluteLinear(b)
        | DistortionBound::AbsoluteAngular(b)
        | DistortionBound::AbsoluteEuclidean(b) => 2.0 * b as f64,
        DistortionBound::RelativeFraction(f) => {
            let scale = region.0.abs().max(region.1.abs()).max(1.0);
            2.0 * f as f64 * scale
        }
        DistortionBound::Exact => 1e-6,
    }
}

/// `2^H` of a code-usage histogram (effective number of codes in use). 0 for an empty
/// histogram; a collapsed codebook → ~1 (§13 alarms below a floor).
pub(crate) fn usage_perplexity(hist: &[u64]) -> f64 {
    let total: u64 = hist.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    let mut h = 0.0f64;
    for &c in hist {
        if c == 0 {
            continue;
        }
        let p = c as f64 / t;
        h -= p * p.log2();
    }
    2f64.powf(h)
}

/// Smallest bit budget that can index `cardinality` codes (min 1).
pub(crate) fn bits_for(cardinality: u32) -> u8 {
    if cardinality <= 1 {
        1
    } else {
        (32 - (cardinality - 1).leading_zeros()) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::FieldRegistry;
    use crate::types::{ArtifactFamily, FieldGeometry as G};

    #[test]
    fn dispatch_builds_a_quantizer_for_every_scalar_registry_field() {
        // Every registered field whose geometry is scalar must yield a quantizer; the four
        // non-scalar geometries must return None (handled elsewhere). This is the registry→
        // quantizer wiring proof.
        let reg = FieldRegistry::build();
        let mut scalar = 0;
        let mut non_scalar = 0;
        for spec in reg.iter() {
            let q = build_scalar_quantizer(&spec.geometry);
            match &spec.geometry {
                G::LatticeVq3d { .. } | G::SmilesGrammar | G::OpcodeIsa | G::Opaque => {
                    assert!(
                        q.is_none(),
                        "{}/{} should be non-scalar",
                        spec_family(spec),
                        spec.field
                    );
                    non_scalar += 1;
                }
                _ => {
                    assert!(
                        q.is_some(),
                        "{}/{} should build a quantizer",
                        spec_family(spec),
                        spec.field
                    );
                    non_scalar += 0;
                    scalar += 1;
                }
            }
        }
        assert!(scalar > 30, "expected most fields scalar, got {scalar}");
        assert!(
            non_scalar >= 1,
            "expected at least the position/SMILES non-scalar fields"
        );
    }

    fn spec_family(s: &crate::types::FieldSpec) -> String {
        format!("{:?}", s.family)
    }

    #[test]
    fn spike_source_dispatches_to_enum_of_five() {
        let reg = FieldRegistry::build();
        let spec = reg
            .lookup(ArtifactFamily::SpikeEvent, "spike_source")
            .unwrap();
        let q = build_scalar_quantizer(&spec.geometry).unwrap();
        // 5 categories → exactly 3 bits, codes 0..4 identity.
        assert_eq!(q.bit_budget(), 3);
        assert_eq!(q.encode(2.0).0, 2);
    }

    #[test]
    fn wavelength_dispatches_to_spectral_line() {
        let reg = FieldRegistry::build();
        let spec = reg
            .lookup(ArtifactFamily::SpikeEvent, "wavelength_nm")
            .unwrap();
        let q = build_scalar_quantizer(&spec.geometry).unwrap();
        let (code, d) = q.encode(280.0);
        assert!(d.value() < 1e-9);
        assert_eq!(q.decode(code), 280.0);
    }

    #[test]
    fn aromatic_residue_id_sentinel_dispatches_through_inner_joinkey() {
        let reg = FieldRegistry::build();
        let spec = reg
            .lookup(ArtifactFamily::SpikeEvent, "aromatic_residue_id")
            .unwrap();
        let q = build_scalar_quantizer(&spec.geometry).unwrap();
        // -1 sentinel decodes back to -1; a real id passes through the inner join key.
        let (s_code, _) = q.encode(-1.0);
        assert_eq!(q.decode(s_code), -1.0);
    }
}
