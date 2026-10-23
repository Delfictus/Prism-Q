//! Raw vector tensor loader for ASC, forces, and aromatic centroid surfaces.
//!
//! These surfaces are producer-verified raw little-endian `f32` XYZ triples:
//! `asc_vectors.bin`, `forces_final.bin`, and `aromatic_centroids_final.bin`.
//! The tokenizer records compact structural invariants and leaves raw byte identity
//! in source lineage; it never treats storage bytes themselves as semantic tokens.

use super::CodecError;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorTensorKind {
    AscVectors,
    ForcesFinal,
    AromaticCentroidsFinal,
}

impl VectorTensorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AscVectors => "asc_vectors",
            Self::ForcesFinal => "forces_final",
            Self::AromaticCentroidsFinal => "aromatic_centroids_final",
        }
    }

    pub fn codec_label(self) -> &'static str {
        match self {
            Self::AscVectors => "asc_vectors.bin",
            Self::ForcesFinal => "forces_final.bin",
            Self::AromaticCentroidsFinal => "aromatic_centroids_final.bin",
        }
    }
}

#[derive(Clone, Debug)]
pub struct VectorTensorDecoded {
    pub kind: VectorTensorKind,
    pub size_bytes: usize,
    pub n_vectors: usize,
    pub finite_vectors: usize,
    pub nonfinite_vectors: usize,
    pub norm_mean: f64,
    pub norm_min: f64,
    pub norm_max: f64,
    pub centroid_xyz: [f64; 3],
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub z_min: f64,
    pub z_max: f64,
    pub span_x: f64,
    pub span_y: f64,
    pub span_z: f64,
    pub radius_gyration: f64,
    pub mean_step_distance: f64,
    pub checksum_fnv1a64: u64,
}

#[inline]
fn fnv_update(mut hash: u64, bytes: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn finite_json(v: f64) -> Value {
    if v.is_finite() {
        json!(v)
    } else {
        Value::Null
    }
}

pub fn decode_vector_tensor(
    raw: &[u8],
    kind: VectorTensorKind,
) -> Result<VectorTensorDecoded, CodecError> {
    if raw.is_empty() {
        return Err(CodecError::Format(format!(
            "{} is empty",
            kind.codec_label()
        )));
    }
    if raw.len() % 12 != 0 {
        return Err(CodecError::Format(format!(
            "{} length {} is not aligned to f32 xyz triples",
            kind.codec_label(),
            raw.len()
        )));
    }

    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    let checksum_fnv1a64 = fnv_update(FNV_OFFSET, raw);

    let n_vectors = raw.len() / 12;
    let mut finite_vectors = 0usize;
    let mut nonfinite_vectors = 0usize;
    let mut norm_sum = 0.0f64;
    let mut norm_min = f64::INFINITY;
    let mut norm_max = f64::NEG_INFINITY;
    let mut x_min = f64::INFINITY;
    let mut y_min = f64::INFINITY;
    let mut z_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    let mut sum = [0.0f64; 3];
    let mut sum_sq_radius = 0.0f64;
    let mut previous: Option<[f64; 3]> = None;
    let mut step_distance_sum = 0.0f64;
    let mut step_distance_count = 0usize;

    let mut finite_points = Vec::with_capacity(n_vectors.min(4096));
    for triple in raw.chunks_exact(12) {
        let x = f32::from_le_bytes(triple[0..4].try_into().unwrap()) as f64;
        let y = f32::from_le_bytes(triple[4..8].try_into().unwrap()) as f64;
        let z = f32::from_le_bytes(triple[8..12].try_into().unwrap()) as f64;
        if x.is_finite() && y.is_finite() && z.is_finite() {
            finite_vectors += 1;
            x_min = x_min.min(x);
            y_min = y_min.min(y);
            z_min = z_min.min(z);
            x_max = x_max.max(x);
            y_max = y_max.max(y);
            z_max = z_max.max(z);
            sum[0] += x;
            sum[1] += y;
            sum[2] += z;
            let norm = (x.mul_add(x, y.mul_add(y, z * z))).sqrt();
            norm_sum += norm;
            norm_min = norm_min.min(norm);
            norm_max = norm_max.max(norm);
            if let Some(prev) = previous {
                let dx = x - prev[0];
                let dy = y - prev[1];
                let dz = z - prev[2];
                step_distance_sum += (dx.mul_add(dx, dy.mul_add(dy, dz * dz))).sqrt();
                step_distance_count += 1;
            }
            previous = Some([x, y, z]);
            finite_points.push([x, y, z]);
        } else {
            nonfinite_vectors += 1;
            previous = None;
        }
    }

    let centroid_xyz = if finite_vectors > 0 {
        [
            sum[0] / finite_vectors as f64,
            sum[1] / finite_vectors as f64,
            sum[2] / finite_vectors as f64,
        ]
    } else {
        [f64::NAN, f64::NAN, f64::NAN]
    };

    for point in &finite_points {
        let dx = point[0] - centroid_xyz[0];
        let dy = point[1] - centroid_xyz[1];
        let dz = point[2] - centroid_xyz[2];
        sum_sq_radius += dx.mul_add(dx, dy.mul_add(dy, dz * dz));
    }

    let norm_mean = if finite_vectors > 0 {
        norm_sum / finite_vectors as f64
    } else {
        f64::NAN
    };
    let radius_gyration = if finite_vectors > 0 {
        (sum_sq_radius / finite_vectors as f64).sqrt()
    } else {
        f64::NAN
    };
    let mean_step_distance = if step_distance_count > 0 {
        step_distance_sum / step_distance_count as f64
    } else {
        f64::NAN
    };

    Ok(VectorTensorDecoded {
        kind,
        size_bytes: raw.len(),
        n_vectors,
        finite_vectors,
        nonfinite_vectors,
        norm_mean,
        norm_min,
        norm_max,
        centroid_xyz,
        x_min,
        x_max,
        y_min,
        y_max,
        z_min,
        z_max,
        span_x: x_max - x_min,
        span_y: y_max - y_min,
        span_z: z_max - z_min,
        radius_gyration,
        mean_step_distance,
        checksum_fnv1a64,
    })
}

impl VectorTensorDecoded {
    pub fn to_summary_json(&self) -> Value {
        json!({
            "kind": self.kind.as_str(),
            "status": "decoded_binary",
            "layout": "raw f32 AoS n_vectors x 3",
            "source_codec": self.kind.codec_label(),
            "size_bytes": self.size_bytes,
            "n_vectors": self.n_vectors,
            "finite_vectors": self.finite_vectors,
            "nonfinite_vectors": self.nonfinite_vectors,
            "norm_mean": finite_json(self.norm_mean),
            "norm_min": finite_json(self.norm_min),
            "norm_max": finite_json(self.norm_max),
            "centroid_xyz": self.centroid_xyz.iter().map(|v| finite_json(*v)).collect::<Vec<_>>(),
            "x_min": finite_json(self.x_min),
            "x_max": finite_json(self.x_max),
            "y_min": finite_json(self.y_min),
            "y_max": finite_json(self.y_max),
            "z_min": finite_json(self.z_min),
            "z_max": finite_json(self.z_max),
            "span_x": finite_json(self.span_x),
            "span_y": finite_json(self.span_y),
            "span_z": finite_json(self.span_z),
            "radius_gyration": finite_json(self.radius_gyration),
            "mean_step_distance": finite_json(self.mean_step_distance),
            "checksum_fnv1a64": format!("{:016x}", self.checksum_fnv1a64),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_xyz(raw: &mut Vec<u8>, xyz: [f32; 3]) {
        for v in xyz {
            raw.extend_from_slice(&v.to_le_bytes());
        }
    }

    #[test]
    fn vector_tensor_decodes_f32_xyz_triples() {
        let mut raw = Vec::new();
        push_xyz(&mut raw, [1.0, 2.0, 2.0]);
        push_xyz(&mut raw, [2.0, 2.0, 1.0]);
        let decoded = decode_vector_tensor(&raw, VectorTensorKind::AscVectors).unwrap();
        assert_eq!(decoded.n_vectors, 2);
        assert_eq!(decoded.finite_vectors, 2);
        assert_eq!(decoded.nonfinite_vectors, 0);
        assert_eq!(decoded.centroid_xyz, [1.5, 2.0, 1.5]);
        assert!((decoded.norm_mean - 3.0).abs() < 1e-6);
        let json = decoded.to_summary_json();
        assert_eq!(json["kind"], "asc_vectors");
        assert_eq!(json["n_vectors"], 2);
    }

    #[test]
    fn vector_tensor_rejects_unaligned_payload() {
        assert!(decode_vector_tensor(&[0, 1, 2, 3], VectorTensorKind::ForcesFinal).is_err());
    }
}
