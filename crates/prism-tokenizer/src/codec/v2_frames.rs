//! Current PRISM V2 trajectory stream loader.
//!
//! `*_v2_frames.bin` is not the legacy `PRISM4D\0` 24-byte `.frames.bin`
//! sidecar. The current writer emits:
//!
//! ```text
//! [0..8] frame_count u64
//! repeated frame_count times:
//!   step u64
//!   n_floats u32
//!   positions [f32; n_floats]  // xyz triples, so n_floats % 3 == 0
//! ```

use super::CodecError;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct V2Frame {
    pub frame_index: usize,
    pub step: u64,
    pub n_floats: usize,
    pub n_atoms: usize,
    pub positions: Vec<f32>,
    pub frame_hash64: u64,
}

#[derive(Clone, Debug)]
pub struct V2FramesDecoded {
    pub frame_count_header: usize,
    pub frames: Vec<V2Frame>,
    pub rolling_hash64: u64,
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

#[inline]
fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

#[inline]
fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes([
        b[o],
        b[o + 1],
        b[o + 2],
        b[o + 3],
        b[o + 4],
        b[o + 5],
        b[o + 6],
        b[o + 7],
    ])
}

fn need(raw: &[u8], end: usize, what: &str) -> Result<(), CodecError> {
    if end > raw.len() {
        Err(CodecError::Format(format!(
            "V2 frames truncated reading {what}"
        )))
    } else {
        Ok(())
    }
}

fn u64_to_usize(v: u64, what: &str) -> Result<usize, CodecError> {
    usize::try_from(v).map_err(|_| CodecError::Format(format!("V2 frames {what} overflows usize")))
}

fn frame_hash64(step: u64, n_floats: u32, payload_bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    let mut hash = fnv_update(FNV_OFFSET, &step.to_le_bytes());
    hash = fnv_update(hash, &n_floats.to_le_bytes());
    fnv_update(hash, payload_bytes)
}

pub fn decode_v2_frames(raw: &[u8]) -> Result<V2FramesDecoded, CodecError> {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    need(raw, 8, "frame_count header")?;
    let frame_count = u64_to_usize(u64_at(raw, 0), "frame_count")?;
    let mut cursor = 8usize;
    let mut frames = Vec::with_capacity(frame_count);
    let mut rolling_hash = FNV_OFFSET;
    for frame_index in 0..frame_count {
        need(raw, cursor + 12, "frame header")?;
        let step = u64_at(raw, cursor);
        cursor += 8;
        let n_floats_u32 = u32_at(raw, cursor);
        cursor += 4;
        let n_floats = n_floats_u32 as usize;
        if n_floats == 0 || n_floats % 3 != 0 {
            return Err(CodecError::Format(format!(
                "V2 frame {frame_index} n_floats={n_floats} is not non-empty xyz triples"
            )));
        }
        let payload_len = n_floats
            .checked_mul(4)
            .ok_or_else(|| CodecError::Format("V2 frame payload length overflow".into()))?;
        need(raw, cursor + payload_len, "frame payload")?;
        let payload = &raw[cursor..cursor + payload_len];
        let hash = frame_hash64(step, n_floats_u32, payload);
        rolling_hash = fnv_update(rolling_hash, &hash.to_le_bytes());
        let positions = payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        frames.push(V2Frame {
            frame_index,
            step,
            n_floats,
            n_atoms: n_floats / 3,
            positions,
            frame_hash64: hash,
        });
        cursor += payload_len;
    }
    if cursor != raw.len() {
        return Err(CodecError::Format(format!(
            "V2 frames has {} trailing bytes",
            raw.len() - cursor
        )));
    }
    Ok(V2FramesDecoded {
        frame_count_header: frame_count,
        frames,
        rolling_hash64: rolling_hash,
    })
}

fn finite_json(v: f64) -> Value {
    if v.is_finite() {
        json!(v)
    } else {
        Value::Null
    }
}

fn frame_summary(frame: &V2Frame) -> Value {
    let mut x_min = f64::INFINITY;
    let mut y_min = f64::INFINITY;
    let mut z_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    let mut norm_sum = 0.0f64;
    let mut norm_max = f64::NEG_INFINITY;
    let mut finite_atoms = 0usize;
    for xyz in frame.positions.chunks_exact(3) {
        let x = xyz[0] as f64;
        let y = xyz[1] as f64;
        let z = xyz[2] as f64;
        if x.is_finite() && y.is_finite() && z.is_finite() {
            x_min = x_min.min(x);
            y_min = y_min.min(y);
            z_min = z_min.min(z);
            x_max = x_max.max(x);
            y_max = y_max.max(y);
            z_max = z_max.max(z);
            let norm = (x.mul_add(x, y.mul_add(y, z * z))).sqrt();
            norm_sum += norm;
            norm_max = norm_max.max(norm);
            finite_atoms += 1;
        }
    }
    let norm_mean = if finite_atoms > 0 {
        norm_sum / finite_atoms as f64
    } else {
        f64::NAN
    };
    json!({
        "frame_index": frame.frame_index,
        "step": frame.step,
        "n_floats": frame.n_floats,
        "n_atoms": frame.n_atoms,
        "finite_atoms": finite_atoms,
        "x_min": finite_json(x_min),
        "x_max": finite_json(x_max),
        "y_min": finite_json(y_min),
        "y_max": finite_json(y_max),
        "z_min": finite_json(z_min),
        "z_max": finite_json(z_max),
        "norm_mean": finite_json(norm_mean),
        "norm_max": finite_json(norm_max),
        "frame_hash64": format!("{:016x}", frame.frame_hash64),
    })
}

impl V2FramesDecoded {
    pub fn to_summary_json(&self) -> Value {
        json!({
            "schema_kind": "PRISM_V2_TRAJECTORY_FRAMES",
            "source_codec": "v2_frames.bin",
            "frame_count_header": self.frame_count_header,
            "decoded_frame_count": self.frames.len(),
            "rolling_hash64": format!("{:016x}", self.rolling_hash64),
            "frames": self.frames.iter().map(frame_summary).collect::<Vec<_>>(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_v2_frames() -> Vec<u8> {
        let frames: [(u64, Vec<f32>); 2] = [
            (5_000, vec![1.0, -2.0, 3.5, 4.0, 5.25, -6.75]),
            (10_000, vec![7.0, 8.5, 9.25, -10.0, 11.5, 12.75]),
        ];
        let mut raw = Vec::new();
        raw.extend_from_slice(&(frames.len() as u64).to_le_bytes());
        for (step, positions) in frames {
            raw.extend_from_slice(&step.to_le_bytes());
            raw.extend_from_slice(&(positions.len() as u32).to_le_bytes());
            for x in positions {
                raw.extend_from_slice(&x.to_le_bytes());
            }
        }
        raw
    }

    #[test]
    fn decodes_current_v2_frames_layout() {
        let decoded = decode_v2_frames(&build_v2_frames()).unwrap();
        assert_eq!(decoded.frame_count_header, 2);
        assert_eq!(decoded.frames[0].step, 5_000);
        assert_eq!(decoded.frames[0].n_atoms, 2);
        assert_eq!(decoded.frames[1].positions[5], 12.75);
        let json = decoded.to_summary_json();
        assert_eq!(json["schema_kind"], "PRISM_V2_TRAJECTORY_FRAMES");
        assert_eq!(json["frames"][0]["n_atoms"], 2);
        assert_eq!(json["frames"][0]["x_min"], 1.0);
    }

    #[test]
    fn rejects_truncated_or_non_xyz_v2_frames() {
        let mut bad = build_v2_frames();
        bad.pop();
        assert!(decode_v2_frames(&bad).is_err());

        let mut non_xyz = Vec::new();
        non_xyz.extend_from_slice(&1u64.to_le_bytes());
        non_xyz.extend_from_slice(&1u64.to_le_bytes());
        non_xyz.extend_from_slice(&2u32.to_le_bytes());
        non_xyz.extend_from_slice(&1.0f32.to_le_bytes());
        non_xyz.extend_from_slice(&2.0f32.to_le_bytes());
        assert!(decode_v2_frames(&non_xyz).is_err());
    }
}
