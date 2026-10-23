//! PRKCC001 — engine-written KCC descriptor envelope.
//!
//! The engine writes `*_kcc_v2full.bin` as a PRISM evidence envelope:
//! common variable-length header, a field-section payload, FNV-1a checksum,
//! and `PRKCCEND` trailer. This decoder validates the envelope, hydrates the
//! typed numeric sections, and exposes the JSON shape already recognized by
//! the tokenizer router as `KccDescriptor`.

use super::CodecError;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub const PRKCC001_MAGIC: &[u8; 8] = b"PRKCC001";
pub const PRKCC001_TRAILER: &[u8; 8] = b"PRKCCEND";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prkcc001Header {
    pub schema_version: u32,
    pub stream_id: u32,
    pub run_id: String,
    pub stem: String,
    pub record_count: usize,
    pub byte_stride: usize,
    pub payload_size: usize,
    pub payload_offset: usize,
}

#[derive(Clone, Debug)]
pub struct Prkcc001Decoded {
    pub header: Prkcc001Header,
    pub f32_fields: BTreeMap<String, Vec<f32>>,
    pub u32_fields: BTreeMap<String, Vec<u32>>,
    pub checksum: u64,
    pub integrity_ok: bool,
}

#[inline]
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = FNV_OFFSET;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
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
            "PRKCC001 truncated reading {what}"
        )))
    } else {
        Ok(())
    }
}

fn u64_to_usize(v: u64, what: &str) -> Result<usize, CodecError> {
    usize::try_from(v).map_err(|_| CodecError::Format(format!("PRKCC001 {what} overflows usize")))
}

fn parse_utf8(raw: &[u8], start: usize, len: usize, what: &str) -> Result<String, CodecError> {
    need(raw, start + len, what)?;
    std::str::from_utf8(&raw[start..start + len])
        .map(|s| s.to_string())
        .map_err(|err| CodecError::Format(format!("PRKCC001 invalid UTF-8 in {what}: {err}")))
}

fn parse_header(raw: &[u8]) -> Result<Prkcc001Header, CodecError> {
    if raw.len() < 8 || &raw[0..8] != PRKCC001_MAGIC {
        return Err(CodecError::Format("not a PRKCC001 envelope".into()));
    }
    need(raw, 20, "fixed header")?;
    let schema_version = u32_at(raw, 8);
    if schema_version != 1 {
        return Err(CodecError::Format(format!(
            "PRKCC001 unsupported schema_version {schema_version}"
        )));
    }
    let endian = u32_at(raw, 12);
    if endian != 0x0102_0304 {
        return Err(CodecError::Format(format!(
            "PRKCC001 endian marker mismatch: 0x{endian:08x}"
        )));
    }
    let stream_id = u32_at(raw, 16);
    let mut c = 20usize;
    need(raw, c + 8, "run_id_len")?;
    let run_id_len = u64_to_usize(u64_at(raw, c), "run_id_len")?;
    c += 8;
    let run_id = parse_utf8(raw, c, run_id_len, "run_id")?;
    c += run_id_len;

    need(raw, c + 8, "stem_len")?;
    let stem_len = u64_to_usize(u64_at(raw, c), "stem_len")?;
    c += 8;
    let stem = parse_utf8(raw, c, stem_len, "stem")?;
    c += stem_len;

    need(raw, c + 24, "record_count/byte_stride/payload_size")?;
    let record_count = u64_to_usize(u64_at(raw, c), "record_count")?;
    let byte_stride = u64_to_usize(u64_at(raw, c + 8), "byte_stride")?;
    let payload_size = u64_to_usize(u64_at(raw, c + 16), "payload_size")?;
    c += 24;
    if byte_stride != 4 {
        return Err(CodecError::Format(format!(
            "PRKCC001 expected byte_stride=4, got {byte_stride}"
        )));
    }
    Ok(Prkcc001Header {
        schema_version,
        stream_id,
        run_id,
        stem,
        record_count,
        byte_stride,
        payload_size,
        payload_offset: c,
    })
}

fn parse_field_name(payload: &[u8], c: &mut usize) -> Result<String, CodecError> {
    need(payload, *c + 8, "field name length")?;
    let name_len = u64_to_usize(u64_at(payload, *c), "field name length")?;
    *c += 8;
    let name = parse_utf8(payload, *c, name_len, "field name")?;
    *c += name_len;
    Ok(name)
}

fn parse_f32_values(bytes: &[u8], record_count: usize, name: &str) -> Result<Vec<f32>, CodecError> {
    if bytes.len() != record_count * 4 {
        return Err(CodecError::Format(format!(
            "PRKCC001 f32 field {name} has {} bytes for {record_count} records",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn parse_u32_values(bytes: &[u8], record_count: usize, name: &str) -> Result<Vec<u32>, CodecError> {
    if bytes.len() != record_count * 4 {
        return Err(CodecError::Format(format!(
            "PRKCC001 u32 field {name} has {} bytes for {record_count} records",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn decode_prkcc001(raw: &[u8]) -> Result<Prkcc001Decoded, CodecError> {
    let header = parse_header(raw)?;
    let payload_end = header
        .payload_offset
        .checked_add(header.payload_size)
        .ok_or_else(|| CodecError::Format("PRKCC001 payload_end overflow".into()))?;
    need(raw, payload_end + 16, "checksum/trailer")?;
    if &raw[payload_end + 8..payload_end + 16] != PRKCC001_TRAILER {
        return Err(CodecError::Format("PRKCC001 trailer mismatch".into()));
    }
    let payload = &raw[header.payload_offset..payload_end];
    let stored_checksum = u64_at(raw, payload_end);
    let computed_checksum = fnv1a_64(payload);
    if stored_checksum != computed_checksum {
        return Err(CodecError::Format(format!(
            "PRKCC001 checksum mismatch: stored=0x{stored_checksum:016x} computed=0x{computed_checksum:016x}"
        )));
    }

    let mut c = 0usize;
    need(payload, c + 16, "payload preamble")?;
    let n_res = u64_to_usize(u64_at(payload, c), "payload n_residues")?;
    c += 8;
    let field_count = u64_to_usize(u64_at(payload, c), "payload field_count")?;
    c += 8;
    if n_res != header.record_count {
        return Err(CodecError::Format(format!(
            "PRKCC001 n_residues {n_res} != header record_count {}",
            header.record_count
        )));
    }

    let mut f32_fields = BTreeMap::new();
    let mut u32_fields = BTreeMap::new();
    for _ in 0..field_count {
        let name = parse_field_name(payload, &mut c)?;
        need(payload, c + 9, "field dtype/size")?;
        let dtype = payload[c];
        c += 1;
        let section_size = u64_to_usize(u64_at(payload, c), "field section_size")?;
        c += 8;
        need(payload, c + section_size, "field section bytes")?;
        let section = &payload[c..c + section_size];
        c += section_size;
        match dtype {
            1 => {
                f32_fields.insert(name.clone(), parse_f32_values(section, n_res, &name)?);
            }
            2 => {
                u32_fields.insert(name.clone(), parse_u32_values(section, n_res, &name)?);
            }
            other => {
                return Err(CodecError::Format(format!(
                    "PRKCC001 field {name} unsupported dtype {other}"
                )));
            }
        }
    }
    if c != payload.len() {
        return Err(CodecError::Format(format!(
            "PRKCC001 payload has {} trailing bytes",
            payload.len() - c
        )));
    }

    Ok(Prkcc001Decoded {
        header,
        f32_fields,
        u32_fields,
        checksum: stored_checksum,
        integrity_ok: true,
    })
}

fn json_f32(v: Option<f32>) -> Value {
    match v {
        Some(x) if x.is_finite() => json!(x),
        Some(_) | None => Value::Null,
    }
}

fn json_u32(v: Option<u32>) -> Value {
    v.map(|x| json!(x)).unwrap_or(Value::Null)
}

impl Prkcc001Decoded {
    pub fn to_kcc_descriptor_json(&self) -> Value {
        let mut residues = Vec::with_capacity(self.header.record_count);
        for i in 0..self.header.record_count {
            let f32v = |name: &str| {
                self.f32_fields
                    .get(name)
                    .and_then(|values| values.get(i).copied())
            };
            let u32v = |name: &str| {
                self.u32_fields
                    .get(name)
                    .and_then(|values| values.get(i).copied())
            };
            residues.push(json!({
                "residue_id": i,
                "source_residue_index": i,
                "direction_score": json_f32(f32v("direction_score")),
                "motion_efficiency": json_f32(f32v("motion_efficiency")),
                "burst_motion": json_f32(f32v("burst_motion")),
                "causal_lag": json_f32(f32v("causal_lag")),
                "lag_corr_peak": json_f32(f32v("lag_corr_peak")),
                "local_cov": json_f32(f32v("local_cov")),
                "sum_motion": json_f32(f32v("sum_m")),
                "active_causal_steps": json_u32(u32v("active_causal")),
                "total_steps": json_u32(u32v("residue_count")),
                "net_dx": json_f32(f32v("net_dx")),
                "net_dy": json_f32(f32v("net_dy")),
                "net_dz": json_f32(f32v("net_dz")),
                "temporal_corr": json_f32(f32v("temporal_corr")),
                "phase_shift": json_f32(f32v("phase_shift")),
                "sum_m": json_f32(f32v("sum_m")),
                "active_causal": json_u32(u32v("active_causal")),
                "residue_count": json_u32(u32v("residue_count")),
            }));
        }
        json!({
            "schema_kind": "PRKCC001_KCC_DESCRIPTOR_DECODE",
            "source_codec": "PRKCC001",
            "stream_id": self.header.stream_id,
            "run_id": self.header.run_id,
            "stem": self.header.stem,
            "record_count": self.header.record_count,
            "byte_stride": self.header.byte_stride,
            "payload_size": self.header.payload_size,
            "checksum_fnv1a64": format!("{:016x}", self.checksum),
            "integrity_ok": self.integrity_ok,
            "vector_field_definition": {
                "source_codec": "PRKCC001",
                "record_order": "producer residue index order",
                "mapped_fields": {
                    "sum_m": "sum_motion",
                    "active_causal": "active_causal_steps",
                    "residue_count": "total_steps"
                },
                "unavailable_in_binary_envelope": [
                    "residue_name",
                    "ca_position",
                    "kcc_score"
                ],
                "raw_f32_fields": self.f32_fields.keys().cloned().collect::<Vec<_>>(),
                "raw_u32_fields": self.u32_fields.keys().cloned().collect::<Vec<_>>()
            },
            "residues": residues,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::Encoder;
    use crate::routing::{detect_family, route_artifact};
    use crate::stream::TokenStream;
    use crate::types::ArtifactFamily as F;

    const SHA: [u8; 32] = [0u8; 32];

    fn push_field_f32(payload: &mut Vec<u8>, name: &str, values: &[f32]) {
        payload.extend_from_slice(&(name.len() as u64).to_le_bytes());
        payload.extend_from_slice(name.as_bytes());
        payload.push(1u8);
        payload.extend_from_slice(&((values.len() * 4) as u64).to_le_bytes());
        for v in values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn push_field_u32(payload: &mut Vec<u8>, name: &str, values: &[u32]) {
        payload.extend_from_slice(&(name.len() as u64).to_le_bytes());
        payload.extend_from_slice(name.as_bytes());
        payload.push(2u8);
        payload.extend_from_slice(&((values.len() * 4) as u64).to_le_bytes());
        for v in values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
    }

    fn synthetic_prkcc001(corrupt_checksum: bool, corrupt_trailer: bool) -> Vec<u8> {
        let n = 3u64;
        let mut payload = Vec::new();
        payload.extend_from_slice(&n.to_le_bytes());
        payload.extend_from_slice(&14u64.to_le_bytes());
        push_field_f32(&mut payload, "temporal_corr", &[0.1, 0.2, 0.3]);
        push_field_f32(&mut payload, "direction_score", &[0.4, 0.5, 0.6]);
        push_field_f32(&mut payload, "motion_efficiency", &[0.7, 0.8, 0.9]);
        push_field_f32(&mut payload, "burst_motion", &[1.0, 1.1, 1.2]);
        push_field_f32(&mut payload, "phase_shift", &[0.0, 0.1, 0.2]);
        push_field_f32(&mut payload, "causal_lag", &[-1.0, 0.0, 1.0]);
        push_field_f32(&mut payload, "lag_corr_peak", &[0.2, 0.3, 0.4]);
        push_field_f32(&mut payload, "local_cov", &[0.5, 0.6, 0.7]);
        push_field_f32(&mut payload, "net_dx", &[1.0, 2.0, 3.0]);
        push_field_f32(&mut payload, "net_dy", &[4.0, 5.0, 6.0]);
        push_field_f32(&mut payload, "net_dz", &[7.0, 8.0, 9.0]);
        push_field_f32(&mut payload, "sum_m", &[10.0, 11.0, 12.0]);
        push_field_u32(&mut payload, "residue_count", &[9500, 9500, 9500]);
        push_field_u32(&mut payload, "active_causal", &[20, 21, 22]);

        let mut raw = Vec::new();
        raw.extend_from_slice(PRKCC001_MAGIC);
        raw.extend_from_slice(&1u32.to_le_bytes());
        raw.extend_from_slice(&0x0102_0304u32.to_le_bytes());
        raw.extend_from_slice(&2u32.to_le_bytes());
        raw.extend_from_slice(&5u64.to_le_bytes());
        raw.extend_from_slice(b"run-a");
        raw.extend_from_slice(&6u64.to_le_bytes());
        raw.extend_from_slice(b"stem-b");
        raw.extend_from_slice(&n.to_le_bytes());
        raw.extend_from_slice(&4u64.to_le_bytes());
        raw.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        raw.extend_from_slice(&payload);
        let mut checksum = fnv1a_64(&payload);
        if corrupt_checksum {
            checksum ^= 1;
        }
        raw.extend_from_slice(&checksum.to_le_bytes());
        raw.extend_from_slice(if corrupt_trailer {
            b"BADKCC!!"
        } else {
            PRKCC001_TRAILER
        });
        raw
    }

    #[test]
    fn synthetic_prkcc001_decodes_and_hydrates_router_shape() {
        let raw = synthetic_prkcc001(false, false);
        let decoded = decode_prkcc001(&raw).unwrap();
        assert_eq!(decoded.header.stream_id, 2);
        assert_eq!(decoded.header.run_id, "run-a");
        assert_eq!(decoded.header.stem, "stem-b");
        assert_eq!(decoded.header.record_count, 3);
        assert_eq!(decoded.f32_fields["sum_m"][2], 12.0);
        assert_eq!(decoded.u32_fields["active_causal"][1], 21);

        let json = decoded.to_kcc_descriptor_json();
        assert_eq!(detect_family(&json), Some((F::KccDescriptor, "/residues")));
        assert_eq!(json["residues"].as_array().unwrap().len(), 3);
        assert_eq!(json["residues"][0]["residue_id"], 0);
        assert_eq!(json["residues"][0]["sum_motion"], 10.0);
        assert!(json["residues"][0].get("residue_name").is_none());
    }

    #[test]
    fn synthetic_prkcc001_rejects_checksum_and_trailer_mismatch() {
        assert!(decode_prkcc001(&synthetic_prkcc001(true, false)).is_err());
        assert!(decode_prkcc001(&synthetic_prkcc001(false, true)).is_err());
    }

    #[test]
    fn synthetic_prkcc001_routes_kcc_descriptor_tokens() {
        let decoded = decode_prkcc001(&synthetic_prkcc001(false, false)).unwrap();
        let json = decoded.to_kcc_descriptor_json();
        let enc = Encoder::new();
        let mut stream = TokenStream::new();
        let (family, report) = route_artifact(&enc, &mut stream, &json, true, SHA, true);
        assert_eq!(family, Some(F::KccDescriptor));
        assert_eq!(report.records, 3);
        assert!(
            report.matched >= 13 * 3,
            "expected mapped KCC fields to route, got {}",
            report.matched
        );
        assert!(!stream.tokens.is_empty());
        assert!(
            stream.sidecar.iter().any(|entry| matches!(
                &entry.exact,
                crate::sidecar::ExactValue::Text(text) if text.starts_with("temporal_corr=")
            )),
            "raw unmapped PRKCC001 fields should be preserved through opaque capture"
        );
    }
}
