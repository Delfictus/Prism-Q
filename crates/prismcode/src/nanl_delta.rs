//! Lossless NANL delta and frame-of-reference coding.
//!
//! Canonical NANL-DELTA never quantizes canonical state. Integer and f64-bit
//! helpers are retained for existing callers, and the byte-frame codec below is
//! the H3 frame-of-reference path for exact canonical byte roundtrip tests.

use serde::{Deserialize, Serialize};

pub type NanlDeltaResult<T> = Result<T, NanlDeltaError>;

#[derive(Debug, thiserror::Error)]
pub enum NanlDeltaError {
    #[error("PC_ERR_NANL_DELTA:{0}")]
    Invalid(String),
}

const MAX_DELTA_COUNT: u64 = 1_000_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum PrecisionReduction {
    None,
    QuantizedBits(u8),
    TruncatedBytes(u8),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeltaEncodingPolicy {
    pub canonical_required: bool,
    pub precision_reduction: PrecisionReduction,
}

impl Default for DeltaEncodingPolicy {
    fn default() -> Self {
        Self {
            canonical_required: true,
            precision_reduction: PrecisionReduction::None,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlDeltaPacket {
    pub magic: String,
    pub version: String,
    pub base_state_hash: String,
    pub delta_count: u64,
    pub frame_of_reference_ref: u64,
    pub delta_stream_hash: String,
    pub canonical_reconstruct_hash: String,
    pub base_value: i64,
    pub zigzag_deltas: Vec<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum FrameDeltaOp {
    EqualRun { len: u64 },
    LiteralRun { bytes: Vec<u8> },
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlFrameDeltaPacket {
    pub magic: String,
    pub version: String,
    pub base_len: u64,
    pub target_len: u64,
    pub base_hash: String,
    pub target_hash: String,
    pub delta_stream_hash: String,
    pub operation_count: u64,
    pub operations: Vec<FrameDeltaOp>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlFrameDeltaReport {
    pub status: String,
    pub mode: String,
    pub canonical_roundtrip_hash_pass: bool,
    pub lossy_quantization_used: bool,
    pub base_hash: String,
    pub target_hash: String,
    pub delta_stream_hash: String,
    pub operation_count: u64,
}

pub fn encode_i64_lossless(values: &[i64]) -> NanlDeltaResult<NanlDeltaPacket> {
    if values.is_empty() {
        return Err(NanlDeltaError::Invalid("delta source is empty".to_string()));
    }
    let base = values[0];
    let mut previous = base;
    let mut zigzag_deltas = Vec::with_capacity(values.len());
    for value in values {
        let delta = value
            .checked_sub(previous)
            .ok_or_else(|| NanlDeltaError::Invalid("delta overflow".to_string()))?;
        zigzag_deltas.push(zigzag(delta));
        previous = *value;
    }
    let canonical = canonical_i64_bytes(values);
    let stream = canonical_u64_bytes(&zigzag_deltas);
    Ok(NanlDeltaPacket {
        magic: "NANL_DELTA".to_string(),
        version: "0.1".to_string(),
        base_state_hash: hash_hex(&base.to_le_bytes()),
        delta_count: zigzag_deltas.len() as u64,
        frame_of_reference_ref: 0,
        delta_stream_hash: hash_hex(&stream),
        canonical_reconstruct_hash: hash_hex(&canonical),
        base_value: base,
        zigzag_deltas,
    })
}

pub fn decode_i64_lossless(packet: &NanlDeltaPacket) -> NanlDeltaResult<Vec<i64>> {
    validate_packet(packet)?;
    let mut values = Vec::with_capacity(packet.zigzag_deltas.len());
    let mut previous = packet.base_value;
    for (idx, encoded) in packet.zigzag_deltas.iter().enumerate() {
        let delta = unzigzag(*encoded);
        let value = if idx == 0 {
            packet.base_value
        } else {
            previous
                .checked_add(delta)
                .ok_or_else(|| NanlDeltaError::Invalid("delta decode overflow".to_string()))?
        };
        values.push(value);
        previous = value;
    }
    if hash_hex(&canonical_i64_bytes(&values)) != packet.canonical_reconstruct_hash {
        return Err(NanlDeltaError::Invalid(
            "canonical reconstruct hash mismatch".to_string(),
        ));
    }
    Ok(values)
}

pub fn encode_f64_bits_lossless(values: &[f64]) -> NanlDeltaResult<NanlDeltaPacket> {
    let bit_values = values
        .iter()
        .map(|value| value.to_bits() as i64)
        .collect::<Vec<_>>();
    encode_i64_lossless(&bit_values)
}

pub fn decode_f64_bits_lossless(packet: &NanlDeltaPacket) -> NanlDeltaResult<Vec<f64>> {
    decode_i64_lossless(packet).map(|values| {
        values
            .into_iter()
            .map(|bits| f64::from_bits(bits as u64))
            .collect()
    })
}

pub fn reject_lossy_canonical_delta(precision_reduction: bool) -> NanlDeltaResult<()> {
    if precision_reduction {
        return Err(NanlDeltaError::Invalid(
            "lossy precision reduction cannot be canonical".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_delta_policy(policy: DeltaEncodingPolicy) -> NanlDeltaResult<()> {
    if policy.canonical_required && policy.precision_reduction != PrecisionReduction::None {
        return Err(NanlDeltaError::Invalid(
            "lossy precision reduction cannot be canonical".to_string(),
        ));
    }
    Ok(())
}

pub fn encode_frame_delta_lossless(
    base: &[u8],
    target: &[u8],
) -> NanlDeltaResult<NanlFrameDeltaPacket> {
    encode_frame_delta_lossless_with_policy(base, target, DeltaEncodingPolicy::default())
}

pub fn encode_frame_delta_lossless_with_policy(
    base: &[u8],
    target: &[u8],
    policy: DeltaEncodingPolicy,
) -> NanlDeltaResult<NanlFrameDeltaPacket> {
    validate_delta_policy(policy)?;
    if target.is_empty() {
        return Err(NanlDeltaError::Invalid(
            "frame delta target is empty".to_string(),
        ));
    }
    let operations = build_frame_delta_ops(base, target)?;
    let delta_stream_hash = hash_frame_delta_ops(&operations);
    let packet = NanlFrameDeltaPacket {
        magic: "NANL_FRAME_DELTA".to_string(),
        version: "0.1".to_string(),
        base_len: base.len() as u64,
        target_len: target.len() as u64,
        base_hash: hash_hex(base),
        target_hash: hash_hex(target),
        delta_stream_hash,
        operation_count: operations.len() as u64,
        operations,
    };
    validate_frame_delta_packet(&packet, base)?;
    Ok(packet)
}

pub fn decode_frame_delta_lossless(
    packet: &NanlFrameDeltaPacket,
    base: &[u8],
) -> NanlDeltaResult<Vec<u8>> {
    validate_frame_delta_header(packet, base)?;
    let mut output = Vec::with_capacity(packet.target_len as usize);
    for operation in &packet.operations {
        match operation {
            FrameDeltaOp::EqualRun { len } => {
                if *len == 0 {
                    return Err(NanlDeltaError::Invalid(
                        "zero-length equal run is invalid".to_string(),
                    ));
                }
                let start = output.len();
                let len = usize::try_from(*len)
                    .map_err(|_| NanlDeltaError::Invalid("equal run overflow".to_string()))?;
                let end = start
                    .checked_add(len)
                    .ok_or_else(|| NanlDeltaError::Invalid("equal run overflow".to_string()))?;
                if end > base.len() {
                    return Err(NanlDeltaError::Invalid(
                        "pathological equal run reads past base frame".to_string(),
                    ));
                }
                output.extend_from_slice(&base[start..end]);
            }
            FrameDeltaOp::LiteralRun { bytes } => {
                if bytes.is_empty() {
                    return Err(NanlDeltaError::Invalid(
                        "zero-length literal run is invalid".to_string(),
                    ));
                }
                output.extend_from_slice(bytes);
            }
        }
        if output.len() as u64 > packet.target_len {
            return Err(NanlDeltaError::Invalid(
                "frame delta reconstructs beyond target length".to_string(),
            ));
        }
    }
    if output.len() as u64 != packet.target_len {
        return Err(NanlDeltaError::Invalid(
            "frame delta reconstructed length mismatch".to_string(),
        ));
    }
    if hash_hex(&output) != packet.target_hash {
        return Err(NanlDeltaError::Invalid(
            "canonical reconstruct hash mismatch".to_string(),
        ));
    }
    Ok(output)
}

pub fn validate_frame_delta_packet(
    packet: &NanlFrameDeltaPacket,
    base: &[u8],
) -> NanlDeltaResult<NanlFrameDeltaReport> {
    validate_frame_delta_header(packet, base)?;
    let decoded = decode_frame_delta_lossless(packet, base)?;
    let roundtrip = hash_hex(&decoded) == packet.target_hash;
    Ok(NanlFrameDeltaReport {
        status: if roundtrip { "PASS" } else { "FAIL" }.to_string(),
        mode: "NANL-DELTA".to_string(),
        canonical_roundtrip_hash_pass: roundtrip,
        lossy_quantization_used: false,
        base_hash: packet.base_hash.clone(),
        target_hash: packet.target_hash.clone(),
        delta_stream_hash: packet.delta_stream_hash.clone(),
        operation_count: packet.operation_count,
    })
}

pub fn validate_packet(packet: &NanlDeltaPacket) -> NanlDeltaResult<()> {
    if packet.magic != "NANL_DELTA" {
        return Err(NanlDeltaError::Invalid("bad delta magic".to_string()));
    }
    if packet.delta_count != packet.zigzag_deltas.len() as u64 {
        return Err(NanlDeltaError::Invalid("delta count mismatch".to_string()));
    }
    if packet.delta_count == 0 || packet.delta_count > MAX_DELTA_COUNT {
        return Err(NanlDeltaError::Invalid(
            "pathological delta count is outside bounded limits".to_string(),
        ));
    }
    if packet.base_state_hash != hash_hex(&packet.base_value.to_le_bytes()) {
        return Err(NanlDeltaError::Invalid(
            "base state hash mismatch".to_string(),
        ));
    }
    if packet.delta_stream_hash != hash_hex(&canonical_u64_bytes(&packet.zigzag_deltas)) {
        return Err(NanlDeltaError::Invalid(
            "delta stream hash mismatch".to_string(),
        ));
    }
    Ok(())
}

fn validate_frame_delta_header(packet: &NanlFrameDeltaPacket, base: &[u8]) -> NanlDeltaResult<()> {
    if packet.magic != "NANL_FRAME_DELTA" {
        return Err(NanlDeltaError::Invalid("bad frame delta magic".to_string()));
    }
    if packet.base_len != base.len() as u64 {
        return Err(NanlDeltaError::Invalid(
            "base frame length mismatch".to_string(),
        ));
    }
    if packet.base_hash != hash_hex(base) {
        return Err(NanlDeltaError::Invalid(
            "base frame hash mismatch".to_string(),
        ));
    }
    if packet.operation_count != packet.operations.len() as u64 {
        return Err(NanlDeltaError::Invalid(
            "frame delta operation count mismatch".to_string(),
        ));
    }
    if packet.operation_count == 0 || packet.operation_count > MAX_DELTA_COUNT {
        return Err(NanlDeltaError::Invalid(
            "pathological frame delta operation count is outside bounded limits".to_string(),
        ));
    }
    if packet.delta_stream_hash != hash_frame_delta_ops(&packet.operations) {
        return Err(NanlDeltaError::Invalid(
            "frame delta stream hash mismatch".to_string(),
        ));
    }
    Ok(())
}

fn build_frame_delta_ops(base: &[u8], target: &[u8]) -> NanlDeltaResult<Vec<FrameDeltaOp>> {
    let mut operations = Vec::new();
    let mut pos = 0usize;
    while pos < target.len() {
        if pos < base.len() && base[pos] == target[pos] {
            let start = pos;
            pos += 1;
            while pos < target.len() && pos < base.len() && base[pos] == target[pos] {
                pos += 1;
            }
            operations.push(FrameDeltaOp::EqualRun {
                len: (pos - start) as u64,
            });
        } else {
            let start = pos;
            pos += 1;
            while pos < target.len() && !(pos < base.len() && base[pos] == target[pos]) {
                pos += 1;
            }
            operations.push(FrameDeltaOp::LiteralRun {
                bytes: target[start..pos].to_vec(),
            });
        }
        if operations.len() as u64 > MAX_DELTA_COUNT {
            return Err(NanlDeltaError::Invalid(
                "pathological frame delta operation count is outside bounded limits".to_string(),
            ));
        }
    }
    Ok(operations)
}

fn hash_frame_delta_ops(operations: &[FrameDeltaOp]) -> String {
    let mut bytes = Vec::new();
    for operation in operations {
        match operation {
            FrameDeltaOp::EqualRun { len } => {
                bytes.push(0);
                bytes.extend_from_slice(&len.to_le_bytes());
            }
            FrameDeltaOp::LiteralRun { bytes: literal } => {
                bytes.push(1);
                bytes.extend_from_slice(&(literal.len() as u64).to_le_bytes());
                bytes.extend_from_slice(literal);
            }
        }
    }
    hash_hex(&bytes)
}

fn zigzag(value: i64) -> u64 {
    ((value << 1) ^ (value >> 63)) as u64
}

fn unzigzag(value: u64) -> i64 {
    ((value >> 1) as i64) ^ (-((value & 1) as i64))
}

fn canonical_i64_bytes(values: &[i64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn canonical_u64_bytes(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn hash_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}
