//! NANL variable-width integer stream coding.
//!
//! The runtime format uses bounded unsigned LEB128 for exact integer roundtrip.
//! H3 also exposes a validator-visible code table type so malformed
//! variable-width tables fail closed before they can be promoted.

use serde::{Deserialize, Serialize};

pub type NanlVarResult<T> = Result<T, NanlVarError>;

#[derive(Debug, thiserror::Error)]
pub enum NanlVarError {
    #[error("PC_ERR_NANL_VAR:{0}")]
    Invalid(String),
}

const MAX_VARINT_BYTES: usize = 10;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlVarPacket {
    pub magic: String,
    pub version: String,
    pub code_table_hash: String,
    pub min_bits: u8,
    pub max_bits: u8,
    pub escape_code: u8,
    pub encoded_stream_hash: String,
    pub canonical_roundtrip_hash: String,
    pub symbol_count: u64,
    pub encoded: Vec<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VarCode {
    pub symbol_id: u64,
    pub bit_width: u8,
    pub code: u64,
    pub frequency: u64,
    pub escape_required: bool,
}

pub fn encode_u64(values: &[u64]) -> NanlVarResult<NanlVarPacket> {
    if values.is_empty() {
        return Err(NanlVarError::Invalid("VAR source is empty".to_string()));
    }
    let mut encoded = Vec::new();
    for value in values {
        encode_leb128(*value, &mut encoded);
    }
    let bit_widths = values
        .iter()
        .map(|value| (64 - value.leading_zeros()).max(1) as u8)
        .collect::<Vec<_>>();
    Ok(NanlVarPacket {
        magic: "NANL_VAR".to_string(),
        version: "0.1".to_string(),
        code_table_hash: hash_hex(&bit_widths),
        min_bits: *bit_widths.iter().min().unwrap_or(&1),
        max_bits: *bit_widths.iter().max().unwrap_or(&1),
        escape_code: 0xff,
        encoded_stream_hash: hash_hex(&encoded),
        canonical_roundtrip_hash: hash_hex(&canonical_u64_bytes(values)),
        symbol_count: values.len() as u64,
        encoded,
    })
}

pub fn decode_u64(packet: &NanlVarPacket) -> NanlVarResult<Vec<u64>> {
    validate_packet(packet)?;
    let mut cursor = 0;
    let mut values = Vec::with_capacity(packet.symbol_count as usize);
    while cursor < packet.encoded.len() {
        values.push(decode_leb128(&packet.encoded, &mut cursor)?);
    }
    if values.len() as u64 != packet.symbol_count {
        return Err(NanlVarError::Invalid("symbol count mismatch".to_string()));
    }
    if hash_hex(&canonical_u64_bytes(&values)) != packet.canonical_roundtrip_hash {
        return Err(NanlVarError::Invalid(
            "canonical VAR roundtrip hash mismatch".to_string(),
        ));
    }
    Ok(values)
}

pub fn validate_packet(packet: &NanlVarPacket) -> NanlVarResult<()> {
    if packet.magic != "NANL_VAR" {
        return Err(NanlVarError::Invalid("bad VAR magic".to_string()));
    }
    if packet.encoded_stream_hash != hash_hex(&packet.encoded) {
        return Err(NanlVarError::Invalid(
            "VAR stream hash mismatch".to_string(),
        ));
    }
    if packet.min_bits == 0 || packet.max_bits == 0 || packet.min_bits > packet.max_bits {
        return Err(NanlVarError::Invalid(
            "invalid VAR table bit widths".to_string(),
        ));
    }
    if packet.max_bits > 64 {
        return Err(NanlVarError::Invalid(
            "invalid VAR table max width".to_string(),
        ));
    }
    let expected_table_hash = hash_hex(&[packet.min_bits, packet.max_bits, packet.escape_code]);
    if packet.code_table_hash.is_empty() || packet.code_table_hash == expected_table_hash {
        return Err(NanlVarError::Invalid(
            "VAR code table hash is not bound to observed symbol widths".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_code_table(table: &[VarCode]) -> NanlVarResult<()> {
    if table.is_empty() {
        return Err(NanlVarError::Invalid("VAR code table is empty".to_string()));
    }
    let mut sorted = table.to_vec();
    sorted.sort_by_key(|code| (code.symbol_id, code.bit_width, code.code));
    for window in sorted.windows(2) {
        if window[0].symbol_id == window[1].symbol_id {
            return Err(NanlVarError::Invalid(format!(
                "duplicate VAR symbol {}",
                window[0].symbol_id
            )));
        }
    }
    for code in table {
        if code.bit_width == 0 || code.bit_width > 63 {
            return Err(NanlVarError::Invalid(format!(
                "invalid VAR code width {}",
                code.bit_width
            )));
        }
        if code.frequency == 0 {
            return Err(NanlVarError::Invalid(
                "VAR code table frequency must be nonzero".to_string(),
            ));
        }
        let max_code = 1u64 << code.bit_width;
        if code.code >= max_code {
            return Err(NanlVarError::Invalid(
                "VAR code does not fit declared bit width".to_string(),
            ));
        }
    }

    let mut prefixes = table.to_vec();
    prefixes.sort_by_key(|code| (code.bit_width, code.code, code.symbol_id));
    for (left_index, left) in prefixes.iter().enumerate() {
        for right in prefixes.iter().skip(left_index + 1) {
            let common_width = left.bit_width.min(right.bit_width);
            let left_prefix = left.code >> u32::from(left.bit_width - common_width);
            let right_prefix = right.code >> u32::from(right.bit_width - common_width);
            if left_prefix == right_prefix {
                return Err(NanlVarError::Invalid(
                    "VAR code table is not prefix-free".to_string(),
                ));
            }
        }
    }
    Ok(())
}

fn encode_leb128(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
}

fn decode_leb128(bytes: &[u8], cursor: &mut usize) -> NanlVarResult<u64> {
    let mut value = 0u64;
    let mut shift = 0u32;
    let start = *cursor;
    for byte_index in 0..MAX_VARINT_BYTES {
        if *cursor >= bytes.len() {
            return Err(NanlVarError::Invalid("truncated VAR integer".to_string()));
        }
        let byte = bytes[*cursor];
        *cursor += 1;
        let payload = (byte & 0x7f) as u64;
        if shift >= 64 && payload != 0 {
            return Err(NanlVarError::Invalid("VAR integer overflow".to_string()));
        }
        value |= payload << shift;
        if byte & 0x80 == 0 {
            if *cursor - start > 1 && byte == 0 {
                return Err(NanlVarError::Invalid("overlong VAR integer".to_string()));
            }
            return Ok(value);
        }
        shift += 7;
        if shift > 63 || byte_index == MAX_VARINT_BYTES - 1 {
            return Err(NanlVarError::Invalid("VAR integer overflow".to_string()));
        }
    }
    Err(NanlVarError::Invalid("VAR integer overflow".to_string()))
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
