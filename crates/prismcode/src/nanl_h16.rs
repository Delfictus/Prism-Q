//! Internal NANL-H16 nibble codec.
//!
//! H16 is a 16-symbol digital alphabet carrying 4 raw bits per symbol. It is
//! intentionally not Q4, not quaternary, and not DNA-like.

use serde::{Deserialize, Serialize};

pub type NanlH16Result<T> = Result<T, NanlH16Error>;

#[derive(Debug, thiserror::Error)]
pub enum NanlH16Error {
    #[error("PC_ERR_NANL_H16:{0}")]
    Invalid(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct H16ClaimDescriptor {
    pub alphabet_size: u8,
    pub bits_per_symbol: u8,
    pub is_quaternary: bool,
    pub dna_like: bool,
    pub described_as: String,
}

impl Default for H16ClaimDescriptor {
    fn default() -> Self {
        Self {
            alphabet_size: 16,
            bits_per_symbol: 4,
            is_quaternary: false,
            dna_like: false,
            described_as: "NANL-H16 internal digital nibble codec".to_string(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlH16Packet {
    pub magic: String,
    pub version: String,
    pub symbol_count: u64,
    pub stream_hash: String,
    pub canonical_roundtrip_hash: String,
    pub packed_nibbles: Vec<u8>,
}

pub fn encode_bytes(bytes: &[u8]) -> NanlH16Result<NanlH16Packet> {
    if bytes.is_empty() {
        return Err(NanlH16Error::Invalid("H16 source is empty".to_string()));
    }
    let packed_nibbles = bytes.to_vec();
    Ok(NanlH16Packet {
        magic: "NANL_H16".to_string(),
        version: "0.1".to_string(),
        symbol_count: (bytes.len() * 2) as u64,
        stream_hash: hash_hex(&packed_nibbles),
        canonical_roundtrip_hash: hash_hex(bytes),
        packed_nibbles,
    })
}

pub fn h16_hex_string(packet: &NanlH16Packet) -> NanlH16Result<String> {
    validate_packet(packet)?;
    Ok(packet
        .packed_nibbles
        .iter()
        .flat_map(|byte| [byte >> 4, byte & 0x0f])
        .map(|nibble| match nibble {
            0..=9 => (b'0' + nibble) as char,
            value => (b'A' + (value - 10)) as char,
        })
        .collect())
}

pub fn decode_bytes(packet: &NanlH16Packet) -> NanlH16Result<Vec<u8>> {
    validate_packet(packet)?;
    let bytes = packet.packed_nibbles.clone();
    if hash_hex(&bytes) != packet.canonical_roundtrip_hash {
        return Err(NanlH16Error::Invalid(
            "H16 canonical roundtrip hash mismatch".to_string(),
        ));
    }
    Ok(bytes)
}

pub fn validate_packet(packet: &NanlH16Packet) -> NanlH16Result<()> {
    if packet.magic != "NANL_H16" {
        return Err(NanlH16Error::Invalid("bad H16 magic".to_string()));
    }
    if packet.symbol_count != (packet.packed_nibbles.len() * 2) as u64 {
        return Err(NanlH16Error::Invalid(
            "H16 symbol count mismatch".to_string(),
        ));
    }
    if packet.stream_hash != hash_hex(&packet.packed_nibbles) {
        return Err(NanlH16Error::Invalid(
            "H16 stream hash mismatch".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_h16_claim(is_quaternary: bool, dna_like: bool) -> NanlH16Result<()> {
    if is_quaternary {
        return Err(NanlH16Error::Invalid(
            "H16 is a 16-symbol alphabet, not quaternary".to_string(),
        ));
    }
    if dna_like {
        return Err(NanlH16Error::Invalid(
            "H16 is internal digital nibble coding, not DNA-like".to_string(),
        ));
    }
    Ok(())
}

pub fn validate_h16_descriptor(descriptor: &H16ClaimDescriptor) -> NanlH16Result<()> {
    if descriptor.alphabet_size != 16 || descriptor.bits_per_symbol != 4 {
        return Err(NanlH16Error::Invalid(
            "H16 must declare a 16-symbol alphabet with 4 raw bits per symbol".to_string(),
        ));
    }
    validate_h16_claim(descriptor.is_quaternary, descriptor.dna_like)?;
    let lower = descriptor.described_as.to_ascii_lowercase();
    if lower.contains("q4") || lower.contains("quaternary") || lower.contains("dna") {
        return Err(NanlH16Error::Invalid(
            "H16 must not be described as Q4, quaternary, or DNA-like".to_string(),
        ));
    }
    Ok(())
}

fn hash_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}
