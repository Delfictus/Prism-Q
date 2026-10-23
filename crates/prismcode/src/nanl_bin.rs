//! NANL-BIN executable token codec.
//!
//! This module is intentionally binary-first: callers pass typed token records,
//! the codec emits a compact byte stream, and validation replays the canonical
//! token records from the byte stream before accepting hashes.

use serde::{Deserialize, Serialize};

pub const NANL_BIN_MAGIC: &[u8; 8] = b"NANLBIN\0";
pub const NANL_BIN_VERSION_MAJOR: u16 = 0;
pub const NANL_BIN_VERSION_MINOR: u16 = 1;
const NANL_BIN_HEADER_LEN: usize = 144;
const STREAM_COUNT: u8 = 3;

pub type NanlBinResult<T> = Result<T, NanlBinError>;

#[derive(Debug, thiserror::Error)]
pub enum NanlBinError {
    #[error("PC_ERR_FORMAT:{0}")]
    Format(String),
    #[error("PC_ERR_HASH:{0}")]
    Hash(String),
    #[error("PC_ERR_INTERNAL:{0}")]
    Internal(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum NanlTokenClass {
    Entity = 1,
    Atom = 2,
    Residue = 3,
    Chain = 4,
    Bond = 5,
    Water = 6,
    TimeDelta = 7,
    EventKind = 8,
    EventMotif = 9,
    Variant = 10,
    Thermo = 11,
    Uncertainty = 12,
    TeacherEvidence = 13,
    GFlowNetAction = 14,
    GFlowNetReward = 15,
    NrfFrame = 16,
    DrfChannel = 17,
    Provenance = 18,
    Seal = 19,
}

impl NanlTokenClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entity => "ENTITY_TOKEN",
            Self::Atom => "ATOM_TOKEN",
            Self::Residue => "RESIDUE_TOKEN",
            Self::Chain => "CHAIN_TOKEN",
            Self::Bond => "BOND_TOKEN",
            Self::Water => "WATER_TOKEN",
            Self::TimeDelta => "TIME_DELTA_TOKEN",
            Self::EventKind => "EVENT_KIND_TOKEN",
            Self::EventMotif => "EVENT_MOTIF_TOKEN",
            Self::Variant => "VARIANT_TOKEN",
            Self::Thermo => "THERMO_TOKEN",
            Self::Uncertainty => "UNCERTAINTY_TOKEN",
            Self::TeacherEvidence => "TEACHER_EVIDENCE_TOKEN",
            Self::GFlowNetAction => "GFLOWNET_ACTION_TOKEN",
            Self::GFlowNetReward => "GFLOWNET_REWARD_TOKEN",
            Self::NrfFrame => "NRF_FRAME_TOKEN",
            Self::DrfChannel => "DRF_CHANNEL_TOKEN",
            Self::Provenance => "PROVENANCE_TOKEN",
            Self::Seal => "SEAL_TOKEN",
        }
    }
}

impl TryFrom<u8> for NanlTokenClass {
    type Error = NanlBinError;

    fn try_from(value: u8) -> NanlBinResult<Self> {
        match value {
            1 => Ok(Self::Entity),
            2 => Ok(Self::Atom),
            3 => Ok(Self::Residue),
            4 => Ok(Self::Chain),
            5 => Ok(Self::Bond),
            6 => Ok(Self::Water),
            7 => Ok(Self::TimeDelta),
            8 => Ok(Self::EventKind),
            9 => Ok(Self::EventMotif),
            10 => Ok(Self::Variant),
            11 => Ok(Self::Thermo),
            12 => Ok(Self::Uncertainty),
            13 => Ok(Self::TeacherEvidence),
            14 => Ok(Self::GFlowNetAction),
            15 => Ok(Self::GFlowNetReward),
            16 => Ok(Self::NrfFrame),
            17 => Ok(Self::DrfChannel),
            18 => Ok(Self::Provenance),
            19 => Ok(Self::Seal),
            other => Err(NanlBinError::Format(format!(
                "unknown NANL-BIN token class {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlToken {
    pub class: NanlTokenClass,
    pub dictionary_id: u16,
    pub value: u32,
}

impl NanlToken {
    pub const fn new(class: NanlTokenClass, dictionary_id: u16, value: u32) -> Self {
        Self {
            class,
            dictionary_id,
            value,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlBinIndexEntry {
    pub token_index: u64,
    pub class: NanlTokenClass,
    pub dictionary_id: u16,
    pub token_span_start: u64,
    pub token_span_len: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanlBinReport {
    pub status: String,
    pub mode: String,
    pub version: String,
    pub token_count: u64,
    pub token_width_bits: u8,
    pub stream_count: u8,
    pub canonical_payload_hash_blake3: String,
    pub nanl_bin_stream_hash_blake3: String,
    pub merkle_root_blake3: String,
    pub encoded_bytes: u64,
    pub canonical_bytes: u64,
    pub compression_ratio_vs_canonical: f64,
    pub roundtrip_canonical_hash_pass: bool,
    pub debug_json_hot_path_used: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NanlBinPacket {
    bytes: Vec<u8>,
    tokens: Vec<NanlToken>,
    report: NanlBinReport,
}

impl NanlBinPacket {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn tokens(&self) -> &[NanlToken] {
        &self.tokens
    }

    pub fn report(&self) -> &NanlBinReport {
        &self.report
    }
}

pub fn encode_tokens(tokens: &[NanlToken]) -> NanlBinResult<NanlBinPacket> {
    if tokens.is_empty() {
        return Err(NanlBinError::Format(
            "NANL-BIN token stream is empty".to_string(),
        ));
    }

    let token_width_bits = token_width_bits(tokens);
    let canonical = canonical_token_bytes(tokens);
    let canonical_hash = hash_bytes(&canonical);

    let class_stream = tokens
        .iter()
        .map(|token| token.class as u8)
        .collect::<Vec<_>>();
    let mut dictionary_stream = Vec::with_capacity(tokens.len() * 2);
    let mut values = Vec::with_capacity(tokens.len());
    for token in tokens {
        dictionary_stream.extend_from_slice(&token.dictionary_id.to_le_bytes());
        values.push(token.value);
    }
    let value_stream = pack_fixed_width(&values, token_width_bits)?;

    let mut stream_bytes =
        Vec::with_capacity(class_stream.len() + dictionary_stream.len() + value_stream.len());
    stream_bytes.extend_from_slice(&class_stream);
    stream_bytes.extend_from_slice(&dictionary_stream);
    stream_bytes.extend_from_slice(&value_stream);
    let stream_hash = hash_bytes(&stream_bytes);
    let merkle_root = merkle_root(&canonical_hash, &stream_hash);

    let mut bytes = vec![0u8; NANL_BIN_HEADER_LEN];
    bytes[0..8].copy_from_slice(NANL_BIN_MAGIC);
    bytes[8..10].copy_from_slice(&NANL_BIN_VERSION_MAJOR.to_le_bytes());
    bytes[10..12].copy_from_slice(&NANL_BIN_VERSION_MINOR.to_le_bytes());
    bytes[12..20].copy_from_slice(&(tokens.len() as u64).to_le_bytes());
    bytes[20] = token_width_bits;
    bytes[21] = STREAM_COUNT;
    bytes[24..32].copy_from_slice(&(class_stream.len() as u64).to_le_bytes());
    bytes[32..40].copy_from_slice(&(dictionary_stream.len() as u64).to_le_bytes());
    bytes[40..48].copy_from_slice(&(value_stream.len() as u64).to_le_bytes());
    bytes[48..80].copy_from_slice(&canonical_hash);
    bytes[80..112].copy_from_slice(&stream_hash);
    bytes[112..144].copy_from_slice(&merkle_root);
    bytes.extend_from_slice(&class_stream);
    bytes.extend_from_slice(&dictionary_stream);
    bytes.extend_from_slice(&value_stream);

    let report = report_from_parts(
        tokens.len(),
        token_width_bits,
        bytes.len(),
        canonical.len(),
        &canonical_hash,
        &stream_hash,
        &merkle_root,
        true,
    );

    Ok(NanlBinPacket {
        bytes,
        tokens: tokens.to_vec(),
        report,
    })
}

pub fn decode_packet(bytes: &[u8]) -> NanlBinResult<Vec<NanlToken>> {
    decode_packet_with_report(bytes).map(|decoded| decoded.tokens)
}

#[derive(Clone, Debug, PartialEq)]
pub struct NanlBinDecoded {
    pub tokens: Vec<NanlToken>,
    pub report: NanlBinReport,
}

pub fn decode_packet_with_report(bytes: &[u8]) -> NanlBinResult<NanlBinDecoded> {
    if bytes.len() < NANL_BIN_HEADER_LEN || &bytes[0..8] != NANL_BIN_MAGIC {
        return Err(NanlBinError::Format(
            "invalid NANL-BIN header magic".to_string(),
        ));
    }
    let version_major = read_u16(bytes, 8)?;
    let version_minor = read_u16(bytes, 10)?;
    if version_major != NANL_BIN_VERSION_MAJOR || version_minor != NANL_BIN_VERSION_MINOR {
        return Err(NanlBinError::Format(format!(
            "unsupported NANL-BIN version {version_major}.{version_minor}"
        )));
    }
    let token_count = read_u64(bytes, 12)? as usize;
    let token_width_bits = bytes[20];
    if !matches!(token_width_bits, 2 | 4 | 8 | 16) {
        return Err(NanlBinError::Format(format!(
            "invalid NANL-BIN token width {token_width_bits}"
        )));
    }
    if bytes[21] != STREAM_COUNT {
        return Err(NanlBinError::Format(format!(
            "invalid NANL-BIN stream count {}",
            bytes[21]
        )));
    }

    let class_len = read_u64(bytes, 24)? as usize;
    let dictionary_len = read_u64(bytes, 32)? as usize;
    let value_len = read_u64(bytes, 40)? as usize;
    if class_len != token_count {
        return Err(NanlBinError::Format(
            "NANL-BIN class stream length does not match token count".to_string(),
        ));
    }
    if dictionary_len
        != token_count.checked_mul(2).ok_or_else(|| {
            NanlBinError::Format("NANL-BIN dictionary stream length overflow".to_string())
        })?
    {
        return Err(NanlBinError::Format(
            "NANL-BIN dictionary stream length does not match token count".to_string(),
        ));
    }
    let expected_value_len = packed_len(token_count, token_width_bits)?;
    if value_len != expected_value_len {
        return Err(NanlBinError::Format(
            "NANL-BIN value stream length does not match packed width".to_string(),
        ));
    }

    let body_len = class_len
        .checked_add(dictionary_len)
        .and_then(|value| value.checked_add(value_len))
        .ok_or_else(|| NanlBinError::Format("NANL-BIN body length overflow".to_string()))?;
    let expected_len = NANL_BIN_HEADER_LEN
        .checked_add(body_len)
        .ok_or_else(|| NanlBinError::Format("NANL-BIN encoded length overflow".to_string()))?;
    if bytes.len() != expected_len {
        return Err(NanlBinError::Format(
            "NANL-BIN encoded length mismatch".to_string(),
        ));
    }

    let canonical_hash_expected = read_hash(bytes, 48)?;
    let stream_hash_expected = read_hash(bytes, 80)?;
    let merkle_root_expected = read_hash(bytes, 112)?;

    let class_start = NANL_BIN_HEADER_LEN;
    let dictionary_start = class_start + class_len;
    let value_start = dictionary_start + dictionary_len;
    let class_stream = &bytes[class_start..dictionary_start];
    let dictionary_stream = &bytes[dictionary_start..value_start];
    let value_stream = &bytes[value_start..];
    let stream_hash = hash_bytes(&bytes[NANL_BIN_HEADER_LEN..]);
    if stream_hash != stream_hash_expected {
        return Err(NanlBinError::Hash(
            "NANL-BIN stream hash mismatch".to_string(),
        ));
    }

    let values = unpack_fixed_width(value_stream, token_count, token_width_bits)?;
    let mut tokens = Vec::with_capacity(token_count);
    for idx in 0..token_count {
        let class = NanlTokenClass::try_from(class_stream[idx])?;
        let dict_offset = idx * 2;
        let dictionary_id = u16::from_le_bytes([
            dictionary_stream[dict_offset],
            dictionary_stream[dict_offset + 1],
        ]);
        tokens.push(NanlToken::new(class, dictionary_id, values[idx]));
    }

    let canonical = canonical_token_bytes(&tokens);
    let canonical_hash = hash_bytes(&canonical);
    if canonical_hash != canonical_hash_expected {
        return Err(NanlBinError::Hash(
            "NANL-BIN canonical roundtrip hash mismatch".to_string(),
        ));
    }
    let merkle_root = merkle_root(&canonical_hash, &stream_hash);
    if merkle_root != merkle_root_expected {
        return Err(NanlBinError::Hash(
            "NANL-BIN Merkle root mismatch".to_string(),
        ));
    }

    Ok(NanlBinDecoded {
        report: report_from_parts(
            tokens.len(),
            token_width_bits,
            bytes.len(),
            canonical.len(),
            &canonical_hash,
            &stream_hash,
            &merkle_root,
            true,
        ),
        tokens,
    })
}

pub fn validate_packet(bytes: &[u8]) -> NanlBinResult<NanlBinReport> {
    decode_packet_with_report(bytes).map(|decoded| decoded.report)
}

pub fn build_index(tokens: &[NanlToken]) -> Vec<NanlBinIndexEntry> {
    tokens
        .iter()
        .enumerate()
        .map(|(idx, token)| NanlBinIndexEntry {
            token_index: idx as u64,
            class: token.class,
            dictionary_id: token.dictionary_id,
            token_span_start: idx as u64,
            token_span_len: 1,
        })
        .collect()
}

pub fn sample_tokens() -> Vec<NanlToken> {
    use NanlTokenClass as C;
    vec![
        NanlToken::new(C::Entity, 1, 0),
        NanlToken::new(C::Atom, 2, 1),
        NanlToken::new(C::Residue, 3, 2),
        NanlToken::new(C::Water, 4, 3),
        NanlToken::new(C::TimeDelta, 5, 1),
        NanlToken::new(C::EventKind, 6, 2),
        NanlToken::new(C::EventMotif, 7, 3),
        NanlToken::new(C::Thermo, 8, 0),
        NanlToken::new(C::TeacherEvidence, 9, 1),
        NanlToken::new(C::GFlowNetReward, 10, 2),
        NanlToken::new(C::NrfFrame, 11, 3),
        NanlToken::new(C::DrfChannel, 12, 0),
        NanlToken::new(C::Provenance, 13, 1),
        NanlToken::new(C::Seal, 14, 2),
    ]
}

fn token_width_bits(tokens: &[NanlToken]) -> u8 {
    let max_value = tokens.iter().map(|token| token.value).max().unwrap_or(0);
    match max_value {
        0..=3 => 2,
        4..=15 => 4,
        16..=255 => 8,
        _ => 16,
    }
}

fn canonical_token_bytes(tokens: &[NanlToken]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 8);
    for token in tokens {
        out.push(token.class as u8);
        out.push(0);
        out.extend_from_slice(&token.dictionary_id.to_le_bytes());
        out.extend_from_slice(&token.value.to_le_bytes());
    }
    out
}

fn pack_fixed_width(values: &[u32], bits: u8) -> NanlBinResult<Vec<u8>> {
    let max_allowed = if bits == 16 {
        u16::MAX as u32
    } else {
        (1u32 << bits) - 1
    };
    let mut out = vec![0u8; packed_len(values.len(), bits)?];
    let mut bit_pos = 0usize;
    for value in values {
        if *value > max_allowed {
            return Err(NanlBinError::Format(format!(
                "NANL-BIN token value {value} exceeds {bits}-bit lane"
            )));
        }
        for lane in 0..bits {
            if (value >> lane) & 1 == 1 {
                let absolute = bit_pos + lane as usize;
                out[absolute / 8] |= 1 << (absolute % 8);
            }
        }
        bit_pos += bits as usize;
    }
    Ok(out)
}

fn unpack_fixed_width(bytes: &[u8], count: usize, bits: u8) -> NanlBinResult<Vec<u32>> {
    if bytes.len() != packed_len(count, bits)? {
        return Err(NanlBinError::Format(
            "NANL-BIN packed value stream length mismatch".to_string(),
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let used_bits = count
        .checked_mul(bits as usize)
        .ok_or_else(|| NanlBinError::Format("NANL-BIN bit count overflow".to_string()))?;
    let unused_bits = bytes.len() * 8 - used_bits;
    if unused_bits > 0 {
        let final_byte = *bytes
            .last()
            .ok_or_else(|| NanlBinError::Format("missing NANL-BIN value byte".to_string()))?;
        let used_in_final = 8 - unused_bits;
        let allowed_mask = ((1u16 << used_in_final) - 1) as u8;
        if final_byte & !allowed_mask != 0 {
            return Err(NanlBinError::Format(
                "NANL-BIN unused high bits are non-zero".to_string(),
            ));
        }
    }

    let mut out = Vec::with_capacity(count);
    let mut bit_pos = 0usize;
    for _ in 0..count {
        let mut value = 0u32;
        for lane in 0..bits {
            let absolute = bit_pos + lane as usize;
            let bit = (bytes[absolute / 8] >> (absolute % 8)) & 1;
            value |= (bit as u32) << lane;
        }
        out.push(value);
        bit_pos += bits as usize;
    }
    Ok(out)
}

fn packed_len(count: usize, bits: u8) -> NanlBinResult<usize> {
    let total_bits = count
        .checked_mul(bits as usize)
        .ok_or_else(|| NanlBinError::Format("NANL-BIN packed length overflow".to_string()))?;
    Ok(total_bits.div_ceil(8))
}

fn report_from_parts(
    token_count: usize,
    token_width_bits: u8,
    encoded_len: usize,
    canonical_len: usize,
    canonical_hash: &[u8; 32],
    stream_hash: &[u8; 32],
    merkle_root: &[u8; 32],
    roundtrip_pass: bool,
) -> NanlBinReport {
    NanlBinReport {
        status: if roundtrip_pass { "PASS" } else { "FAIL" }.to_string(),
        mode: "NANL-BIN".to_string(),
        version: format!("{NANL_BIN_VERSION_MAJOR}.{NANL_BIN_VERSION_MINOR}"),
        token_count: token_count as u64,
        token_width_bits,
        stream_count: STREAM_COUNT,
        canonical_payload_hash_blake3: hash_hex(canonical_hash),
        nanl_bin_stream_hash_blake3: hash_hex(stream_hash),
        merkle_root_blake3: hash_hex(merkle_root),
        encoded_bytes: encoded_len as u64,
        canonical_bytes: canonical_len as u64,
        compression_ratio_vs_canonical: if encoded_len == 0 {
            0.0
        } else {
            canonical_len as f64 / encoded_len as f64
        },
        roundtrip_canonical_hash_pass: roundtrip_pass,
        debug_json_hot_path_used: false,
    }
}

fn merkle_root(canonical_hash: &[u8; 32], stream_hash: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"PRISMCODE:NANL_BIN:MERKLE_ROOT:V0_1");
    hasher.update(canonical_hash);
    hasher.update(stream_hash);
    *hasher.finalize().as_bytes()
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn hash_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn read_u16(bytes: &[u8], offset: usize) -> NanlBinResult<u16> {
    let end = offset + 2;
    if end > bytes.len() {
        return Err(NanlBinError::Format("truncated NANL-BIN u16".to_string()));
    }
    Ok(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> NanlBinResult<u64> {
    let end = offset + 8;
    if end > bytes.len() {
        return Err(NanlBinError::Format("truncated NANL-BIN u64".to_string()));
    }
    Ok(u64::from_le_bytes(bytes[offset..end].try_into().map_err(
        |_| NanlBinError::Internal("NANL-BIN u64 conversion failed".to_string()),
    )?))
}

fn read_hash(bytes: &[u8], offset: usize) -> NanlBinResult<[u8; 32]> {
    let end = offset + 32;
    if end > bytes.len() {
        return Err(NanlBinError::Format(
            "truncated NANL-BIN hash256".to_string(),
        ));
    }
    bytes[offset..end]
        .try_into()
        .map_err(|_| NanlBinError::Internal("NANL-BIN hash conversion failed".to_string()))
}
