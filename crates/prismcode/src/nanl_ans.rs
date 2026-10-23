//! NANL-ANS deterministic lossless entropy core.
//!
//! This module implements a chunked byte-oriented rANS codec over typed PRISM
//! symbols. The binary packet stores the normalized model table, independent
//! chunk checkpoints, fixed little-endian fields, and hash seals for the model,
//! stream, and canonical roundtrip payload.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

pub const NANL_ANS_MAGIC: &[u8; 8] = b"NANLANS\0";
pub const NANL_ANS_VERSION_MAJOR: u16 = 0;
pub const NANL_ANS_VERSION_MINOR: u16 = 1;
pub const NANL_ANS_HEADER_LEN: usize = 256;
pub const NANL_ANS_MODEL_ENTRY_LEN: usize = 24;
pub const NANL_ANS_CHUNK_ENTRY_LEN: usize = 72;

const DEFAULT_SCALE_BITS: u16 = 12;
const DEFAULT_NORMALIZATION_TOTAL: u32 = 1 << DEFAULT_SCALE_BITS;
const DEFAULT_CHUNK_SYMBOL_LIMIT: usize = 1024;
const RANS_STATE_BITS: u16 = 32;
const RANS_BYTE_L: u32 = 1 << 23;
const MAX_MODEL_ENTRIES: usize = DEFAULT_NORMALIZATION_TOTAL as usize;
const MAX_DECODED_SYMBOLS: usize = 10_000_000;
const MAX_PACKET_BYTES: usize = 256 * 1024 * 1024;

pub type NanlAnsResult<T> = Result<T, NanlAnsError>;

#[derive(Debug, thiserror::Error)]
pub enum NanlAnsError {
    #[error("PC_ERR_FORMAT:{0}")]
    Format(String),
    #[error("PC_ERR_MODEL:{0}")]
    Model(String),
    #[error("PC_ERR_HASH:{0}")]
    Hash(String),
    #[error("PC_ERR_CORRUPTION:{0}")]
    Corruption(String),
    #[error("PC_ERR_BOUND:{0}")]
    Bound(String),
    #[error("PC_ERR_INTERNAL:{0}")]
    Internal(String),
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[repr(u16)]
pub enum AnsSymbolClass {
    EventKind = 1,
    ResidueMotif = 2,
    WaterGateMotif = 3,
    DrfChannel = 4,
    GFlowNetAction = 5,
    TeacherLabel = 6,
    TimeDelta = 7,
    VariantKind = 8,
    ThermoQuantity = 9,
    UncertaintyModel = 10,
}

impl AnsSymbolClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::EventKind => "EVENT_KIND",
            Self::ResidueMotif => "RESIDUE_MOTIF",
            Self::WaterGateMotif => "WATER_GATE_MOTIF",
            Self::DrfChannel => "DRF_CHANNEL",
            Self::GFlowNetAction => "GFLOWNET_ACTION",
            Self::TeacherLabel => "TEACHER_LABEL",
            Self::TimeDelta => "TIME_DELTA",
            Self::VariantKind => "VARIANT_KIND",
            Self::ThermoQuantity => "THERMO_QUANTITY",
            Self::UncertaintyModel => "UNCERTAINTY_MODEL",
        }
    }
}

impl TryFrom<u16> for AnsSymbolClass {
    type Error = NanlAnsError;

    fn try_from(value: u16) -> NanlAnsResult<Self> {
        match value {
            1 => Ok(Self::EventKind),
            2 => Ok(Self::ResidueMotif),
            3 => Ok(Self::WaterGateMotif),
            4 => Ok(Self::DrfChannel),
            5 => Ok(Self::GFlowNetAction),
            6 => Ok(Self::TeacherLabel),
            7 => Ok(Self::TimeDelta),
            8 => Ok(Self::VariantKind),
            9 => Ok(Self::ThermoQuantity),
            10 => Ok(Self::UncertaintyModel),
            other => Err(NanlAnsError::Model(format!(
                "unknown NANL-ANS symbol class {other}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct AnsToken {
    pub symbol_class: AnsSymbolClass,
    pub symbol_id: u16,
}

impl AnsToken {
    pub const fn new(symbol_class: AnsSymbolClass, symbol_id: u16) -> Self {
        Self {
            symbol_class,
            symbol_id,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AnsModelEntry {
    pub symbol_class: AnsSymbolClass,
    pub symbol_id: u16,
    pub frequency: u16,
    pub cumulative: u16,
    pub raw_count: u64,
    pub entry_hash_prefix: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlAnsModel {
    pub ans_kind: String,
    pub state_bits: u16,
    pub scale_bits: u16,
    pub normalization_total: u32,
    pub entries: Vec<AnsModelEntry>,
    pub model_hash_blake3: String,
    pub frozen: bool,
    pub endian: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlAnsChunkCheckpoint {
    pub chunk_id: u32,
    pub symbol_offset: u64,
    pub symbol_len: u32,
    pub stream_offset: u64,
    pub stream_len: u32,
    pub final_state: u32,
    pub chunk_hash_blake3: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NanlAnsReport {
    pub status: String,
    pub mode: String,
    pub ans_kind: String,
    pub version: String,
    pub state_bits: u16,
    pub scale_bits: u16,
    pub normalization_total: u32,
    pub symbol_count: u64,
    pub model_count: u32,
    pub stream_count: u32,
    pub chunk_checkpoint_count: u32,
    pub chunk_symbol_limit: u64,
    pub canonical_input_hash_blake3: String,
    pub model_hash_blake3: String,
    pub ans_stream_hash_blake3: String,
    pub canonical_roundtrip_hash_blake3: String,
    pub merkle_root_blake3: String,
    pub encoded_bytes: u64,
    pub canonical_bytes: u64,
    pub model_table_bytes: u64,
    pub bounded_memory_bytes: u64,
    pub compression_ratio_vs_canonical: f64,
    pub deterministic: bool,
    pub streaming_decode_supported: bool,
    pub byte_order: String,
    pub corruption_detection: String,
    pub canonical_roundtrip_hash_pass: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NanlAnsPacket {
    bytes: Vec<u8>,
    pub compressed: Vec<u8>,
    model: NanlAnsModel,
    checkpoints: Vec<NanlAnsChunkCheckpoint>,
    report: NanlAnsReport,
}

impl NanlAnsPacket {
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    pub fn model(&self) -> &NanlAnsModel {
        &self.model
    }

    pub fn checkpoints(&self) -> &[NanlAnsChunkCheckpoint] {
        &self.checkpoints
    }

    pub fn report(&self) -> &NanlAnsReport {
        &self.report
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NanlAnsDecoded {
    pub tokens: Vec<AnsToken>,
    pub model: NanlAnsModel,
    pub checkpoints: Vec<NanlAnsChunkCheckpoint>,
    pub report: NanlAnsReport,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NanlAnsConfig {
    pub scale_bits: u16,
    pub chunk_symbol_limit: usize,
}

impl Default for NanlAnsConfig {
    fn default() -> Self {
        Self {
            scale_bits: DEFAULT_SCALE_BITS,
            chunk_symbol_limit: DEFAULT_CHUNK_SYMBOL_LIMIT,
        }
    }
}

impl NanlAnsConfig {
    pub const fn with_chunk_symbol_limit(chunk_symbol_limit: usize) -> Self {
        Self {
            scale_bits: DEFAULT_SCALE_BITS,
            chunk_symbol_limit,
        }
    }
}

#[derive(Clone, Debug)]
struct EncodedChunk {
    symbol_offset: usize,
    symbol_len: usize,
    stream: Vec<u8>,
    final_state: u32,
}

#[derive(Clone, Debug)]
struct PacketLayout {
    symbol_count: usize,
    chunk_symbol_limit: usize,
    model_table_offset: usize,
    model_table_len: usize,
    chunk_table_offset: usize,
    chunk_table_len: usize,
    stream_offset: usize,
    stream_len: usize,
    canonical_hash: [u8; 32],
    model_hash: [u8; 32],
    stream_hash: [u8; 32],
    roundtrip_hash: [u8; 32],
    merkle_root: [u8; 32],
    normalization_total: u32,
    model_count: usize,
    chunk_count: usize,
}

pub fn build_model(tokens: &[AnsToken]) -> NanlAnsResult<NanlAnsModel> {
    build_model_with_config(tokens, NanlAnsConfig::default())
}

pub fn build_model_with_config(
    tokens: &[AnsToken],
    config: NanlAnsConfig,
) -> NanlAnsResult<NanlAnsModel> {
    validate_config(config)?;
    if tokens.is_empty() {
        return Err(NanlAnsError::Model(
            "NANL-ANS model input is empty".to_string(),
        ));
    }
    if tokens.len() > MAX_DECODED_SYMBOLS {
        return Err(NanlAnsError::Bound(format!(
            "NANL-ANS symbol count {} exceeds bounded limit {MAX_DECODED_SYMBOLS}",
            tokens.len()
        )));
    }

    let mut raw_counts: BTreeMap<AnsToken, u64> = BTreeMap::new();
    for token in tokens {
        *raw_counts.entry(*token).or_default() += 1;
    }
    if raw_counts.len() > MAX_MODEL_ENTRIES {
        return Err(NanlAnsError::Model(format!(
            "NANL-ANS unique symbol count {} exceeds normalization total {DEFAULT_NORMALIZATION_TOTAL}",
            raw_counts.len()
        )));
    }

    let total = tokens.len() as u64;
    let normalization_total = 1u32 << config.scale_bits;
    let mut normalized = Vec::with_capacity(raw_counts.len());
    let mut normalized_sum = 0i64;
    for (token, raw_count) in raw_counts {
        let scaled = raw_count * u64::from(normalization_total);
        let mut frequency = (scaled / total) as u16;
        if frequency == 0 {
            frequency = 1;
        }
        normalized_sum += i64::from(frequency);
        normalized.push((token, raw_count, scaled % total, frequency));
    }

    while normalized_sum < i64::from(normalization_total) {
        let idx = normalized
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.2.cmp(&right.2).then_with(|| right.0.cmp(&left.0)))
            .map(|(idx, _)| idx)
            .ok_or_else(|| NanlAnsError::Internal("missing NANL-ANS model entry".to_string()))?;
        normalized[idx].3 = normalized[idx]
            .3
            .checked_add(1)
            .ok_or_else(|| NanlAnsError::Model("NANL-ANS frequency overflow".to_string()))?;
        normalized_sum += 1;
    }

    while normalized_sum > i64::from(normalization_total) {
        let idx = normalized
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.3 > 1)
            .max_by(|(_, left), (_, right)| {
                left.3
                    .cmp(&right.3)
                    .then_with(|| right.2.cmp(&left.2))
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(idx, _)| idx)
            .ok_or_else(|| {
                NanlAnsError::Model(
                    "NANL-ANS cannot reduce normalized frequencies without zeroing a symbol"
                        .to_string(),
                )
            })?;
        normalized[idx].3 -= 1;
        normalized_sum -= 1;
    }

    let mut cumulative = 0u32;
    let mut entries = Vec::with_capacity(normalized.len());
    for (token, raw_count, _remainder, frequency) in normalized {
        let entry_hash_prefix = entry_hash_prefix(token, frequency, cumulative as u16, raw_count);
        entries.push(AnsModelEntry {
            symbol_class: token.symbol_class,
            symbol_id: token.symbol_id,
            frequency,
            cumulative: cumulative as u16,
            raw_count,
            entry_hash_prefix,
        });
        cumulative += u32::from(frequency);
    }
    if cumulative != normalization_total {
        return Err(NanlAnsError::Model(
            "NANL-ANS normalized model total is not exact".to_string(),
        ));
    }

    let model_table = serialize_model_table(&entries)?;
    let model_hash = hash_bytes(&model_table);
    Ok(NanlAnsModel {
        ans_kind: "rANS".to_string(),
        state_bits: RANS_STATE_BITS,
        scale_bits: config.scale_bits,
        normalization_total,
        entries,
        model_hash_blake3: hash_hex(&model_hash),
        frozen: true,
        endian: "little".to_string(),
    })
}

pub fn encode_tokens(tokens: &[AnsToken]) -> NanlAnsResult<NanlAnsPacket> {
    encode_tokens_with_config(tokens, NanlAnsConfig::default())
}

pub fn encode_tokens_with_config(
    tokens: &[AnsToken],
    config: NanlAnsConfig,
) -> NanlAnsResult<NanlAnsPacket> {
    validate_config(config)?;
    let model = build_model_with_config(tokens, config)?;
    let lookup = build_encode_lookup(&model)?;
    let canonical = canonical_token_bytes(tokens);
    let canonical_hash = hash_bytes(&canonical);

    let mut encoded_chunks = Vec::new();
    for (chunk_idx, chunk) in tokens.chunks(config.chunk_symbol_limit).enumerate() {
        let symbol_offset = chunk_idx
            .checked_mul(config.chunk_symbol_limit)
            .ok_or_else(|| NanlAnsError::Bound("NANL-ANS chunk offset overflow".to_string()))?;
        encoded_chunks.push(encode_chunk(chunk, symbol_offset, &lookup, &model)?);
    }

    let model_table = serialize_model_table(&model.entries)?;
    let model_hash = hash_bytes(&model_table);
    if hash_hex(&model_hash) != model.model_hash_blake3 {
        return Err(NanlAnsError::Internal(
            "NANL-ANS model hash changed during encode".to_string(),
        ));
    }

    let model_table_offset = NANL_ANS_HEADER_LEN;
    let model_table_len = model_table.len();
    let chunk_table_offset = model_table_offset
        .checked_add(model_table_len)
        .ok_or_else(|| NanlAnsError::Bound("NANL-ANS model table offset overflow".to_string()))?;
    let chunk_table_len = encoded_chunks
        .len()
        .checked_mul(NANL_ANS_CHUNK_ENTRY_LEN)
        .ok_or_else(|| NanlAnsError::Bound("NANL-ANS chunk table length overflow".to_string()))?;
    let stream_offset = chunk_table_offset
        .checked_add(chunk_table_len)
        .ok_or_else(|| NanlAnsError::Bound("NANL-ANS stream offset overflow".to_string()))?;

    let mut stream = Vec::new();
    let mut checkpoints = Vec::with_capacity(encoded_chunks.len());
    for (chunk_id, chunk) in encoded_chunks.iter().enumerate() {
        let offset = stream_offset + stream.len();
        let chunk_hash = hash_bytes(&chunk.stream);
        checkpoints.push(NanlAnsChunkCheckpoint {
            chunk_id: chunk_id as u32,
            symbol_offset: chunk.symbol_offset as u64,
            symbol_len: chunk.symbol_len as u32,
            stream_offset: offset as u64,
            stream_len: chunk.stream.len() as u32,
            final_state: chunk.final_state,
            chunk_hash_blake3: hash_hex(&chunk_hash),
        });
        stream.extend_from_slice(&chunk.stream);
    }
    let stream_hash = hash_bytes(&stream);
    let merkle = merkle_root(&canonical_hash, &model_hash, &stream_hash);
    let packet_len = stream_offset
        .checked_add(stream.len())
        .ok_or_else(|| NanlAnsError::Bound("NANL-ANS packet length overflow".to_string()))?;
    if packet_len > MAX_PACKET_BYTES {
        return Err(NanlAnsError::Bound(format!(
            "NANL-ANS packet length {packet_len} exceeds bounded limit {MAX_PACKET_BYTES}"
        )));
    }

    let mut bytes = vec![0u8; NANL_ANS_HEADER_LEN];
    bytes[0..8].copy_from_slice(NANL_ANS_MAGIC);
    write_u16(&mut bytes, 8, NANL_ANS_VERSION_MAJOR)?;
    write_u16(&mut bytes, 10, NANL_ANS_VERSION_MINOR)?;
    write_u16(&mut bytes, 12, 1)?;
    write_u16(&mut bytes, 14, config.scale_bits)?;
    write_u16(&mut bytes, 16, RANS_STATE_BITS)?;
    write_u16(&mut bytes, 18, 0)?;
    write_u32(&mut bytes, 20, model.normalization_total)?;
    write_u32(&mut bytes, 24, model.entries.len() as u32)?;
    write_u32(&mut bytes, 28, checkpoints.len() as u32)?;
    write_u64(&mut bytes, 32, tokens.len() as u64)?;
    write_u64(&mut bytes, 40, config.chunk_symbol_limit as u64)?;
    write_u64(&mut bytes, 48, model_table_offset as u64)?;
    write_u64(&mut bytes, 56, model_table_len as u64)?;
    write_u64(&mut bytes, 64, chunk_table_offset as u64)?;
    write_u64(&mut bytes, 72, chunk_table_len as u64)?;
    write_u64(&mut bytes, 80, stream_offset as u64)?;
    write_u64(&mut bytes, 88, stream.len() as u64)?;
    bytes[96..128].copy_from_slice(&canonical_hash);
    bytes[128..160].copy_from_slice(&model_hash);
    bytes[160..192].copy_from_slice(&stream_hash);
    bytes[192..224].copy_from_slice(&canonical_hash);
    bytes[224..256].copy_from_slice(&merkle);
    bytes.extend_from_slice(&model_table);
    bytes.extend_from_slice(&serialize_chunk_table(&checkpoints)?);
    bytes.extend_from_slice(&stream);

    let report = report_from_parts(
        tokens.len(),
        bytes.len(),
        canonical.len(),
        model_table.len(),
        config.chunk_symbol_limit,
        &model,
        checkpoints.len(),
        &canonical_hash,
        &model_hash,
        &stream_hash,
        &canonical_hash,
        &merkle,
        true,
    );

    Ok(NanlAnsPacket {
        bytes,
        compressed: stream,
        model,
        checkpoints,
        report,
    })
}

pub fn decode_packet(bytes: &[u8]) -> NanlAnsResult<Vec<AnsToken>> {
    decode_packet_with_report(bytes).map(|decoded| decoded.tokens)
}

pub fn decode_packet_with_report(bytes: &[u8]) -> NanlAnsResult<NanlAnsDecoded> {
    let mut tokens = Vec::new();
    let report = decode_packet_streaming(bytes, |chunk| {
        tokens.extend_from_slice(chunk);
        Ok(())
    })?;
    let (layout, model, checkpoints) = parse_packet(bytes)?;
    if tokens.len() != layout.symbol_count {
        return Err(NanlAnsError::Corruption(
            "NANL-ANS decoded symbol count mismatch".to_string(),
        ));
    }
    Ok(NanlAnsDecoded {
        tokens,
        model,
        checkpoints,
        report,
    })
}

pub fn decode_packet_streaming<F>(bytes: &[u8], mut sink: F) -> NanlAnsResult<NanlAnsReport>
where
    F: FnMut(&[AnsToken]) -> NanlAnsResult<()>,
{
    let (layout, model, checkpoints) = parse_packet(bytes)?;
    let slot_table = build_decode_slot_table(&model)?;
    let mut roundtrip_hasher = blake3::Hasher::new();
    let mut decoded_total = 0usize;
    for checkpoint in &checkpoints {
        let stream_start = checkpoint.stream_offset as usize;
        let stream_end = stream_start
            .checked_add(checkpoint.stream_len as usize)
            .ok_or_else(|| NanlAnsError::Bound("NANL-ANS chunk stream overflow".to_string()))?;
        if stream_end > bytes.len() {
            return Err(NanlAnsError::Format(
                "NANL-ANS chunk stream points outside packet".to_string(),
            ));
        }
        let chunk_stream = &bytes[stream_start..stream_end];
        let chunk_hash = hash_bytes(chunk_stream);
        if hash_hex(&chunk_hash) != checkpoint.chunk_hash_blake3 {
            return Err(NanlAnsError::Hash(
                "NANL-ANS chunk hash mismatch".to_string(),
            ));
        }
        let decoded = decode_chunk(
            chunk_stream,
            checkpoint.symbol_len as usize,
            &model,
            &slot_table,
        )?;
        decoded_total = decoded_total
            .checked_add(decoded.len())
            .ok_or_else(|| NanlAnsError::Bound("NANL-ANS decoded count overflow".to_string()))?;
        let canonical = canonical_token_bytes(&decoded);
        roundtrip_hasher.update(&canonical);
        sink(&decoded)?;
    }
    if decoded_total != layout.symbol_count {
        return Err(NanlAnsError::Corruption(
            "NANL-ANS decoded total does not match header symbol count".to_string(),
        ));
    }

    let roundtrip_hash = *roundtrip_hasher.finalize().as_bytes();
    if roundtrip_hash != layout.canonical_hash || roundtrip_hash != layout.roundtrip_hash {
        return Err(NanlAnsError::Hash(
            "NANL-ANS canonical roundtrip hash mismatch".to_string(),
        ));
    }

    Ok(report_from_parts(
        layout.symbol_count,
        bytes.len(),
        layout.symbol_count * 4,
        layout.model_table_len,
        layout.chunk_symbol_limit,
        &model,
        checkpoints.len(),
        &layout.canonical_hash,
        &layout.model_hash,
        &layout.stream_hash,
        &roundtrip_hash,
        &layout.merkle_root,
        true,
    ))
}

pub fn validate_packet(bytes: &[u8]) -> NanlAnsResult<NanlAnsReport> {
    decode_packet_with_report(bytes).map(|decoded| decoded.report)
}

pub fn sample_tokens() -> Vec<AnsToken> {
    use AnsSymbolClass as C;
    let mut tokens = Vec::new();
    for frame in 0..96u16 {
        tokens.push(AnsToken::new(C::EventKind, frame % 5));
        tokens.push(AnsToken::new(C::WaterGateMotif, frame % 3));
        tokens.push(AnsToken::new(C::ResidueMotif, frame % 7));
        tokens.push(AnsToken::new(C::TimeDelta, frame % 4));
        if frame % 2 == 0 {
            tokens.push(AnsToken::new(C::DrfChannel, 2));
        }
        if frame % 8 == 0 {
            tokens.push(AnsToken::new(C::TeacherLabel, 1));
            tokens.push(AnsToken::new(C::GFlowNetAction, frame % 6));
        }
        if frame % 16 == 0 {
            tokens.push(AnsToken::new(C::VariantKind, 3));
            tokens.push(AnsToken::new(C::ThermoQuantity, 4));
            tokens.push(AnsToken::new(C::UncertaintyModel, 1));
        }
    }
    tokens
}

fn validate_config(config: NanlAnsConfig) -> NanlAnsResult<()> {
    if config.scale_bits != DEFAULT_SCALE_BITS {
        return Err(NanlAnsError::Model(format!(
            "NANL-ANS scale_bits {} unsupported in v0.1; expected {DEFAULT_SCALE_BITS}",
            config.scale_bits
        )));
    }
    if config.chunk_symbol_limit == 0 || config.chunk_symbol_limit > 65_535 {
        return Err(NanlAnsError::Bound(
            "NANL-ANS chunk_symbol_limit must be 1..=65535".to_string(),
        ));
    }
    Ok(())
}

fn encode_chunk(
    tokens: &[AnsToken],
    symbol_offset: usize,
    lookup: &HashMap<AnsToken, AnsModelEntry>,
    model: &NanlAnsModel,
) -> NanlAnsResult<EncodedChunk> {
    let mut state = RANS_BYTE_L;
    let mut renorm = Vec::new();
    for token in tokens.iter().rev() {
        let entry = lookup.get(token).ok_or_else(|| {
            NanlAnsError::Model(format!(
                "NANL-ANS token class={} id={} missing from model",
                token.symbol_class.as_str(),
                token.symbol_id
            ))
        })?;
        let frequency = u32::from(entry.frequency);
        let cumulative = u32::from(entry.cumulative);
        let x_max = ((RANS_BYTE_L >> model.scale_bits) << 8)
            .checked_mul(frequency)
            .ok_or_else(|| NanlAnsError::Bound("NANL-ANS x_max overflow".to_string()))?;
        while state >= x_max {
            renorm.push((state & 0xff) as u8);
            state >>= 8;
        }
        let next = ((u64::from(state / frequency)) << model.scale_bits)
            + u64::from(state % frequency)
            + u64::from(cumulative);
        if next > u64::from(u32::MAX) {
            return Err(NanlAnsError::Bound(
                "NANL-ANS encoder state overflow".to_string(),
            ));
        }
        state = next as u32;
    }

    renorm.reverse();
    let mut stream = Vec::with_capacity(4 + renorm.len());
    stream.extend_from_slice(&state.to_le_bytes());
    stream.extend_from_slice(&renorm);
    Ok(EncodedChunk {
        symbol_offset,
        symbol_len: tokens.len(),
        stream,
        final_state: state,
    })
}

fn decode_chunk(
    stream: &[u8],
    symbol_len: usize,
    model: &NanlAnsModel,
    slot_table: &[usize],
) -> NanlAnsResult<Vec<AnsToken>> {
    if stream.len() < 4 {
        return Err(NanlAnsError::Format(
            "NANL-ANS chunk stream is truncated before rANS state".to_string(),
        ));
    }
    let mut state = u32::from_le_bytes(
        stream[0..4]
            .try_into()
            .map_err(|_| NanlAnsError::Internal("NANL-ANS state conversion failed".to_string()))?,
    );
    if state < RANS_BYTE_L {
        return Err(NanlAnsError::Corruption(
            "NANL-ANS chunk final state is below renormalization floor".to_string(),
        ));
    }
    let mask = model.normalization_total - 1;
    let mut cursor = 4usize;
    let mut out = Vec::with_capacity(symbol_len);
    for _ in 0..symbol_len {
        let slot = (state & mask) as usize;
        let entry_idx = *slot_table.get(slot).ok_or_else(|| {
            NanlAnsError::Corruption("NANL-ANS decoder slot outside model".to_string())
        })?;
        let entry = model.entries.get(entry_idx).ok_or_else(|| {
            NanlAnsError::Corruption("NANL-ANS decoder entry outside model".to_string())
        })?;
        out.push(AnsToken::new(entry.symbol_class, entry.symbol_id));
        let slot_delta = slot
            .checked_sub(entry.cumulative as usize)
            .ok_or_else(|| NanlAnsError::Corruption("NANL-ANS slot underflow".to_string()))?;
        state = u32::from(entry.frequency)
            .checked_mul(state >> model.scale_bits)
            .and_then(|value| value.checked_add(slot_delta as u32))
            .ok_or_else(|| {
                NanlAnsError::Corruption("NANL-ANS decoder state overflow".to_string())
            })?;
        while state < RANS_BYTE_L {
            if cursor >= stream.len() {
                return Err(NanlAnsError::Corruption(
                    "NANL-ANS stream ended during renormalization".to_string(),
                ));
            }
            state = (state << 8) | u32::from(stream[cursor]);
            cursor += 1;
        }
    }
    if cursor != stream.len() {
        return Err(NanlAnsError::Corruption(
            "NANL-ANS chunk has trailing renormalization bytes".to_string(),
        ));
    }
    if state != RANS_BYTE_L {
        return Err(NanlAnsError::Corruption(
            "NANL-ANS chunk did not return to canonical initial state".to_string(),
        ));
    }
    Ok(out)
}

fn parse_packet(
    bytes: &[u8],
) -> NanlAnsResult<(PacketLayout, NanlAnsModel, Vec<NanlAnsChunkCheckpoint>)> {
    let layout = parse_layout(bytes)?;
    let model_table =
        &bytes[layout.model_table_offset..layout.model_table_offset + layout.model_table_len];
    let model_hash = hash_bytes(model_table);
    if model_hash != layout.model_hash {
        return Err(NanlAnsError::Hash(
            "NANL-ANS model table hash mismatch".to_string(),
        ));
    }
    let entries = parse_model_table(model_table, layout.model_count)?;
    validate_model_entries(&entries, layout.normalization_total)?;
    let model = NanlAnsModel {
        ans_kind: "rANS".to_string(),
        state_bits: RANS_STATE_BITS,
        scale_bits: DEFAULT_SCALE_BITS,
        normalization_total: layout.normalization_total,
        entries,
        model_hash_blake3: hash_hex(&model_hash),
        frozen: true,
        endian: "little".to_string(),
    };
    let chunk_table =
        &bytes[layout.chunk_table_offset..layout.chunk_table_offset + layout.chunk_table_len];
    let checkpoints = parse_chunk_table(chunk_table, layout.chunk_count)?;
    validate_checkpoints(&checkpoints, &layout, bytes)?;
    let stream = &bytes[layout.stream_offset..layout.stream_offset + layout.stream_len];
    let stream_hash = hash_bytes(stream);
    if stream_hash != layout.stream_hash {
        return Err(NanlAnsError::Hash(
            "NANL-ANS stream hash mismatch".to_string(),
        ));
    }
    let merkle = merkle_root(
        &layout.canonical_hash,
        &layout.model_hash,
        &layout.stream_hash,
    );
    if merkle != layout.merkle_root {
        return Err(NanlAnsError::Hash(
            "NANL-ANS Merkle root mismatch".to_string(),
        ));
    }
    Ok((layout, model, checkpoints))
}

fn parse_layout(bytes: &[u8]) -> NanlAnsResult<PacketLayout> {
    if bytes.len() < NANL_ANS_HEADER_LEN || &bytes[0..8] != NANL_ANS_MAGIC {
        return Err(NanlAnsError::Format(
            "invalid NANL-ANS header magic".to_string(),
        ));
    }
    if bytes.len() > MAX_PACKET_BYTES {
        return Err(NanlAnsError::Bound(format!(
            "NANL-ANS packet length {} exceeds bounded limit {MAX_PACKET_BYTES}",
            bytes.len()
        )));
    }
    let major = read_u16(bytes, 8)?;
    let minor = read_u16(bytes, 10)?;
    if major != NANL_ANS_VERSION_MAJOR || minor != NANL_ANS_VERSION_MINOR {
        return Err(NanlAnsError::Format(format!(
            "unsupported NANL-ANS version {major}.{minor}"
        )));
    }
    let ans_kind = read_u16(bytes, 12)?;
    if ans_kind != 1 {
        return Err(NanlAnsError::Format(format!(
            "unsupported NANL-ANS kind {ans_kind}"
        )));
    }
    let scale_bits = read_u16(bytes, 14)?;
    if scale_bits != DEFAULT_SCALE_BITS {
        return Err(NanlAnsError::Model(format!(
            "unsupported NANL-ANS scale bits {scale_bits}"
        )));
    }
    let state_bits = read_u16(bytes, 16)?;
    if state_bits != RANS_STATE_BITS {
        return Err(NanlAnsError::Model(format!(
            "unsupported NANL-ANS state bits {state_bits}"
        )));
    }
    let normalization_total = read_u32(bytes, 20)?;
    if normalization_total != DEFAULT_NORMALIZATION_TOTAL {
        return Err(NanlAnsError::Model(format!(
            "unsupported NANL-ANS normalization total {normalization_total}"
        )));
    }
    let model_count = read_u32(bytes, 24)? as usize;
    let chunk_count = read_u32(bytes, 28)? as usize;
    let symbol_count = read_u64(bytes, 32)? as usize;
    let chunk_symbol_limit = read_u64(bytes, 40)? as usize;
    if symbol_count == 0 || symbol_count > MAX_DECODED_SYMBOLS {
        return Err(NanlAnsError::Bound(format!(
            "NANL-ANS symbol count {symbol_count} is outside bounded range"
        )));
    }
    if model_count == 0 || model_count > MAX_MODEL_ENTRIES {
        return Err(NanlAnsError::Model(format!(
            "NANL-ANS model count {model_count} is outside bounded range"
        )));
    }
    if chunk_count == 0 {
        return Err(NanlAnsError::Format(
            "NANL-ANS packet has no chunk checkpoints".to_string(),
        ));
    }
    if chunk_symbol_limit == 0 || chunk_symbol_limit > 65_535 {
        return Err(NanlAnsError::Bound(
            "NANL-ANS chunk symbol limit is outside bounded range".to_string(),
        ));
    }

    let model_table_offset = read_u64(bytes, 48)? as usize;
    let model_table_len = read_u64(bytes, 56)? as usize;
    let chunk_table_offset = read_u64(bytes, 64)? as usize;
    let chunk_table_len = read_u64(bytes, 72)? as usize;
    let stream_offset = read_u64(bytes, 80)? as usize;
    let stream_len = read_u64(bytes, 88)? as usize;
    if model_table_offset != NANL_ANS_HEADER_LEN {
        return Err(NanlAnsError::Format(
            "NANL-ANS model table offset is not canonical".to_string(),
        ));
    }
    if model_table_len != model_count * NANL_ANS_MODEL_ENTRY_LEN {
        return Err(NanlAnsError::Format(
            "NANL-ANS model table length mismatch".to_string(),
        ));
    }
    if chunk_table_offset != model_table_offset + model_table_len {
        return Err(NanlAnsError::Format(
            "NANL-ANS chunk table offset is not canonical".to_string(),
        ));
    }
    if chunk_table_len != chunk_count * NANL_ANS_CHUNK_ENTRY_LEN {
        return Err(NanlAnsError::Format(
            "NANL-ANS chunk table length mismatch".to_string(),
        ));
    }
    if stream_offset != chunk_table_offset + chunk_table_len {
        return Err(NanlAnsError::Format(
            "NANL-ANS stream offset is not canonical".to_string(),
        ));
    }
    let expected_len = stream_offset
        .checked_add(stream_len)
        .ok_or_else(|| NanlAnsError::Bound("NANL-ANS packet length overflow".to_string()))?;
    if bytes.len() != expected_len {
        return Err(NanlAnsError::Format(
            "NANL-ANS packet length mismatch".to_string(),
        ));
    }
    Ok(PacketLayout {
        symbol_count,
        chunk_symbol_limit,
        model_table_offset,
        model_table_len,
        chunk_table_offset,
        chunk_table_len,
        stream_offset,
        stream_len,
        canonical_hash: read_hash(bytes, 96)?,
        model_hash: read_hash(bytes, 128)?,
        stream_hash: read_hash(bytes, 160)?,
        roundtrip_hash: read_hash(bytes, 192)?,
        merkle_root: read_hash(bytes, 224)?,
        normalization_total,
        model_count,
        chunk_count,
    })
}

fn parse_model_table(bytes: &[u8], entry_count: usize) -> NanlAnsResult<Vec<AnsModelEntry>> {
    if bytes.len() != entry_count * NANL_ANS_MODEL_ENTRY_LEN {
        return Err(NanlAnsError::Format(
            "NANL-ANS model table length mismatch".to_string(),
        ));
    }
    let mut entries = Vec::with_capacity(entry_count);
    for idx in 0..entry_count {
        let offset = idx * NANL_ANS_MODEL_ENTRY_LEN;
        let symbol_id = read_u16(bytes, offset)?;
        let class = AnsSymbolClass::try_from(read_u16(bytes, offset + 2)?)?;
        let frequency = read_u16(bytes, offset + 4)?;
        let cumulative = read_u16(bytes, offset + 6)?;
        let raw_count = read_u64(bytes, offset + 8)?;
        let stored_entry_hash_prefix = read_u64(bytes, offset + 16)?;
        let expected = entry_hash_prefix(
            AnsToken::new(class, symbol_id),
            frequency,
            cumulative,
            raw_count,
        );
        if stored_entry_hash_prefix != expected {
            return Err(NanlAnsError::Hash(
                "NANL-ANS model entry hash prefix mismatch".to_string(),
            ));
        }
        entries.push(AnsModelEntry {
            symbol_class: class,
            symbol_id,
            frequency,
            cumulative,
            raw_count,
            entry_hash_prefix: stored_entry_hash_prefix,
        });
    }
    Ok(entries)
}

fn validate_model_entries(
    entries: &[AnsModelEntry],
    normalization_total: u32,
) -> NanlAnsResult<()> {
    let mut previous = None;
    let mut cumulative = 0u32;
    for entry in entries {
        if entry.frequency == 0 {
            return Err(NanlAnsError::Model(
                "NANL-ANS model entry has zero frequency".to_string(),
            ));
        }
        if entry.raw_count == 0 {
            return Err(NanlAnsError::Model(
                "NANL-ANS model entry has zero raw count".to_string(),
            ));
        }
        let token = AnsToken::new(entry.symbol_class, entry.symbol_id);
        if previous.is_some_and(|prev| token <= prev) {
            return Err(NanlAnsError::Model(
                "NANL-ANS model entries are not strictly sorted".to_string(),
            ));
        }
        if u32::from(entry.cumulative) != cumulative {
            return Err(NanlAnsError::Model(
                "NANL-ANS model cumulative table is not contiguous".to_string(),
            ));
        }
        cumulative += u32::from(entry.frequency);
        previous = Some(token);
    }
    if cumulative != normalization_total {
        return Err(NanlAnsError::Model(
            "NANL-ANS model frequencies do not sum to normalization total".to_string(),
        ));
    }
    Ok(())
}

fn parse_chunk_table(
    bytes: &[u8],
    chunk_count: usize,
) -> NanlAnsResult<Vec<NanlAnsChunkCheckpoint>> {
    let mut checkpoints = Vec::with_capacity(chunk_count);
    for idx in 0..chunk_count {
        let offset = idx * NANL_ANS_CHUNK_ENTRY_LEN;
        let chunk_id = read_u32(bytes, offset)?;
        let symbol_offset = read_u64(bytes, offset + 8)?;
        let symbol_len = read_u32(bytes, offset + 16)?;
        let stream_offset = read_u64(bytes, offset + 24)?;
        let stream_len = read_u32(bytes, offset + 32)?;
        let final_state = read_u32(bytes, offset + 36)?;
        let chunk_hash = read_hash(bytes, offset + 40)?;
        checkpoints.push(NanlAnsChunkCheckpoint {
            chunk_id,
            symbol_offset,
            symbol_len,
            stream_offset,
            stream_len,
            final_state,
            chunk_hash_blake3: hash_hex(&chunk_hash),
        });
    }
    Ok(checkpoints)
}

fn validate_checkpoints(
    checkpoints: &[NanlAnsChunkCheckpoint],
    layout: &PacketLayout,
    bytes: &[u8],
) -> NanlAnsResult<()> {
    let mut expected_symbol_offset = 0usize;
    let mut expected_stream_offset = layout.stream_offset;
    for (idx, checkpoint) in checkpoints.iter().enumerate() {
        if checkpoint.chunk_id != idx as u32 {
            return Err(NanlAnsError::Format(
                "NANL-ANS chunk ids are not contiguous".to_string(),
            ));
        }
        if checkpoint.symbol_offset as usize != expected_symbol_offset {
            return Err(NanlAnsError::Format(
                "NANL-ANS chunk symbol offsets are not contiguous".to_string(),
            ));
        }
        if checkpoint.symbol_len == 0 || checkpoint.symbol_len as usize > layout.chunk_symbol_limit
        {
            return Err(NanlAnsError::Bound(
                "NANL-ANS chunk symbol length is outside bounded range".to_string(),
            ));
        }
        if checkpoint.stream_offset as usize != expected_stream_offset {
            return Err(NanlAnsError::Format(
                "NANL-ANS chunk stream offsets are not contiguous".to_string(),
            ));
        }
        if checkpoint.stream_len < 4 {
            return Err(NanlAnsError::Format(
                "NANL-ANS chunk stream is shorter than checkpoint state".to_string(),
            ));
        }
        let stream_end = checkpoint
            .stream_offset
            .checked_add(u64::from(checkpoint.stream_len))
            .ok_or_else(|| NanlAnsError::Bound("NANL-ANS checkpoint stream overflow".to_string()))?
            as usize;
        if stream_end > bytes.len() {
            return Err(NanlAnsError::Format(
                "NANL-ANS chunk stream points past packet".to_string(),
            ));
        }
        let state = u32::from_le_bytes(
            bytes[checkpoint.stream_offset as usize..checkpoint.stream_offset as usize + 4]
                .try_into()
                .map_err(|_| {
                    NanlAnsError::Internal(
                        "NANL-ANS checkpoint state conversion failed".to_string(),
                    )
                })?,
        );
        if state != checkpoint.final_state {
            return Err(NanlAnsError::Hash(
                "NANL-ANS checkpoint final state mismatch".to_string(),
            ));
        }
        expected_symbol_offset += checkpoint.symbol_len as usize;
        expected_stream_offset = stream_end;
    }
    if expected_symbol_offset != layout.symbol_count {
        return Err(NanlAnsError::Format(
            "NANL-ANS checkpoints do not cover the decoded symbol range".to_string(),
        ));
    }
    if expected_stream_offset != layout.stream_offset + layout.stream_len {
        return Err(NanlAnsError::Format(
            "NANL-ANS checkpoints do not cover the stream range".to_string(),
        ));
    }
    Ok(())
}

fn serialize_model_table(entries: &[AnsModelEntry]) -> NanlAnsResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(entries.len() * NANL_ANS_MODEL_ENTRY_LEN);
    for entry in entries {
        bytes.extend_from_slice(&entry.symbol_id.to_le_bytes());
        bytes.extend_from_slice(&(entry.symbol_class as u16).to_le_bytes());
        bytes.extend_from_slice(&entry.frequency.to_le_bytes());
        bytes.extend_from_slice(&entry.cumulative.to_le_bytes());
        bytes.extend_from_slice(&entry.raw_count.to_le_bytes());
        bytes.extend_from_slice(&entry.entry_hash_prefix.to_le_bytes());
    }
    Ok(bytes)
}

fn serialize_chunk_table(checkpoints: &[NanlAnsChunkCheckpoint]) -> NanlAnsResult<Vec<u8>> {
    let mut bytes = Vec::with_capacity(checkpoints.len() * NANL_ANS_CHUNK_ENTRY_LEN);
    for checkpoint in checkpoints {
        bytes.extend_from_slice(&checkpoint.chunk_id.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&checkpoint.symbol_offset.to_le_bytes());
        bytes.extend_from_slice(&checkpoint.symbol_len.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&checkpoint.stream_offset.to_le_bytes());
        bytes.extend_from_slice(&checkpoint.stream_len.to_le_bytes());
        bytes.extend_from_slice(&checkpoint.final_state.to_le_bytes());
        let chunk_hash = hash_from_hex(&checkpoint.chunk_hash_blake3)?;
        bytes.extend_from_slice(&chunk_hash);
    }
    Ok(bytes)
}

fn build_encode_lookup(model: &NanlAnsModel) -> NanlAnsResult<HashMap<AnsToken, AnsModelEntry>> {
    let mut lookup = HashMap::with_capacity(model.entries.len());
    for entry in &model.entries {
        let token = AnsToken::new(entry.symbol_class, entry.symbol_id);
        if lookup.insert(token, entry.clone()).is_some() {
            return Err(NanlAnsError::Model(
                "NANL-ANS duplicate model entry".to_string(),
            ));
        }
    }
    Ok(lookup)
}

fn build_decode_slot_table(model: &NanlAnsModel) -> NanlAnsResult<Vec<usize>> {
    let mut slots = vec![usize::MAX; model.normalization_total as usize];
    for (idx, entry) in model.entries.iter().enumerate() {
        let start = entry.cumulative as usize;
        let end = start
            .checked_add(entry.frequency as usize)
            .ok_or_else(|| NanlAnsError::Model("NANL-ANS slot range overflow".to_string()))?;
        if end > slots.len() {
            return Err(NanlAnsError::Model(
                "NANL-ANS model slot range exceeds normalization table".to_string(),
            ));
        }
        for slot in &mut slots[start..end] {
            if *slot != usize::MAX {
                return Err(NanlAnsError::Model(
                    "NANL-ANS overlapping model slot range".to_string(),
                ));
            }
            *slot = idx;
        }
    }
    if slots.iter().any(|slot| *slot == usize::MAX) {
        return Err(NanlAnsError::Model(
            "NANL-ANS model leaves undecodable slots".to_string(),
        ));
    }
    Ok(slots)
}

fn canonical_token_bytes(tokens: &[AnsToken]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        bytes.extend_from_slice(&(token.symbol_class as u16).to_le_bytes());
        bytes.extend_from_slice(&token.symbol_id.to_le_bytes());
    }
    bytes
}

fn report_from_parts(
    symbol_count: usize,
    encoded_len: usize,
    canonical_len: usize,
    model_table_len: usize,
    chunk_symbol_limit: usize,
    model: &NanlAnsModel,
    chunk_count: usize,
    canonical_hash: &[u8; 32],
    model_hash: &[u8; 32],
    stream_hash: &[u8; 32],
    roundtrip_hash: &[u8; 32],
    merkle_root: &[u8; 32],
    roundtrip_pass: bool,
) -> NanlAnsReport {
    NanlAnsReport {
        status: if roundtrip_pass { "PASS" } else { "FAIL" }.to_string(),
        mode: "NANL-ANS".to_string(),
        ans_kind: model.ans_kind.clone(),
        version: format!("{NANL_ANS_VERSION_MAJOR}.{NANL_ANS_VERSION_MINOR}"),
        state_bits: model.state_bits,
        scale_bits: model.scale_bits,
        normalization_total: model.normalization_total,
        symbol_count: symbol_count as u64,
        model_count: model.entries.len() as u32,
        stream_count: chunk_count as u32,
        chunk_checkpoint_count: chunk_count as u32,
        chunk_symbol_limit: chunk_symbol_limit as u64,
        canonical_input_hash_blake3: hash_hex(canonical_hash),
        model_hash_blake3: hash_hex(model_hash),
        ans_stream_hash_blake3: hash_hex(stream_hash),
        canonical_roundtrip_hash_blake3: hash_hex(roundtrip_hash),
        merkle_root_blake3: hash_hex(merkle_root),
        encoded_bytes: encoded_len as u64,
        canonical_bytes: canonical_len as u64,
        model_table_bytes: model_table_len as u64,
        bounded_memory_bytes: bounded_memory_bytes(chunk_symbol_limit, model.normalization_total),
        compression_ratio_vs_canonical: if encoded_len == 0 {
            0.0
        } else {
            canonical_len as f64 / encoded_len as f64
        },
        deterministic: true,
        streaming_decode_supported: true,
        byte_order: "little".to_string(),
        corruption_detection: "BLAKE3(model,stream,chunk,roundtrip)+Merkle+final-state".to_string(),
        canonical_roundtrip_hash_pass: roundtrip_pass,
    }
}

fn bounded_memory_bytes(chunk_symbol_limit: usize, normalization_total: u32) -> u64 {
    let chunk_bytes = chunk_symbol_limit.saturating_mul(std::mem::size_of::<AnsToken>());
    let slot_bytes = (normalization_total as usize).saturating_mul(std::mem::size_of::<usize>());
    (chunk_bytes.saturating_add(slot_bytes)) as u64
}

fn entry_hash_prefix(token: AnsToken, frequency: u16, cumulative: u16, raw_count: u64) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"PRISMCODE:NANL_ANS:MODEL_ENTRY:V0_1");
    hasher.update(&(token.symbol_class as u16).to_le_bytes());
    hasher.update(&token.symbol_id.to_le_bytes());
    hasher.update(&frequency.to_le_bytes());
    hasher.update(&cumulative.to_le_bytes());
    hasher.update(&raw_count.to_le_bytes());
    let hash = hasher.finalize();
    u64::from_le_bytes(
        hash.as_bytes()[0..8]
            .try_into()
            .expect("hash prefix length"),
    )
}

fn merkle_root(
    canonical_hash: &[u8; 32],
    model_hash: &[u8; 32],
    stream_hash: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"PRISMCODE:NANL_ANS:MERKLE_ROOT:V0_1");
    hasher.update(canonical_hash);
    hasher.update(model_hash);
    hasher.update(stream_hash);
    *hasher.finalize().as_bytes()
}

fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

fn hash_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn hash_from_hex(hex: &str) -> NanlAnsResult<[u8; 32]> {
    if hex.len() != 64 {
        return Err(NanlAnsError::Format(
            "NANL-ANS hash hex length is not 64".to_string(),
        ));
    }
    let mut out = [0u8; 32];
    for idx in 0..32 {
        out[idx] = u8::from_str_radix(&hex[idx * 2..idx * 2 + 2], 16).map_err(|_| {
            NanlAnsError::Format("NANL-ANS hash hex contains non-hex digits".to_string())
        })?;
    }
    Ok(out)
}

fn read_u16(bytes: &[u8], offset: usize) -> NanlAnsResult<u16> {
    let end = offset + 2;
    if end > bytes.len() {
        return Err(NanlAnsError::Format("truncated NANL-ANS u16".to_string()));
    }
    Ok(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> NanlAnsResult<u32> {
    let end = offset + 4;
    if end > bytes.len() {
        return Err(NanlAnsError::Format("truncated NANL-ANS u32".to_string()));
    }
    Ok(u32::from_le_bytes(bytes[offset..end].try_into().map_err(
        |_| NanlAnsError::Internal("NANL-ANS u32 conversion failed".to_string()),
    )?))
}

fn read_u64(bytes: &[u8], offset: usize) -> NanlAnsResult<u64> {
    let end = offset + 8;
    if end > bytes.len() {
        return Err(NanlAnsError::Format("truncated NANL-ANS u64".to_string()));
    }
    Ok(u64::from_le_bytes(bytes[offset..end].try_into().map_err(
        |_| NanlAnsError::Internal("NANL-ANS u64 conversion failed".to_string()),
    )?))
}

fn read_hash(bytes: &[u8], offset: usize) -> NanlAnsResult<[u8; 32]> {
    let end = offset + 32;
    if end > bytes.len() {
        return Err(NanlAnsError::Format(
            "truncated NANL-ANS hash256".to_string(),
        ));
    }
    bytes[offset..end]
        .try_into()
        .map_err(|_| NanlAnsError::Internal("NANL-ANS hash conversion failed".to_string()))
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) -> NanlAnsResult<()> {
    let end = offset + 2;
    if end > bytes.len() {
        return Err(NanlAnsError::Internal(
            "NANL-ANS write u16 overflow".to_string(),
        ));
    }
    bytes[offset..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) -> NanlAnsResult<()> {
    let end = offset + 4;
    if end > bytes.len() {
        return Err(NanlAnsError::Internal(
            "NANL-ANS write u32 overflow".to_string(),
        ));
    }
    bytes[offset..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) -> NanlAnsResult<()> {
    let end = offset + 8;
    if end > bytes.len() {
        return Err(NanlAnsError::Internal(
            "NANL-ANS write u64 overflow".to_string(),
        ));
    }
    bytes[offset..end].copy_from_slice(&value.to_le_bytes());
    Ok(())
}
