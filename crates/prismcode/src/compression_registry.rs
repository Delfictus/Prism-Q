//! Runtime compression registry and executable roundtrip validation.

use crate::codecs::{
    delta_time::DeltaTimeCodec, event_dictionary::EventDictionaryCodec, event_rle::EventRleCodec,
    nanl_packed_u2::NanlPackedU2Codec, raw_uncompressed::RawUncompressedCodec,
    section_zstd::SectionZstdCodec, tensor_chunked::TensorChunkedCodec,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

pub const COMPRESSION_ABI_MAJOR: u16 = 0;
pub const COMPRESSION_ABI_MINOR: u16 = 1;

pub const RAW_UNCOMPRESSED_CODEC_ID: u16 = 0x0000;
pub const NANL_PACKED_U2_CODEC_ID: u16 = 0x0001;
pub const SECTION_ZSTD_CODEC_ID: u16 = 0x0002;
pub const DELTA_TIME_CODEC_ID: u16 = 0x0003;
pub const EVENT_RLE_CODEC_ID: u16 = 0x0004;
pub const EVENT_DICTIONARY_CODEC_ID: u16 = 0x0005;
pub const TENSOR_CHUNKED_CODEC_ID: u16 = 0x0006;

pub const REQUIRED_CODEC_KINDS: [CompressionCodecKind; 7] = [
    CompressionCodecKind::RawUncompressed,
    CompressionCodecKind::NanlPackedU2,
    CompressionCodecKind::SectionZstd,
    CompressionCodecKind::DeltaTime,
    CompressionCodecKind::EventRle,
    CompressionCodecKind::EventDictionary,
    CompressionCodecKind::TensorChunked,
];

pub type CompressionResult<T> = Result<T, CompressionError>;

#[derive(Debug, thiserror::Error)]
pub enum CompressionError {
    #[error("PC_ERR_REGISTRY:{0}")]
    Registry(String),
    #[error("PC_ERR_FORMAT:{0}")]
    Format(String),
    #[error("PC_ERR_SECTION:{0}")]
    Section(String),
    #[error("PC_ERR_CODEC:{0}")]
    Codec(String),
    #[error("PC_ERR_HASH:{0}")]
    Hash(String),
    #[error("PC_ERR_INTERNAL:{0}")]
    Internal(String),
}

impl From<std::io::Error> for CompressionError {
    fn from(value: std::io::Error) -> Self {
        CompressionError::Codec(value.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CompressionCodecKind {
    RawUncompressed,
    NanlPackedU2,
    SectionZstd,
    DeltaTime,
    EventRle,
    EventDictionary,
    TensorChunked,
}

impl CompressionCodecKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            CompressionCodecKind::RawUncompressed => "RAW_UNCOMPRESSED",
            CompressionCodecKind::NanlPackedU2 => "NANL_PACKED_U2",
            CompressionCodecKind::SectionZstd => "SECTION_ZSTD",
            CompressionCodecKind::DeltaTime => "DELTA_TIME",
            CompressionCodecKind::EventRle => "EVENT_RLE",
            CompressionCodecKind::EventDictionary => "EVENT_DICTIONARY",
            CompressionCodecKind::TensorChunked => "TENSOR_CHUNKED",
        }
    }
}

impl fmt::Display for CompressionCodecKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CanonicalHashDomain {
    UncompressedCanonical,
    CompressedBytes,
    DualHash,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CompressionCodecRecord {
    pub codec_id: u16,
    pub codec_name: String,
    pub codec_version_major: u16,
    pub codec_version_minor: u16,
    pub codec_kind: CompressionCodecKind,
    pub lossless_required: bool,
    pub lossy_allowed: bool,
    pub section_type_allowed: u64,
    pub state_kind_allowed: u64,
    pub dtype_allowed: u64,
    pub canonical_hash_domain: CanonicalHashDomain,
    pub compressed_hash_required: bool,
    pub uncompressed_hash_required: bool,
    pub decode_required_for_execution: bool,
    pub streaming_decode_supported: bool,
    pub gpu_decode_supported: bool,
    pub deterministic_required: bool,
    pub max_chunk_bytes: u64,
    pub max_window_bytes: u64,
    pub license_id: String,
    pub implementation_hash: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HashPair {
    pub hash_compressed_bytes: [u8; 32],
    pub hash_canonical_uncompressed_bytes: [u8; 32],
}

impl HashPair {
    pub fn compressed_hex(&self) -> String {
        hash_hex(&self.hash_compressed_bytes)
    }

    pub fn canonical_uncompressed_hex(&self) -> String {
        hash_hex(&self.hash_canonical_uncompressed_bytes)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedCompressionSection {
    pub codec_id: u16,
    pub codec_kind: CompressionCodecKind,
    pub codec_version_major: u16,
    pub codec_version_minor: u16,
    pub compressed_bytes: Vec<u8>,
    pub canonical_uncompressed_len: u64,
    pub hashes: HashPair,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutableRoundTripReport {
    pub codec_id: u16,
    pub codec_name: String,
    pub codec_kind: CompressionCodecKind,
    pub codec_version_major: u16,
    pub codec_version_minor: u16,
    pub encoded_bytes: u64,
    pub canonical_uncompressed_bytes: u64,
    pub hash_compressed_bytes: String,
    pub hash_canonical_uncompressed_bytes: String,
    pub compressed_hash_verified: bool,
    pub canonical_hash_verified: bool,
    pub roundtrip_verified: bool,
    pub decode_required_for_execution: bool,
}

pub trait CompressionCodec: Send + Sync {
    fn record(&self) -> CompressionCodecRecord;
    fn encode(&self, canonical: &[u8]) -> CompressionResult<Vec<u8>>;
    fn decode(&self, encoded: &[u8]) -> CompressionResult<Vec<u8>>;
}

pub struct RuntimeCodecRegistry {
    codecs_by_id: BTreeMap<u16, Box<dyn CompressionCodec>>,
    ids_by_kind: BTreeMap<CompressionCodecKind, u16>,
}

impl Default for RuntimeCodecRegistry {
    fn default() -> Self {
        Self::with_builtin_codecs()
    }
}

impl RuntimeCodecRegistry {
    pub fn empty() -> Self {
        Self {
            codecs_by_id: BTreeMap::new(),
            ids_by_kind: BTreeMap::new(),
        }
    }

    pub fn with_builtin_codecs() -> Self {
        let mut registry = Self::empty();
        registry
            .register(Box::<RawUncompressedCodec>::default())
            .expect("RAW_UNCOMPRESSED built-in codec record is valid");
        registry
            .register(Box::<NanlPackedU2Codec>::default())
            .expect("NANL_PACKED_U2 built-in codec record is valid");
        registry
            .register(Box::<SectionZstdCodec>::default())
            .expect("SECTION_ZSTD built-in codec record is valid");
        registry
            .register(Box::<DeltaTimeCodec>::default())
            .expect("DELTA_TIME built-in codec record is valid");
        registry
            .register(Box::<EventRleCodec>::default())
            .expect("EVENT_RLE built-in codec record is valid");
        registry
            .register(Box::<EventDictionaryCodec>::default())
            .expect("EVENT_DICTIONARY built-in codec record is valid");
        registry
            .register(Box::<TensorChunkedCodec>::default())
            .expect("TENSOR_CHUNKED built-in codec record is valid");
        registry
    }

    pub fn register(&mut self, codec: Box<dyn CompressionCodec>) -> CompressionResult<()> {
        let record = codec.record();
        validate_codec_record(&record)?;
        if self.codecs_by_id.contains_key(&record.codec_id) {
            return Err(CompressionError::Registry(format!(
                "duplicate compression codec id {}",
                record.codec_id
            )));
        }
        if self.ids_by_kind.contains_key(&record.codec_kind) {
            return Err(CompressionError::Registry(format!(
                "duplicate compression codec kind {}",
                record.codec_kind
            )));
        }
        self.ids_by_kind.insert(record.codec_kind, record.codec_id);
        self.codecs_by_id.insert(record.codec_id, codec);
        Ok(())
    }

    pub fn records(&self) -> Vec<CompressionCodecRecord> {
        self.codecs_by_id
            .values()
            .map(|codec| codec.record())
            .collect()
    }

    pub fn codec_by_id(&self, codec_id: u16) -> CompressionResult<&dyn CompressionCodec> {
        self.codecs_by_id
            .get(&codec_id)
            .map(|codec| codec.as_ref())
            .ok_or_else(|| {
                CompressionError::Registry(format!(
                    "compression codec id {codec_id} not registered"
                ))
            })
    }

    pub fn codec_by_kind(
        &self,
        kind: CompressionCodecKind,
    ) -> CompressionResult<&dyn CompressionCodec> {
        let codec_id = self.ids_by_kind.get(&kind).ok_or_else(|| {
            CompressionError::Registry(format!("compression codec kind {kind} not registered"))
        })?;
        self.codec_by_id(*codec_id)
    }

    pub fn validate_required_codecs(&self) -> CompressionResult<()> {
        for kind in REQUIRED_CODEC_KINDS {
            let codec = self.codec_by_kind(kind)?;
            validate_codec_record(&codec.record())?;
        }
        Ok(())
    }

    pub fn encode_section(
        &self,
        kind: CompressionCodecKind,
        canonical: &[u8],
    ) -> CompressionResult<EncodedCompressionSection> {
        let codec = self.codec_by_kind(kind)?;
        let record = codec.record();
        let compressed_bytes = codec.encode(canonical)?;
        let hashes = hash_pair(&compressed_bytes, canonical);
        Ok(EncodedCompressionSection {
            codec_id: record.codec_id,
            codec_kind: record.codec_kind,
            codec_version_major: record.codec_version_major,
            codec_version_minor: record.codec_version_minor,
            compressed_bytes,
            canonical_uncompressed_len: canonical.len() as u64,
            hashes,
        })
    }

    pub fn validate_roundtrip(
        &self,
        kind: CompressionCodecKind,
        canonical: &[u8],
    ) -> CompressionResult<ExecutableRoundTripReport> {
        let codec = self.codec_by_kind(kind)?;
        let record = codec.record();
        let compressed = codec.encode(canonical)?;
        let decoded = codec.decode(&compressed)?;
        if decoded != canonical {
            return Err(CompressionError::Hash(format!(
                "{} canonical uncompressed hash mismatch after decode",
                record.codec_name
            )));
        }
        let compressed_hash = hash_bytes(&compressed);
        let canonical_hash = hash_bytes(canonical);
        Ok(roundtrip_report(
            &record,
            compressed.len() as u64,
            canonical.len() as u64,
            &compressed_hash,
            &canonical_hash,
            true,
            true,
            true,
        ))
    }

    pub fn validate_encoded_by_kind(
        &self,
        kind: CompressionCodecKind,
        encoded: &[u8],
        expected_hashes: HashPair,
    ) -> CompressionResult<ExecutableRoundTripReport> {
        let codec = self.codec_by_kind(kind)?;
        self.validate_encoded(codec, encoded, expected_hashes)
    }

    pub fn validate_encoded_by_id(
        &self,
        codec_id: u16,
        encoded: &[u8],
        expected_hashes: HashPair,
    ) -> CompressionResult<ExecutableRoundTripReport> {
        let codec = self.codec_by_id(codec_id)?;
        self.validate_encoded(codec, encoded, expected_hashes)
    }

    fn validate_encoded(
        &self,
        codec: &dyn CompressionCodec,
        encoded: &[u8],
        expected_hashes: HashPair,
    ) -> CompressionResult<ExecutableRoundTripReport> {
        let record = codec.record();
        let compressed_hash = hash_bytes(encoded);
        if compressed_hash != expected_hashes.hash_compressed_bytes {
            return Err(CompressionError::Hash(format!(
                "{} compressed hash mismatch",
                record.codec_name
            )));
        }
        let decoded = codec.decode(encoded)?;
        let canonical_hash = hash_bytes(&decoded);
        if canonical_hash != expected_hashes.hash_canonical_uncompressed_bytes {
            return Err(CompressionError::Hash(format!(
                "{} canonical uncompressed hash mismatch",
                record.codec_name
            )));
        }
        Ok(roundtrip_report(
            &record,
            encoded.len() as u64,
            decoded.len() as u64,
            &compressed_hash,
            &canonical_hash,
            true,
            true,
            true,
        ))
    }
}

pub fn validate_codec_record(record: &CompressionCodecRecord) -> CompressionResult<()> {
    if record.codec_name.trim().is_empty() {
        return Err(CompressionError::Registry(
            "compression codec name is empty".to_string(),
        ));
    }
    if record.codec_version_major != COMPRESSION_ABI_MAJOR {
        return Err(CompressionError::Registry(format!(
            "{} major version {} is unsupported",
            record.codec_name, record.codec_version_major
        )));
    }
    if record.codec_version_minor > COMPRESSION_ABI_MINOR {
        return Err(CompressionError::Registry(format!(
            "{} minor version {} is newer than runtime {}",
            record.codec_name, record.codec_version_minor, COMPRESSION_ABI_MINOR
        )));
    }
    if !record.lossless_required || record.lossy_allowed {
        return Err(CompressionError::Registry(format!(
            "{} must be registered as lossless-only for executable ABI state",
            record.codec_name
        )));
    }
    if record.canonical_hash_domain != CanonicalHashDomain::DualHash {
        return Err(CompressionError::Registry(format!(
            "{} must use DUAL_HASH validation",
            record.codec_name
        )));
    }
    if !record.compressed_hash_required || !record.uncompressed_hash_required {
        return Err(CompressionError::Registry(format!(
            "{} must require compressed and canonical hashes",
            record.codec_name
        )));
    }
    if !record.deterministic_required {
        return Err(CompressionError::Registry(format!(
            "{} must require deterministic decode",
            record.codec_name
        )));
    }
    if record.max_chunk_bytes == 0 || record.max_window_bytes == 0 {
        return Err(CompressionError::Registry(format!(
            "{} has invalid chunk/window limits",
            record.codec_name
        )));
    }
    if !is_valid_hash_hex(&record.implementation_hash) {
        return Err(CompressionError::Registry(format!(
            "{} implementation_hash must be a non-zero hash256 hex value",
            record.codec_name
        )));
    }
    Ok(())
}

pub(crate) fn builtin_codec_record(
    codec_id: u16,
    codec_name: &str,
    codec_kind: CompressionCodecKind,
    max_chunk_bytes: u64,
    max_window_bytes: u64,
    license_id: &str,
    decode_required_for_execution: bool,
    streaming_decode_supported: bool,
) -> CompressionCodecRecord {
    CompressionCodecRecord {
        codec_id,
        codec_name: codec_name.to_string(),
        codec_version_major: COMPRESSION_ABI_MAJOR,
        codec_version_minor: COMPRESSION_ABI_MINOR,
        codec_kind,
        lossless_required: true,
        lossy_allowed: false,
        section_type_allowed: u64::MAX,
        state_kind_allowed: u64::MAX,
        dtype_allowed: u64::MAX,
        canonical_hash_domain: CanonicalHashDomain::DualHash,
        compressed_hash_required: true,
        uncompressed_hash_required: true,
        decode_required_for_execution,
        streaming_decode_supported,
        gpu_decode_supported: false,
        deterministic_required: true,
        max_chunk_bytes,
        max_window_bytes,
        license_id: license_id.to_string(),
        implementation_hash: implementation_hash(codec_id, codec_name, codec_kind),
    }
}

pub fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

pub fn hash_pair(compressed: &[u8], canonical: &[u8]) -> HashPair {
    HashPair {
        hash_compressed_bytes: hash_bytes(compressed),
        hash_canonical_uncompressed_bytes: hash_bytes(canonical),
    }
}

pub fn hash_hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn roundtrip_report(
    record: &CompressionCodecRecord,
    encoded_bytes: u64,
    canonical_uncompressed_bytes: u64,
    compressed_hash: &[u8; 32],
    canonical_hash: &[u8; 32],
    compressed_hash_verified: bool,
    canonical_hash_verified: bool,
    roundtrip_verified: bool,
) -> ExecutableRoundTripReport {
    ExecutableRoundTripReport {
        codec_id: record.codec_id,
        codec_name: record.codec_name.clone(),
        codec_kind: record.codec_kind,
        codec_version_major: record.codec_version_major,
        codec_version_minor: record.codec_version_minor,
        encoded_bytes,
        canonical_uncompressed_bytes,
        hash_compressed_bytes: hash_hex(compressed_hash),
        hash_canonical_uncompressed_bytes: hash_hex(canonical_hash),
        compressed_hash_verified,
        canonical_hash_verified,
        roundtrip_verified,
        decode_required_for_execution: record.decode_required_for_execution,
    }
}

fn implementation_hash(
    codec_id: u16,
    codec_name: &str,
    codec_kind: CompressionCodecKind,
) -> String {
    let manifest = format!(
        "PRISMCODE_COMPRESSION_ABI_V0_1:{codec_id}:{codec_name}:{}",
        codec_kind.as_str()
    );
    hash_hex(&hash_bytes(manifest.as_bytes()))
}

fn is_valid_hash_hex(value: &str) -> bool {
    value.len() == 64
        && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
        && value.as_bytes().iter().any(|byte| *byte != b'0')
}
