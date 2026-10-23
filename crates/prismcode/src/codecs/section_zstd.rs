use crate::codecs::DEFAULT_MAX_CHUNK_BYTES;
use crate::compression_registry::{
    builtin_codec_record, CompressionCodec, CompressionCodecKind, CompressionCodecRecord,
    CompressionError, CompressionResult, SECTION_ZSTD_CODEC_ID,
};
use std::io::Cursor;

#[derive(Default)]
pub struct SectionZstdCodec;

impl CompressionCodec for SectionZstdCodec {
    fn record(&self) -> CompressionCodecRecord {
        builtin_codec_record(
            SECTION_ZSTD_CODEC_ID,
            "SECTION_ZSTD",
            CompressionCodecKind::SectionZstd,
            DEFAULT_MAX_CHUNK_BYTES,
            DEFAULT_MAX_CHUNK_BYTES,
            "BSD-3-Clause",
            true,
            true,
        )
    }

    fn encode(&self, canonical: &[u8]) -> CompressionResult<Vec<u8>> {
        zstd::stream::encode_all(Cursor::new(canonical), 3)
            .map_err(|err| CompressionError::Codec(format!("SECTION_ZSTD encode failed: {err}")))
    }

    fn decode(&self, encoded: &[u8]) -> CompressionResult<Vec<u8>> {
        zstd::stream::decode_all(Cursor::new(encoded))
            .map_err(|err| CompressionError::Codec(format!("SECTION_ZSTD decode failed: {err}")))
    }
}
