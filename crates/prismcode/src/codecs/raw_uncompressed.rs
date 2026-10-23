use crate::codecs::DEFAULT_MAX_CHUNK_BYTES;
use crate::compression_registry::{
    builtin_codec_record, CompressionCodec, CompressionCodecKind, CompressionCodecRecord,
    CompressionResult, RAW_UNCOMPRESSED_CODEC_ID,
};

#[derive(Default)]
pub struct RawUncompressedCodec;

impl CompressionCodec for RawUncompressedCodec {
    fn record(&self) -> CompressionCodecRecord {
        builtin_codec_record(
            RAW_UNCOMPRESSED_CODEC_ID,
            "RAW_UNCOMPRESSED",
            CompressionCodecKind::RawUncompressed,
            DEFAULT_MAX_CHUNK_BYTES,
            DEFAULT_MAX_CHUNK_BYTES,
            "PRISM-INTERNAL",
            false,
            true,
        )
    }

    fn encode(&self, canonical: &[u8]) -> CompressionResult<Vec<u8>> {
        Ok(canonical.to_vec())
    }

    fn decode(&self, encoded: &[u8]) -> CompressionResult<Vec<u8>> {
        Ok(encoded.to_vec())
    }
}
