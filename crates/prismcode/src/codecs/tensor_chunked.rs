use crate::codecs::{read_hash, read_u32_le, read_u64_le, DEFAULT_MAX_CHUNK_BYTES};
use crate::compression_registry::{
    builtin_codec_record, hash_bytes, CompressionCodec, CompressionCodecKind,
    CompressionCodecRecord, CompressionError, CompressionResult, TENSOR_CHUNKED_CODEC_ID,
};

const MAGIC: &[u8; 4] = b"TCHK";
const DEFAULT_TENSOR_CHUNK_BYTES: usize = 64;

#[derive(Default)]
pub struct TensorChunkedCodec;

impl TensorChunkedCodec {
    pub fn encode_with_chunk_size(
        canonical: &[u8],
        chunk_bytes: usize,
    ) -> CompressionResult<Vec<u8>> {
        if chunk_bytes == 0 {
            return Err(CompressionError::Format(
                "TENSOR_CHUNKED chunk size must be non-zero".to_string(),
            ));
        }
        let chunk_bytes_u32 = u32::try_from(chunk_bytes).map_err(|_| {
            CompressionError::Format("TENSOR_CHUNKED chunk size exceeds u32".to_string())
        })?;
        let mut out =
            Vec::with_capacity(canonical.len() + 16 + canonical.len().div_ceil(chunk_bytes) * 40);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(canonical.len() as u64).to_le_bytes());
        out.extend_from_slice(&chunk_bytes_u32.to_le_bytes());
        for (idx, chunk) in canonical.chunks(chunk_bytes).enumerate() {
            let idx_u32 = u32::try_from(idx).map_err(|_| {
                CompressionError::Format("TENSOR_CHUNKED index exceeds u32".to_string())
            })?;
            out.extend_from_slice(&idx_u32.to_le_bytes());
            out.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
            out.extend_from_slice(&hash_bytes(chunk));
            out.extend_from_slice(chunk);
        }
        Ok(out)
    }
}

impl CompressionCodec for TensorChunkedCodec {
    fn record(&self) -> CompressionCodecRecord {
        builtin_codec_record(
            TENSOR_CHUNKED_CODEC_ID,
            "TENSOR_CHUNKED",
            CompressionCodecKind::TensorChunked,
            DEFAULT_MAX_CHUNK_BYTES,
            DEFAULT_MAX_CHUNK_BYTES,
            "PRISM-INTERNAL",
            true,
            true,
        )
    }

    fn encode(&self, canonical: &[u8]) -> CompressionResult<Vec<u8>> {
        Self::encode_with_chunk_size(canonical, DEFAULT_TENSOR_CHUNK_BYTES)
    }

    fn decode(&self, encoded: &[u8]) -> CompressionResult<Vec<u8>> {
        if encoded.len() < 16 || &encoded[0..4] != MAGIC {
            return Err(CompressionError::Format(
                "invalid TENSOR_CHUNKED header".to_string(),
            ));
        }
        let expected_len = usize::try_from(read_u64_le(encoded, 4)?).map_err(|_| {
            CompressionError::Format("TENSOR_CHUNKED decoded length too large".to_string())
        })?;
        let chunk_bytes = read_u32_le(encoded, 12)? as usize;
        if chunk_bytes == 0 {
            return Err(CompressionError::Format(
                "TENSOR_CHUNKED chunk size must be non-zero".to_string(),
            ));
        }

        let mut out = Vec::with_capacity(expected_len);
        let mut cursor = 16usize;
        let mut expected_idx = 0u32;
        while cursor < encoded.len() {
            if cursor + 40 > encoded.len() {
                return Err(CompressionError::Format(
                    "truncated TENSOR_CHUNKED chunk header".to_string(),
                ));
            }
            let idx = read_u32_le(encoded, cursor)?;
            cursor += 4;
            if idx != expected_idx {
                return Err(CompressionError::Format(
                    "TENSOR_CHUNKED index discontinuity".to_string(),
                ));
            }
            let len = read_u32_le(encoded, cursor)? as usize;
            cursor += 4;
            if len == 0 {
                return Err(CompressionError::Format(
                    "TENSOR_CHUNKED zero-length chunk".to_string(),
                ));
            }
            if len > chunk_bytes {
                return Err(CompressionError::Format(
                    "TENSOR_CHUNKED chunk exceeds declared chunk size".to_string(),
                ));
            }
            let expected_hash = read_hash(encoded, cursor)?;
            cursor += 32;
            if cursor + len > encoded.len() {
                return Err(CompressionError::Format(
                    "truncated TENSOR_CHUNKED payload".to_string(),
                ));
            }
            if out.len() + len > expected_len {
                return Err(CompressionError::Format(
                    "TENSOR_CHUNKED decoded length exceeds header".to_string(),
                ));
            }
            let chunk = &encoded[cursor..cursor + len];
            if hash_bytes(chunk) != expected_hash {
                return Err(CompressionError::Hash(
                    "TENSOR_CHUNKED chunk hash mismatch".to_string(),
                ));
            }
            out.extend_from_slice(chunk);
            cursor += len;
            expected_idx = expected_idx.checked_add(1).ok_or_else(|| {
                CompressionError::Format("TENSOR_CHUNKED index overflow".to_string())
            })?;
            if out.len() < expected_len && len != chunk_bytes {
                return Err(CompressionError::Format(
                    "TENSOR_CHUNKED non-final chunk is shorter than declared chunk size"
                        .to_string(),
                ));
            }
        }
        if out.len() != expected_len {
            return Err(CompressionError::Format(
                "TENSOR_CHUNKED decoded length mismatch".to_string(),
            ));
        }
        Ok(out)
    }
}
