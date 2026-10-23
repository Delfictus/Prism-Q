use crate::codecs::{read_u64_le, DEFAULT_MAX_CHUNK_BYTES};
use crate::compression_registry::{
    builtin_codec_record, CompressionCodec, CompressionCodecKind, CompressionCodecRecord,
    CompressionError, CompressionResult, NANL_PACKED_U2_CODEC_ID,
};

const MAGIC: &[u8; 4] = b"NPU2";

#[derive(Default)]
pub struct NanlPackedU2Codec;

impl CompressionCodec for NanlPackedU2Codec {
    fn record(&self) -> CompressionCodecRecord {
        builtin_codec_record(
            NANL_PACKED_U2_CODEC_ID,
            "NANL_PACKED_U2",
            CompressionCodecKind::NanlPackedU2,
            DEFAULT_MAX_CHUNK_BYTES,
            DEFAULT_MAX_CHUNK_BYTES,
            "PRISM-INTERNAL",
            true,
            true,
        )
    }

    fn encode(&self, canonical: &[u8]) -> CompressionResult<Vec<u8>> {
        let mut out = Vec::with_capacity(12 + canonical.len().div_ceil(4));
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(canonical.len() as u64).to_le_bytes());

        let mut byte = 0u8;
        for (idx, token) in canonical.iter().enumerate() {
            if *token > 3 {
                return Err(CompressionError::Format(format!(
                    "NANL_PACKED_U2 token out of range: {token}"
                )));
            }
            byte |= token << ((idx % 4) * 2);
            if idx % 4 == 3 {
                out.push(byte);
                byte = 0;
            }
        }
        if canonical.len() % 4 != 0 {
            out.push(byte);
        }
        Ok(out)
    }

    fn decode(&self, encoded: &[u8]) -> CompressionResult<Vec<u8>> {
        if encoded.len() < 12 || &encoded[0..4] != MAGIC {
            return Err(CompressionError::Format(
                "invalid NANL_PACKED_U2 header".to_string(),
            ));
        }
        let token_count_u64 = read_u64_le(encoded, 4)?;
        let token_count = usize::try_from(token_count_u64).map_err(|_| {
            CompressionError::Format("NANL_PACKED_U2 token count too large".to_string())
        })?;
        let packed_len = token_count.checked_add(3).ok_or_else(|| {
            CompressionError::Format("NANL_PACKED_U2 token count overflow".to_string())
        })? / 4;
        let expected_len = 12usize.checked_add(packed_len).ok_or_else(|| {
            CompressionError::Format("NANL_PACKED_U2 encoded length overflow".to_string())
        })?;
        if encoded.len() != expected_len {
            return Err(CompressionError::Format(
                "NANL_PACKED_U2 encoded length mismatch".to_string(),
            ));
        }

        if token_count % 4 != 0 && packed_len > 0 {
            let used_bits = (token_count % 4) * 2;
            let allowed_mask = ((1u16 << used_bits) - 1) as u8;
            let final_byte = encoded[encoded.len() - 1];
            if final_byte & !allowed_mask != 0 {
                return Err(CompressionError::Format(
                    "NANL_PACKED_U2 unused high bits are non-zero".to_string(),
                ));
            }
        }

        let mut out = Vec::with_capacity(token_count);
        for byte in &encoded[12..] {
            for lane in 0..4 {
                if out.len() == token_count {
                    break;
                }
                out.push((byte >> (lane * 2)) & 0b11);
            }
        }
        Ok(out)
    }
}
