use crate::codecs::{read_u16_le, read_u32_le, DEFAULT_MAX_CHUNK_BYTES, EVENT_RECORD_WIDTH};
use crate::compression_registry::{
    builtin_codec_record, CompressionCodec, CompressionCodecKind, CompressionCodecRecord,
    CompressionError, CompressionResult, EVENT_RLE_CODEC_ID,
};

const MAGIC: &[u8; 4] = b"ERLE";

#[derive(Default)]
pub struct EventRleCodec;

impl CompressionCodec for EventRleCodec {
    fn record(&self) -> CompressionCodecRecord {
        builtin_codec_record(
            EVENT_RLE_CODEC_ID,
            "EVENT_RLE",
            CompressionCodecKind::EventRle,
            DEFAULT_MAX_CHUNK_BYTES,
            DEFAULT_MAX_CHUNK_BYTES,
            "PRISM-INTERNAL",
            true,
            true,
        )
    }

    fn encode(&self, canonical: &[u8]) -> CompressionResult<Vec<u8>> {
        if canonical.len() % EVENT_RECORD_WIDTH != 0 {
            return Err(CompressionError::Section(
                "event table record width mismatch for EVENT_RLE".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(canonical.len() + 12);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(canonical.len() as u64).to_le_bytes());

        let records = canonical.len() / EVENT_RECORD_WIDTH;
        let mut idx = 0usize;
        while idx < records {
            let start = idx * EVENT_RECORD_WIDTH;
            let kind = read_u16_le(canonical, start + 8)?;
            let mut run_len = 1usize;
            while idx + run_len < records {
                let next = (idx + run_len) * EVENT_RECORD_WIDTH;
                if read_u16_le(canonical, next + 8)? != kind {
                    break;
                }
                run_len += 1;
            }
            let run_len_u32 = u32::try_from(run_len).map_err(|_| {
                CompressionError::Format("EVENT_RLE run length exceeds u32".to_string())
            })?;
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&run_len_u32.to_le_bytes());
            let byte_len = run_len * EVENT_RECORD_WIDTH;
            out.extend_from_slice(&canonical[start..start + byte_len]);
            idx += run_len;
        }
        Ok(out)
    }

    fn decode(&self, encoded: &[u8]) -> CompressionResult<Vec<u8>> {
        if encoded.len() < 12 || &encoded[0..4] != MAGIC {
            return Err(CompressionError::Format(
                "invalid EVENT_RLE header".to_string(),
            ));
        }
        let expected_len =
            usize::try_from(crate::codecs::read_u64_le(encoded, 4)?).map_err(|_| {
                CompressionError::Format("EVENT_RLE decoded length too large".to_string())
            })?;
        if expected_len % EVENT_RECORD_WIDTH != 0 {
            return Err(CompressionError::Format(
                "EVENT_RLE decoded length is not record-aligned".to_string(),
            ));
        }

        let mut out = Vec::with_capacity(expected_len);
        let mut cursor = 12usize;
        while cursor < encoded.len() {
            if cursor + 6 > encoded.len() {
                return Err(CompressionError::Format(
                    "truncated EVENT_RLE run header".to_string(),
                ));
            }
            let kind = read_u16_le(encoded, cursor)?;
            cursor += 2;
            let run_len = read_u32_le(encoded, cursor)? as usize;
            cursor += 4;
            if run_len == 0 {
                return Err(CompressionError::Format(
                    "EVENT_RLE zero-length run".to_string(),
                ));
            }
            let byte_len = run_len
                .checked_mul(EVENT_RECORD_WIDTH)
                .ok_or_else(|| CompressionError::Format("EVENT_RLE run overflow".to_string()))?;
            if cursor + byte_len > encoded.len() {
                return Err(CompressionError::Format(
                    "truncated EVENT_RLE records".to_string(),
                ));
            }
            if out.len() + byte_len > expected_len {
                return Err(CompressionError::Format(
                    "EVENT_RLE decoded length exceeds header".to_string(),
                ));
            }
            for record in encoded[cursor..cursor + byte_len].chunks_exact(EVENT_RECORD_WIDTH) {
                if read_u16_le(record, 8)? != kind {
                    return Err(CompressionError::Format(
                        "EVENT_RLE kind mismatch in run".to_string(),
                    ));
                }
                out.extend_from_slice(record);
            }
            cursor += byte_len;
        }
        if out.len() != expected_len {
            return Err(CompressionError::Format(
                "EVENT_RLE decoded length mismatch".to_string(),
            ));
        }
        Ok(out)
    }
}
