use crate::codecs::{read_u64_le, DEFAULT_MAX_CHUNK_BYTES, TIME_RECORD_WIDTH};
use crate::compression_registry::{
    builtin_codec_record, CompressionCodec, CompressionCodecKind, CompressionCodecRecord,
    CompressionError, CompressionResult, DELTA_TIME_CODEC_ID,
};

const MAGIC: &[u8; 4] = b"DTIM";

#[derive(Default)]
pub struct DeltaTimeCodec;

impl CompressionCodec for DeltaTimeCodec {
    fn record(&self) -> CompressionCodecRecord {
        builtin_codec_record(
            DELTA_TIME_CODEC_ID,
            "DELTA_TIME",
            CompressionCodecKind::DeltaTime,
            DEFAULT_MAX_CHUNK_BYTES,
            DEFAULT_MAX_CHUNK_BYTES,
            "PRISM-INTERNAL",
            true,
            true,
        )
    }

    fn encode(&self, canonical: &[u8]) -> CompressionResult<Vec<u8>> {
        if canonical.len() % TIME_RECORD_WIDTH != 0 {
            return Err(CompressionError::Section(
                "time table record width mismatch for DELTA_TIME".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(canonical.len() + 12);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(canonical.len() as u64).to_le_bytes());
        let mut previous_id = 0u64;
        for (idx, record) in canonical.chunks_exact(TIME_RECORD_WIDTH).enumerate() {
            let time_id = read_u64_le(record, 0)?;
            if idx > 0 && time_id < previous_id {
                return Err(CompressionError::Format(
                    "DELTA_TIME source records are not monotonic".to_string(),
                ));
            }
            out.extend_from_slice(&(time_id - previous_id).to_le_bytes());
            out.extend_from_slice(&record[8..TIME_RECORD_WIDTH]);
            previous_id = time_id;
        }
        Ok(out)
    }

    fn decode(&self, encoded: &[u8]) -> CompressionResult<Vec<u8>> {
        if encoded.len() < 12 || &encoded[0..4] != MAGIC {
            return Err(CompressionError::Format(
                "invalid DELTA_TIME header".to_string(),
            ));
        }
        let expected_len = usize::try_from(read_u64_le(encoded, 4)?).map_err(|_| {
            CompressionError::Format("DELTA_TIME decoded length too large".to_string())
        })?;
        if expected_len % TIME_RECORD_WIDTH != 0 {
            return Err(CompressionError::Format(
                "DELTA_TIME decoded length is not record-aligned".to_string(),
            ));
        }
        let record_count = expected_len / TIME_RECORD_WIDTH;
        let expected_encoded_len = 12usize
            .checked_add(record_count.checked_mul(TIME_RECORD_WIDTH).ok_or_else(|| {
                CompressionError::Format("DELTA_TIME encoded length overflow".to_string())
            })?)
            .ok_or_else(|| {
                CompressionError::Format("DELTA_TIME encoded length overflow".to_string())
            })?;
        if encoded.len() != expected_encoded_len {
            return Err(CompressionError::Format(
                "DELTA_TIME encoded length mismatch".to_string(),
            ));
        }

        let mut out = Vec::with_capacity(expected_len);
        let mut previous_id = 0u64;
        let mut cursor = 12;
        for _ in 0..record_count {
            let delta = read_u64_le(encoded, cursor)?;
            cursor += 8;
            let time_id = previous_id
                .checked_add(delta)
                .ok_or_else(|| CompressionError::Format("DELTA_TIME delta overflow".to_string()))?;
            out.extend_from_slice(&time_id.to_le_bytes());
            out.extend_from_slice(&encoded[cursor..cursor + 72]);
            cursor += 72;
            previous_id = time_id;
        }
        Ok(out)
    }
}
