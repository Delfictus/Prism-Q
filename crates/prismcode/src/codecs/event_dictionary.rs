use crate::codecs::{read_u16_le, read_u64_le, DEFAULT_MAX_CHUNK_BYTES, EVENT_RECORD_WIDTH};
use crate::compression_registry::{
    builtin_codec_record, CompressionCodec, CompressionCodecKind, CompressionCodecRecord,
    CompressionError, CompressionResult, EVENT_DICTIONARY_CODEC_ID,
};
use std::collections::{BTreeMap, BTreeSet};

const MAGIC: &[u8; 4] = b"EDIC";

#[derive(Default)]
pub struct EventDictionaryCodec;

impl CompressionCodec for EventDictionaryCodec {
    fn record(&self) -> CompressionCodecRecord {
        builtin_codec_record(
            EVENT_DICTIONARY_CODEC_ID,
            "EVENT_DICTIONARY",
            CompressionCodecKind::EventDictionary,
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
                "event table record width mismatch for EVENT_DICTIONARY".to_string(),
            ));
        }

        let mut kinds = BTreeSet::<u16>::new();
        for record in canonical.chunks_exact(EVENT_RECORD_WIDTH) {
            kinds.insert(read_u16_le(record, 8)?);
        }
        if kinds.len() > u16::MAX as usize {
            return Err(CompressionError::Format(
                "EVENT_DICTIONARY dictionary exceeds u16".to_string(),
            ));
        }

        let kind_to_index = kinds
            .iter()
            .enumerate()
            .map(|(index, kind)| (*kind, index as u16))
            .collect::<BTreeMap<_, _>>();

        let mut out = Vec::with_capacity(canonical.len() + 14 + kind_to_index.len() * 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(canonical.len() as u64).to_le_bytes());
        out.extend_from_slice(&(kind_to_index.len() as u16).to_le_bytes());
        for (kind, index) in &kind_to_index {
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&index.to_le_bytes());
        }
        for record in canonical.chunks_exact(EVENT_RECORD_WIDTH) {
            let kind = read_u16_le(record, 8)?;
            let index = kind_to_index.get(&kind).ok_or_else(|| {
                CompressionError::Internal("EVENT_DICTIONARY missing encoded kind".to_string())
            })?;
            out.extend_from_slice(&index.to_le_bytes());
            out.extend_from_slice(record);
        }
        Ok(out)
    }

    fn decode(&self, encoded: &[u8]) -> CompressionResult<Vec<u8>> {
        if encoded.len() < 14 || &encoded[0..4] != MAGIC {
            return Err(CompressionError::Format(
                "invalid EVENT_DICTIONARY header".to_string(),
            ));
        }
        let expected_len = usize::try_from(read_u64_le(encoded, 4)?).map_err(|_| {
            CompressionError::Format("EVENT_DICTIONARY decoded length too large".to_string())
        })?;
        if expected_len % EVENT_RECORD_WIDTH != 0 {
            return Err(CompressionError::Format(
                "EVENT_DICTIONARY decoded length is not record-aligned".to_string(),
            ));
        }

        let dict_len = read_u16_le(encoded, 12)? as usize;
        let mut cursor = 14usize;
        let dict_bytes = dict_len.checked_mul(4).ok_or_else(|| {
            CompressionError::Format("EVENT_DICTIONARY dictionary overflow".to_string())
        })?;
        if cursor + dict_bytes > encoded.len() {
            return Err(CompressionError::Format(
                "truncated EVENT_DICTIONARY dictionary".to_string(),
            ));
        }

        let mut reverse = BTreeMap::<u16, u16>::new();
        let mut seen_kinds = BTreeSet::<u16>::new();
        for _ in 0..dict_len {
            let kind = read_u16_le(encoded, cursor)?;
            let index = read_u16_le(encoded, cursor + 2)?;
            cursor += 4;
            if !seen_kinds.insert(kind) {
                return Err(CompressionError::Format(
                    "EVENT_DICTIONARY duplicate kind".to_string(),
                ));
            }
            if reverse.insert(index, kind).is_some() {
                return Err(CompressionError::Format(
                    "EVENT_DICTIONARY duplicate index".to_string(),
                ));
            }
        }

        let mut out = Vec::with_capacity(expected_len);
        while cursor < encoded.len() {
            if cursor + 2 + EVENT_RECORD_WIDTH > encoded.len() {
                return Err(CompressionError::Format(
                    "truncated EVENT_DICTIONARY record".to_string(),
                ));
            }
            let index = read_u16_le(encoded, cursor)?;
            cursor += 2;
            let record = &encoded[cursor..cursor + EVENT_RECORD_WIDTH];
            let expected_kind = reverse.get(&index).ok_or_else(|| {
                CompressionError::Format(
                    "EVENT_DICTIONARY index not present in dictionary".to_string(),
                )
            })?;
            if read_u16_le(record, 8)? != *expected_kind {
                return Err(CompressionError::Format(
                    "EVENT_DICTIONARY record kind mismatch".to_string(),
                ));
            }
            if out.len() + EVENT_RECORD_WIDTH > expected_len {
                return Err(CompressionError::Format(
                    "EVENT_DICTIONARY decoded length exceeds header".to_string(),
                ));
            }
            out.extend_from_slice(record);
            cursor += EVENT_RECORD_WIDTH;
        }
        if out.len() != expected_len {
            return Err(CompressionError::Format(
                "EVENT_DICTIONARY decoded length mismatch".to_string(),
            ));
        }
        Ok(out)
    }
}
