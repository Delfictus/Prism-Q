//! Built-in executable compression codecs for PRISMCODE ABI v0.1.

pub mod delta_time;
pub mod event_dictionary;
pub mod event_rle;
pub mod nanl_packed_u2;
pub mod raw_uncompressed;
pub mod section_zstd;
pub mod tensor_chunked;

use crate::compression_registry::{CompressionError, CompressionResult};

pub(crate) const TIME_RECORD_WIDTH: usize = 80;
pub(crate) const EVENT_RECORD_WIDTH: usize = 96;
pub(crate) const DEFAULT_MAX_CHUNK_BYTES: u64 = 64 * 1024 * 1024;

pub(crate) fn read_u16_le(bytes: &[u8], offset: usize) -> CompressionResult<u16> {
    let slice = bytes
        .get(offset..offset + 2)
        .ok_or_else(|| CompressionError::Format("u16 read out of range".to_string()))?;
    Ok(u16::from_le_bytes(slice.try_into().map_err(|_| {
        CompressionError::Internal("u16 conversion failed".to_string())
    })?))
}

pub(crate) fn read_u32_le(bytes: &[u8], offset: usize) -> CompressionResult<u32> {
    let slice = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| CompressionError::Format("u32 read out of range".to_string()))?;
    Ok(u32::from_le_bytes(slice.try_into().map_err(|_| {
        CompressionError::Internal("u32 conversion failed".to_string())
    })?))
}

pub(crate) fn read_u64_le(bytes: &[u8], offset: usize) -> CompressionResult<u64> {
    let slice = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| CompressionError::Format("u64 read out of range".to_string()))?;
    Ok(u64::from_le_bytes(slice.try_into().map_err(|_| {
        CompressionError::Internal("u64 conversion failed".to_string())
    })?))
}

pub(crate) fn read_hash(bytes: &[u8], offset: usize) -> CompressionResult<[u8; 32]> {
    let slice = bytes
        .get(offset..offset + 32)
        .ok_or_else(|| CompressionError::Format("hash read out of range".to_string()))?;
    let mut out = [0u8; 32];
    out.copy_from_slice(slice);
    Ok(out)
}
