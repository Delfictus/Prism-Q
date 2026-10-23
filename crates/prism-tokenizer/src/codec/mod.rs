//! §5 — Stage 0 codec-inverting front-end.
//!
//! INV-1: identity is the *decoded* record. QNANL container decoding delegates to the
//! authoritative `prismcode` decoder; the spike-packet path + byte-plane transpose are
//! reimplemented against the grounded engine layout. No byte reaches a quantizer here.

pub mod container;
pub mod kcc;
pub mod prspk001;
pub mod records;
pub mod v2_frames;
pub mod vector_tensor;

pub use container::{
    decode_qnanl, decode_spike_packet, transpose, untranspose, QnanlDecoded, SpikePacketDecoded,
    QNANL_MAGIC, SPIKE_MAGIC,
};
pub use kcc::{decode_prkcc001, Prkcc001Decoded, Prkcc001Header, PRKCC001_MAGIC};
pub use prspk001::{decode_prspk001, decode_prspk001_head, Prspk001Decoded, PRSPK001_MAGIC};
pub use records::{known_record_stride, GpuSpikeEvent, GPU_SPIKE_EVENT_BYTES};
pub use v2_frames::{decode_v2_frames, V2Frame, V2FramesDecoded};
pub use vector_tensor::{decode_vector_tensor, VectorTensorDecoded, VectorTensorKind};

/// Codec-inversion error. A failure here never aborts the stream (§5.1): the caller emits a
/// quarantined (`verified=0`) record instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// Malformed container / wrong magic / length mismatch / bad header.
    Format(String),
    /// Decompression or seal-codec failure on otherwise well-formed bytes.
    Codec(String),
}

impl core::fmt::Display for CodecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CodecError::Format(m) => write!(f, "codec format error: {m}"),
            CodecError::Codec(m) => write!(f, "codec error: {m}"),
        }
    }
}

impl std::error::Error for CodecError {}
