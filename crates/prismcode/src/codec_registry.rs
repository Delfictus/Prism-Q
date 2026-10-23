//! Codec registry surface for Rust/CUDA readers.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CodecRecord {
    pub codec_id: String,
    pub codec_version: String,
    pub semantic_status: String,
    pub validator_status: String,
}

pub fn required_codec_ids() -> &'static [&'static str] {
    &[
        "NANL_DNA_CODEC",
        "NEUROMORPHIC_TIME_CODEC",
        "MD_BINDING_CODEC",
        "MSM_SPECTRAL_CODEC",
        "TEACHER_STUDENT_CODEC",
        "GFLOWNET_CODEC",
        "PRISMAI_INTERFACE_CODEC",
        "NRF_DSTW_CODEC",
        "FORGE_115K_CHEMISTRY_CODEC",
        "RUNTIME_LOG_STATE_CODEC",
    ]
}
