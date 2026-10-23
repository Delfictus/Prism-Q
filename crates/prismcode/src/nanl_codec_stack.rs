//! NANL-HD codec stack descriptor and validation.

use serde::{Deserialize, Serialize};

pub type NanlCodecStackResult<T> = Result<T, NanlCodecStackError>;

#[derive(Debug, thiserror::Error)]
pub enum NanlCodecStackError {
    #[error("PC_ERR_NANL_CODEC_STACK:{0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[repr(u16)]
pub enum NanlCodecKind {
    Cse = 1,
    Dict = 2,
    Motif = 3,
    Delta = 4,
    Var = 5,
    Ans = 6,
    H16 = 7,
    Bin = 8,
    Q4 = 9,
    DnaSim = 10,
    NrfSparse = 11,
    DrfActiveState = 12,
    Ecc = 13,
    Merkle = 14,
}

impl NanlCodecKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cse => "CSE",
            Self::Dict => "NANL-DICT",
            Self::Motif => "NANL-MOTIF",
            Self::Delta => "NANL-DELTA",
            Self::Var => "NANL-VAR",
            Self::Ans => "NANL-ANS",
            Self::H16 => "NANL-H16",
            Self::Bin => "NANL-BIN",
            Self::Q4 => "NANL-Q4",
            Self::DnaSim => "NANL-DNA-SIM",
            Self::NrfSparse => "NRF-SPARSE",
            Self::DrfActiveState => "DRF-ACTIVE-STATE",
            Self::Ecc => "ECC",
            Self::Merkle => "MERKLE",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlCodecLayer {
    pub layer_id: u64,
    pub stack_id: u64,
    pub layer_order: u16,
    pub codec_kind: NanlCodecKind,
    pub codec_id: u16,
    pub codec_version: u32,
    pub input_ref: u64,
    pub output_ref: u64,
    pub parameter_ref: u64,
    pub model_ref: u64,
    pub hash_before: String,
    pub hash_after: String,
    pub deterministic: bool,
    pub gpu_decode_supported: bool,
    pub streaming_decode_supported: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlCodecStack {
    pub stack_id: u64,
    pub stack_name: String,
    pub version_major: u16,
    pub version_minor: u16,
    pub input_state_kind: u16,
    pub output_state_kind: u16,
    pub layer_count: u16,
    pub canonical_input_hash: String,
    pub canonical_output_hash: String,
    pub compressed_hash: String,
    pub lossless_required: bool,
    pub replay_required: bool,
    pub firmware_profile_ref: u64,
    pub provenance_ref: u64,
    pub seal_ref: u64,
    pub layers: Vec<NanlCodecLayer>,
}

pub fn build_stack(
    name: &str,
    kinds: &[NanlCodecKind],
    canonical_payload: &[u8],
) -> NanlCodecStackResult<NanlCodecStack> {
    if kinds.is_empty() {
        return Err(NanlCodecStackError::Invalid(
            "codec stack is empty".to_string(),
        ));
    }
    let stack_id = stable_id(name);
    let mut previous_hash = hash_hex(canonical_payload);
    let mut layers = Vec::with_capacity(kinds.len());
    for (idx, kind) in kinds.iter().copied().enumerate() {
        let after = hash_hex(format!("{}\0{}\0{}", previous_hash, idx, kind.as_str()).as_bytes());
        layers.push(NanlCodecLayer {
            layer_id: idx as u64,
            stack_id,
            layer_order: idx as u16,
            codec_kind: kind,
            codec_id: kind as u16,
            codec_version: 1,
            input_ref: idx as u64,
            output_ref: idx as u64 + 1,
            parameter_ref: 0,
            model_ref: if matches!(kind, NanlCodecKind::Ans | NanlCodecKind::Q4) {
                idx as u64 + 100
            } else {
                0
            },
            hash_before: previous_hash,
            hash_after: after.clone(),
            deterministic: true,
            gpu_decode_supported: matches!(
                kind,
                NanlCodecKind::Bin | NanlCodecKind::Q4 | NanlCodecKind::DrfActiveState
            ),
            streaming_decode_supported: !matches!(kind, NanlCodecKind::DnaSim),
        });
        previous_hash = after;
    }
    let compressed_hash = hash_layers(&layers);
    Ok(NanlCodecStack {
        stack_id,
        stack_name: name.to_string(),
        version_major: 0,
        version_minor: 1,
        input_state_kind: 1,
        output_state_kind: 2,
        layer_count: layers.len() as u16,
        canonical_input_hash: hash_hex(canonical_payload),
        canonical_output_hash: hash_hex(canonical_payload),
        compressed_hash,
        lossless_required: true,
        replay_required: true,
        firmware_profile_ref: 0,
        provenance_ref: 0,
        seal_ref: 0,
        layers,
    })
}

pub fn fast_execution_stack(payload: &[u8]) -> NanlCodecStackResult<NanlCodecStack> {
    build_stack(
        "FAST_EXECUTION_STACK",
        &[
            NanlCodecKind::Cse,
            NanlCodecKind::Dict,
            NanlCodecKind::Delta,
            NanlCodecKind::Motif,
            NanlCodecKind::Ans,
            NanlCodecKind::Bin,
        ],
        payload,
    )
}

pub fn q4_archival_stack(payload: &[u8]) -> NanlCodecStackResult<NanlCodecStack> {
    build_stack(
        "Q4_ARCHIVAL_STACK",
        &[
            NanlCodecKind::Cse,
            NanlCodecKind::Dict,
            NanlCodecKind::Delta,
            NanlCodecKind::Motif,
            NanlCodecKind::Ans,
            NanlCodecKind::Q4,
            NanlCodecKind::Ecc,
            NanlCodecKind::Merkle,
        ],
        payload,
    )
}

pub fn validate_stack(stack: &NanlCodecStack) -> NanlCodecStackResult<()> {
    if stack.layer_count != stack.layers.len() as u16 {
        return Err(NanlCodecStackError::Invalid(
            "layer count mismatch".to_string(),
        ));
    }
    if !stack.lossless_required || !stack.replay_required {
        return Err(NanlCodecStackError::Invalid(
            "NANL-HD stack must require lossless roundtrip and replay".to_string(),
        ));
    }
    let mut previous_order = None::<u16>;
    for layer in &stack.layers {
        if layer.stack_id != stack.stack_id {
            return Err(NanlCodecStackError::Invalid(
                "layer stack id mismatch".to_string(),
            ));
        }
        if !layer.deterministic {
            return Err(NanlCodecStackError::Invalid(
                "NANL-HD layer is not deterministic".to_string(),
            ));
        }
        if let Some(previous) = previous_order {
            if layer.layer_order <= previous {
                return Err(NanlCodecStackError::Invalid(
                    "codec layer order is not canonical".to_string(),
                ));
            }
        }
        previous_order = Some(layer.layer_order);
    }
    if stack.compressed_hash != hash_layers(&stack.layers) {
        return Err(NanlCodecStackError::Invalid(
            "codec stack compressed hash mismatch".to_string(),
        ));
    }
    Ok(())
}

pub fn required_nanl_hd_opcodes() -> Vec<&'static str> {
    vec![
        "NANL.DICT_BUILD",
        "NANL.DICT_LOOKUP",
        "NANL.DICT_ENCODE",
        "NANL.DICT_DECODE",
        "NANL.DICT_VALIDATE",
        "NANL.DICT_HASH",
        "NANL.DICT_FREEZE",
        "NANL.MOTIF_DISCOVER",
        "NANL.MOTIF_DEFINE",
        "NANL.MOTIF_ENCODE",
        "NANL.MOTIF_DECODE",
        "NANL.MOTIF_VALIDATE",
        "NANL.MOTIF_HASH",
        "NANL.MOTIF_TO_NRF",
        "NANL.DELTA_FRAME",
        "NANL.DELTA_TIME",
        "NANL.DELTA_STATE",
        "NANL.DELTA_EVENT",
        "NANL.DELTA_THERMO",
        "NANL.DELTA_UNCERTAINTY",
        "NANL.DELTA_DECODE",
        "NANL.DELTA_VALIDATE",
        "NANL.VAR_ENCODE",
        "NANL.VAR_DECODE",
        "NANL.VAR_VALIDATE",
        "NANL.VAR_TABLE_BUILD",
        "NANL.VAR_TABLE_HASH",
        "NANL.ANS_MODEL_BUILD",
        "NANL.ANS_MODEL_FREEZE",
        "NANL.ANS_ENCODE",
        "NANL.ANS_DECODE",
        "NANL.ANS_VALIDATE",
        "NANL.ANS_RENORMALIZE",
        "NANL.ANS_HASH",
        "NANL.ANS_STREAM_INDEX",
        "NANL.H16_ENCODE",
        "NANL.H16_DECODE",
        "NANL.H16_VALIDATE",
        "NANL.H16_TO_BIN",
        "NANL.BIN_TO_H16",
        "NANL.Q4_ENCODE",
        "NANL.Q4_DECODE",
        "NANL.Q4_VALIDATE",
        "NANL.Q4_CONSTRAINT_SCORE",
        "NANL.Q4_ECC_ATTACH",
        "NANL.Q4_ECC_VERIFY",
        "NANL.Q4_MERKLE_ROOT",
        "NANL.Q4_TO_BIN",
        "NANL.BIN_TO_Q4",
        "NANL.Q4_TO_DNA_OLIGO",
        "NANL.DNA_SIM_EXPORT",
        "NANL.DNA_SIM_ERROR_CHANNEL",
        "NANL.DNA_SIM_DECODE",
        "NANL.DNA_SIM_VALIDATE",
        "NANL.DNA_SIM_REPORT",
        "NRF.SPARSE_ENCODE",
        "NRF.SPARSE_DECODE",
        "NRF.MOTIF_PACK",
        "NRF.MOTIF_UNPACK",
        "NRF.VALIDATE_REPLAY",
        "NRF.TO_DRF",
        "DRF.MAP_NRF",
        "DRF.FIRE",
        "DRF.RESONATE",
        "DRF.WRITEBACK",
        "DRF.RECONSTRUCT",
        "DRF.CONSISTENCY_ASSERT",
        "DRF.ACTIVE_STATE_ENCODE",
        "DRF.ACTIVE_STATE_VALIDATE",
    ]
}

fn hash_layers(layers: &[NanlCodecLayer]) -> String {
    let mut bytes = Vec::new();
    for layer in layers {
        bytes.extend_from_slice(layer.hash_before.as_bytes());
        bytes.extend_from_slice(layer.hash_after.as_bytes());
        bytes.extend_from_slice(layer.codec_kind.as_str().as_bytes());
        bytes.push(0);
    }
    hash_hex(&bytes)
}

fn stable_id(name: &str) -> u64 {
    let digest = blake3::hash(name.as_bytes());
    let mut bytes = [0_u8; 8];
    bytes.copy_from_slice(&digest.as_bytes()[0..8]);
    u64::from_le_bytes(bytes)
}

fn hash_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}
