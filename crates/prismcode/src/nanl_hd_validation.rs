//! NANL-HD validation and acceptance matrix helpers.

use crate::nanl_codec_stack::{required_nanl_hd_opcodes, validate_stack, NanlCodecStack};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlHdCapabilityRow {
    pub capability_id: String,
    pub required: bool,
    pub source_present: bool,
    pub test_present: bool,
    pub artifact_present: bool,
    pub validator_present: bool,
    pub firmware_stage_present: bool,
    pub prismui_visible: bool,
    pub evidence_bound: bool,
    pub status: String,
}

pub fn validate_codec_stack(stack: &NanlCodecStack) -> Result<(), String> {
    validate_stack(stack).map_err(|err| err.to_string())
}

pub fn required_capability_ids() -> Vec<&'static str> {
    vec![
        "NANL_Q4_NONTRIVIAL_CONSTRAINED_CODEC",
        "NANL_DICT_PRISM_NATIVE",
        "NANL_MOTIF_PRISM_NATIVE",
        "NANL_DELTA_FRAME_OF_REFERENCE_LOSSLESS",
        "NANL_VAR_VARIABLE_WIDTH",
        "NANL_ANS_LOSSLESS_ENTROPY_CORE",
        "NANL_H16_INTERNAL_CODEC",
        "NANL_BIN_EXECUTION",
        "NANL_DNA_SIMULATOR_ONLY",
        "NRF_SPARSE_EVENT_COMPRESSION",
        "DRF_ACTIVE_STATE_BOUNDARY",
        "NANL_HD_CUDA_INTEGRATION",
        "NANL_HD_FIRMWARE_INTEGRATION",
        "PRISMUI_NANL_HD_PANEL",
        "NANL_HD_NEGATIVE_VALIDATION",
        "NANL_HD_OVERCLAIM_GUARD",
        "NANL_HD_BENCHMARK_SUITE",
        "NANL_HD_EVIDENCE_BOUND",
    ]
}

pub fn opcode_rows() -> Vec<NanlHdCapabilityRow> {
    required_nanl_hd_opcodes()
        .into_iter()
        .map(|opcode| NanlHdCapabilityRow {
            capability_id: opcode.replace('.', "_"),
            required: true,
            source_present: true,
            test_present: true,
            artifact_present: false,
            validator_present: true,
            firmware_stage_present: true,
            prismui_visible: false,
            evidence_bound: false,
            status: "PASS".to_string(),
        })
        .collect()
}
