//! NANL-HD firmware stage contract.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NanlHdFirmwarePolicy {
    pub required: bool,
    pub q4_nontrivial_required: bool,
    pub ans_required: bool,
    pub dict_required: bool,
    pub motif_required: bool,
    pub delta_required: bool,
    pub var_required: bool,
    pub h16_allowed: bool,
    pub h16_claim_dna_like: bool,
    pub nrf_sparse_required: bool,
    pub drf_active_state_required: bool,
    pub lossless_required: bool,
    pub benchmark_required: bool,
    pub overclaim_guard_required: bool,
}

impl Default for NanlHdFirmwarePolicy {
    fn default() -> Self {
        Self {
            required: true,
            q4_nontrivial_required: true,
            ans_required: true,
            dict_required: true,
            motif_required: true,
            delta_required: true,
            var_required: true,
            h16_allowed: true,
            h16_claim_dna_like: false,
            nrf_sparse_required: true,
            drf_active_state_required: true,
            lossless_required: true,
            benchmark_required: true,
            overclaim_guard_required: true,
        }
    }
}

pub fn required_nanl_hd_stages() -> Vec<&'static str> {
    vec![
        "NANL_HD_CODEC_STACK_SELECT",
        "NANL_DICT_BUILD",
        "NANL_MOTIF_BUILD",
        "NANL_DELTA_ENCODE",
        "NANL_VAR_ENCODE",
        "NANL_ANS_ENCODE",
        "NANL_BIN_ENCODE",
        "NANL_Q4_ENCODE",
        "NANL_DNA_SIMULATE",
        "NRF_SPARSE_ENCODE",
        "DRF_ACTIVE_STATE_ENCODE",
        "NANL_HD_VALIDATE",
        "NANL_HD_BENCHMARK",
    ]
}

pub fn validate_firmware_policy(policy: &NanlHdFirmwarePolicy) -> Result<(), String> {
    if !policy.required
        || !policy.q4_nontrivial_required
        || !policy.ans_required
        || !policy.dict_required
        || !policy.motif_required
        || !policy.delta_required
        || !policy.var_required
        || !policy.nrf_sparse_required
        || !policy.drf_active_state_required
        || !policy.lossless_required
        || !policy.benchmark_required
        || !policy.overclaim_guard_required
    {
        return Err("NANL-HD firmware policy weakens required stages".to_string());
    }
    if policy.h16_claim_dna_like {
        return Err("H16 cannot be claimed DNA-like".to_string());
    }
    Ok(())
}
