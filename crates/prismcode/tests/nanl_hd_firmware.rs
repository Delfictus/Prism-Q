use prismcode::nanl_hd_firmware::{
    required_nanl_hd_stages, validate_firmware_policy, NanlHdFirmwarePolicy,
};

#[test]
fn firmware_policy_requires_all_nanl_hd_stages() {
    let policy = NanlHdFirmwarePolicy::default();
    validate_firmware_policy(&policy).unwrap();
    let stages = required_nanl_hd_stages();
    for stage in [
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
    ] {
        assert!(stages.contains(&stage), "missing firmware stage {stage}");
    }
}

#[test]
fn firmware_policy_rejects_h16_dna_like_or_missing_ans() {
    let mut policy = NanlHdFirmwarePolicy::default();
    policy.h16_claim_dna_like = true;
    assert!(validate_firmware_policy(&policy).is_err());

    let mut policy = NanlHdFirmwarePolicy::default();
    policy.ans_required = false;
    assert!(validate_firmware_policy(&policy).is_err());
}
