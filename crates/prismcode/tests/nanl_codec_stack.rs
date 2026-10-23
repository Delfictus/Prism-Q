use prismcode::nanl_codec_stack::{
    fast_execution_stack, q4_archival_stack, required_nanl_hd_opcodes, validate_stack,
    NanlCodecKind,
};

#[test]
fn codec_stack_records_lossless_replay_required_layers() {
    let payload = b"canonical PRISM payload";
    let fast = fast_execution_stack(payload).unwrap();
    let q4 = q4_archival_stack(payload).unwrap();

    validate_stack(&fast).unwrap();
    validate_stack(&q4).unwrap();
    assert!(fast.lossless_required);
    assert!(fast.replay_required);
    assert_eq!(fast.layers[0].codec_kind, NanlCodecKind::Cse);
    assert!(fast
        .layers
        .iter()
        .any(|layer| layer.codec_kind == NanlCodecKind::Ans));
    assert!(fast
        .layers
        .iter()
        .any(|layer| layer.codec_kind == NanlCodecKind::Bin));
    assert!(q4
        .layers
        .iter()
        .any(|layer| layer.codec_kind == NanlCodecKind::Q4));
    assert!(q4
        .layers
        .iter()
        .any(|layer| layer.codec_kind == NanlCodecKind::Merkle));
}

#[test]
fn required_opcode_surface_is_validator_visible() {
    let opcodes = required_nanl_hd_opcodes();
    for mnemonic in [
        "NANL.DICT_BUILD",
        "NANL.MOTIF_DISCOVER",
        "NANL.DELTA_FRAME",
        "NANL.VAR_ENCODE",
        "NANL.ANS_ENCODE",
        "NANL.H16_ENCODE",
        "NANL.Q4_VALIDATE",
        "NANL.DNA_SIM_VALIDATE",
        "NRF.SPARSE_ENCODE",
        "DRF.ACTIVE_STATE_VALIDATE",
    ] {
        assert!(opcodes.contains(&mnemonic), "missing opcode {mnemonic}");
    }
}
