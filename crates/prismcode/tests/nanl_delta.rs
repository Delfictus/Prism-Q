use prismcode::nanl_delta::{
    decode_frame_delta_lossless, encode_frame_delta_lossless, encode_i64_lossless,
    reject_lossy_canonical_delta, validate_delta_policy, validate_frame_delta_packet,
    DeltaEncodingPolicy, FrameDeltaOp, PrecisionReduction,
};

#[test]
fn nanl_delta_roundtrips_exact_canonical_bytes() {
    let base = b"timebase=t0;frame=0001;water=11110000;thermo=298.150000";
    let target = b"timebase=t0;frame=0002;water=11110110;thermo=298.150001";

    let packet = encode_frame_delta_lossless(base, target).unwrap();
    let decoded = decode_frame_delta_lossless(&packet, base).unwrap();
    let report = validate_frame_delta_packet(&packet, base).unwrap();

    assert_eq!(decoded, target);
    assert_eq!(report.status, "PASS");
    assert_eq!(report.mode, "NANL-DELTA");
    assert!(report.canonical_roundtrip_hash_pass);
    assert!(!report.lossy_quantization_used);
}

#[test]
fn nanl_delta_rejects_lossy_canonical_quantization() {
    let err = validate_delta_policy(DeltaEncodingPolicy {
        canonical_required: true,
        precision_reduction: PrecisionReduction::QuantizedBits(16),
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("PC_ERR_NANL_DELTA"));
    assert!(err.contains("lossy precision reduction cannot be canonical"));

    let legacy_err = reject_lossy_canonical_delta(true).unwrap_err().to_string();
    assert!(legacy_err.contains("lossy precision reduction cannot be canonical"));
}

#[test]
fn nanl_delta_rejects_pathologically_bad_equal_run() {
    let base = b"abc";
    let target = b"abc";
    let mut packet = encode_frame_delta_lossless(base, target).unwrap();
    packet.operations = vec![FrameDeltaOp::EqualRun { len: 4 }];
    packet.operation_count = 1;
    packet.delta_stream_hash = {
        let mut bytes = Vec::new();
        bytes.push(0);
        bytes.extend_from_slice(&4u64.to_le_bytes());
        blake3::hash(&bytes).to_hex().to_string()
    };

    let err = validate_frame_delta_packet(&packet, base)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_NANL_DELTA"));
    assert!(err.contains("reads past base frame"));
}

#[test]
fn nanl_delta_rejects_zero_length_literal_run() {
    let base = b"abc";
    let target = b"axc";
    let mut packet = encode_frame_delta_lossless(base, target).unwrap();
    packet.operations = vec![FrameDeltaOp::LiteralRun { bytes: Vec::new() }];
    packet.operation_count = 1;
    packet.delta_stream_hash = {
        let mut bytes = Vec::new();
        bytes.push(1);
        bytes.extend_from_slice(&0u64.to_le_bytes());
        blake3::hash(&bytes).to_hex().to_string()
    };

    let err = validate_frame_delta_packet(&packet, base)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_NANL_DELTA"));
    assert!(err.contains("zero-length literal run"));
}

#[test]
fn nanl_delta_i64_compatibility_still_roundtrips() {
    let values = [10, 11, 9, 9, 42, -7, -7, 0];
    let packet = encode_i64_lossless(&values).unwrap();
    let decoded = prismcode::nanl_delta::decode_i64_lossless(&packet).unwrap();
    assert_eq!(decoded, values);
}
