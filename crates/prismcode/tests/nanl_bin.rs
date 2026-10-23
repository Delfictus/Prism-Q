use prismcode::nanl_bin::{
    build_index, decode_packet, encode_tokens, sample_tokens, validate_packet, NanlToken,
    NanlTokenClass,
};

#[test]
fn nanl_bin_roundtrips_typed_tokens_and_hashes() {
    let tokens = sample_tokens();
    let packet = encode_tokens(&tokens).unwrap();
    let decoded = decode_packet(packet.bytes()).unwrap();
    let report = validate_packet(packet.bytes()).unwrap();

    assert_eq!(decoded, tokens);
    assert_eq!(report.status, "PASS");
    assert_eq!(report.mode, "NANL-BIN");
    assert_eq!(report.token_width_bits, 2);
    assert_eq!(report.stream_count, 3);
    assert!(report.roundtrip_canonical_hash_pass);
    assert!(!report.debug_json_hot_path_used);
    assert_eq!(build_index(&tokens).len(), tokens.len());
}

#[test]
fn nanl_bin_selects_wider_token_lanes_when_needed() {
    let tokens = [
        NanlToken::new(NanlTokenClass::Entity, 1, 4),
        NanlToken::new(NanlTokenClass::EventKind, 2, 15),
    ];
    let packet = encode_tokens(&tokens).unwrap();
    assert_eq!(packet.report().token_width_bits, 4);
    assert_eq!(decode_packet(packet.bytes()).unwrap(), tokens);

    let tokens = [NanlToken::new(NanlTokenClass::Entity, 1, 256)];
    let packet = encode_tokens(&tokens).unwrap();
    assert_eq!(packet.report().token_width_bits, 16);
    assert_eq!(decode_packet(packet.bytes()).unwrap(), tokens);
}

#[test]
fn nanl_bin_tampered_stream_hash_fails_closed() {
    let packet = encode_tokens(&sample_tokens()).unwrap();
    let mut bytes = packet.bytes().to_vec();
    let last = bytes.len() - 1;
    bytes[last] ^= 0b01;

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("stream hash mismatch"));
}

#[test]
fn nanl_bin_rejects_unknown_token_class() {
    let packet = encode_tokens(&sample_tokens()).unwrap();
    let mut bytes = packet.bytes().to_vec();
    let class_stream_start = 144;
    bytes[class_stream_start] = 250;
    let stream_hash = blake3::hash(&bytes[144..]);
    bytes[80..112].copy_from_slice(stream_hash.as_bytes());

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_FORMAT"));
    assert!(err.contains("unknown NANL-BIN token class"));
}
