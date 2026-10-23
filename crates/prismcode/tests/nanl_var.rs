use prismcode::nanl_var::{
    decode_u64, encode_u64, validate_code_table, validate_packet, NanlVarPacket, VarCode,
};

#[test]
fn nanl_var_roundtrips_exact_canonical_u64_symbols() {
    let values = [0, 1, 2, 127, 128, 16_384, u32::MAX as u64, u64::MAX - 3];

    let packet = encode_u64(&values).unwrap();
    let decoded = decode_u64(&packet).unwrap();

    assert_eq!(decoded, values);
    assert_eq!(packet.magic, "NANL_VAR");
    assert!(packet.min_bits >= 1);
    assert!(packet.max_bits <= 64);
    assert!(!packet.encoded.is_empty());
}

#[test]
fn nanl_var_rejects_malformed_prefix_table() {
    let table = [
        VarCode {
            symbol_id: 1,
            bit_width: 1,
            code: 0,
            frequency: 10,
            escape_required: false,
        },
        VarCode {
            symbol_id: 2,
            bit_width: 2,
            code: 0,
            frequency: 1,
            escape_required: false,
        },
    ];

    let err = validate_code_table(&table).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_NANL_VAR"));
    assert!(err.contains("prefix-free"));
}

#[test]
fn nanl_var_rejects_duplicate_symbols() {
    let table = [
        VarCode {
            symbol_id: 7,
            bit_width: 1,
            code: 0,
            frequency: 10,
            escape_required: false,
        },
        VarCode {
            symbol_id: 7,
            bit_width: 1,
            code: 1,
            frequency: 9,
            escape_required: false,
        },
    ];

    let err = validate_code_table(&table).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_NANL_VAR"));
    assert!(err.contains("duplicate VAR symbol"));
}

#[test]
fn nanl_var_rejects_overlong_integer_stream() {
    let mut packet = NanlVarPacket {
        magic: "NANL_VAR".to_string(),
        version: "0.1".to_string(),
        code_table_hash: "not-empty-bound-table".to_string(),
        min_bits: 1,
        max_bits: 7,
        escape_code: 0xff,
        encoded_stream_hash: String::new(),
        canonical_roundtrip_hash: blake3::hash(&0u64.to_le_bytes()).to_hex().to_string(),
        symbol_count: 1,
        encoded: vec![0x80, 0x00],
    };
    packet.encoded_stream_hash = blake3::hash(&packet.encoded).to_hex().to_string();

    let err = decode_u64(&packet).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_NANL_VAR"));
    assert!(err.contains("overlong VAR integer"));
}

#[test]
fn nanl_var_tampered_stream_hash_fails_closed() {
    let mut packet = encode_u64(&[1, 127, 128, 4096]).unwrap();
    packet.encoded[0] ^= 0x01;

    let err = validate_packet(&packet).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_NANL_VAR"));
    assert!(err.contains("VAR stream hash mismatch"));
}
