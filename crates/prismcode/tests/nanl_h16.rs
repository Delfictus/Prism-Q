use prismcode::nanl_h16::{
    decode_bytes, encode_bytes, h16_hex_string, validate_h16_claim, validate_h16_descriptor,
    validate_packet, H16ClaimDescriptor,
};

#[test]
fn nanl_h16_roundtrips_exact_canonical_bytes() {
    let payload = b"canonical PRISM H16 internal nibble stream";

    let packet = encode_bytes(payload).unwrap();
    let decoded = decode_bytes(&packet).unwrap();
    let hex = h16_hex_string(&packet).unwrap();

    assert_eq!(decoded, payload);
    assert_eq!(packet.magic, "NANL_H16");
    assert_eq!(packet.symbol_count, (payload.len() * 2) as u64);
    assert!(hex.chars().all(|symbol| symbol.is_ascii_hexdigit()));
}

#[test]
fn nanl_h16_rejects_q4_or_quaternary_description() {
    let err = validate_h16_descriptor(&H16ClaimDescriptor {
        described_as: "Q4-compatible quaternary transport".to_string(),
        ..H16ClaimDescriptor::default()
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("PC_ERR_NANL_H16"));
    assert!(err.contains("must not be described as Q4"));

    let bool_err = validate_h16_claim(true, false).unwrap_err().to_string();
    assert!(bool_err.contains("not quaternary"));
}

#[test]
fn nanl_h16_rejects_dna_like_description() {
    let err = validate_h16_descriptor(&H16ClaimDescriptor {
        dna_like: true,
        described_as: "internal digital nibble codec".to_string(),
        ..H16ClaimDescriptor::default()
    })
    .unwrap_err()
    .to_string();

    assert!(err.contains("PC_ERR_NANL_H16"));
    assert!(err.contains("not DNA-like"));
}

#[test]
fn nanl_h16_tampered_stream_hash_fails_closed() {
    let mut packet = encode_bytes(b"h16 hash protected canonical payload").unwrap();
    packet.packed_nibbles[0] ^= 0x01;

    let err = validate_packet(&packet).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_NANL_H16"));
    assert!(err.contains("H16 stream hash mismatch"));
}

#[test]
fn nanl_h16_rejects_bad_symbol_count() {
    let mut packet = encode_bytes(b"valid packet").unwrap();
    packet.symbol_count += 1;

    let err = validate_packet(&packet).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_NANL_H16"));
    assert!(err.contains("symbol count mismatch"));
}
