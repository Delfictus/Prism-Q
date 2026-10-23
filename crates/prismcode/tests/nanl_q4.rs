use prismcode::nanl_q4::{
    decode_packet, encode_payload, pack_q4_symbols, q4_symbols_to_string, unpack_q4_symbols,
    validate_constraints, validate_packet, Q4ConstraintProfile,
};

#[test]
fn nanl_q4_roundtrips_payload_with_constraints_and_parity() {
    let payload = b"PRISM-CSE-HD canonical payload with typed state refs and no JSON hot path";
    let packet = encode_payload(payload).unwrap();
    let decoded = decode_packet(packet.bytes()).unwrap();
    let report = validate_packet(packet.bytes()).unwrap();

    assert_eq!(decoded, payload);
    assert_eq!(report.status, "PASS");
    assert_eq!(report.mode, "NANL-Q4");
    assert!(report.canonical_roundtrip_hash_pass);
    assert!(report.constraints_pass);
    assert!(report.ecc_enabled);
    assert!(report.ecc_verified);
    assert!(report.gc_min_observed >= 0.40);
    assert!(report.gc_max_observed <= 0.60);
    assert!(report.homopolymer_max_observed <= 3);
    assert!(q4_symbols_to_string(packet.symbols())
        .chars()
        .all(|base| { matches!(base, 'A' | 'C' | 'G' | 'T') }));
}

#[test]
fn nanl_q4_constraints_reject_homopolymer_and_gc_failures() {
    let symbols = unpack_q4_symbols(&[0u8; 16], 64).unwrap();
    let report = validate_constraints(&symbols, Q4ConstraintProfile::default()).unwrap();

    assert!(!report.constraints_pass);
    assert_eq!(report.homopolymer_max_observed, 64);
    assert_eq!(report.gc_min_observed, 0.0);
}

#[test]
fn nanl_q4_parity_tamper_fails_closed() {
    let packet = encode_payload(b"parity protected canonical q4 payload").unwrap();
    let mut bytes = packet.bytes().to_vec();
    let payload_start = 160;
    bytes[payload_start] ^= 0b0000_0001;
    let q4_hash = blake3::hash(&bytes[payload_start..]);
    bytes[64..96].copy_from_slice(q4_hash.as_bytes());

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("ECC/parity verification failed"));
}

#[test]
fn nanl_q4_invalid_constrained_codeword_fails_closed() {
    let packet = encode_payload(b"valid q4 packet").unwrap();
    let mut bytes = packet.bytes().to_vec();
    let payload_start = 160;
    let mut symbols = unpack_q4_symbols(&bytes[payload_start..], packet.symbols().len()).unwrap();
    symbols[0] = symbols[1];
    let packed = pack_q4_symbols(&symbols).unwrap();
    bytes.truncate(payload_start);
    bytes.extend_from_slice(&packed);
    let q4_hash = blake3::hash(&bytes[payload_start..]);
    bytes[64..96].copy_from_slice(q4_hash.as_bytes());
    let mut xor = 0u8;
    let mut sum = 0u32;
    for symbol in symbols {
        let value = symbol as u8;
        xor ^= value;
        sum = (sum + value as u32) % 65_535;
    }
    bytes[128] = xor;
    bytes[130..132].copy_from_slice(&(sum as u16).to_le_bytes());

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(
        err.contains("PC_ERR_HASH") || err.contains("PC_ERR_CONSTRAINT"),
        "{err}"
    );
}
