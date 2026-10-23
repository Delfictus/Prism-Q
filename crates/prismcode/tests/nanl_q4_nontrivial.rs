use prismcode::nanl_q4::{
    decode_packet, encode_payload, pack_q4_symbols, validate_packet, validate_q4_claim, Q4Symbol,
    NANL_Q4_FLAG_CONSTRAINT_PROFILE_PRESENT, NANL_Q4_FLAG_ECC_PRESENT, NANL_Q4_FLAG_MERKLE_PRESENT,
    NANL_Q4_FLAG_NONTRIVIAL_CODEBOOK, NANL_Q4_FLAG_SYMBOL_MODEL_PRESENT, NANL_Q4_HEADER_LEN,
    NANL_Q4_MODEL_KIND_OFFSET, NANL_Q4_PROOF_FLAGS_OFFSET, NANL_Q4_RAW_BITS_PER_SYMBOL_X2_OFFSET,
    NANL_Q4_REQUIRED_PROOF_FLAGS, NANL_Q4_SYMBOL_MODEL_PRISM_NATIVE_V0,
};

#[test]
fn q4_nontrivial_packet_binds_model_profile_ecc_merkle_and_roundtrip() {
    let payload =
        b"typed PRISM canonical payload: molecule protein water time event uncertainty thermo";
    let packet = encode_payload(payload).unwrap();
    let decoded = decode_packet(packet.bytes()).unwrap();
    let report = validate_packet(packet.bytes()).unwrap();

    assert_eq!(decoded, payload);
    assert_eq!(report.status, "PASS");
    assert_eq!(report.mode, "NANL-Q4");
    assert_eq!(report.q4_raw_bits_per_symbol_upper_bound, "2.0");
    assert_eq!(report.proof_flags, NANL_Q4_REQUIRED_PROOF_FLAGS);
    assert_eq!(report.constraint_profile_id, "NANL_Q4_SAFE_V0");
    assert_eq!(
        report.symbol_model_id,
        "NANL_Q4_PRISM_NATIVE_SYMBOL_MODEL_V0"
    );
    assert_eq!(
        packet.bytes()[NANL_Q4_MODEL_KIND_OFFSET],
        NANL_Q4_SYMBOL_MODEL_PRISM_NATIVE_V0
    );
    assert!(report.nontrivial_codec_proof);
    assert!(report.simple_binary_mapper_rejected);
    assert!(report.constraints_pass);
    assert!(report.ecc_enabled);
    assert!(report.ecc_verified);
    assert!(report.merkle_verified);
    assert!(report.canonical_roundtrip_hash_pass);
    assert_ne!(report.symbol_model_hash_blake3, "0".repeat(64));
    assert_ne!(report.constraint_profile_hash_blake3, "0".repeat(64));
    assert_ne!(report.proof_manifest_hash_blake3, "0".repeat(64));
    assert_eq!(packet.symbols().len(), payload.len() * 8);
}

#[test]
fn simple_bits_to_acgt_mapper_is_rejected() {
    let payload = b"simple mapper must fail";
    let packet = encode_payload(payload).unwrap();
    let mut fake = packet.bytes().to_vec();
    let fake_symbols = simple_bits_to_q4_symbols(payload);
    let fake_payload = pack_q4_symbols(&fake_symbols).unwrap();

    fake[20..28].copy_from_slice(&(fake_symbols.len() as u64).to_le_bytes());
    fake.truncate(NANL_Q4_HEADER_LEN);
    fake.extend_from_slice(&fake_payload);
    update_stream_hash_and_parity(&mut fake, &fake_symbols);

    let err = validate_packet(&fake).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_FORMAT"), "{err}");
    assert!(err.contains("constrained code rate"), "{err}");
}

#[test]
fn missing_required_profile_model_ecc_merkle_or_codebook_flags_fail_closed() {
    let packet = encode_payload(b"proof flags are release critical").unwrap();
    for (flag, label) in [
        (
            NANL_Q4_FLAG_CONSTRAINT_PROFILE_PRESENT,
            "constraint profile",
        ),
        (NANL_Q4_FLAG_SYMBOL_MODEL_PRESENT, "symbol model"),
        (NANL_Q4_FLAG_ECC_PRESENT, "ECC/parity"),
        (NANL_Q4_FLAG_MERKLE_PRESENT, "Merkle"),
        (NANL_Q4_FLAG_NONTRIVIAL_CODEBOOK, "nontrivial"),
    ] {
        let mut tampered = packet.bytes().to_vec();
        tampered[NANL_Q4_PROOF_FLAGS_OFFSET] &= !flag;
        let err = validate_packet(&tampered).unwrap_err().to_string();
        assert!(err.contains("PC_ERR_FORMAT"), "{label}: {err}");
    }
}

#[test]
fn missing_or_bad_hash_evidence_fails_closed() {
    let packet = encode_payload(b"hash-bound Q4 payload").unwrap();

    let mut bad_canonical_hash = packet.bytes().to_vec();
    bad_canonical_hash[32..64].fill(0);
    let err = validate_packet(&bad_canonical_hash)
        .unwrap_err()
        .to_string();
    assert!(err.contains("canonical roundtrip hash mismatch"), "{err}");

    let mut bad_stream_hash = packet.bytes().to_vec();
    bad_stream_hash[64..96].fill(0);
    let err = validate_packet(&bad_stream_hash).unwrap_err().to_string();
    assert!(err.contains("stream hash mismatch"), "{err}");

    let mut bad_merkle = packet.bytes().to_vec();
    bad_merkle[96..128].fill(0);
    let err = validate_packet(&bad_merkle).unwrap_err().to_string();
    assert!(err.contains("Merkle root mismatch"), "{err}");

    let mut bad_proof = packet.bytes().to_vec();
    bad_proof[132..160].fill(0);
    let err = validate_packet(&bad_proof).unwrap_err().to_string();
    assert!(err.contains("proof manifest hash mismatch"), "{err}");
}

#[test]
fn gc_and_homopolymer_violations_fail_closed() {
    let packet = encode_payload(b"constraint failure target").unwrap();
    let mut tampered = packet.bytes().to_vec();
    let symbol_count = (read_u64(&tampered[20..28]) as usize).max(64);
    let bad_symbols = vec![Q4Symbol::A; symbol_count];
    let packed = pack_q4_symbols(&bad_symbols).unwrap();
    tampered.truncate(NANL_Q4_HEADER_LEN);
    tampered.extend_from_slice(&packed);
    update_stream_hash_and_parity(&mut tampered, &bad_symbols);

    let err = validate_packet(&tampered).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_CONSTRAINT"), "{err}");
    assert!(
        err.contains("constraint profile validation failed"),
        "{err}"
    );
}

#[test]
fn roundtrip_mismatch_fails_closed() {
    let packet = encode_payload(b"roundtrip mismatch target").unwrap();
    let mut tampered = packet.bytes().to_vec();
    tampered[32] ^= 0x80;

    let err = validate_packet(&tampered).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_HASH"), "{err}");
    assert!(err.contains("canonical roundtrip hash mismatch"), "{err}");
}

#[test]
fn q4_rejects_four_bits_per_symbol_claims() {
    let packet = encode_payload(b"Q4 remains a four-symbol channel").unwrap();
    let mut tampered = packet.bytes().to_vec();
    tampered[NANL_Q4_RAW_BITS_PER_SYMBOL_X2_OFFSET] = 8;

    let err = validate_packet(&tampered).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_OVERCLAIM"), "{err}");
    assert!(
        err.contains("cannot claim more than 2 raw bits per symbol"),
        "{err}"
    );

    let err = validate_q4_claim(4.0).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_OVERCLAIM"), "{err}");
}

fn simple_bits_to_q4_symbols(payload: &[u8]) -> Vec<Q4Symbol> {
    let mut out = Vec::with_capacity(payload.len() * 4);
    for byte in payload {
        for digit in [
            byte >> 6,
            (byte >> 4) & 0b11,
            (byte >> 2) & 0b11,
            byte & 0b11,
        ] {
            out.push(Q4Symbol::try_from(digit).unwrap());
        }
    }
    out
}

fn update_stream_hash_and_parity(packet: &mut [u8], symbols: &[Q4Symbol]) {
    let q4_hash = blake3::hash(&packet[NANL_Q4_HEADER_LEN..]);
    packet[64..96].copy_from_slice(q4_hash.as_bytes());

    let mut xor = 0u8;
    let mut sum = 0u32;
    for symbol in symbols {
        let value = *symbol as u8;
        xor ^= value;
        sum = (sum + value as u32) % 65_535;
    }
    packet[128] = xor;
    packet[130..132].copy_from_slice(&(sum as u16).to_le_bytes());
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}
