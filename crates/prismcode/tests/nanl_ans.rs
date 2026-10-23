#[path = "../src/nanl_ans.rs"]
mod nanl_ans;

use nanl_ans::{
    build_model, decode_packet, decode_packet_streaming, encode_tokens, encode_tokens_with_config,
    sample_tokens, validate_packet, AnsSymbolClass, AnsToken, NanlAnsConfig,
    NANL_ANS_CHUNK_ENTRY_LEN, NANL_ANS_HEADER_LEN, NANL_ANS_MODEL_ENTRY_LEN,
};

#[test]
fn nanl_ans_roundtrips_typed_prism_symbols_exactly() {
    let tokens = sample_tokens();
    let packet = encode_tokens(&tokens).unwrap();
    let decoded = decode_packet(packet.bytes()).unwrap();
    let report = validate_packet(packet.bytes()).unwrap();

    assert_eq!(decoded, tokens);
    assert_eq!(report.status, "PASS");
    assert_eq!(report.mode, "NANL-ANS");
    assert_eq!(report.ans_kind, "rANS");
    assert_eq!(packet.model().ans_kind, "rANS");
    assert!(!packet.clone().into_bytes().is_empty());
    assert_eq!(report.state_bits, 32);
    assert_eq!(report.scale_bits, 12);
    assert_eq!(report.normalization_total, 4096);
    assert!(report.model_count > 1);
    assert!(report.canonical_roundtrip_hash_pass);
    assert!(report.streaming_decode_supported);
    assert!(report.deterministic);
    assert_eq!(report.byte_order, "little");
    assert!(report.corruption_detection.contains("BLAKE3"));
}

#[test]
fn nanl_ans_model_is_frozen_normalized_and_deterministic() {
    let tokens = sample_tokens();
    let model_a = build_model(&tokens).unwrap();
    let model_b = build_model(&tokens).unwrap();

    assert!(model_a.frozen);
    assert_eq!(model_a.model_hash_blake3, model_b.model_hash_blake3);
    assert_eq!(model_a.entries, model_b.entries);
    assert_eq!(model_a.normalization_total, 4096);
    assert_eq!(
        model_a
            .entries
            .iter()
            .map(|entry| u32::from(entry.frequency))
            .sum::<u32>(),
        model_a.normalization_total
    );
    assert!(model_a.entries.iter().all(|entry| entry.frequency >= 1));
    assert!(model_a.entries.windows(2).all(|pair| {
        let left = (pair[0].symbol_class as u16, pair[0].symbol_id);
        let right = (pair[1].symbol_class as u16, pair[1].symbol_id);
        left < right
            && u32::from(pair[0].cumulative) + u32::from(pair[0].frequency)
                == u32::from(pair[1].cumulative)
    }));
}

#[test]
fn nanl_ans_streaming_decode_uses_chunk_checkpoints() {
    let tokens = sample_tokens();
    let packet =
        encode_tokens_with_config(&tokens, NanlAnsConfig::with_chunk_symbol_limit(17)).unwrap();

    assert!(packet.checkpoints().len() > 4);
    assert_eq!(
        packet.report().chunk_checkpoint_count,
        packet.checkpoints().len() as u32
    );

    let mut streamed = Vec::new();
    let mut chunk_count = 0usize;
    let report = decode_packet_streaming(packet.bytes(), |chunk| {
        chunk_count += 1;
        assert!(chunk.len() <= 17);
        streamed.extend_from_slice(chunk);
        Ok(())
    })
    .unwrap();

    assert_eq!(streamed, tokens);
    assert_eq!(chunk_count, packet.checkpoints().len());
    assert_eq!(report.status, "PASS");
    assert!(report.bounded_memory_bytes > 0);
}

#[test]
fn nanl_ans_supports_required_context_symbol_classes() {
    use AnsSymbolClass as C;
    let tokens = vec![
        AnsToken::new(C::EventKind, 1),
        AnsToken::new(C::ResidueMotif, 2),
        AnsToken::new(C::WaterGateMotif, 3),
        AnsToken::new(C::DrfChannel, 4),
        AnsToken::new(C::GFlowNetAction, 5),
        AnsToken::new(C::TeacherLabel, 6),
        AnsToken::new(C::TimeDelta, 7),
        AnsToken::new(C::VariantKind, 8),
        AnsToken::new(C::ThermoQuantity, 9),
        AnsToken::new(C::UncertaintyModel, 10),
    ];
    let mut repeated = Vec::new();
    for _ in 0..12 {
        repeated.extend_from_slice(&tokens);
    }
    let packet = encode_tokens(&repeated).unwrap();
    let decoded = decode_packet(packet.bytes()).unwrap();

    assert_eq!(decoded, repeated);
}

#[test]
fn nanl_ans_tampered_stream_hash_fails_closed() {
    let packet = encode_tokens(&sample_tokens()).unwrap();
    let mut bytes = packet.bytes().to_vec();
    let last = bytes.len() - 1;
    bytes[last] ^= 0x55;

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("stream hash mismatch"));
}

#[test]
fn nanl_ans_truncated_packet_fails_closed() {
    let packet = encode_tokens(&sample_tokens()).unwrap();
    let mut bytes = packet.bytes().to_vec();
    bytes.pop();

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_FORMAT"));
    assert!(err.contains("packet length mismatch"));
}

#[test]
fn nanl_ans_zero_frequency_model_entry_fails_even_with_matching_model_hash() {
    let packet = encode_tokens(&sample_tokens()).unwrap();
    let mut bytes = packet.bytes().to_vec();
    let first_entry = NANL_ANS_HEADER_LEN;
    bytes[first_entry + 4..first_entry + 6].copy_from_slice(&0u16.to_le_bytes());
    let symbol_id = read_u16(&bytes, first_entry);
    let class = read_u16(&bytes, first_entry + 2);
    let cumulative = read_u16(&bytes, first_entry + 6);
    let raw_count = read_u64(&bytes, first_entry + 8);
    let prefix = model_entry_prefix(class, symbol_id, 0, cumulative, raw_count);
    bytes[first_entry + 16..first_entry + 24].copy_from_slice(&prefix.to_le_bytes());
    rewrite_model_hash(&mut bytes);

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_MODEL"));
    assert!(err.contains("zero frequency"));
}

#[test]
fn nanl_ans_bad_model_entry_hash_prefix_fails_closed() {
    let packet = encode_tokens(&sample_tokens()).unwrap();
    let mut bytes = packet.bytes().to_vec();
    let first_entry = NANL_ANS_HEADER_LEN;
    bytes[first_entry + 16] ^= 0x01;
    rewrite_model_hash(&mut bytes);

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("model entry hash prefix mismatch"));
}

#[test]
fn nanl_ans_checkpoint_final_state_mismatch_fails_closed() {
    let packet = encode_tokens(&sample_tokens()).unwrap();
    let mut bytes = packet.bytes().to_vec();
    let model_count = read_u32(&bytes, 24) as usize;
    let chunk_table = NANL_ANS_HEADER_LEN + model_count * NANL_ANS_MODEL_ENTRY_LEN;
    let first_checkpoint_final_state = chunk_table + 36;
    bytes[first_checkpoint_final_state] ^= 0x01;

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("checkpoint final state mismatch"));
}

#[test]
fn nanl_ans_rejects_unsupported_normalization_and_endian_ambiguity() {
    let packet = encode_tokens(&sample_tokens()).unwrap();
    let mut bytes = packet.bytes().to_vec();
    bytes[20..24].copy_from_slice(&2048u32.to_le_bytes());

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_MODEL"));
    assert!(err.contains("normalization total"));
}

#[test]
fn nanl_ans_rejects_empty_and_unbounded_inputs() {
    let err = encode_tokens(&[]).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_MODEL"));
    assert!(err.contains("empty"));

    let err =
        encode_tokens_with_config(&sample_tokens(), NanlAnsConfig::with_chunk_symbol_limit(0))
            .unwrap_err()
            .to_string();
    assert!(err.contains("PC_ERR_BOUND"));
}

#[test]
fn nanl_ans_bad_cumulative_table_fails_with_matching_model_hash() {
    let packet = encode_tokens(&sample_tokens()).unwrap();
    let mut bytes = packet.bytes().to_vec();
    let second_entry = NANL_ANS_HEADER_LEN + NANL_ANS_MODEL_ENTRY_LEN;
    let bad_cumulative = read_u16(&bytes, second_entry + 6).wrapping_add(1);
    bytes[second_entry + 6..second_entry + 8].copy_from_slice(&bad_cumulative.to_le_bytes());
    let symbol_id = read_u16(&bytes, second_entry);
    let class = read_u16(&bytes, second_entry + 2);
    let frequency = read_u16(&bytes, second_entry + 4);
    let raw_count = read_u64(&bytes, second_entry + 8);
    let prefix = model_entry_prefix(class, symbol_id, frequency, bad_cumulative, raw_count);
    bytes[second_entry + 16..second_entry + 24].copy_from_slice(&prefix.to_le_bytes());
    rewrite_model_hash(&mut bytes);

    let err = validate_packet(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_MODEL"));
    assert!(err.contains("cumulative table"));
}

#[test]
fn nanl_ans_chunk_table_entry_size_is_stable() {
    assert_eq!(NANL_ANS_CHUNK_ENTRY_LEN, 72);
    assert_eq!(NANL_ANS_MODEL_ENTRY_LEN, 24);
}

fn rewrite_model_hash(bytes: &mut [u8]) {
    let model_count = read_u32(bytes, 24) as usize;
    let model_start = NANL_ANS_HEADER_LEN;
    let model_end = model_start + model_count * NANL_ANS_MODEL_ENTRY_LEN;
    let model_hash = blake3::hash(&bytes[model_start..model_end]);
    bytes[128..160].copy_from_slice(model_hash.as_bytes());
}

fn model_entry_prefix(
    class: u16,
    symbol_id: u16,
    frequency: u16,
    cumulative: u16,
    raw_count: u64,
) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"PRISMCODE:NANL_ANS:MODEL_ENTRY:V0_1");
    hasher.update(&class.to_le_bytes());
    hasher.update(&symbol_id.to_le_bytes());
    hasher.update(&frequency.to_le_bytes());
    hasher.update(&cumulative.to_le_bytes());
    hasher.update(&raw_count.to_le_bytes());
    let hash = hasher.finalize();
    u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap())
}

fn read_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap())
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn read_u64(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
