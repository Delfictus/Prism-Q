use prismcode::compression_registry::{
    hash_bytes, CompressionCodecKind, HashPair, RuntimeCodecRegistry, REQUIRED_CODEC_KINDS,
};
use std::collections::BTreeSet;

fn bytes_for_kind(kind: CompressionCodecKind) -> Vec<u8> {
    match kind {
        CompressionCodecKind::RawUncompressed => {
            b"PRISMCODE raw executable bytes\0with stable canonical state".to_vec()
        }
        CompressionCodecKind::NanlPackedU2 => vec![0, 1, 2, 3, 3, 2, 1, 0, 1],
        CompressionCodecKind::SectionZstd => b"SECTION_ZSTD canonical payload ".repeat(64),
        CompressionCodecKind::DeltaTime => time_records(&[7, 11, 19, 19, 42]),
        CompressionCodecKind::EventRle => event_records(&[1, 1, 1, 2, 2, 5, 5, 5]),
        CompressionCodecKind::EventDictionary => event_records(&[5, 1, 5, 2, 2, 1, 5]),
        CompressionCodecKind::TensorChunked => (0..257).map(|idx| (idx % 251) as u8).collect(),
    }
}

fn time_records(ids: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * 80);
    for (row, id) in ids.iter().enumerate() {
        let mut record = [0u8; 80];
        record[0..8].copy_from_slice(&id.to_le_bytes());
        for (idx, byte) in record[8..].iter_mut().enumerate() {
            *byte = (row as u8).wrapping_mul(17).wrapping_add(idx as u8);
        }
        out.extend_from_slice(&record);
    }
    out
}

fn event_records(kinds: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(kinds.len() * 96);
    for (row, kind) in kinds.iter().enumerate() {
        let mut record = [0u8; 96];
        record[0..8].copy_from_slice(&(row as u64).to_le_bytes());
        record[8..10].copy_from_slice(&kind.to_le_bytes());
        for (idx, byte) in record[10..].iter_mut().enumerate() {
            *byte = (row as u8).wrapping_mul(31).wrapping_add(idx as u8);
        }
        out.extend_from_slice(&record);
    }
    out
}

#[test]
fn builtin_registry_contains_required_runtime_codecs() {
    let registry = RuntimeCodecRegistry::with_builtin_codecs();
    registry.validate_required_codecs().unwrap();

    let registered = registry
        .records()
        .into_iter()
        .map(|record| record.codec_kind)
        .collect::<BTreeSet<_>>();
    for kind in REQUIRED_CODEC_KINDS {
        assert!(registered.contains(&kind), "missing codec kind {kind}");
    }
}

#[test]
fn required_codecs_roundtrip_and_validate_dual_hashes() {
    let registry = RuntimeCodecRegistry::with_builtin_codecs();

    for kind in REQUIRED_CODEC_KINDS {
        let canonical = bytes_for_kind(kind);
        let roundtrip = registry.validate_roundtrip(kind, &canonical).unwrap();
        assert_eq!(roundtrip.codec_kind, kind);
        assert_eq!(roundtrip.roundtrip_verified, true);
        assert_eq!(roundtrip.compressed_hash_verified, true);
        assert_eq!(roundtrip.canonical_hash_verified, true);

        let encoded = registry.encode_section(kind, &canonical).unwrap();
        let validated = registry
            .validate_encoded_by_id(encoded.codec_id, &encoded.compressed_bytes, encoded.hashes)
            .unwrap();
        assert_eq!(validated.codec_kind, kind);
        assert_eq!(
            validated.hash_compressed_bytes,
            encoded.hashes.compressed_hex()
        );
        assert_eq!(
            validated.hash_canonical_uncompressed_bytes,
            encoded.hashes.canonical_uncompressed_hex()
        );
    }
}

#[test]
fn compressed_hash_mismatch_fails_closed() {
    let registry = RuntimeCodecRegistry::with_builtin_codecs();
    let canonical = bytes_for_kind(CompressionCodecKind::SectionZstd);
    let encoded = registry
        .encode_section(CompressionCodecKind::SectionZstd, &canonical)
        .unwrap();
    let mut wrong_hashes = encoded.hashes;
    wrong_hashes.hash_compressed_bytes[0] ^= 0x80;

    let err = registry
        .validate_encoded_by_id(encoded.codec_id, &encoded.compressed_bytes, wrong_hashes)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("compressed hash mismatch"));
}

#[test]
fn canonical_hash_mismatch_fails_closed_after_decode() {
    let registry = RuntimeCodecRegistry::with_builtin_codecs();
    let canonical = bytes_for_kind(CompressionCodecKind::RawUncompressed);
    let encoded = registry
        .encode_section(CompressionCodecKind::RawUncompressed, &canonical)
        .unwrap();
    let wrong_hashes = HashPair {
        hash_compressed_bytes: encoded.hashes.hash_compressed_bytes,
        hash_canonical_uncompressed_bytes: hash_bytes(b"not the canonical state"),
    };

    let err = registry
        .validate_encoded_by_kind(
            CompressionCodecKind::RawUncompressed,
            &encoded.compressed_bytes,
            wrong_hashes,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("canonical uncompressed hash mismatch"));
}

#[test]
fn malformed_tensor_chunk_hash_fails_closed_even_with_matching_outer_hash() {
    let registry = RuntimeCodecRegistry::with_builtin_codecs();
    let canonical = bytes_for_kind(CompressionCodecKind::TensorChunked);
    let encoded = registry
        .encode_section(CompressionCodecKind::TensorChunked, &canonical)
        .unwrap();
    let mut corrupted = encoded.compressed_bytes.clone();
    let payload_byte = 16 + 4 + 4 + 32;
    corrupted[payload_byte] ^= 0x55;
    let corrupted_hashes = HashPair {
        hash_compressed_bytes: hash_bytes(&corrupted),
        hash_canonical_uncompressed_bytes: encoded.hashes.hash_canonical_uncompressed_bytes,
    };

    let err = registry
        .validate_encoded_by_id(encoded.codec_id, &corrupted, corrupted_hashes)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_HASH"));
    assert!(err.contains("TENSOR_CHUNKED chunk hash mismatch"));
}

#[test]
fn malformed_nanl_padding_fails_closed_even_with_matching_outer_hash() {
    let registry = RuntimeCodecRegistry::with_builtin_codecs();
    let canonical = bytes_for_kind(CompressionCodecKind::NanlPackedU2);
    let encoded = registry
        .encode_section(CompressionCodecKind::NanlPackedU2, &canonical)
        .unwrap();
    let mut corrupted = encoded.compressed_bytes.clone();
    let last = corrupted.len() - 1;
    corrupted[last] |= 0b1100_0000;
    let corrupted_hashes = HashPair {
        hash_compressed_bytes: hash_bytes(&corrupted),
        hash_canonical_uncompressed_bytes: encoded.hashes.hash_canonical_uncompressed_bytes,
    };

    let err = registry
        .validate_encoded_by_kind(
            CompressionCodecKind::NanlPackedU2,
            &corrupted,
            corrupted_hashes,
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_FORMAT"));
    assert!(err.contains("unused high bits"));
}
