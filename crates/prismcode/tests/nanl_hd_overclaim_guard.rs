use prismcode::nanl_hd_overclaim_guard::{
    validate_compression_claim, validate_drf_lossless_claim, validate_h16_not_quaternary,
    validate_lossy_canonical_state, validate_nanl_dna_claim, validate_q4_bits_per_symbol,
};

#[test]
fn overclaim_guard_rejects_known_bad_claims() {
    assert!(validate_q4_bits_per_symbol(4.0).is_err());
    assert!(validate_h16_not_quaternary(true).is_err());
    assert!(validate_nanl_dna_claim(true, false).is_err());
    assert!(validate_drf_lossless_claim(true, true, false).is_err());
    assert!(validate_compression_claim(1, Some(10.0), 10.0).is_err());
    assert!(validate_compression_claim(10, Some(2.0), 10.0).is_err());
    assert!(validate_lossy_canonical_state(true, true).is_err());
}

#[test]
fn overclaim_guard_accepts_bounded_claims() {
    validate_q4_bits_per_symbol(2.0).unwrap();
    validate_h16_not_quaternary(false).unwrap();
    validate_nanl_dna_claim(false, false).unwrap();
    validate_drf_lossless_claim(true, false, false).unwrap();
    validate_compression_claim(10, Some(2.0), 2.0).unwrap();
    validate_lossy_canonical_state(true, false).unwrap();
}
