use prismcode::drf_active_state::{reject_derived_lossless_raw_claim, DrfActiveStateClass};
use prismcode::nanl_h16::validate_h16_claim;
use prismcode::nanl_q4::validate_q4_claim;
use prismcode::nanl_var::{decode_u64, encode_u64};

#[test]
fn nanl_hd_negative_controls_reject_overclaims_and_malformed_var() {
    assert!(validate_q4_claim(4.0).is_err());
    assert!(validate_h16_claim(true, false).is_err());
    assert!(reject_derived_lossless_raw_claim(DrfActiveStateClass::DrfDerived, true).is_err());

    let mut packet = encode_u64(&[1, 2, 3, 300]).unwrap();
    packet.encoded.push(0x80);
    assert!(decode_u64(&packet).is_err());
}
