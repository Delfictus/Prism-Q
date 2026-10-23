use prismcode::nanl_bin::{encode_tokens, sample_tokens};
use prismcode::nanl_dna::{simulate_archival_export, write_nanl_demo_artifacts};
use prismcode::nanl_q4::encode_payload;

#[test]
fn nanl_dna_simulator_exports_q4_without_physical_storage_claim() {
    let bin = encode_tokens(&sample_tokens()).unwrap();
    let q4 = encode_payload(bin.bytes()).unwrap();
    let report = simulate_archival_export(q4.bytes()).unwrap();

    assert_eq!(report.status, "PASS");
    assert!(report.simulator_passed);
    assert!(report.decode_roundtrip_hash_pass);
    assert!(report.ecc_parity_verified);
    assert!(report.gc_constraints_pass);
    assert!(report.homopolymer_constraints_pass);
    assert!(!report.physical_workflow_passed);
    assert!(!report.physical_dna_storage_claimed);
    assert_eq!(report.claim_level, "SIMULATED_ARCHIVAL_EXPORT_ONLY");
    assert!(!report.oligos.is_empty());
    for oligo in &report.oligos {
        assert!(oligo.gc_content >= 0.40 && oligo.gc_content <= 0.60);
        assert!(oligo.homopolymer_max <= 3);
        assert!(oligo.forbidden_motif_flags.is_empty());
        assert_eq!(
            oligo.payload_sequence.len(),
            oligo.payload_q4_span_len as usize
        );
    }
}

#[test]
fn nanl_dna_rejects_tampered_q4_packet() {
    let bin = encode_tokens(&sample_tokens()).unwrap();
    let q4 = encode_payload(bin.bytes()).unwrap();
    let mut bytes = q4.bytes().to_vec();
    let last = bytes.len() - 1;
    bytes[last] ^= 0b01;

    let err = simulate_archival_export(&bytes).unwrap_err().to_string();
    assert!(err.contains("PC_ERR_HASH"));
}

#[test]
fn nanl_demo_writes_artifacts_under_requested_build_path() {
    let tmp = tempfile::tempdir().unwrap();
    let out_dir = tmp.path().join("build").join("demo").join("nanl");
    let report = write_nanl_demo_artifacts(&out_dir).unwrap();

    assert_eq!(report.status, "PASS");
    for artifact in report.artifacts {
        assert!(std::path::Path::new(&artifact).exists(), "{artifact}");
    }
}
