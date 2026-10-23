use prismcode::nanl_hd_bench::{run_nanl_hd_benchmark, write_nanl_hd_artifacts};

#[test]
fn nanl_hd_acceptance_generates_artifacts_and_roundtrips() {
    let temp = tempfile::tempdir().unwrap();
    let payload = b"PRISM NANL-HD acceptance payload with repeated repeated motif motif data";
    let report = write_nanl_hd_artifacts(temp.path(), payload).unwrap();

    assert_eq!(report.status, "PASS");
    assert!(report.roundtrip_hash_pass);
    for artifact in [
        "nanl_codec_stack.json",
        "nanl_dict.nanld",
        "nanl_motif.nanlm",
        "nanl_delta.nanldelta",
        "nanl_var.nanlvar",
        "nanl_ans.nanlans",
        "nanl_h16.nanlh16",
        "nanl_bin.nanlb",
        "nanl_q4.nanlq4",
        "nanl_dna_sim.json",
        "nrf_sparse.nrf",
        "drf_active_state.drf",
    ] {
        assert!(temp.path().join(artifact).metadata().unwrap().len() > 0);
    }
}

#[test]
fn nanl_hd_benchmark_rejects_toy_sample_policy() {
    let err = run_nanl_hd_benchmark(b"payload", 1)
        .unwrap_err()
        .to_string();
    assert!(err.contains("sample_count"));
}
