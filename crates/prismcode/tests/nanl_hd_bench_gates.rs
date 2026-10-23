use prismcode::nanl_hd_bench::{
    load_corpus_manifest, validate_benchmark_report, validate_corpus_manifest,
    validate_overclaim_gate, write_benchmark_report, NanlHdClaim, REQUIRED_BASELINES,
    REQUIRED_MALFORMED_WORKLOAD_IDS, REQUIRED_WORKLOAD_IDS,
};
use serde_json::json;
use std::path::{Path, PathBuf};

#[test]
fn nanl_hd_corpus_manifest_covers_required_workloads_and_baselines() {
    let manifest = load_corpus_manifest(&corpus_root()).unwrap();
    validate_corpus_manifest(&manifest).unwrap();

    let ids = manifest
        .workloads
        .iter()
        .map(|workload| workload.workload_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for required in REQUIRED_WORKLOAD_IDS {
        assert!(ids.contains(required), "missing workload {required}");
    }

    let malformed = manifest
        .workloads
        .iter()
        .filter(|workload| workload.expected_malformed)
        .map(|workload| workload.workload_id.as_str())
        .collect::<std::collections::BTreeSet<_>>();
    for required in REQUIRED_MALFORMED_WORKLOAD_IDS {
        assert!(malformed.contains(required), "missing malformed {required}");
    }

    for baseline in REQUIRED_BASELINES {
        assert!(
            manifest
                .required_baselines
                .iter()
                .any(|item| item == baseline),
            "missing baseline {baseline}"
        );
    }
}

#[test]
fn nanl_hd_benchmark_report_gate_accepts_required_fields_and_sample_stats() {
    let value = json!({
        "reports": [{
            "workload": "bounded_report_contract",
            "sample_count": 10,
            "raw_source_bytes": 1000,
            "canonical_cse_bytes": 700,
            "plain_prismcode_bytes": 1200,
            "zstd_bytes": 400,
            "nanl_bin_bytes": 500,
            "nanl_q4_bytes": 900,
            "nanl_dict_bytes": 450,
            "nanl_motif_bytes": 430,
            "nanl_delta_bytes": 410,
            "nanl_var_bytes": 390,
            "nanl_ans_bytes": 360,
            "nanl_h16_bytes": 1000,
            "nrf_sparse_bytes": 300,
            "drf_active_state_bytes": 260,
            "combined_stack_bytes": 260,
            "encode_MBps": {
                "sample_count": 10,
                "mean": 10.0,
                "median": 10.0,
                "p95": 10.0,
                "stdev": 0.0,
                "min": 10.0,
                "max": 10.0
            },
            "decode_MBps": {
                "sample_count": 10,
                "mean": 11.0,
                "median": 11.0,
                "p95": 11.0,
                "stdev": 0.0,
                "min": 11.0,
                "max": 11.0
            },
            "cpu_runtime_rows_sec": 1.0,
            "events_sec": 1.0,
            "instructions_sec": 1.0
        }],
        "malformed_rejections": 5,
        "baseline_coverage": {"missing_baselines": []}
    });

    validate_benchmark_report(&value).unwrap();
}

#[test]
fn nanl_hd_payload_benchmark_writer_executes_and_records_metrics() {
    let out = build_test_dir("nanl_hd_bench_gates/payload").join("nanl_hd_payload_report.json");
    let payload =
        b"PRISM NANL HD benchmark payload with repeated motif motif motif and delta delta delta";
    let report = write_benchmark_report(&out, payload).unwrap();

    assert!(out.exists());
    assert_eq!(report.status, "PASS");
    assert_eq!(report.sample_count, 10);
    assert!(report.raw_source_bytes > 0);
    assert!(report.nanl_bin_bytes > 0);
    assert!(report.nanl_q4_bytes > 0);
    assert!(report.nanl_ans_bytes > 0);
    assert!(report.roundtrip_hash_pass);
    assert!(report.compression_ratio_vs_raw >= 0.0);
}

#[test]
fn nanl_hd_bench_rejects_toy_sample_count_one() {
    let manifest = json!({
        "schema_version": "PRISM_NANL_HD_CORPUS_MANIFEST_V0_1",
        "min_sample_count": 1,
        "required_baselines": REQUIRED_BASELINES,
        "workloads": [{
            "workload_id": "tiny_cse_state",
            "source_path": "../cse_hd/tiny_cse_state.pcsrc.json",
            "workload_class": "positive_tiny",
            "expected_malformed": false,
            "negative_family": null,
            "sample_count": 1,
            "performance_claim_scope": "performance",
            "required_baselines": REQUIRED_BASELINES
        }]
    });
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("corpus_manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = load_corpus_manifest(tmp.path())
        .and_then(|manifest| validate_corpus_manifest(&manifest))
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_BENCH_SAMPLE_POLICY"), "{err}");
}

#[test]
fn nanl_hd_bench_rejects_missing_malformed_corpora() {
    let manifest = json!({
        "schema_version": "PRISM_NANL_HD_CORPUS_MANIFEST_V0_1",
        "min_sample_count": 10,
        "required_baselines": REQUIRED_BASELINES,
        "workloads": [{
            "workload_id": "tiny_cse_state",
            "source_path": "../cse_hd/tiny_cse_state.pcsrc.json",
            "workload_class": "positive_tiny",
            "expected_malformed": false,
            "negative_family": null,
            "sample_count": 10,
            "performance_claim_scope": "bounded_h6_benchmark",
            "required_baselines": REQUIRED_BASELINES
        }]
    });
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(
        tmp.path().join("corpus_manifest.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();

    let err = load_corpus_manifest(tmp.path())
        .and_then(|manifest| validate_corpus_manifest(&manifest))
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_BENCH_CORPUS_MISSING"), "{err}");
}

#[test]
fn nanl_hd_bench_rejects_missing_size_or_throughput_breakdown() {
    let missing_size = json!({
        "reports": [{
            "workload": "bad",
            "sample_count": 10,
            "raw_source_bytes": 0,
            "encode_MBps": {"sample_count": 10},
            "decode_MBps": {"sample_count": 10},
            "cpu_runtime_rows_sec": 1.0,
            "events_sec": 1.0,
            "instructions_sec": 1.0
        }],
        "malformed_rejections": 5,
        "baseline_coverage": {"missing_baselines": []}
    });
    let err = validate_benchmark_report(&missing_size)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_BENCH_SIZE_BREAKDOWN_MISSING"), "{err}");

    let missing_throughput = json!({
        "reports": [{
            "workload": "bad",
            "sample_count": 10,
            "raw_source_bytes": 1,
            "canonical_cse_bytes": 1,
            "plain_prismcode_bytes": 1,
            "zstd_bytes": 1,
            "nanl_bin_bytes": 1,
            "nanl_q4_bytes": 1,
            "nanl_dict_bytes": 1,
            "nanl_motif_bytes": 1,
            "nanl_delta_bytes": 1,
            "nanl_var_bytes": 1,
            "nanl_ans_bytes": 1,
            "nanl_h16_bytes": 1,
            "nrf_sparse_bytes": 1,
            "drf_active_state_bytes": 1,
            "combined_stack_bytes": 1,
            "decode_MBps": {"sample_count": 10},
            "cpu_runtime_rows_sec": 1.0,
            "events_sec": 1.0,
            "instructions_sec": 1.0
        }],
        "malformed_rejections": 5,
        "baseline_coverage": {"missing_baselines": []}
    });
    let err = validate_benchmark_report(&missing_throughput)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_BENCH_THROUGHPUT_MISSING"), "{err}");
}

#[test]
fn nanl_hd_bench_rejects_overclaiming_ratios_without_evidence() {
    let no_evidence = NanlHdClaim {
        claim_id: "claim_no_evidence".to_string(),
        claim_text: "NANL-HD gives orders of magnitude compression".to_string(),
        sample_count: 10,
        measured_ratio: Some(12.0),
        claimed_ratio: Some(12.0),
        evidence_report_path: None,
    };
    let err = validate_overclaim_gate(&no_evidence)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_NANL_HD_OVERCLAIM"), "{err}");

    let overstated = NanlHdClaim {
        claim_id: "claim_overstated".to_string(),
        claim_text: "NANL-HD measured compression ratio".to_string(),
        sample_count: 10,
        measured_ratio: Some(3.0),
        claimed_ratio: Some(30.0),
        evidence_report_path: Some(
            "build/benchmarks/nanl_high_density_codec_report.json".to_string(),
        ),
    };
    let err = validate_overclaim_gate(&overstated)
        .unwrap_err()
        .to_string();
    assert!(err.contains("exceeds measured"), "{err}");

    let q4_overclaim = NanlHdClaim {
        claim_id: "claim_q4_overclaim".to_string(),
        claim_text: "Q4 stores 4 bits per symbol".to_string(),
        sample_count: 10,
        measured_ratio: None,
        claimed_ratio: None,
        evidence_report_path: None,
    };
    let err = validate_overclaim_gate(&q4_overclaim)
        .unwrap_err()
        .to_string();
    assert!(err.contains("Q4 cannot claim 4 bits"), "{err}");
}

fn corpus_root() -> PathBuf {
    repo_root().join("examples/corpus/nanl_hd")
}

fn build_test_dir(name: &str) -> PathBuf {
    let path = repo_root().join("build/test").join(name);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}
