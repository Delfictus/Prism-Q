use prismcode::binary_runtime::{validate_packet, ValidationLevel};
use prismcode::cse_hd::{compile_scientific_state, write_cse_hd_sidecars, NanlMode};
use std::path::{Path, PathBuf};

const FIXTURES: &[&str] = &[
    "reference.pcsrc.json",
    "protein_water_variant.pcsrc.json",
    "nrf_event_frames.pcsrc.json",
    "drf_graph.pcsrc.json",
];

#[test]
fn cse_hd_fixture_compile_gates_are_bounded_and_validated() {
    let out_dir = build_test_dir("cse_hd_bench_gates/compile");

    for fixture in FIXTURES {
        let input = corpus_path(fixture);
        let output = out_dir.join(fixture.replace(".pcsrc.json", ".pcode"));
        let report = compile_scientific_state(&input, "hd", NanlMode::Both, &output).unwrap();

        assert_eq!(report.status, "PASS", "{fixture}");
        assert!(report.raw_json_size > 0, "{fixture}");
        assert!(report.canonical_table_size > 0, "{fixture}");
        assert!(report.nanl_bin_size > 0, "{fixture}");
        assert!(report.nanl_q4_size > report.nanl_bin_size, "{fixture}");
        assert!(report.nrf_frame_size > 0, "{fixture}");
        assert!(report.drf_state_size > 0, "{fixture}");
        assert!(report.encode_mb_s > 0.0, "{fixture}");
        assert!(report.roundtrip_canonical_hash_pass, "{fixture}");
        assert!(report.raw_json_size > report.nanl_bin_size, "{fixture}");
        assert_eq!(report.canonical_hash.len(), 64, "{fixture}");
        assert_eq!(report.nanl_bin_hash_blake3.len(), 64, "{fixture}");
        assert_eq!(
            report.nanl_q4_hash_blake3.as_ref().unwrap().len(),
            64,
            "{fixture}"
        );

        let pcode_size = std::fs::metadata(&output).unwrap().len();
        assert!(pcode_size > report.raw_json_size, "{fixture}");
        assert!(
            pcode_size < report.raw_json_size + 32 * 1024,
            "{fixture} pcode grew unexpectedly: raw={} pcode={}",
            report.raw_json_size,
            pcode_size
        );

        let validation = validate_packet(&output, ValidationLevel::L3).unwrap();
        assert_eq!(validation.status, "PASS", "{fixture}");
        assert!(validation.errors.is_empty(), "{fixture}");
        assert!(validation.instruction_count >= 22, "{fixture}");
        assert!(validation.nanl_count > 0, "{fixture}");
    }
}

#[test]
fn cse_hd_sidecar_gate_writes_all_real_artifacts_under_build() {
    let out_dir = build_test_dir("cse_hd_bench_gates/sidecars");
    let report = write_cse_hd_sidecars(&corpus_path("reference.pcsrc.json"), &out_dir).unwrap();

    assert_eq!(report.status, "PASS");
    assert_eq!(report.canonical_hash.len(), 64);
    assert_eq!(report.artifacts.len(), 5);

    for artifact in &report.artifacts {
        let path = PathBuf::from(&artifact.path);
        assert!(
            path.starts_with(repo_root().join("build")),
            "sidecar path escaped build symlink: {}",
            artifact.path
        );
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(bytes.len() as u64, artifact.bytes);
        assert_eq!(artifact.hash_blake3.len(), 64);
        assert_eq!(artifact.status, "PASS");

        match artifact.artifact_kind.as_str() {
            "CSE_CANONICAL_TABLES_ARROW_COMPATIBLE_BINARY" => {
                assert!(bytes.starts_with(b"PRISM_CSEHD_TABLES_ARROW_COMPAT_V0_1\0"));
            }
            "NANL_BIN_EXECUTABLE_STREAM" => assert!(bytes.starts_with(b"NANLBIN\0")),
            "NANL_Q4_CONSTRAINED_STREAM" => assert!(bytes.starts_with(b"NANLQ4\0\0")),
            "NRF_SPARSE_EVENT_FRAMES" => {
                assert!(bytes.starts_with(b"PRISM_NRF_FRAME_SET_V0_1\0"));
            }
            "DRF_RESONANCE_GRAPH" => {
                assert!(bytes.starts_with(b"PRISM_DRF_GRAPH_SET_V0_1\0"));
            }
            other => panic!("unexpected sidecar kind {other}"),
        }
    }
}

fn corpus_path(name: &str) -> PathBuf {
    repo_root().join("examples/corpus/cse_hd").join(name)
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
