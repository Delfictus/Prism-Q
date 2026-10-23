use prismcode::binary_runtime::{validate_packet as validate_pcode_packet, ValidationLevel};
use prismcode::cse_hd::{
    compile_cse_hd_from_json_str, compile_scientific_state, nanl_tokens_from_tables, NanlMode,
};
use prismcode::{nanl_bin, nanl_q4};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const REFERENCE: &str = include_str!("../../../examples/corpus/cse_hd/reference.pcsrc.json");

#[test]
fn cse_hd_tables_tokenize_into_nanl_bin_and_q4_without_json_hot_path() {
    let compilation = compile_cse_hd_from_json_str(REFERENCE).unwrap();
    let tables = &compilation.tables;
    let tokens = nanl_tokens_from_tables(tables);

    let expected_token_count = tables.entity_rows.len()
        + tables.atom_rows.len()
        + tables.residue_rows.len()
        + tables.water_site_rows.len()
        + (tables.event_rows.len() * 2)
        + tables.thermo_rows.len()
        + tables.uncertainty_rows.len()
        + tables.teacher_output_rows.len()
        + tables.gflownet_trace_rows.len()
        + tables.nrf_frame_rows.len()
        + tables.drf_state_rows.len();
    assert_eq!(tokens.len(), expected_token_count);

    let counts = class_counts(&tokens);
    assert_eq!(counts.get("ENTITY_TOKEN"), Some(&3));
    assert_eq!(counts.get("ATOM_TOKEN"), Some(&3));
    assert_eq!(counts.get("RESIDUE_TOKEN"), Some(&2));
    assert_eq!(counts.get("WATER_TOKEN"), Some(&1));
    assert_eq!(counts.get("EVENT_KIND_TOKEN"), Some(&2));
    assert_eq!(counts.get("EVENT_MOTIF_TOKEN"), Some(&2));
    assert_eq!(counts.get("THERMO_TOKEN"), Some(&1));
    assert_eq!(counts.get("UNCERTAINTY_TOKEN"), Some(&1));
    assert_eq!(counts.get("TEACHER_EVIDENCE_TOKEN"), Some(&1));
    assert_eq!(counts.get("GFLOWNET_ACTION_TOKEN"), Some(&1));
    assert_eq!(counts.get("NRF_FRAME_TOKEN"), Some(&1));
    assert_eq!(counts.get("DRF_CHANNEL_TOKEN"), Some(&1));

    let nanl_bin_packet = nanl_bin::encode_tokens(&tokens).unwrap();
    assert_eq!(nanl_bin_packet.report().status, "PASS");
    assert_eq!(nanl_bin_packet.report().mode, "NANL-BIN");
    assert_eq!(nanl_bin_packet.report().token_count, tokens.len() as u64);
    assert!(nanl_bin_packet.report().roundtrip_canonical_hash_pass);
    assert!(!nanl_bin_packet.report().debug_json_hot_path_used);
    let bin_report = nanl_bin::validate_packet(nanl_bin_packet.bytes()).unwrap();
    assert_eq!(bin_report.token_count, tokens.len() as u64);
    assert!(bin_report.roundtrip_canonical_hash_pass);

    let q4_packet = nanl_q4::encode_payload(nanl_bin_packet.bytes()).unwrap();
    let q4_report = nanl_q4::validate_packet(q4_packet.bytes()).unwrap();
    assert_eq!(q4_report.status, "PASS");
    assert!(q4_report.constraints_pass);
    assert!(q4_report.ecc_verified);
    assert!(q4_report.canonical_roundtrip_hash_pass);
}

#[test]
fn compile_scientific_state_emits_valid_cse_hd_pcode_packet() {
    let out = build_test_dir("cse_scientific_tokenization").join("reference.pcode");
    let expected_compilation = compile_cse_hd_from_json_str(REFERENCE).unwrap();
    let report = compile_scientific_state(
        &corpus_path("reference.pcsrc.json"),
        "hd",
        NanlMode::Both,
        &out,
    )
    .unwrap();

    assert_eq!(report.status, "PASS");
    assert_eq!(report.cse_mode, "hd");
    assert_eq!(report.nanl_mode, NanlMode::Both);
    assert_eq!(report.canonical_hash, expected_compilation.canonical_hash);
    assert!(report.roundtrip_canonical_hash_pass);
    assert!(report.raw_json_size > report.nanl_bin_size);
    assert!(report.nanl_q4_size > report.nanl_bin_size);
    assert!(out.exists());

    let validation = validate_pcode_packet(&out, ValidationLevel::L3).unwrap();
    assert_eq!(validation.status, "PASS");
    assert_eq!(validation.instruction_count, 24);
    assert_eq!(validation.state_count, 3);
    assert_eq!(validation.time_count, 1);
    assert_eq!(validation.event_count, 2);
    assert_eq!(validation.nanl_count, report.nanl_bin_size);
    assert!(validation.errors.is_empty());
}

fn class_counts(tokens: &[nanl_bin::NanlToken]) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for token in tokens {
        *counts.entry(token.class.as_str()).or_insert(0) += 1;
    }
    counts
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
