use prismcode::binary_runtime::{
    disassemble_packet, parse_packet, validate_packet, ValidationLevel,
};
use prismcode::cse_hd::{compile_scientific_state, NanlMode};
use std::path::{Path, PathBuf};

const EXPECTED_PCASM: &str =
    include_str!("../../../examples/corpus/cse_hd/reference_expected.pcasm");

#[test]
fn cse_hd_reference_packet_disassembly_matches_expected_pcasm() {
    let out_dir = build_test_dir("cse_opcodes");
    let pcode = out_dir.join("reference.pcode");
    let pcasm = out_dir.join("reference.pcasm");

    let report = compile_scientific_state(
        &corpus_path("reference.pcsrc.json"),
        "hd",
        NanlMode::Both,
        &pcode,
    )
    .unwrap();
    assert_eq!(report.status, "PASS");

    let validation = validate_packet(&pcode, ValidationLevel::L3).unwrap();
    assert_eq!(validation.status, "PASS");
    assert_eq!(validation.instruction_count, 24);
    assert_eq!(validation.nanl_count, 206);

    let packet = parse_packet(std::fs::read(&pcode).unwrap()).unwrap();
    assert_eq!(packet.instructions.len(), 24);
    assert_eq!(packet.sections.len(), 14);

    let actual = disassemble_packet(&pcode, Some(&pcasm)).unwrap();
    assert_eq!(actual.trim_end(), EXPECTED_PCASM.trim_end());
    assert_eq!(
        std::fs::read_to_string(&pcasm).unwrap().trim_end(),
        EXPECTED_PCASM.trim_end()
    );
    assert!(actual.contains("NANL.PACK"));
    assert!(actual.contains("DRF.MAP"));
    assert!(actual.contains("DRF.FIRE"));
    assert!(actual.contains("DRF.RESONATE"));
    assert!(actual.contains("THERMO.ENERGY"));
    assert!(actual.contains("ASSERT.DETERMINISTIC"));
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
