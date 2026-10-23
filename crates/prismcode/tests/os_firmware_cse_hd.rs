use prismcode::binary_runtime::{execute_packet, Backend};
use prismcode::cse_hd::{compile_scientific_state, NanlMode};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn cse_hd_reference() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/corpus/cse_hd/reference.pcsrc.json")
}

#[test]
fn prismui_exposes_cse_hd_firmware_os_science_panel() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("cse_hd_reference.pcode");
    let pcout = tmp.path().join("cse_hd_reference.cpu.pcout");
    let report = tmp.path().join("operator/cse_hd_report.txt");
    let missing_status = tmp.path().join("status/prismcode_status.json");

    let compile_report =
        compile_scientific_state(&cse_hd_reference(), "hd", NanlMode::Both, &pcode).unwrap();
    assert_eq!(compile_report.status, "PASS");
    assert!(compile_report.roundtrip_canonical_hash_pass);
    execute_packet(&pcode, Backend::Cpu, &pcout).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_prismui"))
        .arg(&pcout)
        .arg("--report")
        .arg(&report)
        .arg("--status")
        .arg(&missing_status)
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "prismui failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    for needle in [
        "CSE/HD Scientific Capability Panel",
        "CSE/HD: PASS schema=PRISM-CSE/HD ABI v0.1 state=cse_hd_reference_v0_1",
        "artifact_stage: EXECUTED_PCOUT",
        "NRF frames: 1",
        "DRF states: 1",
        "NANL: symbols=",
        "capabilities: typed_molecular_state, canonical_timebase, event_stream",
        "nrf_event_frames",
        "drf_resonance_fabric",
        "firmware_os_boot_capability: PRESENT",
    ] {
        assert!(stdout.contains(needle), "missing {needle}\n{stdout}");
    }
    assert_eq!(fs::read_to_string(report).unwrap(), stdout);
}
