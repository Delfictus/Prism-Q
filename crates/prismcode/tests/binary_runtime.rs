use prismcode::binary_runtime::{
    benchmark_suite, benchmark_suite_with_corpus, compile_reference_json, compression_audit,
    disassemble_packet, execute_packet, export_debug_json, init_file_keyring, parse_packet,
    redact_export, seal_sidecar, security_inspect, ui_report, validate_packet, verify_replay,
    write_dev_encrypted_packet, write_prod_encrypted_packet, Backend, SectionKind, ValidationLevel,
    PCODE_MAGIC, PSK_ENCRYPTED_MAGIC, PSK_PROD_MAGIC,
};
use std::fs;
use std::path::PathBuf;

fn reference_json() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/reference/reference.json")
}

#[test]
fn compile_validate_execute_replay_reference_packet() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let pcout = tmp.path().join("reference.cpu.pcout");
    let pcasm = tmp.path().join("reference.pcasm");
    let debug_json = tmp.path().join("reference.debug.json");
    let replay_report = tmp.path().join("replay_report.json");

    compile_reference_json(&reference_json(), &pcode).unwrap();
    let validation = validate_packet(&pcode, ValidationLevel::L3).unwrap();
    assert_eq!(validation.status, "PASS");
    assert!(validation.instruction_count >= 20);
    assert_eq!(validation.state_count, 3);
    assert_eq!(validation.time_count, 1);
    assert_eq!(validation.event_count, 4);

    let disasm = disassemble_packet(&pcode, Some(&pcasm)).unwrap();
    assert!(disasm.contains("ABI.DECLARE"));
    assert!(disasm.contains("DRF.FIRE"));
    assert!(pcasm.exists());

    let execution = execute_packet(&pcode, Backend::Cpu, &pcout).unwrap();
    assert_eq!(execution.status, "PASS");
    assert_eq!(execution.backend, "cpu");
    assert!(pcout.exists());

    let replay = verify_replay(&pcout, Some(&replay_report)).unwrap();
    assert_eq!(replay.status, "PASS");
    assert_eq!(replay.input_packet_hash, execution.input_packet_hash);
    assert_eq!(replay.output_state_hash, execution.output_state_hash);

    export_debug_json(&pcode, &debug_json).unwrap();
    let exported: serde_json::Value =
        serde_json::from_slice(&fs::read(debug_json).unwrap()).unwrap();
    assert_eq!(exported["synthetic"], true);
}

#[test]
fn packet_parser_rejects_corrupted_instruction_opcode() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    compile_reference_json(&reference_json(), &pcode).unwrap();
    let mut bytes = fs::read(&pcode).unwrap();
    let packet = parse_packet(bytes.clone()).unwrap();
    let instr_entry = packet
        .sections
        .iter()
        .find(|entry| entry.section_type == SectionKind::InstrStream)
        .unwrap();
    let instr_start = instr_entry.offset as usize;
    bytes[instr_start] = 0xff;
    bytes[instr_start + 1] = 0xff;
    assert!(parse_packet(bytes).is_err());
}

#[test]
fn explicit_cuda_backend_does_not_silently_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let pcout = tmp.path().join("reference.cuda.pcout");
    compile_reference_json(&reference_json(), &pcode).unwrap();
    let result = execute_packet(&pcode, Backend::Cuda, &pcout);
    if result.is_err() {
        assert!(!pcout.exists());
    }
}

#[test]
fn benchmark_suite_covers_required_workloads_and_malformed_rejection() {
    let tmp = tempfile::tempdir().unwrap();
    let report = benchmark_suite(Backend::Cpu, tmp.path()).unwrap();
    assert_eq!(report.status, "PASS");
    assert_eq!(report.workload_count, 9);
    assert_eq!(report.passed_workloads, 7);
    assert_eq!(report.rejected_workloads, 1);
    assert_eq!(report.security_rejected_workloads, 1);
    assert_eq!(report.failed_workloads, 0);
    assert_eq!(report.isa_version, "0.1");
    assert!(report.total_instruction_count > 0);
    assert!(report.compression_ratio_vs_json > 0.0);
    assert!(report.latency_percentiles_ns.p50_ns.is_some());
    assert_eq!(
        report.required_workload_coverage["all_required_present"].as_bool(),
        Some(true)
    );

    let ids: std::collections::BTreeSet<_> = report
        .workloads
        .iter()
        .map(|workload| workload.workload_id.as_str())
        .collect();
    for required in [
        "tiny_reference",
        "small_synthetic_nanl",
        "small_synthetic_drf",
        "medium_event_stream",
        "medium_tensor",
        "compression_heavy",
        "replay_heavy",
        "security_negative_unsealed",
        "malformed_rejection",
    ] {
        assert!(
            ids.contains(required),
            "missing benchmark workload {required}"
        );
    }

    let malformed = report
        .workloads
        .iter()
        .find(|workload| workload.workload_id == "malformed_rejection")
        .unwrap();
    assert_eq!(malformed.status, "REJECTED");
    assert!(malformed.error_class.is_some());

    let security_negative = report
        .workloads
        .iter()
        .find(|workload| workload.workload_id == "security_negative_unsealed")
        .unwrap();
    assert_eq!(security_negative.status, "SECURITY_REJECTED");
    assert_eq!(
        security_negative.error_class.as_deref(),
        Some("PC_ERR_SECURITY")
    );
    assert!(security_negative.packet_hash.is_some());
    assert_eq!(security_negative.isa_version.as_deref(), Some("0.1"));
}

#[test]
fn benchmark_suite_accepts_examples_corpus_directory() {
    let tmp = tempfile::tempdir().unwrap();
    let corpus_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/corpus");
    let report = benchmark_suite_with_corpus(Backend::Cpu, tmp.path(), &corpus_dir).unwrap();
    assert_eq!(report.status, "PASS");
    assert_eq!(report.workload_count, 5);
    assert_eq!(report.passed_workloads, 3);
    assert_eq!(report.rejected_workloads, 1);
    assert_eq!(report.security_rejected_workloads, 1);
    assert_eq!(report.failed_workloads, 0);
    assert_eq!(
        report.required_workload_coverage["all_required_present"].as_bool(),
        Some(true)
    );
    assert!(report
        .workloads
        .iter()
        .all(|workload| workload.isa_version.as_deref() == Some("0.1")));
}

#[test]
fn l5_security_validation_fails_closed_until_psk_bindings_exist() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    compile_reference_json(&reference_json(), &pcode).unwrap();

    let result = validate_packet(&pcode, ValidationLevel::L5);
    let err = result.expect_err("L5 must not pass without production PSK bindings");
    let rendered = err.to_string();
    assert!(rendered.contains("L5 security validation requires PRISM Shield Kernel"));
    assert!(rendered.contains("production envelope"));
}

#[test]
fn psk_dev_tools_report_unready_instead_of_overclaiming() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let pcout = tmp.path().join("reference.cpu.pcout");
    let pseal = tmp.path().join("reference.cpu.pcout.pseal");
    let redacted = tmp.path().join("redacted_public_report.md");

    compile_reference_json(&reference_json(), &pcode).unwrap();
    execute_packet(&pcode, Backend::Cpu, &pcout).unwrap();

    let inspection = security_inspect(&pcout).unwrap();
    assert_eq!(inspection["classification"], "SYNTHETIC");
    assert_eq!(inspection["production_security_ready"], false);
    assert_eq!(inspection["encrypted"], false);
    assert_eq!(inspection["signed"], false);
    assert_eq!(inspection["vault_status"], "PRISM_VAULT_UNBOUND");

    let seal = seal_sidecar(&pcout, &pseal).unwrap();
    assert_eq!(seal["signature_status"], "UNSIGNED_DEV");
    assert_eq!(seal["production_security_ready"], false);
    assert!(pseal.exists());

    let report = redact_export(&pcout, "PUBLIC", &redacted).unwrap();
    assert!(report.contains("PUBLIC_SAFE_HASH_ONLY"));
    assert!(fs::read_to_string(redacted)
        .unwrap()
        .contains("production_security_ready: false"));
}

#[test]
fn psk_encrypt_dev_writes_real_encrypted_envelope_not_plaintext_copy() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let encrypted = tmp.path().join("reference.encrypted.pcode");

    compile_reference_json(&reference_json(), &pcode).unwrap();
    let plaintext = fs::read(&pcode).unwrap();
    let report = write_dev_encrypted_packet(&pcode, &encrypted).unwrap();
    let encrypted_bytes = fs::read(&encrypted).unwrap();

    assert_eq!(
        &encrypted_bytes[..PSK_ENCRYPTED_MAGIC.len()],
        PSK_ENCRYPTED_MAGIC
    );
    assert_ne!(&encrypted_bytes[..PCODE_MAGIC.len()], PCODE_MAGIC);
    assert_ne!(encrypted_bytes, plaintext);
    assert_eq!(report["encrypted"], true);
    assert_eq!(report["production_security_ready"], false);

    let inspection = security_inspect(&encrypted).unwrap();
    assert_eq!(inspection["artifact_kind"], "encrypted_pcode_envelope");
    assert_eq!(inspection["encrypted"], true);
    assert_eq!(
        inspection["seal_status"],
        "ENCRYPTED_ENVELOPE_HASH_VERIFIED"
    );
}

#[test]
fn psk_prod_envelope_passes_l5_with_file_keyring_signature() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let keyring = tmp.path().join("keyring/prismcode_file_keyring.json");
    let public_key = tmp.path().join("keyring/release_public_key.ed25519.hex");
    let encrypted = tmp.path().join("reference.encrypted.pcode");

    compile_reference_json(&reference_json(), &pcode).unwrap();
    init_file_keyring(&keyring, &public_key).unwrap();
    let report = write_prod_encrypted_packet(&pcode, &encrypted, &keyring, &public_key).unwrap();
    let encrypted_bytes = fs::read(&encrypted).unwrap();

    assert_eq!(&encrypted_bytes[..PSK_PROD_MAGIC.len()], PSK_PROD_MAGIC);
    assert_eq!(report["encrypted"], true);
    assert_eq!(report["signature_status"], "SIGNED");
    assert_eq!(report["production_security_ready"], true);

    let inspection = security_inspect(&encrypted).unwrap();
    assert_eq!(
        inspection["artifact_kind"],
        "production_encrypted_pcode_envelope"
    );
    assert_eq!(inspection["signed"], true);
    assert_eq!(inspection["production_security_ready"], true);
    assert_eq!(
        inspection["seal_status"],
        "PRODUCTION_ENVELOPE_SIGNATURE_VERIFIED"
    );

    let l5 = validate_packet(&encrypted, ValidationLevel::L5).unwrap();
    assert_eq!(l5.status, "PASS");
    assert_eq!(l5.level, "L5");
    assert!(l5.instruction_count >= 20);
    assert_eq!(l5.event_count, 4);
}

#[test]
fn compression_audit_verifies_required_runtime_codecs() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let report_path = tmp.path().join("compression_report.json");
    compile_reference_json(&reference_json(), &pcode).unwrap();

    let report = compression_audit(&pcode, Some(&report_path)).unwrap();
    assert_eq!(report.status, "PASS");
    assert!(report_path.exists());
    let codecs = report
        .codecs
        .iter()
        .filter_map(|row| row.get("codec").and_then(|value| value.as_str()))
        .collect::<std::collections::BTreeSet<_>>();
    for required in [
        "RAW_UNCOMPRESSED",
        "NANL_PACKED_U2",
        "SECTION_ZSTD",
        "DELTA_TIME",
        "EVENT_RLE",
        "EVENT_DICTIONARY",
        "TENSOR_CHUNKED",
    ] {
        assert!(codecs.contains(required), "missing codec {required}");
    }
    assert!(report
        .codecs
        .iter()
        .all(|row| row["roundtrip_verified"].as_bool() == Some(true)));
}

#[test]
fn ui_report_shows_blocked_security_and_cuda_truth() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let pcout = tmp.path().join("reference.cpu.pcout");
    compile_reference_json(&reference_json(), &pcode).unwrap();
    execute_packet(&pcode, Backend::Cpu, &pcout).unwrap();

    let text = ui_report(&pcout, None).unwrap();
    assert!(text.contains("Runtime: CPU executable"));
    assert!(text.contains("Replay: PASS"));
    assert!(text.contains("Market readiness: BLOCKED"));
    assert!(text.contains("Model runtime: INACTIVE"));
    assert!(text.contains("Model airlock:"));
    assert!(text.contains("CUDA:"));
}
