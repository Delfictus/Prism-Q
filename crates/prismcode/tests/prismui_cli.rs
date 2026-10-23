use prismcode::binary_runtime::{compile_reference_json, execute_packet, Backend};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn reference_json() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/reference/reference.json")
}

#[test]
fn prismui_cli_appends_blocked_operational_dashboard_to_stdout_and_report() {
    let tmp = tempfile::tempdir().unwrap();
    let pcode = tmp.path().join("reference.pcode");
    let pcout = tmp.path().join("reference.cpu.pcout");
    let report = tmp.path().join("reports/operator_view.txt");
    let status_dir = tmp.path().join("build/status");
    fs::create_dir_all(&status_dir).unwrap();
    fs::write(
        status_dir.join("prismcode_status.json"),
        r#"{
  "status_schema_version": "PRISMCODE_OPERATIONALIZATION_STATUS_V0_1",
  "lanes": [
    {"key":"L3_VALIDATION","status":"PASS","evidence":"build/demo/reference.pcode","blockers":[]},
    {"key":"L5_DEMO_ENVELOPE","status":"PASS","evidence":"build/secure-release/reference.encrypted.pcode.security.json","blockers":[]},
    {"key":"CUDA_LINKAGE","status":"FAIL-CLOSED","evidence":"FAIL_CLOSED_UNLINKED","blockers":["CUDA_RUNTIME_LINKED"]},
    {"key":"CUDA_KERNEL_TRUST","status":"FAIL-CLOSED","evidence":"UNBOUND","blockers":["CUDA_KERNEL_TRUSTED"]},
    {"key":"MODEL_RUNTIME","status":"BLOCKED","evidence":"WEIGHTS_INSTALLED_RUNTIME_UNBOUND_SMOKE_REQUIRED","blockers":["MODEL_RUNTIME_SMOKE_PASSED"]},
    {"key":"MARKET_READY_CORE_ISA","status":"BLOCKED","evidence":"cargo xtask market-ready-check --mode CORE_ISA","blockers":["OPENBAO_OR_VAULT_BOUND","TRUSTED_ATTESTATION","MARKET_READY_TRUE"]},
    {"key":"MARKET_READY_AI_ASSISTED_OS","status":"BLOCKED","evidence":"cargo xtask market-ready-check --mode AI_ASSISTED_OS","blockers":["OPENBAO_OR_VAULT_BOUND","TRUSTED_ATTESTATION","MARKET_READY_TRUE","MODEL_RUNTIME_ACTIVE","MODEL_RUNTIME_SMOKE_PASSED"]}
  ],
  "release_modes": {
    "CORE_ISA": {"model_runtime_required": false, "cuda_required": false, "blockers": ["OPENBAO_OR_VAULT_BOUND","TRUSTED_ATTESTATION","MARKET_READY_TRUE"]},
    "AI_ASSISTED_OS": {"model_runtime_required": true, "cuda_required": false, "blockers": ["OPENBAO_OR_VAULT_BOUND","TRUSTED_ATTESTATION","MARKET_READY_TRUE","MODEL_RUNTIME_ACTIVE","MODEL_RUNTIME_SMOKE_PASSED"]},
    "GPU_ACCELERATED_CORE": {"model_runtime_required": false, "cuda_required": true, "blockers": ["OPENBAO_OR_VAULT_BOUND","TRUSTED_ATTESTATION","MARKET_READY_TRUE","CUDA_RUNTIME_LINKED","CUDA_KERNEL_TRUSTED"]},
    "SECURE_PRODUCTION": {"model_runtime_required": false, "cuda_required": false, "blockers": ["OPENBAO_OR_VAULT_BOUND","TRUSTED_ATTESTATION","MARKET_READY_TRUE"]}
  }
}"#,
    )
    .unwrap();

    compile_reference_json(&reference_json(), &pcode).unwrap();
    execute_packet(&pcode, Backend::Cpu, &pcout).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_prismui"))
        .arg(&pcout)
        .arg("--report")
        .arg(&report)
        .current_dir(tmp.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "prismui failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("Status source: build/status/prismcode_status.json"));
    assert!(stdout.contains("Overall readiness: BLOCKED"));
    assert!(!stdout.contains("Overall readiness: PASS"));
    assert!(stdout.contains("L3 validation: PASS blockers=none"));
    assert!(stdout.contains("L5 demo envelope: PASS blockers=none"));
    assert!(stdout.contains("CUDA linkage: FAIL-CLOSED blockers=CUDA_RUNTIME_LINKED"));
    assert!(stdout.contains("Model runtime: BLOCKED blockers=MODEL_RUNTIME_SMOKE_PASSED"));
    assert!(
        stdout.contains("CORE_ISA: OPENBAO_OR_VAULT_BOUND, TRUSTED_ATTESTATION, MARKET_READY_TRUE")
    );
    assert!(stdout.contains(
        "cargo run -p prismcode --bin prismsec -- model-runtime-smoke --out build/demo/model_runtime_smoke_report.json"
    ));
    assert_eq!(fs::read_to_string(report).unwrap(), stdout);
}
