use prismcode::model_governance::{
    inspect_model_registry, smoke_model_runtime, verify_model_airlock,
};
use serde_json::json;
use std::fs;

fn write_registry(tmp: &tempfile::TempDir) -> std::path::PathBuf {
    let model_dir = tmp.path().join("openai_gpt-oss-20b");
    fs::create_dir_all(&model_dir).unwrap();
    let config = model_dir.join("config.json");
    let tokenizer = model_dir.join("tokenizer.json");
    let weights = model_dir.join("model-00000-of-00001.safetensors");
    fs::write(&config, b"{\"model\":\"test\"}").unwrap();
    fs::write(&tokenizer, b"{\"tokenizer\":\"test\"}").unwrap();
    fs::write(&weights, b"test-weights").unwrap();

    let registry = json!({
        "registry_version": "TEST_MODEL_GOVERNANCE",
        "rules": {
            "canonical_state_mutation_allowed": false
        },
        "installed_models": [{
            "model_id": "openai/gpt-oss-20b",
            "install_allowed": true,
            "hardware_gated": false,
            "local_path": model_dir,
            "weights_hash_verified": true,
            "serving_status": "WEIGHTS_INSTALLED_RUNTIME_UNBOUND",
            "runtime_binding_status": "VLLM_OR_TRANSFORMERS_NOT_INSTALLED",
            "runtime_smoke_status": "NOT_RUN",
            "airlock_status": "MODEL_GENERATED_UNVERIFIED_BY_DEFAULT",
            "canonical_state_mutation_allowed": false,
            "files": [
                {
                    "path": "model-00000-of-00001.safetensors",
                    "bytes": 12,
                    "blake3": blake3::hash(b"test-weights").to_hex().to_string()
                },
                {
                    "path": "config.json",
                    "bytes": 16,
                    "blake3": blake3::hash(b"{\"model\":\"test\"}").to_hex().to_string()
                },
                {
                    "path": "tokenizer.json",
                    "bytes": 20,
                    "blake3": blake3::hash(b"{\"tokenizer\":\"test\"}").to_hex().to_string()
                }
            ]
        }],
        "hardware_gated_models": [{
            "model_id": "Qwen/Qwen3-Coder-Next",
            "hardware_gated": true,
            "install_allowed": false,
            "install_gate": "QUANTIZED_OR_OFFLOADED_SERVING_PLAN_REQUIRED",
            "status": "HARDWARE_GATED_INSTALL_DISALLOWED",
            "canonical_state_mutation_allowed": false
        }]
    });
    let path = tmp.path().join("registry.json");
    fs::write(&path, serde_json::to_vec_pretty(&registry).unwrap()).unwrap();
    path
}

#[test]
fn model_inspect_verifies_local_gpt_oss_hashes_and_qwen_gate() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = write_registry(&tmp);

    let report = inspect_model_registry(&registry).unwrap();

    assert_eq!(report.status, "PASS");
    assert_eq!(report.canonical_state_mutation_allowed, false);
    let gpt = report.gpt_oss_20b.unwrap();
    assert_eq!(gpt.installed_hash_verified, true);
    assert_eq!(gpt.model_runtime_active, false);
    assert_eq!(
        gpt.model_serving_gate_status,
        "WEIGHTS_INSTALLED_RUNTIME_UNBOUND_SMOKE_REQUIRED"
    );
    assert!(gpt
        .required_runtime_evidence_missing
        .contains(&"runtime_smoke_status=PASSED".to_string()));
    assert_eq!(gpt.file_checks.len(), 3);
    assert!(gpt.file_checks.iter().all(|check| check.verified));
    let qwen = report.qwen3_coder_next.unwrap();
    assert_eq!(qwen.hardware_gated, true);
    assert_eq!(qwen.install_allowed, false);
    assert_eq!(qwen.canonical_state_mutation_allowed, false);
}

#[test]
fn model_inspect_rejects_claimed_active_runtime_without_smoke_evidence() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = write_registry(&tmp);
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(&registry).unwrap()).unwrap();
    let model = value
        .get_mut("installed_models")
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|rows| rows.first_mut())
        .unwrap();
    model["serving_status"] = json!("RUNTIME_BOUND_SMOKE_PASSED");
    model["runtime_binding_allowed"] = json!(true);
    model["runtime_bound"] = json!(true);
    model["runtime_smoke_status"] = json!("NOT_RUN");
    fs::write(&registry, serde_json::to_vec_pretty(&value).unwrap()).unwrap();

    let report = inspect_model_registry(&registry).unwrap();

    assert_eq!(report.status, "FAIL");
    let gpt = report.gpt_oss_20b.unwrap();
    assert_eq!(gpt.model_runtime_active, false);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.contains("claims active serving")));
}

#[test]
fn model_runtime_smoke_fails_closed_without_compatible_runtime_and_blocks_ai_mode() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = write_registry(&tmp);

    let report = smoke_model_runtime(&registry).unwrap();

    assert_eq!(report.status, "FAIL_CLOSED");
    assert_eq!(report.state, "FAIL_CLOSED_NO_COMPATIBLE_GOVERNED_RUNTIME");
    assert_eq!(report.model_runtime_active, false);
    assert_eq!(
        report.release_mode_gates.core_isa_model_gate,
        "ALLOWED_ADVISORY_RUNTIME_INACTIVE"
    );
    assert_eq!(
        report.release_mode_gates.ai_assisted_os_model_gate,
        "BLOCKED_MODEL_RUNTIME_SMOKE_REQUIRED"
    );
    assert!(report
        .release_mode_gates
        .ai_assisted_os_blockers
        .contains(&"MODEL_RUNTIME_ACTIVE".to_string()));
    assert!(report
        .release_mode_gates
        .ai_assisted_os_blockers
        .contains(&"MODEL_RUNTIME_SMOKE_PASSED".to_string()));
    assert_eq!(report.canonical_mutation_denial_check.status, "PASS");
    assert_eq!(report.canonical_mutation_denial_check.denied, true);
    assert_eq!(
        report
            .canonical_mutation_denial_check
            .probe_canonical_state_mutation_allowed,
        true
    );
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.contains("no compatible governed runtime")));
}

#[test]
fn model_airlock_denies_unaccepted_canonical_mutation() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = write_registry(&tmp);
    let artifact = tmp.path().join("artifact.json");
    fs::write(
        &artifact,
        serde_json::to_vec_pretty(&json!({
            "proposal_source": "model",
            "model_registry_ref": "openai/gpt-oss-20b",
            "output_classification": "MODEL_GENERATED_UNVERIFIED",
            "canonical_state_mutation_allowed": true
        }))
        .unwrap(),
    )
    .unwrap();

    let report = verify_model_airlock(&registry, Some(&artifact)).unwrap();

    assert_eq!(report.status, "FAIL");
    assert_eq!(report.canonical_influence_allowed, false);
    assert!(report
        .issues
        .iter()
        .any(|issue| issue.contains("without ACTION/HIL/SEAL acceptance")));
}

#[test]
fn model_airlock_allows_sealed_accepted_action_influence() {
    let tmp = tempfile::tempdir().unwrap();
    let registry = write_registry(&tmp);
    let artifact = tmp.path().join("accepted_artifact.json");
    fs::write(
        &artifact,
        serde_json::to_vec_pretty(&json!({
            "proposal_source": "model",
            "model_registry_ref": "openai/gpt-oss-20b",
            "output_classification": "ACCEPTED",
            "canonical_state_mutation_allowed": true,
            "parsed_action_schema": "prismcode.action.v0_1",
            "constraints_applied": ["policy", "science", "security", "evidence"],
            "hil_resolution": "ACCEPTED",
            "seal_ref": "AUDIT.SEAL:test"
        }))
        .unwrap(),
    )
    .unwrap();

    let report = verify_model_airlock(&registry, Some(&artifact)).unwrap();

    assert_eq!(report.status, "PASS");
    assert_eq!(report.accepted_action_flow, true);
    assert_eq!(report.hil_accepted, true);
    assert_eq!(report.sealed, true);
    assert_eq!(report.canonical_influence_allowed, true);
}
