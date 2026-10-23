use prismcode::{attestation, key_provider, release_manifest};
use serde_json::{json, Value};
use std::fs;
use std::process::Command;

#[test]
fn file_keyring_dev_state_is_not_market_ready() {
    let matrix = key_provider::provider_matrix();
    let providers = matrix["providers"].as_array().unwrap();
    let file_provider = providers
        .iter()
        .find(|provider| provider["provider"] == key_provider::FILE_KEYRING_DEV)
        .unwrap();

    assert_eq!(file_provider["l5_release_candidate_allowed"], true);
    assert_eq!(file_provider["market_ready_allowed"], false);
    assert!(file_provider["blockers"]
        .as_array()
        .unwrap()
        .iter()
        .any(|blocker| blocker == key_provider::FILE_KEYRING_DEV_BLOCKER));
}

#[test]
fn absent_attestation_reports_unavailable_not_trusted() {
    let report = attestation::mark_runtime_attestation_unavailable(json!({
        "attestation_version": "PRISM_ATTEST_V0_1_DEV",
        "secure_boot": "EVIDENCE_NOT_FOUND",
        "tpm": "EVIDENCE_NOT_FOUND"
    }));

    assert_eq!(report["status"], attestation::ATTESTATION_UNAVAILABLE);
    assert_eq!(report["trust_state"], attestation::ATTESTATION_UNAVAILABLE);
    assert_eq!(report["trusted_attestation"], false);
    assert_ne!(report["trust_state"], attestation::TRUSTED_ATTESTED);
}

#[test]
fn release_policy_blocks_file_keyring_dev_market_ready() {
    let mut report = json!({
        "status": "PASS",
        "key_provider": key_provider::FILE_KEYRING_DEV,
        "attestation_status": attestation::ATTESTATION_UNAVAILABLE
    });
    release_manifest::apply_release_candidate_policy(&mut report);

    assert_eq!(report["market_ready"], false);
    assert_eq!(report["market_ready_eligible"], false);
    assert!(contains_blocker(
        &report,
        "FILE_KEYRING_DEV_NOT_MARKET_READY"
    ));
    assert!(contains_blocker(&report, "TRUSTED_ATTESTATION_UNBOUND"));
}

#[test]
fn prismkey_init_file_keyring_prints_no_raw_key_material() {
    let tmp = tempfile::tempdir().unwrap();
    let keyring = tmp.path().join("prismcode_file_keyring.json");
    let public_key = tmp.path().join("release_public_key.ed25519.hex");
    let output = Command::new(env!("CARGO_BIN_EXE_prismkey"))
        .args([
            "init-file-keyring",
            "--out",
            keyring.to_str().unwrap(),
            "--public-out",
            public_key.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(!key_provider::contains_raw_key_field_name(&stdout));
    assert!(stdout.contains("FILE_KEYRING_DEV_NOT_MARKET_READY"));
    assert!(stdout.contains("\"raw_keys_printed\": false"));

    let keyring_json: Value = serde_json::from_str(&fs::read_to_string(keyring).unwrap()).unwrap();
    for field in key_provider::raw_key_field_names() {
        if let Some(secret) = keyring_json.get(*field).and_then(Value::as_str) {
            assert!(!stdout.contains(secret));
        }
    }
}

#[test]
fn prismattest_runtime_prints_attestation_unavailable() {
    let output = Command::new(env!("CARGO_BIN_EXE_prismattest"))
        .arg("runtime")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("\"status\": \"ATTESTATION_UNAVAILABLE\""));
    assert!(stdout.contains("\"trusted_attestation\": false"));
}

fn contains_blocker(report: &Value, expected: &str) -> bool {
    report["market_blockers"]
        .as_array()
        .unwrap()
        .iter()
        .any(|blocker| blocker.as_str() == Some(expected))
}
