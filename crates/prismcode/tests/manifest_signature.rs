use prismcode::key_provider_local_release_vault::write_json;
use prismcode::manifest_signature::{sign_artifact_report, verify_artifact_report};
use serde_json::json;
use std::fs;
use std::sync::{Mutex, OnceLock};

const POLICY: &str = include_str!("../../../configs/security/release_key_policy.toml");

#[test]
fn manifest_signature_signs_and_verifies_artifact() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let policy_path = tmp.path().join("policy.toml");
    fs::write(&policy_path, POLICY).unwrap();
    std::env::set_var(
        "PRISMCODE_LOCAL_RELEASE_VAULT_STORE",
        tmp.path().join("vault.json"),
    );
    let artifact = tmp.path().join("manifest.json");
    write_json(&artifact, &json!({"cuda_manifest": true, "version": 1})).unwrap();
    let sig = tmp.path().join("manifest.sig");
    let report = sign_artifact_report(&artifact, &policy_path, &sig).unwrap();
    assert_eq!(report["signature_status"], "SIGNED_BY_RELEASE_VAULT");
    let verify = verify_artifact_report(&artifact, &sig).unwrap();
    assert_eq!(verify["signature_status"], "SIGNATURE_VERIFIED");
    assert_eq!(verify["raw_key_exported"], false);
}

#[test]
fn manifest_signature_rejects_tampered_artifact() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let policy_path = tmp.path().join("policy.toml");
    fs::write(&policy_path, POLICY).unwrap();
    std::env::set_var(
        "PRISMCODE_LOCAL_RELEASE_VAULT_STORE",
        tmp.path().join("vault.json"),
    );
    let artifact = tmp.path().join("manifest.json");
    write_json(&artifact, &json!({"cuda_manifest": true, "version": 1})).unwrap();
    let sig = tmp.path().join("manifest.sig");
    sign_artifact_report(&artifact, &policy_path, &sig).unwrap();
    write_json(&artifact, &json!({"cuda_manifest": true, "version": 2})).unwrap();
    assert!(verify_artifact_report(&artifact, &sig).is_err());
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}
