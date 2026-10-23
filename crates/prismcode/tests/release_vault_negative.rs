use prismcode::key_envelope::{KeyContext, SignContext};
use prismcode::key_provider::{ensure_market_provider_state, PrismKeyProvider};
use prismcode::key_provider_local_release_vault::{write_json, LocalReleaseVault};
use prismcode::manifest_signature::{sign_artifact_report, verify_artifact_report};
use prismcode::release_key_policy::ReleaseKeyPolicy;
use serde_json::{json, Value};
use std::fs;
use std::sync::{Mutex, OnceLock};

const POLICY: &str = include_str!("../../../configs/security/release_key_policy.toml");

#[test]
fn market_mode_rejects_test_fixture_and_file_keyring_dev() {
    assert!(ensure_market_provider_state("TEST_FIXTURE_ONLY").is_err());
    assert!(ensure_market_provider_state("FILE_KEYRING_DEV").is_err());
}

#[test]
fn missing_release_key_policy_fails_closed() {
    let missing = tempfile::tempdir().unwrap().path().join("missing.toml");
    assert!(ReleaseKeyPolicy::load(&missing).is_err());
}

#[test]
fn policy_allowing_dev_file_keyring_fails_closed() {
    let bad = POLICY.replace(
        "dev_file_keyring_allowed = false",
        "dev_file_keyring_allowed = true",
    );
    assert!(ReleaseKeyPolicy::parse(&bad).is_err());
}

#[test]
fn policy_allowing_test_fixture_fails_closed() {
    let bad = POLICY.replace(
        "test_fixture_allowed = false",
        "test_fixture_allowed = true",
    );
    assert!(ReleaseKeyPolicy::parse(&bad).is_err());
}

#[test]
fn policy_allowing_raw_key_export_fails_closed() {
    let bad = POLICY.replace(
        "raw_key_export_allowed = false",
        "raw_key_export_allowed = true",
    );
    assert!(ReleaseKeyPolicy::parse(&bad).is_err());
}

#[test]
fn wrap_success_but_wrong_unwrap_context_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = ReleaseKeyPolicy::parse(POLICY).unwrap();
    let vault =
        LocalReleaseVault::open_or_create_at(tmp.path().join("vault.json"), policy).unwrap();
    let dek = [3u8; 32];
    let good = KeyContext::new("artifact", "hash", "policy", "WRAP");
    let bad = KeyContext::new("artifact", "different-hash", "policy", "WRAP");
    let envelope = vault.wrap_dek(&dek, &good).unwrap();
    assert!(vault.unwrap_dek(&envelope, &bad).is_err());
}

#[test]
fn sign_success_but_wrong_digest_verify_fails() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = ReleaseKeyPolicy::parse(POLICY).unwrap();
    let vault =
        LocalReleaseVault::open_or_create_at(tmp.path().join("vault.json"), policy).unwrap();
    let digest = [9u8; 32];
    let signature = vault
        .sign_digest(
            &digest,
            &SignContext::new("artifact", "hash", "policy", "SIGN"),
        )
        .unwrap();
    let wrong_digest = [8u8; 32];
    assert!(vault.verify_signature(&signature, &wrong_digest).is_err());
}

#[test]
fn rotation_missing_in_policy_fails_closed() {
    let bad = POLICY.replace("rotate_required = true", "rotate_required = false");
    assert!(ReleaseKeyPolicy::parse(&bad).is_err());
}

#[test]
fn raw_key_exported_true_in_vault_store_fails_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = ReleaseKeyPolicy::parse(POLICY).unwrap();
    let store = tmp.path().join("vault.json");
    let vault = LocalReleaseVault::open_or_create_at(store.clone(), policy.clone()).unwrap();
    vault.health_check().unwrap();
    let mut value: Value = serde_json::from_slice(&fs::read(&store).unwrap()).unwrap();
    value["raw_key_exported"] = json!(true);
    write_json(&store, &value).unwrap();
    let poisoned = LocalReleaseVault::open_or_create_at(store, policy).unwrap_err();
    assert!(poisoned.to_string().contains("raw_key_exported=true"));
}

#[test]
fn unsigned_cuda_manifest_verification_fails_closed() {
    let tmp = tempfile::tempdir().unwrap();
    let artifact = tmp.path().join("prismcode_cuda_manifest.json");
    write_json(&artifact, &json!({"cuda_manifest": true})).unwrap();
    let missing_sig = tmp.path().join("missing.sig");
    assert!(verify_artifact_report(&artifact, &missing_sig).is_err());
}

#[test]
fn tampered_cuda_manifest_signature_fails_closed() {
    let _guard = env_lock().lock().unwrap();
    let tmp = tempfile::tempdir().unwrap();
    let policy_path = tmp.path().join("policy.toml");
    fs::write(&policy_path, POLICY).unwrap();
    std::env::set_var(
        "PRISMCODE_LOCAL_RELEASE_VAULT_STORE",
        tmp.path().join("vault.json"),
    );
    let artifact = tmp.path().join("prismcode_cuda_manifest.json");
    write_json(&artifact, &json!({"cuda_manifest": true, "version": 1})).unwrap();
    let sig = tmp.path().join("cuda_manifest.sig");
    sign_artifact_report(&artifact, &policy_path, &sig).unwrap();
    let mut sig_json: Value = serde_json::from_slice(&fs::read(&sig).unwrap()).unwrap();
    sig_json["signature"]["signature_hex"] = json!("00");
    write_json(&sig, &sig_json).unwrap();
    assert!(verify_artifact_report(&artifact, &sig).is_err());
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}
