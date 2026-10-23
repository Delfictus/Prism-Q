use prismcode::key_provider::{
    ensure_market_provider_state, PrismKeyProvider, ProviderState, FILE_KEYRING_DEV,
    LOCAL_RELEASE_VAULT_BOUND, TEST_FIXTURE_ONLY,
};
use prismcode::key_provider_local_release_vault::LocalReleaseVault;
use prismcode::release_key_policy::ReleaseKeyPolicy;

const POLICY: &str = r#"
[policy]
name = "PRISMCODE_RELEASE_KEY_POLICY_V0_1"
market_release_allowed = true
dev_file_keyring_allowed = false
test_fixture_allowed = false
raw_key_export_allowed = false
rotation_required = true
audit_log_required = true

[provider]
required_state = "LOCAL_RELEASE_VAULT_BOUND"

[operations]
wrap_dek_required = true
unwrap_dek_required = true
sign_required = true
verify_required = true
rotate_required = true

[artifacts]
cuda_manifest_signing_required = true
release_manifest_signing_required = true
evidence_ledger_signing_required = true
seal_chain_binding_required = true
"#;

#[test]
fn local_release_vault_is_market_bound_provider_state() {
    assert!(ensure_market_provider_state(LOCAL_RELEASE_VAULT_BOUND).is_ok());
    assert!(ensure_market_provider_state(FILE_KEYRING_DEV).is_err());
    assert!(ensure_market_provider_state(TEST_FIXTURE_ONLY).is_err());
    assert!(ProviderState::LocalReleaseVaultBound.market_ready_allowed());
}

#[test]
fn local_release_vault_health_has_no_raw_key_export() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = ReleaseKeyPolicy::parse(POLICY).unwrap();
    let vault =
        LocalReleaseVault::open_or_create_at(tmp.path().join("vault.json"), policy).unwrap();
    let health = vault.health_check().unwrap();
    assert_eq!(health.status, "PASS");
    assert_eq!(health.provider_state, LOCAL_RELEASE_VAULT_BOUND);
    assert!(health.market_release_allowed);
    assert!(!health.raw_key_exported);
    assert!(health.audit_log_present);
}
