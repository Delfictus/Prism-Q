use prismcode::key_envelope::KeyContext;
use prismcode::key_provider::{PrismKeyProvider, LOCAL_RELEASE_VAULT_BOUND};
use prismcode::key_provider_local_release_vault::LocalReleaseVault;
use prismcode::release_key_policy::ReleaseKeyPolicy;

const POLICY: &str = include_str!("../../../configs/security/release_key_policy.toml");

#[test]
fn release_vault_wrap_unwrap_roundtrip_passes() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = ReleaseKeyPolicy::parse(POLICY).unwrap();
    let vault =
        LocalReleaseVault::open_or_create_at(tmp.path().join("vault.json"), policy).unwrap();
    let dek = [7u8; 32];
    let context = KeyContext::new(
        "release-vault-test",
        blake3::hash(&dek).to_hex().to_string(),
        "policy-hash",
        "WRAP_TEST",
    );
    let envelope = vault.wrap_dek(&dek, &context).unwrap();
    assert_eq!(envelope.provider_state, LOCAL_RELEASE_VAULT_BOUND);
    assert!(!envelope.raw_key_exported);
    assert_ne!(
        envelope.ciphertext_hex,
        prismcode::key_provider_local_release_vault::hex_bytes(&dek)
    );
    let unwrapped = vault.unwrap_dek(&envelope, &context).unwrap();
    assert_eq!(unwrapped, dek);
}

#[test]
fn provider_report_declares_local_release_profile_not_external_vault() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = ReleaseKeyPolicy::parse(POLICY).unwrap();
    let vault =
        LocalReleaseVault::open_or_create_at(tmp.path().join("vault.json"), policy).unwrap();
    let report = vault.provider_report().unwrap();
    assert_eq!(report["status"], "PASS");
    assert_eq!(report["provider_state"], LOCAL_RELEASE_VAULT_BOUND);
    assert_eq!(report["enterprise_external_vault"], false);
    assert_eq!(report["market_profile"], "LOCAL_RELEASE");
    assert_eq!(report["raw_key_exported"], false);
    assert_eq!(report["file_keyring_dev_for_market"], false);
}
