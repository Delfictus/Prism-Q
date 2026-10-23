use prismcode::key_envelope::RotationPolicy;
use prismcode::key_provider::PrismKeyProvider;
use prismcode::key_provider_local_release_vault::LocalReleaseVault;
use prismcode::release_key_policy::ReleaseKeyPolicy;

const POLICY: &str = include_str!("../../../configs/security/release_key_policy.toml");

#[test]
fn local_release_vault_rotation_changes_key_version_and_public_hash() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = ReleaseKeyPolicy::parse(POLICY).unwrap();
    let vault =
        LocalReleaseVault::open_or_create_at(tmp.path().join("vault.json"), policy).unwrap();
    let report = vault
        .rotate_key("prismcode-release-key-v0-1", &RotationPolicy::blocker_02())
        .unwrap();
    assert_eq!(report.status, "PASS");
    assert_eq!(report.old_key_version + 1, report.new_key_version);
    assert_ne!(
        report.old_public_key_hash_blake3,
        report.new_public_key_hash_blake3
    );
    assert!(report.audit_log_appended);
    assert!(!report.raw_key_exported);
}
