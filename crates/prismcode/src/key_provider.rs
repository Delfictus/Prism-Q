use crate::binary_runtime::{PcError, PcResult};
use crate::key_envelope::{
    KeyContext, KeyEnvelope, KeyRotationReport, RotationPolicy, SignContext, SignatureEnvelope,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub const TEST_FIXTURE_ONLY: &str = "TEST_FIXTURE_ONLY";
pub const FILE_KEYRING_DEV: &str = "FILE_KEYRING_DEV";
pub const LOCAL_DEV_EPHEMERAL: &str = "LOCAL_DEV_EPHEMERAL";
pub const LOCAL_RELEASE_VAULT_BOUND: &str = "LOCAL_RELEASE_VAULT_BOUND";
pub const PRISM_VAULT_UNBOUND: &str = "PRISM_VAULT_UNBOUND";
pub const OPENBAO: &str = "OPENBAO";
pub const OPENBAO_OR_VAULT_BOUND: &str = "OPENBAO_OR_VAULT_BOUND";
pub const TPM: &str = "TPM";
pub const TPM_BOUND: &str = "TPM_BOUND";
pub const HSM_BOUND: &str = "HSM_BOUND";
pub const CLOUD_KMS_BOUND: &str = "CLOUD_KMS_BOUND";
pub const OFFLINE_RELEASE_SIGNING: &str = "OFFLINE_RELEASE_SIGNING";

pub const FILE_KEYRING_DEV_BLOCKER: &str = "FILE_KEYRING_DEV_NOT_MARKET_READY";
pub const TEST_FIXTURE_ONLY_BLOCKER: &str = "TEST_FIXTURE_ONLY_NOT_MARKET_READY";

const RAW_KEY_FIELDS: &[&str] = &[
    "data_key_hex",
    "signing_pkcs8_hex",
    "private_key_hex",
    "secret_key_hex",
    "wrap_key_hex",
    "vault_payload_plaintext",
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProviderState {
    TestFixtureOnly,
    FileKeyringDev,
    LocalReleaseVaultBound,
    OpenBaoOrVaultBound,
    TpmBound,
    HsmBound,
    CloudKmsBound,
    OfflineReleaseSigning,
    PrismVaultUnbound,
}

impl ProviderState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::TestFixtureOnly => TEST_FIXTURE_ONLY,
            Self::FileKeyringDev => FILE_KEYRING_DEV,
            Self::LocalReleaseVaultBound => LOCAL_RELEASE_VAULT_BOUND,
            Self::OpenBaoOrVaultBound => OPENBAO_OR_VAULT_BOUND,
            Self::TpmBound => TPM_BOUND,
            Self::HsmBound => HSM_BOUND,
            Self::CloudKmsBound => CLOUD_KMS_BOUND,
            Self::OfflineReleaseSigning => OFFLINE_RELEASE_SIGNING,
            Self::PrismVaultUnbound => PRISM_VAULT_UNBOUND,
        }
    }

    pub fn market_ready_allowed(self) -> bool {
        matches!(
            self,
            Self::LocalReleaseVaultBound
                | Self::OpenBaoOrVaultBound
                | Self::TpmBound
                | Self::HsmBound
                | Self::CloudKmsBound
                | Self::OfflineReleaseSigning
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyProviderHealth {
    pub status: String,
    pub provider_state: String,
    pub market_release_allowed: bool,
    pub raw_key_exported: bool,
    pub audit_log_present: bool,
    pub revocation_status: String,
}

pub trait PrismKeyProvider {
    fn provider_id(&self) -> &str;
    fn provider_state(&self) -> ProviderState;
    fn health_check(&self) -> PcResult<KeyProviderHealth>;

    fn wrap_dek(&self, dek: &[u8], context: &KeyContext) -> PcResult<KeyEnvelope>;

    fn unwrap_dek(&self, envelope: &KeyEnvelope, context: &KeyContext) -> PcResult<Vec<u8>>;

    fn sign_digest(&self, digest: &[u8], context: &SignContext) -> PcResult<SignatureEnvelope>;

    fn verify_signature(&self, signature: &SignatureEnvelope, digest: &[u8]) -> PcResult<()>;

    fn rotate_key(&self, key_id: &str, policy: &RotationPolicy) -> PcResult<KeyRotationReport>;
}

pub fn raw_key_field_names() -> &'static [&'static str] {
    RAW_KEY_FIELDS
}

pub fn contains_raw_key_field_name(text: &str) -> bool {
    RAW_KEY_FIELDS.iter().any(|field| text.contains(field))
}

pub fn ensure_market_provider_state(provider_state: &str) -> PcResult<()> {
    match provider_state {
        LOCAL_RELEASE_VAULT_BOUND
        | OPENBAO_OR_VAULT_BOUND
        | TPM_BOUND
        | HSM_BOUND
        | CLOUD_KMS_BOUND
        | OFFLINE_RELEASE_SIGNING => Ok(()),
        TEST_FIXTURE_ONLY => Err(PcError::Security(TEST_FIXTURE_ONLY_BLOCKER.to_string())),
        FILE_KEYRING_DEV => Err(PcError::Security(FILE_KEYRING_DEV_BLOCKER.to_string())),
        other => Err(PcError::Security(format!(
            "provider state {other} is not market-release bound"
        ))),
    }
}

pub fn provider_matrix() -> Value {
    json!({
        "prism_vault_version": "PRISM_VAULT_V0_1_RELEASE_BOUND",
        "provider_states_explicit": true,
        "providers": [
            {
                "provider": LOCAL_DEV_EPHEMERAL,
                "state": "DEV_EPHEMERAL_UNWRAPPED",
                "release_candidate_allowed": false,
                "market_ready_allowed": false,
                "raw_key_output_allowed": false,
                "blockers": ["LOCAL_DEV_EPHEMERAL_NOT_RELEASE_SAFE"]
            },
            {
                "provider": TEST_FIXTURE_ONLY,
                "state": TEST_FIXTURE_ONLY,
                "release_candidate_allowed": false,
                "market_ready_allowed": false,
                "raw_key_output_allowed": false,
                "blockers": [TEST_FIXTURE_ONLY_BLOCKER]
            },
            {
                "provider": FILE_KEYRING_DEV,
                "state": "DEV_FILE_KEYRING_BOUND",
                "release_candidate_allowed": true,
                "l5_release_candidate_allowed": true,
                "market_ready_allowed": false,
                "raw_key_output_allowed": false,
                "blockers": [FILE_KEYRING_DEV_BLOCKER]
            },
            {
                "provider": LOCAL_RELEASE_VAULT_BOUND,
                "state": LOCAL_RELEASE_VAULT_BOUND,
                "release_candidate_allowed": true,
                "market_ready_allowed": true,
                "raw_key_output_allowed": false,
                "blockers": []
            },
            {
                "provider": OPENBAO,
                "state": "UNBOUND",
                "release_candidate_allowed": false,
                "market_ready_allowed": false,
                "raw_key_output_allowed": false,
                "blockers": ["OPENBAO_OR_VAULT_UNBOUND"]
            },
            {
                "provider": TPM,
                "state": "EVIDENCE_NOT_FOUND",
                "release_candidate_allowed": false,
                "market_ready_allowed": false,
                "raw_key_output_allowed": false,
                "blockers": ["TPM_EVIDENCE_NOT_FOUND"]
            }
        ],
        "raw_keys_printed": false,
        "raw_keys_exposed": false
    })
}

pub fn file_keyring_dev_overlay() -> Value {
    json!({
        "provider_state": "DEV_FILE_KEYRING_BOUND",
        "release_candidate_allowed": true,
        "l5_release_candidate_allowed": true,
        "market_ready_allowed": false,
        "market_ready_eligible": false,
        "market_blockers": [FILE_KEYRING_DEV_BLOCKER],
        "raw_key_output_allowed": false,
        "raw_keys_printed": false
    })
}

pub fn local_release_vault_overlay() -> Value {
    json!({
        "provider_state": LOCAL_RELEASE_VAULT_BOUND,
        "release_candidate_allowed": true,
        "l5_release_candidate_allowed": true,
        "market_ready_allowed": true,
        "market_ready_eligible": true,
        "raw_key_output_allowed": false,
        "raw_key_exported": false,
        "raw_keys_printed": false
    })
}

pub fn apply_file_keyring_dev_overlay(report: &mut Value) {
    merge_object(report, file_keyring_dev_overlay());
}

pub fn apply_local_release_vault_overlay(report: &mut Value) {
    merge_object(report, local_release_vault_overlay());
}

pub fn merge_object(target: &mut Value, overlay: Value) {
    let Some(target_obj) = target.as_object_mut() else {
        *target = overlay;
        return;
    };
    let Some(overlay_obj) = overlay.as_object() else {
        return;
    };
    for (key, value) in overlay_obj {
        target_obj.insert(key.clone(), value.clone());
    }
}
