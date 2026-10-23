use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyContext {
    pub artifact_id: String,
    pub artifact_hash: String,
    pub policy_hash: String,
    pub operation: String,
}

impl KeyContext {
    pub fn new(
        artifact_id: impl Into<String>,
        artifact_hash: impl Into<String>,
        policy_hash: impl Into<String>,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            artifact_id: artifact_id.into(),
            artifact_hash: artifact_hash.into(),
            policy_hash: policy_hash.into(),
            operation: operation.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignContext {
    pub artifact_id: String,
    pub artifact_hash: String,
    pub policy_hash: String,
    pub operation: String,
}

impl SignContext {
    pub fn new(
        artifact_id: impl Into<String>,
        artifact_hash: impl Into<String>,
        policy_hash: impl Into<String>,
        operation: impl Into<String>,
    ) -> Self {
        Self {
            artifact_id: artifact_id.into(),
            artifact_hash: artifact_hash.into(),
            policy_hash: policy_hash.into(),
            operation: operation.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyEnvelope {
    pub envelope_version: String,
    pub provider_state: String,
    pub key_id: String,
    pub key_version: u64,
    pub algorithm: String,
    pub nonce_hex: String,
    pub aad_hash_blake3: String,
    pub ciphertext_hex: String,
    pub wrapped_key_hash_blake3: String,
    pub context: KeyContext,
    pub raw_key_exported: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SignatureEnvelope {
    pub signature_version: String,
    pub provider_state: String,
    pub key_id: String,
    pub key_version: u64,
    pub algorithm: String,
    pub public_key_hex: String,
    pub public_key_hash_blake3: String,
    pub digest_hex: String,
    pub signature_hex: String,
    pub context: SignContext,
    pub raw_key_exported: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RotationPolicy {
    pub reason: String,
    pub require_audit_log: bool,
}

impl RotationPolicy {
    pub fn blocker_02() -> Self {
        Self {
            reason: "BLOCKER_02_RELEASE_KEY_ROTATION_TEST".to_string(),
            require_audit_log: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyRotationReport {
    pub status: String,
    pub provider_state: String,
    pub key_id: String,
    pub old_key_version: u64,
    pub new_key_version: u64,
    pub old_public_key_hash_blake3: String,
    pub new_public_key_hash_blake3: String,
    pub audit_log_appended: bool,
    pub raw_key_exported: bool,
}
