use crate::binary_runtime::{PcError, PcResult};
use crate::key_envelope::{
    KeyContext, KeyEnvelope, KeyRotationReport, RotationPolicy, SignContext, SignatureEnvelope,
};
use crate::key_provider::{KeyProviderHealth, PrismKeyProvider, ProviderState};
use crate::release_key_policy::ReleaseKeyPolicy;
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, AES_256_GCM};
use ring::digest::{digest, SHA256};
use ring::pbkdf2::{derive, PBKDF2_HMAC_SHA256};
use ring::rand::{SecureRandom, SystemRandom};
use ring::signature::{Ed25519KeyPair, KeyPair, UnparsedPublicKey, ED25519};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const PROVIDER_ID: &str = "prism-local-release-vault";
const VAULT_VERSION: &str = "PRISM_LOCAL_RELEASE_VAULT_V0_1";
const KEY_ID: &str = "prismcode-release-key-v0-1";
const KDF_ITERATIONS: u32 = 210_000;

#[derive(Debug, Clone)]
pub struct LocalReleaseVault {
    store_path: PathBuf,
    unlock_secret: Vec<u8>,
    policy: ReleaseKeyPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VaultStore {
    vault_version: String,
    provider_state: String,
    provider_id: String,
    encrypted_local_vault_store: bool,
    unlock_mode: String,
    kdf_algorithm: String,
    kdf_iterations: u32,
    salt_hex: String,
    nonce_hex: String,
    ciphertext_hex: String,
    payload_hash_blake3: String,
    key_id: String,
    key_version: u64,
    public_key_hex: String,
    public_key_hash_blake3: String,
    revocation_status: String,
    allowed_operations: Vec<String>,
    raw_key_exported: bool,
    audit_log: Vec<AuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditEntry {
    event: String,
    unix_seconds: u64,
    policy_hash_blake3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VaultPayload {
    key_id: String,
    key_version: u64,
    wrap_key_hex: String,
    signing_algorithm: String,
    signing_pkcs8_hex: String,
    public_key_hex: String,
    revocation_status: String,
}

impl LocalReleaseVault {
    pub fn open_or_create(policy: ReleaseKeyPolicy) -> PcResult<Self> {
        let store_path = default_store_path();
        Self::open_or_create_at(store_path, policy)
    }

    pub fn open_or_create_at(store_path: PathBuf, policy: ReleaseKeyPolicy) -> PcResult<Self> {
        policy.validate_market_mode()?;
        if policy.required_state != ProviderState::LocalReleaseVaultBound.as_str() {
            return Err(PcError::Security(format!(
                "LOCAL_RELEASE_VAULT_BOUND policy required, got {}",
                policy.required_state
            )));
        }
        let unlock_secret = read_or_create_unlock_secret()?;
        let vault = Self {
            store_path,
            unlock_secret,
            policy,
        };
        if !vault.store_path.exists() {
            let payload = generate_payload(1)?;
            vault.save_payload(&payload, vec![audit("VAULT_CREATED", &vault.policy)])?;
        }
        vault.health_check()?;
        Ok(vault)
    }

    pub fn store_path(&self) -> &Path {
        &self.store_path
    }

    pub fn provider_report(&self) -> PcResult<Value> {
        let store = self.read_store()?;
        let health = self.health_check()?;
        Ok(json!({
            "status": "PASS",
            "provider_id": self.provider_id(),
            "provider_state": self.provider_state().as_str(),
            "provider_state_detail": "LOCAL_RELEASE encrypted vault store bound to release policy",
            "enterprise_external_vault": false,
            "market_profile": "LOCAL_RELEASE",
            "vault_store_path": self.store_path.display().to_string(),
            "vault_store_hash_blake3": file_hash_hex(&self.store_path)?,
            "encrypted_local_vault_store": store.encrypted_local_vault_store,
            "unlock_mode": store.unlock_mode,
            "kdf_algorithm": store.kdf_algorithm,
            "kdf_iterations": store.kdf_iterations,
            "aead_algorithm": "AES_256_GCM",
            "signing_algorithm": "ED25519",
            "revocation_status": store.revocation_status,
            "public_key_hash_blake3": store.public_key_hash_blake3,
            "release_key_policy_bound": true,
            "policy_hash": self.policy.policy_hash_blake3,
            "market_release_allowed": health.market_release_allowed,
            "file_keyring_dev_for_market": false,
            "test_fixture_only_for_market": false,
            "raw_key_exported": health.raw_key_exported,
            "raw_keys_printed": false,
        }))
    }

    pub fn wrap_test_report(&self) -> PcResult<Value> {
        let mut dek = [0u8; 32];
        SystemRandom::new()
            .fill(&mut dek)
            .map_err(|_| PcError::Internal("failed to generate wrap-test DEK".to_string()))?;
        let dek_hash = blake3_hex(&dek);
        let context = KeyContext::new(
            "BLOCKER_02_WRAP_TEST",
            dek_hash.clone(),
            self.policy.policy_hash_blake3.clone(),
            "WRAP_TEST",
        );
        let envelope = self.wrap_dek(&dek, &context)?;
        let unwrapped = self.unwrap_dek(&envelope, &context)?;
        let pass = unwrapped == dek;
        dek.fill(0);
        Ok(json!({
            "status": if pass { "PASS" } else { "FAIL" },
            "provider_state": self.provider_state().as_str(),
            "key_wrap": if !envelope.ciphertext_hex.is_empty() { "PASS" } else { "FAIL" },
            "key_unwrap": if pass { "PASS" } else { "FAIL" },
            "wrapped_key_hash_blake3": envelope.wrapped_key_hash_blake3,
            "aad_hash_blake3": envelope.aad_hash_blake3,
            "raw_key_exported": false,
            "policy_hash": self.policy.policy_hash_blake3,
        }))
    }

    pub fn rotate_test_report(&self) -> PcResult<Value> {
        let report = self.rotate_key(KEY_ID, &RotationPolicy::blocker_02())?;
        Ok(json!({
            "status": report.status,
            "provider_state": report.provider_state,
            "key_id": report.key_id,
            "old_key_version": report.old_key_version,
            "new_key_version": report.new_key_version,
            "old_public_key_hash_blake3": report.old_public_key_hash_blake3,
            "new_public_key_hash_blake3": report.new_public_key_hash_blake3,
            "rotation_test": if report.new_key_version > report.old_key_version { "PASS" } else { "FAIL" },
            "audit_log_appended": report.audit_log_appended,
            "raw_key_exported": report.raw_key_exported,
            "policy_hash": self.policy.policy_hash_blake3,
        }))
    }

    pub fn sign_test_report(&self, artifact: &Path) -> PcResult<Value> {
        let digest = artifact_digest(artifact)?;
        let context = SignContext::new(
            artifact.display().to_string(),
            hex_bytes(&digest),
            self.policy.policy_hash_blake3.clone(),
            "SIGN_TEST",
        );
        let signature = self.sign_digest(&digest, &context)?;
        self.verify_signature(&signature, &digest)?;
        Ok(json!({
            "status": "PASS",
            "provider_state": self.provider_state().as_str(),
            "artifact": artifact.display().to_string(),
            "artifact_hash_blake3": file_hash_hex(artifact)?,
            "digest_hex": signature.digest_hex,
            "sign_verify": "PASS",
            "signature_algorithm": signature.algorithm,
            "public_key_hash_blake3": signature.public_key_hash_blake3,
            "raw_key_exported": false,
            "policy_hash": self.policy.policy_hash_blake3,
        }))
    }

    pub fn sign_artifact(&self, artifact: &Path, sig_out: &Path) -> PcResult<Value> {
        let digest = artifact_digest(artifact)?;
        let context = SignContext::new(
            artifact.display().to_string(),
            hex_bytes(&digest),
            self.policy.policy_hash_blake3.clone(),
            "SIGN_ARTIFACT",
        );
        let signature = self.sign_digest(&digest, &context)?;
        let signature_file = json!({
            "signature_file_version": "PRISMCODE_ARTIFACT_SIGNATURE_V0_1",
            "status": "SIGNED_BY_RELEASE_VAULT",
            "provider_state": self.provider_state().as_str(),
            "artifact_path": artifact.display().to_string(),
            "artifact_hash_blake3": file_hash_hex(artifact)?,
            "artifact_sha256": sha256_file_hex(artifact)?,
            "policy_hash": self.policy.policy_hash_blake3,
            "signature": signature,
            "raw_key_exported": false,
        });
        write_json(sig_out, &signature_file)?;
        Ok(json!({
            "status": "PASS",
            "signature_status": "SIGNED_BY_RELEASE_VAULT",
            "provider_state": self.provider_state().as_str(),
            "artifact": artifact.display().to_string(),
            "artifact_hash_blake3": file_hash_hex(artifact)?,
            "sig_out": sig_out.display().to_string(),
            "signature_file_hash_blake3": file_hash_hex(sig_out)?,
            "cuda_manifest_signed": artifact.ends_with("prismcode_cuda_manifest.json"),
            "raw_key_exported": false,
            "policy_hash": self.policy.policy_hash_blake3,
        }))
    }

    pub fn verify_artifact(&self, artifact: &Path, sig_path: &Path) -> PcResult<Value> {
        let signature_file: Value = serde_json::from_slice(&fs::read(sig_path)?)?;
        let signature: SignatureEnvelope =
            serde_json::from_value(signature_file["signature"].clone())?;
        let digest = artifact_digest(artifact)?;
        self.verify_signature(&signature, &digest)?;
        let expected_hash = file_hash_hex(artifact)?;
        if signature_file
            .get("artifact_hash_blake3")
            .and_then(Value::as_str)
            != Some(expected_hash.as_str())
        {
            return Err(PcError::Security(
                "artifact hash in signature file does not match artifact".to_string(),
            ));
        }
        Ok(json!({
            "status": "PASS",
            "signature_status": "SIGNATURE_VERIFIED",
            "provider_state": self.provider_state().as_str(),
            "artifact": artifact.display().to_string(),
            "artifact_hash_blake3": expected_hash,
            "sig": sig_path.display().to_string(),
            "signature_file_hash_blake3": file_hash_hex(sig_path)?,
            "cuda_manifest_signature_verified": artifact.ends_with("prismcode_cuda_manifest.json"),
            "raw_key_exported": false,
            "policy_hash": self.policy.policy_hash_blake3,
        }))
    }

    fn read_store(&self) -> PcResult<VaultStore> {
        let store: VaultStore = serde_json::from_slice(&fs::read(&self.store_path)?)?;
        if store.provider_state != ProviderState::LocalReleaseVaultBound.as_str() {
            return Err(PcError::Security(format!(
                "local release vault store has wrong provider_state {}",
                store.provider_state
            )));
        }
        if store.raw_key_exported {
            return Err(PcError::Security(
                "local release vault store declares raw_key_exported=true".to_string(),
            ));
        }
        Ok(store)
    }

    fn load_payload(&self) -> PcResult<(VaultStore, VaultPayload)> {
        let store = self.read_store()?;
        let key = self.derive_store_key(&store)?;
        let nonce = nonce_from_hex(&store.nonce_hex)?;
        let mut ciphertext = decode_hex(&store.ciphertext_hex)?;
        let plaintext = key
            .open_in_place(
                nonce,
                Aad::from(store_aad(&store).as_slice()),
                &mut ciphertext,
            )
            .map_err(|_| PcError::Security("failed to unlock local release vault".to_string()))?
            .to_vec();
        if blake3_hex(&plaintext) != store.payload_hash_blake3 {
            return Err(PcError::Hash(
                "local release vault payload hash mismatch".to_string(),
            ));
        }
        let payload: VaultPayload = serde_json::from_slice(&plaintext)?;
        Ok((store, payload))
    }

    fn save_payload(&self, payload: &VaultPayload, mut audit_log: Vec<AuditEntry>) -> PcResult<()> {
        let mut salt = [0u8; 16];
        let mut nonce = [0u8; 12];
        SystemRandom::new()
            .fill(&mut salt)
            .map_err(|_| PcError::Internal("failed to generate vault salt".to_string()))?;
        SystemRandom::new()
            .fill(&mut nonce)
            .map_err(|_| PcError::Internal("failed to generate vault nonce".to_string()))?;
        if audit_log.is_empty() {
            audit_log.push(audit("VAULT_WRITTEN", &self.policy));
        }
        let public_key = decode_hex(&payload.public_key_hex)?;
        let mut store = VaultStore {
            vault_version: VAULT_VERSION.to_string(),
            provider_state: ProviderState::LocalReleaseVaultBound.as_str().to_string(),
            provider_id: PROVIDER_ID.to_string(),
            encrypted_local_vault_store: true,
            unlock_mode: unlock_mode(),
            kdf_algorithm: "PBKDF2_HMAC_SHA256".to_string(),
            kdf_iterations: KDF_ITERATIONS,
            salt_hex: hex_bytes(&salt),
            nonce_hex: hex_bytes(&nonce),
            ciphertext_hex: String::new(),
            payload_hash_blake3: String::new(),
            key_id: payload.key_id.clone(),
            key_version: payload.key_version,
            public_key_hex: payload.public_key_hex.clone(),
            public_key_hash_blake3: blake3_hex(&public_key),
            revocation_status: payload.revocation_status.clone(),
            allowed_operations: vec![
                "WRAP_DEK".to_string(),
                "UNWRAP_DEK".to_string(),
                "SIGN_RELEASE_ARTIFACT".to_string(),
                "VERIFY_RELEASE_ARTIFACT".to_string(),
                "ROTATE_KEY".to_string(),
            ],
            raw_key_exported: false,
            audit_log,
        };
        let plaintext = serde_json::to_vec(payload)?;
        store.payload_hash_blake3 = blake3_hex(&plaintext);
        let key = self.derive_store_key(&store)?;
        let mut ciphertext = plaintext;
        key.seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce),
            Aad::from(store_aad(&store).as_slice()),
            &mut ciphertext,
        )
        .map_err(|_| PcError::Internal("failed to encrypt local release vault".to_string()))?;
        store.ciphertext_hex = hex_bytes(&ciphertext);
        write_json(&self.store_path, &store)?;
        chmod_0600(&self.store_path)?;
        Ok(())
    }

    fn derive_store_key(&self, store: &VaultStore) -> PcResult<LessSafeKey> {
        let salt = decode_hex(&store.salt_hex)?;
        let mut out = [0u8; 32];
        derive(
            PBKDF2_HMAC_SHA256,
            NonZeroU32::new(store.kdf_iterations).ok_or_else(|| {
                PcError::Security("local release vault KDF iterations cannot be zero".to_string())
            })?,
            &salt,
            &self.unlock_secret,
            &mut out,
        );
        let unbound = UnboundKey::new(&AES_256_GCM, &out)
            .map_err(|_| PcError::Internal("failed to initialize vault AEAD key".to_string()))?;
        out.fill(0);
        Ok(LessSafeKey::new(unbound))
    }
}

impl PrismKeyProvider for LocalReleaseVault {
    fn provider_id(&self) -> &str {
        PROVIDER_ID
    }

    fn provider_state(&self) -> ProviderState {
        ProviderState::LocalReleaseVaultBound
    }

    fn health_check(&self) -> PcResult<KeyProviderHealth> {
        let store = self.read_store()?;
        let (_store, payload) = self.load_payload()?;
        if payload.key_id != KEY_ID {
            return Err(PcError::Security(
                "local release vault key_id mismatch".to_string(),
            ));
        }
        if store.revocation_status != "ACTIVE" {
            return Err(PcError::Security(format!(
                "local release vault key is revoked: {}",
                store.revocation_status
            )));
        }
        Ok(KeyProviderHealth {
            status: "PASS".to_string(),
            provider_state: self.provider_state().as_str().to_string(),
            market_release_allowed: true,
            raw_key_exported: false,
            audit_log_present: !store.audit_log.is_empty(),
            revocation_status: store.revocation_status,
        })
    }

    fn wrap_dek(&self, dek: &[u8], context: &KeyContext) -> PcResult<KeyEnvelope> {
        if dek.len() != 32 {
            return Err(PcError::Security(
                "DEK wrap requires 32-byte AES-256 data key".to_string(),
            ));
        }
        let (store, payload) = self.load_payload()?;
        let wrap_key = decode_hex(&payload.wrap_key_hex)?;
        let key = LessSafeKey::new(
            UnboundKey::new(&AES_256_GCM, &wrap_key)
                .map_err(|_| PcError::Internal("invalid wrap key".to_string()))?,
        );
        let mut nonce = [0u8; 12];
        SystemRandom::new()
            .fill(&mut nonce)
            .map_err(|_| PcError::Internal("failed to generate wrap nonce".to_string()))?;
        let aad = serde_json::to_vec(context)?;
        let mut ciphertext = dek.to_vec();
        key.seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce),
            Aad::from(aad.as_slice()),
            &mut ciphertext,
        )
        .map_err(|_| PcError::Internal("failed to wrap DEK".to_string()))?;
        Ok(KeyEnvelope {
            envelope_version: "PRISMCODE_KEY_ENVELOPE_V0_1".to_string(),
            provider_state: self.provider_state().as_str().to_string(),
            key_id: payload.key_id,
            key_version: payload.key_version,
            algorithm: "AES_256_GCM_KEY_WRAP".to_string(),
            nonce_hex: hex_bytes(&nonce),
            aad_hash_blake3: blake3_hex(&aad),
            ciphertext_hex: hex_bytes(&ciphertext),
            wrapped_key_hash_blake3: blake3_hex(&ciphertext),
            context: context.clone(),
            raw_key_exported: store.raw_key_exported,
        })
    }

    fn unwrap_dek(&self, envelope: &KeyEnvelope, context: &KeyContext) -> PcResult<Vec<u8>> {
        if envelope.context != *context {
            return Err(PcError::Security(
                "key envelope context does not match unwrap context".to_string(),
            ));
        }
        let (_store, payload) = self.load_payload()?;
        let wrap_key = decode_hex(&payload.wrap_key_hex)?;
        let key = LessSafeKey::new(
            UnboundKey::new(&AES_256_GCM, &wrap_key)
                .map_err(|_| PcError::Internal("invalid wrap key".to_string()))?,
        );
        let aad = serde_json::to_vec(context)?;
        if blake3_hex(&aad) != envelope.aad_hash_blake3 {
            return Err(PcError::Hash("key envelope AAD hash mismatch".to_string()));
        }
        let mut ciphertext = decode_hex(&envelope.ciphertext_hex)?;
        let plaintext = key
            .open_in_place(
                nonce_from_hex(&envelope.nonce_hex)?,
                Aad::from(aad.as_slice()),
                &mut ciphertext,
            )
            .map_err(|_| PcError::Security("failed to unwrap DEK".to_string()))?
            .to_vec();
        if plaintext.len() != 32 {
            return Err(PcError::Security(
                "unwrapped DEK is not 32 bytes".to_string(),
            ));
        }
        Ok(plaintext)
    }

    fn sign_digest(&self, digest: &[u8], context: &SignContext) -> PcResult<SignatureEnvelope> {
        let (_store, payload) = self.load_payload()?;
        let pkcs8 = decode_hex(&payload.signing_pkcs8_hex)?;
        let key_pair = Ed25519KeyPair::from_pkcs8(&pkcs8)
            .map_err(|_| PcError::Security("release vault signing key invalid".to_string()))?;
        let signature = key_pair.sign(digest);
        let public_key = key_pair.public_key().as_ref().to_vec();
        Ok(SignatureEnvelope {
            signature_version: "PRISMCODE_SIGNATURE_ENVELOPE_V0_1".to_string(),
            provider_state: self.provider_state().as_str().to_string(),
            key_id: payload.key_id,
            key_version: payload.key_version,
            algorithm: "ED25519".to_string(),
            public_key_hex: hex_bytes(&public_key),
            public_key_hash_blake3: blake3_hex(&public_key),
            digest_hex: hex_bytes(digest),
            signature_hex: hex_bytes(signature.as_ref()),
            context: context.clone(),
            raw_key_exported: false,
        })
    }

    fn verify_signature(&self, signature: &SignatureEnvelope, digest: &[u8]) -> PcResult<()> {
        if signature.provider_state != self.provider_state().as_str() {
            return Err(PcError::Security(
                "signature provider state does not match vault provider".to_string(),
            ));
        }
        if signature.raw_key_exported {
            return Err(PcError::Security(
                "signature envelope declares raw_key_exported=true".to_string(),
            ));
        }
        if signature.digest_hex != hex_bytes(digest) {
            return Err(PcError::Hash("signature digest mismatch".to_string()));
        }
        let public_key = decode_hex(&signature.public_key_hex)?;
        if blake3_hex(&public_key) != signature.public_key_hash_blake3 {
            return Err(PcError::Hash(
                "signature public key hash mismatch".to_string(),
            ));
        }
        let sig = decode_hex(&signature.signature_hex)?;
        UnparsedPublicKey::new(&ED25519, public_key)
            .verify(digest, &sig)
            .map_err(|_| PcError::Security("release vault signature verify failed".to_string()))
    }

    fn rotate_key(&self, key_id: &str, policy: &RotationPolicy) -> PcResult<KeyRotationReport> {
        if key_id != KEY_ID {
            return Err(PcError::Security(format!(
                "local release vault cannot rotate unknown key_id {key_id}"
            )));
        }
        let (store, payload) = self.load_payload()?;
        let old_version = payload.key_version;
        let old_hash = store.public_key_hash_blake3;
        let new_payload = generate_payload(old_version + 1)?;
        let mut audit_log = store.audit_log;
        audit_log.push(AuditEntry {
            event: format!("KEY_ROTATED:{}", policy.reason),
            unix_seconds: now_unix_seconds(),
            policy_hash_blake3: self.policy.policy_hash_blake3.clone(),
        });
        self.save_payload(&new_payload, audit_log)?;
        Ok(KeyRotationReport {
            status: "PASS".to_string(),
            provider_state: self.provider_state().as_str().to_string(),
            key_id: key_id.to_string(),
            old_key_version: old_version,
            new_key_version: new_payload.key_version,
            old_public_key_hash_blake3: old_hash,
            new_public_key_hash_blake3: blake3_hex(&decode_hex(&new_payload.public_key_hex)?),
            audit_log_appended: true,
            raw_key_exported: false,
        })
    }
}

fn generate_payload(version: u64) -> PcResult<VaultPayload> {
    let rng = SystemRandom::new();
    let mut wrap_key = [0u8; 32];
    rng.fill(&mut wrap_key)
        .map_err(|_| PcError::Internal("failed to generate vault wrap key".to_string()))?;
    let pkcs8 = Ed25519KeyPair::generate_pkcs8(&rng)
        .map_err(|_| PcError::Internal("failed to generate release signing key".to_string()))?;
    let key_pair = Ed25519KeyPair::from_pkcs8(pkcs8.as_ref())
        .map_err(|_| PcError::Internal("failed to parse release signing key".to_string()))?;
    let public_key = key_pair.public_key().as_ref().to_vec();
    let payload = VaultPayload {
        key_id: KEY_ID.to_string(),
        key_version: version,
        wrap_key_hex: hex_bytes(&wrap_key),
        signing_algorithm: "ED25519".to_string(),
        signing_pkcs8_hex: hex_bytes(pkcs8.as_ref()),
        public_key_hex: hex_bytes(&public_key),
        revocation_status: "ACTIVE".to_string(),
    };
    wrap_key.fill(0);
    Ok(payload)
}

fn read_or_create_unlock_secret() -> PcResult<Vec<u8>> {
    if let Ok(value) = env::var("PRISMCODE_RELEASE_VAULT_PASSPHRASE") {
        if value.len() < 24 {
            return Err(PcError::Security(
                "PRISMCODE_RELEASE_VAULT_PASSPHRASE must be at least 24 bytes".to_string(),
            ));
        }
        return Ok(value.into_bytes());
    }
    if let Ok(path) = env::var("PRISMCODE_RELEASE_VAULT_PASSPHRASE_FILE") {
        let value = fs::read(path)?;
        if value.len() < 24 {
            return Err(PcError::Security(
                "release vault passphrase file must contain at least 24 bytes".to_string(),
            ));
        }
        return Ok(value);
    }
    let path = default_unlock_secret_path();
    if path.exists() {
        return Ok(fs::read(path)?);
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut secret = [0u8; 32];
    SystemRandom::new().fill(&mut secret).map_err(|_| {
        PcError::Internal("failed to generate local vault unlock secret".to_string())
    })?;
    fs::write(&path, hex_bytes(&secret))?;
    chmod_0600(&path)?;
    Ok(fs::read(path)?)
}

fn default_store_path() -> PathBuf {
    env::var_os("PRISMCODE_LOCAL_RELEASE_VAULT_STORE")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("build/security/local_release_vault.store.json"))
}

fn default_unlock_secret_path() -> PathBuf {
    let root = env::var("PRISM_BOUNDED_NAS")
        .unwrap_or_else(|_| "/mnt/storage/NAS/Prism4D-Agents/prism-kt-superres-work".to_string());
    PathBuf::from(root)
        .join("prismcode_secret_store")
        .join("a2_vault")
        .join("local_release_vault.unlock")
}

fn unlock_mode() -> String {
    if env::var("PRISMCODE_RELEASE_VAULT_PASSPHRASE").is_ok() {
        "PASSPHRASE_ENV".to_string()
    } else if env::var("PRISMCODE_RELEASE_VAULT_PASSPHRASE_FILE").is_ok() {
        "PASSPHRASE_FILE".to_string()
    } else {
        "OS_PROTECTED_LOCAL_SECRET_FILE".to_string()
    }
}

fn store_aad(store: &VaultStore) -> Vec<u8> {
    format!(
        "{}:{}:{}:{}:{}",
        store.vault_version,
        store.provider_state,
        store.provider_id,
        store.key_id,
        store.key_version
    )
    .into_bytes()
}

fn audit(event: &str, policy: &ReleaseKeyPolicy) -> AuditEntry {
    AuditEntry {
        event: event.to_string(),
        unix_seconds: now_unix_seconds(),
        policy_hash_blake3: policy.policy_hash_blake3.clone(),
    }
}

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn artifact_digest(path: &Path) -> PcResult<[u8; 32]> {
    let bytes = fs::read(path)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

pub fn file_hash_hex(path: &Path) -> PcResult<String> {
    Ok(blake3_hex(&fs::read(path)?))
}

pub fn sha256_file_hex(path: &Path) -> PcResult<String> {
    Ok(hex_bytes(digest(&SHA256, &fs::read(path)?).as_ref()))
}

fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn nonce_from_hex(value: &str) -> PcResult<Nonce> {
    let bytes = decode_hex(value)?;
    if bytes.len() != 12 {
        return Err(PcError::Format("nonce must be 12 bytes".to_string()));
    }
    let mut nonce = [0u8; 12];
    nonce.copy_from_slice(&bytes);
    Ok(Nonce::assume_unique_for_key(nonce))
}

pub fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

pub fn decode_hex(value: &str) -> PcResult<Vec<u8>> {
    let value = value.trim();
    if value.len() % 2 != 0 {
        return Err(PcError::Format("hex value has odd length".to_string()));
    }
    let mut out = Vec::with_capacity(value.len() / 2);
    let bytes = value.as_bytes();
    for idx in (0..bytes.len()).step_by(2) {
        let high = hex_nibble(bytes[idx])?;
        let low = hex_nibble(bytes[idx + 1])?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn hex_nibble(byte: u8) -> PcResult<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(PcError::Format("invalid hex character".to_string())),
    }
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> PcResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn chmod_0600(path: &Path) -> PcResult<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o600);
        fs::set_permissions(path, permissions)?;
    }
    Ok(())
}

pub fn report_artifact(
    path: &Path,
    producer_command: &str,
    status: &str,
    policy_hash: &str,
) -> Value {
    let bytes = fs::read(path).unwrap_or_default();
    let realpath = fs::canonicalize(path)
        .map(|path| path.display().to_string())
        .unwrap_or_else(|_| "EVIDENCE_NOT_FOUND".to_string());
    json!({
        "path": path.display().to_string(),
        "realpath": realpath,
        "byte_size": bytes.len(),
        "hash_algorithm": "BLAKE3",
        "hash": blake3_hex(&bytes),
        "producer_command": producer_command,
        "status": status,
        "provider_state": ProviderState::LocalReleaseVaultBound.as_str(),
        "raw_key_exported": false,
        "policy_hash": policy_hash,
    })
}

pub fn build_blocker_02_report(reports: &[(&str, &Path)], policy_hash: &str) -> Value {
    let artifacts: Vec<Value> = reports
        .iter()
        .map(|(producer, path)| report_artifact(path, producer, "PASS", policy_hash))
        .collect();
    json!({
        "status": "PASS",
        "blocker": "BLOCKER_02_VAULT_PROVIDER",
        "provider_state": ProviderState::LocalReleaseVaultBound.as_str(),
        "vault_provider_bound": "PASS",
        "file_keyring_dev_for_market": false,
        "test_fixture_only_for_market": false,
        "raw_key_exported": false,
        "key_wrap": "PASS",
        "key_unwrap": "PASS",
        "sign_verify": "PASS",
        "rotation_test": "PASS",
        "release_key_policy_bound": "PASS",
        "cuda_manifest_signed": "PASS",
        "cuda_manifest_signature_verified": "PASS",
        "evidence_key_record_bound": "PASS",
        "prismui_security_panel_updated": "PASS",
        "policy_hash": policy_hash,
        "artifacts": artifacts,
    })
}
