use crate::binary_runtime::{PcError, PcResult};
use crate::key_provider_local_release_vault::{write_json, LocalReleaseVault};
use crate::release_key_policy::ReleaseKeyPolicy;
use ring::signature::{UnparsedPublicKey, ED25519};
use serde_json::{json, Value};
use std::path::Path;

pub fn sign_artifact_report(
    artifact: &Path,
    policy_path: &Path,
    sig_out: &Path,
) -> PcResult<Value> {
    let policy = ReleaseKeyPolicy::load(policy_path)?;
    let vault = LocalReleaseVault::open_or_create(policy)?;
    let report = vault.sign_artifact(artifact, sig_out)?;
    Ok(report)
}

pub fn verify_artifact_report(artifact: &Path, sig_path: &Path) -> PcResult<Value> {
    if !sig_path.exists() {
        return Err(PcError::Security(format!(
            "artifact signature missing: {}",
            sig_path.display()
        )));
    }
    let signature_file: Value = serde_json::from_slice(&std::fs::read(sig_path)?)?;
    let signature = signature_file.get("signature").ok_or_else(|| {
        PcError::Security("signature file missing signature envelope".to_string())
    })?;
    if signature.get("raw_key_exported").and_then(Value::as_bool) != Some(false) {
        return Err(PcError::Security(
            "signature envelope declares raw_key_exported=true".to_string(),
        ));
    }
    if signature.get("provider_state").and_then(Value::as_str) != Some("LOCAL_RELEASE_VAULT_BOUND")
    {
        return Err(PcError::Security(
            "signature provider is not LOCAL_RELEASE_VAULT_BOUND".to_string(),
        ));
    }
    let digest = crate::key_provider_local_release_vault::artifact_digest(artifact)?;
    let digest_hex = crate::key_provider_local_release_vault::hex_bytes(&digest);
    if signature.get("digest_hex").and_then(Value::as_str) != Some(digest_hex.as_str()) {
        return Err(PcError::Hash("signature digest mismatch".to_string()));
    }
    let public_key_hex = signature
        .get("public_key_hex")
        .and_then(Value::as_str)
        .ok_or_else(|| PcError::Security("signature missing public_key_hex".to_string()))?;
    let public_key = crate::key_provider_local_release_vault::decode_hex(public_key_hex)?;
    let public_key_hash = blake3::hash(&public_key).to_hex().to_string();
    if signature
        .get("public_key_hash_blake3")
        .and_then(Value::as_str)
        != Some(public_key_hash.as_str())
    {
        return Err(PcError::Hash(
            "signature public key hash mismatch".to_string(),
        ));
    }
    if crate::trust_root::is_release_pinned() {
        crate::trust_root::require_signature_root_hash(&public_key_hash)?;
    }
    let sig = signature
        .get("signature_hex")
        .and_then(Value::as_str)
        .ok_or_else(|| PcError::Security("signature missing signature_hex".to_string()))
        .and_then(crate::key_provider_local_release_vault::decode_hex)?;
    UnparsedPublicKey::new(&ED25519, public_key)
        .verify(&digest, &sig)
        .map_err(|_| PcError::Security("artifact signature verification failed".to_string()))?;
    let expected_hash = crate::key_provider_local_release_vault::file_hash_hex(artifact)?;
    if signature_file
        .get("artifact_hash_blake3")
        .and_then(Value::as_str)
        != Some(expected_hash.as_str())
    {
        return Err(PcError::Hash(
            "artifact hash in signature file does not match artifact".to_string(),
        ));
    }
    Ok(json!({
        "status": "PASS",
        "signature_status": "SIGNATURE_VERIFIED",
        "provider_state": "LOCAL_RELEASE_VAULT_BOUND",
        "artifact": artifact.display().to_string(),
        "artifact_hash_blake3": expected_hash,
        "sig": sig_path.display().to_string(),
        "signature_file_hash_blake3": crate::key_provider_local_release_vault::file_hash_hex(sig_path)?,
        "raw_key_exported": false,
        "policy_hash": signature_policy_hash(sig_path).unwrap_or_else(|| "UNKNOWN".to_string()),
    }))
}

pub fn write_signature_report(out: &Path, mut report: Value) -> PcResult<()> {
    if let Some(obj) = report.as_object_mut() {
        obj.insert("report_path".to_string(), json!(out.display().to_string()));
    }
    write_json(out, &report)
}

fn signature_policy_hash(sig_path: &Path) -> Option<String> {
    let sig: Value = serde_json::from_slice(&std::fs::read(sig_path).ok()?).ok()?;
    sig.get("policy_hash")
        .and_then(Value::as_str)
        .map(str::to_string)
}
