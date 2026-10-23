use crate::binary_runtime::{PcError, PcResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReleaseKeyPolicy {
    pub name: String,
    pub market_release_allowed: bool,
    pub dev_file_keyring_allowed: bool,
    pub test_fixture_allowed: bool,
    pub raw_key_export_allowed: bool,
    pub rotation_required: bool,
    pub audit_log_required: bool,
    pub required_state: String,
    pub wrap_dek_required: bool,
    pub unwrap_dek_required: bool,
    pub sign_required: bool,
    pub verify_required: bool,
    pub rotate_required: bool,
    pub cuda_manifest_signing_required: bool,
    pub release_manifest_signing_required: bool,
    pub evidence_ledger_signing_required: bool,
    pub seal_chain_binding_required: bool,
    pub policy_hash_blake3: String,
}

impl ReleaseKeyPolicy {
    pub fn load(path: &Path) -> PcResult<Self> {
        let text = fs::read_to_string(path)
            .map_err(|err| PcError::Security(format!("release key policy missing: {err}")))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> PcResult<Self> {
        let mut section = String::new();
        let mut values = BTreeMap::new();
        for raw_line in text.lines() {
            let line = raw_line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if line.starts_with('[') && line.ends_with(']') {
                section = line
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .trim()
                    .to_string();
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                return Err(PcError::Format(format!(
                    "invalid release policy line: {raw_line}"
                )));
            };
            let key = format!("{}.{}", section, key.trim());
            values.insert(key, value.trim().trim_matches('"').to_string());
        }

        let policy = Self {
            name: string_value(&values, "policy.name")?,
            market_release_allowed: bool_value(&values, "policy.market_release_allowed")?,
            dev_file_keyring_allowed: bool_value(&values, "policy.dev_file_keyring_allowed")?,
            test_fixture_allowed: bool_value(&values, "policy.test_fixture_allowed")?,
            raw_key_export_allowed: bool_value(&values, "policy.raw_key_export_allowed")?,
            rotation_required: bool_value(&values, "policy.rotation_required")?,
            audit_log_required: bool_value(&values, "policy.audit_log_required")?,
            required_state: string_value(&values, "provider.required_state")?,
            wrap_dek_required: bool_value(&values, "operations.wrap_dek_required")?,
            unwrap_dek_required: bool_value(&values, "operations.unwrap_dek_required")?,
            sign_required: bool_value(&values, "operations.sign_required")?,
            verify_required: bool_value(&values, "operations.verify_required")?,
            rotate_required: bool_value(&values, "operations.rotate_required")?,
            cuda_manifest_signing_required: bool_value(
                &values,
                "artifacts.cuda_manifest_signing_required",
            )?,
            release_manifest_signing_required: bool_value(
                &values,
                "artifacts.release_manifest_signing_required",
            )?,
            evidence_ledger_signing_required: bool_value(
                &values,
                "artifacts.evidence_ledger_signing_required",
            )?,
            seal_chain_binding_required: bool_value(
                &values,
                "artifacts.seal_chain_binding_required",
            )?,
            policy_hash_blake3: blake3::hash(text.as_bytes()).to_hex().to_string(),
        };
        policy.validate_market_mode()?;
        Ok(policy)
    }

    pub fn validate_market_mode(&self) -> PcResult<()> {
        if !self.market_release_allowed {
            return Err(PcError::Security(
                "release key policy does not allow market release".to_string(),
            ));
        }
        if self.dev_file_keyring_allowed {
            return Err(PcError::Security(
                "release key policy illegally allows FILE_KEYRING_DEV".to_string(),
            ));
        }
        if self.test_fixture_allowed {
            return Err(PcError::Security(
                "release key policy illegally allows TEST_FIXTURE_ONLY".to_string(),
            ));
        }
        if self.raw_key_export_allowed {
            return Err(PcError::Security(
                "release key policy illegally allows raw key export".to_string(),
            ));
        }
        if !matches!(
            self.required_state.as_str(),
            "LOCAL_RELEASE_VAULT_BOUND" | "OPENBAO_OR_VAULT_BOUND"
        ) {
            return Err(PcError::Security(format!(
                "release key policy requires unsupported provider state {}",
                self.required_state
            )));
        }
        for (name, required) in [
            ("rotation_required", self.rotation_required),
            ("audit_log_required", self.audit_log_required),
            ("wrap_dek_required", self.wrap_dek_required),
            ("unwrap_dek_required", self.unwrap_dek_required),
            ("sign_required", self.sign_required),
            ("verify_required", self.verify_required),
            ("rotate_required", self.rotate_required),
            (
                "cuda_manifest_signing_required",
                self.cuda_manifest_signing_required,
            ),
        ] {
            if !required {
                return Err(PcError::Security(format!(
                    "release key policy missing required field {name}=true"
                )));
            }
        }
        Ok(())
    }
}

fn string_value(values: &BTreeMap<String, String>, key: &str) -> PcResult<String> {
    values
        .get(key)
        .cloned()
        .ok_or_else(|| PcError::Format(format!("release key policy missing {key}")))
}

fn bool_value(values: &BTreeMap<String, String>, key: &str) -> PcResult<bool> {
    match values.get(key).map(String::as_str) {
        Some("true") => Ok(true),
        Some("false") => Ok(false),
        Some(other) => Err(PcError::Format(format!(
            "release key policy field {key} is not boolean: {other}"
        ))),
        None => Err(PcError::Format(format!("release key policy missing {key}"))),
    }
}
