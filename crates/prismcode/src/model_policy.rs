use crate::binary_runtime::{PcError, PcResult};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize)]
pub struct ModelRuntimePolicy {
    pub path: PathBuf,
    pub model_id: String,
    pub model_artifact_source: String,
    pub bind_address: String,
    pub external_egress_allowed: bool,
    pub remote_api_allowed: bool,
    pub canonical_mutation_allowed: bool,
    pub output_classification: String,
    pub airlock_required: bool,
    pub secret_access_allowed: bool,
    pub vault_access_allowed: bool,
    pub release_key_access_allowed: bool,
    pub source_tree_allowed: bool,
    pub cargo_allowed: bool,
    pub nas_model_path_allowed: bool,
    pub host_cache_allowed: bool,
    pub hf_cache_allowed: bool,
    pub ollama_cache_allowed: bool,
    pub host_path_lookup_allowed: bool,
    pub path_env_allowed: bool,
    pub runtime_hash_required: bool,
    pub model_hash_required: bool,
    pub prompt_hash_required: bool,
    pub output_hash_required: bool,
    pub model_load_timeout_seconds: u64,
    pub model_smoke_timeout_seconds: u64,
    pub kill_on_timeout: bool,
    pub reap_child_processes: bool,
    pub runtime_must_report_loaded_model_digest: bool,
    pub nonce_prompt_required: bool,
    pub canned_response_allowed: bool,
    pub negative_controls_required: bool,
    pub status: String,
}

impl ModelRuntimePolicy {
    pub fn load(path: &Path) -> PcResult<Self> {
        let text = fs::read_to_string(path)?;
        let policy = Self {
            path: path.to_path_buf(),
            model_id: toml_string(&text, "model_id")
                .unwrap_or_else(|| "openai_gpt_oss_20b".to_string()),
            model_artifact_source: toml_string(&text, "model_artifact_source")
                .unwrap_or_else(|| "BUNDLED_MODEL_PACK".to_string()),
            bind_address: toml_string(&text, "bind_address")
                .unwrap_or_else(|| "127.0.0.1".to_string()),
            external_egress_allowed: toml_bool(&text, "external_egress_allowed").unwrap_or(false),
            remote_api_allowed: toml_bool(&text, "remote_api_allowed").unwrap_or(false),
            canonical_mutation_allowed: toml_bool(&text, "canonical_mutation_allowed")
                .unwrap_or(false),
            output_classification: toml_string(&text, "output_classification")
                .unwrap_or_else(|| "MODEL_GENERATED_UNVERIFIED".to_string()),
            airlock_required: toml_bool(&text, "airlock_required").unwrap_or(true),
            secret_access_allowed: toml_bool(&text, "secret_access_allowed").unwrap_or(false),
            vault_access_allowed: toml_bool(&text, "vault_access_allowed").unwrap_or(false),
            release_key_access_allowed: toml_bool(&text, "release_key_access_allowed")
                .unwrap_or(false),
            source_tree_allowed: toml_bool(&text, "source_tree_allowed").unwrap_or(false),
            cargo_allowed: toml_bool(&text, "cargo_allowed").unwrap_or(false),
            nas_model_path_allowed: toml_bool(&text, "nas_model_path_allowed").unwrap_or(false),
            host_cache_allowed: toml_bool(&text, "host_cache_allowed").unwrap_or(false),
            hf_cache_allowed: toml_bool(&text, "hf_cache_allowed").unwrap_or(false),
            ollama_cache_allowed: toml_bool(&text, "ollama_cache_allowed").unwrap_or(false),
            host_path_lookup_allowed: toml_bool(&text, "host_path_lookup_allowed").unwrap_or(false),
            path_env_allowed: toml_bool(&text, "path_env_allowed").unwrap_or(false),
            runtime_hash_required: toml_bool(&text, "runtime_hash_required").unwrap_or(true),
            model_hash_required: toml_bool(&text, "model_hash_required").unwrap_or(true),
            prompt_hash_required: toml_bool(&text, "prompt_hash_required").unwrap_or(true),
            output_hash_required: toml_bool(&text, "output_hash_required").unwrap_or(true),
            model_load_timeout_seconds: toml_u64(&text, "model_load_timeout_seconds")
                .unwrap_or(120),
            model_smoke_timeout_seconds: toml_u64(&text, "model_smoke_timeout_seconds")
                .unwrap_or(180),
            kill_on_timeout: toml_bool(&text, "kill_on_timeout").unwrap_or(true),
            reap_child_processes: toml_bool(&text, "reap_child_processes").unwrap_or(true),
            runtime_must_report_loaded_model_digest: toml_bool(
                &text,
                "runtime_must_report_loaded_model_digest",
            )
            .unwrap_or(true),
            nonce_prompt_required: toml_bool(&text, "nonce_prompt_required").unwrap_or(true),
            canned_response_allowed: toml_bool(&text, "canned_response_allowed").unwrap_or(false),
            negative_controls_required: toml_bool(&text, "negative_controls_required")
                .unwrap_or(true),
            status: "UNVALIDATED".to_string(),
        };
        policy.validate()
    }

    pub fn validate(mut self) -> PcResult<Self> {
        let mut issues = Vec::new();
        // The policy model_id must name an admitted catalog model. This replaced
        // the single hard `== "openai_gpt_oss_20b"` check (see ADR-0016); an id
        // outside `model_catalog::ALLOWED_CANONICAL_MODELS` still fails closed.
        if !crate::model_catalog::is_allowed_canonical(&self.model_id) {
            issues.push(format!(
                "policy model_id must be an admitted catalog model, got {}",
                self.model_id
            ));
        }
        if !matches!(
            self.model_artifact_source.as_str(),
            "BUNDLED_MODEL_PACK" | "INSTALLED_PORTABLE_MODEL_PACK" | "TEST_FIXTURE_ONLY"
        ) {
            issues.push(
                "model_artifact_source must be BUNDLED_MODEL_PACK, INSTALLED_PORTABLE_MODEL_PACK, or TEST_FIXTURE_ONLY"
                    .to_string(),
            );
        }
        if self.bind_address != "127.0.0.1" {
            issues.push("model runtime must bind only to 127.0.0.1".to_string());
        }
        if self.external_egress_allowed || self.remote_api_allowed {
            issues.push("external model runtime egress and remote APIs are forbidden".to_string());
        }
        if self.canonical_mutation_allowed {
            issues.push("canonical mutation must remain forbidden".to_string());
        }
        if self.output_classification != "MODEL_GENERATED_UNVERIFIED" {
            issues.push("output classification must be MODEL_GENERATED_UNVERIFIED".to_string());
        }
        if !self.airlock_required {
            issues.push("airlock_required must be true".to_string());
        }
        if self.secret_access_allowed
            || self.vault_access_allowed
            || self.release_key_access_allowed
        {
            issues.push(
                "model runtime secret/vault/release-key access must be forbidden".to_string(),
            );
        }
        if self.source_tree_allowed
            || self.cargo_allowed
            || self.nas_model_path_allowed
            || self.host_cache_allowed
            || self.hf_cache_allowed
            || self.ollama_cache_allowed
            || self.host_path_lookup_allowed
            || self.path_env_allowed
        {
            issues.push(
                "model runtime policy must forbid NAS/cache/source tree, cargo, PATH, and host lookup"
                    .to_string(),
            );
        }
        if !self.runtime_hash_required
            || !self.model_hash_required
            || !self.prompt_hash_required
            || !self.output_hash_required
            || !self.runtime_must_report_loaded_model_digest
        {
            issues.push(
                "runtime/model/prompt/output hashes and loaded model digest are all required"
                    .to_string(),
            );
        }
        if self.model_load_timeout_seconds == 0 || self.model_smoke_timeout_seconds == 0 {
            issues.push("model runtime load/smoke timeouts must be positive".to_string());
        }
        if self.model_load_timeout_seconds > 120 || self.model_smoke_timeout_seconds > 180 {
            issues.push(
                "model runtime load/smoke timeouts must not exceed portable release bounds"
                    .to_string(),
            );
        }
        if !self.kill_on_timeout || !self.reap_child_processes {
            issues.push("model runtime timeout must kill and reap child processes".to_string());
        }
        if !self.nonce_prompt_required || self.canned_response_allowed {
            issues.push("nonce prompt is required and canned responses are forbidden".to_string());
        }
        if !self.negative_controls_required {
            issues.push("model runtime negative controls are required".to_string());
        }
        if !issues.is_empty() {
            return Err(PcError::Security(format!(
                "model runtime policy invalid: {}",
                issues.join("; ")
            )));
        }
        self.status = "PASS".to_string();
        Ok(self)
    }
}

fn toml_string(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let line = strip_comment(line).trim();
        let (left, right) = line.split_once('=')?;
        if left.trim() != key {
            return None;
        }
        Some(right.trim().trim_matches('"').to_string())
    })
}

fn toml_bool(text: &str, key: &str) -> Option<bool> {
    text.lines().find_map(|line| {
        let line = strip_comment(line).trim();
        let (left, right) = line.split_once('=')?;
        if left.trim() != key {
            return None;
        }
        match right.trim() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    })
}

fn toml_u64(text: &str, key: &str) -> Option<u64> {
    text.lines().find_map(|line| {
        let line = strip_comment(line).trim();
        let (left, right) = line.split_once('=')?;
        if left.trim() != key {
            return None;
        }
        right.trim().parse::<u64>().ok()
    })
}

fn strip_comment(line: &str) -> &str {
    line.split_once('#').map(|(head, _)| head).unwrap_or(line)
}
