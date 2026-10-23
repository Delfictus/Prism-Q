//! Champion PLLM promotion (ADR-0016) coverage: the model-id gate is now an
//! allow-list, the policy admits the champion id, and a fused `.pt` checkpoint
//! artifact class verifies by BLAKE3 + size without a safetensors header — while
//! unknown ids and gpt-oss behaviour are unchanged.

use prismcode::model_catalog::{
    self, ModelArtifactClass, ModelBridgeKind, PRISM_CHAMPION_PLLM_QWEN3_8B,
};
use prismcode::model_policy::ModelRuntimePolicy;
use prismcode::model_runtime::load_model_artifact;
use prismcode::model_runtime_champion::{ChampionBridge, ChampionBridgeConfig, ChampionEndpoint};
use serde_json::json;
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn catalog_admits_champion_and_default_but_rejects_unknown() {
    assert!(model_catalog::is_allowed_canonical("openai_gpt_oss_20b"));
    assert!(model_catalog::is_allowed_canonical(
        "prism_champion_pllm_qwen3_8b"
    ));
    assert!(!model_catalog::is_allowed_canonical("some_random_model"));

    let champ = model_catalog::require_canonical("prism_champion_pllm_qwen3_8b").unwrap();
    assert_eq!(champ.registry_id, "prism/champion-pllm-qwen3-8b");
    assert_eq!(champ.artifact_class, ModelArtifactClass::FusedCheckpointPt);
    assert_eq!(champ.bridge, ModelBridgeKind::PythonDevboxBridge);

    // Unknown id fails closed with the historical rejection text.
    let err = model_catalog::require_canonical("nope")
        .unwrap_err()
        .to_string();
    assert!(err.contains("unsupported model id for bounded runtime"));
}

#[test]
fn default_constants_still_track_gpt_oss() {
    assert_eq!(
        prismcode::model_runtime::MODEL_ID_CANONICAL,
        "openai_gpt_oss_20b"
    );
    assert_eq!(
        prismcode::model_runtime::MODEL_ID_REGISTRY,
        "openai/gpt-oss-20b"
    );
}

#[test]
fn policy_admits_champion_model_id() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("champion_policy.toml");
    fs::write(&path, champion_policy_toml()).unwrap();
    let policy = ModelRuntimePolicy::load(&path).unwrap();
    assert_eq!(policy.status, "PASS");
    assert_eq!(policy.model_id, "prism_champion_pllm_qwen3_8b");
}

#[test]
fn policy_still_rejects_unknown_model_id() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("bad_model_policy.toml");
    fs::write(
        &path,
        champion_policy_toml().replace("prism_champion_pllm_qwen3_8b", "totally_unknown_model"),
    )
    .unwrap();
    let err = ModelRuntimePolicy::load(&path).unwrap_err().to_string();
    assert!(err.contains("admitted catalog model"), "unexpected: {err}");
}

#[test]
fn fused_pt_checkpoint_artifact_verifies_by_hash_without_safetensors() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("champion_model");
    fs::create_dir_all(&model_dir).unwrap();
    // Stand-in fused checkpoint + a config sidecar (small; real ckpt is 17.4 GB).
    fs::write(
        model_dir.join("ckpt_step108500.pt"),
        b"FUSED_CHAMPION_STANDIN",
    )
    .unwrap();
    fs::write(
        model_dir.join("champion_config.json"),
        br#"{"schema":"champion"}"#,
    )
    .unwrap();
    let registry = write_champion_registry(tmp.path(), &model_dir);

    let proof = load_model_artifact(&registry, "prism/champion-pllm-qwen3-8b").unwrap();
    assert_eq!(proof.status, "PASS");
    assert_eq!(proof.artifact_class, "FUSED_CHECKPOINT_PT");
    assert!(proof.all_model_hashes_verified);
    // The fused-pt class passed WITHOUT the safetensors-class structural files
    // (no tokenizer.json, no `config.json`) — only BLAKE3 + size of the listed
    // files (the `.pt` + its config sidecar) were required.
    assert!(!proof.tokenizer_hash_verified);
    assert!(!proof.config_hash_verified);
}

#[test]
fn fused_pt_checkpoint_fails_on_hash_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("champion_model");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("ckpt_step108500.pt"),
        b"FUSED_CHAMPION_STANDIN",
    )
    .unwrap();
    fs::write(
        model_dir.join("champion_config.json"),
        br#"{"schema":"champion"}"#,
    )
    .unwrap();
    // Registry lists a wrong BLAKE3 -> must NOT pass.
    let registry = write_champion_registry_with_bad_hash(tmp.path(), &model_dir);
    let proof = load_model_artifact(&registry, "prism/champion-pllm-qwen3-8b").unwrap();
    assert_eq!(proof.status, "FAIL");
    assert!(!proof.all_model_hashes_verified);
}

#[test]
fn champion_bridge_enforces_loopback_only() {
    // A non-loopback host is rejected before any socket is opened.
    let mut cfg = ChampionBridgeConfig::default();
    cfg.host = "10.0.0.5".to_string();
    let err = ChampionBridge::attach(PRISM_CHAMPION_PLLM_QWEN3_8B, cfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("loopback"), "unexpected: {err}");

    // The default config is loopback and yields the devbox base URL.
    let cfg = ChampionBridgeConfig::default();
    assert_eq!(cfg.base_url().unwrap(), "http://127.0.0.1:8711");
    let bridge = ChampionBridge::attach(PRISM_CHAMPION_PLLM_QWEN3_8B, cfg).unwrap();
    assert_eq!(bridge.base_url(), "http://127.0.0.1:8711");
    assert_eq!(bridge.model().canonical_id, "prism_champion_pllm_qwen3_8b");
}

#[test]
fn advisory_envelope_forces_unverified_and_denies_mutation() {
    use prismcode::model_runtime_champion::ChampionAdvisoryEnvelope;
    // Even a hostile payload claiming mutation is allowed gets forced to deny.
    let hostile = json!({"canonical_mutation_allowed": true, "text": "hi"});
    let env = ChampionAdvisoryEnvelope::wrap(
        &PRISM_CHAMPION_PLLM_QWEN3_8B,
        ChampionEndpoint::Generate,
        &json!({"prompt": "hi"}),
        hostile,
    )
    .unwrap();
    assert_eq!(env.role, "advisory");
    assert_eq!(
        env.model_output_classification,
        "MODEL_GENERATED_UNVERIFIED"
    );
    assert!(!env.canonical_mutation_allowed);
    assert_eq!(env.model_id, "prism_champion_pllm_qwen3_8b");
    assert!(!env.response_hash_blake3.is_empty());
}

#[cfg(not(feature = "champion_bridge"))]
#[test]
fn champion_bridge_live_path_disabled_by_default() {
    let err = ChampionBridge::spawn(
        PRISM_CHAMPION_PLLM_QWEN3_8B,
        ChampionBridgeConfig::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("champion bridge disabled"),
        "unexpected: {err}"
    );
}

fn champion_policy_toml() -> String {
    r#"
[model]
model_id = "prism_champion_pllm_qwen3_8b"
model_artifact_source = "TEST_FIXTURE_ONLY"
canonical_mutation_allowed = false
output_classification = "MODEL_GENERATED_UNVERIFIED"
airlock_required = true

[network]
bind_address = "127.0.0.1"
external_egress_allowed = false
remote_api_allowed = false

[security]
secret_access_allowed = false
vault_access_allowed = false
release_key_access_allowed = false

[runtime]
nas_model_path_allowed = false
host_cache_allowed = false
hf_cache_allowed = false
ollama_cache_allowed = false
source_tree_allowed = false
cargo_allowed = false
host_path_lookup_allowed = false
path_env_allowed = false
runtime_hash_required = true
model_hash_required = true
prompt_hash_required = true
output_hash_required = true
model_load_timeout_seconds = 10
model_smoke_timeout_seconds = 10
kill_on_timeout = true
reap_child_processes = true
runtime_must_report_loaded_model_digest = true

[acceptance]
nonce_prompt_required = true
canned_response_allowed = false
negative_controls_required = true
"#
    .to_string()
}

fn write_champion_registry(root: &Path, model_dir: &Path) -> PathBuf {
    let files = ["ckpt_step108500.pt", "champion_config.json"]
        .into_iter()
        .map(|name| {
            let bytes = fs::read(model_dir.join(name)).unwrap();
            json!({
                "path": name,
                "bytes": bytes.len(),
                "blake3": blake3::hash(&bytes).to_hex().to_string(),
            })
        })
        .collect::<Vec<_>>();
    let registry = json!({
        "registry_version": "TEST_CHAMPION_RUNTIME",
        "rules": {"canonical_state_mutation_allowed": false},
        "installed_models": [{
            "model_id": "prism/champion-pllm-qwen3-8b",
            "artifact_class": "FUSED_CHECKPOINT_PT",
            "local_path": model_dir,
            "install_allowed": true,
            "hardware_gated": false,
            "canonical_state_mutation_allowed": false,
            "files": files
        }]
    });
    let path = root.join("champion_registry.json");
    fs::write(&path, serde_json::to_vec_pretty(&registry).unwrap()).unwrap();
    path
}

fn write_champion_registry_with_bad_hash(root: &Path, model_dir: &Path) -> PathBuf {
    let ckpt_bytes = fs::read(model_dir.join("ckpt_step108500.pt")).unwrap();
    let cfg_bytes = fs::read(model_dir.join("champion_config.json")).unwrap();
    let registry = json!({
        "registry_version": "TEST_CHAMPION_RUNTIME",
        "rules": {"canonical_state_mutation_allowed": false},
        "installed_models": [{
            "model_id": "prism/champion-pllm-qwen3-8b",
            "artifact_class": "FUSED_CHECKPOINT_PT",
            "local_path": model_dir,
            "files": [
                {"path": "ckpt_step108500.pt", "bytes": ckpt_bytes.len(), "blake3": "deadbeef"},
                {"path": "champion_config.json", "bytes": cfg_bytes.len(),
                 "blake3": blake3::hash(&cfg_bytes).to_hex().to_string()}
            ]
        }]
    });
    let path = root.join("champion_registry_bad.json");
    fs::write(&path, serde_json::to_vec_pretty(&registry).unwrap()).unwrap();
    path
}
