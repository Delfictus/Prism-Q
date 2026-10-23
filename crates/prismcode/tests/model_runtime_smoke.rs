use prismcode::model_runtime_smoke::{run_model_runtime_smoke, ModelRuntimeSmokeRequest};
use serde_json::json;
use std::fs;
use std::path::Path;

#[test]
fn bundled_modeld_smoke_passes_with_small_verified_artifact() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    fs::create_dir_all(&model_dir).unwrap();
    write_small_model_files(&model_dir);
    let registry = write_registry(tmp.path(), &model_dir);
    let policy = write_policy(tmp.path());
    let modeld = std::env::var("CARGO_BIN_EXE_prism-modeld").unwrap_or_else(|_| {
        let exe = std::env::current_exe().unwrap();
        exe.parent()
            .and_then(|deps| deps.parent())
            .unwrap()
            .join("prism-modeld")
            .display()
            .to_string()
    });
    std::env::set_var("PRISMCODE_MODELD_BIN", modeld);
    std::env::set_var("TMPDIR", tmp.path());
    let out = tmp.path().join("model_runtime_smoke_report.json");
    let report = run_model_runtime_smoke(&ModelRuntimeSmokeRequest {
        registry,
        policy,
        model_id: "openai_gpt_oss_20b".to_string(),
        out,
    })
    .unwrap();
    assert_eq!(report["status"], "PASS");
    assert_eq!(report["model_runtime_active"], true);
    assert_eq!(report["runtime_bind_address"], "127.0.0.1");
    assert_eq!(
        report["output_classification"],
        "MODEL_GENERATED_UNVERIFIED"
    );
    assert_eq!(report["canonical_mutation_allowed"], false);
}

fn write_policy(root: &Path) -> std::path::PathBuf {
    let path = root.join("model_runtime_policy.toml");
    fs::write(
        &path,
        r#"
[model]
model_id = "openai_gpt_oss_20b"
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
"#,
    )
    .unwrap();
    path
}

fn write_small_model_files(model_dir: &Path) {
    let header = br#"{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut shard = Vec::new();
    shard.extend_from_slice(&(header.len() as u64).to_le_bytes());
    shard.extend_from_slice(header);
    shard.extend_from_slice(&1.0f32.to_le_bytes());
    fs::write(model_dir.join("model-00000-of-00001.safetensors"), shard).unwrap();
    fs::write(
        model_dir.join("config.json"),
        br#"{"model_type":"gpt_oss"}"#,
    )
    .unwrap();
    fs::write(model_dir.join("tokenizer.json"), br#"{"version":"test"}"#).unwrap();
}

fn write_registry(root: &Path, model_dir: &Path) -> std::path::PathBuf {
    let files = [
        "model-00000-of-00001.safetensors",
        "config.json",
        "tokenizer.json",
    ]
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
        "registry_version": "TEST_MODEL_RUNTIME",
        "rules": {"canonical_state_mutation_allowed": false},
        "installed_models": [{
            "model_id": "openai/gpt-oss-20b",
            "local_path": model_dir,
            "install_allowed": true,
            "hardware_gated": false,
            "weights_hash_verified": true,
            "files": files
        }],
        "hardware_gated_models": [{
            "model_id": "Qwen/Qwen3-Coder-Next",
            "hardware_gated": true,
            "install_allowed": false,
            "canonical_state_mutation_allowed": false
        }]
    });
    let path = root.join("registry.json");
    fs::write(&path, serde_json::to_vec_pretty(&registry).unwrap()).unwrap();
    path
}
