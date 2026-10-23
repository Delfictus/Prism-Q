use prismcode::model_runtime::load_model_artifact;
use serde_json::json;
use std::fs;
use std::path::Path;

#[test]
fn small_registered_artifact_loads_and_verifies_hashes() {
    let tmp = tempfile::tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    fs::create_dir_all(&model_dir).unwrap();
    write_small_model_files(&model_dir);
    let registry = write_registry(tmp.path(), &model_dir);
    let proof = load_model_artifact(&registry, "openai/gpt-oss-20b").unwrap();
    assert_eq!(proof.status, "PASS");
    assert!(proof.all_model_hashes_verified);
    assert!(proof.shard_hashes_verified);
    assert!(proof.config_hash_verified);
    assert!(proof.tokenizer_hash_verified);
    assert!(!proof.loaded_model_digest.is_empty());
}

pub fn write_small_model_files(model_dir: &Path) {
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

pub fn write_registry(root: &Path, model_dir: &Path) -> std::path::PathBuf {
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
            "runtime_binding_allowed": false,
            "runtime_bound": false,
            "runtime_smoke_status": "FAILED",
            "canonical_state_mutation_allowed": false,
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
