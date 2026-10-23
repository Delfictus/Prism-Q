use prismcode::model_airlock::reject_forbidden_model_action;
use prismcode::model_policy::ModelRuntimePolicy;
use prismcode::model_runtime::load_model_artifact_with_policy;
use serde_json::json;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

#[test]
fn policy_rejects_non_localhost_runtime() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("bad_policy.toml");
    fs::write(
        &path,
        r#"
[model]
model_id = "openai_gpt_oss_20b"
canonical_mutation_allowed = false
output_classification = "MODEL_GENERATED_UNVERIFIED"
airlock_required = true

[network]
bind_address = "0.0.0.0"
external_egress_allowed = false
remote_api_allowed = false

[security]
secret_access_allowed = false
vault_access_allowed = false
release_key_access_allowed = false

[runtime]
source_tree_allowed = false
cargo_allowed = false
host_path_lookup_allowed = false
path_env_allowed = false
runtime_hash_required = true
model_hash_required = true
prompt_hash_required = true
output_hash_required = true

[acceptance]
nonce_prompt_required = true
canned_response_allowed = false
negative_controls_required = true
"#,
    )
    .unwrap();
    let err = ModelRuntimePolicy::load(&path).unwrap_err().to_string();
    assert!(err.contains("127.0.0.1"));
}

#[test]
fn airlock_rejects_direct_canonical_mutation_and_key_access() {
    let action = json!({
        "model_output_classification": "MODEL_GENERATED_UNVERIFIED",
        "canonical_mutation_allowed": true,
        "release_key_access_requested": true
    });
    let err = reject_forbidden_model_action(&action)
        .unwrap_err()
        .to_string();
    assert!(err.contains("canonical_mutation_allowed"));
}

#[test]
fn airlock_rejects_missing_unverified_label() {
    let action = json!({
        "model_output_classification": "INTERNAL",
        "canonical_mutation_allowed": false
    });
    let err = reject_forbidden_model_action(&action)
        .unwrap_err()
        .to_string();
    assert!(err.contains("MODEL_GENERATED_UNVERIFIED"));
}

#[test]
fn portable_policy_rejects_nas_model_path_before_smoke() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = write_portable_policy(tmp.path());
    let registry =
        write_registry_with_path(tmp.path(), "/mnt/storage/NAS/models/openai_gpt-oss-20b");
    let policy = ModelRuntimePolicy::load(&policy).unwrap();
    let err = load_model_artifact_with_policy(&registry, "openai/gpt-oss-20b", &policy)
        .unwrap_err()
        .to_string();
    assert!(err.contains("PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT"));
}

#[test]
fn portable_policy_rejects_host_model_cache_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let policy = ModelRuntimePolicy::load(&write_portable_policy(tmp.path())).unwrap();
    let home = prismcode::host_roots::home().expect("HOME must be an absolute path for this policy test");
    for rel in [
        ".cache/huggingface/openai_gpt-oss-20b",
        ".huggingface/openai_gpt-oss-20b",
        ".ollama/models/openai_gpt_oss_20b",
        "Desktop/Prism4D-bio/models/openai_gpt_oss_20b",
        ".cargo/models/openai_gpt_oss_20b",
    ] {
        let full = home.join(rel);
        let path = full.to_str().expect("HOME is valid UTF-8");
        let registry = write_registry_with_path(tmp.path(), path);
        let err = load_model_artifact_with_policy(&registry, "openai/gpt-oss-20b", &policy)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT")
                || err.contains("PC_ERR_MODEL_PATH_ESCAPE"),
            "unexpected error for {path}: {err}"
        );
    }
}

#[test]
fn missing_portable_model_artifact_fails_without_hanging() {
    let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
    let tmp = tempfile::tempdir_in("/tmp").unwrap();
    let bundle = tmp.path().join("bundle");
    fs::create_dir_all(bundle.join("models")).unwrap();
    let registry = write_registry_with_path(tmp.path(), "models/openai_gpt_oss_20b_missing");
    let policy = ModelRuntimePolicy::load(&write_portable_policy(tmp.path())).unwrap();
    std::env::set_var("PRISM_BUNDLE_ROOT", &bundle);
    let err = load_model_artifact_with_policy(&registry, "openai/gpt-oss-20b", &policy)
        .unwrap_err()
        .to_string();
    std::env::remove_var("PRISM_BUNDLE_ROOT");
    assert!(err.contains("PC_ERR_MODEL_ARTIFACT_MISSING"));
}

#[cfg(unix)]
#[test]
fn bundled_model_symlink_to_nas_fails() {
    use std::os::unix::fs::symlink;

    let _guard = env_lock().lock().unwrap_or_else(|err| err.into_inner());
    let tmp = tempfile::tempdir_in("/tmp").unwrap();
    let bundle = tmp.path().join("bundle");
    let models = bundle.join("models");
    fs::create_dir_all(&models).unwrap();
    let link = models.join("openai_gpt_oss_20b");
    if symlink("/mnt/storage", &link).is_err() {
        return;
    }
    let registry = write_registry_with_path(tmp.path(), "models/openai_gpt_oss_20b");
    let policy = ModelRuntimePolicy::load(&write_portable_policy(tmp.path())).unwrap();
    std::env::set_var("PRISM_BUNDLE_ROOT", &bundle);
    let err = load_model_artifact_with_policy(&registry, "openai/gpt-oss-20b", &policy)
        .unwrap_err()
        .to_string();
    std::env::remove_var("PRISM_BUNDLE_ROOT");
    assert!(
        err.contains("PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT"),
        "unexpected symlink rejection error: {err}"
    );
}

fn env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn write_portable_policy(root: &Path) -> std::path::PathBuf {
    let path = root.join("model_runtime_policy.toml");
    fs::write(
        &path,
        r#"
[model]
model_id = "openai_gpt_oss_20b"
model_artifact_source = "BUNDLED_MODEL_PACK"
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

fn write_registry_with_path(root: &Path, local_path: &str) -> std::path::PathBuf {
    let registry = json!({
        "registry_version": "TEST_MODEL_RUNTIME",
        "rules": {"canonical_state_mutation_allowed": false},
        "installed_models": [{
            "model_id": "openai/gpt-oss-20b",
            "local_path": local_path,
            "install_allowed": true,
            "hardware_gated": false,
            "weights_hash_verified": true,
            "files": [{
                "path": "model-00000-of-00001.safetensors",
                "bytes": 1,
                "blake3": "invalid"
            }]
        }],
        "hardware_gated_models": [{
            "model_id": "Qwen/Qwen3-Coder-Next",
            "hardware_gated": true,
            "install_allowed": false,
            "canonical_state_mutation_allowed": false
        }]
    });
    let path = root.join("registry_external_path.json");
    fs::write(&path, serde_json::to_vec_pretty(&registry).unwrap()).unwrap();
    path
}
