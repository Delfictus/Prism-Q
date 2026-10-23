use crate::binary_runtime::{PcError, PcResult};
use crate::model_catalog::{self, ModelArtifactClass};
use crate::model_policy::ModelRuntimePolicy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// The historical default advisory model. These remain the single default the
// runtime falls back to, but the runtime is no longer hard-wired to only this
// id: the admitted set now comes from `model_catalog::ALLOWED_CANONICAL_MODELS`
// (see ADR-0016). Sourcing them from the catalog keeps one source of truth.
pub const MODEL_ID_REGISTRY: &str = model_catalog::DEFAULT_MODEL.registry_id;
pub const MODEL_ID_CANONICAL: &str = model_catalog::DEFAULT_MODEL.canonical_id;
pub const MODEL_OUTPUT_CLASSIFICATION: &str = "MODEL_GENERATED_UNVERIFIED";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeModelFileProof {
    pub path: PathBuf,
    pub relative_path: String,
    pub byte_size: u64,
    pub expected_blake3: String,
    pub observed_blake3: String,
    pub hash_verified: bool,
    pub safetensors_header_loaded: bool,
    pub tensor_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifactProof {
    pub status: String,
    pub model_id: String,
    pub artifact_class: String,
    pub local_path: PathBuf,
    pub model_artifact_source: String,
    pub portable: bool,
    pub nas_required: bool,
    pub source_tree_required: bool,
    pub host_cache_required: bool,
    pub model_artifact_not_nas_dependent: bool,
    pub loaded_model_digest: String,
    pub tokenizer_hash_verified: bool,
    pub config_hash_verified: bool,
    pub shard_hashes_verified: bool,
    pub all_model_hashes_verified: bool,
    pub total_bytes_verified: u64,
    pub files: Vec<RuntimeModelFileProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRuntimeProcessReport {
    pub status: String,
    pub runtime_name: String,
    pub runtime_version: String,
    pub runtime_state: String,
    pub runtime_binary_path: PathBuf,
    pub runtime_binary_hash_blake3: String,
    pub bind_address: String,
    pub endpoint: String,
    pub model_id: String,
    pub loaded_model_id: String,
    pub loaded_model_digest: String,
    pub policy_status: String,
    pub network_policy: String,
    pub external_egress_allowed: bool,
    pub secret_access_allowed: bool,
    pub canonical_mutation_allowed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRuntimeRequest {
    pub model_id: String,
    pub nonce: String,
    pub prompt: String,
    pub prompt_hash_blake3: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRuntimeResponse {
    pub runtime_alive: bool,
    pub role: String,
    pub canonical_mutation_allowed: bool,
    pub nonce: String,
    pub model_output_classification: String,
    pub model_id: String,
    pub loaded_model_id: String,
    pub loaded_model_digest: String,
    pub runtime_name: String,
    pub prompt_hash_blake3: String,
    pub response_derivation_hash_blake3: String,
}

pub fn modeld_serve_once(
    registry: &Path,
    policy_path: &Path,
    model_id: &str,
    nonce: &str,
    ready_path: &Path,
    report_path: &Path,
) -> PcResult<()> {
    let policy = ModelRuntimePolicy::load(policy_path)?;
    // Admitted-model resolution replaced the single `model_id != MODEL_ID_CANONICAL`
    // hard gate. Unknown ids still fail closed with the identical error text.
    let model = model_catalog::require_canonical(model_id)?;
    // The policy pins one model_id; the served id must match it (defence in depth
    // so a policy for gpt-oss cannot be reused to serve the champion, or vice versa).
    if policy.model_id != model.canonical_id {
        return Err(PcError::Security(format!(
            "model runtime policy model_id={} does not authorize served model_id={}",
            policy.model_id, model.canonical_id
        )));
    }
    let artifact = load_model_artifact_with_policy(registry, model.registry_id, &policy)?;
    let runtime_binary_path = std::env::current_exe()?;
    let runtime_binary_hash = blake3_file(&runtime_binary_path)?;
    let listener = TcpListener::bind((policy.bind_address.as_str(), 0))?;
    let endpoint = format!("http://{}/v1/prism-smoke", listener.local_addr()?);
    let process = ModelRuntimeProcessReport {
        status: "PASS".to_string(),
        runtime_name: "prism-modeld".to_string(),
        runtime_version: env!("CARGO_PKG_VERSION").to_string(),
        runtime_state: "MODEL_RUNTIME_LOCAL_BOUND_ACTIVE".to_string(),
        runtime_binary_path: runtime_binary_path.clone(),
        runtime_binary_hash_blake3: runtime_binary_hash.clone(),
        bind_address: policy.bind_address.clone(),
        endpoint: endpoint.clone(),
        model_id: model.canonical_id.to_string(),
        loaded_model_id: model.registry_id.to_string(),
        loaded_model_digest: artifact.loaded_model_digest.clone(),
        policy_status: policy.status.clone(),
        network_policy: "LOCALHOST_ONLY".to_string(),
        external_egress_allowed: false,
        secret_access_allowed: false,
        canonical_mutation_allowed: false,
    };
    write_json(
        ready_path,
        &json!({
            "status": "PASS",
            "endpoint": endpoint,
            "nonce": nonce,
            "process": process,
            "model_artifact": artifact,
        }),
    )?;
    let (mut stream, peer) = listener.accept()?;
    if !peer.ip().is_loopback() {
        return Err(PcError::Security(format!(
            "model runtime rejected non-loopback peer: {peer}"
        )));
    }
    let mut request_bytes = Vec::new();
    stream.read_to_end(&mut request_bytes)?;
    let request: ModelRuntimeRequest = serde_json::from_slice(&request_bytes)?;
    if request.model_id != model.canonical_id || request.nonce != nonce {
        return Err(PcError::Security(
            "model runtime request failed model-id/nonce validation".to_string(),
        ));
    }
    let derivation = blake3::hash(
        format!(
            "{}:{}:{}:{}",
            artifact.loaded_model_digest, request.prompt_hash_blake3, request.nonce, request.prompt
        )
        .as_bytes(),
    )
    .to_hex()
    .to_string();
    let response = ModelRuntimeResponse {
        runtime_alive: true,
        role: "advisory".to_string(),
        canonical_mutation_allowed: false,
        nonce: request.nonce,
        model_output_classification: MODEL_OUTPUT_CLASSIFICATION.to_string(),
        model_id: model.canonical_id.to_string(),
        loaded_model_id: model.registry_id.to_string(),
        loaded_model_digest: artifact.loaded_model_digest.clone(),
        runtime_name: "prism-modeld".to_string(),
        prompt_hash_blake3: request.prompt_hash_blake3,
        response_derivation_hash_blake3: derivation,
    };
    let response_bytes = serde_json::to_vec(&response)?;
    stream.write_all(&response_bytes)?;
    write_json(
        report_path,
        &json!({
            "status": "PASS",
            "runtime_process": process,
            "model_artifact": artifact,
            "request_prompt_hash": response.prompt_hash_blake3,
            "response": response,
            "output_hash_blake3": blake3::hash(&response_bytes).to_hex().to_string(),
            "served_at_unix_seconds": unix_seconds(),
        }),
    )?;
    Ok(())
}

pub fn load_model_artifact(registry: &Path, model_id: &str) -> PcResult<ModelArtifactProof> {
    load_model_artifact_inner(registry, model_id, None)
}

pub fn load_model_artifact_with_policy(
    registry: &Path,
    model_id: &str,
    policy: &ModelRuntimePolicy,
) -> PcResult<ModelArtifactProof> {
    load_model_artifact_inner(registry, model_id, Some(policy))
}

fn load_model_artifact_inner(
    registry: &Path,
    model_id: &str,
    policy: Option<&ModelRuntimePolicy>,
) -> PcResult<ModelArtifactProof> {
    let registry_json: Value = serde_json::from_slice(&fs::read(registry)?)?;
    let models = registry_json
        .get("installed_models")
        .and_then(Value::as_array)
        .ok_or_else(|| PcError::Format("model registry missing installed_models".to_string()))?;
    let model = models
        .iter()
        .find(|model| model.get("model_id").and_then(Value::as_str) == Some(model_id))
        .ok_or_else(|| PcError::Format(format!("{model_id} registry record missing")))?;
    let local_path = model
        .get("local_path")
        .and_then(Value::as_str)
        .ok_or_else(|| PcError::Format("model registry missing local_path".to_string()))?;
    let local_path = resolve_model_root(local_path, policy)?;
    let files = model
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| PcError::Format("model registry missing files".to_string()))?;
    // Artifact class: explicit registry field wins (a typo hard-errors rather than
    // silently downgrading verification), else the catalog default for this
    // registry id, else the historical Safetensors class. This keeps existing
    // safetensors registries byte-for-byte unchanged while admitting a `.pt`
    // fused-checkpoint class for the champion (see ADR-0016 / model_catalog).
    let artifact_class = match model.get("artifact_class").and_then(Value::as_str) {
        Some(raw) => ModelArtifactClass::from_registry_str(raw)?,
        None => model_catalog::resolve_by_registry(model_id)
            .map(|model| model.artifact_class)
            .unwrap_or(ModelArtifactClass::Safetensors),
    };
    let mut proofs = Vec::new();
    let mut digest = blake3::Hasher::new();
    digest.update(model_id.as_bytes());
    let mut total = 0_u64;
    for file in files {
        let rel = file
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| PcError::Format("model file missing path".to_string()))?;
        let expected_bytes = file
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| PcError::Format("model file missing byte size".to_string()))?;
        let expected_hash = file
            .get("blake3")
            .and_then(Value::as_str)
            .ok_or_else(|| PcError::Format("model file missing BLAKE3".to_string()))?;
        let path = canonicalize_model_file(&local_path, rel)?;
        if !path.starts_with(&local_path) {
            return Err(PcError::Security(format!(
                "PC_ERR_MODEL_PATH_ESCAPE:model artifact path escapes registered root: {}",
                path.display()
            )));
        }
        let observed_bytes = fs::metadata(&path)?.len();
        let observed_hash = blake3_file(&path)?;
        let (header_loaded, tensor_count) = if rel.ends_with(".safetensors") {
            load_safetensors_header(&path)?
        } else {
            (false, None)
        };
        let verified = observed_bytes == expected_bytes && observed_hash == expected_hash;
        total = total.saturating_add(observed_bytes);
        digest.update(rel.as_bytes());
        digest.update(observed_bytes.to_string().as_bytes());
        digest.update(observed_hash.as_bytes());
        proofs.push(RuntimeModelFileProof {
            path,
            relative_path: rel.to_string(),
            byte_size: observed_bytes,
            expected_blake3: expected_hash.to_string(),
            observed_blake3: observed_hash,
            hash_verified: verified,
            safetensors_header_loaded: header_loaded,
            tensor_count,
        });
    }
    let tokenizer_hash_verified = proofs
        .iter()
        .any(|proof| proof.relative_path == "tokenizer.json" && proof.hash_verified);
    let config_hash_verified = proofs
        .iter()
        .any(|proof| proof.relative_path == "config.json" && proof.hash_verified);
    let shard_hashes_verified = proofs
        .iter()
        .filter(|proof| proof.relative_path.ends_with(".safetensors"))
        .all(|proof| proof.hash_verified && proof.safetensors_header_loaded);
    // A fused `.pt` checkpoint has no safetensors header to parse; it is attested
    // by requiring at least one `.pt` file, all of them hash+size verified.
    let checkpoint_hashes_verified = {
        let pt: Vec<&RuntimeModelFileProof> = proofs
            .iter()
            .filter(|proof| proof.relative_path.ends_with(".pt"))
            .collect();
        !pt.is_empty() && pt.iter().all(|proof| proof.hash_verified)
    };
    // Every class requires that EVERY registered file hash+size verifies (the core
    // security invariant, unchanged). Class only decides which structural files
    // must additionally be present/parsed. Safetensors is bit-for-bit identical to
    // the pre-ADR-0016 requirement.
    let class_structure_verified = match artifact_class {
        ModelArtifactClass::Safetensors => {
            tokenizer_hash_verified && config_hash_verified && shard_hashes_verified
        }
        ModelArtifactClass::FusedCheckpointPt => checkpoint_hashes_verified,
    };
    let all_model_hashes_verified = !proofs.is_empty()
        && proofs.iter().all(|proof| proof.hash_verified)
        && class_structure_verified;
    Ok(ModelArtifactProof {
        status: if all_model_hashes_verified {
            "PASS".to_string()
        } else {
            "FAIL".to_string()
        },
        model_id: model_id.to_string(),
        artifact_class: artifact_class.as_registry_str().to_string(),
        local_path,
        model_artifact_source: policy
            .map(|policy| policy.model_artifact_source.clone())
            .unwrap_or_else(|| "TEST_FIXTURE_ONLY".to_string()),
        portable: policy
            .map(|policy| policy.model_artifact_source != "TEST_FIXTURE_ONLY")
            .unwrap_or(false),
        nas_required: false,
        source_tree_required: false,
        host_cache_required: false,
        model_artifact_not_nas_dependent: true,
        loaded_model_digest: digest.finalize().to_hex().to_string(),
        tokenizer_hash_verified,
        config_hash_verified,
        shard_hashes_verified,
        all_model_hashes_verified,
        total_bytes_verified: total,
        files: proofs,
    })
}

fn resolve_model_root(raw_path: &str, policy: Option<&ModelRuntimePolicy>) -> PcResult<PathBuf> {
    let raw = PathBuf::from(raw_path);
    if let Some(policy) = policy {
        reject_forbidden_model_path(&raw, policy)?;
    }
    let candidate = if raw.is_absolute() {
        raw
    } else {
        bundle_root().join(raw)
    };
    if let Some(policy) = policy {
        reject_forbidden_model_path(&candidate, policy)?;
    }
    let canonical = fs::canonicalize(&candidate).map_err(|err| {
        PcError::Security(format!(
            "PC_ERR_MODEL_ARTIFACT_MISSING:model artifact root missing or inaccessible: {}: {err}",
            candidate.display()
        ))
    })?;
    if !canonical.is_dir() {
        return Err(PcError::Security(format!(
            "PC_ERR_MODEL_ARTIFACT_MISSING:model artifact root is not a directory: {}",
            canonical.display()
        )));
    }
    if let Some(policy) = policy {
        reject_forbidden_model_path(&canonical, policy)?;
        enforce_portable_model_source(&canonical, policy)?;
    }
    Ok(canonical)
}

fn canonicalize_model_file(root: &Path, rel: &str) -> PcResult<PathBuf> {
    let path = root.join(rel);
    fs::canonicalize(&path).map_err(|err| {
        PcError::Security(format!(
            "PC_ERR_MODEL_ARTIFACT_MISSING:model artifact file missing or inaccessible: {}: {err}",
            path.display()
        ))
    })
}

fn enforce_portable_model_source(root: &Path, policy: &ModelRuntimePolicy) -> PcResult<()> {
    match policy.model_artifact_source.as_str() {
        "TEST_FIXTURE_ONLY" => Ok(()),
        "BUNDLED_MODEL_PACK" | "INSTALLED_PORTABLE_MODEL_PACK" => {
            let bundle_models = fs::canonicalize(bundle_root().join("models")).map_err(|err| {
                PcError::Security(format!(
                    "PC_ERR_MODEL_ARTIFACT_MISSING:portable bundle models directory missing: {err}"
                ))
            })?;
            if !root.starts_with(&bundle_models) {
                return Err(PcError::Security(format!(
                    "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT:model_artifact_source={} requires model root under bundled models directory {}; observed {}",
                    policy.model_artifact_source,
                    bundle_models.display(),
                    root.display()
                )));
            }
            Ok(())
        }
        other => Err(PcError::Security(format!(
            "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT:unsupported model_artifact_source: {other}"
        ))),
    }
}

fn bundle_root() -> PathBuf {
    std::env::var_os("PRISM_BUNDLE_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

fn reject_forbidden_model_path(path: &Path, policy: &ModelRuntimePolicy) -> PcResult<()> {
    if policy.model_artifact_source == "TEST_FIXTURE_ONLY" {
        return Ok(());
    }
    let text = path.to_string_lossy();
    let nas_roots = [
        "/mnt/storage",
        "/mnt/prism4d-nas",
        "/mnt/storage/NAS",
        "/mnt/prism4d-nas/Source",
    ]
    .iter()
    .map(|root| root.to_string())
    .collect::<Vec<_>>();
    let forbidden: Vec<(bool, &str, Vec<String>)> = vec![
        (
            !policy.nas_model_path_allowed,
            "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT",
            nas_roots,
        ),
        (
            !policy.host_cache_allowed,
            "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT",
            home_policy_roots(
                !policy.host_cache_allowed,
                &[".cache", ".config", ".local/share"],
            ),
        ),
        (
            !policy.hf_cache_allowed,
            "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT",
            home_policy_roots(
                !policy.hf_cache_allowed,
                &[".huggingface", ".cache/huggingface"],
            ),
        ),
        (
            !policy.ollama_cache_allowed,
            "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT",
            home_policy_roots(!policy.ollama_cache_allowed, &[".ollama"]),
        ),
        (
            !policy.source_tree_allowed,
            "PC_ERR_MODEL_PATH_ESCAPE",
            if policy.source_tree_allowed {
                Vec::new()
            } else {
                crate::host_roots::dev_source_roots().unwrap_or_default()
            },
        ),
        (
            !policy.cargo_allowed,
            "PC_ERR_MODEL_PATH_ESCAPE",
            home_policy_roots(!policy.cargo_allowed, &[".cargo", ".rustup"]),
        ),
    ];
    for (enabled, code, prefixes) in forbidden {
        if enabled && prefixes.iter().any(|prefix| text.starts_with(prefix.as_str())) {
            return Err(PcError::Security(format!(
                "{code}:portable model artifact path is forbidden by policy: {}",
                path.display()
            )));
        }
    }
    // HOME-independent component check: the same per-user cache/toolchain
    // directories and development trees are refused wherever they appear, so
    // the policy still holds when HOME is unset (e.g. under a service manager)
    // or belongs to a different user.
    let components: [(bool, &str, &[&str]); 5] = [
        (!policy.host_cache_allowed, "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT", &["/.cache/", "/.config/", "/.local/share/"]),
        (!policy.hf_cache_allowed, "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT", &["/.huggingface/"]),
        (!policy.ollama_cache_allowed, "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT", &["/.ollama/"]),
        (!policy.source_tree_allowed, "PC_ERR_MODEL_PATH_ESCAPE", &["/Desktop/Prism4D-bio/", "/Desktop/PrismOS/"]),
        (!policy.cargo_allowed, "PC_ERR_MODEL_PATH_ESCAPE", &["/.cargo/", "/.rustup/"]),
    ];
    for (enabled, code, needles) in components {
        if enabled && needles.iter().any(|needle| text.contains(needle)) {
            return Err(PcError::Security(format!(
                "{code}:portable model artifact path is forbidden by policy: {}",
                path.display()
            )));
        }
    }
    if !policy.hf_cache_allowed && text.contains("/huggingface") {
        return Err(PcError::Security(format!(
            "PC_ERR_MODEL_ARTIFACT_NAS_DEPENDENT:portable model artifact path uses Hugging Face cache: {}",
            path.display()
        )));
    }
    Ok(())
}

/// `$HOME`-relative roots for an enabled policy check; empty when the check is
/// disabled or `$HOME` is unusable (the component check below still applies).
fn home_policy_roots(enabled: bool, rels: &[&str]) -> Vec<String> {
    if !enabled {
        return Vec::new();
    }
    crate::host_roots::under_home(rels).unwrap_or_default()
}

fn load_safetensors_header(path: &Path) -> PcResult<(bool, Option<usize>)> {
    let mut file = fs::File::open(path)?;
    let mut len = [0_u8; 8];
    file.read_exact(&mut len)?;
    let header_len = u64::from_le_bytes(len);
    if header_len == 0 || header_len > 64 * 1024 * 1024 {
        return Err(PcError::Format(format!(
            "invalid safetensors header length in {}: {header_len}",
            path.display()
        )));
    }
    let mut header = vec![0_u8; header_len as usize];
    file.read_exact(&mut header)?;
    let value: Value = serde_json::from_slice(&header)?;
    let tensor_count = value.as_object().map(|obj| {
        obj.keys()
            .filter(|key| key.as_str() != "__metadata__")
            .count()
    });
    Ok((true, tensor_count))
}

pub fn blake3_file(path: &Path) -> PcResult<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn write_json(path: &Path, value: &Value) -> PcResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
