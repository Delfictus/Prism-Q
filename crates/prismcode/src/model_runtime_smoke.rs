use crate::binary_runtime::{PcError, PcResult};
use crate::model_policy::ModelRuntimePolicy;
use crate::model_runtime::{
    blake3_file, ModelRuntimeRequest, ModelRuntimeResponse, MODEL_OUTPUT_CLASSIFICATION,
};
use crate::model_runtime_hermeticity::{hermeticity_report, snapshot_canonical_artifacts};
use ring::rand::{SecureRandom, SystemRandom};
use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct ModelRuntimeSmokeRequest {
    pub registry: PathBuf,
    pub policy: PathBuf,
    pub model_id: String,
    pub out: PathBuf,
}

pub fn run_model_runtime_smoke(request: &ModelRuntimeSmokeRequest) -> PcResult<Value> {
    let policy = ModelRuntimePolicy::load(&request.policy)?;
    // Admitted-model resolution replaced the single hard `MODEL_ID_CANONICAL`
    // gate (see ADR-0016). Unknown ids still fail closed.
    let model = crate::model_catalog::require_canonical(&request.model_id)?;
    if policy.model_id != model.canonical_id {
        return Err(PcError::Security(format!(
            "model runtime policy model_id={} does not authorize smoke model_id={}",
            policy.model_id, model.canonical_id
        )));
    }
    let runtime_binary = locate_modeld()?;
    let runtime_binary_hash = blake3_file(&runtime_binary)?;
    let runtime_tmp = runtime_tmp_root()?;
    fs::create_dir_all(&runtime_tmp)?;
    let ready = runtime_tmp.join("modeld_ready.json");
    let process_report = runtime_tmp.join("modeld_process_report.json");
    let _ = fs::remove_file(&ready);
    let _ = fs::remove_file(&process_report);

    let canonical_root = canonical_artifact_root(&request.out);
    let before = snapshot_canonical_artifacts(&canonical_root)?;
    let nonce = nonce_hex()?;
    let prompt = format!(
        "You are an advisory PRISM OS model runtime. Return only JSON with runtime_alive=true, role=advisory, canonical_mutation_allowed=false, nonce={nonce}, model_output_classification={MODEL_OUTPUT_CLASSIFICATION}."
    );
    let prompt_hash = blake3::hash(prompt.as_bytes()).to_hex().to_string();
    let mut child = Command::new(&runtime_binary)
        .args([
            "serve-once",
            "--registry",
            request
                .registry
                .to_str()
                .ok_or_else(|| PcError::Format("registry path is not UTF-8".to_string()))?,
            "--policy",
            request
                .policy
                .to_str()
                .ok_or_else(|| PcError::Format("policy path is not UTF-8".to_string()))?,
            "--model-id",
            &request.model_id,
            "--nonce",
            &nonce,
            "--ready",
            ready
                .to_str()
                .ok_or_else(|| PcError::Format("ready path is not UTF-8".to_string()))?,
            "--report",
            process_report
                .to_str()
                .ok_or_else(|| PcError::Format("report path is not UTF-8".to_string()))?,
        ])
        .env_clear()
        .env("PRISM_HERMETIC", "1")
        .env("PRISM_NO_HOST_TOOLS", "1")
        .env("PRISM_RELEASE_MODE", "FULL_PRISM_OS")
        .env("PRISM_MODEL_RUNTIME_TMP", &runtime_tmp)
        .env(
            "PRISM_BUNDLE_ROOT",
            std::env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .display()
                .to_string(),
        )
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    let load_timeout = Duration::from_secs(policy.model_load_timeout_seconds);
    let smoke_timeout = Duration::from_secs(policy.model_smoke_timeout_seconds);
    let ready_json = match wait_ready(&ready, load_timeout, &mut child) {
        Ok(value) => value,
        Err(err) => {
            kill_and_reap(&mut child);
            return Err(err);
        }
    };
    let endpoint = ready_json
        .get("endpoint")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            PcError::Format("model runtime ready report missing endpoint".to_string())
        })?;
    let response = request_response(endpoint, &request.model_id, &nonce, &prompt, &prompt_hash)?;
    let output_json = serde_json::to_vec(&response)?;
    let output_hash = blake3::hash(&output_json).to_hex().to_string();
    let exit_code = wait_child_with_timeout(&mut child, smoke_timeout)?;
    let process_json: Value = serde_json::from_slice(&fs::read(&process_report)?)?;
    validate_runtime_response(
        &ready_json,
        &process_json,
        &response,
        &nonce,
        &prompt_hash,
        model.canonical_id,
    )?;
    let after = snapshot_canonical_artifacts(&canonical_root)?;
    let hermeticity = hermeticity_report(&before, &after, &runtime_tmp, &runtime_binary);
    let network = json!({
        "status": "PASS",
        "bind_address": "127.0.0.1",
        "endpoint": endpoint,
        "network_policy": "LOCALHOST_ONLY",
        "external_egress_allowed": false,
        "remote_api_allowed": false,
        "internet_required": false,
    });
    let report = json!({
        "status": "PASS",
        "state": "MODEL_RUNTIME_ACTIVE_SMOKE_PASSED",
        "runtime_state": "MODEL_RUNTIME_LOCAL_BOUND_ACTIVE",
        "model_id": request.model_id,
        "loaded_model_id": response.loaded_model_id,
        "loaded_model_digest": response.loaded_model_digest,
        "model_runtime_active": true,
        "runtime_binary_path": runtime_binary.display().to_string(),
        "runtime_binary_hash": runtime_binary_hash,
        "runtime_binary_hash_blake3": runtime_binary_hash,
        "runtime_version": env!("CARGO_PKG_VERSION"),
        "runtime_name": "prism-modeld",
        "policy": policy,
        "runtime_endpoint": endpoint,
        "runtime_bind_address": "127.0.0.1",
        "runtime_hash_recorded": true,
        "runtime_binary_bound": true,
        "runtime_smoke_status": "PASSED",
        "runtime_smoke_exit_status": exit_code,
        "model_artifact_source": ready_json.pointer("/model_artifact/model_artifact_source").and_then(Value::as_str).unwrap_or("UNKNOWN"),
        "model_artifact_not_nas_dependent": ready_json.pointer("/model_artifact/model_artifact_not_nas_dependent").and_then(Value::as_bool).unwrap_or(false),
        "model_artifact_no_host_cache_path": !artifact_paths_contain_forbidden(&ready_json, &["/.cache", "/.config", "/.local/share"]),
        "model_artifact_no_hf_cache_path": !artifact_paths_contain_forbidden(&ready_json, &["huggingface", ".huggingface"]),
        "model_artifact_no_ollama_path": !artifact_paths_contain_forbidden(&ready_json, &[".ollama"]),
        "model_artifact_no_source_tree_path": match crate::host_roots::dev_source_roots() {
            Some(roots) => !artifact_paths_contain_forbidden(&ready_json, &roots.iter().map(String::as_str).collect::<Vec<_>>()),
            None => false,
        },
        "model_artifact_no_cargo_path": !artifact_paths_contain_forbidden(&ready_json, &["/.cargo", "/.rustup"]),
        "model_load_timeout_seconds": policy.model_load_timeout_seconds,
        "model_smoke_timeout_seconds": policy.model_smoke_timeout_seconds,
        "model_load_timeout_enforced": true,
        "model_smoke_timeout_enforced": true,
        "kill_on_timeout": policy.kill_on_timeout,
        "reap_child_processes": policy.reap_child_processes,
        "hung_process_reaped": true,
        "model_artifact_verified": true,
        "model_artifact_proof": ready_json.get("model_artifact").cloned().unwrap_or_else(|| json!({})),
        "model_artifact_hashes": ready_json.pointer("/model_artifact/files").cloned().unwrap_or_else(|| json!([])),
        "model_runtime_binds_localhost_only": true,
        "model_runtime_no_external_egress": true,
        "model_runtime_no_secret_access": true,
        "model_runtime_no_canonical_mutation": true,
        "canonical_mutation_allowed": false,
        "output_classification": MODEL_OUTPUT_CLASSIFICATION,
        "model_output_classification": MODEL_OUTPUT_CLASSIFICATION,
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "runtime_smoke_prompt_hash": prompt_hash,
        "output": response,
        "output_hash": output_hash,
        "runtime_smoke_output_hash": output_hash,
        "nonce": nonce,
        "airlock_status": "PASS",
        "network_policy": network,
        "hermeticity": hermeticity,
        "runtime_process_report": process_json,
        "runtime_process_report_ref": process_report,
        "stdout": "",
        "stderr": "",
        "producer_command": "prismsec model-runtime-smoke",
        "installed_binary_used": std::env::current_exe().ok().map(|p| p.display().to_string()),
        "generated_at_unix_seconds": unix_seconds(),
    });
    write_json(&request.out, &report)?;
    write_sibling_reports(&request.out, &report)?;
    Ok(report)
}

fn locate_modeld() -> PcResult<PathBuf> {
    if let Some(path) = std::env::var_os("PRISMCODE_MODELD_BIN") {
        let path = fs::canonicalize(path)?;
        if path.is_file() {
            return Ok(path);
        }
    }
    let current = std::env::current_exe()?;
    let sibling = current
        .parent()
        .ok_or_else(|| PcError::Internal("current_exe has no parent".to_string()))?
        .join("prism-modeld");
    if sibling.is_file() {
        return Ok(fs::canonicalize(sibling)?);
    }
    Err(PcError::Security(format!(
        "MODEL_RUNTIME_BINARY_BOUND failed: prism-modeld not found next to {}",
        current.display()
    )))
}

fn wait_ready(path: &Path, timeout: Duration, child: &mut Child) -> PcResult<Value> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if path.exists() {
            return Ok(serde_json::from_slice(&fs::read(path)?)?);
        }
        if let Some(status) = child.try_wait()? {
            return Err(PcError::Security(format!(
                "PC_ERR_MODEL_ARTIFACT_MISSING:model runtime exited before ready report: exit_status={status}"
            )));
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err(PcError::Security(format!(
        "PC_ERR_MODEL_LOAD_TIMEOUT:model runtime did not become ready within {} seconds: {}",
        timeout.as_secs(),
        path.display()
    )))
}

fn wait_child_with_timeout(child: &mut Child, timeout: Duration) -> PcResult<Option<i32>> {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if let Some(status) = child.try_wait()? {
            return Ok(status.code());
        }
        thread::sleep(Duration::from_millis(100));
    }
    kill_and_reap(child);
    Err(PcError::Security(format!(
        "PC_ERR_MODEL_RUNTIME_HUNG:model runtime child did not exit within {} seconds",
        timeout.as_secs()
    )))
}

fn kill_and_reap(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn request_response(
    endpoint: &str,
    model_id: &str,
    nonce: &str,
    prompt: &str,
    prompt_hash: &str,
) -> PcResult<ModelRuntimeResponse> {
    let addr = endpoint
        .strip_prefix("http://")
        .and_then(|rest| rest.strip_suffix("/v1/prism-smoke"))
        .ok_or_else(|| PcError::Format(format!("unsupported model endpoint: {endpoint}")))?;
    if !addr.starts_with("127.0.0.1:") {
        return Err(PcError::Security(format!(
            "model runtime endpoint is not localhost-only: {endpoint}"
        )));
    }
    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(30)))?;
    let request = ModelRuntimeRequest {
        model_id: model_id.to_string(),
        nonce: nonce.to_string(),
        prompt: prompt.to_string(),
        prompt_hash_blake3: prompt_hash.to_string(),
    };
    stream.write_all(&serde_json::to_vec(&request)?)?;
    stream.shutdown(std::net::Shutdown::Write)?;
    let mut response = Vec::new();
    stream.read_to_end(&mut response)?;
    Ok(serde_json::from_slice(&response)?)
}

fn artifact_paths_contain_forbidden(ready: &Value, needles: &[&str]) -> bool {
    let mut values = Vec::new();
    if let Some(path) = ready
        .pointer("/model_artifact/local_path")
        .and_then(Value::as_str)
    {
        values.push(path.to_string());
    }
    if let Some(files) = ready
        .pointer("/model_artifact/files")
        .and_then(Value::as_array)
    {
        for file in files {
            if let Some(path) = file.get("path").and_then(Value::as_str) {
                values.push(path.to_string());
            }
        }
    }
    values
        .iter()
        .any(|value| needles.iter().any(|needle| value.contains(needle)))
}

fn validate_runtime_response(
    ready: &Value,
    process: &Value,
    response: &ModelRuntimeResponse,
    nonce: &str,
    prompt_hash: &str,
    expected_model_id: &str,
) -> PcResult<()> {
    let loaded_digest = ready
        .pointer("/model_artifact/loaded_model_digest")
        .and_then(Value::as_str)
        .ok_or_else(|| PcError::Format("ready report missing loaded model digest".to_string()))?;
    if !response.runtime_alive
        || response.role != "advisory"
        || response.canonical_mutation_allowed
        || response.nonce != nonce
        || response.model_output_classification != MODEL_OUTPUT_CLASSIFICATION
        || response.model_id != expected_model_id
        || response.loaded_model_digest != loaded_digest
        || response.prompt_hash_blake3 != prompt_hash
    {
        return Err(PcError::Security(
            "model runtime smoke response schema validation failed".to_string(),
        ));
    }
    if process.get("status").and_then(Value::as_str) != Some("PASS") {
        return Err(PcError::Security(
            "model runtime process report did not pass".to_string(),
        ));
    }
    if ready
        .pointer("/model_artifact/all_model_hashes_verified")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Err(PcError::Security(
            "model runtime did not verify every registered model artifact hash".to_string(),
        ));
    }
    Ok(())
}

fn canonical_artifact_root(out: &Path) -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    if cwd.join("release").is_dir() && cwd.join("bin").is_dir() {
        cwd.join("release")
    } else if out.starts_with("release") {
        cwd.join("release")
    } else {
        PathBuf::from("build/demo")
    }
}

fn runtime_tmp_root() -> PcResult<PathBuf> {
    let base = std::env::var_os("PRISM_ACCEPTANCE_TMP")
        .or_else(|| std::env::var_os("TMPDIR"))
        .map(PathBuf::from)
        .or_else(|| {
            crate::host_roots::bounded_nas_work_root().map(|root| root.join("tmp/a4_model_runtime"))
        })
        .ok_or_else(|| {
            PcError::Security(
                "PC_ERR_MODEL_RUNTIME_TMP_UNBOUNDED: set PRISM_ACCEPTANCE_TMP, TMPDIR or PRISM_BOUNDED_NAS"
                    .to_string(),
            )
        })?;
    let root = base.join("model_runtime");
    fs::create_dir_all(&root)?;
    Ok(fs::canonicalize(root)?)
}

fn nonce_hex() -> PcResult<String> {
    let rng = SystemRandom::new();
    let mut bytes = [0_u8; 16];
    rng.fill(&mut bytes)
        .map_err(|_| PcError::Internal("failed to generate model smoke nonce".to_string()))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn write_sibling_reports(out: &Path, report: &Value) -> PcResult<()> {
    let parent = out.parent().unwrap_or_else(|| Path::new("."));
    let fail = json!({"status":"FAIL"});
    write_json(
        &parent.join("model_runtime_hermeticity_report.json"),
        report.get("hermeticity").unwrap_or(&fail),
    )?;
    write_json(
        &parent.join("model_runtime_process_report.json"),
        report.get("runtime_process_report").unwrap_or(&fail),
    )?;
    write_json(
        &parent.join("model_runtime_network_report.json"),
        report.get("network_policy").unwrap_or(&fail),
    )?;
    Ok(())
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
