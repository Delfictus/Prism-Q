//! Rust -> Python bridge for the champion PRISM PLLM under `prism-modeld`.
//!
//! `prism-modeld` / [`crate::model_runtime`] is an attestation shell: it verifies
//! artifact hashes and bounds a localhost handshake but never runs a forward
//! pass. The champion model ([`crate::model_catalog::PRISM_CHAMPION_PLLM_QWEN3_8B`])
//! DOES have real neural inference, but it lives entirely in Python — the
//! 4-bit devbox server at `$PRISM_CHAMPION_DEVBOX_ROOT/serve.py`
//! (`127.0.0.1:8711`, endpoints `/health` + `/predict`, `/generate` being added).
//!
//! This module is the governed control plane for that Python server. For the
//! champion `model_id` it can spawn `serve.py` as a hermetic localhost
//! subprocess (or attach to an already-running one), health-check it, and proxy
//! a prompt to `/predict` / `/generate`, returning the output under the SAME
//! advisory envelope the attestation shell uses: `role = "advisory"`,
//! `model_output_classification = MODEL_GENERATED_UNVERIFIED`,
//! `canonical_mutation_allowed = false`. The bridge NEVER writes canonical state.
//!
//! Security posture (identical to `prism-modeld`, enforced here):
//!   * localhost only — the base URL host must be a loopback address; any other
//!     host is rejected before a socket is opened (no egress);
//!   * no canonical mutation — output is advisory/unverified and must still pass
//!     the ADR-0013/0014 airlock (HIL.RESOLVE + AUDIT.SEAL) before it can
//!     influence anything.
//!
//! The live subprocess-spawn and network-proxy paths are gated behind the
//! `champion_bridge` cargo feature so the default build stays hermetic and does
//! not require Python at check/test time. With the feature OFF, every OS/network
//! entry point returns a clear "disabled" error; the types, config, and
//! envelope construction remain fully available and testable.
//!
//! REMAINING TO INTEGRATE (see ADR-0016 "Rollout"): the live wiring in
//! `modeld_serve_once` that, for the champion bridge kind, calls
//! [`ChampionBridge::serve_prompt`] instead of the pure attestation handshake,
//! plus the airlock HIL/SEAL step on the returned envelope.

use crate::binary_runtime::{PcError, PcResult};
use crate::model_catalog::{CanonicalModel, ModelBridgeKind};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::net::IpAddr;
use std::path::PathBuf;
use std::process::Child;
use std::time::Duration;

/// Environment variable naming the devbox root (contains `serve.py`,
/// `load_champion.py`, `config/`). There is no compiled-in default: when it is
/// unset the default config carries an empty root and `spawn` fails closed.
pub const DEVBOX_ROOT_ENV: &str = "PRISM_CHAMPION_DEVBOX_ROOT";
/// Default localhost bind of the devbox FastAPI server.
pub const DEFAULT_BRIDGE_HOST: &str = "127.0.0.1";
/// Default devbox port (see `serve.py`).
pub const DEFAULT_BRIDGE_PORT: u16 = 8711;

/// Which champion endpoint a prompt targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChampionEndpoint {
    /// Physics-state reconstruction over a u64 token context (`POST /predict`).
    Predict,
    /// Language generation over a text prompt + PRISM context (`POST /generate`).
    Generate,
}

impl ChampionEndpoint {
    fn path(self) -> &'static str {
        match self {
            Self::Predict => "/predict",
            Self::Generate => "/generate",
        }
    }
}

/// How to reach the Python devbox server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChampionBridgeConfig {
    /// Python interpreter used to launch the server (when spawning).
    pub python_bin: PathBuf,
    /// Devbox root; `serve.py` is expected directly inside it.
    pub devbox_root: PathBuf,
    /// Server script filename inside `devbox_root`.
    pub server_script: String,
    /// Loopback host the server binds. MUST be a loopback address.
    pub host: String,
    /// Server port.
    pub port: u16,
    /// Health endpoint path (GET).
    pub health_path: String,
    /// Seconds to wait for the server to become healthy after spawn.
    pub spawn_timeout_seconds: u64,
    /// Per-request proxy timeout (health/predict/generate).
    pub request_timeout_seconds: u64,
}

impl Default for ChampionBridgeConfig {
    fn default() -> Self {
        Self {
            python_bin: PathBuf::from("python3"),
            devbox_root: std::env::var_os(DEVBOX_ROOT_ENV)
                .map(PathBuf::from)
                .unwrap_or_default(),
            server_script: "serve.py".to_string(),
            host: DEFAULT_BRIDGE_HOST.to_string(),
            port: DEFAULT_BRIDGE_PORT,
            health_path: "/health".to_string(),
            spawn_timeout_seconds: 180,
            request_timeout_seconds: 30,
        }
    }
}

impl ChampionBridgeConfig {
    /// Validate the localhost-only posture. Rejects any non-loopback host BEFORE
    /// a socket is ever opened, mirroring the runtime policy's bind lock.
    pub fn validate(&self) -> PcResult<()> {
        let ip: IpAddr = self.host.parse().map_err(|_| {
            PcError::Security(format!(
                "champion bridge host is not an IP literal (localhost only): {}",
                self.host
            ))
        })?;
        if !ip.is_loopback() {
            return Err(PcError::Security(format!(
                "champion bridge host must be loopback (no egress): {}",
                self.host
            )));
        }
        if self.port == 0 {
            return Err(PcError::Security(
                "champion bridge port must be non-zero".to_string(),
            ));
        }
        Ok(())
    }

    /// `http://host:port` base URL (loopback-validated).
    pub fn base_url(&self) -> PcResult<String> {
        self.validate()?;
        Ok(format!("http://{}:{}", self.host, self.port))
    }
}

/// The advisory envelope every champion output is wrapped in. This is the same
/// contract the attestation shell emits: advisory role, unverified
/// classification, and canonical mutation explicitly denied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChampionAdvisoryEnvelope {
    pub runtime_name: String,
    pub role: String,
    pub model_output_classification: String,
    pub canonical_mutation_allowed: bool,
    pub model_id: String,
    pub loaded_model_id: String,
    pub endpoint: String,
    pub request_hash_blake3: String,
    pub response_hash_blake3: String,
    /// The raw JSON payload returned by the Python server (`/predict` or
    /// `/generate`). Advisory only — never a source of truth.
    pub model_output: Value,
}

impl ChampionAdvisoryEnvelope {
    /// Wrap a raw Python-server payload under the advisory envelope, binding the
    /// request and response by BLAKE3. Hard-sets the advisory/deny fields so a
    /// payload cannot smuggle `canonical_mutation_allowed = true`.
    pub fn wrap(
        model: &CanonicalModel,
        endpoint: ChampionEndpoint,
        request_json: &Value,
        response_json: Value,
    ) -> PcResult<Self> {
        let request_bytes = serde_json::to_vec(request_json)?;
        let response_bytes = serde_json::to_vec(&response_json)?;
        Ok(Self {
            runtime_name: "prism-modeld-champion-bridge".to_string(),
            role: "advisory".to_string(),
            model_output_classification: crate::model_airlock::MODEL_GENERATED_UNVERIFIED
                .to_string(),
            canonical_mutation_allowed: false,
            model_id: model.canonical_id.to_string(),
            loaded_model_id: model.registry_id.to_string(),
            endpoint: endpoint.path().to_string(),
            request_hash_blake3: blake3::hash(&request_bytes).to_hex().to_string(),
            response_hash_blake3: blake3::hash(&response_bytes).to_hex().to_string(),
            model_output: response_json,
        })
    }

    /// Serialize as a `serde_json::Value` (report-friendly).
    pub fn to_value(&self) -> PcResult<Value> {
        Ok(serde_json::to_value(self)?)
    }
}

/// A live (or attached) handle to the Python devbox server.
#[derive(Debug)]
pub struct ChampionBridge {
    model: CanonicalModel,
    config: ChampionBridgeConfig,
    base_url: String,
    /// Present only when this handle spawned the server (so Drop can reap it).
    child: Option<Child>,
}

impl ChampionBridge {
    /// Build a handle bound to an ALREADY-running devbox server. Does not spawn.
    /// The bridge kind must be [`ModelBridgeKind::PythonDevboxBridge`].
    pub fn attach(model: CanonicalModel, config: ChampionBridgeConfig) -> PcResult<Self> {
        if model.bridge != ModelBridgeKind::PythonDevboxBridge {
            return Err(PcError::Security(format!(
                "model {} does not use the python devbox bridge",
                model.canonical_id
            )));
        }
        let base_url = config.base_url()?;
        Ok(Self {
            model,
            config,
            base_url,
            child: None,
        })
    }

    /// The loopback base URL (`http://127.0.0.1:8711`).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The catalog model this bridge serves.
    pub fn model(&self) -> &CanonicalModel {
        &self.model
    }

    fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.config.request_timeout_seconds)
    }

    /// GET `/health`. Returns the parsed JSON body on 2xx.
    pub fn health_check(&self) -> PcResult<Value> {
        http_request(
            &self.base_url,
            "GET",
            &self.config.health_path,
            None,
            self.request_timeout(),
        )
    }

    /// POST a request body to `/predict` or `/generate`, returning the raw
    /// Python payload wrapped in the advisory envelope.
    pub fn serve_prompt(
        &self,
        endpoint: ChampionEndpoint,
        request_json: &Value,
    ) -> PcResult<ChampionAdvisoryEnvelope> {
        let body = serde_json::to_vec(request_json)?;
        let response = http_request(
            &self.base_url,
            "POST",
            endpoint.path(),
            Some(&body),
            self.request_timeout(),
        )?;
        ChampionAdvisoryEnvelope::wrap(&self.model, endpoint, request_json, response)
    }

    /// Convenience: physics-state reconstruction over a u64 token context.
    pub fn predict(&self, u64_context: &[u64], top_k: u32) -> PcResult<ChampionAdvisoryEnvelope> {
        let request = json!({
            "u64_context": u64_context.iter().map(|t| *t as i64).collect::<Vec<_>>(),
            "top_k": top_k,
        });
        self.serve_prompt(ChampionEndpoint::Predict, &request)
    }

    /// Convenience: language generation over a text prompt.
    pub fn generate(
        &self,
        prompt: &str,
        max_new_tokens: u32,
    ) -> PcResult<ChampionAdvisoryEnvelope> {
        let request = json!({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
        });
        self.serve_prompt(ChampionEndpoint::Generate, &request)
    }

    /// Kill and reap a spawned server (no-op for an attached handle).
    pub fn shutdown(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for ChampionBridge {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Governed one-shot query against an ALREADY-running devbox Oracle (attach, not
/// spawn — no per-query re-hash of the 17 GB checkpoint). Resolves the model via
/// the catalog (admission gate), health-checks the loopback server, proxies the
/// request to `/predict` or `/generate`, and returns the advisory-envelope JSON
/// (role=advisory, MODEL_GENERATED_UNVERIFIED, no canonical mutation). This is the
/// entry point `prism-modeld query` exposes for the PrismOS UI adapter.
pub fn champion_query(
    model_id: &str,
    endpoint: ChampionEndpoint,
    request_json: &Value,
) -> PcResult<Value> {
    let model = crate::model_catalog::require_canonical(model_id)?;
    let bridge = ChampionBridge::attach(model, ChampionBridgeConfig::default())?;
    bridge.health_check()?;
    let envelope = bridge.serve_prompt(endpoint, request_json)?;
    envelope.to_value()
}

// ---------------------------------------------------------------------------
// Live OS/network paths — gated behind the `champion_bridge` feature.
// ---------------------------------------------------------------------------

#[cfg(feature = "champion_bridge")]
mod live {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::process::{Command, Stdio};
    use std::thread;
    use std::time::Instant;

    impl ChampionBridge {
        /// Spawn `serve.py` as a hermetic localhost subprocess, wait for
        /// `/health` to report the model loaded, and return the bound handle.
        pub fn spawn(model: CanonicalModel, config: ChampionBridgeConfig) -> PcResult<Self> {
            if model.bridge != ModelBridgeKind::PythonDevboxBridge {
                return Err(PcError::Security(format!(
                    "model {} does not use the python devbox bridge",
                    model.canonical_id
                )));
            }
            let base_url = config.base_url()?;
            if !config.devbox_root.is_absolute() {
                return Err(PcError::Security(format!(
                    "champion devbox root must be absolute (set {DEVBOX_ROOT_ENV}): {}",
                    config.devbox_root.display()
                )));
            }
            let home = crate::host_roots::home().ok_or_else(|| {
                PcError::Security(
                    "champion devbox launch needs an absolute HOME for the hermetic env".to_string(),
                )
            })?;
            let script = config.devbox_root.join(&config.server_script);
            if !script.is_file() {
                return Err(PcError::Security(format!(
                    "champion devbox server script missing: {}",
                    script.display()
                )));
            }
            // Hermetic launch: cleared env + the offline/localhost pins the devbox
            // loader already honours. No secrets, no egress.
            let child = Command::new(&config.python_bin)
                .arg(&script)
                .current_dir(&config.devbox_root)
                .env_clear()
                .env("HOME", &home)
                .env("PATH", "/usr/local/bin:/usr/bin:/bin")
                .env("TRANSFORMERS_OFFLINE", "1")
                .env("HF_HUB_OFFLINE", "1")
                .env("PRISM_CHAMPION_BRIDGE_HOST", &config.host)
                .env("PRISM_CHAMPION_BRIDGE_PORT", config.port.to_string())
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .map_err(|err| {
                    PcError::Security(format!("failed to spawn champion devbox server: {err}"))
                })?;
            let mut bridge = Self {
                model,
                config,
                base_url,
                child: Some(child),
            };
            bridge.wait_healthy()?;
            Ok(bridge)
        }

        fn wait_healthy(&mut self) -> PcResult<()> {
            let deadline = Instant::now() + Duration::from_secs(self.config.spawn_timeout_seconds);
            while Instant::now() < deadline {
                if let Some(child) = self.child.as_mut() {
                    if let Some(status) = child.try_wait()? {
                        return Err(PcError::Security(format!(
                            "champion devbox server exited before healthy: {status}"
                        )));
                    }
                }
                if let Ok(body) = self.health_check() {
                    if body.get("model_loaded").and_then(Value::as_bool) == Some(true) {
                        return Ok(());
                    }
                }
                thread::sleep(Duration::from_millis(500));
            }
            self.shutdown();
            Err(PcError::Security(format!(
                "champion devbox server did not become healthy within {} seconds",
                self.config.spawn_timeout_seconds
            )))
        }
    }

    /// Minimal std-only HTTP/1.1 client over a loopback TcpStream. Speaks just
    /// enough HTTP for the FastAPI server; refuses any non-loopback base URL.
    pub(super) fn http_request(
        base_url: &str,
        method: &str,
        path: &str,
        body: Option<&[u8]>,
        timeout: Duration,
    ) -> PcResult<Value> {
        let addr = base_url
            .strip_prefix("http://")
            .ok_or_else(|| PcError::Security(format!("non-http champion base url: {base_url}")))?;
        let host_port = addr.trim_end_matches('/');
        let ip = host_port
            .rsplit_once(':')
            .and_then(|(h, _)| h.parse::<IpAddr>().ok())
            .ok_or_else(|| PcError::Security(format!("unparseable champion addr: {host_port}")))?;
        if !ip.is_loopback() {
            return Err(PcError::Security(format!(
                "champion bridge endpoint is not loopback (no egress): {host_port}"
            )));
        }
        let mut stream = TcpStream::connect(host_port)?;
        stream.set_read_timeout(Some(timeout))?;
        stream.set_write_timeout(Some(timeout))?;
        let body = body.unwrap_or(&[]);
        let mut request = format!(
            "{method} {path} HTTP/1.1\r\nHost: {host_port}\r\nConnection: close\r\nAccept: application/json\r\n"
        );
        if method != "GET" {
            request.push_str("Content-Type: application/json\r\n");
        }
        request.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        stream.write_all(request.as_bytes())?;
        stream.write_all(body)?;
        stream.flush()?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw)?;
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or_else(|| {
                PcError::Format("champion server response missing header/body split".to_string())
            })?;
        let head = String::from_utf8_lossy(&raw[..split]);
        let status_ok = head
            .lines()
            .next()
            .map(|line| line.contains(" 200") || line.contains(" 2"))
            .unwrap_or(false);
        let payload = &raw[split + 4..];
        let value: Value = serde_json::from_slice(payload).map_err(|err| {
            PcError::Format(format!("champion server returned non-JSON body: {err}"))
        })?;
        if !status_ok {
            return Err(PcError::Security(format!(
                "champion server non-2xx response: {}",
                head.lines().next().unwrap_or("<no status line>")
            )));
        }
        Ok(value)
    }
}

#[cfg(feature = "champion_bridge")]
use live::http_request;

// ---------------------------------------------------------------------------
// Feature-off stubs — keep the crate compiling and hermetic without Python.
// ---------------------------------------------------------------------------

#[cfg(not(feature = "champion_bridge"))]
impl ChampionBridge {
    /// Disabled unless built with `--features champion_bridge`.
    pub fn spawn(_model: CanonicalModel, _config: ChampionBridgeConfig) -> PcResult<Self> {
        Err(champion_bridge_disabled())
    }
}

#[cfg(not(feature = "champion_bridge"))]
fn http_request(
    _base_url: &str,
    _method: &str,
    _path: &str,
    _body: Option<&[u8]>,
    _timeout: Duration,
) -> PcResult<Value> {
    Err(champion_bridge_disabled())
}

#[cfg(not(feature = "champion_bridge"))]
fn champion_bridge_disabled() -> PcError {
    PcError::Security(
        "champion bridge disabled: rebuild prismcode with --features champion_bridge to enable \
         the live Python devbox proxy"
            .to_string(),
    )
}
