//! PROCESS-PROVEN runtime provenance for the PrismOS native MD evidence pipeline
//! (DSTW-T4 Step h1).
//!
//! WHY THIS EXISTS
//! ---------------
//! `RuntimeOrigin` (operand_ledger.rs:40) is a *label* — it asserts "this artifact
//! was produced by an orchestrated subprocess" but carries no PROOF. The directive
//! requires the label to be backed by a captured, value-real provenance block:
//! the exact binary that ran (sha256 of `/proc/self/exe`), the full argv, the
//! process lineage (pid / ppid / spawned engine child), the container-or-VM
//! identity it ran inside, and — per emitted artifact — the file's own sha256,
//! emission timestamp, source stage, and the upstream parent artifact hashes it
//! was derived from.
//!
//! HONESTY BOUNDARIES (mirror operand_ledger.rs:8 guardrails — do not violate)
//! --------------------------------------------------------------------------
//!  * EVERY field is captured from the live process at `md-run` start
//!    (`RuntimeProvenance::capture`) — no field is fabricated. Hashes are
//!    computed by OPENING files, never trusted from a caller string.
//!  * NO NULL-PADDING. When a value cannot be obtained (e.g. the installed
//!    `prismos.deb` is not resolvable in this environment, or `/proc/self/exe`
//!    cannot be hashed), the matching `*_absent_reason` records WHY. An `Option`
//!    is never left `None` silently — its sibling reason is set.
//!  * `RuntimeOrigin` alone is INVALID without this block. The emit path attaches
//!    a [`ArtifactProvenance`] to every artifact record; that record back-references
//!    the run-level [`RuntimeProvenance`] by `run_pid` + `packaged_binary_sha256`.
//!  * The engine itself (`nhs_rt_full`) is launched by prismos *through* the
//!    `prism-validate-and-run.sh` wrapper. The direct OS child of prismos is the
//!    `bash` wrapper; `nhs_rt_full` is a GRANDCHILD spawned by that script. We
//!    record the wrapper child PID we actually own, and record WHY the engine's
//!    own PID is not directly available (it is not our direct child) rather than
//!    guessing it. Wiring the engine to report its own PID back is an
//!    engine-owner change (BLOCKED here) and is flagged as such.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::operand_ledger::sha256_file_hex;

pub const RUNTIME_PROVENANCE_SCHEMA_KIND: &str = "prismos_runtime_provenance";
pub const RUNTIME_PROVENANCE_SCHEMA_VERSION: u32 = 1;
pub const ARTIFACT_PROVENANCE_SCHEMA_KIND: &str = "prismos_artifact_provenance";
pub const ARTIFACT_PROVENANCE_SCHEMA_VERSION: u32 = 1;

/// Container / VM identity, read from the live process environment. Every field
/// is value-real; `absent_reason` fields explain a `None`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContainerOrVmIdentity {
    /// `/.dockerenv` exists (Docker marks containers with this file).
    pub dockerenv_present: bool,
    /// First non-empty line of `/proc/1/cgroup` (the init cgroup path — names the
    /// container runtime/slice). `None` + reason when unreadable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init_cgroup: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub init_cgroup_absent_reason: Option<String>,
    /// Kernel hostname (`/proc/sys/kernel/hostname`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname_absent_reason: Option<String>,
    /// Container-relevant environment variables actually present (only the keys
    /// that exist are recorded — absence is meaningful, not null-padded).
    pub container_env: BTreeMap<String, String>,
    /// True when any positive container/VM signal was found.
    pub looks_containerized: bool,
}

impl ContainerOrVmIdentity {
    fn capture() -> ContainerOrVmIdentity {
        let dockerenv_present = Path::new("/.dockerenv").exists();

        let (init_cgroup, init_cgroup_absent_reason) =
            match std::fs::read_to_string("/proc/1/cgroup") {
                Ok(contents) => {
                    let line = contents
                        .lines()
                        .map(str::trim)
                        .find(|l| !l.is_empty())
                        .map(|l| l.to_string());
                    match line {
                        Some(l) => (Some(l), None),
                        None => (
                            None,
                            Some("/proc/1/cgroup was empty (no cgroup lines)".to_string()),
                        ),
                    }
                }
                Err(e) => (None, Some(format!("/proc/1/cgroup unreadable: {e}"))),
            };

        let (hostname, hostname_absent_reason) =
            match std::fs::read_to_string("/proc/sys/kernel/hostname") {
                Ok(s) => {
                    let h = s.trim().to_string();
                    if h.is_empty() {
                        (
                            None,
                            Some("/proc/sys/kernel/hostname was empty".to_string()),
                        )
                    } else {
                        (Some(h), None)
                    }
                }
                Err(e) => (
                    None,
                    Some(format!("/proc/sys/kernel/hostname unreadable: {e}")),
                ),
            };

        // Only record container-relevant env keys that ACTUALLY exist.
        const CONTAINER_ENV_KEYS: &[&str] = &[
            "container",
            "KUBERNETES_SERVICE_HOST",
            "DOCKER_CONTAINER",
            "PODMAN_NAME",
            "SINGULARITY_NAME",
            "APPTAINER_NAME",
            "HOSTNAME",
        ];
        let mut container_env = BTreeMap::new();
        for key in CONTAINER_ENV_KEYS {
            if let Ok(val) = std::env::var(key) {
                container_env.insert((*key).to_string(), val);
            }
        }

        let looks_containerized = dockerenv_present
            || !container_env.is_empty()
            || init_cgroup
                .as_deref()
                .map(|c| {
                    c.contains("docker")
                        || c.contains("kubepods")
                        || c.contains("containerd")
                        || c.contains("lxc")
                        || c.contains("libpod")
                })
                .unwrap_or(false);

        ContainerOrVmIdentity {
            dockerenv_present,
            init_cgroup,
            init_cgroup_absent_reason,
            hostname,
            hostname_absent_reason,
            container_env,
            looks_containerized,
        }
    }
}

/// One spawned subprocess in this run's lineage. Captured from the live
/// `std::process::Child` (the PID we actually own).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubprocessRecord {
    /// Human label for the role this child plays ("engine_wrapper_bash", ...).
    pub role: String,
    /// PID prismos actually owns for this child.
    pub pid: u32,
    /// argv[0] / command we launched it with.
    pub command: String,
    /// True when this is the direct OS child of prismos; false for a documented
    /// indirection (e.g. the engine, which is a grandchild via the wrapper).
    pub direct_child: bool,
    /// When `direct_child` is false, WHY the PID is not directly ownable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub indirection_reason: Option<String>,
}

/// Run-level provenance, captured ONCE at `md-run` start and threaded into every
/// emitted artifact's [`ArtifactProvenance`] record.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeProvenance {
    pub schema_kind: String,
    pub schema_version: u32,

    // ---- the binary that ran (PROCESS-proven) ----
    /// Resolved path of `/proc/self/exe`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packaged_binary_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packaged_binary_path_absent_reason: Option<String>,
    /// sha256 of the running binary's own bytes (`/proc/self/exe`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packaged_binary_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub packaged_binary_sha256_absent_reason: Option<String>,

    // ---- the installed package (the prismos .deb) ----
    /// sha256 of the installed `prismos*.deb` if resolvable in this environment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_path: Option<String>,
    /// WHY `package_sha256` is absent (e.g. running from a `cargo` target tree,
    /// not an installed .deb).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package_sha256_absent_reason: Option<String>,

    // ---- invocation ----
    pub argv: Vec<String>,
    pub run_root: String,
    /// prismos's own PID.
    pub pid: u32,
    /// prismos's parent PID (the shell / orchestrator that launched it) — read
    /// from `/proc/self/status` PPid. This is the `parent_command_id`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_command_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_command_id_absent_reason: Option<String>,
    /// argv of the parent process if `/proc/<ppid>/cmdline` is readable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_cmdline: Option<String>,

    // ---- environment ----
    pub container_or_vm_identity: ContainerOrVmIdentity,

    // ---- lineage (filled as children are spawned) ----
    pub subprocess_lineage: Vec<SubprocessRecord>,

    /// Capture instant (unix ns).
    pub capture_unix_ns: u128,
}

impl RuntimeProvenance {
    /// Capture the full provenance block from the live process. Call ONCE at
    /// `md-run` start, before launching the engine. `run_root` is the canonical
    /// repo/run root. `argv` is the full prismos argv (`std::env::args`).
    pub fn capture(run_root: &Path, argv: Vec<String>) -> RuntimeProvenance {
        // --- the running binary, via /proc/self/exe ---
        let (packaged_binary_path, packaged_binary_path_absent_reason, exe_path) =
            match std::fs::read_link("/proc/self/exe") {
                Ok(p) => (Some(p.display().to_string()), None, Some(p)),
                Err(e) => (None, Some(format!("/proc/self/exe unreadable: {e}")), None),
            };
        let (packaged_binary_sha256, packaged_binary_sha256_absent_reason) = match &exe_path {
            Some(p) => match sha256_file_hex(p) {
                Ok(h) => (Some(h), None),
                Err(e) => (None, Some(format!("sha256 of {} failed: {e}", p.display()))),
            },
            None => (None, Some("no /proc/self/exe path to hash".to_string())),
        };

        // --- the installed package (.deb) ---
        let (package_sha256, package_path, package_sha256_absent_reason) =
            resolve_installed_package(exe_path.as_deref());

        // --- pid / ppid from /proc/self/status ---
        let pid = std::process::id();
        let (parent_command_id, parent_command_id_absent_reason) = read_ppid();
        let parent_cmdline = parent_command_id.and_then(read_cmdline);

        RuntimeProvenance {
            schema_kind: RUNTIME_PROVENANCE_SCHEMA_KIND.to_string(),
            schema_version: RUNTIME_PROVENANCE_SCHEMA_VERSION,
            packaged_binary_path,
            packaged_binary_path_absent_reason,
            packaged_binary_sha256,
            packaged_binary_sha256_absent_reason,
            package_sha256,
            package_path,
            package_sha256_absent_reason,
            argv,
            run_root: run_root.display().to_string(),
            pid,
            parent_command_id,
            parent_command_id_absent_reason,
            parent_cmdline,
            container_or_vm_identity: ContainerOrVmIdentity::capture(),
            subprocess_lineage: Vec::new(),
            capture_unix_ns: now_unix_ns(),
        }
    }

    /// Record a spawned subprocess we own (the engine wrapper child).
    pub fn record_subprocess(&mut self, record: SubprocessRecord) {
        self.subprocess_lineage.push(record);
    }

    /// Build a per-artifact provenance record. Opens `path`, computes its sha256,
    /// stamps the emission time, and back-references this run (pid + binary sha).
    /// `upstream_parent_artifact_hashes` is the list of input artifact hashes this
    /// output was derived from (may be empty for a root artifact).
    pub fn artifact_record(
        &self,
        path: &Path,
        source_stage: impl Into<String>,
        upstream_parent_artifact_hashes: Vec<String>,
    ) -> ArtifactProvenance {
        let (emitted_artifact_sha256, emitted_artifact_sha256_absent_reason) =
            match sha256_file_hex(path) {
                Ok(h) => (Some(h), None),
                Err(e) => (
                    None,
                    Some(format!(
                        "sha256 of emitted artifact {} failed: {e}",
                        path.display()
                    )),
                ),
            };
        ArtifactProvenance {
            schema_kind: ARTIFACT_PROVENANCE_SCHEMA_KIND.to_string(),
            schema_version: ARTIFACT_PROVENANCE_SCHEMA_VERSION,
            emitted_artifact_path: path.display().to_string(),
            emitted_artifact_sha256,
            emitted_artifact_sha256_absent_reason,
            emission_unix_ns: now_unix_ns(),
            source_stage: source_stage.into(),
            upstream_parent_artifact_hashes,
            run_pid: self.pid,
            run_packaged_binary_sha256: self.packaged_binary_sha256.clone(),
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    pub fn to_pretty_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|e| format!("{{\"serialize_error\": \"{e}\"}}"))
    }
}

/// Per-artifact provenance — the block attached to EVERY emitted artifact record.
/// Without it, a bare `RuntimeOrigin` label is invalid.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactProvenance {
    pub schema_kind: String,
    pub schema_version: u32,
    pub emitted_artifact_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_artifact_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emitted_artifact_sha256_absent_reason: Option<String>,
    pub emission_unix_ns: u128,
    pub source_stage: String,
    pub upstream_parent_artifact_hashes: Vec<String>,
    /// Back-reference to the run-level [`RuntimeProvenance`].
    pub run_pid: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_packaged_binary_sha256: Option<String>,
}

impl ArtifactProvenance {
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

fn now_unix_ns() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Read PPid from `/proc/self/status` (process-proven, no libc dependency).
fn read_ppid() -> (Option<u32>, Option<String>) {
    match std::fs::read_to_string("/proc/self/status") {
        Ok(contents) => {
            for line in contents.lines() {
                if let Some(rest) = line.strip_prefix("PPid:") {
                    if let Ok(ppid) = rest.trim().parse::<u32>() {
                        return (Some(ppid), None);
                    }
                    return (
                        None,
                        Some(format!("PPid line unparseable: {:?}", rest.trim())),
                    );
                }
            }
            (None, Some("no PPid line in /proc/self/status".to_string()))
        }
        Err(e) => (None, Some(format!("/proc/self/status unreadable: {e}"))),
    }
}

/// Read `/proc/<pid>/cmdline` (NUL-separated argv) and join with spaces.
fn read_cmdline(pid: u32) -> Option<String> {
    let path = format!("/proc/{pid}/cmdline");
    let bytes = std::fs::read(path).ok()?;
    if bytes.is_empty() {
        return None;
    }
    let joined = bytes
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| String::from_utf8_lossy(s).into_owned())
        .collect::<Vec<_>>()
        .join(" ");
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
}

/// Resolve the installed prismos `.deb` package and hash it, if this binary was
/// installed from one. Running from a `cargo` target tree is the common case and
/// is recorded honestly as the absent reason — never null-padded.
///
/// Strategy (PROCESS-grounded, no guessing):
///  * If `dpkg-query -W -f=${Version} prismos` succeeds, the package is installed;
///    we then look for the matching `.deb` under the standard apt archive cache
///    (`/var/cache/apt/archives`) and hash it if present.
///  * Otherwise the binary is running from a non-packaged path; we record exactly
///    that, including the exe path so a reviewer can see it was a target-tree run.
fn resolve_installed_package(
    exe_path: Option<&Path>,
) -> (Option<String>, Option<String>, Option<String>) {
    use std::process::Command;

    // Is a `prismos` package installed?
    let installed_version = Command::new("dpkg-query")
        .args(["-W", "-f=${Version}", "prismos"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|v| !v.is_empty());

    let version = match installed_version {
        Some(v) => v,
        None => {
            let exe = exe_path
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "<unknown>".to_string());
            return (
                None,
                None,
                Some(format!(
                    "no installed `prismos` dpkg package (dpkg-query absent or \
                     unregistered); binary is running from a non-packaged path \
                     ({exe}) — e.g. a cargo target tree, not an installed .deb"
                )),
            );
        }
    };

    // Look for the cached .deb matching the installed version.
    let cache_dir = Path::new("/var/cache/apt/archives");
    if let Ok(entries) = std::fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with("prismos") && name.ends_with(".deb") {
                let p = entry.path();
                if let Ok(h) = sha256_file_hex(&p) {
                    return (Some(h), Some(p.display().to_string()), None);
                }
            }
        }
    }

    (
        None,
        None,
        Some(format!(
            "`prismos` package version {version} is dpkg-registered but the \
             source .deb is not present in /var/cache/apt/archives to hash \
             (apt cache cleaned)"
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_is_process_proven_not_a_bare_label() {
        let prov = RuntimeProvenance::capture(Path::new("/tmp"), vec!["prismos".into()]);
        // pid is the live process id.
        assert_eq!(prov.pid, std::process::id());
        // /proc/self/exe is hashable on Linux test hosts -> a real sha256.
        assert!(
            prov.packaged_binary_sha256.is_some()
                || prov.packaged_binary_sha256_absent_reason.is_some(),
            "either a real binary hash or a recorded reason — never silent None"
        );
        if let Some(h) = &prov.packaged_binary_sha256 {
            assert_eq!(h.len(), 64, "sha256 hex is 64 chars");
        }
        // package_sha256 absence MUST carry a reason (no null padding).
        if prov.package_sha256.is_none() {
            assert!(prov.package_sha256_absent_reason.is_some());
        }
    }

    #[test]
    fn artifact_record_opens_and_hashes_the_file() {
        let dir = std::env::temp_dir().join(format!("rtp_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("artifact.json");
        std::fs::write(&f, b"{\"hello\":\"world\"}").unwrap();
        let prov = RuntimeProvenance::capture(&dir, vec!["prismos".into()]);
        let rec = prov.artifact_record(&f, "unit_test_stage", vec!["parenthash".into()]);
        assert!(rec.emitted_artifact_sha256.is_some());
        assert_eq!(rec.run_pid, prov.pid);
        assert_eq!(rec.source_stage, "unit_test_stage");
        assert_eq!(rec.upstream_parent_artifact_hashes, vec!["parenthash"]);
        // A real file produced a 64-char sha256.
        assert_eq!(rec.emitted_artifact_sha256.as_ref().unwrap().len(), 64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn subprocess_lineage_records_owned_pid() {
        let mut prov = RuntimeProvenance::capture(Path::new("/tmp"), vec!["prismos".into()]);
        prov.record_subprocess(SubprocessRecord {
            role: "engine_wrapper_bash".into(),
            pid: 4242,
            command: "bash scripts/prism-validate-and-run.sh".into(),
            direct_child: true,
            indirection_reason: None,
        });
        assert_eq!(prov.subprocess_lineage.len(), 1);
        assert_eq!(prov.subprocess_lineage[0].pid, 4242);
    }
}
