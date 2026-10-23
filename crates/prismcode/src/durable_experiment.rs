//! Durable experiment lifecycle around the canonical Hyper-Runtime owner.
//!
//! Journal recovery never implies restoration of MD state. An interrupted
//! native stage requires a native scientific checkpoint; it is not re-run in
//! place. Committed downstream stages can be verified and resumed separately.
use crate::binary_runtime::{PcError, PcResult};
use crate::hyper_runtime_request::{ArtifactIdentityV1, HyperRuntimeRequestV1};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ExperimentPolicyV1 {
    pub schema_version: u32,
    /// Frozen analysis intent, including selections, references and sampling.
    pub qualification_policy: ArtifactIdentityV1,
    /// Full input dependency inventory admitted before native execution.
    pub qualification_inputs: Vec<ArtifactIdentityV1>,
    pub python: ArtifactIdentityV1,
    /// Hash inventory of the package and scientific dependency installation.
    pub analysis_installation_manifest: ArtifactIdentityV1,
    pub analysis_import_root: PathBuf,
    /// Disk admission is measured before starting the native stage.
    pub minimum_free_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRecord {
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InstallationManifest {
    schema_version: u32,
    root: PathBuf,
    files: Vec<ArtifactRecord>,
    /// Installation-relative paths; site initialization is deliberately disabled.
    loader: PathBuf,
    python_bootstrap: PathBuf,
    python_contract: PathBuf,
    python_home: PathBuf,
    python_paths: Vec<PathBuf>,
    library_paths: Vec<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    Admitted,
    NativeStarted,
    NativeResumed,
    NativeCommitted,
    QualificationStarted,
    QualificationCommitted,
    Failed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct JournalEvent {
    pub schema_version: u32,
    pub sequence: u64,
    pub previous_sha256: Option<String>,
    pub request_id_sha256: String,
    pub kind: EventKind,
    pub stage: String,
    pub artifact_manifest: Option<ArtifactIdentityV1>,
    pub detail: Option<String>,
}

pub struct ExperimentJournal {
    root: PathBuf,
    request_id: String,
    _lock: File,
    writable: bool,
    pub events: Vec<JournalEvent>,
    last_sha256: Option<String>,
}

fn reject(message: impl Into<String>) -> PcError {
    PcError::Security(message.into())
}

fn strict_path(path: &Path, regular: bool) -> PcResult<()> {
    if !path.is_absolute() {
        return Err(reject("experiment paths must be absolute"));
    }
    let mut prefix = PathBuf::new();
    for c in path.components() {
        if matches!(c, Component::ParentDir | Component::CurDir) {
            return Err(reject("experiment path contains traversal"));
        }
        prefix.push(c.as_os_str());
        let m = fs::symlink_metadata(&prefix)?;
        if m.file_type().is_symlink() {
            return Err(reject(format!("symlink: {}", prefix.display())));
        }
        if prefix != path && !m.is_dir() {
            return Err(reject("non-directory ancestor"));
        }
    }
    let m = fs::symlink_metadata(path)?;
    if (regular && !m.is_file()) || (!regular && !m.is_dir()) {
        return Err(reject(format!("wrong file type: {}", path.display())));
    }
    Ok(())
}

fn relative(path: &Path) -> PcResult<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(reject(
            "artifact locator must be a nonempty relative path without traversal",
        ));
    }
    Ok(())
}

fn digest_file(path: &Path, synchronize: bool) -> PcResult<(u64, String)> {
    strict_path(path, true)?;
    let mut f = File::open(path)?;
    let before = f.metadata()?;
    if synchronize {
        f.sync_all()?;
    }
    let mut h = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    let mut size = 0u64;
    loop {
        let n = f.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
        size += n as u64;
    }
    let after = f.metadata()?;
    if before.len() != size || after.len() != size || before.modified()? != after.modified()? {
        return Err(reject(format!(
            "artifact changed during read: {}",
            path.display()
        )));
    }
    Ok((size, format!("{:x}", h.finalize())))
}

fn verify_identity(identity: &ArtifactIdentityV1) -> PcResult<()> {
    let (_, observed) = digest_file(&identity.locator, false)?;
    if observed != identity.sha256 {
        return Err(reject(format!(
            "identity mismatch: {}",
            identity.locator.display()
        )));
    }
    Ok(())
}

/// Shared QNANL publication primitive; publication preserves exact existing bytes.
fn publish(path: &Path, bytes: &[u8]) -> PcResult<()> {
    crate::qnanl_artifact::publish_immutable_artifact(path, bytes)
        .map_err(|error| reject(format!("durable publication: {error}")))
}

impl ExperimentJournal {
    pub fn open(root: &Path, request_id: &str) -> PcResult<Self> {
        Self::open_mode(root, request_id, true)
    }

    fn read(root: &Path, request_id: &str) -> PcResult<Self> {
        Self::open_mode(root, request_id, false)
    }

    fn open_mode(root: &Path, request_id: &str, writable: bool) -> PcResult<Self> {
        if request_id.len() != 64 || !request_id.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(reject("invalid request identity"));
        }
        if !root.exists() {
            if !writable {
                return Err(reject("retained journal does not exist"));
            }
            strict_path(
                root.parent()
                    .ok_or_else(|| reject("journal parent missing"))?,
                false,
            )?;
            fs::create_dir(root)?;
            File::open(root.parent().unwrap())?.sync_all()?;
        }
        strict_path(root, false)?;
        let lock_path = root.join("writer.lock");
        if lock_path.exists() {
            strict_path(&lock_path, true)?;
        }
        let lock = if writable {
            let lock = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(lock_path)?;
            lock.try_lock()
                .map_err(|e| reject(format!("experiment has another writer or verifier: {e}")))?;
            lock
        } else {
            let lock = File::open(lock_path)?;
            lock.try_lock_shared()
                .map_err(|e| reject(format!("experiment has an active writer: {e}")))?;
            lock
        };
        let mut paths = Vec::new();
        for entry in fs::read_dir(root)? {
            let p = entry?.path();
            let name = p.file_name().unwrap().to_string_lossy();
            if name.starts_with("event-") && name.ends_with(".json") {
                paths.push(p);
            }
        }
        paths.sort();
        let mut journal = Self {
            root: root.to_path_buf(),
            request_id: request_id.into(),
            _lock: lock,
            writable,
            events: Vec::new(),
            last_sha256: None,
        };
        for (index, path) in paths.iter().enumerate() {
            if path.file_name().unwrap() != format!("event-{index:020}.json").as_str() {
                return Err(reject("journal event sequence missing or duplicated"));
            }
            strict_path(path, true)?;
            let bytes = fs::read(path)?;
            let event: JournalEvent = serde_json::from_slice(&bytes)?;
            if event.schema_version != 1
                || event.sequence != index as u64
                || event.request_id_sha256 != request_id
                || event.previous_sha256 != journal.last_sha256
            {
                return Err(reject("journal lineage or request mismatch"));
            }
            journal.validate_transition(&event.kind, &event.stage)?;
            if let Some(artifact) = &event.artifact_manifest {
                relative(&artifact.locator)?;
                verify_identity(&ArtifactIdentityV1 {
                    locator: root.join(&artifact.locator),
                    sha256: artifact.sha256.clone(),
                })?;
            } else if matches!(
                event.kind,
                EventKind::NativeCommitted | EventKind::QualificationCommitted
            ) {
                return Err(reject("committed journal event lacks artifact manifest"));
            }
            journal.last_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
            journal.events.push(event);
        }
        Ok(journal)
    }

    fn validate_transition(&self, kind: &EventKind, stage: &str) -> PcResult<()> {
        let native_committed = self
            .events
            .iter()
            .any(|e| e.kind == EventKind::NativeCommitted);
        let qualification_committed = self
            .events
            .iter()
            .any(|e| e.kind == EventKind::QualificationCommitted);
        let last = self.events.last().map(|e| &e.kind);
        let allowed = match kind {
            EventKind::Admitted => self.events.is_empty() && stage == "experiment",
            EventKind::NativeStarted => {
                last == Some(&EventKind::Admitted) && stage == "native_pipeline"
            }
            EventKind::NativeResumed => {
                !native_committed
                    && stage == "native_pipeline"
                    && self
                        .events
                        .last()
                        .map(|e| {
                            e.stage == stage
                                && matches!(
                                    e.kind,
                                    EventKind::NativeStarted
                                        | EventKind::NativeResumed
                                        | EventKind::Failed
                                )
                        })
                        .unwrap_or(false)
            }
            EventKind::NativeCommitted => {
                matches!(
                    last,
                    Some(EventKind::NativeStarted | EventKind::NativeResumed)
                ) && stage == "native_pipeline"
            }
            EventKind::QualificationStarted => {
                native_committed
                    && !qualification_committed
                    && stage == "qualification"
                    && (last == Some(&EventKind::NativeCommitted)
                        || last == Some(&EventKind::Failed)
                        || last == Some(&EventKind::QualificationStarted))
            }
            EventKind::QualificationCommitted => {
                last == Some(&EventKind::QualificationStarted) && stage == "qualification"
            }
            EventKind::Failed => self
                .events
                .last()
                .map(|e| {
                    e.stage == stage
                        && matches!(
                            e.kind,
                            EventKind::NativeStarted
                                | EventKind::NativeResumed
                                | EventKind::QualificationStarted
                        )
                })
                .unwrap_or(false),
        };
        if !allowed {
            return Err(reject(format!(
                "illegal experiment transition {last:?} -> {kind:?}"
            )));
        }
        Ok(())
    }

    fn append(
        &mut self,
        kind: EventKind,
        stage: &str,
        artifact_manifest: Option<ArtifactIdentityV1>,
        detail: Option<String>,
    ) -> PcResult<()> {
        if !self.writable {
            return Err(reject("read-only journal cannot publish events"));
        }
        self.validate_transition(&kind, stage)?;
        let event = JournalEvent {
            schema_version: 1,
            sequence: self.events.len() as u64,
            previous_sha256: self.last_sha256.clone(),
            request_id_sha256: self.request_id.clone(),
            kind,
            stage: stage.into(),
            artifact_manifest,
            detail,
        };
        let mut bytes = serde_json::to_vec_pretty(&event)?;
        bytes.push(b'\n');
        publish(
            &self.root.join(format!("event-{:020}.json", event.sequence)),
            &bytes,
        )?;
        self.last_sha256 = Some(format!("{:x}", Sha256::digest(&bytes)));
        self.events.push(event);
        Ok(())
    }

    fn manifest(
        &self,
        stage: &str,
        base: &Path,
        excluded: &[PathBuf],
    ) -> PcResult<ArtifactIdentityV1> {
        if !self.writable {
            return Err(reject("read-only journal cannot publish manifests"));
        }
        let mut files = Vec::new();
        collect(base, base, excluded, &mut files)?;
        files.sort_by(|a, b| a.path.cmp(&b.path));
        if files.is_empty() {
            return Err(reject("stage produced no artifacts"));
        }
        let bytes = serde_json::to_vec_pretty(&files)?;
        let locator = self
            .root
            .join(format!("{stage}-{:020}.manifest.json", self.events.len()));
        publish(&locator, &bytes)?;
        Ok(ArtifactIdentityV1 {
            locator: locator.strip_prefix(&self.root).unwrap().into(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        })
    }

    fn verify_committed(&self, kind: EventKind, base: &Path) -> PcResult<bool> {
        let Some(event) = self.events.iter().find(|e| e.kind == kind) else {
            return Ok(false);
        };
        let identity = event
            .artifact_manifest
            .as_ref()
            .ok_or_else(|| reject("committed stage missing manifest"))?;
        relative(&identity.locator)?;
        let manifest_path = self.root.join(&identity.locator);
        verify_identity(&ArtifactIdentityV1 {
            locator: manifest_path.clone(),
            sha256: identity.sha256.clone(),
        })?;
        let files: Vec<ArtifactRecord> = serde_json::from_slice(&fs::read(manifest_path)?)?;
        if files.is_empty() {
            return Err(reject("empty committed stage"));
        }
        let resolved_base = if kind == EventKind::QualificationCommitted {
            let attempt = Path::new(
                event
                    .detail
                    .as_deref()
                    .ok_or_else(|| reject("qualification attempt missing"))?,
            );
            relative(attempt)?;
            base.join(attempt)
        } else {
            base.to_path_buf()
        };
        verify_files(&resolved_base, &files)?;
        let excluded = if kind == EventKind::NativeCommitted {
            vec![self.root.clone(), base.join("STRUCTURAL_QUALIFICATION")]
        } else {
            Vec::new()
        };
        let mut actual = std::collections::BTreeSet::new();
        stage_population(&resolved_base, &resolved_base, &excluded, &mut actual)?;
        let expected: std::collections::BTreeSet<_> =
            files.iter().map(|f| f.path.clone()).collect();
        if actual != expected {
            return Err(reject("committed stage population changed"));
        }
        Ok(true)
    }
}

fn collect(
    base: &Path,
    directory: &Path,
    excluded: &[PathBuf],
    files: &mut Vec<ArtifactRecord>,
) -> PcResult<()> {
    strict_path(directory, false)?;
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if excluded.iter().any(|p| path.starts_with(p)) {
            continue;
        }
        let metadata = fs::symlink_metadata(&path)?;
        if metadata.file_type().is_symlink() {
            return Err(reject("stage contains a symlink"));
        }
        if metadata.is_dir() {
            collect(base, &path, excluded, files)?;
        } else if metadata.is_file() {
            let (bytes, sha256) = digest_file(&path, true)?;
            files.push(ArtifactRecord {
                path: path
                    .strip_prefix(base)
                    .map_err(|_| reject("artifact escaped root"))?
                    .into(),
                bytes,
                sha256,
            });
        } else {
            return Err(reject("stage contains a special file"));
        }
    }
    File::open(directory)?.sync_all()?;
    Ok(())
}

fn verify_files(root: &Path, files: &[ArtifactRecord]) -> PcResult<()> {
    let mut seen = std::collections::BTreeSet::new();
    for file in files {
        relative(&file.path)?;
        if !seen.insert(&file.path) {
            return Err(reject("duplicate artifact in manifest"));
        }
        let observed = digest_file(&root.join(&file.path), false)?;
        if observed != (file.bytes, file.sha256.clone()) {
            return Err(reject(format!(
                "committed artifact changed: {}",
                file.path.display()
            )));
        }
    }
    Ok(())
}

fn stage_population(
    root: &Path,
    directory: &Path,
    excluded: &[PathBuf],
    result: &mut std::collections::BTreeSet<PathBuf>,
) -> PcResult<()> {
    strict_path(directory, false)?;
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if excluded.iter().any(|e| path.starts_with(e)) {
            continue;
        }
        let meta = fs::symlink_metadata(&path)?;
        if meta.file_type().is_symlink() {
            return Err(reject("committed stage contains a symlink"));
        }
        if meta.is_dir() {
            stage_population(root, &path, excluded, result)?;
        } else if meta.is_file() {
            result.insert(path.strip_prefix(root).unwrap().into());
        } else {
            return Err(reject("committed stage contains a special file"));
        }
    }
    Ok(())
}

fn member_population(
    root: &Path,
    directory: &Path,
    result: &mut std::collections::BTreeSet<PathBuf>,
) -> PcResult<()> {
    strict_path(directory, false)?;
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        let meta = fs::symlink_metadata(&path)?;
        if meta.file_type().is_symlink() {
            return Err(reject("installation contains an unbound symlink"));
        }
        if meta.is_dir() {
            member_population(root, &path, result)?;
        } else if meta.is_file() {
            result.insert(path.strip_prefix(root).unwrap().into());
        } else {
            return Err(reject("installation contains a special file"));
        }
    }
    Ok(())
}

struct RetainedInputResolver {
    manifest: ArtifactIdentityV1,
    objects: PathBuf,
    sources: std::collections::BTreeMap<PathBuf, RetainedInput>,
}

impl RetainedInputResolver {
    fn from_journal(
        journal: &ExperimentJournal,
        request: &HyperRuntimeRequestV1,
    ) -> PcResult<Self> {
        verify_retained_inputs(journal, request)?;
        let event = journal
            .events
            .first()
            .ok_or_else(|| reject("retained input admission missing"))?;
        let committed = event
            .artifact_manifest
            .as_ref()
            .ok_or_else(|| reject("retained input manifest is not journal-bound"))?;
        if event.kind != EventKind::Admitted
            || committed.locator != Path::new("scientific-inputs.manifest.json")
        {
            return Err(reject("retained input manifest differs from admission"));
        }
        let manifest = ArtifactIdentityV1 {
            locator: journal.root.join(&committed.locator),
            sha256: committed.sha256.clone(),
        };
        verify_identity(&manifest)?;
        let rows: Vec<RetainedInput> = serde_json::from_slice(&fs::read(&manifest.locator)?)?;
        let sources = rows
            .into_iter()
            .map(|r| (r.source.locator.clone(), r))
            .collect();
        Ok(Self {
            manifest,
            objects: journal.root.join("INPUT_OBJECTS"),
            sources,
        })
    }

    fn resolve(&self, identity: &ArtifactIdentityV1) -> PcResult<PathBuf> {
        let row = self
            .sources
            .get(&identity.locator)
            .ok_or_else(|| reject("input absent from admitted retained population"))?;
        if &row.source != identity {
            return Err(reject(
                "retained source identity differs from canonical request",
            ));
        }
        verify_files(&self.objects, std::slice::from_ref(&row.object))?;
        Ok(self.objects.join(&row.object.path))
    }

    fn bind_command(
        &self,
        command: &mut Command,
        journal_root: &Path,
        stage: &str,
    ) -> PcResult<()> {
        static RESOLUTION_SEQUENCE: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let workspace = journal_root.join(format!(
            "retained-{stage}-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| reject("wall clock before epoch for workspace naming"))?
                .as_nanos(),
            RESOLUTION_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        command
            .arg("--retained-inputs")
            .arg(&self.manifest.locator)
            .arg("--retained-inputs-sha256")
            .arg(&self.manifest.sha256)
            .arg("--retained-workspace")
            .arg(workspace);
        Ok(())
    }

    fn verify_receipt(&self, receipt: &serde_json::Value) -> PcResult<()> {
        if receipt
            .pointer("/retained_inputs/manifest_sha256")
            .and_then(|v| v.as_str())
            != Some(self.manifest.sha256.as_str())
            || receipt
                .pointer("/retained_inputs/original_locators_preserved")
                .and_then(|v| v.as_bool())
                != Some(true)
            || receipt
                .pointer("/retained_inputs/verified_binding_count")
                .and_then(|v| v.as_u64())
                != Some(self.sources.len() as u64)
        {
            return Err(reject(
                "qualification retained input receipt differs from admitted archive",
            ));
        }
        verify_identity(&self.manifest)?;
        Ok(())
    }
}

fn source_path(
    identity: &ArtifactIdentityV1,
    retained: Option<&RetainedInputResolver>,
) -> PcResult<PathBuf> {
    if let Some(resolver) = retained {
        resolver.resolve(identity)
    } else {
        verify_identity(identity)?;
        Ok(identity.locator.clone())
    }
}

pub fn validate_policy(policy: &ExperimentPolicyV1) -> PcResult<()> {
    validate_policy_sources(policy, None)?;
    validate_installation(policy)
}

fn validate_policy_sources(
    policy: &ExperimentPolicyV1,
    retained: Option<&RetainedInputResolver>,
) -> PcResult<()> {
    if policy.schema_version != 1 || policy.minimum_free_bytes == 0 {
        return Err(reject("invalid experiment policy"));
    }
    let policy_path = source_path(&policy.qualification_policy, retained)?;
    if policy.qualification_inputs.is_empty() {
        return Err(reject("qualification dependency population is empty"));
    }
    let mut input_paths = std::collections::BTreeSet::new();
    for binding in &policy.qualification_inputs {
        source_path(binding, retained)?;
        if !input_paths.insert(binding.locator.clone()) {
            return Err(reject("duplicate qualification input binding"));
        }
    }
    let scientific: serde_json::Value = serde_json::from_slice(&fs::read(&policy_path)?)?;
    let declared = scientific
        .get("input_bindings")
        .and_then(|v| v.as_array())
        .ok_or_else(|| reject("qualification discovery policy lacks frozen input_bindings"))?;
    let mut declared_bindings = std::collections::BTreeMap::new();
    for item in declared {
        let path = Path::new(
            item.get("path")
                .and_then(|v| v.as_str())
                .ok_or_else(|| reject("qualification binding lacks path"))?,
        );
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            policy
                .qualification_policy
                .locator
                .parent()
                .unwrap()
                .join(path)
        };
        let hash = item
            .get("sha256")
            .and_then(|v| v.as_str())
            .ok_or_else(|| reject("qualification binding lacks sha256"))?
            .to_string();
        if declared_bindings.insert(path, hash).is_some() {
            return Err(reject("duplicate scientific input binding"));
        }
    }
    let bound: std::collections::BTreeMap<_, _> = policy
        .qualification_inputs
        .iter()
        .map(|v| (v.locator.clone(), v.sha256.clone()))
        .collect();
    if declared_bindings != bound {
        return Err(reject(
            "qualification dependency set differs from canonical request",
        ));
    }
    for key in ["topology", "fpocket", "prismqnanl", "reference_manifest"] {
        if let Some(value) = scientific.get(key) {
            let value = value
                .as_str()
                .ok_or_else(|| reject(format!("qualification {key} must be a path")))?;
            let path = Path::new(value);
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                policy
                    .qualification_policy
                    .locator
                    .parent()
                    .unwrap()
                    .join(path)
            };
            if !input_paths.contains(&path) {
                return Err(reject(format!(
                    "qualification dependency not bound before run: {key}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_installation(policy: &ExperimentPolicyV1) -> PcResult<()> {
    if policy.schema_version != 1 || policy.minimum_free_bytes == 0 {
        return Err(reject("invalid experiment policy"));
    }
    verify_identity(&policy.python)?;
    verify_identity(&policy.analysis_installation_manifest)?;
    strict_path(&policy.analysis_import_root, false)?;
    let installation: InstallationManifest =
        serde_json::from_slice(&fs::read(&policy.analysis_installation_manifest.locator)?)?;
    if installation.schema_version != 1 || installation.files.is_empty() {
        return Err(reject("invalid analysis installation manifest"));
    }
    strict_path(&installation.root, false)?;
    validate_runtime_layout(&installation)?;
    verify_files(&installation.root, &installation.files)?;
    let mut actual = std::collections::BTreeSet::new();
    member_population(&installation.root, &installation.root, &mut actual)?;
    // Only the pinned inventory itself is excluded from its own population.
    if let Ok(path) = policy
        .analysis_installation_manifest
        .locator
        .strip_prefix(&installation.root)
    {
        actual.remove(path);
    }
    let expected: std::collections::BTreeSet<_> =
        installation.files.iter().map(|f| f.path.clone()).collect();
    if actual != expected {
        return Err(reject(
            "analysis installation has missing or unlisted executable/dependency files",
        ));
    }
    let contract: serde_json::Value = serde_json::from_slice(&fs::read(
        installation.root.join(&installation.python_contract),
    )?)?;
    let runtime = contract
        .get("runtime")
        .ok_or_else(|| reject("Python launch contract lacks runtime"))?;
    if runtime.get("python_home") != Some(&serde_json::to_value(&installation.python_home)?)
        || runtime.get("python_paths") != Some(&serde_json::to_value(&installation.python_paths)?)
        || runtime
            .pointer("/entrypoints/python")
            .and_then(|v| v.as_str())
            .map(|p| installation.root.join(p))
            != Some(policy.python.locator.clone())
    {
        return Err(reject(
            "Python launch contract disagrees with installation admission",
        ));
    }
    let files = contract
        .get("files")
        .and_then(|v| v.as_array())
        .ok_or_else(|| reject("Python launch contract member population absent"))?;
    for file in files {
        let path = Path::new(
            file.as_str()
                .ok_or_else(|| reject("invalid Python contract member"))?,
        );
        relative(path)?;
        if !expected.contains(path) {
            return Err(reject("Python contract admits an unpinned member"));
        }
    }
    let python_relative = policy
        .python
        .locator
        .strip_prefix(&installation.root)
        .map_err(|_| reject("Python interpreter outside pinned installation"))?;
    if !expected.contains(python_relative) {
        return Err(reject(
            "Python interpreter absent from installation inventory",
        ));
    }
    let required = policy
        .analysis_import_root
        .join("prism_dstw/qualification/__main__.py");
    let relative_required = required
        .strip_prefix(&installation.root)
        .map_err(|_| reject("analysis import root outside pinned installation"))?;
    if !installation
        .files
        .iter()
        .any(|f| f.path == relative_required)
    {
        return Err(reject(
            "qualification entry point absent from pinned installation",
        ));
    }
    Ok(())
}

fn validate_runtime_layout(installation: &InstallationManifest) -> PcResult<()> {
    if installation.python_paths.is_empty() || installation.library_paths.is_empty() {
        return Err(reject("Python runtime search population must be explicit"));
    }
    for file in [
        &installation.loader,
        &installation.python_bootstrap,
        &installation.python_contract,
    ] {
        relative(file)?;
        strict_path(&installation.root.join(file), true)?;
    }
    relative(&installation.python_home)?;
    strict_path(&installation.root.join(&installation.python_home), false)?;
    for path in installation
        .python_paths
        .iter()
        .chain(&installation.library_paths)
    {
        relative(path)?;
        strict_path(&installation.root.join(path), false)?;
    }
    Ok(())
}

fn qualification_command(policy: &ExperimentPolicyV1) -> PcResult<Command> {
    let installation: InstallationManifest =
        serde_json::from_slice(&fs::read(&policy.analysis_installation_manifest.locator)?)?;
    validate_runtime_layout(&installation)?;
    let home = installation.root.join(&installation.python_home);
    let paths: Vec<_> = installation
        .python_paths
        .iter()
        .map(|p| installation.root.join(p))
        .collect();
    if !paths.contains(&policy.analysis_import_root) {
        return Err(reject(
            "qualification import root absent from explicit runtime path",
        ));
    }
    let library_paths = std::env::join_paths(
        installation
            .library_paths
            .iter()
            .map(|p| installation.root.join(p)),
    )
    .map_err(|_| reject("invalid native library search path"))?;
    let mut command = Command::new(installation.root.join(&installation.loader));
    command
        .arg("--library-path")
        .arg(&library_paths)
        .arg(&policy.python.locator)
        .args(["-S", "-P", "-B"])
        .arg(installation.root.join(&installation.python_bootstrap))
        .arg(&installation.root)
        .arg(installation.root.join(&installation.python_contract))
        .args(["-m", "prism_dstw.qualification"])
        .env_clear()
        .env("PATH", installation.root.join("bin"))
        .env("LANG", "C.UTF-8")
        .env("PYTHONHOME", &home)
        .env("LD_LIBRARY_PATH", library_paths)
        .env("PYTHONNOUSERSITE", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::null());
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        let parent = unsafe { libc::getpid() };
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0
                    || libc::setpgid(0, 0) != 0
                {
                    return Err(std::io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "experiment parent exited",
                    ));
                }
                Ok(())
            });
        }
    }
    Ok(command)
}

fn validate_transitive_policy(
    policy: &ExperimentPolicyV1,
    retained: Option<&RetainedInputResolver>,
    journal_root: &Path,
) -> PcResult<()> {
    let mut command = qualification_command(policy)?;
    if let Some(resolver) = retained {
        resolver.bind_command(&mut command, journal_root, "admission")?;
    }
    let output = command
        .arg("--validate-policy")
        .arg(&policy.qualification_policy.locator)
        .current_dir(&policy.analysis_import_root)
        .output()?;
    if !output.status.success() {
        return Err(reject(format!(
            "pre-MD qualification admission failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let receipt: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    if let Some(resolver) = retained {
        resolver.verify_receipt(&receipt)?;
    }
    if receipt.get("status").and_then(|v| v.as_str()) != Some("INPUTS_VERIFIED")
        || receipt.get("policy_sha256").and_then(|v| v.as_str())
            != Some(&policy.qualification_policy.sha256)
    {
        return Err(reject(
            "qualification pre-MD input admission receipt mismatched",
        ));
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn free_bytes(directory: &Path) -> PcResult<u64> {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(directory.as_os_str().as_bytes())
        .map_err(|_| reject("NUL in path"))?;
    let mut value = std::mem::MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::statvfs(path.as_ptr(), value.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let v = unsafe { value.assume_init() };
    v.f_bavail
        .checked_mul(v.f_frsize)
        .ok_or_else(|| reject("filesystem capacity overflow"))
}

#[cfg(not(target_os = "linux"))]
fn free_bytes(_: &Path) -> PcResult<u64> {
    Err(reject("durable native experiments currently require Linux"))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RetainedInput {
    source: ArtifactIdentityV1,
    object: ArtifactRecord,
}

fn copy_retained_input(identity: &ArtifactIdentityV1, objects: &Path) -> PcResult<ArtifactRecord> {
    verify_identity(identity)?;
    let destination = objects.join(&identity.sha256);
    let (expected_bytes, _) = digest_file(&identity.locator, false)?;
    if destination.exists() {
        if digest_file(&destination, false)? != (expected_bytes, identity.sha256.clone()) {
            return Err(reject(
                "retained input object conflicts with source identity",
            ));
        }
    } else {
        static TEMP_SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let temporary = objects.join(format!(
            "{}.{}.{}.{}.pending",
            identity.sha256,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|_| reject("wall clock before epoch for temporary file naming"))?
                .as_nanos(),
            TEMP_SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));

        let mut source = File::open(&identity.locator)?;
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temporary)?;
        let before = source.metadata()?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut size = 0u64;
        loop {
            let n = source.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            output.write_all(&buffer[..n])?;
            digest.update(&buffer[..n]);
            size += n as u64;
        }
        output.sync_all()?;
        let after = source.metadata()?;
        if size != expected_bytes
            || before.modified()? != after.modified()?
            || format!("{:x}", digest.finalize()) != identity.sha256
        {
            return Err(reject("source input changed during retention"));
        }
        // No-clobber link preserves an earlier committed object. The journal
        // writer lock owns this namespace; orphan pending bytes are retained.
        fs::hard_link(&temporary, &destination)?;
        File::open(objects)?.sync_all()?;
        fs::remove_file(&temporary)?;
        File::open(objects)?.sync_all()?;
    }
    Ok(ArtifactRecord {
        path: PathBuf::from(&identity.sha256),
        bytes: expected_bytes,
        sha256: identity.sha256.clone(),
    })
}

fn scientific_input_sources(
    request: &HyperRuntimeRequestV1,
    policy: &ExperimentPolicyV1,
) -> PcResult<std::collections::BTreeMap<PathBuf, ArtifactIdentityV1>> {
    let q = &request.request;
    let mut sources = vec![
        q.topology.clone(),
        q.input.clone(),
        q.packet_authority.prismcode_packet.clone(),
        q.packet_authority.qnanl_binding.clone(),
        policy.qualification_policy.clone(),
    ];
    sources.extend(policy.qualification_inputs.clone());
    if let Some(branch) = &q.governed_branch {
        for source in [
            &branch.governed_branch_selection,
            &branch.explicit_selection,
            &branch.selected_proposal,
            &branch.normalization_receipt,
            &branch.validation_receipt,
        ] {
            sources.push(ArtifactIdentityV1 {
                locator: source.locator.clone(),
                sha256: source.sha256.clone(),
            });
        }
    }
    if let Some(coupled) = &q.coupled {
        sources.push(coupled.topology_b.clone());
    }
    if let Some(native) = &q.native_engine_intent {
        sources.extend(native.transport_input_bindings.clone());
    }
    let mut unique = std::collections::BTreeMap::new();
    for source in sources {
        if let Some(existing) = unique.insert(source.locator.clone(), source.clone()) {
            if existing != source {
                return Err(reject("conflicting retained input path bindings"));
            }
        }
    }
    Ok(unique)
}

fn retain_scientific_inputs(
    request: &HyperRuntimeRequestV1,
    policy: &ExperimentPolicyV1,
    journal: &ExperimentJournal,
) -> PcResult<()> {
    let objects = journal.root.join("INPUT_OBJECTS");
    if !objects.exists() {
        fs::create_dir(&objects)?;
        File::open(&journal.root)?.sync_all()?;
    }
    strict_path(&objects, false)?;
    let unique = scientific_input_sources(request, policy)?;
    let mut additional = policy.minimum_free_bytes;
    let mut byte_objects = std::collections::BTreeSet::new();
    for source in unique.values() {
        if byte_objects.insert(&source.sha256) && !objects.join(&source.sha256).exists() {
            strict_path(&source.locator, true)?;
            additional = additional
                .checked_add(fs::metadata(&source.locator)?.len())
                .ok_or_else(|| reject("input retention byte budget overflow"))?;
        }
    }
    if free_bytes(&journal.root)? < additional {
        return Err(reject("insufficient filesystem capacity for retained inputs and requested native output reserve"));
    }
    let mut retained = Vec::new();
    for source in unique.into_values() {
        let object = copy_retained_input(&source, &objects)?;
        retained.push(RetainedInput { source, object });
    }
    publish(
        &journal.root.join("scientific-inputs.manifest.json"),
        &serde_json::to_vec_pretty(&retained)?,
    )?;
    Ok(())
}

fn verify_retained_inputs(
    journal: &ExperimentJournal,
    request: &HyperRuntimeRequestV1,
) -> PcResult<usize> {
    let path = journal.root.join("scientific-inputs.manifest.json");
    strict_path(&path, true)?;
    let retained: Vec<RetainedInput> = serde_json::from_slice(&fs::read(path)?)?;
    if retained.is_empty() {
        return Err(reject("retained input population empty"));
    }
    let policy = request
        .request
        .experiment
        .as_ref()
        .ok_or_else(|| reject("retained experiment policy missing"))?;
    let expected = scientific_input_sources(request, policy)?;
    let mut sources = std::collections::BTreeSet::new();
    for row in &retained {
        if expected.get(&row.source.locator) != Some(&row.source) {
            return Err(reject(
                "retained input differs from canonical request dependency",
            ));
        }
        if !sources.insert(&row.source.locator)
            || row.object.sha256 != row.source.sha256
            || row.object.path != PathBuf::from(&row.source.sha256)
        {
            return Err(reject("retained input identity or membership differs"));
        }
        verify_files(
            &journal.root.join("INPUT_OBJECTS"),
            std::slice::from_ref(&row.object),
        )?;
    }
    if sources.len() != expected.len() {
        return Err(reject("retained input dependency population incomplete"));
    }
    Ok(retained.len())
}

fn bind_policy_topology(
    request: &HyperRuntimeRequestV1,
    policy: &ExperimentPolicyV1,
    retained: Option<&RetainedInputResolver>,
) -> PcResult<()> {
    let scientific: serde_json::Value = serde_json::from_slice(&fs::read(source_path(
        &policy.qualification_policy,
        retained,
    )?)?)?;
    let raw = scientific
        .get("topology")
        .and_then(|v| v.as_str())
        .ok_or_else(|| reject("qualification topology missing"))?;
    let path = Path::new(raw);
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        policy
            .qualification_policy
            .locator
            .parent()
            .unwrap()
            .join(path)
    };
    let identity = policy
        .qualification_inputs
        .iter()
        .find(|v| v.locator == path)
        .ok_or_else(|| reject("qualification topology is not an admitted source input"))?;
    let (_, sha) = digest_file(&source_path(identity, retained)?, false)?;
    if sha != request.request.topology.sha256 {
        return Err(reject(
            "qualification topology differs from native topology before MD admission",
        ));
    }
    Ok(())
}

/// Verify a retained or relocated result without changing the scientific
/// request's original provenance paths or consulting the current ACTIVE.
pub fn verify_output(output: &Path) -> PcResult<serde_json::Value> {
    strict_path(output, false)?;
    let request_file = output.join("OPERATOR_CONTROL/PRISMOS_HYPER_RUNTIME_REQUEST.json");
    strict_path(&request_file, true)?;
    let request: HyperRuntimeRequestV1 = serde_json::from_slice(&fs::read(request_file)?)?;
    if crate::hyper_runtime_request::canonical_request_identity_sha256(&request.request)?
        != request.request_id_sha256
    {
        return Err(reject("retained request identity mismatch"));
    }
    let journal_root = output.join("OPERATOR_CONTROL/DURABLE_EXPERIMENT");
    strict_path(&journal_root, false)?;
    let journal = ExperimentJournal::read(&journal_root, &request.request_id_sha256)?;
    let retained_input_count = verify_retained_inputs(&journal, &request)?;
    let native = journal.verify_committed(EventKind::NativeCommitted, output)?;
    let qualification = journal.verify_committed(
        EventKind::QualificationCommitted,
        &output.join("STRUCTURAL_QUALIFICATION"),
    )?;
    Ok(serde_json::json!({
        "schema_kind": "prismos.durable_experiment.verification",
        "schema_version": 1,
        "status": if native && qualification { "IMPLEMENTATION_VERIFIED" } else { "IMPLEMENTATION_PARTIAL" },
        "scope": "complete recorded artifact population and committed journal lineage",
        "request_id_sha256": request.request_id_sha256,
        "source_output": request.request.output,
        "verified_output": output,
        "native_pipeline_committed": native,
        "qualification_committed": qualification,
        "scientific_acceptance_recomputed": false,
        "retained_input_count": retained_input_count,
        "event_count": journal.events.len(),
        "last_event_sha256": journal.last_sha256,
    }))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NativeExecutionContext {
    pub request_id_sha256: String,
    pub package_root: PathBuf,
    pub checkpoint_root: PathBuf,
    pub resume: bool,
    pub checkpoint_every_chunks: u64,
    pub checkpoint_max_bytes: u64,
}

fn native_context(
    journal: &ExperimentJournal,
    policy: &ExperimentPolicyV1,
) -> PcResult<NativeExecutionContext> {
    let installation: InstallationManifest =
        serde_json::from_slice(&fs::read(&policy.analysis_installation_manifest.locator)?)?;
    let resume = match journal.events.last() {
        Some(e) if e.kind == EventKind::Admitted => false,
        Some(e)
            if e.stage == "native_pipeline"
                && matches!(
                    e.kind,
                    EventKind::NativeStarted | EventKind::NativeResumed | EventKind::Failed
                ) =>
        {
            true
        }
        _ => return Err(reject("invalid native continuation journal state")),
    };
    Ok(NativeExecutionContext {
        request_id_sha256: journal.request_id.clone(),
        package_root: installation.root,
        checkpoint_root: journal.root.join("NATIVE_CHECKPOINT"),
        resume,
        checkpoint_every_chunks: 100,
        checkpoint_max_bytes: 64 * 1024 * 1024 * 1024,
    })
}

pub fn validate_request_policy(
    request: &HyperRuntimeRequestV1,
    policy: &ExperimentPolicyV1,
) -> PcResult<()> {
    validate_installation(policy)?;
    let root = request
        .request
        .output
        .join("OPERATOR_CONTROL/DURABLE_EXPERIMENT");
    if root.join("event-00000000000000000000.json").exists() {
        let journal = ExperimentJournal::open(&root, &request.request_id_sha256)?;
        let retained = RetainedInputResolver::from_journal(&journal, request)?;
        validate_policy_sources(policy, Some(&retained))?;
        bind_policy_topology(request, policy, Some(&retained))
    } else {
        validate_policy_sources(policy, None)?;
        bind_policy_topology(request, policy, None)
    }
}

/// The callback is the existing canonical native pipeline. It must retain all
/// existing force, graph, Path-B/DSTW and corpus acceptance gates.
pub fn execute<F>(
    request: &HyperRuntimeRequestV1,
    policy: &ExperimentPolicyV1,
    native: F,
) -> PcResult<()>
where
    F: FnOnce(&NativeExecutionContext) -> PcResult<()>,
{
    validate_installation(policy)?;
    let output = &request.request.output;
    let control = output.join("OPERATOR_CONTROL");
    strict_path(&control, false)?;
    let journal_root = control.join("DURABLE_EXPERIMENT");
    let mut journal = ExperimentJournal::open(&journal_root, &request.request_id_sha256)?;
    let qualification_root = output.join("STRUCTURAL_QUALIFICATION");
    if journal.events.is_empty() {
        validate_policy_sources(policy, None)?;
        bind_policy_topology(request, policy, None)?;
        validate_transitive_policy(policy, None, &journal_root)?;
        retain_scientific_inputs(request, policy, &journal)?;
        let path = journal.root.join("scientific-inputs.manifest.json");
        let (_, sha256) = digest_file(&path, false)?;
        journal.append(
            EventKind::Admitted,
            "experiment",
            Some(ArtifactIdentityV1 {
                locator: PathBuf::from("scientific-inputs.manifest.json"),
                sha256,
            }),
            None,
        )?;
    }
    let retained = RetainedInputResolver::from_journal(&journal, request)?;
    validate_policy_sources(policy, Some(&retained))?;
    bind_policy_topology(request, policy, Some(&retained))?;
    validate_transitive_policy(policy, Some(&retained), &journal_root)?;
    if !journal.verify_committed(EventKind::NativeCommitted, output)? {
        let context = native_context(&journal, policy)?;
        let available = free_bytes(output)?;
        if available < policy.minimum_free_bytes {
            return Err(reject(format!(
                "insufficient output capacity: available={available}, required={}",
                policy.minimum_free_bytes
            )));
        }
        journal.append(
            if context.resume {
                EventKind::NativeResumed
            } else {
                EventKind::NativeStarted
            },
            "native_pipeline",
            None,
            Some(serde_json::to_string(&context)?),
        )?;
        if let Err(e) = native(&context) {
            journal.append(
                EventKind::Failed,
                "native_pipeline",
                None,
                Some(e.to_string()),
            )?;
            return Err(e);
        }
        let manifest = journal.manifest(
            "native",
            output,
            &[journal_root.clone(), qualification_root.clone()],
        )?;
        journal.append(
            EventKind::NativeCommitted,
            "native_pipeline",
            Some(manifest),
            None,
        )?;
    }
    if journal.verify_committed(EventKind::QualificationCommitted, &qualification_root)? {
        return Ok(());
    }
    validate_installation(policy)?;
    if !qualification_root.exists() {
        fs::create_dir(&qualification_root)?;
        File::open(output)?.sync_all()?;
    }
    strict_path(&qualification_root, false)?;
    let attempt = qualification_root.join(format!("attempt-{:020}", journal.events.len()));
    journal.append(
        EventKind::QualificationStarted,
        "qualification",
        None,
        Some(attempt.file_name().unwrap().to_string_lossy().to_string()),
    )?;
    let result = (|| -> PcResult<()> {
        let log_path = journal_root.join(format!("qualification-{:020}.log", journal.events.len()));
        let log = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(log_path)?;
        let mut command = qualification_command(policy)?;
        retained.bind_command(&mut command, &journal_root, "qualification")?;
        let cache = journal_root.join(format!("qualification-cache-{:020}", journal.events.len()));
        fs::create_dir(&cache)?;
        File::open(&journal_root)?.sync_all()?;
        command.env("MPLCONFIGDIR", &cache);
        command
            .arg("--native-run")
            .arg(output)
            .arg("--policy")
            .arg(&policy.qualification_policy.locator)
            .arg("--policy-sha256")
            .arg(&policy.qualification_policy.sha256)
            .arg("--output")
            .arg(&attempt)
            .current_dir(&journal_root)
            .stdout(log.try_clone()?)
            .stderr(log);
        let status = command.status()?;
        if !status.success() {
            return Err(PcError::Internal(format!(
                "structural qualification exited {status}"
            )));
        }
        let receipt = attempt.join("OPERATOR_CONTROL/QUALIFICATION_EXECUTION.json");
        strict_path(&receipt, true)?;
        let value: serde_json::Value = serde_json::from_slice(&fs::read(receipt)?)?;
        retained.verify_receipt(&value)?;
        if value.get("status").and_then(|v| v.as_str()) != Some("IMPLEMENTATION_VERIFIED") {
            return Err(reject(
                "qualification exit did not produce a verified execution receipt",
            ));
        }
        if value.get("schema").and_then(|v| v.as_str())
            != Some("prism.structural_qualification.execution.v1")
            || value.get("topology_sha256").and_then(|v| v.as_str())
                != Some(request.request.topology.sha256.as_str())
            || value
                .pointer("/discovery/policy_sha256")
                .and_then(|v| v.as_str())
                != Some(policy.qualification_policy.sha256.as_str())
        {
            return Err(reject(
                "qualification receipt does not bind canonical topology and frozen analysis policy",
            ));
        }
        let md_manifest = output.join("md_evidence_manifest.json");
        let (_, md_sha) = digest_file(&md_manifest, false)?;
        let md: serde_json::Value = serde_json::from_slice(&fs::read(&md_manifest)?)?;
        if value
            .pointer("/discovery/manifest_sha256")
            .and_then(|v| v.as_str())
            != Some(md_sha.as_str())
            || value.pointer("/discovery/native_run_id") != md.get("run_id")
        {
            return Err(reject(
                "qualification receipt native run identity differs from committed engine output",
            ));
        }
        let ledger: Vec<serde_json::Value> = serde_json::from_slice(&fs::read(
            attempt.join("OPERATOR_CONTROL/INPUT_LEDGER.json"),
        )?)?;
        for binding in &policy.qualification_inputs {
            if !ledger.iter().any(|v| {
                v.get("path").and_then(|v| v.as_str()) == binding.locator.to_str()
                    && v.get("sha256").and_then(|v| v.as_str()) == Some(binding.sha256.as_str())
            }) {
                return Err(reject(
                    "qualification input ledger omits or changes a bound scientific dependency",
                ));
            }
        }
        validate_installation(policy)?;
        validate_policy_sources(policy, Some(&retained))?;
        // An interrupted attempt remains available for inspection. Only this
        // completed child owns the committed population; an older orphan's
        // writes cannot enter the new scientific result manifest.
        let manifest = journal.manifest("qualification", &attempt, &[])?;
        journal.append(
            EventKind::QualificationCommitted,
            "qualification",
            Some(manifest),
            Some(attempt.file_name().unwrap().to_string_lossy().to_string()),
        )?;
        Ok(())
    })();
    if let Err(e) = result {
        journal.append(
            EventKind::Failed,
            "qualification",
            None,
            Some(e.to_string()),
        )?;
        return Err(e);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn journal_rejects_concurrent_writer_identity_change_and_corruption() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("journal");
        let identity = "a".repeat(64);
        let mut journal = ExperimentJournal::open(&root, &identity).unwrap();
        journal
            .append(EventKind::Admitted, "experiment", None, None)
            .unwrap();
        assert!(ExperimentJournal::open(&root, &identity).is_err());
        drop(journal);
        assert!(ExperimentJournal::open(&root, &"b".repeat(64)).is_err());
        let mut journal = ExperimentJournal::open(&root, &identity).unwrap();
        journal
            .append(EventKind::NativeStarted, "native_pipeline", None, None)
            .unwrap();
        assert!(journal
            .append(
                EventKind::QualificationCommitted,
                "qualification",
                None,
                None
            )
            .is_err());
        drop(journal);
        fs::write(root.join(".unpublished.pending"), b"torn").unwrap();
        assert_eq!(
            ExperimentJournal::open(&root, &identity)
                .unwrap()
                .events
                .len(),
            2
        );
        fs::write(root.join("event-00000000000000000001.json"), b"torn").unwrap();
        assert!(ExperimentJournal::open(&root, &identity).is_err());
    }
    #[test]
    fn committed_population_rejects_changed_missing_or_escaped_files() {
        let temp = tempfile::tempdir().unwrap();
        let data = temp.path().join("data");
        fs::create_dir(&data).unwrap();
        fs::write(data.join("measurement"), b"source bytes").unwrap();
        let mut journal =
            ExperimentJournal::open(&temp.path().join("journal"), &"c".repeat(64)).unwrap();
        journal
            .append(EventKind::Admitted, "experiment", None, None)
            .unwrap();
        journal
            .append(EventKind::NativeStarted, "native_pipeline", None, None)
            .unwrap();
        let manifest = journal.manifest("native", &data, &[]).unwrap();
        journal
            .append(
                EventKind::NativeCommitted,
                "native_pipeline",
                Some(manifest),
                None,
            )
            .unwrap();
        assert!(journal
            .verify_committed(EventKind::NativeCommitted, &data)
            .unwrap());
        fs::write(data.join("measurement"), b"different bytes").unwrap();
        assert!(journal
            .verify_committed(EventKind::NativeCommitted, &data)
            .is_err());
        assert!(relative(Path::new("../escape")).is_err());
        assert!(relative(Path::new("/absolute")).is_err());
    }
    #[test]
    fn qualification_commits_only_the_completed_attempt_and_survives_relocation() {
        let temp = tempfile::tempdir().unwrap();
        let native = temp.path().join("native");
        fs::create_dir(&native).unwrap();
        fs::write(native.join("observations"), b"retained bytes").unwrap();
        let root = temp.path().join("qualification");
        let failed = root.join("attempt-0");
        let completed = root.join("attempt-1");
        fs::create_dir_all(&failed).unwrap();
        fs::create_dir(&completed).unwrap();
        fs::write(failed.join("unfinished"), b"partial bytes").unwrap();
        fs::write(completed.join("observations"), b"complete bytes").unwrap();
        let mut journal =
            ExperimentJournal::open(&temp.path().join("journal"), &"d".repeat(64)).unwrap();
        journal
            .append(EventKind::Admitted, "experiment", None, None)
            .unwrap();
        journal
            .append(EventKind::NativeStarted, "native_pipeline", None, None)
            .unwrap();
        let manifest = journal.manifest("native", &native, &[]).unwrap();
        journal
            .append(
                EventKind::NativeCommitted,
                "native_pipeline",
                Some(manifest),
                None,
            )
            .unwrap();
        journal
            .append(
                EventKind::QualificationStarted,
                "qualification",
                None,
                Some("attempt-0".into()),
            )
            .unwrap();
        journal
            .append(
                EventKind::QualificationStarted,
                "qualification",
                None,
                Some("attempt-1".into()),
            )
            .unwrap();
        let manifest = journal.manifest("qualification", &completed, &[]).unwrap();
        journal
            .append(
                EventKind::QualificationCommitted,
                "qualification",
                Some(manifest),
                Some("attempt-1".into()),
            )
            .unwrap();
        fs::write(failed.join("unfinished"), b"late orphan bytes").unwrap();
        assert!(journal
            .verify_committed(EventKind::QualificationCommitted, &root)
            .unwrap());
        let relocated = temp.path().join("relocated");
        fs::rename(&root, &relocated).unwrap();
        assert!(journal
            .verify_committed(EventKind::QualificationCommitted, &relocated)
            .unwrap());
        fs::remove_file(relocated.join("attempt-1/observations")).unwrap();
        assert!(journal
            .verify_committed(EventKind::QualificationCommitted, &relocated)
            .is_err());
    }

    #[test]
    fn runtime_paths_refuse_escape_and_symlink() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir(temp.path().join("python")).unwrap();
        for name in ["loader", "bootstrap.py", "contract.json"] {
            fs::write(temp.path().join(name), b"test IO bytes").unwrap();
        }
        let mut manifest = InstallationManifest {
            schema_version: 1,
            root: temp.path().into(),
            files: vec![],
            loader: "loader".into(),
            python_bootstrap: "bootstrap.py".into(),
            python_contract: "contract.json".into(),
            python_home: "python".into(),
            python_paths: vec!["python".into()],
            library_paths: vec!["python".into()],
        };
        validate_runtime_layout(&manifest).unwrap();
        manifest.python_paths = vec!["../other".into()];
        assert!(validate_runtime_layout(&manifest).is_err());
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(temp.path().join("python"), temp.path().join("link"))
                .unwrap();
            manifest.python_paths = vec!["link".into()];
            assert!(validate_runtime_layout(&manifest).is_err());
        }
    }
    #[test]
    fn retained_input_survives_original_removal_and_refuses_corruption() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("prepared-input");
        let objects = temp.path().join("objects");
        fs::create_dir(&objects).unwrap();
        let bytes: Vec<u8> = (0..3_000_001u64).map(|i| (i % 251) as u8).collect();
        fs::write(&input, &bytes).unwrap();
        let identity = ArtifactIdentityV1 {
            locator: input.clone(),
            sha256: format!("{:x}", Sha256::digest(&bytes)),
        };
        let record = copy_retained_input(&identity, &objects).unwrap();
        assert_eq!(record, copy_retained_input(&identity, &objects).unwrap());
        fs::remove_file(&input).unwrap();
        let relocated = temp.path().join("relocated");
        fs::rename(&objects, &relocated).unwrap();
        verify_files(&relocated, std::slice::from_ref(&record)).unwrap();
        assert_eq!(fs::read(relocated.join(&record.path)).unwrap(), bytes);
        fs::write(relocated.join(&record.path), b"corrupt").unwrap();
        assert!(verify_files(&relocated, &[record]).is_err());
    }
    #[test]
    fn unlisted_completed_observation_is_refused() {
        let temp = tempfile::tempdir().unwrap();
        let data = temp.path().join("data");
        fs::create_dir(&data).unwrap();
        fs::write(data.join("observed"), b"actual retained IO bytes").unwrap();
        let mut j = ExperimentJournal::open(&temp.path().join("journal"), &"e".repeat(64)).unwrap();
        j.append(EventKind::Admitted, "experiment", None, None)
            .unwrap();
        j.append(EventKind::NativeStarted, "native_pipeline", None, None)
            .unwrap();
        let m = j.manifest("native", &data, &[]).unwrap();
        j.append(EventKind::NativeCommitted, "native_pipeline", Some(m), None)
            .unwrap();
        assert!(j
            .verify_committed(EventKind::NativeCommitted, &data)
            .unwrap());
        fs::write(data.join("later-observation"), b"uncommitted IO bytes").unwrap();
        assert!(j
            .verify_committed(EventKind::NativeCommitted, &data)
            .is_err());
    }

    #[test]
    fn retained_input_ignores_preserved_partial_copy_and_retries_exactly() {
        let temp = tempfile::tempdir().unwrap();
        let objects = temp.path().join("objects");
        fs::create_dir(&objects).unwrap();
        let input = temp.path().join("input");
        fs::write(&input, b"retained bytes").unwrap();
        let identity = ArtifactIdentityV1 {
            locator: input,
            sha256: format!("{:x}", Sha256::digest(b"retained bytes")),
        };
        let stale = objects.join(format!(
            "{}.{}.pending",
            identity.sha256,
            std::process::id()
        ));
        fs::write(&stale, b"interrupted prefix").unwrap();
        let r = copy_retained_input(&identity, &objects).unwrap();
        assert_eq!(fs::read(&stale).unwrap(), b"interrupted prefix");
        assert_eq!(r, copy_retained_input(&identity, &objects).unwrap());
    }

    #[test]
    #[ignore = "invoked only as a real crash-test child"]
    fn journal_crash_child() {
        let Ok(path) = std::env::var("PRISM_DURABLE_TEST_JOURNAL") else {
            return;
        };
        let mut j = ExperimentJournal::open(Path::new(&path), &"f".repeat(64)).unwrap();
        j.append(EventKind::NativeStarted, "native_pipeline", None, None)
            .unwrap();
        // Actual process death bypasses Rust destructors and drops the OS lease.
        std::process::exit(91);
    }

    #[test]
    fn killed_process_reopens_same_request_and_records_resumption() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("journal");
        let mut j = ExperimentJournal::open(&root, &"f".repeat(64)).unwrap();
        j.append(EventKind::Admitted, "experiment", None, None)
            .unwrap();
        drop(j);
        let status = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "durable_experiment::tests::journal_crash_child",
                "--ignored",
                "--nocapture",
            ])
            .env("PRISM_DURABLE_TEST_JOURNAL", &root)
            .status()
            .unwrap();
        assert_eq!(status.code(), Some(91));
        let mut j = ExperimentJournal::open(&root, &"f".repeat(64)).unwrap();
        assert_eq!(j.events.len(), 2);
        assert!(j
            .append(EventKind::NativeStarted, "native_pipeline", None, None)
            .is_err());
        j.append(EventKind::NativeResumed, "native_pipeline", None, None)
            .unwrap();
        j.append(
            EventKind::Failed,
            "native_pipeline",
            None,
            Some("observed refusal".into()),
        )
        .unwrap();
        j.append(EventKind::NativeResumed, "native_pipeline", None, None)
            .unwrap();
        assert_eq!(j.events.len(), 5);
        assert!(j
            .append(EventKind::NativeResumed, "qualification", None, None)
            .is_err());
    }
    #[test]
    fn retained_resolver_preserves_logical_identity_and_refuses_wrong_source() {
        let temp = tempfile::tempdir().unwrap();
        let objects = temp.path().join("INPUT_OBJECTS");
        fs::create_dir(&objects).unwrap();
        let source = temp.path().join("original-data");
        fs::write(&source, b"actual archived source bytes").unwrap();
        let id = ArtifactIdentityV1 {
            locator: source.clone(),
            sha256: format!("{:x}", Sha256::digest(b"actual archived source bytes")),
        };
        let object = copy_retained_input(&id, &objects).unwrap();
        let row = RetainedInput {
            source: id.clone(),
            object,
        };
        let manifest = temp.path().join("scientific-inputs.manifest.json");
        fs::write(&manifest, serde_json::to_vec(&vec![row.clone()]).unwrap()).unwrap();
        let (_, sha) = digest_file(&manifest, false).unwrap();
        let resolver = RetainedInputResolver {
            manifest: ArtifactIdentityV1 {
                locator: manifest,
                sha256: sha.clone(),
            },
            objects,
            sources: std::collections::BTreeMap::from([(source.clone(), row)]),
        };
        fs::remove_file(&source).unwrap();
        assert_eq!(
            fs::read(source_path(&id, Some(&resolver)).unwrap()).unwrap(),
            b"actual archived source bytes"
        );
        assert!(source_path(&id, None).is_err());
        let mut other = id.clone();
        other.sha256 = "0".repeat(64);
        assert!(source_path(&other, Some(&resolver)).is_err());
        let receipt = serde_json::json!({"retained_inputs":{"manifest_sha256":sha,"original_locators_preserved":true,"verified_binding_count":1}});
        resolver.verify_receipt(&receipt).unwrap();
        let mut wrong = receipt;
        wrong["retained_inputs"]["verified_binding_count"] = serde_json::json!(2);
        assert!(resolver.verify_receipt(&wrong).is_err());
    }
    #[test]
    fn verification_uses_read_only_shared_lease_without_creating_files() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("journal");
        let id = "9".repeat(64);
        assert!(ExperimentJournal::read(&root, &id).is_err());
        assert!(!root.exists());
        let mut writer = ExperimentJournal::open(&root, &id).unwrap();
        writer
            .append(EventKind::Admitted, "experiment", None, None)
            .unwrap();
        assert!(ExperimentJournal::read(&root, &id).is_err());
        drop(writer);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(root.join("writer.lock"), fs::Permissions::from_mode(0o444))
                .unwrap();
            fs::set_permissions(&root, fs::Permissions::from_mode(0o555)).unwrap();
        }
        let before: std::collections::BTreeSet<_> = fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        let mut first = ExperimentJournal::read(&root, &id).unwrap();
        let second = ExperimentJournal::read(&root, &id).unwrap();
        assert_eq!(first.events.len(), 1);
        assert_eq!(second.events.len(), 1);
        assert!(first
            .append(EventKind::NativeStarted, "native_pipeline", None, None)
            .is_err());
        assert!(ExperimentJournal::open(&root, &id).is_err());
        let after: std::collections::BTreeSet<_> = fs::read_dir(&root)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(before, after);
        drop(first);
        drop(second);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&root, fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
}
