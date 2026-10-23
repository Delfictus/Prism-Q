//! Typed ensemble manifest schema, validator, and finalizer.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Current ensemble manifest schema version.
pub const ENSEMBLE_MANIFEST_SCHEMA_VERSION: &str = "1.0.0";

/// Current ensemble manifest validator version.
pub const ENSEMBLE_VALIDATOR_VERSION: &str = "1.0.0";

/// Campaign-level metadata shared by all replicas in a manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CampaignMetadata {
    /// Stable campaign identifier.
    pub campaign_id: String,
    /// ISO-8601 start timestamp.
    pub started_at: String,
    /// ISO-8601 completion timestamp.
    pub completed_at: String,
    /// Pod, host, or node identifier.
    pub pod_id: String,
    /// Operator identifier.
    pub operator: String,
}

/// Engine provenance for a replica or finalized ensemble.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineMetadata {
    /// SHA-256 of the exact engine binary.
    pub binary_sha256: String,
    /// Best-effort binary build/modification timestamp.
    pub binary_built_at: String,
    /// Logical binary version label.
    pub binary_version: String,
    /// Cargo/features compiled into the binary.
    pub build_features: Vec<String>,
    /// CUDA architecture label, when known.
    pub cuda_arch: String,
    /// Git commit or explicit unknown marker.
    pub git_commit: String,
    /// SHA-256 of docs/DETERMINISM_BUDGET.md.
    pub determinism_budget_doc_sha256: String,
}

/// Target identity and input provenance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TargetMetadata {
    /// Target identifier derived from the topology stem unless overridden.
    pub pdb_id: String,
    /// Target chain identifier.
    pub target_chain: String,
    /// Source PDB path relative to the run directory when available.
    pub input_pdb_path_relative: String,
    /// SHA-256 of the source PDB when present.
    pub input_pdb_sha256: String,
    /// Topology path relative to the run directory when possible.
    pub topology_json_path_relative: String,
    /// SHA-256 of the topology JSON.
    pub topology_json_sha256: String,
    /// Residue count from topology.
    pub n_residues: usize,
    /// Atom count from topology.
    pub n_atoms: usize,
    /// Whether a ground-truth sidecar was present.
    pub ground_truth_present: bool,
    /// Optional UniProt identifier from ground truth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ground_truth_uniprot_id: Option<String>,
    /// Optional holo PDB identifier from ground truth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ground_truth_pdb_holo: Option<String>,
    /// Optional ligand centroid from ground truth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ground_truth_ligand_centroid: Option<[f64; 3]>,
    /// Optional ligand residue name from ground truth.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ground_truth_ligand_resname: Option<String>,
    /// Whether ground truth is valid for DCC.
    pub ground_truth_valid_for_dcc: bool,
}

/// Ensemble-level run configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EnsembleConfig {
    /// Number of replicas planned.
    pub n_replicas_planned: usize,
    /// Number of replicas completed.
    pub n_replicas_completed: usize,
    /// Number of replicas failed or excluded.
    pub n_replicas_failed: usize,
    /// True iff all included replicas passed integrity audit.
    pub all_replicas_passed_audit: bool,
    /// Seed strategy label.
    pub replica_seed_strategy: String,
    /// Base seed.
    pub base_seed: u64,
    /// Per-replica seed offsets from base_seed.
    pub seed_offsets: Vec<i64>,
    /// Concurrent replicas per pod.
    pub concurrent_replicas_per_pod: usize,
    /// Canonical engine flags used for every replica.
    pub engine_flags_per_replica: Vec<String>,
}

/// Replica status values accepted by the manifest validator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReplicaStatus {
    /// Replica completed and all integrity gates passed.
    Ok,
    /// Some streams failed or evidence is incomplete.
    PartialStreams,
    /// Replica failed.
    Failed,
    /// Replica was aborted.
    Aborted,
}

impl ReplicaStatus {
    fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }
}

/// Per-replica memory plan summary.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct MemoryPlan {
    /// Voxel grid dimension used by the run.
    pub voxel_grid_dim: Option<u32>,
    /// Voxel spacing in angstrom.
    pub voxel_spacing_angstrom: Option<f32>,
    /// Padded coverage in angstrom.
    pub padded_coverage_angstrom: Option<f32>,
    /// VRAM used in MiB.
    pub vram_used_mib: Option<u64>,
    /// VRAM free in MiB.
    pub vram_free_mib: Option<u64>,
    /// Whether adaptive grid sizing was used.
    pub adaptive_grid_used: bool,
}

/// V2 trajectory frame audit summary.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct FrameAudit {
    /// Producer frame count per stream.
    pub producer_count_per_stream: Vec<u64>,
    /// Writer frame count per stream.
    pub writer_count_per_stream: Vec<u64>,
    /// Disk frame-count header per stream.
    pub disk_count_per_stream: Vec<u64>,
    /// Producer rolling hash per stream.
    pub producer_hash_per_stream: Vec<String>,
    /// Writer rolling hash per stream.
    pub writer_hash_per_stream: Vec<String>,
    /// Disk payload hash per stream.
    pub disk_hash_per_stream: Vec<String>,
    /// True iff every count and hash check passed.
    pub all_hashes_match: bool,
    /// Primary audit sidecar path for compatibility with the prose schema.
    pub audit_sidecar_relative_path: String,
    /// All per-stream audit sidecar paths.
    #[serde(default)]
    pub audit_sidecar_relative_paths: Vec<String>,
}

impl FrameAudit {
    /// Validate count and hash integrity for a frame audit block.
    pub fn validate(&self) -> Result<()> {
        let n = self.producer_count_per_stream.len();
        if n == 0 {
            bail!("frame_audit has no producer_count_per_stream entries");
        }
        for (name, len) in [
            (
                "writer_count_per_stream",
                self.writer_count_per_stream.len(),
            ),
            ("disk_count_per_stream", self.disk_count_per_stream.len()),
            (
                "producer_hash_per_stream",
                self.producer_hash_per_stream.len(),
            ),
            ("writer_hash_per_stream", self.writer_hash_per_stream.len()),
            ("disk_hash_per_stream", self.disk_hash_per_stream.len()),
        ] {
            if len != n {
                bail!(
                    "frame_audit length mismatch: producer_count_per_stream has {}, {} has {}",
                    n,
                    name,
                    len
                );
            }
        }
        for i in 0..n {
            let p = self.producer_count_per_stream[i];
            let w = self.writer_count_per_stream[i];
            let d = self.disk_count_per_stream[i];
            if p != w || p != d {
                bail!(
                    "frame_audit count mismatch stream {}: producer={}, writer={}, disk={}",
                    i,
                    p,
                    w,
                    d
                );
            }
            let ph = &self.producer_hash_per_stream[i];
            let wh = &self.writer_hash_per_stream[i];
            let dh = &self.disk_hash_per_stream[i];
            if ph != wh || ph != dh {
                bail!(
                    "frame_audit hash mismatch stream {}: producer={}, writer={}, disk={}",
                    i,
                    ph,
                    wh,
                    dh
                );
            }
        }
        if !self.all_hashes_match {
            bail!("frame_audit.all_hashes_match=false");
        }
        Ok(())
    }
}

/// Output artifact paths and hashes for one replica.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct ReplicaOutputs {
    /// Stage-B/v5 residue feature Parquet path for teacher aggregation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_parquet_relative: Option<String>,
    /// Stage-B/v5 residue feature Parquet SHA-256.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_parquet_sha256: Option<String>,
    /// Stage-B/v5 residue feature Parquet byte size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_parquet_bytes: Option<u64>,
    /// Spike Arrow path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_arrow_relative: Option<String>,
    /// Spike Arrow SHA-256.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_arrow_sha256: Option<String>,
    /// Spike Arrow byte size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_arrow_bytes: Option<u64>,
    /// Binding sites JSON path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binding_sites_json_relative: Option<String>,
    /// Binding sites JSON SHA-256.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binding_sites_json_sha256: Option<String>,
    /// KCC visualization JSON path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kcc_visualization_json_relative: Option<String>,
    /// KCC visualization JSON SHA-256.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kcc_visualization_json_sha256: Option<String>,
    /// KCC validation JSON path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kcc_validation_json_relative: Option<String>,
    /// PRISM-Therm JSON path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_prism_therm_json_relative: Option<String>,
    /// PRISM-Therm JSON SHA-256.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_prism_therm_json_sha256: Option<String>,
    /// ASC consensus JSON path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_asc_consensus_json_relative: Option<String>,
    /// GCPID synergy JSON path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_gcpid_synergy_json_relative: Option<String>,
    /// Phasors binary path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_phasors_bin_relative: Option<String>,
    /// Phasors binary SHA-256.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_phasors_bin_sha256: Option<String>,
    /// ACL contrast binary path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_acl_contrast_bin_relative: Option<String>,
    /// Druggability PDB path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub topology_druggability_pdb_relative: Option<String>,
    /// Ensemble trajectory JSON path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble_trajectory_json_relative: Option<String>,
}

/// Engine telemetry captured for a replica.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct EngineTelemetry {
    /// Total spikes.
    pub total_spikes: u64,
    /// Total pockets detected.
    pub total_pockets_detected: u64,
    /// Cryptic pocket count.
    pub n_cryptic_pockets: u64,
    /// TIDE residues mapped.
    pub tide_residues_mapped: u64,
    /// KV top1/top2 separation.
    pub kv_top1_vs_top2_separation: Option<f64>,
    /// Consensus residue count.
    pub n_consensus_residues: u64,
}

/// Physical time metadata for one replica.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhysicalTime {
    /// Base timestep in ps.
    pub dt_ps: f32,
    /// Trajectory save interval in steps.
    pub save_interval_steps: u32,
    /// Total simulated steps.
    pub n_steps_total: u64,
    /// Physical duration in ps.
    pub physical_duration_ps: f32,
    /// Whether HMR was used.
    pub hmr_used: bool,
    /// Effective HMR timestep in ps.
    pub hmr_dt_ps_effective: Option<f32>,
    /// Whether adaptive dt was used.
    pub adaptive_dt_used: bool,
    /// Average adaptive dt in ps when known.
    pub adaptive_dt_average_ps: Option<f32>,
}

/// One replica entry inside an ensemble manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReplicaManifest {
    /// Replica id.
    pub replica_id: u32,
    /// Seed used by this run.
    pub run_seed: u64,
    /// Completion status.
    pub status: ReplicaStatus,
    /// ISO-8601 start timestamp.
    pub started_at: String,
    /// ISO-8601 completion timestamp.
    pub completed_at: String,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Pod identifier.
    pub pod_id: String,
    /// Planned stream count.
    pub n_streams_planned: usize,
    /// Completed stream count.
    pub n_streams_completed: usize,
    /// Failed stream count.
    pub n_streams_failed: usize,
    /// Stream failure descriptions.
    pub stream_failures: Vec<String>,
    /// Memory plan.
    pub memory_plan: MemoryPlan,
    /// Frame audit.
    pub frame_audit: FrameAudit,
    /// Output artifact map.
    pub outputs: ReplicaOutputs,
    /// Engine telemetry.
    pub engine_telemetry: EngineTelemetry,
    /// Physical time.
    pub physical_time: PhysicalTime,
}

impl ReplicaManifest {
    /// Validate this replica entry.
    pub fn validate(&self) -> Result<()> {
        if self.status.is_ok() {
            if self.n_streams_failed != 0 {
                bail!(
                    "replica {} status=ok but n_streams_failed={}",
                    self.replica_id,
                    self.n_streams_failed
                );
            }
            if self.n_streams_planned != self.n_streams_completed {
                bail!(
                    "replica {} status=ok but streams completed/planned differ: {}/{}",
                    self.replica_id,
                    self.n_streams_completed,
                    self.n_streams_planned
                );
            }
            self.frame_audit
                .validate()
                .with_context(|| format!("replica {} frame_audit invalid", self.replica_id))?;
        }
        Ok(())
    }
}

/// One validation step in the manifest audit log.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationStep {
    /// Step identifier.
    pub step: String,
    /// Pass/fail status.
    pub passed: bool,
    /// Optional human-readable details.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

/// Manifest validation audit log.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuditLog {
    /// Validation steps.
    pub validation_steps: Vec<ValidationStep>,
    /// Validator version.
    pub validator_version: String,
    /// ISO-8601 validation timestamp.
    pub validated_at: String,
}

impl AuditLog {
    /// Validate every audit step passed.
    pub fn validate(&self) -> Result<()> {
        for step in &self.validation_steps {
            if !step.passed {
                bail!(
                    "audit_log validation step failed: {} ({})",
                    step.step,
                    step.details.as_deref().unwrap_or("no details")
                );
            }
        }
        Ok(())
    }
}

/// Final ensemble manifest.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnsembleManifest {
    /// Schema version.
    pub schema_version: String,
    /// Campaign metadata.
    pub campaign: CampaignMetadata,
    /// Engine metadata.
    pub engine: EngineMetadata,
    /// Target metadata.
    pub target: TargetMetadata,
    /// Ensemble config.
    pub ensemble_config: EnsembleConfig,
    /// Replica entries.
    pub replicas: Vec<ReplicaManifest>,
    /// Consensus block, populated by the assembler.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ensemble_consensus: Option<serde_json::Value>,
    /// Audit log.
    pub audit_log: AuditLog,
}

impl EnsembleManifest {
    /// Read and validate a manifest from disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)
            .with_context(|| format!("read ensemble manifest {}", path.display()))?;
        let manifest: Self = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse ensemble manifest {}", path.display()))?;
        manifest
            .validate()
            .with_context(|| format!("validate ensemble manifest {}", path.display()))?;
        Ok(manifest)
    }

    /// Write a manifest to disk.
    pub fn write_pretty(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        let bytes = serde_json::to_vec_pretty(self)?;
        std::fs::write(path, bytes)
            .with_context(|| format!("write ensemble manifest {}", path.display()))?;
        Ok(())
    }

    /// Validate schema, audit gates, and all replicas.
    pub fn validate(&self) -> Result<()> {
        validate_schema_version(&self.schema_version)?;
        if self.replicas.is_empty() {
            bail!("ensemble manifest has no replicas");
        }
        let ok_count = self.replicas.iter().filter(|r| r.status.is_ok()).count();
        if self.ensemble_config.n_replicas_completed != ok_count {
            bail!(
                "ensemble_config.n_replicas_completed={} but {} replicas have status=ok",
                self.ensemble_config.n_replicas_completed,
                ok_count
            );
        }
        if self.ensemble_config.all_replicas_passed_audit && ok_count != self.replicas.len() {
            bail!("all_replicas_passed_audit=true but not every replica has status=ok");
        }
        for replica in &self.replicas {
            replica.validate()?;
        }
        self.audit_log.validate()?;
        Ok(())
    }
}

/// On-disk per-run replica record written by the engine before finalization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EnsembleReplicaRecord {
    /// Schema version.
    pub schema_version: String,
    /// Optional campaign block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub campaign: Option<CampaignMetadata>,
    /// Engine metadata.
    pub engine: EngineMetadata,
    /// Target metadata.
    pub target: TargetMetadata,
    /// Replica config fragment.
    pub ensemble_config: EnsembleConfig,
    /// Replica payload.
    pub replica: ReplicaManifest,
    /// Audit log for this replica record.
    pub audit_log: AuditLog,
}

impl EnsembleReplicaRecord {
    /// Validate this per-run record.
    pub fn validate(&self) -> Result<()> {
        validate_schema_version(&self.schema_version)?;
        self.replica.validate()?;
        self.audit_log.validate()?;
        Ok(())
    }

    /// Write this record to disk.
    pub fn write_pretty(&self, path: impl AsRef<Path>) -> Result<()> {
        self.validate()?;
        let path = path.as_ref();
        std::fs::write(path, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("write ensemble replica record {}", path.display()))?;
        Ok(())
    }
}

/// Options for finalizing an ensemble manifest from replica records.
#[derive(Debug, Clone)]
pub struct EnsembleFinalizeOptions {
    /// Directory containing ensemble_replica_*.json files.
    pub target_dir: PathBuf,
    /// Campaign id for the final manifest.
    pub campaign_id: String,
    /// Base seed.
    pub base_seed: u64,
    /// Expected replica count.
    pub n_replicas_expected: usize,
}

/// Finalize an ensemble manifest from per-replica records.
pub fn finalize_manifest_from_replica_records(
    opts: &EnsembleFinalizeOptions,
) -> Result<EnsembleManifest> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(&opts.target_dir)
        .with_context(|| format!("read target dir {}", opts.target_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name.starts_with("ensemble_replica_") && name.ends_with(".json") {
            paths.push(path);
        }
    }
    paths.sort();
    if paths.len() != opts.n_replicas_expected {
        bail!(
            "expected {} ensemble_replica_*.json files in {}, found {}",
            opts.n_replicas_expected,
            opts.target_dir.display(),
            paths.len()
        );
    }

    let mut records = Vec::with_capacity(paths.len());
    for path in &paths {
        let bytes = std::fs::read(path)
            .with_context(|| format!("read replica record {}", path.display()))?;
        let record: EnsembleReplicaRecord = serde_json::from_slice(&bytes)
            .with_context(|| format!("parse replica record {}", path.display()))?;
        record
            .validate()
            .with_context(|| format!("validate replica record {}", path.display()))?;
        records.push(record);
    }
    if records.is_empty() {
        bail!("no replica records found");
    }

    let first = records[0].clone();
    for record in &records[1..] {
        if record.engine.binary_sha256 != first.engine.binary_sha256 {
            bail!(
                "replica {} engine binary hash mismatch: {} != {}",
                record.replica.replica_id,
                record.engine.binary_sha256,
                first.engine.binary_sha256
            );
        }
        if record.target.topology_json_sha256 != first.target.topology_json_sha256 {
            bail!(
                "replica {} topology hash mismatch: {} != {}",
                record.replica.replica_id,
                record.target.topology_json_sha256,
                first.target.topology_json_sha256
            );
        }
        if record.ensemble_config.engine_flags_per_replica
            != first.ensemble_config.engine_flags_per_replica
        {
            bail!(
                "replica {} engine flags differ from replica {}",
                record.replica.replica_id,
                first.replica.replica_id
            );
        }
    }

    let replicas: Vec<ReplicaManifest> = records.iter().map(|r| r.replica.clone()).collect();
    let ok_count = replicas.iter().filter(|r| r.status.is_ok()).count();
    let failed_count = replicas.len().saturating_sub(ok_count);
    if failed_count != 0 {
        bail!(
            "{} replica records are not status=ok; refusing to finalize contaminated ensemble",
            failed_count
        );
    }

    let seed_offsets = replicas
        .iter()
        .map(|r| r.run_seed as i128 - opts.base_seed as i128)
        .map(|d| {
            if d < i64::MIN as i128 || d > i64::MAX as i128 {
                bail!("seed offset out of i64 range");
            }
            Ok(d as i64)
        })
        .collect::<Result<Vec<_>>>()?;

    let now = now_utc_string();
    let campaign = first.campaign.unwrap_or(CampaignMetadata {
        campaign_id: opts.campaign_id.clone(),
        started_at: replicas
            .iter()
            .map(|r| r.started_at.clone())
            .min()
            .unwrap_or_else(|| now.clone()),
        completed_at: replicas
            .iter()
            .map(|r| r.completed_at.clone())
            .max()
            .unwrap_or_else(|| now.clone()),
        pod_id: std::env::var("HOSTNAME").unwrap_or_else(|_| "unknown".to_string()),
        operator: std::env::var("USER").unwrap_or_else(|_| "unknown".to_string()),
    });

    let frame_count_pass = replicas.iter().all(|r| r.frame_audit.validate().is_ok());
    let frame_hash_pass = replicas.iter().all(|r| r.frame_audit.all_hashes_match);
    let audit_log = AuditLog {
        validation_steps: vec![
            ValidationStep {
                step: "all_replicas_completed".to_string(),
                passed: ok_count == opts.n_replicas_expected,
                details: Some(format!("{} of {} ok", ok_count, opts.n_replicas_expected)),
            },
            ValidationStep {
                step: "frame_count_match_per_replica".to_string(),
                passed: frame_count_pass,
                details: Some("producer == writer == disk for every stream".to_string()),
            },
            ValidationStep {
                step: "frame_hash_match_per_replica".to_string(),
                passed: frame_hash_pass,
                details: Some("producer == writer == disk hash for every stream".to_string()),
            },
            ValidationStep {
                step: "binary_hash_consistent".to_string(),
                passed: true,
                details: Some(first.engine.binary_sha256.clone()),
            },
            ValidationStep {
                step: "topology_hash_consistent".to_string(),
                passed: true,
                details: Some(first.target.topology_json_sha256.clone()),
            },
        ],
        validator_version: ENSEMBLE_VALIDATOR_VERSION.to_string(),
        validated_at: now,
    };

    let manifest = EnsembleManifest {
        schema_version: ENSEMBLE_MANIFEST_SCHEMA_VERSION.to_string(),
        campaign,
        engine: first.engine,
        target: first.target,
        ensemble_config: EnsembleConfig {
            n_replicas_planned: opts.n_replicas_expected,
            n_replicas_completed: ok_count,
            n_replicas_failed: failed_count,
            all_replicas_passed_audit: failed_count == 0 && frame_count_pass && frame_hash_pass,
            replica_seed_strategy: "deterministic_offsets".to_string(),
            base_seed: opts.base_seed,
            seed_offsets,
            concurrent_replicas_per_pod: first.ensemble_config.concurrent_replicas_per_pod,
            engine_flags_per_replica: first.ensemble_config.engine_flags_per_replica,
        },
        replicas,
        ensemble_consensus: None,
        audit_log,
    };
    manifest.validate()?;
    Ok(manifest)
}

/// A single row from docs/cuda_determinism_audit_results.csv.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeterminismAuditRow {
    /// Kernel identifier.
    pub kernel_id: String,
    /// Operation site.
    pub op_site: String,
    /// Operation kind.
    pub op_kind: String,
    /// Data type.
    pub data_type: String,
    /// Criticality.
    pub criticality: String,
    /// Current determinism classification.
    pub current_determinism: String,
    /// Status.
    pub status: String,
}

/// Enforce the CUDA determinism gate before multi-replica execution.
pub fn enforce_cuda_determinism_gate(
    replica_count: usize,
    explicit_audit_path: Option<&Path>,
) -> Result<()> {
    if replica_count <= 1 {
        return Ok(());
    }
    let audit_path = explicit_audit_path
        .map(PathBuf::from)
        .or_else(|| std::env::var("PRISM_CUDA_DETERMINISM_AUDIT_CSV").ok().map(PathBuf::from))
        .or_else(default_audit_csv_path)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "multi-replica execution requested (replicas={}) but docs/cuda_determinism_audit_results.csv was not found",
                replica_count
            )
        })?;
    let rows = read_determinism_audit_csv(&audit_path)?;
    if rows.is_empty() {
        bail!(
            "multi-replica execution requested (replicas={}) but {} has no audit rows",
            replica_count,
            audit_path.display()
        );
    }
    let blockers: Vec<_> = rows
        .iter()
        .filter(|row| {
            let critical = matches!(row.criticality.as_str(), "HIGH" | "MEDIUM");
            let nondeterministic = row.current_determinism == "nondeterministic";
            let accepted = matches!(row.status.as_str(), "done" | "accepted-as-is");
            critical && nondeterministic && !accepted
        })
        .collect();
    if !blockers.is_empty() {
        let detail = blockers
            .iter()
            .take(8)
            .map(|row| {
                format!(
                    "{} {} {} status={}",
                    row.kernel_id, row.op_site, row.op_kind, row.status
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        bail!(
            "multi-replica execution is ensemble-unsafe: {} HIGH/MEDIUM nondeterministic CUDA audit rows are unresolved in {}. Blockers: {}",
            blockers.len(),
            audit_path.display(),
            detail
        );
    }
    Ok(())
}

/// Read determinism audit CSV rows.
pub fn read_determinism_audit_csv(path: impl AsRef<Path>) -> Result<Vec<CudaDeterminismAuditRow>> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read CUDA determinism audit CSV {}", path.display()))?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("{} is empty", path.display()))?;
    let headers = split_csv_line(header);
    let idx = |name: &str| -> Result<usize> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| anyhow::anyhow!("{} missing CSV column {}", path.display(), name))
    };
    let kernel_idx = idx("kernel_id")?;
    let site_idx = idx("op_site")?;
    let kind_idx = idx("op_kind")?;
    let dtype_idx = idx("data_type")?;
    let crit_idx = idx("criticality")?;
    let det_idx = idx("current_determinism")?;
    let status_idx = idx("status")?;

    let mut rows = Vec::new();
    for (line_no, line) in lines.enumerate() {
        if line.trim_start().starts_with('#') {
            continue;
        }
        let cols = split_csv_line(line);
        let need = *[
            kernel_idx, site_idx, kind_idx, dtype_idx, crit_idx, det_idx, status_idx,
        ]
        .iter()
        .max()
        .unwrap();
        if cols.len() <= need {
            bail!(
                "{}:{} has {} columns, expected at least {}",
                path.display(),
                line_no + 2,
                cols.len(),
                need + 1
            );
        }
        rows.push(CudaDeterminismAuditRow {
            kernel_id: cols[kernel_idx].clone(),
            op_site: cols[site_idx].clone(),
            op_kind: cols[kind_idx].clone(),
            data_type: cols[dtype_idx].clone(),
            criticality: cols[crit_idx].clone(),
            current_determinism: cols[det_idx].clone(),
            status: cols[status_idx].clone(),
        });
    }
    Ok(rows)
}

/// SHA-256 hex digest for a file.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Best-effort SHA-256 for a file, returning "absent" when missing.
pub fn sha256_file_or_absent(path: impl AsRef<Path>) -> String {
    sha256_file(path).unwrap_or_else(|_| "absent".to_string())
}

/// Return current UTC timestamp.
pub fn now_utc_string() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Relative path from a base directory when possible.
pub fn relative_path_string(base: &Path, path: &Path) -> String {
    path.strip_prefix(base)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// Build a frame audit block by reading V2 trajectory sidecars.
pub fn frame_audit_from_v2_sidecars(output_dir: &Path, stem: &str, n_streams: usize) -> FrameAudit {
    let mut audit = FrameAudit::default();
    let mut all_ok = n_streams > 0;
    for stream_id in 0..n_streams {
        let (bin_path, audit_path) = resolve_v2_sidecar_paths(output_dir, stem, stream_id);
        let rel_audit = relative_path_string(output_dir, &audit_path);
        if audit.audit_sidecar_relative_path.is_empty() {
            audit.audit_sidecar_relative_path = rel_audit.clone();
        }
        audit.audit_sidecar_relative_paths.push(rel_audit);

        let parsed = std::fs::read(&audit_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok());
        let producer = parsed
            .as_ref()
            .and_then(|v| v.get("producer_frames_enqueued"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let writer = parsed
            .as_ref()
            .and_then(|v| v.get("frames_written"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let mismatches = parsed
            .as_ref()
            .and_then(|v| v.get("hash_mismatches"))
            .and_then(|v| v.as_u64())
            .unwrap_or(u64::MAX);
        let rolling = parsed
            .as_ref()
            .and_then(|v| v.get("rolling_hash64"))
            .and_then(|v| v.as_str())
            .map(|s| format!("fnv1a64:{}", s))
            .unwrap_or_else(|| "absent".to_string());

        let disk_count = read_v2_frame_count_header(&bin_path).unwrap_or(0);
        let disk_hash = read_v2_disk_rolling_hash(&bin_path)
            .map(|h| format!("fnv1a64:{}", h))
            .unwrap_or_else(|_| "absent".to_string());
        let stream_ok = producer == writer
            && producer == disk_count
            && mismatches == 0
            && producer > 0
            && rolling != "absent"
            && rolling == disk_hash;
        all_ok &= stream_ok;

        audit.producer_count_per_stream.push(producer);
        audit.writer_count_per_stream.push(writer);
        audit.disk_count_per_stream.push(disk_count);
        audit.producer_hash_per_stream.push(rolling.clone());
        audit.writer_hash_per_stream.push(rolling.clone());
        audit.disk_hash_per_stream.push(disk_hash);
    }
    audit.all_hashes_match = all_ok;
    audit
}

fn resolve_v2_sidecar_paths(output_dir: &Path, stem: &str, stream_id: usize) -> (PathBuf, PathBuf) {
    let final_bin = output_dir.join(format!("{}_stream{:02}_v2_frames.bin", stem, stream_id));
    let final_audit = output_dir.join(format!(
        "{}_stream{:02}_v2_frames.audit.json",
        stem, stream_id
    ));
    if final_bin.exists() || final_audit.exists() {
        return (final_bin, final_audit);
    }

    let temp_suffix = format!("_{}.bin", stream_id);
    let mut candidates: Vec<(SystemTime, PathBuf, PathBuf)> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(output_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !file_name.starts_with("prism_v2_") || !file_name.ends_with(&temp_suffix) {
                continue;
            }
            let audit_path = output_dir.join(format!("{}.audit.json", file_name));
            if !audit_path.exists() {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|meta| meta.modified())
                .unwrap_or(UNIX_EPOCH);
            candidates.push((modified, path, audit_path));
        }
    }
    candidates.sort_by_key(|(modified, _, _)| *modified);
    candidates
        .pop()
        .map(|(_, bin, audit)| (bin, audit))
        .unwrap_or((final_bin, final_audit))
}

/// Read the V2 frames.bin frame-count header.
pub fn read_v2_frame_count_header(path: &Path) -> Result<u64> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let header = bytes
        .get(..8)
        .ok_or_else(|| anyhow::anyhow!("{} shorter than 8-byte header", path.display()))?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(header);
    Ok(u64::from_le_bytes(buf))
}

fn read_v2_disk_rolling_hash(path: &Path) -> Result<String> {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut cursor = 0usize;
    let read_u64 = |bytes: &[u8], cursor: &mut usize| -> Result<u64> {
        let end = cursor.saturating_add(8);
        let slice = bytes
            .get(*cursor..end)
            .ok_or_else(|| anyhow::anyhow!("{} truncated while reading u64", path.display()))?;
        let mut buf = [0u8; 8];
        buf.copy_from_slice(slice);
        *cursor = end;
        Ok(u64::from_le_bytes(buf))
    };
    let read_u32 = |bytes: &[u8], cursor: &mut usize| -> Result<u32> {
        let end = cursor.saturating_add(4);
        let slice = bytes
            .get(*cursor..end)
            .ok_or_else(|| anyhow::anyhow!("{} truncated while reading u32", path.display()))?;
        let mut buf = [0u8; 4];
        buf.copy_from_slice(slice);
        *cursor = end;
        Ok(u32::from_le_bytes(buf))
    };
    let frame_count = read_u64(&bytes, &mut cursor)?;
    let mut rolling_hash = FNV_OFFSET;
    for _ in 0..frame_count {
        let step_start = cursor;
        let _step = read_u64(&bytes, &mut cursor)?;
        let n_start = cursor;
        let n_floats = read_u32(&bytes, &mut cursor)?;
        let payload_len = n_floats as usize * 4;
        let payload_end = cursor.saturating_add(payload_len);
        let payload = bytes.get(cursor..payload_end).ok_or_else(|| {
            anyhow::anyhow!("{} truncated while reading frame payload", path.display())
        })?;
        let mut frame_hash = FNV_OFFSET;
        frame_hash = v2_hash_update(frame_hash, &bytes[step_start..step_start + 8]);
        frame_hash = v2_hash_update(frame_hash, &bytes[n_start..n_start + 4]);
        frame_hash = v2_hash_update(frame_hash, payload);
        rolling_hash = v2_hash_update(rolling_hash, &frame_hash.to_le_bytes());
        cursor = payload_end;
    }
    if cursor != bytes.len() {
        bail!(
            "{} has {} trailing bytes after {} V2 trajectory frames",
            path.display(),
            bytes.len() - cursor,
            frame_count
        );
    }
    Ok(format!("{:016x}", rolling_hash))
}

fn v2_hash_update(mut hash: u64, bytes: &[u8]) -> u64 {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn validate_schema_version(version: &str) -> Result<()> {
    let major = version
        .split('.')
        .next()
        .ok_or_else(|| anyhow::anyhow!("invalid schema_version '{}'", version))?;
    if major != "1" {
        bail!(
            "unsupported ensemble manifest major schema version '{}'",
            version
        );
    }
    Ok(())
}

fn split_csv_line(line: &str) -> Vec<String> {
    line.split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect()
}

fn default_audit_csv_path() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let from_cwd = cwd.join("docs/cuda_determinism_audit_results.csv");
    if from_cwd.exists() {
        return Some(from_cwd);
    }
    let from_manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/cuda_determinism_audit_results.csv");
    if from_manifest.exists() {
        return Some(from_manifest);
    }
    None
}

#[cfg(test)]
mod ensemble_manifest_tests {
    use super::*;

    fn valid_frame_audit() -> FrameAudit {
        FrameAudit {
            producer_count_per_stream: vec![2, 2],
            writer_count_per_stream: vec![2, 2],
            disk_count_per_stream: vec![2, 2],
            producer_hash_per_stream: vec!["fnv1a64:a".to_string(), "fnv1a64:b".to_string()],
            writer_hash_per_stream: vec!["fnv1a64:a".to_string(), "fnv1a64:b".to_string()],
            disk_hash_per_stream: vec!["fnv1a64:a".to_string(), "fnv1a64:b".to_string()],
            all_hashes_match: true,
            audit_sidecar_relative_path: "replica_0/audit.json".to_string(),
            audit_sidecar_relative_paths: vec![
                "replica_0/s0.audit.json".to_string(),
                "replica_0/s1.audit.json".to_string(),
            ],
        }
    }

    fn replica(id: u32, binary_hash: &str) -> EnsembleReplicaRecord {
        let now = "2026-05-16T00:00:00Z".to_string();
        EnsembleReplicaRecord {
            schema_version: ENSEMBLE_MANIFEST_SCHEMA_VERSION.to_string(),
            campaign: None,
            engine: EngineMetadata {
                binary_sha256: binary_hash.to_string(),
                binary_built_at: now.clone(),
                binary_version: "v2_ignition".to_string(),
                build_features: vec!["gpu".to_string(), "v2_ignition".to_string()],
                cuda_arch: "sm_100".to_string(),
                git_commit: "abc".to_string(),
                determinism_budget_doc_sha256: "budget".to_string(),
            },
            target: TargetMetadata {
                pdb_id: "1zvd".to_string(),
                target_chain: "A".to_string(),
                input_pdb_path_relative: "1zvd.pdb".to_string(),
                input_pdb_sha256: "pdb".to_string(),
                topology_json_path_relative: "1zvd.topology.json".to_string(),
                topology_json_sha256: "topo".to_string(),
                n_residues: 10,
                n_atoms: 100,
                ground_truth_present: false,
                ground_truth_uniprot_id: None,
                ground_truth_pdb_holo: None,
                ground_truth_ligand_centroid: None,
                ground_truth_ligand_resname: None,
                ground_truth_valid_for_dcc: false,
            },
            ensemble_config: EnsembleConfig {
                n_replicas_planned: 2,
                n_replicas_completed: 1,
                n_replicas_failed: 0,
                all_replicas_passed_audit: true,
                replica_seed_strategy: "deterministic_offsets".to_string(),
                base_seed: 42,
                seed_offsets: vec![id as i64],
                concurrent_replicas_per_pod: 1,
                engine_flags_per_replica: vec!["--fast-25k".to_string()],
            },
            replica: ReplicaManifest {
                replica_id: id,
                run_seed: 42 + id as u64,
                status: ReplicaStatus::Ok,
                started_at: now.clone(),
                completed_at: now,
                duration_seconds: 1.0,
                pod_id: "pod".to_string(),
                n_streams_planned: 2,
                n_streams_completed: 2,
                n_streams_failed: 0,
                stream_failures: Vec::new(),
                memory_plan: MemoryPlan::default(),
                frame_audit: valid_frame_audit(),
                outputs: ReplicaOutputs::default(),
                engine_telemetry: EngineTelemetry::default(),
                physical_time: PhysicalTime {
                    dt_ps: 0.002,
                    save_interval_steps: 100,
                    n_steps_total: 25_000,
                    physical_duration_ps: 50.0,
                    hmr_used: false,
                    hmr_dt_ps_effective: None,
                    adaptive_dt_used: false,
                    adaptive_dt_average_ps: None,
                },
            },
            audit_log: AuditLog {
                validation_steps: vec![ValidationStep {
                    step: "frame_hash_match_per_replica".to_string(),
                    passed: true,
                    details: None,
                }],
                validator_version: ENSEMBLE_VALIDATOR_VERSION.to_string(),
                validated_at: "2026-05-16T00:00:00Z".to_string(),
            },
        }
    }

    #[test]
    fn manifest_rejects_ok_replica_with_failed_frame_audit() {
        let mut r = replica(0, "bin").replica;
        r.frame_audit.all_hashes_match = false;
        let err = r.validate().unwrap_err().to_string();
        assert!(err.contains("frame_audit"));
    }

    #[test]
    fn finalizer_rejects_mixed_binary_hashes() -> Result<()> {
        let dir = tempfile::tempdir()?;
        replica(0, "bin-a").write_pretty(dir.path().join("ensemble_replica_0.json"))?;
        replica(1, "bin-b").write_pretty(dir.path().join("ensemble_replica_1.json"))?;
        let opts = EnsembleFinalizeOptions {
            target_dir: dir.path().to_path_buf(),
            campaign_id: "c".to_string(),
            base_seed: 42,
            n_replicas_expected: 2,
        };
        let err = finalize_manifest_from_replica_records(&opts)
            .unwrap_err()
            .to_string();
        assert!(err.contains("engine binary hash mismatch"));
        Ok(())
    }

    #[test]
    fn finalizer_writes_valid_manifest_from_two_records() -> Result<()> {
        let dir = tempfile::tempdir()?;
        replica(0, "bin").write_pretty(dir.path().join("ensemble_replica_0.json"))?;
        replica(1, "bin").write_pretty(dir.path().join("ensemble_replica_1.json"))?;
        let opts = EnsembleFinalizeOptions {
            target_dir: dir.path().to_path_buf(),
            campaign_id: "c".to_string(),
            base_seed: 42,
            n_replicas_expected: 2,
        };
        let manifest = finalize_manifest_from_replica_records(&opts)?;
        assert_eq!(manifest.replicas.len(), 2);
        manifest.validate()?;
        Ok(())
    }

    #[test]
    fn determinism_gate_blocks_unresolved_high_nondeterminism() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let csv = dir.path().join("audit.csv");
        std::fs::write(
            &csv,
            "kernel_id,op_site,op_kind,data_type,affects_training_label,criticality,current_determinism,replacement_strategy,epsilon_budget,owner,eta,status,pr_link\n\
             k,line1,atomicAdd,f32,yes,HIGH,nondeterministic,redesign,,eng,2026-05-16,pending,\n",
        )?;
        let err = enforce_cuda_determinism_gate(2, Some(&csv))
            .unwrap_err()
            .to_string();
        assert!(err.contains("ensemble-unsafe"));
        Ok(())
    }
}
