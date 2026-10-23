//! Production driver for PRISM Twin teacher consensus packaging.
//!
//! This binary turns the previously manual Stage-B -> ensemble -> Stage-C
//! chain into one fail-closed command surface.  It may consume raw
//! `md_evidence_manifest.json` files or already-materialized Stage-B parquet
//! files, validates the contract at each boundary, and emits a provenance
//! manifest beside the final teacher/student bundles.

use anyhow::{bail, Context, Result};
use arrow_array::{Array, BooleanArray, Float64Array, RecordBatch};
use chrono::Utc;
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Parser, Debug)]
#[command(name = "prism-teacher-pipeline")]
#[command(about = "Run validated PRISM Twin Stage-B/C teacher consensus packaging")]
struct Args {
    /// Raw MD evidence manifests. Each one is materialized through prism-teacher-tags.
    #[arg(long = "replica-manifest")]
    replica_manifests: Vec<PathBuf>,

    /// Existing Stage-B teacher_residue_tags.parquet files.
    #[arg(long = "stage-b-parquet")]
    stage_b_parquets: Vec<PathBuf>,

    #[arg(long)]
    target_id: String,

    #[arg(long)]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 42)]
    base_seed: u64,

    #[arg(long, default_value_t = 3)]
    min_replicas: usize,

    #[arg(long, default_value_t = 0.05)]
    min_trainable_fraction: f64,

    #[arg(long, default_value_t = 2.0)]
    strict_rhat: f64,

    #[arg(long, default_value_t = 5.0)]
    soft_rhat: f64,

    #[arg(long, default_value_t = 3)]
    min_replicas_per_feature: u32,

    #[arg(long, default_value_t = 1)]
    max_outlier_features: u32,

    #[arg(long, default_value_t = 0.55)]
    min_phase_score: f64,

    #[arg(long, default_value_t = 0.60)]
    core_support_fraction: f64,

    #[arg(long, default_value_t = 0.35)]
    fringe_support_fraction: f64,

    #[arg(long, default_value_t = false)]
    allow_partial_stage_b: bool,

    /// R2 key prefix for the archive manifest. Defaults to a unique teacher-consensus prefix.
    #[arg(long)]
    archive_prefix: Option<String>,

    #[arg(long, default_value_t = 4096)]
    parquet_batch_size: usize,
}

#[derive(Debug, Clone, Serialize)]
struct StageBInput {
    replica_id: usize,
    source_kind: &'static str,
    source_path: String,
    parquet_path: String,
    parquet_sha256: String,
}

#[derive(Debug, Serialize)]
struct CommandRecord {
    tool: String,
    argv: Vec<String>,
    status_code: i32,
}

#[derive(Debug, Serialize)]
struct PipelineManifest {
    schema_kind: &'static str,
    schema_version: &'static str,
    computed_at: String,
    computed_by: &'static str,
    archive_prefix: String,
    target_id: String,
    n_replicas: usize,
    base_seed: u64,
    inputs: Vec<StageBInput>,
    commands: Vec<CommandRecord>,
    outputs: BTreeMap<&'static str, ArtifactSummary>,
    acceptance: AcceptanceSummary,
    student_bundle_audit: StudentBundleAudit,
}

#[derive(Debug, Serialize)]
struct ArtifactSummary {
    path: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Debug, Serialize)]
struct AcceptanceSummary {
    n_residues: u64,
    n_trainable: u64,
    n_uncertain: u64,
    n_bimodal: u64,
    n_outlier_masked: u64,
    min_trainable_fraction: f64,
    observed_trainable_fraction: f64,
}

#[derive(Debug, Serialize)]
struct StudentBundleAudit {
    row_count: u64,
    trainable_rows: u64,
    non_trainable_rows: u64,
    bimodal_masked_rows: u64,
    outlier_masked_rows: u64,
    hard_negative_rows: u64,
    invalid_trainable_mask_rows: u64,
    finite_label_failures: u64,
    required_columns_checked: Vec<&'static str>,
    finite_columns_checked: Vec<&'static str>,
}

#[derive(Debug, Serialize)]
struct ArchiveManifest {
    schema_kind: &'static str,
    schema_version: &'static str,
    computed_at: String,
    computed_by: &'static str,
    archive_profile: &'static str,
    bucket: &'static str,
    r2_prefix: String,
    target_id: String,
    n_replicas: usize,
    base_seed: u64,
    objects: Vec<ArchiveObject>,
}

#[derive(Debug, Serialize)]
struct ArchiveObject {
    kind: String,
    relative_path: String,
    object_key: String,
    local_path: String,
    sha256: String,
    size_bytes: u64,
}

const STUDENT_REQUIRED_COLUMNS: &[&str] = &[
    "residue_id",
    "y_cryptic",
    "y_phase_score",
    "y_spike_log1p",
    "y_uv_fraction",
    "y_lif_fraction",
    "y_signal_coupled_fraction",
    "y_active_causal_steps",
    "y_burst_motion",
    "y_direction_score",
    "y_motion_efficiency",
    "dominant_mechanism_tag",
    "dominant_mechanism_fraction",
    "label_confidence",
    "student_weight",
    "uncertainty_target",
    "train_mask",
    "bimodal_mask",
    "outlier_mask",
    "hard_negative_mask",
];

const STUDENT_FINITE_COLUMNS: &[&str] = &[
    "y_cryptic",
    "y_phase_score",
    "y_spike_log1p",
    "y_uv_fraction",
    "y_lif_fraction",
    "y_signal_coupled_fraction",
    "y_active_causal_steps",
    "y_burst_motion",
    "y_direction_score",
    "y_motion_efficiency",
    "dominant_mechanism_fraction",
    "label_confidence",
    "student_weight",
    "uncertainty_target",
];

fn main() -> Result<()> {
    let args = Args::parse();
    validate_args(&args)?;
    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create {}", args.output_dir.display()))?;
    let computed_at = Utc::now().to_rfc3339();

    let tools = ToolPaths::discover()?;
    let mut commands = Vec::new();
    let mut stage_b_inputs = Vec::new();

    for (idx, manifest) in args.replica_manifests.iter().enumerate() {
        validate_md_manifest(manifest, args.allow_partial_stage_b)?;
        let stage_b_dir = args
            .output_dir
            .join("stage_b")
            .join(format!("replica_{idx:03}"));
        std::fs::create_dir_all(&stage_b_dir)
            .with_context(|| format!("create {}", stage_b_dir.display()))?;

        let argv = vec![
            "--manifest".into(),
            manifest.as_os_str().to_os_string(),
            "--output-dir".into(),
            stage_b_dir.as_os_str().to_os_string(),
            "--no-update-replica-record".into(),
        ];
        commands.push(run_tool("prism-teacher-tags", &tools.teacher_tags, &argv)?);

        let parquet = stage_b_dir.join("teacher_residue_tags.parquet");
        validate_existing_file(&parquet)?;
        stage_b_inputs.push(StageBInput {
            replica_id: stage_b_inputs.len(),
            source_kind: "md_evidence_manifest",
            source_path: manifest.to_string_lossy().into_owned(),
            parquet_sha256: sha256_file(&parquet)?,
            parquet_path: parquet.to_string_lossy().into_owned(),
        });
    }

    for parquet in &args.stage_b_parquets {
        validate_existing_file(parquet)?;
        stage_b_inputs.push(StageBInput {
            replica_id: stage_b_inputs.len(),
            source_kind: "stage_b_parquet",
            source_path: parquet.to_string_lossy().into_owned(),
            parquet_sha256: sha256_file(parquet)?,
            parquet_path: parquet.to_string_lossy().into_owned(),
        });
    }

    if stage_b_inputs.len() < args.min_replicas {
        bail!(
            "only {} Stage-B inputs after materialization; require at least {}",
            stage_b_inputs.len(),
            args.min_replicas
        );
    }

    let ensemble_dir = args.output_dir.join("ensemble");
    let stage_c_dir = args.output_dir.join("stage_c");
    std::fs::create_dir_all(&ensemble_dir)
        .with_context(|| format!("create {}", ensemble_dir.display()))?;
    std::fs::create_dir_all(&stage_c_dir)
        .with_context(|| format!("create {}", stage_c_dir.display()))?;

    let mut ensemble_argv = Vec::<OsString>::new();
    for input in &stage_b_inputs {
        ensemble_argv.push("--feature-parquet".into());
        ensemble_argv.push(input.parquet_path.clone().into());
    }
    ensemble_argv.extend([
        "--target-id".into(),
        args.target_id.clone().into(),
        "--base-seed".into(),
        args.base_seed.to_string().into(),
        "--output-dir".into(),
        ensemble_dir.as_os_str().to_os_string(),
    ]);
    commands.push(run_tool(
        "prism-v004-ensemble",
        &tools.v004_ensemble,
        &ensemble_argv,
    )?);

    let aggregates = ensemble_dir.join("ensemble_aggregates.parquet");
    let consensus = ensemble_dir.join("ensemble_consensus.json");
    validate_existing_file(&aggregates)?;
    validate_existing_file(&consensus)?;

    let finalize_argv = vec![
        "--aggregates".into(),
        aggregates.as_os_str().to_os_string(),
        "--consensus".into(),
        consensus.as_os_str().to_os_string(),
        "--output-dir".into(),
        stage_c_dir.as_os_str().to_os_string(),
        "--strict-rhat".into(),
        args.strict_rhat.to_string().into(),
        "--soft-rhat".into(),
        args.soft_rhat.to_string().into(),
        "--min-replicas".into(),
        args.min_replicas_per_feature.to_string().into(),
        "--max-outlier-features".into(),
        args.max_outlier_features.to_string().into(),
        "--min-phase-score".into(),
        args.min_phase_score.to_string().into(),
        "--core-support-fraction".into(),
        args.core_support_fraction.to_string().into(),
        "--fringe-support-fraction".into(),
        args.fringe_support_fraction.to_string().into(),
    ];
    commands.push(run_tool(
        "prism-teacher-finalize",
        &tools.teacher_finalize,
        &finalize_argv,
    )?);

    let summary_path = stage_c_dir.join("teacher_finalize_summary.json");
    validate_existing_file(&summary_path)?;
    let acceptance = read_acceptance(&summary_path, args.min_trainable_fraction)?;

    let mut outputs = BTreeMap::new();
    let teacher_labels = stage_c_dir.join("teacher_consensus_residue_labels.parquet");
    let student_bundle = stage_c_dir.join("student_label_bundle.parquet");
    let regions = stage_c_dir.join("teacher_consensus_regions.json");
    for (kind, path) in [
        ("ensemble_aggregates", aggregates),
        ("ensemble_consensus", consensus),
        ("teacher_consensus_residue_labels", teacher_labels),
        ("student_label_bundle", student_bundle),
        ("teacher_consensus_regions", regions),
        ("teacher_finalize_summary", summary_path),
    ] {
        validate_existing_file(&path)?;
        outputs.insert(kind, artifact_summary(&path)?);
    }

    let student_bundle_path = outputs
        .get("student_label_bundle")
        .map(|s| PathBuf::from(&s.path))
        .context("student_label_bundle output missing")?;
    let student_bundle_audit =
        validate_student_bundle(&student_bundle_path, args.parquet_batch_size)?;
    let archive_prefix = args.archive_prefix.clone().unwrap_or_else(|| {
        default_archive_prefix(
            &args.target_id,
            stage_b_inputs.len(),
            args.base_seed,
            &computed_at,
        )
    });

    let manifest = PipelineManifest {
        schema_kind: "prism_twin_teacher_pipeline",
        schema_version: "1.0.0",
        computed_at: computed_at.clone(),
        computed_by: "prism-teacher-pipeline",
        archive_prefix: archive_prefix.clone(),
        target_id: args.target_id,
        n_replicas: stage_b_inputs.len(),
        base_seed: args.base_seed,
        inputs: stage_b_inputs,
        commands,
        outputs,
        acceptance,
        student_bundle_audit,
    };
    let pipeline_manifest_path = args
        .output_dir
        .join("teacher_consensus_pipeline_manifest.json");
    std::fs::write(
        &pipeline_manifest_path,
        serde_json::to_vec_pretty(&manifest)?,
    )
    .with_context(|| format!("write {}", pipeline_manifest_path.display()))?;
    let archive_manifest_path = write_archive_manifest(
        &args.output_dir,
        &archive_prefix,
        &manifest,
        &pipeline_manifest_path,
    )?;

    println!("wrote: {}", pipeline_manifest_path.display());
    println!("wrote: {}", archive_manifest_path.display());
    println!(
        "accepted: residues={} trainable={} uncertain={} trainable_fraction={:.3} archive_prefix={}",
        manifest.acceptance.n_residues,
        manifest.acceptance.n_trainable,
        manifest.acceptance.n_uncertain,
        manifest.acceptance.observed_trainable_fraction,
        manifest.archive_prefix
    );
    Ok(())
}

struct ToolPaths {
    teacher_tags: PathBuf,
    v004_ensemble: PathBuf,
    teacher_finalize: PathBuf,
}

impl ToolPaths {
    fn discover() -> Result<Self> {
        Ok(Self {
            teacher_tags: resolve_tool("prism-teacher-tags")?,
            v004_ensemble: resolve_tool("prism-v004-ensemble")?,
            teacher_finalize: resolve_tool("prism-teacher-finalize")?,
        })
    }
}

fn resolve_tool(name: &str) -> Result<PathBuf> {
    let exe = std::env::current_exe().context("resolve current executable")?;
    if let Some(dir) = exe.parent() {
        let sibling = dir.join(name);
        if sibling.is_file() {
            return Ok(sibling);
        }
    }
    for dir in std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
        .unwrap_or_default()
    {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Ok(candidate);
        }
    }
    bail!("could not locate required sibling/PATH tool {name}");
}

fn run_tool(name: &str, path: &Path, argv: &[OsString]) -> Result<CommandRecord> {
    let display_argv = argv
        .iter()
        .map(|s| s.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    log::info!("running {} {:?}", path.display(), display_argv);
    let status = Command::new(path)
        .args(argv)
        .status()
        .with_context(|| format!("spawn {}", path.display()))?;
    let code = status.code().unwrap_or(-1);
    if !status.success() {
        bail!("{name} failed with status {code}: {:?}", display_argv);
    }
    Ok(CommandRecord {
        tool: name.to_string(),
        argv: display_argv,
        status_code: code,
    })
}

fn validate_args(args: &Args) -> Result<()> {
    if args.replica_manifests.is_empty() && args.stage_b_parquets.is_empty() {
        bail!("provide at least one --replica-manifest or --stage-b-parquet");
    }
    if !(0.0..=1.0).contains(&args.min_trainable_fraction) {
        bail!("--min-trainable-fraction must be in [0, 1]");
    }
    if args.strict_rhat <= 0.0 || args.soft_rhat <= 1.0 || args.strict_rhat > args.soft_rhat {
        bail!("require 0 < --strict-rhat <= --soft-rhat and --soft-rhat > 1");
    }
    Ok(())
}

fn validate_md_manifest(path: &Path, allow_partial: bool) -> Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let complete = value
        .get("required_artifacts_complete")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let status = value
        .get("validation_status")
        .and_then(Value::as_str)
        .unwrap_or("");
    if !allow_partial && (!complete || status != "accepted_required_artifacts_present") {
        bail!(
            "{} is not accepted complete MD evidence: status={status:?} required_artifacts_complete={complete}",
            path.display()
        );
    }
    let stream_count = value
        .get("stream_count")
        .and_then(Value::as_u64)
        .context("manifest missing stream_count")?;
    let streams_serialized = value
        .get("streams_serialized")
        .and_then(Value::as_u64)
        .context("manifest missing streams_serialized")?;
    if !allow_partial && streams_serialized != stream_count {
        bail!(
            "{} streams_serialized {} != stream_count {}",
            path.display(),
            streams_serialized,
            stream_count
        );
    }
    validate_manifest_artifacts(&value, stream_count, allow_partial)
        .with_context(|| format!("validate artifacts in {}", path.display()))
}

fn validate_manifest_artifacts(
    value: &Value,
    stream_count: u64,
    allow_partial: bool,
) -> Result<()> {
    let artifacts = value
        .get("artifacts")
        .and_then(Value::as_array)
        .context("manifest missing artifacts[]")?;
    for kind in ["spikes", "signal_grid", "kcc_v2full"] {
        let present = artifacts
            .iter()
            .filter(|a| {
                a.get("kind").and_then(Value::as_str) == Some(kind)
                    && a.get("present").and_then(Value::as_bool).unwrap_or(false)
                    && a.get("size_bytes").and_then(Value::as_u64).unwrap_or(0) > 0
            })
            .count() as u64;
        if !allow_partial && present != stream_count {
            bail!(
                "{kind} present streams {} != expected {}",
                present,
                stream_count
            );
        }
    }
    Ok(())
}

fn validate_existing_file(path: &Path) -> Result<()> {
    let meta = path
        .metadata()
        .with_context(|| format!("missing file {}", path.display()))?;
    if !meta.is_file() || meta.len() == 0 {
        bail!("{} is not a non-empty file", path.display());
    }
    Ok(())
}

fn artifact_summary(path: &Path) -> Result<ArtifactSummary> {
    let meta = path
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?;
    Ok(ArtifactSummary {
        path: path.to_string_lossy().into_owned(),
        sha256: sha256_file(path)?,
        size_bytes: meta.len(),
    })
}

fn read_acceptance(path: &Path, min_trainable_fraction: f64) -> Result<AcceptanceSummary> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let value: Value =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    let read_u64 = |name: &str| -> Result<u64> {
        value
            .get(name)
            .and_then(Value::as_u64)
            .with_context(|| format!("teacher summary missing {name}"))
    };
    let n_residues = read_u64("n_residues")?;
    let n_trainable = read_u64("n_trainable")?;
    let observed_trainable_fraction = if n_residues == 0 {
        0.0
    } else {
        n_trainable as f64 / n_residues as f64
    };
    if observed_trainable_fraction < min_trainable_fraction {
        bail!(
            "trainable fraction {:.3} < required {:.3}",
            observed_trainable_fraction,
            min_trainable_fraction
        );
    }
    Ok(AcceptanceSummary {
        n_residues,
        n_trainable,
        n_uncertain: read_u64("n_uncertain")?,
        n_bimodal: read_u64("n_bimodal")?,
        n_outlier_masked: read_u64("n_outlier_masked")?,
        min_trainable_fraction,
        observed_trainable_fraction,
    })
}

fn validate_student_bundle(path: &Path, batch_size: usize) -> Result<StudentBundleAudit> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open parquet reader {}", path.display()))?
        .with_batch_size(batch_size);
    let mut reader = builder.build()?;
    let mut audit = StudentBundleAudit {
        row_count: 0,
        trainable_rows: 0,
        non_trainable_rows: 0,
        bimodal_masked_rows: 0,
        outlier_masked_rows: 0,
        hard_negative_rows: 0,
        invalid_trainable_mask_rows: 0,
        finite_label_failures: 0,
        required_columns_checked: STUDENT_REQUIRED_COLUMNS.to_vec(),
        finite_columns_checked: STUDENT_FINITE_COLUMNS.to_vec(),
    };
    let mut checked_schema = false;

    while let Some(batch) = reader.next().transpose()? {
        if !checked_schema {
            validate_student_schema(&batch)
                .with_context(|| format!("validate schema {}", path.display()))?;
            checked_schema = true;
        }
        audit_student_batch(&batch, &mut audit)
            .with_context(|| format!("audit student bundle {}", path.display()))?;
    }

    if audit.row_count == 0 {
        bail!("{} contains no student label rows", path.display());
    }
    if audit.trainable_rows == 0 {
        bail!("{} contains no trainable student labels", path.display());
    }
    if audit.invalid_trainable_mask_rows > 0 {
        bail!(
            "{} has {} trainable rows masked by bimodal/outlier flags",
            path.display(),
            audit.invalid_trainable_mask_rows
        );
    }
    if audit.finite_label_failures > 0 {
        bail!(
            "{} has {} null/non-finite numeric label values",
            path.display(),
            audit.finite_label_failures
        );
    }
    Ok(audit)
}

fn validate_student_schema(batch: &RecordBatch) -> Result<()> {
    let schema = batch.schema();
    for name in STUDENT_REQUIRED_COLUMNS {
        schema
            .index_of(name)
            .with_context(|| format!("student bundle missing required column {name}"))?;
    }
    Ok(())
}

fn audit_student_batch(batch: &RecordBatch, audit: &mut StudentBundleAudit) -> Result<()> {
    let train = bool_column(batch, "train_mask")?;
    let bimodal = bool_column(batch, "bimodal_mask")?;
    let outlier = bool_column(batch, "outlier_mask")?;
    let hard_negative = bool_column(batch, "hard_negative_mask")?;
    let finite_columns = STUDENT_FINITE_COLUMNS
        .iter()
        .map(|name| float64_column(batch, name).map(|array| (*name, array)))
        .collect::<Result<Vec<_>>>()?;

    for row in 0..batch.num_rows() {
        let train_mask = bool_value(train, row, "train_mask")?;
        let bimodal_mask = bool_value(bimodal, row, "bimodal_mask")?;
        let outlier_mask = bool_value(outlier, row, "outlier_mask")?;
        let hard_negative_mask = bool_value(hard_negative, row, "hard_negative_mask")?;
        audit.row_count += 1;
        if train_mask {
            audit.trainable_rows += 1;
        } else {
            audit.non_trainable_rows += 1;
        }
        if bimodal_mask {
            audit.bimodal_masked_rows += 1;
        }
        if outlier_mask {
            audit.outlier_masked_rows += 1;
        }
        if hard_negative_mask {
            audit.hard_negative_rows += 1;
        }
        if train_mask && (bimodal_mask || outlier_mask) {
            audit.invalid_trainable_mask_rows += 1;
        }
        for (name, array) in &finite_columns {
            if float64_value(array, row, name).is_err() {
                audit.finite_label_failures += 1;
            }
        }
    }
    Ok(())
}

fn bool_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a BooleanArray> {
    let idx = batch
        .schema()
        .index_of(name)
        .with_context(|| format!("missing boolean column {name}"))?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .with_context(|| format!("{name} is not a Boolean column"))
}

fn float64_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Float64Array> {
    let idx = batch
        .schema()
        .index_of(name)
        .with_context(|| format!("missing Float64 column {name}"))?;
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .with_context(|| format!("{name} is not a Float64 column"))
}

fn bool_value(array: &BooleanArray, row: usize, name: &str) -> Result<bool> {
    if array.is_null(row) {
        bail!("{name} has null at row {row}");
    }
    Ok(array.value(row))
}

fn float64_value(array: &Float64Array, row: usize, name: &str) -> Result<f64> {
    if array.is_null(row) {
        bail!("{name} has null at row {row}");
    }
    let value = array.value(row);
    if !value.is_finite() {
        bail!("{name} has non-finite value at row {row}");
    }
    Ok(value)
}

fn write_archive_manifest(
    output_dir: &Path,
    archive_prefix: &str,
    manifest: &PipelineManifest,
    pipeline_manifest_path: &Path,
) -> Result<PathBuf> {
    let mut objects = Vec::new();
    for (kind, summary) in &manifest.outputs {
        let path = PathBuf::from(&summary.path);
        objects.push(archive_object(output_dir, archive_prefix, kind, &path)?);
    }
    objects.push(archive_object(
        output_dir,
        archive_prefix,
        "teacher_consensus_pipeline_manifest",
        pipeline_manifest_path,
    )?);

    let archive = ArchiveManifest {
        schema_kind: "prism_twin_teacher_archive_manifest",
        schema_version: "1.0.0",
        computed_at: manifest.computed_at.clone(),
        computed_by: "prism-teacher-pipeline",
        archive_profile: "prism_twin_teacher_v1",
        bucket: "prism-archive",
        r2_prefix: archive_prefix.to_string(),
        target_id: manifest.target_id.clone(),
        n_replicas: manifest.n_replicas,
        base_seed: manifest.base_seed,
        objects,
    };
    let path = output_dir.join("teacher_archive_manifest.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&archive)?)
        .with_context(|| format!("write {}", path.display()))?;
    Ok(path)
}

fn archive_object(
    output_dir: &Path,
    archive_prefix: &str,
    kind: &str,
    path: &Path,
) -> Result<ArchiveObject> {
    validate_existing_file(path)?;
    let relative_path = relative_output_path(output_dir, path);
    Ok(ArchiveObject {
        kind: kind.to_string(),
        object_key: format!("{}/{}", archive_prefix.trim_matches('/'), relative_path),
        relative_path,
        local_path: path.to_string_lossy().into_owned(),
        sha256: sha256_file(path)?,
        size_bytes: path.metadata()?.len(),
    })
}

fn relative_output_path(output_dir: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(output_dir).unwrap_or_else(|_| {
        path.file_name()
            .map(Path::new)
            .unwrap_or_else(|| Path::new("artifact"))
    });
    path_to_key(rel)
}

fn path_to_key(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn default_archive_prefix(
    target_id: &str,
    n_replicas: usize,
    base_seed: u64,
    computed_at: &str,
) -> String {
    let stamp = computed_at
        .chars()
        .filter(|c| c.is_ascii_digit())
        .take(14)
        .collect::<String>();
    format!(
        "teacher-consensus/{}/replicas-{}/seed-{}/{}",
        sanitize_key_component(target_id),
        n_replicas,
        base_seed,
        stamp
    )
}

fn sanitize_key_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        "unknown".to_string()
    } else {
        out
    }
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .with_context(|| format!("read {}", path.display()))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_artifact_validation_requires_required_streams() {
        let value = serde_json::json!({
            "artifacts": [
                {"kind": "spikes", "present": true, "size_bytes": 10},
                {"kind": "signal_grid", "present": true, "size_bytes": 10},
                {"kind": "kcc_v2full", "present": true, "size_bytes": 10}
            ]
        });
        validate_manifest_artifacts(&value, 1, false).unwrap();
        let err = validate_manifest_artifacts(&value, 2, false).unwrap_err();
        assert!(err.to_string().contains("present streams 1 != expected 2"));
    }

    #[test]
    fn acceptance_rejects_too_few_trainable_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("summary.json");
        std::fs::write(
            &path,
            r#"{
                "n_residues": 100,
                "n_trainable": 4,
                "n_uncertain": 96,
                "n_bimodal": 0,
                "n_outlier_masked": 0
            }"#,
        )
        .unwrap();
        assert!(read_acceptance(&path, 0.05).is_err());
        assert_eq!(read_acceptance(&path, 0.04).unwrap().n_trainable, 4);
    }

    #[test]
    fn archive_prefix_is_r2_key_safe_and_unique_to_run_shape() {
        let prefix = default_archive_prefix(
            "Mpro monomer/prism twin",
            5,
            42,
            "2026-05-16T22:19:31.123456Z",
        );
        assert_eq!(
            prefix,
            "teacher-consensus/Mpro_monomer_prism_twin/replicas-5/seed-42/20260516221931"
        );
    }

    #[test]
    fn trainable_mask_rejects_outlier_or_bimodal_rows() {
        let mut columns: Vec<(&str, arrow_array::ArrayRef)> = Vec::new();
        columns.push((
            "residue_id",
            std::sync::Arc::new(arrow_array::Int32Array::from(vec![101, 102]))
                as arrow_array::ArrayRef,
        ));
        for name in STUDENT_FINITE_COLUMNS {
            columns.push((
                name,
                std::sync::Arc::new(Float64Array::from(vec![1.0, 0.5])) as arrow_array::ArrayRef,
            ));
        }
        columns.push((
            "dominant_mechanism_tag",
            std::sync::Arc::new(arrow_array::StringArray::from(vec![
                "UV_AROMATIC_PERTURBATION",
                "LIF_THERMAL_SHAPE",
            ])) as arrow_array::ArrayRef,
        ));
        columns.push((
            "train_mask",
            std::sync::Arc::new(BooleanArray::from(vec![true, true])) as arrow_array::ArrayRef,
        ));
        columns.push((
            "bimodal_mask",
            std::sync::Arc::new(BooleanArray::from(vec![false, true])) as arrow_array::ArrayRef,
        ));
        columns.push((
            "outlier_mask",
            std::sync::Arc::new(BooleanArray::from(vec![false, false])) as arrow_array::ArrayRef,
        ));
        columns.push((
            "hard_negative_mask",
            std::sync::Arc::new(BooleanArray::from(vec![false, false])) as arrow_array::ArrayRef,
        ));
        let batch = RecordBatch::try_from_iter(columns).unwrap();
        validate_student_schema(&batch).unwrap();
        let mut audit = StudentBundleAudit {
            row_count: 0,
            trainable_rows: 0,
            non_trainable_rows: 0,
            bimodal_masked_rows: 0,
            outlier_masked_rows: 0,
            hard_negative_rows: 0,
            invalid_trainable_mask_rows: 0,
            finite_label_failures: 0,
            required_columns_checked: STUDENT_REQUIRED_COLUMNS.to_vec(),
            finite_columns_checked: STUDENT_FINITE_COLUMNS.to_vec(),
        };
        audit_student_batch(&batch, &mut audit).unwrap();
        assert_eq!(audit.row_count, 2);
        assert_eq!(audit.trainable_rows, 2);
        assert_eq!(audit.invalid_trainable_mask_rows, 1);
    }
}
