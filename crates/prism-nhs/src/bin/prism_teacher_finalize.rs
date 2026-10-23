//! Stage-C teacher consensus finalizer.
//!
//! This binary consumes the compiled Stage-B ensemble aggregate outputs and
//! emits the training-facing teacher label contract.  It does not rank sites by
//! a hidden composite score: physics heads remain separate, while masks and
//! confidence fields describe convergence quality.

use anyhow::{bail, Context, Result};
use arrow_array::{
    builder::{BooleanBuilder, Float64Builder, Int32Builder, StringBuilder, UInt32Builder},
    Array, BooleanArray, Float64Array, Int32Array, RecordBatch, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use chrono::Utc;
use clap::Parser;
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter},
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

const CRITICAL_FEATURES: &[&str] = &[
    "max_phase_manifold_score",
    "cryptic_likelihood_proxy",
    "teacher_spike_hits",
    "teacher_mean_intensity",
    "teacher_uv_fraction",
    "teacher_lif_fraction",
    "teacher_wavelength_diversity",
    "teacher_wavelength_entropy",
    "teacher_signal_coupled_fraction",
    "active_causal_steps",
    "burst_motion",
    "direction_score",
    "motion_efficiency",
    "stream_entropy",
    "effective_n_streams",
];

const CORE_LABEL_FEATURES: &[&str] = &[
    "max_phase_manifold_score",
    "cryptic_likelihood_proxy",
    "teacher_spike_hits",
    "teacher_signal_coupled_fraction",
];

const REGION_RESIDUE_GAP: i32 = 3;

const MECHANISM_FEATURES: &[(&str, &str)] = &[
    (
        "teacher_mechanism_uv_aromatic_fraction",
        "UV_AROMATIC_PERTURBATION",
    ),
    (
        "teacher_mechanism_lif_thermal_shape_fraction",
        "LIF_THERMAL_SHAPE",
    ),
    (
        "teacher_mechanism_lif_local_intensity_fraction",
        "LIF_LOCAL_INTENSITY",
    ),
    ("teacher_mechanism_efp_fraction", "EFP"),
    ("teacher_mechanism_ladd_fraction", "LADD"),
    ("teacher_mechanism_cofire_fraction", "COFIRE"),
    ("teacher_mechanism_other_fraction", "OTHER"),
];

#[derive(Parser, Debug)]
#[command(name = "prism-teacher-finalize")]
#[command(about = "Finalize Stage-B PRISM Twin consensus aggregates into teacher/student labels")]
struct Args {
    /// ensemble_aggregates.parquet emitted by prism-v004-ensemble.
    #[arg(long)]
    aggregates: PathBuf,
    /// ensemble_consensus.json emitted by prism-v004-ensemble.
    #[arg(long)]
    consensus: Option<PathBuf>,
    #[arg(long)]
    output_dir: PathBuf,
    /// Strict p95 R-hat gate for hard supervised labels.
    #[arg(long, default_value_t = 2.0)]
    strict_rhat: f64,
    /// Soft R-hat ceiling used for confidence decay.
    #[arg(long, default_value_t = 5.0)]
    soft_rhat: f64,
    /// Minimum contributing replicas per critical feature.
    #[arg(long, default_value_t = 3)]
    min_replicas: u32,
    /// Maximum severe critical-feature outlier count before masking hard supervision.
    #[arg(long, default_value_t = 1)]
    max_outlier_features: u32,
    /// Minimum phase score for positive cryptic-support labels.
    #[arg(long, default_value_t = 0.55)]
    min_phase_score: f64,
    /// Mean support fraction for region core assignment.
    #[arg(long, default_value_t = 0.60)]
    core_support_fraction: f64,
    /// Mean support fraction for region fringe assignment.
    #[arg(long, default_value_t = 0.35)]
    fringe_support_fraction: f64,
    #[arg(long, default_value_t = 4096)]
    parquet_batch_size: usize,
}

#[derive(Debug, Clone)]
struct AggregateRow {
    mean: f64,
    std: f64,
    rhat: f64,
    ess: f64,
    is_bimodal: bool,
    n_replicas_used: u32,
    n_outliers: u32,
}

#[derive(Debug, Clone)]
struct LabelRow {
    residue_id: i32,
    cryptic_likelihood_label: f64,
    phase_manifold_score_mean: f64,
    phase_manifold_score_std: f64,
    phase_support_fraction: f64,
    core_support_fraction: f64,
    top1_site_rank: f64,
    top1_site_score: f64,
    spike_hits_mean: f64,
    spike_hits_std: f64,
    spike_log1p_label: f64,
    uv_fraction: f64,
    lif_fraction: f64,
    wavelength_diversity: f64,
    wavelength_entropy: f64,
    signal_coupled_fraction: f64,
    active_causal_steps: f64,
    burst_motion: f64,
    direction_score: f64,
    motion_efficiency: f64,
    stream_entropy: f64,
    effective_n_streams: f64,
    dominant_mechanism_tag: String,
    dominant_mechanism_fraction: f64,
    mechanism_uv_aromatic_fraction: f64,
    mechanism_lif_thermal_shape_fraction: f64,
    mechanism_lif_local_intensity_fraction: f64,
    mechanism_efp_fraction: f64,
    mechanism_ladd_fraction: f64,
    mechanism_cofire_fraction: f64,
    mechanism_other_fraction: f64,
    important_rhat_p95: f64,
    important_rhat_max: f64,
    important_ess_min: f64,
    n_important_features: u32,
    n_missing_important_features: u32,
    n_bimodal_important: u32,
    n_outlier_important: u32,
    n_severe_outlier_important: u32,
    min_replicas_used_important: u32,
    label_confidence: f64,
    student_weight: f64,
    uncertainty_target: f64,
    train_mask: bool,
    bimodal_mask: bool,
    outlier_mask: bool,
    uncertainty_mask: bool,
    hard_negative_mask: bool,
}

#[derive(Serialize)]
struct FinalizeSummary {
    schema_kind: &'static str,
    schema_version: &'static str,
    computed_at: String,
    computed_by: &'static str,
    target_id: String,
    n_replicas: u64,
    n_residues: usize,
    n_trainable: usize,
    n_uncertain: usize,
    n_bimodal: usize,
    n_outlier_masked: usize,
    n_hard_negatives: usize,
    thresholds: ThresholdSummary,
    outputs: OutputSummary,
    label_policy: LabelPolicy,
}

#[derive(Serialize)]
struct ThresholdSummary {
    strict_rhat: f64,
    soft_rhat: f64,
    min_replicas: u32,
    max_outlier_features: u32,
    min_phase_score: f64,
    core_support_fraction: f64,
    fringe_support_fraction: f64,
}

#[derive(Serialize)]
struct OutputSummary {
    teacher_consensus_residue_labels_parquet: String,
    teacher_consensus_residue_labels_sha256: String,
    teacher_consensus_regions_json: String,
    student_label_bundle_parquet: String,
    student_label_bundle_sha256: String,
}

#[derive(Serialize)]
struct LabelPolicy {
    hard_supervision: &'static str,
    uncertainty_supervision: &'static str,
    region_object: &'static str,
    centroid_policy: &'static str,
    mechanism_policy: &'static str,
}

#[derive(Serialize)]
struct RegionsDoc {
    schema_kind: &'static str,
    schema_version: &'static str,
    computed_at: String,
    target_id: String,
    n_replicas: u64,
    region_policy: &'static str,
    thresholds: ThresholdSummary,
    regions: Vec<RegionSummary>,
}

#[derive(Serialize)]
struct RegionSummary {
    region_id: u32,
    rank_key: i32,
    support_source: String,
    support_residue_ids: Vec<i32>,
    core_residue_ids: Vec<i32>,
    fringe_residue_ids: Vec<i32>,
    kcc_driver_residue_ids: Vec<i32>,
    hot_phase_residue_ids: Vec<i32>,
    burst_motion_residue_ids: Vec<i32>,
    n_support: usize,
    n_core: usize,
    n_trainable: usize,
    max_phase_score: f64,
    mean_phase_score: f64,
    mean_label_confidence: f64,
    mean_support_fraction: f64,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    if args.soft_rhat <= 1.0 || args.strict_rhat <= 0.0 || args.strict_rhat > args.soft_rhat {
        bail!("require 0 < --strict-rhat <= --soft-rhat and --soft-rhat > 1");
    }

    let consensus = read_consensus(args.consensus.as_deref())?;
    let target_id = consensus
        .get("target_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown_target")
        .to_string();
    let n_replicas = consensus
        .get("n_replicas")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let aggregates = read_aggregates(&args.aggregates, args.parquet_batch_size)?;
    if aggregates.is_empty() {
        bail!("{} contained no aggregate rows", args.aggregates.display());
    }

    let labels = finalize_labels(&aggregates, &args);
    if labels.is_empty() {
        bail!("no labels produced from {}", args.aggregates.display());
    }

    std::fs::create_dir_all(&args.output_dir)?;
    let teacher_path = args
        .output_dir
        .join("teacher_consensus_residue_labels.parquet");
    let student_path = args.output_dir.join("student_label_bundle.parquet");
    let regions_path = args.output_dir.join("teacher_consensus_regions.json");
    let summary_path = args.output_dir.join("teacher_finalize_summary.json");

    write_teacher_labels(&labels, &teacher_path)?;
    write_student_bundle(&labels, &student_path)?;
    let thresholds = thresholds(&args);
    let regions = build_regions(&labels, &aggregates, &args, &target_id, n_replicas);
    std::fs::write(&regions_path, serde_json::to_vec_pretty(&regions)?)?;

    let teacher_sha = sha256_file(&teacher_path)?;
    let student_sha = sha256_file(&student_path)?;
    let summary = FinalizeSummary {
        schema_kind: "prism_teacher_stage_c",
        schema_version: "1.0.0",
        computed_at: Utc::now().to_rfc3339(),
        computed_by: "prism-teacher-finalize",
        target_id,
        n_replicas,
        n_residues: labels.len(),
        n_trainable: labels.iter().filter(|r| r.train_mask).count(),
        n_uncertain: labels.iter().filter(|r| r.uncertainty_mask).count(),
        n_bimodal: labels.iter().filter(|r| r.bimodal_mask).count(),
        n_outlier_masked: labels.iter().filter(|r| r.outlier_mask).count(),
        n_hard_negatives: labels.iter().filter(|r| r.hard_negative_mask).count(),
        thresholds,
        outputs: OutputSummary {
            teacher_consensus_residue_labels_parquet: teacher_path.to_string_lossy().into_owned(),
            teacher_consensus_residue_labels_sha256: teacher_sha,
            teacher_consensus_regions_json: regions_path.to_string_lossy().into_owned(),
            student_label_bundle_parquet: student_path.to_string_lossy().into_owned(),
            student_label_bundle_sha256: student_sha,
        },
        label_policy: LabelPolicy {
            hard_supervision:
                "stable consensus rows become hard supervised labels only when critical features pass convergence, replica, bimodality, and severe-outlier gates",
            uncertainty_supervision:
                "raw one-replica outliers decay confidence but do not hard-mask labels unless they hit core label features repeatedly or coincide with convergence/bimodality instability",
            region_object:
                "residue support sets grouped by phase-manifold rank/support flags; centroids are not required",
            centroid_policy:
                "no centroid dependency for teacher/student labels; centroid views remain optional diagnostics",
            mechanism_policy:
                "dominant mechanism tag is selected from replicated Stage-B mechanism fractions; no XGB-derived tags are used",
        },
    };
    std::fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;

    println!("wrote: {}", teacher_path.display());
    println!("wrote: {}", regions_path.display());
    println!("wrote: {}", student_path.display());
    println!(
        "labels: residues={} trainable={} uncertain={} bimodal={}",
        summary.n_residues, summary.n_trainable, summary.n_uncertain, summary.n_bimodal
    );
    Ok(())
}

fn read_consensus(path: Option<&Path>) -> Result<Value> {
    match path {
        Some(p) => serde_json::from_slice(
            &std::fs::read(p).with_context(|| format!("read {}", p.display()))?,
        )
        .with_context(|| format!("parse {}", p.display())),
        None => Ok(Value::Object(Default::default())),
    }
}

fn read_aggregates(
    path: &Path,
    batch_size: usize,
) -> Result<BTreeMap<i32, HashMap<String, AggregateRow>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open parquet reader {}", path.display()))?
        .with_batch_size(batch_size);
    let mut reader = builder.build()?;
    let mut out: BTreeMap<i32, HashMap<String, AggregateRow>> = BTreeMap::new();

    while let Some(batch) = reader.next().transpose()? {
        let schema = batch.schema();
        let idx = |name: &str| -> Result<usize> {
            schema
                .index_of(name)
                .with_context(|| format!("aggregate parquet missing column {name}"))
        };
        let residue_id = batch.column(idx("residue_id")?);
        let feature_name = batch.column(idx("feature_name")?);
        let mean = batch.column(idx("mean")?);
        let std = batch.column(idx("std")?);
        let rhat = batch.column(idx("rhat")?);
        let ess = batch.column(idx("ess")?);
        let is_bimodal = batch.column(idx("is_bimodal")?);
        let n_replicas_used = batch.column(idx("n_replicas_used")?);
        let n_outliers = batch.column(idx("n_outliers")?);

        for row in 0..batch.num_rows() {
            let rid = int32_value(residue_id.as_ref(), row, "residue_id")?;
            let fname = string_value(feature_name.as_ref(), row, "feature_name")?;
            let agg = AggregateRow {
                mean: float_value(mean.as_ref(), row, "mean")?,
                std: float_value(std.as_ref(), row, "std")?,
                rhat: float_value(rhat.as_ref(), row, "rhat")?,
                ess: float_value(ess.as_ref(), row, "ess")?,
                is_bimodal: bool_value(is_bimodal.as_ref(), row, "is_bimodal")?,
                n_replicas_used: uint32_value(n_replicas_used.as_ref(), row, "n_replicas_used")?,
                n_outliers: uint32_value(n_outliers.as_ref(), row, "n_outliers")?,
            };
            out.entry(rid).or_default().insert(fname, agg);
        }
    }
    Ok(out)
}

fn int32_value(array: &dyn Array, row: usize, name: &str) -> Result<i32> {
    if array.is_null(row) {
        bail!("null {name} at row {row}");
    }
    match array.data_type() {
        DataType::Int32 => Ok(array
            .as_any()
            .downcast_ref::<Int32Array>()
            .context("Int32 downcast")?
            .value(row)),
        other => bail!("column {name} expected Int32, got {other:?}"),
    }
}

fn uint32_value(array: &dyn Array, row: usize, name: &str) -> Result<u32> {
    if array.is_null(row) {
        bail!("null {name} at row {row}");
    }
    match array.data_type() {
        DataType::UInt32 => Ok(array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .context("UInt32 downcast")?
            .value(row)),
        other => bail!("column {name} expected UInt32, got {other:?}"),
    }
}

fn float_value(array: &dyn Array, row: usize, name: &str) -> Result<f64> {
    if array.is_null(row) {
        bail!("null {name} at row {row}");
    }
    match array.data_type() {
        DataType::Float64 => Ok(array
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("Float64 downcast")?
            .value(row)),
        other => bail!("column {name} expected Float64, got {other:?}"),
    }
}

fn string_value(array: &dyn Array, row: usize, name: &str) -> Result<String> {
    if array.is_null(row) {
        bail!("null {name} at row {row}");
    }
    match array.data_type() {
        DataType::Utf8 => Ok(array
            .as_any()
            .downcast_ref::<StringArray>()
            .context("Utf8 downcast")?
            .value(row)
            .to_string()),
        other => bail!("column {name} expected Utf8, got {other:?}"),
    }
}

fn bool_value(array: &dyn Array, row: usize, name: &str) -> Result<bool> {
    if array.is_null(row) {
        bail!("null {name} at row {row}");
    }
    match array.data_type() {
        DataType::Boolean => Ok(array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .context("Boolean downcast")?
            .value(row)),
        other => bail!("column {name} expected Boolean, got {other:?}"),
    }
}

fn finalize_labels(
    aggregates: &BTreeMap<i32, HashMap<String, AggregateRow>>,
    args: &Args,
) -> Vec<LabelRow> {
    aggregates
        .iter()
        .map(|(&rid, features)| finalize_residue(rid, features, args))
        .collect()
}

fn finalize_residue(rid: i32, features: &HashMap<String, AggregateRow>, args: &Args) -> LabelRow {
    let phase = agg(features, "max_phase_manifold_score");
    let cryptic_proxy = value(features, "cryptic_likelihood_proxy")
        .or_else(|| value(features, "max_phase_manifold_score"))
        .unwrap_or(0.0);
    let phase_score = phase.map(|r| r.mean).unwrap_or(0.0);
    let phase_std = phase.map(|r| r.std).unwrap_or(0.0);
    let top10_support = mean_or(features, "is_in_top10_phase_site", 0.0);
    let top5_support = mean_or(features, "is_in_top5_phase_site", 0.0);
    let lining = mean_or(features, "is_lining", 0.0);
    let kcc_driver = mean_or(features, "is_kcc_driver", 0.0);
    let hot_phase = mean_or(features, "is_hot_phase", 0.0);
    let burst_flag = mean_or(features, "is_burst_motion", 0.0);
    let phase_support = top10_support.max(top5_support);
    let core_support = lining.max(kcc_driver).max(hot_phase).max(burst_flag);

    let critical = CRITICAL_FEATURES
        .iter()
        .filter_map(|name| agg(features, name))
        .collect::<Vec<_>>();
    let n_missing = (CRITICAL_FEATURES.len() - critical.len()) as u32;
    let rhats = critical
        .iter()
        .map(|r| finite_or(r.rhat, args.soft_rhat))
        .collect::<Vec<_>>();
    let important_rhat_p95 = percentile(rhats.clone(), 0.95).unwrap_or(args.soft_rhat);
    let important_rhat_max = rhats.into_iter().fold(1.0, f64::max);
    let important_ess_min = critical
        .iter()
        .map(|r| finite_or(r.ess, 0.0))
        .fold(f64::INFINITY, f64::min);
    let important_ess_min = if important_ess_min.is_finite() {
        important_ess_min
    } else {
        0.0
    };
    let n_bimodal_important = critical.iter().filter(|r| r.is_bimodal).count() as u32;
    let n_outlier_important = CRITICAL_FEATURES
        .iter()
        .filter_map(|name| agg(features, name))
        .filter(|r| r.n_outliers > 0)
        .count() as u32;
    let n_severe_outlier_important = CRITICAL_FEATURES
        .iter()
        .filter_map(|name| agg(features, name).map(|row| (*name, row)))
        .filter(|(name, row)| severe_outlier(name, row, args.strict_rhat))
        .count() as u32;
    let min_replicas_used_important = critical
        .iter()
        .map(|r| r.n_replicas_used)
        .min()
        .unwrap_or(0);

    let all_bimodal = features.values().any(|r| r.is_bimodal);
    let bimodal_mask = all_bimodal || n_bimodal_important > 0;
    let outlier_mask = n_severe_outlier_important > args.max_outlier_features;
    let evidence_complete = n_missing <= 2 && min_replicas_used_important >= args.min_replicas;
    let train_mask = evidence_complete
        && !bimodal_mask
        && !outlier_mask
        && important_rhat_p95 <= args.strict_rhat
        && phase.is_some();
    let hard_negative_mask = train_mask
        && phase_score < 0.25
        && phase_support < 0.20
        && core_support < 0.20
        && mean_or(features, "teacher_spike_hits", 0.0) > 0.0;
    let uncertainty_mask = !train_mask && evidence_complete;
    let label_confidence = confidence(
        important_rhat_p95,
        args.soft_rhat,
        phase_support,
        core_support,
        dominant_mechanism(features).1,
        n_bimodal_important,
        n_outlier_important,
        n_missing,
    );
    let uncertainty_target = (1.0 - label_confidence)
        .max((phase_std / (phase_score.abs() + 1.0)).min(1.0))
        .clamp(0.0, 1.0);
    let (mechanism_tag, mechanism_fraction) = dominant_mechanism(features);

    LabelRow {
        residue_id: rid,
        cryptic_likelihood_label: if phase_score >= args.min_phase_score || phase_support >= 0.5 {
            cryptic_proxy.max(phase_score).clamp(0.0, 1.0)
        } else {
            cryptic_proxy.min(phase_score).clamp(0.0, 1.0)
        },
        phase_manifold_score_mean: phase_score,
        phase_manifold_score_std: phase_std,
        phase_support_fraction: phase_support.clamp(0.0, 1.0),
        core_support_fraction: core_support.clamp(0.0, 1.0),
        top1_site_rank: mean_or(features, "top1_site_rank", 0.0),
        top1_site_score: mean_or(features, "top1_site_score", 0.0),
        spike_hits_mean: mean_or(features, "teacher_spike_hits", 0.0),
        spike_hits_std: std_or(features, "teacher_spike_hits", 0.0),
        spike_log1p_label: mean_or(features, "teacher_spike_hits", 0.0)
            .max(0.0)
            .ln_1p(),
        uv_fraction: mean_or(features, "teacher_uv_fraction", 0.0).clamp(0.0, 1.0),
        lif_fraction: mean_or(features, "teacher_lif_fraction", 0.0).clamp(0.0, 1.0),
        wavelength_diversity: mean_or(features, "teacher_wavelength_diversity", 0.0),
        wavelength_entropy: mean_or(features, "teacher_wavelength_entropy", 0.0).clamp(0.0, 1.0),
        signal_coupled_fraction: mean_or(features, "teacher_signal_coupled_fraction", 0.0)
            .clamp(0.0, 1.0),
        active_causal_steps: mean_or(features, "active_causal_steps", 0.0),
        burst_motion: mean_or(features, "burst_motion", 0.0),
        direction_score: mean_or(features, "direction_score", 0.0),
        motion_efficiency: mean_or(features, "motion_efficiency", 0.0),
        stream_entropy: mean_or(features, "stream_entropy", 0.0).clamp(0.0, 1.0),
        effective_n_streams: mean_or(features, "effective_n_streams", 0.0),
        dominant_mechanism_tag: mechanism_tag,
        dominant_mechanism_fraction: mechanism_fraction,
        mechanism_uv_aromatic_fraction: mean_or(
            features,
            "teacher_mechanism_uv_aromatic_fraction",
            0.0,
        ),
        mechanism_lif_thermal_shape_fraction: mean_or(
            features,
            "teacher_mechanism_lif_thermal_shape_fraction",
            0.0,
        ),
        mechanism_lif_local_intensity_fraction: mean_or(
            features,
            "teacher_mechanism_lif_local_intensity_fraction",
            0.0,
        ),
        mechanism_efp_fraction: mean_or(features, "teacher_mechanism_efp_fraction", 0.0),
        mechanism_ladd_fraction: mean_or(features, "teacher_mechanism_ladd_fraction", 0.0),
        mechanism_cofire_fraction: mean_or(features, "teacher_mechanism_cofire_fraction", 0.0),
        mechanism_other_fraction: mean_or(features, "teacher_mechanism_other_fraction", 0.0),
        important_rhat_p95,
        important_rhat_max,
        important_ess_min,
        n_important_features: critical.len() as u32,
        n_missing_important_features: n_missing,
        n_bimodal_important,
        n_outlier_important,
        n_severe_outlier_important,
        min_replicas_used_important,
        label_confidence,
        student_weight: if train_mask {
            label_confidence.max(0.05)
        } else {
            0.0
        },
        uncertainty_target,
        train_mask,
        bimodal_mask,
        outlier_mask,
        uncertainty_mask,
        hard_negative_mask,
    }
}

fn agg<'a>(features: &'a HashMap<String, AggregateRow>, name: &str) -> Option<&'a AggregateRow> {
    features.get(name)
}

fn value(features: &HashMap<String, AggregateRow>, name: &str) -> Option<f64> {
    features.get(name).map(|r| r.mean).filter(|v| v.is_finite())
}

fn mean_or(features: &HashMap<String, AggregateRow>, name: &str, default: f64) -> f64 {
    value(features, name).unwrap_or(default)
}

fn std_or(features: &HashMap<String, AggregateRow>, name: &str, default: f64) -> f64 {
    features
        .get(name)
        .map(|r| r.std)
        .filter(|v| v.is_finite())
        .unwrap_or(default)
}

fn dominant_mechanism(features: &HashMap<String, AggregateRow>) -> (String, f64) {
    let mut best_tag = "NO_SPIKE_EVIDENCE";
    let mut best = 0.0;
    for (feature, tag) in MECHANISM_FEATURES {
        let v = mean_or(features, feature, 0.0).clamp(0.0, 1.0);
        if v > best {
            best = v;
            best_tag = tag;
        }
    }
    if mean_or(features, "teacher_spike_hits", 0.0) <= 0.0 {
        ("NO_SPIKE_EVIDENCE".to_string(), 0.0)
    } else if best < 0.15 {
        ("MIXED_OR_LOW_SIGNAL".to_string(), best)
    } else {
        (best_tag.to_string(), best)
    }
}

fn severe_outlier(feature_name: &str, row: &AggregateRow, strict_rhat: f64) -> bool {
    if row.n_outliers == 0 {
        return false;
    }

    let core_label_feature = CORE_LABEL_FEATURES.contains(&feature_name);
    let repeated_outlier = row.n_outliers >= 2;
    let high_rhat = finite_or(row.rhat, f64::INFINITY) > strict_rhat;

    if core_label_feature {
        repeated_outlier || high_rhat || row.is_bimodal
    } else {
        repeated_outlier && (high_rhat || row.is_bimodal)
    }
}

fn confidence(
    rhat_p95: f64,
    soft_rhat: f64,
    support: f64,
    core_support: f64,
    mechanism_confidence: f64,
    n_bimodal: u32,
    n_outlier: u32,
    n_missing: u32,
) -> f64 {
    let convergence = if rhat_p95 <= 1.0 {
        1.0
    } else {
        ((soft_rhat - rhat_p95) / (soft_rhat - 1.0)).clamp(0.0, 1.0)
    };
    let support_conf =
        (0.35 + 0.45 * support.clamp(0.0, 1.0) + 0.20 * core_support.clamp(0.0, 1.0))
            .clamp(0.0, 1.0);
    let mechanism_conf = mechanism_confidence.max(0.25).min(1.0);
    let penalty = (0.70_f64).powi(n_bimodal as i32)
        * (0.85_f64).powi(n_outlier as i32)
        * (0.90_f64).powi(n_missing as i32);
    (convergence * support_conf * mechanism_conf * penalty).clamp(0.0, 1.0)
}

fn finite_or(v: f64, default: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        default
    }
}

fn percentile(mut values: Vec<f64>, q: f64) -> Option<f64> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = ((values.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    values.get(idx).copied()
}

fn build_regions(
    labels: &[LabelRow],
    aggregates: &BTreeMap<i32, HashMap<String, AggregateRow>>,
    args: &Args,
    target_id: &str,
    n_replicas: u64,
) -> RegionsDoc {
    let mut groups: BTreeMap<i32, Vec<&LabelRow>> = BTreeMap::new();
    let mut unranked = Vec::new();
    for row in labels {
        let support = row.phase_support_fraction >= args.fringe_support_fraction
            || row.phase_manifold_score_mean >= args.min_phase_score;
        if !support {
            continue;
        }
        if row.top1_site_rank.is_finite() && row.top1_site_rank >= 0.5 {
            let rank_key = (row.top1_site_rank.round() as i32).max(1);
            groups.entry(rank_key).or_default().push(row);
        } else {
            unranked.push(row);
        }
    }

    let mut region_inputs = groups
        .into_iter()
        .map(|(rank_key, rows)| (rank_key, "phase_manifold_rank".to_string(), rows))
        .collect::<Vec<_>>();

    unranked.sort_by_key(|r| r.residue_id);
    let mut current = Vec::new();
    let mut previous_residue_id: Option<i32> = None;
    let mut unranked_idx = 0;
    for row in unranked {
        let starts_new = previous_residue_id
            .map(|prev| row.residue_id - prev > REGION_RESIDUE_GAP)
            .unwrap_or(false);
        if starts_new && !current.is_empty() {
            region_inputs.push((
                1000 + unranked_idx,
                "sequence_contiguous_phase_support".to_string(),
                std::mem::take(&mut current),
            ));
            unranked_idx += 1;
        }
        current.push(row);
        previous_residue_id = Some(row.residue_id);
    }
    if !current.is_empty() {
        region_inputs.push((
            1000 + unranked_idx,
            "sequence_contiguous_phase_support".to_string(),
            current,
        ));
    }

    let mut regions = region_inputs
        .into_iter()
        .map(|(rank_key, source, rows)| {
            make_region_summary(rank_key, source, rows, aggregates, args)
        })
        .collect::<Vec<_>>();

    regions.sort_by(|a, b| {
        b.max_phase_score
            .total_cmp(&a.max_phase_score)
            .then_with(|| a.rank_key.cmp(&b.rank_key))
    });
    for (idx, region) in regions.iter_mut().enumerate() {
        region.region_id = (idx + 1) as u32;
    }

    RegionsDoc {
        schema_kind: "prism_teacher_consensus_regions",
        schema_version: "1.0.0",
        computed_at: Utc::now().to_rfc3339(),
        target_id: target_id.to_string(),
        n_replicas,
        region_policy:
            "rank-grouped residue support sets from phase-manifold support flags; no centroid required",
        thresholds: thresholds(args),
        regions,
    }
}

fn make_region_summary(
    rank_key: i32,
    support_source: String,
    rows: Vec<&LabelRow>,
    aggregates: &BTreeMap<i32, HashMap<String, AggregateRow>>,
    args: &Args,
) -> RegionSummary {
    let mut support = rows.iter().map(|r| r.residue_id).collect::<Vec<_>>();
    support.sort_unstable();
    support.dedup();
    let mut core = rows
        .iter()
        .filter(|r| {
            r.core_support_fraction >= args.core_support_fraction
                || (r.phase_manifold_score_mean >= 0.85 && r.train_mask)
        })
        .map(|r| r.residue_id)
        .collect::<Vec<_>>();
    core.sort_unstable();
    core.dedup();
    let fringe = support
        .iter()
        .copied()
        .filter(|rid| !core.contains(rid))
        .collect::<Vec<_>>();
    let selected = |feature: &str, threshold: f64| {
        let mut selected = rows
            .iter()
            .filter(|r| {
                aggregates
                    .get(&r.residue_id)
                    .and_then(|m| m.get(feature))
                    .map(|a| a.mean >= threshold)
                    .unwrap_or(false)
            })
            .map(|r| r.residue_id)
            .collect::<Vec<_>>();
        selected.sort_unstable();
        selected.dedup();
        selected
    };
    let max_phase_score = rows
        .iter()
        .map(|r| r.phase_manifold_score_mean)
        .fold(0.0, f64::max);
    RegionSummary {
        region_id: 0,
        rank_key,
        support_source,
        n_support: support.len(),
        n_core: core.len(),
        n_trainable: rows.iter().filter(|r| r.train_mask).count(),
        max_phase_score,
        mean_phase_score: mean(rows.iter().map(|r| r.phase_manifold_score_mean)),
        mean_label_confidence: mean(rows.iter().map(|r| r.label_confidence)),
        mean_support_fraction: mean(rows.iter().map(|r| r.phase_support_fraction)),
        support_residue_ids: support,
        core_residue_ids: core,
        fringe_residue_ids: fringe,
        kcc_driver_residue_ids: selected("is_kcc_driver", 0.5),
        hot_phase_residue_ids: selected("is_hot_phase", 0.5),
        burst_motion_residue_ids: selected("is_burst_motion", 0.5),
    }
}

fn mean<I>(values: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut n = 0usize;
    let mut sum = 0.0;
    for v in values.filter(|v| v.is_finite()) {
        n += 1;
        sum += v;
    }
    if n == 0 {
        0.0
    } else {
        sum / n as f64
    }
}

fn thresholds(args: &Args) -> ThresholdSummary {
    ThresholdSummary {
        strict_rhat: args.strict_rhat,
        soft_rhat: args.soft_rhat,
        min_replicas: args.min_replicas,
        max_outlier_features: args.max_outlier_features,
        min_phase_score: args.min_phase_score,
        core_support_fraction: args.core_support_fraction,
        fringe_support_fraction: args.fringe_support_fraction,
    }
}

fn write_teacher_labels(rows: &[LabelRow], path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("residue_id", DataType::Int32, false),
        Field::new("cryptic_likelihood_label", DataType::Float64, false),
        Field::new("phase_manifold_score_mean", DataType::Float64, false),
        Field::new("phase_manifold_score_std", DataType::Float64, false),
        Field::new("phase_support_fraction", DataType::Float64, false),
        Field::new("core_support_fraction", DataType::Float64, false),
        Field::new("top1_site_rank", DataType::Float64, false),
        Field::new("top1_site_score", DataType::Float64, false),
        Field::new("spike_hits_mean", DataType::Float64, false),
        Field::new("spike_hits_std", DataType::Float64, false),
        Field::new("spike_log1p_label", DataType::Float64, false),
        Field::new("uv_fraction", DataType::Float64, false),
        Field::new("lif_fraction", DataType::Float64, false),
        Field::new("wavelength_diversity", DataType::Float64, false),
        Field::new("wavelength_entropy", DataType::Float64, false),
        Field::new("signal_coupled_fraction", DataType::Float64, false),
        Field::new("active_causal_steps", DataType::Float64, false),
        Field::new("burst_motion", DataType::Float64, false),
        Field::new("direction_score", DataType::Float64, false),
        Field::new("motion_efficiency", DataType::Float64, false),
        Field::new("stream_entropy", DataType::Float64, false),
        Field::new("effective_n_streams", DataType::Float64, false),
        Field::new("dominant_mechanism_tag", DataType::Utf8, false),
        Field::new("dominant_mechanism_fraction", DataType::Float64, false),
        Field::new("mechanism_uv_aromatic_fraction", DataType::Float64, false),
        Field::new(
            "mechanism_lif_thermal_shape_fraction",
            DataType::Float64,
            false,
        ),
        Field::new(
            "mechanism_lif_local_intensity_fraction",
            DataType::Float64,
            false,
        ),
        Field::new("mechanism_efp_fraction", DataType::Float64, false),
        Field::new("mechanism_ladd_fraction", DataType::Float64, false),
        Field::new("mechanism_cofire_fraction", DataType::Float64, false),
        Field::new("mechanism_other_fraction", DataType::Float64, false),
        Field::new("important_rhat_p95", DataType::Float64, false),
        Field::new("important_rhat_max", DataType::Float64, false),
        Field::new("important_ess_min", DataType::Float64, false),
        Field::new("n_important_features", DataType::UInt32, false),
        Field::new("n_missing_important_features", DataType::UInt32, false),
        Field::new("n_bimodal_important", DataType::UInt32, false),
        Field::new("n_outlier_important", DataType::UInt32, false),
        Field::new("n_severe_outlier_important", DataType::UInt32, false),
        Field::new("min_replicas_used_important", DataType::UInt32, false),
        Field::new("label_confidence", DataType::Float64, false),
        Field::new("student_weight", DataType::Float64, false),
        Field::new("uncertainty_target", DataType::Float64, false),
        Field::new("train_mask", DataType::Boolean, false),
        Field::new("bimodal_mask", DataType::Boolean, false),
        Field::new("outlier_mask", DataType::Boolean, false),
        Field::new("uncertainty_mask", DataType::Boolean, false),
        Field::new("hard_negative_mask", DataType::Boolean, false),
    ]));
    write_label_parquet(schema, teacher_columns(rows), path)
}

fn write_student_bundle(rows: &[LabelRow], path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("residue_id", DataType::Int32, false),
        Field::new("y_cryptic", DataType::Float64, false),
        Field::new("y_phase_score", DataType::Float64, false),
        Field::new("y_spike_log1p", DataType::Float64, false),
        Field::new("y_uv_fraction", DataType::Float64, false),
        Field::new("y_lif_fraction", DataType::Float64, false),
        Field::new("y_signal_coupled_fraction", DataType::Float64, false),
        Field::new("y_active_causal_steps", DataType::Float64, false),
        Field::new("y_burst_motion", DataType::Float64, false),
        Field::new("y_direction_score", DataType::Float64, false),
        Field::new("y_motion_efficiency", DataType::Float64, false),
        Field::new("dominant_mechanism_tag", DataType::Utf8, false),
        Field::new("dominant_mechanism_fraction", DataType::Float64, false),
        Field::new("label_confidence", DataType::Float64, false),
        Field::new("student_weight", DataType::Float64, false),
        Field::new("uncertainty_target", DataType::Float64, false),
        Field::new("train_mask", DataType::Boolean, false),
        Field::new("bimodal_mask", DataType::Boolean, false),
        Field::new("outlier_mask", DataType::Boolean, false),
        Field::new("hard_negative_mask", DataType::Boolean, false),
    ]));

    let mut residue_id = Int32Builder::new();
    let mut y_cryptic = Float64Builder::new();
    let mut y_phase = Float64Builder::new();
    let mut y_spike = Float64Builder::new();
    let mut y_uv = Float64Builder::new();
    let mut y_lif = Float64Builder::new();
    let mut y_signal = Float64Builder::new();
    let mut y_active = Float64Builder::new();
    let mut y_burst = Float64Builder::new();
    let mut y_direction = Float64Builder::new();
    let mut y_motion = Float64Builder::new();
    let mut mechanism = StringBuilder::new();
    let mut mechanism_fraction = Float64Builder::new();
    let mut confidence = Float64Builder::new();
    let mut student_weight = Float64Builder::new();
    let mut uncertainty = Float64Builder::new();
    let mut train_mask = BooleanBuilder::new();
    let mut bimodal_mask = BooleanBuilder::new();
    let mut outlier_mask = BooleanBuilder::new();
    let mut hard_negative_mask = BooleanBuilder::new();

    for r in rows {
        residue_id.append_value(r.residue_id);
        y_cryptic.append_value(r.cryptic_likelihood_label);
        y_phase.append_value(r.phase_manifold_score_mean);
        y_spike.append_value(r.spike_log1p_label);
        y_uv.append_value(r.uv_fraction);
        y_lif.append_value(r.lif_fraction);
        y_signal.append_value(r.signal_coupled_fraction);
        y_active.append_value(r.active_causal_steps);
        y_burst.append_value(r.burst_motion);
        y_direction.append_value(r.direction_score);
        y_motion.append_value(r.motion_efficiency);
        mechanism.append_value(&r.dominant_mechanism_tag);
        mechanism_fraction.append_value(r.dominant_mechanism_fraction);
        confidence.append_value(r.label_confidence);
        student_weight.append_value(r.student_weight);
        uncertainty.append_value(r.uncertainty_target);
        train_mask.append_value(r.train_mask);
        bimodal_mask.append_value(r.bimodal_mask);
        outlier_mask.append_value(r.outlier_mask);
        hard_negative_mask.append_value(r.hard_negative_mask);
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(residue_id.finish()),
            Arc::new(y_cryptic.finish()),
            Arc::new(y_phase.finish()),
            Arc::new(y_spike.finish()),
            Arc::new(y_uv.finish()),
            Arc::new(y_lif.finish()),
            Arc::new(y_signal.finish()),
            Arc::new(y_active.finish()),
            Arc::new(y_burst.finish()),
            Arc::new(y_direction.finish()),
            Arc::new(y_motion.finish()),
            Arc::new(mechanism.finish()),
            Arc::new(mechanism_fraction.finish()),
            Arc::new(confidence.finish()),
            Arc::new(student_weight.finish()),
            Arc::new(uncertainty.finish()),
            Arc::new(train_mask.finish()),
            Arc::new(bimodal_mask.finish()),
            Arc::new(outlier_mask.finish()),
            Arc::new(hard_negative_mask.finish()),
        ],
    )?;
    write_parquet_batch(schema, batch, path)
}

fn teacher_columns(rows: &[LabelRow]) -> Vec<Arc<dyn Array>> {
    let mut residue_id = Int32Builder::new();
    let mut cryptic = Float64Builder::new();
    let mut phase_mean = Float64Builder::new();
    let mut phase_std = Float64Builder::new();
    let mut phase_support = Float64Builder::new();
    let mut core_support = Float64Builder::new();
    let mut top_rank = Float64Builder::new();
    let mut top_score = Float64Builder::new();
    let mut spike_mean = Float64Builder::new();
    let mut spike_std = Float64Builder::new();
    let mut spike_log = Float64Builder::new();
    let mut uv = Float64Builder::new();
    let mut lif = Float64Builder::new();
    let mut wave_div = Float64Builder::new();
    let mut wave_ent = Float64Builder::new();
    let mut signal = Float64Builder::new();
    let mut active = Float64Builder::new();
    let mut burst = Float64Builder::new();
    let mut direction = Float64Builder::new();
    let mut motion = Float64Builder::new();
    let mut stream_ent = Float64Builder::new();
    let mut eff_streams = Float64Builder::new();
    let mut mechanism = StringBuilder::new();
    let mut mechanism_frac = Float64Builder::new();
    let mut mech_uv = Float64Builder::new();
    let mut mech_lif_thermal = Float64Builder::new();
    let mut mech_lif_local = Float64Builder::new();
    let mut mech_efp = Float64Builder::new();
    let mut mech_ladd = Float64Builder::new();
    let mut mech_cofire = Float64Builder::new();
    let mut mech_other = Float64Builder::new();
    let mut rhat_p95 = Float64Builder::new();
    let mut rhat_max = Float64Builder::new();
    let mut ess_min = Float64Builder::new();
    let mut n_important = UInt32Builder::new();
    let mut n_missing = UInt32Builder::new();
    let mut n_bimodal = UInt32Builder::new();
    let mut n_outlier = UInt32Builder::new();
    let mut n_severe_outlier = UInt32Builder::new();
    let mut min_replicas = UInt32Builder::new();
    let mut confidence = Float64Builder::new();
    let mut student_weight = Float64Builder::new();
    let mut uncertainty = Float64Builder::new();
    let mut train = BooleanBuilder::new();
    let mut bimodal = BooleanBuilder::new();
    let mut outlier = BooleanBuilder::new();
    let mut uncertain = BooleanBuilder::new();
    let mut hard_negative = BooleanBuilder::new();

    for r in rows {
        residue_id.append_value(r.residue_id);
        cryptic.append_value(r.cryptic_likelihood_label);
        phase_mean.append_value(r.phase_manifold_score_mean);
        phase_std.append_value(r.phase_manifold_score_std);
        phase_support.append_value(r.phase_support_fraction);
        core_support.append_value(r.core_support_fraction);
        top_rank.append_value(r.top1_site_rank);
        top_score.append_value(r.top1_site_score);
        spike_mean.append_value(r.spike_hits_mean);
        spike_std.append_value(r.spike_hits_std);
        spike_log.append_value(r.spike_log1p_label);
        uv.append_value(r.uv_fraction);
        lif.append_value(r.lif_fraction);
        wave_div.append_value(r.wavelength_diversity);
        wave_ent.append_value(r.wavelength_entropy);
        signal.append_value(r.signal_coupled_fraction);
        active.append_value(r.active_causal_steps);
        burst.append_value(r.burst_motion);
        direction.append_value(r.direction_score);
        motion.append_value(r.motion_efficiency);
        stream_ent.append_value(r.stream_entropy);
        eff_streams.append_value(r.effective_n_streams);
        mechanism.append_value(&r.dominant_mechanism_tag);
        mechanism_frac.append_value(r.dominant_mechanism_fraction);
        mech_uv.append_value(r.mechanism_uv_aromatic_fraction);
        mech_lif_thermal.append_value(r.mechanism_lif_thermal_shape_fraction);
        mech_lif_local.append_value(r.mechanism_lif_local_intensity_fraction);
        mech_efp.append_value(r.mechanism_efp_fraction);
        mech_ladd.append_value(r.mechanism_ladd_fraction);
        mech_cofire.append_value(r.mechanism_cofire_fraction);
        mech_other.append_value(r.mechanism_other_fraction);
        rhat_p95.append_value(r.important_rhat_p95);
        rhat_max.append_value(r.important_rhat_max);
        ess_min.append_value(r.important_ess_min);
        n_important.append_value(r.n_important_features);
        n_missing.append_value(r.n_missing_important_features);
        n_bimodal.append_value(r.n_bimodal_important);
        n_outlier.append_value(r.n_outlier_important);
        n_severe_outlier.append_value(r.n_severe_outlier_important);
        min_replicas.append_value(r.min_replicas_used_important);
        confidence.append_value(r.label_confidence);
        student_weight.append_value(r.student_weight);
        uncertainty.append_value(r.uncertainty_target);
        train.append_value(r.train_mask);
        bimodal.append_value(r.bimodal_mask);
        outlier.append_value(r.outlier_mask);
        uncertain.append_value(r.uncertainty_mask);
        hard_negative.append_value(r.hard_negative_mask);
    }

    vec![
        Arc::new(residue_id.finish()),
        Arc::new(cryptic.finish()),
        Arc::new(phase_mean.finish()),
        Arc::new(phase_std.finish()),
        Arc::new(phase_support.finish()),
        Arc::new(core_support.finish()),
        Arc::new(top_rank.finish()),
        Arc::new(top_score.finish()),
        Arc::new(spike_mean.finish()),
        Arc::new(spike_std.finish()),
        Arc::new(spike_log.finish()),
        Arc::new(uv.finish()),
        Arc::new(lif.finish()),
        Arc::new(wave_div.finish()),
        Arc::new(wave_ent.finish()),
        Arc::new(signal.finish()),
        Arc::new(active.finish()),
        Arc::new(burst.finish()),
        Arc::new(direction.finish()),
        Arc::new(motion.finish()),
        Arc::new(stream_ent.finish()),
        Arc::new(eff_streams.finish()),
        Arc::new(mechanism.finish()),
        Arc::new(mechanism_frac.finish()),
        Arc::new(mech_uv.finish()),
        Arc::new(mech_lif_thermal.finish()),
        Arc::new(mech_lif_local.finish()),
        Arc::new(mech_efp.finish()),
        Arc::new(mech_ladd.finish()),
        Arc::new(mech_cofire.finish()),
        Arc::new(mech_other.finish()),
        Arc::new(rhat_p95.finish()),
        Arc::new(rhat_max.finish()),
        Arc::new(ess_min.finish()),
        Arc::new(n_important.finish()),
        Arc::new(n_missing.finish()),
        Arc::new(n_bimodal.finish()),
        Arc::new(n_outlier.finish()),
        Arc::new(n_severe_outlier.finish()),
        Arc::new(min_replicas.finish()),
        Arc::new(confidence.finish()),
        Arc::new(student_weight.finish()),
        Arc::new(uncertainty.finish()),
        Arc::new(train.finish()),
        Arc::new(bimodal.finish()),
        Arc::new(outlier.finish()),
        Arc::new(uncertain.finish()),
        Arc::new(hard_negative.finish()),
    ]
}

fn write_label_parquet(
    schema: Arc<Schema>,
    columns: Vec<Arc<dyn Array>>,
    path: &Path,
) -> Result<()> {
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    write_parquet_batch(schema, batch, path)
}

fn write_parquet_batch(schema: Arc<Schema>, batch: RecordBatch, path: &Path) -> Result<()> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(6)?))
        .build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aggregate(n_outliers: u32, rhat: f64, is_bimodal: bool) -> AggregateRow {
        AggregateRow {
            mean: 0.0,
            std: 0.0,
            rhat,
            ess: 5.0,
            is_bimodal,
            n_replicas_used: 5,
            n_outliers,
        }
    }

    fn args() -> Args {
        Args {
            aggregates: PathBuf::from("unused.parquet"),
            consensus: None,
            output_dir: PathBuf::from("unused"),
            strict_rhat: 2.0,
            soft_rhat: 5.0,
            min_replicas: 3,
            max_outlier_features: 1,
            min_phase_score: 0.55,
            core_support_fraction: 0.60,
            fringe_support_fraction: 0.35,
            parquet_batch_size: 4096,
        }
    }

    fn stable_row(mean: f64) -> AggregateRow {
        AggregateRow {
            mean,
            std: 0.01,
            rhat: 1.05,
            ess: 5.0,
            is_bimodal: false,
            n_replicas_used: 5,
            n_outliers: 0,
        }
    }

    fn stable_features() -> HashMap<String, AggregateRow> {
        let mut features = HashMap::new();
        for name in CRITICAL_FEATURES {
            features.insert((*name).to_string(), stable_row(0.7));
        }
        features.insert("teacher_spike_hits".to_string(), stable_row(100.0));
        features.insert("teacher_lif_fraction".to_string(), stable_row(0.8));
        features.insert("teacher_uv_fraction".to_string(), stable_row(0.1));
        features.insert(
            "teacher_mechanism_lif_thermal_shape_fraction".to_string(),
            stable_row(0.9),
        );
        features
    }

    #[test]
    fn severe_outlier_core_feature_masks_repeated_or_high_rhat_instability() {
        assert!(!severe_outlier(
            "max_phase_manifold_score",
            &aggregate(1, 1.05, false),
            2.0
        ));
        assert!(severe_outlier(
            "max_phase_manifold_score",
            &aggregate(2, 1.05, false),
            2.0
        ));
        assert!(severe_outlier(
            "teacher_spike_hits",
            &aggregate(1, 2.01, false),
            2.0
        ));
    }

    #[test]
    fn severe_outlier_noncore_feature_requires_repeated_instability_with_convergence_signal() {
        assert!(!severe_outlier(
            "teacher_wavelength_entropy",
            &aggregate(1, 4.0, false),
            2.0
        ));
        assert!(!severe_outlier(
            "teacher_wavelength_entropy",
            &aggregate(2, 1.05, false),
            2.0
        ));
        assert!(severe_outlier(
            "teacher_wavelength_entropy",
            &aggregate(2, 2.01, false),
            2.0
        ));
        assert!(severe_outlier(
            "teacher_wavelength_entropy",
            &aggregate(2, 1.05, true),
            2.0
        ));
    }

    #[test]
    fn finalize_residue_keeps_raw_single_replica_outlier_as_confidence_signal() {
        let args = args();
        let mut features = stable_features();
        features
            .get_mut("teacher_wavelength_entropy")
            .unwrap()
            .n_outliers = 1;
        let row = finalize_residue(42, &features, &args);
        assert!(row.train_mask);
        assert!(!row.outlier_mask);
        assert_eq!(row.n_outlier_important, 1);
        assert_eq!(row.n_severe_outlier_important, 0);
        assert!(row.label_confidence < 1.0);
    }

    #[test]
    fn finalize_residue_masks_repeated_core_instability_but_keeps_uncertainty_target() {
        let args = args();
        let mut features = stable_features();
        features.get_mut("teacher_spike_hits").unwrap().n_outliers = 2;
        features
            .get_mut("teacher_signal_coupled_fraction")
            .unwrap()
            .n_outliers = 2;
        let row = finalize_residue(42, &features, &args);
        assert!(!row.train_mask);
        assert!(row.outlier_mask);
        assert!(row.uncertainty_mask);
        assert_eq!(row.n_severe_outlier_important, 2);
        assert_eq!(row.student_weight, 0.0);
        assert!(row.uncertainty_target > 0.0);
    }

    #[test]
    fn finalize_residue_global_bimodality_blocks_hard_supervision() {
        let args = args();
        let mut features = stable_features();
        features.insert(
            "noncritical_diagnostic".to_string(),
            AggregateRow {
                is_bimodal: true,
                ..stable_row(0.5)
            },
        );
        let row = finalize_residue(42, &features, &args);
        assert!(!row.train_mask);
        assert!(row.bimodal_mask);
        assert!(row.uncertainty_mask);
    }
}
