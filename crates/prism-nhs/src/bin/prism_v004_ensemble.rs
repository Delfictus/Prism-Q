//! Native v004 ensemble teacher aggregate assembler.
//!
//! This is the compiled replacement for the Python aggregate path. It consumes
//! a validated ensemble_manifest.json plus per-replica v5 Parquet feature files
//! and emits:
//!   - ensemble_aggregates.parquet
//!   - convergence_rhat.parquet
//!   - ensemble_consensus.json
//!
//! Training frameworks can still consume the Parquet outputs, but the expensive
//! integrity validation and ensemble math live in typed Rust.

use anyhow::{bail, Context, Result};
use arrow_array::{
    builder::{BooleanBuilder, Float64Builder, Int32Builder, StringBuilder, UInt32Builder},
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    RecordBatch, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow_schema::{DataType, Field, Schema};
use chrono::Utc;
use clap::Parser;
use parquet::{
    arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter},
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};
use prism_nhs::ensemble::manifest::{EnsembleManifest, ReplicaManifest};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::File,
    path::{Path, PathBuf},
    sync::Arc,
};

const RESIDUE_COLUMNS: &[&str] = &[
    "residue_id",
    "res_id",
    "residue_idx",
    "residue_index",
    "rid",
];

const TARGET_GROUPS: &[(&str, &[&str])] = &[
    (
        "teacher_spike",
        &[
            "teacher_spike_hits",
            "teacher_total_intensity",
            "teacher_mean_intensity",
            "teacher_max_intensity",
            "teacher_aromatic_hits",
        ],
    ),
    (
        "teacher_protocol",
        &[
            "teacher_uv_fraction",
            "teacher_lif_fraction",
            "teacher_source_diversity",
            "teacher_source_entropy",
            "teacher_wavelength_diversity",
            "teacher_wavelength_entropy",
            "teacher_uv_lif_concordance",
            "teacher_mechanism_uv_aromatic_fraction",
            "teacher_mechanism_lif_thermal_shape_fraction",
            "teacher_mechanism_lif_local_intensity_fraction",
            "teacher_mechanism_efp_fraction",
            "teacher_mechanism_ladd_fraction",
            "teacher_mechanism_cofire_fraction",
            "teacher_mechanism_other_fraction",
            "teacher_mean_abs_wd_change",
            "teacher_mean_water_density",
            "teacher_mean_vibrational_energy",
            "teacher_mean_nearby_excited",
            "teacher_signal_voxel_hits",
            "teacher_signal_primary_count",
            "teacher_signal_coupled_count",
            "teacher_signal_coupled_fraction",
        ],
    ),
    (
        "kcc",
        &[
            "kcc_score",
            "active_causal_steps",
            "burst_motion",
            "causal_lag",
            "direction_score",
            "lag_corr_peak",
            "local_cov",
            "motion_efficiency",
            "phase_shift",
            "kcc_selected_stream",
            "sum_motion",
            "total_steps",
            "net_dx_norm",
            "nearest_site_gtck",
            "nearest_kcc_confidence",
            "nearest_temporal_corr",
            "nearest_site_burst_motion",
            "nearest_site_causal_lag",
            "nearest_site_direction_score",
            "nearest_site_lag_corr_peak",
            "nearest_site_local_cov",
            "nearest_site_motion_efficiency",
        ],
    ),
    (
        "therm",
        &[
            "therm_class",
            "is_cryptic",
            "pocket_ccns_tau",
            "pocket_druggability",
            "pocket_hysteresis_asym",
            "nearest_pocket_dist",
            "max_transfer_entropy",
            "sum_transfer_entropy",
            "pocket_top_count",
        ],
    ),
    ("asc", &["asc_s_pc", "asc_n_groups", "asc_in_consensus"]),
    (
        "gcpid",
        &[
            "gcpid_n_samples",
            "gcpid_redundancy_nats",
            "gcpid_synergy_nats",
            "gcpid_synergy_fraction",
            "gcpid_total_mi_nats",
            "gcpid_unique_a_nats",
            "gcpid_unique_b_nats",
        ],
    ),
    (
        "phasors",
        &[
            "phasor_mag_0",
            "phasor_mag_1",
            "phasor_mag_2",
            "phasor_mag_3",
            "phasor_phase_0",
            "phasor_phase_1",
            "phasor_phase_2",
            "phasor_phase_3",
            "phasor_count_0",
            "phasor_count_1",
            "phasor_count_2",
            "phasor_count_3",
            "phasor_coherence_01",
            "phasor_coherence_02",
            "phasor_coherence_03",
            "phasor_coherence_12",
            "phasor_coherence_13",
            "phasor_coherence_23",
            "phasor_phase_diff_01",
            "phasor_phase_diff_02",
            "phasor_phase_diff_03",
            "phasor_phase_diff_12",
            "phasor_phase_diff_13",
            "phasor_phase_diff_23",
            "phasor_mean_mag",
            "phasor_total_count",
            "phasor_scout_observer_coherence",
        ],
    ),
    (
        "phase_manifold",
        &[
            "max_phase_manifold_score",
            "top1_site_score",
            "top1_site_classification",
            "top1_site_centroid_view",
            "top1_site_rank",
            "is_in_top1_phase_site",
            "is_in_top5_phase_site",
            "is_in_top10_phase_site",
            "n_sites_containing_residue",
            "is_all_region",
            "n_sites_as_all_region",
            "is_lining",
            "n_sites_as_lining",
            "is_kcc_driver",
            "n_sites_as_kcc_driver",
            "is_hot_phase",
            "n_sites_as_hot_phase",
            "is_cold_phase",
            "n_sites_as_cold_phase",
            "is_burst_motion",
            "n_sites_as_burst_motion",
            "is_validation_contact",
            "n_sites_as_validation_contact",
            "cryptic_likelihood_proxy",
        ],
    ),
    (
        "stream",
        &[
            "stream_entropy",
            "stream_dominant_id",
            "stream_max_fraction",
            "effective_n_streams",
            "scout_mean_spikes",
            "observer_mean_spikes",
            "scout_observer_contrast",
        ],
    ),
    (
        "phase_bit",
        &[
            "phase_bit_entropy",
            "ccns_phase_entropy",
            "ccns_dominant_phase",
            "ccns_max_fraction",
            "phase_popcount_mean",
        ],
    ),
    (
        "druggability",
        &[
            "residue_druggability_pdb",
            "residue_druggability_seen",
            "nearest_site_druggability",
            "nearest_site_classification",
        ],
    ),
    (
        "ground_truth",
        &[
            "gt_dist_to_ligand",
            "gt_in_contact_5A",
            "gt_in_contact_8A",
            "gt_has_ground_truth",
            "gt_valid_for_dcc",
            "gt_ligand_n_atoms",
        ],
    ),
    (
        "p2rank",
        &[
            "p2rank_score",
            "p2rank_zscore",
            "p2rank_probability",
            "p2rank_pocket",
            "p2rank_has_data",
        ],
    ),
];

#[derive(Parser, Debug)]
#[command(name = "prism-v004-ensemble")]
#[command(about = "Native Rust v004 ensemble aggregate assembler")]
struct Args {
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long = "feature-parquet")]
    feature_parquets: Vec<PathBuf>,
    #[arg(long, default_value = "teacher_stage_b")]
    target_id: String,
    #[arg(long, default_value_t = 42)]
    base_seed: u64,
    #[arg(long)]
    output_dir: PathBuf,
    #[arg(long, default_value_t = 3.5)]
    mad_threshold: f64,
    #[arg(long, default_value_t = 1024)]
    parquet_batch_size: usize,
}

#[derive(Debug)]
struct ReplicaFeatures {
    replica_id: u32,
    run_seed: u64,
    parquet_path: PathBuf,
    values: HashMap<String, BTreeMap<i32, f64>>,
}

#[derive(Debug)]
struct AggregateRow {
    residue_id: i32,
    feature_group: &'static str,
    feature_name: &'static str,
    mean: f64,
    std: f64,
    median: f64,
    q05: f64,
    q25: f64,
    q75: f64,
    q95: f64,
    rhat: f64,
    ess: f64,
    is_bimodal: bool,
    n_replicas_used: u32,
    n_outliers: u32,
}

#[derive(Serialize)]
struct Consensus<'a> {
    computed_at: String,
    computed_by: &'a str,
    target_id: &'a str,
    n_replicas: usize,
    n_residues: usize,
    feature_count: usize,
    per_residue_aggregates_relative_path: &'a str,
    per_residue_aggregates_sha256: String,
    convergence_rhat_relative_path: &'a str,
    rhat_summary: RhatSummary,
    outlier_detection: OutlierSummary,
    bimodality: BimodalitySummary,
    replica_feature_parquets: Vec<ReplicaParquetSummary>,
}

#[derive(Serialize)]
struct RhatSummary {
    method: &'static str,
    min: f64,
    median: f64,
    p95: f64,
    max: f64,
}

#[derive(Serialize)]
struct OutlierSummary {
    method: &'static str,
    mad_threshold: f64,
    n_outlier_replicas_max: u32,
    outliers_detected: bool,
}

#[derive(Serialize)]
struct BimodalitySummary {
    method: &'static str,
    n_residues_bimodal: usize,
    bimodal_residue_ids: Vec<i32>,
}

#[derive(Serialize)]
struct ReplicaParquetSummary {
    replica_id: u32,
    run_seed: u64,
    path: String,
    sha256: String,
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    if args.manifest.is_none() && args.feature_parquets.is_empty() {
        bail!("provide either --manifest <ensemble_manifest.json> or at least two --feature-parquet paths");
    }

    let (target_id, replicas, fallback_n_residues) = if !args.feature_parquets.is_empty() {
        if args.feature_parquets.len() < 2 {
            bail!("direct v004 ensemble assembly requires at least two --feature-parquet inputs");
        }
        let mut replicas = Vec::with_capacity(args.feature_parquets.len());
        for (idx, parquet) in args.feature_parquets.iter().enumerate() {
            let values = read_feature_parquet(parquet, args.parquet_batch_size)
                .with_context(|| format!("read feature parquet {}", parquet.display()))?;
            if values.is_empty() {
                bail!(
                    "{} contains no v004 target feature columns",
                    parquet.display()
                );
            }
            replicas.push(ReplicaFeatures {
                replica_id: idx as u32,
                run_seed: args.base_seed + idx as u64,
                parquet_path: parquet.clone(),
                values,
            });
        }
        (args.target_id.clone(), replicas, 0usize)
    } else {
        let manifest_path = args.manifest.as_ref().expect("checked above");
        let manifest_dir = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let manifest = EnsembleManifest::from_path(manifest_path)?;
        if manifest.replicas.len() < 2 {
            bail!("v004 ensemble assembly requires at least two replicas");
        }
        let raw_manifest: Value = serde_json::from_slice(
            &std::fs::read(manifest_path)
                .with_context(|| format!("read {}", manifest_path.display()))?,
        )?;

        let mut replicas = Vec::with_capacity(manifest.replicas.len());
        for replica in &manifest.replicas {
            let parquet = find_replica_parquet(&manifest_dir, replica, &raw_manifest)?;
            let values =
                read_feature_parquet(&parquet, args.parquet_batch_size).with_context(|| {
                    format!("read feature parquet for replica {}", replica.replica_id)
                })?;
            if values.is_empty() {
                bail!(
                    "{} contains no v004 target feature columns",
                    parquet.display()
                );
            }
            replicas.push(ReplicaFeatures {
                replica_id: replica.replica_id,
                run_seed: replica.run_seed,
                parquet_path: parquet,
                values,
            });
        }
        (
            manifest.target.pdb_id.clone(),
            replicas,
            manifest.target.n_residues,
        )
    };

    let residue_axis = residue_axis(&replicas, fallback_n_residues);
    let rows = aggregate_rows(&residue_axis, &replicas, args.mad_threshold);
    if rows.is_empty() {
        bail!("no aggregate rows produced; check per-replica feature parquet columns");
    }

    std::fs::create_dir_all(&args.output_dir)?;

    let agg_path = args.output_dir.join("ensemble_aggregates.parquet");
    let rhat_path = args.output_dir.join("convergence_rhat.parquet");
    write_aggregate_parquet(&rows, &agg_path)?;
    write_rhat_parquet(&rows, &rhat_path)?;
    let consensus = build_consensus(&target_id, &replicas, &rows, &agg_path, args.mad_threshold)?;
    let consensus_path = args.output_dir.join("ensemble_consensus.json");
    std::fs::write(&consensus_path, serde_json::to_vec_pretty(&consensus)?)?;

    println!("wrote: {}", agg_path.display());
    println!("features: {}", consensus.feature_count);
    println!("rhat p95: {:.6}", consensus.rhat_summary.p95);
    println!(
        "bimodal residues: {}",
        consensus.bimodality.n_residues_bimodal
    );
    Ok(())
}

fn find_replica_parquet(
    manifest_dir: &Path,
    replica: &ReplicaManifest,
    raw_manifest: &Value,
) -> Result<PathBuf> {
    let rid = replica.replica_id;
    let outputs = raw_manifest
        .get("replicas")
        .and_then(Value::as_array)
        .and_then(|arr| {
            arr.iter().find(|r| {
                r.get("replica_id")
                    .and_then(Value::as_u64)
                    .map(|v| v as u32 == rid)
                    .unwrap_or(false)
            })
        })
        .and_then(|r| r.get("outputs"));

    for key in [
        "feature_parquet_relative",
        "v5_feature_parquet_relative",
        "training_feature_parquet_relative",
    ] {
        if let Some(p) = outputs.and_then(|o| o.get(key)).and_then(Value::as_str) {
            let path = resolve_path(manifest_dir, p);
            if path.exists() {
                return Ok(path);
            }
        }
    }

    let patterns = [
        format!("replica_{}/*_v5.parquet", rid),
        format!("replica_{}/**/*_v5.parquet", rid),
        format!("features/*replica_{}*_v5.parquet", rid),
        format!("*replica_{}*_v5.parquet", rid),
    ];
    for pattern in patterns {
        let glob_pattern = manifest_dir.join(pattern).to_string_lossy().into_owned();
        for hit in glob::glob(&glob_pattern).with_context(|| format!("glob {}", glob_pattern))? {
            let path = hit?;
            if path.exists() {
                return Ok(path);
            }
        }
    }
    bail!("no v5 feature parquet found for replica {}", rid)
}

fn resolve_path(base: &Path, value: &str) -> PathBuf {
    let p = PathBuf::from(value);
    if p.is_absolute() {
        p
    } else {
        base.join(p)
    }
}

fn read_feature_parquet(
    path: &Path,
    batch_size: usize,
) -> Result<HashMap<String, BTreeMap<i32, f64>>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("open parquet reader {}", path.display()))?
        .with_batch_size(batch_size);
    let mut reader = builder.build()?;
    let mut out: HashMap<String, BTreeMap<i32, f64>> = HashMap::new();
    let features = target_feature_names();

    let mut row_offset = 0usize;
    while let Some(batch) = reader.next().transpose()? {
        let residue_ids = batch_residue_ids(&batch, row_offset)?;
        let schema = batch.schema();
        for feature in &features {
            let Ok(idx) = schema.index_of(feature) else {
                continue;
            };
            let array = batch.column(idx);
            if !is_numeric(array.data_type()) {
                continue;
            }
            let metric = out.entry((*feature).to_string()).or_default();
            for row in 0..batch.num_rows() {
                if let Some(v) = array_value_f64(array.as_ref(), row) {
                    metric.insert(residue_ids[row], v);
                }
            }
        }
        row_offset += batch.num_rows();
    }
    Ok(out)
}

fn batch_residue_ids(batch: &RecordBatch, row_offset: usize) -> Result<Vec<i32>> {
    for name in RESIDUE_COLUMNS {
        if let Ok(idx) = batch.schema().index_of(name) {
            let array = batch.column(idx);
            let mut out = Vec::with_capacity(batch.num_rows());
            for row in 0..batch.num_rows() {
                let v = array_value_i64(array.as_ref(), row)
                    .with_context(|| format!("invalid residue id in column {}", name))?;
                out.push(v as i32);
            }
            return Ok(out);
        }
    }
    Ok((0..batch.num_rows())
        .map(|i| (row_offset + i) as i32)
        .collect())
}

fn is_numeric(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Float32
            | DataType::Float64
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean
    )
}

fn array_value_f64(array: &dyn Array, row: usize) -> Option<f64> {
    if array.is_null(row) {
        return None;
    }
    let v = match array.data_type() {
        DataType::Float32 => array.as_any().downcast_ref::<Float32Array>()?.value(row) as f64,
        DataType::Float64 => array.as_any().downcast_ref::<Float64Array>()?.value(row),
        DataType::Int16 => array.as_any().downcast_ref::<Int16Array>()?.value(row) as f64,
        DataType::Int32 => array.as_any().downcast_ref::<Int32Array>()?.value(row) as f64,
        DataType::Int64 => array.as_any().downcast_ref::<Int64Array>()?.value(row) as f64,
        DataType::UInt8 => array.as_any().downcast_ref::<UInt8Array>()?.value(row) as f64,
        DataType::UInt16 => array.as_any().downcast_ref::<UInt16Array>()?.value(row) as f64,
        DataType::UInt32 => array.as_any().downcast_ref::<UInt32Array>()?.value(row) as f64,
        DataType::UInt64 => array.as_any().downcast_ref::<UInt64Array>()?.value(row) as f64,
        DataType::Boolean => {
            if array.as_any().downcast_ref::<BooleanArray>()?.value(row) {
                1.0
            } else {
                0.0
            }
        }
        _ => return None,
    };
    v.is_finite().then_some(v)
}

fn array_value_i64(array: &dyn Array, row: usize) -> Option<i64> {
    if array.is_null(row) {
        return None;
    }
    match array.data_type() {
        DataType::Int16 => Some(array.as_any().downcast_ref::<Int16Array>()?.value(row) as i64),
        DataType::Int32 => Some(array.as_any().downcast_ref::<Int32Array>()?.value(row) as i64),
        DataType::Int64 => Some(array.as_any().downcast_ref::<Int64Array>()?.value(row)),
        DataType::UInt8 => Some(array.as_any().downcast_ref::<UInt8Array>()?.value(row) as i64),
        DataType::UInt16 => Some(array.as_any().downcast_ref::<UInt16Array>()?.value(row) as i64),
        DataType::UInt32 => Some(array.as_any().downcast_ref::<UInt32Array>()?.value(row) as i64),
        DataType::UInt64 => Some(array.as_any().downcast_ref::<UInt64Array>()?.value(row) as i64),
        _ => None,
    }
}

fn target_feature_names() -> BTreeSet<&'static str> {
    TARGET_GROUPS
        .iter()
        .flat_map(|(_, features)| features.iter().copied())
        .collect()
}

fn residue_axis(replicas: &[ReplicaFeatures], fallback_n_residues: usize) -> Vec<i32> {
    let observed = replicas
        .iter()
        .flat_map(|r| r.values.values())
        .flat_map(|m| m.keys().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if !observed.is_empty() {
        return observed;
    }
    if fallback_n_residues > 0 {
        return (0..fallback_n_residues).map(|i| i as i32).collect();
    }
    Vec::new()
}

fn aggregate_rows(
    residue_axis: &[i32],
    replicas: &[ReplicaFeatures],
    mad_k: f64,
) -> Vec<AggregateRow> {
    let mut rows = Vec::new();
    for (group, features) in TARGET_GROUPS {
        for feature in *features {
            if !replicas.iter().any(|r| r.values.contains_key(*feature)) {
                continue;
            }
            for &rid in residue_axis {
                let vals = replicas
                    .iter()
                    .filter_map(|r| r.values.get(*feature).and_then(|m| m.get(&rid)).copied())
                    .filter(|v| v.is_finite())
                    .collect::<Vec<_>>();
                if vals.is_empty() {
                    continue;
                }
                let outliers = outlier_mask(&vals, mad_k);
                let kept = vals
                    .iter()
                    .zip(outliers.iter())
                    .filter_map(|(v, is_outlier)| (!*is_outlier).then_some(*v))
                    .collect::<Vec<_>>();
                if kept.is_empty() {
                    continue;
                }
                let mean = mean(&kept);
                let std = sample_std(&kept, mean);
                let rhat = rhat_proxy(mean, std, kept.len());
                let ess = if rhat.is_finite() {
                    kept.len() as f64 / (rhat * rhat)
                } else {
                    0.0
                };
                rows.push(AggregateRow {
                    residue_id: rid,
                    feature_group: group,
                    feature_name: feature,
                    mean,
                    std,
                    median: quantile(&kept, 0.50),
                    q05: quantile(&kept, 0.05),
                    q25: quantile(&kept, 0.25),
                    q75: quantile(&kept, 0.75),
                    q95: quantile(&kept, 0.95),
                    rhat,
                    ess,
                    is_bimodal: bimodal_gap_heuristic(&kept),
                    n_replicas_used: kept.len() as u32,
                    n_outliers: outliers.iter().filter(|v| **v).count() as u32,
                });
            }
        }
    }
    rows.sort_by(|a, b| {
        (a.residue_id, a.feature_group, a.feature_name).cmp(&(
            b.residue_id,
            b.feature_group,
            b.feature_name,
        ))
    });
    rows
}

fn mean(vals: &[f64]) -> f64 {
    vals.iter().sum::<f64>() / vals.len().max(1) as f64
}

fn sample_std(vals: &[f64], mean: f64) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
    var.max(0.0).sqrt()
}

fn quantile(vals: &[f64], q: f64) -> f64 {
    let mut x = vals
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if x.is_empty() {
        return f64::NAN;
    }
    x.sort_by(|a, b| a.total_cmp(b));
    if x.len() == 1 {
        return x[0];
    }
    let pos = q.clamp(0.0, 1.0) * (x.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        x[lo]
    } else {
        x[lo] * (hi as f64 - pos) + x[hi] * (pos - lo as f64)
    }
}

fn outlier_mask(vals: &[f64], mad_k: f64) -> Vec<bool> {
    if vals.len() < 5 {
        return vec![false; vals.len()];
    }
    let med = quantile(vals, 0.50);
    let devs = vals.iter().map(|v| (v - med).abs()).collect::<Vec<_>>();
    let mad = quantile(&devs, 0.50);
    if mad <= 1.0e-12 {
        return vec![false; vals.len()];
    }
    vals.iter()
        .map(|v| 0.6745 * (v - med).abs() / mad > mad_k)
        .collect()
}

fn bimodal_gap_heuristic(vals: &[f64]) -> bool {
    let mut x = vals
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if x.len() < 5 {
        return false;
    }
    x.sort_by(|a, b| a.total_cmp(b));
    let mut best_i = 0usize;
    let mut best_gap = f64::NEG_INFINITY;
    for i in 0..x.len() - 1 {
        let gap = x[i + 1] - x[i];
        if gap > best_gap {
            best_gap = gap;
            best_i = i;
        }
    }
    let left = best_i + 1;
    let right = x.len() - left;
    if left < 2 || right < 2 {
        return false;
    }
    let med = quantile(&x, 0.50);
    let devs = x.iter().map(|v| (v - med).abs()).collect::<Vec<_>>();
    let robust_scale = quantile(&devs, 0.50) * 1.4826;
    robust_scale > 1.0e-12 && best_gap > 3.0 * robust_scale
}

fn rhat_proxy(mean: f64, std: f64, n_used: usize) -> f64 {
    if n_used <= 1 || !std.is_finite() {
        return f64::INFINITY;
    }
    1.0 + (std / mean.abs().max(1.0e-6)).min(99.0)
}

fn aggregate_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("residue_id", DataType::Int32, false),
        Field::new("feature_group", DataType::Utf8, false),
        Field::new("feature_name", DataType::Utf8, false),
        Field::new("mean", DataType::Float64, false),
        Field::new("std", DataType::Float64, false),
        Field::new("median", DataType::Float64, false),
        Field::new("q05", DataType::Float64, false),
        Field::new("q25", DataType::Float64, false),
        Field::new("q75", DataType::Float64, false),
        Field::new("q95", DataType::Float64, false),
        Field::new("rhat", DataType::Float64, false),
        Field::new("ess", DataType::Float64, false),
        Field::new("is_bimodal", DataType::Boolean, false),
        Field::new("n_replicas_used", DataType::UInt32, false),
        Field::new("n_outliers", DataType::UInt32, false),
    ]))
}

fn write_aggregate_parquet(rows: &[AggregateRow], path: &Path) -> Result<()> {
    let schema = aggregate_schema();
    let mut residue_id = Int32Builder::with_capacity(rows.len());
    let mut feature_group = StringBuilder::with_capacity(rows.len(), rows.len() * 8);
    let mut feature_name = StringBuilder::with_capacity(rows.len(), rows.len() * 24);
    let mut mean_b = Float64Builder::with_capacity(rows.len());
    let mut std_b = Float64Builder::with_capacity(rows.len());
    let mut median_b = Float64Builder::with_capacity(rows.len());
    let mut q05_b = Float64Builder::with_capacity(rows.len());
    let mut q25_b = Float64Builder::with_capacity(rows.len());
    let mut q75_b = Float64Builder::with_capacity(rows.len());
    let mut q95_b = Float64Builder::with_capacity(rows.len());
    let mut rhat_b = Float64Builder::with_capacity(rows.len());
    let mut ess_b = Float64Builder::with_capacity(rows.len());
    let mut bimodal_b = BooleanBuilder::with_capacity(rows.len());
    let mut nrep_b = UInt32Builder::with_capacity(rows.len());
    let mut outlier_b = UInt32Builder::with_capacity(rows.len());

    for row in rows {
        residue_id.append_value(row.residue_id);
        feature_group.append_value(row.feature_group);
        feature_name.append_value(row.feature_name);
        mean_b.append_value(row.mean);
        std_b.append_value(row.std);
        median_b.append_value(row.median);
        q05_b.append_value(row.q05);
        q25_b.append_value(row.q25);
        q75_b.append_value(row.q75);
        q95_b.append_value(row.q95);
        rhat_b.append_value(row.rhat);
        ess_b.append_value(row.ess);
        bimodal_b.append_value(row.is_bimodal);
        nrep_b.append_value(row.n_replicas_used);
        outlier_b.append_value(row.n_outliers);
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(residue_id.finish()) as ArrayRef,
            Arc::new(feature_group.finish()) as ArrayRef,
            Arc::new(feature_name.finish()) as ArrayRef,
            Arc::new(mean_b.finish()) as ArrayRef,
            Arc::new(std_b.finish()) as ArrayRef,
            Arc::new(median_b.finish()) as ArrayRef,
            Arc::new(q05_b.finish()) as ArrayRef,
            Arc::new(q25_b.finish()) as ArrayRef,
            Arc::new(q75_b.finish()) as ArrayRef,
            Arc::new(q95_b.finish()) as ArrayRef,
            Arc::new(rhat_b.finish()) as ArrayRef,
            Arc::new(ess_b.finish()) as ArrayRef,
            Arc::new(bimodal_b.finish()) as ArrayRef,
            Arc::new(nrep_b.finish()) as ArrayRef,
            Arc::new(outlier_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet_batch(schema, batch, path)
}

fn write_rhat_parquet(rows: &[AggregateRow], path: &Path) -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("residue_id", DataType::Int32, false),
        Field::new("feature_group", DataType::Utf8, false),
        Field::new("feature_name", DataType::Utf8, false),
        Field::new("rhat", DataType::Float64, false),
        Field::new("ess", DataType::Float64, false),
    ]));
    let mut residue_id = Int32Builder::with_capacity(rows.len());
    let mut feature_group = StringBuilder::with_capacity(rows.len(), rows.len() * 8);
    let mut feature_name = StringBuilder::with_capacity(rows.len(), rows.len() * 24);
    let mut rhat_b = Float64Builder::with_capacity(rows.len());
    let mut ess_b = Float64Builder::with_capacity(rows.len());
    for row in rows {
        residue_id.append_value(row.residue_id);
        feature_group.append_value(row.feature_group);
        feature_name.append_value(row.feature_name);
        rhat_b.append_value(row.rhat);
        ess_b.append_value(row.ess);
    }
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(residue_id.finish()) as ArrayRef,
            Arc::new(feature_group.finish()) as ArrayRef,
            Arc::new(feature_name.finish()) as ArrayRef,
            Arc::new(rhat_b.finish()) as ArrayRef,
            Arc::new(ess_b.finish()) as ArrayRef,
        ],
    )?;
    write_parquet_batch(schema, batch, path)
}

fn write_parquet_batch(schema: Arc<Schema>, batch: RecordBatch, path: &Path) -> Result<()> {
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .set_dictionary_enabled(true)
        .build();
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn build_consensus<'a>(
    target_id: &'a str,
    replicas: &[ReplicaFeatures],
    rows: &[AggregateRow],
    agg_path: &Path,
    mad_threshold: f64,
) -> Result<Consensus<'a>> {
    let rhats = rows
        .iter()
        .map(|r| r.rhat)
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    let bimodal_ids = rows
        .iter()
        .filter(|r| r.is_bimodal)
        .map(|r| r.residue_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    Ok(Consensus {
        computed_at: Utc::now().to_rfc3339(),
        computed_by: "prism-v004-ensemble",
        target_id,
        n_replicas: replicas.len(),
        n_residues: rows
            .iter()
            .map(|r| r.residue_id)
            .collect::<BTreeSet<_>>()
            .len(),
        feature_count: rows
            .iter()
            .map(|r| (r.feature_group, r.feature_name))
            .collect::<BTreeSet<_>>()
            .len(),
        per_residue_aggregates_relative_path: "ensemble_aggregates.parquet",
        per_residue_aggregates_sha256: sha256_file(agg_path)?,
        convergence_rhat_relative_path: "convergence_rhat.parquet",
        rhat_summary: RhatSummary {
            method: "single_draw_replica_dispersion_proxy",
            min: quantile(&rhats, 0.0),
            median: quantile(&rhats, 0.50),
            p95: quantile(&rhats, 0.95),
            max: quantile(&rhats, 1.0),
        },
        outlier_detection: OutlierSummary {
            method: "median_absolute_deviation",
            mad_threshold,
            n_outlier_replicas_max: rows.iter().map(|r| r.n_outliers).max().unwrap_or(0),
            outliers_detected: rows.iter().any(|r| r.n_outliers > 0),
        },
        bimodality: BimodalitySummary {
            method: "largest_gap_robust_heuristic",
            n_residues_bimodal: bimodal_ids.len(),
            bimodal_residue_ids: bimodal_ids,
        },
        replica_feature_parquets: replicas
            .iter()
            .map(|r| {
                Ok(ReplicaParquetSummary {
                    replica_id: r.replica_id,
                    run_seed: r.run_seed,
                    path: r.parquet_path.display().to_string(),
                    sha256: sha256_file(&r.parquet_path)?,
                })
            })
            .collect::<Result<Vec<_>>>()?,
    })
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut h = Sha256::new();
    std::io::copy(&mut file, &mut h).with_context(|| format!("hash {}", path.display()))?;
    Ok(format!("{:x}", h.finalize()))
}
