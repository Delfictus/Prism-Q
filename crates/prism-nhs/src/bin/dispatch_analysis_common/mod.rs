use std::{collections::BTreeMap, fs::File, path::Path};

use anyhow::{Context, Result};
use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray, UInt32Array, UInt64Array,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SignalGridSummary {
    pub row_count: u64,
    pub columns: Vec<String>,
    pub classification_counts: BTreeMap<String, u64>,
    pub cold_mean: f64,
    pub warm_mean: f64,
    pub mean_abs_delta: f64,
    pub hysteresis_score: f64,
    pub reversibility_mean: f64,
    pub thermally_activated_count: u64,
    pub stable_occupied_count: u64,
    pub thermally_destabilized_count: u64,
    pub void_count: u64,
    pub explicit_pathway_voxel_count: u64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize)]
pub struct WarpSummary {
    pub row_count: u64,
    pub columns: Vec<String>,
    pub metric_column: String,
    pub metric_mean: f64,
    pub metric_max: f64,
    pub positive_metric_count: u64,
}

pub fn read_signal_grid_summary(path: &Path) -> Result<SignalGridSummary> {
    let file = File::open(path).with_context(|| format!("open signal grid {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("read parquet metadata {}", path.display()))?
        .build()?;
    let mut out = SignalGridSummary {
        row_count: 0,
        columns: Vec::new(),
        classification_counts: BTreeMap::new(),
        cold_mean: 0.0,
        warm_mean: 0.0,
        mean_abs_delta: 0.0,
        hysteresis_score: 0.0,
        reversibility_mean: 0.0,
        thermally_activated_count: 0,
        stable_occupied_count: 0,
        thermally_destabilized_count: 0,
        void_count: 0,
        explicit_pathway_voxel_count: 0,
    };
    let mut cold_sum = 0.0;
    let mut warm_sum = 0.0;
    let mut abs_delta_sum = 0.0;
    let mut reversibility_sum = 0.0;

    for batch in reader {
        let batch = batch?;
        if out.columns.is_empty() {
            out.columns = batch
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect();
        }
        for row in 0..batch.num_rows() {
            let cold = optional_f64(
                &batch,
                &["hit_count_cold_mean", "cold_mean", "cold_normalized"],
                row,
            )
            .unwrap_or(0.0);
            let warm = optional_f64(
                &batch,
                &["hit_count_warm_mean", "warm_mean", "warm_normalized"],
                row,
            )
            .unwrap_or(0.0);
            let delta =
                optional_f64(&batch, &["hit_count_delta", "delta"], row).unwrap_or(warm - cold);
            let class = optional_string(
                &batch,
                &[
                    "variance_classification",
                    "variance_class",
                    "consensus_raw_variance_class",
                ],
                row,
            )
            .unwrap_or_else(|| classify_from_signal(cold, warm).to_owned());
            let class = normalize_class(&class);
            *out.classification_counts.entry(class.clone()).or_insert(0) += 1;
            match class.as_str() {
                "thermally_activated" => out.thermally_activated_count += 1,
                "stable_occupied" => out.stable_occupied_count += 1,
                "thermally_destabilized" => out.thermally_destabilized_count += 1,
                _ => out.void_count += 1,
            }
            if optional_bool(
                &batch,
                &[
                    "on_activation_pathway",
                    "activation_pathway",
                    "pathway_voxel",
                ],
                row,
            )
            .unwrap_or(false)
            {
                out.explicit_pathway_voxel_count += 1;
            }
            cold_sum += cold;
            warm_sum += warm;
            abs_delta_sum += delta.abs();
            let denom = cold.abs().max(warm.abs()).max(1.0);
            reversibility_sum += (1.0 - (delta.abs() / denom)).clamp(0.0, 1.0);
            out.row_count += 1;
        }
    }

    if out.row_count == 0 {
        anyhow::bail!("signal grid parquet contains zero rows: {}", path.display());
    }
    let denom = out.row_count as f64;
    out.cold_mean = cold_sum / denom;
    out.warm_mean = warm_sum / denom;
    out.mean_abs_delta = abs_delta_sum / denom;
    out.reversibility_mean = reversibility_sum / denom;
    out.hysteresis_score = 1.0 - out.reversibility_mean;
    Ok(out)
}

#[allow(dead_code)]
pub fn read_warp_summary(path: &Path) -> Result<WarpSummary> {
    let file = File::open(path).with_context(|| format!("open warp field {}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("read parquet metadata {}", path.display()))?
        .build()?;
    let mut out = WarpSummary {
        row_count: 0,
        columns: Vec::new(),
        metric_column: String::new(),
        metric_mean: 0.0,
        metric_max: 0.0,
        positive_metric_count: 0,
    };
    let mut metric_sum = 0.0;
    for batch in reader {
        let batch = batch?;
        if out.columns.is_empty() {
            out.columns = batch
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().clone())
                .collect();
            out.metric_column = first_existing_column(
                &batch,
                &[
                    "shear_stress",
                    "force_concentration_gradient",
                    "asc_directional_shear",
                ],
            )
            .unwrap_or("missing_metric")
            .to_owned();
        }
        for row in 0..batch.num_rows() {
            let metric = optional_f64(&batch, &[out.metric_column.as_str()], row).unwrap_or(0.0);
            metric_sum += metric;
            out.metric_max = out.metric_max.max(metric);
            if metric > 0.0 {
                out.positive_metric_count += 1;
            }
            out.row_count += 1;
        }
    }
    if out.row_count > 0 {
        out.metric_mean = metric_sum / out.row_count as f64;
    }
    Ok(out)
}

pub fn write_json_atomic(path: &Path, payload: &serde_json::Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }
    let tmp = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|value| value.to_str())
            .unwrap_or("json")
    ));
    std::fs::write(&tmp, serde_json::to_vec_pretty(payload)?)
        .with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    Ok(())
}

pub fn parse_bifurcation_channels(raw: Option<&str>) -> Result<Vec<String>> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(vec![
            "pocket".to_owned(),
            "lock".to_owned(),
            "pathway".to_owned(),
        ]);
    };
    let mut channels = Vec::new();
    for item in raw.split(',') {
        let channel = item.trim().to_ascii_lowercase();
        match channel.as_str() {
            "pocket" | "lock" | "pathway" => {
                if !channels.contains(&channel) {
                    channels.push(channel);
                }
            }
            other => anyhow::bail!(
                "unsupported bifurcation channel {other}; expected pocket, lock, or pathway"
            ),
        }
    }
    if channels.is_empty() {
        anyhow::bail!("bifurcation channel list is empty");
    }
    Ok(channels)
}

#[allow(dead_code)]
pub fn phase_filter_weight(raw: Option<&str>) -> Result<f64> {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(1.0);
    };
    let frames = raw
        .split(',')
        .map(|item| {
            let frame = item
                .trim()
                .parse::<f64>()
                .with_context(|| format!("parse phase-filter frame {item:?}"))?;
            if !frame.is_finite() || frame < 0.0 {
                anyhow::bail!("phase-filter frames must be finite non-negative values");
            }
            Ok(frame)
        })
        .collect::<Result<Vec<_>>>()?;
    if frames.is_empty() {
        anyhow::bail!("phase-filter frame list is empty");
    }
    let mean = frames.iter().sum::<f64>() / frames.len() as f64;
    Ok((mean / (mean + 10_000.0)).clamp(0.05, 1.0))
}

pub fn bifurcated_signal_scores(
    summary: &SignalGridSummary,
    channels: &[String],
) -> BTreeMap<String, f64> {
    let denom = summary.row_count.max(1) as f64;
    channels
        .iter()
        .map(|channel| {
            let score = match channel.as_str() {
                "pocket" => summary.thermally_activated_count as f64 / denom,
                "lock" => summary.stable_occupied_count as f64 / denom,
                "pathway" => {
                    let count = if summary.explicit_pathway_voxel_count > 0 {
                        summary.explicit_pathway_voxel_count
                    } else {
                        summary.thermally_activated_count
                    };
                    count as f64 / denom
                }
                _ => 0.0,
            };
            (channel.clone(), score)
        })
        .collect()
}

fn first_existing_column<'a>(batch: &RecordBatch, names: &'a [&'a str]) -> Option<&'a str> {
    names
        .iter()
        .copied()
        .find(|name| batch.schema().index_of(name).is_ok())
}

fn optional_f64(batch: &RecordBatch, names: &[&str], row: usize) -> Option<f64> {
    let name = first_existing_column(batch, names)?;
    let array = batch.column(batch.schema().index_of(name).ok()?);
    if array.is_null(row) {
        return None;
    }
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        Some(values.value(row))
    } else if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        Some(f64::from(values.value(row)))
    } else if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        Some(values.value(row) as f64)
    } else if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        Some(f64::from(values.value(row)))
    } else if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        Some(values.value(row) as f64)
    } else {
        array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|values| f64::from(values.value(row)))
    }
}

fn optional_string(batch: &RecordBatch, names: &[&str], row: usize) -> Option<String> {
    let name = first_existing_column(batch, names)?;
    let array = batch.column(batch.schema().index_of(name).ok()?);
    if array.is_null(row) {
        return None;
    }
    if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
        Some(values.value(row).to_owned())
    } else {
        array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|values| values.value(row).to_owned())
    }
}

fn optional_bool(batch: &RecordBatch, names: &[&str], row: usize) -> Option<bool> {
    let name = first_existing_column(batch, names)?;
    let array = batch.column(batch.schema().index_of(name).ok()?);
    if array.is_null(row) {
        return None;
    }
    array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .map(|values| values.value(row))
}

fn normalize_class(raw: &str) -> String {
    let lower = raw.trim().to_ascii_lowercase();
    match lower.as_str() {
        "stable_occupied" | "stableoccupied" => "stable_occupied".to_owned(),
        "thermally_destabilized" | "thermallydestabilized" => "thermally_destabilized".to_owned(),
        "thermally_activated" | "thermallyactivated" => "thermally_activated".to_owned(),
        "thermally_released" | "thermallyreleased" => "thermally_released".to_owned(),
        _ => "void".to_owned(),
    }
}

fn classify_from_signal(cold: f64, warm: f64) -> &'static str {
    match (cold > 0.0, warm > 0.0) {
        (true, true) => "stable_occupied",
        (true, false) => "thermally_destabilized",
        (false, true) => "thermally_activated",
        (false, false) => "void",
    }
}
