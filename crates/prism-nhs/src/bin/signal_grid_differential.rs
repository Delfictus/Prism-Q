use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use arrow_schema::DataType;
use clap::Parser;
use prism_nhs::io::provenance::{
    f64s, filter_streams, i32s, load_protocol_phase_map, read_signal_hit_grid, record_batch,
    schema, strings, u64s, HitCount, HitMean, ProvenanceOptions, ProvenanceParquetWriter,
    StreamPath, VoxelIdx, CAMPAIGN_ID, DEFAULT_OUTPUT_DIR,
};
use rayon::prelude::*;
use serde_json::json;

#[derive(Parser, Debug)]
struct Args {
    /// Raw campaign root (required; there is no compiled-in host default).
    #[arg(long)]
    raw_root: PathBuf,
    #[arg(long, default_value = DEFAULT_OUTPUT_DIR)]
    out_dir: PathBuf,
    #[arg(
        long,
        default_value = "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/protocol_state_summary.parquet"
    )]
    protocol_state_summary: PathBuf,
    #[arg(long, default_value_t = 0.0)]
    hit_threshold: f64,
    #[arg(long)]
    condition_id: Option<String>,
    #[arg(long)]
    replica_id: Option<u16>,
    #[arg(long)]
    max_streams: Option<usize>,
    #[arg(long)]
    frame_scope: Option<String>,
    #[arg(long)]
    bifurcate: Option<String>,
}

#[derive(Debug, Clone)]
struct GridAccumulator {
    grid_dim: u64,
    cold_sum: Vec<f64>,
    warm_sum: Vec<f64>,
    cold_streams: u64,
    warm_streams: u64,
}

impl GridAccumulator {
    fn new(grid_dim: u64, voxel_count: usize) -> Self {
        Self {
            grid_dim,
            cold_sum: vec![0.0; voxel_count],
            warm_sum: vec![0.0; voxel_count],
            cold_streams: 0,
            warm_streams: 0,
        }
    }

    fn add_grid(&mut self, hits: &[HitCount], thermal_class: &str) {
        if thermal_class == "Warm_Phase" {
            self.warm_streams += 1;
            for (idx, hit) in hits.iter().enumerate() {
                self.warm_sum[idx] += f64::from(hit.0);
            }
        } else {
            self.cold_streams += 1;
            for (idx, hit) in hits.iter().enumerate() {
                self.cold_sum[idx] += f64::from(hit.0);
            }
        }
    }

    fn merge(&mut self, other: GridAccumulator) -> Result<()> {
        if self.grid_dim != other.grid_dim || self.cold_sum.len() != other.cold_sum.len() {
            anyhow::bail!("signal grid dimension mismatch during reduce");
        }
        self.cold_streams += other.cold_streams;
        self.warm_streams += other.warm_streams;
        for idx in 0..self.cold_sum.len() {
            self.cold_sum[idx] += other.cold_sum[idx];
            self.warm_sum[idx] += other.warm_sum[idx];
        }
        Ok(())
    }
}

fn classify(cold: HitMean, warm: HitMean, threshold: HitMean) -> &'static str {
    let cold_high = cold.0 > threshold.0;
    let warm_high = warm.0 > threshold.0;
    match (cold_high, warm_high) {
        (true, true) => "stable_occupied",
        (true, false) => "thermally_destabilized",
        (false, true) => "thermally_activated",
        (false, false) => "void",
    }
}

fn process_file(item: &StreamPath, thermal_class: &str) -> Result<(String, GridAccumulator)> {
    let (header, hits) = read_signal_hit_grid(&item.path)?;
    let mut acc = GridAccumulator::new(header.grid_dim, header.voxel_count as usize);
    acc.add_grid(&hits, thermal_class);
    Ok((item.condition_id.clone(), acc))
}

fn merge_maps(
    mut left: HashMap<String, GridAccumulator>,
    right: HashMap<String, GridAccumulator>,
) -> HashMap<String, GridAccumulator> {
    for (condition, acc) in right {
        if let Some(existing) = left.get_mut(&condition) {
            existing.merge(acc).expect("grid accumulator merge");
        } else {
            left.insert(condition, acc);
        }
    }
    left
}

fn frame_scope_accepts_phase(
    frame_scope: Option<&str>,
    phase: &str,
    thermal_class: &str,
) -> Result<bool> {
    let Some(scope) = frame_scope.map(str::trim).filter(|scope| !scope.is_empty()) else {
        return Ok(true);
    };
    let phase_lower = phase.to_ascii_lowercase();
    match scope {
        "all" => Ok(true),
        "equilibrated" => Ok(
            phase_lower.contains("hold")
                || phase_lower.contains("return")
                || phase_lower.contains("equilibrated"),
        ),
        "ramp" | "ramp_phase" => Ok(phase_lower.contains("ramp")),
        "cold" | "cold_phase" => Ok(thermal_class != "Warm_Phase"),
        "warm" | "warm_phase" => Ok(thermal_class == "Warm_Phase"),
        other => anyhow::bail!(
            "unsupported signal grid frame_scope {other}; expected all, equilibrated, ramp, cold, or warm"
        ),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let all = prism_nhs::io::provenance::discover_stream_files(&args.raw_root, "signal_grid.bin")?;
    let selected = filter_streams(
        &all,
        args.condition_id.as_deref(),
        args.replica_id,
        args.max_streams,
    );
    let phase_map = load_protocol_phase_map(&args.protocol_state_summary)?;
    let accumulators: HashMap<String, GridAccumulator> = selected
        .par_iter()
        .try_fold(
            HashMap::<String, GridAccumulator>::new,
            |mut local, item| -> Result<_> {
                let phase = phase_map
                    .get(&(
                        item.condition_id.clone(),
                        item.replica_id.0,
                        item.stream_id.0,
                    ))
                    .with_context(|| {
                        format!("missing protocol phase for {}", item.path.display())
                    })?;
                if !frame_scope_accepts_phase(
                    args.frame_scope.as_deref(),
                    &phase.thermal_phase,
                    &phase.thermal_class,
                )? {
                    return Ok(local);
                }
                let (condition, acc) = process_file(item, &phase.thermal_class)?;
                if let Some(existing) = local.get_mut(&condition) {
                    existing.merge(acc)?;
                } else {
                    local.insert(condition, acc);
                }
                Ok(local)
            },
        )
        .try_reduce(
            HashMap::new,
            |left, right| -> Result<HashMap<String, GridAccumulator>> {
                Ok(merge_maps(left, right))
            },
        )?;
    if accumulators.is_empty() {
        anyhow::bail!(
            "frame_scope {:?} selected no signal-grid streams",
            args.frame_scope
        );
    }
    let threshold = HitMean(args.hit_threshold);
    let out = args.out_dir.join("signal_grid_variance_channel.parquet");
    let schema_ref = schema(vec![
        ("campaign_id", DataType::Utf8, false),
        ("condition_id", DataType::Utf8, false),
        ("voxel_idx", DataType::UInt64, false),
        ("x_idx", DataType::Int32, false),
        ("y_idx", DataType::Int32, false),
        ("z_idx", DataType::Int32, false),
        ("hit_count_cold_mean", DataType::Float64, false),
        ("hit_count_warm_mean", DataType::Float64, false),
        ("variance_class", DataType::Utf8, false),
        ("cold_stream_count", DataType::UInt64, false),
        ("warm_stream_count", DataType::UInt64, false),
    ]);
    let mut input_paths = selected
        .iter()
        .map(|item| item.path.clone())
        .collect::<Vec<_>>();
    input_paths.push(args.protocol_state_summary.clone());
    let mut writer = ProvenanceParquetWriter::try_new(
        &out,
        schema_ref.clone(),
        ProvenanceOptions {
            module: "phase4_signal_grid_differential".to_string(),
            schema_version: "prism.signal_grid_variance_channel.v1".to_string(),
            producer: "crates/prism-nhs/src/bin/signal_grid_differential.rs".to_string(),
            input_paths,
            source_parquets: vec![args.protocol_state_summary],
            partition_keys: vec!["condition_id".to_string(), "voxel_idx".to_string()],
            parameters: json!({
                "hit_threshold": args.hit_threshold,
                "classification": "cold/warm mean hit count thresholded into stable_occupied, thermally_destabilized, thermally_activated, void",
                "selected_input_count": selected.len(),
                "voxel_output_row_group_size": 65536,
                "max_streams": args.max_streams,
                "frame_scope": args.frame_scope,
                "bifurcate": args.bifurcate
            }),
        },
    )?;
    for (condition, acc) in accumulators {
        let dim = acc.grid_dim as usize;
        for start in (0..acc.cold_sum.len()).step_by(65_536) {
            let end = (start + 65_536).min(acc.cold_sum.len());
            let mut campaign_id = Vec::with_capacity(end - start);
            let mut condition_id = Vec::with_capacity(end - start);
            let mut voxel_idx = Vec::with_capacity(end - start);
            let mut x_idx = Vec::with_capacity(end - start);
            let mut y_idx = Vec::with_capacity(end - start);
            let mut z_idx = Vec::with_capacity(end - start);
            let mut cold_mean = Vec::with_capacity(end - start);
            let mut warm_mean = Vec::with_capacity(end - start);
            let mut classification = Vec::with_capacity(end - start);
            let mut cold_streams = Vec::with_capacity(end - start);
            let mut warm_streams = Vec::with_capacity(end - start);
            for idx in start..end {
                let voxel = VoxelIdx(idx as u32);
                let cold = HitMean(if acc.cold_streams == 0 {
                    0.0
                } else {
                    acc.cold_sum[idx] / acc.cold_streams as f64
                });
                let warm = HitMean(if acc.warm_streams == 0 {
                    0.0
                } else {
                    acc.warm_sum[idx] / acc.warm_streams as f64
                });
                campaign_id.push(CAMPAIGN_ID.to_string());
                condition_id.push(condition.clone());
                voxel_idx.push(u64::from(voxel.0));
                x_idx.push((idx % dim) as i32);
                y_idx.push(((idx / dim) % dim) as i32);
                z_idx.push((idx / (dim * dim)) as i32);
                cold_mean.push(cold.0);
                warm_mean.push(warm.0);
                classification.push(classify(cold, warm, threshold).to_string());
                cold_streams.push(acc.cold_streams);
                warm_streams.push(acc.warm_streams);
            }
            let batch = record_batch(
                schema_ref.clone(),
                vec![
                    strings(campaign_id),
                    strings(condition_id),
                    u64s(voxel_idx),
                    i32s(x_idx),
                    i32s(y_idx),
                    i32s(z_idx),
                    f64s(cold_mean),
                    f64s(warm_mean),
                    strings(classification),
                    u64s(cold_streams),
                    u64s(warm_streams),
                ],
            )?;
            writer.write(&batch)?;
        }
    }
    writer.close()?;
    println!("WROTE {}", out.display());
    Ok(())
}
