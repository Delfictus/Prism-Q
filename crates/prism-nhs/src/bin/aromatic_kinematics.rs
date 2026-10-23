use std::{collections::BTreeMap, fs::File, io::Read, path::PathBuf};

use anyhow::{Context, Result};
use arrow_schema::DataType;
use clap::Parser;
use prism_nhs::io::provenance::{
    f32s, f64s, filter_streams, i32s, load_protocol_phase_map, record_batch, schema, strings,
    ProvenanceOptions, RingIdx, StreamPath, Vec3, CAMPAIGN_ID, DEFAULT_OUTPUT_DIR,
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
    #[arg(long)]
    condition_id: Option<String>,
    #[arg(long)]
    replica_id: Option<u16>,
    #[arg(long)]
    max_streams: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct PhaseStats {
    count: u32,
    sum: Vec3,
    sum_sq_norm: f64,
}

impl Default for PhaseStats {
    fn default() -> Self {
        Self {
            count: 0,
            sum: Vec3::new(0.0, 0.0, 0.0),
            sum_sq_norm: 0.0,
        }
    }
}

impl PhaseStats {
    fn add(&mut self, value: Vec3) {
        self.count += 1;
        self.sum = Vec3::new(
            self.sum.x + value.x,
            self.sum.y + value.y,
            self.sum.z + value.z,
        );
        self.sum_sq_norm += f64::from(value.x * value.x + value.y * value.y + value.z * value.z);
    }

    fn merge(&mut self, other: PhaseStats) {
        self.count += other.count;
        self.sum = Vec3::new(
            self.sum.x + other.sum.x,
            self.sum.y + other.sum.y,
            self.sum.z + other.sum.z,
        );
        self.sum_sq_norm += other.sum_sq_norm;
    }

    fn mean(&self) -> Vec3 {
        let n = self.count as f32;
        Vec3::new(self.sum.x / n, self.sum.y / n, self.sum.z / n)
    }

    fn variance_around_mean(&self, mean: Vec3) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let mean_norm = f64::from(mean.x * mean.x + mean.y * mean.y + mean.z * mean.z);
        (self.sum_sq_norm / f64::from(self.count) - mean_norm).max(0.0)
    }
}

#[derive(Debug, Clone, Default)]
struct RingStats {
    cold: PhaseStats,
    warm: PhaseStats,
}

#[derive(Debug, Clone)]
struct AromaticTensorRow {
    condition_id: String,
    ring_idx: RingIdx,
    cold_n: u32,
    warm_n: u32,
    cold_mean: Vec3,
    warm_mean: Vec3,
    displacement: Vec3,
    centroid_displacement_std: f64,
}

fn add_centroid_file(
    grouped: &mut BTreeMap<(String, RingIdx), RingStats>,
    item: &StreamPath,
    thermal_class: &str,
) -> Result<()> {
    let len = item.path.metadata()?.len();
    if len % 12 != 0 {
        anyhow::bail!(
            "{} aromatic centroid file is not xyz f32 triples",
            item.path.display()
        );
    }
    let mut file =
        File::open(&item.path).with_context(|| format!("open {}", item.path.display()))?;
    let mut triple = [0u8; 12];
    for idx in 0..(len / 12) {
        file.read_exact(&mut triple)?;
        let centroid = Vec3::new(
            f32::from_le_bytes(triple[0..4].try_into()?),
            f32::from_le_bytes(triple[4..8].try_into()?),
            f32::from_le_bytes(triple[8..12].try_into()?),
        );
        let entry = grouped
            .entry((item.condition_id.clone(), RingIdx(idx as u32)))
            .or_default();
        if thermal_class == "Warm_Phase" {
            entry.warm.add(centroid);
        } else {
            entry.cold.add(centroid);
        }
    }
    Ok(())
}

fn merge_grouped(
    mut left: BTreeMap<(String, RingIdx), RingStats>,
    right: BTreeMap<(String, RingIdx), RingStats>,
) -> BTreeMap<(String, RingIdx), RingStats> {
    for (key, stats) in right {
        let entry = left.entry(key).or_default();
        entry.cold.merge(stats.cold);
        entry.warm.merge(stats.warm);
    }
    left
}

fn displacement_std(cold: PhaseStats, warm: PhaseStats, cold_mean: Vec3, warm_mean: Vec3) -> f64 {
    let displacement = Vec3::new(
        warm_mean.x - cold_mean.x,
        warm_mean.y - cold_mean.y,
        warm_mean.z - cold_mean.z,
    );
    let drift = f64::from(
        displacement.x * displacement.x
            + displacement.y * displacement.y
            + displacement.z * displacement.z,
    );
    let total = f64::from(cold.count + warm.count);
    let pooled_variance = if total == 0.0 {
        0.0
    } else {
        (cold.variance_around_mean(cold_mean) * f64::from(cold.count)
            + warm.variance_around_mean(warm_mean) * f64::from(warm.count))
            / total
    };
    (pooled_variance + drift).sqrt()
}

fn tensor_rows(grouped: BTreeMap<(String, RingIdx), RingStats>) -> Vec<AromaticTensorRow> {
    grouped
        .into_iter()
        .filter_map(|((condition_id, ring_idx), stats)| {
            if stats.cold.count == 0 || stats.warm.count == 0 {
                return None;
            }
            let cold_mean = stats.cold.mean();
            let warm_mean = stats.warm.mean();
            let displacement = Vec3::new(
                warm_mean.x - cold_mean.x,
                warm_mean.y - cold_mean.y,
                warm_mean.z - cold_mean.z,
            );
            Some(AromaticTensorRow {
                condition_id,
                ring_idx,
                cold_n: stats.cold.count,
                warm_n: stats.warm.count,
                cold_mean,
                warm_mean,
                displacement,
                centroid_displacement_std: displacement_std(
                    stats.cold, stats.warm, cold_mean, warm_mean,
                ),
            })
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let all = prism_nhs::io::provenance::discover_stream_files(
        &args.raw_root,
        "aromatic_centroids_final.bin",
    )?;
    let selected = filter_streams(
        &all,
        args.condition_id.as_deref(),
        args.replica_id,
        args.max_streams,
    );
    let phase_map = load_protocol_phase_map(&args.protocol_state_summary)?;
    let grouped = selected
        .par_iter()
        .try_fold(BTreeMap::new, |mut local, item| -> Result<_> {
            let phase = phase_map
                .get(&(
                    item.condition_id.clone(),
                    item.replica_id.0,
                    item.stream_id.0,
                ))
                .with_context(|| format!("missing protocol phase for {}", item.path.display()))?;
            add_centroid_file(&mut local, item, &phase.thermal_class)?;
            Ok(local)
        })
        .try_reduce(BTreeMap::new, |left, right| -> Result<_> {
            Ok(merge_grouped(left, right))
        })?;
    let rows = tensor_rows(grouped);
    let out = args.out_dir.join("aromatic_reorganization_tensor.parquet");
    let schema = schema(vec![
        ("campaign_id", DataType::Utf8, false),
        ("condition_id", DataType::Utf8, false),
        ("ring_idx", DataType::Int32, false),
        ("cold_stream_count", DataType::Int32, false),
        ("warm_stream_count", DataType::Int32, false),
        ("cold_mean_x", DataType::Float32, false),
        ("cold_mean_y", DataType::Float32, false),
        ("cold_mean_z", DataType::Float32, false),
        ("warm_mean_x", DataType::Float32, false),
        ("warm_mean_y", DataType::Float32, false),
        ("warm_mean_z", DataType::Float32, false),
        ("centroid_displacement_x", DataType::Float32, false),
        ("centroid_displacement_y", DataType::Float32, false),
        ("centroid_displacement_z", DataType::Float32, false),
        ("centroid_displacement_std", DataType::Float64, false),
    ]);
    let batch = record_batch(
        schema.clone(),
        vec![
            strings(vec![CAMPAIGN_ID.to_string(); rows.len()]),
            strings(rows.iter().map(|row| row.condition_id.clone()).collect()),
            i32s(rows.iter().map(|row| row.ring_idx.0 as i32).collect()),
            i32s(rows.iter().map(|row| row.cold_n as i32).collect()),
            i32s(rows.iter().map(|row| row.warm_n as i32).collect()),
            f32s(rows.iter().map(|row| row.cold_mean.x).collect()),
            f32s(rows.iter().map(|row| row.cold_mean.y).collect()),
            f32s(rows.iter().map(|row| row.cold_mean.z).collect()),
            f32s(rows.iter().map(|row| row.warm_mean.x).collect()),
            f32s(rows.iter().map(|row| row.warm_mean.y).collect()),
            f32s(rows.iter().map(|row| row.warm_mean.z).collect()),
            f32s(rows.iter().map(|row| row.displacement.x).collect()),
            f32s(rows.iter().map(|row| row.displacement.y).collect()),
            f32s(rows.iter().map(|row| row.displacement.z).collect()),
            f64s(
                rows.iter()
                    .map(|row| row.centroid_displacement_std)
                    .collect(),
            ),
        ],
    )?;
    let mut input_paths = selected
        .iter()
        .map(|item| item.path.clone())
        .collect::<Vec<_>>();
    input_paths.push(args.protocol_state_summary.clone());
    prism_nhs::io::provenance::write_single_batch_with_provenance(
        &out,
        schema,
        batch,
        ProvenanceOptions {
            module: "phase4_aromatic_kinematics".to_string(),
            schema_version: "prism.aromatic_reorganization_tensor.v1".to_string(),
            producer: "crates/prism-nhs/src/bin/aromatic_kinematics.rs".to_string(),
            input_paths,
            source_parquets: vec![args.protocol_state_summary],
            partition_keys: vec!["condition_id".to_string(), "ring_idx".to_string()],
            parameters: json!({
                "math": "warm_centroid_mean - cold_centroid_mean per aromatic ring; std is sqrt(pooled within-phase spatial variance + displacement magnitude squared)",
                "thermal_class_source": "protocol_state_summary.parquet",
                "selected_input_count": selected.len(),
                "max_streams": args.max_streams
            }),
        },
    )?;
    println!("WROTE {}", out.display());
    Ok(())
}
