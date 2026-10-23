use std::{collections::BTreeMap, path::PathBuf};

use anyhow::Result;
use clap::Parser;
use serde_json::json;

mod dispatch_analysis_common;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    candidate_id: String,
    #[arg(long)]
    signal_grid: PathBuf,
    #[arg(long)]
    protocol_state_summary: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    bifurcate: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let grid = dispatch_analysis_common::read_signal_grid_summary(&args.signal_grid)?;
    let bifurcation_channels =
        dispatch_analysis_common::parse_bifurcation_channels(args.bifurcate.as_deref())?;
    let bifurcated_signal_scores =
        dispatch_analysis_common::bifurcated_signal_scores(&grid, &bifurcation_channels);
    let channel_weights: BTreeMap<String, f64> = [
        ("pocket".to_owned(), 1.0),
        ("lock".to_owned(), 0.65),
        ("pathway".to_owned(), 1.35),
    ]
    .into_iter()
    .collect();
    let bifurcation_scale = if bifurcated_signal_scores.is_empty() {
        1.0
    } else {
        bifurcated_signal_scores
            .iter()
            .map(|(channel, score)| score * channel_weights.get(channel).copied().unwrap_or(1.0))
            .sum::<f64>()
            / bifurcated_signal_scores.len() as f64
    };
    let bifurcated_hysteresis_score = grid.hysteresis_score * (1.0 + bifurcation_scale);
    let payload = json!({
        "mode": "hysteresis_analysis",
        "schema_version": "prism.gpu_dispatch.hysteresis_analysis.v1",
        "analysis_provenance": "rust_signal_grid_hysteresis",
        "candidate_id": args.candidate_id,
        "signal_grid": args.signal_grid,
        "protocol_state_summary": args.protocol_state_summary,
        "bifurcate": args.bifurcate,
        "bifurcation_channels": bifurcation_channels,
        "bifurcated_signal_scores": bifurcated_signal_scores,
        "bifurcated_hysteresis_channel_weights": channel_weights,
        "row_count": grid.row_count,
        "columns": grid.columns,
        "classification_counts": grid.classification_counts,
        "cold_mean": grid.cold_mean,
        "warm_mean": grid.warm_mean,
        "mean_abs_delta": grid.mean_abs_delta,
        "hysteresis_score": grid.hysteresis_score,
        "bifurcated_hysteresis_score": bifurcated_hysteresis_score,
        "reversibility_mean": grid.reversibility_mean,
        "coulombic_inefficiency_proxy": grid.hysteresis_score,
        "thermally_activated_count": grid.thermally_activated_count,
        "stable_occupied_count": grid.stable_occupied_count,
        "thermally_destabilized_count": grid.thermally_destabilized_count,
        "void_count": grid.void_count
    });
    dispatch_analysis_common::write_json_atomic(&args.output, &payload)?;
    println!(
        "hysteresis_analysis_complete output={}",
        args.output.display()
    );
    Ok(())
}
