use std::path::PathBuf;

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
    warp_jacobian: Option<PathBuf>,
    #[arg(long)]
    protocol_state_summary: Option<PathBuf>,
    #[arg(long)]
    phase_filter: Option<String>,
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
    let phase_weight = dispatch_analysis_common::phase_filter_weight(args.phase_filter.as_deref())?;
    let warp = args
        .warp_jacobian
        .as_ref()
        .filter(|path| path.is_file())
        .map(|path| dispatch_analysis_common::read_warp_summary(path))
        .transpose()?;
    let inferred_pathway_voxels = if grid.explicit_pathway_voxel_count > 0 {
        grid.explicit_pathway_voxel_count
    } else {
        grid.thermally_activated_count
    };
    let shear_positive_voxels = warp
        .as_ref()
        .map(|summary| summary.positive_metric_count)
        .unwrap_or(0);
    let pathway_enabled = bifurcation_channels
        .iter()
        .any(|channel| channel.as_str() == "pathway");
    let phase_filtered_pathway_voxels = (inferred_pathway_voxels as f64 * phase_weight).round();
    let pathway_control_score = if grid.row_count == 0 {
        0.0
    } else if !pathway_enabled {
        0.0
    } else {
        let kinetic_fraction = phase_filtered_pathway_voxels / grid.row_count as f64;
        let shear_fraction = shear_positive_voxels as f64 / grid.row_count as f64;
        kinetic_fraction * (1.0 + shear_fraction)
    };
    let payload = json!({
        "mode": "pathway_analysis",
        "schema_version": "prism.gpu_dispatch.pathway_analysis.v1",
        "analysis_provenance": "rust_signal_grid_pathway_warp_analysis",
        "candidate_id": args.candidate_id,
        "signal_grid": args.signal_grid,
        "warp_jacobian": args.warp_jacobian,
        "protocol_state_summary": args.protocol_state_summary,
        "phase_filter": args.phase_filter,
        "phase_filter_weight": phase_weight,
        "bifurcate": args.bifurcate,
        "bifurcation_channels": bifurcation_channels,
        "bifurcated_signal_scores": bifurcated_signal_scores,
        "row_count": grid.row_count,
        "columns": grid.columns,
        "classification_counts": grid.classification_counts,
        "explicit_pathway_voxel_count": grid.explicit_pathway_voxel_count,
        "inferred_pathway_voxel_count": inferred_pathway_voxels,
        "phase_filtered_pathway_voxel_count": phase_filtered_pathway_voxels,
        "thermally_activated_count": grid.thermally_activated_count,
        "pathway_control_score": pathway_control_score,
        "warp_summary": warp
    });
    dispatch_analysis_common::write_json_atomic(&args.output, &payload)?;
    println!("pathway_analysis_complete output={}", args.output.display());
    Ok(())
}
