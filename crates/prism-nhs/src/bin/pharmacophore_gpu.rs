//! PRISM-4D GPU Pharmacophore Hotspot Map Generator
//! Usage: pharmacophore_gpu <spike_events.json> <output_dir> [receptor.pdb]

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "pharmacophore_gpu",
    about = "GPU-accelerated pharmacophore hotspot extraction"
)]
struct Args {
    /// Path to spike_events.json
    spike_events: PathBuf,
    /// Output directory for .dx grids and PyMOL script
    output_dir: PathBuf,
    /// Receptor PDB for PyMOL script (optional)
    #[arg(short, long, default_value = "receptor.pdb")]
    receptor: String,
    /// Grid spacing in Angstroms
    #[arg(long, default_value_t = 1.0)]
    spacing: f32,
    /// Gaussian sigma in Angstroms
    #[arg(long, default_value_t = 1.5)]
    sigma: f32,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    log::info!("PRISM-4D GPU Pharmacophore Extraction");
    log::info!("  Input:  {}", args.spike_events.display());
    log::info!("  Output: {}", args.output_dir.display());

    // Load spike events from JSON
    let (spikes, centroid, site_id) =
        prism_nhs::pharmacophore_gpu::SpikeData::from_json(&args.spike_events)?;
    log::info!(
        "  Loaded {} spikes, site {} @ [{:.1}, {:.1}, {:.1}]",
        spikes.len(),
        site_id,
        centroid[0],
        centroid[1],
        centroid[2]
    );

    // Init CUDA
    let context = cudarc::driver::CudaContext::new(0)?;
    let stream = context.default_stream();

    // Build engine
    let engine = prism_nhs::pharmacophore_gpu::PharmacophoreGpu::new(
        Arc::clone(&context),
        Arc::clone(&stream),
    )?;

    // Run full pipeline
    prism_nhs::pharmacophore_gpu::extract_pharmacophore_gpu(
        &engine,
        &spikes,
        centroid,
        site_id,
        &args.output_dir,
        &args.receptor,
    )?;

    log::info!("Done. View in PyMOL:");
    log::info!("  cd {}", args.output_dir.display());
    log::info!("  pymol @site{}_visualize.pml", site_id);
    Ok(())
}
