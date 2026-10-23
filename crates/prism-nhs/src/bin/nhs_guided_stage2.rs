//! NHS Guided Stage 2 Runner
//!
//! Uses Stage 1 hotspots to guide targeted UV probing in Stage 2.
//!
//! ## Two-Stage Pipeline
//! 1. Stage 1 (aggressive cryo-UV) → generates hotspot map
//! 2. Stage 2 (this binary) → uses hotspots for targeted 10ns production
//!
//! ## Usage
//! ```bash
//! nhs_guided_stage2 \
//!   --topology /tmp/ensemble_prep/4B7Q_apo_topology.json \
//!   --stage1-spikes /tmp/ensemble_results/stage1/4B7Q_apo_spikes.csv \
//!   --output /tmp/ensemble_results/stage2/4B7Q_apo \
//!   --steps 5000000  # 10ns
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_nhs::{
    EnsembleSnapshot, NhsAmberFusedEngine, PrismPrepTopology, TemperatureProtocol, UvProbeConfig,
};
use std::sync::Arc;

#[derive(Parser, Debug)]
#[command(name = "nhs_guided_stage2")]
#[command(about = "Stage 2 guided production run using Stage 1 hotspots")]
struct Args {
    /// Input topology JSON
    #[arg(short, long)]
    topology: PathBuf,

    /// Stage 1 spike CSV file (for hotspot extraction)
    #[arg(long)]
    stage1_spikes: PathBuf,

    /// Output prefix for results
    #[arg(short, long)]
    output: PathBuf,

    /// Total simulation steps (default 5M = 10ns)
    #[arg(long, default_value = "5000000")]
    steps: i32,

    /// Temperature (K)
    #[arg(long, default_value = "300.0")]
    temperature: f32,

    /// Equilibration steps before production
    #[arg(long, default_value = "100000")]
    equilibration: i32,

    /// Top N hotspots to target (default all with >10 spikes)
    #[arg(long, default_value = "20")]
    top_hotspots: usize,

    /// Hotspot clustering radius (Angstroms)
    #[arg(long, default_value = "8.0")]
    cluster_radius: f32,

    /// Grid dimension
    #[arg(long, default_value = "64")]
    grid_dim: usize,

    /// Number of parallel replicas (3-5 recommended for 3-5x speedup)
    #[arg(long, default_value = "0")]
    replicas: usize,
}

/// Hotspot from Stage 1
#[derive(Debug, Clone)]
struct Hotspot {
    center: [f32; 3],
    spike_count: usize,
    avg_intensity: f32,
    max_intensity: f32,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    log::info!("╔══════════════════════════════════════════════════════════════╗");
    log::info!("║     NHS GUIDED STAGE 2 - Targeted Production Run            ║");
    log::info!("╚══════════════════════════════════════════════════════════════╝");

    // Load and cluster Stage 1 hotspots
    let hotspots =
        load_and_cluster_hotspots(&args.stage1_spikes, args.cluster_radius, args.top_hotspots)?;
    log::info!("Loaded {} hotspots from Stage 1", hotspots.len());

    for (i, hs) in hotspots.iter().take(5).enumerate() {
        log::info!(
            "  Hotspot {}: ({:.1}, {:.1}, {:.1}) - {} spikes, max intensity {:.3}",
            i + 1,
            hs.center[0],
            hs.center[1],
            hs.center[2],
            hs.spike_count,
            hs.max_intensity
        );
    }

    #[cfg(feature = "gpu")]
    {
        run_guided_stage2(&args, &hotspots)?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        bail!("GPU feature required");
    }

    Ok(())
}

/// Load spike CSV and cluster into hotspots
fn load_and_cluster_hotspots(
    spike_path: &PathBuf,
    cluster_radius: f32,
    top_n: usize,
) -> Result<Vec<Hotspot>> {
    let file = fs::File::open(spike_path)
        .with_context(|| format!("Failed to open spike file: {}", spike_path.display()))?;
    let reader = BufReader::new(file);

    // Parse spikes
    let mut spikes: Vec<([f32; 3], f32)> = Vec::new();

    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        if i == 0 || line.starts_with('#') {
            continue; // Skip header
        }

        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 6 {
            let x: f32 = parts[2].parse().unwrap_or(0.0);
            let y: f32 = parts[3].parse().unwrap_or(0.0);
            let z: f32 = parts[4].parse().unwrap_or(0.0);
            let intensity: f32 = parts[5].parse().unwrap_or(0.0);
            spikes.push(([x, y, z], intensity));
        }
    }

    log::info!("Parsed {} spikes from Stage 1", spikes.len());

    if spikes.is_empty() {
        return Ok(Vec::new());
    }

    // Cluster spikes by proximity
    let mut clusters: Vec<(Vec<([f32; 3], f32)>, [f32; 3])> = Vec::new();
    let radius_sq = cluster_radius * cluster_radius;

    for (pos, intensity) in spikes {
        let mut found_cluster = false;

        for (cluster_spikes, center) in &mut clusters {
            let dx = pos[0] - center[0];
            let dy = pos[1] - center[1];
            let dz = pos[2] - center[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < radius_sq {
                // Add to existing cluster and update center
                cluster_spikes.push((pos, intensity));
                let n = cluster_spikes.len() as f32;
                center[0] = (center[0] * (n - 1.0) + pos[0]) / n;
                center[1] = (center[1] * (n - 1.0) + pos[1]) / n;
                center[2] = (center[2] * (n - 1.0) + pos[2]) / n;
                found_cluster = true;
                break;
            }
        }

        if !found_cluster {
            clusters.push((vec![(pos, intensity)], pos));
        }
    }

    // Convert to Hotspot structs
    let mut hotspots: Vec<Hotspot> = clusters
        .into_iter()
        .filter(|(spikes, _)| spikes.len() >= 5) // Minimum 5 spikes
        .map(|(spikes, center)| {
            let spike_count = spikes.len();
            let avg_intensity = spikes.iter().map(|(_, i)| i).sum::<f32>() / spike_count as f32;
            let max_intensity = spikes.iter().map(|(_, i)| *i).fold(0.0f32, |a, b| a.max(b));
            Hotspot {
                center,
                spike_count,
                avg_intensity,
                max_intensity,
            }
        })
        .collect();

    // Sort by spike count (descending)
    hotspots.sort_by(|a, b| b.spike_count.cmp(&a.spike_count));

    // Take top N
    hotspots.truncate(top_n);

    Ok(hotspots)
}

#[cfg(feature = "gpu")]
fn run_guided_stage2(args: &Args, hotspots: &[Hotspot]) -> Result<()> {
    let total_start = Instant::now();

    // Load topology
    log::info!("\nLoading topology: {}", args.topology.display());
    let topology = PrismPrepTopology::load(&args.topology)?;
    log::info!(
        "  Atoms: {}, Aromatics: {}",
        topology.n_atoms,
        topology.aromatic_residues().len()
    );

    // Create CUDA context
    let context = CudaContext::new(0).context("Failed to create CUDA context")?;

    // Create engine
    log::info!("Creating NHS engine...");
    let mut engine = NhsAmberFusedEngine::new(
        context,
        &topology,
        args.grid_dim,
        1.5, // grid_spacing
    )?;

    // [STAGE-2A-PERF] Initialize parallel replicas if requested (3-5x speedup)
    if args.replicas > 0 {
        log::info!(
            "🚀 Initializing {} parallel replicas (expected 3-5x speedup)...",
            args.replicas
        );
        engine.init_parallel_streams(args.replicas, &topology)?;
        log::info!("✅ Parallel replicas initialized");
    }

    // Configure for production (no cryo, stable temperature)
    let temp_protocol = TemperatureProtocol {
        start_temp: args.temperature,
        end_temp: args.temperature,
        ramp_steps: 0,
        hold_steps: args.steps + args.equilibration,
        cold_hold_steps: 0,
        current_step: 0,
    };
    engine.set_temperature_protocol(temp_protocol)?;

    // Configure UV probe - target hotspot regions
    // The UV config targets aromatics, but we'll focus on those near hotspots
    let uv_config = UvProbeConfig {
        burst_energy: 2.0,    // Moderate energy for production
        burst_interval: 2000, // Less frequent than Stage 1
        burst_duration: 50,
        target_sequence: find_aromatics_near_hotspots(&topology, hotspots),
        current_target: 0,
        timestep_counter: 0,
        ..Default::default()
    };
    engine.set_uv_config(uv_config);
    log::info!(
        "UV targeting {} aromatic residues near hotspots",
        find_aromatics_near_hotspots(&topology, hotspots).len()
    );

    // Equilibration
    log::info!(
        "\nEquilibration: {} steps at {:.0}K...",
        args.equilibration,
        args.temperature
    );
    let eq_start = Instant::now();
    let _eq_summary = engine.run(args.equilibration)?;
    log::info!("  Equilibration: {:.1}s", eq_start.elapsed().as_secs_f64());

    // Production
    log::info!(
        "\nProduction: {} steps ({:.1}ns)...",
        args.steps,
        args.steps as f64 * 2e-6
    ); // 2fs timestep
    let prod_start = Instant::now();
    let summary = engine.run(args.steps)?;
    let prod_time = prod_start.elapsed();

    log::info!(
        "  Production: {:.1}s ({:.0} steps/sec)",
        prod_time.as_secs_f64(),
        args.steps as f64 / prod_time.as_secs_f64()
    );

    // Get results
    let spikes = engine.get_spike_events();
    let snapshots = engine.get_ensemble_snapshots();

    log::info!("\nResults:");
    log::info!("  Total spikes: {}", spikes.len());
    log::info!("  Ensemble snapshots: {}", snapshots.len());
    log::info!("  Final temperature: {:.1}K", summary.end_temperature);

    // Save outputs
    let output_dir = args.output.parent().unwrap_or(std::path::Path::new("."));
    fs::create_dir_all(output_dir)?;

    let base_name = args
        .output
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "stage2".to_string());

    // Save spike CSV
    let spike_path = output_dir.join(format!("{}_stage2_spikes.csv", base_name));
    let mut spike_file = fs::File::create(&spike_path)?;
    writeln!(
        spike_file,
        "timestep,voxel_idx,x,y,z,intensity,temperature,uv_active"
    )?;
    for spike in spikes {
        writeln!(
            spike_file,
            "{},{},{:.3},{:.3},{:.3},{:.4},{:.1},{}",
            spike.timestep,
            spike.voxel_idx,
            spike.position[0],
            spike.position[1],
            spike.position[2],
            spike.intensity,
            spike.temperature,
            spike.uv_burst_active
        )?;
    }
    log::info!("  Wrote: {}", spike_path.display());

    // Save ensemble PDB
    if !snapshots.is_empty() {
        let pdb_path = output_dir.join(format!("{}_stage2_ensemble.pdb", base_name));
        write_ensemble_pdb(&pdb_path, &snapshots, &topology)?;
        log::info!("  Wrote: {}", pdb_path.display());
    }

    // Save summary JSON
    let summary_path = output_dir.join(format!("{}_stage2_summary.json", base_name));
    let summary_json = serde_json::json!({
        "structure": args.topology.display().to_string(),
        "stage1_spikes_file": args.stage1_spikes.display().to_string(),
        "hotspots_used": hotspots.len(),
        "total_steps": args.steps,
        "equilibration_steps": args.equilibration,
        "temperature": args.temperature,
        "total_spikes": engine.get_spike_events().len(),
        "ensemble_frames": snapshots.len(),
        "wall_time_sec": total_start.elapsed().as_secs_f64(),
        "throughput_steps_per_sec": args.steps as f64 / prod_time.as_secs_f64(),
    });
    fs::write(&summary_path, serde_json::to_string_pretty(&summary_json)?)?;
    log::info!("  Wrote: {}", summary_path.display());

    log::info!(
        "\n✓ Stage 2 complete in {:.1}s",
        total_start.elapsed().as_secs_f64()
    );

    Ok(())
}

/// Find aromatic residues near hotspot centers
#[cfg(feature = "gpu")]
fn find_aromatics_near_hotspots(topology: &PrismPrepTopology, hotspots: &[Hotspot]) -> Vec<usize> {
    let aromatic_residues = topology.aromatic_residues();
    let cutoff = 15.0f32; // 15Å from hotspot center
    let cutoff_sq = cutoff * cutoff;

    let mut targeted_aromatics: Vec<usize> = Vec::new();

    for (aro_idx, &res_id) in aromatic_residues.iter().enumerate() {
        // Get center of this aromatic residue
        let mut center = [0.0f32; 3];
        let mut count = 0;

        for (i, &rid) in topology.residue_ids.iter().enumerate() {
            if rid as usize == res_id {
                center[0] += topology.positions[i * 3];
                center[1] += topology.positions[i * 3 + 1];
                center[2] += topology.positions[i * 3 + 2];
                count += 1;
            }
        }

        if count > 0 {
            center[0] /= count as f32;
            center[1] /= count as f32;
            center[2] /= count as f32;

            // Check if near any hotspot
            for hs in hotspots {
                let dx = center[0] - hs.center[0];
                let dy = center[1] - hs.center[1];
                let dz = center[2] - hs.center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    targeted_aromatics.push(aro_idx);
                    break;
                }
            }
        }
    }

    targeted_aromatics
}

#[cfg(feature = "gpu")]
fn write_ensemble_pdb(
    path: &std::path::Path,
    snapshots: &[EnsembleSnapshot],
    topology: &PrismPrepTopology,
) -> Result<()> {
    let mut file = fs::File::create(path)?;

    for (model_idx, snapshot) in snapshots.iter().enumerate() {
        writeln!(file, "MODEL     {:>4}", model_idx + 1)?;

        let positions = &snapshot.positions;
        let n_atoms = positions.len() / 3;

        for i in 0..n_atoms.min(topology.n_atoms) {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            let atom_name = topology
                .atom_names
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("CA");
            let residue_name = topology
                .residue_names
                .get(i)
                .map(|s| s.as_str())
                .unwrap_or("UNK");
            let residue_id = topology.residue_ids.get(i).copied().unwrap_or(1);
            let chain_id = topology
                .chain_ids
                .get(i)
                .map(|s| s.chars().next().unwrap_or('A'))
                .unwrap_or('A');

            writeln!(
                file,
                "ATOM  {:>5} {:>4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00",
                (i + 1) % 100000,
                atom_name,
                residue_name,
                chain_id,
                residue_id % 10000,
                x,
                y,
                z
            )?;
        }

        writeln!(file, "ENDMDL")?;
    }

    writeln!(file, "END")?;

    Ok(())
}
