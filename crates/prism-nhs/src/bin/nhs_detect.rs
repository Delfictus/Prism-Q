//! NHS Cryptic Site Detection CLI
//!
//! GPU-accelerated cryptic binding site detection using the holographic negative principle.
//!
//! ## Usage
//!
//! ```bash
//! # Run detection on PRISM-PREP topology
//! nhs-detect input_topology.json --output results/
//!
//! # With custom parameters
//! nhs-detect input_topology.json --output results/ --frames 500 --spacing 0.5
//! ```
//!
//! ## Input Requirements
//!
//! **CRITICAL**: Input MUST be a PRISM-PREP sanitized topology JSON.
//! Raw PDB files must go through `prism-prep` first:
//!
//! ```bash
//! prism-prep input.pdb output_topology.json --use-amber --strict
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use prism_nhs::{avalanche::CrypticSiteEvent, config::SolventMode, input::NhsPreparedInput};
use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_nhs::NhsGpuEngine;

/// NHS Cryptic Site Detection - GPU-accelerated holographic water inference
#[derive(Parser, Debug)]
#[command(name = "nhs-detect")]
#[command(author = "PRISM-4D Team")]
#[command(version)]
#[command(about = "Detect cryptic binding sites using neuromorphic holographic streaming")]
struct Args {
    /// Input PRISM-PREP topology JSON file
    #[arg(required = true)]
    input: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "nhs_results")]
    output: PathBuf,

    /// Number of frames to simulate
    #[arg(short, long, default_value = "200")]
    frames: usize,

    /// Grid spacing in Angstroms
    #[arg(short, long, default_value = "0.5")]
    spacing: f32,

    /// Grid padding around protein (Angstroms)
    #[arg(short, long, default_value = "5.0")]
    padding: f32,

    /// Enable UV bias perturbation for causal validation
    #[arg(long, default_value = "true")]
    uv_bias: bool,

    /// LIF membrane time constant (frames)
    #[arg(long, default_value = "10.0")]
    tau_mem: f32,

    /// Dewetting sensitivity multiplier
    #[arg(long, default_value = "1.0")]
    sensitivity: f32,

    /// CUDA device ID
    #[arg(long, default_value = "0")]
    device: i32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    // === RT Integration [STAGE-1-INTEGRATION] ===
    /// Solvent mode: implicit (fast), explicit (high-fidelity), or hybrid (adaptive)
    ///
    /// - implicit: No explicit waters, fastest (~1 μs/day simulation)
    /// - explicit: Full water box, most accurate (~50-100 ns/day simulation)
    /// - hybrid: Start implicit, switch to explicit for detected pockets
    ///
    /// Example: --solvent-mode explicit --water-padding 10.0
    #[arg(long, default_value = "implicit")]
    solvent_mode: String,

    /// Water box padding (Angstroms) for explicit mode
    ///
    /// Typical: 10-15 Å creates ~20K-50K water molecules for small proteins
    #[arg(long, default_value = "10.0")]
    water_padding: f32,

    /// Enable RT probe scanning (requires RTX GPU with RT cores)
    ///
    /// Uses 84 RT cores on RTX 5080 for spatial sensing:
    /// - Solvation disruption detection (earliest signal)
    /// - Geometric void detection via ray tracing
    /// - Aromatic LIF (laser-induced fluorescence)
    #[arg(long, default_value = "false")]
    enable_rt: bool,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     PRISM-NHS: Neuromorphic Holographic Stream Detection       ║");
    println!("║              Cryptic Binding Site Discovery                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    // Verify input exists
    if !args.input.exists() {
        anyhow::bail!(
            "Input file not found: {}\n\
             Note: Input must be a PRISM-PREP topology JSON.\n\
             Run: prism-prep your_structure.pdb output.json --use-amber --strict",
            args.input.display()
        );
    }

    // Create output directory
    std::fs::create_dir_all(&args.output).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            args.output.display()
        )
    })?;

    // Parse solvent mode [STAGE-1-INTEGRATION]
    let solvent_mode = match args.solvent_mode.to_lowercase().as_str() {
        "implicit" => SolventMode::Implicit,
        "explicit" => SolventMode::Explicit {
            padding_angstroms: args.water_padding,
        },
        "hybrid" => SolventMode::Hybrid {
            exploration_steps: 100,
            characterization_steps: 500,
            switch_threshold: 0.6,
        },
        mode => anyhow::bail!(
            "Invalid solvent mode: '{}'. Must be 'implicit', 'explicit', or 'hybrid'",
            mode
        ),
    };

    log::info!("Solvent mode: {:?}", solvent_mode);
    if args.enable_rt {
        log::info!("RT probe scanning: ENABLED (requires RTX GPU)");
    } else {
        log::info!("RT probe scanning: DISABLED");
    }

    log::info!("Loading PRISM-PREP topology: {}", args.input.display());
    let start_load = Instant::now();
    let prepared = NhsPreparedInput::load(&args.input, args.spacing, args.padding, &solvent_mode)?;
    let load_time = start_load.elapsed();

    log::info!(
        "Structure: {} atoms ({} total with waters), {} residues",
        prepared.topology.n_atoms,
        prepared.total_atoms,
        prepared.topology.n_residues
    );
    log::info!("Topology loading time: {:.2}s", load_time.as_secs_f64());

    // RT Integration info [STAGE-1-INTEGRATION]
    if let Some(ref water_atoms) = prepared.water_atoms {
        println!("\nSolvation:");
        println!("  Mode:             {:?}", prepared.solvent_mode);
        println!("  Water molecules:  {}", water_atoms.len());
        println!(
            "  Total atoms:      {} ({} protein + {} waters)",
            prepared.total_atoms,
            prepared.total_atoms - water_atoms.len(),
            water_atoms.len()
        );
    } else {
        println!("\nSolvation:");
        println!("  Mode:             Implicit (no explicit waters)");
        println!(
            "  Total atoms:      {} (protein only)",
            prepared.total_atoms
        );
    }

    println!("\nRT Probe Targets:");
    println!("  {}", prepared.rt_targets.summary());
    if args.enable_rt {
        println!("  Status:           RT scanning ENABLED");
    } else {
        println!("  Status:           RT scanning DISABLED (use --enable-rt to activate)");
    }
    println!();

    // Print atom type statistics
    let stats = prepared.topology.atom_type_stats();
    println!("\nAtom Classification:");
    println!(
        "  Hydrophobic:      {:>5}",
        stats
            .get(&prism_nhs::NhsAtomType::Hydrophobic)
            .unwrap_or(&0)
    );
    println!(
        "  Polar:            {:>5}",
        stats.get(&prism_nhs::NhsAtomType::Polar).unwrap_or(&0)
    );
    println!(
        "  Charged (+):      {:>5}",
        stats
            .get(&prism_nhs::NhsAtomType::ChargedPositive)
            .unwrap_or(&0)
    );
    println!(
        "  Charged (-):      {:>5}",
        stats
            .get(&prism_nhs::NhsAtomType::ChargedNegative)
            .unwrap_or(&0)
    );
    println!(
        "  Aromatic (UV):    {:>5}",
        stats.get(&prism_nhs::NhsAtomType::Aromatic).unwrap_or(&0)
    );
    println!(
        "  Backbone:         {:>5}",
        stats.get(&prism_nhs::NhsAtomType::Backbone).unwrap_or(&0)
    );
    println!();

    // List aromatic residues (UV bias targets)
    let aromatic = prepared.topology.aromatic_residues();
    if !aromatic.is_empty() {
        println!("UV Bias Targets ({} aromatic residues):", aromatic.len());
        for (i, res_id) in aromatic.iter().take(10).enumerate() {
            // Get residue name by residue ID (not atom index)
            let res_name = if *res_id < prepared.topology.residue_names.len() {
                prepared.topology.residue_names[*res_id].as_str()
            } else {
                "???"
            };
            println!("  {} - Residue {} ({})", i + 1, res_id, res_name);
        }
        if aromatic.len() > 10 {
            println!("  ... and {} more", aromatic.len() - 10);
        }
        println!();
    }

    println!("Grid Configuration:");
    println!("  Dimension:  {}³ voxels", prepared.grid_dim);
    println!("  Spacing:    {:.2} Å", args.spacing);
    println!(
        "  Origin:     [{:.1}, {:.1}, {:.1}]",
        prepared.grid_origin[0], prepared.grid_origin[1], prepared.grid_origin[2]
    );
    println!("  Frames:     {}", args.frames);
    println!();

    // Run GPU detection
    #[cfg(feature = "gpu")]
    {
        run_gpu_detection(&args, &prepared)?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        log::error!("NHS requires GPU support. Compile with --features gpu");
        anyhow::bail!("GPU feature not enabled");
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_gpu_detection(args: &Args, prepared: &NhsPreparedInput) -> Result<()> {
    println!("Initializing CUDA device {}...", args.device);

    let context = CudaContext::new(args.device as usize)
        .context("Failed to initialize CUDA. Ensure GPU is available.")?;

    log::info!("Creating NHS GPU Engine...");
    let mut engine = NhsGpuEngine::new(context, prepared.grid_dim, prepared.topology.n_atoms)?;

    engine.set_lif_params(args.tau_mem, args.sensitivity);
    engine.initialize(prepared.grid_origin)?;

    println!("Running NHS detection ({} frames)...\n", args.frames);

    let mut total_spikes = 0;
    let _cryptic_events: Vec<CrypticSiteEvent> = Vec::new(); // For future use
    let start_time = Instant::now();

    // For now, we run multiple "frames" with the same positions
    // In real use, this would process trajectory frames
    for frame in 0..args.frames {
        let frame_result = engine.process_frame(
            &prepared.positions,
            &prepared.types,
            &prepared.charges,
            &prepared.residues,
        )?;

        total_spikes += frame_result.spike_count;

        if frame_result.spike_count > 0 && args.verbose {
            log::info!(
                "Frame {}: {} spikes detected",
                frame,
                frame_result.spike_count
            );
        }

        // Progress indicator
        if (frame + 1) % 50 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let fps = (frame + 1) as f64 / elapsed;
            println!(
                "  Frame {}/{} ({:.1} fps, {} total spikes)",
                frame + 1,
                args.frames,
                fps,
                total_spikes
            );
        }
    }

    let elapsed = start_time.elapsed();
    let fps = args.frames as f64 / elapsed.as_secs_f64();
    let ms_per_frame = elapsed.as_secs_f64() * 1000.0 / args.frames as f64;

    println!("\n════════════════════════════════════════════════════════════════");
    println!("                         RESULTS                                 ");
    println!("════════════════════════════════════════════════════════════════\n");

    println!("Performance:");
    println!("  Total time:       {:.2}s", elapsed.as_secs_f64());
    println!("  Frames/second:    {:.1}", fps);
    println!("  ms/frame:         {:.2}", ms_per_frame);
    println!();

    println!("Detection Summary:");
    println!("  Total spikes:     {}", total_spikes);
    println!(
        "  Avg spikes/frame: {:.1}",
        total_spikes as f64 / args.frames as f64
    );
    println!();

    // Get final pocket probability
    let pocket_prob = engine.finalize_pocket_probability(0.05)?;
    let high_prob_voxels: usize = pocket_prob.iter().filter(|&&p| p > 0.5).count();
    let cryptic_voxels: usize = pocket_prob.iter().filter(|&&p| p > 0.7).count();

    println!("Pocket Analysis:");
    println!("  High probability voxels (>0.5): {}", high_prob_voxels);
    println!("  Cryptic site voxels (>0.7):     {}", cryptic_voxels);
    println!();

    // Performance target check
    if ms_per_frame < 2.0 {
        println!(
            "✓ Performance target MET: {:.2} ms/frame < 2.0 ms",
            ms_per_frame
        );
    } else {
        println!(
            "✗ Performance target MISSED: {:.2} ms/frame > 2.0 ms",
            ms_per_frame
        );
    }

    // Save results (with RT integration info) [STAGE-1-INTEGRATION]
    let output_json = args.output.join("nhs_results.json");
    let results = serde_json::json!({
        "input": args.input.to_string_lossy(),
        "n_atoms": prepared.topology.n_atoms,
        "n_residues": prepared.topology.n_residues,
        "total_atoms": prepared.total_atoms,
        "n_waters": prepared.water_atoms.as_ref().map(|w| w.len()).unwrap_or(0),
        "solvent_mode": format!("{:?}", prepared.solvent_mode),
        "rt_targets": {
            "protein_atoms": prepared.rt_targets.protein_atoms.len(),
            "water_atoms": prepared.rt_targets.water_atoms.as_ref().map(|w| w.len()).unwrap_or(0),
            "aromatic_centers": prepared.rt_targets.aromatic_centers.len(),
            "total_targets": prepared.rt_targets.total_targets,
        },
        "rt_enabled": args.enable_rt,
        "n_frames": args.frames,
        "grid_dim": prepared.grid_dim,
        "grid_spacing": args.spacing,
        "total_spikes": total_spikes,
        "avg_spikes_per_frame": total_spikes as f64 / args.frames as f64,
        "high_prob_voxels": high_prob_voxels,
        "cryptic_voxels": cryptic_voxels,
        "elapsed_seconds": elapsed.as_secs_f64(),
        "ms_per_frame": ms_per_frame,
        "performance_target_met": ms_per_frame < 2.0,
    });

    std::fs::write(&output_json, serde_json::to_string_pretty(&results)?)?;
    println!("\nResults saved to: {}", output_json.display());

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_gpu_detection(_args: &Args, _prepared: &NhsPreparedInput) -> Result<()> {
    anyhow::bail!("GPU feature not enabled. Compile with --features gpu")
}
