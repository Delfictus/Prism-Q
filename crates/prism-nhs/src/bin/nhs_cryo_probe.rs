//! NHS Cryogenic Probing CLI
//!
//! Fused NHS-AMBER pipeline with dynamic temperature protocols for
//! extreme-contrast cryptic site detection.
//!
//! ## Cryogenic Contrast Principle
//!
//! At cold temperatures:
//! - Water approaches freeze (discrete event)
//! - Aromatics/hydrophobics become sluggish (continuous)
//! - UV bursts create STARK contrast against frozen background
//! - Binding site nucleation points are mapped with extreme precision
//!
//! ## Usage
//!
//! ```bash
//! # Standard cryogenic probe (100K → 300K)
//! nhs-cryo-probe input_topology.json --output results/ --protocol cryo
//!
//! # Deep freeze for maximum contrast (50K → 300K)
//! nhs-cryo-probe input_topology.json --output results/ --protocol deep-freeze
//!
//! # Flash freeze then slow warm
//! nhs-cryo-probe input_topology.json --output results/ --protocol flash-freeze
//!
//! # Custom temperature protocol
//! nhs-cryo-probe input_topology.json --start-temp 80 --end-temp 310 --ramp-steps 50000
//! ```

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;

use prism_nhs::input::PrismPrepTopology;
use prism_nhs::trajectory::{TrajectoryConfig, TrajectoryFrame, TrajectoryWriter};

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_nhs::fused_engine::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

/// Temperature protocol presets
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProtocolPreset {
    /// Standard physiological (300K constant)
    Physiological,
    /// Cryogenic probe (100K → 300K)
    Cryo,
    /// Deep freeze (50K → 300K) - maximum contrast
    DeepFreeze,
    /// Flash freeze then slow warm (100K → 300K slow)
    FlashFreeze,
    /// Custom (use --start-temp, --end-temp, --ramp-steps)
    Custom,
}

/// NHS Cryogenic Probing - Fused AMBER+NHS with temperature protocols
#[derive(Parser, Debug)]
#[command(name = "nhs-cryo-probe")]
#[command(author = "PRISM-4D Team")]
#[command(version)]
#[command(about = "Cryogenic contrast probing for cryptic site detection")]
struct Args {
    /// Input PRISM-PREP topology JSON file
    #[arg(required = true)]
    input: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "nhs_cryo_results")]
    output: PathBuf,

    /// Temperature protocol preset
    #[arg(short, long, value_enum, default_value = "cryo")]
    protocol: ProtocolPreset,

    /// Custom start temperature (K) - only with --protocol custom
    #[arg(long, default_value = "100.0")]
    start_temp: f32,

    /// Custom end temperature (K) - only with --protocol custom
    #[arg(long, default_value = "300.0")]
    end_temp: f32,

    /// Ramp steps for temperature transition
    #[arg(long, default_value = "50000")]
    ramp_steps: i32,

    /// Hold steps at final temperature
    #[arg(long, default_value = "50000")]
    hold_steps: i32,

    /// Cold hold steps at start temperature before ramping (cryo baseline)
    #[arg(long, default_value = "0")]
    cold_hold_steps: i32,

    /// Grid spacing in Angstroms
    #[arg(short, long, default_value = "1.0")]
    spacing: f32,

    /// UV burst energy (kcal/mol)
    #[arg(long, default_value = "5.0")]
    uv_energy: f32,

    /// UV burst interval (timesteps)
    #[arg(long, default_value = "1000")]
    uv_interval: i32,

    /// CUDA device ID
    #[arg(long, default_value = "0")]
    device: i32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Live monitor address (e.g., 127.0.0.1:9999)
    #[arg(long)]
    live_monitor: Option<String>,

    // =========================================================================
    // Trajectory Output Options
    // =========================================================================
    /// Save ensemble trajectory for later analysis
    #[arg(long)]
    save_trajectory: bool,

    /// Save trajectory frame every N steps (default: 1000 = 2ps)
    #[arg(long, default_value = "1000")]
    trajectory_interval: i32,

    /// Save snapshot when spike is detected
    #[arg(long)]
    save_on_spike: bool,

    /// Maximum frames to save (0 = unlimited)
    #[arg(long, default_value = "10000")]
    max_frames: usize,

    // =========================================================================
    // UV Spectroscopy Options
    // =========================================================================
    /// Enable publication-quality UV spectroscopy with frequency hopping
    /// Scans wavelengths: 250nm (S-S), 258nm (PHE), 265nm, 274nm (TYR), 280nm (TRP), 290nm
    /// Provides chromophore-specific excitation for higher signal/noise
    #[arg(long)]
    spectroscopy: bool,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_millis()
        .init();

    let args = Args::parse();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     PRISM-NHS: Cryogenic Contrast Probing                      ║");
    println!("║     Fused AMBER + Holographic + Neuromorphic Pipeline          ║");
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

    log::info!("Loading PRISM-PREP topology: {}", args.input.display());
    let topology = PrismPrepTopology::load(&args.input)?;

    log::info!(
        "Structure: {} atoms, {} residues, {} chains",
        topology.n_atoms,
        topology.n_residues,
        topology.n_chains
    );

    // Print structure info
    println!("Structure Summary:");
    println!("  Atoms:     {}", topology.n_atoms);
    println!("  Residues:  {}", topology.n_residues);
    println!("  Chains:    {}", topology.n_chains);
    println!("  Bonds:     {}", topology.bonds.len());
    println!("  Angles:    {}", topology.angles.len());
    println!("  Dihedrals: {}", topology.dihedrals.len());
    println!();

    // List aromatic residues (UV targets)
    let aromatics = topology.aromatic_residues();
    println!("UV Probe Targets ({} aromatic residues):", aromatics.len());
    for (i, res_id) in aromatics.iter().take(5).enumerate() {
        let res_name = topology
            .residue_ids
            .iter()
            .position(|&r| r == *res_id)
            .map(|atom_idx| topology.residue_names[atom_idx].as_str())
            .unwrap_or("???");
        println!("  {} - Residue {} ({})", i + 1, res_id, res_name);
    }
    if aromatics.len() > 5 {
        println!("  ... and {} more", aromatics.len() - 5);
    }
    println!();

    // Run GPU pipeline
    #[cfg(feature = "gpu")]
    {
        run_fused_pipeline(&args, &topology)?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        log::error!("NHS Cryo Probe requires GPU support. Compile with --features gpu");
        anyhow::bail!("GPU feature not enabled");
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_fused_pipeline(args: &Args, topology: &PrismPrepTopology) -> Result<()> {
    println!("Initializing CUDA device {}...", args.device);

    let context = CudaContext::new(args.device as usize)
        .context("Failed to initialize CUDA. Ensure GPU is available.")?;

    // Determine grid dimension from structure size
    let (min_pos, max_pos) = topology.bounding_box();
    let padding = 5.0f32;
    let box_size = [
        max_pos[0] - min_pos[0] + 2.0 * padding,
        max_pos[1] - min_pos[1] + 2.0 * padding,
        max_pos[2] - min_pos[2] + 2.0 * padding,
    ];
    let max_dim = box_size[0].max(box_size[1]).max(box_size[2]);
    let grid_dim = ((max_dim / args.spacing).ceil() as usize).min(128).max(32);

    println!("Grid Configuration:");
    println!("  Dimension: {}³ voxels", grid_dim);
    println!("  Spacing:   {:.2} Å", args.spacing);
    println!();

    // Create temperature protocol
    let temp_protocol = match args.protocol {
        ProtocolPreset::Physiological => {
            println!("Temperature Protocol: PHYSIOLOGICAL (300K constant)");
            TemperatureProtocol::physiological()
        }
        ProtocolPreset::Cryo => {
            println!("Temperature Protocol: CRYOGENIC (100K → 300K)");
            TemperatureProtocol::cryogenic_probe(100.0, 50000, 50000)
        }
        ProtocolPreset::DeepFreeze => {
            println!("Temperature Protocol: DEEP FREEZE (50K → 300K)");
            println!("  ⚠ Maximum contrast mode - hunting primary hydrophobics");
            TemperatureProtocol::deep_freeze()
        }
        ProtocolPreset::FlashFreeze => {
            println!("Temperature Protocol: FLASH FREEZE (100K → 300K slow)");
            println!("  ⚠ Slow warming for transient state capture");
            TemperatureProtocol::flash_freeze_slow_warm()
        }
        ProtocolPreset::Custom => {
            println!("Temperature Protocol: CUSTOM");
            println!("  Start: {:.1}K", args.start_temp);
            println!("  End:   {:.1}K", args.end_temp);
            println!("  Ramp:  {} steps", args.ramp_steps);
            println!("  Cold hold: {} steps", args.cold_hold_steps);
            TemperatureProtocol {
                start_temp: args.start_temp,
                end_temp: args.end_temp,
                ramp_steps: args.ramp_steps,
                hold_steps: args.hold_steps,
                cold_hold_steps: args.cold_hold_steps,
                current_step: 0,
            }
        }
    };

    let total_steps = temp_protocol.total_steps();
    println!(
        "  Total:  {} steps ({:.1} ns at 2fs timestep)",
        total_steps,
        total_steps as f64 * 0.002 / 1000.0
    );
    println!();

    // UV probe configuration
    let aromatics = topology.aromatic_residues();
    let uv_config = if args.spectroscopy {
        // Publication-quality UV spectroscopy with frequency hopping
        let mut config = UvProbeConfig::publication_quality();
        config.target_sequence = (0..aromatics.len()).collect();
        config.burst_energy = args.uv_energy;
        config.burst_interval = args.uv_interval;
        config
    } else {
        // Standard UV probing (localized heating)
        UvProbeConfig {
            burst_energy: args.uv_energy,
            burst_interval: args.uv_interval,
            burst_duration: 10,
            target_sequence: (0..aromatics.len()).collect(),
            current_target: 0,
            timestep_counter: 0,
            ..Default::default()
        }
    };

    println!("UV Probe Configuration:");
    if args.spectroscopy {
        println!("  Mode:           SPECTROSCOPY (frequency hopping)");
        println!("  Wavelengths:    250, 258, 265, 274, 280, 290 nm");
        println!("                  (S-S, PHE, -, TYR, TRP, -)");
        println!("  Dwell steps:    {} per wavelength", uv_config.dwell_steps);
    } else {
        println!("  Mode:           STANDARD (localized heating)");
    }
    println!("  Burst energy:   {:.1} kcal/mol", args.uv_energy);
    println!("  Burst interval: {} steps", args.uv_interval);
    println!("  Targets:        {} aromatic residues", aromatics.len());
    println!();

    // Create fused engine
    log::info!("Creating NHS-AMBER Fused Engine...");
    let mut engine = NhsAmberFusedEngine::new(context, topology, grid_dim, args.spacing)?;

    // Connect live monitor if specified
    if let Some(addr) = &args.live_monitor {
        engine.connect_live_monitor(addr)?;
        println!("Live monitor: Connected to {}", addr);
    }

    // Save spectroscopy info before moving config
    let spectroscopy_wavelengths = if args.spectroscopy {
        Some(uv_config.scan_wavelengths.clone())
    } else {
        None
    };
    let frequency_hopping_enabled = uv_config.frequency_hopping_enabled;

    // Set protocols
    engine.set_temperature_protocol(temp_protocol.clone())?;
    engine.set_uv_config(uv_config);

    // Setup trajectory writer if enabled
    let mut trajectory_writer = if args.save_trajectory || args.save_on_spike {
        let base_name = args
            .input
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "trajectory".to_string());

        let config = TrajectoryConfig {
            save_interval: args.trajectory_interval,
            save_on_spike: args.save_on_spike,
            max_memory_snapshots: 1000,
            output_dir: args.output.to_string_lossy().to_string(),
            base_name,
        };
        println!("Trajectory Output:");
        println!(
            "  Interval:     every {} steps ({:.2} ps)",
            args.trajectory_interval,
            args.trajectory_interval as f32 * 0.002
        );
        println!(
            "  Save on spike: {}",
            if args.save_on_spike { "YES" } else { "NO" }
        );
        println!(
            "  Max frames:   {}",
            if args.max_frames == 0 {
                "unlimited".to_string()
            } else {
                args.max_frames.to_string()
            }
        );
        println!();
        Some(TrajectoryWriter::new(config)?)
    } else {
        None
    };

    println!("════════════════════════════════════════════════════════════════");
    println!("                 RUNNING CRYOGENIC PROBE");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    let start_time = Instant::now();
    let report_interval = total_steps / 20; // Report 20 times

    // Run simulation
    let mut total_spikes = 0usize;
    let mut phase_spikes = [0usize; 3]; // Cold, Ramp, Warm phases

    let mut frames_saved = 0usize;

    for step in 0..total_steps {
        let result = engine.step()?;
        total_spikes += result.spike_count;

        // Track spikes by phase
        let phase = if step < temp_protocol.cold_hold_steps {
            0 // Cold hold phase
        } else if step < temp_protocol.cold_hold_steps + temp_protocol.ramp_steps {
            1 // Ramp phase
        } else {
            2 // Warm hold phase
        };
        phase_spikes[phase] += result.spike_count;

        // Save trajectory frames
        if let Some(ref mut writer) = trajectory_writer {
            let should_save_interval = writer.should_save(step);
            let should_save_spike = args.save_on_spike && result.spike_count > 0;
            let under_limit = args.max_frames == 0 || frames_saved < args.max_frames;

            if (should_save_interval || should_save_spike) && under_limit {
                let positions = engine.get_positions()?;
                let frame = TrajectoryFrame {
                    frame_idx: frames_saved,
                    timestep: step,
                    temperature: result.temperature,
                    time_ps: step as f32 * 0.002,
                    positions,
                    spike_triggered: should_save_spike,
                    spike_count: if should_save_spike {
                        Some(result.spike_count)
                    } else {
                        None
                    },
                    spike_voxels: None,
                    wavelength_nm: result.current_wavelength_nm,
                };
                writer.add_frame(frame);
                frames_saved += 1;
            }
        }

        // Progress report
        if step > 0 && step % report_interval == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let steps_per_sec = step as f64 / elapsed;
            let eta = (total_steps - step) as f64 / steps_per_sec;

            let frame_info = if trajectory_writer.is_some() {
                format!(" | Frames={}", frames_saved)
            } else {
                String::new()
            };

            println!(
                "  Step {:>7}/{} | T={:>5.1}K | Spikes={:>5} | {:.0} steps/s | ETA {:.0}s{}",
                step,
                total_steps,
                result.temperature,
                result.spike_count,
                steps_per_sec,
                eta,
                frame_info
            );

            if result.uv_burst_active && args.verbose {
                println!("    └─ UV burst active");
            }
        }
    }

    let elapsed = start_time.elapsed();
    let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
    let ns_per_day = (steps_per_sec * 0.002 * 86400.0) / 1_000_000.0;

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("                         RESULTS");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    println!("Performance:");
    println!("  Total time:     {:.2}s", elapsed.as_secs_f64());
    println!("  Steps/second:   {:.0}", steps_per_sec);
    println!("  ns/day:         {:.2}", ns_per_day);
    println!();

    println!("Spike Detection:");
    println!("  Total spikes:   {}", total_spikes);
    println!(
        "  Cold phase:     {} (primary hydrophobic mapping)",
        phase_spikes[0]
    );
    println!(
        "  Ramp phase:     {} (cryptic site emergence)",
        phase_spikes[1]
    );
    println!(
        "  Warm phase:     {} (physiological validation)",
        phase_spikes[2]
    );
    println!();

    // Interpretation
    if phase_spikes[0] > phase_spikes[2] {
        println!("✓ Cryogenic contrast successful: more spikes at cold temps");
        println!("  Primary hydrophobic sites mapped with enhanced clarity");
    }

    if phase_spikes[1] > (phase_spikes[0] + phase_spikes[2]) / 2 {
        println!("✓ Cryptic site emergence detected during temperature ramp");
        println!("  Binding sites opening as structure warms");
    }

    // Finalize trajectory and write ensemble PDB
    let trajectory_stats = if let Some(mut writer) = trajectory_writer {
        println!();
        println!("Trajectory Output:");
        println!("  Total frames:   {}", frames_saved);
        let stats = writer.finalize(&topology)?;
        println!("  Ensemble PDB:   {}", stats.ensemble_pdb);
        println!("  Spike frames:   {}", stats.spike_triggered_frames);
        println!(
            "  Time range:     {:.2} - {:.2} ps",
            stats.time_range_ps.0, stats.time_range_ps.1
        );
        println!(
            "  Temp range:     {:.1} - {:.1} K",
            stats.temperature_range.0, stats.temperature_range.1
        );
        Some(stats)
    } else {
        None
    };
    println!();

    // Save results
    let output_json = args.output.join("cryo_probe_results.json");
    let mut results = serde_json::json!({
        "input": args.input.to_string_lossy(),
        "n_atoms": topology.n_atoms,
        "n_residues": topology.n_residues,
        "protocol": format!("{:?}", args.protocol),
        "start_temp": temp_protocol.start_temp,
        "end_temp": temp_protocol.end_temp,
        "cold_hold_steps": temp_protocol.cold_hold_steps,
        "total_steps": total_steps,
        "total_spikes": total_spikes,
        "phase_spikes": {
            "cold": phase_spikes[0],
            "ramp": phase_spikes[1],
            "warm": phase_spikes[2],
        },
        "elapsed_seconds": elapsed.as_secs_f64(),
        "steps_per_second": steps_per_sec,
        "ns_per_day": ns_per_day,
        "uv_config": {
            "burst_energy": args.uv_energy,
            "burst_interval": args.uv_interval,
            "n_targets": aromatics.len(),
            "spectroscopy_mode": args.spectroscopy,
            "frequency_hopping": frequency_hopping_enabled,
            "wavelengths_nm": spectroscopy_wavelengths,
        },
    });

    // Add trajectory info if saved
    if let Some(ref stats) = trajectory_stats {
        results["trajectory"] = serde_json::json!({
            "total_frames": stats.total_frames,
            "spike_triggered_frames": stats.spike_triggered_frames,
            "interval_frames": stats.interval_frames,
            "time_range_ps": [stats.time_range_ps.0, stats.time_range_ps.1],
            "temperature_range_k": [stats.temperature_range.0, stats.temperature_range.1],
            "ensemble_pdb": stats.ensemble_pdb,
        });
    }

    std::fs::write(&output_json, serde_json::to_string_pretty(&results)?)?;
    println!();
    println!("Results saved to: {}", output_json.display());

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_fused_pipeline(_args: &Args, _topology: &PrismPrepTopology) -> Result<()> {
    anyhow::bail!("GPU feature not enabled. Compile with --features gpu")
}
