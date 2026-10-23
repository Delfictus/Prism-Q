//! NHS Adaptive Detection CLI
//!
//! Jitter-based cryptic site detection with adaptive resolution protocol.
//!
//! ## Strategy: Jitter as Signal
//!
//! In a quiet (equilibrated) landscape, controlled perturbations become the
//! meaningful signal. UV-induced "jitter" at aromatic residues propagates
//! through hidden pathways, revealing cryptic binding sites.
//!
//! ## Three-Phase Protocol
//!
//! 1. **Survey** (2Å grid, 20k steps): Establish baseline, broad exploration
//! 2. **Convergence** (1Å grid, 40k steps): Signal-guided hot zone targeting
//! 3. **Precision** (0.5Å grid, 40k steps): Validate cascade events
//!
//! ## Usage
//!
//! ```bash
//! # Run adaptive detection on KRAS G12C
//! nhs-adaptive data/kras/6OIM_topology.json --output results/kras_adaptive/
//!
//! # Custom phase durations
//! nhs-adaptive input.json --survey-steps 30000 --convergence-steps 60000
//!
//! # With verbose jitter logging
//! nhs-adaptive input.json --output results/ --verbose
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use prism_nhs::adaptive::{
    AdaptiveGridProtocol, AdaptiveNhsEngine, CascadeDetector, ExplorationPhase, JitterConfig,
};
use prism_nhs::input::PrismPrepTopology;
use prism_nhs::mapping::{ExperimentalCondition, MappedHotspot, NhsSiteMapper, ProtocolType};

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_nhs::fused_engine::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

/// NHS Adaptive Detection - Jitter-based cryptic site discovery
#[derive(Parser, Debug)]
#[command(name = "nhs-adaptive")]
#[command(author = "PRISM-4D Team")]
#[command(version)]
#[command(about = "Adaptive resolution cryptic site detection with jitter analysis")]
struct Args {
    /// Input PRISM-PREP topology JSON file
    #[arg(required = true)]
    input: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "nhs_adaptive_results")]
    output: PathBuf,

    /// Survey phase duration (steps) - 1ns default for frozen baseline
    /// Note: dt=0.002ps, so 500000 steps = 1ns
    #[arg(long, default_value = "500000")]
    survey_steps: i32,

    /// Convergence phase duration (steps) - 2ns default for warming/probing
    #[arg(long, default_value = "1000000")]
    convergence_steps: i32,

    /// Precision phase duration (steps) - 2ns default for hot validation
    #[arg(long, default_value = "1000000")]
    precision_steps: i32,

    /// Survey phase grid spacing (Angstroms)
    #[arg(long, default_value = "2.0")]
    survey_grid: f32,

    /// Convergence phase grid spacing (Angstroms)
    #[arg(long, default_value = "1.0")]
    convergence_grid: f32,

    /// Precision phase grid spacing (Angstroms)
    #[arg(long, default_value = "0.5")]
    precision_grid: f32,

    /// Equilibration steps before baseline measurement
    /// Default: 50000 steps = 100ps equilibration
    #[arg(long, default_value = "50000")]
    equilibration: i32,

    /// Jitter amplitude threshold
    #[arg(long, default_value = "0.1")]
    jitter_threshold: f32,

    /// Minimum cascade size for cryptic site
    #[arg(long, default_value = "3")]
    min_cascade: usize,

    /// Cascade spatial radius (Angstroms)
    #[arg(long, default_value = "8.0")]
    cascade_radius: f32,

    /// Target temperature (K) - final temperature after warming (minimum 300K)
    #[arg(long, default_value = "300.0")]
    temperature: f32,

    /// Enable cryogenic probing protocol (freeze → probe → warm)
    #[arg(long)]
    cryo: bool,

    /// Quick test mode (reduced steps for debugging, NOT for production)
    #[arg(long)]
    quick: bool,

    /// Cryogenic start temperature (K) - requires --cryo
    #[arg(long, default_value = "100.0")]
    cryo_temp: f32,

    /// Steps to hold at cryogenic temperature before warming
    /// Default: 500000 steps = 1ns frozen (entire survey phase)
    #[arg(long, default_value = "500000")]
    cryo_hold: i32,

    /// CUDA device ID
    #[arg(long, default_value = "0")]
    device: i32,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<()> {
    // Initialize logging
    let log_level = if std::env::var("RUST_LOG").is_ok() {
        None
    } else {
        Some("info")
    };
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(log_level.unwrap_or("info")),
    )
    .format_timestamp_millis()
    .init();

    let mut args = Args::parse();

    // Validate minimum temperature (300K required for physiological relevance)
    if args.temperature < 300.0 {
        log::warn!(
            "Temperature {} K is below minimum 300K, setting to 300K",
            args.temperature
        );
        args.temperature = 300.0;
    }

    // Quick mode: reduced steps for debugging (NOT for production)
    if args.quick {
        log::warn!("QUICK MODE: Using reduced steps for testing (NOT production quality)");
        args.survey_steps = 10000; // 20ps
        args.convergence_steps = 20000; // 40ps
        args.precision_steps = 10000; // 20ps
        args.cryo_hold = 10000; // 20ps frozen
        args.equilibration = 5000; // 10ps equilibration
    }

    // Calculate total simulation time for logging
    let total_steps = args.survey_steps + args.convergence_steps + args.precision_steps;
    let total_time_ns = total_steps as f64 * 0.002 / 1000.0; // dt=0.002ps

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     PRISM-NHS: Adaptive Jitter Detection                       ║");
    println!("║     Quiet Landscape → UV Perturbation → Cascade Discovery      ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();

    if args.quick {
        println!("⚠️  QUICK TEST MODE - NOT FOR PRODUCTION");
        println!();
    }

    println!("Simulation Parameters:");
    println!(
        "  Total steps:    {} ({:.2} ns)",
        total_steps, total_time_ns
    );
    println!("  Final temp:     {} K", args.temperature);
    println!(
        "  Cryogenic:      {}",
        if args.cryo { "ENABLED" } else { "disabled" }
    );
    println!();

    // Verify input exists
    if !args.input.exists() {
        anyhow::bail!(
            "Input file not found: {}\n\
             Note: Input must be a PRISM-PREP topology JSON.\n\
             Run: prism-prep your_structure.pdb output.json --use-amber",
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
    for (i, res_id) in aromatics.iter().take(10).enumerate() {
        let res_name = topology
            .residue_ids
            .iter()
            .position(|&r| r == *res_id)
            .map(|atom_idx| topology.residue_names[atom_idx].as_str())
            .unwrap_or("???");
        println!("  {} - Residue {} ({})", i + 1, res_id, res_name);
    }
    if aromatics.len() > 10 {
        println!("  ... and {} more", aromatics.len() - 10);
    }
    println!();

    // Run GPU pipeline
    #[cfg(feature = "gpu")]
    {
        run_adaptive_pipeline(&args, &topology)?;
    }

    #[cfg(not(feature = "gpu"))]
    {
        log::error!("NHS Adaptive requires GPU support. Compile with --features gpu");
        anyhow::bail!("GPU feature not enabled");
    }

    Ok(())
}

#[cfg(feature = "gpu")]
fn run_adaptive_pipeline(args: &Args, topology: &PrismPrepTopology) -> Result<()> {
    use std::sync::Arc;

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

    // Create adaptive protocol with custom settings
    let mut protocol = AdaptiveGridProtocol::new();
    protocol.phase1_survey.duration = args.survey_steps;
    protocol.phase1_survey.resolution = args.survey_grid;
    protocol.phase2_convergence.duration = args.convergence_steps;
    protocol.phase2_convergence.resolution = args.convergence_grid;
    protocol.phase3_precision.duration = args.precision_steps;
    protocol.phase3_precision.resolution = args.precision_grid;

    // Calculate grid dimensions for each phase
    let survey_dim = ((max_dim / args.survey_grid).ceil() as usize)
        .min(64)
        .max(16);
    let convergence_dim = ((max_dim / args.convergence_grid).ceil() as usize)
        .min(128)
        .max(32);
    let precision_dim = ((max_dim / args.precision_grid).ceil() as usize)
        .min(128)
        .max(32);

    println!("Adaptive Protocol:");
    println!(
        "  Phase 1 - Survey:      {}³ voxels @ {:.1}Å ({} steps)",
        survey_dim, args.survey_grid, args.survey_steps
    );
    println!(
        "  Phase 2 - Convergence: {}³ voxels @ {:.1}Å ({} steps)",
        convergence_dim, args.convergence_grid, args.convergence_steps
    );
    println!(
        "  Phase 3 - Precision:   {}³ voxels @ {:.1}Å ({} steps)",
        precision_dim, args.precision_grid, args.precision_steps
    );
    println!();

    // Configure jitter detection
    let mut jitter_config = JitterConfig::default();
    jitter_config.amplitude_threshold = args.jitter_threshold;

    // Configure cascade detection
    let cascade_detector = CascadeDetector::new(
        args.min_cascade,
        50, // cascade window
        args.cascade_radius,
    );

    println!("Jitter Detection:");
    println!("  Equilibration:    {} steps", args.equilibration);
    println!("  Amplitude threshold: {:.3}", args.jitter_threshold);
    println!("  Min cascade size: {}", args.min_cascade);
    println!("  Cascade radius:   {:.1}Å", args.cascade_radius);
    println!();

    // Create fused engine (using survey grid initially)
    log::info!("Creating NHS-AMBER Fused Engine (Survey Phase)...");
    let mut engine =
        NhsAmberFusedEngine::new(context.clone(), topology, survey_dim, args.survey_grid)?;

    // Set temperature protocol
    let temp_protocol = if args.cryo {
        // CRYOGENIC PROBING PROTOCOL
        // 1. Start frozen at cryo_temp (e.g., 100K)
        // 2. Hold frozen during survey phase (quiet baseline)
        // 3. Begin UV probing during hold
        // 4. Warm up during convergence/precision (detect opening events)
        println!("CRYOGENIC PROBING ENABLED:");
        println!("  Start temp:  {}K (frozen)", args.cryo_temp);
        println!("  Hold frozen: {} steps (survey phase)", args.cryo_hold);
        println!(
            "  Warm to:     {}K (during convergence/precision)",
            args.temperature
        );
        println!();

        let warm_steps = protocol.total_steps() - args.cryo_hold;
        TemperatureProtocol {
            start_temp: args.cryo_temp,
            end_temp: args.temperature,
            ramp_steps: warm_steps.max(1000), // Warm up over remaining steps
            hold_steps: args.cryo_hold,       // Hold frozen first
            cold_hold_steps: 0,
            current_step: 0,
        }
    } else {
        // Standard constant temperature
        TemperatureProtocol {
            start_temp: args.temperature,
            end_temp: args.temperature,
            ramp_steps: 0,
            hold_steps: protocol.total_steps(),
            cold_hold_steps: 0,
            current_step: 0,
        }
    };
    engine.set_temperature_protocol(temp_protocol.clone())?;

    // Set UV config
    let aromatics = topology.aromatic_residues();
    let uv_config = UvProbeConfig {
        burst_energy: 4.0, // Start low for survey
        burst_interval: 2000,
        burst_duration: 10,
        target_sequence: (0..aromatics.len()).collect(),
        current_target: 0,
        timestep_counter: 0,
        ..Default::default()
    };
    engine.set_uv_config(uv_config);

    println!("════════════════════════════════════════════════════════════════");
    println!("              PHASE 1: SURVEY (Quiet Baseline)");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    let start_time = Instant::now();
    let total_steps = protocol.total_steps();

    // Create adaptive engine for tracking
    let total_voxels = survey_dim * survey_dim * survey_dim;
    let mut adaptive = AdaptiveNhsEngine::new(total_voxels);
    adaptive.protocol = protocol.clone();
    adaptive.jitter_detector.config = jitter_config;
    adaptive.cascade_detector = cascade_detector;

    // Phase tracking
    let mut current_phase = ExplorationPhase::Survey;
    let mut phase_spikes = [0usize; 3];
    let mut phase_jitter_signals = [0usize; 3];
    let mut phase_cascades = [0usize; 3];

    // Spike-based spatial analysis with aromatic-distance weighting
    // Track which voxels spike frequently → these are candidate cryptic sites
    // Weighted accumulator: (weighted_count, raw_count, aromatic_adjacent_count, position)
    let mut spike_accumulator: std::collections::HashMap<i32, (f32, usize, usize, [f32; 3])> =
        std::collections::HashMap::new();
    let spike_clustering_radius = 5.0f32; // Angstroms - cluster nearby spikes
    let aromatic_cutoff = 5.0f32; // Spikes within 5Å of aromatic count as "aromatic-adjacent"

    // Pre-compute aromatic residue center positions for weighting
    let aromatic_residue_ids = topology.aromatic_residues();
    let aromatic_centers: Vec<[f32; 3]> = aromatic_residue_ids
        .iter()
        .map(|&res_id| {
            // Average position of all atoms in this aromatic residue
            let mut sum = [0.0f32; 3];
            let mut count = 0;
            for (i, &rid) in topology.residue_ids.iter().enumerate() {
                if rid == res_id {
                    sum[0] += topology.positions[i * 3];
                    sum[1] += topology.positions[i * 3 + 1];
                    sum[2] += topology.positions[i * 3 + 2];
                    count += 1;
                }
            }
            if count > 0 {
                [
                    sum[0] / count as f32,
                    sum[1] / count as f32,
                    sum[2] / count as f32,
                ]
            } else {
                [0.0, 0.0, 0.0]
            }
        })
        .collect();

    log::info!(
        "Computed {} aromatic centers for spike weighting",
        aromatic_centers.len()
    );

    // Run simulation using batched stepping for maximum GPU throughput
    // Batch size of 500 steps minimizes CPU-GPU sync overhead while allowing progress reporting
    let batch_size = 500;
    // Sample water density periodically for jitter analysis (not every step - too slow)
    let jitter_sample_interval = 100; // Sample every 100 steps for jitter analysis
    let report_interval = total_steps / 20; // Report ~20 times
    let mut step = 0;
    let mut last_report_step = 0;
    let mut last_jitter_sample = 0;

    while step < total_steps {
        // Determine batch size (don't overshoot total or phase boundary)
        let phase_remaining = match adaptive.protocol.current_phase {
            ExplorationPhase::Survey => args.survey_steps - adaptive.protocol.current_step,
            ExplorationPhase::Convergence => {
                args.convergence_steps - adaptive.protocol.current_step
            }
            ExplorationPhase::Precision => args.precision_steps - adaptive.protocol.current_step,
        };
        let remaining = total_steps - step;
        let this_batch = batch_size.min(remaining).min(phase_remaining);

        // Run batch of steps (single GPU sync at end)
        let result = engine.step_batch(this_batch)?;
        step += this_batch;

        // Track by phase
        let phase_idx = match adaptive.protocol.current_phase {
            ExplorationPhase::Survey => 0,
            ExplorationPhase::Convergence => 1,
            ExplorationPhase::Precision => 2,
        };
        phase_spikes[phase_idx] += result.spike_count;

        // Advance protocol counter for all steps in this batch
        for _ in 0..this_batch {
            adaptive.protocol.advance();
        }

        // Download water density field periodically for jitter/cascade analysis
        // This is the key connection - GPU spikes → Rust jitter/cascade detectors
        if step - last_jitter_sample >= jitter_sample_interval {
            last_jitter_sample = step;

            // Download water density from GPU
            let water_density = engine.get_water_density()?;

            // Download spike events for spatial analysis with aromatic weighting
            if let Ok(spike_events) = engine.download_spike_events(1000) {
                if args.verbose && !spike_events.is_empty() {
                    log::info!(
                        "Step {}: Downloaded {} spike events for analysis",
                        step,
                        spike_events.len()
                    );
                }

                for (voxel_idx, pos) in &spike_events {
                    // Find distance to nearest aromatic residue
                    let min_aromatic_dist = aromatic_centers
                        .iter()
                        .map(|ac| {
                            let dx = pos[0] - ac[0];
                            let dy = pos[1] - ac[1];
                            let dz = pos[2] - ac[2];
                            (dx * dx + dy * dy + dz * dz).sqrt()
                        })
                        .fold(f32::MAX, |a, b| a.min(b));

                    // Weight by inverse distance (spikes near aromatics count more)
                    // weight = 1.0 at 0Å, ~0.5 at 5Å, ~0.2 at 10Å
                    let weight = 1.0 / (1.0 + min_aromatic_dist / aromatic_cutoff);

                    // Track if this spike is aromatic-adjacent (within cutoff)
                    let is_aromatic_adjacent = min_aromatic_dist <= aromatic_cutoff;

                    // Accumulate: (weighted_count, raw_count, aromatic_adjacent_count, position)
                    let entry = spike_accumulator
                        .entry(*voxel_idx)
                        .or_insert((0.0, 0, 0, *pos));
                    entry.0 += weight; // weighted count
                    entry.1 += 1; // raw count
                    if is_aromatic_adjacent {
                        entry.2 += 1; // aromatic-adjacent count
                    }
                }
            }

            // Update jitter detector with current field state
            adaptive.jitter_detector.update(&water_density, step);

            // Detect jitter signals (only in convergence and precision phases)
            if adaptive.protocol.current_phase != ExplorationPhase::Survey {
                let jitter_signals = adaptive.jitter_detector.detect_jitter(&water_density, step);

                // Detect cascades from jitter signals
                let cascade_events = adaptive.cascade_detector.detect_cascades(&jitter_signals);

                // Track jitter signals and cascades by phase
                phase_jitter_signals[phase_idx] += jitter_signals.len();
                phase_cascades[phase_idx] += cascade_events.len();

                // Validate cascades in precision phase
                if adaptive.protocol.current_phase == ExplorationPhase::Precision {
                    for event in &cascade_events {
                        adaptive.cascade_detector.validate_cascade(event);
                    }
                }

                // Log significant events
                if args.verbose && !jitter_signals.is_empty() {
                    log::info!(
                        "Step {}: {} jitter signals detected",
                        step,
                        jitter_signals.len()
                    );
                }
                if !cascade_events.is_empty() {
                    log::info!(
                        "Step {}: {} CASCADE EVENTS detected!",
                        step,
                        cascade_events.len()
                    );
                }
            } else {
                // Survey phase: build baseline
                // Store history for baseline establishment
                if step >= args.equilibration {
                    adaptive
                        .jitter_detector
                        .establish_baseline(&[water_density]);
                }
            }
        }

        // Check for phase transition
        let new_phase = adaptive.protocol.current_phase;
        if new_phase != current_phase {
            println!();
            println!("════════════════════════════════════════════════════════════════");
            match new_phase {
                ExplorationPhase::Convergence => {
                    println!("        PHASE 2: CONVERGENCE (Signal-Guided Exploration)");
                    println!("        Survey jitter signals: {}", phase_jitter_signals[0]);
                }
                ExplorationPhase::Precision => {
                    println!("        PHASE 3: PRECISION (Cascade Validation)");
                    println!(
                        "        Convergence jitter signals: {}",
                        phase_jitter_signals[1]
                    );
                    println!("        Convergence cascades: {}", phase_cascades[1]);
                }
                _ => {}
            }
            println!("════════════════════════════════════════════════════════════════");
            println!();
            current_phase = new_phase;
        }

        // Progress report
        if step - last_report_step >= report_interval || step >= total_steps {
            last_report_step = step;
            let elapsed = start_time.elapsed().as_secs_f64();
            let steps_per_sec = step as f64 / elapsed;
            let eta = (total_steps - step) as f64 / steps_per_sec;

            let phase_name = match current_phase {
                ExplorationPhase::Survey => "SURVEY",
                ExplorationPhase::Convergence => "CONVERGE",
                ExplorationPhase::Precision => "PRECISION",
            };

            // Get current temperature from protocol
            let current_temp = temp_protocol.start_temp
                + (temp_protocol.end_temp - temp_protocol.start_temp)
                    * (step as f32 / total_steps as f32).min(1.0);

            println!(
                "  Step {:>7}/{} | {:>9} | T={:>5.0}K | Spikes={:>3} | {:.0} steps/s | ETA {:.0}s",
                step, total_steps, phase_name, current_temp, result.spike_count, steps_per_sec, eta
            );
        }
    }

    let elapsed = start_time.elapsed();
    let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();

    println!();
    println!("════════════════════════════════════════════════════════════════");
    println!("                         RESULTS");
    println!("════════════════════════════════════════════════════════════════");
    println!();

    println!("Performance:");
    println!("  Total time:     {:.2}s", elapsed.as_secs_f64());
    println!("  Steps/second:   {:.0}", steps_per_sec);
    println!();

    println!("Phase Analysis:");
    println!(
        "  Survey phase:      {} spikes, {} jitter signals (baseline establishment)",
        phase_spikes[0], phase_jitter_signals[0]
    );
    println!(
        "  Convergence phase: {} spikes, {} jitter signals, {} cascades (hot zone detection)",
        phase_spikes[1], phase_jitter_signals[1], phase_cascades[1]
    );
    println!(
        "  Precision phase:   {} spikes, {} jitter signals, {} cascades (cascade validation)",
        phase_spikes[2], phase_jitter_signals[2], phase_cascades[2]
    );
    println!();

    // Get summary from adaptive engine
    let summary = adaptive.get_summary();
    println!("Jitter Detection:");
    println!("  Baseline noise:    {:.4}", summary.baseline_noise);
    println!("  Sensitivity boost: {:.2}x", summary.current_sensitivity);
    println!("  Jitter signals:    {}", summary.jitter_signals_detected);
    println!("  Cascade events:    {}", summary.cascade_events_detected);
    println!("  Validated sites:   {}", summary.validated_sites);
    println!();

    // Create site mapper for residue correlation
    let pdb_id = args
        .input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("UNKNOWN")
        .split('_')
        .next()
        .unwrap_or("UNKNOWN");

    // Convert residue_ids from usize to i32 for mapping
    let residue_ids_i32: Vec<i32> = topology.residue_ids.iter().map(|&id| id as i32).collect();

    let mapper = NhsSiteMapper::from_topology(
        pdb_id,
        &topology.positions, // Already flat Vec<f32>
        &residue_ids_i32,
        &topology.residue_names,
        &topology.chain_ids,
        8.0, // 8Å mapping radius
    );

    // Create experimental condition
    let condition = if args.cryo {
        ExperimentalCondition::new(args.cryo_temp, ProtocolType::CryoUvProbe)
    } else {
        ExperimentalCondition::standard(args.temperature)
    };

    // Convert weighted accumulator to old format for mapping compatibility
    // (weighted_count, raw_count, aromatic_adjacent_count, pos) -> (raw_count, pos)
    let spike_accumulator_compat: std::collections::HashMap<i32, (usize, [f32; 3])> =
        spike_accumulator
            .iter()
            .map(|(&voxel_idx, &(_, raw_count, _, pos))| (voxel_idx, (raw_count, pos)))
            .collect();

    // Map all hotspots to residues
    let mapped_hotspots = mapper.map_all_hotspots(&spike_accumulator_compat, condition.clone(), 3);

    // Compute aromatic weighting statistics
    let total_weighted: f32 = spike_accumulator.values().map(|(w, _, _, _)| w).sum();
    let total_raw: usize = spike_accumulator.values().map(|(_, r, _, _)| r).sum();
    let total_aromatic_adj: usize = spike_accumulator.values().map(|(_, _, a, _)| a).sum();

    // Filter hotspots by aromatic adjacency (>50% of spikes near aromatics)
    let aromatic_weighted_hotspots: Vec<_> = spike_accumulator
        .iter()
        .filter(|(_, (_, raw, adj, _))| *raw >= 3 && *adj as f32 / *raw as f32 > 0.5)
        .map(|(&idx, &(weighted, raw, adj, pos))| (idx, weighted, raw, adj, pos))
        .collect();

    // Analyze spike spatial distribution with AMBER residue correlation
    println!("Spike Spatial Analysis (AMBER-mapped with Aromatic Weighting):");
    if spike_accumulator.is_empty() {
        println!("  No spike events captured for spatial analysis");
    } else {
        println!(
            "  Total unique voxels with spikes: {}",
            spike_accumulator.len()
        );
        println!(
            "  Total spikes: {} raw, {:.1} weighted",
            total_raw, total_weighted
        );
        println!(
            "  Aromatic-adjacent spikes: {} ({:.1}%)",
            total_aromatic_adj,
            100.0 * total_aromatic_adj as f32 / total_raw.max(1) as f32
        );
        println!("  Mapped hotspots (3+ spikes): {}", mapped_hotspots.len());
        println!(
            "  Aromatic-weighted hotspots (>50% aromatic): {}",
            aromatic_weighted_hotspots.len()
        );
        println!("  Condition: {}", condition.condition_id);

        if !mapped_hotspots.is_empty() {
            println!("\n  Top mapped hotspots (AMBER residue correlation):");
            for (i, hs) in mapped_hotspots.iter().take(10).enumerate() {
                let residue_info = if let Some(ref primary) = hs.primary_residue {
                    format!("{} ({})", primary.full_id, primary.residue_name)
                } else {
                    format!("HS_{} (no nearby residue)", hs.voxel_idx)
                };

                let aromatic_marker = if hs.nearby_residues.iter().any(|r| r.is_aromatic) {
                    " [UV-target]"
                } else {
                    ""
                };

                println!(
                    "    {} {} @ ({:.1}, {:.1}, {:.1})Å - {} spikes{}",
                    if hs.spike_count >= 10 { "★" } else { "○" },
                    residue_info,
                    hs.position_angstrom[0],
                    hs.position_angstrom[1],
                    hs.position_angstrom[2],
                    hs.spike_count,
                    aromatic_marker
                );

                // Show nearby residues for top 3 hotspots
                if i < 3 && !hs.nearby_residues.is_empty() {
                    println!(
                        "      Nearby: {}",
                        hs.nearby_residues
                            .iter()
                            .take(5)
                            .map(|r| format!(
                                "{} ({:.1}Å)",
                                r.site_id.residue_name, r.distance_angstrom
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
            }

            // Count aromatic-adjacent hotspots (potential allosteric sites)
            let aromatic_adjacent = mapped_hotspots
                .iter()
                .filter(|hs| hs.nearby_residues.iter().any(|r| r.is_aromatic))
                .count();

            println!(
                "\n  Aromatic-adjacent hotspots: {} / {} ({:.0}%)",
                aromatic_adjacent,
                mapped_hotspots.len(),
                100.0 * aromatic_adjacent as f32 / mapped_hotspots.len() as f32
            );

            if aromatic_adjacent > 0 {
                println!("    These may respond to UV probing (allosteric candidates)");
            }

            // Display aromatic-weighted hotspots (candidate cryptic sites)
            if !aromatic_weighted_hotspots.is_empty() {
                println!("\n  ═══════════════════════════════════════════════════");
                println!("  AROMATIC-WEIGHTED HOTSPOTS (Candidate Cryptic Sites)");
                println!("  ═══════════════════════════════════════════════════");
                println!("  These hotspots have >50% of spikes near aromatic residues");
                println!("  (UV-correlated activity indicates allosteric communication)\n");

                // Sort by weighted score descending
                let mut sorted_weighted = aromatic_weighted_hotspots.clone();
                sorted_weighted
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (rank, (voxel_idx, weighted, raw, adj, pos)) in
                    sorted_weighted.iter().take(10).enumerate()
                {
                    let aromatic_pct = 100.0 * *adj as f32 / (*raw).max(1) as f32;
                    let marker = if aromatic_pct >= 75.0 {
                        "★★"
                    } else if aromatic_pct >= 50.0 {
                        "★ "
                    } else {
                        "○ "
                    };

                    println!(
                        "    {} #{:<3} @ ({:>6.1}, {:>6.1}, {:>6.1})Å",
                        marker,
                        rank + 1,
                        pos[0],
                        pos[1],
                        pos[2]
                    );
                    println!(
                        "            Weighted: {:.1} | Raw: {} | Aromatic: {}% | Voxel: {}",
                        weighted, raw, aromatic_pct as u32, voxel_idx
                    );
                }

                if aromatic_weighted_hotspots.len() > 10 {
                    println!(
                        "\n    ... and {} more aromatic-weighted hotspots",
                        aromatic_weighted_hotspots.len() - 10
                    );
                }
            } else {
                println!("\n  ⚠ No aromatic-weighted hotspots found");
                println!("    (No voxels with >50% aromatic-adjacent spikes)");
            }
        }
    }
    println!();

    // Report validated cryptic sites from jitter analysis
    let validated = adaptive.cascade_detector.get_validated_sites();
    if !validated.is_empty() {
        println!("✓ Validated Cryptic Sites (cascade analysis):");
        for (i, site) in validated.iter().enumerate() {
            println!(
                "  Site {}: ({:.1}, {:.1}, {:.1}) Å",
                i + 1,
                site[0],
                site[1],
                site[2]
            );
        }
    } else {
        println!("○ No cascade-validated sites (jitter threshold not met)");
    }
    println!();

    // Interpretation
    if phase_spikes[1] > phase_spikes[0] * 2 {
        println!("✓ Convergence phase showed increased activity");
        println!("  Signal-guided exploration found responsive regions");
    }

    if phase_spikes[2] > 0 {
        println!("✓ Precision phase detected activity");
        println!("  Cascade events may indicate cryptic site opening");
    }

    // Save results
    let output_json = args.output.join("adaptive_results.json");
    let results = serde_json::json!({
        "input": args.input.to_string_lossy(),
        "n_atoms": topology.n_atoms,
        "n_residues": topology.n_residues,
        "n_aromatics": aromatics.len(),
        "protocol": {
            "survey_steps": args.survey_steps,
            "survey_grid": args.survey_grid,
            "convergence_steps": args.convergence_steps,
            "convergence_grid": args.convergence_grid,
            "precision_steps": args.precision_steps,
            "precision_grid": args.precision_grid,
        },
        "phase_spikes": {
            "survey": phase_spikes[0],
            "convergence": phase_spikes[1],
            "precision": phase_spikes[2],
        },
        "phase_jitter_signals": {
            "survey": phase_jitter_signals[0],
            "convergence": phase_jitter_signals[1],
            "precision": phase_jitter_signals[2],
        },
        "phase_cascades": {
            "survey": phase_cascades[0],
            "convergence": phase_cascades[1],
            "precision": phase_cascades[2],
        },
        "jitter_detection": {
            "baseline_noise": summary.baseline_noise,
            "sensitivity_boost": summary.current_sensitivity,
            "jitter_signals": summary.jitter_signals_detected,
            "cascade_events": summary.cascade_events_detected,
            "validated_sites": summary.validated_sites,
        },
        "validated_sites": validated,
        "mapped_hotspots": mapped_hotspots.iter()
            .map(|hs| serde_json::json!({
                "hotspot_id": hs.hotspot_id,
                "comparison_id": hs.comparison_id,
                "voxel_idx": hs.voxel_idx,
                "spike_count": hs.spike_count,
                "position_angstrom": hs.position_angstrom,
                "primary_residue": hs.primary_residue.as_ref().map(|r| serde_json::json!({
                    "site_id": r.full_id,
                    "chain": r.chain_id,
                    "residue_number": r.residue_number,
                    "residue_name": r.residue_name,
                })),
                "nearby_residues": hs.nearby_residues.iter().take(10).map(|r| serde_json::json!({
                    "site_id": r.site_id.full_id,
                    "distance_angstrom": r.distance_angstrom,
                    "is_aromatic": r.is_aromatic,
                })).collect::<Vec<_>>(),
            }))
            .collect::<Vec<_>>(),
        "n_persistent_regions": mapped_hotspots.len(),
        "n_aromatic_adjacent": mapped_hotspots.iter()
            .filter(|hs| hs.nearby_residues.iter().any(|r| r.is_aromatic))
            .count(),
        "aromatic_weighting": {
            "total_raw_spikes": total_raw,
            "total_weighted_spikes": total_weighted,
            "total_aromatic_adjacent": total_aromatic_adj,
            "aromatic_adjacency_pct": if total_raw > 0 {
                100.0 * total_aromatic_adj as f64 / total_raw as f64
            } else { 0.0 },
            "n_weighted_hotspots": aromatic_weighted_hotspots.len(),
        },
        "aromatic_weighted_hotspots": aromatic_weighted_hotspots.iter()
            .map(|(voxel_idx, weighted, raw, adj, pos)| serde_json::json!({
                "voxel_idx": voxel_idx,
                "weighted_score": weighted,
                "raw_count": raw,
                "aromatic_adjacent_count": adj,
                "aromatic_pct": 100.0 * *adj as f64 / (*raw).max(1) as f64,
                "position_angstrom": pos,
            }))
            .collect::<Vec<_>>(),
        "experimental_condition": {
            "temperature_k": condition.temperature_k,
            "protocol": condition.protocol.as_str(),
            "condition_id": condition.condition_id,
        },
        "cryo_protocol": args.cryo,
        "cryo_temp_k": if args.cryo { Some(args.cryo_temp) } else { None },
        "elapsed_seconds": elapsed.as_secs_f64(),
        "steps_per_second": steps_per_sec,
    });

    std::fs::write(&output_json, serde_json::to_string_pretty(&results)?)?;
    println!("Results saved to: {}", output_json.display());

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_adaptive_pipeline(_args: &Args, _topology: &PrismPrepTopology) -> Result<()> {
    anyhow::bail!("GPU feature not enabled. Compile with --features gpu")
}
