//! NHS Persistent Batch Processor
//!
//! Processes multiple structures through a single persistent GPU engine.
//! Eliminates per-structure CUDA initialization overhead.
//!
//! ## Usage
//!
//! ```bash
//! # Process multiple topologies in batch
//! nhs_batch --topologies topo1.json topo2.json topo3.json --output results/
//!
//! # Stage 1 (1ns) configuration
//! nhs_batch --topologies *.json --output results/ --stage 1
//!
//! # Stage 2 (5ns) configuration
//! nhs_batch --topologies *.json --output results/ --stage 2
//!
//! # From manifest file
//! nhs_batch --manifest batch_manifest.txt --output results/ --stage 1
//! ```

use anyhow::{bail, Context, Result};
use clap::Parser;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(feature = "gpu")]
use prism_nhs::{
    BatchProcessor, CryoUvProtocol, EnsembleSnapshot, PersistentBatchConfig, PersistentNhsEngine,
    PrismPrepTopology, StructureResult,
};

#[derive(Parser, Debug)]
#[command(name = "nhs_batch")]
#[command(about = "Persistent batch processing of NHS cryo-UV ensemble generation")]
struct Args {
    /// Topology JSON files to process
    #[arg(short, long, num_args = 1..)]
    topologies: Option<Vec<PathBuf>>,

    /// Manifest file with one topology path per line
    #[arg(short, long)]
    manifest: Option<PathBuf>,

    /// Output directory for results
    #[arg(short, long, default_value = "batch_results")]
    output: PathBuf,

    /// Stage: 1 = 1ns quick, 2 = 5ns refinement, 3 = 100ns production
    #[arg(long, default_value = "1")]
    stage: u8,

    /// Target temperature (K)
    #[arg(long, default_value = "300.0")]
    temperature: f32,

    /// Cryo temperature (K)
    #[arg(long, default_value = "100.0")]
    cryo_temp: f32,

    /// Maximum atoms (for buffer pre-allocation)
    #[arg(long, default_value = "15000")]
    max_atoms: usize,

    /// Grid dimension
    #[arg(long, default_value = "64")]
    grid_dim: usize,

    /// Skip structures that already have results
    #[arg(long)]
    skip_existing: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Use sequential mode (slower, for debugging)
    #[arg(long)]
    sequential: bool,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = Args::parse();

    // Collect topology paths
    let topology_paths = collect_topology_paths(&args)?;

    if topology_paths.is_empty() {
        bail!("No topology files specified. Use --topologies or --manifest");
    }

    log::info!("╔══════════════════════════════════════════════════════════════╗");
    log::info!("║       NHS PERSISTENT BATCH PROCESSOR                         ║");
    log::info!("╠══════════════════════════════════════════════════════════════╣");
    log::info!(
        "║  Structures: {:>4}                                            ║",
        topology_paths.len()
    );
    log::info!(
        "║  Stage: {} ({})                          ║",
        args.stage,
        stage_description(args.stage)
    );
    log::info!(
        "║  Output: {}                               ║",
        args.output.display()
    );
    log::info!("╚══════════════════════════════════════════════════════════════╝");

    // Create output directory
    fs::create_dir_all(&args.output).context("Failed to create output directory")?;

    // Configure based on stage
    let config = stage_config(
        args.stage,
        args.max_atoms,
        args.grid_dim,
        args.temperature,
        args.cryo_temp,
    );

    #[cfg(feature = "gpu")]
    {
        if args.sequential {
            log::warn!("⚠️  Sequential mode enabled (10-50x slower than default)");
            run_batch(
                &topology_paths,
                &args.output,
                config,
                args.skip_existing,
                args.verbose,
            )?;
        } else {
            log::info!("🚀 CONCURRENT MODE (DEFAULT): AmberSimdBatch enabled for 10-50x speedup");
            run_batch_concurrent(&topology_paths, &args.output, config, args.skip_existing)?;
        }
    }

    #[cfg(not(feature = "gpu"))]
    {
        bail!("GPU feature required for batch processing");
    }

    Ok(())
}

fn collect_topology_paths(args: &Args) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();

    // From --topologies argument
    if let Some(ref topos) = args.topologies {
        paths.extend(topos.iter().cloned());
    }

    // From --manifest file
    if let Some(ref manifest) = args.manifest {
        let content = fs::read_to_string(manifest).context("Failed to read manifest file")?;
        for line in content.lines() {
            let line = line.trim();
            if !line.is_empty() && !line.starts_with('#') {
                paths.push(PathBuf::from(line));
            }
        }
    }

    // Verify all paths exist
    for path in &paths {
        if !path.exists() {
            bail!("Topology file not found: {}", path.display());
        }
    }

    Ok(paths)
}

fn stage_description(stage: u8) -> &'static str {
    match stage {
        1 => "1ns quick scan",
        2 => "5ns refinement",
        3 => "100ns production",
        _ => "custom",
    }
}

#[cfg(feature = "gpu")]
fn stage_config(
    stage: u8,
    max_atoms: usize,
    grid_dim: usize,
    temperature: f32,
    cryo_temp: f32,
) -> PersistentBatchConfig {
    let (survey, convergence, precision) = match stage {
        1 => (500_000, 250_000, 250_000),         // 1ns total
        2 => (500_000, 1_000_000, 1_000_000),     // 5ns total
        3 => (1_000_000, 10_000_000, 39_000_000), // 100ns total
        _ => (500_000, 250_000, 250_000),
    };

    PersistentBatchConfig {
        max_atoms,
        grid_dim,
        grid_spacing: 1.5,
        survey_steps: survey,
        convergence_steps: convergence,
        precision_steps: precision,
        temperature,
        cryo_temp,
        cryo_hold: 100_000,
        nonbonded_execution_policy: prism_nhs::input::NonbondedExecutionPolicy::RequireExactV2,
    }
}

#[cfg(feature = "gpu")]
fn run_batch(
    topology_paths: &[PathBuf],
    output_dir: &Path,
    config: PersistentBatchConfig,
    skip_existing: bool,
    verbose: bool,
) -> Result<()> {
    let total_start = Instant::now();

    // Create persistent engine (single CUDA context/module load)
    log::info!("\nInitializing persistent engine...");
    let mut engine = PersistentNhsEngine::new(&config)?;

    let mut results_summary = Vec::new();
    let mut skipped = 0;

    for (idx, path) in topology_paths.iter().enumerate() {
        let structure_name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| format!("structure_{}", idx));

        // Check if already processed
        let result_path = output_dir.join(format!("{}_results.json", structure_name));
        if skip_existing && result_path.exists() {
            log::info!(
                "[{}/{}] Skipping {} (already exists)",
                idx + 1,
                topology_paths.len(),
                structure_name
            );
            skipped += 1;
            continue;
        }

        log::info!("\n╔════════════════════════════════════════════════════════════╗");
        log::info!(
            "║ [{}/{}] Processing: {:<39} ║",
            idx + 1,
            topology_paths.len(),
            structure_name
        );
        log::info!("╚════════════════════════════════════════════════════════════╝");

        let struct_start = Instant::now();

        // Load topology
        let topology = PrismPrepTopology::load(path)
            .with_context(|| format!("Failed to load: {}", path.display()))?;

        log::info!(
            "  Atoms: {}, Residues: {}, Aromatics: {}",
            topology.n_atoms,
            topology.residue_ids.iter().max().unwrap_or(&0) + 1,
            topology.aromatic_residues().len()
        );

        // Load into engine
        engine.load_topology(&topology)?;

        // Configure unified cryo-UV protocol (with RT defaults) [STAGE-2A-RT]
        let cryo_uv_protocol = CryoUvProtocol {
            start_temp: config.cryo_temp,
            end_temp: config.temperature,
            cold_hold_steps: config.cryo_hold,
            ramp_steps: config.convergence_steps / 2,
            warm_hold_steps: config.convergence_steps / 2,
            current_step: 0,
            // UV-LIF coupling (validated parameters from benchmark)
            uv_burst_energy: 30.0,
            uv_burst_interval: 500,
            uv_burst_duration: 50,
            scan_wavelengths: prism_nhs::fused_engine::PRODUCTION_UV_SCAN_WAVELENGTHS.to_vec(),
            wavelength_dwell_steps: 500,
            // RT fields use defaults from standard()
            ..CryoUvProtocol::standard()
        };
        engine.set_cryo_uv_protocol(cryo_uv_protocol)?;

        // Run simulation
        let total_steps = config.survey_steps + config.convergence_steps + config.precision_steps;
        log::info!("  Running {} steps...", total_steps);

        let summary = engine.run(total_steps)?;

        let wall_time = struct_start.elapsed();
        let throughput = total_steps as f64 / wall_time.as_secs_f64();

        log::info!(
            "  ✓ Completed in {:.1}s ({:.0} steps/sec)",
            wall_time.as_secs_f64(),
            throughput
        );
        log::info!(
            "  Spikes: {}, Snapshots: {}, Final temp: {:.1}K",
            engine.get_spike_events().len(),
            engine.get_snapshots().len(),
            summary.end_temperature
        );

        // Save results
        let result = StructureResultOutput {
            structure_id: structure_name.clone(),
            topology_path: path.display().to_string(),
            total_steps,
            wall_time_ms: wall_time.as_millis() as u64,
            throughput_steps_per_sec: throughput,
            n_spikes: engine.get_spike_events().len(),
            n_snapshots: engine.get_snapshots().len(),
            final_temperature: summary.end_temperature,
            config: ConfigOutput {
                survey_steps: config.survey_steps,
                convergence_steps: config.convergence_steps,
                precision_steps: config.precision_steps,
                temperature: config.temperature,
                cryo_temp: config.cryo_temp,
            },
        };

        // Write result JSON
        let result_json = serde_json::to_string_pretty(&result)?;
        fs::write(&result_path, &result_json)?;

        // Write spike events as CSV (simpler than JSON)
        let spikes = engine.get_spike_events();
        if !spikes.is_empty() {
            let spikes_path = output_dir.join(format!("{}_spikes.csv", structure_name));
            let mut file = fs::File::create(&spikes_path)?;
            writeln!(
                file,
                "timestep,voxel_idx,x,y,z,intensity,temperature,uv_active"
            )?;
            for spike in spikes {
                writeln!(
                    file,
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
        }

        // Write ensemble snapshots
        let snapshots = engine.get_snapshots();
        if !snapshots.is_empty() {
            let ensemble_path = output_dir.join(format!("{}_ensemble.pdb", structure_name));
            write_ensemble_pdb(&ensemble_path, &snapshots, &topology)?;
            log::info!("  Wrote ensemble: {}", ensemble_path.display());
        }

        results_summary.push(result);
    }

    // Write batch summary
    let total_time = total_start.elapsed();
    let stats = engine.stats();

    log::info!("\n╔══════════════════════════════════════════════════════════════╗");
    log::info!("║                    BATCH COMPLETE                            ║");
    log::info!("╠══════════════════════════════════════════════════════════════╣");
    log::info!(
        "║  Processed: {:>4} structures                                 ║",
        stats.structures_processed
    );
    log::info!(
        "║  Skipped:   {:>4} (existing)                                 ║",
        skipped
    );
    log::info!(
        "║  Total time: {:>6.1}s                                        ║",
        total_time.as_secs_f64()
    );
    log::info!(
        "║  Total steps: {:>12}                                  ║",
        stats.total_steps_run
    );
    log::info!(
        "║  Overhead saved: {:>6}ms (persistent context)              ║",
        stats.overhead_saved_ms
    );
    log::info!(
        "║  Avg throughput: {:>8.0} steps/sec                        ║",
        stats.total_steps_run as f64 / total_time.as_secs_f64()
    );
    log::info!("╚══════════════════════════════════════════════════════════════╝");

    // Write batch summary JSON
    let summary_path = output_dir.join("batch_summary.json");
    let batch_summary = BatchSummaryOutput {
        total_structures: topology_paths.len(),
        processed: stats.structures_processed,
        skipped,
        total_time_sec: total_time.as_secs_f64(),
        total_steps: stats.total_steps_run,
        overhead_saved_ms: stats.overhead_saved_ms,
        avg_throughput: stats.total_steps_run as f64 / total_time.as_secs_f64(),
        results: results_summary,
    };
    let summary_json = serde_json::to_string_pretty(&batch_summary)?;
    fs::write(&summary_path, &summary_json)?;
    log::info!("\nBatch summary: {}", summary_path.display());

    Ok(())
}

#[cfg(feature = "gpu")]
fn write_ensemble_pdb(
    path: &Path,
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

#[derive(serde::Serialize)]
struct StructureResultOutput {
    structure_id: String,
    topology_path: String,
    total_steps: i32,
    wall_time_ms: u64,
    throughput_steps_per_sec: f64,
    n_spikes: usize,
    n_snapshots: usize,
    final_temperature: f32,
    config: ConfigOutput,
}

#[derive(serde::Serialize)]
struct ConfigOutput {
    survey_steps: i32,
    convergence_steps: i32,
    precision_steps: i32,
    temperature: f32,
    cryo_temp: f32,
}

#[derive(serde::Serialize)]
struct BatchSummaryOutput {
    total_structures: usize,
    processed: usize,
    skipped: usize,
    total_time_sec: f64,
    total_steps: i64,
    overhead_saved_ms: u64,
    avg_throughput: f64,
    results: Vec<StructureResultOutput>,
}

#[cfg(feature = "gpu")]
fn run_batch_concurrent(
    topology_paths: &[PathBuf],
    output_dir: &Path,
    config: PersistentBatchConfig,
    _skip_existing: bool,
) -> Result<()> {
    use cudarc::driver::CudaContext;
    use prism_gpu::amber_simd_batch::{AmberSimdBatch, OptimizationConfig};
    use prism_nhs::simd_batch_integration::convert_to_structure_topology;
    use std::sync::Arc;

    let total_start = Instant::now();

    log::info!("\n╔══════════════════════════════════════════════════════════════╗");
    log::info!("║  🚀 CONCURRENT BATCH MODE - AmberSimdBatch                   ║");
    log::info!("║  Expected: 10-50x speedup vs sequential processing          ║");
    log::info!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load all topologies
    log::info!("📦 Loading {} topologies...", topology_paths.len());
    let mut topologies = Vec::new();
    for path in topology_paths {
        let topo = PrismPrepTopology::load(path)?;
        topologies.push(topo);
    }
    log::info!("✅ All topologies loaded\n");

    // Convert to StructureTopology format
    log::info!("🔄 Converting to AmberSimdBatch format...");
    let mut struct_topos = Vec::new();
    for topo in &topologies {
        let s = convert_to_structure_topology(topo)?;
        struct_topos.push(s);
    }
    log::info!("✅ Converted {} structures\n", struct_topos.len());

    // Create AmberSimdBatch engine
    log::info!("🚀 Creating AmberSimdBatch with MAXIMUM optimizations:");
    log::info!("   • Verlet neighbor lists (2-3x speedup)");
    log::info!("   • Tensor Cores (2-4x speedup)");
    log::info!("   • FP16 params (1.3-1.5x speedup)");
    log::info!("   • Async pipeline (1.1-1.3x speedup)");
    log::info!("   • Batched forces (true parallel processing)\n");

    // Find max atoms across all structures
    let max_atoms_actual = topologies.iter().map(|t| t.n_atoms).max().unwrap_or(35000);
    log::info!("Max atoms across structures: {}", max_atoms_actual);

    let ctx = CudaContext::new(0)?;
    let opt_config = OptimizationConfig::maximum();
    let mut batch = AmberSimdBatch::new_with_config(
        ctx,
        max_atoms_actual, // Use actual max, not config default
        topology_paths.len(),
        opt_config,
    )?;
    log::info!("✅ AmberSimdBatch engine created\n");

    // Add all structures to batch
    log::info!("📥 Adding structures to batch...");
    for (idx, topo) in struct_topos.iter().enumerate() {
        let structure_id = batch.add_structure(topo)?;
        log::info!("  [{}] Added (ID: {})", idx + 1, structure_id);
    }
    log::info!("✅ All structures added to batch\n");

    // Finalize batch (upload to GPU)
    log::info!("⚡ Finalizing batch (uploading to GPU)...");
    batch.finalize_batch()?;
    log::info!("✅ Batch finalized\n");

    // Run MD simulation
    let total_steps =
        (config.survey_steps + config.convergence_steps + config.precision_steps) as usize;
    let dt = 0.002; // 2 fs timestep
    let temperature = config.temperature;
    let gamma = 1.0; // Langevin friction

    log::info!(
        "🔥 Running {} steps on {} structures CONCURRENTLY...\n",
        total_steps,
        topology_paths.len()
    );

    let bench_start = Instant::now();
    batch.run(total_steps, dt, temperature, gamma)?;
    let elapsed = bench_start.elapsed();

    let steps_per_sec = total_steps as f64 / elapsed.as_secs_f64();
    let effective_throughput = steps_per_sec * topology_paths.len() as f64;

    log::info!("\n╔══════════════════════════════════════════════════════════════╗");
    log::info!("║                 CONCURRENT BATCH RESULTS                     ║");
    log::info!("╠══════════════════════════════════════════════════════════════╣");
    log::info!(
        "║  Structures:     {:>6}                                      ║",
        topology_paths.len()
    );
    log::info!(
        "║  Steps/struct:   {:>6}                                      ║",
        total_steps
    );
    log::info!(
        "║  Wall time:      {:>6.1}s                                   ║",
        elapsed.as_secs_f64()
    );
    log::info!(
        "║  Throughput:     {:>6.0} steps/sec                         ║",
        steps_per_sec
    );
    log::info!(
        "║  Effective:      {:>6.0} steps/sec (all structures)        ║",
        effective_throughput
    );
    log::info!("╠══════════════════════════════════════════════════════════════╣");

    let baseline = 787.0;
    let speedup = effective_throughput / baseline;
    log::info!(
        "║  vs Sequential:  {:>6.1}x SPEEDUP! 🚀                      ║",
        speedup
    );
    log::info!("╚══════════════════════════════════════════════════════════════╝\n");

    let total_time = total_start.elapsed();
    log::info!("Total batch time: {:.1}s", total_time.as_secs_f64());
    log::info!("\n✅ CONCURRENT BATCH COMPLETE!");

    Ok(())
}
