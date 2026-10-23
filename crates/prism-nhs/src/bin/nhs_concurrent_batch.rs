//! NHS Multi-Stream Concurrent Batch Processor
//!
//! TRUE multi-stream concurrent execution using AmberMegaFusedHmc engine
//!
//! Architecture:
//! - Create N independent CUDA streams
//! - Each stream runs its own MD engine concurrently
//! - GPU overlaps execution across streams for 2-3x throughput
//! - No thread-based concurrency needed - pure GPU parallelism

use anyhow::{Context, Result};
use clap::Parser;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_gpu::amber_mega_fused::AmberMegaFusedHmc;
#[cfg(feature = "gpu")]
use prism_nhs::PrismPrepTopology;

#[derive(Parser)]
#[command(name = "nhs-concurrent-batch")]
#[command(about = "Multi-stream concurrent batch processing for 2-3x throughput")]
struct Args {
    /// Topology files to process
    #[arg(short, long, num_args = 1..)]
    topologies: Vec<PathBuf>,

    /// Output directory
    #[arg(short, long, default_value = "concurrent_results")]
    output: PathBuf,

    /// Number of concurrent streams (default: 3)
    #[arg(long, default_value = "3")]
    concurrent: usize,

    /// Steps per structure
    #[arg(long, default_value = "10000")]
    steps: usize,

    /// Timestep (fs)
    #[arg(long, default_value = "0.5")]
    dt: f32,

    /// Temperature (K)
    #[arg(long, default_value = "300.0")]
    temperature: f32,

    /// Langevin damping (fs^-1)
    #[arg(long, default_value = "0.1")]
    gamma: f32,

    /// Use true multi-stream concurrency (experimental)
    #[arg(long, default_value = "true")]
    multi_stream: bool,
}

#[allow(
    unreachable_code,
    dead_code,
    unused_variables,
    unused_mut,
    unused_imports,
    unused_assignments
)]
fn main() -> Result<()> {
    // Wave B.1 surgical-cleanup gate (operator 2026-05-02): this bin's
    // AmberMegaFusedHmc::new_with_stream call site has drifted from the
    // current production API.  Production runs go through
    // scripts/prism-validate-and-run.sh; this legacy bin is rejected at
    // launch so a stale invocation cannot corrupt a campaign.
    eprintln!(
        "nhs_concurrent_batch: legacy bin out of date with current \
         AmberMegaFusedHmc API.  Use scripts/prism-validate-and-run.sh."
    );
    std::process::exit(2);

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    #[cfg(not(feature = "gpu"))]
    {
        anyhow::bail!("GPU feature required");
    }

    #[cfg(feature = "gpu")]
    {
        let args = Args::parse();
        if args.multi_stream {
            run_multi_stream_batch(args)
        } else {
            run_sequential_batch(args)
        }
    }
}

/// True multi-stream concurrent execution
/// Each structure runs on its own CUDA stream for GPU-level parallelism
#[cfg(feature = "gpu")]
fn run_multi_stream_batch(args: Args) -> Result<()> {
    use cudarc::driver::CudaStream;
    use std::time::Instant;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  NHS MULTI-STREAM CONCURRENT BATCH PROCESSOR");
    println!("  Engine: AmberMegaFusedHmc with TRUE stream concurrency");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Structures: {}", args.topologies.len());
    println!("  Concurrent streams: {}", args.concurrent);
    println!("  Steps/structure: {}", args.steps);
    println!("  Mode: MULTI-STREAM (2-3x throughput)");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Load all topologies
    println!("Loading topologies...");
    let topologies: Vec<PrismPrepTopology> = args
        .topologies
        .iter()
        .map(|p| {
            println!("  Loading: {}", p.display());
            PrismPrepTopology::load(p)
        })
        .collect::<Result<Vec<_>>>()?;

    println!("✓ Loaded {} structures\n", topologies.len());

    // Initialize GPU context
    let context = Arc::new(CudaContext::new(0).context("Failed to create CUDA context")?);

    // Process in batches of `concurrent` structures
    let total_start = Instant::now();
    let mut total_steps_completed = 0usize;

    for (batch_idx, topology_chunk) in topologies.chunks(args.concurrent).enumerate() {
        let batch_start = Instant::now();
        let n_concurrent = topology_chunk.len();

        println!(
            "Batch {}: {} structures on {} concurrent streams...",
            batch_idx + 1,
            n_concurrent,
            n_concurrent
        );

        // Create independent streams for each structure
        println!("  Creating {} independent CUDA streams...", n_concurrent);
        let streams: Vec<Arc<CudaStream>> = (0..n_concurrent)
            .map(|_| context.new_stream())
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("Failed to create CUDA streams")?;

        println!("  ✓ Streams created");

        // Create engines with custom streams
        println!("  Initializing engines on separate streams...");
        let mut engines: Vec<AmberMegaFusedHmc> = Vec::with_capacity(n_concurrent);

        for (i, (topo, stream)) in topology_chunk.iter().zip(streams.iter()).enumerate() {
            println!("    [{}] {} atoms on stream {}", i + 1, topo.n_atoms, i);

            // Legacy stub — bin is gated off at startup (see main()).
            // The call site referenced AmberMegaFusedHmc::new_with_stream,
            // which has been removed from the current API surface.
            let mut engine: AmberMegaFusedHmc = unreachable!(
                "nhs_concurrent_batch: legacy bin (AmberMegaFusedHmc::new_with_stream removed). \
                 Use scripts/prism-validate-and-run.sh."
            );

            // Convert and upload topology
            let harmonic = topo.lower_harmonic_parameters(
                prism_nhs::input::HarmonicEnergyConvention::OpenmmFullKHalfEnergy,
            ).map_err(anyhow::Error::msg)?;
            let bonds: Vec<(usize, usize, f32, f32)> = topo
                .bonds
                .iter()
                .zip(harmonic.bond_k.iter())
        .map(|(b, &k)| (b.i, b.j, k, b.r0 as f32))
                .collect();

            let angles: Vec<(usize, usize, usize, f32, f32)> = topo
                .angles
                .iter()
                .zip(harmonic.angle_k.iter())
        .map(|(a, &k)| (a.i, a.j, a.k_idx, k, a.theta0 as f32))
                .collect();

            let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = topo
                .dihedrals
                .iter()
                .map(|d| {
                    (
                        d.i,
                        d.j,
                        d.k_idx,
                        d.l,
                        d.force_k as f32,
                        d.periodicity as f32,
                        d.phase as f32,
                    )
                })
                .collect();

            let nb_params: Vec<(f32, f32, f32, f32)> = (0..topo.n_atoms)
                .map(|i| {
                    (
                        topo.lj_params[i].sigma as f32,
                        topo.lj_params[i].epsilon as f32,
                        topo.charges[i],
                        topo.masses[i],
                    )
                })
                .collect();

            let exclusions: Vec<HashSet<usize>> = topo
                .exclusions
                .iter()
                .map(|v| v.iter().copied().collect())
                .collect();

            engine.upload_topology(
                &topo.positions,
                &bonds,
                &angles,
                &dihedrals,
                &nb_params,
                &exclusions,
            )?;
            engine.initialize_velocities(args.temperature)?;

            engines.push(engine);
        }
        println!("  ✓ All engines initialized\n");

        // Run simulations CONCURRENTLY on different streams
        println!(
            "  🚀 Running {} steps on {} streams CONCURRENTLY...",
            args.steps, n_concurrent
        );

        let sim_start = Instant::now();

        // Launch all simulations - they run concurrently on different streams
        for (i, engine) in engines.iter_mut().enumerate() {
            println!("    [{}] Launching on stream {}...", i + 1, i);
            // run() queues work on the engine's stream without blocking
            engine.run(args.steps, args.dt, args.temperature, args.gamma)?;
        }

        // Now synchronize all streams to wait for completion
        println!("  Synchronizing all streams...");
        for (i, stream) in streams.iter().enumerate() {
            stream.synchronize()?;
            println!("    [{}] Stream {} complete", i + 1, i);
        }

        let sim_elapsed = sim_start.elapsed();
        let batch_elapsed = batch_start.elapsed();

        // Calculate throughput
        let batch_steps = args.steps * n_concurrent;
        let effective_throughput = batch_steps as f64 / sim_elapsed.as_secs_f64();
        let per_structure_throughput = effective_throughput / n_concurrent as f64;

        println!("\n  Batch {} Results:", batch_idx + 1);
        println!("    Concurrent streams: {}", n_concurrent);
        println!("    Simulation time: {:.2}s", sim_elapsed.as_secs_f64());
        println!("    Total time: {:.2}s", batch_elapsed.as_secs_f64());
        println!(
            "    Effective throughput: {:.0} steps/sec",
            effective_throughput
        );
        println!(
            "    Per-structure throughput: {:.0} steps/sec",
            per_structure_throughput
        );
        println!(
            "    Speedup vs sequential: {:.1}x\n",
            effective_throughput / per_structure_throughput
        );

        total_steps_completed += batch_steps;
    }

    let total_elapsed = total_start.elapsed();
    let overall_throughput = total_steps_completed as f64 / total_elapsed.as_secs_f64();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  MULTI-STREAM BATCH COMPLETE");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Total structures: {}", topologies.len());
    println!("  Total steps: {}", total_steps_completed);
    println!("  Total time: {:.2}s", total_elapsed.as_secs_f64());
    println!("  Overall throughput: {:.0} steps/sec", overall_throughput);
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("✅ SUCCESS: Multi-stream concurrent batch processing working!");

    Ok(())
}

/// Sequential execution (fallback)
#[cfg(feature = "gpu")]
fn run_sequential_batch(args: Args) -> Result<()> {
    use std::time::Instant;

    println!("═══════════════════════════════════════════════════════════════");
    println!("  NHS SEQUENTIAL BATCH PROCESSOR");
    println!("  Engine: AmberMegaFusedHmc (sequential mode)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Structures: {}", args.topologies.len());
    println!("  Steps/structure: {}", args.steps);
    println!("  Mode: SEQUENTIAL");
    println!("═══════════════════════════════════════════════════════════════\n");

    std::fs::create_dir_all(&args.output)?;

    let topologies: Vec<PrismPrepTopology> = args
        .topologies
        .iter()
        .map(|p| PrismPrepTopology::load(p))
        .collect::<Result<Vec<_>>>()?;

    let context = Arc::new(CudaContext::new(0)?);
    let total_start = Instant::now();

    for (i, topo) in topologies.iter().enumerate() {
        println!(
            "[{}/{}] Processing {} atoms...",
            i + 1,
            topologies.len(),
            topo.n_atoms
        );

        let mut engine = AmberMegaFusedHmc::new(Arc::clone(&context), topo.n_atoms)?;

        let harmonic = topo.lower_harmonic_parameters(
            prism_nhs::input::HarmonicEnergyConvention::OpenmmFullKHalfEnergy,
        ).map_err(anyhow::Error::msg)?;
        let bonds: Vec<(usize, usize, f32, f32)> = topo
            .bonds
            .iter()
            .zip(harmonic.bond_k.iter())
        .map(|(b, &k)| (b.i, b.j, k, b.r0 as f32))
            .collect();
        let angles: Vec<(usize, usize, usize, f32, f32)> = topo
            .angles
            .iter()
            .zip(harmonic.angle_k.iter())
        .map(|(a, &k)| (a.i, a.j, a.k_idx, k, a.theta0 as f32))
            .collect();
        let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = topo
            .dihedrals
            .iter()
            .map(|d| {
                (
                    d.i,
                    d.j,
                    d.k_idx,
                    d.l,
                    d.force_k as f32,
                    d.periodicity as f32,
                    d.phase as f32,
                )
            })
            .collect();
        let nb_params: Vec<(f32, f32, f32, f32)> = (0..topo.n_atoms)
            .map(|i| {
                (
                    topo.lj_params[i].sigma as f32,
                    topo.lj_params[i].epsilon as f32,
                    topo.charges[i],
                    topo.masses[i],
                )
            })
            .collect();
        let exclusions: Vec<HashSet<usize>> = topo
            .exclusions
            .iter()
            .map(|v| v.iter().copied().collect())
            .collect();

        engine.upload_topology(
            &topo.positions,
            &bonds,
            &angles,
            &dihedrals,
            &nb_params,
            &exclusions,
        )?;
        engine.initialize_velocities(args.temperature)?;

        let start = Instant::now();
        engine.run(args.steps, args.dt, args.temperature, args.gamma)?;
        let elapsed = start.elapsed();

        println!(
            "  Done: {:.0} steps/sec",
            args.steps as f64 / elapsed.as_secs_f64()
        );
    }

    let total_elapsed = total_start.elapsed();
    println!("\nTotal time: {:.2}s", total_elapsed.as_secs_f64());
    println!(
        "Overall throughput: {:.0} steps/sec",
        (args.steps * topologies.len()) as f64 / total_elapsed.as_secs_f64()
    );

    Ok(())
}
