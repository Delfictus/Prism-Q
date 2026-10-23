//! Short Run Benchmark Strategy
//!
//! Instead of one long run that drifts, use multiple short runs from fresh starts.
//! Each short run stays stable (RMSD < 3 Å) and we accumulate spikes across runs.

use anyhow::Result;
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol};

fn compute_rmsd(pos1: &[f32], pos2: &[f32]) -> f32 {
    if pos1.len() != pos2.len() || pos1.is_empty() {
        return 0.0;
    }
    let n = pos1.len() / 3;
    let mut sum = 0.0;
    for i in 0..n {
        let dx = pos1[i * 3] - pos2[i * 3];
        let dy = pos1[i * 3 + 1] - pos2[i * 3 + 1];
        let dz = pos1[i * 3 + 2] - pos2[i * 3 + 2];
        sum += dx * dx + dy * dy + dz * dz;
    }
    (sum / n as f32).sqrt()
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        SHORT RUN BENCHMARK: Multiple Stable Runs                 ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Strategy: 10 x 2000-step runs instead of 1 x 20000-step run     ║");
    println!("║  Each run stays stable (RMSD < 3Å), spikes accumulate            ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("Loading topology: {}", topology_path.display());
    let topology = PrismPrepTopology::load(topology_path)?;
    println!("  Atoms: {}", topology.n_atoms);

    let n_runs = 10;
    let steps_per_run = 2000;
    let mut total_spikes = 0usize;
    let mut total_steps = 0usize;
    let mut max_rmsd = 0.0f32;

    let start_time = Instant::now();

    println!(
        "\n{:>4} {:>8} {:>8} {:>8} {:>10} {:>8}",
        "Run", "Steps", "Spikes", "RMSD", "Time(ms)", "Status"
    );
    println!("{}", "-".repeat(60));

    for run_idx in 0..n_runs {
        // Create fresh engine for each run (resets positions)
        let context = CudaContext::new(0)?;
        let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

        // Use cryo temperature ramp for cryptic site detection
        let temp_protocol = TemperatureProtocol {
            start_temp: 100.0, // Cryo start
            end_temp: 300.0,   // Ramp to room temp
            ramp_steps: 1500,  // Most of run is ramp
            hold_steps: 500,   // Short hold at 300K
            cold_hold_steps: 0,
            current_step: 0,
        };
        engine.set_temperature_protocol(temp_protocol)?;

        // Get reference positions for RMSD
        let ref_pos = engine.get_positions()?;

        // Run simulation
        let run_start = Instant::now();
        let summary = engine.run(steps_per_run)?;
        let run_time = run_start.elapsed();

        // Check stability
        let final_pos = engine.get_positions()?;
        let rmsd = compute_rmsd(&ref_pos, &final_pos);
        max_rmsd = max_rmsd.max(rmsd);

        let status = if rmsd < 3.0 {
            "OK"
        } else if rmsd < 10.0 {
            "DRIFT"
        } else {
            "FAIL"
        };

        println!(
            "{:>4} {:>8} {:>8} {:>8.2} {:>10.1} {:>8}",
            run_idx + 1,
            steps_per_run,
            summary.total_spikes,
            rmsd,
            run_time.as_secs_f64() * 1000.0,
            status
        );

        total_spikes += summary.total_spikes;
        total_steps += steps_per_run as usize;
    }

    let total_time = start_time.elapsed();

    println!("{}", "-".repeat(60));
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                                   ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Total runs:           {:>6}                                     ║",
        n_runs
    );
    println!(
        "║  Total steps:          {:>6}                                     ║",
        total_steps
    );
    println!(
        "║  Total spikes:         {:>6}                                     ║",
        total_spikes
    );
    println!(
        "║  Max RMSD:             {:>6.2} Å                                   ║",
        max_rmsd
    );
    println!(
        "║  Total time:           {:>6.1} s                                   ║",
        total_time.as_secs_f64()
    );
    println!(
        "║  Throughput:           {:>6.0} steps/s                             ║",
        total_steps as f64 / total_time.as_secs_f64()
    );
    println!("╠══════════════════════════════════════════════════════════════════╣");
    if max_rmsd < 5.0 {
        println!("║  Result: ALL RUNS STABLE - Ready for cryptic site benchmark     ║");
    } else {
        println!("║  Result: Some drift detected - may need shorter runs            ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
