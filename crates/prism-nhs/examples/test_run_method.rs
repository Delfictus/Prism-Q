//! Quick test to verify run() method stability

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

    println!("Loading topology: {}", topology_path.display());
    let topology = PrismPrepTopology::load(topology_path)?;
    println!("  Atoms: {}", topology.n_atoms);

    println!("Creating CUDA context...");
    let context = CudaContext::new(0)?;

    println!("Creating engine...");
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

    // Test 1: Constant 300K (should be stable)
    println!("\n=== TEST 1: Constant 300K ===");
    engine.set_temperature_protocol(TemperatureProtocol {
        start_temp: 300.0,
        end_temp: 300.0,
        ramp_steps: 0,
        hold_steps: 1000,
        cold_hold_steps: 0,
        current_step: 0,
    })?;

    let initial_pos = engine.get_positions()?;
    let start = Instant::now();
    let summary = engine.run(1000)?;
    let elapsed = start.elapsed();
    let final_pos = engine.get_positions()?;
    let rmsd = compute_rmsd(&initial_pos, &final_pos);

    println!(
        "  Steps: {} | Time: {:.2}s | Rate: {:.0} steps/s",
        summary.steps_completed,
        elapsed.as_secs_f32(),
        1000.0 / elapsed.as_secs_f32()
    );
    println!("  RMSD: {:.2} Å | Spikes: {}", rmsd, summary.total_spikes);
    println!("  Status: {}", if rmsd < 5.0 { "PASS" } else { "FAIL" });

    // Test 2: More steps
    println!("\n=== TEST 2: 5000 more steps at 300K ===");
    let initial_pos = engine.get_positions()?;
    let start = Instant::now();
    let summary = engine.run(5000)?;
    let elapsed = start.elapsed();
    let final_pos = engine.get_positions()?;
    let rmsd = compute_rmsd(&initial_pos, &final_pos);

    println!(
        "  Steps: {} | Time: {:.2}s | Rate: {:.0} steps/s",
        summary.steps_completed,
        elapsed.as_secs_f32(),
        5000.0 / elapsed.as_secs_f32()
    );
    println!("  RMSD: {:.2} Å | Spikes: {}", rmsd, summary.total_spikes);
    println!("  Status: {}", if rmsd < 5.0 { "PASS" } else { "FAIL" });

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
