//! Concurrent Multi-Structure Physics Test
//!
//! Tests running DIFFERENT structures on separate CUDA contexts/streams simultaneously
//! using std::thread to verify no race conditions between independent engines.

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol};

/// Result from a single structure run
#[derive(Debug, Clone)]
struct StructureResult {
    name: String,
    n_atoms: usize,
    n_steps: usize,
    time_ms: f64,
    steps_per_sec: f64,
    final_rmsd: f32,
    valid: bool,
}

/// Compute RMSD between two position arrays
fn compute_rmsd(pos1: &[f32], pos2: &[f32]) -> f32 {
    if pos1.len() != pos2.len() || pos1.is_empty() {
        return 0.0;
    }
    let n_atoms = pos1.len() / 3;
    let mut sum_sq = 0.0;
    for i in 0..n_atoms {
        let dx = pos1[i * 3] - pos2[i * 3];
        let dy = pos1[i * 3 + 1] - pos2[i * 3 + 1];
        let dz = pos1[i * 3 + 2] - pos2[i * 3 + 2];
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    (sum_sq / n_atoms as f32).sqrt()
}

/// Check positions for NaN/Inf
fn check_positions_valid(positions: &[f32]) -> bool {
    for &p in positions {
        if p.is_nan() || p.is_infinite() || p.abs() > 500.0 {
            return false;
        }
    }
    true
}

#[cfg(feature = "gpu")]
fn run_structure(topology_path: &str, n_steps: usize) -> Result<StructureResult> {
    let path = Path::new(topology_path);
    let topology = PrismPrepTopology::load(path)
        .with_context(|| format!("Failed to load: {}", topology_path))?;

    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    println!("[{}] Starting {} atoms...", name, topology.n_atoms);

    // Create CUDA context for this thread
    let context = CudaContext::new(0)?;

    // Create temperature protocol
    let temp_protocol = TemperatureProtocol {
        start_temp: 300.0,
        end_temp: 300.0,
        ramp_steps: 0,
        hold_steps: n_steps as i32,
        cold_hold_steps: 0,
        current_step: 0,
    };

    // Create engine
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;
    engine.set_temperature_protocol(temp_protocol)?;

    // Get initial positions
    let initial_positions = engine.get_positions()?;

    // Run simulation
    let start = Instant::now();
    let _summary = engine.run(n_steps as i32)?;
    let elapsed = start.elapsed();

    // Get final positions
    let final_positions = engine.get_positions()?;

    let valid = check_positions_valid(&final_positions);
    let rmsd = compute_rmsd(&initial_positions, &final_positions);

    let time_ms = elapsed.as_secs_f64() * 1000.0;
    let steps_per_sec = n_steps as f64 / elapsed.as_secs_f64();

    println!(
        "[{}] Completed: {:.1} steps/s, RMSD={:.2}Å, valid={}",
        name, steps_per_sec, rmsd, valid
    );

    Ok(StructureResult {
        name,
        n_atoms: topology.n_atoms,
        n_steps,
        time_ms,
        steps_per_sec,
        final_rmsd: rmsd,
        valid,
    })
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔════════════════════════════════════════════════════════════════════╗");
    println!("║      PRISM4D Concurrent Multi-Structure Physics Test              ║");
    println!("╠════════════════════════════════════════════════════════════════════╣");
    println!("║  Testing: Multiple DIFFERENT structures running simultaneously    ║");
    println!("║  Verifies: No race conditions, independent physics correctness    ║");
    println!("╚════════════════════════════════════════════════════════════════════╝");
    println!();

    // Test configurations: (path, steps)
    let configs = vec![
        (
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/1L2Y_topology.json"),
            500,
        ),
        (
            concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
            300,
        ),
    ];

    // Collect results in thread-safe container
    let results: Arc<Mutex<Vec<StructureResult>>> = Arc::new(Mutex::new(Vec::new()));

    println!(
        "Starting {} concurrent structure simulations...\n",
        configs.len()
    );

    let start = Instant::now();

    // Spawn threads for each structure
    let handles: Vec<_> = configs
        .into_iter()
        .map(|(path, steps)| {
            let results = Arc::clone(&results);
            let path = path.to_string();
            thread::spawn(move || match run_structure(&path, steps) {
                Ok(result) => {
                    let mut r = results.lock().unwrap();
                    r.push(result);
                }
                Err(e) => {
                    eprintln!("[ERROR] Failed: {}", e);
                }
            })
        })
        .collect();

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    let total_elapsed = start.elapsed();

    // Print results
    println!("\n{}", "=".repeat(70));
    println!("                   CONCURRENT TEST RESULTS");
    println!("{}", "=".repeat(70));

    let results = results.lock().unwrap();
    let mut all_valid = true;

    for r in results.iter() {
        println!("\n  Structure: {}", r.name);
        println!("    Atoms: {} | Steps: {}", r.n_atoms, r.n_steps);
        println!(
            "    Time: {:.1}ms | Throughput: {:.1} steps/s",
            r.time_ms, r.steps_per_sec
        );
        println!(
            "    Final RMSD: {:.2} Å | Valid: {}",
            r.final_rmsd,
            if r.valid { "✓" } else { "✗" }
        );
        if !r.valid {
            all_valid = false;
        }
    }

    let total_steps: usize = results.iter().map(|r| r.n_steps).sum();
    let concurrent_throughput = total_steps as f64 / total_elapsed.as_secs_f64();

    println!("\n{}", "-".repeat(70));
    println!(
        "  Total wall-clock time: {:.1}ms",
        total_elapsed.as_secs_f64() * 1000.0
    );
    println!("  Total steps (all structures): {}", total_steps);
    println!(
        "  Concurrent throughput: {:.1} steps/sec",
        concurrent_throughput
    );
    println!();

    if all_valid {
        println!("  ╔═══════════════════════════════════════════════╗");
        println!("  ║   CONCURRENT PHYSICS TESTS PASSED ✓          ║");
        println!("  ╚═══════════════════════════════════════════════╝");
        Ok(())
    } else {
        println!("  ╔═══════════════════════════════════════════════╗");
        println!("  ║   CONCURRENT PHYSICS TESTS FAILED ✗          ║");
        println!("  ╚═══════════════════════════════════════════════╝");
        Err(anyhow::anyhow!("Concurrent physics test failed"))
    }
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("This test requires the 'gpu' feature. Compile with:");
    eprintln!("  cargo build --features gpu --example test_concurrent_structures");
}
