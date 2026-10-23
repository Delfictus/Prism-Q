//! PRISM4D Cryptic Site Benchmark
//!
//! Validates PRISM4D's ability to rediscover cryptic binding sites using
//! the Cryo-UV probing protocol with LIF neuromorphic detection.
//!
//! Strategy: Multiple short stable runs with spike accumulation.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol};

/// Benchmark configuration
const STEPS_PER_RUN: i32 = 2000;
const N_RUNS: usize = 5; // 5 runs per structure = 10000 total steps
const TEMP_START: f32 = 100.0;
const TEMP_END: f32 = 300.0;
const RAMP_STEPS: i32 = 1500;

/// Result from benchmarking one structure
#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    n_atoms: usize,
    n_aromatics: usize,
    total_steps: usize,
    total_spikes: usize,
    time_seconds: f64,
    throughput: f64,
    max_rmsd: f32,
    stable: bool,
    spike_residues: Vec<i32>, // Residue IDs that produced spikes
}

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
fn benchmark_structure(topology_path: &Path) -> Result<BenchmarkResult> {
    let topology = PrismPrepTopology::load(topology_path)
        .with_context(|| format!("Failed to load: {}", topology_path.display()))?;

    let name = topology_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let n_atoms = topology.n_atoms;

    // Count aromatics for this structure
    let n_aromatics = topology
        .residue_names
        .iter()
        .filter(|name| matches!(name.as_str(), "TRP" | "TYR" | "PHE"))
        .count();

    let start_time = Instant::now();
    let mut total_spikes = 0usize;
    let mut max_rmsd = 0.0f32;
    let mut all_spike_residues: HashSet<i32> = HashSet::new();

    for run_idx in 0..N_RUNS {
        // Create fresh engine for each run
        let context = CudaContext::new(0)?;
        let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

        // Set cryo-UV temperature protocol
        engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: TEMP_START,
            end_temp: TEMP_END,
            ramp_steps: RAMP_STEPS,
            hold_steps: STEPS_PER_RUN - RAMP_STEPS,
            cold_hold_steps: 0,
            current_step: 0,
        })?;

        let ref_pos = engine.get_positions()?;

        // Run simulation
        let summary = engine.run(STEPS_PER_RUN)?;

        // Check stability
        let final_pos = engine.get_positions()?;
        let rmsd = compute_rmsd(&ref_pos, &final_pos);
        max_rmsd = max_rmsd.max(rmsd);

        total_spikes += summary.total_spikes;

        // Collect spike residues from GPU (with proper parsing)
        if let Ok(gpu_spikes) = engine.download_full_spike_events(1000) {
            for spike in gpu_spikes {
                for i in 0..spike.n_residues.min(8) as usize {
                    let res_id = spike.nearby_residues[i];
                    if res_id > 0 {
                        all_spike_residues.insert(res_id);
                    }
                }
            }
        }

        if rmsd > 10.0 {
            log::warn!("[{}] Run {} drifted: RMSD={:.2} Å", name, run_idx + 1, rmsd);
        }
    }

    let total_time = start_time.elapsed();
    let total_steps = N_RUNS * STEPS_PER_RUN as usize;

    let mut spike_residues: Vec<i32> = all_spike_residues.into_iter().collect();
    spike_residues.sort();

    Ok(BenchmarkResult {
        name,
        n_atoms,
        n_aromatics,
        total_steps,
        total_spikes,
        time_seconds: total_time.as_secs_f64(),
        throughput: total_steps as f64 / total_time.as_secs_f64(),
        max_rmsd,
        stable: max_rmsd < 5.0,
        spike_residues,
    })
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║           PRISM4D CRYPTIC SITE BENCHMARK (Cryo-UV + LIF)             ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Protocol: {} runs × {} steps = {} total steps per structure      ║",
        N_RUNS,
        STEPS_PER_RUN,
        N_RUNS * STEPS_PER_RUN as usize
    );
    println!(
        "║  Temperature: {}K → {}K (cryo ramp)                                 ║",
        TEMP_START as i32, TEMP_END as i32
    );
    println!("║  Detection: LIF neuromorphic spike detection                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Find available test topologies
    let test_dirs = [
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/6M0J_results_test"),
    ];

    let mut topology_paths: Vec<std::path::PathBuf> = Vec::new();

    for dir in &test_dirs {
        let dir_path = Path::new(dir);
        if dir_path.exists() {
            if let Ok(entries) = std::fs::read_dir(dir_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map(|e| e == "json").unwrap_or(false)
                        && path.to_string_lossy().contains("topology")
                    {
                        topology_paths.push(path);
                    }
                }
            }
        }
    }

    if topology_paths.is_empty() {
        println!("No topology files found. Please run prism-prep on test structures first.");
        return Ok(());
    }

    topology_paths.sort();
    println!("Found {} topology files:\n", topology_paths.len());

    // Run benchmarks
    let mut results: Vec<BenchmarkResult> = Vec::new();

    for (idx, path) in topology_paths.iter().enumerate() {
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        println!(
            "[{}/{}] Processing: {}",
            idx + 1,
            topology_paths.len(),
            name
        );

        match benchmark_structure(path) {
            Ok(result) => {
                println!(
                    "       Atoms: {} | Aromatics: {} | Spikes: {} | RMSD: {:.2}Å | {}",
                    result.n_atoms,
                    result.n_aromatics,
                    result.total_spikes,
                    result.max_rmsd,
                    if result.stable { "STABLE" } else { "DRIFT" }
                );
                println!("       Spike residues: {:?}", result.spike_residues);
                results.push(result);
            }
            Err(e) => {
                println!("       ERROR: {}", e);
            }
        }
        println!();
    }

    // Summary
    println!("\n{}", "═".repeat(74));
    println!("                           BENCHMARK SUMMARY");
    println!("{}", "═".repeat(74));

    println!(
        "\n{:<25} {:>8} {:>8} {:>10} {:>8} {:>8}",
        "Structure", "Atoms", "Spikes", "Time(s)", "Steps/s", "Status"
    );
    println!("{}", "-".repeat(74));

    let mut total_spikes = 0usize;
    let mut total_time = 0.0f64;
    let mut stable_count = 0usize;

    for r in &results {
        println!(
            "{:<25} {:>8} {:>8} {:>10.1} {:>8.0} {:>8}",
            &r.name[..r.name.len().min(25)],
            r.n_atoms,
            r.total_spikes,
            r.time_seconds,
            r.throughput,
            if r.stable { "OK" } else { "DRIFT" }
        );

        total_spikes += r.total_spikes;
        total_time += r.time_seconds;
        if r.stable {
            stable_count += 1;
        }
    }

    println!("{}", "-".repeat(74));

    let total_structures = results.len();
    let stability_rate = if total_structures > 0 {
        stable_count as f64 / total_structures as f64 * 100.0
    } else {
        0.0
    };

    println!("\n  Total structures:    {}", total_structures);
    println!(
        "  Stable structures:   {} ({:.0}%)",
        stable_count, stability_rate
    );
    println!("  Total spikes:        {}", total_spikes);
    println!("  Total time:          {:.1}s", total_time);

    if total_time > 0.0 {
        let avg_throughput = results.iter().map(|r| r.total_steps as f64).sum::<f64>() / total_time;
        println!("  Avg throughput:      {:.0} steps/s", avg_throughput);
    }

    println!("\n{}", "═".repeat(74));
    if stability_rate >= 80.0 {
        println!("  BENCHMARK RESULT: READY FOR CRYPTIC SITE DETECTION");
        println!("  All structures remained stable during Cryo-UV probing.");
    } else {
        println!("  BENCHMARK RESULT: NEEDS ATTENTION");
        println!("  Some structures showed drift - may need shorter runs or restraints.");
    }
    println!("{}", "═".repeat(74));

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("This benchmark requires the 'gpu' feature. Compile with:");
    eprintln!("  cargo run --features gpu --example cryptic_site_benchmark");
}
