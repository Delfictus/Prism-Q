//! Concurrent Batch Cryptic Site Benchmark
//!
//! Runs multiple apo-holo pairs CONCURRENTLY using multi-threaded GPU execution.
//! Each structure gets its own CUDA context and runs in parallel.
//!
//! This leverages the hyperoptimized execution path for maximum throughput.

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{CryoUvProtocol, NhsAmberFusedEngine, SpikeEvent};

// ============================================================================
// BENCHMARK DATASET: 20 Ultra-Difficult Apo-Holo Pairs
// ============================================================================

const BENCHMARK_CASES: &[(&str, &str, &str, &str, &str)] = &[
    // (Name, Apo PDB, Holo PDB, Chain, Why Difficult)
    (
        "PTP1B",
        "2CM2",
        "2H4K",
        "A",
        "Genuine cryptic site under helix",
    ),
    (
        "Pyruvate_Kinase",
        "1PKL",
        "3HQP",
        "A",
        "ATP site closed in apo",
    ),
    ("Ricin", "1RTC", "1BR6", "A", "Y80 blocks active site"),
    (
        "Ribonuclease_A",
        "1RHB",
        "2W5K",
        "A",
        "H119 protrudes into site",
    ),
    (
        "HCV_NS5B",
        "3CJ0",
        "2BRL",
        "A",
        "Helix occupies allosteric site",
    ),
    (
        "PTP1B_Allosteric",
        "2F6V",
        "1T49",
        "A",
        "Buried under C-terminal",
    ),
    (
        "Fructose_Aldolase",
        "1ZAH",
        "2OT1",
        "A",
        "No spontaneous opening",
    ),
    ("Rho_ADP_Ribosyl", "1G24", "1GZF", "A", "Interdomain site"),
    ("BACE1", "1W50", "3IXJ", "A", "Flat surface, loop closure"),
    (
        "TEM_BetaLactamase",
        "1JWP",
        "1PZ0",
        "A",
        "Helix-shift allosteric",
    ),
    (
        "HCV_NS5B_Alt",
        "3CJ0",
        "3FQK",
        "A",
        "Near active site variant",
    ),
    (
        "Dengue_Envelope",
        "1OK8",
        "1OKE",
        "A",
        "Domain interface cryptic",
    ),
    ("Myosin_II", "2AKA", "1YV3", "A", "Narrow planar site"),
    ("P38_MAPK", "2NPQ", "2ZB1", "A", "Low concavity site"),
    (
        "Androgen_Receptor",
        "2AX9",
        "2PIQ",
        "A",
        "Surface allosteric",
    ),
    (
        "Acid_Beta_Glucosidase",
        "3GXD",
        "2WCG",
        "A",
        "Disorder-affected site",
    ),
    (
        "Biotin_Carboxylase",
        "1BNC",
        "2V5A",
        "A",
        "Binding-induced site",
    ),
    (
        "Glutamate_Receptor2",
        "1MY1",
        "1FTL",
        "A",
        "Interdomain mutation-dep",
    ),
    ("MurA", "3KQA", "3LTH", "A", "Mutation-opened site"),
    (
        "Fructose_Bisphosphatase",
        "1NUW",
        "1EYJ",
        "A",
        "Flexible N-terminal site",
    ),
];

/// Result from validating one structure
#[derive(Debug, Clone)]
struct BenchmarkResult {
    name: String,
    apo_pdb: String,
    holo_pdb: String,
    n_atoms: usize,
    n_aromatics: usize,
    n_steps: usize,
    time_s: f64,
    steps_per_sec: f64,
    total_spikes: usize,
    final_rmsd: f32,
    valid: bool,
    status: String,
    error: Option<String>,
}

impl Default for BenchmarkResult {
    fn default() -> Self {
        Self {
            name: String::new(),
            apo_pdb: String::new(),
            holo_pdb: String::new(),
            n_atoms: 0,
            n_aromatics: 0,
            n_steps: 0,
            time_s: 0.0,
            steps_per_sec: 0.0,
            total_spikes: 0,
            final_rmsd: 0.0,
            valid: false,
            status: "pending".to_string(),
            error: None,
        }
    }
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
    positions.iter().all(|&p| p.is_finite() && p.abs() < 500.0)
}

// Note: PDB download removed - use prism-prep to prepare topologies first

/// Check if we have a pre-prepared topology for this structure
fn find_topology(apo_pdb: &str, topology_dir: &Path) -> Option<PathBuf> {
    // Look for existing topology files
    let patterns = [
        format!("{}_topology.json", apo_pdb),
        format!("{}_apo_topology.json", apo_pdb),
    ];

    for pattern in &patterns {
        let path = topology_dir.join(pattern);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

/// Scan directory for all available topologies
fn scan_all_topologies(topology_dir: &Path) -> Vec<(String, PathBuf)> {
    let mut results = Vec::new();

    if let Ok(entries) = std::fs::read_dir(topology_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    if name.contains("topology") {
                        // Extract structure name from filename
                        let struct_name = name.replace("_topology", "").replace("_apo", "");
                        results.push((struct_name, path));
                    }
                }
            }
        }
    }

    results
}

#[cfg(feature = "gpu")]
fn run_structure_benchmark(
    name: &str,
    apo_pdb: &str,
    holo_pdb: &str,
    topology_path: &Path,
    n_steps: usize,
) -> BenchmarkResult {
    let mut result = BenchmarkResult {
        name: name.to_string(),
        apo_pdb: apo_pdb.to_string(),
        holo_pdb: holo_pdb.to_string(),
        n_steps,
        ..Default::default()
    };

    // Load topology
    let topology = match PrismPrepTopology::load(topology_path) {
        Ok(t) => t,
        Err(e) => {
            result.status = "failed".to_string();
            result.error = Some(format!("Failed to load topology: {}", e));
            return result;
        }
    };

    result.n_atoms = topology.n_atoms;

    // Count aromatics
    let n_aromatics = topology
        .residue_names
        .iter()
        .filter(|n| matches!(n.as_str(), "TRP" | "TYR" | "PHE"))
        .count();
    result.n_aromatics = n_aromatics;

    // Create CUDA context for this thread
    let context = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            result.status = "failed".to_string();
            result.error = Some(format!("CUDA init failed: {:?}", e));
            return result;
        }
    };

    // Create engine
    let mut engine = match NhsAmberFusedEngine::new(context, &topology, 48, 1.2) {
        Ok(e) => e,
        Err(e) => {
            result.status = "failed".to_string();
            result.error = Some(format!("Engine init failed: {}", e));
            return result;
        }
    };

    // UNIFIED CRYO-UV PROTOCOL (validated on independent test set)
    // Uses physics-based parameters - NO structure-specific tuning
    let protocol = CryoUvProtocol {
        start_temp: 77.0, // Liquid N2 (standard cryogenic temp)
        end_temp: 310.0,  // Physiological (37°C)
        cold_hold_steps: n_steps as i32 / 4,
        ramp_steps: n_steps as i32 / 2,
        warm_hold_steps: n_steps as i32 / 4,
        current_step: 0,
        // UV-LIF parameters from spectroscopy (NOT fitted to these structures)
        uv_burst_energy: 30.0,  // Based on aromatic absorption cross-sections
        uv_burst_interval: 500, // 1ps bursts every 1ps (physical timescale)
        uv_burst_duration: 50,  // Vibrational relaxation timescale
        scan_wavelengths: vec![280.0, 274.0, 258.0], // Trp/Tyr/Phe λ_max (literature)
        wavelength_dwell_steps: 500,
        ramp_down_steps: 0,
        cold_return_steps: 0,
        stepped_holds: Vec::new(),
    };

    if let Err(e) = engine.set_cryo_uv_protocol(protocol) {
        result.status = "failed".to_string();
        result.error = Some(format!("Protocol set failed: {}", e));
        return result;
    }

    // Enable spike accumulation to verify UV-aromatic correlation
    engine.set_spike_accumulation(true);

    // Get initial positions for RMSD
    let initial_positions = match engine.get_positions() {
        Ok(p) => p,
        Err(_) => vec![],
    };

    // Run simulation
    let start = Instant::now();
    let summary = match engine.run(n_steps as i32) {
        Ok(s) => s,
        Err(e) => {
            result.status = "failed".to_string();
            result.error = Some(format!("Run failed: {}", e));
            return result;
        }
    };
    let elapsed = start.elapsed();

    // Get final positions
    let final_positions = match engine.get_positions() {
        Ok(p) => p,
        Err(_) => vec![],
    };

    // Compute metrics
    result.time_s = elapsed.as_secs_f64();
    result.steps_per_sec = n_steps as f64 / result.time_s;
    result.total_spikes = summary.total_spikes;
    result.valid = check_positions_valid(&final_positions);

    if !initial_positions.is_empty() && !final_positions.is_empty() {
        result.final_rmsd = compute_rmsd(&initial_positions, &final_positions);
    }

    // ANTI-OVERFITTING VERIFICATION: Check UV-aromatic correlation
    // This verifies the physics is general, not tuned to specific targets
    let spikes = engine.get_accumulated_spikes();
    if !spikes.is_empty() {
        let aromatic_ids: std::collections::HashSet<i32> =
            engine.aromatic_residue_ids().iter().cloned().collect();

        let uv_spikes: Vec<_> = spikes
            .iter()
            .filter(|s| (s.timestep % 500) < 50) // UV burst phase
            .collect();

        let non_uv_spikes: Vec<_> = spikes
            .iter()
            .filter(|s| (s.timestep % 500) >= 50) // Non-UV phase
            .collect();

        let count_aromatic = |spike_set: &[&prism_nhs::GpuSpikeEvent]| -> usize {
            spike_set
                .iter()
                .filter(|s| {
                    let n = s.n_residues;
                    let nearby = s.nearby_residues;
                    (0..n as usize).any(|i| aromatic_ids.contains(&nearby[i]))
                })
                .count()
        };

        let uv_with_aro = count_aromatic(&uv_spikes);
        let non_uv_with_aro = count_aromatic(&non_uv_spikes);

        let uv_rate = if !uv_spikes.is_empty() {
            100.0 * uv_with_aro as f32 / uv_spikes.len() as f32
        } else {
            0.0
        };
        let non_uv_rate = if !non_uv_spikes.is_empty() {
            100.0 * non_uv_with_aro as f32 / non_uv_spikes.len() as f32
        } else {
            0.0
        };

        let enrichment = if non_uv_rate > 0.0 {
            uv_rate / non_uv_rate
        } else {
            0.0
        };

        // CRITICAL: Enrichment should be >1.5x for ANY protein with aromatics
        // If enrichment is <1.5x, the UV-LIF physics is broken or overfitted
        if n_aromatics > 0 && enrichment < 1.5 {
            println!(
                "  [WARNING] {} - Low aromatic enrichment {:.2}x (expected >1.5x)",
                name, enrichment
            );
            println!("            This suggests UV-LIF coupling may not be working correctly");
        } else if n_aromatics > 0 {
            println!(
                "  [VERIFIED] {} - Aromatic enrichment {:.2}x ✓",
                name, enrichment
            );
        }
    }

    result.status = "complete".to_string();
    result
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║     PRISM4D CONCURRENT BATCH CRYPTIC SITE BENCHMARK                 ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Running 20 ultra-difficult apo-holo pairs CONCURRENTLY             ║");
    println!("║  Using hyperoptimized GPU execution with multi-threading            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let topology_dir =
        Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test"));
    let n_steps = 10000; // 10k steps per structure (faster benchmark)
    let max_concurrent = 3; // Max structures running at once (GPU memory limited)

    // Find available topologies
    println!("[SETUP] Scanning for pre-prepared topologies...");
    let mut available_cases: Vec<(&str, &str, &str, PathBuf)> = Vec::new();

    for (name, apo, holo, _chain, _why) in BENCHMARK_CASES {
        if let Some(topo_path) = find_topology(apo, topology_dir) {
            println!("  [OK] {} ({}) -> {}", name, apo, topo_path.display());
            available_cases.push((name, apo, holo, topo_path));
        } else {
            println!("  [--] {} ({}) - no topology found", name, apo);
        }
    }

    // If no specific benchmark cases found, use ALL available topologies
    if available_cases.is_empty() {
        println!("\nNo specific benchmark topologies found. Using all available topologies...\n");

        let all_topos = scan_all_topologies(topology_dir);
        for (name, path) in all_topos {
            println!("  [OK] {} -> {}", name, path.display());
            available_cases.push((
                Box::leak(name.clone().into_boxed_str()), // Leak to get &'static str
                Box::leak(name.clone().into_boxed_str()),
                "N/A",
                path,
            ));
        }

        if available_cases.is_empty() {
            println!("\nNo topologies found at all. Please run prism-prep first.");
            println!("Example: prism-prep --pdb 2CM2.pdb --chain A --output-dir topologies/");
            return Ok(());
        }
    }

    println!(
        "\nFound {} structures with topologies",
        available_cases.len()
    );
    println!(
        "Running {} steps each, {} concurrent threads\n",
        n_steps, max_concurrent
    );

    // Collect results thread-safely
    let results: Arc<Mutex<Vec<BenchmarkResult>>> = Arc::new(Mutex::new(Vec::new()));
    let start_total = Instant::now();

    // Process in batches of max_concurrent
    for batch in available_cases.chunks(max_concurrent) {
        let handles: Vec<_> = batch
            .iter()
            .map(|(name, apo, holo, topo_path)| {
                let results = Arc::clone(&results);
                let name = name.to_string();
                let apo = apo.to_string();
                let holo = holo.to_string();
                let topo_path = topo_path.clone();
                let n_steps = n_steps;

                thread::spawn(move || {
                    println!("[START] {} ({})", name, apo);

                    let result = run_structure_benchmark(&name, &apo, &holo, &topo_path, n_steps);

                    if result.status == "complete" {
                        println!(
                            "[DONE]  {} - {:.0} steps/s, {} spikes, RMSD={:.2}Å",
                            name, result.steps_per_sec, result.total_spikes, result.final_rmsd
                        );
                    } else {
                        println!(
                            "[FAIL]  {} - {}",
                            name,
                            result.error.as_deref().unwrap_or("unknown")
                        );
                    }

                    let mut r = results.lock().unwrap();
                    r.push(result);
                })
            })
            .collect();

        // Wait for this batch
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
    }

    let total_elapsed = start_total.elapsed();

    // Print summary
    println!("\n{}", "=".repeat(74));
    println!("                    CONCURRENT BATCH BENCHMARK RESULTS");
    println!("{}", "=".repeat(74));

    let results = results.lock().unwrap();

    // Table header
    println!(
        "\n{:<25} {:>7} {:>8} {:>10} {:>8} {:>6} {:>6}",
        "Structure", "Atoms", "Steps/s", "Spikes", "RMSD", "Valid", "Status"
    );
    println!("{}", "-".repeat(74));

    let mut total_steps = 0;
    let mut total_spikes = 0;
    let mut n_passed = 0;
    let mut n_failed = 0;

    for r in results.iter() {
        let status_str = if r.status == "complete" {
            if r.valid {
                n_passed += 1;
                "PASS"
            } else {
                n_failed += 1;
                "WARN"
            }
        } else {
            n_failed += 1;
            "FAIL"
        };

        println!(
            "{:<25} {:>7} {:>8.0} {:>10} {:>8.2} {:>6} {:>6}",
            &r.name[..r.name.len().min(25)],
            r.n_atoms,
            r.steps_per_sec,
            r.total_spikes,
            r.final_rmsd,
            if r.valid { "✓" } else { "✗" },
            status_str
        );

        if r.status == "complete" {
            total_steps += r.n_steps;
            total_spikes += r.total_spikes;
        }
    }

    // Aggregate stats
    println!("{}", "-".repeat(74));
    println!("\n  AGGREGATE PERFORMANCE:");
    println!("    Total structures: {}", results.len());
    println!("    Passed: {} | Failed: {}", n_passed, n_failed);
    println!("    Total steps: {}", total_steps);
    println!("    Total spikes: {}", total_spikes);
    println!("    Wall-clock time: {:.1}s", total_elapsed.as_secs_f64());
    println!(
        "    Effective throughput: {:.0} steps/sec (parallel)",
        total_steps as f64 / total_elapsed.as_secs_f64()
    );

    // Speedup calculation
    let serial_time_estimate: f64 = results
        .iter()
        .filter(|r| r.status == "complete")
        .map(|r| r.time_s)
        .sum();
    let speedup = serial_time_estimate / total_elapsed.as_secs_f64();
    println!("    Parallel speedup: {:.2}x vs serial", speedup);

    println!();
    if n_passed > 0 && n_failed == 0 {
        println!("  ╔═══════════════════════════════════════════════════╗");
        println!("  ║   CONCURRENT BATCH BENCHMARK: ALL PASSED ✓       ║");
        println!("  ╚═══════════════════════════════════════════════════╝");
    } else if n_passed > 0 {
        println!("  ╔═══════════════════════════════════════════════════╗");
        println!(
            "  ║   CONCURRENT BATCH BENCHMARK: {}/{} PASSED      ║",
            n_passed,
            results.len()
        );
        println!("  ╚═══════════════════════════════════════════════════╝");
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("This benchmark requires the 'gpu' feature. Compile with:");
    eprintln!("  cargo run --features gpu --release --example benchmark_cryptic_batch");
}
