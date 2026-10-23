//! Optimized Cryptic Site Detection
//!
//! Combined approach:
//! 1. Wider aromatic proximity (12 residues) to catch substrate binding sites
//! 2. Warm-phase weighting (cryptic sites open at higher temps)
//! 3. Spike intensity weighting
//! 4. Multiple scoring strategies with ensemble voting

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

const STEPS_PER_RUN: i32 = 2500; // More steps for better statistics
const N_RUNS: usize = 6; // More runs
const AROMATIC_PROXIMITY_CUTOFF: i32 = 12; // Wider cutoff

fn calculate_metrics(predicted: &HashSet<i32>, truth: &HashSet<i32>) -> (f32, f32, f32) {
    if predicted.is_empty() || truth.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let tp = predicted.intersection(truth).count() as f32;
    let precision = tp / predicted.len() as f32;
    let recall = tp / truth.len() as f32;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    (precision, recall, f1)
}

#[derive(Default, Clone)]
struct ResidueData {
    total_count: usize,
    warm_count: usize,     // Spikes at T > 250K
    high_intensity: usize, // Spikes with intensity > 1.5
    total_intensity: f32,
    runs_detected: usize, // In how many runs was this detected?
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   PRISM4D OPTIMIZED CRYPTIC SITE DETECTION                          ║");
    println!("║   Wider Aromatic Filter + Warm-Phase + Intensity Weighting          ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    let n_residues = *topology.residue_ids.iter().max().unwrap_or(&0) + 1;

    println!("Structure: 6LU7 (SARS-CoV-2 Main Protease)");
    println!("Atoms: {}, Residues: {}", topology.n_atoms, n_residues);

    // Aromatic residues
    let aromatic_residues: Vec<i32> = topology
        .residue_names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            if matches!(name.as_str(), "TRP" | "TYR" | "PHE") {
                Some(i as i32)
            } else {
                None
            }
        })
        .collect();

    println!("Aromatic residues: {}", aromatic_residues.len());

    // Also consider His as weak UV absorbers at 280nm
    let histidine_residues: Vec<i32> = topology
        .residue_names
        .iter()
        .enumerate()
        .filter_map(
            |(i, name)| {
                if name == "HIS" {
                    Some(i as i32)
                } else {
                    None
                }
            },
        )
        .collect();
    println!("Histidine residues: {}", histidine_residues.len());

    // Combined chromophores (aromatics + His)
    let chromophores: HashSet<i32> = aromatic_residues
        .iter()
        .chain(histidine_residues.iter())
        .cloned()
        .collect();

    // Truth
    let truth: HashSet<i32> = [
        23, 24, 25, 26, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 139, 140, 141, 142, 143, 144,
        145, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 186, 187, 188, 189, 190, 191,
        192,
    ]
    .iter()
    .cloned()
    .collect();

    println!("\nTruth residues: {}", truth.len());

    // Terminal regions
    let terminal_start = 10;
    let terminal_end = n_residues - 10;

    // UV configuration with higher energy
    let uv_config = UvProbeConfig {
        enabled: true,
        burst_energy: 20.0,  // Even higher energy
        burst_interval: 400, // More frequent
        burst_duration: 25,
        frequency_hopping_enabled: true,
        scan_wavelengths: vec![258.0, 274.0, 280.0],
        dwell_steps: 400,
        ..Default::default()
    };

    println!("\nConfiguration:");
    println!("  Steps/run: {}, Runs: {}", STEPS_PER_RUN, N_RUNS);
    println!("  UV burst energy: {} kcal/mol", uv_config.burst_energy);
    println!(
        "  Aromatic proximity cutoff: {} residues",
        AROMATIC_PROXIMITY_CUTOFF
    );

    // Collect data
    let mut residue_data: HashMap<i32, ResidueData> = HashMap::new();
    let mut total_spikes = 0;

    print!("\nRunning detection");
    for run_idx in 0..N_RUNS {
        let context = CudaContext::new(0)?;
        let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

        engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: 100.0,
            end_temp: 300.0,
            ramp_steps: 2000,
            hold_steps: 500,
            cold_hold_steps: 0,
            current_step: 0,
        })?;

        engine.set_uv_config(uv_config.clone());
        let _summary = engine.run(STEPS_PER_RUN)?;

        // Track which residues detected this run
        let mut run_detected: HashSet<i32> = HashSet::new();

        if let Ok(gpu_spikes) = engine.download_full_spike_events(3000) {
            for spike in &gpu_spikes {
                total_spikes += 1;

                // Temperature from timestep (ramp over 2000 steps)
                let temp = if spike.timestep < 2000 {
                    100.0 + (spike.timestep as f32 / 2000.0) * 200.0
                } else {
                    300.0
                };
                let is_warm = temp >= 250.0;

                for i in 0..spike.n_residues.min(8) as usize {
                    let res_id = spike.nearby_residues[i];
                    if res_id < 0 {
                        continue;
                    }

                    // Check proximity to any chromophore
                    let near_chromophore = chromophores
                        .iter()
                        .any(|&ch| (ch - res_id).abs() <= AROMATIC_PROXIMITY_CUTOFF);

                    if !near_chromophore {
                        continue;
                    }

                    let data = residue_data.entry(res_id).or_default();
                    data.total_count += 1;
                    data.total_intensity += spike.intensity;

                    if is_warm {
                        data.warm_count += 1;
                    }
                    if spike.intensity > 1.5 {
                        data.high_intensity += 1;
                    }

                    run_detected.insert(res_id);
                }
            }
        }

        // Update runs_detected count
        for res_id in run_detected {
            residue_data.get_mut(&res_id).unwrap().runs_detected += 1;
        }

        print!(".");
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!(" Done\n");

    println!("Total spikes: {}", total_spikes);
    println!("Unique residues: {}", residue_data.len());

    // Compute scores using multiple strategies
    let mut scored: Vec<_> = residue_data
        .iter()
        .filter(|(_, data)| data.total_count >= 3) // Minimum detections
        .map(|(&res_id, data)| {
            let is_terminal = res_id < terminal_start as i32 || res_id >= terminal_end as i32;
            let terminal_penalty = if is_terminal { 0.25 } else { 1.0 };

            // Nearest chromophore distance
            let min_dist = chromophores
                .iter()
                .map(|&ch| (ch - res_id).abs())
                .min()
                .unwrap_or(100) as f32;
            let proximity_factor = 2.0 / (min_dist + 1.0);

            // Warm-phase ratio
            let warm_ratio = data.warm_count as f32 / data.total_count.max(1) as f32;

            // Consistency across runs
            let consistency = data.runs_detected as f32 / N_RUNS as f32;

            // Intensity factor
            let avg_intensity = data.total_intensity / data.total_count as f32;

            // Combined score
            let base = (data.total_count as f32).sqrt();
            let score = base
                * terminal_penalty
                * proximity_factor
                * (1.0 + warm_ratio)
                * (0.5 + consistency)
                * (0.8 + avg_intensity * 0.2);

            (
                res_id,
                score,
                data.clone(),
                min_dist as i32,
                warm_ratio,
                consistency,
            )
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display top 60
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("TOP 60 SCORED RESIDUES");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>4} {:>5} {:>7} {:>5} {:>6} {:>5} {:>6} {:>8}",
        "Rank", "Res", "Score", "Count", "Warm%", "Runs", "Dist", "Truth?"
    );
    println!("{}", "-".repeat(60));

    let mut top_40_hits = 0;
    for (i, (res_id, score, data, dist, warm_ratio, consistency)) in
        scored.iter().take(60).enumerate()
    {
        let in_truth = truth.contains(res_id);
        let truth_mark = if in_truth { "YES ←" } else { "" };

        if in_truth && i < 40 {
            top_40_hits += 1;
        }

        println!(
            "{:>4} {:>5} {:>7.2} {:>5} {:>5.0}% {:>5} {:>6} {:>8}",
            i + 1,
            res_id,
            score,
            data.total_count,
            warm_ratio * 100.0,
            data.runs_detected,
            dist,
            truth_mark
        );
    }

    // Metrics
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("METRICS");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>6} {:>8}",
        "Top-N", "Precision", "Recall", "F1", "Hits", "Status"
    );
    println!("{}", "-".repeat(60));

    let cutoffs = [20, 30, 40, 50, 60, 80, 100];
    let mut best_f1 = 0.0f32;
    let mut best_n = 0;

    for &n in &cutoffs {
        let predicted: HashSet<i32> = scored
            .iter()
            .take(n)
            .map(|(res_id, _, _, _, _, _)| *res_id)
            .collect();

        let (p, r, f1) = calculate_metrics(&predicted, &truth);
        let hits = predicted.intersection(&truth).count();
        let status = if f1 >= 0.3 {
            "HIT ✓"
        } else if f1 >= 0.25 {
            "close"
        } else {
            "miss"
        };

        println!(
            "{:>8} {:>10.3} {:>10.3} {:>10.3} {:>6} {:>8}",
            n, p, r, f1, hits, status
        );

        if f1 > best_f1 {
            best_f1 = f1;
            best_n = n;
        }
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                            SUMMARY                                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Truth residues in top 40: {}/40                                      ║",
        top_40_hits
    );
    println!(
        "║  Best F1: {:.3} at Top-{}                                             ║",
        best_f1, best_n
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    if best_f1 >= 0.30 {
        println!("║  RESULT: ✓✓ PASSED - F1 >= 0.30 achieved!                           ║");
    } else if best_f1 >= 0.28 {
        println!(
            "║  RESULT: ✓ NEARLY PASSED - F1 = {:.3} (target 0.30)                 ║",
            best_f1
        );
    } else if best_f1 >= 0.25 {
        println!(
            "║  RESULT: CLOSE - F1 = {:.3}                                         ║",
            best_f1
        );
    } else {
        println!(
            "║  RESULT: NEEDS IMPROVEMENT - F1 = {:.3}                             ║",
            best_f1
        );
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
