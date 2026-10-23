//! Refined Cryptic Site Detection
//!
//! Improvements over previous tests:
//! 1. Filter out N/C-terminal residues (common false positives)
//! 2. Proper UV burst timing within each run
//! 3. Aromatic-adjacency weighting (cryptic sites near aromatics)
//! 4. Warm-phase emergence scoring

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

const STEPS_PER_RUN: i32 = 2000;
const N_RUNS: usize = 5;

// Temperature thresholds
const RAMP_STEPS: i32 = 1500; // Steps to ramp from 100K to 300K
const WARM_STEP_THRESHOLD: i32 = 1250; // ~233K and above

// UV burst parameters (must match engine config)
const UV_BURST_INTERVAL: i32 = 500;
const UV_BURST_DURATION: i32 = 20;

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

/// Check if a timestep is during a UV burst
fn is_uv_active(timestep: i32) -> bool {
    (timestep % UV_BURST_INTERVAL) < UV_BURST_DURATION
}

/// Get temperature from timestep (linear ramp 100K->300K over RAMP_STEPS)
fn temp_from_step(step: i32) -> f32 {
    if step >= RAMP_STEPS {
        300.0
    } else {
        100.0 + (step as f32 / RAMP_STEPS as f32) * 200.0
    }
}

#[derive(Default, Clone)]
struct ResidueScore {
    total_count: usize,
    cold_count: usize,     // Spikes at temp < 200K
    warm_count: usize,     // Spikes at temp >= 233K
    uv_burst_count: usize, // Spikes during UV burst
    near_aromatic_count: usize,
    total_intensity: f32,
}

impl ResidueScore {
    fn weighted_score(&self, is_terminal: bool) -> f32 {
        if self.total_count == 0 {
            return 0.0;
        }

        // Penalize terminal residues (known flexible regions)
        let terminal_penalty = if is_terminal { 0.3 } else { 1.0 };

        // Warm emergence: residues that appear primarily when warm
        let warm_ratio = self.warm_count as f32 / self.total_count as f32;
        let warm_boost = 1.0 + warm_ratio; // 1.0 to 2.0

        // UV correlation: higher weight for UV-correlated spikes
        let uv_ratio = self.uv_burst_count as f32 / self.total_count as f32;
        let uv_boost = 1.0 + uv_ratio * 2.0; // 1.0 to 3.0

        // Aromatic proximity: cryptic sites often near Trp/Tyr/Phe
        let arom_ratio = self.near_aromatic_count as f32 / self.total_count as f32;
        let arom_boost = 1.0 + arom_ratio * 0.5; // 1.0 to 1.5

        let base = (self.total_count as f32).sqrt() * (1.0 + self.total_intensity * 0.1);
        base * terminal_penalty * warm_boost * uv_boost * arom_boost
    }
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   PRISM4D REFINED CRYPTIC SITE DETECTION                            ║");
    println!("║   Terminal Filtering + UV Correlation + Warm Emergence              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    let n_residues = *topology.residue_ids.iter().max().unwrap_or(&0) + 1;

    println!("Structure: 6LU7 (SARS-CoV-2 Main Protease)");
    println!("Atoms: {}, Residues: {}", topology.n_atoms, n_residues);

    // Find aromatic residues
    let mut aromatic_residues: HashSet<i32> = HashSet::new();
    for (i, name) in topology.residue_names.iter().enumerate() {
        if matches!(name.as_str(), "TRP" | "TYR" | "PHE") {
            aromatic_residues.insert(i as i32);
        }
    }

    // Define terminal regions to penalize (first/last 15 residues typically flexible)
    let terminal_residues: HashSet<i32> = (0..15)
        .chain((n_residues - 15)..n_residues)
        .map(|r| r as i32)
        .collect();

    println!("Aromatic residues: {}", aromatic_residues.len());
    println!(
        "Terminal residues (penalized): {} to {}, {} to {}",
        0,
        14,
        n_residues - 15,
        n_residues - 1
    );

    // Truth set (0-indexed)
    let truth: HashSet<i32> = [
        23, 24, 25, 26, // S1' subsite
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, // Catalytic His41
        139, 140, 141, 142, 143, 144, 145, // Catalytic Cys145
        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, // S1 subsite
        186, 187, 188, 189, 190, 191, 192, // S2 subsite
    ]
    .iter()
    .cloned()
    .collect();

    println!("Truth (active site): {} residues\n", truth.len());

    // Enhanced UV config
    let uv_config = UvProbeConfig {
        enabled: true,
        burst_energy: 15.0,
        burst_interval: UV_BURST_INTERVAL,
        burst_duration: UV_BURST_DURATION,
        frequency_hopping_enabled: true,
        scan_wavelengths: vec![258.0, 274.0, 280.0],
        dwell_steps: 500,
        ..Default::default()
    };

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("UV Configuration:");
    println!("  Burst energy: {} kcal/mol", uv_config.burst_energy);
    println!(
        "  Burst interval: {} steps ({} ps)",
        UV_BURST_INTERVAL,
        UV_BURST_INTERVAL as f32 * 0.002
    );
    println!(
        "  Burst duration: {} steps ({} fs)",
        UV_BURST_DURATION,
        UV_BURST_DURATION as f32 * 2.0
    );
    println!("  UV bursts per run: {}", STEPS_PER_RUN / UV_BURST_INTERVAL);
    println!();

    // Collect spike data
    let mut residue_scores: HashMap<i32, ResidueScore> = HashMap::new();
    let mut total_spikes = 0;
    let mut uv_correlated_spikes = 0;

    print!("Running detection");
    for _run in 0..N_RUNS {
        let context = CudaContext::new(0)?;
        let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

        engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: 100.0,
            end_temp: 300.0,
            ramp_steps: RAMP_STEPS,
            hold_steps: STEPS_PER_RUN - RAMP_STEPS,
            cold_hold_steps: 0,
            current_step: 0,
        })?;

        engine.set_uv_config(uv_config.clone());

        let _summary = engine.run(STEPS_PER_RUN)?;

        // Download spike events
        if let Ok(gpu_spikes) = engine.download_full_spike_events(2000) {
            for spike in &gpu_spikes {
                total_spikes += 1;
                let temp = temp_from_step(spike.timestep);
                let uv_active = is_uv_active(spike.timestep);

                if uv_active {
                    uv_correlated_spikes += 1;
                }

                for i in 0..spike.n_residues.min(8) as usize {
                    let res_id = spike.nearby_residues[i];
                    if res_id >= 0 {
                        let score = residue_scores.entry(res_id).or_default();
                        score.total_count += 1;
                        score.total_intensity += spike.intensity;

                        if temp < 200.0 {
                            score.cold_count += 1;
                        }
                        if temp >= 233.0 {
                            score.warm_count += 1;
                        }
                        if uv_active {
                            score.uv_burst_count += 1;
                        }

                        // Check aromatic proximity (within 5 residues)
                        for &ar in &aromatic_residues {
                            if (ar - res_id).abs() <= 5 {
                                score.near_aromatic_count += 1;
                                break; // Count once per spike
                            }
                        }
                    }
                }
            }
        }
        print!(".");
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!(" Done\n");

    println!("Total spikes: {}", total_spikes);
    println!(
        "UV-correlated spikes: {} ({:.1}%)\n",
        uv_correlated_spikes,
        uv_correlated_spikes as f32 / total_spikes as f32 * 100.0
    );

    // Rank residues
    let mut ranked: Vec<_> = residue_scores
        .iter()
        .map(|(&res_id, score)| {
            let is_terminal = terminal_residues.contains(&res_id);
            (
                res_id,
                score.weighted_score(is_terminal),
                score.clone(),
                is_terminal,
            )
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display results
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("TOP 50 RANKED RESIDUES (Terminal-Penalized + UV-Weighted)");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>4} {:>6} {:>8} {:>5} {:>5} {:>5} {:>6} {:>8}",
        "Rank", "ResID", "Score", "Cold", "Warm", "UV", "Arom", "Truth?"
    );
    println!("{}", "-".repeat(70));

    let mut hits_in_top_40 = 0;
    let mut hits_in_top_50 = 0;
    for (i, (res_id, score, data, is_term)) in ranked.iter().take(50).enumerate() {
        let term_mark = if *is_term { "T" } else { "" };
        let truth_mark = if truth.contains(res_id) {
            if i < 40 {
                hits_in_top_40 += 1;
            }
            hits_in_top_50 += 1;
            "YES ←"
        } else {
            ""
        };

        println!(
            "{:>4} {:>5}{:1} {:>8.1} {:>5} {:>5} {:>5} {:>6} {:>8}",
            i + 1,
            res_id,
            term_mark,
            score,
            data.cold_count,
            data.warm_count,
            data.uv_burst_count,
            data.near_aromatic_count,
            truth_mark
        );
    }

    // Metrics
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("PRECISION-RECALL METRICS");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>8}",
        "Top-N", "Precision", "Recall", "F1", "Status"
    );
    println!("{}", "-".repeat(50));

    let cutoffs = [20, 30, 40, 50, 60, 80, 100];
    let mut best_f1 = 0.0f32;
    let mut best_n = 0;

    for &n in &cutoffs {
        let predicted: HashSet<i32> = ranked
            .iter()
            .take(n)
            .map(|(res_id, _, _, _)| *res_id)
            .collect();

        let (p, r, f1) = calculate_metrics(&predicted, &truth);
        let status = if f1 > 0.3 {
            "HIT ✓"
        } else if f1 > 0.2 {
            "near"
        } else {
            "miss"
        };

        println!(
            "{:>8} {:>10.3} {:>10.3} {:>10.3} {:>8}",
            n, p, r, f1, status
        );

        if f1 > best_f1 {
            best_f1 = f1;
            best_n = n;
        }
    }

    // Hit@N analysis
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("HIT@N ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════");

    for n in [1, 3, 5, 10, 20, 40] {
        let top_n: HashSet<i32> = ranked.iter().take(n).map(|(r, _, _, _)| *r).collect();
        let hits = top_n.intersection(&truth).count();
        let hit = hits > 0;
        println!(
            "  Hit@{:2}: {} ({} truth residues in top {})",
            n,
            if hit { "YES ✓" } else { "NO   " },
            hits,
            n
        );
    }

    // Which truth residues are we finding?
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("TRUTH RESIDUE DETECTION");
    println!("═══════════════════════════════════════════════════════════════════════");

    let mut truth_ranks: Vec<_> = truth
        .iter()
        .filter_map(|&tr| {
            ranked
                .iter()
                .position(|(r, _, _, _)| *r == tr)
                .map(|pos| (tr, pos + 1))
        })
        .collect();
    truth_ranks.sort_by_key(|&(_, rank)| rank);

    println!("Truth residues found (with rank):");
    for (res_id, rank) in &truth_ranks {
        let region = if *res_id <= 26 {
            "S1'"
        } else if *res_id <= 49 {
            "His41"
        } else if *res_id <= 145 {
            "Cys145"
        } else if *res_id <= 172 {
            "S1"
        } else {
            "S2"
        };
        println!("  Res {:>3} ({}): Rank {:>3}", res_id, region, rank);
    }

    let found_truth = truth_ranks.len();
    let missing = truth.len() - found_truth;
    println!("\nFound: {}/{} truth residues", found_truth, truth.len());
    if missing > 0 {
        let missing_residues: Vec<_> = truth
            .iter()
            .filter(|&tr| !truth_ranks.iter().any(|(r, _)| r == tr))
            .collect();
        println!("Missing: {:?}", missing_residues);
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                            SUMMARY                                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Truth residues in top 40: {}/40                                      ║",
        hits_in_top_40
    );
    println!(
        "║  Truth residues in top 50: {}/40                                      ║",
        hits_in_top_50
    );
    println!(
        "║  Best F1: {:.3} at Top-{}                                             ║",
        best_f1, best_n
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    if best_f1 > 0.3 {
        println!("║  RESULT: ✓ PASSED - Cryptic site detection working!                 ║");
    } else if best_f1 > 0.2 {
        println!("║  RESULT: PARTIAL - Near threshold, needs tuning                     ║");
    } else {
        println!("║  RESULT: NEEDS IMPROVEMENT                                          ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
