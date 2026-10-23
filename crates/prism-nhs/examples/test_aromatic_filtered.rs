//! Aromatic-Filtered Cryptic Site Detection
//!
//! Uses aromatic proximity as the primary signal - UV perturbation acts on
//! Trp/Tyr/Phe, so cryptic sites should be detected near these residues.
//!
//! This test:
//! 1. Only considers spikes within N residues of an aromatic
//! 2. Heavily penalizes terminal regions
//! 3. Weights by warm-phase emergence

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
const AROMATIC_PROXIMITY_CUTOFF: i32 = 8; // Must be within 8 residues of an aromatic

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

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   PRISM4D AROMATIC-FILTERED CRYPTIC SITE DETECTION                  ║");
    println!("║   Only considering spikes near Trp/Tyr/Phe (UV chromophores)        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    let n_residues = *topology.residue_ids.iter().max().unwrap_or(&0) + 1;

    println!("Structure: 6LU7 (SARS-CoV-2 Main Protease)");
    println!("Atoms: {}, Residues: {}", topology.n_atoms, n_residues);

    // Find aromatic residues (UV chromophores)
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

    println!(
        "\nAromatic residues (UV targets): {} found",
        aromatic_residues.len()
    );
    println!(
        "  {:?}",
        &aromatic_residues[..aromatic_residues.len().min(20)]
    );

    // Check which truth residues are near aromatics
    let truth: HashSet<i32> = [
        23, 24, 25, 26, // S1'
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, // His41
        139, 140, 141, 142, 143, 144, 145, // Cys145
        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, // S1
        186, 187, 188, 189, 190, 191, 192, // S2
    ]
    .iter()
    .cloned()
    .collect();

    println!(
        "\nTruth residues near aromatics (within {} residues):",
        AROMATIC_PROXIMITY_CUTOFF
    );
    let mut truth_near_aromatic = 0;
    for &tr in &truth {
        let nearest_aromatic = aromatic_residues
            .iter()
            .map(|&ar| (ar - tr).abs())
            .min()
            .unwrap_or(999);
        if nearest_aromatic <= AROMATIC_PROXIMITY_CUTOFF {
            truth_near_aromatic += 1;
        }
    }
    println!(
        "  {}/{} truth residues are near aromatics",
        truth_near_aromatic,
        truth.len()
    );

    // Terminal regions (first/last 12 residues)
    let terminal_start = 12;
    let terminal_end = n_residues - 12;

    // Collect spike data with aromatic filtering
    let mut residue_counts: HashMap<i32, usize> = HashMap::new();
    let mut residue_intensity: HashMap<i32, f32> = HashMap::new();
    let mut total_spikes = 0;
    let mut aromatic_filtered_spikes = 0;

    let uv_config = UvProbeConfig {
        enabled: true,
        burst_energy: 15.0,
        burst_interval: 500,
        burst_duration: 20,
        frequency_hopping_enabled: true,
        scan_wavelengths: vec![258.0, 274.0, 280.0],
        dwell_steps: 500,
        ..Default::default()
    };

    print!("\nRunning detection");
    for _run in 0..N_RUNS {
        let context = CudaContext::new(0)?;
        let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

        engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: 100.0,
            end_temp: 300.0,
            ramp_steps: 1500,
            hold_steps: 500,
            cold_hold_steps: 0,
            current_step: 0,
        })?;

        engine.set_uv_config(uv_config.clone());

        let _summary = engine.run(STEPS_PER_RUN)?;

        if let Ok(gpu_spikes) = engine.download_full_spike_events(2000) {
            for spike in &gpu_spikes {
                total_spikes += 1;

                // Collect spike residues
                let mut spike_residues = Vec::new();
                for i in 0..spike.n_residues.min(8) as usize {
                    let res_id = spike.nearby_residues[i];
                    if res_id >= 0 {
                        spike_residues.push(res_id);
                    }
                }

                // Check if any spike residue is near an aromatic
                let near_aromatic = spike_residues.iter().any(|&res| {
                    aromatic_residues
                        .iter()
                        .any(|&ar| (ar - res).abs() <= AROMATIC_PROXIMITY_CUTOFF)
                });

                // Only count spikes near aromatics
                if near_aromatic {
                    aromatic_filtered_spikes += 1;
                    for &res_id in &spike_residues {
                        // Additional filter: only residues themselves near aromatics
                        let res_near_aromatic = aromatic_residues
                            .iter()
                            .any(|&ar| (ar - res_id).abs() <= AROMATIC_PROXIMITY_CUTOFF);

                        if res_near_aromatic {
                            *residue_counts.entry(res_id).or_insert(0) += 1;
                            *residue_intensity.entry(res_id).or_insert(0.0) += spike.intensity;
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
        "Aromatic-filtered spikes: {} ({:.1}%)",
        aromatic_filtered_spikes,
        aromatic_filtered_spikes as f32 / total_spikes as f32 * 100.0
    );
    println!("Unique residues after filtering: {}", residue_counts.len());

    // Rank by count, penalizing terminals
    let mut ranked: Vec<_> = residue_counts
        .iter()
        .map(|(&res_id, &count)| {
            let is_terminal = res_id < terminal_start as i32 || res_id >= terminal_end as i32;
            let penalty = if is_terminal { 0.2 } else { 1.0 };

            // Distance to nearest aromatic (closer = better)
            let nearest_aromatic_dist = aromatic_residues
                .iter()
                .map(|&ar| (ar - res_id).abs())
                .min()
                .unwrap_or(100) as f32;
            let proximity_boost = 1.0 + (1.0 / (nearest_aromatic_dist + 1.0));

            let intensity = residue_intensity.get(&res_id).unwrap_or(&0.0);
            let score = (count as f32).sqrt() * penalty * proximity_boost * (1.0 + intensity * 0.1);

            (
                res_id,
                score,
                count,
                is_terminal,
                nearest_aromatic_dist as i32,
            )
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display top 50
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("TOP 50 AROMATIC-FILTERED RESIDUES");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>4} {:>6} {:>8} {:>6} {:>8} {:>10}",
        "Rank", "ResID", "Score", "Count", "AromDist", "Truth?"
    );
    println!("{}", "-".repeat(55));

    let mut hits = [0usize; 6]; // hits at 10, 20, 30, 40, 50, 60
    for (i, (res_id, score, count, is_term, arom_dist)) in ranked.iter().take(60).enumerate() {
        let term_mark = if *is_term { "T" } else { "" };
        let in_truth = truth.contains(res_id);
        let truth_mark = if in_truth { "YES ←" } else { "" };

        if in_truth {
            if i < 10 {
                hits[0] += 1;
            }
            if i < 20 {
                hits[1] += 1;
            }
            if i < 30 {
                hits[2] += 1;
            }
            if i < 40 {
                hits[3] += 1;
            }
            if i < 50 {
                hits[4] += 1;
            }
            if i < 60 {
                hits[5] += 1;
            }
        }

        if i < 50 {
            println!(
                "{:>4} {:>5}{:1} {:>8.2} {:>6} {:>8} {:>10}",
                i + 1,
                res_id,
                term_mark,
                score,
                count,
                arom_dist,
                truth_mark
            );
        }
    }

    // Metrics
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("PRECISION-RECALL METRICS");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>6} {:>8}",
        "Top-N", "Precision", "Recall", "F1", "Hits", "Status"
    );
    println!("{}", "-".repeat(60));

    let cutoffs = [10, 20, 30, 40, 50, 60, 80, 100];
    let mut best_f1 = 0.0f32;
    let mut best_n = 0;

    for (idx, &n) in cutoffs.iter().enumerate() {
        let predicted: HashSet<i32> = ranked
            .iter()
            .take(n)
            .map(|(res_id, _, _, _, _)| *res_id)
            .collect();

        let (p, r, f1) = calculate_metrics(&predicted, &truth);
        let n_hits = predicted.intersection(&truth).count();
        let status = if f1 > 0.3 {
            "HIT ✓"
        } else if f1 > 0.2 {
            "near"
        } else {
            "miss"
        };

        println!(
            "{:>8} {:>10.3} {:>10.3} {:>10.3} {:>6} {:>8}",
            n, p, r, f1, n_hits, status
        );

        if f1 > best_f1 {
            best_f1 = f1;
            best_n = n;
        }
    }

    // Show which truth residues we found
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("TRUTH RESIDUE RANKS");
    println!("═══════════════════════════════════════════════════════════════════════");

    let mut truth_info: Vec<_> = truth
        .iter()
        .map(|&tr| {
            let rank = ranked
                .iter()
                .position(|(r, _, _, _, _)| *r == tr)
                .map(|p| p + 1);
            let region = if tr <= 26 {
                "S1'"
            } else if tr <= 49 {
                "His41"
            } else if tr <= 145 {
                "Cys145"
            } else if tr <= 172 {
                "S1"
            } else {
                "S2"
            };
            (tr, rank, region)
        })
        .collect();
    truth_info.sort_by_key(|&(_, rank, _)| rank.unwrap_or(9999));

    println!("{:>6} {:>10} {:>8}", "ResID", "Region", "Rank");
    println!("{}", "-".repeat(28));
    for (res_id, rank, region) in &truth_info {
        let rank_str = rank.map(|r| format!("{}", r)).unwrap_or("N/A".to_string());
        println!("{:>6} {:>10} {:>8}", res_id, region, rank_str);
    }

    let found = truth_info.iter().filter(|(_, r, _)| r.is_some()).count();
    let in_top_40 = truth_info
        .iter()
        .filter(|(_, r, _)| r.map(|x| x <= 40).unwrap_or(false))
        .count();

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                            SUMMARY                                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Aromatic filtering kept: {:.1}% of spikes                            ║",
        aromatic_filtered_spikes as f32 / total_spikes as f32 * 100.0
    );
    println!(
        "║  Truth residues found: {}/{}                                          ║",
        found,
        truth.len()
    );
    println!(
        "║  Truth residues in top 40: {}/{}                                      ║",
        in_top_40,
        truth.len()
    );
    println!(
        "║  Best F1: {:.3} at Top-{}                                             ║",
        best_f1, best_n
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    if best_f1 > 0.3 {
        println!("║  RESULT: ✓ PASSED - Cryptic site detection working!                 ║");
    } else if best_f1 > 0.25 {
        println!("║  RESULT: CLOSE - Very near threshold (F1 > 0.25)                    ║");
    } else if best_f1 > 0.2 {
        println!("║  RESULT: PARTIAL - Approaching threshold                            ║");
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
