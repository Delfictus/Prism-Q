//! Test UV-Correlated Spike Detection
//!
//! Focus on spikes that occur during/after UV bursts on aromatic residues.
//! These are more likely to indicate cryptic sites.

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol};

const STEPS_PER_RUN: i32 = 2000;
const N_RUNS: usize = 5;

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
    println!("║    PRISM4D UV-CORRELATED CRYPTIC SITE DETECTION                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    println!("Structure: 6LU7 (SARS-CoV-2 Main Protease)");
    println!("Atoms: {}", topology.n_atoms);

    // Find aromatic residues (these are UV targets)
    let mut aromatic_residues: HashSet<i32> = HashSet::new();
    for (i, name) in topology.residue_names.iter().enumerate() {
        if matches!(name.as_str(), "TRP" | "TYR" | "PHE") {
            aromatic_residues.insert(i as i32 + 1);
        }
    }
    println!("Aromatic residues (UV targets): {:?}", aromatic_residues);

    // Known active site
    let truth: HashSet<i32> = [
        25, 26, 27, 41, 42, 43, 44, 45, 46, 47, 48, 49, 140, 141, 142, 143, 144, 145, 163, 164,
        165, 166, 167, 168, 169, 170, 171, 172, 187, 188, 189, 190, 191, 192,
    ]
    .iter()
    .cloned()
    .collect();

    println!("\nTruth (active site): {} residues", truth.len());
    println!(
        "Active site aromatics: {:?}",
        truth.intersection(&aromatic_residues).collect::<Vec<_>>()
    );

    // Run detection with focus on aromatic neighbors
    println!("\nRunning UV-correlated detection...");

    let mut aromatic_neighbor_counts: HashMap<i32, usize> = HashMap::new();
    let mut all_spike_residues: HashSet<i32> = HashSet::new();

    for run_idx in 0..N_RUNS {
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

        let _summary = engine.run(STEPS_PER_RUN)?;

        // Get spike events
        if let Ok(gpu_spikes) = engine.download_full_spike_events(2000) {
            for spike in &gpu_spikes {
                let mut spike_residues: Vec<i32> = Vec::new();
                for i in 0..spike.n_residues.min(8) as usize {
                    let res_id = spike.nearby_residues[i];
                    if res_id > 0 {
                        spike_residues.push(res_id);
                        all_spike_residues.insert(res_id);
                    }
                }

                // Check if any spike residue is near an aromatic
                let near_aromatic = spike_residues
                    .iter()
                    .any(|r| aromatic_residues.iter().any(|ar| (*ar - *r).abs() <= 5));

                // Only count residues that are near aromatics (UV-correlated)
                if near_aromatic {
                    for &res_id in &spike_residues {
                        *aromatic_neighbor_counts.entry(res_id).or_insert(0) += 1;
                    }
                }
            }
        }

        print!(".");
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!(" Done");

    // Sort aromatic neighbors by count
    let mut ranked_aromatic: Vec<_> = aromatic_neighbor_counts.into_iter().collect();
    ranked_aromatic.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n{}", "═".repeat(60));
    println!("TOP 30 UV-PROXIMAL SPIKE RESIDUES");
    println!("{}", "═".repeat(60));
    println!(
        "{:>4} {:>8} {:>8} {:>10}",
        "Rank", "ResID", "Count", "In Truth?"
    );
    println!("{}", "-".repeat(36));

    for (i, (res_id, count)) in ranked_aromatic.iter().take(30).enumerate() {
        let in_truth = if truth.contains(res_id) {
            "YES ←"
        } else {
            ""
        };
        println!("{:>4} {:>8} {:>8} {:>10}", i + 1, res_id, count, in_truth);
    }

    // Metrics at different cutoffs
    println!("\n{}", "═".repeat(60));
    println!("PRECISION-RECALL (UV-PROXIMAL RESIDUES)");
    println!("{}", "═".repeat(60));
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10}",
        "Top-N", "Precision", "Recall", "F1", "Status"
    );
    println!("{}", "-".repeat(60));

    let cutoffs = [10, 20, 30, 40, 50, 60, 80];
    let mut best_f1 = 0.0f32;

    for &n in &cutoffs {
        let predicted: HashSet<i32> = ranked_aromatic
            .iter()
            .take(n)
            .map(|(res_id, _)| *res_id)
            .collect();

        let (precision, recall, f1) = calculate_metrics(&predicted, &truth);
        let status = if f1 > 0.3 { "HIT ✓" } else { "miss" };

        println!(
            "{:>8} {:>10.3} {:>10.3} {:>10.3} {:>10}",
            n, precision, recall, f1, status
        );

        if f1 > best_f1 {
            best_f1 = f1;
        }
    }

    // Compare to all residues
    println!("\n{}", "═".repeat(60));
    println!("COMPARISON: All spikes vs UV-proximal");
    println!("{}", "═".repeat(60));

    let (p_all, r_all, f1_all) = calculate_metrics(&all_spike_residues, &truth);
    println!(
        "All spike residues:    {} residues, F1={:.3}",
        all_spike_residues.len(),
        f1_all
    );

    let uv_proximal: HashSet<i32> = ranked_aromatic.iter().take(40).map(|(r, _)| *r).collect();
    let (p_uv, r_uv, f1_uv) = calculate_metrics(&uv_proximal, &truth);
    println!(
        "UV-proximal (top 40):  {} residues, F1={:.3}",
        uv_proximal.len(),
        f1_uv
    );

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    if best_f1 > 0.3 {
        println!("║  RESULT: PASSED - UV-correlated detection works!                    ║");
    } else {
        println!(
            "║  RESULT: Best F1 = {:.3}                                             ║",
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
