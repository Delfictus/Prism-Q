//! Final Cryptic Site Validation Test
//!
//! Uses correct 0-indexed residue IDs matching topology

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
    println!("║        PRISM4D CRYPTIC SITE VALIDATION (Final Test)                  ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    println!("Structure: 6LU7 (SARS-CoV-2 Main Protease)");
    println!(
        "Atoms: {}, Residues: {}\n",
        topology.n_atoms,
        topology.residue_ids.iter().max().unwrap_or(&0) + 1
    );

    // Truth: 6LU7 active site (0-indexed to match topology)
    // PDB His41 = topology 40, PDB Cys145 = topology 144, etc.
    // Active site spans: catalytic dyad (His41, Cys145) + substrate binding pocket
    let truth_0idx: HashSet<i32> = [
        // S1' subsite (PDB 24-27 = topo 23-26)
        23, 24, 25, 26, // Catalytic His41 region (PDB 40-50 = topo 39-49)
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        // Catalytic Cys145 region (PDB 140-146 = topo 139-145)
        139, 140, 141, 142, 143, 144, 145, // S1 subsite (PDB 163-173 = topo 162-172)
        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
        // S2 subsite (PDB 187-193 = topo 186-192)
        186, 187, 188, 189, 190, 191, 192,
    ]
    .iter()
    .cloned()
    .collect();

    println!(
        "Truth (0-indexed): {} residues in active site",
        truth_0idx.len()
    );

    // Run detection
    print!("Running Cryo-UV detection");
    let mut residue_counts: HashMap<i32, usize> = HashMap::new();

    for _run_idx in 0..N_RUNS {
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

        if let Ok(gpu_spikes) = engine.download_full_spike_events(2000) {
            for spike in gpu_spikes {
                for i in 0..spike.n_residues.min(8) as usize {
                    let res_id = spike.nearby_residues[i];
                    if res_id >= 0 {
                        // Include 0-indexed
                        *residue_counts.entry(res_id).or_insert(0) += 1;
                    }
                }
            }
        }
        print!(".");
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!(" Done\n");

    // Sort by count
    let mut ranked: Vec<_> = residue_counts.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1));

    // Show top residues with truth marking
    println!("{}", "═".repeat(60));
    println!("TOP 40 SPIKE RESIDUES (0-indexed)");
    println!("{}", "═".repeat(60));
    println!(
        "{:>4} {:>8} {:>8} {:>12}",
        "Rank", "ResID", "Count", "In Truth?"
    );
    println!("{}", "-".repeat(40));

    let mut hits_in_top_40 = 0;
    for (i, (&res_id, &count)) in ranked.iter().take(40).enumerate() {
        let in_truth = if truth_0idx.contains(&res_id) {
            hits_in_top_40 += 1;
            "YES ←"
        } else {
            ""
        };
        println!("{:>4} {:>8} {:>8} {:>12}", i + 1, res_id, count, in_truth);
    }

    // Metrics at different cutoffs
    println!("\n{}", "═".repeat(60));
    println!("PRECISION-RECALL AT CUTOFFS");
    println!("{}", "═".repeat(60));
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>8}",
        "Top-N", "Precision", "Recall", "F1", "Status"
    );
    println!("{}", "-".repeat(50));

    let cutoffs = [20, 30, 40, 50, 60, 80, 100];
    let mut best_f1 = 0.0f32;
    let mut best_n = 0;

    for &n in &cutoffs {
        let predicted: HashSet<i32> = ranked.iter().take(n).map(|(&res_id, _)| res_id).collect();

        let (precision, recall, f1) = calculate_metrics(&predicted, &truth_0idx);
        let status = if f1 > 0.3 { "HIT ✓" } else { "miss" };

        println!(
            "{:>8} {:>10.3} {:>10.3} {:>10.3} {:>8}",
            n, precision, recall, f1, status
        );

        if f1 > best_f1 {
            best_f1 = f1;
            best_n = n;
        }
    }

    // Hit@N metrics
    println!("\n{}", "═".repeat(60));
    println!("HIT@N (Is any top-N residue in truth set?)");
    println!("{}", "═".repeat(60));

    for n in [1, 3, 5, 10, 20] {
        let top_n: HashSet<i32> = ranked.iter().take(n).map(|(&r, _)| r).collect();
        let hit = !top_n.is_disjoint(&truth_0idx);
        let count = top_n.intersection(&truth_0idx).count();
        println!(
            "  Hit@{:2}: {} ({} truth residues)",
            n,
            if hit { "YES ✓" } else { "NO   " },
            count
        );
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                            SUMMARY                                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Truth residues in top 40: {}/{}                                      ║",
        hits_in_top_40,
        truth_0idx.len()
    );
    println!(
        "║  Best F1: {:.3} at Top-{}                                             ║",
        best_f1, best_n
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    if best_f1 > 0.3 {
        println!("║  RESULT: ✓ PASSED - Cryptic site detection working!                 ║");
    } else if best_f1 > 0.2 {
        println!("║  RESULT: PARTIAL - Detection working but below threshold            ║");
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
