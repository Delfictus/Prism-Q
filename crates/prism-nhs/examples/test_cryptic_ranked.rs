//! Test Cryptic Site Detection with Ranked Residues
//!
//! Instead of using all spike residues, rank by frequency and use top-N

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol};

const STEPS_PER_RUN: i32 = 2000;
const N_RUNS: usize = 5;

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
fn run_detection_ranked(topology_path: &Path) -> Result<Vec<(i32, usize)>> {
    let topology = PrismPrepTopology::load(topology_path)?;
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

        // Count spike occurrences per residue
        if let Ok(gpu_spikes) = engine.download_full_spike_events(2000) {
            for spike in gpu_spikes {
                for i in 0..spike.n_residues.min(8) as usize {
                    let res_id = spike.nearby_residues[i];
                    if res_id > 0 {
                        *residue_counts.entry(res_id).or_insert(0) += 1;
                    }
                }
            }
        }
        print!(".");
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!();

    // Sort by count descending
    let mut ranked: Vec<_> = residue_counts.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    Ok(ranked)
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║      PRISM4D CRYPTIC SITE DETECTION - RANKED RESIDUES                ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Test Case: 6LU7 (SARS-CoV-2 Mpro)
    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let truth: HashSet<i32> = [
        25, 26, 27, 41, 42, 43, 44, 45, 46, 47, 48, 49, 140, 141, 142, 143, 144, 145, 163, 164,
        165, 166, 167, 168, 169, 170, 171, 172, 187, 188, 189, 190, 191, 192,
    ]
    .iter()
    .cloned()
    .collect();

    println!("Structure: 6LU7 (SARS-CoV-2 Main Protease)");
    println!("Truth residues: {} residues in active site\n", truth.len());

    print!("Running Cryo-UV detection");
    let start = Instant::now();
    let ranked = run_detection_ranked(topology_path)?;
    let elapsed = start.elapsed();

    println!("\nCompleted in {:.1}s", elapsed.as_secs_f64());
    println!("\nTop 20 spike residues (by frequency):");
    println!("{:>4} {:>8} {:>8}", "Rank", "ResID", "Count");
    println!("{}", "-".repeat(24));

    for (i, (res_id, count)) in ranked.iter().take(20).enumerate() {
        let in_truth = if truth.contains(res_id) {
            " ← TRUTH"
        } else {
            ""
        };
        println!("{:>4} {:>8} {:>8}{}", i + 1, res_id, count, in_truth);
    }

    // Calculate metrics at different cutoffs
    println!("\n{}", "═".repeat(60));
    println!("PRECISION-RECALL AT DIFFERENT CUTOFFS");
    println!("{}", "═".repeat(60));
    println!(
        "{:>8} {:>10} {:>10} {:>10} {:>10}",
        "Top-N", "Precision", "Recall", "F1", "Status"
    );
    println!("{}", "-".repeat(60));

    let cutoffs = [20, 30, 40, 50, 60, 80, 100];
    let mut best_f1 = 0.0f32;
    let mut best_cutoff = 0;

    for &n in &cutoffs {
        let predicted: HashSet<i32> = ranked.iter().take(n).map(|(res_id, _)| *res_id).collect();

        let (precision, recall, f1) = calculate_metrics(&predicted, &truth);
        let status = if f1 > 0.3 { "HIT" } else { "miss" };

        println!(
            "{:>8} {:>10.3} {:>10.3} {:>10.3} {:>10}",
            n, precision, recall, f1, status
        );

        if f1 > best_f1 {
            best_f1 = f1;
            best_cutoff = n;
        }
    }

    println!("{}", "-".repeat(60));
    println!("\nBest F1: {:.3} at Top-{}", best_f1, best_cutoff);

    // Check hit@N (is any truth residue in top N?)
    println!("\n{}", "═".repeat(60));
    println!("HIT@N (Is top-N residue in truth set?)");
    println!("{}", "═".repeat(60));

    for n in [1, 3, 5, 10] {
        let top_n: HashSet<i32> = ranked.iter().take(n).map(|(r, _)| *r).collect();
        let hit = !top_n.is_disjoint(&truth);
        println!("  Hit@{}: {}", n, if hit { "YES ✓" } else { "NO" });
    }

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    if best_f1 > 0.3 {
        println!("║  RESULT: PASSED - F1 > 0.3 at optimal cutoff                        ║");
    } else {
        println!(
            "║  RESULT: NEEDS IMPROVEMENT - Best F1 = {:.3}                        ║",
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
