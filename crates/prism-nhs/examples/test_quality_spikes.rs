//! Test using Quality-Scored Spikes from Ensemble Snapshots
//!
//! Uses the spike quality scoring system to filter for high-confidence
//! UV-correlated spikes that are more likely to indicate cryptic sites.

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
    println!("║   PRISM4D CRYPTIC SITE DETECTION - Quality-Scored Spikes             ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    println!("Structure: 6LU7 (SARS-CoV-2 Mpro)");

    // Truth (0-indexed)
    let truth: HashSet<i32> = [
        23, 24, 25, 26, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 139, 140, 141, 142, 143, 144,
        145, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 186, 187, 188, 189, 190, 191,
        192,
    ]
    .iter()
    .cloned()
    .collect();

    // Collect spikes with quality scores
    let mut high_quality_residues: HashMap<i32, f32> = HashMap::new();
    let mut medium_quality_residues: HashMap<i32, f32> = HashMap::new();
    let mut all_residues: HashMap<i32, usize> = HashMap::new();

    print!("Running detection");
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

        let _summary = engine.run(STEPS_PER_RUN)?;

        // Get ensemble snapshots with quality-scored spikes
        for snapshot in engine.get_ensemble_snapshots() {
            for (spike, quality) in snapshot
                .trigger_spikes
                .iter()
                .zip(snapshot.spike_quality_scores.iter())
            {
                for &res_id in &spike.nearby_residues {
                    if res_id >= 0 {
                        *all_residues.entry(res_id).or_insert(0) += 1;

                        // Weight by UV correlation and overall confidence
                        let weight = quality.uv_correlation * quality.overall_confidence;

                        if quality.overall_confidence >= 0.5 {
                            *high_quality_residues.entry(res_id).or_insert(0.0) += weight;
                        } else if quality.overall_confidence >= 0.25 {
                            *medium_quality_residues.entry(res_id).or_insert(0.0) += weight;
                        }
                    }
                }
            }
        }
        print!(".");
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!(" Done\n");

    // Show high quality results
    println!("{}", "═".repeat(60));
    println!("HIGH CONFIDENCE SPIKE RESIDUES (confidence >= 0.5)");
    println!("{}", "═".repeat(60));

    let mut hq_ranked: Vec<_> = high_quality_residues.iter().collect();
    hq_ranked.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    if hq_ranked.is_empty() {
        println!("No high-confidence spikes found.");
    } else {
        println!(
            "{:>4} {:>8} {:>12} {:>12}",
            "Rank", "ResID", "Score", "In Truth?"
        );
        println!("{}", "-".repeat(40));

        for (i, (&res_id, &score)) in hq_ranked.iter().take(20).enumerate() {
            let in_truth = if truth.contains(&res_id) {
                "YES ←"
            } else {
                ""
            };
            println!(
                "{:>4} {:>8} {:>12.3} {:>12}",
                i + 1,
                res_id,
                score,
                in_truth
            );
        }

        // Metrics for high-quality
        let hq_predicted: HashSet<i32> = hq_ranked.iter().take(40).map(|(&r, _)| r).collect();
        let (p, r, f1) = calculate_metrics(&hq_predicted, &truth);
        println!(
            "\nTop-40 HQ: Precision={:.3}, Recall={:.3}, F1={:.3}",
            p, r, f1
        );
    }

    // Compare with all residues
    println!("\n{}", "═".repeat(60));
    println!("COMPARISON: All vs High-Quality Spikes");
    println!("{}", "═".repeat(60));

    let all_set: HashSet<i32> = all_residues.keys().cloned().collect();
    let hq_set: HashSet<i32> = high_quality_residues.keys().cloned().collect();

    let (p_all, r_all, f1_all) = calculate_metrics(&all_set, &truth);
    let (p_hq, r_hq, f1_hq) = calculate_metrics(&hq_set, &truth);

    println!(
        "All residues:    {} detected, F1={:.3}",
        all_set.len(),
        f1_all
    );
    println!(
        "High-quality:    {} detected, F1={:.3}",
        hq_set.len(),
        f1_hq
    );

    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    if f1_hq > 0.3 {
        println!("║  RESULT: ✓ High-quality spike filtering improves detection!         ║");
    } else {
        println!("║  RESULT: Quality scoring helps but more tuning needed               ║");
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
