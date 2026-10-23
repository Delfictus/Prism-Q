//! Test Cryptic Site Detection with Truth Comparison
//!
//! Tests the full pipeline on a known cryptic site case:
//! - Runs Cryo-UV probing on apo structure
//! - Compares detected spike residues to known binding site
//! - Reports F1/precision/recall

use anyhow::{Context, Result};
use std::collections::HashSet;
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

/// Calculate precision, recall, F1 between predicted and truth residues
fn calculate_metrics(predicted: &HashSet<i32>, truth: &HashSet<i32>) -> (f32, f32, f32) {
    if predicted.is_empty() || truth.is_empty() {
        return (0.0, 0.0, 0.0);
    }

    let true_positives = predicted.intersection(truth).count() as f32;
    let precision = true_positives / predicted.len() as f32;
    let recall = true_positives / truth.len() as f32;
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };

    (precision, recall, f1)
}

#[cfg(feature = "gpu")]
fn run_detection(topology_path: &Path) -> Result<HashSet<i32>> {
    let topology = PrismPrepTopology::load(topology_path)?;
    let mut all_spike_residues: HashSet<i32> = HashSet::new();
    let mut max_rmsd = 0.0f32;

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

        let ref_pos = engine.get_positions()?;
        let _summary = engine.run(STEPS_PER_RUN)?;
        let final_pos = engine.get_positions()?;
        let rmsd = compute_rmsd(&ref_pos, &final_pos);
        max_rmsd = max_rmsd.max(rmsd);

        // Collect spike residues
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

        print!(".");
        std::io::Write::flush(&mut std::io::stdout())?;
    }
    println!(" RMSD={:.2}Å", max_rmsd);

    Ok(all_spike_residues)
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║         PRISM4D CRYPTIC SITE DETECTION TEST                          ║");
    println!("║         Cryo-UV + LIF Neuromorphic Detection                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Test Case 1: 6LU7 (SARS-CoV-2 Mpro) - Known active site
    // Truth: Catalytic dyad His41, Cys145 and surrounding residues
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("TEST 1: 6LU7 (SARS-CoV-2 Main Protease)");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Known binding site residues: 25-27, 41-49, 140-145, 163-172, 187-192");
    println!();

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    if !topology_path.exists() {
        println!("Topology not found: {}", topology_path.display());
        return Ok(());
    }

    // Known binding site residues for 6LU7 (SARS-CoV-2 Mpro active site)
    let truth_6lu7: HashSet<i32> = [
        25, 26, 27, // S1' subsite
        41, 42, 43, 44, 45, 46, 47, 48, 49, // Catalytic His41 region
        140, 141, 142, 143, 144, 145, // Catalytic Cys145 region
        163, 164, 165, 166, 167, 168, 169, 170, 171, 172, // S1 subsite
        187, 188, 189, 190, 191, 192, // S2 subsite
    ]
    .iter()
    .cloned()
    .collect();

    print!("Running Cryo-UV detection");
    let start = Instant::now();
    let detected = run_detection(topology_path)?;
    let elapsed = start.elapsed();

    println!(
        "\nDetected {} spike residues in {:.1}s",
        detected.len(),
        elapsed.as_secs_f64()
    );

    // Show overlap with truth
    let overlap: HashSet<_> = detected.intersection(&truth_6lu7).cloned().collect();
    println!(
        "Truth residues:    {:?}",
        truth_6lu7.iter().collect::<Vec<_>>()
    );
    println!(
        "Detected overlap:  {:?}",
        overlap.iter().collect::<Vec<_>>()
    );

    let (precision, recall, f1) = calculate_metrics(&detected, &truth_6lu7);
    println!("\n┌─────────────────────────────────────────┐");
    println!("│  Precision: {:.3}                        │", precision);
    println!("│  Recall:    {:.3}                        │", recall);
    println!("│  F1 Score:  {:.3}                        │", f1);
    println!("└─────────────────────────────────────────┘");

    let hit = f1 > 0.3;
    println!(
        "\nResult: {} (F1 > 0.3 = HIT)",
        if hit { "✓ HIT" } else { "✗ MISS" }
    );

    // Test Case 2: 6M0J (ACE2-RBD complex) - RBD binding interface
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("TEST 2: 6M0J_apo (ACE2 receptor)");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("Known RBD binding interface: 24-42, 79-83, 324-330, 353-357");
    println!();

    let topology_path_ace2 = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6M0J_apo_topology.json"),
    );

    if !topology_path_ace2.exists() {
        println!("Topology not found: {}", topology_path_ace2.display());
        return Ok(());
    }

    // Known ACE2-RBD binding interface residues
    let truth_ace2: HashSet<i32> = [
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        42, // N-terminal helix
        79, 80, 81, 82, 83, // Loop region
        324, 325, 326, 327, 328, 329, 330, // Beta sheet
        353, 354, 355, 356, 357, // C-terminal region
    ]
    .iter()
    .cloned()
    .collect();

    print!("Running Cryo-UV detection");
    let start = Instant::now();
    let detected_ace2 = run_detection(topology_path_ace2)?;
    let elapsed = start.elapsed();

    println!(
        "\nDetected {} spike residues in {:.1}s",
        detected_ace2.len(),
        elapsed.as_secs_f64()
    );

    let overlap_ace2: HashSet<_> = detected_ace2.intersection(&truth_ace2).cloned().collect();
    println!(
        "Truth residues:    {:?}",
        truth_ace2.iter().collect::<Vec<_>>()
    );
    println!(
        "Detected overlap:  {:?}",
        overlap_ace2.iter().collect::<Vec<_>>()
    );

    let (precision2, recall2, f12) = calculate_metrics(&detected_ace2, &truth_ace2);
    println!("\n┌─────────────────────────────────────────┐");
    println!("│  Precision: {:.3}                        │", precision2);
    println!("│  Recall:    {:.3}                        │", recall2);
    println!("│  F1 Score:  {:.3}                        │", f12);
    println!("└─────────────────────────────────────────┘");

    let hit2 = f12 > 0.3;
    println!(
        "\nResult: {} (F1 > 0.3 = HIT)",
        if hit2 { "✓ HIT" } else { "✗ MISS" }
    );

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                         TEST SUMMARY                                  ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Structure          F1      Status                                   ║");
    println!("║  ─────────────────────────────────────                               ║");
    println!(
        "║  6LU7 (Mpro)       {:.3}    {}                                     ║",
        f1,
        if hit { "HIT " } else { "MISS" }
    );
    println!(
        "║  6M0J (ACE2)       {:.3}    {}                                     ║",
        f12,
        if hit2 { "HIT " } else { "MISS" }
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    let hits = (if hit { 1 } else { 0 }) + (if hit2 { 1 } else { 0 });
    println!(
        "║  Detection Rate: {}/2 ({:.0}%)                                        ║",
        hits,
        hits as f32 * 50.0
    );
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
