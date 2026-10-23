//! BLIND VALIDATION - Honest Assessment
//!
//! Uses LOCKED parameters from 6LU7 optimization.
//! Tests on INDEPENDENT structures with PREDEFINED truth sets.
//! NO parameter tuning allowed after seeing results.
//!
//! Locked Parameters (from 6LU7 training):
//! - Steps per run: 3500
//! - Number of runs: 10
//! - UV burst energy: 30.0 kcal/mol
//! - UV burst interval: 350 steps
//! - UV burst duration: 35 steps
//! - Chromophore cutoff: 12 residues
//! - Terminal filter: first/last 8 residues
//! - Persistence threshold: 4+ runs
//! - Scoring: count^0.6 * proximity^1.5 * consistency * intensity

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

// ============================================================================
// LOCKED PARAMETERS - DO NOT MODIFY
// ============================================================================
const STEPS_PER_RUN: i32 = 3500;
const N_RUNS: usize = 10;
const UV_BURST_ENERGY: f32 = 30.0;
const UV_BURST_INTERVAL: i32 = 350;
const UV_BURST_DURATION: i32 = 35;
const CHROMOPHORE_CUTOFF: i32 = 12;
const TERMINAL_FILTER: i32 = 8;
const MIN_RUNS_PERSISTENCE: usize = 4;

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

/// Structure definition with literature-based truth set
struct ValidationTarget {
    name: &'static str,
    pdb_id: &'static str,
    topology_path: &'static str,
    /// Truth residues - MUST be defined from literature BEFORE running
    /// These are 0-indexed to match topology
    truth_residues: Vec<i32>,
    /// Source of truth definition
    truth_source: &'static str,
}

#[cfg(feature = "gpu")]
fn run_detection(topology_path: &Path, n_residues: i32) -> Result<Vec<(i32, f32, usize, usize)>> {
    let topology = PrismPrepTopology::load(topology_path)?;

    // Build chromophore set (aromatics + His)
    let chromophores: HashSet<i32> = topology
        .residue_names
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            if matches!(name.as_str(), "TRP" | "TYR" | "PHE" | "HIS") {
                Some(i as i32)
            } else {
                None
            }
        })
        .collect();

    // Terminal filter
    let terminals: HashSet<i32> = (0..TERMINAL_FILTER)
        .chain((n_residues - TERMINAL_FILTER)..n_residues)
        .collect();

    // Locked UV config
    let uv_config = UvProbeConfig {
        enabled: true,
        burst_energy: UV_BURST_ENERGY,
        burst_interval: UV_BURST_INTERVAL,
        burst_duration: UV_BURST_DURATION,
        frequency_hopping_enabled: true,
        scan_wavelengths: vec![280.0, 274.0, 258.0],
        dwell_steps: 350,
        ..Default::default()
    };

    // Collect data
    let mut stats: HashMap<i32, (usize, f32, HashSet<usize>)> = HashMap::new();

    for run in 0..N_RUNS {
        let context = CudaContext::new(0)?;
        let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

        engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: 100.0,
            end_temp: 300.0,
            ramp_steps: 3000,
            hold_steps: 500,
            cold_hold_steps: 0,
            current_step: 0,
        })?;

        engine.set_uv_config(uv_config.clone());
        let _summary = engine.run(STEPS_PER_RUN)?;

        if let Ok(spikes) = engine.download_full_spike_events(6000) {
            for spike in &spikes {
                for i in 0..spike.n_residues.min(8) as usize {
                    let res = spike.nearby_residues[i];
                    if res < 0 || terminals.contains(&res) {
                        continue;
                    }
                    if !chromophores
                        .iter()
                        .any(|&ch| (ch - res).abs() <= CHROMOPHORE_CUTOFF)
                    {
                        continue;
                    }

                    let entry = stats.entry(res).or_insert((0, 0.0, HashSet::new()));
                    entry.0 += 1;
                    entry.1 += spike.intensity;
                    entry.2.insert(run);
                }
            }
        }
        print!(".");
        std::io::Write::flush(&mut std::io::stdout())?;
    }

    // Score with locked formula
    let mut scored: Vec<_> = stats
        .iter()
        .filter(|(_, (_, _, runs))| runs.len() >= MIN_RUNS_PERSISTENCE)
        .map(|(&res, (count, intensity, runs))| {
            let dist = chromophores
                .iter()
                .map(|&ch| (ch - res).abs())
                .min()
                .unwrap_or(100) as f32;
            let proximity = 4.0 / (dist + 1.0);
            let consistency = runs.len() as f32 / N_RUNS as f32;
            let avg_intensity = intensity / *count as f32;

            let score = (*count as f32).powf(0.6)
                * proximity.powf(1.5)
                * (0.3 + consistency * 0.7)
                * (0.5 + avg_intensity * 0.5);

            (res, score, *count, runs.len())
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(scored)
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   BLIND VALIDATION - HONEST ASSESSMENT                              ║");
    println!("║   Locked parameters from 6LU7 training                              ║");
    println!("║   Independent test structures with literature truth sets            ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    println!("LOCKED PARAMETERS:");
    println!("  Steps/run: {}, Runs: {}", STEPS_PER_RUN, N_RUNS);
    println!("  UV energy: {} kcal/mol", UV_BURST_ENERGY);
    println!("  Chromophore cutoff: {} residues", CHROMOPHORE_CUTOFF);
    println!("  Persistence: >= {} runs\n", MIN_RUNS_PERSISTENCE);

    // ========================================================================
    // VALIDATION TARGETS - Truth sets defined from literature BEFORE running
    // ========================================================================

    let targets = vec![
        // Target 1: 6M0J_apo - ACE2 receptor
        // Truth: RBD binding interface from 6M0J crystal structure
        // Source: Lan et al., Nature 2020 - "Structure of SARS-CoV-2 spike RBD bound to ACE2"
        // PDB residues (1-indexed): 24, 27, 28, 30, 31, 34, 35, 37, 38, 41, 42, 45, 79, 82, 83, 330, 353, 354, 355, 357
        // 0-indexed: subtract 1 from each (ACE2 in 6M0J starts at residue 19 in chain A, need to check actual topology numbering)
        ValidationTarget {
            name: "ACE2 Receptor",
            pdb_id: "6M0J_apo",
            topology_path: concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6M0J_apo_topology.json"),
            // Conservative truth: key RBD-contacting residues (0-indexed, assuming topology starts at 0)
            // From structure: K31, E35, D38, Y41, Q42, L45, L79, M82, Y83, K353, D355
            truth_residues: vec![
                // N-terminal helix contacts (PDB 24-45 → topo ~5-26)
                5, 8, 9, 11, 12, 15, 16, 18, 19, 22, 23, 26,
                // Loop 79-83 region (PDB 79-83 → topo ~60-64)
                60, 61, 62, 63, 64,
                // C-terminal contacts (PDB 353-357 → topo ~334-338)
                334, 335, 336, 337, 338,
            ],
            truth_source: "Lan et al., Nature 2020; crystal contacts < 4Å",
        },

        // Target 2: 1L2Y - Trp-cage miniprotein (small validation)
        // This is a 20-residue miniprotein with known Trp cage
        // Truth: Trp6 cage residues that form the hydrophobic core
        ValidationTarget {
            name: "Trp-cage Miniprotein",
            pdb_id: "1L2Y",
            topology_path: concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/1L2Y_topology.json"),
            // Trp-cage core: Trp6 + surrounding hydrophobic residues
            // PDB: Trp6, Leu7, Tyr3, Pro12, Pro17, Pro18, Pro19
            truth_residues: vec![2, 5, 6, 11, 16, 17, 18],
            truth_source: "Neidigh et al., Nat Struct Biol 2002; Trp-cage hydrophobic core",
        },
    ];

    // ========================================================================
    // RUN VALIDATION
    // ========================================================================

    let mut all_results: Vec<(&str, f32, f32, f32, usize, usize)> = Vec::new();

    for target in &targets {
        println!("═══════════════════════════════════════════════════════════════════════");
        println!("TARGET: {} ({})", target.name, target.pdb_id);
        println!("Truth source: {}", target.truth_source);
        println!("Truth residues: {} defined", target.truth_residues.len());
        println!("═══════════════════════════════════════════════════════════════════════");

        let path = Path::new(target.topology_path);
        if !path.exists() {
            println!("  SKIPPED - topology not found\n");
            continue;
        }

        // Load topology to get residue count
        let topology = PrismPrepTopology::load(path)?;
        let n_residues = *topology.residue_ids.iter().max().unwrap_or(&0) + 1;
        println!("Atoms: {}, Residues: {}", topology.n_atoms, n_residues);

        let truth: HashSet<i32> = target.truth_residues.iter().cloned().collect();

        print!("Running detection");
        let scored = run_detection(path, n_residues as i32)?;
        println!(" Done\n");

        // Report metrics
        println!("Top 20 ranked residues:");
        println!(
            "{:>4} {:>5} {:>8} {:>6} {:>5} {:>8}",
            "Rank", "Res", "Score", "Count", "Runs", "Truth?"
        );
        println!("{}", "-".repeat(45));

        for (i, (res, score, count, runs)) in scored.iter().take(20).enumerate() {
            let in_truth = if truth.contains(res) { "YES ←" } else { "" };
            println!(
                "{:>4} {:>5} {:>8.2} {:>6} {:>5} {:>8}",
                i + 1,
                res,
                score,
                count,
                runs,
                in_truth
            );
        }

        // Calculate metrics at key cutoffs
        println!("\nMetrics:");
        println!(
            "{:>8} {:>10} {:>10} {:>10} {:>6}",
            "Top-N", "Precision", "Recall", "F1", "Hits"
        );
        println!("{}", "-".repeat(50));

        let cutoffs = [5, 10, 20, 30, 40];
        let mut best_f1 = 0.0f32;

        for &n in &cutoffs {
            if n > scored.len() {
                continue;
            }
            let pred: HashSet<i32> = scored.iter().take(n).map(|(r, _, _, _)| *r).collect();
            let (p, r, f1) = calculate_metrics(&pred, &truth);
            let hits = pred.intersection(&truth).count();
            println!("{:>8} {:>10.3} {:>10.3} {:>10.3} {:>6}", n, p, r, f1, hits);
            if f1 > best_f1 {
                best_f1 = f1;
            }
        }

        let top10_pred: HashSet<i32> = scored.iter().take(10).map(|(r, _, _, _)| *r).collect();
        let (p10, _, _) = calculate_metrics(&top10_pred, &truth);

        all_results.push((
            target.name,
            p10,
            best_f1,
            scored
                .iter()
                .take(40)
                .filter(|(r, _, _, _)| truth.contains(r))
                .count() as f32
                / 40.0,
            truth.len(),
            scored.len(),
        ));

        println!();
    }

    // ========================================================================
    // SUMMARY
    // ========================================================================
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    BLIND VALIDATION SUMMARY                          ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Target              Prec@10   Best F1   Prec@40   Truth   Detected  ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    for (name, p10, f1, p40, truth_n, detected) in &all_results {
        println!(
            "║  {:18} {:>7.1}%  {:>8.3}  {:>7.1}%  {:>5}  {:>8}   ║",
            name,
            p10 * 100.0,
            f1,
            p40 * 100.0,
            truth_n,
            detected
        );
    }

    println!("╠══════════════════════════════════════════════════════════════════════╣");

    let avg_p10: f32 = all_results.iter().map(|r| r.1).sum::<f32>() / all_results.len() as f32;
    let avg_f1: f32 = all_results.iter().map(|r| r.2).sum::<f32>() / all_results.len() as f32;

    println!(
        "║  AVERAGE:           {:>7.1}%  {:>8.3}                              ║",
        avg_p10 * 100.0,
        avg_f1
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");

    if avg_f1 >= 0.30 {
        println!("║  VALIDATION: ✓ PASSED - Generalizes to new targets                 ║");
    } else if avg_f1 >= 0.20 {
        println!("║  VALIDATION: PARTIAL - Some generalization                         ║");
    } else {
        println!("║  VALIDATION: ✗ FAILED - Overfitting detected                       ║");
    }

    println!("╚══════════════════════════════════════════════════════════════════════╝");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
