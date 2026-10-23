//! Enhanced UV-Correlated Cryptic Site Detection
//!
//! This test addresses the data fusion gap and applies:
//! 1. Increased UV energy (higher fluence, more frequent bursts)
//! 2. Quality-weighted residue scoring (uv_correlation, aromatic_proximity)
//! 3. Temperature-differential analysis (cryo vs warm phase spikes)
//! 4. Focused wavelength targeting for aromatic residues

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

// Temperature thresholds for differential analysis
const CRYO_TEMP_MAX: f32 = 150.0; // Cold phase: 100-150K
const WARM_TEMP_MIN: f32 = 250.0; // Warm phase: 250-300K

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

/// Residue score accumulator with phase-differential tracking
#[derive(Default, Clone)]
struct ResidueScore {
    total_count: usize,
    cryo_count: usize,          // Spikes during cold phase (100-150K)
    warm_count: usize,          // Spikes during warm phase (250-300K)
    uv_correlated_count: usize, // Spikes during active UV burst
    aromatic_proximity_sum: f32,
    total_intensity: f32,
}

impl ResidueScore {
    /// Compute weighted score favoring:
    /// 1. Differential spikes (warm phase emergence)
    /// 2. UV correlation
    /// 3. High intensity
    fn weighted_score(&self) -> f32 {
        if self.total_count == 0 {
            return 0.0;
        }

        // Differential: warm-only spikes indicate cryptic site opening
        let differential = if self.cryo_count == 0 && self.warm_count > 0 {
            2.0 // Strong signal: only appears when warm
        } else if self.warm_count > self.cryo_count * 2 {
            1.5 // Good signal: much stronger when warm
        } else {
            1.0
        };

        // UV correlation boost
        let uv_ratio = self.uv_correlated_count as f32 / self.total_count as f32;
        let uv_boost = 1.0 + uv_ratio; // 1.0 to 2.0x

        // Aromatic proximity boost
        let aromatic_boost = 1.0 + (self.aromatic_proximity_sum / self.total_count as f32).min(1.0);

        // Combine
        let base = self.total_count as f32 + self.total_intensity * 0.5;
        base * differential * uv_boost * aromatic_boost
    }
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   PRISM4D ENHANCED UV-CORRELATED CRYPTIC SITE DETECTION             ║");
    println!("║   With Increased UV Energy + Temperature-Differential Analysis      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    println!("Structure: 6LU7 (SARS-CoV-2 Main Protease)");
    println!("Atoms: {}", topology.n_atoms);

    // Find aromatic residues (UV chromophores)
    let mut aromatic_residues: HashSet<i32> = HashSet::new();
    for (i, name) in topology.residue_names.iter().enumerate() {
        if matches!(name.as_str(), "TRP" | "TYR" | "PHE") {
            aromatic_residues.insert(i as i32); // 0-indexed
        }
    }
    println!(
        "Aromatic residues (UV targets): {} found",
        aromatic_residues.len()
    );

    // Truth set (0-indexed to match topology)
    let truth: HashSet<i32> = [
        // S1' subsite
        23, 24, 25, 26, // Catalytic His41 region
        39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, // Catalytic Cys145 region
        139, 140, 141, 142, 143, 144, 145, // S1 subsite
        162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, // S2 subsite
        186, 187, 188, 189, 190, 191, 192,
    ]
    .iter()
    .cloned()
    .collect();

    println!(
        "Truth (active site): {} residues (0-indexed)\n",
        truth.len()
    );

    // Enhanced UV configuration
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("UV CONFIGURATION (Enhanced)");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("• Burst energy:    15.0 kcal/mol (3x default)");
    println!("• Burst interval:  500 steps (2x frequency)");
    println!("• Burst duration:  20 steps (2x duration)");
    println!("• Wavelengths:     258nm (Phe), 274nm (Tyr), 280nm (Trp)");
    println!();

    // Collect spikes with enhanced tracking
    let mut residue_scores: HashMap<i32, ResidueScore> = HashMap::new();

    print!("Running enhanced detection");
    for _run in 0..N_RUNS {
        let context = CudaContext::new(0)?;
        let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

        // Configure temperature ramp
        engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: 100.0,
            end_temp: 300.0,
            ramp_steps: 1500,
            hold_steps: 500,
            cold_hold_steps: 0,
            current_step: 0,
        })?;

        // Configure enhanced UV probing
        let uv_config = UvProbeConfig {
            enabled: true,
            burst_energy: 15.0,  // 3x default (was 5.0)
            burst_interval: 500, // 2x frequency (was 1000)
            burst_duration: 20,  // 2x duration (was 10)
            frequency_hopping_enabled: true,
            scan_wavelengths: vec![258.0, 274.0, 280.0], // Phe, Tyr, Trp
            dwell_steps: 500,
            ..Default::default()
        };
        engine.set_uv_config(uv_config);

        // Run simulation
        let _summary = engine.run(STEPS_PER_RUN)?;

        // Download full spike events with residue mapping
        if let Ok(gpu_spikes) = engine.download_full_spike_events(2000) {
            for spike in &gpu_spikes {
                // Determine temperature phase from timestep
                // Temperature ramps from 100K to 300K over 1500 steps
                let temp = if spike.timestep < 1500 {
                    100.0 + (spike.timestep as f32 / 1500.0) * 200.0
                } else {
                    300.0
                };

                let is_cryo = temp < CRYO_TEMP_MAX;
                let is_warm = temp >= WARM_TEMP_MIN;

                // Check if UV burst was likely active
                // With burst_interval=500, burst_duration=20
                let steps_since_burst = spike.timestep % 500;
                let uv_active = steps_since_burst < 20;

                // Process each nearby residue
                for i in 0..spike.n_residues.min(8) as usize {
                    let res_id = spike.nearby_residues[i];
                    if res_id >= 0 {
                        let score = residue_scores.entry(res_id).or_default();
                        score.total_count += 1;
                        score.total_intensity += spike.intensity;

                        if is_cryo {
                            score.cryo_count += 1;
                        }
                        if is_warm {
                            score.warm_count += 1;
                        }
                        if uv_active {
                            score.uv_correlated_count += 1;
                        }

                        // Check aromatic proximity
                        for &ar in &aromatic_residues {
                            if (ar - res_id).abs() <= 5 {
                                score.aromatic_proximity_sum +=
                                    1.0 / ((ar - res_id).abs() as f32 + 1.0);
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

    // Rank by weighted score
    let mut ranked: Vec<_> = residue_scores
        .iter()
        .map(|(&res_id, score)| (res_id, score.weighted_score(), score.clone()))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display results
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("TOP 40 WEIGHTED SPIKE RESIDUES");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>4} {:>6} {:>8} {:>6} {:>6} {:>6} {:>10}",
        "Rank", "ResID", "Score", "Cryo", "Warm", "UV%", "Truth?"
    );
    println!("{}", "-".repeat(60));

    let mut hits_in_top_40 = 0;
    for (i, (res_id, score, data)) in ranked.iter().take(40).enumerate() {
        let uv_pct = if data.total_count > 0 {
            (data.uv_correlated_count as f32 / data.total_count as f32) * 100.0
        } else {
            0.0
        };

        let in_truth = if truth.contains(res_id) {
            hits_in_top_40 += 1;
            "YES ←"
        } else {
            ""
        };

        println!(
            "{:>4} {:>6} {:>8.1} {:>6} {:>6} {:>5.1}% {:>10}",
            i + 1,
            res_id,
            score,
            data.cryo_count,
            data.warm_count,
            uv_pct,
            in_truth
        );
    }

    // Metrics at cutoffs
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("PRECISION-RECALL AT CUTOFFS (Weighted Scoring)");
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
            .map(|(res_id, _, _)| *res_id)
            .collect();

        let (p, r, f1) = calculate_metrics(&predicted, &truth);
        let status = if f1 > 0.3 { "HIT ✓" } else { "miss" };

        println!(
            "{:>8} {:>10.3} {:>10.3} {:>10.3} {:>8}",
            n, p, r, f1, status
        );

        if f1 > best_f1 {
            best_f1 = f1;
            best_n = n;
        }
    }

    // Temperature differential analysis
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("TEMPERATURE DIFFERENTIAL ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════");

    // Find residues that appear mainly in warm phase (cryptic site opening)
    let mut warm_differential: Vec<_> = residue_scores
        .iter()
        .filter(|(_, s)| s.warm_count > s.cryo_count * 2 && s.total_count >= 5)
        .map(|(&res_id, score)| {
            (
                res_id,
                score.warm_count as f32 / (score.cryo_count as f32 + 1.0),
            )
        })
        .collect();
    warm_differential.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Residues appearing preferentially in WARM phase (cryptic opening signal):");
    println!("{:>6} {:>12} {:>10}", "ResID", "Warm/Cryo", "Truth?");
    println!("{}", "-".repeat(30));

    for (res_id, ratio) in warm_differential.iter().take(20) {
        let in_truth = if truth.contains(res_id) {
            "YES ←"
        } else {
            ""
        };
        println!("{:>6} {:>12.1}x {:>10}", res_id, ratio, in_truth);
    }

    // UV correlation analysis
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("UV CORRELATION ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════");

    let mut high_uv_corr: Vec<_> = residue_scores
        .iter()
        .filter(|(_, s)| s.total_count >= 5)
        .map(|(&res_id, score)| {
            let uv_ratio = score.uv_correlated_count as f32 / score.total_count as f32;
            (res_id, uv_ratio, score.uv_correlated_count)
        })
        .collect();
    high_uv_corr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("Residues with highest UV correlation (spikes during UV bursts):");
    println!(
        "{:>6} {:>12} {:>10} {:>10}",
        "ResID", "UV%", "UV Count", "Truth?"
    );
    println!("{}", "-".repeat(42));

    for (res_id, uv_ratio, uv_count) in high_uv_corr.iter().take(20) {
        let in_truth = if truth.contains(res_id) {
            "YES ←"
        } else {
            ""
        };
        println!(
            "{:>6} {:>11.1}% {:>10} {:>10}",
            res_id,
            uv_ratio * 100.0,
            uv_count,
            in_truth
        );
    }

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                            SUMMARY                                    ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Truth residues in top 40: {}/{}                                      ║",
        hits_in_top_40,
        truth.len()
    );
    println!(
        "║  Best F1: {:.3} at Top-{}                                             ║",
        best_f1, best_n
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    if best_f1 > 0.3 {
        println!("║  RESULT: ✓ PASSED - Enhanced UV detection working!                  ║");
    } else if best_f1 > 0.2 {
        println!("║  RESULT: PARTIAL - Improved but needs more tuning                   ║");
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
