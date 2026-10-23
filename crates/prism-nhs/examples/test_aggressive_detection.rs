//! Aggressive Cryptic Site Detection
//!
//! Maximizes detection performance through:
//! 1. Post-UV correlation (spikes 20-200 steps AFTER UV burst = energy transfer time)
//! 2. Strict aromatic proximity (≤6 residues from Trp/Tyr/Phe)
//! 3. Spike persistence (must appear in multiple runs)
//! 4. Warm-only spikes (cryptic sites open at high temp)
//! 5. Intensity thresholding (high-intensity spikes only)
//! 6. Extended simulation for better statistics

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

// Aggressive configuration
const STEPS_PER_RUN: i32 = 3000;
const N_RUNS: usize = 8;
const AROMATIC_CUTOFF: i32 = 6; // Strict: must be within 6 residues of aromatic
const MIN_INTENSITY: f32 = 0.8; // Only high-intensity spikes
const MIN_RUNS_DETECTED: usize = 3; // Must appear in at least 3 runs
const UV_BURST_INTERVAL: i32 = 400;
const UV_BURST_DURATION: i32 = 30;
const POST_UV_WINDOW_START: i32 = 20; // Start looking 20 steps after burst
const POST_UV_WINDOW_END: i32 = 200; // End looking 200 steps after burst

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

/// Check if spike occurred in post-UV window (after energy transfer)
fn is_post_uv_spike(timestep: i32) -> bool {
    let steps_since_burst = timestep % UV_BURST_INTERVAL;
    steps_since_burst >= POST_UV_WINDOW_START && steps_since_burst <= POST_UV_WINDOW_END
}

/// Check if spike occurred during UV burst
fn is_during_uv_spike(timestep: i32) -> bool {
    let steps_since_burst = timestep % UV_BURST_INTERVAL;
    steps_since_burst < UV_BURST_DURATION
}

#[derive(Default, Clone)]
struct AggregatedResidueData {
    total_spikes: usize,
    post_uv_spikes: usize,   // Spikes in post-UV window (key signal!)
    during_uv_spikes: usize, // Spikes during UV burst
    high_intensity_spikes: usize,
    warm_spikes: usize, // Spikes at T > 250K
    runs_detected: HashSet<usize>,
    total_intensity: f32,
    max_intensity: f32,
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   PRISM4D AGGRESSIVE CRYPTIC SITE DETECTION                         ║");
    println!("║   Post-UV Correlation + Strict Aromatic Filter + Persistence        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    let n_residues = *topology.residue_ids.iter().max().unwrap_or(&0) + 1;

    println!("Structure: 6LU7 (SARS-CoV-2 Main Protease)");
    println!("Atoms: {}, Residues: {}", topology.n_atoms, n_residues);

    // Build aromatic residue set
    let aromatic_residues: HashSet<i32> = topology
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

    // Also include Histidine (weak UV absorber, but His41 is catalytic!)
    let histidine_residues: HashSet<i32> = topology
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

    let chromophores: HashSet<i32> = aromatic_residues
        .iter()
        .chain(histidine_residues.iter())
        .cloned()
        .collect();

    println!(
        "Chromophores: {} aromatics + {} histidines = {}",
        aromatic_residues.len(),
        histidine_residues.len(),
        chromophores.len()
    );

    // Truth set (0-indexed)
    let truth: HashSet<i32> = [
        23, 24, 25, 26, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 139, 140, 141, 142, 143, 144,
        145, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 186, 187, 188, 189, 190, 191,
        192,
    ]
    .iter()
    .cloned()
    .collect();

    // Terminal regions (first/last 10 residues - often flexible noise)
    let terminal_residues: HashSet<i32> = (0..10)
        .chain((n_residues as i32 - 10)..(n_residues as i32))
        .collect();

    println!("Truth residues: {}", truth.len());
    println!("\nConfiguration:");
    println!("  Steps/run: {}, Runs: {}", STEPS_PER_RUN, N_RUNS);
    println!("  Aromatic cutoff: {} residues", AROMATIC_CUTOFF);
    println!("  Min intensity: {}", MIN_INTENSITY);
    println!("  Min runs detected: {}", MIN_RUNS_DETECTED);
    println!(
        "  Post-UV window: {}-{} steps after burst",
        POST_UV_WINDOW_START, POST_UV_WINDOW_END
    );

    // Enhanced UV config
    let uv_config = UvProbeConfig {
        enabled: true,
        burst_energy: 25.0, // High energy
        burst_interval: UV_BURST_INTERVAL,
        burst_duration: UV_BURST_DURATION,
        frequency_hopping_enabled: true,
        scan_wavelengths: vec![280.0, 274.0, 258.0], // Trp, Tyr, Phe
        dwell_steps: 400,
        ..Default::default()
    };

    // Collect data
    let mut residue_data: HashMap<i32, AggregatedResidueData> = HashMap::new();
    let mut total_spikes = 0;
    let mut filtered_spikes = 0;

    print!("\nRunning {} runs", N_RUNS);
    for run_idx in 0..N_RUNS {
        let context = CudaContext::new(0)?;
        let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

        engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: 100.0,
            end_temp: 300.0,
            ramp_steps: 2500,
            hold_steps: 500,
            cold_hold_steps: 0,
            current_step: 0,
        })?;

        engine.set_uv_config(uv_config.clone());
        let _summary = engine.run(STEPS_PER_RUN)?;

        if let Ok(gpu_spikes) = engine.download_full_spike_events(5000) {
            for spike in &gpu_spikes {
                total_spikes += 1;

                // Temperature from timestep
                let temp = if spike.timestep < 2500 {
                    100.0 + (spike.timestep as f32 / 2500.0) * 200.0
                } else {
                    300.0
                };
                let is_warm = temp >= 250.0;

                // Check UV timing
                let is_post_uv = is_post_uv_spike(spike.timestep);
                let is_during_uv = is_during_uv_spike(spike.timestep);

                // Process each residue
                for i in 0..spike.n_residues.min(8) as usize {
                    let res_id = spike.nearby_residues[i];
                    if res_id < 0 {
                        continue;
                    }

                    // STRICT: Must be within AROMATIC_CUTOFF of a chromophore
                    let near_chromophore = chromophores
                        .iter()
                        .any(|&ch| (ch - res_id).abs() <= AROMATIC_CUTOFF);

                    if !near_chromophore {
                        continue;
                    }

                    // Skip terminal residues
                    if terminal_residues.contains(&res_id) {
                        continue;
                    }

                    filtered_spikes += 1;

                    let data = residue_data.entry(res_id).or_default();
                    data.total_spikes += 1;
                    data.total_intensity += spike.intensity;
                    data.max_intensity = data.max_intensity.max(spike.intensity);
                    data.runs_detected.insert(run_idx);

                    if is_warm {
                        data.warm_spikes += 1;
                    }
                    if is_post_uv {
                        data.post_uv_spikes += 1;
                    }
                    if is_during_uv {
                        data.during_uv_spikes += 1;
                    }
                    if spike.intensity >= MIN_INTENSITY {
                        data.high_intensity_spikes += 1;
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
        "After aromatic + terminal filter: {} ({:.1}%)",
        filtered_spikes,
        filtered_spikes as f32 / total_spikes as f32 * 100.0
    );

    // Apply persistence filter
    let persistent_residues: Vec<_> = residue_data
        .iter()
        .filter(|(_, data)| data.runs_detected.len() >= MIN_RUNS_DETECTED)
        .collect();

    println!(
        "After persistence filter (>= {} runs): {} residues",
        MIN_RUNS_DETECTED,
        persistent_residues.len()
    );

    // Score residues with aggressive weighting
    let mut scored: Vec<_> = persistent_residues
        .iter()
        .map(|(&res_id, data)| {
            // Distance to nearest chromophore
            let min_dist = chromophores
                .iter()
                .map(|&ch| (ch - res_id).abs())
                .min()
                .unwrap_or(100) as f32;

            // Post-UV ratio - THE KEY SIGNAL!
            // Spikes that occur after UV energy transfer are the real cryptic site indicators
            let post_uv_ratio = data.post_uv_spikes as f32 / data.total_spikes.max(1) as f32;

            // Warm ratio
            let warm_ratio = data.warm_spikes as f32 / data.total_spikes.max(1) as f32;

            // High intensity ratio
            let intensity_ratio =
                data.high_intensity_spikes as f32 / data.total_spikes.max(1) as f32;

            // Consistency (detected in multiple runs)
            let consistency = data.runs_detected.len() as f32 / N_RUNS as f32;

            // AGGRESSIVE SCORING:
            // - Post-UV is the PRIMARY signal (3x weight)
            // - Warm-only is REQUIRED (2x weight)
            // - Intensity matters (1.5x weight)
            // - Proximity to chromophore (1.5x weight)
            // - Consistency (1.5x weight)

            let proximity_score = 3.0 / (min_dist + 1.0); // Closer = better

            let score = (data.total_spikes as f32).sqrt()
                * (1.0 + post_uv_ratio * 3.0)    // Post-UV: huge boost
                * (1.0 + warm_ratio * 2.0)       // Warm-only: big boost
                * (1.0 + intensity_ratio * 1.5)  // High intensity: moderate boost
                * proximity_score                 // Chromophore proximity
                * (0.5 + consistency * 1.5); // Consistency across runs

            (
                res_id,
                score,
                data,
                min_dist as i32,
                post_uv_ratio,
                warm_ratio,
                consistency,
            )
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Display top results
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("TOP 60 AGGRESSIVELY FILTERED RESIDUES");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>4} {:>5} {:>8} {:>5} {:>6} {:>6} {:>6} {:>4} {:>8}",
        "Rank", "Res", "Score", "Count", "PostUV", "Warm%", "Runs", "Dist", "Truth?"
    );
    println!("{}", "-".repeat(70));

    let mut top_40_hits = 0;
    for (i, (res_id, score, data, dist, post_uv, warm, consistency)) in
        scored.iter().take(60).enumerate()
    {
        let in_truth = truth.contains(res_id);
        let truth_mark = if in_truth { "YES ←" } else { "" };
        if in_truth && i < 40 {
            top_40_hits += 1;
        }

        println!(
            "{:>4} {:>5} {:>8.2} {:>5} {:>5.0}% {:>5.0}% {:>6} {:>4} {:>8}",
            i + 1,
            res_id,
            score,
            data.total_spikes,
            post_uv * 100.0,
            warm * 100.0,
            data.runs_detected.len(),
            dist,
            truth_mark
        );
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

    let cutoffs = [10, 20, 30, 40, 50, 60, 80];
    let mut best_f1 = 0.0f32;
    let mut best_n = 0;

    for &n in &cutoffs {
        let predicted: HashSet<i32> = scored
            .iter()
            .take(n)
            .map(|(res_id, _, _, _, _, _, _)| *res_id)
            .collect();

        let (p, r, f1) = calculate_metrics(&predicted, &truth);
        let hits = predicted.intersection(&truth).count();
        let status = if f1 >= 0.40 {
            "GREAT ✓✓"
        } else if f1 >= 0.30 {
            "GOOD ✓"
        } else if f1 >= 0.25 {
            "OK"
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

    // Post-UV analysis
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("POST-UV CORRELATION ANALYSIS (Key Signal)");
    println!("═══════════════════════════════════════════════════════════════════════");

    let mut high_post_uv: Vec<_> = scored
        .iter()
        .filter(|(_, _, data, _, _, _, _)| data.post_uv_spikes >= 5)
        .collect();
    high_post_uv.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap()); // Sort by post_uv_ratio

    println!("Residues with highest post-UV spike ratio (> 5 spikes):");
    println!(
        "{:>5} {:>8} {:>8} {:>8}",
        "Res", "PostUV%", "PostUV#", "Truth?"
    );
    for (res_id, _, data, _, post_uv, _, _) in high_post_uv.iter().take(20) {
        let in_truth = if truth.contains(res_id) {
            "YES ←"
        } else {
            ""
        };
        println!(
            "{:>5} {:>7.1}% {:>8} {:>8}",
            res_id,
            post_uv * 100.0,
            data.post_uv_spikes,
            in_truth
        );
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
    if best_f1 >= 0.40 {
        println!("║  RESULT: ✓✓✓ EXCELLENT - F1 >= 0.40!                                ║");
    } else if best_f1 >= 0.35 {
        println!("║  RESULT: ✓✓ VERY GOOD - F1 >= 0.35                                  ║");
    } else if best_f1 >= 0.30 {
        println!("║  RESULT: ✓ GOOD - F1 >= 0.30                                        ║");
    } else {
        println!(
            "║  RESULT: NEEDS WORK - F1 = {:.3}                                    ║",
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
