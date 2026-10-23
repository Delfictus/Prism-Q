//! UV Physics Debugging
//!
//! Traces the complete UV perturbation pathway:
//! 1. UV burst activation → aromatic excitation
//! 2. Energy absorption → local heating
//! 3. Thermal perturbation → spike generation
//! 4. Spike-UV temporal correlation
//!
//! This will reveal if the physics is working or broken.

use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

const STEPS: i32 = 2000;
const UV_BURST_INTERVAL: i32 = 200; // Frequent bursts for debugging
const UV_BURST_DURATION: i32 = 20;

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   UV PHYSICS DEBUGGING                                              ║");
    println!("║   Tracing UV burst → spike correlation                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    // Test on both 6LU7 (working) and 6M0J (failing)
    let targets = [
        ("6LU7", concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json")),
        ("6M0J_apo", concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6M0J_apo_topology.json")),
    ];

    for (name, path) in &targets {
        println!("═══════════════════════════════════════════════════════════════════════");
        println!("TARGET: {}", name);
        println!("═══════════════════════════════════════════════════════════════════════\n");

        let topology_path = Path::new(path);
        if !topology_path.exists() {
            println!("  SKIPPED - file not found\n");
            continue;
        }

        let topology = PrismPrepTopology::load(topology_path)?;
        println!("Atoms: {}", topology.n_atoms);

        // Find aromatics
        let aromatics: Vec<(i32, &str)> = topology
            .residue_names
            .iter()
            .enumerate()
            .filter_map(|(i, name)| {
                if matches!(name.as_str(), "TRP" | "TYR" | "PHE") {
                    Some((i as i32, name.as_str()))
                } else {
                    None
                }
            })
            .collect();

        println!(
            "Aromatic residues: {} (TRP: {}, TYR: {}, PHE: {})",
            aromatics.len(),
            aromatics.iter().filter(|(_, n)| *n == "TRP").count(),
            aromatics.iter().filter(|(_, n)| *n == "TYR").count(),
            aromatics.iter().filter(|(_, n)| *n == "PHE").count()
        );

        // Show first few aromatics
        println!(
            "First 10 aromatics: {:?}",
            aromatics
                .iter()
                .take(10)
                .map(|(id, n)| format!("{}:{}", id, n))
                .collect::<Vec<_>>()
        );

        // Configure UV
        let uv_config = UvProbeConfig {
            enabled: true,
            burst_energy: 30.0,
            burst_interval: UV_BURST_INTERVAL,
            burst_duration: UV_BURST_DURATION,
            frequency_hopping_enabled: true,
            scan_wavelengths: vec![280.0, 274.0, 258.0],
            dwell_steps: 200,
            ..Default::default()
        };

        println!("\nUV Config:");
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
        println!(
            "  Expected bursts in {} steps: {}",
            STEPS,
            STEPS / UV_BURST_INTERVAL
        );

        // Run simulation
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

        engine.set_uv_config(uv_config);

        println!("\nRunning {} steps...", STEPS);
        let summary = engine.run(STEPS)?;

        println!("\nRun Summary:");
        println!("  Total spikes: {}", summary.total_spikes);
        println!("  Expected UV bursts: {}", STEPS / UV_BURST_INTERVAL);

        // Download all spike events
        let spikes = engine.download_full_spike_events(5000)?;
        println!("  Spikes downloaded: {}", spikes.len());

        if spikes.is_empty() {
            println!("\n  ⚠ NO SPIKES DETECTED - Physics may be broken!");
            continue;
        }

        // Analyze spike timing relative to UV bursts
        println!("\n--- SPIKE TIMING ANALYSIS ---");

        // Categorize spikes by timing relative to UV burst
        let mut during_burst = 0;
        let mut post_burst_0_50 = 0; // 0-50 steps after burst ends
        let mut post_burst_50_100 = 0; // 50-100 steps after burst ends
        let mut post_burst_100_150 = 0;
        let mut between_bursts = 0;

        let mut timestep_histogram: HashMap<i32, usize> = HashMap::new();

        for spike in &spikes {
            let phase = spike.timestep % UV_BURST_INTERVAL;
            *timestep_histogram.entry(phase).or_insert(0) += 1;

            if phase < UV_BURST_DURATION {
                during_burst += 1;
            } else if phase < UV_BURST_DURATION + 50 {
                post_burst_0_50 += 1;
            } else if phase < UV_BURST_DURATION + 100 {
                post_burst_50_100 += 1;
            } else if phase < UV_BURST_DURATION + 150 {
                post_burst_100_150 += 1;
            } else {
                between_bursts += 1;
            }
        }

        let total = spikes.len();
        println!(
            "Spike timing (relative to UV burst cycle of {} steps):",
            UV_BURST_INTERVAL
        );
        println!(
            "  During UV burst (0-{}):        {:>5} ({:>5.1}%)",
            UV_BURST_DURATION,
            during_burst,
            during_burst as f32 / total as f32 * 100.0
        );
        println!(
            "  Post-burst 0-50 steps:         {:>5} ({:>5.1}%)",
            post_burst_0_50,
            post_burst_0_50 as f32 / total as f32 * 100.0
        );
        println!(
            "  Post-burst 50-100 steps:       {:>5} ({:>5.1}%)",
            post_burst_50_100,
            post_burst_50_100 as f32 / total as f32 * 100.0
        );
        println!(
            "  Post-burst 100-150 steps:      {:>5} ({:>5.1}%)",
            post_burst_100_150,
            post_burst_100_150 as f32 / total as f32 * 100.0
        );
        println!(
            "  Between bursts (>{} steps):   {:>5} ({:>5.1}%)",
            UV_BURST_DURATION + 150,
            between_bursts,
            between_bursts as f32 / total as f32 * 100.0
        );

        // If UV is working, we should see MORE spikes shortly after UV bursts
        let uv_correlated = during_burst + post_burst_0_50 + post_burst_50_100;
        let expected_if_random =
            total as f32 * (UV_BURST_DURATION + 100) as f32 / UV_BURST_INTERVAL as f32;

        println!("\nUV Correlation Check:");
        println!(
            "  UV-correlated spikes (during + 100 steps after): {}",
            uv_correlated
        );
        println!("  Expected if random (uniform): {:.0}", expected_if_random);
        let enrichment = uv_correlated as f32 / expected_if_random;
        println!("  Enrichment ratio: {:.2}x", enrichment);

        if enrichment > 1.5 {
            println!("  ✓ UV correlation detected (enrichment > 1.5x)");
        } else if enrichment > 1.1 {
            println!("  ~ Weak UV correlation (1.1-1.5x)");
        } else {
            println!("  ✗ NO UV correlation - spikes are random relative to UV bursts!");
        }

        // Analyze spike proximity to aromatics
        println!("\n--- AROMATIC PROXIMITY ANALYSIS ---");

        let aromatic_set: HashSet<i32> = aromatics.iter().map(|(id, _)| *id).collect();
        let mut near_aromatic = 0;
        let mut far_from_aromatic = 0;
        let mut aromatic_distance_sum = 0i32;

        for spike in &spikes {
            let mut min_dist = i32::MAX;
            for i in 0..spike.n_residues.min(8) as usize {
                let res = spike.nearby_residues[i];
                if res >= 0 {
                    for &(ar_id, _) in &aromatics {
                        let dist = (ar_id - res).abs();
                        if dist < min_dist {
                            min_dist = dist;
                        }
                    }
                }
            }
            if min_dist <= 6 {
                near_aromatic += 1;
            } else {
                far_from_aromatic += 1;
            }
            if min_dist < i32::MAX {
                aromatic_distance_sum += min_dist;
            }
        }

        println!(
            "Spikes near aromatics (≤6 residues): {} ({:.1}%)",
            near_aromatic,
            near_aromatic as f32 / total as f32 * 100.0
        );
        println!(
            "Spikes far from aromatics (>6 residues): {} ({:.1}%)",
            far_from_aromatic,
            far_from_aromatic as f32 / total as f32 * 100.0
        );
        println!(
            "Average distance to nearest aromatic: {:.1} residues",
            aromatic_distance_sum as f32 / total as f32
        );

        // If UV is working, spikes should be enriched near aromatics
        let aromatic_fraction = aromatics.len() as f32 / topology.residue_names.len() as f32;
        let expected_near_if_random = total as f32 * aromatic_fraction * 13.0; // ±6 residues = 13 residue window
        let aromatic_enrichment = near_aromatic as f32 / expected_near_if_random.max(1.0);

        println!("Aromatic enrichment: {:.2}x vs random", aromatic_enrichment);
        if aromatic_enrichment > 2.0 {
            println!("  ✓ Strong aromatic enrichment - UV targeting works");
        } else if aromatic_enrichment > 1.2 {
            println!("  ~ Moderate aromatic enrichment");
        } else {
            println!("  ✗ NO aromatic enrichment - spikes not UV-driven!");
        }

        // Spike intensity analysis
        println!("\n--- SPIKE INTENSITY ANALYSIS ---");
        let intensities: Vec<f32> = spikes.iter().map(|s| s.intensity).collect();
        let avg_intensity = intensities.iter().sum::<f32>() / intensities.len() as f32;
        let max_intensity = intensities.iter().cloned().fold(0.0f32, f32::max);
        let min_intensity = intensities.iter().cloned().fold(f32::MAX, f32::min);

        println!("Intensity stats:");
        println!(
            "  Min: {:.3}, Max: {:.3}, Avg: {:.3}",
            min_intensity, max_intensity, avg_intensity
        );

        // Intensity during vs after UV
        let intensity_during: f32 = spikes
            .iter()
            .filter(|s| (s.timestep % UV_BURST_INTERVAL) < UV_BURST_DURATION)
            .map(|s| s.intensity)
            .sum::<f32>();
        let count_during = during_burst.max(1) as f32;

        let intensity_after: f32 = spikes
            .iter()
            .filter(|s| {
                let phase = s.timestep % UV_BURST_INTERVAL;
                phase >= UV_BURST_DURATION && phase < UV_BURST_DURATION + 100
            })
            .map(|s| s.intensity)
            .sum::<f32>();
        let count_after = (post_burst_0_50 + post_burst_50_100).max(1) as f32;

        println!(
            "Avg intensity during UV: {:.3}",
            intensity_during / count_during
        );
        println!(
            "Avg intensity 0-100 after UV: {:.3}",
            intensity_after / count_after
        );

        // Check which residues are spiking
        println!("\n--- TOP SPIKING RESIDUES ---");
        let mut residue_counts: HashMap<i32, usize> = HashMap::new();
        for spike in &spikes {
            for i in 0..spike.n_residues.min(8) as usize {
                let res = spike.nearby_residues[i];
                if res >= 0 {
                    *residue_counts.entry(res).or_insert(0) += 1;
                }
            }
        }

        let mut ranked: Vec<_> = residue_counts.iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(a.1));

        println!("Top 15 residues by spike count:");
        println!(
            "{:>5} {:>8} {:>8} {:>10}",
            "Res", "Count", "ResName", "Aromatic?"
        );
        for (i, (&res, &count)) in ranked.iter().take(15).enumerate() {
            let name = topology
                .residue_names
                .get(res as usize)
                .map(|s| s.as_str())
                .unwrap_or("?");
            let is_aromatic = if aromatic_set.contains(&res) {
                "YES"
            } else {
                ""
            };
            let near_ar = aromatics
                .iter()
                .map(|(id, _)| (id - res).abs())
                .min()
                .unwrap_or(999);
            println!(
                "{:>5} {:>8} {:>8} {:>10} (dist to arom: {})",
                res, count, name, is_aromatic, near_ar
            );
        }

        println!();
    }

    println!("═══════════════════════════════════════════════════════════════════════");
    println!("PHYSICS DEBUGGING COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
