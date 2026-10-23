//! Debug Spike Timing - Analyze actual spike timestamps
//!
//! Downloads all spike events and analyzes their timestep distribution
//! to understand when spikes are actually occurring relative to UV bursts.

use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   SPIKE TIMESTAMP ANALYSIS                                           ║");
    println!("║   Analyzing actual spike timing relative to UV bursts                ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    println!("Loaded topology: {} atoms\n", topology.n_atoms);

    // UV config
    let burst_interval = 200i32;
    let burst_duration = 20i32;

    let uv_config = UvProbeConfig {
        enabled: true,
        burst_energy: 30.0,
        burst_interval,
        burst_duration,
        frequency_hopping_enabled: false,
        scan_wavelengths: vec![280.0],
        dwell_steps: 200,
        ..Default::default()
    };

    println!("UV Config:");
    println!("  Burst interval: {} steps", burst_interval);
    println!("  Burst duration: {} steps", burst_duration);
    println!("  UV active at phases: 0-{}", burst_duration - 1);
    println!();

    let context = CudaContext::new(0)?;
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

    engine.set_temperature_protocol(TemperatureProtocol {
        start_temp: 200.0,
        end_temp: 300.0,
        ramp_steps: 500,
        hold_steps: 500,
        cold_hold_steps: 0,
        current_step: 0,
    })?;

    engine.set_uv_config(uv_config);

    // Enable spike accumulation to collect spikes across all sync intervals
    engine.set_spike_accumulation(true);

    println!("Running 1000 steps with spike accumulation...");
    let _summary = engine.run(1000)?;

    // Get accumulated spike events (collected during each sync interval)
    let spikes = engine.get_accumulated_spikes().to_vec();
    println!(
        "Accumulated {} spike events across all sync intervals\n",
        spikes.len()
    );

    if spikes.is_empty() {
        println!("No spikes to analyze!");
        return Ok(());
    }

    // Analyze timestep distribution
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("TIMESTEP PHASE HISTOGRAM");
    println!("═══════════════════════════════════════════════════════════════════════");

    let mut phase_counts: HashMap<i32, usize> = HashMap::new();
    let mut intensity_sum: HashMap<i32, f32> = HashMap::new();

    for spike in &spikes {
        // Copy fields to avoid packed struct alignment issues
        let ts = spike.timestep;
        let intensity = spike.intensity;
        let phase = ts % burst_interval;
        *phase_counts.entry(phase).or_insert(0) += 1;
        *intensity_sum.entry(phase).or_insert(0.0) += intensity;
    }

    // Print histogram
    println!("\nPhase distribution (mod {} steps):", burst_interval);
    println!(
        "{:>6} {:>8} {:>10} {:>50}",
        "Phase", "Count", "Intensity", "Bar"
    );
    println!("{}", "-".repeat(80));

    let max_count = *phase_counts.values().max().unwrap_or(&1);

    for phase in 0..burst_interval {
        let count = *phase_counts.get(&phase).unwrap_or(&0);
        let intensity = *intensity_sum.get(&phase).unwrap_or(&0.0);
        let avg_intensity = if count > 0 {
            intensity / count as f32
        } else {
            0.0
        };

        let bar_len = if max_count > 0 {
            (count * 40) / max_count
        } else {
            0
        };
        let bar = "█".repeat(bar_len);

        let uv_marker = if phase < burst_duration {
            " ← UV"
        } else {
            ""
        };

        if count > 0 || phase < burst_duration || phase >= burst_interval - 5 {
            println!(
                "{:>6} {:>8} {:>10.3} {:>50}{}",
                phase, count, avg_intensity, bar, uv_marker
            );
        }
    }

    // Summary statistics
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("TIMING ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════");

    let during_uv: usize = (0..burst_duration)
        .map(|p| phase_counts.get(&p).unwrap_or(&0))
        .sum();
    let after_uv: usize = (burst_duration..(burst_duration + 50))
        .map(|p| phase_counts.get(&p).unwrap_or(&0))
        .sum();
    let between: usize = ((burst_duration + 50)..burst_interval)
        .map(|p| phase_counts.get(&p).unwrap_or(&0))
        .sum();
    let total = spikes.len();

    println!("\nSpike timing categories:");
    println!(
        "  During UV (phase 0-{}):       {:>6} ({:>5.1}%)",
        burst_duration - 1,
        during_uv,
        100.0 * during_uv as f32 / total as f32
    );
    println!(
        "  Post-UV (phase {}-{}):       {:>6} ({:>5.1}%)",
        burst_duration,
        burst_duration + 49,
        after_uv,
        100.0 * after_uv as f32 / total as f32
    );
    println!(
        "  Between bursts (phase {}-{}): {:>6} ({:>5.1}%)",
        burst_duration + 50,
        burst_interval - 1,
        between,
        100.0 * between as f32 / total as f32
    );

    // UV correlation metric
    let uv_related = during_uv + after_uv;
    let expected_if_uniform = total as f32 * (burst_duration + 50) as f32 / burst_interval as f32;
    let enrichment = uv_related as f32 / expected_if_uniform.max(1.0);

    println!("\nUV Correlation:");
    println!("  UV-related spikes: {}", uv_related);
    println!("  Expected if uniform: {:.0}", expected_if_uniform);
    println!("  Enrichment ratio: {:.2}x", enrichment);

    if enrichment > 1.5 {
        println!("  ✓ STRONG UV correlation (enrichment > 1.5x)");
    } else if enrichment > 1.1 {
        println!("  ~ Moderate UV correlation");
    } else {
        println!("  ✗ NO UV correlation - spikes are NOT UV-triggered");
    }

    // Check intensity correlation
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("INTENSITY ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════");

    let intensities: Vec<f32> = spikes
        .iter()
        .map(|s| {
            let i = s.intensity;
            i
        })
        .collect();
    let avg_intensity = intensities.iter().sum::<f32>() / intensities.len() as f32;
    let max_intensity = intensities.iter().cloned().fold(0.0f32, f32::max);
    let min_intensity = intensities.iter().cloned().fold(f32::MAX, f32::min);
    let nonzero = intensities.iter().filter(|&&i| i > 0.0).count();

    println!("\nIntensity statistics:");
    println!("  Min: {:.4}", min_intensity);
    println!("  Max: {:.4}", max_intensity);
    println!("  Avg: {:.4}", avg_intensity);
    println!(
        "  Non-zero: {}/{} ({:.1}%)",
        nonzero,
        total,
        100.0 * nonzero as f32 / total as f32
    );

    // Sample some spikes
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("SAMPLE SPIKES (first 20)");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>6} {:>8} {:>10} {:>8}",
        "Index", "Timestep", "Phase", "Intensity"
    );
    println!("{}", "-".repeat(40));

    for (i, spike) in spikes.iter().take(20).enumerate() {
        // Copy fields to avoid packed struct alignment issues
        let ts = spike.timestep;
        let intensity = spike.intensity;
        let phase = ts % burst_interval;
        let uv_marker = if phase < burst_duration {
            " ← UV"
        } else {
            ""
        };
        println!(
            "{:>6} {:>8} {:>10} {:>8.4}{}",
            i, ts, phase, intensity, uv_marker
        );
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
