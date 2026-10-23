//! Debug spike timesteps to understand the timing pattern

use anyhow::Result;
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("=== SPIKE TIMESTEP DEBUGGING ===\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );
    let topology = PrismPrepTopology::load(topology_path)?;

    // Minimal UV config
    let uv_config = UvProbeConfig {
        enabled: true,
        burst_energy: 30.0,
        burst_interval: 200,
        burst_duration: 20,
        ..Default::default()
    };

    let context = CudaContext::new(0)?;
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

    engine.set_temperature_protocol(TemperatureProtocol {
        start_temp: 300.0, // Start warm for more spikes
        end_temp: 300.0,
        ramp_steps: 0,
        hold_steps: 1000,
        cold_hold_steps: 0,
        current_step: 0,
    })?;

    engine.set_uv_config(uv_config);

    println!("Running 1000 steps at 300K...");
    let _summary = engine.run(1000)?;

    let spikes = engine.download_full_spike_events(1000)?;
    println!("Downloaded {} spikes\n", spikes.len());

    if spikes.is_empty() {
        println!("No spikes to analyze!");
        return Ok(());
    }

    // Show raw timesteps (copy fields to avoid packed struct alignment issues)
    println!("First 50 spike timesteps:");
    for (i, spike) in spikes.iter().take(50).enumerate() {
        let ts = spike.timestep;
        let phase = ts % 200;
        let uv_status = if phase < 20 { "DURING UV" } else { "" };
        println!(
            "  Spike {}: timestep={}, phase={} {}",
            i, ts, phase, uv_status
        );
    }

    // Histogram
    println!("\nTimestep phase histogram (mod 200):");
    let mut bins = [0usize; 20]; // 10-step bins
    for spike in &spikes {
        let ts = spike.timestep;
        let phase = ts % 200;
        let bin = (phase / 10) as usize;
        if bin < 20 {
            bins[bin] += 1;
        }
    }

    for (i, &count) in bins.iter().enumerate() {
        let start = i * 10;
        let end = start + 9;
        let bar = "█".repeat((count as f32 / spikes.len() as f32 * 50.0) as usize);
        let uv_mark = if start < 20 { " ← UV BURST" } else { "" };
        println!("  {:3}-{:3}: {:4} {:50}{}", start, end, count, bar, uv_mark);
    }

    // Check if timesteps are sequential or all the same
    let unique_timesteps: std::collections::HashSet<i32> = spikes
        .iter()
        .map(|s| {
            let ts = s.timestep;
            ts
        })
        .collect();
    println!("\nUnique timesteps: {}", unique_timesteps.len());

    let min_ts = spikes
        .iter()
        .map(|s| {
            let ts = s.timestep;
            ts
        })
        .min()
        .unwrap_or(0);
    let max_ts = spikes
        .iter()
        .map(|s| {
            let ts = s.timestep;
            ts
        })
        .max()
        .unwrap_or(0);
    println!("Timestep range: {} to {}", min_ts, max_ts);

    // Check intensity values
    let non_zero_intensity = spikes
        .iter()
        .filter(|s| {
            let i = s.intensity;
            i > 0.0
        })
        .count();
    println!(
        "\nSpikes with intensity > 0: {}/{}",
        non_zero_intensity,
        spikes.len()
    );

    // Check nearby_residues
    let with_residues = spikes
        .iter()
        .filter(|s| {
            let n = s.n_residues;
            n > 0
        })
        .count();
    println!("Spikes with residues: {}/{}", with_residues, spikes.len());

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
