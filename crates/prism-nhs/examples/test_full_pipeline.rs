//! Full PRISM4D Pipeline Test - UV-LIF Coupling Validation
//!
//! Tests the complete UV physics pipeline on a real protein structure

use anyhow::Result;
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{CryoUvProtocol, GpuSpikeEvent, NhsAmberFusedEngine};

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║     PRISM4D FULL PIPELINE TEST - UV-LIF Coupling Validation              ║");
    println!("║     Testing on 6M0J (SARS-CoV-2 Spike RBD)                               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Load topology
    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6M0J_topology.json"),
    );
    let topology = PrismPrepTopology::load(topology_path)?;
    println!("✓ Loaded topology: {} atoms", topology.n_atoms);

    // Create engine
    let context = CudaContext::new(0)?;
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;
    println!("✓ Created NHS-AMBER Fused Engine");
    println!("  Aromatics detected: {}", engine.n_aromatics());

    // Configure unified cryo-UV protocol
    let protocol = CryoUvProtocol {
        start_temp: 77.0, // Liquid nitrogen temp
        end_temp: 310.0,  // Physiological
        cold_hold_steps: 0,
        ramp_steps: 5000,
        warm_hold_steps: 5000,
        current_step: 0,
        uv_burst_energy: 30.0,
        uv_burst_interval: 500,
        uv_burst_duration: 50,
        scan_wavelengths: vec![280.0, 274.0, 258.0], // TRP, TYR, PHE
        wavelength_dwell_steps: 500,
        ramp_down_steps: 0,
        cold_return_steps: 0,
        stepped_holds: Vec::new(),
    };
    engine.set_cryo_uv_protocol(protocol)?;
    println!("✓ Unified cryo-UV protocol: 77K → 310K with UV-LIF coupling");

    // Enable spike accumulation
    engine.set_spike_accumulation(true);
    println!("✓ Spike accumulation enabled\n");

    // Run simulation
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("RUNNING SIMULATION (10,000 steps)");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let summary = engine.run(10000)?;
    println!(
        "\n✓ Simulation complete: {} total spikes",
        summary.total_spikes
    );

    // Analyze accumulated spikes
    let spikes = engine.get_accumulated_spikes();
    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("SPIKE ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    println!("Total accumulated spikes: {}", spikes.len());

    // Categorize by UV phase
    let burst_interval = 500i32;
    let burst_duration = 50i32;

    let uv_spikes: Vec<_> = spikes
        .iter()
        .filter(|s| s.timestep % burst_interval < burst_duration)
        .collect();
    let non_uv_spikes: Vec<_> = spikes
        .iter()
        .filter(|s| s.timestep % burst_interval >= burst_duration)
        .collect();

    println!(
        "UV-phase spikes:     {:>8} ({:.1}%)",
        uv_spikes.len(),
        100.0 * uv_spikes.len() as f32 / spikes.len().max(1) as f32
    );
    println!(
        "Non-UV-phase spikes: {:>8} ({:.1}%)",
        non_uv_spikes.len(),
        100.0 * non_uv_spikes.len() as f32 / spikes.len().max(1) as f32
    );

    // Check aromatic enrichment
    let aromatic_set: std::collections::HashSet<i32> =
        engine.aromatic_residue_ids().iter().cloned().collect();

    fn count_with_aromatics(
        spikes: &[&GpuSpikeEvent],
        aromatic_set: &std::collections::HashSet<i32>,
    ) -> usize {
        spikes
            .iter()
            .filter(|s| {
                let n = s.n_residues;
                let nearby = s.nearby_residues;
                (0..n as usize).any(|i| aromatic_set.contains(&nearby[i]))
            })
            .count()
    }

    let uv_with_aromatics = count_with_aromatics(&uv_spikes, &aromatic_set);
    let non_uv_with_aromatics = count_with_aromatics(&non_uv_spikes, &aromatic_set);

    let uv_rate = if uv_spikes.len() > 0 {
        100.0 * uv_with_aromatics as f32 / uv_spikes.len() as f32
    } else {
        0.0
    };
    let non_uv_rate = if non_uv_spikes.len() > 0 {
        100.0 * non_uv_with_aromatics as f32 / non_uv_spikes.len() as f32
    } else {
        0.0
    };

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("AROMATIC ENRICHMENT");
    println!("═══════════════════════════════════════════════════════════════════════════\n");
    println!(
        "UV spikes near aromatics:     {:>6}/{:<6} ({:.1}%)",
        uv_with_aromatics,
        uv_spikes.len(),
        uv_rate
    );
    println!(
        "Non-UV spikes near aromatics: {:>6}/{:<6} ({:.1}%)",
        non_uv_with_aromatics,
        non_uv_spikes.len(),
        non_uv_rate
    );

    let enrichment = if non_uv_rate > 0.0 {
        uv_rate / non_uv_rate
    } else {
        f32::INFINITY
    };
    println!("\nAromatic enrichment: {:.2}x", enrichment);

    println!("\n═══════════════════════════════════════════════════════════════════════════");
    println!("VALIDATION RESULTS");
    println!("═══════════════════════════════════════════════════════════════════════════\n");

    let uv_timing_ok = uv_spikes.len() > 0;
    let aromatic_ok = uv_rate > 90.0; // Should be ~100% for UV spikes
    let enrichment_ok = enrichment > 1.5;

    println!(
        "{} UV-timing correlation: {} UV-phase spikes detected",
        if uv_timing_ok { "✓" } else { "✗" },
        uv_spikes.len()
    );
    println!(
        "{} Aromatic localization: {:.1}% of UV spikes at aromatic sites",
        if aromatic_ok { "✓" } else { "✗" },
        uv_rate
    );
    println!(
        "{} Aromatic enrichment: {:.2}x (target > 1.5x)",
        if enrichment_ok { "✓" } else { "✗" },
        enrichment
    );

    if uv_timing_ok && aromatic_ok && enrichment_ok {
        println!("\n🎉 UV-LIF COUPLING VALIDATED SUCCESSFULLY!");
    } else {
        println!("\n⚠️  Some validation checks failed");
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
