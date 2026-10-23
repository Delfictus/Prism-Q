//! Test aromatic enrichment in spike positions
//!
//! Verifies that spikes occurring during UV bursts are
//! enriched near aromatic residues.

use anyhow::Result;
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{CryoUvProtocol, GpuSpikeEvent, NhsAmberFusedEngine};

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   AROMATIC ENRICHMENT TEST                                           ║");
    println!("║   Checking if UV spikes are near aromatics                           ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    let topology = PrismPrepTopology::load(topology_path)?;
    println!("Loaded topology: {} atoms\n", topology.n_atoms);

    let protocol = CryoUvProtocol {
        start_temp: 200.0,
        end_temp: 300.0,
        cold_hold_steps: 0,
        ramp_steps: 500,
        warm_hold_steps: 500,
        current_step: 0,
        uv_burst_energy: 30.0,
        uv_burst_interval: 200,
        uv_burst_duration: 20,
        scan_wavelengths: vec![280.0],
        wavelength_dwell_steps: 200,
        ramp_down_steps: 0,
        cold_return_steps: 0,
        stepped_holds: Vec::new(),
    };

    let context = CudaContext::new(0)?;
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;
    engine.set_cryo_uv_protocol(protocol)?;
    engine.set_spike_accumulation(true);

    println!("Running 1000 steps...");
    let _summary = engine.run(1000)?;

    let spikes = engine.get_accumulated_spikes().to_vec();
    println!("Accumulated {} spike events\n", spikes.len());

    // Get aromatic residue IDs for comparison
    let aromatic_residues = engine.aromatic_residue_ids();
    println!(
        "Aromatic residues ({}): {:?}\n",
        aromatic_residues.len(),
        &aromatic_residues[..aromatic_residues.len().min(10)]
    );

    // Separate UV and non-UV spikes
    let burst_interval = 200i32;
    let burst_duration = 20i32;

    let uv_spikes: Vec<_> = spikes
        .iter()
        .filter(|s| s.timestep % burst_interval < burst_duration)
        .collect();
    let non_uv_spikes: Vec<_> = spikes
        .iter()
        .filter(|s| s.timestep % burst_interval >= burst_duration)
        .collect();

    println!("UV spikes: {}", uv_spikes.len());
    println!("Non-UV spikes: {}", non_uv_spikes.len());

    // Check how many spikes have aromatic residues in their nearby_residues field
    fn count_with_aromatics(
        spikes: &[&GpuSpikeEvent],
        aromatic_set: &std::collections::HashSet<i32>,
    ) -> usize {
        spikes
            .iter()
            .filter(|s| {
                // Copy fields to avoid packed struct alignment issues
                let n = s.n_residues;
                let nearby = s.nearby_residues; // Copy the array
                (0..n as usize).any(|i| aromatic_set.contains(&nearby[i]))
            })
            .count()
    }

    let aromatic_set: std::collections::HashSet<i32> = aromatic_residues.iter().cloned().collect();

    let uv_with_aromatics = count_with_aromatics(&uv_spikes, &aromatic_set);
    let non_uv_with_aromatics = count_with_aromatics(&non_uv_spikes, &aromatic_set);

    let uv_aromatic_rate = if uv_spikes.len() > 0 {
        100.0 * uv_with_aromatics as f32 / uv_spikes.len() as f32
    } else {
        0.0
    };
    let non_uv_aromatic_rate = if non_uv_spikes.len() > 0 {
        100.0 * non_uv_with_aromatics as f32 / non_uv_spikes.len() as f32
    } else {
        0.0
    };

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("AROMATIC ENRICHMENT ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "\nUV spikes near aromatics: {}/{} ({:.1}%)",
        uv_with_aromatics,
        uv_spikes.len(),
        uv_aromatic_rate
    );
    println!(
        "Non-UV spikes near aromatics: {}/{} ({:.1}%)",
        non_uv_with_aromatics,
        non_uv_spikes.len(),
        non_uv_aromatic_rate
    );

    let enrichment = if non_uv_aromatic_rate > 0.0 {
        uv_aromatic_rate / non_uv_aromatic_rate
    } else {
        f32::INFINITY
    };

    println!("\nAromatic enrichment (UV vs Non-UV): {:.2}x", enrichment);

    if enrichment > 1.5 {
        println!("\n✓ STRONG aromatic enrichment in UV spikes!");
    } else if enrichment > 1.1 {
        println!("\n~ Moderate aromatic enrichment");
    } else {
        println!("\n✗ NO aromatic enrichment - UV spikes not localized to aromatics");
    }

    // Sample some UV spikes with their nearby residues
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("SAMPLE UV SPIKES (first 10)");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>6} {:>8} {:>8} {:>30}",
        "Index", "Timestep", "n_res", "Nearby Residues"
    );
    println!("{}", "-".repeat(60));

    for (i, spike) in uv_spikes.iter().take(10).enumerate() {
        // Copy fields to avoid packed struct alignment issues
        let ts = spike.timestep;
        let n_res = spike.n_residues;
        let nearby_arr = spike.nearby_residues; // Copy the array

        let nearby: Vec<i32> = (0..n_res as usize)
            .map(|j| nearby_arr[j])
            .filter(|&r| r >= 0)
            .collect();
        let is_aromatic: Vec<bool> = nearby.iter().map(|r| aromatic_set.contains(r)).collect();
        let residue_str: Vec<String> = nearby
            .iter()
            .zip(is_aromatic.iter())
            .map(|(r, &is_aro)| format!("{}{}", r, if is_aro { "*" } else { "" }))
            .collect();
        println!(
            "{:>6} {:>8} {:>8} {:>30}",
            i,
            ts,
            n_res,
            residue_str.join(", ")
        );
    }
    println!("(* = aromatic residue)");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
