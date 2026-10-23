//! UV Physics Chain Debug
//!
//! Traces the complete UV excitation pathway step-by-step:
//! 1. UV burst active flag
//! 2. Aromatic excitation state (d_is_excited)
//! 3. Vibrational energy (d_vibrational_energy)
//! 4. Energy transfer to neighbors
//! 5. LIF spike detection
//!
//! This will identify exactly where the physics chain breaks.

use anyhow::Result;
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig};

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   UV PHYSICS CHAIN DEBUG                                             ║");
    println!("║   Tracing: UV burst → excitation → energy → spikes                   ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    if !topology_path.exists() {
        println!("ERROR: Topology file not found");
        return Ok(());
    }

    let topology = PrismPrepTopology::load(topology_path)?;
    println!("Loaded topology: {} atoms", topology.n_atoms);

    // UV config with short burst interval for rapid testing
    let uv_config = UvProbeConfig {
        enabled: true,
        burst_energy: 30.0,
        burst_interval: 50, // Short interval for rapid testing
        burst_duration: 10, // 10-step bursts
        frequency_hopping_enabled: false,
        scan_wavelengths: vec![280.0],
        dwell_steps: 100,
        ..Default::default()
    };

    println!("\nUV Config:");
    println!(
        "  Burst interval: {} steps (burst at 0-9, 50-59, 100-109, ...)",
        uv_config.burst_interval
    );
    println!("  Burst duration: {} steps", uv_config.burst_duration);
    println!("  Burst energy: {} kcal/mol\n", uv_config.burst_energy);

    let context = CudaContext::new(0)?;
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

    // Start at moderate temperature
    engine.set_temperature_protocol(TemperatureProtocol {
        start_temp: 200.0,
        end_temp: 200.0,
        ramp_steps: 0,
        hold_steps: 500,
        cold_hold_steps: 0,
        current_step: 0,
    })?;

    engine.set_uv_config(uv_config);

    let n_aromatics = engine.n_aromatics();
    println!("Aromatics detected: {}", n_aromatics);
    if n_aromatics == 0 {
        println!("ERROR: No aromatics detected! UV physics cannot work.");
        return Ok(());
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("STEP-BY-STEP UV CHAIN TRACE");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!(
        "{:>5} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "Step", "Phase", "UV Active", "Excited", "VibEnergy", "Spikes"
    );
    println!("{}", "-".repeat(65));

    let mut total_spikes_during_uv = 0;
    let mut total_spikes_after_uv = 0;
    let mut total_spikes_between = 0;
    let mut steps_with_excitation = 0;
    let mut max_vib_energy = 0.0f32;

    for step in 0..200 {
        // Get state BEFORE step
        let is_excited_before = engine.get_is_excited()?;
        let vib_energy_before = engine.get_vibrational_energy()?;

        // Run one step
        let result = engine.step()?;

        // Get state AFTER step
        let is_excited_after = engine.get_is_excited()?;
        let vib_energy_after = engine.get_vibrational_energy()?;

        // Analyze
        let phase = step % 50;
        let is_uv_burst = phase < 10;

        let n_excited_before: i32 = is_excited_before.iter().sum();
        let n_excited_after: i32 = is_excited_after.iter().sum();
        let sum_vib_before: f32 = vib_energy_before.iter().sum();
        let sum_vib_after: f32 = vib_energy_after.iter().sum();

        max_vib_energy = max_vib_energy.max(sum_vib_after);
        if n_excited_after > 0 {
            steps_with_excitation += 1;
        }

        // Count spikes by timing
        if phase < 10 {
            total_spikes_during_uv += result.spike_count;
        } else if phase < 30 {
            total_spikes_after_uv += result.spike_count;
        } else {
            total_spikes_between += result.spike_count;
        }

        // Print every step for first 20, then every 10
        if step < 20 || step % 10 == 0 || result.spike_count > 0 {
            let uv_marker = if is_uv_burst { "YES ←" } else { "" };
            println!(
                "{:>5} {:>8} {:>10} {:>5}/{:>5} {:>10.4} {:>8}",
                step,
                phase,
                uv_marker,
                n_excited_before,
                n_excited_after,
                sum_vib_after,
                result.spike_count
            );

            // Detailed debug for UV burst transitions
            if phase == 0 || phase == 10 {
                println!(
                    "      [Excitation: {} → {}, VibEnergy: {:.4} → {:.4}]",
                    n_excited_before, n_excited_after, sum_vib_before, sum_vib_after
                );
            }
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("SUMMARY");
    println!("═══════════════════════════════════════════════════════════════════════");

    println!("\nExcitation State:");
    println!("  Steps with excitation: {}/200", steps_with_excitation);
    println!("  Max vibrational energy: {:.4} kcal/mol", max_vib_energy);

    println!("\nSpike Timing:");
    println!(
        "  During UV burst (phase 0-9):   {}",
        total_spikes_during_uv
    );
    println!(
        "  Shortly after UV (phase 10-29): {}",
        total_spikes_after_uv
    );
    println!("  Between bursts (phase 30-49):   {}", total_spikes_between);

    // Diagnose
    println!("\n═══════════════════════════════════════════════════════════════════════");
    println!("DIAGNOSIS");
    println!("═══════════════════════════════════════════════════════════════════════");

    if steps_with_excitation == 0 {
        println!("✗ BUG: Aromatics are NEVER excited!");
        println!("  → Check: Is d_is_excited array being set on GPU?");
        println!("  → Check: Is excite_aromatic_wavelength() being called?");
    } else if steps_with_excitation < 40 {
        // Should be ~40 steps with burst
        println!(
            "✗ BUG: Aromatics excited less than expected ({} steps)",
            steps_with_excitation
        );
        println!("  → Check: Is excitation decaying too fast?");
    } else {
        println!("✓ Excitation is working ({} steps)", steps_with_excitation);
    }

    if max_vib_energy < 0.01 {
        println!(
            "✗ BUG: No vibrational energy deposited (max={:.6})",
            max_vib_energy
        );
        println!("  → Check: compute_deposited_energy_wavelength() return value");
        println!("  → Check: vibrational_energy array initialization");
    } else if max_vib_energy < 0.1 {
        println!(
            "~ WARNING: Vibrational energy is low ({:.4} kcal/mol)",
            max_vib_energy
        );
        println!("  → May need to increase burst_energy");
    } else {
        println!(
            "✓ Vibrational energy is being deposited (max={:.4})",
            max_vib_energy
        );
    }

    let total_spikes = total_spikes_during_uv + total_spikes_after_uv + total_spikes_between;
    if total_spikes == 0 {
        println!("✗ BUG: No spikes at all!");
        println!("  → LIF neurons not detecting anything");
    } else if total_spikes_during_uv + total_spikes_after_uv < total_spikes_between {
        println!("✗ BUG: Spikes NOT correlated with UV");
        println!(
            "  → UV: {} vs Non-UV: {} spikes",
            total_spikes_during_uv + total_spikes_after_uv,
            total_spikes_between
        );
        println!("  → Spikes are thermal noise, not UV-induced");
    } else {
        println!("✓ Spikes ARE correlated with UV");
        println!(
            "  → UV: {} vs Non-UV: {} spikes",
            total_spikes_during_uv + total_spikes_after_uv,
            total_spikes_between
        );
    }

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
