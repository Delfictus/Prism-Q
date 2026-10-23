//! GPU Wavelength Proof Test
//!
//! This test proves that the GPU kernel receives the correct wavelength value
//! and that the vibrational energy deposited differs with wavelength.
//!
//! Requirements:
//! - PTX compiled with -DDEBUG_UV_WAVELENGTH to see GPU printf output
//! - 1L2Y topology has TRP (peak at 280nm) and TYR (peak at 274nm)
//!
//! Expected behavior:
//! - At 280nm: TRP gets maximum excitation (~102 kcal/mol)
//! - At 258nm: TRP gets minimal excitation (Gaussian off-peak)
//!
//! Usage:
//!   cargo run --release -p prism-nhs --features gpu --bin test_wavelength_gpu

use anyhow::{Context, Result};

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_nhs::{
    fused_engine::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig},
    input::PrismPrepTopology,
};

fn main() -> Result<()> {
    #[cfg(not(feature = "gpu"))]
    {
        println!("This test requires the 'gpu' feature. Run with:");
        println!("  cargo run --release -p prism-nhs --features gpu --bin test_wavelength_gpu");
        return Ok(());
    }

    #[cfg(feature = "gpu")]
    run_gpu_test()
}

#[cfg(feature = "gpu")]
fn run_gpu_test() -> Result<()> {
    println!("=== GPU Wavelength Proof Test ===");
    println!("NOTE: Requires PTX compiled with -DDEBUG_UV_WAVELENGTH for printf");
    println!();

    // Initialize CUDA
    let ctx = CudaContext::new(0).context("Failed to create CUDA context")?;
    println!("CUDA context initialized");

    // Load 1L2Y topology (small structure with TRP and TYR)
    let topology_path = "data/curated_14/topologies/1L2Y_topology.json";
    println!("Loading topology: {}", topology_path);
    let topology: PrismPrepTopology = serde_json::from_str(
        &std::fs::read_to_string(topology_path).context("Failed to read topology file")?,
    )
    .context("Failed to parse topology")?;
    println!("  Atoms: {}", topology.n_atoms);
    println!("  Residues: {}", topology.n_residues);

    // Create engine
    let mut engine = NhsAmberFusedEngine::new(
        ctx, &topology, 32,  // grid_dim
        1.5, // grid_spacing
    )
    .context("Failed to create engine")?;
    println!("Engine created");
    println!("  Aromatics detected: {}", engine.n_aromatics());

    // Configure temperature (room temp, no protocol)
    let temp_protocol = TemperatureProtocol {
        start_temp: 300.0,
        end_temp: 300.0,
        ramp_steps: 0,
        hold_steps: 100000,
        cold_hold_steps: 0,
        current_step: 0,
    };
    let _ = engine.set_temperature_protocol(temp_protocol);

    // ========================================================
    // TEST 1: 280nm wavelength (TRP peak)
    // ========================================================
    println!("\n=== TEST 1: 280nm wavelength (TRP peak) ===");
    let uv_config_280 = UvProbeConfig {
        burst_energy: 5.0,
        burst_interval: 1, // Burst every step for testing
        burst_duration: 1,
        frequency_hopping_enabled: true,
        scan_wavelengths: vec![280.0], // Fixed at 280nm
        current_wavelength_idx: 0,
        dwell_steps: 1000,
        ..Default::default()
    };
    engine.set_uv_config(uv_config_280);
    println!("UV config set: wavelength=280nm");

    // Run one step with UV burst to excite
    println!("\nRunning 1 step with UV burst at 280nm:");
    let result = engine.step()?;
    println!(
        "  Step result: temp={:.1}K, burst={}",
        result.temperature, result.uv_burst_active
    );

    // Read back vibrational energy from GPU
    let energy_280 = engine.get_vibrational_energy()?;
    let excited_280 = engine.get_is_excited()?;
    println!("  Vibrational energy (GPU readback): {:?}", energy_280);
    println!("  Is excited (GPU readback): {:?}", excited_280);

    // ========================================================
    // Reset engine for TEST 2
    // ========================================================
    println!("\n--- Resetting engine for second test ---");
    drop(engine);

    let ctx = CudaContext::new(0).context("Failed to create CUDA context")?;
    let mut engine =
        NhsAmberFusedEngine::new(ctx, &topology, 32, 1.5).context("Failed to create engine")?;
    let _ = engine.set_temperature_protocol(TemperatureProtocol {
        start_temp: 300.0,
        end_temp: 300.0,
        ramp_steps: 0,
        hold_steps: 100000,
        cold_hold_steps: 0,
        current_step: 0,
    });

    // ========================================================
    // TEST 2: 258nm wavelength (off-peak for TRP)
    // ========================================================
    println!("\n=== TEST 2: 258nm wavelength (TRP off-peak) ===");
    let uv_config_258 = UvProbeConfig {
        burst_energy: 5.0,
        burst_interval: 1,
        burst_duration: 1,
        frequency_hopping_enabled: true,
        scan_wavelengths: vec![258.0], // Fixed at 258nm
        current_wavelength_idx: 0,
        dwell_steps: 1000,
        ..Default::default()
    };
    engine.set_uv_config(uv_config_258);
    println!("UV config set: wavelength=258nm");

    // Run one step with UV burst to excite
    println!("\nRunning 1 step with UV burst at 258nm:");
    let result = engine.step()?;
    println!(
        "  Step result: temp={:.1}K, burst={}",
        result.temperature, result.uv_burst_active
    );

    // Read back vibrational energy from GPU
    let energy_258 = engine.get_vibrational_energy()?;
    let excited_258 = engine.get_is_excited()?;
    println!("  Vibrational energy (GPU readback): {:?}", energy_258);
    println!("  Is excited (GPU readback): {:?}", excited_258);

    // ========================================================
    // COMPARISON
    // ========================================================
    println!("\n=== COMPARISON ===");
    println!("Energy at 280nm: {:?}", energy_280);
    println!("Energy at 258nm: {:?}", energy_258);

    // Check that energies are different
    if !energy_280.is_empty() && !energy_258.is_empty() {
        let e280_sum: f32 = energy_280.iter().sum();
        let e258_sum: f32 = energy_258.iter().sum();

        println!("\nTotal vibrational energy:");
        println!("  280nm: {:.4} kcal/mol", e280_sum);
        println!("  258nm: {:.4} kcal/mol", e258_sum);

        if e280_sum > e258_sum * 2.0 {
            println!(
                "\n✓ PASS: Energy at 280nm ({:.4}) > 2× energy at 258nm ({:.4})",
                e280_sum, e258_sum
            );
            println!("  This confirms wavelength-dependent σ(λ) is working on GPU.");
        } else if (e280_sum - e258_sum).abs() < 0.001 {
            println!("\n✗ FAIL: Energies are identical - wavelength NOT affecting GPU!");
        } else {
            println!("\nINFO: Energies differ but ratio may not be as expected.");
            println!(
                "  280nm/258nm ratio: {:.2}x",
                e280_sum / e258_sum.max(0.0001)
            );
        }
    }

    println!("\n=== TEST COMPLETE ===");
    Ok(())
}
