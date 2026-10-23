//! GPU Wavelength Integration Test
//!
//! GOAL B: Verify that the GPU kernel receives and uses the wavelength parameter correctly.
//!
//! This test:
//! - Launches CUDA kernel (not CPU-only)
//! - Sets wavelength to 258nm then 280nm
//! - Reads back d_vibrational_energy buffer
//! - Asserts energy_280 > energy_258 for TRP (ratio > 1.5)
//! - Must FAIL if kernel doesn't receive wavelength correctly
//!
//! Run with: cargo test -p prism-nhs --features gpu --test gpu_wavelength_integration

#[cfg(feature = "gpu")]
mod gpu_tests {
    use anyhow::{Context, Result};
    use cudarc::driver::CudaContext;
    use prism_nhs::{
        fused_engine::{NhsAmberFusedEngine, TemperatureProtocol, UvProbeConfig},
        input::PrismPrepTopology,
    };
    use std::sync::Arc;
    use std::time::Instant;

    /// Integration test: GPU wavelength-dependent excitation
    ///
    /// This test PROVES that wavelength affects the GPU excitation path by:
    /// 1. Running a UV burst at 280nm (TRP peak) and measuring deposited energy
    /// 2. Running a UV burst at 258nm (TRP off-peak) and measuring deposited energy
    /// 3. Asserting that 280nm deposits significantly more energy than 258nm
    ///
    /// If the kernel ignores wavelength, energies would be identical → test FAILS.
    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn test_gpu_wavelength_launch_readback() -> Result<()> {
        let start = Instant::now();

        // Initialize CUDA context
        let ctx =
            CudaContext::new(0).context("CUDA context required - this test must run on GPU")?;

        // Load 1L2Y topology (has TRP for wavelength-dependent testing)
        // Try workspace root first, then package-relative path
        let topology_paths = [
            "data/curated_14/topologies/1L2Y_topology.json",
            "../../data/curated_14/topologies/1L2Y_topology.json",
        ];
        let topology_content = topology_paths
            .iter()
            .find_map(|p| std::fs::read_to_string(p).ok())
            .context("Failed to read 1L2Y topology - ensure data/curated_14/topologies/ exists")?;
        let topology: PrismPrepTopology =
            serde_json::from_str(&topology_content).context("Failed to parse topology")?;

        // Must have aromatics for this test to be valid
        assert!(topology.n_residues > 0, "Topology must have residues");

        // Test at 280nm (TRP peak)
        let energy_280 = run_uv_burst_and_get_energy(ctx.clone(), &topology, 280.0)?;

        // Re-initialize for 258nm test (fresh state)
        let ctx = CudaContext::new(0)?;
        let energy_258 = run_uv_burst_and_get_energy(ctx, &topology, 258.0)?;

        // === CRITICAL ASSERTION ===
        // At 280nm (TRP peak), energy should be significantly higher than at 258nm (off-peak)
        // TRP has peak absorption at 280nm with σ ≈ 8.5nm bandwidth
        // At 258nm, the Gaussian absorption cross-section is much lower
        //
        // Expected ratio: ~2-3x (depending on exact σ(λ) implementation)
        // We use 1.5 as a conservative threshold
        let ratio = energy_280 / energy_258.max(0.0001);

        println!("\n=== GPU Wavelength Test Results ===");
        println!("Energy at 280nm: {:.5} kcal/mol", energy_280);
        println!("Energy at 258nm: {:.5} kcal/mol", energy_258);
        println!("Ratio (280/258): {:.2}x", ratio);
        println!("Test duration: {:?}", start.elapsed());

        // The test MUST fail if wavelength is ignored (ratio would be ~1.0)
        assert!(
            ratio > 1.5,
            "FAIL: Wavelength not affecting GPU excitation! \
             Energy ratio {:.2} should be > 1.5. \
             280nm={:.5}, 258nm={:.5}",
            ratio,
            energy_280,
            energy_258
        );

        // Sanity check: both should be non-zero
        assert!(energy_280 > 0.0, "280nm excitation produced no energy");
        assert!(energy_258 > 0.0, "258nm excitation produced no energy");

        println!("PASS: GPU wavelength-dependent σ(λ) confirmed working");

        // Performance check: should complete in < 2s (allows for cold CUDA init)
        assert!(
            start.elapsed().as_secs_f64() < 2.0,
            "Test took too long: {:?}",
            start.elapsed()
        );

        Ok(())
    }

    /// Helper: Run a UV burst at specified wavelength and return total vibrational energy
    fn run_uv_burst_and_get_energy(
        ctx: Arc<CudaContext>,
        topology: &PrismPrepTopology,
        wavelength_nm: f32,
    ) -> Result<f32> {
        // Create engine
        let mut engine = NhsAmberFusedEngine::new(
            ctx, topology, 32,  // grid_dim
            1.5, // grid_spacing
        )
        .context("Failed to create NHS engine")?;

        // Skip test if no aromatics detected (can't test wavelength without them)
        if engine.n_aromatics() == 0 {
            return Err(anyhow::anyhow!(
                "No aromatics in topology - cannot test wavelength"
            ));
        }

        // Configure temperature (room temp, no ramp)
        let _ = engine.set_temperature_protocol(TemperatureProtocol {
            start_temp: 300.0,
            end_temp: 300.0,
            ramp_steps: 0,
            hold_steps: 100,
            cold_hold_steps: 0,
            current_step: 0,
        });

        // Configure UV burst at specific wavelength
        engine.set_uv_config(UvProbeConfig {
            burst_energy: 5.0,
            burst_interval: 1, // Burst every step
            burst_duration: 1,
            frequency_hopping_enabled: true,
            scan_wavelengths: vec![wavelength_nm], // Fixed wavelength
            current_wavelength_idx: 0,
            dwell_steps: 100,
            ..Default::default()
        });

        // Run one step with UV burst
        engine.step()?;

        // Read back vibrational energy from GPU
        let energies = engine.get_vibrational_energy()?;
        let total_energy: f32 = energies.iter().sum();

        Ok(total_energy)
    }
}

/// Stub test for non-GPU builds
#[cfg(not(feature = "gpu"))]
#[test]
fn test_gpu_wavelength_requires_gpu_feature() {
    println!("This test requires the 'gpu' feature.");
    println!("Run with: cargo test -p prism-nhs --features gpu --test gpu_wavelength_integration");
    // Not a failure - just skip on non-GPU builds
}
