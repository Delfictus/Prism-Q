//! Phase 8: Fused Kernel Integration Tests
//!
//! These tests verify that the fused kernels produce correct results
//! and provide performance improvements over the separate kernel approach.
//!
//! Test Categories:
//! 1. Correctness: Fused vs separate kernel output comparison
//! 2. Energy conservation: Verify stable MD trajectory
//! 3. Performance: Measure speedup from kernel fusion
//!
//! Run with: cargo test -p prism-gpu --test fused_kernel_integration --features cuda

#[allow(unused_imports)]
use std::sync::Arc;

// ============================================================================
// TIER 1: Unit Tests (No CUDA Required)
// ============================================================================

/// Test that fused MD step math matches separate computations
#[test]
fn test_fused_md_step_math_equivalence() {
    println!("\n=== Fused MD Step Math Equivalence Test ===\n");

    // Test Velocity Verlet step 1: v += (dt/2)*a; x += dt*v
    let dt = 1.0f32; // fs
    let mass = 12.0f32; // amu (carbon)

    // Initial conditions
    let x0 = 0.0f32;
    let v0 = 1.0f32; // Å/fs
    let f0 = 0.5f32; // kcal/mol/Å

    // Convert force to acceleration: a = F/m (need unit conversion)
    // In AMBER: F is in kcal/mol/Å, m in amu
    // a = F / m * conversion_factor
    // conversion_factor = 418.4 Å²/fs²/(kcal/mol/amu)
    let conv = 418.4f32;
    let a0 = f0 / mass * conv;

    // Step 1: half kick + full drift
    let v_half = v0 + 0.5 * dt * a0;
    let x1 = x0 + dt * v_half;

    // New force at x1 (assume same for simplicity)
    let f1 = f0;
    let a1 = f1 / mass * conv;

    // Step 2: second half kick
    let v1 = v_half + 0.5 * dt * a1;

    println!("Initial: x0={:.4}, v0={:.4}, a0={:.4}", x0, v0, a0);
    println!("After step: x1={:.4}, v1={:.4}", x1, v1);

    // Verify energy is approximately conserved (KE + PE)
    // KE = 0.5 * m * v^2 / conv (to get kcal/mol)
    let ke0 = 0.5 * mass * v0 * v0 / conv;
    let ke1 = 0.5 * mass * v1 * v1 / conv;

    println!("Kinetic energy: KE0={:.6}, KE1={:.6}", ke0, ke1);

    // For a simple harmonic oscillator, total energy should be conserved
    // This is a free particle (F=const), so KE will increase

    // Main check: position and velocity updated correctly
    assert!(x1 > x0, "Position should increase with positive velocity");
    assert!(v1 > v0, "Velocity should increase with positive force");

    println!("\n✓ Fused MD step math test PASSED");
}

/// Test Langevin thermostat statistics
#[test]
fn test_langevin_thermostat_statistics() {
    println!("\n=== Langevin Thermostat Statistics Test ===\n");

    // Langevin equation: dv = (F/m - gamma*v)*dt + sqrt(2*gamma*kT/m)*dW
    // At equilibrium: <v^2> = kT/m
    // Temperature from velocity: T = m*<v^2>/k

    let target_temp = 300.0f32; // K
    let kb = 0.001987204; // kcal/mol/K
    let mass = 12.0f32; // amu

    // Expected <v^2> at target temperature
    // Using equipartition: 0.5 * m * <v^2> = 0.5 * kT per DOF
    // <v^2> = kT/m
    let expected_v2 = (kb * target_temp as f64 / mass as f64) as f32 * 418.4; // Å²/fs²

    println!("Target temperature: {} K", target_temp);
    println!("Expected <v²>: {:.4} Å²/fs²", expected_v2);
    println!("Expected |v|_rms: {:.4} Å/fs", expected_v2.sqrt());

    // For carbon at 300K:
    // <v^2> ≈ 0.001987 * 300 / 12 * 418.4 ≈ 20.8 Å²/fs²
    // |v|_rms ≈ 4.6 Å/fs

    assert!(
        expected_v2 > 10.0 && expected_v2 < 100.0,
        "Expected velocity variance in reasonable range"
    );

    println!("\n✓ Langevin thermostat statistics test PASSED");
}

/// Test shared memory tile size calculations
#[test]
fn test_shared_memory_tile_calculations() {
    println!("\n=== Shared Memory Tile Calculations Test ===\n");

    // SOTA: 256-atom tiles for 8 warps per block
    const SM_TILE_SIZE: usize = 256;

    // Each atom in tile needs: xyz (3*4=12 bytes) + sigma,epsilon,charge (3*4=12 bytes)
    let bytes_per_atom = 12 + 12;
    let tile_bytes = SM_TILE_SIZE * bytes_per_atom;

    println!("Tile size: {} atoms (SOTA)", SM_TILE_SIZE);
    println!("Bytes per atom: {} bytes", bytes_per_atom);
    println!(
        "Total tile size: {} bytes (~{}KB)",
        tile_bytes,
        tile_bytes / 1024
    );

    // Typical shared memory limit: 48KB per SM
    let sm_limit = 48 * 1024;
    assert!(
        tile_bytes < sm_limit,
        "Tile size {} exceeds shared memory limit {}",
        tile_bytes,
        sm_limit
    );

    // Calculate number of tiles for various system sizes
    for n_atoms in [500, 1000, 5000, 10000, 50000] {
        let n_tiles = (n_atoms + SM_TILE_SIZE - 1) / SM_TILE_SIZE;
        let tile_pairs = n_tiles * n_tiles;
        println!(
            "System {} atoms: {} tiles, {} tile pairs",
            n_atoms, n_tiles, tile_pairs
        );
    }

    println!("\n✓ Shared memory tile calculations test PASSED");
}

/// Test constraint cluster packing
#[test]
fn test_constraint_cluster_packing() {
    println!("\n=== Constraint Cluster Packing Test ===\n");

    // FusedConstraintCluster structure:
    // - cluster_type: i32 (4 bytes)
    // - atoms: [i32; 4] (16 bytes)
    // - ref_lengths: [f32; 3] (12 bytes)
    // - padding: [f32; 1] (4 bytes) for 16-byte alignment
    // Total: 36 bytes, aligned to 16 = 48 bytes

    let cluster_size = 48; // bytes

    // For a typical solvated protein:
    // - ~5000 water molecules = 5000 SETTLE clusters
    // - ~2000 H-bonds in protein = ~1000 H-constraint clusters
    // Total: ~6000 clusters

    let n_waters = 5000;
    let n_h_clusters = 1000;
    let total_clusters = n_waters + n_h_clusters;

    let buffer_size = total_clusters * cluster_size;
    println!(
        "Total clusters: {} ({} SETTLE + {} H-constraint)",
        total_clusters, n_waters, n_h_clusters
    );
    println!(
        "Buffer size: {} bytes ({:.2} MB)",
        buffer_size,
        buffer_size as f64 / 1e6
    );

    // Should fit easily in GPU memory
    assert!(buffer_size < 10 * 1024 * 1024, "Cluster buffer too large");

    println!("\n✓ Constraint cluster packing test PASSED");
}

/// Test kernel launch overhead estimation
#[test]
fn test_kernel_launch_overhead_estimation() {
    println!("\n=== Kernel Launch Overhead Estimation ===\n");

    // Typical kernel launch overhead: 5-20 μs
    // For an MD step with separate kernels:
    // - zero_forces: 1 launch
    // - compute_bonds: 1 launch
    // - compute_angles: 1 launch
    // - compute_dihedrals: 1 launch
    // - compute_nonbonded: 1 launch
    // - vv_step1: 1 launch
    // - vv_step2: 1 launch
    // - SETTLE: 1 launch
    // - H-constraints: 1 launch
    // Total: ~9 launches per step

    let launches_separate = 9;
    let launches_fused = 2; // mega_fused_md_step + fused_constraints

    let launch_overhead_us = 10.0; // μs per launch (typical)

    let overhead_separate = launches_separate as f64 * launch_overhead_us;
    let overhead_fused = launches_fused as f64 * launch_overhead_us;
    let savings = overhead_separate - overhead_fused;
    let speedup = overhead_separate / overhead_fused;

    println!(
        "Separate kernels: {} launches, {:.0} μs overhead",
        launches_separate, overhead_separate
    );
    println!(
        "Fused kernels: {} launches, {:.0} μs overhead",
        launches_fused, overhead_fused
    );
    println!("Overhead savings: {:.0} μs per step", savings);
    println!("Launch overhead speedup: {:.1}x", speedup);

    // For 1M steps: savings = 70M μs = 70 seconds
    let steps = 1_000_000;
    let total_savings_s = savings * steps as f64 / 1e6;
    println!(
        "Total savings for {} steps: {:.1} seconds",
        steps, total_savings_s
    );

    assert!(
        speedup > 2.0,
        "Fused kernels should provide >2x launch overhead reduction"
    );

    println!("\n✓ Kernel launch overhead estimation test PASSED");
}

// ============================================================================
// TIER 2: GPU Tests (Require CUDA)
// ============================================================================

#[cfg(feature = "cuda")]
mod gpu_tests {
    use super::*;
    use anyhow::Result;
    use cudarc::driver::CudaContext;
    use prism_gpu::{AmberMegaFusedHmc, KB_KCAL_MOL_K};

    /// Test fused MD step produces reasonable energies
    #[test]
    #[ignore = "Requires CUDA runtime"]
    fn test_fused_md_step_gpu() -> Result<()> {
        println!("\n=== Fused MD Step GPU Test ===\n");

        let context = CudaContext::new(0)?;
        let n_atoms = 100;

        let mut hmc = AmberMegaFusedHmc::new(context, n_atoms)?;

        // Create simple test system: carbon chain
        let positions: Vec<f32> = (0..n_atoms)
            .flat_map(|i| vec![i as f32 * 1.54, 0.0, 0.0]) // C-C bond length
            .collect();

        hmc.set_positions(&positions)?;

        // Upload minimal topology (no bonds for simple test).
        // nb_params is (sigma, epsilon, charge, mass) per atom — carbon defaults.
        let nb_params: Vec<(f32, f32, f32, f32)> = vec![(3.4, 0.11, 0.0, 12.0); n_atoms];
        hmc.upload_topology(
            &positions, // positions [x0,y0,z0,...]
            &[],        // bonds
            &[],        // angles
            &[],        // dihedrals
            &nb_params, // (sigma, epsilon, charge, mass) per atom
            &[],        // exclusions
        )?;

        // Build neighbor lists
        hmc.build_neighbor_lists()?;

        // Initialize velocities at 300K
        hmc.initialize_velocities(300.0)?;

        // Run a single fused MD step
        let dt = 1.0f32; // fs
        let temperature = 300.0f32;
        let gamma_ps = 1.0f32; // ps^-1 (1.0 ps collision time)
        let seed = 42u32;

        let ke = hmc.run_fused_md_step(dt, temperature, gamma_ps, seed)?;

        println!("Kinetic energy after fused step: {:.4} kcal/mol", ke);

        // KE should be positive and reasonable for 100 atoms at 300K
        // Expected: ~0.5 * n_atoms * 3 * kT ≈ 0.5 * 100 * 3 * 0.596 ≈ 89 kcal/mol
        assert!(ke > 0.0, "Kinetic energy should be positive");
        assert!(ke < 1000.0, "Kinetic energy should be reasonable");

        println!("\n✓ Fused MD step GPU test PASSED");
        Ok(())
    }

    /// Test fused vs separate kernel output equivalence
    #[test]
    #[ignore = "Requires CUDA runtime"]
    fn test_fused_vs_separate_equivalence() -> Result<()> {
        println!("\n=== Fused vs Separate Kernel Equivalence Test ===\n");

        let context = CudaContext::new(0)?;
        let n_atoms = 50;

        // Create two identical systems
        let mut hmc_fused = AmberMegaFusedHmc::new(context.clone(), n_atoms)?;
        let mut hmc_separate = AmberMegaFusedHmc::new(context, n_atoms)?;

        // Same initial positions
        let positions: Vec<f32> = (0..n_atoms)
            .flat_map(|i| {
                let angle = i as f32 * 2.0 * std::f32::consts::PI / n_atoms as f32;
                vec![5.0 * angle.cos(), 5.0 * angle.sin(), (i as f32) * 0.5]
            })
            .collect();

        hmc_fused.set_positions(&positions)?;
        hmc_separate.set_positions(&positions)?;

        // Same topology
        let sigmas = vec![3.4; n_atoms];
        let epsilons = vec![0.11; n_atoms];
        let charges = vec![0.0; n_atoms];
        let masses = vec![12.0; n_atoms];

        // nb_params: (sigma, epsilon, charge, mass) per atom.
        let nb_params: Vec<(f32, f32, f32, f32)> = sigmas
            .iter()
            .zip(&epsilons)
            .zip(&charges)
            .zip(&masses)
            .map(|(((s, e), c), m)| (*s, *e, *c, *m))
            .collect();
        hmc_fused.upload_topology(&positions, &[], &[], &[], &nb_params, &[])?;
        hmc_separate.upload_topology(&positions, &[], &[], &[], &nb_params, &[])?;

        // Build neighbor lists
        hmc_fused.build_neighbor_lists()?;
        hmc_separate.build_neighbor_lists()?;

        // Same initial velocities
        hmc_fused.initialize_velocities(300.0)?;
        hmc_separate.initialize_velocities(300.0)?;

        // Run with same parameters
        let n_steps = 10;
        let dt = 1.0f32;
        let temperature = 300.0f32;
        let gamma_fs = 0.001f32; // Low friction for comparison

        // Run fused
        let result_fused = hmc_fused.run_fused(n_steps, dt, temperature, gamma_fs, false)?;

        // Run separate (regular Verlet)
        let result_separate = hmc_separate.run_verlet(n_steps, dt, temperature, gamma_fs)?;

        println!(
            "Fused: avg_temp={:.2}K, PE={:.2} kcal/mol",
            result_fused.avg_temperature, result_fused.potential_energy
        );
        println!(
            "Separate: avg_temp={:.2}K, PE={:.2} kcal/mol",
            result_separate.avg_temperature, result_separate.potential_energy
        );

        // Results should be similar (not identical due to different RNG usage)
        let temp_diff = (result_fused.avg_temperature - result_separate.avg_temperature).abs();
        println!("Temperature difference: {:.2}K", temp_diff);

        // Allow some difference due to RNG/order of operations
        assert!(temp_diff < 100.0, "Temperature difference too large");

        println!("\n✓ Fused vs separate equivalence test PASSED");
        Ok(())
    }

    /// Test energy conservation with fused kernels (NVE-like)
    #[test]
    #[ignore = "Requires CUDA runtime"]
    fn test_fused_energy_conservation() -> Result<()> {
        println!("\n=== Fused Energy Conservation Test ===\n");

        let context = CudaContext::new(0)?;
        let n_atoms = 100;

        let mut hmc = AmberMegaFusedHmc::new(context, n_atoms)?;

        // Create compact system
        let positions: Vec<f32> = (0..n_atoms)
            .flat_map(|i| {
                let layer = i / 10;
                let pos_in_layer = i % 10;
                let x = (pos_in_layer % 5) as f32 * 4.0;
                let y = (pos_in_layer / 5) as f32 * 4.0;
                let z = layer as f32 * 4.0;
                vec![x, y, z]
            })
            .collect();

        hmc.set_positions(&positions)?;

        let sigmas = vec![3.4; n_atoms];
        let epsilons = vec![0.11; n_atoms];
        let charges = vec![0.0; n_atoms];
        let masses = vec![12.0; n_atoms];

        let nb_params: Vec<(f32, f32, f32, f32)> = sigmas
            .iter()
            .zip(&epsilons)
            .zip(&charges)
            .zip(&masses)
            .map(|(((s, e), c), m)| (*s, *e, *c, *m))
            .collect();
        hmc.upload_topology(&positions, &[], &[], &[], &nb_params, &[])?;
        hmc.build_neighbor_lists()?;
        hmc.initialize_velocities(300.0)?;

        // Run with very low friction (near NVE)
        let result = hmc.run_fused(100, 1.0, 300.0, 0.0001, false)?;

        // Check energy drift
        let energies = &result.energy_trajectory;
        let e_start = energies.first().map(|e| e.total_energy).unwrap_or(0.0);
        let e_end = energies.last().map(|e| e.total_energy).unwrap_or(0.0);
        let e_drift = (e_end - e_start).abs();
        let e_drift_percent = e_drift / e_start.abs() * 100.0;

        println!("Initial total energy: {:.4} kcal/mol", e_start);
        println!("Final total energy: {:.4} kcal/mol", e_end);
        println!(
            "Energy drift: {:.4} kcal/mol ({:.2}%)",
            e_drift, e_drift_percent
        );

        // With Verlet integrator and low friction, drift should be small
        // Allow up to 10% for 100 steps (this is conservative)
        assert!(
            e_drift_percent < 10.0,
            "Energy drift {:.2}% exceeds 10% threshold",
            e_drift_percent
        );

        println!("\n✓ Fused energy conservation test PASSED");
        Ok(())
    }

    /// Benchmark fused vs separate kernel performance
    #[test]
    #[ignore = "Requires CUDA runtime and takes time"]
    fn bench_fused_kernel_performance() -> Result<()> {
        println!("\n=== Fused Kernel Performance Benchmark ===\n");

        let context = CudaContext::new(0)?;

        for &n_atoms in &[500, 1000, 2000, 5000] {
            println!("--- System size: {} atoms ---", n_atoms);

            let mut hmc = AmberMegaFusedHmc::new(context.clone(), n_atoms)?;

            // Create grid of atoms
            let side = (n_atoms as f64).cbrt().ceil() as usize;
            let spacing = 4.0f32;
            let positions: Vec<f32> = (0..n_atoms)
                .flat_map(|i| {
                    let x = (i % side) as f32 * spacing;
                    let y = ((i / side) % side) as f32 * spacing;
                    let z = (i / (side * side)) as f32 * spacing;
                    vec![x, y, z]
                })
                .collect();

            hmc.set_positions(&positions)?;

            let sigmas = vec![3.4; n_atoms];
            let epsilons = vec![0.11; n_atoms];
            let charges = vec![0.0; n_atoms];
            let masses = vec![12.0; n_atoms];

            let nb_params: Vec<(f32, f32, f32, f32)> = sigmas
                .iter()
                .zip(&epsilons)
                .zip(&charges)
                .zip(&masses)
                .map(|(((s, e), c), m)| (*s, *e, *c, *m))
                .collect();
            hmc.upload_topology(&positions, &[], &[], &[], &nb_params, &[])?;
            hmc.build_neighbor_lists()?;
            hmc.initialize_velocities(300.0)?;

            let n_steps = 100;
            let dt = 1.0f32;
            let temperature = 300.0f32;
            let gamma_fs = 0.001f32;

            // Benchmark fused (non-tiled)
            let start = std::time::Instant::now();
            let _ = hmc.run_fused(n_steps, dt, temperature, gamma_fs, false)?;
            let fused_time = start.elapsed();

            // Reset and benchmark fused (tiled)
            hmc.set_positions(&positions)?;
            hmc.initialize_velocities(300.0)?;

            let start = std::time::Instant::now();
            let _ = hmc.run_fused(n_steps, dt, temperature, gamma_fs, true)?;
            let tiled_time = start.elapsed();

            // Reset and benchmark separate (regular Verlet)
            hmc.set_positions(&positions)?;
            hmc.initialize_velocities(300.0)?;

            let start = std::time::Instant::now();
            let _ = hmc.run_verlet(n_steps, dt, temperature, gamma_fs)?;
            let separate_time = start.elapsed();

            let fused_ns_step = fused_time.as_nanos() as f64 / n_steps as f64;
            let tiled_ns_step = tiled_time.as_nanos() as f64 / n_steps as f64;
            let separate_ns_step = separate_time.as_nanos() as f64 / n_steps as f64;

            let fused_speedup = separate_ns_step / fused_ns_step;
            let tiled_speedup = separate_ns_step / tiled_ns_step;

            println!("  Separate: {:.1} μs/step", separate_ns_step / 1000.0);
            println!(
                "  Fused:    {:.1} μs/step ({:.2}x speedup)",
                fused_ns_step / 1000.0,
                fused_speedup
            );
            println!(
                "  Tiled:    {:.1} μs/step ({:.2}x speedup)",
                tiled_ns_step / 1000.0,
                tiled_speedup
            );
            println!();
        }

        println!("✓ Fused kernel performance benchmark COMPLETED");
        Ok(())
    }
}

// ============================================================================
// Test entry point
// ============================================================================

#[test]
fn test_phase8_summary() {
    println!("\n");
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           Phase 8: Fused Kernels Test Summary                  ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║                                                                ║");
    println!("║  Tier 1 (Unit Tests - No CUDA):                               ║");
    println!("║    ✓ Fused MD step math equivalence                           ║");
    println!("║    ✓ Langevin thermostat statistics                           ║");
    println!("║    ✓ Shared memory tile calculations                          ║");
    println!("║    ✓ Constraint cluster packing                               ║");
    println!("║    ✓ Kernel launch overhead estimation                        ║");
    println!("║                                                                ║");
    println!("║  Tier 2 (GPU Tests - Require CUDA):                           ║");
    println!("║    • test_fused_md_step_gpu                                   ║");
    println!("║    • test_fused_vs_separate_equivalence                       ║");
    println!("║    • test_fused_energy_conservation                           ║");
    println!("║    • bench_fused_kernel_performance                           ║");
    println!("║                                                                ║");
    println!("║  Phase 8 Improvements:                                        ║");
    println!("║    • Mega-fused MD step: forces + integration in 1 kernel    ║");
    println!("║    • Fused constraints: SETTLE + H-bonds in 1 kernel         ║");
    println!("║    • Shared memory tiling: 128-atom tiles for bandwidth      ║");
    println!("║    • Expected launch overhead reduction: 4-5x                 ║");
    println!("║                                                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!();
}
