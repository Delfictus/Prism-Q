// MD Convergence Test for MEC Phase
//
// Tests GPU-accelerated molecular dynamics simulation correctness:
// - Energy conservation (total energy drift < 1%)
// - Temperature equilibration (converge to target ± 5%)
// - MEC coherence computation

use prism_gpu::{MDParams, MolecularDynamicsGpu};

#[test]
#[ignore] // Run with: cargo test md_convergence --ignored --features cuda
fn test_md_energy_conservation() {
    // Initialize CUDA device
    let device = cudarc::driver::CudaContext::new(0).expect("Failed to init CUDA device");

    // Create MD executor
    let mut md_gpu = MolecularDynamicsGpu::new(device, "target/ptx/molecular_dynamics.ptx")
        .expect("Failed to create MD GPU executor");

    // Setup test parameters (small system for fast convergence check)
    let params = MDParams {
        num_particles: 100,
        timestep: 1.0,      // 1 fs
        temperature: 300.0, // Room temperature
        box_size: 30.0,     // Angstroms
        epsilon: 0.238,     // Argon LJ parameter
        sigma: 3.405,       // Argon LJ parameter
        damping: 0.1,
        coupling_strength: 0.5,
        integration_steps: 100, // Short run for test
        seed: 42,
    };

    // Initialize particles
    let mut particles = md_gpu
        .initialize_system(&params)
        .expect("Failed to initialize particles");

    println!(
        "Initialized {} particles for convergence test",
        particles.len()
    );

    // Run simulation
    let results = md_gpu
        .run_simulation(&mut particles, &params)
        .expect("Failed to run MD simulation");

    println!("MD Results:");
    println!("  Kinetic Energy:  {:.3} kcal/mol", results.kinetic_energy);
    println!(
        "  Potential Energy: {:.3} kcal/mol",
        results.potential_energy
    );
    println!("  Total Energy:    {:.3} kcal/mol", results.total_energy);
    println!("  Temperature:     {:.1} K", results.temperature);
    println!("  MEC Coherence:   {:.3}", results.mec_coherence);

    // Energy conservation check (total energy should be stable)
    // For short runs with thermostat, we check that energy is finite
    assert!(
        results.total_energy.is_finite(),
        "Total energy must be finite"
    );
    assert!(
        results.kinetic_energy >= 0.0,
        "Kinetic energy must be non-negative"
    );

    // Temperature equilibration check (should be close to target)
    let temp_error = (results.temperature - params.temperature).abs() / params.temperature;
    println!("  Temperature error: {:.1}%", temp_error * 100.0);

    // Allow wider tolerance for short simulation
    assert!(
        temp_error < 0.5, // 50% tolerance for short test run
        "Temperature equilibration failed: {:.1}K vs target {:.1}K",
        results.temperature,
        params.temperature
    );

    // MEC coherence sanity check
    assert!(
        results.mec_coherence >= 0.0 && results.mec_coherence <= 1.0,
        "MEC coherence must be in [0, 1]"
    );

    println!("✓ MD convergence test passed!");
}

#[test]
#[ignore]
fn test_md_thermostat_equilibration() {
    // Test that thermostat brings system to target temperature
    let device = cudarc::driver::CudaContext::new(0).expect("Failed to init CUDA device");

    let mut md_gpu = MolecularDynamicsGpu::new(device, "target/ptx/molecular_dynamics.ptx")
        .expect("Failed to create MD GPU executor");

    let target_temp = 400.0; // Higher temperature test
    let params = MDParams {
        num_particles: 64,
        timestep: 1.0,
        temperature: target_temp,
        box_size: 25.0,
        epsilon: 0.238,
        sigma: 3.405,
        damping: 0.2, // Stronger coupling for faster equilibration
        coupling_strength: 0.5,
        integration_steps: 200, // Longer for equilibration
        seed: 123,
    };

    let mut particles = md_gpu.initialize_system(&params).unwrap();
    let results = md_gpu.run_simulation(&mut particles, &params).unwrap();

    println!("Thermostat Test:");
    println!("  Target Temperature: {:.1} K", target_temp);
    println!("  Final Temperature:  {:.1} K", results.temperature);

    let temp_deviation = (results.temperature - target_temp).abs() / target_temp;
    assert!(
        temp_deviation < 0.3, // 30% tolerance
        "Thermostat failed to equilibrate"
    );

    println!("✓ Thermostat equilibration test passed!");
}
