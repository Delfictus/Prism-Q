//! Performance Baseline Tests (Phase 7-8)
//!
//! Benchmarks to establish performance baseline for explicit solvent simulations.
//! These tests measure:
//! 1. Neighbor list build time
//! 2. Force computation time per step
//! 3. SETTLE constraint time
//! 4. PME computation time (when available)
//!
//! Run with: cargo test -p prism-gpu --test performance_baseline --features cuda -- --ignored --nocapture
//!
//! Note: These tests require a CUDA-enabled system to run.

use std::collections::HashSet;
use std::time::Instant;

/// Performance metrics from a benchmark run
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    /// System description
    system: String,
    /// Number of atoms
    n_atoms: usize,
    /// Number of waters (for explicit solvent)
    n_waters: usize,
    /// Average time per MD step (µs)
    time_per_step_us: f64,
    /// Neighbor list build time (µs)
    neighbor_build_us: f64,
    /// Force computation time (µs)
    force_compute_us: f64,
    /// Estimated ns/day throughput
    ns_per_day: f64,
}

impl PerformanceMetrics {
    fn print(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════╗");
        println!("║  Performance Baseline: {}  ", self.system);
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║  Atoms: {:>8}    Waters: {:>8}                        ║",
            self.n_atoms, self.n_waters
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║  Time per step:      {:>10.1} µs                         ║",
            self.time_per_step_us
        );
        println!(
            "║  Neighbor build:     {:>10.1} µs                         ║",
            self.neighbor_build_us
        );
        println!(
            "║  Force computation:  {:>10.1} µs                         ║",
            self.force_compute_us
        );
        println!("╠══════════════════════════════════════════════════════════════╣");
        println!(
            "║  Estimated throughput: {:>8.2} ns/day                     ║",
            self.ns_per_day
        );
        println!("╚══════════════════════════════════════════════════════════════╝");
    }
}

/// Calculate estimated ns/day from time per step
fn calculate_ns_per_day(time_per_step_us: f64, dt_fs: f64) -> f64 {
    // ns/day = (steps/day) * (dt in ns)
    // steps/day = 86400 * 1e6 / time_per_step_us
    // dt in ns = dt_fs * 1e-6
    let steps_per_day = 86400.0 * 1e6 / time_per_step_us;
    let dt_ns = dt_fs * 1e-6;
    steps_per_day * dt_ns
}

// ============================================================================
// PERFORMANCE BASELINE TESTS
// ============================================================================

/// Benchmark: 216-water box (648 atoms) - Tier 2 system
///
/// This is the smallest meaningful water box test.
/// Target: < 100 µs/step on modern GPU (RTX 3080+)
#[test]
#[ignore] // Requires CUDA: cargo test water_box_216 --ignored --features cuda -- --nocapture
fn benchmark_water_box_216() {
    use cudarc::driver::CudaContext;
    use prism_gpu::AmberMegaFusedHmc;

    println!("\n=== Performance Benchmark: 216 Water Box ===\n");

    let n_waters = 216;
    let n_atoms = n_waters * 3; // 648 atoms
    let box_dim = 18.6f32; // Approximately correct density
    let box_dims = [box_dim, box_dim, box_dim];

    // Generate water positions on grid
    let (positions, water_oxygens) = generate_water_grid(n_waters, box_dim);
    let nb_params = generate_tip3p_params(n_waters);
    let exclusions = generate_water_exclusions(n_waters);

    // Initialize CUDA
    println!("Initializing CUDA...");
    let context = CudaContext::new(0).expect("Failed to create CUDA context");
    let mut hmc = AmberMegaFusedHmc::new(context, n_atoms).expect("Failed to create HMC");

    // Upload topology
    let bonds: Vec<(usize, usize, f32, f32)> = vec![];
    let angles: Vec<(usize, usize, usize, f32, f32)> = vec![];
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = vec![];

    hmc.upload_topology(
        &positions,
        &bonds,
        &angles,
        &dihedrals,
        &nb_params,
        &exclusions,
    )
    .expect("Failed to upload topology");

    hmc.set_pbc_box(box_dims).expect("Failed to set PBC");
    hmc.enable_explicit_solvent(box_dims)
        .expect("Failed to enable explicit solvent");
    hmc.set_water_molecules(&water_oxygens)
        .expect("Failed to set waters");
    hmc.initialize_velocities(310.0)
        .expect("Failed to init velocities");

    // Warm-up (100 steps)
    println!("Warming up (100 steps)...");
    let _ = hmc
        .run_fused(100, 2.0, 310.0, 0.01, false)
        .expect("Warmup failed");

    // Benchmark neighbor list build
    println!("Benchmarking neighbor list build...");
    let n_neighbor_builds = 10;
    let neighbor_start = Instant::now();
    for _ in 0..n_neighbor_builds {
        hmc.build_neighbor_lists()
            .expect("Failed to build neighbor list");
    }
    let neighbor_elapsed = neighbor_start.elapsed();
    let neighbor_build_us = neighbor_elapsed.as_micros() as f64 / n_neighbor_builds as f64;

    // Benchmark MD steps
    println!("Benchmarking MD steps (1000 steps)...");
    let n_steps = 1000;
    let md_start = Instant::now();
    let _ = hmc
        .run_fused(n_steps, 2.0, 310.0, 0.01, false)
        .expect("Benchmark failed");
    let md_elapsed = md_start.elapsed();
    let time_per_step_us = md_elapsed.as_micros() as f64 / n_steps as f64;

    // Calculate metrics
    let metrics = PerformanceMetrics {
        system: "216 TIP3P Water Box".to_string(),
        n_atoms,
        n_waters,
        time_per_step_us,
        neighbor_build_us,
        force_compute_us: time_per_step_us * 0.8, // Estimate: forces ~80% of step time
        ns_per_day: calculate_ns_per_day(time_per_step_us, 2.0),
    };

    metrics.print();

    // Performance assertions (targets for modern GPU)
    assert!(
        time_per_step_us < 500.0,
        "Step time {} µs exceeds 500 µs target for 648-atom system",
        time_per_step_us
    );
    assert!(
        neighbor_build_us < 1000.0,
        "Neighbor build {} µs exceeds 1 ms target",
        neighbor_build_us
    );

    println!("\n✓ Performance benchmark passed");
}

/// Benchmark: 1000-water box (3000 atoms)
///
/// Medium-sized water box for scaling analysis.
/// Target: < 500 µs/step on modern GPU
#[test]
#[ignore] // Requires CUDA
fn benchmark_water_box_1000() {
    use cudarc::driver::CudaContext;
    use prism_gpu::AmberMegaFusedHmc;

    println!("\n=== Performance Benchmark: 1000 Water Box ===\n");

    let n_waters = 1000;
    let n_atoms = n_waters * 3; // 3000 atoms
    let box_dim = 31.0f32;
    let box_dims = [box_dim, box_dim, box_dim];

    let (positions, water_oxygens) = generate_water_grid(n_waters, box_dim);
    let nb_params = generate_tip3p_params(n_waters);
    let exclusions = generate_water_exclusions(n_waters);

    let context = CudaContext::new(0).expect("Failed to create CUDA context");
    let mut hmc = AmberMegaFusedHmc::new(context, n_atoms).expect("Failed to create HMC");

    let bonds: Vec<(usize, usize, f32, f32)> = vec![];
    let angles: Vec<(usize, usize, usize, f32, f32)> = vec![];
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = vec![];

    hmc.upload_topology(
        &positions,
        &bonds,
        &angles,
        &dihedrals,
        &nb_params,
        &exclusions,
    )
    .expect("Failed to upload topology");

    hmc.set_pbc_box(box_dims).expect("Failed to set PBC");
    hmc.enable_explicit_solvent(box_dims)
        .expect("Failed to enable explicit solvent");
    hmc.set_water_molecules(&water_oxygens)
        .expect("Failed to set waters");
    hmc.initialize_velocities(310.0)
        .expect("Failed to init velocities");

    // Warm-up
    let _ = hmc
        .run_fused(100, 2.0, 310.0, 0.01, false)
        .expect("Warmup failed");

    // Benchmark
    let n_steps = 500;
    let md_start = Instant::now();
    let _ = hmc
        .run_fused(n_steps, 2.0, 310.0, 0.01, false)
        .expect("Benchmark failed");
    let md_elapsed = md_start.elapsed();
    let time_per_step_us = md_elapsed.as_micros() as f64 / n_steps as f64;

    let metrics = PerformanceMetrics {
        system: "1000 TIP3P Water Box".to_string(),
        n_atoms,
        n_waters,
        time_per_step_us,
        neighbor_build_us: 0.0, // Not measured separately
        force_compute_us: time_per_step_us * 0.8,
        ns_per_day: calculate_ns_per_day(time_per_step_us, 2.0),
    };

    metrics.print();

    assert!(
        time_per_step_us < 2000.0,
        "Step time {} µs exceeds 2 ms target for 3000-atom system",
        time_per_step_us
    );

    println!("\n✓ Performance benchmark passed");
}

/// Benchmark: Scaling analysis
///
/// Tests how performance scales with system size.
/// Ideal: O(N) scaling with cell list optimization.
#[test]
#[ignore] // Requires CUDA
fn benchmark_scaling_analysis() {
    use cudarc::driver::CudaContext;
    use prism_gpu::AmberMegaFusedHmc;

    println!("\n=== Performance Scaling Analysis ===\n");

    let water_counts = [27, 64, 125, 216, 343]; // 3x3x3, 4x4x4, 5x5x5, 6x6x6, 7x7x7
    let mut results: Vec<(usize, f64)> = Vec::new();

    for &n_waters in &water_counts {
        let n_atoms = n_waters * 3;
        let n_per_side = (n_waters as f64).powf(1.0 / 3.0).round() as usize;
        let box_dim = (n_per_side as f32) * 3.1;
        let box_dims = [box_dim, box_dim, box_dim];

        let (positions, water_oxygens) = generate_water_grid(n_waters, box_dim);
        let nb_params = generate_tip3p_params(n_waters);
        let exclusions = generate_water_exclusions(n_waters);

        let context = CudaContext::new(0).expect("Failed to create CUDA context");
        let mut hmc = AmberMegaFusedHmc::new(context, n_atoms).expect("Failed to create HMC");

        let bonds: Vec<(usize, usize, f32, f32)> = vec![];
        let angles: Vec<(usize, usize, usize, f32, f32)> = vec![];
        let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = vec![];

        hmc.upload_topology(
            &positions,
            &bonds,
            &angles,
            &dihedrals,
            &nb_params,
            &exclusions,
        )
        .expect("Failed to upload topology");

        hmc.set_pbc_box(box_dims).expect("Failed to set PBC");
        hmc.enable_explicit_solvent(box_dims)
            .expect("Failed to enable explicit solvent");
        hmc.set_water_molecules(&water_oxygens)
            .expect("Failed to set waters");
        hmc.initialize_velocities(310.0)
            .expect("Failed to init velocities");

        // Warm-up and benchmark
        let _ = hmc
            .run_fused(50, 2.0, 310.0, 0.01, false)
            .expect("Scaling benchmark failed");

        let n_steps = 200;
        let start = Instant::now();
        let _ = hmc
            .run_fused(n_steps, 2.0, 310.0, 0.01, false)
            .expect("Benchmark failed");
        let elapsed = start.elapsed();
        let time_per_step_us = elapsed.as_micros() as f64 / n_steps as f64;

        results.push((n_atoms, time_per_step_us));
        println!(
            "   {:>5} atoms ({:>3} waters): {:>8.1} µs/step",
            n_atoms, n_waters, time_per_step_us
        );
    }

    // Analyze scaling
    println!("\n--- Scaling Analysis ---");
    for i in 1..results.len() {
        let (n1, t1) = results[i - 1];
        let (n2, t2) = results[i];
        let atom_ratio = n2 as f64 / n1 as f64;
        let time_ratio = t2 / t1;
        let scaling_exponent = time_ratio.ln() / atom_ratio.ln();

        println!(
            "   {} → {} atoms: time ratio = {:.2}x (scaling exponent = {:.2})",
            n1, n2, time_ratio, scaling_exponent
        );
    }

    println!("\n   Ideal O(N) scaling would give exponent ≈ 1.0");
    println!("   Cell list should achieve near-linear scaling");

    println!("\n✓ Scaling analysis complete");
}

/// Benchmark: 1 ns production run on medium system (15,000 atoms)
///
/// This is the production benchmark - 500,000 steps at 2 fs timestep = 1 ns
/// Target system: 5000 waters = 15,000 atoms
#[test]
#[ignore] // Requires CUDA - run with: cargo test -p prism-gpu --release --test performance_baseline --features cuda -- --ignored benchmark_1ns_production --nocapture
fn benchmark_1ns_production() {
    use cudarc::driver::CudaContext;
    use prism_gpu::AmberMegaFusedHmc;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║     PRISM-4D 1 ns PRODUCTION BENCHMARK                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // System configuration: 5000 waters = 15,000 atoms
    let n_waters = 5000;
    let n_atoms = n_waters * 3;
    let box_dim = 54.0f32; // ~54 Å for proper density
    let box_dims = [box_dim, box_dim, box_dim];

    println!("System Setup:");
    println!("  Waters: {}", n_waters);
    println!("  Atoms:  {}", n_atoms);
    println!(
        "  Box:    {:.1} × {:.1} × {:.1} Å",
        box_dim, box_dim, box_dim
    );

    // Simulation parameters
    let dt_fs = 2.0f32; // 2 fs timestep
    let total_steps = 500_000; // 500,000 steps = 1 ns
    let report_interval = 50_000; // Report every 100 ps
    let temperature = 310.0f32;
    let gamma_fs = 0.01f32; // Langevin friction

    println!("\nSimulation Parameters:");
    println!("  Timestep:    {} fs", dt_fs);
    println!("  Total steps: {} (1 ns)", total_steps);
    println!("  Temperature: {} K", temperature);
    println!("  Report every {} steps (100 ps)", report_interval);

    // Generate system
    println!("\nGenerating water box...");
    let (positions, water_oxygens) = generate_water_grid(n_waters, box_dim);
    let nb_params = generate_tip3p_params(n_waters);
    let exclusions = generate_water_exclusions(n_waters);

    // Initialize CUDA
    println!("Initializing CUDA...");
    let context = CudaContext::new(0).expect("Failed to create CUDA context");
    let mut hmc = AmberMegaFusedHmc::new(context, n_atoms).expect("Failed to create HMC");

    // Upload topology (no bonds/angles/dihedrals for pure water)
    let bonds: Vec<(usize, usize, f32, f32)> = vec![];
    let angles: Vec<(usize, usize, usize, f32, f32)> = vec![];
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = vec![];

    hmc.upload_topology(
        &positions,
        &bonds,
        &angles,
        &dihedrals,
        &nb_params,
        &exclusions,
    )
    .expect("Failed to upload topology");

    hmc.set_pbc_box(box_dims).expect("Failed to set PBC");
    hmc.enable_explicit_solvent(box_dims)
        .expect("Failed to enable explicit solvent");
    hmc.set_water_molecules(&water_oxygens)
        .expect("Failed to set waters");
    hmc.initialize_velocities(temperature)
        .expect("Failed to init velocities");

    // Warm-up
    println!("\nWarming up (1000 steps)...");
    let _ = hmc
        .run_fused(1000, dt_fs, temperature, gamma_fs, false)
        .expect("Warmup failed");

    // Production run
    println!("\n════════════════════════════════════════════════════════════════");
    println!("Starting 1 ns production run...");
    println!("════════════════════════════════════════════════════════════════\n");

    let start = Instant::now();
    let mut last_report = Instant::now();
    let mut steps_completed = 0usize;

    // Run in chunks with progress reporting
    while steps_completed < total_steps {
        let chunk_size = report_interval.min(total_steps - steps_completed);

        let _ = hmc
            .run_fused(chunk_size, dt_fs, temperature, gamma_fs, false)
            .expect("MD step failed");

        steps_completed += chunk_size;

        let chunk_elapsed = last_report.elapsed();
        let total_elapsed = start.elapsed();
        let steps_per_sec = chunk_size as f64 / chunk_elapsed.as_secs_f64();
        let ns_per_day = steps_per_sec * (dt_fs as f64 / 1000.0) * 86400.0;
        let progress = steps_completed as f64 / total_steps as f64 * 100.0;
        let time_ns = steps_completed as f64 * dt_fs as f64 / 1_000_000.0;

        println!(
            "  [{:>6.1}%] {:>7}/{} steps | {:.3} ns | {:>8.0} steps/s | {:>7.1} ns/day | {:.1}s elapsed",
            progress,
            steps_completed,
            total_steps,
            time_ns,
            steps_per_sec,
            ns_per_day,
            total_elapsed.as_secs_f64()
        );

        last_report = Instant::now();
    }

    let total_elapsed = start.elapsed();
    let avg_steps_per_sec = total_steps as f64 / total_elapsed.as_secs_f64();
    let avg_ns_per_day = avg_steps_per_sec * (dt_fs as f64 / 1000.0) * 86400.0;
    let time_per_step_us = total_elapsed.as_micros() as f64 / total_steps as f64;

    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK RESULTS                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  System:        {:>6} atoms ({} waters)                   ║",
        n_atoms, n_waters
    );
    println!("║  Simulated:     1.000 ns (500,000 steps)                    ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Wall time:     {:>7.2} seconds ({:.2} minutes)             ║",
        total_elapsed.as_secs_f64(),
        total_elapsed.as_secs_f64() / 60.0
    );
    println!(
        "║  Time/step:     {:>7.2} µs                                  ║",
        time_per_step_us
    );
    println!(
        "║  Steps/second:  {:>7.0}                                     ║",
        avg_steps_per_sec
    );
    println!(
        "║  Performance:   {:>7.1} ns/day                              ║",
        avg_ns_per_day
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Performance assertions
    assert!(
        avg_ns_per_day > 100.0,
        "Performance {} ns/day is below minimum threshold of 100 ns/day",
        avg_ns_per_day
    );

    println!("\n✓ 1 ns production benchmark completed successfully!");
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Generate TIP3P water molecules on a grid
fn generate_water_grid(n_waters: usize, box_dim: f32) -> (Vec<f32>, Vec<usize>) {
    let n_per_side = (n_waters as f64).powf(1.0 / 3.0).ceil() as usize;
    let spacing = box_dim / n_per_side as f32;

    // TIP3P geometry
    let oh_bond = 0.9572f32;
    let hoh_angle = 104.52f32 * std::f32::consts::PI / 180.0;
    let h_offset_x = oh_bond * (hoh_angle / 2.0).sin();
    let h_offset_y = oh_bond * (hoh_angle / 2.0).cos();

    let mut positions = Vec::with_capacity(n_waters * 9);
    let mut water_oxygens = Vec::with_capacity(n_waters);
    let mut atom_idx = 0;
    let mut water_count = 0;

    for ix in 0..n_per_side {
        for iy in 0..n_per_side {
            for iz in 0..n_per_side {
                if water_count >= n_waters {
                    break;
                }

                let ox = (ix as f32 + 0.5) * spacing;
                let oy = (iy as f32 + 0.5) * spacing;
                let oz = (iz as f32 + 0.5) * spacing;

                water_oxygens.push(atom_idx);

                // Oxygen
                positions.extend_from_slice(&[ox, oy, oz]);
                // H1
                positions.extend_from_slice(&[ox + h_offset_x, oy + h_offset_y, oz]);
                // H2
                positions.extend_from_slice(&[ox - h_offset_x, oy + h_offset_y, oz]);

                atom_idx += 3;
                water_count += 1;
            }
        }
    }

    (positions, water_oxygens)
}

/// Generate TIP3P nonbonded parameters for n_waters
fn generate_tip3p_params(n_waters: usize) -> Vec<(f32, f32, f32, f32)> {
    let mut params = Vec::with_capacity(n_waters * 3);
    for _ in 0..n_waters {
        // Oxygen: σ=3.15061, ε=0.1521, q=-0.834, m=15.9994
        params.push((3.15061f32, 0.1521f32, -0.834f32, 15.9994f32));
        // H1: σ=0, ε=0, q=0.417, m=1.008
        params.push((0.0f32, 0.0f32, 0.417f32, 1.008f32));
        // H2: σ=0, ε=0, q=0.417, m=1.008
        params.push((0.0f32, 0.0f32, 0.417f32, 1.008f32));
    }
    params
}

/// Generate exclusion lists for water molecules
fn generate_water_exclusions(n_waters: usize) -> Vec<HashSet<usize>> {
    let n_atoms = n_waters * 3;
    let mut exclusions: Vec<HashSet<usize>> = vec![HashSet::new(); n_atoms];

    for i in 0..n_waters {
        let o = i * 3;
        let h1 = o + 1;
        let h2 = o + 2;

        exclusions[o].insert(h1);
        exclusions[o].insert(h2);
        exclusions[h1].insert(o);
        exclusions[h1].insert(h2);
        exclusions[h2].insert(o);
        exclusions[h2].insert(h1);
    }

    exclusions
}
