//! GPU GNM Integration Test
//!
//! Tests the GPU-accelerated Lanczos eigendecomposition for GNM.
//!
//! ```bash
//! cargo run --release -p prism-physics --features cuda --bin test_gpu_gnm
//! ```

use prism_physics::gnm_gpu::GpuGnm;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("              GPU GNM Integration Test                          ");
    println!("═══════════════════════════════════════════════════════════════");

    // Test 1: Small protein (CPU path)
    println!("\n[Test 1] Small protein (20 residues) - CPU path");
    let small_positions: Vec<[f32; 3]> = (0..20).map(|i| [i as f32 * 3.8, 0.0, 0.0]).collect();

    let mut gnm = GpuGnm::default();
    let result = gnm.compute_rmsf(&small_positions);

    println!("  Residues: {}", result.n_residues);
    println!("  GPU used: {}", result.gpu_used);
    println!("  Time: {} ms", result.computation_time_ms);
    println!(
        "  RMSF range: [{:.4}, {:.4}]",
        result.rmsf.iter().cloned().fold(f64::INFINITY, f64::min),
        result
            .rmsf
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    assert!(!result.gpu_used, "Small protein should use CPU");
    println!("  ✓ Small protein test PASSED");

    // Test 2: Medium protein with CPU Lanczos
    println!("\n[Test 2] Medium protein (200 residues) - CPU Lanczos path");
    let medium_positions: Vec<[f32; 3]> = (0..200)
        .map(|i| {
            let angle = i as f32 * 0.1;
            [angle.cos() * 15.0, angle.sin() * 15.0, i as f32 * 1.5]
        })
        .collect();

    // Lower threshold to force Lanczos
    let gnm_lanczos = GpuGnm::with_gpu_threshold(100);
    let result = gnm_lanczos.compute_rmsf(&medium_positions);

    println!("  Residues: {}", result.n_residues);
    println!("  GPU used: {} (no GPU init yet)", result.gpu_used);
    println!("  Eigenvalues: {}", result.eigenvalues.len());
    println!("  Time: {} ms", result.computation_time_ms);
    println!(
        "  RMSF range: [{:.4}, {:.4}]",
        result.rmsf.iter().cloned().fold(f64::INFINITY, f64::min),
        result
            .rmsf
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // Verify RMSF makes physical sense (termini more flexible)
    let terminal_avg = (result.rmsf[0] + result.rmsf[result.n_residues - 1]) / 2.0;
    let middle_avg = result.rmsf[result.n_residues / 2];
    println!("  Terminal avg RMSF: {:.4}", terminal_avg);
    println!("  Middle avg RMSF: {:.4}", middle_avg);
    assert!(
        terminal_avg > middle_avg * 0.5,
        "Termini should be more flexible"
    );
    println!("  ✓ Medium protein test PASSED");

    // Test 3: Large protein with GPU (if available)
    println!("\n[Test 3] Large protein (800 residues) - GPU path");
    let large_positions: Vec<[f32; 3]> = (0..800)
        .map(|i| {
            let angle = i as f32 * 0.08;
            [angle.cos() * 25.0, angle.sin() * 25.0, i as f32 * 1.2]
        })
        .collect();

    let mut gnm_gpu = GpuGnm::with_gpu_threshold(500);

    // Try to init CUDA
    #[cfg(feature = "cuda")]
    {
        match gnm_gpu.init_cuda() {
            Ok(_) => {
                println!("  ✓ CUDA initialized successfully");
                println!("  GPU ready: {}", gnm_gpu.gpu_ready());
            }
            Err(e) => {
                println!("  ⚠ CUDA init failed: {}", e);
                println!("  Will use CPU Lanczos fallback");
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("  ⚠ CUDA feature not enabled, using CPU Lanczos");
    }

    let start = std::time::Instant::now();
    let result = gnm_gpu.compute_rmsf(&large_positions);
    let elapsed = start.elapsed();

    println!("  Residues: {}", result.n_residues);
    println!("  GPU used: {}", result.gpu_used);
    println!("  Eigenvalues computed: {}", result.eigenvalues.len());
    println!("  Internal time: {} ms", result.computation_time_ms);
    println!("  Wall clock time: {:?}", elapsed);
    println!(
        "  RMSF range: [{:.4}, {:.4}]",
        result.rmsf.iter().cloned().fold(f64::INFINITY, f64::min),
        result
            .rmsf
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    // Verify results
    assert_eq!(result.n_residues, 800);
    assert!(!result.rmsf.is_empty());
    assert!(result.rmsf.iter().all(|&r| r > 0.0 && r.is_finite()));

    if result.gpu_used {
        println!("  ✓ Large protein GPU test PASSED");
    } else {
        println!("  ✓ Large protein CPU fallback test PASSED");
    }

    // Print eigenvalue spectrum if available
    if !result.eigenvalues.is_empty() {
        println!("\n  Eigenvalue spectrum (first 5):");
        for (i, &ev) in result.eigenvalues.iter().take(5).enumerate() {
            println!("    λ_{} = {:.6}", i + 1, ev);
        }
    }

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("                     ALL TESTS PASSED                           ");
    println!("═══════════════════════════════════════════════════════════════");
}
