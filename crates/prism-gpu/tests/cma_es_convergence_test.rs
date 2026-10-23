#![cfg(any())]
// DISABLED (obsolete): references module intentionally disabled in crates/prism-gpu/src/lib.rs — pending operator re-enable decision
//! CMA-ES convergence test - verifies the algorithm actually optimizes
//!
//! Tests CMA-ES on standard optimization problems to ensure:
//! - Algorithm converges to optimal solutions
//! - GPU kernels produce correct results
//! - Performance meets targets
//!
//! Run with: cargo test -p prism-gpu --features cuda cma_es_convergence

use anyhow::Result;
use cudarc::driver::CudaContext;
use prism_gpu::cma_es::{CmaOptimizer, CmaParams};
use std::sync::Arc;

/// Test CMA-ES on sphere function: f(x) = sum(x_i^2)
/// Optimal solution: x* = [0, 0, ..., 0], f(x*) = 0
#[test]
fn test_cma_es_sphere_function() -> Result<()> {
    // Skip test if no CUDA device available
    let device = match CudaContext::new(0) {
        Ok(dev) => dev, // CudaContext::new already returns Arc<CudaContext>
        Err(_) => {
            eprintln!("Skipping test: No CUDA device available");
            return Ok(());
        }
    };

    println!("Testing CMA-ES on sphere function...");

    // Test parameters
    let dimensions = 10;
    let seed = 42;

    // Create optimizer
    let mut optimizer = CmaOptimizer::new(device, dimensions, seed)?;

    // Define sphere function
    let sphere_fitness = |params: &[f32]| -> f32 { params.iter().map(|x| x * x).sum() };

    // Run optimization
    let result = optimizer.optimize(
        sphere_fitness,
        100,  // max generations
        1e-6, // target fitness
        1e-8, // convergence tolerance
        1e14, // max condition number
    )?;

    // Verify convergence
    println!("Final generation: {}", result.generation);
    println!("Best fitness: {:.6e}", result.best_fitness);
    println!("Sigma: {:.6e}", result.sigma);
    println!("Condition number: {:.2e}", result.covariance_condition);
    println!(
        "Best solution (first 5): {:?}",
        &result.best_solution[..5.min(dimensions)]
    );

    // Assert convergence criteria
    assert!(
        result.best_fitness < 1e-4,
        "Failed to minimize sphere function: fitness = {}",
        result.best_fitness
    );

    // Check solution is near optimal
    for &x in &result.best_solution {
        assert!(x.abs() < 0.1, "Solution component too far from zero: {}", x);
    }

    println!("✓ CMA-ES successfully minimized sphere function");
    Ok(())
}

/// Test CMA-ES on Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
/// Optimal solution: x* = [1, 1], f(x*) = 0
#[test]
fn test_cma_es_rosenbrock() -> Result<()> {
    // Skip test if no CUDA device available
    let device = match CudaContext::new(0) {
        Ok(dev) => dev, // CudaContext::new already returns Arc<CudaContext>
        Err(_) => {
            eprintln!("Skipping test: No CUDA device available");
            return Ok(());
        }
    };

    println!("Testing CMA-ES on Rosenbrock function...");

    // Create 2D optimizer
    let mut optimizer = CmaOptimizer::new(device, 2, 123)?;

    // Define Rosenbrock function
    let rosenbrock_fitness = |params: &[f32]| -> f32 {
        let x = params[0];
        let y = params[1];
        (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
    };

    // Run optimization with more generations (Rosenbrock is harder)
    let result = optimizer.optimize(
        rosenbrock_fitness,
        500,  // max generations
        1e-4, // target fitness
        1e-6, // convergence tolerance
        1e14, // max condition number
    )?;

    println!("Final generation: {}", result.generation);
    println!("Best fitness: {:.6e}", result.best_fitness);
    println!(
        "Best solution: [{:.4}, {:.4}]",
        result.best_solution[0], result.best_solution[1]
    );

    // Check if close to optimal
    let x_opt = result.best_solution[0];
    let y_opt = result.best_solution[1];

    assert!(
        (x_opt - 1.0).abs() < 0.1,
        "X not close to optimal: {} (expected 1.0)",
        x_opt
    );
    assert!(
        (y_opt - 1.0).abs() < 0.1,
        "Y not close to optimal: {} (expected 1.0)",
        y_opt
    );

    println!("✓ CMA-ES successfully optimized Rosenbrock function");
    Ok(())
}

/// Test CMA-ES on high-dimensional problem
#[test]
fn test_cma_es_high_dimensional() -> Result<()> {
    // Skip test if no CUDA device available
    let device = match CudaContext::new(0) {
        Ok(dev) => dev, // CudaContext::new already returns Arc<CudaContext>
        Err(_) => {
            eprintln!("Skipping test: No CUDA device available");
            return Ok(());
        }
    };

    println!("Testing CMA-ES on 100-dimensional sphere...");

    let dimensions = 100;
    let mut optimizer = CmaOptimizer::new(device, dimensions, 999)?;

    // Shifted sphere: f(x) = sum((x_i - 0.5)^2)
    let shifted_sphere = |params: &[f32]| -> f32 { params.iter().map(|x| (x - 0.5).powi(2)).sum() };

    // Run with relaxed convergence (high-dim is harder)
    let result = optimizer.optimize(
        shifted_sphere,
        200,  // max generations
        1e-2, // relaxed target
        1e-4, // relaxed convergence
        1e14, // max condition
    )?;

    println!("Final generation: {}", result.generation);
    println!("Best fitness: {:.6e}", result.best_fitness);
    println!(
        "Mean distance from optimum: {:.4}",
        result
            .best_solution
            .iter()
            .map(|x| (x - 0.5).abs())
            .sum::<f32>()
            / dimensions as f32
    );

    // Check convergence (relaxed for high dimension)
    assert!(
        result.best_fitness < 1.0,
        "Failed to optimize high-dimensional problem: {}",
        result.best_fitness
    );

    println!("✓ CMA-ES handled high-dimensional optimization");
    Ok(())
}

/// Benchmark CMA-ES performance
#[test]
fn test_cma_es_performance() -> Result<()> {
    // Skip test if no CUDA device available
    let device = match CudaContext::new(0) {
        Ok(dev) => dev, // CudaContext::new already returns Arc<CudaContext>
        Err(_) => {
            eprintln!("Skipping test: No CUDA device available");
            return Ok(());
        }
    };

    println!("Benchmarking CMA-ES performance...");

    let dimensions = 50;
    let mut optimizer = CmaOptimizer::new(device, dimensions, 42)?;

    let start = std::time::Instant::now();

    // Simple sphere function
    let fitness_fn = |params: &[f32]| -> f32 { params.iter().map(|x| x * x).sum() };

    // Run fixed number of generations
    for _ in 0..10 {
        optimizer.step(fitness_fn)?;
    }

    let elapsed = start.elapsed();
    let ms_per_generation = elapsed.as_millis() / 10;

    println!(
        "Time per generation ({}D): {}ms",
        dimensions, ms_per_generation
    );

    // Performance assertions
    assert!(
        ms_per_generation < 100,
        "CMA-ES too slow: {}ms per generation (target < 100ms)",
        ms_per_generation
    );

    // Check GPU memory usage
    let state = optimizer.get_state();
    println!(
        "Final state - Generation: {}, Fitness: {:.6e}",
        state.generation, state.best_fitness
    );

    println!("✓ CMA-ES performance meets targets");
    Ok(())
}

/// Test CMA-ES state persistence and recovery
#[test]
fn test_cma_es_state_management() -> Result<()> {
    // Skip test if no CUDA device available
    let device = match CudaContext::new(0) {
        Ok(dev) => dev, // CudaContext::new already returns Arc<CudaContext>
        Err(_) => {
            eprintln!("Skipping test: No CUDA device available");
            return Ok(());
        }
    };

    println!("Testing CMA-ES state management...");

    let dimensions = 5;
    let mut optimizer = CmaOptimizer::new(device.clone(), dimensions, 42)?;

    let fitness_fn = |params: &[f32]| -> f32 { params.iter().map(|x| x * x).sum() };

    // Run 5 generations
    for _ in 0..5 {
        optimizer.step(fitness_fn)?;
    }

    // Get state
    let state1 = optimizer.get_state().clone();
    println!("State after 5 generations:");
    println!("  Generation: {}", state1.generation);
    println!("  Best fitness: {:.6e}", state1.best_fitness);
    println!("  Sigma: {:.6e}", state1.sigma);

    // Verify state fields
    assert_eq!(state1.generation, 5);
    assert!(state1.best_fitness < f32::INFINITY);
    assert!(state1.sigma > 0.0);
    assert_eq!(state1.best_solution.len(), dimensions);

    // Run 5 more generations
    for _ in 0..5 {
        optimizer.step(fitness_fn)?;
    }

    let state2 = optimizer.get_state();

    // Verify progress
    assert_eq!(state2.generation, 10);
    assert!(
        state2.best_fitness <= state1.best_fitness,
        "Fitness should improve or stay same"
    );

    // Test telemetry emission
    let telemetry = optimizer.emit_telemetry();
    assert!(telemetry.contains_key("cma_generation"));
    assert!(telemetry.contains_key("cma_best_fitness"));
    assert!(telemetry.contains_key("cma_sigma"));
    assert!(telemetry.contains_key("cma_condition"));

    println!("✓ CMA-ES state management works correctly");
    Ok(())
}

/// Test CMA-ES convergence detection
#[test]
fn test_cma_es_convergence_detection() -> Result<()> {
    // Skip test if no CUDA device available
    let device = match CudaContext::new(0) {
        Ok(dev) => dev, // CudaContext::new already returns Arc<CudaContext>
        Err(_) => {
            eprintln!("Skipping test: No CUDA device available");
            return Ok(());
        }
    };

    println!("Testing CMA-ES convergence detection...");

    let dimensions = 3;
    let mut optimizer = CmaOptimizer::new(device, dimensions, 42)?;

    // Easy problem that should converge quickly
    let easy_fitness = |params: &[f32]| -> f32 {
        params.iter().map(|x| x * x).sum::<f32>() * 0.01f32 // Scaled to converge faster
    };

    let result = optimizer.optimize(
        easy_fitness,
        1000,  // max generations (shouldn't need all)
        1e-10, // very small target
        1e-8,  // tight convergence
        1e14,  // max condition
    )?;

    println!("Converged at generation: {}", result.generation);
    println!("Final fitness: {:.6e}", result.best_fitness);
    println!("Convergence metric: {:.6e}", result.convergence_metric);

    // Should converge well before max generations
    assert!(
        result.generation < 500,
        "Should converge quickly on easy problem"
    );

    // Should reach very low fitness
    assert!(
        result.best_fitness < 1e-6,
        "Should achieve very low fitness on easy problem"
    );

    // Check convergence metric
    assert!(
        result.convergence_metric < 1e-6 || result.is_converged(1e-10, 1e-8, 1e14),
        "Convergence should be detected"
    );

    println!("✓ CMA-ES convergence detection works");
    Ok(())
}
