#![cfg(any())]
// DISABLED (obsolete): references module intentionally disabled in crates/prism-gpu/src/lib.rs — pending operator re-enable decision
//! Integration tests for Floyd-Warshall GPU kernel
//!
//! TESTING STRATEGY:
//! 1. Small synthetic graphs (10-100 vertices)
//! 2. Compare GPU output against CPU reference implementation
//! 3. Numerical equivalence within f32 tolerance (1e-5)
//! 4. Edge cases: empty, single vertex, disconnected components
//! 5. Performance benchmarks on target datasets
//!
//! TEST FIXTURES:
//! - Complete graphs (K5, K10)
//! - Path graphs (linear chains)
//! - Disconnected graphs
//! - Random graphs with known properties
//!
//! REFERENCE: PRISM GPU Plan §4.4 (Phase 4 APSP Testing)

use cudarc::driver::CudaContext;
use prism_gpu::FloydWarshallGpu;
use std::sync::Arc;

/// Tolerance for f32 floating-point comparison
const F32_TOLERANCE: f32 = 1e-5;

/// CPU reference implementation of Floyd-Warshall
///
/// Used for validating GPU results.
fn floyd_warshall_cpu(adjacency: &[Vec<usize>], num_vertices: usize) -> Vec<Vec<f32>> {
    let n = num_vertices;
    let mut dist = vec![vec![f32::INFINITY; n]; n];

    // Initialize diagonal and edges
    for i in 0..n {
        dist[i][i] = 0.0;
        for &j in &adjacency[i] {
            dist[i][j] = 1.0;
        }
    }

    // Floyd-Warshall algorithm
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[i][k] < f32::INFINITY && dist[k][j] < f32::INFINITY {
                    let new_dist = dist[i][k] + dist[k][j];
                    if new_dist < dist[i][j] {
                        dist[i][j] = new_dist;
                    }
                }
            }
        }
    }

    dist
}

/// Compares two distance matrices for numerical equivalence
fn assert_distances_equal(gpu: &[Vec<f32>], cpu: &[Vec<f32>], tolerance: f32) {
    assert_eq!(gpu.len(), cpu.len(), "Matrix size mismatch");

    for i in 0..gpu.len() {
        for j in 0..gpu[i].len() {
            let diff = (gpu[i][j] - cpu[i][j]).abs();

            if gpu[i][j].is_infinite() && cpu[i][j].is_infinite() {
                continue; // Both infinite is OK
            }

            assert!(
                diff < tolerance,
                "Distance mismatch at ({}, {}): GPU={}, CPU={}, diff={}",
                i,
                j,
                gpu[i][j],
                cpu[i][j],
                diff
            );
        }
    }
}

/// TEST: Small directed graph (4 vertices)
///
/// Graph: 0 -> 1 -> 2
///        |         ^
///        +----3----+
#[test]
#[ignore] // Requires GPU hardware
fn test_small_directed_graph() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
        .expect("Failed to create FloydWarshallGpu");

    let adjacency = vec![
        vec![1, 3], // 0 -> 1, 3
        vec![2],    // 1 -> 2
        vec![],     // 2 -> (none)
        vec![2],    // 3 -> 2
    ];
    let num_vertices = 4;

    // Compute on GPU
    let gpu_dist = fw
        .compute_apsp(&adjacency, num_vertices)
        .expect("GPU APSP failed");

    // Compute on CPU
    let cpu_dist = floyd_warshall_cpu(&adjacency, num_vertices);

    // Verify equivalence
    assert_distances_equal(&gpu_dist, &cpu_dist, F32_TOLERANCE);

    // Verify specific paths
    assert_eq!(gpu_dist[0][0], 0.0);
    assert_eq!(gpu_dist[0][1], 1.0);
    assert_eq!(gpu_dist[0][2], 2.0);
    assert_eq!(gpu_dist[0][3], 1.0);
    assert_eq!(gpu_dist[1][2], 1.0);
    assert_eq!(gpu_dist[3][2], 1.0);

    // Verify no path from 2 to others
    assert!(gpu_dist[2][0].is_infinite());
    assert!(gpu_dist[2][1].is_infinite());
    assert!(gpu_dist[2][3].is_infinite());

    println!("PASS: Small directed graph test");
}

/// TEST: Complete graph K5 (5 vertices, all connected)
#[test]
#[ignore]
fn test_complete_graph_k5() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
        .expect("Failed to create FloydWarshallGpu");

    let n = 5;
    let mut adjacency = vec![vec![]; n];

    // Connect all vertices (complete graph)
    for i in 0..n {
        for j in 0..n {
            if i != j {
                adjacency[i].push(j);
            }
        }
    }

    let gpu_dist = fw.compute_apsp(&adjacency, n).expect("GPU APSP failed");
    let cpu_dist = floyd_warshall_cpu(&adjacency, n);

    assert_distances_equal(&gpu_dist, &cpu_dist, F32_TOLERANCE);

    // In complete graph, all distances should be 1 or 0
    for i in 0..n {
        for j in 0..n {
            if i == j {
                assert_eq!(gpu_dist[i][j], 0.0);
            } else {
                assert_eq!(gpu_dist[i][j], 1.0);
            }
        }
    }

    println!("PASS: Complete graph K5 test");
}

/// TEST: Path graph (linear chain)
///
/// Graph: 0 -> 1 -> 2 -> 3 -> 4
#[test]
#[ignore]
fn test_path_graph() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
        .expect("Failed to create FloydWarshallGpu");

    let n = 10;
    let mut adjacency = vec![vec![]; n];

    // Create linear path
    for i in 0..(n - 1) {
        adjacency[i].push(i + 1);
    }

    let gpu_dist = fw.compute_apsp(&adjacency, n).expect("GPU APSP failed");
    let cpu_dist = floyd_warshall_cpu(&adjacency, n);

    assert_distances_equal(&gpu_dist, &cpu_dist, F32_TOLERANCE);

    // Verify distances are path lengths
    for i in 0..n {
        for j in i..n {
            assert_eq!(gpu_dist[i][j], (j - i) as f32);
        }
    }

    println!("PASS: Path graph test");
}

/// TEST: Disconnected graph (two components)
///
/// Graph: 0 <-> 1    2 <-> 3
#[test]
#[ignore]
fn test_disconnected_graph() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
        .expect("Failed to create FloydWarshallGpu");

    let adjacency = vec![
        vec![1], // 0 -> 1
        vec![0], // 1 -> 0
        vec![3], // 2 -> 3
        vec![2], // 3 -> 2
    ];
    let n = 4;

    let gpu_dist = fw.compute_apsp(&adjacency, n).expect("GPU APSP failed");
    let cpu_dist = floyd_warshall_cpu(&adjacency, n);

    assert_distances_equal(&gpu_dist, &cpu_dist, F32_TOLERANCE);

    // Verify intra-component distances
    assert_eq!(gpu_dist[0][1], 1.0);
    assert_eq!(gpu_dist[1][0], 1.0);
    assert_eq!(gpu_dist[2][3], 1.0);
    assert_eq!(gpu_dist[3][2], 1.0);

    // Verify no paths between components
    assert!(gpu_dist[0][2].is_infinite());
    assert!(gpu_dist[0][3].is_infinite());
    assert!(gpu_dist[1][2].is_infinite());
    assert!(gpu_dist[1][3].is_infinite());

    println!("PASS: Disconnected graph test");
}

/// TEST: Medium-sized random graph (100 vertices)
#[test]
#[ignore]
fn test_medium_random_graph() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
        .expect("Failed to create FloydWarshallGpu");

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 100;
    let edge_prob = 0.1; // 10% edge probability
    let mut adjacency = vec![vec![]; n];

    // Generate random graph
    for i in 0..n {
        for j in 0..n {
            if i != j && rng.gen::<f64>() < edge_prob {
                adjacency[i].push(j);
            }
        }
    }

    let start = std::time::Instant::now();
    let gpu_dist = fw.compute_apsp(&adjacency, n).expect("GPU APSP failed");
    let gpu_time = start.elapsed();

    let start = std::time::Instant::now();
    let cpu_dist = floyd_warshall_cpu(&adjacency, n);
    let cpu_time = start.elapsed();

    assert_distances_equal(&gpu_dist, &cpu_dist, F32_TOLERANCE);

    println!(
        "PASS: Medium random graph (n={}): GPU={:.3}s, CPU={:.3}s, speedup={:.2}x",
        n,
        gpu_time.as_secs_f64(),
        cpu_time.as_secs_f64(),
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );
}

/// BENCHMARK: Large graph (500 vertices, DSJC500 target)
///
/// Performance target: < 1.5 seconds on RTX 3060
#[test]
#[ignore]
fn benchmark_large_graph_500() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
        .expect("Failed to create FloydWarshallGpu");

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    let n = 500;
    let edge_prob = 0.05; // Sparse graph
    let mut adjacency = vec![vec![]; n];

    for i in 0..n {
        for j in 0..n {
            if i != j && rng.gen::<f64>() < edge_prob {
                adjacency[i].push(j);
            }
        }
    }

    // Warmup run
    let _ = fw.compute_apsp(&adjacency, n).expect("GPU APSP failed");

    // Benchmark run
    let start = std::time::Instant::now();
    let gpu_dist = fw.compute_apsp(&adjacency, n).expect("GPU APSP failed");
    let gpu_time = start.elapsed();

    println!("\n=== BENCHMARK: DSJC500 Target ===");
    println!("Graph size: {} vertices", n);
    println!("GPU time: {:.3}s", gpu_time.as_secs_f64());
    println!("Target: < 1.5s");

    if gpu_time.as_secs_f64() < 1.5 {
        println!("PASS: Target met!");
    } else {
        println!("WARN: Target not met (consider optimization)");
    }

    // Verify result is valid
    assert_eq!(gpu_dist.len(), n);
    assert_eq!(gpu_dist[0].len(), n);

    // Basic sanity checks
    for i in 0..n {
        assert_eq!(gpu_dist[i][i], 0.0, "Diagonal should be zero");
    }
}

/// TEST: Edge case - single vertex graph
#[test]
#[ignore]
fn test_single_vertex() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
        .expect("Failed to create FloydWarshallGpu");

    let adjacency = vec![vec![]];
    let gpu_dist = fw.compute_apsp(&adjacency, 1).expect("GPU APSP failed");

    assert_eq!(gpu_dist.len(), 1);
    assert_eq!(gpu_dist[0][0], 0.0);

    println!("PASS: Single vertex test");
}

/// TEST: Edge case - two isolated vertices
#[test]
#[ignore]
fn test_two_isolated_vertices() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
        .expect("Failed to create FloydWarshallGpu");

    let adjacency = vec![vec![], vec![]];
    let gpu_dist = fw.compute_apsp(&adjacency, 2).expect("GPU APSP failed");

    assert_eq!(gpu_dist[0][0], 0.0);
    assert_eq!(gpu_dist[1][1], 0.0);
    assert!(gpu_dist[0][1].is_infinite());
    assert!(gpu_dist[1][0].is_infinite());

    println!("PASS: Two isolated vertices test");
}
