//! Benchmark and integration tests for TDA GPU kernel
//!
//! Tests performance on DSJC benchmarks and reports timing.

use cudarc::driver::CudaContext;
use prism_gpu::tda::TdaGpu;
use std::time::Instant;

/// Helper: Initialize GPU for benchmarks
fn init_gpu() -> anyhow::Result<TdaGpu> {
    let device = CudaContext::new(0)?;
    TdaGpu::new(device, "target/ptx/tda.ptx")
}

/// Generate synthetic DSJC-like graph
///
/// Creates a random graph with specified density for benchmarking.
/// DSJC graphs are random graphs with edge probability p.
fn generate_synthetic_graph(n: usize, edge_probability: f32, seed: u64) -> Vec<Vec<usize>> {
    use std::collections::HashSet;

    // Simple LCG for reproducible randomness
    let mut rng_state = seed;
    let mut next_rand = || {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng_state >> 16) as f32 / 65536.0
    };

    let mut adjacency = vec![vec![]; n];
    let mut edges = HashSet::new();

    // Generate edges with probability p
    for u in 0..n {
        for v in (u + 1)..n {
            if next_rand() < edge_probability {
                adjacency[u].push(v);
                adjacency[v].push(u);
                edges.insert((u, v));
            }
        }
    }

    println!(
        "Generated synthetic graph: {} vertices, {} edges (density: {:.2}%)",
        n,
        edges.len(),
        (edges.len() as f64 / ((n * (n - 1) / 2) as f64)) * 100.0
    );

    adjacency
}

#[test]
#[ignore] // Requires CUDA hardware - run with: cargo test --features cuda bench_dsjc250 -- --ignored
fn bench_dsjc250_synthetic() {
    let tda = init_gpu().expect("Failed to initialize GPU");

    // DSJC250.5: 250 vertices, ~50% edge density
    let n = 250;
    let density = 0.5;
    let adjacency = generate_synthetic_graph(n, density, 12345);
    let num_edges: usize = adjacency.iter().map(|v| v.len()).sum::<usize>() / 2;

    println!("\n=== DSJC250 Benchmark ===");
    println!("Vertices: {}, Edges: {}", n, num_edges);

    // Warmup run
    let _ = tda.compute_betti_numbers(&adjacency, n, num_edges);

    // Benchmark Betti number computation
    let start = Instant::now();
    let (betti_0, betti_1) = tda
        .compute_betti_numbers(&adjacency, n, num_edges)
        .expect("Failed to compute Betti numbers");
    let betti_time = start.elapsed();

    println!("Betti numbers: β₀={}, β₁={}", betti_0, betti_1);
    println!(
        "Betti computation time: {:.3}ms",
        betti_time.as_secs_f64() * 1000.0
    );

    // Benchmark persistence/importance computation
    let start = Instant::now();
    let (persistence, importance) = tda
        .compute_persistence_and_importance(&adjacency, betti_0, betti_1)
        .expect("Failed to compute persistence");
    let persistence_time = start.elapsed();

    println!(
        "Persistence computation time: {:.3}ms",
        persistence_time.as_secs_f64() * 1000.0
    );

    // Total time
    let total_time = betti_time + persistence_time;
    println!("Total TDA time: {:.3}ms", total_time.as_secs_f64() * 1000.0);

    // Performance targets (from spec)
    let target_time_ms = 50.0;
    if total_time.as_secs_f64() * 1000.0 < target_time_ms {
        println!("✓ PASSED: Performance target met (<{}ms)", target_time_ms);
    } else {
        println!(
            "⚠ MISSED: Performance target (target: {}ms, actual: {:.3}ms)",
            target_time_ms,
            total_time.as_secs_f64() * 1000.0
        );
    }

    // Verify output validity
    assert!(betti_0 > 0 && betti_0 <= n, "Betti-0 out of valid range");
    assert!(betti_1 <= num_edges, "Betti-1 cannot exceed edge count");
    assert_eq!(persistence.len(), n, "Persistence vector wrong size");
    assert_eq!(importance.len(), n, "Importance vector wrong size");

    // Check top-k anchors
    let mut ranked: Vec<(usize, f32)> = importance.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 5 anchors by importance: {:?}", &ranked[..5.min(n)]);
}

#[test]
#[ignore] // Requires CUDA hardware
fn bench_dsjc500_synthetic() {
    let tda = init_gpu().expect("Failed to initialize GPU");

    // DSJC500.5: 500 vertices, ~50% edge density
    let n = 500;
    let density = 0.5;
    let adjacency = generate_synthetic_graph(n, density, 54321);
    let num_edges: usize = adjacency.iter().map(|v| v.len()).sum::<usize>() / 2;

    println!("\n=== DSJC500 Benchmark ===");
    println!("Vertices: {}, Edges: {}", n, num_edges);

    // Warmup
    let _ = tda.compute_betti_numbers(&adjacency, n, num_edges);

    // Benchmark
    let start = Instant::now();
    let (betti_0, betti_1) = tda
        .compute_betti_numbers(&adjacency, n, num_edges)
        .expect("Failed to compute Betti numbers");
    let betti_time = start.elapsed();

    let start = Instant::now();
    let (persistence, importance) = tda
        .compute_persistence_and_importance(&adjacency, betti_0, betti_1)
        .expect("Failed to compute persistence");
    let persistence_time = start.elapsed();

    let total_time = betti_time + persistence_time;

    println!("Betti numbers: β₀={}, β₁={}", betti_0, betti_1);
    println!("Betti time: {:.3}ms", betti_time.as_secs_f64() * 1000.0);
    println!(
        "Persistence time: {:.3}ms",
        persistence_time.as_secs_f64() * 1000.0
    );
    println!("Total TDA time: {:.3}ms", total_time.as_secs_f64() * 1000.0);

    // Performance target: <200ms for DSJC500
    let target_time_ms = 200.0;
    if total_time.as_secs_f64() * 1000.0 < target_time_ms {
        println!("✓ PASSED: Performance target met (<{}ms)", target_time_ms);
    } else {
        println!(
            "⚠ MISSED: Performance target (target: {}ms, actual: {:.3}ms)",
            target_time_ms,
            total_time.as_secs_f64() * 1000.0
        );
    }

    // Verify outputs
    assert!(betti_0 > 0 && betti_0 <= n);
    assert!(betti_1 <= num_edges);
    assert_eq!(persistence.len(), n);
    assert_eq!(importance.len(), n);

    // Check top-k anchors
    let mut ranked: Vec<(usize, f32)> = importance.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 5 anchors: {:?}", &ranked[..5.min(n)]);
}

#[test]
#[ignore] // Requires CUDA hardware
fn bench_scaling_analysis() {
    let tda = init_gpu().expect("Failed to initialize GPU");

    println!("\n=== TDA Scaling Analysis ===");
    println!(
        "{:<10} {:<10} {:<15} {:<15} {:<15}",
        "Vertices", "Edges", "Betti (ms)", "Persist (ms)", "Total (ms)"
    );
    println!("{:-<70}", "");

    let test_sizes = vec![
        (50, 0.5),
        (100, 0.5),
        (250, 0.5),
        (500, 0.5),
        (1000, 0.3), // Lower density for larger graphs
    ];

    for (n, density) in test_sizes {
        let adjacency = generate_synthetic_graph(n, density, n as u64 * 42);
        let num_edges: usize = adjacency.iter().map(|v| v.len()).sum::<usize>() / 2;

        // Warmup
        let _ = tda.compute_betti_numbers(&adjacency, n, num_edges);

        // Measure Betti
        let start = Instant::now();
        let (betti_0, betti_1) = tda
            .compute_betti_numbers(&adjacency, n, num_edges)
            .expect("Failed");
        let betti_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Measure persistence
        let start = Instant::now();
        let _ = tda
            .compute_persistence_and_importance(&adjacency, betti_0, betti_1)
            .expect("Failed");
        let persist_ms = start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = betti_ms + persist_ms;

        println!(
            "{:<10} {:<10} {:<15.3} {:<15.3} {:<15.3}",
            n, num_edges, betti_ms, persist_ms, total_ms
        );
    }

    println!("{:-<70}", "");
    println!("Scaling analysis complete. Review for performance regressions.");
}

/// CPU reference for performance comparison
#[test]
#[ignore] // Requires CUDA hardware
fn bench_gpu_vs_cpu_speedup() {
    let tda = init_gpu().expect("Failed to initialize GPU");

    let n = 250;
    let adjacency = generate_synthetic_graph(n, 0.5, 99999);
    let num_edges: usize = adjacency.iter().map(|v| v.len()).sum::<usize>() / 2;

    println!("\n=== GPU vs CPU Speedup (n={}, m={}) ===", n, num_edges);

    // GPU timing
    let start = Instant::now();
    let (gpu_b0, gpu_b1) = tda
        .compute_betti_numbers(&adjacency, n, num_edges)
        .expect("GPU failed");
    let gpu_time = start.elapsed();

    // CPU reference (union-find)
    let start = Instant::now();
    let (cpu_b0, cpu_b1) = cpu_union_find(&adjacency, n);
    let cpu_time = start.elapsed();

    println!(
        "GPU time: {:.3}ms (β₀={}, β₁={})",
        gpu_time.as_secs_f64() * 1000.0,
        gpu_b0,
        gpu_b1
    );
    println!(
        "CPU time: {:.3}ms (β₀={}, β₁={})",
        cpu_time.as_secs_f64() * 1000.0,
        cpu_b0,
        cpu_b1
    );

    // Verify correctness
    assert_eq!(gpu_b0, cpu_b0, "GPU and CPU Betti-0 mismatch");
    assert_eq!(gpu_b1, cpu_b1, "GPU and CPU Betti-1 mismatch");

    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    println!("Speedup: {:.2}x", speedup);

    if speedup > 1.0 {
        println!("✓ GPU faster than CPU");
    } else {
        println!("⚠ GPU slower than CPU (may indicate overhead for small graphs)");
    }
}

/// CPU union-find reference implementation
fn cpu_union_find(adjacency: &[Vec<usize>], n: usize) -> (usize, usize) {
    use std::collections::HashSet;

    let mut parent: Vec<usize> = (0..n).collect();

    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut [usize], x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px != py {
            parent[px] = py;
        }
    }

    let mut edge_count = 0;
    for (u, neighbors) in adjacency.iter().enumerate() {
        for &v in neighbors {
            if u < v {
                union(&mut parent, u, v);
                edge_count += 1;
            }
        }
    }

    let mut roots = HashSet::new();
    for i in 0..n {
        roots.insert(find(&mut parent, i));
    }
    let betti_0 = roots.len();
    let betti_1 = if edge_count >= n {
        edge_count - n + betti_0
    } else {
        0
    };

    (betti_0, betti_1)
}
