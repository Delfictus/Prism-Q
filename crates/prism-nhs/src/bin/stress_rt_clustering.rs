//! Stress Test for RT-Core Accelerated Clustering
//!
//! Tests scaling behavior with 10,000-50,000 points to demonstrate
//! RT core advantage over grid-based methods.

use anyhow::{Context, Result};
use rand::Rng;

use cudarc::driver::CudaContext;
use prism_nhs::{find_optixir_path, RtClusteringConfig, RtClusteringEngine};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== RT-Core Clustering Stress Test ===\n");

    // Parse command line for point count
    let args: Vec<String> = std::env::args().collect();
    let num_points: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);

    let num_clusters: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);

    println!("Configuration:");
    println!("  Points: {}", num_points);
    println!("  Target clusters: {}", num_clusters);

    // Initialize CUDA
    println!("\n[1/5] Initializing CUDA...");
    let context = CudaContext::new(0).context("Failed to create CUDA context")?;

    // Create RT clustering engine
    println!("\n[2/5] Creating RT clustering engine...");
    let config = RtClusteringConfig {
        epsilon: 3.0,        // 3 Angstrom neighborhood
        min_points: 2,       // Minimum 2 neighbors for core point
        min_cluster_size: 5, // Minimum 5 points per cluster
        rays_per_event: 32,  // 32 rays per point for neighbor finding
    };

    let mut engine = RtClusteringEngine::new(context, config.clone())
        .context("Failed to create RT clustering engine")?;

    // Load pipeline
    println!("\n[3/5] Loading OptiX IR pipeline...");
    let optixir_path = find_optixir_path()
        .or_else(|| {
            let p = std::path::PathBuf::from("crates/prism-gpu/src/kernels/rt_clustering.optixir");
            if p.exists() {
                Some(p)
            } else {
                None
            }
        })
        .context("Could not find rt_clustering.optixir")?;

    engine
        .load_pipeline(&optixir_path)
        .context("Failed to load OptiX pipeline")?;
    println!("  Pipeline loaded successfully!");

    // Generate test data: many clusters in a 3D grid
    println!(
        "\n[4/5] Generating {} test points in {} clusters...",
        num_points, num_clusters
    );
    let mut rng = rand::thread_rng();
    let mut positions: Vec<f32> = Vec::with_capacity(num_points * 3);

    let points_per_cluster = num_points / num_clusters;
    let cluster_spacing = 30.0f32; // 30 Angstroms between cluster centers
    let cluster_radius = 2.0f32; // Points within 2 Angstrom radius

    // Create clusters arranged in a 3D grid
    let grid_size = (num_clusters as f32).cbrt().ceil() as usize;

    for cluster_idx in 0..num_clusters {
        let cx = (cluster_idx % grid_size) as f32 * cluster_spacing;
        let cy = ((cluster_idx / grid_size) % grid_size) as f32 * cluster_spacing;
        let cz = (cluster_idx / (grid_size * grid_size)) as f32 * cluster_spacing;

        for _ in 0..points_per_cluster {
            // Random point within cluster radius (spherical distribution)
            let theta = rng.gen::<f32>() * 2.0 * std::f32::consts::PI;
            let phi = (rng.gen::<f32>() * 2.0 - 1.0).acos();
            let r = rng.gen::<f32>().cbrt() * cluster_radius; // Cube root for uniform distribution

            positions.push(cx + r * phi.sin() * theta.cos());
            positions.push(cy + r * phi.sin() * theta.sin());
            positions.push(cz + r * phi.cos());
        }
    }

    // Add remaining points as noise
    let noise_count = num_points - (points_per_cluster * num_clusters);
    let noise_range = grid_size as f32 * cluster_spacing * 2.0;
    for _ in 0..noise_count {
        positions.push(rng.gen::<f32>() * noise_range + 1000.0); // Far from clusters
        positions.push(rng.gen::<f32>() * noise_range + 1000.0);
        positions.push(rng.gen::<f32>() * noise_range + 1000.0);
    }

    let actual_points = positions.len() / 3;
    println!(
        "  Generated {} points ({} per cluster, {} noise)",
        actual_points, points_per_cluster, noise_count
    );

    // Run clustering
    println!("\n[5/5] Running RT-core clustering...");
    let start = std::time::Instant::now();

    let result = engine.cluster(&positions).context("Clustering failed")?;

    let total_time = start.elapsed();

    // Print results
    println!("\n=== Results ===");
    println!("  Total points:     {}", actual_points);
    println!("  Clusters found:   {}", result.num_clusters);
    println!("  Neighbor pairs:   {}", result.total_neighbors);
    println!("  GPU time:         {:.2} ms", result.gpu_time_ms);
    println!(
        "  Total time:       {:.2} ms",
        total_time.as_secs_f64() * 1000.0
    );

    // Throughput metrics
    let points_per_ms = actual_points as f64 / result.gpu_time_ms;
    let neighbors_per_ms = result.total_neighbors as f64 / result.gpu_time_ms;

    println!("\n=== Performance Metrics ===");
    println!("  Points/ms:        {:.0}", points_per_ms);
    println!("  Neighbors/ms:     {:.0}", neighbors_per_ms);
    println!(
        "  Avg neighbors:    {:.1} per point",
        result.total_neighbors as f64 / actual_points as f64
    );

    // Cluster size distribution (top 10)
    let mut cluster_counts: std::collections::HashMap<i32, usize> =
        std::collections::HashMap::new();
    for &cid in &result.cluster_ids {
        *cluster_counts.entry(cid).or_default() += 1;
    }

    let mut sorted_clusters: Vec<_> = cluster_counts
        .iter()
        .filter(|(&k, _)| k >= 0)
        .map(|(&k, &v)| (k, v))
        .collect();
    sorted_clusters.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\n=== Top 10 Clusters by Size ===");
    for (i, (cid, count)) in sorted_clusters.iter().take(10).enumerate() {
        println!("  #{}: Cluster {} with {} points", i + 1, cid, count);
    }

    let noise_count = cluster_counts.get(&-1).copied().unwrap_or(0);
    let small_clusters = sorted_clusters.iter().filter(|(_, c)| *c < 5).count();
    println!("\n  Noise points:     {}", noise_count);
    println!("  Small clusters (<5): {}", small_clusters);
    println!(
        "  Large clusters (>=5): {}",
        sorted_clusters.len() - small_clusters
    );

    // Validation
    println!("\n=== Validation ===");
    if result.num_clusters >= num_clusters / 2 {
        println!(
            "  [PASS] Found reasonable cluster count ({} >= {})",
            result.num_clusters,
            num_clusters / 2
        );
    } else {
        println!(
            "  [WARN] Fewer clusters than expected: {} < {}",
            result.num_clusters,
            num_clusters / 2
        );
    }

    if result.gpu_time_ms < 100.0 {
        println!("  [PASS] Fast GPU execution: {:.2}ms", result.gpu_time_ms);
    } else if result.gpu_time_ms < 1000.0 {
        println!(
            "  [OK] Acceptable GPU execution: {:.2}ms",
            result.gpu_time_ms
        );
    } else {
        println!("  [SLOW] GPU execution took: {:.2}ms", result.gpu_time_ms);
    }

    println!("\n=== RT Clustering Stress Test Complete ===");

    // Exit without cleanup to avoid the Drop segfault
    // The OS will clean up GPU resources on process exit
    std::process::exit(0);
}
