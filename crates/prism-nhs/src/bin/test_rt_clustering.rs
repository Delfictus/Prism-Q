//! Test RT-Core Accelerated Clustering
//!
//! Validates the full OptiX pipeline:
//! 1. Load rt_clustering.optixir module
//! 2. Build BVH from test positions
//! 3. Launch RT neighbor finding
//! 4. Verify clustering results

use anyhow::{Context, Result};
use std::sync::Arc;

use cudarc::driver::CudaContext;
use prism_nhs::{find_optixir_path, RtClusteringConfig, RtClusteringEngine};

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("=== RT-Core Clustering Test ===\n");

    // Initialize CUDA
    println!("[1/5] Initializing CUDA...");
    let context = CudaContext::new(0).context("Failed to create CUDA context")?;

    // Get device info
    println!("  GPU: CUDA Device 0");

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

    // Find and load the OptiX IR module
    println!("\n[3/5] Loading OptiX IR pipeline...");
    let optixir_path = find_optixir_path()
        .or_else(|| {
            // Explicit operator override; no compiled-in host path.
            std::env::var_os("PRISM_RT_CLUSTERING_OPTIXIR")
                .map(std::path::PathBuf::from)
                .filter(|p| p.is_absolute() && p.exists())
        })
        .context("Could not find rt_clustering.optixir")?;

    println!("  Loading from: {}", optixir_path.display());
    engine
        .load_pipeline(&optixir_path)
        .context("Failed to load OptiX pipeline")?;
    println!("  Pipeline loaded successfully!");

    // Generate test data: 3 clusters of points
    println!("\n[4/5] Generating test data...");
    let mut positions: Vec<f32> = Vec::new();

    // Cluster 1: centered at (0, 0, 0) - 50 points
    for i in 0..50 {
        let angle = (i as f32) * 0.125 * std::f32::consts::PI;
        let r = 1.0 + (i as f32) * 0.02;
        positions.push(r * angle.cos()); // x
        positions.push(r * angle.sin()); // y
        positions.push((i as f32) * 0.05); // z
    }

    // Cluster 2: centered at (20, 0, 0) - 50 points
    for i in 0..50 {
        let angle = (i as f32) * 0.125 * std::f32::consts::PI;
        let r = 1.0 + (i as f32) * 0.02;
        positions.push(20.0 + r * angle.cos());
        positions.push(r * angle.sin());
        positions.push((i as f32) * 0.05);
    }

    // Cluster 3: centered at (10, 20, 0) - 50 points
    for i in 0..50 {
        let angle = (i as f32) * 0.125 * std::f32::consts::PI;
        let r = 1.0 + (i as f32) * 0.02;
        positions.push(10.0 + r * angle.cos());
        positions.push(20.0 + r * angle.sin());
        positions.push((i as f32) * 0.05);
    }

    // Add some noise points (scattered far from clusters)
    for i in 0..10 {
        positions.push(50.0 + (i as f32) * 5.0);
        positions.push(50.0 + (i as f32) * 5.0);
        positions.push(50.0 + (i as f32) * 5.0);
    }

    let num_points = positions.len() / 3;
    println!("  Generated {} test points:", num_points);
    println!("    - Cluster 1: 50 points at (0, 0, 0)");
    println!("    - Cluster 2: 50 points at (20, 0, 0)");
    println!("    - Cluster 3: 50 points at (10, 20, 0)");
    println!("    - Noise: 10 scattered points");

    // Run clustering
    println!("\n[5/5] Running RT-core clustering...");
    let result = engine.cluster(&positions).context("Clustering failed")?;

    // Print results
    println!("\n=== Results ===");
    println!("  Total points:     {}", num_points);
    println!("  Clusters found:   {}", result.num_clusters);
    println!("  Neighbor pairs:   {}", result.total_neighbors);
    println!("  GPU time:         {:.2} ms", result.gpu_time_ms);

    // Analyze cluster assignments
    let mut cluster_counts: std::collections::HashMap<i32, usize> =
        std::collections::HashMap::new();
    for &cid in &result.cluster_ids {
        *cluster_counts.entry(cid).or_default() += 1;
    }

    println!("\n  Cluster distribution:");
    let mut sorted_clusters: Vec<_> = cluster_counts.iter().collect();
    sorted_clusters.sort_by_key(|(k, _)| *k);
    for (cid, count) in sorted_clusters {
        if *cid == -1 {
            println!("    Noise:      {} points", count);
        } else {
            println!("    Cluster {}: {} points", cid, count);
        }
    }

    // Validate results
    let noise_count = cluster_counts.get(&-1).copied().unwrap_or(0);
    let clustered_count = num_points - noise_count;

    println!("\n=== Validation ===");
    if result.num_clusters >= 1 {
        println!("  [PASS] Found at least 1 cluster");
    } else {
        println!(
            "  [FAIL] Expected at least 1 cluster, found {}",
            result.num_clusters
        );
    }

    if clustered_count > 0 {
        println!("  [PASS] {} points were clustered", clustered_count);
    } else {
        println!("  [WARN] No points were clustered");
    }

    if result.gpu_time_ms < 1000.0 {
        println!("  [PASS] Completed in {:.2}ms (< 1s)", result.gpu_time_ms);
    } else {
        println!("  [WARN] Slow: {:.2}ms", result.gpu_time_ms);
    }

    println!("\n=== RT Clustering Test Complete ===");
    Ok(())
}
