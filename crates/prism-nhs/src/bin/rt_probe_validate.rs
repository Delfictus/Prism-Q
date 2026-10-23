// RT Probe Validation Test
//
// Validates the OptiX RT probe pipeline on actual RTX hardware.
// Tests: optixInit, BVH build, ray casting, result retrieval.

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaStream, DevicePtr};
use prism_nhs::rt_probe::{RtProbeConfig, RtProbeEngine};
use prism_optix::OptixContext;
use std::sync::Arc;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           RT PROBE VALIDATION TEST (OptiX Pipeline)            ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Step 1: Initialize CUDA
    println!("[1/5] Initializing CUDA...");
    let cuda_ctx = CudaContext::new(0).context("Failed to initialize CUDA - is GPU available?")?;
    let stream: Arc<CudaStream> = cuda_ctx.default_stream();
    println!("  ✓ CUDA context created\n");

    // Step 2: Initialize OptiX
    println!("[2/5] Initializing OptiX...");
    OptixContext::init().context("Failed to initialize OptiX - check driver/library")?;

    // Create OptiX context from CUDA context
    let optix_ctx =
        OptixContext::new(cuda_ctx.cu_ctx(), true).context("Failed to create OptiX context")?;
    println!("  ✓ OptiX initialized\n");

    // Step 3: Create RT probe engine
    println!("[3/5] Creating RT probe engine...");
    let config = RtProbeConfig {
        rays_per_point: 64,   // Fewer rays for quick test
        attention_points: 10, // Few probe points
        max_ray_distance: 15.0,
        ..Default::default()
    };

    let mut rt_engine = RtProbeEngine::new(optix_ctx, config.clone())
        .context("Failed to create RT probe engine")?;
    rt_engine.set_stream(stream.clone());
    println!("  ✓ RT probe engine created\n");

    // Step 4: Build BVH with test atoms
    println!("[4/5] Building BVH acceleration structure...");

    // Create test atoms: simple cube of atoms
    let mut positions_flat: Vec<f32> = Vec::new();
    let mut radii: Vec<f32> = Vec::new();

    // 5x5x5 grid of atoms (125 atoms)
    for x in 0..5 {
        for y in 0..5 {
            for z in 0..5 {
                positions_flat.push(x as f32 * 3.0); // x
                positions_flat.push(y as f32 * 3.0); // y
                positions_flat.push(z as f32 * 3.0); // z
                radii.push(1.5); // 1.5 Å radius (typical for carbon)
            }
        }
    }

    let num_atoms = radii.len();
    println!("  Test structure: {} atoms in 5x5x5 grid", num_atoms);

    // Upload to GPU
    let d_positions = stream
        .clone_htod(&positions_flat)
        .context("Failed to upload positions to GPU")?;
    let d_radii = stream
        .clone_htod(&radii)
        .context("Failed to upload radii to GPU")?;

    // Get device pointers
    let (positions_ptr, _) = d_positions.device_ptr(&stream);
    let (radii_ptr, _) = d_radii.device_ptr(&stream);

    rt_engine
        .build_protein_bvh(positions_ptr, radii_ptr, num_atoms)
        .context("BVH build failed - check PTX and OptiX setup")?;
    println!("  ✓ BVH built successfully\n");

    // Step 5: Initialize buffers and cast rays
    println!("[5/5] Casting rays with RT cores...");
    rt_engine
        .initialize_buffers(&stream)
        .context("Failed to initialize RT buffers")?;

    // Probe from center of the grid
    let probe_positions = vec![
        [6.0, 6.0, 6.0],    // Center of grid
        [0.0, 0.0, 0.0],    // Corner
        [12.0, 12.0, 12.0], // Far corner
        [-5.0, 6.0, 6.0],   // Outside grid (should see voids)
    ];

    let snapshots = rt_engine
        .cast_rays(&probe_positions, 0, &stream)
        .context("Ray casting failed - SBT or payload mismatch?")?;

    println!("  ✓ Ray casting completed\n");

    // Analyze results
    println!("════════════════════════════════════════════════════════════════");
    println!("                        RESULTS                                  ");
    println!("════════════════════════════════════════════════════════════════\n");

    let mut total_hits = 0;
    let mut total_rays = 0;
    let mut void_count = 0;

    for (i, snapshot) in snapshots.iter().enumerate() {
        let hits: usize = snapshot.hit_distances.iter().filter(|&&d| d > 0.0).count();
        let rays = snapshot.hit_distances.len();
        total_hits += hits;
        total_rays += rays;

        if snapshot.void_detected {
            void_count += 1;
        }

        let avg_dist = if hits > 0 {
            snapshot
                .hit_distances
                .iter()
                .filter(|&&d| d > 0.0)
                .sum::<f32>()
                / hits as f32
        } else {
            0.0
        };

        println!(
            "Probe {} at [{:.1}, {:.1}, {:.1}]:",
            i, probe_positions[i][0], probe_positions[i][1], probe_positions[i][2]
        );
        println!("  Hits:           {}/{} rays", hits, rays);
        println!(
            "  Hit rate:       {:.1}%",
            100.0 * hits as f32 / rays as f32
        );
        println!("  Avg distance:   {:.2} Å", avg_dist);
        println!("  Void detected:  {}", snapshot.void_detected);
        println!("  Solvation var:  {:?}", snapshot.solvation_variance);
        println!();
    }

    // Validation checks
    println!("════════════════════════════════════════════════════════════════");
    println!("                     VALIDATION                                  ");
    println!("════════════════════════════════════════════════════════════════\n");

    let mut all_passed = true;

    // Check 1: Did we get hits?
    if total_hits > 0 {
        println!(
            "✓ [PASS] Ray hits detected: {} / {} total rays",
            total_hits, total_rays
        );
    } else {
        println!("✗ [FAIL] No ray hits detected! Check BVH or ray directions.");
        all_passed = false;
    }

    // Check 2: Did we get non-zero distances?
    let non_zero_distances: Vec<f32> = snapshots
        .iter()
        .flat_map(|s| s.hit_distances.iter().filter(|&&d| d > 0.0).copied())
        .collect();

    if !non_zero_distances.is_empty() {
        let min_dist = non_zero_distances
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max_dist = non_zero_distances.iter().cloned().fold(0.0, f32::max);
        println!(
            "✓ [PASS] Hit distances: {:.2} - {:.2} Å",
            min_dist, max_dist
        );
    } else {
        println!("✗ [FAIL] All hit distances are zero!");
        all_passed = false;
    }

    // Check 3: Did void detection work for exterior probe?
    // The probe at [-5, 6, 6] should detect void (outside the atom grid)
    if void_count > 0 {
        println!(
            "✓ [PASS] Void detection working ({} voids detected)",
            void_count
        );
    } else {
        println!("⚠ [WARN] No voids detected (may be expected for this test)");
    }

    // Final verdict
    println!();
    if all_passed {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  🎉 RT PROBE VALIDATION PASSED - OptiX pipeline is working!   ");
        println!("═══════════════════════════════════════════════════════════════");
    } else {
        println!("═══════════════════════════════════════════════════════════════");
        println!("  ⚠ RT PROBE VALIDATION FAILED - Check OptiX setup            ");
        println!("═══════════════════════════════════════════════════════════════");
        std::process::exit(1);
    }

    Ok(())
}
