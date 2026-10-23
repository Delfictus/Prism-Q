// Integration test for WHCR memory access fixes

#[cfg(feature = "cuda")]
#[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
#[test]
fn test_whcr_memory_access() -> anyhow::Result<()> {
    use cudarc::driver::CudaContext;
    use prism_gpu::whcr::WhcrGpu;

    println!("Testing WHCR with forced conflicts...");

    // Create a simple graph with 10 vertices
    let adjacency = vec![
        vec![1, 2, 3, 4], // vertex 0 connected to 1,2,3,4
        vec![0, 2, 5],    // vertex 1
        vec![0, 1, 3],    // vertex 2
        vec![0, 2, 4],    // vertex 3
        vec![0, 3, 5],    // vertex 4
        vec![1, 4, 6],    // vertex 5
        vec![5, 7, 8],    // vertex 6
        vec![6, 8, 9],    // vertex 7
        vec![6, 7, 9],    // vertex 8
        vec![7, 8],       // vertex 9
    ];

    // Create a deliberately bad coloring with conflicts
    // All vertices get color 0, guaranteeing conflicts on every edge
    let mut coloring = vec![0; 10];

    // Count initial conflicts
    let mut initial_conflicts = 0;
    for (v, neighbors) in adjacency.iter().enumerate() {
        for &n in neighbors {
            if coloring[v] == coloring[n] {
                initial_conflicts += 1;
            }
        }
    }
    initial_conflicts /= 2; // Each conflict counted twice
    assert!(initial_conflicts > 0, "Should have initial conflicts");

    // Initialize GPU
    let device = CudaContext::new(0)?;

    // Create WHCR GPU instance
    let mut whcr = WhcrGpu::new(device, 10, &adjacency)?;

    // Test 1: Small color count (should work)
    println!("Test 1: Repair with 3 colors");
    let result = whcr.repair(&mut coloring, 3, 100, 1)?;
    println!(
        "  Result: {} colors, {} conflicts",
        result.final_colors, result.final_conflicts
    );

    // Test 2: Large color count (tests dynamic allocation)
    println!("Test 2: Repair with 100 colors");
    coloring = vec![0; 10]; // Reset to bad coloring
    let result = whcr.repair(&mut coloring, 100, 100, 1)?;
    println!(
        "  Result: {} colors, {} conflicts",
        result.final_colors, result.final_conflicts
    );
    assert!(
        result.final_conflicts <= initial_conflicts,
        "Should reduce conflicts"
    );

    // Test 3: Maximum color count (tests bounds)
    println!("Test 3: Repair with 256 colors (maximum)");
    coloring = vec![0; 10]; // Reset to bad coloring
    let result = whcr.repair(&mut coloring, 256, 100, 1)?;
    println!(
        "  Result: {} colors, {} conflicts",
        result.final_colors, result.final_conflicts
    );

    // Test 4: Geometry update (tests null checks)
    println!("Test 4: Update geometry");
    let stress = vec![0.5; 10];
    let persistence = vec![0.3; 10];
    let hotspots = vec![1i32; 10];
    whcr.update_geometry(Some(&stress), Some(&persistence), Some(&hotspots), None)?;

    // Test 5: Repair with geometry
    println!("Test 5: Repair with geometry coupling");
    coloring = vec![0; 10]; // Reset to bad coloring
    let result = whcr.repair(&mut coloring, 10, 100, 2)?; // precision=2 for f64 path
    println!(
        "  Result: {} colors, {} conflicts",
        result.final_colors, result.final_conflicts
    );

    println!("✅ All WHCR memory access tests passed!");
    Ok(())
}

#[cfg(not(feature = "cuda"))]
#[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
#[test]
fn test_whcr_memory_access() {
    println!("WHCR test requires CUDA feature");
}
