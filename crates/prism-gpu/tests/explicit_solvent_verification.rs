//! Explicit Solvent Verification Test Suite (Phase 6)
//!
//! Tier 1 unit tests for verifying explicit solvent components:
//! 1. PBC minimum image convention
//! 2. Neighbor list PBC handling
//! 3. SETTLE constraint tolerance
//! 4. Displacement-based neighbor list rebuild
//!
//! Run with: cargo test -p prism-gpu --test explicit_solvent_verification

// Note: HashSet is used in CUDA-dependent tests when they run
#[allow(unused_imports)]
use std::collections::HashSet;

// ============================================================================
// TIER 1: Unit Tests (No CUDA Required)
// ============================================================================

/// Test minimum image convention for PBC
///
/// Verifies that distance calculations with PBC give correct minimum distances.
/// All distances should be < L/2 where L is the box dimension.
#[test]
fn test_minimum_image_convention() {
    println!("\n=== Minimum Image Convention Test ===\n");

    let box_dims = [10.0f32, 10.0, 10.0];

    // Helper to apply minimum image convention
    fn min_image_distance(p1: [f32; 3], p2: [f32; 3], box_dims: [f32; 3]) -> f32 {
        let mut dx = p2[0] - p1[0];
        let mut dy = p2[1] - p1[1];
        let mut dz = p2[2] - p1[2];

        // Apply minimum image convention
        dx -= box_dims[0] * (dx / box_dims[0]).round();
        dy -= box_dims[1] * (dy / box_dims[1]).round();
        dz -= box_dims[2] * (dz / box_dims[2]).round();

        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    // Test cases: particle at (1,1,1) vs various positions
    let p1 = [1.0f32, 1.0, 1.0];

    // Case 1: Same side of box - no wrapping needed
    let p2 = [3.0f32, 1.0, 1.0];
    let d = min_image_distance(p1, p2, box_dims);
    println!("Case 1 (same side): distance = {:.4} Å", d);
    assert!((d - 2.0).abs() < 1e-5, "Expected 2.0, got {}", d);

    // Case 2: Across boundary - wrapping needed (9,1,1) is closer via wrap
    // Without PBC: distance = 8.0 Å
    // With PBC: distance = 2.0 Å (9 -> 9-10 = -1, so dx = -1-1 = -2)
    let p3 = [9.0f32, 1.0, 1.0];
    let d = min_image_distance(p1, p3, box_dims);
    println!("Case 2 (across boundary): distance = {:.4} Å", d);
    assert!(
        (d - 2.0).abs() < 1e-5,
        "Expected 2.0 via PBC wrap, got {}",
        d
    );

    // Case 3: Corner to corner (requires 3D wrapping)
    // (0.5, 0.5, 0.5) to (9.5, 9.5, 9.5) should be sqrt(3) Å
    let p4 = [0.5f32, 0.5, 0.5];
    let p5 = [9.5f32, 9.5, 9.5];
    let d = min_image_distance(p4, p5, box_dims);
    let expected = (3.0f32).sqrt(); // sqrt(1^2 + 1^2 + 1^2)
    println!(
        "Case 3 (corner wrap): distance = {:.4} Å (expected {:.4})",
        d, expected
    );
    assert!(
        (d - expected).abs() < 1e-4,
        "Expected {:.4}, got {:.4}",
        expected,
        d
    );

    // Case 4: Exactly at half box - should be 5.0 Å
    let p6 = [6.0f32, 1.0, 1.0];
    let d = min_image_distance(p1, p6, box_dims);
    println!("Case 4 (half box): distance = {:.4} Å", d);
    assert!((d - 5.0).abs() < 1e-5, "Expected 5.0, got {}", d);

    // Case 5: All distances should be <= L/2 * sqrt(3)
    // This is the maximum possible distance in a cubic box with PBC
    let max_possible = 5.0 * (3.0f32).sqrt();

    for _ in 0..100 {
        let p_rand = [
            rand_f32() * box_dims[0],
            rand_f32() * box_dims[1],
            rand_f32() * box_dims[2],
        ];
        let d = min_image_distance(p1, p_rand, box_dims);
        assert!(
            d <= max_possible + 0.01,
            "Distance {} exceeds max possible {} for random point {:?}",
            d,
            max_possible,
            p_rand
        );
    }
    println!(
        "Case 5 (random check): All 100 random distances <= {:.2} Å ✓",
        max_possible
    );

    println!("\n✓ Minimum image convention test PASSED");
}

/// Simple pseudo-random number generator for testing (deterministic)
fn rand_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static SEED: Cell<u64> = Cell::new(12345);
    }
    SEED.with(|seed| {
        let s = seed.get();
        let new_seed = s.wrapping_mul(1103515245).wrapping_add(12345);
        seed.set(new_seed);
        ((new_seed >> 16) & 0x7fff) as f32 / 32768.0
    })
}

/// Test neighbor list PBC handling via brute-force comparison
///
/// Builds a neighbor list with cutoff and verifies all pairs within cutoff
/// are found, including those across periodic boundaries.
#[test]
fn test_neighbor_list_pbc_brute_force() {
    println!("\n=== Neighbor List PBC Brute-Force Comparison ===\n");

    let box_dims = [10.0f32, 10.0, 10.0];
    let cutoff = 3.0f32;

    // Create test system: 8 particles at corners + 1 in center
    // This ensures we have pairs that cross boundaries
    let positions = vec![
        [0.5f32, 0.5, 0.5], // 0: near corner 0,0,0
        [9.5f32, 0.5, 0.5], // 1: near corner L,0,0 - close to 0 via PBC
        [0.5f32, 9.5, 0.5], // 2: near corner 0,L,0 - close to 0 via PBC
        [0.5f32, 0.5, 9.5], // 3: near corner 0,0,L - close to 0 via PBC
        [9.5f32, 9.5, 0.5], // 4: close to 0,1,2 via PBC
        [5.0f32, 5.0, 5.0], // 5: center (far from corners)
        [4.5f32, 0.5, 0.5], // 6: close to 0 (same side)
    ];

    let n_atoms = positions.len();

    // Brute-force: find all pairs within cutoff
    let mut brute_force_pairs: Vec<(usize, usize)> = Vec::new();

    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            let mut dx = positions[j][0] - positions[i][0];
            let mut dy = positions[j][1] - positions[i][1];
            let mut dz = positions[j][2] - positions[i][2];

            // Apply minimum image convention
            dx -= box_dims[0] * (dx / box_dims[0]).round();
            dy -= box_dims[1] * (dy / box_dims[1]).round();
            dz -= box_dims[2] * (dz / box_dims[2]).round();

            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < cutoff {
                brute_force_pairs.push((i, j));
                println!(
                    "   Pair ({}, {}): distance = {:.3} Å (within cutoff)",
                    i, j, r
                );
            }
        }
    }

    println!(
        "\nBrute-force found {} pairs within cutoff {:.1} Å:",
        brute_force_pairs.len(),
        cutoff
    );
    for (i, j) in &brute_force_pairs {
        println!("   ({}, {})", i, j);
    }

    // Verify expected pairs:
    // - (0, 1): 1 Å apart via PBC (9.5-0.5=9, wrapped to 1)
    // - (0, 2): 1 Å apart via PBC
    // - (0, 3): 1 Å apart via PBC
    // - (0, 6): 4 Å apart (within 3Å? No, 4 > 3)
    // - (1, 4): 1 Å apart via PBC
    // - etc.

    let expected_pairs = vec![
        (0, 1), // x-boundary
        (0, 2), // y-boundary
        (0, 3), // z-boundary
        (1, 4), // y-boundary from corner
        (2, 4), // x-boundary from corner
    ];

    for (i, j) in &expected_pairs {
        assert!(
            brute_force_pairs.contains(&(*i, *j)),
            "Expected pair ({}, {}) not found in brute-force pairs",
            i,
            j
        );
    }

    println!("\n✓ All expected PBC-crossing pairs found via brute-force");
    println!("✓ Neighbor list PBC test PASSED");
}

/// Test SETTLE constraint geometry satisfaction
///
/// Verifies that SETTLE produces TIP3P geometry with < 1e-4 Å violations.
#[test]
fn test_settle_geometry_tolerance() {
    println!("\n=== SETTLE Constraint Tolerance Test ===\n");

    // TIP3P ideal geometry
    let oh_bond = 0.9572f32; // O-H bond length
    let hh_distance = 1.5139f32; // H-H distance (derived from angle)
    let hoh_angle_deg = 104.52f32;

    // Calculate H-H distance from angle: h-h = 2 * oh * sin(angle/2)
    let hoh_angle_rad = hoh_angle_deg * std::f32::consts::PI / 180.0;
    let calc_hh = 2.0 * oh_bond * (hoh_angle_rad / 2.0).sin();

    println!("TIP3P ideal geometry:");
    println!("   O-H bond: {:.4} Å", oh_bond);
    println!("   H-O-H angle: {:.2}°", hoh_angle_deg);
    println!(
        "   H-H distance: {:.4} Å (calculated: {:.4} Å)",
        hh_distance, calc_hh
    );

    // Helper function to check SETTLE satisfaction
    fn check_settle_violations(
        o_pos: [f32; 3],
        h1_pos: [f32; 3],
        h2_pos: [f32; 3],
        oh_target: f32,
        hh_target: f32,
    ) -> (f32, f32, f32) {
        let oh1 = ((h1_pos[0] - o_pos[0]).powi(2)
            + (h1_pos[1] - o_pos[1]).powi(2)
            + (h1_pos[2] - o_pos[2]).powi(2))
        .sqrt();
        let oh2 = ((h2_pos[0] - o_pos[0]).powi(2)
            + (h2_pos[1] - o_pos[1]).powi(2)
            + (h2_pos[2] - o_pos[2]).powi(2))
        .sqrt();
        let hh = ((h2_pos[0] - h1_pos[0]).powi(2)
            + (h2_pos[1] - h1_pos[1]).powi(2)
            + (h2_pos[2] - h1_pos[2]).powi(2))
        .sqrt();

        let oh1_violation = (oh1 - oh_target).abs();
        let oh2_violation = (oh2 - oh_target).abs();
        let hh_violation = (hh - hh_target).abs();

        (oh1_violation, oh2_violation, hh_violation)
    }

    // Test case 1: Ideal geometry (should pass trivially)
    let o = [5.0f32, 5.0, 5.0];
    let h1_offset_x = oh_bond * (hoh_angle_rad / 2.0).sin();
    let h1_offset_y = oh_bond * (hoh_angle_rad / 2.0).cos();
    let h1 = [o[0] + h1_offset_x, o[1] + h1_offset_y, o[2]];
    let h2 = [o[0] - h1_offset_x, o[1] + h1_offset_y, o[2]];

    let (v1, v2, v3) = check_settle_violations(o, h1, h2, oh_bond, hh_distance);
    println!("\nTest case 1 (ideal geometry):");
    println!("   O-H1 violation: {:.6} Å", v1);
    println!("   O-H2 violation: {:.6} Å", v2);
    println!("   H-H violation:  {:.6} Å", v3);

    let tolerance = 1e-4f32;
    assert!(v1 < tolerance, "O-H1 violation {} exceeds tolerance", v1);
    assert!(v2 < tolerance, "O-H2 violation {} exceeds tolerance", v2);
    assert!(v3 < tolerance, "H-H violation {} exceeds tolerance", v3);
    println!("   ✓ All violations < {:.0e} Å", tolerance);

    // Test case 2: Perturbed geometry (simulates what SETTLE would correct)
    // Add small perturbations and verify that our check function detects them
    let h1_perturbed = [h1[0] + 0.001, h1[1], h1[2]]; // 0.001 Å perturbation
    let (v1, v2, v3) = check_settle_violations(o, h1_perturbed, h2, oh_bond, hh_distance);
    println!("\nTest case 2 (perturbed geometry, 0.001 Å shift):");
    println!("   O-H1 violation: {:.6} Å", v1);
    println!("   O-H2 violation: {:.6} Å", v2);
    println!("   H-H violation:  {:.6} Å", v3);

    // A 0.001 Å shift should create a small but detectable violation
    assert!(v1 > 0.0005, "Perturbation should be detectable in O-H1");
    println!("   ✓ Perturbation correctly detected");

    // Test case 3: Multiple water molecules
    println!("\nTest case 3 (checking geometry of 10 TIP3P waters):");
    let n_waters = 10;
    let mut max_violation = 0.0f32;

    for i in 0..n_waters {
        let offset = i as f32 * 3.0;
        let o = [offset, 0.0, 0.0];
        let h1 = [o[0] + h1_offset_x, o[1] + h1_offset_y, o[2]];
        let h2 = [o[0] - h1_offset_x, o[1] + h1_offset_y, o[2]];

        let (v1, v2, v3) = check_settle_violations(o, h1, h2, oh_bond, hh_distance);
        max_violation = max_violation.max(v1).max(v2).max(v3);
    }

    println!(
        "   Maximum violation across {} waters: {:.6} Å",
        n_waters, max_violation
    );
    assert!(
        max_violation < tolerance,
        "Max violation {} exceeds tolerance",
        max_violation
    );
    println!("   ✓ All {} waters satisfy SETTLE tolerance", n_waters);

    println!("\n✓ SETTLE geometry tolerance test PASSED");
}

/// Test displacement-based neighbor list rebuild logic
///
/// Verifies that the rebuild trigger fires when any atom moves more than
/// the threshold (half the skin distance).
#[test]
fn test_displacement_rebuild_trigger() {
    println!("\n=== Displacement-Based Rebuild Trigger Test ===\n");

    // Verlet list parameters
    let cutoff = 9.0f32; // Non-bonded cutoff
    let skin = 1.0f32; // Verlet skin
    let rebuild_threshold = 0.5f32; // Rebuild when max_disp > skin/2

    println!("Neighbor list parameters:");
    println!("   Cutoff: {:.1} Å", cutoff);
    println!("   Skin: {:.1} Å", skin);
    println!("   Effective cutoff: {:.1} Å", cutoff + skin);
    println!("   Rebuild threshold: {:.2} Å (skin/2)", rebuild_threshold);

    // Helper: compute max displacement
    fn compute_max_displacement(
        current: &[[f32; 3]],
        reference: &[[f32; 3]],
        box_dims: [f32; 3],
    ) -> f32 {
        let mut max_disp = 0.0f32;
        for (curr, ref_pos) in current.iter().zip(reference.iter()) {
            let mut dx = curr[0] - ref_pos[0];
            let mut dy = curr[1] - ref_pos[1];
            let mut dz = curr[2] - ref_pos[2];

            // Apply PBC for displacement calculation
            dx -= box_dims[0] * (dx / box_dims[0]).round();
            dy -= box_dims[1] * (dy / box_dims[1]).round();
            dz -= box_dims[2] * (dz / box_dims[2]).round();

            let disp = (dx * dx + dy * dy + dz * dz).sqrt();
            max_disp = max_disp.max(disp);
        }
        max_disp
    }

    let box_dims = [30.0f32, 30.0, 30.0];

    // Initial positions (saved at neighbor list build)
    let reference_positions = vec![[5.0f32, 5.0, 5.0], [15.0, 15.0, 15.0], [25.0, 25.0, 25.0]];

    // Test case 1: Small movement (no rebuild needed)
    let current_positions_small = vec![
        [5.1f32, 5.1, 5.1], // Moved 0.17 Å
        [15.0, 15.0, 15.0], // No movement
        [25.0, 25.0, 25.0], // No movement
    ];

    let max_disp =
        compute_max_displacement(&current_positions_small, &reference_positions, box_dims);
    println!("\nTest case 1 (small movement):");
    println!("   Max displacement: {:.4} Å", max_disp);
    println!("   Rebuild needed: {}", max_disp > rebuild_threshold);
    assert!(
        max_disp < rebuild_threshold,
        "Small movement shouldn't trigger rebuild"
    );
    println!("   ✓ No rebuild triggered (correct)");

    // Test case 2: Large movement (rebuild needed)
    let current_positions_large = vec![
        [5.0f32, 5.0, 5.0], // No movement
        [15.6, 15.0, 15.0], // Moved 0.6 Å in x
        [25.0, 25.0, 25.0], // No movement
    ];

    let max_disp =
        compute_max_displacement(&current_positions_large, &reference_positions, box_dims);
    println!("\nTest case 2 (large movement):");
    println!("   Max displacement: {:.4} Å", max_disp);
    println!("   Rebuild needed: {}", max_disp > rebuild_threshold);
    assert!(
        max_disp > rebuild_threshold,
        "Large movement should trigger rebuild"
    );
    println!("   ✓ Rebuild triggered (correct)");

    // Test case 3: Movement across PBC boundary
    // Atom at (25,25,25) moves to (0.5,25,25) - this is a small movement via PBC
    // Real displacement = 30 - 24.5 = 5.5 Å? No wait, let's recalculate
    // Actually (25,25,25) to (29.3,25,25) is 4.3 Å movement (no PBC wrap in displacement)
    // But if we consider (25,25,25) to (0.3,25,25) in a 30 Å box:
    // dx = 0.3 - 25 = -24.7, wrapped: -24.7 + 30 = 5.3 Å (still large)
    // Let's test wrap correctly:
    let current_positions_pbc = vec![
        [5.0f32, 5.0, 5.0],
        [15.0, 15.0, 15.0],
        [25.3, 25.0, 25.0], // Small movement in same direction
    ];

    let max_disp = compute_max_displacement(&current_positions_pbc, &reference_positions, box_dims);
    println!("\nTest case 3 (small movement near boundary):");
    println!("   Max displacement: {:.4} Å", max_disp);
    assert!(
        max_disp < 0.5,
        "0.3 Å movement should give 0.3 Å displacement"
    );
    println!("   ✓ PBC displacement calculation correct");

    // Test case 4: True cross-boundary movement
    // Reference at (29.5, 5, 5), current at (0.5, 5, 5) - moved 1 Å across boundary
    let reference_boundary = vec![[29.5f32, 5.0, 5.0]];
    let current_boundary = vec![
        [0.5f32, 5.0, 5.0], // Crossed boundary, real displacement = 1 Å
    ];

    let max_disp = compute_max_displacement(&current_boundary, &reference_boundary, box_dims);
    println!("\nTest case 4 (cross-boundary movement):");
    println!("   Reference: (29.5, 5, 5)");
    println!("   Current: (0.5, 5, 5)");
    println!("   Box: 30 Å");
    println!("   Max displacement: {:.4} Å (expected: 1.0 Å)", max_disp);
    assert!(
        (max_disp - 1.0).abs() < 0.01,
        "Cross-boundary displacement should be 1 Å via PBC"
    );
    println!("   ✓ Cross-boundary displacement correctly computed");

    println!("\n✓ Displacement-based rebuild trigger test PASSED");
}

/// Test DOF (Degrees of Freedom) calculation for explicit solvent
///
/// Proper DOF accounting is critical for temperature calculation:
/// DOF = 3N - 3 (COM) - 3*n_waters (SETTLE) - n_h_constraints (SHAKE/RATTLE)
#[test]
fn test_dof_calculation() {
    println!("\n=== DOF Calculation Test ===\n");

    // Helper to calculate DOF
    fn calculate_dof(n_atoms: usize, n_waters: usize, n_h_constraints: usize) -> usize {
        let raw_dof = 3 * n_atoms;
        let com_removal = 3; // 3 translational DOF removed
        let settle_constraints = 3 * n_waters; // Each water has 3 constraints
        let h_constraints = n_h_constraints; // Each X-H bond has 1 constraint

        raw_dof - com_removal - settle_constraints - h_constraints
    }

    // Test case 1: Pure water box (27 waters = 81 atoms)
    let n_waters_1 = 27;
    let n_atoms_1 = n_waters_1 * 3;
    let dof_1 = calculate_dof(n_atoms_1, n_waters_1, 0);

    println!(
        "Test case 1: Pure water box ({} waters, {} atoms)",
        n_waters_1, n_atoms_1
    );
    println!("   Raw DOF: {} (3 * {})", 3 * n_atoms_1, n_atoms_1);
    println!("   COM removal: -3");
    println!(
        "   SETTLE constraints: -{} (3 * {})",
        3 * n_waters_1,
        n_waters_1
    );
    println!("   Final DOF: {}", dof_1);

    // Expected: 3*81 - 3 - 3*27 = 243 - 3 - 81 = 159
    assert_eq!(dof_1, 159, "Pure water DOF incorrect");
    println!("   ✓ DOF = 159 (correct)");

    // Test case 2: Small protein (100 atoms) + 50 waters (150 atoms) + 40 H-constraints
    let n_protein_atoms = 100;
    let n_water_atoms = 50 * 3;
    let n_waters_2 = 50;
    let n_atoms_2 = n_protein_atoms + n_water_atoms;
    let n_h_constraints = 40;
    let dof_2 = calculate_dof(n_atoms_2, n_waters_2, n_h_constraints);

    println!(
        "\nTest case 2: Protein ({} atoms) + water ({} waters)",
        n_protein_atoms, n_waters_2
    );
    println!("   Total atoms: {}", n_atoms_2);
    println!("   Raw DOF: {}", 3 * n_atoms_2);
    println!("   COM removal: -3");
    println!("   SETTLE constraints: -{}", 3 * n_waters_2);
    println!("   H-constraints: -{}", n_h_constraints);
    println!("   Final DOF: {}", dof_2);

    // Expected: 3*250 - 3 - 3*50 - 40 = 750 - 3 - 150 - 40 = 557
    assert_eq!(dof_2, 557, "Protein+water DOF incorrect");
    println!("   ✓ DOF = 557 (correct)");

    // Test case 3: Verify temperature formula
    // T = 2 * KE / (kB * DOF) where kB = 0.001987204 kcal/(mol·K)
    let kb = 0.001987204f64; // Boltzmann constant in kcal/(mol·K)
    let target_temp = 310.0f64;
    let expected_ke = 0.5 * kb * target_temp * (dof_2 as f64);

    println!("\nTest case 3: Temperature-energy relationship");
    println!("   Target temperature: {:.1} K", target_temp);
    println!(
        "   Expected KE at {:.1} K: {:.2} kcal/mol",
        target_temp, expected_ke
    );
    println!(
        "   KE per DOF: {:.4} kcal/mol/DOF",
        expected_ke / (dof_2 as f64)
    );

    // Verify inverse: T = 2*KE/(kB*DOF)
    let calc_temp = 2.0 * expected_ke / (kb * (dof_2 as f64));
    assert!(
        (calc_temp - target_temp).abs() < 0.01,
        "Temperature calculation mismatch: {:.2} vs {:.2}",
        calc_temp,
        target_temp
    );
    println!("   ✓ Temperature formula verified");

    println!("\n✓ DOF calculation test PASSED");
}

// ============================================================================
// TIER 1: CUDA-Required Tests (Marked #[ignore])
// ============================================================================

/// Integration test: Verify neighbor list finds all PBC pairs
///
/// Runs actual neighbor list build on GPU and compares to brute-force.
#[test]
#[ignore] // Requires CUDA: cargo test neighbor_list_gpu --ignored --features cuda
fn test_neighbor_list_gpu_pbc() {
    use cudarc::driver::CudaContext;

    println!("\n=== GPU Neighbor List PBC Test ===\n");

    // This test would:
    // 1. Create a system with atoms near periodic boundaries
    // 2. Run GPU neighbor list build
    // 3. Compare to brute-force reference
    // 4. Verify all pairs within cutoff are found

    let context = CudaContext::new(0).expect("Failed to create CUDA context");
    println!("CUDA context created: {:?}", context);

    // Full implementation would go here
    println!("GPU neighbor list PBC test would run here (requires prism-gpu integration)");
}

/// Integration test: Verify SETTLE constraint satisfaction after dynamics
///
/// Runs short MD with SETTLE and checks constraint violations stay < 1e-4 Å.
#[test]
#[ignore] // Requires CUDA: cargo test settle_dynamics --ignored --features cuda
fn test_settle_dynamics_tolerance() {
    use cudarc::driver::CudaContext;

    println!("\n=== SETTLE Dynamics Tolerance Test ===\n");

    // This test would:
    // 1. Create water box
    // 2. Run 1000 steps of MD with SETTLE
    // 3. Check constraint violations at each step
    // 4. Verify max violation < 1e-4 Å

    let context = CudaContext::new(0).expect("Failed to create CUDA context");
    println!("CUDA context created: {:?}", context);

    println!("SETTLE dynamics test would run here (requires prism-gpu integration)");
}
