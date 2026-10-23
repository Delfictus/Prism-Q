//! RT Integration Tests [STAGE-1-INTEGRATION]
//!
//! Comprehensive integration tests validating the COMPLETE RT integration pipeline
//! from Stage 1 (PREP) configuration through RT target identification.
//!
//! ## Test Philosophy: GOLD STANDARD
//!
//! - NO shortcuts: Every integration point tested
//! - NO mocks: Real data structures, real algorithms
//! - NO skipped edge cases: Comprehensive coverage
//! - FULL validation: Not just "it compiles"
//!
//! ## Test Coverage
//!
//! 1. Config → Solvate → RT Targets → PreparedInput (full pipeline)
//! 2. Implicit mode (no waters, protein only)
//! 3. Explicit mode (waters added, full RT targets)
//! 4. Hybrid mode (starts implicit, metadata for explicit)
//! 5. Error paths (invalid configs, edge cases)
//! 6. Performance validation (prep time <5s for explicit)
//! 7. Data integrity (water metadata, RT target counts)

use prism_nhs::{
    config::SolventMode,
    input::{NhsPreparedInput, PrismPrepTopology},
    rt_targets::{identify_rt_targets, RtTargets},
    solvate::solvate_protein,
};

/// Create a realistic test protein topology
///
/// This is NOT a minimal toy - this is a realistic small protein:
/// - 50 atoms (mix of backbone + sidechains)
/// - 10 residues (ALA, PHE, TYR, GLY, etc.)
/// - 2 chains (A, B)
/// - Proper geometry (realistic distances)
fn create_realistic_protein() -> PrismPrepTopology {
    // 50 atoms: realistic small protein domain
    let n_atoms = 50;

    // Generate realistic atomic coordinates in a compact globular structure
    // Center at origin, ~15Å radius
    let mut positions = Vec::with_capacity(n_atoms * 3);
    let mut elements = Vec::with_capacity(n_atoms);
    let mut atom_names = Vec::with_capacity(n_atoms);
    let mut residue_ids = Vec::with_capacity(n_atoms);
    let mut chain_ids = Vec::with_capacity(n_atoms);
    let mut charges = Vec::with_capacity(n_atoms);
    let mut masses = Vec::with_capacity(n_atoms);

    // 10 residues: ALA, PHE, TYR, GLY, ALA, TRP, HIS, ALA, GLY, ALA
    let residues = vec![
        (
            "ALA",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("C", "C", 12.0, 0.5),
                ("O", "O", 16.0, -0.5),
                ("CB", "C", 12.0, 0.0),
            ],
        ),
        (
            "PHE",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("CG", "C", 12.0, 0.0),
                ("CD1", "C", 12.0, 0.0),
                ("CE1", "C", 12.0, 0.0),
            ],
        ),
        (
            "TYR",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("CG", "C", 12.0, 0.0),
                ("CD1", "C", 12.0, 0.0),
                ("OH", "O", 16.0, -0.5),
            ],
        ),
        (
            "GLY",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("C", "C", 12.0, 0.5),
                ("O", "O", 16.0, -0.5),
            ],
        ),
        (
            "ALA",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("C", "C", 12.0, 0.5),
                ("O", "O", 16.0, -0.5),
                ("CB", "C", 12.0, 0.0),
            ],
        ),
        (
            "TRP",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("CG", "C", 12.0, 0.0),
                ("CD1", "C", 12.0, 0.0),
                ("NE1", "N", 14.0, -0.3),
            ],
        ),
        (
            "HIS",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("CG", "C", 12.0, 0.0),
                ("ND1", "N", 14.0, -0.3),
                ("CE1", "C", 12.0, 0.0),
            ],
        ),
        (
            "ALA",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("C", "C", 12.0, 0.5),
                ("O", "O", 16.0, -0.5),
                ("CB", "C", 12.0, 0.0),
            ],
        ),
        (
            "GLY",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("C", "C", 12.0, 0.5),
                ("O", "O", 16.0, -0.5),
            ],
        ),
        (
            "ALA",
            vec![
                ("N", "N", 14.0, -0.4),
                ("CA", "C", 12.0, 0.1),
                ("C", "C", 12.0, 0.5),
                ("O", "O", 16.0, -0.5),
                ("CB", "C", 12.0, 0.0),
            ],
        ),
    ];

    let mut atom_idx = 0;
    let mut residue_names = Vec::new();

    for (res_id, (res_name, atoms)) in residues.iter().enumerate() {
        for (atom_name, element, mass, charge) in atoms {
            // residue_names is ATOM-indexed (one entry per atom) per the PrismPrepTopology
            // contract (input.rs validate(): residue_names.len() must equal n_atoms). Pushing it
            // once per residue produced residue_names.len()==n_residues < n_atoms and panicked
            // classify_atoms() at input.rs:313 (index out of bounds).
            residue_names.push(res_name.to_string());
            // Generate realistic coordinates in a compact sphere
            // Use spherical coordinates for distribution
            let theta = (atom_idx as f32 * 2.4) % (2.0 * std::f32::consts::PI);
            let phi = (atom_idx as f32 * 1.6) % std::f32::consts::PI;
            let r = 5.0 + (atom_idx as f32 % 10.0); // 5-15Å radius

            let x = r * phi.sin() * theta.cos();
            let y = r * phi.sin() * theta.sin();
            let z = r * phi.cos();

            positions.extend_from_slice(&[x, y, z]);
            elements.push(element.to_string());
            atom_names.push(atom_name.to_string());
            residue_ids.push(res_id);
            chain_ids.push(if res_id < 5 { "A" } else { "B" }.to_string());
            charges.push(*charge);
            masses.push(*mass);

            atom_idx += 1;
        }
    }

    // CA indices (one per residue, typically second atom)
    let ca_indices = vec![1, 6, 11, 15, 20, 25, 30, 35, 39, 44];

    PrismPrepTopology {
        source_pdb: "test_protein.pdb".to_string(),
        n_atoms: atom_idx,
        n_residues: residues.len(),
        total_physical_residue_count: None,
        n_analysis_residues: None,
        n_chains: 2,
        positions,
        elements,
        atom_names,
        residue_names,
        residue_ids,
        atom_physical_residue_ids: None,
        atom_analysis_residue_ids: None,
        analysis_residue_contract: None,
        chain_ids,
        charges,
        masses,
        ca_indices,
        bonds: Vec::new(),
        angles: Vec::new(),
        dihedrals: Vec::new(),
        lj_params: Vec::new(),
        exclusions: Vec::new(),
        nonbonded_exceptions: Vec::new(),
        nonbonded_exception_contract: None,
        generated_prmtop_sha256: None,
        h_clusters: Vec::new(),
        hmr_applied: Some(false),
        recommended_timestep_fs: None,
        water_oxygens: Vec::new(),
        residues: Vec::new(),
        dimer_dyad: None,
        lambda_fragment_system: None,
        harmonic_parameter_contract: None,
        schema_kind: None,
        schema_version: None,
        periodic_cell: None,
        water_model_contract: None,
        molecule_identity_table: None,
        topology_provenance: None,
        npt_frozen_cell_contract: None,
    }
}

// ============================================================================
// INTEGRATION TEST 1: Full Pipeline - Implicit Mode
// ============================================================================

#[test]
fn test_full_pipeline_implicit_mode() {
    println!("\n=== INTEGRATION TEST: Full Pipeline - Implicit Mode ===\n");

    // Step 1: Create realistic protein
    let topology = create_realistic_protein();
    println!(
        "Created test protein: {} atoms, {} residues",
        topology.n_atoms, topology.n_residues
    );

    // Step 2: Configure implicit solvent mode
    let solvent_mode = SolventMode::Implicit;
    println!("Solvent mode: {:?}", solvent_mode);

    // Step 3: Prepare system (full pipeline)
    let prepared = NhsPreparedInput::from_topology(
        topology.clone(),
        0.5,  // grid_spacing
        10.0, // padding
        &solvent_mode,
    )
    .expect("Failed to prepare input");

    println!("\n--- Prepared System ---");
    println!("Total atoms: {}", prepared.total_atoms);
    println!("Solvent mode: {:?}", prepared.solvent_mode);
    println!("Water atoms: {:?}", prepared.water_atoms);
    println!("{}", prepared.rt_targets.summary());

    // VALIDATION: Implicit mode should have NO waters
    assert!(
        prepared.water_atoms.is_none(),
        "Implicit mode should have no waters"
    );
    assert_eq!(
        prepared.total_atoms, topology.n_atoms,
        "Total atoms should equal protein atoms"
    );

    // VALIDATION: RT targets should have protein atoms only (no waters)
    assert!(
        prepared.rt_targets.water_atoms.is_none(),
        "RT targets should have no water atoms"
    );
    assert!(
        !prepared.rt_targets.protein_atoms.is_empty(),
        "RT targets must have protein atoms"
    );

    // VALIDATION: Aromatic centers should be detected (PHE, TYR, TRP, HIS present)
    assert!(
        prepared.rt_targets.aromatic_centers.len() >= 4,
        "Should detect at least 4 aromatic residues (PHE, TYR, TRP, HIS), found {}",
        prepared.rt_targets.aromatic_centers.len()
    );

    // VALIDATION: Total target count should be protein + aromatics (no waters)
    let expected_total =
        prepared.rt_targets.protein_atoms.len() + prepared.rt_targets.aromatic_centers.len();
    assert_eq!(
        prepared.rt_targets.total_targets, expected_total,
        "Total target count mismatch"
    );

    println!("\n✅ PASS: Implicit mode pipeline validated");
}

// ============================================================================
// INTEGRATION TEST 2: Full Pipeline - Explicit Mode
// ============================================================================

#[test]
fn test_full_pipeline_explicit_mode() {
    println!("\n=== INTEGRATION TEST: Full Pipeline - Explicit Mode ===\n");

    // Step 1: Create realistic protein
    let topology = create_realistic_protein();
    println!(
        "Created test protein: {} atoms, {} residues",
        topology.n_atoms, topology.n_residues
    );

    // Step 2: Configure explicit solvent mode (10Å padding)
    let solvent_mode = SolventMode::Explicit {
        padding_angstroms: 10.0,
    };
    println!("Solvent mode: {:?}", solvent_mode);

    // Step 3: Prepare system (full pipeline with solvation)
    let start_time = std::time::Instant::now();
    let prepared = NhsPreparedInput::from_topology(
        topology.clone(),
        0.5,  // grid_spacing
        10.0, // padding
        &solvent_mode,
    )
    .expect("Failed to prepare input");
    let prep_time = start_time.elapsed();

    println!("\n--- Prepared System ---");
    println!("Total atoms: {}", prepared.total_atoms);
    println!("Solvent mode: {:?}", prepared.solvent_mode);
    println!("Preparation time: {:.3}s", prep_time.as_secs_f64());

    if let Some(ref waters) = prepared.water_atoms {
        println!("Water molecules: {}", waters.len());
        println!("Protein atoms: {}", prepared.total_atoms - waters.len());
    }
    println!("{}", prepared.rt_targets.summary());

    // VALIDATION: Explicit mode MUST have waters
    assert!(
        prepared.water_atoms.is_some(),
        "Explicit mode must have waters"
    );
    let n_waters = prepared.water_atoms.as_ref().unwrap().len();
    assert!(n_waters > 0, "Must add at least some waters");

    // VALIDATION: Water count should be reasonable for 10Å padding
    // Protein radius ~15Å, padded box ~35Å diameter
    // Volume ~42,000 Å³ → ~1,400 waters expected
    // Allow 500-3000 range (generous for test)
    assert!(n_waters >= 500, "Too few waters added: {}", n_waters);
    assert!(n_waters <= 3000, "Too many waters added: {}", n_waters);
    println!(
        "Water count validation: {} waters (expected 500-3000) ✓",
        n_waters
    );

    // VALIDATION: Total atoms = protein + waters
    assert_eq!(
        prepared.total_atoms,
        topology.n_atoms + n_waters,
        "Total atoms mismatch: {} != {} + {}",
        prepared.total_atoms,
        topology.n_atoms,
        n_waters
    );

    // VALIDATION: RT targets should include water atoms
    assert!(
        prepared.rt_targets.water_atoms.is_some(),
        "RT targets must include water atoms"
    );
    assert_eq!(
        prepared.rt_targets.water_atoms.as_ref().unwrap().len(),
        n_waters,
        "RT water target count mismatch"
    );

    // VALIDATION: Aromatic centers still detected
    assert!(
        prepared.rt_targets.aromatic_centers.len() >= 4,
        "Should still detect aromatic residues in explicit mode"
    );

    // VALIDATION: Performance target (<5s for solvation)
    assert!(
        prep_time.as_secs_f64() < 5.0,
        "Preparation time {} exceeds 5s target",
        prep_time.as_secs_f64()
    );
    println!(
        "Performance validation: {:.3}s < 5.0s ✓",
        prep_time.as_secs_f64()
    );

    // VALIDATION: Water metadata in topology
    let protein_start = topology.n_atoms;
    for water_idx in prepared.water_atoms.as_ref().unwrap().iter().take(5) {
        assert!(
            *water_idx >= protein_start,
            "Water index {} should be >= protein count {}",
            water_idx,
            protein_start
        );
    }
    println!(
        "Water metadata validation: indices start at {} ✓",
        protein_start
    );

    println!("\n✅ PASS: Explicit mode pipeline validated");
}

// ============================================================================
// INTEGRATION TEST 3: Full Pipeline - Hybrid Mode
// ============================================================================

#[test]
fn test_full_pipeline_hybrid_mode() {
    println!("\n=== INTEGRATION TEST: Full Pipeline - Hybrid Mode ===\n");

    let topology = create_realistic_protein();
    println!(
        "Created test protein: {} atoms, {} residues",
        topology.n_atoms, topology.n_residues
    );

    // Hybrid mode: starts implicit, switches to explicit during simulation
    let solvent_mode = SolventMode::Hybrid {
        exploration_steps: 1000,
        characterization_steps: 5000,
        switch_threshold: 0.6,
    };
    println!("Solvent mode: {:?}", solvent_mode);

    // For Stage 1 (PREP), hybrid starts as implicit
    let prepared = NhsPreparedInput::from_topology(topology.clone(), 0.5, 10.0, &solvent_mode)
        .expect("Failed to prepare input");

    println!("\n--- Prepared System (Hybrid Start) ---");
    println!("Total atoms: {}", prepared.total_atoms);
    println!("Solvent mode: {:?}", prepared.solvent_mode);
    println!("{}", prepared.rt_targets.summary());

    // VALIDATION: Hybrid starts implicit (no waters initially)
    // Waters will be added dynamically during Stage 2b when pockets detected
    assert!(
        prepared.water_atoms.is_none() || prepared.water_atoms.as_ref().unwrap().is_empty(),
        "Hybrid mode should start without explicit waters"
    );

    // VALIDATION: Configuration preserved for later explicit switch
    if let SolventMode::Hybrid {
        exploration_steps,
        characterization_steps,
        switch_threshold,
    } = prepared.solvent_mode
    {
        assert_eq!(exploration_steps, 1000);
        assert_eq!(characterization_steps, 5000);
        assert!((switch_threshold - 0.6).abs() < 0.001);
    } else {
        panic!("Solvent mode should be Hybrid");
    }

    println!("\n✅ PASS: Hybrid mode pipeline validated");
}

// ============================================================================
// INTEGRATION TEST 4: Water Density Validation
// ============================================================================

#[test]
fn test_water_density_validation() {
    println!("\n=== INTEGRATION TEST: Water Density Validation ===\n");

    let topology = create_realistic_protein();
    let solvent_mode = SolventMode::Explicit {
        padding_angstroms: 15.0,
    };

    let prepared = NhsPreparedInput::from_topology(topology.clone(), 0.5, 15.0, &solvent_mode)
        .expect("Failed to prepare input");

    let n_waters = prepared.water_atoms.as_ref().unwrap().len();

    // Calculate box volume
    // Protein radius ~15Å, padded box ~45Å diameter
    // Approximate box volume (sphere): 4/3 * π * r³
    let padded_radius: f32 = 15.0 + 15.0; // protein radius + padding
    let box_volume = (4.0 / 3.0) * std::f32::consts::PI * padded_radius.powi(3);

    // Expected water density: 0.0334 molecules/Å³
    let expected_waters = (box_volume * 0.0334) as usize;

    // Calculate actual density
    let actual_density = n_waters as f32 / box_volume;

    println!("Box volume: {:.0} Å³", box_volume);
    println!("Expected waters: {} (at 0.0334 mol/Å³)", expected_waters);
    println!("Actual waters: {}", n_waters);
    println!("Actual density: {:.4} mol/Å³", actual_density);

    // VALIDATION: Density should be close to TIP3P target
    // Allow ±50% tolerance: overlap removal reduces count (0.5-0.9x typical),
    // but spherical packing can increase local density (1.0-1.5x)
    // Real systems vary based on protein topology and packing geometry
    let density_ratio = actual_density / 0.0334;
    assert!(
        density_ratio >= 0.4 && density_ratio <= 1.6,
        "Water density ratio {:.2} outside acceptable range [0.4, 1.6]",
        density_ratio
    );

    println!("Density ratio: {:.2} (target: 1.0) ✓", density_ratio);
    println!("\n✅ PASS: Water density within acceptable range");
}

// ============================================================================
// INTEGRATION TEST 5: Aromatic Center Accuracy
// ============================================================================

#[test]
fn test_aromatic_center_accuracy() {
    println!("\n=== INTEGRATION TEST: Aromatic Center Accuracy ===\n");

    let topology = create_realistic_protein();
    let solvent_mode = SolventMode::Implicit;

    let prepared = NhsPreparedInput::from_topology(topology.clone(), 0.5, 10.0, &solvent_mode)
        .expect("Failed to prepare input");

    let aromatics = &prepared.rt_targets.aromatic_centers;
    println!("Detected {} aromatic centers", aromatics.len());

    // VALIDATION: Should detect PHE, TYR, TRP, HIS (4 aromatic residues)
    assert_eq!(
        aromatics.len(),
        4,
        "Should detect exactly 4 aromatic residues"
    );

    // VALIDATION: Each center should be within protein bounds
    // Protein is in sphere of radius ~15Å centered at origin
    for (i, center) in aromatics.iter().enumerate() {
        let distance_from_origin =
            (center[0].powi(2) + center[1].powi(2) + center[2].powi(2)).sqrt();
        println!(
            "Aromatic {}: [{:.2}, {:.2}, {:.2}] distance from origin: {:.2}Å",
            i, center[0], center[1], center[2], distance_from_origin
        );

        assert!(
            distance_from_origin < 20.0,
            "Aromatic center {} at {:.2}Å is outside protein bounds",
            i,
            distance_from_origin
        );
    }

    println!("\n✅ PASS: Aromatic centers within protein bounds");
}

// ============================================================================
// INTEGRATION TEST 6: RT Target Count Validation
// ============================================================================

#[test]
fn test_rt_target_count_validation() {
    println!("\n=== INTEGRATION TEST: RT Target Count Validation ===\n");

    let topology = create_realistic_protein();

    // Test both implicit and explicit modes
    for (mode_name, solvent_mode) in &[
        ("Implicit", SolventMode::Implicit),
        (
            "Explicit",
            SolventMode::Explicit {
                padding_angstroms: 10.0,
            },
        ),
    ] {
        println!("\n--- Testing {} Mode ---", mode_name);

        let prepared = NhsPreparedInput::from_topology(topology.clone(), 0.5, 10.0, solvent_mode)
            .expect("Failed to prepare input");

        let targets = &prepared.rt_targets;

        // VALIDATION: compute_total() matches actual counts
        let manual_total = targets.protein_atoms.len()
            + targets.water_atoms.as_ref().map_or(0, |w| w.len())
            + targets.aromatic_centers.len();

        assert_eq!(
            targets.total_targets, manual_total,
            "{} mode: total_targets mismatch",
            mode_name
        );

        println!("Protein atoms: {}", targets.protein_atoms.len());
        println!(
            "Water atoms: {}",
            targets.water_atoms.as_ref().map_or(0, |w| w.len())
        );
        println!("Aromatic centers: {}", targets.aromatic_centers.len());
        println!("Total: {} ✓", targets.total_targets);
    }

    println!("\n✅ PASS: RT target counts validated for all modes");
}

// ============================================================================
// INTEGRATION TEST 7: Error Handling Validation
// ============================================================================

#[test]
fn test_error_handling_validation() {
    println!("\n=== INTEGRATION TEST: Error Handling Validation ===\n");

    let topology = create_realistic_protein();

    // Test 1: Invalid padding (negative)
    println!("Testing invalid padding (negative)...");
    let result = NhsPreparedInput::from_topology(
        topology.clone(),
        0.5,
        10.0,
        &SolventMode::Explicit {
            padding_angstroms: -5.0,
        },
    );
    assert!(result.is_err(), "Should reject negative padding");
    println!("✓ Correctly rejected negative padding");

    // Test 2: Invalid padding (zero)
    println!("Testing invalid padding (zero)...");
    let result = NhsPreparedInput::from_topology(
        topology.clone(),
        0.5,
        10.0,
        &SolventMode::Explicit {
            padding_angstroms: 0.0,
        },
    );
    assert!(result.is_err(), "Should reject zero padding");
    println!("✓ Correctly rejected zero padding");

    // Test 3: Invalid hybrid config (negative steps)
    println!("Testing invalid hybrid config (negative steps)...");
    let invalid_hybrid = SolventMode::Hybrid {
        exploration_steps: -100,
        characterization_steps: 1000,
        switch_threshold: 0.6,
    };
    assert!(
        invalid_hybrid.validate().is_err(),
        "Should reject negative steps"
    );
    println!("✓ Correctly rejected negative steps");

    // Test 4: Invalid threshold (> 1.0)
    println!("Testing invalid threshold (> 1.0)...");
    let invalid_threshold = SolventMode::Hybrid {
        exploration_steps: 1000,
        characterization_steps: 1000,
        switch_threshold: 1.5,
    };
    assert!(
        invalid_threshold.validate().is_err(),
        "Should reject threshold > 1.0"
    );
    println!("✓ Correctly rejected invalid threshold");

    println!("\n✅ PASS: Error handling validated");
}

// ============================================================================
// INTEGRATION TEST 8: Performance Regression Check
// ============================================================================

#[test]
fn test_performance_regression() {
    println!("\n=== INTEGRATION TEST: Performance Regression Check ===\n");

    let topology = create_realistic_protein();

    // Baseline: Implicit mode (no solvation overhead)
    let start_implicit = std::time::Instant::now();
    let _prepared_implicit =
        NhsPreparedInput::from_topology(topology.clone(), 0.5, 10.0, &SolventMode::Implicit)
            .expect("Failed to prepare implicit");
    let time_implicit = start_implicit.elapsed();
    println!(
        "Implicit mode prep time: {:.3}s",
        time_implicit.as_secs_f64()
    );

    // With solvation: Explicit mode
    let start_explicit = std::time::Instant::now();
    let _prepared_explicit = NhsPreparedInput::from_topology(
        topology.clone(),
        0.5,
        10.0,
        &SolventMode::Explicit {
            padding_angstroms: 10.0,
        },
    )
    .expect("Failed to prepare explicit");
    let time_explicit = start_explicit.elapsed();
    println!(
        "Explicit mode prep time: {:.3}s",
        time_explicit.as_secs_f64()
    );

    // VALIDATION: Explicit overhead should be <5s absolute
    assert!(
        time_explicit.as_secs_f64() < 5.0,
        "Explicit mode prep time {:.3}s exceeds 5s target",
        time_explicit.as_secs_f64()
    );

    // Calculate overhead
    let overhead = time_explicit.as_secs_f64() - time_implicit.as_secs_f64();
    println!("Solvation overhead: {:.3}s", overhead);

    // VALIDATION: Overhead should be reasonable (<5s)
    assert!(
        overhead < 5.0,
        "Solvation overhead {:.3}s exceeds 5s",
        overhead
    );

    println!("\n✅ PASS: Performance targets met");
}
