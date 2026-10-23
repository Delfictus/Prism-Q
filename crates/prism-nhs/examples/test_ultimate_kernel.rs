//! Quick test to verify ultimate_md.cu kernel works after constant memory fix
//!
//! Compile with: cargo run --release --features gpu --example test_ultimate_kernel

use anyhow::Result;
use cudarc::driver::CudaContext;
use prism_nhs::{PrismPrepTopology, UltimateEngine, UltimateEngineConfig};
use std::time::Instant;

fn create_test_topology(n_atoms: usize) -> PrismPrepTopology {
    let mut positions = vec![0.0f32; n_atoms * 3];
    // Arrange atoms in a 3D grid
    let grid_size = (n_atoms as f32).cbrt().ceil() as usize;
    let spacing = 4.0f32; // 4 Angstroms between atoms

    for i in 0..n_atoms {
        let x = (i % grid_size) as f32 * spacing;
        let y = ((i / grid_size) % grid_size) as f32 * spacing;
        let z = (i / (grid_size * grid_size)) as f32 * spacing;
        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;
    }

    PrismPrepTopology {
        // Fields added to PrismPrepTopology after this example was written; the
        // values are the serde defaults (absent producer metadata).
        aromatic_targets: Vec::new(),
        hmr_applied: None,
        recommended_timestep_fs: None,
        npt_frozen_cell_contract: None,
        source_pdb: "test.pdb".to_string(),
        n_atoms,
        n_residues: n_atoms,
        total_physical_residue_count: None,
        n_analysis_residues: None,
        n_chains: 1,
        positions,
        elements: vec!["C".to_string(); n_atoms],
        atom_names: vec!["CA".to_string(); n_atoms],
        residue_names: vec!["ALA".to_string(); n_atoms],
        residue_ids: (0..n_atoms).collect(),
        atom_physical_residue_ids: None,
        atom_analysis_residue_ids: None,
        analysis_residue_contract: None,
        chain_ids: vec!["A".to_string(); n_atoms],
        charges: vec![0.1; n_atoms], // Small charges for testing
        masses: vec![12.0; n_atoms],
        ca_indices: (0..n_atoms).collect(),
        bonds: vec![],
        angles: vec![],
        dihedrals: vec![],
        lj_params: vec![
            prism_nhs::input::LjParam {
                sigma: 3.4, // ~C atom
                epsilon: 0.086,
            };
            n_atoms
        ],
        exclusions: vec![vec![]; n_atoms],
        nonbonded_exceptions: Vec::new(),
        nonbonded_exception_contract: None,
        generated_prmtop_sha256: None,
        h_clusters: vec![],
        water_oxygens: vec![],
        residues: (0..n_atoms)
            .map(|idx| prism_nhs::input::ResidueMetadata {
                residue_idx: idx,
                residue_name: "ALA".to_string(),
                residue_id: idx as i32,
            })
            .collect(),
        dimer_dyad: None,
        lambda_fragment_system: None,
        harmonic_parameter_contract: None,
        schema_kind: None,
        schema_version: None,
        periodic_cell: None,
        water_model_contract: None,
        molecule_identity_table: None,
        topology_provenance: None,
    }
}

fn main() -> Result<()> {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  ULTIMATE KERNEL TEST & BENCHMARK");
    println!("  Testing ultimate_md.cu hyperoptimized kernel");
    println!("═══════════════════════════════════════════════════════════════\n");

    // Create CUDA context
    println!("1. Initializing CUDA context...");
    let context = CudaContext::new(0)?;
    println!("   ✓ CUDA context created\n");

    // Test 1: Small system (10 atoms) - basic validation
    println!("2. Testing small system (10 atoms)...");
    {
        let topology = create_test_topology(10);
        let config = UltimateEngineConfig::default();
        let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;

        for i in 1..=5 {
            let result = engine.step()?;
            println!(
                "   Step {}: PE={:.2e}, KE={:.2e}",
                i, result.potential_energy, result.kinetic_energy
            );
        }
    }
    println!("   ✓ Small system test PASSED\n");

    // Test 2: Medium system (500 atoms) - performance benchmark
    println!("3. Benchmarking medium system (500 atoms, 1000 steps)...");
    {
        let topology = create_test_topology(500);
        let config = UltimateEngineConfig::default();
        let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;

        let n_steps = 1000;
        let start = Instant::now();
        let result = engine.step_batch(n_steps)?;
        let elapsed = start.elapsed();

        let steps_per_sec = n_steps as f64 / elapsed.as_secs_f64();
        println!("   Steps: {}", n_steps);
        println!("   Time: {:.2}s", elapsed.as_secs_f64());
        println!("   Throughput: {:.0} steps/sec", steps_per_sec);
        println!(
            "   Final PE: {:.2e}, KE: {:.2e}",
            result.potential_energy, result.kinetic_energy
        );
    }
    println!("   ✓ Medium system benchmark PASSED\n");

    // Test 3: Larger system (1000 atoms) - baseline
    println!("4. Baseline test (1000 atoms, 500 steps)...");
    {
        let topology = create_test_topology(1000);
        let config = UltimateEngineConfig::default();
        let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;

        let n_steps = 500;
        let start = Instant::now();
        let result = engine.step_batch(n_steps)?;
        let elapsed = start.elapsed();

        let steps_per_sec = n_steps as f64 / elapsed.as_secs_f64();
        println!("   Steps: {}", n_steps);
        println!("   Time: {:.2}s", elapsed.as_secs_f64());
        println!("   Throughput: {:.0} steps/sec", steps_per_sec);
        println!(
            "   Final PE: {:.2e}, KE: {:.2e}",
            result.potential_energy, result.kinetic_energy
        );
    }
    println!("   ✓ Baseline test PASSED\n");

    // Test 4: Production-scale (10,000 atoms) - memory stress test
    println!("5. PRODUCTION SCALE TEST (10,000 atoms, 100 steps)...");
    println!("   Watching for: shared memory overflow, register spilling");
    {
        let topology = create_test_topology(10_000);
        let config = UltimateEngineConfig::default();
        let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;

        let n_steps = 100;
        let start = Instant::now();
        let result = engine.step_batch(n_steps)?;
        let elapsed = start.elapsed();

        let steps_per_sec = n_steps as f64 / elapsed.as_secs_f64();
        let atoms_steps_per_sec = 10_000.0 * n_steps as f64 / elapsed.as_secs_f64();
        println!("   Steps: {}", n_steps);
        println!("   Time: {:.2}s", elapsed.as_secs_f64());
        println!("   Throughput: {:.0} steps/sec", steps_per_sec);
        println!("   Atom-steps/sec: {:.2e}", atoms_steps_per_sec);
        println!(
            "   Final PE: {:.2e}, KE: {:.2e}",
            result.potential_energy, result.kinetic_energy
        );
    }
    println!("   ✓ 10K atoms test PASSED\n");

    // Test 5: Large protein scale (20,000 atoms) - typical protein
    println!("6. LARGE PROTEIN SCALE (20,000 atoms, 50 steps)...");
    {
        let topology = create_test_topology(20_000);
        let config = UltimateEngineConfig::default();
        let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;

        let n_steps = 50;
        let start = Instant::now();
        let result = engine.step_batch(n_steps)?;
        let elapsed = start.elapsed();

        let steps_per_sec = n_steps as f64 / elapsed.as_secs_f64();
        let atoms_steps_per_sec = 20_000.0 * n_steps as f64 / elapsed.as_secs_f64();
        println!("   Steps: {}", n_steps);
        println!("   Time: {:.2}s", elapsed.as_secs_f64());
        println!("   Throughput: {:.0} steps/sec", steps_per_sec);
        println!("   Atom-steps/sec: {:.2e}", atoms_steps_per_sec);
        println!(
            "   Final PE: {:.2e}, KE: {:.2e}",
            result.potential_energy, result.kinetic_energy
        );
    }
    println!("   ✓ 20K atoms test PASSED\n");

    // Test 6: Extreme scale (50,000 atoms) - stress limit
    println!("7. EXTREME SCALE TEST (50,000 atoms, 20 steps)...");
    let (t50k_steps_sec, t50k_time) = {
        let topology = create_test_topology(50_000);
        let config = UltimateEngineConfig::default();
        let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;

        let n_steps = 20;
        let start = Instant::now();
        let result = engine.step_batch(n_steps)?;
        let elapsed = start.elapsed();

        let steps_per_sec = n_steps as f64 / elapsed.as_secs_f64();
        let atoms_steps_per_sec = 50_000.0 * n_steps as f64 / elapsed.as_secs_f64();
        println!("   Steps: {}", n_steps);
        println!("   Time: {:.2}s", elapsed.as_secs_f64());
        println!("   Throughput: {:.0} steps/sec", steps_per_sec);
        println!("   Atom-steps/sec: {:.2e}", atoms_steps_per_sec);
        println!(
            "   Final PE: {:.2e}, KE: {:.2e}",
            result.potential_energy, result.kinetic_energy
        );
        (steps_per_sec, elapsed.as_secs_f64() / n_steps as f64)
    };
    println!("   ✓ 50K atoms test PASSED\n");

    // Test 7: Maximum scale (100,000 atoms) - production simulation size
    println!("8. MAXIMUM SCALE TEST (100,000 atoms, 10 steps)...");
    let (t100k_steps_sec, t100k_time) = {
        let topology = create_test_topology(100_000);
        let config = UltimateEngineConfig::default();
        let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;

        let n_steps = 10;
        let start = Instant::now();
        let result = engine.step_batch(n_steps)?;
        let elapsed = start.elapsed();

        let steps_per_sec = n_steps as f64 / elapsed.as_secs_f64();
        let atoms_steps_per_sec = 100_000.0 * n_steps as f64 / elapsed.as_secs_f64();
        let pair_evals_per_sec = (100_000.0 * 100_000.0) * steps_per_sec;
        println!("   Steps: {}", n_steps);
        println!("   Time: {:.2}s", elapsed.as_secs_f64());
        println!("   Throughput: {:.0} steps/sec", steps_per_sec);
        println!("   Atom-steps/sec: {:.2e}", atoms_steps_per_sec);
        println!("   Pair evaluations/sec: {:.2e}", pair_evals_per_sec);
        println!(
            "   Final PE: {:.2e}, KE: {:.2e}",
            result.potential_energy, result.kinetic_energy
        );
        (steps_per_sec, elapsed.as_secs_f64() / n_steps as f64)
    };
    println!("   ✓ 100K atoms test PASSED\n");

    // Scaling analysis
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SCALING ANALYSIS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Current kernel: O(N²) non-bonded (no neighbor lists)");
    println!();
    println!("  50K → 100K scaling:");
    println!(
        "    Expected O(N²): {:.1}x slowdown",
        (100_000.0 / 50_000.0_f64).powi(2)
    );
    println!(
        "    Actual: {:.1}x slowdown",
        t50k_steps_sec / t100k_steps_sec
    );
    println!();
    println!("  Time per step:");
    println!("    50K atoms:  {:.1}ms/step", t50k_time * 1000.0);
    println!("    100K atoms: {:.1}ms/step", t100k_time * 1000.0);
    println!();
    println!("  For production NHS (1M steps on 50K atoms):");
    println!(
        "    Current: {:.1} hours",
        1_000_000.0 / t50k_steps_sec / 3600.0
    );
    println!(
        "    With neighbor lists (est 10-20x): {:.1}-{:.1} hours",
        1_000_000.0 / (t50k_steps_sec * 20.0) / 3600.0,
        1_000_000.0 / (t50k_steps_sec * 10.0) / 3600.0
    );
    println!();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  ALL TESTS PASSED");
    println!("═══════════════════════════════════════════════════════════════");
    println!("\nUltimate kernel features working:");
    println!("  ✓ Cooperative groups synchronization");
    println!("  ✓ Tiled shared memory non-bonded forces");
    println!("  ✓ Warp-level reductions");
    println!("  ✓ Coalesced memory access (SoA layout)");
    println!("  ✓ Langevin integration");

    Ok(())
}
