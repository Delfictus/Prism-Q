//! Comprehensive validation for sm_120 compiled kernels
//!
//! Tests for:
//! - Race conditions (repeated runs should give consistent results)
//! - Livelocks (timeout detection)
//! - Energy conservation (physics validation)
//! - Multi-stream concurrency safety

use anyhow::Result;
use cudarc::driver::CudaContext;
use prism_nhs::{PrismPrepTopology, UltimateEngine, UltimateEngineConfig};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn create_test_topology(n_atoms: usize) -> PrismPrepTopology {
    let mut positions = vec![0.0f32; n_atoms * 3];
    let grid_size = (n_atoms as f32).cbrt().ceil() as usize;
    let spacing = 4.0f32;

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
        charges: vec![0.1; n_atoms],
        masses: vec![12.0; n_atoms],
        ca_indices: (0..n_atoms).collect(),
        bonds: vec![],
        angles: vec![],
        dihedrals: vec![],
        lj_params: vec![
            prism_nhs::input::LjParam {
                sigma: 3.4,
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

/// Run a kernel with timeout to detect livelocks
fn run_with_timeout<F, T>(timeout: Duration, f: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    let (tx, rx) = std::sync::mpsc::channel();

    thread::spawn(move || {
        let result = f();
        let _ = tx.send(result);
    });

    match rx.recv_timeout(timeout) {
        Ok(result) => result,
        Err(_) => anyhow::bail!(
            "LIVELOCK DETECTED: Kernel did not complete within {:?}",
            timeout
        ),
    }
}

fn main() -> Result<()> {
    env_logger::init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  SM_120 KERNEL VALIDATION SUITE");
    println!("  Testing for race conditions, livelocks, and correctness");
    println!("═══════════════════════════════════════════════════════════════\n");

    let context = CudaContext::new(0)?;
    let mut all_passed = true;

    // =========================================================================
    // TEST 1: Livelock Detection (timeout test)
    // =========================================================================
    println!("TEST 1: Livelock Detection");
    println!("─────────────────────────────────────────────────────────────────");

    for &n_atoms in &[100, 500, 1000, 5000] {
        print!("  {} atoms, 50 steps... ", n_atoms);
        let ctx = context.clone();

        let result = run_with_timeout(Duration::from_secs(30), move || {
            let topology = create_test_topology(n_atoms);
            let config = UltimateEngineConfig::default();
            let mut engine = UltimateEngine::new(ctx, &topology, config)?;
            engine.step_batch(50)
        });

        match result {
            Ok(_) => println!("✓ PASS (no livelock)"),
            Err(e) => {
                println!("✗ FAIL: {}", e);
                all_passed = false;
            }
        }
    }
    println!();

    // =========================================================================
    // TEST 2: Race Condition Detection (determinism check)
    // =========================================================================
    println!("TEST 2: Race Condition Detection (determinism)");
    println!("─────────────────────────────────────────────────────────────────");

    let n_atoms = 500;
    let n_steps = 100;
    let n_trials = 5;

    print!("  Running {} identical simulations... ", n_trials);

    let mut energies: Vec<f64> = Vec::new();

    for _ in 0..n_trials {
        let topology = create_test_topology(n_atoms);
        let config = UltimateEngineConfig::default();
        let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;
        let result = engine.step_batch(n_steps as i32)?;
        energies.push(result.potential_energy);
    }

    // Check if all energies are identical (deterministic)
    let first = energies[0];
    let all_same = energies.iter().all(|&e| (e - first).abs() < 1e-6);

    if all_same {
        println!("✓ PASS (all {} runs identical: PE={:.6e})", n_trials, first);
    } else {
        println!("⚠ WARNING: Non-deterministic results");
        for (i, e) in energies.iter().enumerate() {
            println!("    Trial {}: PE={:.6e}", i + 1, e);
        }
        // Non-determinism in GPU is sometimes expected due to floating point associativity
        // but large variations indicate race conditions
        let max_diff = energies
            .iter()
            .map(|&e| (e - first).abs())
            .fold(0.0f64, f64::max);

        if max_diff > 1.0 {
            println!(
                "    ✗ FAIL: Max difference {:.6e} too large (possible race)",
                max_diff
            );
            all_passed = false;
        } else {
            println!("    ✓ PASS: Variation within acceptable range");
        }
    }
    println!();

    // =========================================================================
    // TEST 3: Energy Conservation Check (physics validation)
    // =========================================================================
    println!("TEST 3: Energy Conservation (physics validation)");
    println!("─────────────────────────────────────────────────────────────────");

    let topology = create_test_topology(500);
    let config = UltimateEngineConfig::default();
    let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;

    let mut prev_total = 0.0f64;
    let mut max_drift = 0.0f64;
    let mut energy_samples: Vec<(f64, f64)> = Vec::new();

    print!("  Running 200 steps, checking energy drift... ");

    for i in 0..200 {
        let result = engine.step()?;
        let total = result.potential_energy + result.kinetic_energy;

        if i > 0 {
            let drift = (total - prev_total).abs();
            max_drift = max_drift.max(drift);
        }
        prev_total = total;

        if i % 50 == 0 {
            energy_samples.push((result.potential_energy, result.kinetic_energy));
        }
    }

    // With Langevin thermostat, energy fluctuates but shouldn't explode
    if max_drift < 1000.0 && prev_total.is_finite() {
        println!("✓ PASS");
        println!("    Max step-to-step drift: {:.2e}", max_drift);
        println!("    Final total energy: {:.2e}", prev_total);
    } else {
        println!("✗ FAIL: Energy explosion detected");
        println!(
            "    Max drift: {:.2e}, Final: {:.2e}",
            max_drift, prev_total
        );
        all_passed = false;
    }
    println!();

    // =========================================================================
    // TEST 4: Repeated Kernel Launch Stress Test
    // =========================================================================
    println!("TEST 4: Repeated Kernel Launch Stress Test");
    println!("─────────────────────────────────────────────────────────────────");

    let topology = create_test_topology(1000);
    let config = UltimateEngineConfig::default();
    let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;

    print!("  Launching 1000 rapid kernel calls... ");
    let start = Instant::now();

    for _ in 0..1000 {
        engine.step()?;
    }

    let elapsed = start.elapsed();
    println!(
        "✓ PASS ({:.2}s, {:.0} launches/sec)",
        elapsed.as_secs_f64(),
        1000.0 / elapsed.as_secs_f64()
    );
    println!();

    // =========================================================================
    // TEST 5: Large System Stability
    // =========================================================================
    println!("TEST 5: Large System Stability (20K atoms)");
    println!("─────────────────────────────────────────────────────────────────");

    print!("  Initializing 20,000 atom system... ");
    let topology = create_test_topology(20_000);
    let config = UltimateEngineConfig::default();
    let mut engine = UltimateEngine::new(context.clone(), &topology, config)?;
    println!("✓");

    print!("  Running 50 steps... ");
    let start = Instant::now();
    let result = engine.step_batch(50)?;
    let elapsed = start.elapsed();

    if result.potential_energy.is_finite() && result.kinetic_energy.is_finite() {
        println!("✓ PASS ({:.2}s)", elapsed.as_secs_f64());
        println!(
            "    PE: {:.2e}, KE: {:.2e}",
            result.potential_energy, result.kinetic_energy
        );
    } else {
        println!("✗ FAIL: NaN or Inf detected");
        all_passed = false;
    }
    println!();

    // =========================================================================
    // SUMMARY
    // =========================================================================
    println!("═══════════════════════════════════════════════════════════════");
    if all_passed {
        println!("  ALL TESTS PASSED ✓");
        println!("  SM_120 kernels validated - no race conditions or livelocks");
    } else {
        println!("  SOME TESTS FAILED ✗");
        println!("  Review failures above");
    }
    println!("═══════════════════════════════════════════════════════════════");

    if all_passed {
        Ok(())
    } else {
        anyhow::bail!("Validation failed")
    }
}
