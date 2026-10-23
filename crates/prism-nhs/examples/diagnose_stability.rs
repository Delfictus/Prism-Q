//! Diagnose physics stability issues
//!
//! Runs detailed step-by-step analysis to find when/why physics explodes

use anyhow::Result;
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol};

fn compute_stats(positions: &[f32]) -> (f32, f32, f32, bool) {
    // Returns (max_abs, mean_abs, max_velocity_proxy, has_nan)
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f32;
    let mut has_nan = false;

    for &p in positions {
        if p.is_nan() || p.is_infinite() {
            has_nan = true;
            continue;
        }
        let abs_p = p.abs();
        max_abs = max_abs.max(abs_p);
        sum_abs += abs_p;
    }

    let mean_abs = if positions.len() > 0 {
        sum_abs / positions.len() as f32
    } else {
        0.0
    };
    (max_abs, mean_abs, 0.0, has_nan)
}

fn compute_rmsd(pos1: &[f32], pos2: &[f32]) -> f32 {
    if pos1.len() != pos2.len() || pos1.is_empty() {
        return 0.0;
    }
    let n = pos1.len() / 3;
    let mut sum = 0.0;
    for i in 0..n {
        let dx = pos1[i * 3] - pos2[i * 3];
        let dy = pos1[i * 3 + 1] - pos2[i * 3 + 1];
        let dz = pos1[i * 3 + 2] - pos2[i * 3 + 2];
        sum += dx * dx + dy * dy + dz * dz;
    }
    (sum / n as f32).sqrt()
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    println!("======================================================================");
    println!("              PRISM4D PHYSICS STABILITY DIAGNOSTIC");
    println!("======================================================================\n");

    println!("Loading topology: {}", topology_path.display());
    let topology = PrismPrepTopology::load(topology_path)?;
    println!("  Atoms: {}", topology.n_atoms);
    println!("  Bonds: {}", topology.bonds.len());
    println!("  Angles: {}", topology.angles.len());
    println!("  Dihedrals: {}", topology.dihedrals.len());

    // Check initial positions
    let (max_pos, mean_pos, _, has_nan) = compute_stats(&topology.positions);
    println!("\nInitial position stats:");
    println!("  Max |pos|: {:.2} Å", max_pos);
    println!("  Mean |pos|: {:.2} Å", mean_pos);
    println!("  Has NaN: {}", has_nan);

    if max_pos > 200.0 {
        println!("  [WARN] Structure may not be centered (max > 200 Å)");
    }

    println!("\nCreating CUDA context...");
    let context = CudaContext::new(0)?;

    println!("Creating engine (48x48x48 grid, 1.2nm cutoff)...");
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

    // Set constant temperature for stability test
    engine.set_temperature_protocol(TemperatureProtocol {
        start_temp: 300.0,
        end_temp: 300.0,
        ramp_steps: 0,
        hold_steps: 100000,
        cold_hold_steps: 0,
        current_step: 0,
    })?;

    println!("\n=== Running 10000 steps with detailed logging ===\n");
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>8} {:>10}",
        "Step", "MaxPos", "MeanPos", "RMSD", "Spikes", "Status"
    );
    println!("{}", "-".repeat(66));

    let reference_pos = engine.get_positions()?;
    let mut prev_pos = reference_pos.clone();
    let mut explosion_step = None;

    // Run in batches of 100 steps
    for batch in 0..100 {
        let step_start = batch * 100;

        // Run 100 steps
        let summary = engine.run(100)?;

        // Get current positions
        let current_pos = engine.get_positions()?;
        let (max_pos, mean_pos, _, has_nan) = compute_stats(&current_pos);
        let rmsd_from_ref = compute_rmsd(&reference_pos, &current_pos);
        let rmsd_from_prev = compute_rmsd(&prev_pos, &current_pos);

        let status = if has_nan {
            "NaN!"
        } else if max_pos > 500.0 {
            "EXPLODED"
        } else if rmsd_from_prev > 20.0 {
            "UNSTABLE"
        } else if rmsd_from_ref > 50.0 {
            "DRIFTED"
        } else {
            "OK"
        };

        println!(
            "{:>6} {:>10.2} {:>10.2} {:>10.2} {:>8} {:>10}",
            step_start + 100,
            max_pos,
            mean_pos,
            rmsd_from_ref,
            summary.total_spikes,
            status
        );

        if (has_nan || max_pos > 500.0) && explosion_step.is_none() {
            explosion_step = Some(step_start + 100);
            println!("\n[ERROR] Explosion detected at step {}!", step_start + 100);

            // Print some diagnostic info
            let mut nan_count = 0;
            let mut inf_count = 0;
            let mut huge_count = 0;
            for &p in &current_pos {
                if p.is_nan() {
                    nan_count += 1;
                } else if p.is_infinite() {
                    inf_count += 1;
                } else if p.abs() > 1000.0 {
                    huge_count += 1;
                }
            }
            println!("  NaN positions: {}", nan_count);
            println!("  Inf positions: {}", inf_count);
            println!("  |pos| > 1000: {}", huge_count);
            break;
        }

        prev_pos = current_pos;
    }

    println!("\n{}", "=".repeat(66));
    if explosion_step.is_some() {
        println!(
            "RESULT: Physics explosion detected at step {}",
            explosion_step.unwrap()
        );
        println!("\nPossible causes:");
        println!("  1. Equilibration friction not applied correctly");
        println!("  2. Force calculation overflow");
        println!("  3. SHAKE constraint failure");
        println!("  4. Bad initial geometry (overlapping atoms)");
    } else {
        println!("RESULT: Physics remained stable for 10000 steps");
    }
    println!("{}", "=".repeat(66));

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
