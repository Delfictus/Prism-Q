//! Diagnose Cryo-UV physics stability
//!
//! Tests the actual cryo protocol starting at low temperature

use anyhow::Result;
use std::path::Path;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
use prism_nhs::input::PrismPrepTopology;
#[cfg(feature = "gpu")]
use prism_nhs::{NhsAmberFusedEngine, TemperatureProtocol};

fn compute_stats(positions: &[f32]) -> (f32, f32, bool) {
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
    (max_abs, mean_abs, has_nan)
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
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let topology_path = Path::new(
        concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/prism_prep_test/6LU7_topology.json"),
    );

    println!("======================================================================");
    println!("         PRISM4D CRYO-UV PHYSICS STABILITY DIAGNOSTIC");
    println!("======================================================================\n");

    println!("Loading topology: {}", topology_path.display());
    let topology = PrismPrepTopology::load(topology_path)?;
    println!("  Atoms: {}", topology.n_atoms);

    println!("\nCreating CUDA context and engine...");
    let context = CudaContext::new(0)?;
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

    // === TEST 1: CRYO START (100K constant) ===
    println!("\n=== TEST 1: Cryo 100K constant (10000 steps) ===\n");

    engine.set_temperature_protocol(TemperatureProtocol {
        start_temp: 100.0,
        end_temp: 100.0,
        ramp_steps: 0,
        hold_steps: 100000,
        cold_hold_steps: 0,
        current_step: 0,
    })?;

    let reference_pos = engine.get_positions()?;
    println!(
        "{:>6} {:>10} {:>10} {:>10} {:>8}",
        "Step", "MaxPos", "MeanPos", "RMSD", "Status"
    );
    println!("{}", "-".repeat(50));

    let mut stable = true;
    for batch in 0..100 {
        let summary = engine.run(100)?;
        let current_pos = engine.get_positions()?;
        let (max_pos, mean_pos, has_nan) = compute_stats(&current_pos);
        let rmsd = compute_rmsd(&reference_pos, &current_pos);

        let status = if has_nan || max_pos > 500.0 {
            "EXPLODED"
        } else if rmsd > 20.0 {
            "DRIFTED"
        } else {
            "OK"
        };

        if batch % 10 == 9 {
            println!(
                "{:>6} {:>10.2} {:>10.2} {:>10.2} {:>8}",
                (batch + 1) * 100,
                max_pos,
                mean_pos,
                rmsd,
                status
            );
        }

        if has_nan || max_pos > 500.0 || rmsd > 50.0 {
            stable = false;
            break;
        }
    }
    println!(
        "Result: {}",
        if stable { "STABLE at 100K" } else { "UNSTABLE" }
    );

    // Reset engine for next test
    let context = CudaContext::new(0)?;
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

    // === TEST 2: CRYO RAMP (100K -> 300K over 10000 steps) ===
    println!("\n=== TEST 2: Cryo ramp 100K -> 300K (10000 steps) ===\n");

    engine.set_temperature_protocol(TemperatureProtocol {
        start_temp: 100.0,
        end_temp: 300.0,
        ramp_steps: 10000,
        hold_steps: 0,
        cold_hold_steps: 0,
        current_step: 0,
    })?;

    let reference_pos = engine.get_positions()?;
    println!(
        "{:>6} {:>6} {:>10} {:>10} {:>8}",
        "Step", "Temp", "MaxPos", "RMSD", "Status"
    );
    println!("{}", "-".repeat(50));

    let mut stable = true;
    for batch in 0..100 {
        let summary = engine.run(100)?;
        let current_pos = engine.get_positions()?;
        let (max_pos, mean_pos, has_nan) = compute_stats(&current_pos);
        let rmsd = compute_rmsd(&reference_pos, &current_pos);
        let temp = 100.0 + (batch as f32 + 1.0) * 2.0; // Approximate temp

        let status = if has_nan || max_pos > 500.0 {
            "EXPLODED"
        } else if rmsd > 20.0 {
            "DRIFTED"
        } else {
            "OK"
        };

        if batch % 10 == 9 {
            println!(
                "{:>6} {:>6.0} {:>10.2} {:>10.2} {:>8}",
                (batch + 1) * 100,
                temp,
                max_pos,
                rmsd,
                status
            );
        }

        if has_nan || max_pos > 500.0 || rmsd > 50.0 {
            stable = false;
            break;
        }
    }
    println!(
        "Result: {}",
        if stable {
            "STABLE during ramp"
        } else {
            "UNSTABLE"
        }
    );

    // === TEST 3: Very cold (50K constant) ===
    let context = CudaContext::new(0)?;
    let mut engine = NhsAmberFusedEngine::new(context, &topology, 48, 1.2)?;

    println!("\n=== TEST 3: Ultra-Cryo 50K constant (10000 steps) ===\n");

    engine.set_temperature_protocol(TemperatureProtocol {
        start_temp: 50.0,
        end_temp: 50.0,
        ramp_steps: 0,
        hold_steps: 100000,
        cold_hold_steps: 0,
        current_step: 0,
    })?;

    let reference_pos = engine.get_positions()?;
    println!(
        "{:>6} {:>10} {:>10} {:>8}",
        "Step", "MaxPos", "RMSD", "Status"
    );
    println!("{}", "-".repeat(38));

    let mut stable = true;
    for batch in 0..100 {
        let summary = engine.run(100)?;
        let current_pos = engine.get_positions()?;
        let (max_pos, _, has_nan) = compute_stats(&current_pos);
        let rmsd = compute_rmsd(&reference_pos, &current_pos);

        let status = if has_nan || max_pos > 500.0 {
            "EXPLODED"
        } else if rmsd > 10.0 {
            "DRIFTED"
        } else {
            "OK"
        };

        if batch % 10 == 9 {
            println!(
                "{:>6} {:>10.2} {:>10.2} {:>8}",
                (batch + 1) * 100,
                max_pos,
                rmsd,
                status
            );
        }

        if has_nan || max_pos > 500.0 {
            stable = false;
            break;
        }
    }
    println!(
        "Result: {}",
        if stable { "STABLE at 50K" } else { "UNSTABLE" }
    );

    println!("\n======================================================================");
    println!("                           SUMMARY");
    println!("======================================================================");
    println!("The cryo protocol (starting at low T) should be more stable than 300K");
    println!("because thermal motion is reduced and the equilibration friction is");
    println!("enhanced by the cryo scaling (gamma * sqrt(300/T)).");
    println!("======================================================================\n");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("Requires --features gpu");
}
