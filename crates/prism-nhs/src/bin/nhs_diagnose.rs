//! NHS NaN Diagnostic Tool
//!
//! Runs minimal steps to identify where NaN values first appear in the NHS fused engine.

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_nhs::{fused_engine::NhsAmberFusedEngine, input::PrismPrepTopology};

#[derive(Parser, Debug)]
#[command(name = "nhs-diagnose")]
#[command(about = "Diagnose NaN issues in NHS fused engine")]
struct Args {
    /// Input PRISM-PREP topology JSON file
    #[arg(required = true)]
    input: PathBuf,

    /// Number of steps to run (default: 1)
    #[arg(short, long, default_value = "1")]
    steps: usize,

    /// Grid dimension
    #[arg(long, default_value = "32")]
    grid_dim: usize,
}

fn count_nan(values: &[f32], label: &str) -> usize {
    let nan_count = values.iter().filter(|&&x| x.is_nan()).count();
    let inf_count = values.iter().filter(|&&x| x.is_infinite()).count();
    let total = values.len();

    if nan_count > 0 || inf_count > 0 {
        println!(
            "  {} NaN: {}/{}, Inf: {}/{}",
            label, nan_count, total, inf_count, total
        );

        // Find first NaN/Inf index
        for (i, &v) in values.iter().enumerate() {
            if v.is_nan() || v.is_infinite() {
                let atom_idx = i / 3;
                let component = ["X", "Y", "Z"][i % 3];
                println!(
                    "    First bad value at atom {} component {}: {}",
                    atom_idx, component, v
                );
                break;
            }
        }
    } else {
        // Print range for valid values
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        println!("  {} OK - range: [{:.2}, {:.2}]", label, min, max);
    }

    nan_count + inf_count
}

fn check_masses(masses: &[f32]) -> bool {
    let zero_count = masses.iter().filter(|&&m| m == 0.0).count();
    let negative_count = masses.iter().filter(|&&m| m < 0.0).count();
    let nan_count = masses.iter().filter(|&&m| m.is_nan()).count();

    println!(
        "  Masses: {} atoms, {} zeros, {} negative, {} NaN",
        masses.len(),
        zero_count,
        negative_count,
        nan_count
    );

    if zero_count > 0 {
        // Find first zero mass
        for (i, &m) in masses.iter().enumerate() {
            if m == 0.0 {
                println!("    First zero mass at atom {}", i);
                break;
            }
        }
    }

    zero_count == 0 && negative_count == 0 && nan_count == 0
}

#[cfg(feature = "gpu")]
fn run_diagnostic(args: Args) -> Result<()> {
    println!("=== NHS NaN Diagnostic Tool ===\n");

    // Load topology
    println!("Loading topology: {:?}", args.input);
    let topology = PrismPrepTopology::load(&args.input).context("Failed to load topology")?;

    let n_atoms = topology.n_atoms;
    println!("  Atoms: {}", n_atoms);
    println!("  Bonds: {}", topology.bonds.len());
    println!("  Angles: {}", topology.angles.len());
    println!("  Dihedrals: {}", topology.dihedrals.len());

    // Check input positions
    println!("\n--- Input Data Check ---");
    count_nan(&topology.positions, "Input positions");
    let masses_ok = check_masses(&topology.masses);
    count_nan(&topology.charges, "Charges");

    if !masses_ok {
        println!("\n[ERROR] Zero masses detected - this will cause division by zero!");
        return Ok(());
    }

    // Initialize CUDA
    println!("\n--- CUDA Initialization ---");
    let ctx = CudaContext::new(0).context("Failed to create CUDA context")?;
    println!("  CUDA context created");

    // Create engine
    println!("\n--- Engine Creation ---");
    println!("  Grid dim: {}", args.grid_dim);
    let mut engine = NhsAmberFusedEngine::new(
        ctx,
        &topology,
        args.grid_dim,
        1.5, // grid_spacing
    )
    .context("Failed to create NHS engine")?;

    println!("  Engine created successfully");

    // Get positions BEFORE any steps
    println!("\n--- Position Check Before Steps ---");
    let pos_before = engine.get_positions().context("Failed to get positions")?;
    let bad_before = count_nan(&pos_before, "Positions");

    if bad_before > 0 {
        println!("\n[ERROR] Positions are already bad BEFORE running any steps!");
        println!("This indicates a problem with position upload to GPU.");
        return Ok(());
    }

    // Run steps one at a time
    println!("\n--- Running {} Steps ---", args.steps);
    for step in 0..args.steps {
        engine.step().context("Failed to execute step")?;

        let pos_after = engine.get_positions().context("Failed to get positions")?;
        let bad_count = count_nan(&pos_after, &format!("Step {}", step + 1));

        if bad_count > 0 {
            println!("\n[ERROR] NaN/Inf detected after step {}!", step + 1);

            // Try to identify which atoms went bad
            let mut first_bad_atoms = Vec::new();
            for (i, &v) in pos_after.iter().enumerate() {
                if v.is_nan() || v.is_infinite() {
                    let atom_idx = i / 3;
                    if !first_bad_atoms.contains(&atom_idx) {
                        first_bad_atoms.push(atom_idx);
                        if first_bad_atoms.len() >= 10 {
                            break;
                        }
                    }
                }
            }

            println!("  First bad atoms: {:?}", first_bad_atoms);

            // Check what properties these atoms have
            for &atom in &first_bad_atoms[..first_bad_atoms.len().min(5)] {
                let mass = topology.masses[atom];
                let charge = topology.charges[atom];
                let orig_pos = [
                    topology.positions[atom * 3],
                    topology.positions[atom * 3 + 1],
                    topology.positions[atom * 3 + 2],
                ];

                // Get residue info if available
                let res_name = if atom < topology.residue_names.len() {
                    &topology.residue_names[atom]
                } else {
                    "???"
                };
                let res_id = if atom < topology.residue_ids.len() {
                    topology.residue_ids[atom]
                } else {
                    0
                };

                println!("    Atom {}: mass={:.3}, charge={:.3}, res={} ({}), orig_pos=[{:.2}, {:.2}, {:.2}]",
                         atom, mass, charge, res_name, res_id, orig_pos[0], orig_pos[1], orig_pos[2]);
            }

            return Ok(());
        }
    }

    println!(
        "\n[SUCCESS] No NaN/Inf values detected after {} steps!",
        args.steps
    );

    // Print summary
    let pos_final = engine.get_positions()?;
    println!("\n--- Final Position Statistics ---");
    count_nan(&pos_final, "Final positions");

    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn run_diagnostic(_args: Args) -> Result<()> {
    anyhow::bail!("GPU feature not enabled. Rebuild with: cargo build --release -p prism-nhs --features gpu --bin nhs_diagnose");
}

fn main() -> Result<()> {
    env_logger::init();
    let args = Args::parse();
    run_diagnostic(args)
}
