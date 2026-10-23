//! Physics Sanity Check - AmberMegaFusedHmc Reference Implementation

use anyhow::{bail, Context, Result};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;
#[cfg(feature = "gpu")]
use prism_gpu::amber_mega_fused::AmberMegaFusedHmc;
#[cfg(feature = "gpu")]
use prism_nhs::PrismPrepTopology;

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  PHYSICS SANITY CHECK - AmberMegaFusedHmc");
    println!("═══════════════════════════════════════════════════════════════");

    #[cfg(not(feature = "gpu"))]
    bail!("GPU feature required");

    #[cfg(feature = "gpu")]
    run_sanity_check()
}

#[cfg(feature = "gpu")]
fn run_sanity_check() -> Result<()> {
    let topo_path = PathBuf::from("production_test/targets/16_GBA_apo.topology.json");
    println!("Loading: {}", topo_path.display());
    let topo = PrismPrepTopology::load(&topo_path)?;
    println!("  Atoms: {}\n", topo.n_atoms);

    let initial_coord = [topo.positions[0], topo.positions[1], topo.positions[2]];

    let cuda_ctx = CudaContext::new(0)?;
    let mut engine = AmberMegaFusedHmc::new(cuda_ctx, topo.n_atoms)?;

    // Convert topology
    let harmonic = topo.lower_harmonic_parameters(
        prism_nhs::input::HarmonicEnergyConvention::OpenmmFullKHalfEnergy,
    ).map_err(anyhow::Error::msg)?;
    let bonds: Vec<(usize, usize, f32, f32)> = topo
        .bonds
        .iter()
        .zip(harmonic.bond_k.iter())
        .map(|(b, &k)| (b.i, b.j, k, b.r0 as f32))
        .collect();
    let angles: Vec<(usize, usize, usize, f32, f32)> = topo
        .angles
        .iter()
        .zip(harmonic.angle_k.iter())
        .map(|(a, &k)| (a.i, a.j, a.k_idx, k, a.theta0 as f32))
        .collect();
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = topo
        .dihedrals
        .iter()
        .map(|d| {
            (
                d.i,
                d.j,
                d.k_idx,
                d.l,
                d.force_k as f32,
                d.periodicity as f32,
                d.phase as f32,
            )
        })
        .collect();
    let nb_params: Vec<(f32, f32, f32, f32)> = (0..topo.n_atoms)
        .map(|i| {
            (
                topo.lj_params[i].sigma as f32,
                topo.lj_params[i].epsilon as f32,
                topo.charges[i],
                topo.masses[i],
            )
        })
        .collect();
    let exclusions: Vec<HashSet<usize>> = topo
        .exclusions
        .iter()
        .map(|v| v.iter().copied().collect())
        .collect();

    engine.upload_topology(
        &topo.positions,
        &bonds,
        &angles,
        &dihedrals,
        &nb_params,
        &exclusions,
    )?;

    println!("Initializing velocities at 300K...");
    engine.initialize_velocities(300.0)?;

    println!("Running 1000 steps (dt=0.5fs, T=300K, gamma=0.1)...");
    let start = std::time::Instant::now();
    let result = engine.run(1000, 0.5, 300.0, 0.1)?; // Stable params for implicit solvent
    let elapsed = start.elapsed();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Results");
    println!("═══════════════════════════════════════════════════════════════");

    let final_coord = [
        result.positions[0],
        result.positions[1],
        result.positions[2],
    ];
    let dx = final_coord[0] - initial_coord[0];
    let dy = final_coord[1] - initial_coord[1];
    let dz = final_coord[2] - initial_coord[2];
    let disp = (dx * dx + dy * dy + dz * dz).sqrt();

    println!("  Displacement: {:.2} Å", disp);
    println!("  Temperature: {:.1} K", result.avg_temperature);
    println!(
        "  Potential Energy: {:.1} kcal/mol",
        result.potential_energy
    );
    println!("  Kinetic Energy: {:.1} kcal/mol", result.kinetic_energy);
    println!(
        "  Throughput: {:.0} steps/sec\n",
        1000.0 / elapsed.as_secs_f64()
    );

    if result.potential_energy > 1e6 || disp > 10.0 {
        println!(
            "❌ FAIL: Physics explosion (PE={:.1e}, disp={:.2}Å)",
            result.potential_energy, disp
        );
        std::process::exit(1);
    } else if result.potential_energy < 0.0 && disp < 3.0 {
        println!("✅ PASS: Physics HEALTHY!");
        println!("  → Negative PE, reasonable displacement");
        println!("  → PROCEED to 5-Target Validation");
        std::process::exit(0);
    } else {
        println!(
            "⚠️  Marginal (PE={:.1e}, disp={:.2}Å)",
            result.potential_energy, disp
        );
        std::process::exit(2);
    }
}
