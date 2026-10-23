use crate::input::PrismPrepTopology;
///! Integration layer for AmberSimdBatch with NHS batch processor
///!
///! Converts PrismPrepTopology to StructureTopology format for concurrent batch MD
use anyhow::{Context, Result};
use prism_gpu::amber_simd_batch::StructureTopology;
use std::collections::HashSet;

/// Convert PrismPrepTopology to StructureTopology for AmberSimdBatch
pub fn convert_to_structure_topology(prism_topo: &PrismPrepTopology) -> Result<StructureTopology> {
    let harmonic = prism_topo.lower_harmonic_parameters(
        crate::input::HarmonicEnergyConvention::OpenmmFullKHalfEnergy,
    ).map_err(anyhow::Error::msg)?;
    // Extract LJ parameters (sigma, epsilon) from lj_params
    let mut sigmas = Vec::with_capacity(prism_topo.n_atoms);
    let mut epsilons = Vec::with_capacity(prism_topo.n_atoms);

    for lj in &prism_topo.lj_params {
        sigmas.push(lj.sigma as f32);
        epsilons.push(lj.epsilon as f32);
    }

    // Consumer tuple is (i, j, k, r0), matching AmberSimdBatch/EnsembleWarp uploads.
    let bonds: Vec<(usize, usize, f32, f32)> = prism_topo
        .bonds
        .iter()
        .zip(harmonic.bond_k.iter())
        .map(|(b, &k)| (b.i, b.j, k, b.r0 as f32))
        .collect();

    // Consumer tuple is (i, j, k_idx, force_k, theta0).
    let angles: Vec<(usize, usize, usize, f32, f32)> = prism_topo
        .angles
        .iter()
        .zip(harmonic.angle_k.iter())
        .map(|(a, &k)| (a.i, a.j, a.k_idx, k, a.theta0 as f32))
        .collect();

    // Consumer tuple is (i, j, k, l, force_k, periodicity, phase).
    let dihedrals: Vec<(usize, usize, usize, usize, f32, f32, f32)> = prism_topo
        .dihedrals
        .iter()
        .map(|d| {
            (
                d.i,
                d.j,
                d.k_idx,
                d.l, // Fourth atom index
                d.force_k as f32,
                d.periodicity as f32,
                d.phase as f32,
            )
        })
        .collect();

    // Convert exclusions Vec<Vec<usize>> to Vec<HashSet<usize>>
    let exclusions: Vec<HashSet<usize>> = prism_topo
        .exclusions
        .iter()
        .map(|excl_vec| excl_vec.iter().cloned().collect())
        .collect();

    Ok(StructureTopology {
        positions: prism_topo.positions.clone(),
        masses: prism_topo.masses.clone(),
        charges: prism_topo.charges.clone(),
        sigmas,
        epsilons,
        bonds,
        angles,
        dihedrals,
        exclusions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_basic() {
        // Test that conversion preserves atom count and basic properties
        // This will be tested with actual topology files
    }
}
