//! Water Box Solvation for Explicit Solvent MD
//!
//! [STAGE-1-SOLVATE]
//!
//! Generates TIP3P water molecules around protein structures for explicit solvent simulations.
//!
//! ## Physical Model
//! - Water model: TIP3P (3-point rigid water)
//! - Density: ~1 g/cm³ = ~0.0334 molecules/Å³
//! - O-O spacing: ~3.1 Å (face-centered cubic packing)
//! - Overlap cutoff: 2.4 Å (minimum O-protein distance)
//!
//! ## Usage
//! ```ignore
//! use prism_nhs::solvate::solvate_protein;
//!
//! let (water_coords, water_indices) = solvate_protein(
//!     &topology,
//!     &protein_coords,
//!     10.0,  // 10 Å padding
//! )?;
//! // water_coords: [x1, y1, z1, x2, y2, z2, ...]  (oxygen positions only)
//! // water_indices: [atom_idx1, atom_idx2, ...] (indices in combined system)
//! ```

use crate::input::PrismPrepTopology;
use anyhow::{Context, Result};

/// 3D vector type
type Vec3 = [f32; 3];

/// TIP3P water O-O spacing for ~1 g/cm³ density (Angstroms)
/// Face-centered cubic packing with 30 waters/nm³
const WATER_SPACING: f32 = 3.1;

/// Minimum distance between water oxygen and protein heavy atom (Angstroms)
/// Waters closer than this are considered overlapping and removed
const OVERLAP_CUTOFF: f32 = 2.4;

// =============================================================================
// BOUNDING BOX COMPUTATION
// =============================================================================

/// Compute axis-aligned bounding box (AABB) of atomic coordinates
///
/// Returns (min_corner, max_corner) where each corner is [x, y, z]
///
/// # Arguments
/// * `coordinates` - Flat array [x1, y1, z1, x2, y2, z2, ...]
///
/// # Returns
/// * `(Vec3, Vec3)` - (min_corner, max_corner)
///
/// # Panics
/// * If coordinates length is not divisible by 3
pub fn compute_bbox(coordinates: &[f32]) -> Result<(Vec3, Vec3)> {
    if coordinates.is_empty() {
        anyhow::bail!("Cannot compute bounding box of empty coordinate array");
    }

    if coordinates.len() % 3 != 0 {
        anyhow::bail!(
            "Coordinate array length {} is not divisible by 3",
            coordinates.len()
        );
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for i in (0..coordinates.len()).step_by(3) {
        for j in 0..3 {
            let coord = coordinates[i + j];
            min[j] = min[j].min(coord);
            max[j] = max[j].max(coord);
        }
    }

    // Sanity check: box should be reasonable
    for i in 0..3 {
        if !min[i].is_finite() || !max[i].is_finite() {
            anyhow::bail!("Bounding box contains infinite/NaN coordinates");
        }
        if max[i] < min[i] {
            anyhow::bail!("Invalid bounding box: max < min on axis {}", i);
        }
    }

    let dims = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    log::debug!(
        "Computed bounding box: min=[{:.2}, {:.2}, {:.2}], max=[{:.2}, {:.2}, {:.2}], dims=[{:.1}, {:.1}, {:.1}] Å",
        min[0], min[1], min[2], max[0], max[1], max[2], dims[0], dims[1], dims[2]
    );

    Ok((min, max))
}

// =============================================================================
// OVERLAP DETECTION
// =============================================================================

/// Check if a position overlaps with any protein atom
///
/// Returns true if distance to ANY protein atom is less than cutoff.
/// Uses squared distances to avoid sqrt() in inner loop.
///
/// # Arguments
/// * `pos` - Position to check [x, y, z]
/// * `protein_coords` - Protein coordinates [x1, y1, z1, x2, y2, z2, ...]
/// * `cutoff` - Minimum allowed distance (Angstroms)
///
/// # Returns
/// * `true` if position overlaps (too close to protein)
/// * `false` if position is valid (no overlap)
pub fn overlaps_protein(pos: Vec3, protein_coords: &[f32], cutoff: f32) -> bool {
    let cutoff_sq = cutoff * cutoff;

    for i in (0..protein_coords.len()).step_by(3) {
        let dx = pos[0] - protein_coords[i];
        let dy = pos[1] - protein_coords[i + 1];
        let dz = pos[2] - protein_coords[i + 2];
        let dist_sq = dx * dx + dy * dy + dz * dz;

        if dist_sq < cutoff_sq {
            return true; // Overlap detected
        }
    }

    false // No overlap
}

// =============================================================================
// PROTEIN SOLVATION (FULL BOX)
// =============================================================================

/// Solvate protein with TIP3P water box
///
/// Generates water molecules in a rectangular box around the protein with specified padding.
/// Waters overlapping with protein atoms are removed.
///
/// # Algorithm
/// 1. Compute protein bounding box
/// 2. Expand by padding on all sides
/// 3. Fill box with waters on regular grid (WATER_SPACING = 3.1 Å)
/// 4. Remove waters overlapping with protein (< OVERLAP_CUTOFF = 2.4 Å)
///
/// # Arguments
/// * `topology` - Protein topology (for atom count)
/// * `coordinates` - Protein coordinates [x1, y1, z1, ...]
/// * `padding` - Padding around protein (Angstroms), typical: 10-15 Å
///
/// # Returns
/// * `(Vec<f32>, Vec<usize>)` - (water_coords, water_indices)
///   - water_coords: Flat array of oxygen positions [x1, y1, z1, x2, y2, z2, ...]
///   - water_indices: Atom indices in combined system [protein_n, protein_n+1, ...]
///
/// # Example
/// ```ignore
/// let (waters, indices) = solvate_protein(&topo, &coords, 10.0)?;
/// println!("Added {} water molecules", waters.len() / 3);
/// ```
pub fn solvate_protein(
    topology: &PrismPrepTopology,
    coordinates: &[f32],
    padding: f32,
) -> Result<(Vec<f32>, Vec<usize>)> {
    if padding <= 0.0 {
        anyhow::bail!("Padding must be positive, got {}", padding);
    }

    log::info!(
        "Solvating protein with {}Å padding (TIP3P water model)",
        padding
    );

    // Step 1: Compute protein bounding box
    let (min_coords, max_coords) =
        compute_bbox(coordinates).context("Failed to compute protein bounding box")?;

    // Step 2: Expand by padding
    let box_min = [
        min_coords[0] - padding,
        min_coords[1] - padding,
        min_coords[2] - padding,
    ];
    let box_max = [
        max_coords[0] + padding,
        max_coords[1] + padding,
        max_coords[2] + padding,
    ];

    let box_dims = [
        box_max[0] - box_min[0],
        box_max[1] - box_min[1],
        box_max[2] - box_min[2],
    ];

    let box_volume = box_dims[0] * box_dims[1] * box_dims[2];
    let expected_waters = (box_volume * 0.0334) as usize; // ~30 waters/nm³

    log::info!(
        "Solvation box: [{:.1} x {:.1} x {:.1}] Å³, volume={:.1} Å³, expected ~{} waters",
        box_dims[0],
        box_dims[1],
        box_dims[2],
        box_volume,
        expected_waters
    );

    // Step 3: Fill box with water grid
    let mut water_coords = Vec::with_capacity(expected_waters * 3);
    let mut water_indices = Vec::with_capacity(expected_waters);
    let mut waters_generated = 0usize;
    let mut waters_removed = 0usize;

    let nx = ((box_dims[0] / WATER_SPACING).ceil()) as usize;
    let ny = ((box_dims[1] / WATER_SPACING).ceil()) as usize;
    let nz = ((box_dims[2] / WATER_SPACING).ceil()) as usize;

    log::debug!(
        "Water grid: {}x{}x{} = {} potential positions",
        nx,
        ny,
        nz,
        nx * ny * nz
    );

    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let x = box_min[0] + (ix as f32) * WATER_SPACING;
                let y = box_min[1] + (iy as f32) * WATER_SPACING;
                let z = box_min[2] + (iz as f32) * WATER_SPACING;

                let pos = [x, y, z];
                waters_generated += 1;

                // Step 4: Check for overlap with protein
                if overlaps_protein(pos, coordinates, OVERLAP_CUTOFF) {
                    waters_removed += 1;
                    continue;
                }

                // Accept this water
                water_coords.extend_from_slice(&pos);
                let water_idx = topology.n_atoms + (water_indices.len());
                water_indices.push(water_idx);
            }
        }
    }

    let final_water_count = water_indices.len();
    let removal_percent = (waters_removed as f32 / waters_generated as f32) * 100.0;

    log::info!(
        "Solvation complete: {} waters added ({} generated, {} removed due to overlap = {:.1}%)",
        final_water_count,
        waters_generated,
        waters_removed,
        removal_percent
    );

    if final_water_count == 0 {
        log::warn!("No waters added! Check padding and overlap cutoff.");
    }

    // Sanity check: density should be reasonable
    let actual_density = (final_water_count as f32) / box_volume;
    let expected_density = 0.0334; // molecules/Å³
    let density_ratio = actual_density / expected_density;

    if density_ratio < 0.5 || density_ratio > 1.5 {
        log::warn!(
            "Water density {:.4} molecules/Å³ is unusual (expected ~{:.4}, ratio={:.2}x)",
            actual_density,
            expected_density,
            density_ratio
        );
    }

    Ok((water_coords, water_indices))
}

// =============================================================================
// REGIONAL SOLVATION (HYBRID MODE)
// =============================================================================

/// Solvate a spherical region around a point (for hybrid mode)
///
/// Used in hybrid mode to add explicit waters only around interesting regions
/// detected by RT probes, rather than solvating the entire protein.
///
/// # Arguments
/// * `center` - Center of region [x, y, z]
/// * `radius` - Radius of solvation sphere (Angstroms)
/// * `protein_coords` - Protein coordinates (to avoid overlaps)
///
/// # Returns
/// * `Vec<f32>` - Water coordinates [x1, y1, z1, x2, y2, z2, ...]
///
/// # Example
/// ```ignore
/// // RT probes detected void at [50.0, 30.0, 20.0]
/// let local_waters = solvate_region([50.0, 30.0, 20.0], 10.0, &protein_coords)?;
/// println!("Added {} local waters", local_waters.len() / 3);
/// ```
pub fn solvate_region(center: Vec3, radius: f32, protein_coords: &[f32]) -> Result<Vec<f32>> {
    if radius <= 0.0 {
        anyhow::bail!("Solvation radius must be positive, got {}", radius);
    }

    log::debug!(
        "Solvating region: center=[{:.2}, {:.2}, {:.2}], radius={:.1}Å",
        center[0],
        center[1],
        center[2],
        radius
    );

    // Bounding box for the region
    let box_min = [center[0] - radius, center[1] - radius, center[2] - radius];
    let box_max = [center[0] + radius, center[1] + radius, center[2] + radius];

    let mut water_coords = Vec::new();
    let radius_sq = radius * radius;

    // Fill bounding box, then filter by spherical distance
    let nx = ((2.0 * radius / WATER_SPACING).ceil()) as usize;
    let ny = nx;
    let nz = nx;

    for ix in 0..=nx {
        for iy in 0..=ny {
            for iz in 0..=nz {
                let x = box_min[0] + (ix as f32) * WATER_SPACING;
                let y = box_min[1] + (iy as f32) * WATER_SPACING;
                let z = box_min[2] + (iz as f32) * WATER_SPACING;

                // Check if within sphere
                let dx = x - center[0];
                let dy = y - center[1];
                let dz = z - center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq > radius_sq {
                    continue; // Outside sphere
                }

                let pos = [x, y, z];

                // Check for overlap with protein
                if overlaps_protein(pos, protein_coords, OVERLAP_CUTOFF) {
                    continue;
                }

                // Accept this water
                water_coords.extend_from_slice(&pos);
            }
        }
    }

    let water_count = water_coords.len() / 3;
    let sphere_volume = (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius;
    let density = (water_count as f32) / sphere_volume;

    log::debug!(
        "Regional solvation complete: {} waters in {:.1}Å radius sphere (density={:.4} molecules/Å³)",
        water_count, radius, density
    );

    Ok(water_coords)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bbox_simple() {
        // Simple 2-atom system
        let coords = vec![
            0.0, 0.0, 0.0, // atom 1 at origin
            10.0, 5.0, 3.0, // atom 2
        ];

        let (min, max) = compute_bbox(&coords).unwrap();

        assert_eq!(min, [0.0, 0.0, 0.0]);
        assert_eq!(max, [10.0, 5.0, 3.0]);
    }

    #[test]
    fn test_compute_bbox_negative_coords() {
        let coords = vec![-5.0, -3.0, -2.0, 5.0, 3.0, 2.0];

        let (min, max) = compute_bbox(&coords).unwrap();

        assert_eq!(min, [-5.0, -3.0, -2.0]);
        assert_eq!(max, [5.0, 3.0, 2.0]);
    }

    #[test]
    fn test_compute_bbox_empty() {
        let coords: Vec<f32> = vec![];
        assert!(compute_bbox(&coords).is_err());
    }

    #[test]
    fn test_compute_bbox_invalid_length() {
        // Not divisible by 3
        let coords = vec![1.0, 2.0];
        assert!(compute_bbox(&coords).is_err());
    }

    #[test]
    fn test_overlaps_protein_detected() {
        let protein = vec![
            0.0, 0.0, 0.0, // atom at origin
        ];

        // Position 1.0 Å away - should overlap with cutoff 2.4 Å
        assert!(overlaps_protein([1.0, 0.0, 0.0], &protein, 2.4));

        // Position 2.0 Å away - should overlap
        assert!(overlaps_protein([2.0, 0.0, 0.0], &protein, 2.4));
    }

    #[test]
    fn test_overlaps_protein_no_overlap() {
        let protein = vec![0.0, 0.0, 0.0];

        // Position 3.0 Å away - no overlap with cutoff 2.4 Å
        assert!(!overlaps_protein([3.0, 0.0, 0.0], &protein, 2.4));

        // Position 10.0 Å away - definitely no overlap
        assert!(!overlaps_protein([10.0, 0.0, 0.0], &protein, 2.4));
    }

    #[test]
    fn test_overlaps_protein_multiple_atoms() {
        let protein = vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0];

        // Near first atom
        assert!(overlaps_protein([1.0, 0.0, 0.0], &protein, 2.4));

        // Near second atom
        assert!(overlaps_protein([11.0, 0.0, 0.0], &protein, 2.4));

        // Midpoint - far from both
        assert!(!overlaps_protein([5.0, 0.0, 0.0], &protein, 2.4));
    }

    #[test]
    fn test_solvate_protein_basic() {
        // Create minimal topology
        let topo = PrismPrepTopology {
            aromatic_targets: Vec::new(),
            source_pdb: String::new(),
            n_atoms: 2,
            n_residues: 1,
            total_physical_residue_count: None,
            n_analysis_residues: None,
            n_chains: 1,
            positions: Vec::new(),
            elements: Vec::new(),
            atom_names: Vec::new(),
            residue_names: Vec::new(),
            residue_ids: Vec::new(),
            atom_physical_residue_ids: None,
            atom_analysis_residue_ids: None,
            analysis_residue_contract: None,
            chain_ids: Vec::new(),
            charges: Vec::new(),
            masses: Vec::new(),
            ca_indices: Vec::new(),
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
        };

        // Simple 2-atom protein (10 Å apart)
        let coords = vec![0.0, 0.0, 0.0, 10.0, 0.0, 0.0];

        let (waters, indices) = solvate_protein(&topo, &coords, 5.0).unwrap();

        // Should have added some waters
        assert!(waters.len() > 0, "No waters added");
        assert_eq!(waters.len() % 3, 0, "Water coords not multiple of 3");
        assert_eq!(waters.len() / 3, indices.len(), "Coords/indices mismatch");

        // All water indices should be >= n_atoms
        for &idx in &indices {
            assert!(
                idx >= topo.n_atoms,
                "Water index {} < n_atoms {}",
                idx,
                topo.n_atoms
            );
        }

        println!("test_solvate_protein_basic: Added {} waters", indices.len());
    }

    #[test]
    fn test_solvate_protein_density() {
        let topo = PrismPrepTopology {
            aromatic_targets: Vec::new(),
            source_pdb: String::new(),
            n_atoms: 1,
            n_residues: 1,
            total_physical_residue_count: None,
            n_analysis_residues: None,
            n_chains: 1,
            positions: Vec::new(),
            elements: Vec::new(),
            atom_names: Vec::new(),
            residue_names: Vec::new(),
            residue_ids: Vec::new(),
            atom_physical_residue_ids: None,
            atom_analysis_residue_ids: None,
            analysis_residue_contract: None,
            chain_ids: Vec::new(),
            charges: Vec::new(),
            masses: Vec::new(),
            ca_indices: Vec::new(),
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
        };

        // Single atom at origin
        let coords = vec![0.0, 0.0, 0.0];

        // Large padding to ensure many waters
        let (waters, _) = solvate_protein(&topo, &coords, 20.0).unwrap();
        let water_count = waters.len() / 3;

        // Box dimensions: ~40x40x40 Å
        // Volume: ~64,000 Å³
        // Expected: ~64,000 * 0.0334 ≈ 2,138 waters
        // With overlap removal: ~1,500-2,000 waters

        assert!(water_count > 1000, "Water count {} too low", water_count);
        assert!(water_count < 3000, "Water count {} too high", water_count);

        println!(
            "test_solvate_protein_density: {} waters in large box",
            water_count
        );
    }

    #[test]
    fn test_solvate_protein_no_overlap() {
        let topo = PrismPrepTopology {
            aromatic_targets: Vec::new(),
            source_pdb: String::new(),
            n_atoms: 1,
            n_residues: 1,
            total_physical_residue_count: None,
            n_analysis_residues: None,
            n_chains: 1,
            positions: Vec::new(),
            elements: Vec::new(),
            atom_names: Vec::new(),
            residue_names: Vec::new(),
            residue_ids: Vec::new(),
            atom_physical_residue_ids: None,
            atom_analysis_residue_ids: None,
            analysis_residue_contract: None,
            chain_ids: Vec::new(),
            charges: Vec::new(),
            masses: Vec::new(),
            ca_indices: Vec::new(),
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
        };

        let coords = vec![0.0, 0.0, 0.0];
        let (waters, _) = solvate_protein(&topo, &coords, 10.0).unwrap();

        // Check that no water is too close to protein atom at origin
        for i in (0..waters.len()).step_by(3) {
            let dx = waters[i] - 0.0;
            let dy = waters[i + 1] - 0.0;
            let dz = waters[i + 2] - 0.0;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            assert!(
                dist >= OVERLAP_CUTOFF,
                "Water at {:.2} Å too close (cutoff {:.2} Å)",
                dist,
                OVERLAP_CUTOFF
            );
        }

        println!(
            "test_solvate_protein_no_overlap: All {} waters validated",
            waters.len() / 3
        );
    }

    #[test]
    fn test_solvate_region_basic() {
        let protein = vec![0.0, 0.0, 0.0];
        let center = [10.0, 10.0, 10.0];
        let radius = 5.0;

        let waters = solvate_region(center, radius, &protein).unwrap();

        assert!(waters.len() > 0, "No waters in region");
        assert_eq!(waters.len() % 3, 0);

        println!(
            "test_solvate_region_basic: {} waters in 5Å sphere",
            waters.len() / 3
        );
    }

    #[test]
    fn test_solvate_region_all_within_radius() {
        let protein = vec![100.0, 100.0, 100.0]; // Far away
        let center = [0.0, 0.0, 0.0];
        let radius = 10.0;

        let waters = solvate_region(center, radius, &protein).unwrap();

        // Check all waters are within radius
        for i in (0..waters.len()).step_by(3) {
            let dx = waters[i] - center[0];
            let dy = waters[i + 1] - center[1];
            let dz = waters[i + 2] - center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            assert!(
                dist <= radius,
                "Water at {:.2} Å outside radius {:.2} Å",
                dist,
                radius
            );
        }

        println!(
            "test_solvate_region_all_within_radius: {} waters validated",
            waters.len() / 3
        );
    }

    #[test]
    fn test_solvate_region_zero_radius() {
        let protein = vec![0.0, 0.0, 0.0];
        let result = solvate_region([0.0, 0.0, 0.0], 0.0, &protein);
        assert!(result.is_err(), "Zero radius should fail");
    }

    #[test]
    fn test_solvate_region_negative_radius() {
        let protein = vec![0.0, 0.0, 0.0];
        let result = solvate_region([0.0, 0.0, 0.0], -5.0, &protein);
        assert!(result.is_err(), "Negative radius should fail");
    }
}
