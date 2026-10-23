//! Hydrophobic Exclusion Field Computation
//!
//! Computes a 3D grid where each voxel contains the probability that
//! water is EXCLUDED from that location based on nearby hydrophobic atoms.
//!
//! # The Holographic Negative Principle
//!
//! Instead of simulating water (expensive), we map what EXCLUDES water:
//! - Hydrophobic atoms create exclusion zones
//! - Polar atoms create attraction zones
//! - The "negative" of this field reveals where water CAN exist
//!
//! # Physical Basis
//!
//! The exclusion field E(r) is computed as:
//!
//! E(r) = 1 - ∏ᵢ (1 - Gᵢ(r))
//!
//! where Gᵢ(r) is a soft Gaussian exclusion around each hydrophobic atom i,
//! weighted by that atom's hydrophobicity.

use crate::config::{NhsConfig, WATER_PROBE_RADIUS};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// =============================================================================
// VAN DER WAALS RADII
// =============================================================================

/// Van der Waals radii by element (Å)
/// Source: Bondi, 1964
pub const VDW_RADII: [f32; 18] = [
    1.20, // H  (1)
    1.40, // He (2)
    1.82, // Li (3)
    1.53, // Be (4)
    1.92, // B  (5)
    1.70, // C  (6) ← IMPORTANT
    1.55, // N  (7) ← IMPORTANT
    1.52, // O  (8) ← IMPORTANT
    1.47, // F  (9)
    1.54, // Ne (10)
    2.27, // Na (11)
    1.73, // Mg (12)
    1.84, // Al (13)
    2.10, // Si (14)
    1.80, // P  (15) ← IMPORTANT
    1.80, // S  (16) ← IMPORTANT
    1.75, // Cl (17)
    1.88, // Ar (18)
];

/// Get VdW radius for an element by atomic number
#[inline]
pub fn get_vdw_radius(element: u8) -> f32 {
    if element == 0 || element > 18 {
        1.70 // Default to carbon
    } else {
        VDW_RADII[(element - 1) as usize]
    }
}

// =============================================================================
// HYDROPHOBICITY SCALE (Kyte-Doolittle)
// =============================================================================

/// Kyte-Doolittle hydrophobicity scale (normalized to 0-1)
/// Higher = more hydrophobic
pub fn kyte_doolittle_normalized(residue: &str) -> f32 {
    // Original scale: -4.5 (Arg) to +4.5 (Ile)
    // Normalized: 0.0 (most hydrophilic) to 1.0 (most hydrophobic)
    let raw = match residue.to_uppercase().as_str() {
        "ILE" | "I" => 4.5,
        "VAL" | "V" => 4.2,
        "LEU" | "L" => 3.8,
        "PHE" | "F" => 2.8,
        "CYS" | "C" => 2.5,
        "MET" | "M" => 1.9,
        "ALA" | "A" => 1.8,
        "GLY" | "G" => -0.4,
        "THR" | "T" => -0.7,
        "SER" | "S" => -0.8,
        "TRP" | "W" => -0.9,
        "TYR" | "Y" => -1.3,
        "PRO" | "P" => -1.6,
        "HIS" | "H" => -3.2,
        "GLU" | "E" => -3.5,
        "GLN" | "Q" => -3.5,
        "ASP" | "D" => -3.5,
        "ASN" | "N" => -3.5,
        "LYS" | "K" => -3.9,
        "ARG" | "R" => -4.5,
        _ => 0.0, // Unknown
    };

    // Normalize from [-4.5, 4.5] to [0, 1]
    (raw + 4.5) / 9.0
}

// =============================================================================
// CLASSIFIED ATOM
// =============================================================================

/// Classified atom for exclusion computation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C, align(16))]
pub struct ClassifiedAtom {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub radius: f32,         // VdW radius + water probe
    pub hydrophobicity: f32, // 0.0 (hydrophilic) to 1.0 (hydrophobic)
    pub charge: f32,         // Partial atomic charge
    pub is_polar: u32,       // 1 if polar/charged, 0 otherwise
    pub residue_idx: u32,    // Residue index
}

impl ClassifiedAtom {
    /// Create from raw atom data
    pub fn new(
        x: f32,
        y: f32,
        z: f32,
        element: u8,
        charge: f32,
        residue_name: &str,
        residue_idx: usize,
    ) -> Self {
        let vdw = get_vdw_radius(element);
        let hydrophobicity = kyte_doolittle_normalized(residue_name);
        let is_polar = if charge.abs() > 0.3 || hydrophobicity < 0.4 {
            1
        } else {
            0
        };

        Self {
            x,
            y,
            z,
            radius: vdw + WATER_PROBE_RADIUS,
            hydrophobicity,
            charge,
            is_polar,
            residue_idx: residue_idx as u32,
        }
    }
}

// =============================================================================
// EXCLUSION GRID
// =============================================================================

/// 3D Grid for exclusion and polar fields
#[derive(Debug, Clone)]
pub struct ExclusionGrid {
    /// Grid dimensions
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,

    /// Grid origin (minimum corner)
    pub origin: [f32; 3],

    /// Grid spacing (Å)
    pub spacing: f32,

    /// Exclusion field values [0, 1] where 1 = fully excluded (no water)
    pub exclusion: Vec<f32>,

    /// Polar attraction field (positive = favorable for water)
    pub polar_field: Vec<f32>,

    /// Inferred water density (computed from exclusion + polar)
    pub water_density: Vec<f32>,

    /// Pocket probability (derived from exclusion gradient)
    pub pocket_probability: Vec<f32>,
}

impl ExclusionGrid {
    /// Create grid from bounding box
    pub fn new(min: [f32; 3], max: [f32; 3], spacing: f32, padding: f32) -> Self {
        let origin = [min[0] - padding, min[1] - padding, min[2] - padding];

        let extent = [
            max[0] - min[0] + 2.0 * padding,
            max[1] - min[1] + 2.0 * padding,
            max[2] - min[2] + 2.0 * padding,
        ];

        let nx = (extent[0] / spacing).ceil() as usize + 1;
        let ny = (extent[1] / spacing).ceil() as usize + 1;
        let nz = (extent[2] / spacing).ceil() as usize + 1;

        let total = nx * ny * nz;

        log::info!(
            "ExclusionGrid: {}×{}×{} = {} voxels ({:.1} MB)",
            nx,
            ny,
            nz,
            total,
            (total * 16) as f32 / 1e6 // 4 fields × 4 bytes
        );

        Self {
            nx,
            ny,
            nz,
            origin,
            spacing,
            exclusion: vec![0.0; total],
            polar_field: vec![0.0; total],
            water_density: vec![0.0; total],
            pocket_probability: vec![0.0; total],
        }
    }

    /// Create from atoms
    pub fn from_atoms(atoms: &[ClassifiedAtom], config: &NhsConfig) -> Self {
        let (min, max) = compute_bounding_box(atoms);
        Self::new(min, max, config.grid_spacing, config.grid_padding)
    }

    /// Get total number of voxels
    #[inline]
    pub fn total_voxels(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Get linear index from 3D coordinates
    #[inline]
    pub fn index(&self, ix: usize, iy: usize, iz: usize) -> usize {
        iz * (self.nx * self.ny) + iy * self.nx + ix
    }

    /// Get 3D coordinates from linear index
    #[inline]
    pub fn coords(&self, idx: usize) -> (usize, usize, usize) {
        let iz = idx / (self.nx * self.ny);
        let remainder = idx % (self.nx * self.ny);
        let iy = remainder / self.nx;
        let ix = remainder % self.nx;
        (ix, iy, iz)
    }

    /// Get world position of voxel center
    #[inline]
    pub fn voxel_position(&self, ix: usize, iy: usize, iz: usize) -> [f32; 3] {
        [
            self.origin[0] + (ix as f32 + 0.5) * self.spacing,
            self.origin[1] + (iy as f32 + 0.5) * self.spacing,
            self.origin[2] + (iz as f32 + 0.5) * self.spacing,
        ]
    }

    /// Check if position is inside grid
    pub fn contains(&self, pos: &[f32; 3]) -> bool {
        let max = [
            self.origin[0] + self.nx as f32 * self.spacing,
            self.origin[1] + self.ny as f32 * self.spacing,
            self.origin[2] + self.nz as f32 * self.spacing,
        ];

        pos[0] >= self.origin[0]
            && pos[0] < max[0]
            && pos[1] >= self.origin[1]
            && pos[1] < max[1]
            && pos[2] >= self.origin[2]
            && pos[2] < max[2]
    }

    /// Get voxel index for a position (None if outside grid)
    pub fn position_to_index(&self, pos: &[f32; 3]) -> Option<usize> {
        if !self.contains(pos) {
            return None;
        }

        let ix = ((pos[0] - self.origin[0]) / self.spacing) as usize;
        let iy = ((pos[1] - self.origin[1]) / self.spacing) as usize;
        let iz = ((pos[2] - self.origin[2]) / self.spacing) as usize;

        // Clamp to valid range
        let ix = ix.min(self.nx - 1);
        let iy = iy.min(self.ny - 1);
        let iz = iz.min(self.nz - 1);

        Some(self.index(ix, iy, iz))
    }
}

// =============================================================================
// EXCLUSION COMPUTER
// =============================================================================

/// Computes exclusion field from classified atoms
pub struct ExclusionComputer {
    config: NhsConfig,
}

impl ExclusionComputer {
    pub fn new(config: NhsConfig) -> Self {
        Self { config }
    }

    /// Classify raw atoms
    pub fn classify_atoms(
        &self,
        positions: &[f32],
        elements: &[u8],
        charges: &[f32],
        residue_names: &[String],
        atom_residues: &[usize],
    ) -> Vec<ClassifiedAtom> {
        let n_atoms = positions.len() / 3;
        let mut classified = Vec::with_capacity(n_atoms);

        for i in 0..n_atoms {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            let element = elements.get(i).copied().unwrap_or(6); // Default carbon
            let charge = charges.get(i).copied().unwrap_or(0.0);
            let res_idx = atom_residues.get(i).copied().unwrap_or(0);
            let res_name = residue_names
                .get(res_idx)
                .map(|s| s.as_str())
                .unwrap_or("UNK");

            classified.push(ClassifiedAtom::new(
                x, y, z, element, charge, res_name, res_idx,
            ));
        }

        classified
    }

    /// Compute exclusion field (CPU implementation)
    ///
    /// For GPU implementation, see cuda/nhs_exclusion.cu
    pub fn compute(&self, atoms: &[ClassifiedAtom], grid: &mut ExclusionGrid) -> Result<()> {
        let sigma_scale = self.config.exclusion_sigma_scale;
        let polar_scale = self.config.polar_attraction_scale;
        let n_voxels = grid.total_voxels();

        // Reset grids
        grid.exclusion.fill(0.0);
        grid.polar_field.fill(0.0);

        log::debug!(
            "Computing exclusion field: {} atoms, {} voxels",
            atoms.len(),
            n_voxels
        );

        // Process each voxel
        for iz in 0..grid.nz {
            for iy in 0..grid.ny {
                for ix in 0..grid.nx {
                    let pos = grid.voxel_position(ix, iy, iz);
                    let idx = grid.index(ix, iy, iz);

                    let mut exclusion = 0.0f32;
                    let mut polar = 0.0f32;

                    for atom in atoms {
                        let dx = pos[0] - atom.x;
                        let dy = pos[1] - atom.y;
                        let dz = pos[2] - atom.z;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        // Early exit for far atoms
                        let cutoff = atom.radius + 3.0 * atom.radius * sigma_scale;
                        if dist_sq > cutoff * cutoff {
                            continue;
                        }

                        let dist = dist_sq.sqrt();

                        // Soft Gaussian exclusion
                        let sigma = atom.radius * sigma_scale;
                        let d_surface = dist - atom.radius;
                        let gaussian = (-0.5 * (d_surface / sigma).powi(2)).exp();

                        // Weight by hydrophobicity
                        exclusion += gaussian * atom.hydrophobicity;

                        // Polar attraction (screened)
                        if atom.is_polar == 1 && dist > 0.1 {
                            let screening = (-dist / 4.0).exp();
                            polar += atom.charge.abs() * polar_scale * screening / dist;
                        }
                    }

                    grid.exclusion[idx] = exclusion.min(1.0);
                    grid.polar_field[idx] = polar;
                }
            }
        }

        // Compute water density from exclusion + polar
        self.compute_water_density(grid);

        // Compute pocket probability from exclusion gradient
        self.compute_pocket_probability(grid);

        Ok(())
    }

    /// Infer water density from exclusion field
    fn compute_water_density(&self, grid: &mut ExclusionGrid) {
        use crate::config::BULK_WATER_DENSITY;

        let polar_boost = self.config.polar_attraction_scale;

        for i in 0..grid.total_voxels() {
            let exclusion = grid.exclusion[i];
            let polar = grid.polar_field[i];

            // Accessible volume
            let accessible = (1.0 - exclusion).max(0.0);

            // Boltzmann-weighted density near polar atoms
            let polar_factor = 1.0 + polar * polar_boost * 0.1;

            grid.water_density[i] = BULK_WATER_DENSITY * accessible * polar_factor;
        }
    }

    /// Compute pocket probability from exclusion gradient
    ///
    /// High probability = regions where exclusion is intermediate (surface)
    /// and has high gradient (cavity edge)
    fn compute_pocket_probability(&self, grid: &mut ExclusionGrid) {
        for iz in 1..grid.nz - 1 {
            for iy in 1..grid.ny - 1 {
                for ix in 1..grid.nx - 1 {
                    let idx = grid.index(ix, iy, iz);
                    let e = grid.exclusion[idx];

                    // Surface indicator: exclusion between 0.2 and 0.8
                    let surface_score = if e > 0.2 && e < 0.8 {
                        1.0 - 2.0 * (e - 0.5).abs()
                    } else {
                        0.0
                    };

                    // Gradient magnitude
                    let grad_x = grid.exclusion[grid.index(ix + 1, iy, iz)]
                        - grid.exclusion[grid.index(ix - 1, iy, iz)];
                    let grad_y = grid.exclusion[grid.index(ix, iy + 1, iz)]
                        - grid.exclusion[grid.index(ix, iy - 1, iz)];
                    let grad_z = grid.exclusion[grid.index(ix, iy, iz + 1)]
                        - grid.exclusion[grid.index(ix, iy, iz - 1)];

                    let grad_mag = (grad_x * grad_x + grad_y * grad_y + grad_z * grad_z).sqrt();

                    // Pocket probability: surface × gradient
                    grid.pocket_probability[idx] = surface_score * grad_mag.min(1.0);
                }
            }
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Compute bounding box of atoms
pub fn compute_bounding_box(atoms: &[ClassifiedAtom]) -> ([f32; 3], [f32; 3]) {
    if atoms.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }

    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for atom in atoms {
        min[0] = min[0].min(atom.x);
        min[1] = min[1].min(atom.y);
        min[2] = min[2].min(atom.z);
        max[0] = max[0].max(atom.x);
        max[1] = max[1].max(atom.y);
        max[2] = max[2].max(atom.z);
    }

    (min, max)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdw_radii() {
        assert!((get_vdw_radius(6) - 1.70).abs() < 0.01); // Carbon
        assert!((get_vdw_radius(7) - 1.55).abs() < 0.01); // Nitrogen
        assert!((get_vdw_radius(8) - 1.52).abs() < 0.01); // Oxygen
        assert!((get_vdw_radius(1) - 1.20).abs() < 0.01); // Hydrogen
    }

    #[test]
    fn test_hydrophobicity_scale() {
        // Most hydrophobic
        assert!(kyte_doolittle_normalized("ILE") > 0.9);
        assert!(kyte_doolittle_normalized("VAL") > 0.9);

        // Most hydrophilic
        assert!(kyte_doolittle_normalized("ARG") < 0.1);
        assert!(kyte_doolittle_normalized("LYS") < 0.1);

        // Middle
        let gly = kyte_doolittle_normalized("GLY");
        assert!(gly > 0.3 && gly < 0.6);
    }

    #[test]
    fn test_grid_indexing() {
        let grid = ExclusionGrid::new([0.0; 3], [10.0; 3], 0.5, 2.0);

        // Test round-trip
        for idx in [0, 100, 1000, grid.total_voxels() - 1] {
            let (ix, iy, iz) = grid.coords(idx);
            assert_eq!(grid.index(ix, iy, iz), idx);
        }
    }

    #[test]
    fn test_position_to_index() {
        let grid = ExclusionGrid::new([0.0; 3], [10.0; 3], 1.0, 0.0);

        // Center should map to valid index
        let idx = grid.position_to_index(&[5.0, 5.0, 5.0]);
        assert!(idx.is_some());

        // Outside should return None
        let idx = grid.position_to_index(&[-5.0, 5.0, 5.0]);
        assert!(idx.is_none());
    }
}
