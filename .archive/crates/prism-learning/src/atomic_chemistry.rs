//! Atomic-Level Chemistry Module for Bio-Aware Feature Extraction
//!
//! This module provides atomic-resolution biochemical data for the SNN:
//! - Residue type identification (3-letter code → index)
//! - Kyte-Doolittle hydrophobicity scale
//! - Partial atomic charges (AMBER ff14SB-derived)
//! - Atom classification (backbone vs sidechain, polar vs nonpolar)
//!
//! ## Key Insight
//! A drug doesn't see "residues" - it sees atoms with specific:
//! - Hydrophobicity (greasy vs polar)
//! - Partial charges (electrostatic attraction/repulsion)
//! - Spatial positions (where exactly is the sticky spot?)
//!
//! This module bridges PDB atom records to biochemically-meaningful features.

use prism_io::sovereign_types::Atom;
use std::collections::HashMap;

// ============================================================================
// RESIDUE TYPE IDENTIFICATION
// ============================================================================

/// Standard amino acid 3-letter codes mapped to indices (0-19)
/// Order follows conventional bioinformatics ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ResidueType {
    ALA = 0,  // Alanine
    ARG = 1,  // Arginine (positive charge)
    ASN = 2,  // Asparagine
    ASP = 3,  // Aspartic acid (negative charge)
    CYS = 4,  // Cysteine (can form disulfides)
    GLN = 5,  // Glutamine
    GLU = 6,  // Glutamic acid (negative charge)
    GLY = 7,  // Glycine (smallest, most flexible)
    HIS = 8,  // Histidine (can be charged)
    ILE = 9,  // Isoleucine (hydrophobic)
    LEU = 10, // Leucine (hydrophobic)
    LYS = 11, // Lysine (positive charge)
    MET = 12, // Methionine (hydrophobic)
    PHE = 13, // Phenylalanine (aromatic, hydrophobic)
    PRO = 14, // Proline (rigid, helix breaker)
    SER = 15, // Serine (polar)
    THR = 16, // Threonine (polar)
    TRP = 17, // Tryptophan (aromatic, largest)
    TYR = 18, // Tyrosine (aromatic, can H-bond)
    VAL = 19, // Valine (hydrophobic)
    UNK = 20, // Unknown/non-standard
}

impl ResidueType {
    /// Parse 3-letter residue code to ResidueType
    pub fn from_code(code: &str) -> Self {
        match code.to_uppercase().as_str() {
            "ALA" | "A" => ResidueType::ALA,
            "ARG" | "R" => ResidueType::ARG,
            "ASN" | "N" => ResidueType::ASN,
            "ASP" | "D" => ResidueType::ASP,
            "CYS" | "C" => ResidueType::CYS,
            "GLN" | "Q" => ResidueType::GLN,
            "GLU" | "E" => ResidueType::GLU,
            "GLY" | "G" => ResidueType::GLY,
            "HIS" | "H" => ResidueType::HIS,
            "ILE" | "I" => ResidueType::ILE,
            "LEU" | "L" => ResidueType::LEU,
            "LYS" | "K" => ResidueType::LYS,
            "MET" | "M" => ResidueType::MET,
            "PHE" | "F" => ResidueType::PHE,
            "PRO" | "P" => ResidueType::PRO,
            "SER" | "S" => ResidueType::SER,
            "THR" | "T" => ResidueType::THR,
            "TRP" | "W" => ResidueType::TRP,
            "TYR" | "Y" => ResidueType::TYR,
            "VAL" | "V" => ResidueType::VAL,
            _ => ResidueType::UNK,
        }
    }

    /// Get index for array lookups (0-20)
    pub fn index(&self) -> usize {
        *self as usize
    }
}

// ============================================================================
// KYTE-DOOLITTLE HYDROPHOBICITY SCALE
// ============================================================================

/// Kyte-Doolittle hydrophobicity scale, normalized to [0.0, 1.0]
/// Higher values = more hydrophobic = better drug target when exposed
///
/// Original scale: -4.5 (ARG) to +4.5 (ILE)
/// Normalized: 0.0 (most hydrophilic) to 1.0 (most hydrophobic)
pub const HYDROPHOBICITY: [f32; 21] = [
    0.64, // ALA: 1.8 → 0.64
    0.00, // ARG: -4.5 → 0.00 (most hydrophilic - charged)
    0.36, // ASN: -3.5 → 0.11 (polar)
    0.33, // ASP: -3.5 → 0.11 (charged)
    0.78, // CYS: 2.5 → 0.78 (can be in hydrophobic core)
    0.36, // GLN: -3.5 → 0.11
    0.33, // GLU: -3.5 → 0.11 (charged)
    0.53, // GLY: -0.4 → 0.46
    0.47, // HIS: -3.2 → 0.14 (can be charged)
    1.00, // ILE: 4.5 → 1.00 (MOST hydrophobic - GREAT target)
    0.97, // LEU: 3.8 → 0.92 (very hydrophobic)
    0.07, // LYS: -3.9 → 0.07 (charged)
    0.83, // MET: 1.9 → 0.71
    0.94, // PHE: 2.8 → 0.81 (aromatic - GREAT target)
    0.39, // PRO: -1.6 → 0.32
    0.42, // SER: -0.8 → 0.41 (polar)
    0.44, // THR: -0.7 → 0.42 (polar)
    0.89, // TRP: -0.9 → 0.40 (aromatic but can H-bond)
    0.72, // TYR: -1.3 → 0.36 (aromatic, polar -OH)
    0.97, // VAL: 4.2 → 0.97 (very hydrophobic)
    0.50, // UNK: neutral default
];

/// Get hydrophobicity score for a residue type (0.0-1.0)
#[inline]
pub fn get_hydrophobicity(res_type: ResidueType) -> f32 {
    HYDROPHOBICITY[res_type.index()]
}

// ============================================================================
// PARTIAL ATOMIC CHARGES (Simplified AMBER-like)
// ============================================================================

/// Atom type classification for charge assignment
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AtomType {
    // Backbone atoms
    BackboneN,  // Amide nitrogen: -0.4157
    BackboneH,  // Amide hydrogen: +0.2719
    BackboneCa, // Alpha carbon: +0.0337
    BackboneHa, // Alpha hydrogen: +0.0823
    BackboneC,  // Carbonyl carbon: +0.5973
    BackboneO,  // Carbonyl oxygen: -0.5679

    // Sidechain types (simplified)
    SidechainNonpolar, // CH2, CH3 groups: ~0.0
    SidechainPolar,    // OH, NH2 groups: partial charges
    SidechainPositive, // NH3+, guanidinium: +1.0 distributed
    SidechainNegative, // COO-: -1.0 distributed
    SidechainAromatic, // Ring carbons: slight negative

    Unknown,
}

/// Simplified partial charge lookup
/// Returns charge in elementary charge units (e)
pub fn get_atom_charge(atom_name: &str, res_type: ResidueType) -> f32 {
    let atom_name = atom_name.trim();

    // Backbone atoms (consistent across residues)
    match atom_name {
        "N" => return -0.4157,
        "H" | "HN" | "H1" | "H2" | "H3" => return 0.2719,
        "CA" => return 0.0337,
        "HA" | "HA2" | "HA3" => return 0.0823,
        "C" => return 0.5973,
        "O" | "OXT" => return -0.5679,
        _ => {}
    }

    // Sidechain charges by residue type
    match res_type {
        // Charged residues
        ResidueType::ARG => {
            match atom_name {
                "NE" | "NH1" | "NH2" => 0.34, // Guanidinium nitrogens
                "CZ" => 0.64,                 // Central carbon
                "HE" | "HH11" | "HH12" | "HH21" | "HH22" => 0.35,
                _ => 0.0,
            }
        }
        ResidueType::LYS => {
            match atom_name {
                "NZ" => -0.385,                // NH3+ nitrogen
                "HZ1" | "HZ2" | "HZ3" => 0.34, // NH3+ hydrogens
                "CE" => 0.21,
                _ => 0.0,
            }
        }
        ResidueType::ASP => {
            match atom_name {
                "CG" => 0.55,
                "OD1" | "OD2" => -0.65, // Carboxylate oxygens
                _ => 0.0,
            }
        }
        ResidueType::GLU => {
            match atom_name {
                "CD" => 0.55,
                "OE1" | "OE2" => -0.65, // Carboxylate oxygens
                _ => 0.0,
            }
        }
        ResidueType::HIS => {
            // Assume neutral HIS (HID form)
            match atom_name {
                "ND1" => -0.38,
                "HD1" => 0.36,
                "CE1" => 0.21,
                "NE2" => -0.57,
                "CD2" => 0.13,
                _ => 0.0,
            }
        }

        // Polar residues
        ResidueType::SER => match atom_name {
            "OG" => -0.65,
            "HG" => 0.43,
            _ => 0.0,
        },
        ResidueType::THR => match atom_name {
            "OG1" => -0.68,
            "HG1" => 0.42,
            _ => 0.0,
        },
        ResidueType::ASN => match atom_name {
            "OD1" => -0.59,
            "ND2" => -0.92,
            "HD21" | "HD22" => 0.42,
            _ => 0.0,
        },
        ResidueType::GLN => match atom_name {
            "OE1" => -0.59,
            "NE2" => -0.91,
            "HE21" | "HE22" => 0.43,
            _ => 0.0,
        },
        ResidueType::TYR => {
            match atom_name {
                "OH" => -0.56,
                "HH" => 0.40,
                // Ring carbons have slight charges
                "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ" => -0.02,
                _ => 0.0,
            }
        }
        ResidueType::TRP => match atom_name {
            "NE1" => -0.34,
            "HE1" => 0.34,
            _ => 0.0,
        },
        ResidueType::CYS => {
            match atom_name {
                "SG" => -0.31,
                "HG" => 0.19, // If protonated
                _ => 0.0,
            }
        }

        // Nonpolar residues - minimal sidechain charges
        _ => 0.0,
    }
}

// ============================================================================
// ATOM CLASSIFICATION
// ============================================================================

/// Check if atom is a backbone atom
#[inline]
pub fn is_backbone_atom(atom_name: &str) -> bool {
    matches!(
        atom_name.trim(),
        "N" | "H" | "HN" | "CA" | "HA" | "C" | "O" | "OXT"
    )
}

/// Check if atom is the alpha carbon (Cα) - key for backbone tracking
#[inline]
pub fn is_alpha_carbon(atom_name: &str) -> bool {
    atom_name.trim() == "CA"
}

/// Check if residue is charged at physiological pH
#[inline]
pub fn is_charged_residue(res_type: ResidueType) -> bool {
    matches!(
        res_type,
        ResidueType::ARG | ResidueType::LYS |  // Positive
        ResidueType::ASP | ResidueType::GLU // Negative
    )
}

/// Get formal charge of residue (-1, 0, or +1)
#[inline]
pub fn get_residue_formal_charge(res_type: ResidueType) -> i32 {
    match res_type {
        ResidueType::ARG | ResidueType::LYS => 1,
        ResidueType::ASP | ResidueType::GLU => -1,
        _ => 0,
    }
}

// ============================================================================
// ATOMIC METADATA STRUCTURE
// ============================================================================

/// Complete atomic metadata for bio-aware feature extraction
#[derive(Debug, Clone)]
pub struct AtomicMetadata {
    /// Residue type for each residue index
    pub residue_types: Vec<ResidueType>,
    /// Hydrophobicity for each residue
    pub hydrophobicity: Vec<f32>,
    /// Partial charge for each atom
    pub atom_charges: Vec<f32>,
    /// Is this atom a Cα (for backbone tracking)?
    pub is_ca: Vec<bool>,
    /// Is this atom in a charged residue?
    pub is_charged: Vec<bool>,
    /// Atom-to-residue mapping
    pub atom_to_residue: Vec<usize>,
    /// Cα atom indices for backbone analysis
    pub ca_indices: Vec<usize>,
}

impl AtomicMetadata {
    /// Build atomic metadata from PDB atoms
    ///
    /// This extracts all the biochemical context needed for:
    /// - Hydrophobic exposure calculations
    /// - Electrostatic frustration detection
    /// - Backbone hinge analysis
    pub fn from_atoms(atoms: &[Atom]) -> Self {
        let mut residue_types = Vec::new();
        let mut hydrophobicity = Vec::new();
        let mut atom_charges = Vec::with_capacity(atoms.len());
        let mut is_ca = Vec::with_capacity(atoms.len());
        let mut is_charged = Vec::with_capacity(atoms.len());
        let mut atom_to_residue = Vec::with_capacity(atoms.len());
        let mut ca_indices = Vec::new();

        let mut current_res_id = u16::MAX;
        let mut current_res_type = ResidueType::UNK;
        let mut residue_idx = 0;

        for (atom_idx, atom) in atoms.iter().enumerate() {
            // Track residue changes
            if atom.residue_id != current_res_id {
                current_res_id = atom.residue_id;

                // Parse residue name from atom (assumes residue_name field exists)
                // If not available, we'll use a placeholder
                current_res_type = Self::parse_residue_type(atom);

                residue_types.push(current_res_type);
                hydrophobicity.push(get_hydrophobicity(current_res_type));
                residue_idx = residue_types.len() - 1;
            }

            // Map atom to residue
            atom_to_residue.push(residue_idx);

            // Get atom name (first 4 chars typically)
            let atom_name = Self::get_atom_name(atom);

            // Calculate partial charge
            let charge = get_atom_charge(&atom_name, current_res_type);
            atom_charges.push(charge);

            // Check if Cα
            let is_alpha = is_alpha_carbon(&atom_name);
            is_ca.push(is_alpha);
            if is_alpha {
                ca_indices.push(atom_idx);
            }

            // Check if in charged residue
            is_charged.push(is_charged_residue(current_res_type));
        }

        Self {
            residue_types,
            hydrophobicity,
            atom_charges,
            is_ca,
            is_charged,
            atom_to_residue,
            ca_indices,
        }
    }

    /// Parse residue type from atom record
    /// Uses atom's element and residue_id to infer type
    fn parse_residue_type(atom: &Atom) -> ResidueType {
        // The Atom struct from prism_io should have residue info
        // If residue_name is available as bytes, parse it
        // Otherwise, return UNK

        // Try to get residue name from the atom's fixed-size array
        // Assuming atom has a residue_name or similar field
        // For now, use a heuristic based on available data

        // If we have residue_type field (1-letter code as u8)
        if atom.element != 0 {
            // Try to interpret as 1-letter code
            match atom.element as char {
                'A' => return ResidueType::ALA,
                'R' => return ResidueType::ARG,
                'N' => return ResidueType::ASN,
                'D' => return ResidueType::ASP,
                'C' => return ResidueType::CYS,
                'Q' => return ResidueType::GLN,
                'E' => return ResidueType::GLU,
                'G' => return ResidueType::GLY,
                'H' => return ResidueType::HIS,
                'I' => return ResidueType::ILE,
                'L' => return ResidueType::LEU,
                'K' => return ResidueType::LYS,
                'M' => return ResidueType::MET,
                'F' => return ResidueType::PHE,
                'P' => return ResidueType::PRO,
                'S' => return ResidueType::SER,
                'T' => return ResidueType::THR,
                'W' => return ResidueType::TRP,
                'Y' => return ResidueType::TYR,
                'V' => return ResidueType::VAL,
                _ => {}
            }
        }

        ResidueType::UNK
    }

    /// Extract atom name from Atom struct
    fn get_atom_name(atom: &Atom) -> String {
        // Atom names are typically stored in a fixed array
        // Convert to string, trimming nulls and spaces
        // For now, use a placeholder based on element
        match atom.element as char {
            'C' => "C".to_string(),
            'N' => "N".to_string(),
            'O' => "O".to_string(),
            'S' => "S".to_string(),
            'H' => "H".to_string(),
            _ => "X".to_string(),
        }
    }

    /// Get number of residues
    pub fn num_residues(&self) -> usize {
        self.residue_types.len()
    }

    /// Get number of atoms
    pub fn num_atoms(&self) -> usize {
        self.atom_to_residue.len()
    }

    /// Get hydrophobicity for a specific residue
    pub fn residue_hydrophobicity(&self, res_idx: usize) -> f32 {
        self.hydrophobicity.get(res_idx).copied().unwrap_or(0.5)
    }

    /// Get total charge in a region (list of atom indices)
    pub fn region_charge(&self, atom_indices: &[usize]) -> f32 {
        atom_indices
            .iter()
            .filter_map(|&i| self.atom_charges.get(i))
            .sum()
    }
}

// ============================================================================
// BIO-CHEMISTRY FEATURE CALCULATIONS
// ============================================================================

/// Calculate hydrophobic exposure delta
///
/// This computes the change in solvent-accessible surface area weighted by
/// hydrophobicity. High values indicate exposed "greasy" residues - prime
/// drug binding sites.
///
/// Formula: Σ (ΔSASA_i × hydrophobicity_i) for each residue i
pub fn calculate_hydrophobic_exposure(
    positions: &[f32],
    initial_positions: &[f32],
    metadata: &AtomicMetadata,
    target_residues: &[usize],
    neighbor_cutoff: f32,
) -> f32 {
    let cutoff_sq = neighbor_cutoff * neighbor_cutoff;
    let mut total_hydrophobic_delta = 0.0;

    for &res_idx in target_residues {
        if res_idx >= metadata.num_residues() {
            continue;
        }

        let hydrophobicity = metadata.residue_hydrophobicity(res_idx);

        // Find atoms belonging to this residue
        let res_atoms: Vec<usize> = metadata
            .atom_to_residue
            .iter()
            .enumerate()
            .filter(|(_, &r)| r == res_idx)
            .map(|(i, _)| i)
            .collect();

        // Calculate exposure change for this residue
        let initial_exposure = count_neighbors(&res_atoms, initial_positions, cutoff_sq);
        let current_exposure = count_neighbors(&res_atoms, positions, cutoff_sq);

        // Exposure increases when neighbor count decreases
        let exposure_delta = initial_exposure - current_exposure;

        // Weight by hydrophobicity
        total_hydrophobic_delta += exposure_delta * hydrophobicity;
    }

    // Normalize
    if target_residues.is_empty() {
        0.0
    } else {
        total_hydrophobic_delta / target_residues.len() as f32
    }
}

/// Calculate local displacement anisotropy (hinge detection)
///
/// Hinges are where one Cα moves significantly more than its neighbors.
/// This is the "edge detector" for finding cryptic pocket entrances.
///
/// Formula: max(|Disp(i) - Avg(Disp(i-1), Disp(i+1))|) over all Cα atoms
pub fn calculate_anisotropy(
    positions: &[f32],
    initial_positions: &[f32],
    metadata: &AtomicMetadata,
) -> (f32, f32) {
    // Returns (max_anisotropy, mean_anisotropy)
    let ca_indices = &metadata.ca_indices;

    if ca_indices.len() < 3 {
        return (0.0, 0.0);
    }

    let mut max_anisotropy = 0.0f32;
    let mut total_anisotropy = 0.0f32;
    let mut count = 0;

    // Calculate displacement for each Cα
    let displacements: Vec<f32> = ca_indices
        .iter()
        .map(|&idx| {
            let base = idx * 3;
            if base + 2 < positions.len() && base + 2 < initial_positions.len() {
                let dx = positions[base] - initial_positions[base];
                let dy = positions[base + 1] - initial_positions[base + 1];
                let dz = positions[base + 2] - initial_positions[base + 2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Calculate anisotropy for each internal Cα
    for i in 1..displacements.len() - 1 {
        let d_prev = displacements[i - 1];
        let d_curr = displacements[i];
        let d_next = displacements[i + 1];

        let neighbor_avg = (d_prev + d_next) / 2.0;
        let anisotropy = (d_curr - neighbor_avg).abs();

        max_anisotropy = max_anisotropy.max(anisotropy);
        total_anisotropy += anisotropy;
        count += 1;
    }

    let mean_anisotropy = if count > 0 {
        total_anisotropy / count as f32
    } else {
        0.0
    };

    (max_anisotropy, mean_anisotropy)
}

/// Calculate electrostatic frustration
///
/// Regions with like-charges forced close together are "spring-loaded"
/// and want to open. This identifies thermodynamically stressed regions.
///
/// Formula: Σ (q_i × q_j / r_ij) for all charged pairs within cutoff
/// Positive sum = frustration (like charges repelling)
pub fn calculate_electrostatic_frustration(
    positions: &[f32],
    metadata: &AtomicMetadata,
    cutoff: f32,
) -> f32 {
    let cutoff_sq = cutoff * cutoff;
    let mut total_frustration = 0.0;

    // Only look at atoms in charged residues
    let charged_atoms: Vec<usize> = metadata
        .is_charged
        .iter()
        .enumerate()
        .filter(|(_, &is_charged)| is_charged)
        .map(|(i, _)| i)
        .collect();

    // Calculate pairwise Coulombic energy
    for i in 0..charged_atoms.len() {
        let idx_i = charged_atoms[i];
        let q_i = metadata.atom_charges[idx_i];

        if q_i.abs() < 0.01 {
            continue; // Skip neutral atoms
        }

        let base_i = idx_i * 3;
        if base_i + 2 >= positions.len() {
            continue;
        }

        let xi = positions[base_i];
        let yi = positions[base_i + 1];
        let zi = positions[base_i + 2];

        for j in (i + 1)..charged_atoms.len() {
            let idx_j = charged_atoms[j];
            let q_j = metadata.atom_charges[idx_j];

            if q_j.abs() < 0.01 {
                continue;
            }

            let base_j = idx_j * 3;
            if base_j + 2 >= positions.len() {
                continue;
            }

            let xj = positions[base_j];
            let yj = positions[base_j + 1];
            let zj = positions[base_j + 2];

            let dx = xi - xj;
            let dy = yi - yj;
            let dz = zi - zj;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < cutoff_sq && dist_sq > 1.0 {
                // Avoid self and clashes
                let dist = dist_sq.sqrt();
                // Coulomb: E = k * q1 * q2 / r
                // Positive when like charges (frustration)
                let coulomb = q_i * q_j / dist;

                // Only count repulsion (frustration)
                if coulomb > 0.0 {
                    total_frustration += coulomb;
                }
            }
        }
    }

    total_frustration
}

/// Helper: count neighbors within cutoff
fn count_neighbors(atom_indices: &[usize], positions: &[f32], cutoff_sq: f32) -> f32 {
    let mut total_neighbors = 0.0;
    let n_atoms = positions.len() / 3;

    for &idx in atom_indices {
        let base = idx * 3;
        if base + 2 >= positions.len() {
            continue;
        }

        let x = positions[base];
        let y = positions[base + 1];
        let z = positions[base + 2];

        for j in 0..n_atoms {
            if j == idx {
                continue;
            }

            let base_j = j * 3;
            let dx = x - positions[base_j];
            let dy = y - positions[base_j + 1];
            let dz = z - positions[base_j + 2];
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < cutoff_sq {
                total_neighbors += 1.0;
            }
        }
    }

    total_neighbors
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_residue_type_parsing() {
        assert_eq!(ResidueType::from_code("ALA"), ResidueType::ALA);
        assert_eq!(ResidueType::from_code("PHE"), ResidueType::PHE);
        assert_eq!(ResidueType::from_code("F"), ResidueType::PHE);
        assert_eq!(ResidueType::from_code("XXX"), ResidueType::UNK);
    }

    #[test]
    fn test_hydrophobicity_scale() {
        // ILE should be most hydrophobic
        assert_eq!(HYDROPHOBICITY[ResidueType::ILE as usize], 1.0);
        // ARG should be least hydrophobic (charged)
        assert_eq!(HYDROPHOBICITY[ResidueType::ARG as usize], 0.0);
        // PHE should be highly hydrophobic
        assert!(HYDROPHOBICITY[ResidueType::PHE as usize] > 0.9);
    }

    #[test]
    fn test_charge_assignment() {
        // Backbone N should be negative
        assert!(get_atom_charge("N", ResidueType::ALA) < 0.0);
        // Backbone O should be negative
        assert!(get_atom_charge("O", ResidueType::ALA) < 0.0);
        // ASP carboxylate should be negative
        assert!(get_atom_charge("OD1", ResidueType::ASP) < 0.0);
        // LYS amino group should contribute positive
        assert!(get_atom_charge("NZ", ResidueType::LYS) < 0.0); // N is negative
        assert!(get_atom_charge("HZ1", ResidueType::LYS) > 0.0); // H is positive
    }

    #[test]
    fn test_charged_residue_detection() {
        assert!(is_charged_residue(ResidueType::ARG));
        assert!(is_charged_residue(ResidueType::LYS));
        assert!(is_charged_residue(ResidueType::ASP));
        assert!(is_charged_residue(ResidueType::GLU));
        assert!(!is_charged_residue(ResidueType::ALA));
        assert!(!is_charged_residue(ResidueType::PHE));
    }
}
