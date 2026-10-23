//! AMBER ff14SB Force Field Implementation
//!
//! Complete implementation of the AMBER ff14SB force field for protein simulations.
//! Includes bond, angle, dihedral parameters and automatic topology generation.
//!
//! Reference: Maier et al. (2015) JCTC - "ff14SB: Improving the Accuracy of
//! Protein Side Chain and Backbone Parameters from ff99SB"

use std::collections::HashMap;
use std::f32::consts::PI;

// ============================================================================
// AMBER ATOM TYPES
// ============================================================================

/// AMBER ff14SB atom types for proteins
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AmberAtomType {
    // Backbone atoms
    N = 0,  // sp2 nitrogen in amide groups
    H = 1,  // H bonded to nitrogen atoms
    CT = 2, // sp3 aliphatic carbon
    H1 = 3, // H aliph. bonded to C with 1 electrwd. group
    HC = 4, // H aliph. bonded to C without electrwd. group
    C = 5,  // sp2 carbonyl carbon
    O = 6,  // carbonyl oxygen

    // Charged/polar sidechains
    O2 = 7, // carboxyl oxygen (ASP, GLU)
    N2 = 8, // sp2 N in guanidinium (ARG)
    N3 = 9, // sp3 N in amino groups (LYS N-terminus)

    // Hydroxyl groups
    OH = 10, // oxygen in hydroxyl group
    HO = 11, // hydrogen in hydroxyl group

    // Sulfur
    S = 12,  // sulfur in disulfide
    SH = 13, // sulfur in thiol (CYS)
    HS = 14, // hydrogen on thiol sulfur

    // Aromatic
    CA = 15, // sp2 aromatic carbon
    HA = 16, // H bonded to aromatic carbon

    // Special
    HP = 17, // H bonded to N in charged amino (LYS, ARG)
    CW = 18, // sp2 C in 5-ring of TRP, next to N
    NA = 19, // sp2 N in HIS with H
    NB = 20, // sp2 N in aromatic 5-ring, no H
    CC = 21, // sp2 C in aromatic 5-ring (HIS)
    CR = 22, // sp2 aromatic C between 2 N (HIS)
    CV = 23, // sp2 aromatic C 5-ring (TRP)
    CN = 24, // sp2 junction C in TRP
    CB = 25, // sp2 aromatic C in PHE, TYR, fused ring of TRP

    // Water atoms (TIP3P)
    OW = 26, // TIP3P water oxygen
    HW = 27, // TIP3P water hydrogen

    // Ions (Joung & Cheatham parameters)
    IP = 28, // Sodium cation (Na+)
    IM = 29, // Chloride anion (Cl-)

    Unknown = 255,
}

impl AmberAtomType {
    /// Get AMBER atom type from residue name and atom name
    pub fn from_pdb(residue: &str, atom_name: &str) -> Self {
        let atom = atom_name.trim();
        let res = residue.to_uppercase();

        // Backbone atoms (same across most residues)
        match atom {
            "N" => return AmberAtomType::N,
            "H" | "HN" | "H1" | "H2" | "H3" => {
                // N-terminal has H1, H2, H3
                if atom.starts_with("H")
                    && atom.len() == 2
                    && atom.chars().nth(1).map_or(false, |c| c.is_numeric())
                {
                    return AmberAtomType::HP; // Charged amino H
                }
                return AmberAtomType::H;
            }
            "CA" => return AmberAtomType::CT,
            "HA" | "HA2" | "HA3" => return AmberAtomType::H1,
            "C" => return AmberAtomType::C,
            "O" | "OXT" => return AmberAtomType::O,
            _ => {}
        }

        // Residue-specific sidechain atoms
        match res.as_str() {
            "GLY" => match atom {
                _ => AmberAtomType::H1, // GLY only has HA2, HA3
            },

            "ALA" => match atom {
                "CB" => AmberAtomType::CT,
                "HB1" | "HB2" | "HB3" => AmberAtomType::HC,
                _ => AmberAtomType::CT,
            },

            "VAL" => match atom {
                "CB" => AmberAtomType::CT,
                "HB" => AmberAtomType::HC,
                "CG1" | "CG2" => AmberAtomType::CT,
                "HG11" | "HG12" | "HG13" | "HG21" | "HG22" | "HG23" => AmberAtomType::HC,
                _ => AmberAtomType::CT,
            },

            "LEU" => match atom {
                "CB" | "CG" | "CD1" | "CD2" => AmberAtomType::CT,
                "HB2" | "HB3" | "HG" => AmberAtomType::HC,
                "HD11" | "HD12" | "HD13" | "HD21" | "HD22" | "HD23" => AmberAtomType::HC,
                _ => AmberAtomType::CT,
            },

            "ILE" => match atom {
                "CB" | "CG1" | "CG2" | "CD1" => AmberAtomType::CT,
                "HB" | "HG12" | "HG13" | "HG21" | "HG22" | "HG23" => AmberAtomType::HC,
                "HD11" | "HD12" | "HD13" => AmberAtomType::HC,
                _ => AmberAtomType::CT,
            },

            "PRO" => match atom {
                "CB" | "CG" | "CD" => AmberAtomType::CT,
                "HB2" | "HB3" | "HG2" | "HG3" | "HD2" | "HD3" => AmberAtomType::HC,
                _ => AmberAtomType::CT,
            },

            "PHE" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::HC,
                "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ" => AmberAtomType::CA,
                "HD1" | "HD2" | "HE1" | "HE2" | "HZ" => AmberAtomType::HA,
                _ => AmberAtomType::CA,
            },

            "TYR" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::HC,
                "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ" => AmberAtomType::CA,
                "HD1" | "HD2" | "HE1" | "HE2" => AmberAtomType::HA,
                "OH" => AmberAtomType::OH,
                "HH" => AmberAtomType::HO,
                _ => AmberAtomType::CA,
            },

            "TRP" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::HC,
                "CG" => AmberAtomType::CB, // junction carbon
                "CD1" => AmberAtomType::CW,
                "HD1" => AmberAtomType::H,
                "NE1" => AmberAtomType::NA,
                "HE1" => AmberAtomType::H,
                "CE2" => AmberAtomType::CN,
                "CE3" | "CZ2" | "CZ3" | "CH2" => AmberAtomType::CA,
                "CD2" => AmberAtomType::CB,
                "HE3" | "HZ2" | "HZ3" | "HH2" => AmberAtomType::HA,
                _ => AmberAtomType::CA,
            },

            "SER" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::H1,
                "OG" => AmberAtomType::OH,
                "HG" => AmberAtomType::HO,
                _ => AmberAtomType::CT,
            },

            "THR" => match atom {
                "CB" => AmberAtomType::CT,
                "HB" => AmberAtomType::H1,
                "OG1" => AmberAtomType::OH,
                "HG1" => AmberAtomType::HO,
                "CG2" => AmberAtomType::CT,
                "HG21" | "HG22" | "HG23" => AmberAtomType::HC,
                _ => AmberAtomType::CT,
            },

            "CYS" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::H1,
                "SG" => AmberAtomType::SH,
                "HG" => AmberAtomType::HS,
                _ => AmberAtomType::CT,
            },

            // Cystine (disulfide bonded)
            "CYX" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::H1,
                "SG" => AmberAtomType::S,
                _ => AmberAtomType::CT,
            },

            "MET" => match atom {
                "CB" | "CG" | "CE" => AmberAtomType::CT,
                "HB2" | "HB3" | "HG2" | "HG3" => AmberAtomType::H1,
                "SD" => AmberAtomType::S,
                "HE1" | "HE2" | "HE3" => AmberAtomType::H1,
                _ => AmberAtomType::CT,
            },

            "ASN" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::HC,
                "CG" => AmberAtomType::C,
                "OD1" => AmberAtomType::O,
                "ND2" => AmberAtomType::N,
                "HD21" | "HD22" => AmberAtomType::H,
                _ => AmberAtomType::CT,
            },

            "GLN" => match atom {
                "CB" | "CG" => AmberAtomType::CT,
                "HB2" | "HB3" | "HG2" | "HG3" => AmberAtomType::HC,
                "CD" => AmberAtomType::C,
                "OE1" => AmberAtomType::O,
                "NE2" => AmberAtomType::N,
                "HE21" | "HE22" => AmberAtomType::H,
                _ => AmberAtomType::CT,
            },

            "ASP" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::HC,
                "CG" => AmberAtomType::C,
                "OD1" | "OD2" => AmberAtomType::O2,
                _ => AmberAtomType::CT,
            },

            "GLU" => match atom {
                "CB" | "CG" => AmberAtomType::CT,
                "HB2" | "HB3" | "HG2" | "HG3" => AmberAtomType::HC,
                "CD" => AmberAtomType::C,
                "OE1" | "OE2" => AmberAtomType::O2,
                _ => AmberAtomType::CT,
            },

            "LYS" => match atom {
                "CB" | "CG" | "CD" | "CE" => AmberAtomType::CT,
                "HB2" | "HB3" | "HG2" | "HG3" | "HD2" | "HD3" | "HE2" | "HE3" => AmberAtomType::HC,
                "NZ" => AmberAtomType::N3,
                "HZ1" | "HZ2" | "HZ3" => AmberAtomType::HP,
                _ => AmberAtomType::CT,
            },

            "ARG" => match atom {
                "CB" | "CG" | "CD" => AmberAtomType::CT,
                "HB2" | "HB3" | "HG2" | "HG3" | "HD2" | "HD3" => AmberAtomType::HC,
                "NE" => AmberAtomType::N2,
                "HE" => AmberAtomType::H,
                "CZ" => AmberAtomType::CA,
                "NH1" | "NH2" => AmberAtomType::N2,
                "HH11" | "HH12" | "HH21" | "HH22" => AmberAtomType::H,
                _ => {
                    log::warn!("⚠️ UNKNOWN ATOM in ARG: '{}' → defaulting to CT", atom);
                    AmberAtomType::CT
                }
            },

            "HIS" | "HID" | "HIE" | "HIP" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::HC,
                "CG" => AmberAtomType::CC,
                "ND1" => AmberAtomType::NA,
                "HD1" => AmberAtomType::H,
                "CE1" => AmberAtomType::CR,
                "HE1" => AmberAtomType::H,
                "NE2" => AmberAtomType::NB,
                "HE2" => AmberAtomType::H,
                "CD2" => AmberAtomType::CV,
                "HD2" => AmberAtomType::H,
                _ => {
                    log::warn!("⚠️ UNKNOWN ATOM in HIS: '{}' → defaulting to CC", atom);
                    AmberAtomType::CC
                }
            },

            // Water residues (TIP3P)
            "WAT" | "HOH" | "SOL" | "TIP3" | "TIP3P" => match atom {
                "O" | "OW" | "OH2" => AmberAtomType::OW,
                "H1" | "H2" | "HW1" | "HW2" => AmberAtomType::HW,
                _ => AmberAtomType::OW,
            },

            // Ions
            "NA" | "NA+" | "SOD" => AmberAtomType::IP,
            "CL" | "CL-" | "CLA" => AmberAtomType::IM,

            _ => {
                // DEBUG: This catch-all may be hiding the real problem!
                log::warn!(
                    "⚠️ UNKNOWN RESIDUE/ATOM: res='{}' atom='{}' → defaulting to CT",
                    res,
                    atom
                );
                AmberAtomType::CT
            }
        }
    }
}

// ============================================================================
// FORCE FIELD PARAMETERS
// ============================================================================

/// Bond parameter (harmonic potential)
/// E = k * (r - r0)²
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BondParam {
    /// Equilibrium bond length (Å)
    pub r0: f32,
    /// Force constant (kcal/mol/Å²)
    pub k: f32,
}

/// Angle parameter (harmonic potential)
/// E = k * (θ - θ0)²
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AngleParam {
    /// Equilibrium angle (radians)
    pub theta0: f32,
    /// Force constant (kcal/mol/rad²)
    pub k: f32,
}

/// Dihedral/torsion parameter (periodic potential)
/// E = k * (1 + cos(n*φ - δ))
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DihedralParam {
    /// Barrier height (kcal/mol)
    pub k: f32,
    /// Periodicity
    pub n: u8,
    /// Phase offset (radians)
    pub phase: f32,
    /// Number of paths dividing the torsion (for scaling)
    pub paths: u8,
}

/// Lennard-Jones parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LJParam {
    /// Well depth (kcal/mol)
    pub epsilon: f32,
    /// Collision diameter (Å) - actually r_min/2
    pub rmin_half: f32,
}

// ============================================================================
// AMBER ff14SB PARAMETER TABLES
// ============================================================================

/// Get bond parameters for a pair of atom types
/// Returns (r0, k) in (Å, kcal/mol/Å²)
pub fn get_bond_param(type1: AmberAtomType, type2: AmberAtomType) -> Option<BondParam> {
    use AmberAtomType::*;

    // Canonicalize order (smaller type first)
    let (t1, t2) = if (type1 as u8) <= (type2 as u8) {
        (type1, type2)
    } else {
        (type2, type1)
    };

    let result = match (t1, t2) {
        // Backbone bonds
        (N, H) | (N, HP) => (1.010, 434.0),
        (N, CT) => (1.449, 337.0), // N-CA
        (CT, C) => (1.522, 317.0), // CA-C
        (C, O) => (1.229, 570.0),  // C=O
        (C, O2) => (1.250, 656.0), // Carboxylate
        (N, C) => (1.335, 490.0),  // Peptide bond (N=0 < C=5)

        // Aliphatic bonds
        (CT, CT) => (1.526, 310.0),
        (CT, HC) => (1.090, 340.0),
        (CT, H1) => (1.090, 340.0),

        // Hydroxyl bonds
        (CT, OH) => (1.410, 320.0),
        (OH, HO) => (0.960, 553.0),

        // Sulfur bonds
        (CT, SH) => (1.810, 237.0),
        (CT, S) => (1.810, 227.0),
        (SH, HS) => (1.336, 274.0),
        (S, S) => (2.038, 166.0), // Disulfide

        // Aromatic bonds (ordered by enum values: CT=2, N2=8, OH=10, CA=15, HA=16, CB=25)
        (CA, CA) => (1.400, 469.0),
        (CA, HA) => (1.080, 367.0),
        (CT, CA) => (1.510, 317.0), // CT=2 < CA=15
        (CT, CB) => (1.510, 317.0), // CT=2 < CB=25 - aliphatic to aromatic β-carbon (PHE/TYR/TRP)
        (OH, CA) => (1.364, 450.0), // Tyrosine: OH=10 < CA=15
        (N2, CA) => (1.340, 481.0), // Arginine guanidinium: N2=8 < CA=15
        (CA, CB) => (1.404, 469.0), // CA=15 < CB=25
        (CB, CB) => (1.370, 520.0),
        (CN, CB) => (1.419, 447.0), // Tryptophan: CN=24 < CB=25

        // Charged residue bonds
        (CT, N3) => (1.471, 367.0), // Lysine: CT=2 < N3=9
        (CT, N2) => (1.463, 337.0), // Arginine: CT=2 < N2=8
        (H, N2) => (1.010, 434.0),  // Arginine guanidinium N-H: H=1 < N2=8
        (N3, HP) => (1.010, 434.0), // Lysine charged amino N-H: N3=9 < HP=17

        // Histidine specific (CT=2, NA=19, NB=20, CC=21, CR=22, CV=23)
        (CT, CC) => (1.504, 317.0), // CT=2 < CC=21
        (NA, CC) => (1.385, 422.0), // NA=19 < CC=21
        (NB, CC) => (1.394, 410.0), // NB=20 < CC=21
        (CC, CV) => (1.375, 512.0), // CC=21 < CV=23
        (NB, CV) => (1.350, 488.0), // NB=20 < CV=23 - HIS ring NB-CV bond (AMBER standard)
        (NA, CR) => (1.343, 477.0), // NA=19 < CR=22
        (NB, CR) => (1.335, 488.0), // NB=20 < CR=22
        (H, CR) => (1.080, 367.0),  // H=1 < CR=22
        (H, CV) => (1.080, 367.0),  // H=1 < CV=23
        (H, CW) => (1.080, 367.0),  // H=1 < CW=18
        (CW, NA) => (1.381, 427.0), // CW=18 < NA=19
        (H, NA) => (1.010, 434.0),  // H=1 < NA=19

        // Tryptophan specific
        (CW, CB) => (1.365, 546.0), // CW=18 < CB=25
        (CA, CN) => (1.400, 469.0), // CA=15 < CN=24
        (NA, CN) => (1.380, 428.0), // NA=19 < CN=24

        // TIP3P Water bonds (OW=26, HW=27)
        (OW, HW) => (0.9572, 553.0), // TIP3P O-H bond
        (HW, HW) => (1.5139, 553.0), // Virtual H-H bond (for SETTLE reference)

        _ => {
            // DEBUG: Log every failed bond parameter lookup
            log::warn!(
                "⚠️ MISSING BOND PARAM: {:?}-{:?} (canonicalized from {:?}-{:?})",
                t1,
                t2,
                type1,
                type2
            );
            return None;
        }
    };

    Some(BondParam {
        r0: result.0,
        k: result.1,
    })
}

/// Get angle parameters for a triplet of atom types
/// Returns (theta0, k) in (radians, kcal/mol/rad²)
pub fn get_angle_param(
    type1: AmberAtomType,
    type2: AmberAtomType,
    type3: AmberAtomType,
) -> Option<AngleParam> {
    use AmberAtomType::*;

    let deg2rad = |deg: f32| deg * PI / 180.0;

    let (theta0_deg, k) = match (type1, type2, type3) {
        // Backbone angles
        (H, N, CT) | (HP, N, CT) | (CT, N, H) | (CT, N, HP) => (118.0, 50.0), // symmetric
        (CT, N, C) => (121.9, 50.0),                                          // CA-N-C (peptide)
        (H, N, C) => (119.8, 50.0),
        (N, CT, C) | (C, CT, N) => (110.1, 63.0), // N-CA-C (symmetric)
        (N, CT, CT) | (CT, CT, N) => (109.7, 80.0), // N-CA-CB (symmetric)
        (CT, CT, C) | (C, CT, CT) => (111.1, 63.0), // CB-CA-C (symmetric)
        (CT, C, O) | (O, C, CT) => (120.4, 80.0), // CA-C=O (symmetric)
        (CT, C, N) | (N, C, CT) => (116.6, 70.0), // CA-C-N (symmetric)
        (O, C, N) | (N, C, O) => (122.9, 80.0),   // O=C-N (symmetric)
        (C, N, CT) => (121.9, 50.0),              // C-N-CA (next)
        (CT, N, CT) => (118.0, 70.0),             // Proline ring N

        // Alpha hydrogen (symmetric)
        (N, CT, H1) | (H1, CT, N) => (109.5, 50.0),
        (C, CT, H1) | (H1, CT, C) => (109.5, 50.0),
        (H1, CT, CT) | (CT, CT, H1) => (109.5, 50.0),
        (H1, CT, H1) => (109.5, 35.0),
        (N, CT, HC) | (HC, CT, N) => (109.5, 50.0), // Backbone N to sidechain H
        (C, CT, HC) | (HC, CT, C) => (109.5, 50.0), // Carbonyl to methyl H

        // Aliphatic angles
        (CT, CT, CT) => (109.5, 40.0),
        (CT, CT, HC) | (HC, CT, CT) => (109.5, 50.0),
        (HC, CT, HC) => (109.5, 35.0),

        // Carboxylate angles
        (CT, C, O2) => (117.0, 70.0),
        (O2, C, O2) => (126.0, 80.0),

        // Hydroxyl angles
        (CT, CT, OH) | (OH, CT, CT) => (109.5, 50.0),
        (CT, OH, HO) | (HO, OH, CT) => (108.5, 55.0),
        (H1, CT, OH) | (OH, CT, H1) => (109.5, 50.0),

        // Sulfur angles (symmetric)
        (CT, CT, SH) | (SH, CT, CT) => (108.6, 50.0),
        (CT, SH, HS) | (HS, SH, CT) => (96.0, 43.0),
        (CT, CT, S) | (S, CT, CT) => (114.7, 50.0),
        (CT, S, CT) => (98.9, 62.0),
        (CT, S, S) | (S, S, CT) => (103.7, 68.0), // Disulfide
        (H1, CT, S) | (S, CT, H1) => (109.5, 50.0), // Sulfur-alpha H
        (H1, CT, SH) | (SH, CT, H1) => (109.5, 50.0), // Thiol-alpha H

        // Aromatic angles (symmetric)
        (CA, CA, CA) => (120.0, 63.0),
        (CA, CA, HA) | (HA, CA, CA) => (120.0, 50.0),
        (CA, CA, CT) | (CT, CA, CA) => (120.0, 70.0),
        (CA, CA, OH) | (OH, CA, CA) => (120.0, 70.0),
        (CA, OH, HO) | (HO, OH, CA) => (113.0, 50.0),
        (CT, CT, CA) | (CA, CT, CT) => (109.5, 63.0), // Aliphatic to aromatic
        (HC, CT, CA) | (CA, CT, HC) => (109.5, 50.0), // H on CB next to aromatic

        // Charged sidechain angles (symmetric)
        (CT, CT, N3) | (N3, CT, CT) => (111.2, 80.0), // Lysine
        (CT, N3, HP) | (HP, N3, CT) => (109.5, 50.0),
        (HP, N3, HP) => (109.5, 35.0),
        (HC, CT, N3) | (N3, CT, HC) => (109.5, 50.0), // Lysine CE-H to NZ (CT-centered)
        (CT, N3, HC) | (HC, N3, CT) => (109.5, 50.0), // Lysine C-N-H (N3-centered)
        (CT, CT, N2) | (N2, CT, CT) => (111.2, 80.0), // Arginine
        (CT, N2, H) | (H, N2, CT) => (118.4, 50.0),
        (CA, N2, H) | (H, N2, CA) => (120.0, 50.0),
        (N2, CA, N2) => (120.0, 70.0),
        (CT, N2, CA) | (CA, N2, CT) => (123.2, 50.0),
        (HC, CT, N2) | (N2, CT, HC) => (109.5, 50.0), // Arginine CD-H to NE (CT-centered)
        (CT, N2, HC) | (HC, N2, CT) => (120.0, 50.0), // Arginine C-N-H (N2-centered)
        (H, N2, H) => (120.0, 35.0),                  // Guanidinium H-N-H
        (H, N2, HC) | (HC, N2, H) => (120.0, 35.0),   // Guanidinium mixed H

        // Histidine angles (symmetric)
        (CT, CT, CC) | (CC, CT, CT) => (109.5, 63.0), // CA-CB-CG angle
        (CT, CC, NA) | (NA, CC, CT) => (120.0, 70.0),
        (CT, CC, CV) | (CV, CC, CT) => (120.0, 70.0),
        (CT, CC, NB) | (NB, CC, CT) => (120.0, 70.0),
        (NA, CC, CV) | (CV, CC, NA) => (120.0, 70.0),
        (NA, CC, NB) | (NB, CC, NA) => (120.0, 70.0),
        (CC, NA, CR) | (CR, NA, CC) => (105.4, 70.0),
        (CC, NA, H) | (H, NA, CC) => (126.4, 50.0),
        (CR, NA, H) | (H, NA, CR) => (128.2, 50.0),
        (NA, CR, NB) | (NB, CR, NA) => (111.6, 70.0),
        (NA, CR, H) | (H, CR, NA) => (124.2, 50.0),
        (NB, CR, H) | (H, CR, NB) => (124.2, 50.0),
        (CC, NB, CR) | (CR, NB, CC) => (103.8, 70.0),
        (CC, NB, CV) | (CV, NB, CC) => (105.0, 70.0), // HIS ring NB-CV connection
        (CR, NB, CV) | (CV, NB, CR) => (110.0, 70.0), // HIS ring
        (CC, CV, H) | (H, CV, CC) => (130.0, 50.0),
        (NB, CV, H) | (H, CV, NB) => (120.0, 50.0),
        (CC, CV, NB) | (NB, CV, CC) => (110.0, 70.0),
        (HC, CT, CC) | (CC, CT, HC) => (109.5, 50.0), // HIS CB-H angles

        // Amide angles (ASN, GLN) - note: some patterns covered above
        (C, N, H) => (119.8, 50.0),
        (H, N, H) => (120.0, 35.0),

        // Tryptophan angles (with symmetric patterns)
        (CA, CB, CB) | (CB, CB, CA) => (117.0, 63.0), // CB-centered: CA and CB outer
        (CA, CB, CW) | (CW, CB, CA) => (133.0, 63.0), // CB-centered: CA and CW outer
        (CT, CB, CB) | (CB, CB, CT) => (117.0, 63.0), // CB-centered: CT and CB outer (backbone junction)
        (CT, CB, CW) | (CW, CB, CT) => (126.0, 63.0), // CB-centered: CT and CW outer (TRP Cβ)
        (CN, CB, CA) | (CA, CB, CN) => (117.0, 63.0), // CB-centered: CN and CA outer
        (CB, CB, CN) | (CN, CB, CB) => (116.0, 63.0), // CB-centered: CB and CN outer
        (CB, CW, NA) | (NA, CW, CB) => (108.7, 70.0), // CW-centered
        (CB, CW, H) | (H, CW, CB) => (130.0, 50.0),   // CW-centered
        (NA, CW, H) | (H, CW, NA) => (121.0, 50.0),   // CW-centered
        (CW, NA, CN) | (CN, NA, CW) => (111.6, 70.0), // NA-centered
        (CW, NA, H) | (H, NA, CW) => (125.0, 50.0),   // NA-centered
        (CN, NA, H) | (H, NA, CN) => (123.0, 50.0),   // NA-centered
        (NA, CN, CA) | (CA, CN, NA) => (132.0, 70.0), // CN-centered
        (NA, CN, CB) | (CB, CN, NA) => (108.0, 70.0), // CN-centered
        (CA, CN, CB) | (CB, CN, CA) => (120.0, 63.0), // CN-centered
        (CN, CA, CA) | (CA, CA, CN) => (120.0, 63.0), // CA-centered

        // Aromatic CB with aliphatic CT (PHE/TYR/TRP Cα-Cβ junction)
        (CT, CT, CB) | (CB, CT, CT) => (109.5, 63.0), // CT-centered: backbone to aromatic
        (CB, CT, HC) | (HC, CT, CB) => (109.5, 50.0), // CT-centered: H on CT next to aromatic CB
        (CB, CA, HA) | (HA, CA, CB) => (120.0, 50.0), // CA-centered: aromatic ring H
        (CB, CA, CA) | (CA, CA, CB) => (120.0, 63.0), // CA-centered: CB to CA aromatic ring
        (CN, CA, HA) | (HA, CA, CN) => (120.0, 50.0), // CA-centered: TRP ring junction
        (CW, CB, CB) | (CB, CB, CW) => (107.0, 63.0), // CB-centered: TRP 5-ring/6-ring junction

        // TIP3P Water angle (HW-OW-HW)
        (HW, OW, HW) => (104.52, 100.0), // TIP3P H-O-H angle

        _ => {
            // DEBUG: Log every failed angle parameter lookup
            log::warn!(
                "⚠️ MISSING ANGLE PARAM: {:?}-{:?}-{:?}",
                type1,
                type2,
                type3
            );
            return None;
        }
    };

    Some(AngleParam {
        theta0: deg2rad(theta0_deg),
        k,
    })
}

/// Get dihedral parameters for a quartet of atom types
/// Returns Vec of (k, n, phase, paths) for multiple terms
pub fn get_dihedral_params(
    type1: AmberAtomType,
    type2: AmberAtomType,
    type3: AmberAtomType,
    type4: AmberAtomType,
) -> Vec<DihedralParam> {
    use AmberAtomType::*;

    let deg2rad = |deg: f32| deg * PI / 180.0;

    // Helper to create DihedralParam
    let dp = |k: f32, n: u8, phase_deg: f32, paths: u8| DihedralParam {
        k,
        n,
        phase: deg2rad(phase_deg),
        paths,
    };

    // Check for specific matches first, then wildcards
    // AMBER uses X (any) for wildcards

    // Specific backbone dihedrals
    match (type1, type2, type3, type4) {
        // Phi (C-N-CA-C)
        (C, N, CT, C) => return vec![dp(0.0, 1, 0.0, 1)],

        // Psi (N-CA-C-N)
        (N, CT, C, N) => return vec![dp(0.4, 1, 0.0, 1), dp(2.0, 2, 180.0, 1)],

        // Omega (CA-C-N-CA) - peptide bond planarity
        (CT, C, N, CT) => return vec![dp(2.5, 2, 180.0, 1)],

        _ => {}
    }

    // Generic dihedrals by central bond type
    match (type2, type3) {
        // X-C-N-X (peptide bonds)
        (C, N) => return vec![dp(2.5, 2, 180.0, 4)],

        // X-C-CT-X
        (C, CT) => return vec![dp(0.0, 2, 0.0, 4)],

        // X-CT-CT-X (aliphatic)
        (CT, CT) => return vec![dp(0.156, 3, 0.0, 9)],

        // X-CT-N-X (general CT-N)
        (CT, N) => return vec![dp(0.0, 2, 0.0, 6)],
        (N, CT) => return vec![dp(0.0, 2, 0.0, 6)],

        // X-CT-OH-X (hydroxyl)
        (CT, OH) => return vec![dp(0.167, 3, 0.0, 3)],
        (OH, CT) => return vec![dp(0.167, 3, 0.0, 3)],

        // X-CT-SH-X (thiol)
        (CT, SH) => return vec![dp(0.75, 3, 0.0, 3)],
        (SH, CT) => return vec![dp(0.75, 3, 0.0, 3)],

        // X-CT-S-X (thioether)
        (CT, S) | (S, CT) => return vec![dp(0.333, 3, 0.0, 3)],

        // X-S-S-X (disulfide)
        (S, S) => return vec![dp(3.5, 2, 0.0, 1), dp(0.6, 3, 0.0, 1)],

        // X-CT-N3-X (lysine amino)
        (CT, N3) | (N3, CT) => return vec![dp(0.156, 3, 0.0, 9)],

        // X-CA-CA-X (aromatic)
        (CA, CA) => return vec![dp(3.625, 2, 180.0, 4)],

        // X-CA-CT-X
        (CA, CT) | (CT, CA) => return vec![dp(0.0, 2, 0.0, 6)],

        // X-CA-OH-X (tyrosine)
        (CA, OH) | (OH, CA) => return vec![dp(0.9, 2, 180.0, 2)],

        // Histidine ring
        (CC, NA) | (NA, CC) => return vec![dp(1.4, 2, 180.0, 4)],
        (CR, NA) | (NA, CR) => return vec![dp(1.4, 2, 180.0, 4)],
        (CR, NB) | (NB, CR) => return vec![dp(2.4, 2, 180.0, 4)],
        (CC, NB) | (NB, CC) => return vec![dp(2.4, 2, 180.0, 4)],
        (CC, CV) | (CV, CC) => return vec![dp(2.1, 2, 180.0, 4)],

        // Tryptophan ring
        (CB, CW) | (CW, CB) => return vec![dp(5.0, 2, 180.0, 4)],
        (CW, NA) | (NA, CW) => return vec![dp(1.5, 2, 180.0, 4)],
        (CN, NA) | (NA, CN) => return vec![dp(1.5, 2, 180.0, 4)],
        (CN, CA) | (CA, CN) => return vec![dp(3.625, 2, 180.0, 4)],
        (CN, CB) | (CB, CN) => return vec![dp(3.0, 2, 180.0, 4)],

        _ => {}
    }

    // Default: return empty (no dihedral)
    vec![]
}

/// Get Lennard-Jones parameters for an atom type
pub fn get_lj_param(atom_type: AmberAtomType) -> LJParam {
    use AmberAtomType::*;

    // AMBER ff14SB LJ parameters (epsilon in kcal/mol, rmin/2 in Å)
    let (epsilon, rmin_half) = match atom_type {
        N | N2 | N3 | NA | NB => (0.170, 1.824),
        H | HP | HS => (0.0157, 0.600),
        H1 => (0.0157, 1.387),
        HC | HA => (0.0157, 1.487),
        HO => (0.0000, 0.000), // No LJ for hydroxyl H
        CT => (0.1094, 1.908),
        C => (0.0860, 1.908),
        O | O2 => (0.2100, 1.661),
        OH => (0.2104, 1.721),
        S | SH => (0.2500, 2.000),
        CA | CB | CC | CR | CV | CW | CN => (0.0860, 1.908),

        // TIP3P Water (modified for stability with simplified SETTLE)
        // Standard TIP3P has no LJ on H, but this causes collapse issues
        // during minimization when waters get too close. Adding LJ
        // for stability, similar to CHARMM TIP3P which gives H a radius.
        // Using rmin_half = 0.8 Å gives combined H-O distance ~2.5 Å minimum.
        OW => (0.1521, 1.7683), // TIP3P oxygen: ε=0.1521, rmin/2=σ*2^(1/6)/2
        HW => (0.0460, 0.8000), // TIP3P-mod: LJ for stability (rmin/2=0.8Å)

        // Ions (Joung & Cheatham parameters for TIP3P water)
        IP => (0.0874393, 1.212), // Na+
        IM => (0.0355910, 2.711), // Cl-

        Unknown => (0.1094, 1.908), // Default to CT
    };

    LJParam { epsilon, rmin_half }
}

/// 1-4 scaling factors for AMBER ff14SB
pub const SCALE_14_LJ: f32 = 0.5;
pub const SCALE_14_ELEC: f32 = 0.8333; // 1/1.2

// ============================================================================
// TOPOLOGY DATA STRUCTURES
// ============================================================================

/// Complete molecular topology for simulation
#[derive(Debug, Clone)]
pub struct AmberTopology {
    // Atom data
    pub n_atoms: usize,
    pub atom_types: Vec<AmberAtomType>,
    pub masses: Vec<f32>,
    pub charges: Vec<f32>,
    pub lj_params: Vec<LJParam>,

    // Bond connectivity and parameters
    pub bonds: Vec<(u32, u32)>,
    pub bond_params: Vec<BondParam>,

    // Angle connectivity and parameters
    pub angles: Vec<(u32, u32, u32)>,
    pub angle_params: Vec<AngleParam>,

    // Dihedral connectivity and parameters
    pub dihedrals: Vec<(u32, u32, u32, u32)>,
    pub dihedral_params: Vec<Vec<DihedralParam>>,

    // Improper dihedrals (for planar groups)
    pub impropers: Vec<(u32, u32, u32, u32)>,
    pub improper_params: Vec<DihedralParam>,

    // 1-4 pairs for special non-bonded scaling
    pub pairs_14: Vec<(u32, u32)>,

    // Exclusions (1-2 and 1-3 pairs to exclude from non-bonded)
    pub exclusions: Vec<(u32, u32)>,
}

impl Default for AmberTopology {
    fn default() -> Self {
        Self {
            n_atoms: 0,
            atom_types: Vec::new(),
            masses: Vec::new(),
            charges: Vec::new(),
            lj_params: Vec::new(),
            bonds: Vec::new(),
            bond_params: Vec::new(),
            angles: Vec::new(),
            angle_params: Vec::new(),
            dihedrals: Vec::new(),
            dihedral_params: Vec::new(),
            impropers: Vec::new(),
            improper_params: Vec::new(),
            pairs_14: Vec::new(),
            exclusions: Vec::new(),
        }
    }
}

/// GPU-uploadable topology format (flattened arrays)
#[derive(Debug, Clone)]
pub struct GpuTopology {
    /// Bond list: [atom_i, atom_j, ...] flattened pairs
    pub bond_list: Vec<i32>,
    /// Bond params: [r0, k, ...] flattened pairs
    pub bond_params: Vec<f32>,

    /// Angle list: [atom_i, atom_j, atom_k, ...] flattened triplets
    pub angle_list: Vec<i32>,
    /// Angle params: [theta0, k, ...] flattened pairs
    pub angle_params: Vec<f32>,

    /// Dihedral list: [atom_i, atom_j, atom_k, atom_l, ...] flattened quartets
    pub dihedral_list: Vec<i32>,
    /// Dihedral params: [k, n, phase, paths, ...] flattened quartets
    /// Note: Each dihedral can have multiple terms, so we need a term count
    pub dihedral_params: Vec<f32>,
    /// Number of terms per dihedral
    pub dihedral_term_counts: Vec<i32>,

    /// 1-4 pair list: [atom_i, atom_j, ...] flattened pairs
    pub pair_14_list: Vec<i32>,

    /// Exclusion list (1-2 and 1-3): [atom_i, atom_j, ...] flattened pairs
    pub exclusion_list: Vec<i32>,

    /// Counts
    pub n_bonds: i32,
    pub n_angles: i32,
    pub n_dihedrals: i32,
    pub n_pairs_14: i32,
    pub n_exclusions: i32,
}

impl GpuTopology {
    /// Convert from AmberTopology to GPU-uploadable format
    pub fn from_amber(topo: &AmberTopology) -> Self {
        let mut gpu = GpuTopology {
            bond_list: Vec::with_capacity(topo.bonds.len() * 2),
            bond_params: Vec::with_capacity(topo.bonds.len() * 2),
            angle_list: Vec::with_capacity(topo.angles.len() * 3),
            angle_params: Vec::with_capacity(topo.angles.len() * 2),
            dihedral_list: Vec::with_capacity(topo.dihedrals.len() * 4),
            dihedral_params: Vec::new(),
            dihedral_term_counts: Vec::with_capacity(topo.dihedrals.len()),
            pair_14_list: Vec::with_capacity(topo.pairs_14.len() * 2),
            exclusion_list: Vec::with_capacity(topo.exclusions.len() * 2),
            n_bonds: topo.bonds.len() as i32,
            n_angles: topo.angles.len() as i32,
            n_dihedrals: topo.dihedrals.len() as i32,
            n_pairs_14: topo.pairs_14.len() as i32,
            n_exclusions: topo.exclusions.len() as i32,
        };

        // Flatten bonds
        for ((a, b), param) in topo.bonds.iter().zip(&topo.bond_params) {
            gpu.bond_list.push(*a as i32);
            gpu.bond_list.push(*b as i32);
            gpu.bond_params.push(param.r0);
            gpu.bond_params.push(param.k);
        }

        // Flatten angles
        for ((a, b, c), param) in topo.angles.iter().zip(&topo.angle_params) {
            gpu.angle_list.push(*a as i32);
            gpu.angle_list.push(*b as i32);
            gpu.angle_list.push(*c as i32);
            gpu.angle_params.push(param.theta0);
            gpu.angle_params.push(param.k);
        }

        // Flatten dihedrals (with multiple terms per dihedral)
        for ((a, b, c, d), params) in topo.dihedrals.iter().zip(&topo.dihedral_params) {
            gpu.dihedral_list.push(*a as i32);
            gpu.dihedral_list.push(*b as i32);
            gpu.dihedral_list.push(*c as i32);
            gpu.dihedral_list.push(*d as i32);
            gpu.dihedral_term_counts.push(params.len() as i32);
            for param in params {
                gpu.dihedral_params.push(param.k);
                gpu.dihedral_params.push(param.n as f32);
                gpu.dihedral_params.push(param.phase);
                gpu.dihedral_params.push(param.paths as f32);
            }
        }

        // Flatten 1-4 pairs
        for (a, b) in &topo.pairs_14 {
            gpu.pair_14_list.push(*a as i32);
            gpu.pair_14_list.push(*b as i32);
        }

        // Flatten exclusions
        for (a, b) in &topo.exclusions {
            gpu.exclusion_list.push(*a as i32);
            gpu.exclusion_list.push(*b as i32);
        }

        gpu
    }
}

// ============================================================================
// ATOM MASS LOOKUP
// ============================================================================

/// Get atomic mass from atom type (in amu/Daltons)
pub fn get_atom_mass(atom_type: AmberAtomType) -> f32 {
    use AmberAtomType::*;

    match atom_type {
        H | H1 | HC | HA | HO | HP | HS => 1.008,
        C | CT | CA | CB | CC | CR | CV | CW | CN => 12.01,
        N | N2 | N3 | NA | NB => 14.01,
        O | O2 | OH => 16.00,
        S | SH => 32.07,

        // TIP3P Water
        OW => 15.9994, // Water oxygen
        HW => 1.008,   // Water hydrogen

        // Ions
        IP => 22.9898, // Na+
        IM => 35.453,  // Cl-

        Unknown => 12.01, // Default to carbon
    }
}

// ============================================================================
// PARTIAL CHARGES (from atomic_chemistry.rs integration)
// ============================================================================

/// Get partial charge for an atom (in elementary charge units)
/// These are derived from AMBER ff14SB
pub fn get_atom_charge(residue: &str, atom_name: &str) -> f32 {
    let atom = atom_name.trim();
    let res = residue.to_uppercase();

    // Backbone charges (same for all residues)
    match atom {
        "N" => return -0.4157,
        "H" | "HN" => return 0.2719,
        "CA" => return 0.0337,
        "HA" | "HA2" | "HA3" => return 0.0823,
        "C" => return 0.5973,
        "O" => return -0.5679,
        "OXT" => return -0.8055, // C-terminal
        _ => {}
    }

    // Sidechain charges
    match res.as_str() {
        "ALA" => match atom {
            "CB" => -0.1825,
            "HB1" | "HB2" | "HB3" => 0.0603,
            _ => 0.0,
        },

        "ARG" => match atom {
            "CB" => -0.0007,
            "HB2" | "HB3" => 0.0327,
            "CG" => 0.0390,
            "HG2" | "HG3" => 0.0285,
            "CD" => 0.0486,
            "HD2" | "HD3" => 0.0687,
            "NE" => -0.5295,
            "HE" => 0.3456,
            "CZ" => 0.8076,
            "NH1" | "NH2" => -0.8627,
            "HH11" | "HH12" | "HH21" | "HH22" => 0.4478,
            _ => 0.0,
        },

        "ASN" => match atom {
            "CB" => -0.2041,
            "HB2" | "HB3" => 0.0797,
            "CG" => 0.7130,
            "OD1" => -0.5931,
            "ND2" => -0.9191,
            "HD21" | "HD22" => 0.4196,
            _ => 0.0,
        },

        "ASP" => match atom {
            "CB" => -0.0303,
            "HB2" | "HB3" => -0.0122,
            "CG" => 0.7994,
            "OD1" | "OD2" => -0.8014,
            _ => 0.0,
        },

        "CYS" => match atom {
            "CB" => -0.1231,
            "HB2" | "HB3" => 0.1112,
            "SG" => -0.3119,
            "HG" => 0.1933,
            _ => 0.0,
        },

        "GLN" => match atom {
            "CB" => -0.0036,
            "HB2" | "HB3" => 0.0171,
            "CG" => -0.0645,
            "HG2" | "HG3" => 0.0352,
            "CD" => 0.6951,
            "OE1" => -0.6086,
            "NE2" => -0.9407,
            "HE21" | "HE22" => 0.4251,
            _ => 0.0,
        },

        "GLU" => match atom {
            "CB" => 0.0560,
            "HB2" | "HB3" => -0.0173,
            "CG" => 0.0136,
            "HG2" | "HG3" => -0.0425,
            "CD" => 0.8054,
            "OE1" | "OE2" => -0.8188,
            _ => 0.0,
        },

        "GLY" => match atom {
            "HA2" | "HA3" => 0.0698,
            _ => 0.0,
        },

        "HIS" | "HID" => match atom {
            // Neutral, proton on ND1
            "CB" => -0.0414,
            "HB2" | "HB3" => 0.0367,
            "CG" => -0.0012,
            "ND1" => -0.3811,
            "HD1" => 0.3649,
            "CE1" => 0.2057,
            "HE1" => 0.1392,
            "NE2" => -0.5727,
            "CD2" => 0.1292,
            "HD2" => 0.1147,
            _ => 0.0,
        },

        "HIE" => match atom {
            // Neutral, proton on NE2
            "CB" => -0.0581,
            "HB2" | "HB3" => 0.0367,
            "CG" => 0.1868,
            "ND1" => -0.5432,
            "CE1" => 0.1635,
            "HE1" => 0.1435,
            "NE2" => -0.2795,
            "HE2" => 0.3339,
            "CD2" => -0.2207,
            "HD2" => 0.1862,
            _ => 0.0,
        },

        "HIP" => match atom {
            // Charged (+1)
            "CB" => -0.0236,
            "HB2" | "HB3" => 0.0519,
            "CG" => -0.0017,
            "ND1" => -0.1513,
            "HD1" => 0.3866,
            "CE1" => -0.0170,
            "HE1" => 0.2681,
            "NE2" => -0.1718,
            "HE2" => 0.3911,
            "CD2" => -0.1141,
            "HD2" => 0.2317,
            _ => 0.0,
        },

        "ILE" => match atom {
            "CB" => 0.1303,
            "HB" => 0.0187,
            "CG1" => -0.0430,
            "HG12" | "HG13" => 0.0236,
            "CG2" => -0.3204,
            "HG21" | "HG22" | "HG23" => 0.0882,
            "CD1" => -0.0660,
            "HD11" | "HD12" | "HD13" => 0.0186,
            _ => 0.0,
        },

        "LEU" => match atom {
            "CB" => -0.2106,
            "HB2" | "HB3" => 0.0457,
            "CG" => 0.3531,
            "HG" => -0.0361,
            "CD1" | "CD2" => -0.4121,
            "HD11" | "HD12" | "HD13" | "HD21" | "HD22" | "HD23" => 0.1000,
            _ => 0.0,
        },

        "LYS" => match atom {
            "CB" => -0.0094,
            "HB2" | "HB3" => 0.0362,
            "CG" => 0.0187,
            "HG2" | "HG3" => 0.0103,
            "CD" => -0.0479,
            "HD2" | "HD3" => 0.0621,
            "CE" => -0.0143,
            "HE2" | "HE3" => 0.1135,
            "NZ" => -0.3854,
            "HZ1" | "HZ2" | "HZ3" => 0.3400,
            _ => 0.0,
        },

        "MET" => match atom {
            "CB" => 0.0342,
            "HB2" | "HB3" => 0.0241,
            "CG" => 0.0018,
            "HG2" | "HG3" => 0.0440,
            "SD" => -0.2737,
            "CE" => -0.0536,
            "HE1" | "HE2" | "HE3" => 0.0684,
            _ => 0.0,
        },

        "PHE" => match atom {
            "CB" => -0.0343,
            "HB2" | "HB3" => 0.0295,
            "CG" => 0.0118,
            "CD1" | "CD2" => -0.1256,
            "HD1" | "HD2" => 0.1330,
            "CE1" | "CE2" => -0.1704,
            "HE1" | "HE2" => 0.1430,
            "CZ" => -0.1072,
            "HZ" => 0.1297,
            _ => 0.0,
        },

        "PRO" => match atom {
            "CB" => -0.0070,
            "HB2" | "HB3" => 0.0253,
            "CG" => 0.0189,
            "HG2" | "HG3" => 0.0213,
            "CD" => 0.0192,
            "HD2" | "HD3" => 0.0391,
            _ => 0.0,
        },

        "SER" => match atom {
            "CB" => 0.2117,
            "HB2" | "HB3" => 0.0352,
            "OG" => -0.6546,
            "HG" => 0.4275,
            _ => 0.0,
        },

        "THR" => match atom {
            "CB" => 0.3654,
            "HB" => 0.0043,
            "OG1" => -0.6761,
            "HG1" => 0.4102,
            "CG2" => -0.2438,
            "HG21" | "HG22" | "HG23" => 0.0642,
            _ => 0.0,
        },

        "TRP" => match atom {
            "CB" => -0.0050,
            "HB2" | "HB3" => 0.0339,
            "CG" => -0.1415,
            "CD1" => -0.1638,
            "HD1" => 0.2062,
            "NE1" => -0.3418,
            "HE1" => 0.3412,
            "CE2" => 0.1380,
            "CZ2" => -0.2601,
            "HZ2" => 0.1572,
            "CH2" => -0.1134,
            "HH2" => 0.1417,
            "CZ3" => -0.1972,
            "HZ3" => 0.1447,
            "CE3" => -0.2387,
            "HE3" => 0.1700,
            "CD2" => 0.1243,
            _ => 0.0,
        },

        "TYR" => match atom {
            "CB" => -0.0152,
            "HB2" | "HB3" => 0.0295,
            "CG" => -0.0011,
            "CD1" | "CD2" => -0.1906,
            "HD1" | "HD2" => 0.1699,
            "CE1" | "CE2" => -0.2341,
            "HE1" | "HE2" => 0.1656,
            "CZ" => 0.3226,
            "OH" => -0.5579,
            "HH" => 0.3992,
            _ => 0.0,
        },

        "VAL" => match atom {
            "CB" => 0.2985,
            "HB" => -0.0297,
            "CG1" | "CG2" => -0.3192,
            "HG11" | "HG12" | "HG13" | "HG21" | "HG22" | "HG23" => 0.0791,
            _ => 0.0,
        },

        // TIP3P Water charges
        "WAT" | "HOH" | "SOL" | "TIP3" | "TIP3P" => match atom {
            "O" | "OW" | "OH2" => -0.834,
            "H1" | "H2" | "HW1" | "HW2" => 0.417,
            _ => 0.0,
        },

        // Ion charges
        "NA" | "NA+" | "SOD" => 1.0,  // Na+
        "CL" | "CL-" | "CLA" => -1.0, // Cl-

        _ => 0.0,
    }
}

// ============================================================================
// RESIDUE TEMPLATES
// ============================================================================

/// Internal connectivity template for a residue (which atoms are bonded)
/// Atom names are canonical PDB names
#[derive(Debug, Clone)]
pub struct ResidueTemplate {
    pub name: &'static str,
    /// Internal bonds within residue (atom_name_1, atom_name_2)
    pub bonds: &'static [(&'static str, &'static str)],
    /// Heavy atoms (for CA-only builds)
    pub heavy_atoms: &'static [&'static str],
}

/// Get residue template for standard amino acids
pub fn get_residue_template(res_name: &str) -> Option<ResidueTemplate> {
    match res_name.to_uppercase().as_str() {
        "ALA" => Some(ResidueTemplate {
            name: "ALA",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB1"),
                ("CB", "HB2"),
                ("CB", "HB3"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB"],
        }),

        "ARG" => Some(ResidueTemplate {
            name: "ARG",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "HG2"),
                ("CG", "HG3"),
                ("CG", "CD"),
                ("CD", "HD2"),
                ("CD", "HD3"),
                ("CD", "NE"),
                ("NE", "HE"),
                ("NE", "CZ"),
                ("CZ", "NH1"),
                ("CZ", "NH2"),
                ("NH1", "HH11"),
                ("NH1", "HH12"),
                ("NH2", "HH21"),
                ("NH2", "HH22"),
            ],
            heavy_atoms: &[
                "N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2",
            ],
        }),

        "ASN" => Some(ResidueTemplate {
            name: "ASN",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "OD1"),
                ("CG", "ND2"),
                ("ND2", "HD21"),
                ("ND2", "HD22"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
        }),

        "ASP" => Some(ResidueTemplate {
            name: "ASP",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "OD1"),
                ("CG", "OD2"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
        }),

        "CYS" => Some(ResidueTemplate {
            name: "CYS",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "SG"),
                ("SG", "HG"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "SG"],
        }),

        "GLN" => Some(ResidueTemplate {
            name: "GLN",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "HG2"),
                ("CG", "HG3"),
                ("CG", "CD"),
                ("CD", "OE1"),
                ("CD", "NE2"),
                ("NE2", "HE21"),
                ("NE2", "HE22"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
        }),

        "GLU" => Some(ResidueTemplate {
            name: "GLU",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "HG2"),
                ("CG", "HG3"),
                ("CG", "CD"),
                ("CD", "OE1"),
                ("CD", "OE2"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
        }),

        "GLY" => Some(ResidueTemplate {
            name: "GLY",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA2"),
                ("CA", "HA3"),
            ],
            heavy_atoms: &["N", "CA", "C", "O"],
        }),

        "HIS" | "HID" | "HIE" | "HIP" => Some(ResidueTemplate {
            name: "HIS",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "ND1"),
                ("CG", "CD2"),
                ("ND1", "HD1"),
                ("ND1", "CE1"),
                ("CE1", "HE1"),
                ("CE1", "NE2"),
                ("NE2", "CD2"),
                ("CD2", "HD2"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG", "ND1", "CE1", "NE2", "CD2"],
        }),

        "ILE" => Some(ResidueTemplate {
            name: "ILE",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB"),
                ("CB", "CG1"),
                ("CB", "CG2"),
                ("CG1", "HG12"),
                ("CG1", "HG13"),
                ("CG1", "CD1"),
                ("CG2", "HG21"),
                ("CG2", "HG22"),
                ("CG2", "HG23"),
                ("CD1", "HD11"),
                ("CD1", "HD12"),
                ("CD1", "HD13"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
        }),

        "LEU" => Some(ResidueTemplate {
            name: "LEU",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "HG"),
                ("CG", "CD1"),
                ("CG", "CD2"),
                ("CD1", "HD11"),
                ("CD1", "HD12"),
                ("CD1", "HD13"),
                ("CD2", "HD21"),
                ("CD2", "HD22"),
                ("CD2", "HD23"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
        }),

        "LYS" => Some(ResidueTemplate {
            name: "LYS",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "HG2"),
                ("CG", "HG3"),
                ("CG", "CD"),
                ("CD", "HD2"),
                ("CD", "HD3"),
                ("CD", "CE"),
                ("CE", "HE2"),
                ("CE", "HE3"),
                ("CE", "NZ"),
                ("NZ", "HZ1"),
                ("NZ", "HZ2"),
                ("NZ", "HZ3"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
        }),

        "MET" => Some(ResidueTemplate {
            name: "MET",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "HG2"),
                ("CG", "HG3"),
                ("CG", "SD"),
                ("SD", "CE"),
                ("CE", "HE1"),
                ("CE", "HE2"),
                ("CE", "HE3"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
        }),

        "PHE" => Some(ResidueTemplate {
            name: "PHE",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "CD1"),
                ("CG", "CD2"),
                ("CD1", "HD1"),
                ("CD1", "CE1"),
                ("CD2", "HD2"),
                ("CD2", "CE2"),
                ("CE1", "HE1"),
                ("CE1", "CZ"),
                ("CE2", "HE2"),
                ("CE2", "CZ"),
                ("CZ", "HZ"),
            ],
            heavy_atoms: &[
                "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ",
            ],
        }),

        "PRO" => Some(ResidueTemplate {
            name: "PRO",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("N", "CD"), // Proline ring closes here
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "HG2"),
                ("CG", "HG3"),
                ("CG", "CD"),
                ("CD", "HD2"),
                ("CD", "HD3"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG", "CD"],
        }),

        "SER" => Some(ResidueTemplate {
            name: "SER",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "OG"),
                ("OG", "HG"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "OG"],
        }),

        "THR" => Some(ResidueTemplate {
            name: "THR",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB"),
                ("CB", "OG1"),
                ("CB", "CG2"),
                ("OG1", "HG1"),
                ("CG2", "HG21"),
                ("CG2", "HG22"),
                ("CG2", "HG23"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "OG1", "CG2"],
        }),

        "TRP" => Some(ResidueTemplate {
            name: "TRP",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "CD1"),
                ("CG", "CD2"),
                ("CD1", "HD1"),
                ("CD1", "NE1"),
                ("NE1", "HE1"),
                ("NE1", "CE2"),
                ("CE2", "CZ2"),
                ("CE2", "CD2"),
                ("CZ2", "HZ2"),
                ("CZ2", "CH2"),
                ("CH2", "HH2"),
                ("CH2", "CZ3"),
                ("CZ3", "HZ3"),
                ("CZ3", "CE3"),
                ("CE3", "HE3"),
                ("CE3", "CD2"),
            ],
            heavy_atoms: &[
                "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3",
                "CH2",
            ],
        }),

        "TYR" => Some(ResidueTemplate {
            name: "TYR",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB2"),
                ("CB", "HB3"),
                ("CB", "CG"),
                ("CG", "CD1"),
                ("CG", "CD2"),
                ("CD1", "HD1"),
                ("CD1", "CE1"),
                ("CD2", "HD2"),
                ("CD2", "CE2"),
                ("CE1", "HE1"),
                ("CE1", "CZ"),
                ("CE2", "HE2"),
                ("CE2", "CZ"),
                ("CZ", "OH"),
                ("OH", "HH"),
            ],
            heavy_atoms: &[
                "N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH",
            ],
        }),

        "VAL" => Some(ResidueTemplate {
            name: "VAL",
            bonds: &[
                ("N", "CA"),
                ("CA", "C"),
                ("C", "O"),
                ("N", "H"),
                ("CA", "HA"),
                ("CA", "CB"),
                ("CB", "HB"),
                ("CB", "CG1"),
                ("CB", "CG2"),
                ("CG1", "HG11"),
                ("CG1", "HG12"),
                ("CG1", "HG13"),
                ("CG2", "HG21"),
                ("CG2", "HG22"),
                ("CG2", "HG23"),
            ],
            heavy_atoms: &["N", "CA", "C", "O", "CB", "CG1", "CG2"],
        }),

        _ => None,
    }
}

// ============================================================================
// TOPOLOGY GENERATOR
// ============================================================================

/// Input atom for topology generation
#[derive(Debug, Clone)]
pub struct PdbAtom {
    pub index: usize,
    pub name: String,
    pub residue_name: String,
    pub residue_id: i32,
    pub chain_id: char,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl AmberTopology {
    /// Build topology from a list of PDB atoms
    /// This is the main entry point for topology generation
    pub fn from_pdb_atoms(atoms: &[PdbAtom]) -> Self {
        let mut topo = AmberTopology::default();
        topo.n_atoms = atoms.len();

        // Step 1: Assign atom types, masses, charges, and LJ params
        for atom in atoms {
            let atom_type = AmberAtomType::from_pdb(&atom.residue_name, &atom.name);
            topo.atom_types.push(atom_type);
            topo.masses.push(get_atom_mass(atom_type));
            topo.charges
                .push(get_atom_charge(&atom.residue_name, &atom.name));
            topo.lj_params.push(get_lj_param(atom_type));
        }

        // Step 2: Build atom lookup by (residue_id, chain_id, atom_name)
        let mut atom_lookup: HashMap<(i32, char, String), usize> = HashMap::new();
        let mut h_atoms_count = 0;
        for (i, atom) in atoms.iter().enumerate() {
            let name = atom.name.trim().to_string();
            if name.starts_with('H') {
                h_atoms_count += 1;
            }
            atom_lookup.insert((atom.residue_id, atom.chain_id, name), i);
        }
        log::info!(
            "Topology: {} total atoms, {} hydrogen atoms in lookup",
            atoms.len(),
            h_atoms_count
        );

        // Step 3: Detect bonds
        let mut bond_set: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();

        // 3a: Intra-residue bonds from templates
        let residues = Self::group_by_residue(atoms);
        for (res_key, res_atoms) in &residues {
            let (res_id, chain_id) = res_key;
            if res_atoms.is_empty() {
                continue;
            }
            let res_name = &res_atoms[0].residue_name;

            if let Some(template) = get_residue_template(res_name) {
                for (name1, name2) in template.bonds {
                    let key1 = (*res_id, *chain_id, name1.to_string());
                    let key2 = (*res_id, *chain_id, name2.to_string());

                    let idx1 = atom_lookup.get(&key1);
                    let idx2 = atom_lookup.get(&key2);

                    if let (Some(&i1), Some(&i2)) = (idx1, idx2) {
                        let (a, b) = if i1 < i2 {
                            (i1 as u32, i2 as u32)
                        } else {
                            (i2 as u32, i1 as u32)
                        };
                        bond_set.insert((a, b));
                    } else if name1.starts_with('H') || name2.starts_with('H') {
                        // Log failed hydrogen bond lookups (first few only)
                        static LOGGED: std::sync::atomic::AtomicUsize =
                            std::sync::atomic::AtomicUsize::new(0);
                        if LOGGED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 5 {
                            log::warn!("H bond lookup failed: res {} chain {} atoms {}-{} (found: {:?}, {:?})",
                                res_id, chain_id, name1, name2, idx1.is_some(), idx2.is_some());
                        }
                    }
                }
            }
        }

        // 3b: Inter-residue peptide bonds (C-N between consecutive residues)
        let mut sorted_res: Vec<_> = residues.keys().collect();
        sorted_res.sort();

        for i in 0..sorted_res.len().saturating_sub(1) {
            let (res1_id, chain1) = sorted_res[i];
            let (res2_id, chain2) = sorted_res[i + 1];

            // Only connect if same chain and consecutive
            if chain1 == chain2 && *res2_id == *res1_id + 1 {
                let key_c = (*res1_id, *chain1, "C".to_string());
                let key_n = (*res2_id, *chain2, "N".to_string());

                if let (Some(&idx_c), Some(&idx_n)) =
                    (atom_lookup.get(&key_c), atom_lookup.get(&key_n))
                {
                    let (a, b) = if idx_c < idx_n {
                        (idx_c as u32, idx_n as u32)
                    } else {
                        (idx_n as u32, idx_c as u32)
                    };
                    bond_set.insert((a, b));
                }
            }
        }

        // 3c: Disulfide bonds (CYS SG-SG within 2.5 Å)
        let mut cys_sg: Vec<(usize, f32, f32, f32)> = Vec::new();
        for atom in atoms {
            if (atom.residue_name == "CYS" || atom.residue_name == "CYX")
                && atom.name.trim() == "SG"
            {
                cys_sg.push((atom.index, atom.x, atom.y, atom.z));
            }
        }

        for i in 0..cys_sg.len() {
            for j in (i + 1)..cys_sg.len() {
                let (idx1, x1, y1, z1) = cys_sg[i];
                let (idx2, x2, y2, z2) = cys_sg[j];
                let dx = x2 - x1;
                let dy = y2 - y1;
                let dz = z2 - z1;
                let dist_sq = dx * dx + dy * dy + dz * dz;
                if dist_sq < 6.25 {
                    // 2.5^2
                    let (a, b) = if idx1 < idx2 {
                        (idx1 as u32, idx2 as u32)
                    } else {
                        (idx2 as u32, idx1 as u32)
                    };
                    bond_set.insert((a, b));
                }
            }
        }

        // Convert bonds to vectors with parameters
        let mut h_bonds_found = 0;
        let mut h_bonds_missing_params = 0;

        for (a, b) in &bond_set {
            let type1 = topo.atom_types[*a as usize];
            let type2 = topo.atom_types[*b as usize];

            // Track hydrogen bonds
            let is_h_bond = matches!(
                type1,
                AmberAtomType::H
                    | AmberAtomType::H1
                    | AmberAtomType::HC
                    | AmberAtomType::HP
                    | AmberAtomType::HA
                    | AmberAtomType::HO
            ) || matches!(
                type2,
                AmberAtomType::H
                    | AmberAtomType::H1
                    | AmberAtomType::HC
                    | AmberAtomType::HP
                    | AmberAtomType::HA
                    | AmberAtomType::HO
            );

            let param = get_bond_param(type1, type2);

            if is_h_bond {
                if param.is_some() {
                    h_bonds_found += 1;
                } else {
                    h_bonds_missing_params += 1;
                    log::warn!("Missing bond param for H bond: {:?}-{:?}", type1, type2);
                }
            }

            let param = param.unwrap_or(BondParam {
                r0: 1.5,
                k: 300.0, // Default values if not found
            });

            topo.bonds.push((*a, *b));
            topo.bond_params.push(param);
        }

        log::info!(
            "Topology: {} bonds total, {} H-bonds with params, {} H-bonds missing params",
            bond_set.len(),
            h_bonds_found,
            h_bonds_missing_params
        );

        // Step 4: Build bond graph for angle/dihedral detection
        let mut bond_graph: HashMap<u32, Vec<u32>> = HashMap::new();
        for (a, b) in &topo.bonds {
            bond_graph.entry(*a).or_default().push(*b);
            bond_graph.entry(*b).or_default().push(*a);
        }

        // Step 5: Generate angles (i-j-k where i-j and j-k are bonds)
        let mut angle_set: std::collections::HashSet<(u32, u32, u32)> =
            std::collections::HashSet::new();

        for (&center, neighbors) in &bond_graph {
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let atom_i = neighbors[i];
                    let atom_k = neighbors[j];
                    // Canonicalize: smaller index first
                    let (a, c) = if atom_i < atom_k {
                        (atom_i, atom_k)
                    } else {
                        (atom_k, atom_i)
                    };
                    angle_set.insert((a, center, c));
                }
            }
        }

        for (a, b, c) in &angle_set {
            let type1 = topo.atom_types[*a as usize];
            let type2 = topo.atom_types[*b as usize];
            let type3 = topo.atom_types[*c as usize];

            // Try both orderings for parameter lookup
            let param = get_angle_param(type1, type2, type3)
                .or_else(|| get_angle_param(type3, type2, type1))
                .unwrap_or(AngleParam {
                    theta0: 109.5 * PI / 180.0,
                    k: 50.0, // Default tetrahedral angle
                });

            topo.angles.push((*a, *b, *c));
            topo.angle_params.push(param);
        }

        // Step 6: Generate dihedrals (i-j-k-l where i-j, j-k, k-l are bonds)
        let mut dihedral_set: std::collections::HashSet<(u32, u32, u32, u32)> =
            std::collections::HashSet::new();

        // For each bond j-k (the central bond)
        for (j, k) in &topo.bonds {
            // Find all atoms bonded to j (except k)
            let neighbors_j: Vec<u32> = bond_graph
                .get(j)
                .map(|v| v.iter().filter(|&&x| x != *k).cloned().collect())
                .unwrap_or_default();

            // Find all atoms bonded to k (except j)
            let neighbors_k: Vec<u32> = bond_graph
                .get(k)
                .map(|v| v.iter().filter(|&&x| x != *j).cloned().collect())
                .unwrap_or_default();

            // Create all i-j-k-l dihedrals
            for &i in &neighbors_j {
                for &l in &neighbors_k {
                    // Canonicalize: ensure j < k, and if j == k (shouldn't happen), i < l
                    let (a, b, c, d) = if j < k {
                        (i, *j, *k, l)
                    } else {
                        (l, *k, *j, i)
                    };
                    dihedral_set.insert((a, b, c, d));
                }
            }
        }

        for (a, b, c, d) in &dihedral_set {
            let type1 = topo.atom_types[*a as usize];
            let type2 = topo.atom_types[*b as usize];
            let type3 = topo.atom_types[*c as usize];
            let type4 = topo.atom_types[*d as usize];

            // Try both orderings for parameter lookup
            let params = {
                let p = get_dihedral_params(type1, type2, type3, type4);
                if !p.is_empty() {
                    p
                } else {
                    let p_rev = get_dihedral_params(type4, type3, type2, type1);
                    if !p_rev.is_empty() {
                        p_rev
                    } else {
                        // Default dihedral
                        vec![DihedralParam {
                            k: 0.0,
                            n: 2,
                            phase: 0.0,
                            paths: 1,
                        }]
                    }
                }
            };

            topo.dihedrals.push((*a, *b, *c, *d));
            topo.dihedral_params.push(params);
        }

        // Step 7: Generate 1-4 pairs (atoms at dihedral ends)
        let mut pair_14_set: std::collections::HashSet<(u32, u32)> =
            std::collections::HashSet::new();
        for (a, _, _, d) in &topo.dihedrals {
            let (i, j) = if a < d { (*a, *d) } else { (*d, *a) };
            pair_14_set.insert((i, j));
        }
        topo.pairs_14 = pair_14_set.into_iter().collect();

        // Step 8: Generate exclusions (1-2 and 1-3 pairs)
        let mut exclusion_set: std::collections::HashSet<(u32, u32)> =
            std::collections::HashSet::new();

        // 1-2 exclusions (bonded pairs)
        for (a, b) in &topo.bonds {
            let (i, j) = if a < b { (*a, *b) } else { (*b, *a) };
            exclusion_set.insert((i, j));
        }

        // 1-3 exclusions (angle ends)
        for (a, _, c) in &topo.angles {
            let (i, j) = if a < c { (*a, *c) } else { (*c, *a) };
            exclusion_set.insert((i, j));
        }

        topo.exclusions = exclusion_set.into_iter().collect();

        topo
    }

    /// Group atoms by residue (residue_id, chain_id)
    fn group_by_residue(atoms: &[PdbAtom]) -> HashMap<(i32, char), Vec<&PdbAtom>> {
        let mut groups: HashMap<(i32, char), Vec<&PdbAtom>> = HashMap::new();
        for atom in atoms {
            groups
                .entry((atom.residue_id, atom.chain_id))
                .or_default()
                .push(atom);
        }
        groups
    }

    /// Build topology for coarse-grained (CA-only) model using parameter-free ANM (pfANM)
    ///
    /// SOTA Implementation based on Yang et al. (2009) "Protein elastic network models
    /// and the ranges of cooperativity" PNAS 106(30):12347-12352
    ///
    /// Key improvements over uniform ENM:
    /// 1. Distance-dependent spring constants: k ∝ 1/r² (pfANM)
    /// 2. Backbone enhancement: 2x stronger springs for sequential residues
    /// 3. Extended cutoff: 15Å (ANM standard) vs 10Å (GNM standard)
    ///
    /// Expected correlation with experimental B-factors: ρ ≈ 0.55-0.65
    pub fn from_ca_only(ca_atoms: &[PdbAtom], cutoff_angstrom: f32) -> Self {
        let mut topo = AmberTopology::default();
        topo.n_atoms = ca_atoms.len();

        // pfANM spring constant scaling factor (kcal/mol·Å²)
        // Calibrated for HMC sampling at 310K with 8Å cutoff
        //
        // With sparse network (fewer springs), need stronger individual springs
        // to maintain appropriate total restraint while preserving flexibility differences
        //
        // Target: RMSF ~ sqrt(kT / k_eff) where k_eff is effective spring constant
        // At T=310K, kT ≈ 0.62 kcal/mol
        // For RMSF ~ 0.5-1.0 Å: k_eff ~ 0.5-2.0 kcal/mol/Å²
        //
        // With pfANM k = C/r², at r=5Å: k = C/25
        // For k ~ 0.2 kcal/mol/Å²: C ~ 5 (balance between stiffness and acceptance)
        const PFANM_C: f32 = 5.0;

        // Backbone enhancement factor for sequential residues
        // Sequential CA-CA bonds are ~3.8Å and have stronger coupling
        const BACKBONE_ENHANCEMENT: f32 = 2.0;

        // All CA atoms get CT type with average residue mass
        for _atom in ca_atoms {
            topo.atom_types.push(AmberAtomType::CT);
            topo.masses.push(110.0); // Average residue mass in Daltons
            topo.charges.push(0.0); // No electrostatics in CG (handled by ENM)
            topo.lj_params.push(get_lj_param(AmberAtomType::CT));
        }

        // Parameter-free ANM: connect all pairs within cutoff with k = C/r²
        let cutoff_sq = cutoff_angstrom * cutoff_angstrom;

        for i in 0..ca_atoms.len() {
            for j in (i + 1)..ca_atoms.len() {
                let dx = ca_atoms[j].x - ca_atoms[i].x;
                let dy = ca_atoms[j].y - ca_atoms[i].y;
                let dz = ca_atoms[j].z - ca_atoms[i].z;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq && dist_sq > 1.0 {
                    // Avoid division by tiny distances
                    let dist = dist_sq.sqrt();

                    // pfANM: k = C / r² (distance-dependent spring constant)
                    // This naturally captures the hierarchy of protein interactions:
                    // - Nearby residues: strong coupling
                    // - Distant residues: weak coupling
                    let k_base = PFANM_C / dist_sq;

                    // Backbone enhancement: sequential residues have stronger coupling
                    // This improves agreement with all-atom MD (Lezon & Bahar, 2010)
                    let is_sequential = (j - i) == 1;
                    let k_final = if is_sequential {
                        k_base * BACKBONE_ENHANCEMENT
                    } else {
                        k_base
                    };

                    topo.bonds.push((i as u32, j as u32));
                    topo.bond_params.push(BondParam {
                        r0: dist,
                        k: k_final,
                    });
                }
            }
        }

        // Log statistics
        if !topo.bonds.is_empty() {
            let k_values: Vec<f32> = topo.bond_params.iter().map(|p| p.k).collect();
            let k_min = k_values.iter().cloned().fold(f32::INFINITY, f32::min);
            let k_max = k_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let k_mean = k_values.iter().sum::<f32>() / k_values.len() as f32;
            log::debug!(
                "pfANM topology: {} springs, k range [{:.3}, {:.3}], mean {:.3} kcal/mol/Å²",
                topo.bonds.len(),
                k_min,
                k_max,
                k_mean
            );
        }

        // No angles/dihedrals for CG model (ENM captures collective motions)
        topo
    }

    /// Print topology summary
    pub fn summary(&self) -> String {
        format!(
            "AmberTopology: {} atoms, {} bonds, {} angles, {} dihedrals, {} 1-4 pairs, {} exclusions",
            self.n_atoms,
            self.bonds.len(),
            self.angles.len(),
            self.dihedrals.len(),
            self.pairs_14.len(),
            self.exclusions.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atom_typing() {
        assert_eq!(AmberAtomType::from_pdb("ALA", "N"), AmberAtomType::N);
        assert_eq!(AmberAtomType::from_pdb("ALA", "CA"), AmberAtomType::CT);
        assert_eq!(AmberAtomType::from_pdb("ALA", "C"), AmberAtomType::C);
        assert_eq!(AmberAtomType::from_pdb("ALA", "O"), AmberAtomType::O);
        assert_eq!(AmberAtomType::from_pdb("ALA", "CB"), AmberAtomType::CT);
        assert_eq!(AmberAtomType::from_pdb("PHE", "CG"), AmberAtomType::CA);
        assert_eq!(AmberAtomType::from_pdb("CYS", "SG"), AmberAtomType::SH);
    }

    #[test]
    fn test_bond_params() {
        // Test backbone N-CA bond
        let param = get_bond_param(AmberAtomType::N, AmberAtomType::CT);
        assert!(param.is_some(), "N-CT bond param should exist");
        let param = param.unwrap_or(BondParam { k: 0.0, r0: 0.0 });
        assert!((param.r0 - 1.449).abs() < 0.01);
        assert!((param.k - 337.0).abs() < 1.0);

        // Test peptide bond C-N
        let param = get_bond_param(AmberAtomType::C, AmberAtomType::N);
        assert!(param.is_some(), "C-N bond param should exist");
        let param = param.unwrap_or(BondParam { k: 0.0, r0: 0.0 });
        assert!((param.r0 - 1.335).abs() < 0.01);
    }

    #[test]
    fn test_angle_params() {
        // Test N-CA-C angle
        let param = get_angle_param(AmberAtomType::N, AmberAtomType::CT, AmberAtomType::C);
        assert!(param.is_some(), "N-CT-C angle param should exist");
        let param = param.unwrap_or(AngleParam {
            k: 0.0,
            theta0: 0.0,
        });
        assert!((param.theta0 - 110.1 * PI / 180.0).abs() < 0.01);
    }

    #[test]
    fn test_charges() {
        // Test backbone charges
        assert!((get_atom_charge("ALA", "N") - (-0.4157)).abs() < 0.001);
        assert!((get_atom_charge("ALA", "C") - 0.5973).abs() < 0.001);

        // Test sidechain charges
        assert!((get_atom_charge("LYS", "NZ") - (-0.3854)).abs() < 0.001);
    }

    #[test]
    fn test_topology_generation_dipeptide() {
        // Create a simple ALA-GLY dipeptide
        let atoms = vec![
            // ALA residue 1
            PdbAtom {
                index: 0,
                name: "N".to_string(),
                residue_name: "ALA".to_string(),
                residue_id: 1,
                chain_id: 'A',
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            PdbAtom {
                index: 1,
                name: "CA".to_string(),
                residue_name: "ALA".to_string(),
                residue_id: 1,
                chain_id: 'A',
                x: 1.449,
                y: 0.0,
                z: 0.0,
            },
            PdbAtom {
                index: 2,
                name: "C".to_string(),
                residue_name: "ALA".to_string(),
                residue_id: 1,
                chain_id: 'A',
                x: 2.0,
                y: 1.5,
                z: 0.0,
            },
            PdbAtom {
                index: 3,
                name: "O".to_string(),
                residue_name: "ALA".to_string(),
                residue_id: 1,
                chain_id: 'A',
                x: 1.5,
                y: 2.5,
                z: 0.0,
            },
            PdbAtom {
                index: 4,
                name: "CB".to_string(),
                residue_name: "ALA".to_string(),
                residue_id: 1,
                chain_id: 'A',
                x: 1.8,
                y: -0.8,
                z: 1.2,
            },
            // GLY residue 2
            PdbAtom {
                index: 5,
                name: "N".to_string(),
                residue_name: "GLY".to_string(),
                residue_id: 2,
                chain_id: 'A',
                x: 3.3,
                y: 1.5,
                z: 0.0,
            },
            PdbAtom {
                index: 6,
                name: "CA".to_string(),
                residue_name: "GLY".to_string(),
                residue_id: 2,
                chain_id: 'A',
                x: 4.0,
                y: 2.7,
                z: 0.0,
            },
            PdbAtom {
                index: 7,
                name: "C".to_string(),
                residue_name: "GLY".to_string(),
                residue_id: 2,
                chain_id: 'A',
                x: 5.5,
                y: 2.7,
                z: 0.0,
            },
            PdbAtom {
                index: 8,
                name: "O".to_string(),
                residue_name: "GLY".to_string(),
                residue_id: 2,
                chain_id: 'A',
                x: 6.2,
                y: 1.7,
                z: 0.0,
            },
        ];

        let topo = AmberTopology::from_pdb_atoms(&atoms);

        // Verify basic counts
        assert_eq!(topo.n_atoms, 9);
        assert!(topo.bonds.len() > 0, "Should have bonds");
        assert!(topo.angles.len() > 0, "Should have angles");
        assert!(topo.dihedrals.len() > 0, "Should have dihedrals");

        // Check that peptide bond was detected (C of res1 to N of res2)
        let has_peptide_bond = topo
            .bonds
            .iter()
            .any(|(a, b)| (*a == 2 && *b == 5) || (*a == 5 && *b == 2));
        assert!(
            has_peptide_bond,
            "Should detect peptide bond between residues"
        );

        // Check that atom types were assigned correctly
        assert_eq!(topo.atom_types[0], AmberAtomType::N);
        assert_eq!(topo.atom_types[1], AmberAtomType::CT);
        assert_eq!(topo.atom_types[2], AmberAtomType::C);
        assert_eq!(topo.atom_types[3], AmberAtomType::O);

        println!("{}", topo.summary());
    }

    #[test]
    fn test_ca_only_topology() {
        // Create CA-only atoms
        let ca_atoms = vec![
            PdbAtom {
                index: 0,
                name: "CA".to_string(),
                residue_name: "ALA".to_string(),
                residue_id: 1,
                chain_id: 'A',
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            PdbAtom {
                index: 1,
                name: "CA".to_string(),
                residue_name: "GLY".to_string(),
                residue_id: 2,
                chain_id: 'A',
                x: 3.8,
                y: 0.0,
                z: 0.0,
            },
            PdbAtom {
                index: 2,
                name: "CA".to_string(),
                residue_name: "VAL".to_string(),
                residue_id: 3,
                chain_id: 'A',
                x: 7.6,
                y: 0.0,
                z: 0.0,
            },
            PdbAtom {
                index: 3,
                name: "CA".to_string(),
                residue_name: "LEU".to_string(),
                residue_id: 4,
                chain_id: 'A',
                x: 11.4,
                y: 0.0,
                z: 0.0,
            },
        ];

        // 10 Å cutoff
        let topo = AmberTopology::from_ca_only(&ca_atoms, 10.0);

        assert_eq!(topo.n_atoms, 4);
        // At 10 Å cutoff, CA atoms at 3.8 Å apart should all be connected
        // Bonds: 0-1 (3.8Å), 0-2 (7.6Å), 1-2 (3.8Å), 1-3 (7.6Å), 2-3 (3.8Å)
        // 0-3 is 11.4Å - outside cutoff
        assert!(topo.bonds.len() >= 3, "Should have at least 3 bonds");

        println!("CA-only: {}", topo.summary());
    }

    #[test]
    fn test_gpu_topology_conversion() {
        // Create minimal topology
        let mut topo = AmberTopology::default();
        topo.n_atoms = 4;
        topo.bonds = vec![(0, 1), (1, 2), (2, 3)];
        topo.bond_params = vec![
            BondParam { r0: 1.5, k: 300.0 },
            BondParam { r0: 1.5, k: 300.0 },
            BondParam { r0: 1.5, k: 300.0 },
        ];
        topo.angles = vec![(0, 1, 2), (1, 2, 3)];
        topo.angle_params = vec![
            AngleParam {
                theta0: 1.91,
                k: 50.0,
            },
            AngleParam {
                theta0: 1.91,
                k: 50.0,
            },
        ];

        let gpu_topo = GpuTopology::from_amber(&topo);

        assert_eq!(gpu_topo.n_bonds, 3);
        assert_eq!(gpu_topo.n_angles, 2);
        assert_eq!(gpu_topo.bond_list.len(), 6); // 3 bonds * 2 indices
        assert_eq!(gpu_topo.angle_list.len(), 6); // 2 angles * 3 indices
    }
}
