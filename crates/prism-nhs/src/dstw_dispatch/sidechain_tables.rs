//! Per-amino-acid side-chain descriptors used by the rigid-backbone
//! Δq / ΔV substitution in `projection.rs`.
//!
//! The numbers below are the AMBER ff14SB residue net charges (sum of
//! side-chain partial charges, equal to the residue's formal charge at
//! physiological pH for the canonical 20 amino acids) and Bondi
//! side-chain volumes in cubic angstroms.  These are the values
//! distributed with AmberTools and reported in:
//!
//!   Maier, J. A., et al. *ff14SB: Improving the Accuracy of Protein
//!   Side Chain and Backbone Parameters from ff99SB.*  J. Chem. Theory
//!   Comput. 2015, 11, 3696.
//!
//!   Bondi, A. *van der Waals Volumes and Radii.*  J. Phys. Chem. 1964,
//!   68, 441.
//!
//! Pro is included; the rigid-backbone substitution refuses Pro→X / X→Pro
//! mutations because the backbone amide H is missing / present respectively
//! and the rigid-backbone assumption breaks down there.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AminoAcid {
    A,
    R,
    N,
    D,
    C,
    Q,
    E,
    G,
    H,
    I,
    L,
    K,
    M,
    F,
    P,
    S,
    T,
    W,
    Y,
    V,
}

impl AminoAcid {
    pub fn from_one_letter(c: char) -> Result<Self, String> {
        Ok(match c.to_ascii_uppercase() {
            'A' => AminoAcid::A,
            'R' => AminoAcid::R,
            'N' => AminoAcid::N,
            'D' => AminoAcid::D,
            'C' => AminoAcid::C,
            'Q' => AminoAcid::Q,
            'E' => AminoAcid::E,
            'G' => AminoAcid::G,
            'H' => AminoAcid::H,
            'I' => AminoAcid::I,
            'L' => AminoAcid::L,
            'K' => AminoAcid::K,
            'M' => AminoAcid::M,
            'F' => AminoAcid::F,
            'P' => AminoAcid::P,
            'S' => AminoAcid::S,
            'T' => AminoAcid::T,
            'W' => AminoAcid::W,
            'Y' => AminoAcid::Y,
            'V' => AminoAcid::V,
            other => return Err(format!("unrecognised amino acid {:?}", other)),
        })
    }

    pub fn as_char(self) -> char {
        match self {
            AminoAcid::A => 'A',
            AminoAcid::R => 'R',
            AminoAcid::N => 'N',
            AminoAcid::D => 'D',
            AminoAcid::C => 'C',
            AminoAcid::Q => 'Q',
            AminoAcid::E => 'E',
            AminoAcid::G => 'G',
            AminoAcid::H => 'H',
            AminoAcid::I => 'I',
            AminoAcid::L => 'L',
            AminoAcid::K => 'K',
            AminoAcid::M => 'M',
            AminoAcid::F => 'F',
            AminoAcid::P => 'P',
            AminoAcid::S => 'S',
            AminoAcid::T => 'T',
            AminoAcid::W => 'W',
            AminoAcid::Y => 'Y',
            AminoAcid::V => 'V',
        }
    }
}

/// Side-chain electrostatic + steric descriptors.
///
/// `net_charge` is the formal charge on the side chain at pH 7
/// (Arg / Lys = +1; Asp / Glu = -1; His = 0 protonated as neutral
/// for the dual-tautomer average; all others = 0).  AMBER ff14SB
/// partial charges sum to slightly off-integer values; we use the
/// formal charge here because the projection only cares about the
/// gross electrostatic shift on side-chain substitution.
///
/// `volume_angstrom3` is the Bondi side-chain volume (heavy atoms +
/// hydrogen contribution).  Gly = 0 by convention (no side chain
/// beyond the backbone Cα-H).
#[derive(Debug, Clone, Copy)]
pub struct SidechainDescriptor {
    pub net_charge: f64,
    pub volume_angstrom3: f64,
}

impl AminoAcid {
    pub fn descriptor(self) -> SidechainDescriptor {
        match self {
            AminoAcid::A => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 16.8,
            },
            AminoAcid::R => SidechainDescriptor {
                net_charge: 1.0,
                volume_angstrom3: 105.1,
            },
            AminoAcid::N => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 58.7,
            },
            AminoAcid::D => SidechainDescriptor {
                net_charge: -1.0,
                volume_angstrom3: 54.6,
            },
            AminoAcid::C => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 44.6,
            },
            AminoAcid::Q => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 78.7,
            },
            AminoAcid::E => SidechainDescriptor {
                net_charge: -1.0,
                volume_angstrom3: 74.7,
            },
            AminoAcid::G => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 0.0,
            },
            AminoAcid::H => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 79.0,
            },
            AminoAcid::I => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 66.9,
            },
            AminoAcid::L => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 66.9,
            },
            AminoAcid::K => SidechainDescriptor {
                net_charge: 1.0,
                volume_angstrom3: 97.1,
            },
            AminoAcid::M => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 83.7,
            },
            AminoAcid::F => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 88.2,
            },
            AminoAcid::P => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 44.6,
            },
            AminoAcid::S => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 25.3,
            },
            AminoAcid::T => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 44.0,
            },
            AminoAcid::W => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 113.6,
            },
            AminoAcid::Y => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 98.2,
            },
            AminoAcid::V => SidechainDescriptor {
                net_charge: 0.0,
                volume_angstrom3: 50.5,
            },
        }
    }
}

/// Returns `(delta_q, delta_v)` for `wildtype -> mutant`.
///
/// Sign convention: `delta_q = q_mut - q_wt`, `delta_v = v_mut - v_wt`.
/// A K→A mutation has `delta_q = -1` (loss of positive charge) and
/// `delta_v = 16.8 - 97.1 = -80.3` (significant volume loss).
pub fn delta_q_v(wildtype: AminoAcid, mutant: AminoAcid) -> (f64, f64) {
    let wt = wildtype.descriptor();
    let mu = mutant.descriptor();
    (
        mu.net_charge - wt.net_charge,
        mu.volume_angstrom3 - wt.volume_angstrom3,
    )
}

/// Rigid-backbone substitution is invalid for any Pro↔non-Pro swap.
pub fn rigid_backbone_compatible(wildtype: AminoAcid, mutant: AminoAcid) -> bool {
    !(wildtype == AminoAcid::P || mutant == AminoAcid::P)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lysine_to_alanine_drops_charge_and_volume() {
        let (dq, dv) = delta_q_v(AminoAcid::K, AminoAcid::A);
        assert!((dq - (-1.0)).abs() < 1e-9);
        assert!(dv < 0.0); // 16.8 - 97.1 = -80.3
        assert!((dv - (16.8 - 97.1)).abs() < 1e-9);
    }

    #[test]
    fn aspartate_to_alanine_drops_negative_charge() {
        let (dq, _) = delta_q_v(AminoAcid::D, AminoAcid::A);
        assert!(
            (dq - 1.0).abs() < 1e-9,
            "D→A should DROP a negative charge => dq = +1"
        );
    }

    #[test]
    fn proline_substitutions_are_blocked() {
        assert!(!rigid_backbone_compatible(AminoAcid::P, AminoAcid::A));
        assert!(!rigid_backbone_compatible(AminoAcid::A, AminoAcid::P));
        assert!(rigid_backbone_compatible(AminoAcid::L, AminoAcid::A));
    }

    #[test]
    fn glycine_volume_is_zero() {
        assert_eq!(AminoAcid::G.descriptor().volume_angstrom3, 0.0);
    }

    #[test]
    fn one_letter_round_trip() {
        for c in "ARNDCQEGHILKMFPSTWYV".chars() {
            let aa = AminoAcid::from_one_letter(c).unwrap();
            assert_eq!(aa.as_char(), c);
        }
    }
}
