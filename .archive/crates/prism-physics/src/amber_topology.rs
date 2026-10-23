//! AMBER Topology Generator
//!
//! Generates molecular topology (bonds, angles, dihedrals) from PDB structures.
//! Uses standard amino acid templates to determine connectivity.
//!
//! # Architecture Note
//! This is a NEW file for Phase 2 (AMBER MD). It does NOT modify any locked files.

use std::collections::{HashMap, HashSet};

/// AMBER atom type for force field parameter lookup
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AmberAtomType {
    // Backbone atoms
    N,      // Backbone nitrogen
    H,      // Backbone amide hydrogen
    CA,     // Alpha carbon
    HA,     // Alpha hydrogen
    C,      // Backbone carbonyl carbon
    O,      // Backbone carbonyl oxygen

    // Terminal atoms
    N3,     // N-terminal nitrogen (charged)
    H1,     // N-terminal hydrogen
    OXT,    // C-terminal oxygen

    // Sidechain - aliphatic
    CT,     // sp3 carbon (general)
    HC,     // Hydrogen on CT
    HP,     // Hydrogen on aromatic carbon

    // Sidechain - polar
    OH,     // Hydroxyl oxygen (Ser, Thr, Tyr)
    HO,     // Hydroxyl hydrogen
    SH,     // Thiol sulfur (Cys)
    HS,     // Thiol hydrogen
    S,      // Sulfur (Met, disulfide)

    // Sidechain - aromatic
    CW,     // Aromatic carbon (Trp indole)
    CV,     // Aromatic carbon (His)
    CR,     // Aromatic carbon (His)
    NB,     // Aromatic nitrogen (His)
    NA,     // Aromatic nitrogen with H

    // Sidechain - charged
    N2,     // Guanidinium nitrogen (Arg)
    N3Plus, // Charged nitrogen (Lys)
    O2,     // Carboxylate oxygen (Asp, Glu)

    // Special
    Unknown,
}

impl AmberAtomType {
    /// Get AMBER atom type from residue name and atom name
    pub fn from_pdb(residue: &str, atom: &str) -> Self {
        let atom = atom.trim();
        let residue = residue.trim().to_uppercase();

        // Backbone atoms (same for all residues)
        match atom {
            "N" => return AmberAtomType::N,
            "H" | "HN" => return AmberAtomType::H,
            "CA" => return AmberAtomType::CA,
            "HA" | "HA2" | "HA3" => return AmberAtomType::HA,
            "C" => return AmberAtomType::C,
            "O" | "OC1" => return AmberAtomType::O,
            "OXT" | "OC2" => return AmberAtomType::OXT,
            _ => {}
        }

        // Sidechain atoms by residue type
        match residue.as_str() {
            "ALA" => match atom {
                "CB" => AmberAtomType::CT,
                "HB1" | "HB2" | "HB3" => AmberAtomType::HC,
                _ => AmberAtomType::Unknown,
            },
            "GLY" => AmberAtomType::Unknown, // No sidechain
            "SER" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::HC,
                "OG" => AmberAtomType::OH,
                "HG" => AmberAtomType::HO,
                _ => AmberAtomType::Unknown,
            },
            "CYS" => match atom {
                "CB" => AmberAtomType::CT,
                "HB2" | "HB3" => AmberAtomType::HC,
                "SG" => AmberAtomType::SH,
                "HG" => AmberAtomType::HS,
                _ => AmberAtomType::Unknown,
            },
            "PRO" => match atom {
                "CB" | "CG" | "CD" => AmberAtomType::CT,
                _ if atom.starts_with('H') => AmberAtomType::HC,
                _ => AmberAtomType::Unknown,
            },
            "VAL" | "ILE" | "LEU" => match atom {
                "CB" | "CG" | "CG1" | "CG2" | "CD" | "CD1" => AmberAtomType::CT,
                _ if atom.starts_with('H') => AmberAtomType::HC,
                _ => AmberAtomType::Unknown,
            },
            "MET" => match atom {
                "CB" | "CG" | "CE" => AmberAtomType::CT,
                "SD" => AmberAtomType::S,
                _ if atom.starts_with('H') => AmberAtomType::HC,
                _ => AmberAtomType::Unknown,
            },
            "PHE" | "TYR" | "TRP" => match atom {
                "CB" => AmberAtomType::CT,
                "CG" | "CD1" | "CD2" | "CE1" | "CE2" | "CZ" | "CE3" | "CZ2" | "CZ3" | "CH2" => AmberAtomType::CW,
                "OH" => AmberAtomType::OH,
                "HH" => AmberAtomType::HO,
                "NE1" => AmberAtomType::NA,
                _ if atom.starts_with('H') => AmberAtomType::HP,
                _ => AmberAtomType::Unknown,
            },
            "HIS" => match atom {
                "CB" => AmberAtomType::CT,
                "CG" | "CE1" => AmberAtomType::CR,
                "CD2" => AmberAtomType::CV,
                "ND1" => AmberAtomType::NB,
                "NE2" => AmberAtomType::NA,
                _ if atom.starts_with('H') => AmberAtomType::HP,
                _ => AmberAtomType::Unknown,
            },
            "LYS" => match atom {
                "CB" | "CG" | "CD" | "CE" => AmberAtomType::CT,
                "NZ" => AmberAtomType::N3Plus,
                _ if atom.starts_with('H') => AmberAtomType::HC,
                _ => AmberAtomType::Unknown,
            },
            "ARG" => match atom {
                "CB" | "CG" | "CD" => AmberAtomType::CT,
                "NE" | "NH1" | "NH2" => AmberAtomType::N2,
                "CZ" => AmberAtomType::CA, // Guanidinium carbon
                _ if atom.starts_with('H') => AmberAtomType::HC,
                _ => AmberAtomType::Unknown,
            },
            "ASP" | "GLU" => match atom {
                "CB" | "CG" => AmberAtomType::CT,
                "OD1" | "OD2" | "OE1" | "OE2" => AmberAtomType::O2,
                "CD" | "CG" => AmberAtomType::C, // Carboxyl carbon
                _ if atom.starts_with('H') => AmberAtomType::HC,
                _ => AmberAtomType::Unknown,
            },
            "ASN" | "GLN" => match atom {
                "CB" | "CG" => AmberAtomType::CT,
                "OD1" | "OE1" => AmberAtomType::O,
                "ND2" | "NE2" => AmberAtomType::N,
                "CD" | "CG" => AmberAtomType::C,
                _ if atom.starts_with('H') => AmberAtomType::HC,
                _ => AmberAtomType::Unknown,
            },
            "THR" => match atom {
                "CB" => AmberAtomType::CT,
                "OG1" => AmberAtomType::OH,
                "CG2" => AmberAtomType::CT,
                "HG1" => AmberAtomType::HO,
                _ if atom.starts_with('H') => AmberAtomType::HC,
                _ => AmberAtomType::Unknown,
            },
            _ => AmberAtomType::Unknown,
        }
    }
}

/// Atom in the topology
#[derive(Debug, Clone)]
pub struct TopologyAtom {
    pub index: usize,
    pub name: String,
    pub residue_name: String,
    pub residue_index: usize,
    pub chain_id: char,
    pub position: [f32; 3],
    pub amber_type: AmberAtomType,
    pub mass: f32,
    pub charge: f32,
}

/// Bond between two atoms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TopologyBond {
    pub atom1: usize,
    pub atom2: usize,
    pub order: u8, // 1 = single, 2 = double, 3 = aromatic
}

/// Angle between three atoms (atom1-atom2-atom3, where atom2 is central)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TopologyAngle {
    pub atom1: usize,
    pub atom2: usize, // Central atom
    pub atom3: usize,
}

/// Proper dihedral (torsion) between four atoms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TopologyDihedral {
    pub atom1: usize,
    pub atom2: usize,
    pub atom3: usize,
    pub atom4: usize,
    pub is_improper: bool,
}

/// Complete molecular topology
#[derive(Debug, Clone)]
pub struct AmberTopology {
    pub atoms: Vec<TopologyAtom>,
    pub bonds: Vec<TopologyBond>,
    pub angles: Vec<TopologyAngle>,
    pub dihedrals: Vec<TopologyDihedral>,
    pub pairs_14: Vec<(usize, usize)>, // 1-4 pairs for scaled interactions
    pub exclusions: HashSet<(usize, usize)>, // 1-2 and 1-3 pairs to exclude
}

impl AmberTopology {
    /// Create topology from parsed atoms
    pub fn from_atoms(atoms: Vec<TopologyAtom>) -> Self {
        let mut topology = Self {
            atoms,
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            pairs_14: Vec::new(),
            exclusions: HashSet::new(),
        };

        topology.generate_bonds();
        topology.generate_angles();
        topology.generate_dihedrals();
        topology.generate_exclusions();

        topology
    }

    /// Generate bonds from amino acid templates and distances
    fn generate_bonds(&mut self) {
        let n = self.atoms.len();

        // Build residue groups
        let mut residue_atoms: HashMap<(char, usize), Vec<usize>> = HashMap::new();
        for (i, atom) in self.atoms.iter().enumerate() {
            residue_atoms
                .entry((atom.chain_id, atom.residue_index))
                .or_default()
                .push(i);
        }

        // Generate intra-residue bonds from templates
        for ((chain, res_idx), atom_indices) in &residue_atoms {
            self.add_residue_bonds(atom_indices);
        }

        // Generate inter-residue peptide bonds (C-N)
        self.add_peptide_bonds(&residue_atoms);

        // Detect disulfide bonds (CYS SG-SG < 2.5Å)
        self.add_disulfide_bonds();
    }

    /// Add bonds within a single residue using templates
    fn add_residue_bonds(&mut self, atom_indices: &[usize]) {
        if atom_indices.is_empty() { return; }

        // Build name -> index map for this residue
        let mut name_to_idx: HashMap<&str, usize> = HashMap::new();
        for &idx in atom_indices {
            name_to_idx.insert(&self.atoms[idx].name, idx);
        }

        let residue_name = &self.atoms[atom_indices[0]].residue_name;

        // Backbone bonds (same for all residues except PRO)
        let backbone_bonds = if residue_name == "PRO" {
            // Proline: N is bonded to CD (ring)
            vec![
                ("N", "CA"), ("CA", "C"), ("C", "O"),
                ("CA", "HA"), ("N", "CD"),
            ]
        } else {
            vec![
                ("N", "H"), ("N", "CA"), ("CA", "C"), ("C", "O"),
                ("CA", "HA"),
            ]
        };

        for (a1, a2) in backbone_bonds {
            if let (Some(&i1), Some(&i2)) = (name_to_idx.get(a1), name_to_idx.get(a2)) {
                self.bonds.push(TopologyBond { atom1: i1, atom2: i2, order: 1 });
            }
        }

        // Sidechain bonds by residue type
        let sidechain_bonds: Vec<(&str, &str)> = match residue_name.as_str() {
            "ALA" => vec![("CA", "CB"), ("CB", "HB1"), ("CB", "HB2"), ("CB", "HB3")],
            "GLY" => vec![("CA", "HA2"), ("CA", "HA3")], // Two HA
            "SER" => vec![("CA", "CB"), ("CB", "OG"), ("OG", "HG"), ("CB", "HB2"), ("CB", "HB3")],
            "CYS" => vec![("CA", "CB"), ("CB", "SG"), ("SG", "HG"), ("CB", "HB2"), ("CB", "HB3")],
            "VAL" => vec![
                ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CB", "HB"),
                ("CG1", "HG11"), ("CG1", "HG12"), ("CG1", "HG13"),
                ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23"),
            ],
            "LEU" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
                ("CB", "HB2"), ("CB", "HB3"), ("CG", "HG"),
                ("CD1", "HD11"), ("CD1", "HD12"), ("CD1", "HD13"),
                ("CD2", "HD21"), ("CD2", "HD22"), ("CD2", "HD23"),
            ],
            "ILE" => vec![
                ("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1"),
                ("CB", "HB"), ("CG1", "HG12"), ("CG1", "HG13"),
                ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23"),
                ("CD1", "HD11"), ("CD1", "HD12"), ("CD1", "HD13"),
            ],
            "MET" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE"),
                ("CB", "HB2"), ("CB", "HB3"), ("CG", "HG2"), ("CG", "HG3"),
                ("CE", "HE1"), ("CE", "HE2"), ("CE", "HE3"),
            ],
            "PRO" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N"),
                ("CB", "HB2"), ("CB", "HB3"), ("CG", "HG2"), ("CG", "HG3"),
                ("CD", "HD2"), ("CD", "HD3"),
            ],
            "PHE" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
                ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
                ("CB", "HB2"), ("CB", "HB3"),
                ("CD1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("CE2", "HE2"), ("CZ", "HZ"),
            ],
            "TYR" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
                ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
                ("CZ", "OH"), ("OH", "HH"),
                ("CB", "HB2"), ("CB", "HB3"),
                ("CD1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("CE2", "HE2"),
            ],
            "TRP" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
                ("CD1", "NE1"), ("NE1", "CE2"), ("CD2", "CE2"), ("CD2", "CE3"),
                ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"), ("CZ3", "CH2"),
                ("NE1", "HE1"),
                ("CB", "HB2"), ("CB", "HB3"), ("CD1", "HD1"),
                ("CE3", "HE3"), ("CZ2", "HZ2"), ("CZ3", "HZ3"), ("CH2", "HH2"),
            ],
            "HIS" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"),
                ("ND1", "CE1"), ("CD2", "NE2"), ("CE1", "NE2"),
                ("CB", "HB2"), ("CB", "HB3"),
                ("ND1", "HD1"), ("CD2", "HD2"), ("CE1", "HE1"), ("NE2", "HE2"),
            ],
            "LYS" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ"),
                ("CB", "HB2"), ("CB", "HB3"), ("CG", "HG2"), ("CG", "HG3"),
                ("CD", "HD2"), ("CD", "HD3"), ("CE", "HE2"), ("CE", "HE3"),
                ("NZ", "HZ1"), ("NZ", "HZ2"), ("NZ", "HZ3"),
            ],
            "ARG" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"), ("NE", "CZ"),
                ("CZ", "NH1"), ("CZ", "NH2"),
                ("CB", "HB2"), ("CB", "HB3"), ("CG", "HG2"), ("CG", "HG3"),
                ("CD", "HD2"), ("CD", "HD3"), ("NE", "HE"),
                ("NH1", "HH11"), ("NH1", "HH12"), ("NH2", "HH21"), ("NH2", "HH22"),
            ],
            "ASP" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2"),
                ("CB", "HB2"), ("CB", "HB3"),
            ],
            "GLU" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2"),
                ("CB", "HB2"), ("CB", "HB3"), ("CG", "HG2"), ("CG", "HG3"),
            ],
            "ASN" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2"),
                ("ND2", "HD21"), ("ND2", "HD22"),
                ("CB", "HB2"), ("CB", "HB3"),
            ],
            "GLN" => vec![
                ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2"),
                ("NE2", "HE21"), ("NE2", "HE22"),
                ("CB", "HB2"), ("CB", "HB3"), ("CG", "HG2"), ("CG", "HG3"),
            ],
            "THR" => vec![
                ("CA", "CB"), ("CB", "OG1"), ("CB", "CG2"), ("OG1", "HG1"), ("CB", "HB"),
                ("CG2", "HG21"), ("CG2", "HG22"), ("CG2", "HG23"),
            ],
            _ => vec![],
        };

        for (a1, a2) in sidechain_bonds {
            if let (Some(&i1), Some(&i2)) = (name_to_idx.get(a1), name_to_idx.get(a2)) {
                self.bonds.push(TopologyBond { atom1: i1, atom2: i2, order: 1 });
            }
        }
    }

    /// Add peptide bonds between consecutive residues
    fn add_peptide_bonds(&mut self, residue_atoms: &HashMap<(char, usize), Vec<usize>>) {
        // Find C and N atoms for each residue
        let mut c_atoms: HashMap<(char, usize), usize> = HashMap::new();
        let mut n_atoms: HashMap<(char, usize), usize> = HashMap::new();

        for (i, atom) in self.atoms.iter().enumerate() {
            let key = (atom.chain_id, atom.residue_index);
            if atom.name == "C" {
                c_atoms.insert(key, i);
            } else if atom.name == "N" {
                n_atoms.insert(key, i);
            }
        }

        // Connect C(i) to N(i+1) within same chain
        for ((chain, res_idx), &c_idx) in &c_atoms {
            let next_key = (*chain, res_idx + 1);
            if let Some(&n_idx) = n_atoms.get(&next_key) {
                // Verify distance is reasonable (< 2.0 Å)
                let dist = self.distance(c_idx, n_idx);
                if dist < 2.0 {
                    self.bonds.push(TopologyBond { atom1: c_idx, atom2: n_idx, order: 1 });
                }
            }
        }
    }

    /// Detect and add disulfide bonds
    fn add_disulfide_bonds(&mut self) {
        // Find all CYS SG atoms
        let sg_atoms: Vec<usize> = self.atoms.iter()
            .enumerate()
            .filter(|(_, a)| a.residue_name == "CYS" && a.name == "SG")
            .map(|(i, _)| i)
            .collect();

        // Check all pairs for disulfide distance (< 2.5 Å)
        for i in 0..sg_atoms.len() {
            for j in (i + 1)..sg_atoms.len() {
                let dist = self.distance(sg_atoms[i], sg_atoms[j]);
                if dist < 2.5 {
                    self.bonds.push(TopologyBond {
                        atom1: sg_atoms[i],
                        atom2: sg_atoms[j],
                        order: 1,
                    });
                }
            }
        }
    }

    /// Generate angles from connected bonds
    fn generate_angles(&mut self) {
        // Build adjacency list
        let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
        for bond in &self.bonds {
            neighbors.entry(bond.atom1).or_default().push(bond.atom2);
            neighbors.entry(bond.atom2).or_default().push(bond.atom1);
        }

        // For each atom with 2+ neighbors, create angles
        let mut angles_set: HashSet<(usize, usize, usize)> = HashSet::new();

        for (&center, connected) in &neighbors {
            if connected.len() < 2 { continue; }

            for i in 0..connected.len() {
                for j in (i + 1)..connected.len() {
                    let (a1, a3) = if connected[i] < connected[j] {
                        (connected[i], connected[j])
                    } else {
                        (connected[j], connected[i])
                    };
                    angles_set.insert((a1, center, a3));
                }
            }
        }

        self.angles = angles_set.into_iter()
            .map(|(a1, a2, a3)| TopologyAngle { atom1: a1, atom2: a2, atom3: a3 })
            .collect();
    }

    /// Generate dihedrals from connected angles
    fn generate_dihedrals(&mut self) {
        // Build bond adjacency
        let mut bonded: HashSet<(usize, usize)> = HashSet::new();
        for bond in &self.bonds {
            let (a, b) = if bond.atom1 < bond.atom2 {
                (bond.atom1, bond.atom2)
            } else {
                (bond.atom2, bond.atom1)
            };
            bonded.insert((a, b));
        }

        let is_bonded = |a: usize, b: usize| -> bool {
            let (x, y) = if a < b { (a, b) } else { (b, a) };
            bonded.contains(&(x, y))
        };

        // Build neighbor list
        let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
        for bond in &self.bonds {
            neighbors.entry(bond.atom1).or_default().push(bond.atom2);
            neighbors.entry(bond.atom2).or_default().push(bond.atom1);
        }

        // Find proper dihedrals: i-j-k-l where i-j, j-k, k-l are bonded
        let mut dihedral_set: HashSet<(usize, usize, usize, usize)> = HashSet::new();

        for bond in &self.bonds {
            let j = bond.atom1;
            let k = bond.atom2;

            // Find atoms bonded to j (but not k)
            let j_neighbors = neighbors.get(&j).cloned().unwrap_or_default();
            let k_neighbors = neighbors.get(&k).cloned().unwrap_or_default();

            for &i in &j_neighbors {
                if i == k { continue; }
                for &l in &k_neighbors {
                    if l == j || l == i { continue; }

                    // Canonical ordering
                    let (a1, a2, a3, a4) = if i < l {
                        (i, j, k, l)
                    } else {
                        (l, k, j, i)
                    };
                    dihedral_set.insert((a1, a2, a3, a4));
                }
            }
        }

        self.dihedrals = dihedral_set.into_iter()
            .map(|(a1, a2, a3, a4)| TopologyDihedral {
                atom1: a1, atom2: a2, atom3: a3, atom4: a4,
                is_improper: false,
            })
            .collect();

        // Add improper dihedrals for planar groups (backbone, aromatics)
        self.add_improper_dihedrals(&neighbors);
    }

    /// Add improper dihedrals for planar groups
    fn add_improper_dihedrals(&mut self, neighbors: &HashMap<usize, Vec<usize>>) {
        for (i, atom) in self.atoms.iter().enumerate() {
            // Backbone C (carbonyl) - needs to stay planar
            if atom.name == "C" {
                if let Some(nbrs) = neighbors.get(&i) {
                    if nbrs.len() >= 3 {
                        // Improper: CA-C-O-N (keeps carbonyl planar)
                        self.dihedrals.push(TopologyDihedral {
                            atom1: nbrs[0],
                            atom2: i,
                            atom3: nbrs[1],
                            atom4: nbrs[2],
                            is_improper: true,
                        });
                    }
                }
            }

            // Aromatic carbons
            if matches!(atom.amber_type, AmberAtomType::CW | AmberAtomType::CV | AmberAtomType::CR) {
                if let Some(nbrs) = neighbors.get(&i) {
                    if nbrs.len() >= 3 {
                        self.dihedrals.push(TopologyDihedral {
                            atom1: nbrs[0],
                            atom2: i,
                            atom3: nbrs[1],
                            atom4: nbrs[2],
                            is_improper: true,
                        });
                    }
                }
            }
        }
    }

    /// Generate 1-4 pairs and exclusions
    fn generate_exclusions(&mut self) {
        // Build adjacency for path finding
        let mut neighbors: HashMap<usize, Vec<usize>> = HashMap::new();
        for bond in &self.bonds {
            neighbors.entry(bond.atom1).or_default().push(bond.atom2);
            neighbors.entry(bond.atom2).or_default().push(bond.atom1);
        }

        let n = self.atoms.len();

        // 1-2 pairs (directly bonded)
        for bond in &self.bonds {
            let (a, b) = if bond.atom1 < bond.atom2 {
                (bond.atom1, bond.atom2)
            } else {
                (bond.atom2, bond.atom1)
            };
            self.exclusions.insert((a, b));
        }

        // 1-3 pairs (separated by 2 bonds)
        for angle in &self.angles {
            let (a, c) = if angle.atom1 < angle.atom3 {
                (angle.atom1, angle.atom3)
            } else {
                (angle.atom3, angle.atom1)
            };
            self.exclusions.insert((a, c));
        }

        // 1-4 pairs (separated by 3 bonds) - these get scaled interactions
        for dih in &self.dihedrals {
            if dih.is_improper { continue; }

            let (a, d) = if dih.atom1 < dih.atom4 {
                (dih.atom1, dih.atom4)
            } else {
                (dih.atom4, dih.atom1)
            };

            // Only add if not already excluded (not also 1-2 or 1-3)
            if !self.exclusions.contains(&(a, d)) {
                self.pairs_14.push((a, d));
            }
        }
    }

    /// Calculate distance between two atoms
    fn distance(&self, i: usize, j: usize) -> f32 {
        let a = &self.atoms[i].position;
        let b = &self.atoms[j].position;
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Get summary statistics
    pub fn summary(&self) -> TopologySummary {
        TopologySummary {
            n_atoms: self.atoms.len(),
            n_bonds: self.bonds.len(),
            n_angles: self.angles.len(),
            n_dihedrals: self.dihedrals.iter().filter(|d| !d.is_improper).count(),
            n_impropers: self.dihedrals.iter().filter(|d| d.is_improper).count(),
            n_pairs_14: self.pairs_14.len(),
            n_exclusions: self.exclusions.len(),
        }
    }
}

/// Summary statistics for topology
#[derive(Debug, Clone)]
pub struct TopologySummary {
    pub n_atoms: usize,
    pub n_bonds: usize,
    pub n_angles: usize,
    pub n_dihedrals: usize,
    pub n_impropers: usize,
    pub n_pairs_14: usize,
    pub n_exclusions: usize,
}

impl std::fmt::Display for TopologySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Topology: {} atoms, {} bonds, {} angles, {} dihedrals ({} improper), {} 1-4 pairs",
               self.n_atoms, self.n_bonds, self.n_angles, self.n_dihedrals,
               self.n_impropers, self.n_pairs_14)
    }
}

/// Parse PDB file and create topology
pub fn parse_pdb_topology(content: &str, chain_filter: Option<char>) -> AmberTopology {
    let mut atoms = Vec::new();
    let mut last_res_key = String::new();
    let mut residue_counter = 0usize;

    for line in content.lines() {
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            continue;
        }

        // Parse PDB fields
        let atom_name = line.get(12..16).unwrap_or("").trim();
        let residue_name = line.get(17..20).unwrap_or("").trim();
        let chain_id = line.get(21..22).unwrap_or(" ").chars().next().unwrap_or(' ');
        let res_seq = line.get(22..26).unwrap_or("0").trim();

        // Apply chain filter
        if let Some(target_chain) = chain_filter {
            if chain_id != target_chain {
                continue;
            }
        }

        // Handle alternate conformations
        let alt_loc = line.get(16..17).unwrap_or(" ");
        if alt_loc != " " && alt_loc != "A" {
            continue;
        }

        // Track residue changes
        let res_key = format!("{}{}", chain_id, res_seq);
        if res_key != last_res_key {
            residue_counter += 1;
            last_res_key = res_key;
        }

        // Parse coordinates
        let x: f32 = line.get(30..38).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let y: f32 = line.get(38..46).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let z: f32 = line.get(46..54).unwrap_or("0").trim().parse().unwrap_or(0.0);

        // Get AMBER atom type
        let amber_type = AmberAtomType::from_pdb(residue_name, atom_name);

        // Get mass from atom type
        let mass = atom_mass(atom_name);

        atoms.push(TopologyAtom {
            index: atoms.len(),
            name: atom_name.to_string(),
            residue_name: residue_name.to_string(),
            residue_index: residue_counter,
            chain_id,
            position: [x, y, z],
            amber_type,
            mass,
            charge: 0.0, // Will be set from force field
        });
    }

    AmberTopology::from_atoms(atoms)
}

/// Get atomic mass from atom name
fn atom_mass(atom_name: &str) -> f32 {
    let first_char = atom_name.chars().next().unwrap_or('X');
    match first_char {
        'H' => 1.008,
        'C' => 12.011,
        'N' => 14.007,
        'O' => 15.999,
        'S' => 32.065,
        'P' => 30.974,
        _ => 12.0, // Default to carbon mass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alanine_topology() {
        // Minimal alanine PDB
        let pdb = r#"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760   1.220  1.00  0.00           C
"#;

        let topology = parse_pdb_topology(pdb, None);

        assert_eq!(topology.atoms.len(), 5);
        assert!(topology.bonds.len() >= 4); // N-CA, CA-C, C-O, CA-CB minimum
        assert!(!topology.angles.is_empty());
    }

    #[test]
    fn test_peptide_bond() {
        // Two-residue peptide
        let pdb = r#"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   GLY A   2       3.300   1.600   0.000  1.00  0.00           N
ATOM      6  CA  GLY A   2       3.900   2.900   0.000  1.00  0.00           C
ATOM      7  C   GLY A   2       5.400   2.900   0.000  1.00  0.00           C
ATOM      8  O   GLY A   2       6.000   3.950   0.000  1.00  0.00           O
"#;

        let topology = parse_pdb_topology(pdb, None);

        // Should have peptide bond between C(1) and N(2)
        let has_peptide_bond = topology.bonds.iter().any(|b| {
            let a1 = &topology.atoms[b.atom1];
            let a2 = &topology.atoms[b.atom2];
            (a1.name == "C" && a2.name == "N" && a1.residue_index == 1 && a2.residue_index == 2) ||
            (a1.name == "N" && a2.name == "C" && a1.residue_index == 2 && a2.residue_index == 1)
        });

        assert!(has_peptide_bond, "Peptide bond should be detected");
    }

    #[test]
    fn test_amber_atom_type() {
        assert_eq!(AmberAtomType::from_pdb("ALA", "N"), AmberAtomType::N);
        assert_eq!(AmberAtomType::from_pdb("ALA", "CA"), AmberAtomType::CA);
        assert_eq!(AmberAtomType::from_pdb("ALA", "CB"), AmberAtomType::CT);
        assert_eq!(AmberAtomType::from_pdb("SER", "OG"), AmberAtomType::OH);
        assert_eq!(AmberAtomType::from_pdb("CYS", "SG"), AmberAtomType::SH);
    }
}
