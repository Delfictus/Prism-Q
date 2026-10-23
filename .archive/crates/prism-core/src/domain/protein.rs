//! Protein Structure Parser
//!
//! Parses PDB/CIF files and extracts residue contact graphs for graph coloring

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Amino acid residue with 3D coordinates
#[derive(Debug, Clone)]
pub struct Residue {
    pub chain: char,
    pub residue_number: i32,
    pub residue_name: String,
    pub ca_coords: Option<(f64, f64, f64)>, // C-alpha coordinates
    pub atoms: Vec<Atom>,
}

/// Individual atom
#[derive(Debug, Clone)]
pub struct Atom {
    pub name: String,
    pub coords: (f64, f64, f64),
}

/// Protein contact graph
#[derive(Debug)]
pub struct ProteinContactGraph {
    pub residues: Vec<Residue>,
    pub contacts: Vec<(usize, usize)>, // Edges between residue indices
    pub num_vertices: usize,
    pub num_edges: usize,
}

impl ProteinContactGraph {
    /// Parse PDB file and build contact graph
    pub fn from_pdb_file<P: AsRef<Path>>(path: P, contact_distance: f64) -> Result<Self> {
        let file = File::open(path.as_ref()).context("Failed to open PDB file")?;
        let reader = BufReader::new(file);

        Self::parse_pdb(reader, contact_distance)
    }

    /// Parse PDB format from a reader
    fn parse_pdb<R: BufRead>(reader: R, contact_distance: f64) -> Result<Self> {
        let mut residues_map: HashMap<(char, i32), Residue> = HashMap::new();

        // Parse ATOM records
        for line in reader.lines() {
            let line = line?;

            if !line.starts_with("ATOM") {
                continue;
            }

            // PDB format specification:
            // ATOM record: columns 1-6 = "ATOM  "
            // columns 7-11 = atom serial number
            // columns 13-16 = atom name
            // columns 18-20 = residue name
            // column 22 = chain ID
            // columns 23-26 = residue sequence number
            // columns 31-38 = x coordinate
            // columns 39-46 = y coordinate
            // columns 47-54 = z coordinate

            if line.len() < 54 {
                continue;
            }

            let atom_name = line[12..16].trim().to_string();
            let residue_name = line[17..20].trim().to_string();
            let chain = line[21..22].chars().next().unwrap_or('A');

            let residue_number: i32 = match line[22..26].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };

            let x: f64 = match line[30..38].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let y: f64 = match line[38..46].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let z: f64 = match line[46..54].trim().parse() {
                Ok(v) => v,
                Err(_) => continue,
            };

            let coords = (x, y, z);

            // Get or create residue
            let key = (chain, residue_number);
            let residue = residues_map.entry(key).or_insert_with(|| Residue {
                chain,
                residue_number,
                residue_name: residue_name.clone(),
                ca_coords: None,
                atoms: Vec::new(),
            });

            // Store C-alpha coordinates for distance calculation
            if atom_name == "CA" {
                residue.ca_coords = Some(coords);
            }

            // Store all atoms
            residue.atoms.push(Atom {
                name: atom_name,
                coords,
            });
        }

        // Convert to sorted vector
        let mut residues: Vec<Residue> = residues_map.into_values().collect();
        residues.sort_by_key(|r| (r.chain, r.residue_number));

        // Filter out residues without C-alpha
        residues.retain(|r| r.ca_coords.is_some());

        let num_vertices = residues.len();

        // Build contact graph based on C-alpha distances
        let mut contacts = Vec::new();

        for i in 0..num_vertices {
            let ca1 = residues[i].ca_coords.unwrap();

            for j in (i + 1)..num_vertices {
                let ca2 = residues[j].ca_coords.unwrap();

                let distance = euclidean_distance(ca1, ca2);

                // Create edge if residues are in contact
                // Skip sequential neighbors (i+1) as they're always close
                if distance <= contact_distance && (j - i) > 1 {
                    contacts.push((i, j));
                }
            }
        }

        let num_edges = contacts.len();

        Ok(Self {
            residues,
            contacts,
            num_vertices,
            num_edges,
        })
    }

    /// Get edge list for graph coloring
    pub fn get_edges(&self) -> Vec<(usize, usize)> {
        self.contacts.clone()
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let chains: std::collections::HashSet<char> =
            self.residues.iter().map(|r| r.chain).collect();

        format!(
            "{} residues across {} chain(s), {} contacts",
            self.num_vertices,
            chains.len(),
            self.num_edges
        )
    }
}

/// Calculate Euclidean distance between two 3D points
fn euclidean_distance(p1: (f64, f64, f64), p2: (f64, f64, f64)) -> f64 {
    let dx = p1.0 - p2.0;
    let dy = p1.1 - p2.1;
    let dz = p1.2 - p2.2;

    (dx * dx + dy * dy + dz * dz).sqrt()
}
