//! Solvation Box Builder
//!
//! Provides tools for solvating protein structures with explicit TIP3P water
//! molecules and counterions for charge neutralization.
//!
//! The solvation process:
//! 1. Calculate protein bounding box
//! 2. Add padding to create water box dimensions
//! 3. Place water molecules on a grid, removing overlaps
//! 4. Add counterions (Na+/Cl-) to neutralize net charge
//! 5. Generate combined topology

use crate::amber_ff14sb::{
    get_atom_mass, get_lj_param, AmberAtomType, AmberTopology, AngleParam, BondParam, LJParam,
};
use crate::water_model::{distance_sq, Ion, IonType, TIP3PWater, WaterMolecule};

// ============================================================================
// SOLVATION BOX CONFIGURATION
// ============================================================================

/// Configuration for solvation box building
#[derive(Debug, Clone)]
pub struct SolvationConfig {
    /// Padding distance from protein to box edge (Å)
    pub padding: f32,
    /// Minimum distance between water and protein atoms (Å)
    pub min_protein_distance: f32,
    /// Minimum distance between water molecules (Å)
    pub min_water_distance: f32,
    /// Water density target (g/mL)
    pub target_density: f32,
    /// Maximum box dimension (Å) - prevents memory explosion
    pub max_box_dimension: f32,
    /// Salt concentration for ionic strength (M) - adds extra Na+/Cl-
    pub salt_concentration: f32,
}

impl Default for SolvationConfig {
    fn default() -> Self {
        Self {
            padding: 10.0,
            min_protein_distance: 2.8,
            min_water_distance: 2.5,
            target_density: 0.997, // TIP3P water at 298K
            max_box_dimension: 100.0,
            salt_concentration: 0.0, // No extra salt by default
        }
    }
}

// ============================================================================
// SOLVATION BOX
// ============================================================================

/// A solvation box containing protein, water, and ions
#[derive(Debug, Clone)]
pub struct SolvationBox {
    /// Original protein atom positions
    pub protein_positions: Vec<[f32; 3]>,
    /// Original protein atom types
    pub protein_types: Vec<AmberAtomType>,
    /// Original protein charges
    pub protein_charges: Vec<f32>,
    /// Original protein masses
    pub protein_masses: Vec<f32>,

    /// Water molecules added
    pub waters: Vec<WaterMolecule>,

    /// Ions for neutralization and ionic strength
    pub ions: Vec<Ion>,

    /// Box dimensions [x, y, z] in Å
    pub box_dimensions: [f32; 3],

    /// Box origin (minimum corner)
    pub box_origin: [f32; 3],

    /// Total number of atoms (protein + water + ions)
    pub total_atoms: usize,

    /// Indices of water oxygen atoms (for SETTLE)
    pub water_oxygen_indices: Vec<usize>,
}

impl SolvationBox {
    /// Create a solvation box from protein positions
    ///
    /// # Arguments
    /// * `positions` - Protein atom positions as flat array [x0, y0, z0, x1, y1, z1, ...]
    /// * `atom_types` - AMBER atom types for each protein atom
    /// * `charges` - Partial charges for each protein atom
    /// * `config` - Solvation configuration
    pub fn from_protein(
        positions: &[f32],
        atom_types: &[AmberAtomType],
        charges: &[f32],
        config: &SolvationConfig,
    ) -> anyhow::Result<Self> {
        let n_protein_atoms = atom_types.len();

        if positions.len() != n_protein_atoms * 3 {
            anyhow::bail!(
                "Position array length {} doesn't match {} atoms * 3",
                positions.len(),
                n_protein_atoms
            );
        }

        // Convert flat positions to [x,y,z] arrays
        let protein_positions: Vec<[f32; 3]> = positions
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        // Get masses
        let protein_masses: Vec<f32> = atom_types.iter().map(|&t| get_atom_mass(t)).collect();

        // Calculate bounding box
        let (min_corner, max_corner) = bounding_box(&protein_positions);

        // Add padding
        let box_origin = [
            min_corner[0] - config.padding,
            min_corner[1] - config.padding,
            min_corner[2] - config.padding,
        ];
        let box_dimensions = [
            (max_corner[0] - min_corner[0] + 2.0 * config.padding).min(config.max_box_dimension),
            (max_corner[1] - min_corner[1] + 2.0 * config.padding).min(config.max_box_dimension),
            (max_corner[2] - min_corner[2] + 2.0 * config.padding).min(config.max_box_dimension),
        ];

        log::info!(
            "Solvation box: origin=[{:.1}, {:.1}, {:.1}], dims=[{:.1}, {:.1}, {:.1}] Å",
            box_origin[0],
            box_origin[1],
            box_origin[2],
            box_dimensions[0],
            box_dimensions[1],
            box_dimensions[2]
        );

        Ok(Self {
            protein_positions,
            protein_types: atom_types.to_vec(),
            protein_charges: charges.to_vec(),
            protein_masses,
            waters: Vec::new(),
            ions: Vec::new(),
            box_dimensions,
            box_origin,
            total_atoms: n_protein_atoms,
            water_oxygen_indices: Vec::new(),
        })
    }

    /// Add water molecules to fill the box
    ///
    /// Returns the number of waters added
    pub fn add_waters(&mut self, config: &SolvationConfig) -> usize {
        let tip3p = TIP3PWater::new();

        // Grid spacing for water placement (based on density)
        // Water density = 0.997 g/mL = 0.0334 mol/Å³
        // Volume per water = 29.9 Å³, cube root ≈ 3.1 Å
        let grid_spacing = 3.1_f32;

        let nx = (self.box_dimensions[0] / grid_spacing).ceil() as usize;
        let ny = (self.box_dimensions[1] / grid_spacing).ceil() as usize;
        let nz = (self.box_dimensions[2] / grid_spacing).ceil() as usize;

        // Add buffer for hydrogen atoms: H extends ~0.96 Å from O, so add 2*0.96 ≈ 2.0 Å
        // This ensures even H-H contacts are above the minimum distance
        const H_EXTENSION: f32 = 1.0; // Conservative buffer for H atoms
        let effective_protein_dist = config.min_protein_distance + H_EXTENSION;
        let effective_water_dist = config.min_water_distance + 2.0 * H_EXTENSION; // Both waters have H

        let min_dist_sq_protein = effective_protein_dist * effective_protein_dist;
        let min_dist_sq_water = effective_water_dist * effective_water_dist;

        let mut placed = 0;
        let mut seed = 12345u64; // Deterministic RNG seed

        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let center = [
                        self.box_origin[0] + (ix as f32 + 0.5) * grid_spacing,
                        self.box_origin[1] + (iy as f32 + 0.5) * grid_spacing,
                        self.box_origin[2] + (iz as f32 + 0.5) * grid_spacing,
                    ];

                    // Check distance to protein atoms
                    let mut too_close = false;
                    for pos in &self.protein_positions {
                        if distance_sq(center, *pos) < min_dist_sq_protein {
                            too_close = true;
                            break;
                        }
                    }
                    if too_close {
                        continue;
                    }

                    // Check distance to existing waters
                    for water in &self.waters {
                        if distance_sq(center, water.o_pos) < min_dist_sq_water {
                            too_close = true;
                            break;
                        }
                    }
                    if too_close {
                        continue;
                    }

                    // Place water with random orientation
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let water = WaterMolecule::at_position_random(center, seed);
                    self.waters.push(water);
                    placed += 1;
                }
            }
        }

        // Update total atoms
        self.total_atoms = self.protein_positions.len() + self.waters.len() * 3;

        log::info!("Added {} water molecules ({} atoms)", placed, placed * 3);

        placed
    }

    /// Neutralize the system by adding counterions
    ///
    /// Returns (n_sodium, n_chloride) added
    pub fn neutralize(&mut self, config: &SolvationConfig) -> (usize, usize) {
        // Calculate net charge of protein
        let net_charge: f32 = self.protein_charges.iter().sum();
        let net_charge_rounded = net_charge.round() as i32;

        log::info!(
            "Protein net charge: {:.2} (rounded: {})",
            net_charge,
            net_charge_rounded
        );

        let mut n_sodium = 0;
        let mut n_chloride = 0;

        // Add counterions to neutralize
        if net_charge_rounded > 0 {
            // Add Cl- to neutralize positive charge
            n_chloride = net_charge_rounded as usize;
        } else if net_charge_rounded < 0 {
            // Add Na+ to neutralize negative charge
            n_sodium = (-net_charge_rounded) as usize;
        }

        // Add extra salt if requested
        if config.salt_concentration > 0.0 {
            // Volume in liters
            let volume_l =
                self.box_dimensions[0] * self.box_dimensions[1] * self.box_dimensions[2] * 1e-27; // Å³ to L

            // Number of ion pairs for desired concentration
            let n_pairs = (config.salt_concentration * volume_l * 6.022e23).round() as usize;
            n_sodium += n_pairs;
            n_chloride += n_pairs;
        }

        // Place ions by replacing random waters
        let mut ion_positions: Vec<[f32; 3]> = Vec::new();

        // Place sodium ions
        for _ in 0..n_sodium {
            if let Some(pos) = self.find_ion_position(&ion_positions) {
                self.ions.push(Ion::sodium(pos));
                ion_positions.push(pos);
            }
        }

        // Place chloride ions
        for _ in 0..n_chloride {
            if let Some(pos) = self.find_ion_position(&ion_positions) {
                self.ions.push(Ion::chloride(pos));
                ion_positions.push(pos);
            }
        }

        // Remove waters that overlap with ions
        // CRITICAL: Minimum distance must be >= r_min for ion-water LJ interaction
        // Na+-O: σ_combined ~ 2.65 Å, r_min ~ 3.0 Å
        // Cl--O: σ_combined ~ 4.0 Å, r_min ~ 4.5 Å
        // Use 4.5 Å to accommodate the larger Cl- ion
        let min_dist_sq = 4.5_f32 * 4.5_f32;
        self.waters.retain(|water| {
            !ion_positions
                .iter()
                .any(|ion_pos| distance_sq(water.o_pos, *ion_pos) < min_dist_sq)
        });

        // Update total atoms
        self.total_atoms = self.protein_positions.len() + self.waters.len() * 3 + self.ions.len();

        log::info!("Added {} Na+ and {} Cl- ions", n_sodium, n_chloride);

        (n_sodium, n_chloride)
    }

    /// Find a suitable position for an ion (away from protein and other ions)
    fn find_ion_position(&self, existing_ions: &[[f32; 3]]) -> Option<[f32; 3]> {
        let min_dist_protein_sq = 5.0_f32 * 5.0_f32; // 5 Å from protein
        let min_dist_ion_sq = 4.0_f32 * 4.0_f32; // 4 Å from other ions

        // Try random water positions
        for water in &self.waters {
            let pos = water.o_pos;

            // Check distance to protein
            let mut valid = true;
            for prot_pos in &self.protein_positions {
                if distance_sq(pos, *prot_pos) < min_dist_protein_sq {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }

            // Check distance to existing ions
            for ion_pos in existing_ions {
                if distance_sq(pos, *ion_pos) < min_dist_ion_sq {
                    valid = false;
                    break;
                }
            }
            if !valid {
                continue;
            }

            return Some(pos);
        }

        None
    }

    /// Get indices of water oxygen atoms (for SETTLE constraints)
    pub fn water_oxygen_indices(&self) -> Vec<usize> {
        let protein_atoms = self.protein_positions.len();
        (0..self.waters.len())
            .map(|i| protein_atoms + i * 3) // O is first atom of each water
            .collect()
    }

    /// Generate complete topology including water and ions
    pub fn to_topology(&self) -> AmberTopology {
        let tip3p = TIP3PWater::new();
        let n_protein = self.protein_positions.len();
        let n_waters = self.waters.len();
        let n_ions = self.ions.len();
        let n_total = n_protein + n_waters * 3 + n_ions;

        let mut topo = AmberTopology {
            n_atoms: n_total,
            atom_types: Vec::with_capacity(n_total),
            masses: Vec::with_capacity(n_total),
            charges: Vec::with_capacity(n_total),
            lj_params: Vec::with_capacity(n_total),
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
        };

        // Add protein atoms
        topo.atom_types.extend_from_slice(&self.protein_types);
        topo.masses.extend_from_slice(&self.protein_masses);
        topo.charges.extend_from_slice(&self.protein_charges);
        for &t in &self.protein_types {
            topo.lj_params.push(get_lj_param(t));
        }

        // Add water molecules
        for (wi, water) in self.waters.iter().enumerate() {
            let base_idx = n_protein + wi * 3;

            // Oxygen
            topo.atom_types.push(AmberAtomType::OW);
            topo.masses.push(tip3p.o_mass);
            topo.charges.push(tip3p.o_charge);
            topo.lj_params.push(get_lj_param(AmberAtomType::OW));

            // Hydrogen 1
            topo.atom_types.push(AmberAtomType::HW);
            topo.masses.push(tip3p.h_mass);
            topo.charges.push(tip3p.h_charge);
            topo.lj_params.push(get_lj_param(AmberAtomType::HW));

            // Hydrogen 2
            topo.atom_types.push(AmberAtomType::HW);
            topo.masses.push(tip3p.h_mass);
            topo.charges.push(tip3p.h_charge);
            topo.lj_params.push(get_lj_param(AmberAtomType::HW));

            // CRITICAL: Do NOT add O-H bonds or H-O-H angles for water molecules!
            // SETTLE handles the rigid water geometry as holonomic constraints.
            // If we add bonds/angles here, the force calculation will compute huge forces
            // (k=553 for bonds!) that accelerate water atoms, but SETTLE then moves
            // positions back to the constraint surface. This creates an energy imbalance
            // where forces inject energy that isn't properly dissipated, causing
            // temperature explosion.
            //
            // SETTLE-constrained water only needs:
            // 1. LJ parameters (O-O interactions)
            // 2. Charges (electrostatics via PME)
            // 3. Exclusions (no intramolecular non-bonded)

            let o_idx = base_idx as u32;
            let h1_idx = (base_idx + 1) as u32;
            let h2_idx = (base_idx + 2) as u32;

            // Water exclusions (all atoms exclude each other) - store as pairs
            topo.exclusions.push((o_idx, h1_idx));
            topo.exclusions.push((o_idx, h2_idx));
            topo.exclusions.push((h1_idx, h2_idx));
        }

        // Add ions
        for ion in &self.ions {
            match ion.ion_type {
                IonType::Sodium => {
                    topo.atom_types.push(AmberAtomType::IP);
                    topo.masses.push(ion.mass());
                    topo.charges.push(ion.charge());
                    topo.lj_params.push(LJParam {
                        epsilon: ion.epsilon(),
                        rmin_half: ion.rmin_half(),
                    });
                }
                IonType::Chloride => {
                    topo.atom_types.push(AmberAtomType::IM);
                    topo.masses.push(ion.mass());
                    topo.charges.push(ion.charge());
                    topo.lj_params.push(LJParam {
                        epsilon: ion.epsilon(),
                        rmin_half: ion.rmin_half(),
                    });
                }
            }
        }

        topo
    }

    /// Get all positions as a flat array [x0, y0, z0, x1, ...]
    pub fn all_positions_flat(&self) -> Vec<f32> {
        let n_total = self.protein_positions.len() + self.waters.len() * 3 + self.ions.len();
        let mut positions = Vec::with_capacity(n_total * 3);

        // Protein positions
        for pos in &self.protein_positions {
            positions.extend_from_slice(pos);
        }

        // Water positions (O, H1, H2)
        for water in &self.waters {
            positions.extend_from_slice(&water.o_pos);
            positions.extend_from_slice(&water.h1_pos);
            positions.extend_from_slice(&water.h2_pos);
        }

        // Ion positions
        for ion in &self.ions {
            positions.extend_from_slice(&ion.position);
        }

        positions
    }

    /// Get number of water molecules
    pub fn n_waters(&self) -> usize {
        self.waters.len()
    }

    /// Get number of ions
    pub fn n_ions(&self) -> usize {
        self.ions.len()
    }

    /// Get total system volume in Å³
    pub fn volume(&self) -> f32 {
        self.box_dimensions[0] * self.box_dimensions[1] * self.box_dimensions[2]
    }

    /// Get water density in g/mL
    pub fn water_density(&self) -> f32 {
        let tip3p = TIP3PWater::new();
        let water_mass = self.waters.len() as f32 * tip3p.molecule_mass(); // amu
        let water_mass_g = water_mass / 6.022e23; // grams
        let volume_ml = self.volume() * 1e-24; // mL
        if volume_ml > 0.0 {
            water_mass_g / volume_ml
        } else {
            0.0
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Calculate bounding box of a set of positions
fn bounding_box(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    if positions.is_empty() {
        return ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    }

    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];

    for pos in positions {
        for i in 0..3 {
            min[i] = min[i].min(pos[i]);
            max[i] = max[i].max(pos[i]);
        }
    }

    (min, max)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box() {
        let positions = vec![[0.0, 0.0, 0.0], [10.0, 5.0, 3.0], [5.0, 10.0, 8.0]];

        let (min, max) = bounding_box(&positions);
        assert_eq!(min, [0.0, 0.0, 0.0]);
        assert_eq!(max, [10.0, 10.0, 8.0]);
    }

    #[test]
    fn test_solvation_box_creation() {
        // Simple 3-atom protein
        let positions = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 2.5, 5.0, 0.0];
        let atom_types = vec![AmberAtomType::N, AmberAtomType::CT, AmberAtomType::C];
        let charges = vec![-0.4, 0.1, 0.6];

        let config = SolvationConfig {
            padding: 5.0,
            ..Default::default()
        };

        let solvbox = SolvationBox::from_protein(&positions, &atom_types, &charges, &config)
            .expect("Should create solvation box");

        assert_eq!(solvbox.protein_positions.len(), 3);
        assert!((solvbox.box_dimensions[0] - 15.0).abs() < 0.1); // 5 + 5*2 padding
    }

    #[test]
    fn test_water_addition() {
        let positions = vec![0.0; 30]; // 10 atoms at origin
        let atom_types = vec![AmberAtomType::CT; 10];
        let charges = vec![0.0; 10];

        let config = SolvationConfig {
            padding: 10.0,
            min_protein_distance: 3.0,
            min_water_distance: 2.5,
            max_box_dimension: 30.0,
            ..Default::default()
        };

        let mut solvbox = SolvationBox::from_protein(&positions, &atom_types, &charges, &config)
            .expect("Should create solvation box");

        let n_waters = solvbox.add_waters(&config);
        assert!(n_waters > 0, "Should add some waters");
        assert_eq!(solvbox.waters.len(), n_waters);
    }

    #[test]
    fn test_neutralization() {
        let positions = vec![0.0; 9]; // 3 atoms
        let atom_types = vec![AmberAtomType::N, AmberAtomType::CT, AmberAtomType::C];
        let charges = vec![-1.0, 0.0, -1.0]; // Net charge = -2

        let config = SolvationConfig {
            padding: 15.0,
            max_box_dimension: 40.0,
            ..Default::default()
        };

        let mut solvbox = SolvationBox::from_protein(&positions, &atom_types, &charges, &config)
            .expect("Should create solvation box");

        solvbox.add_waters(&config);
        let (n_na, n_cl) = solvbox.neutralize(&config);

        assert_eq!(n_na, 2, "Should add 2 Na+ to neutralize -2 charge");
        assert_eq!(n_cl, 0, "Should add 0 Cl-");
    }

    #[test]
    fn test_topology_generation() {
        let positions = vec![0.0; 9];
        let atom_types = vec![AmberAtomType::N, AmberAtomType::CT, AmberAtomType::C];
        let charges = vec![-0.4, 0.1, 0.3];

        let config = SolvationConfig {
            padding: 8.0,
            max_box_dimension: 20.0,
            ..Default::default()
        };

        let mut solvbox = SolvationBox::from_protein(&positions, &atom_types, &charges, &config)
            .expect("Should create solvation box");

        solvbox.add_waters(&config);

        let topo = solvbox.to_topology();
        assert_eq!(topo.n_atoms, solvbox.total_atoms);
        assert!(
            topo.atom_types.contains(&AmberAtomType::OW),
            "Should have water oxygens"
        );
        assert!(
            topo.atom_types.contains(&AmberAtomType::HW),
            "Should have water hydrogens"
        );
    }
}
