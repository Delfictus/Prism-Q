//! TIP3P Water Model Implementation
//!
//! Provides the TIP3P (Transferable Intermolecular Potential 3-Point) water model
//! parameters and geometry generation for explicit solvent simulations.
//!
//! Reference: Jorgensen et al. (1983) JCP - "Comparison of simple potential functions
//! for simulating liquid water"
//!
//! TIP3P Parameters:
//! - O charge: -0.834 e
//! - H charge: +0.417 e
//! - O-H bond: 0.9572 Å
//! - H-O-H angle: 104.52°
//! - O LJ: σ = 3.15061 Å, ε = 0.1521 kcal/mol

use std::f32::consts::PI;

// ============================================================================
// TIP3P WATER MODEL PARAMETERS
// ============================================================================

/// TIP3P water model parameters (AMBER-compatible)
#[derive(Debug, Clone, Copy)]
pub struct TIP3PWater {
    /// Oxygen partial charge (e)
    pub o_charge: f32,
    /// Hydrogen partial charge (e)
    pub h_charge: f32,
    /// Oxygen LJ sigma (Å)
    pub o_sigma: f32,
    /// Oxygen LJ epsilon (kcal/mol)
    pub o_epsilon: f32,
    /// Oxygen LJ rmin/2 (Å) - AMBER format
    pub o_rmin_half: f32,
    /// O-H equilibrium bond length (Å)
    pub oh_bond: f32,
    /// H-O-H equilibrium angle (degrees)
    pub hoh_angle_deg: f32,
    /// H-H distance (Å) - derived from geometry
    pub hh_distance: f32,
    /// Oxygen mass (amu)
    pub o_mass: f32,
    /// Hydrogen mass (amu)
    pub h_mass: f32,
}

impl Default for TIP3PWater {
    fn default() -> Self {
        let oh_bond = 0.9572;
        let hoh_angle_deg = 104.52;
        let hoh_angle_rad = hoh_angle_deg * PI / 180.0;

        // H-H distance from law of cosines: c² = a² + b² - 2ab·cos(C)
        // where a = b = OH bond, C = HOH angle
        let hh_distance = (2.0 * oh_bond * oh_bond * (1.0 - hoh_angle_rad.cos())).sqrt();

        Self {
            o_charge: -0.834,
            h_charge: 0.417,
            o_sigma: 3.15061,
            o_epsilon: 0.1521,
            // rmin/2 = sigma * 2^(1/6) / 2
            o_rmin_half: 3.15061 * 2.0_f32.powf(1.0 / 6.0) / 2.0,
            oh_bond,
            hoh_angle_deg,
            hh_distance,
            o_mass: 15.9994,
            h_mass: 1.008,
        }
    }
}

impl TIP3PWater {
    /// Create TIP3P water model with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total water molecule mass (amu)
    pub fn molecule_mass(&self) -> f32 {
        self.o_mass + 2.0 * self.h_mass
    }

    /// Get H-O-H angle in radians
    pub fn hoh_angle_rad(&self) -> f32 {
        self.hoh_angle_deg * PI / 180.0
    }
}

// ============================================================================
// WATER MOLECULE REPRESENTATION
// ============================================================================

/// A single water molecule with 3 atoms
#[derive(Debug, Clone, Copy)]
pub struct WaterMolecule {
    /// Oxygen position [x, y, z] in Å
    pub o_pos: [f32; 3],
    /// First hydrogen position [x, y, z] in Å
    pub h1_pos: [f32; 3],
    /// Second hydrogen position [x, y, z] in Å
    pub h2_pos: [f32; 3],
    /// Global atom index of oxygen (set during solvation)
    pub o_index: usize,
}

impl WaterMolecule {
    /// Generate water molecule at a given center position
    ///
    /// The molecule is placed with:
    /// - Oxygen at center
    /// - Hydrogens in the xy-plane, symmetric about x-axis
    pub fn at_position(center: [f32; 3]) -> Self {
        let tip3p = TIP3PWater::new();
        let half_angle = tip3p.hoh_angle_rad() / 2.0;

        // H1 above x-axis, H2 below
        let h_x = tip3p.oh_bond * half_angle.cos();
        let h_y = tip3p.oh_bond * half_angle.sin();

        Self {
            o_pos: center,
            h1_pos: [center[0] + h_x, center[1] + h_y, center[2]],
            h2_pos: [center[0] + h_x, center[1] - h_y, center[2]],
            o_index: 0,
        }
    }

    /// Generate water molecule with random orientation at center
    pub fn at_position_random(center: [f32; 3], rng_seed: u64) -> Self {
        let tip3p = TIP3PWater::new();
        let half_angle = tip3p.hoh_angle_rad() / 2.0;

        // Simple pseudo-random rotation using seed
        // This is deterministic for reproducibility
        let seed = rng_seed.wrapping_mul(0x5DEECE66D).wrapping_add(0xB);
        let theta = (seed as f32 / u64::MAX as f32) * 2.0 * PI; // Rotation around z
        let phi = ((seed >> 16) as f32 / (u64::MAX >> 16) as f32) * PI; // Tilt

        let h_x = tip3p.oh_bond * half_angle.cos();
        let h_y = tip3p.oh_bond * half_angle.sin();

        // Rotate H1 position
        let (sin_t, cos_t) = theta.sin_cos();
        let (sin_p, cos_p) = phi.sin_cos();

        // Simple rotation: first rotate in xy plane, then tilt
        let h1_local = [h_x, h_y, 0.0];
        let h2_local = [h_x, -h_y, 0.0];

        let rotate = |p: [f32; 3]| -> [f32; 3] {
            // Rotate around z-axis by theta
            let rx = p[0] * cos_t - p[1] * sin_t;
            let ry = p[0] * sin_t + p[1] * cos_t;
            let rz = p[2];

            // Rotate around y-axis by phi
            let rrx = rx * cos_p + rz * sin_p;
            let rry = ry;
            let rrz = -rx * sin_p + rz * cos_p;

            [rrx, rry, rrz]
        };

        let h1_rot = rotate(h1_local);
        let h2_rot = rotate(h2_local);

        Self {
            o_pos: center,
            h1_pos: [
                center[0] + h1_rot[0],
                center[1] + h1_rot[1],
                center[2] + h1_rot[2],
            ],
            h2_pos: [
                center[0] + h2_rot[0],
                center[1] + h2_rot[1],
                center[2] + h2_rot[2],
            ],
            o_index: 0,
        }
    }

    /// Get center of mass
    pub fn center_of_mass(&self) -> [f32; 3] {
        let tip3p = TIP3PWater::new();
        let total_mass = tip3p.molecule_mass();

        [
            (self.o_pos[0] * tip3p.o_mass
                + self.h1_pos[0] * tip3p.h_mass
                + self.h2_pos[0] * tip3p.h_mass)
                / total_mass,
            (self.o_pos[1] * tip3p.o_mass
                + self.h1_pos[1] * tip3p.h_mass
                + self.h2_pos[1] * tip3p.h_mass)
                / total_mass,
            (self.o_pos[2] * tip3p.o_mass
                + self.h1_pos[2] * tip3p.h_mass
                + self.h2_pos[2] * tip3p.h_mass)
                / total_mass,
        ]
    }

    /// Check if molecule overlaps with a point within given distance
    pub fn overlaps_with(&self, point: [f32; 3], min_distance: f32) -> bool {
        let dist_sq = min_distance * min_distance;

        for pos in [self.o_pos, self.h1_pos, self.h2_pos] {
            let dx = pos[0] - point[0];
            let dy = pos[1] - point[1];
            let dz = pos[2] - point[2];
            if dx * dx + dy * dy + dz * dz < dist_sq {
                return true;
            }
        }
        false
    }
}

// ============================================================================
// ION MODELS
// ============================================================================

/// Ion type for system neutralization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IonType {
    /// Sodium cation (Na+)
    Sodium,
    /// Chloride anion (Cl-)
    Chloride,
}

/// An ion particle
#[derive(Debug, Clone, Copy)]
pub struct Ion {
    /// Ion type
    pub ion_type: IonType,
    /// Position [x, y, z] in Å
    pub position: [f32; 3],
    /// Global atom index (set during solvation)
    pub index: usize,
}

impl Ion {
    /// Create a sodium ion at position
    pub fn sodium(position: [f32; 3]) -> Self {
        Self {
            ion_type: IonType::Sodium,
            position,
            index: 0,
        }
    }

    /// Create a chloride ion at position
    pub fn chloride(position: [f32; 3]) -> Self {
        Self {
            ion_type: IonType::Chloride,
            position,
            index: 0,
        }
    }

    /// Get ion charge
    pub fn charge(&self) -> f32 {
        match self.ion_type {
            IonType::Sodium => 1.0,
            IonType::Chloride => -1.0,
        }
    }

    /// Get ion mass (amu)
    pub fn mass(&self) -> f32 {
        match self.ion_type {
            IonType::Sodium => 22.9898,
            IonType::Chloride => 35.453,
        }
    }

    /// Get ion LJ epsilon (kcal/mol) - Joung & Cheatham parameters
    pub fn epsilon(&self) -> f32 {
        match self.ion_type {
            IonType::Sodium => 0.0874393,
            IonType::Chloride => 0.0355910,
        }
    }

    /// Get ion LJ rmin/2 (Å) - Joung & Cheatham parameters
    pub fn rmin_half(&self) -> f32 {
        match self.ion_type {
            IonType::Sodium => 1.212,
            IonType::Chloride => 2.711,
        }
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Calculate distance between two 3D points
pub fn distance(p1: [f32; 3], p2: [f32; 3]) -> f32 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let dz = p2[2] - p1[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Calculate squared distance between two 3D points (faster for comparisons)
pub fn distance_sq(p1: [f32; 3], p2: [f32; 3]) -> f32 {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let dz = p2[2] - p1[2];
    dx * dx + dy * dy + dz * dz
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tip3p_defaults() {
        let tip3p = TIP3PWater::new();

        // Verify standard TIP3P parameters
        assert!((tip3p.o_charge - (-0.834)).abs() < 0.001);
        assert!((tip3p.h_charge - 0.417).abs() < 0.001);
        assert!((tip3p.oh_bond - 0.9572).abs() < 0.0001);
        assert!((tip3p.hoh_angle_deg - 104.52).abs() < 0.01);

        // Verify derived parameters
        assert!((tip3p.hh_distance - 1.5139).abs() < 0.001); // Known value for TIP3P
        assert!((tip3p.molecule_mass() - 18.0154).abs() < 0.001);
    }

    #[test]
    fn test_water_geometry() {
        let water = WaterMolecule::at_position([0.0, 0.0, 0.0]);
        let tip3p = TIP3PWater::new();

        // Verify O-H bond lengths
        let oh1 = distance(water.o_pos, water.h1_pos);
        let oh2 = distance(water.o_pos, water.h2_pos);
        assert!((oh1 - tip3p.oh_bond).abs() < 0.0001);
        assert!((oh2 - tip3p.oh_bond).abs() < 0.0001);

        // Verify H-H distance
        let hh = distance(water.h1_pos, water.h2_pos);
        assert!((hh - tip3p.hh_distance).abs() < 0.001);
    }

    #[test]
    fn test_water_overlap() {
        let water = WaterMolecule::at_position([0.0, 0.0, 0.0]);

        // Point at oxygen should overlap
        assert!(water.overlaps_with([0.0, 0.0, 0.0], 0.5));

        // Point far away should not overlap
        assert!(!water.overlaps_with([10.0, 0.0, 0.0], 2.0));
    }

    #[test]
    fn test_ion_parameters() {
        let na = Ion::sodium([0.0, 0.0, 0.0]);
        let cl = Ion::chloride([0.0, 0.0, 0.0]);

        assert!((na.charge() - 1.0).abs() < 0.001);
        assert!((cl.charge() - (-1.0)).abs() < 0.001);
        assert!((na.mass() - 22.9898).abs() < 0.001);
        assert!((cl.mass() - 35.453).abs() < 0.001);
    }

    #[test]
    fn test_charge_neutrality() {
        let tip3p = TIP3PWater::new();
        let total_charge = tip3p.o_charge + 2.0 * tip3p.h_charge;
        assert!(
            total_charge.abs() < 0.001,
            "TIP3P water should be charge neutral"
        );
    }
}
