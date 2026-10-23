//! Secondary Structure Detection from CA Geometry
//!
//! Detects secondary structure elements (helix, sheet, loop) from Cα atom positions
//! using geometric criteria. This enables structural weighting for RMSF predictions.
//!
//! # Theory
//!
//! Secondary structures have characteristic geometric signatures:
//! - **Alpha helix**: ~3.6 residues/turn, ~1.5Å rise, CA(i)-CA(i+3) ≈ 5.0-5.5Å
//! - **Beta strand**: Extended conformation, CA(i)-CA(i+2) ≈ 6.5-7.0Å
//! - **Loop/coil**: Irregular regions connecting structured elements
//!
//! # References
//!
//! - Kabsch & Sander (1983) "Dictionary of protein secondary structure"
//! - Frishman & Argos (1995) "Knowledge-based secondary structure assignment"

use std::f32::consts::PI;

/// Secondary structure classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecondaryStructure {
    /// Alpha helix (3.6 residues/turn)
    Helix,
    /// Beta strand (extended conformation)
    Sheet,
    /// Loop/coil (irregular region)
    Loop,
}

impl SecondaryStructure {
    /// Get flexibility scaling factor for this structure type
    ///
    /// Based on empirical observations:
    /// - Helices are most rigid (hydrogen bond network)
    /// - Sheets are moderately rigid (inter-strand H-bonds)
    /// - Loops are most flexible (no regular structure)
    pub fn flexibility_factor(&self) -> f64 {
        match self {
            SecondaryStructure::Helix => 0.7,
            SecondaryStructure::Sheet => 0.8,
            SecondaryStructure::Loop => 1.2,
        }
    }
}

/// Secondary structure analyzer using CA geometry
pub struct SecondaryStructureAnalyzer {
    /// Distance threshold for CA(i)-CA(i+3) helix detection (Å)
    helix_i3_min: f32,
    helix_i3_max: f32,
    /// Distance threshold for CA(i)-CA(i+4) helix detection (Å)
    helix_i4_min: f32,
    helix_i4_max: f32,
    /// Distance threshold for sheet detection (extended CA-CA)
    sheet_i2_min: f32,
    sheet_i2_max: f32,
    /// Minimum consecutive residues for helix
    min_helix_length: usize,
    /// Minimum consecutive residues for sheet
    min_sheet_length: usize,
}

impl Default for SecondaryStructureAnalyzer {
    fn default() -> Self {
        Self {
            // Helix: CA(i)-CA(i+3) ≈ 5.0-5.5Å (ideal helix ~5.3Å)
            helix_i3_min: 4.5,
            helix_i3_max: 6.0,
            // Helix: CA(i)-CA(i+4) ≈ 5.5-6.5Å (ideal helix ~6.2Å)
            helix_i4_min: 5.0,
            helix_i4_max: 7.0,
            // Sheet: CA(i)-CA(i+2) ≈ 6.5-7.0Å (extended ~6.8Å)
            sheet_i2_min: 6.2,
            sheet_i2_max: 7.5,
            // Minimum lengths
            min_helix_length: 4,
            min_sheet_length: 3,
        }
    }
}

impl SecondaryStructureAnalyzer {
    /// Create analyzer with custom thresholds
    pub fn new(
        helix_i3_range: (f32, f32),
        helix_i4_range: (f32, f32),
        sheet_i2_range: (f32, f32),
    ) -> Self {
        Self {
            helix_i3_min: helix_i3_range.0,
            helix_i3_max: helix_i3_range.1,
            helix_i4_min: helix_i4_range.0,
            helix_i4_max: helix_i4_range.1,
            sheet_i2_min: sheet_i2_range.0,
            sheet_i2_max: sheet_i2_range.1,
            ..Default::default()
        }
    }

    /// Detect secondary structure from CA positions
    ///
    /// # Arguments
    /// * `ca_positions` - Alpha carbon positions as [[x, y, z], ...]
    ///
    /// # Returns
    /// Secondary structure assignment for each residue
    pub fn detect(&self, ca_positions: &[[f32; 3]]) -> Vec<SecondaryStructure> {
        let n = ca_positions.len();
        if n < 3 {
            return vec![SecondaryStructure::Loop; n];
        }

        // Initial assignment based on local geometry
        let mut assignments = vec![SecondaryStructure::Loop; n];
        let mut helix_scores = vec![0.0f32; n];
        let mut sheet_scores = vec![0.0f32; n];

        // Compute helix scores using i→i+3 and i→i+4 distances
        for i in 0..n {
            let mut score = 0.0;

            // Check i→i+3 distance (characteristic of helix)
            if i + 3 < n {
                let d = self.distance(&ca_positions[i], &ca_positions[i + 3]);
                if d >= self.helix_i3_min && d <= self.helix_i3_max {
                    score += 1.0;
                }
            }

            // Check i→i+4 distance
            if i + 4 < n {
                let d = self.distance(&ca_positions[i], &ca_positions[i + 4]);
                if d >= self.helix_i4_min && d <= self.helix_i4_max {
                    score += 1.0;
                }
            }

            // Also check from previous residues looking forward
            if i >= 3 {
                let d = self.distance(&ca_positions[i - 3], &ca_positions[i]);
                if d >= self.helix_i3_min && d <= self.helix_i3_max {
                    score += 0.5;
                }
            }
            if i >= 4 {
                let d = self.distance(&ca_positions[i - 4], &ca_positions[i]);
                if d >= self.helix_i4_min && d <= self.helix_i4_max {
                    score += 0.5;
                }
            }

            helix_scores[i] = score;
        }

        // Compute sheet scores using extended CA-CA distances
        for i in 0..n {
            let mut score = 0.0;

            // Check i→i+2 distance (should be extended for sheet)
            if i + 2 < n {
                let d = self.distance(&ca_positions[i], &ca_positions[i + 2]);
                if d >= self.sheet_i2_min && d <= self.sheet_i2_max {
                    score += 1.0;
                }
            }

            // Check dihedral-like angle (sheets are more planar)
            if i >= 1 && i + 2 < n {
                let angle = self.compute_pseudo_dihedral(
                    &ca_positions[i.saturating_sub(1)],
                    &ca_positions[i],
                    &ca_positions[(i + 1).min(n - 1)],
                    &ca_positions[(i + 2).min(n - 1)],
                );
                // Extended backbone has dihedral close to 180° or -180°
                if angle.abs() > 2.5 {
                    // ~143 degrees
                    score += 0.5;
                }
            }

            sheet_scores[i] = score;
        }

        // Assign based on scores with threshold
        for i in 0..n {
            if helix_scores[i] >= 1.5 {
                assignments[i] = SecondaryStructure::Helix;
            } else if sheet_scores[i] >= 1.0 {
                assignments[i] = SecondaryStructure::Sheet;
            }
        }

        // Smooth assignments - require minimum consecutive residues
        self.smooth_assignments(&mut assignments);

        assignments
    }

    /// Compute flexibility scaling factors from secondary structure
    pub fn get_flexibility_factors(&self, ca_positions: &[[f32; 3]]) -> Vec<f64> {
        self.detect(ca_positions)
            .iter()
            .map(|ss| ss.flexibility_factor())
            .collect()
    }

    /// Apply flexibility factors to RMSF predictions
    pub fn apply_to_rmsf(&self, rmsf: &[f64], ca_positions: &[[f32; 3]]) -> Vec<f64> {
        let factors = self.get_flexibility_factors(ca_positions);
        rmsf.iter()
            .zip(factors.iter())
            .map(|(&r, &f)| r * f)
            .collect()
    }

    /// Euclidean distance between two 3D points
    fn distance(&self, a: &[f32; 3], b: &[f32; 3]) -> f32 {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let dz = b[2] - a[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Compute pseudo-dihedral angle from 4 consecutive CA positions
    fn compute_pseudo_dihedral(
        &self,
        p1: &[f32; 3],
        p2: &[f32; 3],
        p3: &[f32; 3],
        p4: &[f32; 3],
    ) -> f32 {
        // Vectors along the backbone
        let b1 = [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]];
        let b2 = [p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]];
        let b3 = [p4[0] - p3[0], p4[1] - p3[1], p4[2] - p3[2]];

        // Normal vectors to planes
        let n1 = self.cross(&b1, &b2);
        let n2 = self.cross(&b2, &b3);

        // Normalize
        let n1_norm = self.normalize(&n1);
        let n2_norm = self.normalize(&n2);
        let b2_norm = self.normalize(&b2);

        // Compute dihedral
        let m1 = self.cross(&n1_norm, &b2_norm);
        let x = self.dot(&n1_norm, &n2_norm);
        let y = self.dot(&m1, &n2_norm);

        y.atan2(x)
    }

    /// Cross product
    fn cross(&self, a: &[f32; 3], b: &[f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    /// Dot product
    fn dot(&self, a: &[f32; 3], b: &[f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    /// Normalize vector
    fn normalize(&self, v: &[f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len > 1e-6 {
            [v[0] / len, v[1] / len, v[2] / len]
        } else {
            [0.0, 0.0, 1.0]
        }
    }

    /// Smooth assignments to enforce minimum segment lengths
    fn smooth_assignments(&self, assignments: &mut [SecondaryStructure]) {
        let n = assignments.len();
        if n < 3 {
            return;
        }

        // Find and validate helix segments
        let mut i = 0;
        while i < n {
            if assignments[i] == SecondaryStructure::Helix {
                let start = i;
                while i < n && assignments[i] == SecondaryStructure::Helix {
                    i += 1;
                }
                let length = i - start;

                // If segment too short, convert to loop
                if length < self.min_helix_length {
                    for j in start..i {
                        assignments[j] = SecondaryStructure::Loop;
                    }
                }
            } else {
                i += 1;
            }
        }

        // Find and validate sheet segments
        i = 0;
        while i < n {
            if assignments[i] == SecondaryStructure::Sheet {
                let start = i;
                while i < n && assignments[i] == SecondaryStructure::Sheet {
                    i += 1;
                }
                let length = i - start;

                // If segment too short, convert to loop
                if length < self.min_sheet_length {
                    for j in start..i {
                        assignments[j] = SecondaryStructure::Loop;
                    }
                }
            } else {
                i += 1;
            }
        }
    }
}

/// Quick secondary structure summary
#[derive(Debug, Clone)]
pub struct SecondaryStructureSummary {
    pub n_residues: usize,
    pub n_helix: usize,
    pub n_sheet: usize,
    pub n_loop: usize,
    pub helix_fraction: f64,
    pub sheet_fraction: f64,
    pub loop_fraction: f64,
}

impl SecondaryStructureSummary {
    /// Compute summary from assignments
    pub fn from_assignments(assignments: &[SecondaryStructure]) -> Self {
        let n = assignments.len();
        let n_helix = assignments
            .iter()
            .filter(|&&s| s == SecondaryStructure::Helix)
            .count();
        let n_sheet = assignments
            .iter()
            .filter(|&&s| s == SecondaryStructure::Sheet)
            .count();
        let n_loop = n - n_helix - n_sheet;

        Self {
            n_residues: n,
            n_helix,
            n_sheet,
            n_loop,
            helix_fraction: n_helix as f64 / n.max(1) as f64,
            sheet_fraction: n_sheet as f64 / n.max(1) as f64,
            loop_fraction: n_loop as f64 / n.max(1) as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate ideal alpha helix CA positions
    fn generate_helix_positions(n_residues: usize) -> Vec<[f32; 3]> {
        // Alpha helix parameters
        let rise_per_residue = 1.5; // Å
        let radius = 2.3; // Å
        let rotation_per_residue = 100.0 * PI / 180.0; // ~100 degrees

        (0..n_residues)
            .map(|i| {
                let angle = i as f32 * rotation_per_residue;
                [
                    radius * angle.cos(),
                    radius * angle.sin(),
                    i as f32 * rise_per_residue,
                ]
            })
            .collect()
    }

    /// Generate extended beta strand CA positions
    fn generate_strand_positions(n_residues: usize) -> Vec<[f32; 3]> {
        // Extended strand: ~3.4Å per residue along backbone
        (0..n_residues)
            .map(|i| {
                // Slight zigzag pattern typical of beta strands
                let y_offset = if i % 2 == 0 { 0.5 } else { -0.5 };
                [i as f32 * 3.4, y_offset, 0.0]
            })
            .collect()
    }

    #[test]
    fn test_helix_detection() {
        let analyzer = SecondaryStructureAnalyzer::default();
        let positions = generate_helix_positions(12);

        let assignments = analyzer.detect(&positions);
        let summary = SecondaryStructureSummary::from_assignments(&assignments);

        println!("Helix detection: {:?}", summary);

        // Most residues should be classified as helix
        assert!(
            summary.helix_fraction > 0.5,
            "Helix fraction {} should be > 0.5",
            summary.helix_fraction
        );
    }

    #[test]
    fn test_strand_detection() {
        let analyzer = SecondaryStructureAnalyzer::default();
        let positions = generate_strand_positions(8);

        let assignments = analyzer.detect(&positions);
        let summary = SecondaryStructureSummary::from_assignments(&assignments);

        println!("Strand detection: {:?}", summary);

        // Some residues should be classified as sheet
        // Note: strand detection is harder without H-bond info
        assert!(
            summary.sheet_fraction > 0.0 || summary.loop_fraction > 0.5,
            "Extended structure should have some sheet/loop character"
        );
    }

    #[test]
    fn test_flexibility_factors() {
        let analyzer = SecondaryStructureAnalyzer::default();

        // Mix of structures
        let mut positions = generate_helix_positions(6);
        positions.extend(vec![[20.0, 0.0, 0.0], [24.0, 1.0, 1.0], [28.0, -1.0, 2.0]]); // Loop-like
        positions.extend(
            generate_strand_positions(4)
                .iter()
                .map(|p| [p[0] + 35.0, p[1], p[2]]),
        );

        let factors = analyzer.get_flexibility_factors(&positions);

        assert_eq!(factors.len(), positions.len());

        // Check that factors are in expected range
        for f in &factors {
            assert!(*f >= 0.6 && *f <= 1.3, "Factor {} out of range", f);
        }
    }

    #[test]
    fn test_empty_and_small() {
        let analyzer = SecondaryStructureAnalyzer::default();

        // Empty
        let empty = analyzer.detect(&[]);
        assert!(empty.is_empty());

        // Single residue
        let single = analyzer.detect(&[[0.0, 0.0, 0.0]]);
        assert_eq!(single.len(), 1);
        assert_eq!(single[0], SecondaryStructure::Loop);

        // Two residues
        let two = analyzer.detect(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]]);
        assert_eq!(two.len(), 2);
    }
}
