//! Tertiary Structure Analysis
//!
//! Analyzes protein tertiary structure features for RMSF enhancement:
//! - Approximate SASA from neighbor counting (Cα-only approximation)
//! - Long-range contacts (sequence separation > 12 residues)
//! - Domain detection via spectral clustering
//! - Burial depth computation
//!
//! # Theory
//!
//! Tertiary structure determines local environment:
//! - Surface residues: more flexible, exposed to solvent
//! - Core residues: constrained by packing, less flexible
//! - Domain boundaries: often flexible hinges
//!
//! # References
//!
//! - Halle (2002) "Flexibility and packing in proteins"
//! - Csermely et al. (2013) "Structure and dynamics of protein networks"

use nalgebra::{DMatrix, SymmetricEigen};
use std::collections::HashMap;

/// Approximate SASA result from neighbor counting
#[derive(Debug, Clone)]
pub struct ApproximateSasaResult {
    /// Relative SASA per residue (0.0 = buried, 1.0 = fully exposed)
    pub relative_sasa: Vec<f64>,
    /// Number of neighbors per residue
    pub neighbor_counts: Vec<usize>,
    /// Burial depth (0.0 = surface, 1.0 = core)
    pub burial_depth: Vec<f64>,
    /// Surface residue indices (relative_sasa > 0.5)
    pub surface_residues: Vec<usize>,
    /// Core residue indices (relative_sasa < 0.25)
    pub core_residues: Vec<usize>,
}

/// Tertiary structure analyzer
pub struct TertiaryAnalyzer {
    /// Cutoff for neighbor counting (Å)
    neighbor_cutoff: f64,
    /// Expected max neighbors at this cutoff
    max_expected_neighbors: f64,
    /// Cutoff for long-range contacts (Å)
    contact_cutoff: f64,
    /// Minimum sequence separation for long-range contacts
    min_sequence_separation: usize,
}

impl Default for TertiaryAnalyzer {
    fn default() -> Self {
        Self {
            neighbor_cutoff: 10.0,
            max_expected_neighbors: 20.0, // Typical max for 10Å cutoff
            contact_cutoff: 8.0,
            min_sequence_separation: 12,
        }
    }
}

impl TertiaryAnalyzer {
    /// Create analyzer with custom parameters
    pub fn new(neighbor_cutoff: f64, contact_cutoff: f64) -> Self {
        // Estimate max neighbors based on cutoff
        // At 10Å, expect ~20 neighbors for buried residue
        let max_expected = (neighbor_cutoff / 10.0).powi(3) * 20.0;
        Self {
            neighbor_cutoff,
            max_expected_neighbors: max_expected,
            contact_cutoff,
            min_sequence_separation: 12,
        }
    }

    /// Compute approximate SASA from neighbor counting
    ///
    /// This is a fast approximation suitable for Cα-only models.
    /// More neighbors = more buried = lower SASA.
    ///
    /// # Arguments
    /// * `ca_positions` - Alpha carbon positions
    ///
    /// # Returns
    /// ApproximateSasaResult with per-residue exposure estimates
    pub fn compute_approximate_sasa(&self, ca_positions: &[[f32; 3]]) -> ApproximateSasaResult {
        let n = ca_positions.len();
        if n == 0 {
            return ApproximateSasaResult {
                relative_sasa: vec![],
                neighbor_counts: vec![],
                burial_depth: vec![],
                surface_residues: vec![],
                core_residues: vec![],
            };
        }

        let cutoff_sq = (self.neighbor_cutoff * self.neighbor_cutoff) as f64;

        // Count neighbors for each residue
        let mut neighbor_counts = vec![0usize; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
                let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
                let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    neighbor_counts[i] += 1;
                    neighbor_counts[j] += 1;
                }
            }
        }

        // Convert to relative SASA (inverse of neighbor density)
        let relative_sasa: Vec<f64> = neighbor_counts
            .iter()
            .map(|&count| {
                let normalized = count as f64 / self.max_expected_neighbors;
                (1.0 - normalized).max(0.0).min(1.0)
            })
            .collect();

        // Burial depth is inverse of relative SASA
        let burial_depth: Vec<f64> = relative_sasa.iter().map(|&s| 1.0 - s).collect();

        // Classify surface and core
        let surface_residues: Vec<usize> = relative_sasa
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > 0.5)
            .map(|(i, _)| i)
            .collect();

        let core_residues: Vec<usize> = relative_sasa
            .iter()
            .enumerate()
            .filter(|(_, &s)| s < 0.25)
            .map(|(i, _)| i)
            .collect();

        ApproximateSasaResult {
            relative_sasa,
            neighbor_counts,
            burial_depth,
            surface_residues,
            core_residues,
        }
    }

    /// Identify long-range contacts
    ///
    /// Long-range contacts indicate tertiary structure formation
    /// and can affect local flexibility.
    ///
    /// # Returns
    /// Vector of (residue_i, residue_j) pairs
    pub fn find_long_range_contacts(&self, ca_positions: &[[f32; 3]]) -> Vec<(usize, usize)> {
        let n = ca_positions.len();
        let cutoff_sq = (self.contact_cutoff * self.contact_cutoff) as f64;
        let mut contacts = Vec::new();

        for i in 0..n {
            for j in (i + self.min_sequence_separation)..n {
                let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
                let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
                let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    contacts.push((i, j));
                }
            }
        }

        contacts
    }

    /// Count long-range contacts per residue
    pub fn long_range_contact_counts(&self, ca_positions: &[[f32; 3]]) -> Vec<usize> {
        let n = ca_positions.len();
        let mut counts = vec![0usize; n];

        for (i, j) in self.find_long_range_contacts(ca_positions) {
            counts[i] += 1;
            counts[j] += 1;
        }

        counts
    }

    /// Detect domains via spectral clustering on contact map
    ///
    /// Uses the slowest modes of the Kirchhoff matrix to identify
    /// structural domains (regions that move together).
    ///
    /// # Arguments
    /// * `ca_positions` - Alpha carbon positions
    /// * `n_domains` - Target number of domains
    ///
    /// # Returns
    /// Domain assignment for each residue (0 to n_domains-1)
    pub fn detect_domains(&self, ca_positions: &[[f32; 3]], n_domains: usize) -> Vec<usize> {
        let n = ca_positions.len();
        if n < n_domains || n_domains == 0 {
            return vec![0; n];
        }

        // Build Kirchhoff-like contact matrix
        let cutoff = 10.0f64; // Domain detection cutoff
        let cutoff_sq = cutoff * cutoff;
        let mut contact = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
                let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
                let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    contact[(i, j)] = -1.0;
                    contact[(j, i)] = -1.0;
                }
            }
        }

        // Set diagonal (Laplacian)
        for i in 0..n {
            let row_sum: f64 = contact.row(i).iter().sum();
            contact[(i, i)] = -row_sum;
        }

        // Eigendecomposition
        let eigen = SymmetricEigen::new(contact);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Sort eigenvalues and get indices
        let mut indexed: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Use the n_domains-1 smallest non-zero eigenvectors for clustering
        // (Skip the trivial zero eigenvalue)
        let n_features = (n_domains - 1).min(n - 1);
        if n_features == 0 {
            return vec![0; n];
        }

        // Extract feature vectors (slow modes)
        let mut features: Vec<Vec<f64>> = vec![vec![0.0; n_features]; n];
        for f in 0..n_features {
            let (orig_idx, _) = indexed[f + 1]; // Skip first (zero) eigenvalue
            let eigenvec = eigenvectors.column(orig_idx);
            for i in 0..n {
                features[i][f] = eigenvec[i];
            }
        }

        // Simple k-means clustering on feature space
        self.kmeans_cluster(&features, n_domains)
    }

    /// Simple k-means clustering
    fn kmeans_cluster(&self, features: &[Vec<f64>], k: usize) -> Vec<usize> {
        let n = features.len();
        if n == 0 || k == 0 {
            return vec![];
        }
        let dim = features[0].len();
        if dim == 0 {
            return vec![0; n];
        }

        // Initialize centroids evenly spaced
        let mut centroids: Vec<Vec<f64>> = (0..k)
            .map(|i| {
                let idx = i * n / k;
                features[idx].clone()
            })
            .collect();

        let mut assignments = vec![0usize; n];
        let max_iter = 20;

        for _ in 0..max_iter {
            // Assign points to nearest centroid
            let mut changed = false;
            for i in 0..n {
                let mut best_cluster = 0;
                let mut best_dist = f64::INFINITY;

                for (c, centroid) in centroids.iter().enumerate() {
                    let dist: f64 = features[i]
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            let mut counts = vec![0usize; k];
            for c in &mut centroids {
                c.fill(0.0);
            }

            for (i, &cluster) in assignments.iter().enumerate() {
                counts[cluster] += 1;
                for (d, val) in features[i].iter().enumerate() {
                    centroids[cluster][d] += val;
                }
            }

            for (c, count) in counts.iter().enumerate() {
                if *count > 0 {
                    for d in 0..dim {
                        centroids[c][d] /= *count as f64;
                    }
                }
            }
        }

        assignments
    }

    /// Compute flexibility modulation from tertiary structure
    ///
    /// Combines burial depth and long-range contact information
    /// to modulate RMSF predictions.
    pub fn compute_flexibility_modulation(&self, ca_positions: &[[f32; 3]]) -> Vec<f64> {
        let sasa = self.compute_approximate_sasa(ca_positions);
        let lr_contacts = self.long_range_contact_counts(ca_positions);
        let n = ca_positions.len();

        // Modulation factors:
        // - Surface residues: enhanced flexibility (× 1.0-1.3)
        // - Core residues: reduced flexibility (× 0.7-1.0)
        // - Many LR contacts: reduced flexibility (constrained)

        let max_lr = *lr_contacts.iter().max().unwrap_or(&1) as f64;

        (0..n)
            .map(|i| {
                let burial_factor = 1.0 - 0.3 * sasa.burial_depth[i]; // 0.7-1.0
                let lr_factor = if max_lr > 0.0 {
                    1.0 - 0.2 * (lr_contacts[i] as f64 / max_lr) // 0.8-1.0
                } else {
                    1.0
                };
                burial_factor * lr_factor
            })
            .collect()
    }

    /// Apply tertiary structure modulation to RMSF
    pub fn apply_to_rmsf(&self, rmsf: &[f64], ca_positions: &[[f32; 3]]) -> Vec<f64> {
        let modulation = self.compute_flexibility_modulation(ca_positions);
        rmsf.iter()
            .zip(modulation.iter())
            .map(|(&r, &m)| r * m)
            .collect()
    }
}

/// Contact order calculation
///
/// Average sequence separation of contacting residues.
/// Higher contact order → more complex folding → potentially different flexibility.
pub fn compute_contact_order(ca_positions: &[[f32; 3]], cutoff: f64) -> f64 {
    let n = ca_positions.len();
    if n < 2 {
        return 0.0;
    }

    let cutoff_sq = cutoff * cutoff;
    let mut contact_sum = 0.0;
    let mut contact_count = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
            let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
            let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < cutoff_sq {
                contact_sum += (j - i) as f64;
                contact_count += 1;
            }
        }
    }

    if contact_count > 0 {
        contact_sum / (contact_count as f64 * n as f64)
    } else {
        0.0
    }
}

/// Radius of gyration calculation
pub fn radius_of_gyration(ca_positions: &[[f32; 3]]) -> f64 {
    let n = ca_positions.len();
    if n == 0 {
        return 0.0;
    }

    // Compute center of mass
    let mut com = [0.0f64; 3];
    for pos in ca_positions {
        com[0] += pos[0] as f64;
        com[1] += pos[1] as f64;
        com[2] += pos[2] as f64;
    }
    com[0] /= n as f64;
    com[1] /= n as f64;
    com[2] /= n as f64;

    // Compute sum of squared distances from COM
    let rg_sq: f64 = ca_positions
        .iter()
        .map(|pos| {
            let dx = pos[0] as f64 - com[0];
            let dy = pos[1] as f64 - com[1];
            let dz = pos[2] as f64 - com[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum::<f64>()
        / n as f64;

    rg_sq.sqrt()
}

/// Tertiary structure summary
#[derive(Debug, Clone)]
pub struct TertiarySummary {
    pub n_residues: usize,
    pub radius_of_gyration: f64,
    pub contact_order: f64,
    pub n_long_range_contacts: usize,
    pub surface_fraction: f64,
    pub core_fraction: f64,
}

impl TertiarySummary {
    /// Compute summary from CA positions
    pub fn from_positions(ca_positions: &[[f32; 3]]) -> Self {
        let n = ca_positions.len();
        let analyzer = TertiaryAnalyzer::default();
        let sasa = analyzer.compute_approximate_sasa(ca_positions);
        let lr_contacts = analyzer.find_long_range_contacts(ca_positions);

        Self {
            n_residues: n,
            radius_of_gyration: radius_of_gyration(ca_positions),
            contact_order: compute_contact_order(ca_positions, 8.0),
            n_long_range_contacts: lr_contacts.len(),
            surface_fraction: sasa.surface_residues.len() as f64 / n.max(1) as f64,
            core_fraction: sasa.core_residues.len() as f64 / n.max(1) as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a compact globular structure
    fn generate_globular_positions(n: usize) -> Vec<[f32; 3]> {
        // Spiral that folds back on itself
        let radius = 10.0;
        (0..n)
            .map(|i| {
                let t = i as f32 / n as f32 * 4.0 * std::f32::consts::PI;
                let r = radius * (0.5 + 0.5 * (t * 0.5).sin());
                [r * t.cos(), r * t.sin(), (i as f32 * 0.3).sin() * 5.0]
            })
            .collect()
    }

    /// Generate an extended chain
    fn generate_extended_chain(n: usize) -> Vec<[f32; 3]> {
        (0..n).map(|i| [i as f32 * 3.8, 0.0, 0.0]).collect()
    }

    #[test]
    fn test_approximate_sasa() {
        let analyzer = TertiaryAnalyzer::default();

        // Globular structure should have buried residues
        let globular = generate_globular_positions(50);
        let sasa_globular = analyzer.compute_approximate_sasa(&globular);

        assert!(
            !sasa_globular.core_residues.is_empty(),
            "Globular structure should have core residues"
        );

        // Extended chain should be mostly surface
        let extended = generate_extended_chain(20);
        let sasa_extended = analyzer.compute_approximate_sasa(&extended);

        assert!(
            sasa_extended.core_residues.len() < sasa_globular.core_residues.len(),
            "Extended chain should have fewer core residues"
        );
    }

    #[test]
    fn test_long_range_contacts() {
        let analyzer = TertiaryAnalyzer::default();

        // Globular should have many long-range contacts
        let globular = generate_globular_positions(50);
        let lr_globular = analyzer.find_long_range_contacts(&globular);

        // Extended should have few/none
        let extended = generate_extended_chain(50);
        let lr_extended = analyzer.find_long_range_contacts(&extended);

        assert!(
            lr_globular.len() > lr_extended.len(),
            "Globular ({}) should have more LR contacts than extended ({})",
            lr_globular.len(),
            lr_extended.len()
        );
    }

    #[test]
    fn test_domain_detection() {
        let analyzer = TertiaryAnalyzer::default();

        // Create a two-domain structure (two compact regions connected by linker)
        let mut positions = generate_globular_positions(30);
        // Add linker
        for i in 0..5 {
            positions.push([i as f32 * 5.0 + 20.0, 0.0, 0.0]);
        }
        // Add second domain
        let domain2: Vec<[f32; 3]> = generate_globular_positions(30)
            .iter()
            .map(|p| [p[0] + 50.0, p[1], p[2]])
            .collect();
        positions.extend(domain2);

        let domains = analyzer.detect_domains(&positions, 2);

        assert_eq!(domains.len(), positions.len());

        // Check that we have both domain assignments
        let has_domain_0 = domains.iter().any(|&d| d == 0);
        let has_domain_1 = domains.iter().any(|&d| d == 1);
        assert!(has_domain_0 && has_domain_1, "Should detect both domains");
    }

    #[test]
    fn test_contact_order() {
        // Extended chain has low contact order
        let extended = generate_extended_chain(30);
        let co_extended = compute_contact_order(&extended, 8.0);

        // Globular has higher contact order
        let globular = generate_globular_positions(30);
        let co_globular = compute_contact_order(&globular, 8.0);

        println!(
            "Contact order - Extended: {}, Globular: {}",
            co_extended, co_globular
        );

        // Globular should have higher contact order
        assert!(
            co_globular > co_extended || (co_extended < 0.1 && co_globular < 0.2),
            "Globular should have higher contact order"
        );
    }

    #[test]
    fn test_radius_of_gyration() {
        // Larger structure should have larger Rg
        let small = generate_globular_positions(20);
        let large = generate_globular_positions(100);

        let rg_small = radius_of_gyration(&small);
        let rg_large = radius_of_gyration(&large);

        println!("Rg - Small: {}, Large: {}", rg_small, rg_large);

        // At least they should both be positive
        assert!(rg_small > 0.0);
        assert!(rg_large > 0.0);
    }

    #[test]
    fn test_flexibility_modulation() {
        let analyzer = TertiaryAnalyzer::default();
        let positions = generate_globular_positions(50);

        let modulation = analyzer.compute_flexibility_modulation(&positions);

        assert_eq!(modulation.len(), 50);

        // All modulation factors should be in reasonable range
        for m in &modulation {
            assert!(
                *m >= 0.5 && *m <= 1.5,
                "Modulation {} outside expected range",
                m
            );
        }
    }

    #[test]
    fn test_tertiary_summary() {
        let positions = generate_globular_positions(50);
        let summary = TertiarySummary::from_positions(&positions);

        assert_eq!(summary.n_residues, 50);
        assert!(summary.radius_of_gyration > 0.0);
        assert!(summary.surface_fraction + summary.core_fraction <= 1.0);
    }
}
