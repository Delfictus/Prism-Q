//! Gaussian Network Model (GNM) for protein flexibility prediction
//!
//! GNM predicts residue fluctuations from the protein contact network topology.
//! The key insight is that low-frequency collective modes dominate thermal fluctuations.
//!
//! # Theory
//!
//! The Kirchhoff (connectivity) matrix Γ encodes the network:
//! - Γᵢⱼ = -1 if residues i,j are within cutoff distance
//! - Γᵢᵢ = coordination number (number of contacts)
//!
//! RMSF is computed from eigendecomposition:
//! ```text
//! RMSF_i² = (kT/γ) × Σⱼ (1/λⱼ) × [Uⱼ]ᵢ²
//! ```
//!
//! Where λⱼ are eigenvalues and Uⱼ are eigenvectors.
//! The first mode (λ₀=0) is excluded as it represents rigid-body translation.
//!
//! # References
//!
//! - Bahar et al. (1997) "Direct evaluation of thermal fluctuations in proteins
//!   using a single-parameter harmonic potential" Folding & Design 2:173-181
//! - Yang et al. (2009) "Protein elastic network models and the ranges of
//!   cooperativity" PNAS 106:12347-12352

use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// GNM analysis results
#[derive(Debug, Clone)]
pub struct GnmResult {
    /// Number of residues
    pub n_residues: usize,
    /// Predicted RMSF per residue (Ångstroms)
    pub rmsf: Vec<f64>,
    /// Eigenvalues (excluding zero mode)
    pub eigenvalues: Vec<f64>,
    /// Number of contacts per residue (coordination number)
    pub coordination: Vec<usize>,
    /// Cutoff distance used (Ångstroms)
    pub cutoff: f64,
}

/// Gaussian Network Model for flexibility prediction
pub struct GaussianNetworkModel {
    /// Distance cutoff for contacts (Ångstroms)
    cutoff: f64,
    /// Spring constant (kcal/mol/Ų)
    gamma: f64,
    /// Temperature (Kelvin)
    temperature: f64,
}

impl Default for GaussianNetworkModel {
    fn default() -> Self {
        Self {
            cutoff: 7.3,        // Optimal cutoff from literature (Bahar 1997)
            gamma: 1.0,         // Arbitrary units (cancels in correlation)
            temperature: 310.0, // 37°C
        }
    }
}

impl GaussianNetworkModel {
    /// Create GNM with custom cutoff
    pub fn with_cutoff(cutoff: f64) -> Self {
        Self {
            cutoff,
            ..Default::default()
        }
    }

    /// Compute GNM-predicted RMSF from CA positions
    ///
    /// # Arguments
    /// * `ca_positions` - Alpha carbon positions as [[x, y, z], ...]
    ///
    /// # Returns
    /// GnmResult containing predicted RMSF and eigenmode information
    pub fn compute_rmsf(&self, ca_positions: &[[f32; 3]]) -> GnmResult {
        let n = ca_positions.len();

        if n < 3 {
            return GnmResult {
                n_residues: n,
                rmsf: vec![1.0; n],
                eigenvalues: vec![],
                coordination: vec![0; n],
                cutoff: self.cutoff,
            };
        }

        // Build Kirchhoff matrix
        let (kirchhoff, coordination) = self.build_kirchhoff_matrix(ca_positions);

        // Eigendecomposition
        let eigen = SymmetricEigen::new(kirchhoff);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Sort eigenvalues and get indices (nalgebra doesn't guarantee order)
        let mut indexed_eigenvalues: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed_eigenvalues
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute RMSF from inverse eigenvalue weighting
        // Skip first mode (zero eigenvalue = rigid body motion)
        let kb = 0.001987204; // Boltzmann constant kcal/(mol·K)
        let kt = kb * self.temperature;
        let scale = kt / self.gamma;

        let mut rmsf_sq = vec![0.0f64; n];
        let mut valid_eigenvalues = Vec::new();

        for (orig_idx, lambda) in indexed_eigenvalues.iter().skip(1) {
            // Skip near-zero eigenvalues (numerical noise)
            if *lambda < 1e-6 {
                continue;
            }

            valid_eigenvalues.push(*lambda);

            // Get eigenvector components
            let eigenvec = eigenvectors.column(*orig_idx);

            // Add contribution: (1/λ) × uᵢ²
            for i in 0..n {
                let u_i = eigenvec[i];
                rmsf_sq[i] += (1.0 / lambda) * u_i * u_i;
            }
        }

        // Scale and sqrt
        let rmsf: Vec<f64> = rmsf_sq.iter().map(|&sq| (scale * sq).sqrt()).collect();

        // Normalize to typical B-factor range (~0.5-2.0 Å)
        // GNM gives relative fluctuations; we scale to match experimental range
        let rmsf_mean: f64 = rmsf.iter().sum::<f64>() / n as f64;
        let target_mean = 0.8; // Target mean RMSF in Ångstroms
        let scale_factor = if rmsf_mean > 1e-6 {
            target_mean / rmsf_mean
        } else {
            1.0
        };

        let rmsf_scaled: Vec<f64> = rmsf.iter().map(|&r| r * scale_factor).collect();

        GnmResult {
            n_residues: n,
            rmsf: rmsf_scaled,
            eigenvalues: valid_eigenvalues,
            coordination,
            cutoff: self.cutoff,
        }
    }

    /// Build the Kirchhoff (connectivity) matrix
    ///
    /// Γᵢⱼ = -1 if i≠j and distance < cutoff
    /// Γᵢᵢ = -Σⱼ≠ᵢ Γᵢⱼ (ensures row sums to zero)
    fn build_kirchhoff_matrix(&self, ca_positions: &[[f32; 3]]) -> (DMatrix<f64>, Vec<usize>) {
        let n = ca_positions.len();
        let cutoff_sq = (self.cutoff * self.cutoff) as f64;

        let mut kirchhoff = DMatrix::zeros(n, n);
        let mut coordination = vec![0usize; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
                let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
                let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    // Off-diagonal: -1 for connected pairs
                    kirchhoff[(i, j)] = -1.0;
                    kirchhoff[(j, i)] = -1.0;

                    // Track coordination
                    coordination[i] += 1;
                    coordination[j] += 1;
                }
            }
        }

        // Diagonal: sum of connections (makes each row sum to zero)
        for i in 0..n {
            kirchhoff[(i, i)] = coordination[i] as f64;
        }

        (kirchhoff, coordination)
    }

    /// Compute correlation between predicted and experimental RMSF
    pub fn correlation(predicted: &[f64], experimental: &[f64]) -> f64 {
        if predicted.len() != experimental.len() || predicted.is_empty() {
            return 0.0;
        }

        let n = predicted.len() as f64;
        let mean_p: f64 = predicted.iter().sum::<f64>() / n;
        let mean_e: f64 = experimental.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_p = 0.0;
        let mut var_e = 0.0;

        for (p, e) in predicted.iter().zip(experimental.iter()) {
            let dp = p - mean_p;
            let de = e - mean_e;
            cov += dp * de;
            var_p += dp * dp;
            var_e += de * de;
        }

        if var_p < 1e-10 || var_e < 1e-10 {
            return 0.0;
        }

        cov / (var_p.sqrt() * var_e.sqrt())
    }
}

/// Anisotropic Network Model (ANM) for directional fluctuations
///
/// ANM extends GNM to predict 3D displacement vectors, not just magnitudes.
/// This is more expensive (3N×3N matrix) but provides mode shapes.
pub struct AnisotropicNetworkModel {
    /// Distance cutoff for contacts (Ångstroms)
    cutoff: f64,
    /// Spring constant
    gamma: f64,
    /// Temperature (Kelvin)
    temperature: f64,
}

impl Default for AnisotropicNetworkModel {
    fn default() -> Self {
        Self {
            cutoff: 15.0, // ANM typically uses larger cutoff
            gamma: 1.0,
            temperature: 310.0,
        }
    }
}

impl AnisotropicNetworkModel {
    /// Create ANM with custom cutoff
    pub fn with_cutoff(cutoff: f64) -> Self {
        Self {
            cutoff,
            ..Default::default()
        }
    }

    /// Compute ANM-predicted RMSF from CA positions
    ///
    /// ANM uses a 3N×3N Hessian matrix for full directional information.
    /// This is more expensive but provides richer mode information.
    pub fn compute_rmsf(&self, ca_positions: &[[f32; 3]]) -> GnmResult {
        let n = ca_positions.len();

        if n < 3 {
            return GnmResult {
                n_residues: n,
                rmsf: vec![1.0; n],
                eigenvalues: vec![],
                coordination: vec![0; n],
                cutoff: self.cutoff,
            };
        }

        // Build 3N×3N Hessian matrix
        let (hessian, coordination) = self.build_hessian_matrix(ca_positions);

        // Eigendecomposition (expensive for large proteins)
        let eigen = SymmetricEigen::new(hessian);
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Sort eigenvalues
        let mut indexed_eigenvalues: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed_eigenvalues
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute RMSF from modes
        // Skip first 6 modes (rigid body: 3 translations + 3 rotations)
        let kb = 0.001987204;
        let kt = kb * self.temperature;
        let scale = kt / self.gamma;

        let mut rmsf_sq = vec![0.0f64; n];
        let mut valid_eigenvalues = Vec::new();

        for (orig_idx, lambda) in indexed_eigenvalues.iter().skip(6) {
            if *lambda < 1e-6 {
                continue;
            }

            valid_eigenvalues.push(*lambda);

            let eigenvec = eigenvectors.column(*orig_idx);

            // Sum x², y², z² components for each residue
            for i in 0..n {
                let ux = eigenvec[3 * i];
                let uy = eigenvec[3 * i + 1];
                let uz = eigenvec[3 * i + 2];
                rmsf_sq[i] += (1.0 / lambda) * (ux * ux + uy * uy + uz * uz);
            }
        }

        // Scale and sqrt
        let rmsf: Vec<f64> = rmsf_sq.iter().map(|&sq| (scale * sq).sqrt()).collect();

        // Normalize
        let rmsf_mean: f64 = rmsf.iter().sum::<f64>() / n as f64;
        let target_mean = 0.8;
        let scale_factor = if rmsf_mean > 1e-6 {
            target_mean / rmsf_mean
        } else {
            1.0
        };

        let rmsf_scaled: Vec<f64> = rmsf.iter().map(|&r| r * scale_factor).collect();

        GnmResult {
            n_residues: n,
            rmsf: rmsf_scaled,
            eigenvalues: valid_eigenvalues,
            coordination,
            cutoff: self.cutoff,
        }
    }

    /// Build the 3N×3N Hessian matrix for ANM
    ///
    /// H_ij = -γ/r² × (r_ij ⊗ r_ij) for i≠j and connected
    /// H_ii = -Σⱼ≠ᵢ H_ij
    fn build_hessian_matrix(&self, ca_positions: &[[f32; 3]]) -> (DMatrix<f64>, Vec<usize>) {
        let n = ca_positions.len();
        let n3 = 3 * n;
        let cutoff_sq = (self.cutoff * self.cutoff) as f64;

        let mut hessian = DMatrix::zeros(n3, n3);
        let mut coordination = vec![0usize; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
                let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
                let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq && dist_sq > 1e-6 {
                    coordination[i] += 1;
                    coordination[j] += 1;

                    // Build 3×3 super-element: -γ/r² × (r ⊗ r)
                    let scale = -self.gamma / dist_sq;

                    let h_xx = scale * dx * dx;
                    let h_xy = scale * dx * dy;
                    let h_xz = scale * dx * dz;
                    let h_yy = scale * dy * dy;
                    let h_yz = scale * dy * dz;
                    let h_zz = scale * dz * dz;

                    // Off-diagonal blocks (i,j) and (j,i)
                    let i3 = 3 * i;
                    let j3 = 3 * j;

                    // Block (i,j)
                    hessian[(i3, j3)] = h_xx;
                    hessian[(i3, j3 + 1)] = h_xy;
                    hessian[(i3, j3 + 2)] = h_xz;
                    hessian[(i3 + 1, j3)] = h_xy;
                    hessian[(i3 + 1, j3 + 1)] = h_yy;
                    hessian[(i3 + 1, j3 + 2)] = h_yz;
                    hessian[(i3 + 2, j3)] = h_xz;
                    hessian[(i3 + 2, j3 + 1)] = h_yz;
                    hessian[(i3 + 2, j3 + 2)] = h_zz;

                    // Block (j,i) - symmetric
                    hessian[(j3, i3)] = h_xx;
                    hessian[(j3, i3 + 1)] = h_xy;
                    hessian[(j3, i3 + 2)] = h_xz;
                    hessian[(j3 + 1, i3)] = h_xy;
                    hessian[(j3 + 1, i3 + 1)] = h_yy;
                    hessian[(j3 + 1, i3 + 2)] = h_yz;
                    hessian[(j3 + 2, i3)] = h_xz;
                    hessian[(j3 + 2, i3 + 1)] = h_yz;
                    hessian[(j3 + 2, i3 + 2)] = h_zz;

                    // Diagonal blocks: subtract from both i and j
                    hessian[(i3, i3)] -= h_xx;
                    hessian[(i3, i3 + 1)] -= h_xy;
                    hessian[(i3, i3 + 2)] -= h_xz;
                    hessian[(i3 + 1, i3)] -= h_xy;
                    hessian[(i3 + 1, i3 + 1)] -= h_yy;
                    hessian[(i3 + 1, i3 + 2)] -= h_yz;
                    hessian[(i3 + 2, i3)] -= h_xz;
                    hessian[(i3 + 2, i3 + 1)] -= h_yz;
                    hessian[(i3 + 2, i3 + 2)] -= h_zz;

                    hessian[(j3, j3)] -= h_xx;
                    hessian[(j3, j3 + 1)] -= h_xy;
                    hessian[(j3, j3 + 2)] -= h_xz;
                    hessian[(j3 + 1, j3)] -= h_xy;
                    hessian[(j3 + 1, j3 + 1)] -= h_yy;
                    hessian[(j3 + 1, j3 + 2)] -= h_yz;
                    hessian[(j3 + 2, j3)] -= h_xz;
                    hessian[(j3 + 2, j3 + 1)] -= h_yz;
                    hessian[(j3 + 2, j3 + 2)] -= h_zz;
                }
            }
        }

        (hessian, coordination)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gnm_basic() {
        // Simple 4-residue chain
        let positions = vec![
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
        ];

        let gnm = GaussianNetworkModel::with_cutoff(10.0);
        let result = gnm.compute_rmsf(&positions);

        assert_eq!(result.n_residues, 4);
        assert_eq!(result.rmsf.len(), 4);

        // Terminal residues should be more flexible than internal
        // (they have fewer connections)
        println!("GNM RMSF: {:?}", result.rmsf);
        println!("Coordination: {:?}", result.coordination);

        // Terminal (index 0, 3) should have higher RMSF than internal (1, 2)
        let terminal_avg = (result.rmsf[0] + result.rmsf[3]) / 2.0;
        let internal_avg = (result.rmsf[1] + result.rmsf[2]) / 2.0;
        assert!(
            terminal_avg > internal_avg,
            "Terminal residues should be more flexible: {} vs {}",
            terminal_avg,
            internal_avg
        );
    }

    #[test]
    fn test_gnm_correlation() {
        let predicted = vec![1.0, 0.5, 0.8, 1.2, 0.6];
        let experimental = vec![1.1, 0.4, 0.9, 1.1, 0.7];

        let corr = GaussianNetworkModel::correlation(&predicted, &experimental);
        assert!(corr > 0.9, "High correlation expected: {}", corr);
    }

    #[test]
    fn test_anm_basic() {
        let positions = vec![
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
            [11.4, 0.0, 0.0],
        ];

        let anm = AnisotropicNetworkModel::with_cutoff(15.0);
        let result = anm.compute_rmsf(&positions);

        assert_eq!(result.n_residues, 4);
        println!("ANM RMSF: {:?}", result.rmsf);
    }
}
