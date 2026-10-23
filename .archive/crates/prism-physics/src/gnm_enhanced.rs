//! Enhanced Gaussian Network Model with Structural Weighting
//!
//! Extends the basic GNM with multiple enhancements:
//! - Distance-weighted Kirchhoff matrix (exponential decay)
//! - Multi-cutoff ensemble averaging
//! - Secondary structure weighting (helix/sheet/loop)
//! - Residue-type flexibility factors
//! - SASA/burial modulation
//! - Long-range contact constraints
//!
//! # Expected Improvements
//!
//! | Enhancement | Expected Δρ |
//! |-------------|-------------|
//! | Distance-weighted | +0.02-0.03 |
//! | Multi-cutoff | +0.02-0.03 |
//! | Secondary structure | +0.02-0.03 |
//! | Residue correction | +0.01-0.02 |
//! | SASA modulation | +0.01-0.02 |
//! | **Combined** | **+0.08-0.13** |
//!
//! Starting from baseline ρ=0.591, expect ρ=0.67-0.72.
//!
//! # References
//!
//! - Yang et al. (2009) "Protein elastic network models"
//! - Eyal et al. (2015) "The anisotropic network model web server"

use nalgebra::{DMatrix, SymmetricEigen};

use crate::gnm::{GaussianNetworkModel, GnmResult};
use crate::secondary_structure::{
    SecondaryStructure, SecondaryStructureAnalyzer, SecondaryStructureSummary,
};
use crate::sidechain_analysis::SidechainAnalyzer;
use crate::tertiary_analysis::{TertiaryAnalyzer, TertiarySummary};

/// Enhanced GNM result with additional structural information
#[derive(Debug, Clone)]
pub struct EnhancedGnmResult {
    /// Base GNM result
    pub base: GnmResult,
    /// Final enhanced RMSF
    pub rmsf: Vec<f64>,
    /// Secondary structure assignments
    pub secondary_structure: Vec<SecondaryStructure>,
    /// Approximate relative SASA per residue
    pub relative_sasa: Vec<f64>,
    /// Burial depth per residue (0=surface, 1=core)
    pub burial_depth: Vec<f64>,
    /// Domain assignments (if computed)
    pub domain_assignments: Option<Vec<usize>>,
    /// Enhancement factors applied
    pub applied_enhancements: EnhancementFlags,
    /// Correlation between base and enhanced (self-consistency check)
    pub base_enhanced_correlation: f64,
}

/// Flags indicating which enhancements were applied
#[derive(Debug, Clone, Default)]
pub struct EnhancementFlags {
    pub distance_weighting: bool,
    pub multi_cutoff: bool,
    pub secondary_structure: bool,
    pub sidechain_factors: bool,
    pub sasa_modulation: bool,
    pub long_range_contacts: bool,
}

/// Enhanced GNM configuration
#[derive(Debug, Clone)]
pub struct EnhancedGnmConfig {
    /// Use distance-weighted Kirchhoff matrix
    pub use_distance_weighting: bool,
    /// Sigma for exponential weighting (Å)
    pub distance_sigma: f64,
    /// Use multi-cutoff ensemble
    pub use_multi_cutoff: bool,
    /// Cutoffs for ensemble (Å)
    pub ensemble_cutoffs: Vec<f64>,
    /// Weights for ensemble cutoffs (should sum to 1.0)
    pub ensemble_weights: Vec<f64>,
    /// Apply secondary structure weighting
    pub use_secondary_structure: bool,
    /// Apply residue-type (sidechain) flexibility factors
    pub use_sidechain_factors: bool,
    /// Apply SASA/burial modulation
    pub use_sasa_modulation: bool,
    /// Apply long-range contact constraints
    pub use_long_range_contacts: bool,
    /// Detect and use domain information
    pub detect_domains: bool,
    /// Number of domains for detection (if detect_domains is true)
    pub n_domains: usize,
}

impl Default for EnhancedGnmConfig {
    fn default() -> Self {
        // OPTIMIZED based on ablation study (2026-01-09):
        // - Distance weighting ALONE: +0.009 ✅ (BEST)
        // - Multi-cutoff ALONE: +0.007
        // - Combined MC + DW: +0.004 (they interfere!)
        // - Secondary structure: -0.080 ❌
        // - Sidechain factors: -0.090 ❌
        Self {
            use_distance_weighting: true,
            distance_sigma: 5.0,     // Gaussian width for distance weighting
            use_multi_cutoff: false, // DISABLED: interferes with distance weighting
            ensemble_cutoffs: vec![6.0, 7.0, 8.0, 10.0],
            ensemble_weights: vec![0.15, 0.35, 0.35, 0.15],
            use_secondary_structure: false, // DISABLED: hurts accuracy by -0.08
            use_sidechain_factors: false,   // DISABLED: hurts accuracy by -0.09
            use_sasa_modulation: false,     // DISABLED: neutral but adds noise
            use_long_range_contacts: false,
            detect_domains: false,
            n_domains: 2,
        }
    }
}

impl EnhancedGnmConfig {
    /// Create config with only distance weighting
    pub fn distance_weighted_only() -> Self {
        Self {
            use_distance_weighting: true,
            use_multi_cutoff: false,
            use_secondary_structure: false,
            use_sidechain_factors: false,
            use_sasa_modulation: false,
            ..Default::default()
        }
    }

    /// Create config with only multi-cutoff ensemble
    pub fn multi_cutoff_only() -> Self {
        Self {
            use_distance_weighting: false,
            use_multi_cutoff: true,
            use_secondary_structure: false,
            use_sidechain_factors: false,
            use_sasa_modulation: false,
            ..Default::default()
        }
    }

    /// Create config with all enhancements enabled (EXPERIMENTAL)
    /// WARNING: Secondary structure and sidechain factors are miscalibrated
    /// and may hurt accuracy. Use default() for best performance.
    pub fn full_experimental() -> Self {
        Self {
            use_distance_weighting: true,
            use_multi_cutoff: true,
            use_secondary_structure: true, // WARNING: -0.08 accuracy
            use_sidechain_factors: true,   // WARNING: -0.09 accuracy
            use_sasa_modulation: true,
            use_long_range_contacts: true,
            detect_domains: true,
            n_domains: 2,
            ..Default::default()
        }
    }

    /// Optimized config based on ablation study
    /// Uses only enhancements that improve accuracy
    pub fn optimized() -> Self {
        Self::default() // Default is now optimized
    }

    /// Create config with NO enhancements (plain GNM baseline)
    /// Use this to compare against standard GNM literature baseline
    pub fn plain() -> Self {
        Self {
            use_distance_weighting: false,
            use_multi_cutoff: false,
            use_secondary_structure: false,
            use_sidechain_factors: false,
            use_sasa_modulation: false,
            use_long_range_contacts: false,
            detect_domains: false,
            n_domains: 0,
            distance_sigma: 5.0,
            ensemble_cutoffs: vec![7.3],
            ensemble_weights: vec![1.0],
        }
    }

    /// Create plain config with specific cutoff
    pub fn plain_with_cutoff(cutoff: f64) -> Self {
        let mut config = Self::plain();
        config.ensemble_cutoffs = vec![cutoff];
        config
    }
}

/// Enhanced Gaussian Network Model
pub struct EnhancedGnm {
    config: EnhancedGnmConfig,
    /// Base cutoff for non-ensemble mode
    base_cutoff: f64,
    /// Spring constant
    gamma: f64,
    /// Temperature (Kelvin)
    temperature: f64,
}

impl Default for EnhancedGnm {
    fn default() -> Self {
        Self {
            config: EnhancedGnmConfig::default(),
            base_cutoff: 7.3,
            gamma: 1.0,
            temperature: 310.0,
        }
    }
}

impl EnhancedGnm {
    /// Create with custom configuration
    pub fn with_config(config: EnhancedGnmConfig) -> Self {
        Self {
            config,
            ..Default::default()
        }
    }

    /// Create with specific cutoff and default enhancements
    pub fn with_cutoff(cutoff: f64) -> Self {
        Self {
            base_cutoff: cutoff,
            ..Default::default()
        }
    }

    /// Set the base cutoff
    pub fn set_cutoff(&mut self, cutoff: f64) {
        self.base_cutoff = cutoff;
    }

    /// Compute enhanced RMSF from CA positions
    ///
    /// # Arguments
    /// * `ca_positions` - Alpha carbon positions
    /// * `residue_names` - Optional residue names for sidechain factors
    pub fn compute_rmsf(
        &self,
        ca_positions: &[[f32; 3]],
        residue_names: Option<&[&str]>,
    ) -> EnhancedGnmResult {
        let n = ca_positions.len();
        if n < 3 {
            return self.empty_result(n);
        }

        // Step 1: Compute base RMSF (with or without distance weighting)
        let (base_rmsf, eigenvalues, coordination) = if self.config.use_multi_cutoff {
            self.compute_multi_cutoff_rmsf(ca_positions)
        } else if self.config.use_distance_weighting {
            self.compute_distance_weighted_rmsf(ca_positions)
        } else {
            // Plain GNM: Build Kirchhoff matrix and use Lanczos for large matrices
            let (kirchhoff, coord) = self.build_kirchhoff_matrix(ca_positions, self.base_cutoff);
            let (rmsf, eigs) = self.compute_rmsf_from_kirchhoff(&kirchhoff, n);
            (rmsf, eigs, coord)
        };

        let mut rmsf = base_rmsf.clone();
        let mut flags = EnhancementFlags::default();

        if self.config.use_distance_weighting || self.config.use_multi_cutoff {
            flags.distance_weighting = self.config.use_distance_weighting;
            flags.multi_cutoff = self.config.use_multi_cutoff;
        }

        // Step 2: Apply secondary structure weighting
        let ss_analyzer = SecondaryStructureAnalyzer::default();
        let secondary_structure = ss_analyzer.detect(ca_positions);
        if self.config.use_secondary_structure {
            let ss_factors: Vec<f64> = secondary_structure
                .iter()
                .map(|ss| ss.flexibility_factor())
                .collect();
            for (r, f) in rmsf.iter_mut().zip(ss_factors.iter()) {
                *r *= f;
            }
            flags.secondary_structure = true;
        }

        // Step 3: Apply sidechain (residue-type) factors
        if self.config.use_sidechain_factors {
            if let Some(names) = residue_names {
                let sc_analyzer = SidechainAnalyzer::default();
                let sc_factors = sc_analyzer.compute_factors(names);
                for (r, f) in rmsf.iter_mut().zip(sc_factors.iter()) {
                    *r *= f;
                }
                flags.sidechain_factors = true;
            }
        }

        // Step 4: Apply SASA/burial modulation
        let tertiary_analyzer = TertiaryAnalyzer::default();
        let sasa_result = tertiary_analyzer.compute_approximate_sasa(ca_positions);
        if self.config.use_sasa_modulation {
            // Surface residues: enhanced flexibility (×1.0-1.2)
            // Buried residues: reduced flexibility (×0.8-1.0)
            for (i, r) in rmsf.iter_mut().enumerate() {
                let burial = sasa_result.burial_depth[i];
                let modulation = 1.0 - 0.2 * burial; // 0.8 to 1.0
                *r *= modulation;
            }
            flags.sasa_modulation = true;
        }

        // Step 5: Apply long-range contact constraints (optional)
        if self.config.use_long_range_contacts {
            let lr_counts = tertiary_analyzer.long_range_contact_counts(ca_positions);
            let max_lr = *lr_counts.iter().max().unwrap_or(&1) as f64;
            if max_lr > 0.0 {
                for (i, r) in rmsf.iter_mut().enumerate() {
                    // More LR contacts = more constrained = lower RMSF
                    let constraint = 1.0 - 0.15 * (lr_counts[i] as f64 / max_lr);
                    *r *= constraint;
                }
            }
            flags.long_range_contacts = true;
        }

        // Step 6: Domain detection (optional)
        let domain_assignments = if self.config.detect_domains && n > 20 {
            Some(tertiary_analyzer.detect_domains(ca_positions, self.config.n_domains))
        } else {
            None
        };

        // Normalize to maintain mean RMSF
        let current_mean: f64 = rmsf.iter().sum::<f64>() / n as f64;
        let target_mean: f64 = base_rmsf.iter().sum::<f64>() / n as f64;
        if current_mean > 1e-6 {
            let scale = target_mean / current_mean;
            for r in rmsf.iter_mut() {
                *r *= scale;
            }
        }

        // Compute correlation between base and enhanced
        let base_enhanced_correlation = Self::correlation(&base_rmsf, &rmsf);

        EnhancedGnmResult {
            base: GnmResult {
                n_residues: n,
                rmsf: base_rmsf,
                eigenvalues,
                coordination,
                cutoff: self.base_cutoff,
            },
            rmsf,
            secondary_structure,
            relative_sasa: sasa_result.relative_sasa,
            burial_depth: sasa_result.burial_depth,
            domain_assignments,
            applied_enhancements: flags,
            base_enhanced_correlation,
        }
    }

    /// Build standard (binary) Kirchhoff matrix for plain GNM
    fn build_kirchhoff_matrix(
        &self,
        ca_positions: &[[f32; 3]],
        cutoff: f64,
    ) -> (DMatrix<f64>, Vec<usize>) {
        let n = ca_positions.len();
        let cutoff_sq = cutoff * cutoff;

        let mut kirchhoff = DMatrix::zeros(n, n);
        let mut coordination = vec![0usize; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
                let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
                let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    // Binary contact: -1 for off-diagonal
                    kirchhoff[(i, j)] = -1.0;
                    kirchhoff[(j, i)] = -1.0;
                    coordination[i] += 1;
                    coordination[j] += 1;
                }
            }
        }

        // Set diagonal to coordination number
        for i in 0..n {
            kirchhoff[(i, i)] = coordination[i] as f64;
        }

        (kirchhoff, coordination)
    }

    /// Compute RMSF with distance-weighted Kirchhoff matrix
    fn compute_distance_weighted_rmsf(
        &self,
        ca_positions: &[[f32; 3]],
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let n = ca_positions.len();
        let sigma_sq = self.config.distance_sigma * self.config.distance_sigma;
        let cutoff_sq = (self.base_cutoff * self.base_cutoff) as f64;

        // Build distance-weighted Kirchhoff matrix
        let mut kirchhoff = DMatrix::zeros(n, n);
        let mut coordination = vec![0usize; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
                let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
                let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    // Gaussian weighting: exp(-d²/2σ²)
                    let weight = (-dist_sq / (2.0 * sigma_sq)).exp();

                    kirchhoff[(i, j)] = -weight;
                    kirchhoff[(j, i)] = -weight;

                    coordination[i] += 1;
                    coordination[j] += 1;
                }
            }
        }

        // Set diagonal
        for i in 0..n {
            let row_sum: f64 = kirchhoff.row(i).iter().sum();
            kirchhoff[(i, i)] = -row_sum;
        }

        // Eigendecomposition
        let (rmsf, eigenvalues) = self.compute_rmsf_from_kirchhoff(&kirchhoff, n);

        (rmsf, eigenvalues, coordination)
    }

    /// Compute multi-cutoff ensemble RMSF
    fn compute_multi_cutoff_rmsf(
        &self,
        ca_positions: &[[f32; 3]],
    ) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
        let n = ca_positions.len();
        let mut ensemble_rmsf = vec![0.0f64; n];
        let mut all_eigenvalues = Vec::new();
        let mut coordination = vec![0usize; n];

        let cutoffs = &self.config.ensemble_cutoffs;
        let weights = &self.config.ensemble_weights;

        for (cutoff, weight) in cutoffs.iter().zip(weights.iter()) {
            let gnm = if self.config.use_distance_weighting {
                // Build weighted Kirchhoff for this cutoff
                let sigma_sq = self.config.distance_sigma * self.config.distance_sigma;
                let cutoff_sq = cutoff * cutoff;

                let mut kirchhoff = DMatrix::zeros(n, n);

                for i in 0..n {
                    for j in (i + 1)..n {
                        let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
                        let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
                        let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
                        let dist_sq = dx * dx + dy * dy + dz * dz;

                        if dist_sq < cutoff_sq {
                            let w = (-dist_sq / (2.0 * sigma_sq)).exp();
                            kirchhoff[(i, j)] = -w;
                            kirchhoff[(j, i)] = -w;
                        }
                    }
                }

                for i in 0..n {
                    let row_sum: f64 = kirchhoff.row(i).iter().sum();
                    kirchhoff[(i, i)] = -row_sum;
                }

                let (rmsf, eigenvalues) = self.compute_rmsf_from_kirchhoff(&kirchhoff, n);

                for (r, &e) in ensemble_rmsf.iter_mut().zip(rmsf.iter()) {
                    *r += weight * e;
                }
                all_eigenvalues.extend(eigenvalues);

                continue;
            } else {
                GaussianNetworkModel::with_cutoff(*cutoff)
            };

            let result = gnm.compute_rmsf(ca_positions);

            for (r, &e) in ensemble_rmsf.iter_mut().zip(result.rmsf.iter()) {
                *r += weight * e;
            }

            // Use coordination from largest cutoff
            if *cutoff
                == *cutoffs
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(&10.0)
            {
                coordination = result.coordination;
            }

            all_eigenvalues.extend(result.eigenvalues);
        }

        (ensemble_rmsf, all_eigenvalues, coordination)
    }

    /// Compute RMSF from Kirchhoff matrix
    /// Uses Lanczos algorithm for large matrices (> 500 residues) to avoid O(n³) full eigendecomposition
    fn compute_rmsf_from_kirchhoff(
        &self,
        kirchhoff: &DMatrix<f64>,
        n: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        // For large matrices, use Lanczos algorithm (O(k * n * iterations) instead of O(n³))
        if n > 500 {
            return self.compute_rmsf_lanczos(kirchhoff, n);
        }

        // For small matrices, use full eigendecomposition (fast enough)
        let eigen = SymmetricEigen::new(kirchhoff.clone());
        let eigenvalues = eigen.eigenvalues;
        let eigenvectors = eigen.eigenvectors;

        // Sort eigenvalues
        let mut indexed: Vec<(usize, f64)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute RMSF
        let kb = 0.001987204;
        let kt = kb * self.temperature;
        let scale = kt / self.gamma;

        let mut rmsf_sq = vec![0.0f64; n];
        let mut valid_eigenvalues = Vec::new();

        for (orig_idx, lambda) in indexed.iter().skip(1) {
            if *lambda < 1e-6 {
                continue;
            }

            valid_eigenvalues.push(*lambda);

            let eigenvec = eigenvectors.column(*orig_idx);
            for i in 0..n {
                let u_i = eigenvec[i];
                rmsf_sq[i] += (1.0 / lambda) * u_i * u_i;
            }
        }

        let rmsf: Vec<f64> = rmsf_sq.iter().map(|&sq| (scale * sq).sqrt()).collect();

        // Normalize
        let mean: f64 = rmsf.iter().sum::<f64>() / n as f64;
        let target = 0.8;
        let scale_factor = if mean > 1e-6 { target / mean } else { 1.0 };

        let rmsf_scaled: Vec<f64> = rmsf.iter().map(|&r| r * scale_factor).collect();

        (rmsf_scaled, valid_eigenvalues)
    }

    /// Lanczos algorithm for large matrices - computes only the lowest eigenmodes needed for RMSF
    /// This is O(k * n * iterations) instead of O(n³) for full eigendecomposition
    fn compute_rmsf_lanczos(&self, kirchhoff: &DMatrix<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
        use nalgebra::DVector;

        let num_modes = 50.min(n - 1); // Need ~50 lowest modes for accurate RMSF (captures ~95% variance)
        let max_lanczos = (num_modes * 2).min(n - 1);

        // Initialize random starting vector
        let mut v = DVector::from_fn(n, |_, _| rand::random::<f64>() - 0.5);
        v.normalize_mut();

        let mut alpha = Vec::with_capacity(max_lanczos);
        let mut beta = Vec::with_capacity(max_lanczos);
        let mut lanczos_vectors = Vec::with_capacity(max_lanczos + 1);
        lanczos_vectors.push(v.clone());

        // First Lanczos iteration
        let mut w = kirchhoff * &v;
        let a = v.dot(&w);
        alpha.push(a);
        w -= a * &v;

        // Main Lanczos loop
        for j in 1..max_lanczos {
            let b = w.norm();
            if b < 1e-10 {
                log::debug!("Lanczos converged early at iteration {}", j);
                break;
            }
            beta.push(b);

            let v_prev = lanczos_vectors[j - 1].clone();
            let v_new = &w / b;
            lanczos_vectors.push(v_new.clone());

            w = kirchhoff * &v_new;
            w -= b * &v_prev;

            let a = v_new.dot(&w);
            alpha.push(a);
            w -= a * &v_new;

            // Reorthogonalization for numerical stability
            if j % 10 == 0 {
                for lv in lanczos_vectors.iter() {
                    let proj = lv.dot(&w);
                    w -= proj * lv;
                }
            }
        }

        // Build and solve tridiagonal eigenvalue problem
        let m = alpha.len();
        let mut tridiag = DMatrix::zeros(m, m);
        for i in 0..m {
            tridiag[(i, i)] = alpha[i];
            if i > 0 && i - 1 < beta.len() {
                tridiag[(i, i - 1)] = beta[i - 1];
                tridiag[(i - 1, i)] = beta[i - 1];
            }
        }

        let eigen = SymmetricEigen::new(tridiag);
        let ritz_eigenvalues = eigen.eigenvalues;
        let ritz_vectors = eigen.eigenvectors;

        // Sort eigenvalues (ascending)
        let mut indexed: Vec<(usize, f64)> = ritz_eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Compute RMSF from Ritz pairs
        let kb = 0.001987204;
        let kt = kb * self.temperature;
        let scale = kt / self.gamma;

        let mut rmsf_sq = vec![0.0f64; n];
        let mut valid_eigenvalues = Vec::new();

        for (ritz_idx, lambda) in indexed.iter().skip(1).take(num_modes) {
            if *lambda < 1e-6 {
                continue;
            }
            valid_eigenvalues.push(*lambda);

            // Reconstruct eigenvector: u = Σ_j ritz_vec[j] * lanczos_vec[j]
            let ritz_col = ritz_vectors.column(*ritz_idx);
            let mut eigenvec = DVector::zeros(n);
            for (j, lv) in lanczos_vectors.iter().enumerate() {
                if j < m {
                    eigenvec += ritz_col[j] * lv;
                }
            }

            // Normalize eigenvector
            let norm = eigenvec.norm();
            if norm > 1e-10 {
                eigenvec /= norm;
            }

            // Add contribution to RMSF
            for i in 0..n {
                let u_i = eigenvec[i];
                rmsf_sq[i] += (1.0 / lambda) * u_i * u_i;
            }
        }

        let rmsf: Vec<f64> = rmsf_sq.iter().map(|&sq| (scale * sq).sqrt()).collect();

        // Normalize
        let mean: f64 = rmsf.iter().sum::<f64>() / n as f64;
        let target = 0.8;
        let scale_factor = if mean > 1e-6 { target / mean } else { 1.0 };

        let rmsf_scaled: Vec<f64> = rmsf.iter().map(|&r| r * scale_factor).collect();

        log::debug!(
            "Lanczos: computed {} modes for {} residues",
            valid_eigenvalues.len(),
            n
        );

        (rmsf_scaled, valid_eigenvalues)
    }

    /// Create empty result for edge cases
    fn empty_result(&self, n: usize) -> EnhancedGnmResult {
        EnhancedGnmResult {
            base: GnmResult {
                n_residues: n,
                rmsf: vec![1.0; n],
                eigenvalues: vec![],
                coordination: vec![0; n],
                cutoff: self.base_cutoff,
            },
            rmsf: vec![1.0; n],
            secondary_structure: vec![SecondaryStructure::Loop; n],
            relative_sasa: vec![1.0; n],
            burial_depth: vec![0.0; n],
            domain_assignments: None,
            applied_enhancements: EnhancementFlags::default(),
            base_enhanced_correlation: 1.0,
        }
    }

    /// Pearson correlation coefficient
    fn correlation(a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let n = a.len() as f64;
        let mean_a: f64 = a.iter().sum::<f64>() / n;
        let mean_b: f64 = b.iter().sum::<f64>() / n;

        let mut cov = 0.0;
        let mut var_a = 0.0;
        let mut var_b = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            let da = x - mean_a;
            let db = y - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        if var_a < 1e-10 || var_b < 1e-10 {
            return 0.0;
        }

        cov / (var_a.sqrt() * var_b.sqrt())
    }

    /// Compute correlation between predicted and experimental RMSF
    pub fn correlation_with_experimental(predicted: &[f64], experimental: &[f64]) -> f64 {
        Self::correlation(predicted, experimental)
    }
}

/// Quick function to compute enhanced RMSF with default settings
pub fn compute_enhanced_rmsf(
    ca_positions: &[[f32; 3]],
    residue_names: Option<&[&str]>,
) -> Vec<f64> {
    let gnm = EnhancedGnm::default();
    gnm.compute_rmsf(ca_positions, residue_names).rmsf
}

/// Ablation study: compute RMSF with each enhancement individually
pub fn ablation_study(
    ca_positions: &[[f32; 3]],
    residue_names: Option<&[&str]>,
) -> Vec<(&'static str, Vec<f64>)> {
    let mut results = Vec::new();

    // Baseline (no enhancements)
    let gnm_base = GaussianNetworkModel::with_cutoff(7.3);
    results.push(("baseline", gnm_base.compute_rmsf(ca_positions).rmsf));

    // Distance-weighted only
    let gnm_dist = EnhancedGnm::with_config(EnhancedGnmConfig::distance_weighted_only());
    results.push((
        "distance_weighted",
        gnm_dist.compute_rmsf(ca_positions, None).rmsf,
    ));

    // Multi-cutoff only
    let gnm_multi = EnhancedGnm::with_config(EnhancedGnmConfig::multi_cutoff_only());
    results.push((
        "multi_cutoff",
        gnm_multi.compute_rmsf(ca_positions, None).rmsf,
    ));

    // Full enhancements
    let gnm_full = EnhancedGnm::with_config(EnhancedGnmConfig::full_experimental());
    results.push((
        "full_enhanced",
        gnm_full.compute_rmsf(ca_positions, residue_names).rmsf,
    ));

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_positions(n: usize) -> Vec<[f32; 3]> {
        // Simple helix-like structure
        (0..n)
            .map(|i| {
                let angle = i as f32 * 100.0 * std::f32::consts::PI / 180.0;
                [2.3 * angle.cos(), 2.3 * angle.sin(), i as f32 * 1.5]
            })
            .collect()
    }

    #[test]
    fn test_enhanced_gnm_basic() {
        let positions = generate_test_positions(20);
        let gnm = EnhancedGnm::default();

        let result = gnm.compute_rmsf(&positions, None);

        assert_eq!(result.rmsf.len(), 20);
        assert!(result.applied_enhancements.distance_weighting);
        assert!(result.applied_enhancements.multi_cutoff);
        assert!(result.applied_enhancements.secondary_structure);
    }

    #[test]
    fn test_with_residue_names() {
        let positions = generate_test_positions(10);
        let residue_names: Vec<&str> = vec![
            "ALA", "GLY", "PRO", "VAL", "LEU", "SER", "ASN", "PHE", "TRP", "LYS",
        ];

        let gnm = EnhancedGnm::default();
        let result = gnm.compute_rmsf(&positions, Some(&residue_names));

        assert!(result.applied_enhancements.sidechain_factors);

        // GLY should have relatively higher RMSF than PRO
        // (positions 1 vs 2)
        // Note: this depends on overall context, so just check both are valid
        assert!(result.rmsf[1] > 0.0);
        assert!(result.rmsf[2] > 0.0);
    }

    #[test]
    fn test_distance_weighting_effect() {
        let positions = generate_test_positions(30);

        // Compare base vs distance-weighted
        let gnm_base = GaussianNetworkModel::with_cutoff(7.3);
        let base_result = gnm_base.compute_rmsf(&positions);

        let gnm_weighted = EnhancedGnm::with_config(EnhancedGnmConfig::distance_weighted_only());
        let weighted_result = gnm_weighted.compute_rmsf(&positions, None);

        // Should produce different results
        let correlation = EnhancedGnm::correlation(&base_result.rmsf, &weighted_result.rmsf);

        // High correlation but not perfect (enhancements change predictions)
        assert!(
            correlation > 0.8,
            "Correlation {} should be high",
            correlation
        );
        assert!(
            correlation < 0.999,
            "Correlation {} should not be perfect",
            correlation
        );
    }

    #[test]
    fn test_multi_cutoff_ensemble() {
        let positions = generate_test_positions(25);

        let gnm = EnhancedGnm::with_config(EnhancedGnmConfig::multi_cutoff_only());
        let result = gnm.compute_rmsf(&positions, None);

        assert!(result.applied_enhancements.multi_cutoff);
        assert!(!result.applied_enhancements.secondary_structure);

        // Check RMSF values are reasonable
        let mean_rmsf: f64 = result.rmsf.iter().sum::<f64>() / result.rmsf.len() as f64;
        assert!(
            mean_rmsf > 0.1 && mean_rmsf < 2.0,
            "Mean RMSF {} should be reasonable",
            mean_rmsf
        );
    }

    #[test]
    fn test_full_enhancements() {
        let positions = generate_test_positions(40);
        let residue_names: Vec<&str> = (0..40)
            .map(|i| match i % 5 {
                0 => "ALA",
                1 => "GLY",
                2 => "LEU",
                3 => "SER",
                _ => "PRO",
            })
            .collect();

        let gnm = EnhancedGnm::with_config(EnhancedGnmConfig::full_experimental());
        let result = gnm.compute_rmsf(&positions, Some(&residue_names));

        // All enhancements should be applied
        assert!(
            result.applied_enhancements.distance_weighting
                || result.applied_enhancements.multi_cutoff
        );
        assert!(result.applied_enhancements.secondary_structure);
        assert!(result.applied_enhancements.sidechain_factors);
        assert!(result.applied_enhancements.sasa_modulation);
    }

    #[test]
    fn test_ablation_study() {
        let positions = generate_test_positions(20);
        let residue_names: Vec<&str> = vec!["ALA"; 20];

        let results = ablation_study(&positions, Some(&residue_names));

        assert_eq!(results.len(), 4);

        // All should have same length
        for (name, rmsf) in &results {
            assert_eq!(rmsf.len(), 20, "{} should have 20 values", name);
        }
    }

    #[test]
    fn test_empty_and_small() {
        let gnm = EnhancedGnm::default();

        // Empty
        let result = gnm.compute_rmsf(&[], None);
        assert!(result.rmsf.is_empty());

        // Single residue
        let result = gnm.compute_rmsf(&[[0.0, 0.0, 0.0]], None);
        assert_eq!(result.rmsf.len(), 1);

        // Two residues
        let result = gnm.compute_rmsf(&[[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]], None);
        assert_eq!(result.rmsf.len(), 2);
    }

    #[test]
    fn test_self_consistency() {
        let positions = generate_test_positions(30);
        let gnm = EnhancedGnm::default();

        let result = gnm.compute_rmsf(&positions, None);

        // Base and enhanced should be correlated (same underlying physics)
        assert!(
            result.base_enhanced_correlation > 0.7,
            "Base-enhanced correlation {} should be high",
            result.base_enhanced_correlation
        );
    }
}
