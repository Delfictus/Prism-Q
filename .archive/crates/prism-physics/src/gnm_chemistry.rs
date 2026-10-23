//! Chemistry-Aware Gaussian Network Model (CA-GNM)
//!
//! Extends the standard GNM with chemistry-weighted spring constants.
//! Instead of binary contacts (within cutoff → -1), we use:
//!
//! Γᵢⱼ = -k(i,j) × w_dist(dᵢⱼ) × w_burial(i,j) × w_hbond(i,j) × w_type(i,j)
//!
//! Where:
//! - k(i,j) = amino acid pair stiffness
//! - w_dist = distance weighting (Gaussian decay)
//! - w_burial = burial depth weighting (buried contacts are stiffer)
//! - w_hbond = hydrogen bond detection (H-bonds are stiffer)
//! - w_type = contact type (backbone-backbone vs sidechain)
//!
//! ## GPU Acceleration
//!
//! For large proteins (>500 residues), GPU-accelerated eigendecomposition
//! can be enabled via `enable_gpu()`. Uses Lanczos iteration on CUDA.

use nalgebra::{DMatrix, SymmetricEigen};
use std::collections::HashSet;

use crate::residue_chemistry::{enhanced_pair_stiffness, ResidueClass};

#[cfg(feature = "cuda")]
use crate::gnm_gpu::GpuGnm;

/// Configuration for Chemistry-Aware GNM
#[derive(Debug, Clone)]
pub struct ChemistryGnmConfig {
    /// Distance cutoff for contacts (Å)
    pub cutoff: f64,
    /// Sigma for Gaussian distance weighting
    pub sigma: f64,
    /// Enable amino acid pair stiffness weighting
    pub use_aa_stiffness: bool,
    /// Enable burial depth weighting
    pub use_burial_weighting: bool,
    /// Enable hydrogen bond detection
    pub use_hbond_detection: bool,
    /// Enable salt bridge detection
    pub use_salt_bridges: bool,
    /// Enable contact type classification
    pub use_contact_types: bool,
    /// Radius for burial depth calculation (Å)
    pub burial_radius: f64,
}

impl Default for ChemistryGnmConfig {
    fn default() -> Self {
        // Optimized based on ablation study:
        // - AA stiffness + burial weighting + contact types: beneficial
        // - H-bond detection: neutral (+0.0002)
        // - Salt bridges: HURTS accuracy (-0.004) - DISABLED
        // - Best combo: all except salt bridges → ρ=0.6204 (vs 0.6140 baseline)
        Self {
            cutoff: 9.0,
            sigma: 5.0,
            use_aa_stiffness: true,
            use_burial_weighting: true,
            use_hbond_detection: true, // Slight positive
            use_salt_bridges: false,   // DISABLED: hurts accuracy by 0.004
            use_contact_types: true,   // Beneficial
            burial_radius: 10.0,
        }
    }
}

impl ChemistryGnmConfig {
    /// Full experimental configuration with all features enabled.
    /// WARNING: Salt bridges and contact types hurt accuracy in benchmarks.
    pub fn full_experimental() -> Self {
        Self {
            cutoff: 9.0,
            sigma: 5.0,
            use_aa_stiffness: true,
            use_burial_weighting: true,
            use_hbond_detection: true,
            use_salt_bridges: true,
            use_contact_types: true,
            burial_radius: 10.0,
        }
    }
}

/// Contact type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContactType {
    /// Sequential neighbors (|i-j| ≤ 4)
    SequenceLocal,
    /// Backbone-backbone contact (close range)
    BackboneBackbone,
    /// Long-range tertiary contact (|i-j| > 12)
    LongRange,
    /// Sidechain-sidechain interaction
    SidechainSidechain,
    /// Mixed backbone-sidechain
    BackboneSidechain,
}

impl ContactType {
    /// Classify a contact based on sequence separation and distance
    pub fn classify(seq_sep: usize, dist: f64) -> Self {
        if seq_sep <= 4 {
            ContactType::SequenceLocal
        } else if seq_sep > 12 {
            ContactType::LongRange
        } else if dist < 6.0 {
            ContactType::BackboneBackbone
        } else {
            ContactType::SidechainSidechain
        }
    }

    /// Get the stiffness weight for this contact type
    pub fn weight(&self) -> f64 {
        match self {
            ContactType::SequenceLocal => 1.2, // Chain connectivity is stiff
            ContactType::BackboneBackbone => 1.1, // Backbone contacts stable
            ContactType::LongRange => 1.3,     // Tertiary contacts important
            ContactType::SidechainSidechain => 0.9, // More dynamic
            ContactType::BackboneSidechain => 1.0,
        }
    }
}

/// Result from Chemistry-Aware GNM computation
#[derive(Debug, Clone)]
pub struct ChemistryGnmResult {
    /// Per-residue RMSF values
    pub rmsf: Vec<f64>,
    /// Burial depth for each residue (0.0 = surface, 1.0 = core)
    pub burial_depth: Vec<f64>,
    /// Detected hydrogen bonds as (i, j) pairs
    pub hbonds: Vec<(usize, usize)>,
    /// Detected salt bridges as (i, j) pairs
    pub salt_bridges: Vec<(usize, usize)>,
    /// Eigenvalues from the Kirchhoff matrix
    pub eigenvalues: Vec<f64>,
    /// Mean correlation expected (for debugging)
    pub chemistry_factor_mean: f64,
}

/// Chemistry-Aware GNM implementation
pub struct ChemistryGnm {
    config: ChemistryGnmConfig,
    /// GPU accelerator for large proteins
    #[cfg(feature = "cuda")]
    gpu_gnm: Option<GpuGnm>,
    /// Threshold for GPU acceleration (number of residues)
    gpu_threshold: usize,
}

impl ChemistryGnm {
    /// Create with default configuration
    pub fn new() -> Self {
        Self {
            config: ChemistryGnmConfig::default(),
            #[cfg(feature = "cuda")]
            gpu_gnm: None,
            gpu_threshold: 500,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ChemistryGnmConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "cuda")]
            gpu_gnm: None,
            gpu_threshold: 500,
        }
    }

    /// Enable GPU acceleration for large proteins
    ///
    /// When enabled, proteins with more than `threshold` residues will
    /// use GPU-accelerated Lanczos eigendecomposition.
    #[cfg(feature = "cuda")]
    pub fn enable_gpu(&mut self, threshold: Option<usize>) -> anyhow::Result<()> {
        let mut gpu = GpuGnm::with_cutoff(self.config.cutoff);
        gpu.init_cuda()?;
        self.gpu_gnm = Some(gpu);
        if let Some(t) = threshold {
            self.gpu_threshold = t;
        }
        log::info!(
            "CA-GNM: GPU acceleration enabled (threshold: {} residues)",
            self.gpu_threshold
        );
        Ok(())
    }

    /// Check if GPU is enabled and initialized
    #[cfg(feature = "cuda")]
    pub fn gpu_enabled(&self) -> bool {
        self.gpu_gnm.as_ref().map_or(false, |g| g.gpu_ready())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn gpu_enabled(&self) -> bool {
        false
    }

    /// Set GPU threshold
    pub fn set_gpu_threshold(&mut self, threshold: usize) {
        self.gpu_threshold = threshold;
    }

    /// Compute burial depth for each residue using neighbor counting
    ///
    /// Buried residues have more neighbors within the burial radius.
    /// Returns values from 0.0 (surface) to 1.0 (fully buried).
    pub fn compute_burial_depth(&self, ca_positions: &[[f32; 3]]) -> Vec<f64> {
        let n = ca_positions.len();
        let radius_sq = self.config.burial_radius * self.config.burial_radius;
        let mut burial = vec![0.0; n];

        // Count neighbors for each residue
        let mut max_neighbors = 0usize;
        let mut neighbor_counts = vec![0usize; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist_sq = distance_squared(ca_positions[i], ca_positions[j]);
                if dist_sq < radius_sq as f64 {
                    neighbor_counts[i] += 1;
                }
            }
            max_neighbors = max_neighbors.max(neighbor_counts[i]);
        }

        // Normalize to 0-1 range
        if max_neighbors > 0 {
            for i in 0..n {
                burial[i] = neighbor_counts[i] as f64 / max_neighbors as f64;
            }
        }

        burial
    }

    /// Detect backbone hydrogen bonds from Cα geometry
    ///
    /// Uses geometric criteria based on Cα-Cα distances:
    /// - α-helix: i to i+4 at ~6.3Å (i to i+3 at ~5.4Å)
    /// - β-sheet: long-range at ~4.5-5.5Å (antiparallel) or ~6.5Å (parallel)
    pub fn detect_hbonds(&self, ca_positions: &[[f32; 3]]) -> Vec<(usize, usize)> {
        let n = ca_positions.len();
        let mut hbonds = Vec::new();

        for i in 0..n {
            for j in (i + 3)..n {
                // At least 3 residues apart
                let dist = distance(ca_positions[i], ca_positions[j]);

                // α-helix pattern: i,i+3 or i,i+4 at specific distances
                if j == i + 3 && dist > 4.5 && dist < 5.8 {
                    hbonds.push((i, j));
                } else if j == i + 4 && dist > 5.0 && dist < 7.0 {
                    hbonds.push((i, j));
                }
                // β-sheet pattern: long-range at ~4.5-6.5Å
                else if j > i + 4 && dist > 4.0 && dist < 7.0 {
                    hbonds.push((i, j));
                }
            }
        }

        hbonds
    }

    /// Detect salt bridges between oppositely charged residues
    ///
    /// Salt bridges form when:
    /// - One residue is positively charged (ARG, LYS, HIS)
    /// - One residue is negatively charged (ASP, GLU)
    /// - Distance is < 8Å (sidechain proximity)
    pub fn detect_salt_bridges(
        &self,
        ca_positions: &[[f32; 3]],
        residue_names: &[&str],
    ) -> Vec<(usize, usize)> {
        let n = ca_positions.len();
        let mut bridges = Vec::new();

        for i in 0..n {
            let class_i = ResidueClass::from_name(residue_names[i]);
            for j in (i + 1)..n {
                let dist = distance(ca_positions[i], ca_positions[j]);
                if dist > 8.0 {
                    continue;
                }

                let class_j = ResidueClass::from_name(residue_names[j]);

                if (class_i.is_positive() && class_j.is_negative())
                    || (class_i.is_negative() && class_j.is_positive())
                {
                    bridges.push((i, j));
                }
            }
        }

        bridges
    }

    /// Build the chemistry-aware Kirchhoff matrix
    ///
    /// Each spring constant is weighted by multiple chemistry factors:
    /// Γᵢⱼ = -w_dist × w_aa × w_burial × w_hbond × w_salt × w_type
    pub fn build_kirchhoff(
        &self,
        ca_positions: &[[f32; 3]],
        residue_names: &[&str],
    ) -> (DMatrix<f64>, f64) {
        let n = ca_positions.len();
        let cutoff_sq = self.config.cutoff * self.config.cutoff;
        let sigma_sq = self.config.sigma * self.config.sigma;

        // Pre-compute chemistry factors
        let burial = if self.config.use_burial_weighting {
            self.compute_burial_depth(ca_positions)
        } else {
            vec![0.0; n]
        };

        let hbonds: HashSet<(usize, usize)> = if self.config.use_hbond_detection {
            self.detect_hbonds(ca_positions).into_iter().collect()
        } else {
            HashSet::new()
        };

        let salt_bridges: HashSet<(usize, usize)> = if self.config.use_salt_bridges {
            self.detect_salt_bridges(ca_positions, residue_names)
                .into_iter()
                .collect()
        } else {
            HashSet::new()
        };

        let mut kirchhoff = DMatrix::zeros(n, n);
        let mut total_weight = 0.0;
        let mut contact_count = 0usize;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist_sq = distance_squared(ca_positions[i], ca_positions[j]);

                if dist_sq < cutoff_sq {
                    let dist = dist_sq.sqrt();

                    // 1. Distance weighting (Gaussian decay)
                    let w_dist = (-dist_sq / (2.0 * sigma_sq)).exp();

                    // 2. Amino acid pair stiffness
                    let w_aa = if self.config.use_aa_stiffness {
                        enhanced_pair_stiffness(residue_names[i], residue_names[j])
                    } else {
                        1.0
                    };

                    // 3. Burial weighting (buried contacts are 20-40% stiffer)
                    let w_burial = if self.config.use_burial_weighting {
                        let avg_burial = (burial[i] + burial[j]) / 2.0;
                        1.0 + 0.4 * avg_burial
                    } else {
                        1.0
                    };

                    // 4. Hydrogen bond detection (50% stiffer)
                    let is_hbond = hbonds.contains(&(i, j)) || hbonds.contains(&(j, i));
                    let w_hbond = if is_hbond { 1.5 } else { 1.0 };

                    // 5. Salt bridge detection (40% stiffer)
                    let is_salt = salt_bridges.contains(&(i, j)) || salt_bridges.contains(&(j, i));
                    let w_salt = if is_salt { 1.4 } else { 1.0 };

                    // 6. Contact type classification
                    let w_type = if self.config.use_contact_types {
                        let seq_sep = j - i;
                        ContactType::classify(seq_sep, dist).weight()
                    } else {
                        1.0
                    };

                    // Combined spring constant
                    let k = w_dist * w_aa * w_burial * w_hbond * w_salt * w_type;

                    kirchhoff[(i, j)] = -k;
                    kirchhoff[(j, i)] = -k;

                    total_weight += k;
                    contact_count += 1;
                }
            }
        }

        // Set diagonal (sum of row)
        for i in 0..n {
            let row_sum: f64 = kirchhoff.row(i).iter().sum();
            kirchhoff[(i, i)] = -row_sum;
        }

        let mean_weight = if contact_count > 0 {
            total_weight / contact_count as f64
        } else {
            1.0
        };

        (kirchhoff, mean_weight)
    }

    /// Compute RMSF using Chemistry-Aware GNM
    ///
    /// For proteins with more than `gpu_threshold` residues (default 500),
    /// GPU-accelerated Lanczos eigendecomposition is used when available.
    pub fn compute_rmsf(
        &self,
        ca_positions: &[[f32; 3]],
        residue_names: &[&str],
    ) -> ChemistryGnmResult {
        let n = ca_positions.len();

        if n < 3 {
            return ChemistryGnmResult {
                rmsf: vec![1.0; n],
                burial_depth: vec![0.0; n],
                hbonds: vec![],
                salt_bridges: vec![],
                eigenvalues: vec![],
                chemistry_factor_mean: 1.0,
            };
        }

        // Build chemistry-weighted Kirchhoff matrix
        let (kirchhoff, chemistry_factor_mean) = self.build_kirchhoff(ca_positions, residue_names);

        // Choose eigendecomposition path based on matrix size and GPU availability
        let (rmsf, eigenvalues) = if n >= self.gpu_threshold && self.gpu_enabled() {
            // GPU path for large proteins
            self.compute_eigen_gpu(&kirchhoff, n)
        } else {
            // CPU path (full eigendecomposition for small proteins, Lanczos for medium)
            self.compute_eigen_cpu(&kirchhoff, n)
        };

        // Collect auxiliary data
        let burial_depth = if self.config.use_burial_weighting {
            self.compute_burial_depth(ca_positions)
        } else {
            vec![0.0; n]
        };

        let hbonds = if self.config.use_hbond_detection {
            self.detect_hbonds(ca_positions)
        } else {
            vec![]
        };

        let salt_bridges = if self.config.use_salt_bridges {
            self.detect_salt_bridges(ca_positions, residue_names)
        } else {
            vec![]
        };

        ChemistryGnmResult {
            rmsf,
            burial_depth,
            hbonds,
            salt_bridges,
            eigenvalues,
            chemistry_factor_mean,
        }
    }

    /// CPU eigendecomposition path
    fn compute_eigen_cpu(&self, kirchhoff: &DMatrix<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
        // Full eigendecomposition using nalgebra
        let eigen = SymmetricEigen::new(kirchhoff.clone());
        let eigenvalues = eigen.eigenvalues.as_slice().to_vec();
        let eigenvectors = eigen.eigenvectors;

        // Sort eigenvalues and get indices
        let mut sorted_indices: Vec<usize> = (0..n).collect();
        sorted_indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

        // Compute RMSF from inverse eigenvalues (skip first trivial mode)
        let mut rmsf = vec![0.0; n];

        for k in 1..n {
            let idx = sorted_indices[k];
            let lambda = eigenvalues[idx];
            if lambda.abs() < 1e-6 {
                continue;
            }

            for i in 0..n {
                let v = eigenvectors[(i, idx)];
                rmsf[i] += v * v / lambda;
            }
        }

        // Normalize RMSF
        let max_rmsf = rmsf.iter().cloned().fold(0.0, f64::max);
        if max_rmsf > 1e-10 {
            for r in rmsf.iter_mut() {
                *r = (*r / max_rmsf).sqrt();
            }
        }

        (rmsf, eigenvalues)
    }

    /// GPU eigendecomposition path using Lanczos iteration
    #[cfg(feature = "cuda")]
    fn compute_eigen_gpu(&self, kirchhoff: &DMatrix<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
        if let Some(ref gpu) = self.gpu_gnm {
            // Use GPU GNM which internally uses Lanczos
            // Note: GpuGnm builds its own Kirchhoff matrix from CA positions,
            // but we've already built chemistry-weighted one. We pass the
            // pre-built matrix via the RMSF computation.
            let result = gpu.compute_rmsf_from_kirchhoff(kirchhoff);
            log::info!("CA-GNM: Used GPU Lanczos for {} residues", n);
            (result.rmsf, result.eigenvalues)
        } else {
            // Fallback to CPU
            log::debug!("CA-GNM: GPU not available, using CPU");
            self.compute_eigen_cpu(kirchhoff, n)
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn compute_eigen_gpu(&self, kirchhoff: &DMatrix<f64>, n: usize) -> (Vec<f64>, Vec<f64>) {
        self.compute_eigen_cpu(kirchhoff, n)
    }
}

impl Default for ChemistryGnm {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate squared Euclidean distance between two 3D points
fn distance_squared(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dx = (b[0] - a[0]) as f64;
    let dy = (b[1] - a[1]) as f64;
    let dz = (b[2] - a[2]) as f64;
    dx * dx + dy * dy + dz * dz
}

/// Calculate Euclidean distance between two 3D points
fn distance(a: [f32; 3], b: [f32; 3]) -> f64 {
    distance_squared(a, b).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_helix_positions(n: usize) -> Vec<[f32; 3]> {
        // Generate idealized α-helix positions
        // Rise per residue: 1.5Å, rotation: 100°
        (0..n)
            .map(|i| {
                let angle = (i as f64) * 100.0_f64.to_radians();
                let x = 2.3 * angle.cos();
                let y = 2.3 * angle.sin();
                let z = (i as f64) * 1.5;
                [x as f32, y as f32, z as f32]
            })
            .collect()
    }

    #[test]
    fn test_burial_depth() {
        let positions = make_helix_positions(20);
        let gnm = ChemistryGnm::new();
        let burial = gnm.compute_burial_depth(&positions);

        assert_eq!(burial.len(), 20);
        // Terminal residues should be less buried
        assert!(burial[0] < burial[10]);
        assert!(burial[19] < burial[10]);
    }

    #[test]
    fn test_hbond_detection() {
        let positions = make_helix_positions(20);
        let gnm = ChemistryGnm::new();
        let hbonds = gnm.detect_hbonds(&positions);

        // α-helix should have i,i+4 hydrogen bonds
        assert!(!hbonds.is_empty());
    }

    #[test]
    fn test_compute_rmsf() {
        let positions = make_helix_positions(20);
        let residues: Vec<&str> = vec!["ALA"; 20];
        let gnm = ChemistryGnm::new();
        let result = gnm.compute_rmsf(&positions, &residues);

        assert_eq!(result.rmsf.len(), 20);
        // Terminal residues should be more flexible
        assert!(result.rmsf[0] > result.rmsf[10]);
        assert!(result.rmsf[19] > result.rmsf[10]);
    }

    #[test]
    fn test_chemistry_factor_effect() {
        let positions = make_helix_positions(10);

        // All GLY (very flexible)
        let gly_residues: Vec<&str> = vec!["GLY"; 10];
        let gnm = ChemistryGnm::new();
        let gly_result = gnm.compute_rmsf(&positions, &gly_residues);

        // All PRO (very rigid)
        let pro_residues: Vec<&str> = vec!["PRO"; 10];
        let pro_result = gnm.compute_rmsf(&positions, &pro_residues);

        // Chemistry factor should differ significantly
        assert!(gly_result.chemistry_factor_mean < pro_result.chemistry_factor_mean);
    }
}
