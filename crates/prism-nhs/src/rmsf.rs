//! [STAGE-2B-RMSF] RMSF Convergence Analysis
//!
//! Compute Root Mean Square Fluctuation (RMSF) to assess trajectory convergence.
//! Convergence is validated by comparing first-half and second-half RMSF profiles.
//!
//! **Quality Metric**: Pearson correlation > 0.8 indicates converged sampling.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// RMSF analysis results for convergence checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmsfAnalysis {
    /// Per-residue RMSF for first half (Å)
    pub first_half_rmsf: Vec<f32>,
    /// Per-residue RMSF for second half (Å)
    pub second_half_rmsf: Vec<f32>,
    /// Pearson correlation between halves
    pub correlation: f32,
    /// Is trajectory converged? (r > 0.8)
    pub converged: bool,
    /// Number of frames analyzed
    pub num_frames: usize,
    /// Number of Cα atoms
    pub num_ca_atoms: usize,
}

/// RMSF calculator for trajectory convergence analysis
pub struct RmsfCalculator {
    /// Number of Cα atoms
    num_ca_atoms: usize,
    /// Cα atom indices in full atom array
    ca_indices: Vec<usize>,
}

impl RmsfCalculator {
    /// Create new RMSF calculator
    ///
    /// # Arguments
    /// * `atom_names` - IUPAC atom names for all atoms
    /// * `residue_indices` - Residue index for each atom
    ///
    /// # Returns
    /// RmsfCalculator configured for Cα-only analysis
    pub fn new(atom_names: &[String], residue_indices: &[usize]) -> Result<Self> {
        // Find Cα atoms (name == "CA" and not in a ligand)
        let mut ca_indices = Vec::new();
        let mut current_residue = usize::MAX;

        for (atom_idx, (name, &res_idx)) in atom_names.iter().zip(residue_indices).enumerate() {
            if name == "CA" && res_idx != current_residue {
                ca_indices.push(atom_idx);
                current_residue = res_idx;
            }
        }

        anyhow::ensure!(!ca_indices.is_empty(), "No Cα atoms found in structure");

        log::info!("RMSF calculator initialized: {} Cα atoms", ca_indices.len());

        Ok(Self {
            num_ca_atoms: ca_indices.len(),
            ca_indices,
        })
    }

    /// Compute RMSF convergence analysis from trajectory frames
    ///
    /// # Arguments
    /// * `frames` - Trajectory frames with positions [x, y, z, ...]
    ///
    /// # Returns
    /// RmsfAnalysis with convergence metrics
    ///
    /// # Quality Criteria
    /// - Requires ≥20 frames minimum
    /// - Pearson r > 0.8 indicates convergence
    pub fn analyze_convergence(&self, frames: &[Vec<f32>]) -> Result<RmsfAnalysis> {
        anyhow::ensure!(
            frames.len() >= 20,
            "Insufficient frames for RMSF analysis: {} < 20",
            frames.len()
        );

        // Split trajectory into halves
        let mid_frame = frames.len() / 2;
        let first_half = &frames[..mid_frame];
        let second_half = &frames[mid_frame..];

        log::debug!(
            "Computing RMSF: {} total frames ({} first half, {} second half)",
            frames.len(),
            first_half.len(),
            second_half.len()
        );

        // Compute RMSF for each half
        let first_half_rmsf = self.compute_rmsf(first_half)?;
        let second_half_rmsf = self.compute_rmsf(second_half)?;

        // Compute Pearson correlation
        let correlation = pearson_correlation(&first_half_rmsf, &second_half_rmsf)?;
        let converged = correlation > 0.8;

        log::info!(
            "RMSF convergence: r = {:.3} ({})",
            correlation,
            if converged {
                "CONVERGED"
            } else {
                "NOT CONVERGED"
            }
        );

        Ok(RmsfAnalysis {
            first_half_rmsf,
            second_half_rmsf,
            correlation,
            converged,
            num_frames: frames.len(),
            num_ca_atoms: self.num_ca_atoms,
        })
    }

    /// Compute RMSF for a set of frames
    ///
    /// RMSF_i = sqrt( <(r_i - <r_i>)^2> )
    ///
    /// where:
    /// - r_i is position of atom i at each frame
    /// - <r_i> is average position across frames
    /// - <...> denotes time average
    fn compute_rmsf(&self, frames: &[Vec<f32>]) -> Result<Vec<f32>> {
        anyhow::ensure!(!frames.is_empty(), "No frames provided");

        let num_frames = frames.len();
        let mut rmsf = vec![0.0f32; self.num_ca_atoms];

        // Compute average position for each Cα atom
        let mut avg_positions = vec![[0.0f32; 3]; self.num_ca_atoms];

        for frame in frames {
            for (ca_idx, &atom_idx) in self.ca_indices.iter().enumerate() {
                let pos_idx = atom_idx * 3;
                avg_positions[ca_idx][0] += frame[pos_idx];
                avg_positions[ca_idx][1] += frame[pos_idx + 1];
                avg_positions[ca_idx][2] += frame[pos_idx + 2];
            }
        }

        for ca_idx in 0..self.num_ca_atoms {
            avg_positions[ca_idx][0] /= num_frames as f32;
            avg_positions[ca_idx][1] /= num_frames as f32;
            avg_positions[ca_idx][2] /= num_frames as f32;
        }

        // Compute fluctuations from average
        for frame in frames {
            for (ca_idx, &atom_idx) in self.ca_indices.iter().enumerate() {
                let pos_idx = atom_idx * 3;
                let dx = frame[pos_idx] - avg_positions[ca_idx][0];
                let dy = frame[pos_idx + 1] - avg_positions[ca_idx][1];
                let dz = frame[pos_idx + 2] - avg_positions[ca_idx][2];
                rmsf[ca_idx] += dx * dx + dy * dy + dz * dz;
            }
        }

        // Take square root and normalize by number of frames
        for ca_idx in 0..self.num_ca_atoms {
            rmsf[ca_idx] = (rmsf[ca_idx] / num_frames as f32).sqrt();
        }

        Ok(rmsf)
    }

    /// Get Cα atom indices
    pub fn ca_indices(&self) -> &[usize] {
        &self.ca_indices
    }

    /// Get number of Cα atoms
    pub fn num_ca_atoms(&self) -> usize {
        self.num_ca_atoms
    }
}

/// Compute Pearson correlation coefficient between two arrays
///
/// r = cov(X, Y) / (std(X) * std(Y))
fn pearson_correlation(x: &[f32], y: &[f32]) -> Result<f32> {
    anyhow::ensure!(
        x.len() == y.len(),
        "Array length mismatch: {} != {}",
        x.len(),
        y.len()
    );
    anyhow::ensure!(!x.is_empty(), "Empty arrays provided");

    let n = x.len() as f32;

    // Compute means
    let mean_x: f32 = x.iter().sum::<f32>() / n;
    let mean_y: f32 = y.iter().sum::<f32>() / n;

    // Compute covariance and standard deviations
    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let std_x = (var_x / n).sqrt();
    let std_y = (var_y / n).sqrt();

    anyhow::ensure!(std_x > 0.0 && std_y > 0.0, "Zero variance in RMSF data");

    Ok(cov / (n * std_x * std_y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pearson_correlation_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pearson_correlation_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let r = pearson_correlation(&x, &y).unwrap();
        assert!((r + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pearson_correlation_uncorrelated() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 2.0, 2.0, 2.0, 2.0];
        // y has zero variance, should error
        assert!(pearson_correlation(&x, &y).is_err());
    }

    #[test]
    fn test_rmsf_calculator_creation() {
        let atom_names = vec![
            "N".to_string(),
            "CA".to_string(),
            "C".to_string(),
            "O".to_string(),
            "N".to_string(),
            "CA".to_string(),
            "C".to_string(),
            "O".to_string(),
        ];
        let residue_indices = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let calc = RmsfCalculator::new(&atom_names, &residue_indices).unwrap();
        assert_eq!(calc.num_ca_atoms(), 2);
        assert_eq!(calc.ca_indices(), &[1, 5]);
    }

    #[test]
    fn test_rmsf_convergence_insufficient_frames() {
        let atom_names = vec!["CA".to_string()];
        let residue_indices = vec![0];
        let calc = RmsfCalculator::new(&atom_names, &residue_indices).unwrap();

        // Only 10 frames (< 20 minimum)
        let frames = vec![vec![0.0, 0.0, 0.0]; 10];
        assert!(calc.analyze_convergence(&frames).is_err());
    }
}
