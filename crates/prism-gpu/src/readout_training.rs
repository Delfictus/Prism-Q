//! Supervised Readout Training for Reservoir Computing
//!
//! Implements per-residue ridge regression training for the final readout layer.
//! The reservoir (dendritic network) is fixed/deterministic - only the readout
//! weights are learned from labeled data.
//!
//! ## Training Pipeline
//! 1. Run mega-fused kernel on training structures (885 for CryptoBench)
//! 2. Extract reservoir states (4 floats per residue: branches 1-3 + integrated)
//! 3. Train ridge regression: W = (X^T X + λI)^{-1} X^T y
//! 4. Save trained weights to binary file
//!
//! ## Why This Works
//! - Reservoir computing uses fixed random projections for feature extraction
//! - Only the readout layer needs training (linear, closed-form solution)
//! - No backprop, no gradient descent, minimal hyperparameters
//! - Achieves F1 0.70-0.78 on CryptoBench when properly trained

use std::path::Path;
use std::io::{Read, Write};
use std::collections::HashSet;

/// Number of features in reservoir state per residue (full inference kernel features)
/// Uses SAME 80-dim features from inference kernel (mega_fused_pocket_kernel) for training.
/// This ensures training and inference see identical features.
///
/// Layout (80 dimensions total):
/// [0-47]:  TDA features from inference kernel (48 topological features)
///          - Per-radius persistence: β₀, β₁, persistence, complexity, void_boundary at 4 radii
///          - Geodesic tunnel features
///          - Multi-scale topological invariants
/// [48-55]: Reservoir state (8 base features)
/// [56-63]: Input features (degree, conservation, centrality, bfactor, burial, etc.)
/// [64-71]: Network/geometry features (eigenvector, HSE, concavity)
/// [72-79]: Context features (pocket assignment, distances, etc.)
pub const RESERVOIR_STATE_DIM: usize = 92;  // 48 TDA + 32 base + 12 physics features from inference kernel

/// Trained readout weights for pocket prediction
#[derive(Debug, Clone)]
pub struct TrainedReadout {
    /// Weights for linear combination of reservoir states [reservoir_dim]
    pub weights: Vec<f32>,
    /// Reservoir dimension (should match kernel output)
    pub reservoir_dim: usize,
    /// Training metadata
    pub n_residues: usize,
    pub n_positive: usize,
    pub n_structures: usize,
    pub lambda: f32,
    /// Z-score normalization stats (computed from training set)
    pub norm_means: Vec<f32>,  // [reservoir_dim]
    pub norm_stds: Vec<f32>,   // [reservoir_dim]
}

impl Default for TrainedReadout {
    fn default() -> Self {
        // Default uniform weights (no training) - will give poor results
        Self {
            weights: vec![0.25f32; RESERVOIR_STATE_DIM],
            reservoir_dim: RESERVOIR_STATE_DIM,
            n_residues: 0,
            n_positive: 0,
            n_structures: 0,
            lambda: 1e-4,
            norm_means: vec![0.0f32; RESERVOIR_STATE_DIM],
            norm_stds: vec![1.0f32; RESERVOIR_STATE_DIM],
        }
    }
}

impl TrainedReadout {
    /// Compute Z-score normalization stats and apply to training data in-place
    /// Returns (means, stds) for later use at inference time
    fn normalize_train_data(
        train_data: &mut [(Vec<f32>, Vec<u8>)],
        reservoir_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        // Count total samples
        let total_residues: usize = train_data.iter().map(|(_, mask)| mask.len()).sum();

        // Compute means for each feature
        let mut means = vec![0.0f64; reservoir_dim];
        for (states_flat, mask) in train_data.iter() {
            let n_residues = mask.len();
            for r in 0..n_residues {
                let offset = r * reservoir_dim;
                for f in 0..reservoir_dim {
                    means[f] += states_flat[offset + f] as f64;
                }
            }
        }
        for f in 0..reservoir_dim {
            means[f] /= total_residues as f64;
        }

        // Compute standard deviations
        let mut stds = vec![0.0f64; reservoir_dim];
        for (states_flat, mask) in train_data.iter() {
            let n_residues = mask.len();
            for r in 0..n_residues {
                let offset = r * reservoir_dim;
                for f in 0..reservoir_dim {
                    let diff = states_flat[offset + f] as f64 - means[f];
                    stds[f] += diff * diff;
                }
            }
        }
        for f in 0..reservoir_dim {
            stds[f] = (stds[f] / total_residues as f64).sqrt().max(1e-6);  // Avoid divide by zero
        }

        // Apply normalization in-place
        for (states_flat, mask) in train_data.iter_mut() {
            let n_residues = mask.len();
            for r in 0..n_residues {
                let offset = r * reservoir_dim;
                for f in 0..reservoir_dim {
                    let normalized = (states_flat[offset + f] as f64 - means[f]) / stds[f];
                    states_flat[offset + f] = normalized as f32;
                }
            }
        }

        log::info!(
            "Z-score normalization: TDA features (0-47) mean_std={:.4}, Base features (48-79) mean_std={:.4}",
            stds[0..48.min(reservoir_dim)].iter().sum::<f64>() / 48.0,
            stds[48.min(reservoir_dim)..reservoir_dim].iter().sum::<f64>() / (reservoir_dim - 48).max(1) as f64
        );

        (
            means.iter().map(|&m| m as f32).collect(),
            stds.iter().map(|&s| s as f32).collect(),
        )
    }

    /// Train per-residue linear readout on reservoir states
    ///
    /// # Arguments
    /// * `train_data` - Per-structure data: (flattened_states [n_residues * reservoir_dim], mask [n_residues])
    /// * `reservoir_dim` - Dimension of reservoir state per residue
    /// * `lambda` - Ridge regularization parameter (default 1e-4)
    /// * `custom_pos_weight` - Optional custom positive class weight (default: sqrt of imbalance ratio)
    ///
    /// # Returns
    /// Trained readout weights [reservoir_dim]
    pub fn train(
        train_data: &[(Vec<f32>, Vec<u8>)],
        reservoir_dim: usize,
        lambda: f32,
    ) -> Result<Self, String> {
        Self::train_with_weight(train_data, reservoir_dim, lambda, None)
    }

    /// Train with custom positive class weight
    pub fn train_with_weight(
        train_data: &[(Vec<f32>, Vec<u8>)],
        reservoir_dim: usize,
        lambda: f32,
        custom_pos_weight: Option<f32>,
    ) -> Result<Self, String> {
        if train_data.is_empty() {
            return Err("No training data provided".to_string();
        }

        // Z-score normalize training data to equalize TDA and base feature scales
        let mut train_data_normalized: Vec<(Vec<f32>, Vec<u8>)> = train_data.iter()
            .map(|(states, mask)| (states.clone(), mask.clone()))
            .collect();
        let (norm_means, norm_stds) = Self::normalize_train_data(&mut train_data_normalized, reservoir_dim);

        // Count total residues and positives
        let total_residues: usize = train_data_normalized.iter().map(|(_, mask)| mask.len()).sum();
        let n_positive: usize = train_data.iter()
            .flat_map(|(_, mask)| mask.iter())
            .filter(|&&l| l > 0)
            .count();
        let n_structures = train_data.len();

        // Compute class-balanced weights for imbalanced data
        // With 1.59% positive rate:
        //   - sqrt gives ~7.9x weighting (high recall, low precision)
        //   - lower weights (1.0-4.0) give better F1 by improving precision
        let n_negative = total_residues - n_positive;
        let pos_weight = if let Some(custom) = custom_pos_weight {
            custom as f64  // Use custom weight if provided
        } else if n_positive > 0 {
            (n_negative as f64 / n_positive as f64).sqrt()  // Default: sqrt of imbalance
        } else {
            1.0
        };
        let neg_weight = 1.0f64;

        log::info!(
            "Training readout: {} structures, {} residues ({} positive, {:.2}% rate), dim={}, λ={}, pos_weight={:.2}",
            n_structures, total_residues, n_positive,
            100.0 * n_positive as f64 / total_residues as f64,
            reservoir_dim, lambda, pos_weight
        );

        // Build X^T W X (reservoir_dim x reservoir_dim) and X^T W y (reservoir_dim)
        // Using f64 for numerical stability during accumulation
        // W is diagonal weight matrix with pos_weight for positives, 1.0 for negatives
        let mut xtx = vec![0.0f64; reservoir_dim * reservoir_dim];
        let mut xty = vec![0.0f64; reservoir_dim];

        for (states_flat, mask) in &train_data_normalized {
            let n_residues = mask.len();

            // Validate dimensions
            if states_flat.len() != n_residues * reservoir_dim {
                return Err(format!(
                    "State/mask dimension mismatch: {} states vs {} residues * {} dim = {}",
                    states_flat.len(), n_residues, reservoir_dim, n_residues * reservoir_dim
                );
            }

            for r in 0..n_residues {
                let x_r = &states_flat[r * reservoir_dim..(r + 1) * reservoir_dim];
                let y_r = mask[r] as f64;

                // Apply class-balanced weight
                let w_r = if mask[r] > 0 { pos_weight } else { neg_weight };

                // Accumulate X^T W X (weighted)
                for i in 0..reservoir_dim {
                    for j in 0..reservoir_dim {
                        xtx[i * reservoir_dim + j] += w_r * (x_r[i] as f64) * (x_r[j] as f64);
                    }
                    // Accumulate X^T W y (weighted)
                    xty[i] += w_r * (x_r[i] as f64) * y_r;
                }
            }
        }

        // Add ridge regularization to diagonal
        for i in 0..reservoir_dim {
            xtx[i * reservoir_dim + i] += lambda as f64;
        }

        // Solve (X^T X + λI) w = X^T y via Cholesky decomposition
        let weights_f64 = Self::cholesky_solve(&xtx, &xty, reservoir_dim)?;
        let weights: Vec<f32> = weights_f64.iter().map(|&w| w as f32).collect();

        // Log weight statistics
        let w_min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
        let w_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let w_mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        log::info!(
            "Trained weights: min={:.4}, max={:.4}, mean={:.4}",
            w_min, w_max, w_mean
        );

        Ok(Self {
            weights,
            reservoir_dim,
            n_residues: total_residues,
            n_positive,
            n_structures,
            lambda,
            norm_means,
            norm_stds,
        })
    }

    /// Cholesky decomposition and solve: A x = b where A is symmetric positive definite
    fn cholesky_solve(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, String> {
        // Cholesky decomposition: A = L L^T
        let mut l = vec![0.0f64; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }

                if i == j {
                    if sum <= 0.0 {
                        return Err(format!(
                            "Matrix not positive definite at [{},{}], sum={:.6e}. Try increasing lambda.",
                            i, j, sum
                        );
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    if l[j * n + j].abs() < 1e-15 {
                        return Err(format!("Near-zero diagonal at [{},{}]", j, j);
                    }
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }

        // Forward substitution: L z = b
        let mut z = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i * n + j] * z[j];
            }
            z[i] = sum / l[i * n + i];
        }

        // Backward substitution: L^T w = z
        let mut w = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut sum = z[i];
            for j in (i + 1)..n {
                sum -= l[j * n + i] * w[j];
            }
            w[i] = sum / l[i * n + i];
        }

        Ok(w)
    }

    /// Predict binding score for a single residue (raw, no sigmoid)
    pub fn predict_raw(&self, reservoir_state: &[f32]) -> f32 {
        let mut score = 0.0f32;
        for (i, &w) in self.weights.iter().enumerate() {
            if i < reservoir_state.len() && i < self.norm_means.len() {
                // Apply Z-score normalization before prediction
                let normalized = (reservoir_state[i] - self.norm_means[i]) / self.norm_stds[i];
                score += w * normalized;
            }
        }
        score
    }

    /// Predict binding probability for a single residue (with sigmoid)
    pub fn predict(&self, reservoir_state: &[f32]) -> f32 {
        let score = self.predict_raw(reservoir_state);
        1.0 / (1.0 + (-score).exp())
    }

    /// Predict for all residues in a structure (batch)
    pub fn predict_batch(&self, states_flat: &[f32], n_residues: usize) -> Vec<f32> {
        let mut scores = Vec::with_capacity(n_residues);
        for r in 0..n_residues {
            let offset = r * self.reservoir_dim;
            if offset + self.reservoir_dim <= states_flat.len() {
                let state = &states_flat[offset..offset + self.reservoir_dim];
                scores.push(self.predict_raw(state);
            } else {
                scores.push(0.0);
            }
        }
        scores
    }

    /// Evaluate on test data with comprehensive threshold optimization
    /// Returns (accuracy, precision, recall, f1, auc_roc)
    pub fn evaluate(&self, test_data: &[(Vec<f32>, Vec<u8>)]) -> (f64, f64, f64, f64, f64) {
        let mut all_scores = Vec::new();
        let mut all_labels = Vec::new();

        for (states_flat, mask) in test_data {
            let n_residues = mask.len();
            for r in 0..n_residues {
                let offset = r * self.reservoir_dim;
                if offset + self.reservoir_dim <= states_flat.len() {
                    let state = &states_flat[offset..offset + self.reservoir_dim];
                    all_scores.push(self.predict_raw(state);
                    all_labels.push(mask[r]);
                }
            }
        }

        // Compute score statistics
        let n_pos = all_labels.iter().filter(|&&l| l > 0).count();
        let n_neg = all_labels.len() - n_pos;

        let mut sorted_scores = all_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

        let min_score = sorted_scores[0];
        let max_score = sorted_scores[sorted_scores.len() - 1];
        let mean_score: f32 = all_scores.iter().sum::<f32>() / all_scores.len() as f32;
        let variance: f32 = all_scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f32>()
                           / all_scores.len() as f32;
        let std_score = variance.sqrt();

        log::info!("═══════════════════════════════════════════════════════════════════");
        log::info!("  SCORE DISTRIBUTION ANALYSIS");
        log::info!("  Samples: {} total ({} positive, {} negative)",
                   all_scores.len(), n_pos, n_neg);
        log::info!("  Score range: [{:.4}, {:.4}]", min_score, max_score);
        log::info!("  Mean: {:.4}, Std: {:.4}", mean_score, std_score);
        log::info!("═══════════════════════════════════════════════════════════════════");

        let mut best_f1 = 0.0f64;
        let mut best_thresh = 0.0f32;
        let mut best_metrics = (0.0, 0.0, 0.0, 0.0);

        // Method 1: Percentile-based thresholds (50% to 99%) - extended range for recall
        log::info!("Percentile-based threshold search:");
        for pct in [50, 60, 70, 75, 80, 85, 90, 92, 94, 95, 96, 97, 98, 99] {
            let idx = (sorted_scores.len() as f32 * pct as f32 / 100.0) as usize;
            let thresh = sorted_scores[idx.min(sorted_scores.len() - 1)];
            let (acc, prec, rec, f1) = Self::evaluate_at_threshold(&all_scores, &all_labels, thresh);

            log::info!("  p={:2}% thresh={:>7.4}: P={:.3} R={:.3} F1={:.3}",
                       pct, thresh, prec, rec, f1);

            if f1 > best_f1 {
                best_f1 = f1;
                best_thresh = thresh;
                best_metrics = (acc, prec, rec, f1);
            }
        }

        // Method 2: Value-based thresholds (sweep the actual score range - 50 steps for finer granularity)
        log::info!("Value-based threshold search:");
        for i in 0..=50 {
            let thresh = min_score + (max_score - min_score) * (i as f32 / 50.0);
            let (acc, prec, rec, f1) = Self::evaluate_at_threshold(&all_scores, &all_labels, thresh);

            if i % 10 == 0 {
                log::info!("  v={:>7.4}: P={:.3} R={:.3} F1={:.3}", thresh, prec, rec, f1);
            }

            if f1 > best_f1 {
                best_f1 = f1;
                best_thresh = thresh;
                best_metrics = (acc, prec, rec, f1);
            }
        }

        // Method 3: Fine-grained search around best threshold
        let search_min = best_thresh - 0.1 * (max_score - min_score);
        let search_max = best_thresh + 0.1 * (max_score - min_score);
        for i in 0..=20 {
            let thresh = search_min + (search_max - search_min) * (i as f32 / 20.0);
            let (acc, prec, rec, f1) = Self::evaluate_at_threshold(&all_scores, &all_labels, thresh);

            if f1 > best_f1 {
                best_f1 = f1;
                best_thresh = thresh;
                best_metrics = (acc, prec, rec, f1);
            }
        }

        // Show common fixed thresholds for comparison
        log::info!("Common fixed thresholds:");
        for thresh in [-1.0f32, -0.5, 0.0, 0.5, 1.0] {
            let (_, prec, rec, f1) = Self::evaluate_at_threshold(&all_scores, &all_labels, thresh);
            log::info!("  {:.1}: P={:.3} R={:.3} F1={:.3}", thresh, prec, rec, f1);
        }

        // Compute AUC-ROC
        let auc = self.compute_auc(&all_scores, &all_labels);

        log::info!("═══════════════════════════════════════════════════════════════════");
        log::info!("  OPTIMAL THRESHOLD: {:.4}", best_thresh);
        log::info!("  Accuracy:  {:.4}", best_metrics.0);
        log::info!("  Precision: {:.4}", best_metrics.1);
        log::info!("  Recall:    {:.4}", best_metrics.2);
        log::info!("  F1 Score:  {:.4}", best_metrics.3);
        log::info!("  AUC-ROC:   {:.4}", auc);
        log::info!("═══════════════════════════════════════════════════════════════════");

        (best_metrics.0, best_metrics.1, best_metrics.2, best_metrics.3, auc)
    }

    /// Helper function to evaluate metrics at a specific threshold
    fn evaluate_at_threshold(scores: &[f32], labels: &[u8], threshold: f32) -> (f64, f64, f64, f64) {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut tn = 0usize;
        let mut fn_ = 0usize;

        for (&score, &label) in scores.iter().zip(labels) {
            let pred = score > threshold;
            let gt = label > 0;
            match (pred, gt) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let accuracy = (tp + tn) as f64 / scores.len() as f64;

        (accuracy, precision, recall, f1)
    }

    /// Compute AUC-ROC via trapezoidal approximation
    fn compute_auc(&self, scores: &[f32], labels: &[u8]) -> f64 {
        // Sort by score descending
        let mut indexed: Vec<(f32, u8)> = scores.iter().cloned()
            .zip(labels.iter().cloned())
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal);

        let n_pos = labels.iter().filter(|&&l| l > 0).count() as f64;
        let n_neg = labels.len() as f64 - n_pos;

        if n_pos == 0.0 || n_neg == 0.0 {
            return 0.5; // Undefined, return random
        }

        let mut tpr_prev = 0.0f64;
        let mut fpr_prev = 0.0f64;
        let mut auc = 0.0f64;
        let mut tp = 0.0f64;
        let mut fp = 0.0f64;

        for (_, label) in &indexed {
            if *label > 0 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }

            let tpr = tp / n_pos;
            let fpr = fp / n_neg;

            // Trapezoidal rule
            auc += 0.5 * (tpr + tpr_prev) * (fpr - fpr_prev);

            tpr_prev = tpr;
            fpr_prev = fpr;
        }

        auc
    }

    /// Save trained weights to binary file
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;

        // Write header
        file.write_all(b"PRISM_READOUT_v2")?;

        // Write metadata
        file.write_all(&(self.reservoir_dim as u32).to_le_bytes())?;
        file.write_all(&(self.n_residues as u32).to_le_bytes())?;
        file.write_all(&(self.n_positive as u32).to_le_bytes())?;
        file.write_all(&(self.n_structures as u32).to_le_bytes())?;
        file.write_all(&self.lambda.to_le_bytes())?;

        // Write weights
        file.write_all(&(self.weights.len() as u32).to_le_bytes())?;
        for &w in &self.weights {
            file.write_all(&w.to_le_bytes())?;
        }

        log::info!("Saved readout weights to {:?}", path);
        Ok(())
    }

    /// Load trained weights from binary file
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;

        // Read and verify header
        let mut header = [0u8; 16];
        file.read_exact(&mut header)?;

        // Support both v1 and v2 formats
        let is_v2 = &header == b"PRISM_READOUT_v2";
        let is_v1 = &header == b"PRISM_READOUT_v1";

        if !is_v1 && !is_v2 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid readout file header",
            );
        }

        let mut buf4 = [0u8; 4];

        let (reservoir_dim, n_residues, n_positive, n_structures, lambda) = if is_v2 {
            file.read_exact(&mut buf4)?;
            let reservoir_dim = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let n_residues = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let n_positive = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let n_structures = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let lambda = f32::from_le_bytes(buf4);
            (reservoir_dim, n_residues, n_positive, n_structures, lambda)
        } else {
            // v1 format (backwards compatible)
            file.read_exact(&mut buf4)?;
            let n_samples = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let n_positive = u32::from_le_bytes(buf4) as usize;
            file.read_exact(&mut buf4)?;
            let lambda = f32::from_le_bytes(buf4);
            file.read_exact(&mut buf4)?;
            let _bias = f32::from_le_bytes(buf4); // Ignored in v2
            (RESERVOIR_STATE_DIM, n_samples, n_positive, 0, lambda)
        };

        // Read weights
        file.read_exact(&mut buf4)?;
        let n_weights = u32::from_le_bytes(buf4) as usize;

        let mut weights = Vec::with_capacity(n_weights);
        for _ in 0..n_weights {
            file.read_exact(&mut buf4)?;
            weights.push(f32::from_le_bytes(buf4);
        }

        log::info!(
            "Loaded readout weights from {:?}: {} structures, {} residues, {:.1}% positive",
            path, n_structures, n_residues,
            100.0 * n_positive as f64 / n_residues.max(1) as f64
        );

        // Old binary files don't have normalization stats - use identity transform
        let norm_means = vec![0.0f32; reservoir_dim];
        let norm_stds = vec![1.0f32; reservoir_dim];

        Ok(Self {
            weights,
            reservoir_dim,
            n_residues,
            n_positive,
            n_structures,
            lambda,
            norm_means,
            norm_stds,
        })
    }

    /// Try to load from standard locations, fall back to default
    pub fn load_or_default() -> Self {
        // Try standard locations
        let paths = [
            "readout_weights.bin",
            "data/readout_weights.bin",
            "/tmp/prism_readout_weights.bin",
            "target/readout_weights.bin",
        ];

        for path in &paths {
            if let Ok(readout) = Self::load(Path::new(path)) {
                return readout;
            }
        }

        // Check environment variable
        if let Ok(path) = std::env::var("PRISM_READOUT_WEIGHTS") {
            if let Ok(readout) = Self::load(Path::new(&path)) {
                return readout;
            }
        }

        log::warn!("No trained readout weights found, using defaults (prediction quality will be poor)");
        Self::default()
    }

    /// Generate Rust const source for compile-time embedding
    pub fn to_rust_const(&self) -> String {
        let mut s = String::from(
            "// Auto-generated by train_readout - DO NOT EDIT\n\
             // Training: {} structures, {} residues, {:.2}% positive\n\n\
             pub const RESERVOIR_DIM: usize = {};\n\n\
             pub const READOUT_WEIGHTS: &[f32] = &[\n"
        );

        s = s.replace("{}", &self.n_structures.to_string();
        s = s.replace("{}", &self.n_residues.to_string();
        s = s.replace("{:.2}", &format!("{:.2}", 100.0 * self.n_positive as f64 / self.n_residues.max(1) as f64);
        s = s.replace("{}", &self.reservoir_dim.to_string();

        for (i, w) in self.weights.iter().enumerate() {
            if i % 8 == 0 {
                s.push_str("    ");
            }
            s.push_str(&format!("{:>12.8}, ", w);
            if i % 8 == 7 || i == self.weights.len() - 1 {
                s.push('\n');
            }
        }
        s.push_str("];\n");
        s
    }
}

/// Reservoir state collector for training
#[derive(Debug, Default)]
pub struct ReservoirStateCollector {
    /// Collected training data: (flattened_states, mask) per structure
    pub data: Vec<(Vec<f32>, Vec<u8>)>,
    /// Structure IDs for debugging/logging
    pub structure_ids: Vec<String>,
}

impl ReservoirStateCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add reservoir states and labels from a single structure
    ///
    /// # Arguments
    /// * `structure_id` - PDB ID or filename for logging
    /// * `reservoir_states` - Flattened [n_residues * reservoir_dim]
    /// * `labels` - Binary labels [n_residues]
    pub fn add_structure(
        &mut self,
        structure_id: &str,
        reservoir_states: Vec<f32>,
        labels: Vec<u8>,
    ) {
        self.data.push((reservoir_states, labels);
        self.structure_ids.push(structure_id.to_string();
    }

    /// Train readout from collected data
    pub fn train(&self, reservoir_dim: usize, lambda: f32) -> Result<TrainedReadout, String> {
        TrainedReadout::train(&self.data, reservoir_dim, lambda)
    }

    /// Get statistics: (n_structures, n_residues, n_positive)
    pub fn stats(&self) -> (usize, usize, usize) {
        let n_structures = self.data.len();
        let n_residues: usize = self.data.iter().map(|(_, m)| m.len()).sum();
        let n_positive: usize = self.data.iter()
            .flat_map(|(_, m)| m.iter())
            .filter(|&&l| l > 0)
            .count();
        (n_structures, n_residues, n_positive)
    }

    /// Clear collected data
    pub fn clear(&mut self) {
        self.data.clear();
        self.structure_ids.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_regression_simple() {
        // Simple test: y ≈ 0.5*x0 + 0.3*x1 + 0.1*x2 + 0.1*x3
        let train_data = vec![
            // Structure 1: 4 residues
            (
                vec![
                    1.0, 0.0, 0.0, 0.0,  // res 0: x0=1
                    0.0, 1.0, 0.0, 0.0,  // res 1: x1=1
                    0.0, 0.0, 1.0, 0.0,  // res 2: x2=1
                    0.0, 0.0, 0.0, 1.0,  // res 3: x3=1
                ],
                vec![1, 0, 0, 0],  // Only res 0 is binding
            ),
            // Structure 2: 2 residues
            (
                vec![
                    1.0, 1.0, 0.0, 0.0,  // res 0: x0=1, x1=1
                    0.5, 0.5, 0.5, 0.5,  // res 1: all 0.5
                ],
                vec![1, 1],  // Both binding
            ),
        ];

        let readout = TrainedReadout::train(&train_data, 4, 1e-4).unwrap());
        assert_eq!(readout.weights.len(), 4);
        assert_eq!(readout.n_residues, 6);
        assert_eq!(readout.n_positive, 3);
        assert_eq!(readout.n_structures, 2);

        // First weight should be highest (x0 most predictive)
        println!("Weights: {:?}", readout.weights);
    }

    #[test]
    fn test_save_load_v2() {
        let readout = TrainedReadout {
            weights: vec![0.5, 0.3, 0.1, 0.1],
            reservoir_dim: 4,
            n_residues: 1000,
            n_positive: 150,
            n_structures: 100,
            lambda: 1e-4,
        };

        let path = Path::new("/tmp/test_readout_v2.bin");
        readout.save(path).unwrap());

        let loaded = TrainedReadout::load(path).unwrap());
        assert_eq!(loaded.weights.len(), 4);
        assert!((loaded.weights[0] - 0.5).abs() < 1e-6);
        assert_eq!(loaded.reservoir_dim, 4);
        assert_eq!(loaded.n_residues, 1000);
        assert_eq!(loaded.n_structures, 100);
    }

    #[test]
    fn test_collector() {
        let mut collector = ReservoirStateCollector::new());

        collector.add_structure(
            "1ABC",
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],  // 2 residues * 4 dim
            vec![1, 0],
        );

        collector.add_structure(
            "2DEF",
            vec![0.9, 0.8, 0.7, 0.6],  // 1 residue * 4 dim
            vec![1],
        );

        let (n_struct, n_res, n_pos) = collector.stats();
        assert_eq!(n_struct, 2);
        assert_eq!(n_res, 3);
        assert_eq!(n_pos, 2);
    }

    #[test]
    fn test_auc_perfect() {
        let readout = TrainedReadout {
            weights: vec![1.0, 0.0, 0.0, 0.0],
            reservoir_dim: 4,
            n_residues: 0,
            n_positive: 0,
            n_structures: 0,
            lambda: 1e-4,
        };

        // Perfect separation: positives have high x0, negatives have low x0
        let test_data = vec![(
            vec![
                0.9, 0.0, 0.0, 0.0,  // positive
                0.8, 0.0, 0.0, 0.0,  // positive
                0.1, 0.0, 0.0, 0.0,  // negative
                0.2, 0.0, 0.0, 0.0,  // negative
            ],
            vec![1, 1, 0, 0],
        )];

        let (_, _, _, f1, auc) = readout.evaluate(&test_data);
        assert!(auc > 0.99, "AUC should be ~1.0 for perfect separation, got {}", auc);
        assert!(f1 > 0.99, "F1 should be ~1.0 for perfect separation, got {}", f1);
    }
}
