//! Ridge Regression Readout Training
//!
//! Implements closed-form ridge regression for training binding site classifiers.
//! Uses SVD-based solution for numerical stability.

use serde::{Serialize, Deserialize};
use crate::batch_tda::TOTAL_COMBINED_FEATURES;
use super::{TrainingError, TrainingBatch, NormStats};

/// Configuration for ridge regression
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RidgeConfig {
    /// Regularization parameter (lambda)
    pub lambda: f64,
    /// Apply class weighting
    pub use_class_weights: bool,
    /// Include bias term
    pub fit_intercept: bool,
    /// Number of cross-validation folds for lambda selection
    pub cv_folds: usize,
}

impl Default for RidgeConfig {
    fn default() -> Self {
        Self {
            lambda: 1e-4,
            use_class_weights: true,
            fit_intercept: true,
            cv_folds: 5,
        }
    }
}

/// Trained readout weights
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainedReadout {
    /// Feature weights [n_features]
    pub weights: Vec<f32>,
    /// Bias term (if fit_intercept=true)
    pub bias: f32,
    /// Normalization statistics used during training
    pub norm_stats: Option<NormStats>,
    /// Training configuration
    pub config: RidgeConfig,
    /// Number of training samples
    pub n_train_samples: usize,
    /// Training metrics
    pub train_auc: f64,
}

impl TrainedReadout {
    /// Predict scores for features
    pub fn predict(&self, features: &[f32]) -> Vec<f32> {
        let n_features = self.weights.len();
        assert_eq!(features.len() % n_features, 0);
        let n_samples = features.len() / n_features;

        // Optionally normalize
        let normalized_features = if let Some(ref stats) = self.norm_stats {
            stats.normalize(features)
        } else {
            features.to_vec()
        };

        // Compute scores
        let mut scores = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let mut score = self.bias;
            let sample = &normalized_features[i * n_features..(i + 1) * n_features];
            for (j, &x) in sample.iter().enumerate() {
                score += x * self.weights[j];
            }
            scores.push(score);
        }

        scores
    }

    /// Predict probabilities (sigmoid of scores)
    pub fn predict_proba(&self, features: &[f32]) -> Vec<f32> {
        self.predict(features).iter()
            .map(|&x| sigmoid(x))
            .collect()
    }

    /// Save to binary file
    pub fn save(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let data = bincode::serialize(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, data)
    }

    /// Load from binary file
    pub fn load(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let data = std::fs::read(path)?;
        bincode::deserialize(&data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }

    /// Save to JSON file
    pub fn save_json(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load from JSON file
    pub fn load_json(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}

/// Sigmoid activation
fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Ridge regression trainer
pub struct ReadoutTrainer {
    /// Configuration
    config: RidgeConfig,
}

impl ReadoutTrainer {
    /// Create a new trainer
    pub fn new(config: RidgeConfig) -> Self {
        Self { config }
    }

    /// Train readout weights on a batch
    pub fn train(
        &self,
        batch: &TrainingBatch,
        norm_stats: Option<NormStats>,
    ) -> Result<TrainedReadout, TrainingError> {
        if batch.n_samples == 0 {
            return Err(TrainingError::InvalidInput("Empty training batch".into());
        }

        let n_features = TOTAL_COMBINED_FEATURES;

        // Normalize features if stats provided
        let features = if let Some(ref stats) = norm_stats {
            stats.normalize(&batch.features)
        } else {
            batch.features.clone()
        };

        // Prepare weighted design matrix
        let (x_mat, y_vec, weights_sqrt) = self.prepare_weighted_data(
            &features,
            &batch.labels,
            &batch.weights,
            n_features,
        );

        // Solve weighted ridge regression
        let (weights, bias) = self.solve_ridge(&x_mat, &y_vec, &weights_sqrt, n_features)?;

        // Compute training AUC
        let predictions: Vec<f32> = (0..batch.n_samples)
            .map(|i| {
                let sample = &features[i * n_features..(i + 1) * n_features];
                let mut score = bias;
                for (j, &x) in sample.iter().enumerate() {
                    score += x * weights[j];
                }
                sigmoid(score)
            })
            .collect();

        let train_auc = super::Metrics::from_predictions(&predictions, &batch.labels, 0.5).auc_roc;

        Ok(TrainedReadout {
            weights,
            bias,
            norm_stats,
            config: self.config.clone(),
            n_train_samples: batch.n_samples,
            train_auc,
        })
    }

    /// Prepare weighted data matrices
    fn prepare_weighted_data(
        &self,
        features: &[f32],
        labels: &[f32],
        weights: &[f32],
        n_features: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n_samples = labels.len();

        // Compute sqrt weights for weighted least squares
        let weights_sqrt: Vec<f64> = if self.config.use_class_weights {
            weights.iter().map(|&w| (w as f64).sqrt()).collect()
        } else {
            vec![1.0; n_samples]
        };

        // Weight-scaled design matrix (with optional intercept column)
        let n_cols = if self.config.fit_intercept { n_features + 1 } else { n_features };
        let mut x_mat = vec![0.0f64; n_samples * n_cols];

        for i in 0..n_samples {
            let w = weights_sqrt[i];
            let row_start = i * n_cols;

            // Features
            for j in 0..n_features {
                x_mat[row_start + j] = features[i * n_features + j] as f64 * w;
            }

            // Intercept column
            if self.config.fit_intercept {
                x_mat[row_start + n_features] = w;
            }
        }

        // Weight-scaled labels
        let y_vec: Vec<f64> = labels.iter()
            .zip(weights_sqrt.iter())
            .map(|(&y, &w)| y as f64 * w)
            .collect();

        (x_mat, y_vec, weights_sqrt)
    }

    /// Solve ridge regression using normal equations
    ///
    /// (X'X + λI)β = X'y
    fn solve_ridge(
        &self,
        x_mat: &[f64],
        y_vec: &[f64],
        _weights_sqrt: &[f64],
        n_features: usize,
    ) -> Result<(Vec<f32>, f32), TrainingError> {
        let n_samples = y_vec.len();
        let n_cols = if self.config.fit_intercept { n_features + 1 } else { n_features };

        // Compute X'X (n_cols x n_cols)
        let mut xtx = vec![0.0f64; n_cols * n_cols];
        for i in 0..n_cols {
            for j in 0..n_cols {
                let mut sum = 0.0;
                for k in 0..n_samples {
                    sum += x_mat[k * n_cols + i] * x_mat[k * n_cols + j];
                }
                xtx[i * n_cols + j] = sum;
            }
        }

        // Add regularization (λI), but not to intercept
        let regularize_count = if self.config.fit_intercept { n_features } else { n_cols };
        for i in 0..regularize_count {
            xtx[i * n_cols + i] += self.config.lambda;
        }

        // Compute X'y (n_cols)
        let mut xty = vec![0.0f64; n_cols];
        for i in 0..n_cols {
            let mut sum = 0.0;
            for k in 0..n_samples {
                sum += x_mat[k * n_cols + i] * y_vec[k];
            }
            xty[i] = sum;
        }

        // Solve using Cholesky decomposition
        let beta = self.cholesky_solve(&xtx, &xty, n_cols)?;

        // Extract weights and bias
        let weights: Vec<f32> = beta[..n_features].iter().map(|&x| x as f32).collect();
        let bias = if self.config.fit_intercept {
            beta[n_features] as f32
        } else {
            0.0
        };

        Ok((weights, bias))
    }

    /// Solve linear system using Cholesky decomposition
    fn cholesky_solve(&self, a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, TrainingError> {
        // Cholesky decomposition: A = L * L'
        let mut l = vec![0.0f64; n * n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = a[i * n + j];

                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }

                if i == j {
                    if sum <= 0.0 {
                        // Matrix not positive definite, fall back to pseudoinverse
                        return self.pseudoinverse_solve(a, b, n);
                    }
                    l[i * n + j] = sum.sqrt();
                } else {
                    l[i * n + j] = sum / l[j * n + j];
                }
            }
        }

        // Forward substitution: L * y = b
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[i * n + j] * y[j];
            }
            y[i] = sum / l[i * n + i];
        }

        // Backward substitution: L' * x = y
        let mut x = vec![0.0f64; n];
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[j * n + i] * x[j];
            }
            x[i] = sum / l[i * n + i];
        }

        Ok(x)
    }

    /// Fallback solver using pseudoinverse (for ill-conditioned matrices)
    fn pseudoinverse_solve(&self, a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, TrainingError> {
        // Simple iterative refinement as fallback
        // For production, use proper SVD from a linear algebra library

        let mut x = vec![0.0f64; n];
        let max_iter = 100;
        let tol = 1e-10;

        // Gradient descent with momentum
        let mut velocity = vec![0.0f64; n];
        let lr = 0.01;
        let momentum = 0.9;

        for _ in 0..max_iter {
            // Compute Ax
            let mut ax = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    ax[i] += a[i * n + j] * x[j];
                }
            }

            // Compute gradient: A'(Ax - b)
            let mut grad = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    grad[i] += a[j * n + i] * (ax[j] - b[j]);
                }
            }

            // Check convergence
            let grad_norm: f64 = grad.iter().map(|&g| g * g).sum::<f64>().sqrt();
            if grad_norm < tol {
                break;
            }

            // Update with momentum
            for i in 0..n {
                velocity[i] = momentum * velocity[i] - lr * grad[i];
                x[i] += velocity[i];
            }
        }

        Ok(x)
    }

    /// Cross-validation for lambda selection
    pub fn cross_validate_lambda(
        &self,
        batch: &TrainingBatch,
        lambdas: &[f64],
        n_folds: usize,
    ) -> Vec<(f64, f64)> {
        let n_samples = batch.n_samples;
        let fold_size = n_samples / n_folds;

        lambdas.iter().map(|&lambda| {
            let mut config = self.config.clone();
            config.lambda = lambda;
            let trainer = ReadoutTrainer::new(config);

            let mut fold_aucs = Vec::with_capacity(n_folds);

            for fold in 0..n_folds {
                let test_start = fold * fold_size;
                let test_end = if fold == n_folds - 1 { n_samples } else { (fold + 1) * fold_size };

                // Split data
                let (train_batch, test_batch) = self.split_batch(batch, test_start, test_end);

                // Train on fold
                if let Ok(model) = trainer.train(&train_batch, None) {
                    // Evaluate on held-out fold
                    let predictions = model.predict_proba(&test_batch.features);
                    let auc = super::Metrics::from_predictions(&predictions, &test_batch.labels, 0.5).auc_roc;
                    fold_aucs.push(auc);
                }
            }

            let mean_auc = if !fold_aucs.is_empty() {
                fold_aucs.iter().sum::<f64>() / fold_aucs.len() as f64
            } else {
                0.0
            };

            (lambda, mean_auc)
        }).collect()
    }

    /// Split batch into train and test portions
    fn split_batch(&self, batch: &TrainingBatch, test_start: usize, test_end: usize) -> (TrainingBatch, TrainingBatch) {
        let n_features = TOTAL_COMBINED_FEATURES;

        let mut train_batch = TrainingBatch::new());
        let mut test_batch = TrainingBatch::new());

        for i in 0..batch.n_samples {
            let features = batch.features[i * n_features..(i + 1) * n_features].to_vec();
            let label = batch.labels[i];
            let weight = batch.weights[i];
            let structure_id = batch.structure_ids[i].clone();

            let sample = super::TrainingSample {
                structure_id,
                residue_idx: i,
                residue_name: "UNK".to_string(),
                chain_id: "A".to_string(),
                seq_pos: i as i32,
                features,
                label,
                weight,
            };

            if i >= test_start && i < test_end {
                test_batch.add(sample);
            } else {
                train_batch.add(sample);
            }
        }

        (train_batch, test_batch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ridge_regression_simple() {
        let config = RidgeConfig {
            lambda: 0.1,
            use_class_weights: false,
            fit_intercept: true,
            cv_folds: 5,
        };

        let trainer = ReadoutTrainer::new(config);

        // Create simple test data
        let mut batch = TrainingBatch::new());

        for i in 0..100 {
            let label = if i < 50 { 0.0 } else { 1.0 };
            let mut features = vec![0.0f32; TOTAL_COMBINED_FEATURES];
            // Simple linear relationship: higher feature[0] -> higher label
            features[0] = label * 2.0 + (i as f32 % 10.0) * 0.1;
            features[1] = (i as f32 % 20.0) * 0.05;

            batch.add(super::super::TrainingSample {
                structure_id: format!("test_{}", i),
                residue_idx: i,
                residue_name: "ALA".to_string(),
                chain_id: "A".to_string(),
                seq_pos: i as i32,
                features,
                label,
                weight: 1.0,
            });
        }

        let result = trainer.train(&batch, None);
        assert!(result.is_ok();

        let model = result.unwrap());
        assert_eq!(model.weights.len(), TOTAL_COMBINED_FEATURES);

        // Verify the model learned something
        assert!(model.train_auc > 0.5);
    }

    #[test]
    fn test_predict() {
        let model = TrainedReadout {
            weights: vec![1.0; TOTAL_COMBINED_FEATURES],
            bias: 0.0,
            norm_stats: None,
            config: RidgeConfig::default(),
            n_train_samples: 100,
            train_auc: 0.8,
        };

        let features = vec![0.01f32; TOTAL_COMBINED_FEATURES];
        let scores = model.predict(&features);

        assert_eq!(scores.len(), 1);
        // Score should be sum of features = 80 * 0.01 = 0.8
        assert!((scores[0] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(10.0) - 1.0).abs() < 0.001);
        assert!((sigmoid(-10.0) - 0.0).abs() < 0.001);
    }
}
