//! Training Pipeline for Hybrid TDA + Base Features
//!
//! This module provides training infrastructure for the integrated
//! 80-dimensional feature space (32 base + 48 TDA).
//!
//! ## Components
//!
//! - `FeaturePipeline`: Extracts and normalizes features from PDB structures
//! - `Normalizer`: Computes and applies z-score normalization
//! - `ReadoutTrainer`: Ridge regression for binding site prediction
//! - `CryptoBenchRunner`: Benchmarking harness for CryptoBench dataset

pub mod feature_pipeline;
pub mod normalization;
pub mod readout;
pub mod benchmark;
pub mod reorthogonalization;
pub mod simulated_annealing;
pub mod two_pass;

// Re-exports
pub use feature_pipeline::{FeaturePipeline, FeatureConfig, StructureFeatures};
pub use normalization::{Normalizer, WelfordStats, NormStats};
pub use readout::{ReadoutTrainer, TrainedReadout, RidgeConfig};
pub use benchmark::{CryptoBenchRunner, BenchmarkConfig, BenchmarkResult};
pub use reorthogonalization::WhiteningParams;
pub use simulated_annealing::{SAClassifier, SAConfig};
pub use two_pass::{TwoPassBenchmark, TwoPassConfig, ComprehensiveReport};

use crate::batch_tda::{BASE_FEATURES, TDA_FEATURE_COUNT, TOTAL_COMBINED_FEATURES};

/// Feature dimension constants (re-exported for convenience)
pub mod dims {
    pub use crate::batch_tda::{
        BASE_FEATURES,
        TDA_FEATURE_COUNT,
        TOTAL_COMBINED_FEATURES,
    };
}

/// Training-related error types
#[derive(Debug, Clone)]
pub enum TrainingError {
    /// Invalid input data
    InvalidInput(String),
    /// Feature extraction failed
    FeatureExtraction(String),
    /// Training failed
    TrainingFailed(String),
    /// IO error
    Io(String),
    /// GPU error
    Gpu(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            TrainingError::FeatureExtraction(s) => write!(f, "Feature extraction: {}", s),
            TrainingError::TrainingFailed(s) => write!(f, "Training failed: {}", s),
            TrainingError::Io(s) => write!(f, "IO error: {}", s),
            TrainingError::Gpu(s) => write!(f, "GPU error: {}", s),
        }
    }
}

impl std::error::Error for TrainingError {}

impl From<prism_core::PrismError> for TrainingError {
    fn from(e: prism_core::PrismError) -> Self {
        TrainingError::Gpu(e.to_string())
    }
}

impl From<std::io::Error> for TrainingError {
    fn from(e: std::io::Error) -> Self {
        TrainingError::Io(e.to_string())
    }
}

/// Training sample for binding site prediction
#[derive(Clone, Debug)]
pub struct TrainingSample {
    /// Structure identifier
    pub structure_id: String,
    /// Residue index
    pub residue_idx: usize,
    /// Residue name (e.g., "ALA", "GLY")
    pub residue_name: String,
    /// Chain ID
    pub chain_id: String,
    /// Sequence position
    pub seq_pos: i32,
    /// Feature vector [TOTAL_COMBINED_FEATURES]
    pub features: Vec<f32>,
    /// Ground truth label (1 = binding site, 0 = not)
    pub label: f32,
    /// Weight for class balancing
    pub weight: f32,
}

/// Batch of training samples
#[derive(Clone, Debug)]
pub struct TrainingBatch {
    /// Number of samples
    pub n_samples: usize,
    /// Feature matrix [n_samples × TOTAL_COMBINED_FEATURES]
    pub features: Vec<f32>,
    /// Labels [n_samples]
    pub labels: Vec<f32>,
    /// Weights [n_samples]
    pub weights: Vec<f32>,
    /// Structure IDs for each sample
    pub structure_ids: Vec<String>,
}

impl TrainingBatch {
    /// Create an empty batch
    pub fn new() -> Self {
        Self {
            n_samples: 0,
            features: Vec::new(),
            labels: Vec::new(),
            weights: Vec::new(),
            structure_ids: Vec::new(),
        }
    }

    /// Add a sample to the batch
    pub fn add(&mut self, sample: TrainingSample) {
        self.features.extend_from_slice(&sample.features);
        self.labels.push(sample.label);
        self.weights.push(sample.weight);
        self.structure_ids.push(sample.structure_id);
        self.n_samples += 1;
    }

    /// Merge another batch into this one
    pub fn extend(&mut self, other: TrainingBatch) {
        self.features.extend(other.features);
        self.labels.extend(other.labels);
        self.weights.extend(other.weights);
        self.structure_ids.extend(other.structure_ids);
        self.n_samples += other.n_samples;
    }

    /// Get feature matrix as 2D array reference
    pub fn feature_matrix(&self) -> Vec<&[f32]> {
        (0..self.n_samples)
            .map(|i| {
                let start = i * TOTAL_COMBINED_FEATURES;
                &self.features[start..start + TOTAL_COMBINED_FEATURES]
            })
            .collect()
    }
}

impl Default for TrainingBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluation metrics for binding site prediction
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Metrics {
    /// Area under ROC curve
    pub auc_roc: f64,
    /// Precision at various thresholds
    pub precision: f64,
    /// Recall at various thresholds
    pub recall: f64,
    /// F1 score
    pub f1: f64,
    /// Accuracy
    pub accuracy: f64,
    /// Number of true positives
    pub tp: usize,
    /// Number of false positives
    pub fp: usize,
    /// Number of true negatives
    pub tn: usize,
    /// Number of false negatives
    pub fn_: usize,
}

impl Metrics {
    /// Compute metrics from predictions and labels
    pub fn from_predictions(predictions: &[f32], labels: &[f32], threshold: f32) -> Self {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut tn = 0usize;
        let mut fn_ = 0usize;

        for (pred, &label) in predictions.iter().zip(labels.iter()) {
            let predicted = if *pred >= threshold { 1.0 } else { 0.0 };
            if predicted == 1.0 && label == 1.0 {
                tp += 1;
            } else if predicted == 1.0 && label == 0.0 {
                fp += 1;
            } else if predicted == 0.0 && label == 0.0 {
                tn += 1;
            } else {
                fn_ += 1;
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let accuracy = (tp + tn) as f64 / predictions.len().max(1) as f64;

        // Compute AUC-ROC
        let auc_roc = Self::compute_auc_roc(predictions, labels);

        Self {
            auc_roc,
            precision,
            recall,
            f1,
            accuracy,
            tp,
            fp,
            tn,
            fn_,
        }
    }

    /// Compute AUC-ROC using trapezoidal rule
    fn compute_auc_roc(predictions: &[f32], labels: &[f32]) -> f64 {
        if predictions.is_empty() || labels.is_empty() {
            return 0.5;
        }

        // Create (score, label) pairs and sort by score descending
        let mut pairs: Vec<(f32, f32)> = predictions.iter()
            .zip(labels.iter())
            .map(|(&p, &l)| (p, l))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal);

        let n_pos = labels.iter().filter(|&&l| l > 0.5).count() as f64;
        let n_neg = labels.iter().filter(|&&l| l < 0.5).count() as f64;

        if n_pos == 0.0 || n_neg == 0.0 {
            return 0.5;
        }

        let mut auc = 0.0;
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut prev_tpr = 0.0;
        let mut prev_fpr = 0.0;

        for (_, label) in pairs {
            if label > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }

            let tpr = tp / n_pos;
            let fpr = fp / n_neg;

            // Trapezoidal rule
            auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

            prev_tpr = tpr;
            prev_fpr = fpr;
        }

        auc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_batch() {
        let mut batch = TrainingBatch::new());

        let sample = TrainingSample {
            structure_id: "test".to_string(),
            residue_idx: 0,
            residue_name: "ALA".to_string(),
            chain_id: "A".to_string(),
            seq_pos: 1,
            features: vec![0.0; TOTAL_COMBINED_FEATURES],
            label: 1.0,
            weight: 1.0,
        };

        batch.add(sample);

        assert_eq!(batch.n_samples, 1);
        assert_eq!(batch.features.len(), TOTAL_COMBINED_FEATURES);
    }

    #[test]
    fn test_metrics() {
        let predictions = vec![0.9, 0.8, 0.3, 0.2];
        let labels = vec![1.0, 1.0, 0.0, 0.0];

        let metrics = Metrics::from_predictions(&predictions, &labels, 0.5);

        assert_eq!(metrics.tp, 2);
        assert_eq!(metrics.tn, 2);
        assert_eq!(metrics.fp, 0);
        assert_eq!(metrics.fn_, 0);
        assert!((metrics.precision - 1.0).abs() < 1e-6);
        assert!((metrics.recall - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_auc_roc() {
        // Perfect classifier
        let predictions = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![1.0, 1.0, 0.0, 0.0];

        let auc = Metrics::compute_auc_roc(&predictions, &labels);
        assert!((auc - 1.0).abs() < 1e-6);

        // Random classifier
        let predictions = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![1.0, 0.0, 1.0, 0.0];

        let auc = Metrics::compute_auc_roc(&predictions, &labels);
        assert!((auc - 0.5).abs() < 0.1);
    }
}
