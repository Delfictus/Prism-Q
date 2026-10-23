//! CryptoBench Benchmark Runner
//!
//! Benchmarking harness for evaluating binding site prediction on the
//! CryptoBench dataset (885 train / 222 test structures).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use serde::{Serialize, Deserialize};

use super::{
    TrainingError, TrainingBatch, Metrics,
    FeaturePipeline, FeatureConfig, StructureFeatures,
    Normalizer, NormStats,
    ReadoutTrainer, RidgeConfig, TrainedReadout,
};
use crate::batch_tda::TOTAL_COMBINED_FEATURES;

/// Configuration for benchmark run
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Path to PDB files directory
    pub pdb_dir: PathBuf,
    /// Path to dataset JSON (structure IDs and binding residues)
    pub dataset_path: PathBuf,
    /// Path to folds JSON (train/test split)
    pub folds_path: Option<PathBuf>,
    /// Feature extraction configuration
    pub feature_config: FeatureConfig,
    /// Ridge regression configuration
    pub ridge_config: RidgeConfig,
    /// Apply normalization
    pub normalize: bool,
    /// Positive class weight
    pub positive_weight: f32,
    /// Batch size for feature extraction
    pub batch_size: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            pdb_dir: PathBuf::from("pdb-files"),
            dataset_path: PathBuf::from("dataset.json"),
            folds_path: None,
            feature_config: FeatureConfig::default(),
            ridge_config: RidgeConfig::default(),
            normalize: true,
            positive_weight: 10.0,
            batch_size: 8,
        }
    }
}

/// Dataset entry for a structure
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetEntry {
    /// Structure ID (PDB ID)
    pub structure_id: String,
    /// Binding residues (chain:resnum format, e.g., "A:123")
    #[serde(default)]
    pub binding_residues: Vec<String>,
    /// Alternative: list of residue indices
    #[serde(default)]
    pub binding_indices: Vec<usize>,
}

/// Folds definition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Folds {
    /// Training structure IDs
    pub train: Vec<String>,
    /// Test structure IDs
    pub test: Vec<String>,
}

/// Result of a benchmark run
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Test set metrics
    pub test_metrics: Metrics,
    /// Train set metrics
    pub train_metrics: Metrics,
    /// Per-structure results
    pub per_structure: Vec<StructureResult>,
    /// Trained model
    #[serde(skip)]
    pub model: Option<TrainedReadout>,
    /// Normalization statistics
    #[serde(skip)]
    pub norm_stats: Option<NormStats>,
    /// Total runtime (seconds)
    pub runtime_seconds: f64,
    /// Number of training samples
    pub n_train_samples: usize,
    /// Number of test samples
    pub n_test_samples: usize,
}

/// Per-structure result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StructureResult {
    /// Structure ID
    pub structure_id: String,
    /// Number of residues
    pub n_residues: usize,
    /// Number of binding residues
    pub n_binding: usize,
    /// Predictions (per-residue scores)
    pub predictions: Vec<f32>,
    /// Labels (per-residue)
    pub labels: Vec<f32>,
    /// AUC for this structure
    pub auc: f64,
}

/// CryptoBench benchmark runner
pub struct CryptoBenchRunner {
    /// Configuration
    config: BenchmarkConfig,
    /// Feature pipeline
    pipeline: FeaturePipeline,
}

impl CryptoBenchRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchmarkConfig) -> Self {
        let pipeline = FeaturePipeline::new(config.feature_config.clone();
        Self { config, pipeline }
    }

    /// Run the full benchmark
    pub fn run(&self) -> Result<BenchmarkResult, TrainingError> {
        let start = std::time::Instant::now();

        // Load dataset
        let dataset = self.load_dataset()?;
        log::info!("Loaded {} structures from dataset", dataset.len();

        // Load or create folds
        let folds = self.load_or_create_folds(&dataset)?;
        log::info!("Train: {} structures, Test: {} structures",
                  folds.train.len(), folds.test.len();

        // Extract features and build training batch
        let (train_batch, train_features) = self.build_batch(&folds.train, &dataset)?;
        log::info!("Train batch: {} samples", train_batch.n_samples);

        // Compute normalization statistics from training data
        let norm_stats = if self.config.normalize {
            let mut normalizer = Normalizer::new(TOTAL_COMBINED_FEATURES);
            normalizer.fit(&train_batch.features);
            Some(normalizer.finalize().clone())
        } else {
            None
        };

        // Train model
        let trainer = ReadoutTrainer::new(self.config.ridge_config.clone();
        let model = trainer.train(&train_batch, norm_stats.clone())?;
        log::info!("Training AUC: {:.4}", model.train_auc);

        // Evaluate on training set
        let train_predictions = model.predict_proba(&train_batch.features);
        let train_metrics = Metrics::from_predictions(&train_predictions, &train_batch.labels, 0.5);

        // Evaluate on test set
        let (test_batch, test_features) = self.build_batch(&folds.test, &dataset)?;
        log::info!("Test batch: {} samples", test_batch.n_samples);

        let test_predictions = model.predict_proba(&test_batch.features);
        let test_metrics = Metrics::from_predictions(&test_predictions, &test_batch.labels, 0.5);

        // Per-structure results
        let mut per_structure = Vec::new();

        let mut pred_idx = 0;
        for (features, labels) in test_features.iter().zip(
            self.get_labels_per_structure(&folds.test, &dataset)?.iter()
        ) {
            let n_residues = features.n_residues;
            let n_binding = labels.iter().filter(|&&l| l > 0.5).count();

            let structure_preds = &test_predictions[pred_idx..pred_idx + n_residues];
            let structure_labels = labels;

            let auc = Metrics::compute_auc_roc(structure_preds, structure_labels);

            per_structure.push(StructureResult {
                structure_id: features.structure_id.clone(),
                n_residues,
                n_binding,
                predictions: structure_preds.to_vec(),
                labels: structure_labels.clone(),
                auc,
            });

            pred_idx += n_residues;
        }

        let runtime_seconds = start.elapsed().as_secs_f64();

        Ok(BenchmarkResult {
            test_metrics,
            train_metrics,
            per_structure,
            model: Some(model),
            norm_stats,
            runtime_seconds,
            n_train_samples: train_batch.n_samples,
            n_test_samples: test_batch.n_samples,
        })
    }

    /// Load dataset from JSON
    fn load_dataset(&self) -> Result<HashMap<String, DatasetEntry>, TrainingError> {
        let content = std::fs::read_to_string(&self.config.dataset_path)
            .map_err(|e| TrainingError::Io(format!("Read dataset: {}", e)))?;

        let entries: Vec<DatasetEntry> = serde_json::from_str(&content)
            .map_err(|e| TrainingError::InvalidInput(format!("Parse dataset: {}", e)))?;

        let mut dataset = HashMap::new());
        for entry in entries {
            dataset.insert(entry.structure_id.clone(), entry);
        }

        Ok(dataset)
    }

    /// Load folds or create default 80/20 split
    fn load_or_create_folds(&self, dataset: &HashMap<String, DatasetEntry>) -> Result<Folds, TrainingError> {
        if let Some(ref folds_path) = self.config.folds_path {
            let content = std::fs::read_to_string(folds_path)
                .map_err(|e| TrainingError::Io(format!("Read folds: {}", e)))?;

            serde_json::from_str(&content)
                .map_err(|e| TrainingError::InvalidInput(format!("Parse folds: {}", e)))
        } else {
            // Create default 80/20 split
            let mut structure_ids: Vec<_> = dataset.keys().cloned().collect();
            structure_ids.sort(); // Deterministic order

            let train_count = (structure_ids.len() * 4) / 5;
            let (train, test) = structure_ids.split_at(train_count);

            Ok(Folds {
                train: train.to_vec(),
                test: test.to_vec(),
            })
        }
    }

    /// Build training batch from structure IDs
    fn build_batch(
        &self,
        structure_ids: &[String],
        dataset: &HashMap<String, DatasetEntry>,
    ) -> Result<(TrainingBatch, Vec<StructureFeatures>), TrainingError> {
        let mut batch = TrainingBatch::new());
        let mut all_features = Vec::new();

        for structure_id in structure_ids {
            let pdb_path = self.config.pdb_dir.join(format!("{}.pdb", structure_id);

            if !pdb_path.exists() {
                log::warn!("PDB file not found: {}", pdb_path.display();
                continue;
            }

            // Extract features
            let features = match self.pipeline.extract_from_pdb(&pdb_path) {
                Ok(f) => f,
                Err(e) => {
                    log::warn!("Failed to extract features for {}: {}", structure_id, e);
                    continue;
                }
            };

            // Get labels
            let labels = self.get_labels(structure_id, &features, dataset)?;

            // Convert to training samples
            let samples = features.to_training_samples(&labels, self.config.positive_weight);

            for sample in samples {
                batch.add(sample);
            }

            all_features.push(features);
        }

        Ok((batch, all_features))
    }

    /// Get labels for a structure
    fn get_labels(
        &self,
        structure_id: &str,
        features: &StructureFeatures,
        dataset: &HashMap<String, DatasetEntry>,
    ) -> Result<Vec<f32>, TrainingError> {
        let entry = dataset.get(structure_id)
            .ok_or_else(|| TrainingError::InvalidInput(format!("Structure {} not in dataset", structure_id)))?;

        let mut labels = vec![0.0f32; features.n_residues];

        // Check binding_residues (chain:resnum format)
        for binding_res in &entry.binding_residues {
            // Parse chain:resnum
            let parts: Vec<&str> = binding_res.split(':').collect();
            if parts.len() != 2 {
                continue;
            }
            let chain = parts[0];
            let resnum: i32 = match parts[1].parse() {
                Ok(n) => n,
                Err(_) => continue,
            };

            // Find matching residue
            for (i, (c, &r)) in features.chain_ids.iter()
                .zip(features.seq_positions.iter())
                .enumerate()
            {
                if c == chain && r == resnum {
                    labels[i] = 1.0;
                }
            }
        }

        // Check binding_indices (direct index)
        for &idx in &entry.binding_indices {
            if idx < labels.len() {
                labels[idx] = 1.0;
            }
        }

        Ok(labels)
    }

    /// Get labels per structure for test set
    fn get_labels_per_structure(
        &self,
        structure_ids: &[String],
        dataset: &HashMap<String, DatasetEntry>,
    ) -> Result<Vec<Vec<f32>>, TrainingError> {
        let mut all_labels = Vec::new();

        for structure_id in structure_ids {
            let pdb_path = self.config.pdb_dir.join(format!("{}.pdb", structure_id);

            if !pdb_path.exists() {
                continue;
            }

            let features = match self.pipeline.extract_from_pdb(&pdb_path) {
                Ok(f) => f,
                Err(_) => continue,
            };

            let labels = self.get_labels(structure_id, &features, dataset)?;
            all_labels.push(labels);
        }

        Ok(all_labels)
    }

    /// Save benchmark results to JSON
    pub fn save_results(result: &BenchmarkResult, path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(result)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
}

impl Metrics {
    // compute_auc_roc is already defined in mod.rs at line 240
    // We use that implementation via Metrics::compute_auc_roc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.batch_size, 8);
        assert!(config.normalize);
    }

    #[test]
    fn test_metrics_auc() {
        // Perfect separation
        let predictions = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![1.0, 1.0, 0.0, 0.0];

        let auc = Metrics::compute_auc_roc(&predictions, &labels);
        assert!((auc - 1.0).abs() < 1e-6);
    }
}
