//! Feature Extraction Pipeline
//!
//! Extracts integrated 80-dim features from PDB structures.

use std::path::Path;

use crate::batch_tda::{
    TOTAL_COMBINED_FEATURES, HybridTdaConfig,
};
use crate::mega_fused_integrated::{IntegratedCpu, IntegratedConfig};

use super::{TrainingError, TrainingSample, TrainingBatch};

/// Configuration for feature extraction
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FeatureConfig {
    /// Use GPU if available
    pub use_gpu: bool,
    /// TDA configuration
    pub tda_config: HybridTdaConfig,
    /// Apply per-feature normalization
    pub normalize: bool,
    /// Class weight for positive samples (for imbalanced data)
    pub positive_weight: f32,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            use_gpu: true,
            tda_config: HybridTdaConfig::default(),
            normalize: false,
            positive_weight: 10.0, // ~10:1 imbalance typical for binding sites
        }
    }
}

/// Features extracted for a structure
#[derive(Clone, Debug)]
pub struct StructureFeatures {
    /// Structure identifier
    pub structure_id: String,
    /// Number of residues
    pub n_residues: usize,
    /// Feature matrix [n_residues × TOTAL_COMBINED_FEATURES]
    pub features: Vec<f32>,
    /// Cα coordinates
    pub coords: Vec<[f32; 3]>,
    /// Residue names
    pub residue_names: Vec<String>,
    /// Chain IDs
    pub chain_ids: Vec<String>,
    /// Sequence positions
    pub seq_positions: Vec<i32>,
    /// Extraction time (μs)
    pub extraction_time_us: u64,
}

impl StructureFeatures {
    /// Get features for a specific residue
    pub fn get_residue(&self, idx: usize) -> &[f32] {
        let start = idx * TOTAL_COMBINED_FEATURES;
        &self.features[start..start + TOTAL_COMBINED_FEATURES]
    }

    /// Convert to training samples given ground truth labels
    pub fn to_training_samples(
        &self,
        labels: &[f32],
        positive_weight: f32,
    ) -> Vec<TrainingSample> {
        assert_eq!(labels.len(), self.n_residues);

        (0..self.n_residues)
            .map(|i| {
                let label = labels[i];
                let weight = if label > 0.5 { positive_weight } else { 1.0 };

                TrainingSample {
                    structure_id: self.structure_id.clone(),
                    residue_idx: i,
                    residue_name: self.residue_names.get(i)
                        .cloned()
                        .unwrap_or_else(|| "UNK".to_string()),
                    chain_id: self.chain_ids.get(i)
                        .cloned()
                        .unwrap_or_else(|| "A".to_string()),
                    seq_pos: self.seq_positions.get(i).copied().unwrap_or(i as i32),
                    features: self.get_residue(i).to_vec(),
                    label,
                    weight,
                }
            })
            .collect()
    }
}

/// Feature extraction pipeline
pub struct FeaturePipeline {
    /// CPU feature extractor
    cpu_extractor: IntegratedCpu,
    /// Configuration
    config: FeatureConfig,
}

impl FeaturePipeline {
    /// Create a new feature pipeline
    pub fn new(config: FeatureConfig) -> Self {
        let integrated_config = IntegratedConfig {
            tda: config.tda_config.clone(),
            ..Default::default()
        };

        let cpu_extractor = IntegratedCpu::new().with_config(integrated_config);

        Self {
            cpu_extractor,
            config,
        }
    }

    /// Extract features from Cα coordinates
    pub fn extract_from_coords(
        &self,
        structure_id: &str,
        coords: &[[f32; 3]],
    ) -> Result<StructureFeatures, TrainingError> {
        let start = std::time::Instant::now();

        let output = self.cpu_extractor.extract(coords)?;

        let extraction_time_us = start.elapsed().as_micros() as u64;

        Ok(StructureFeatures {
            structure_id: structure_id.to_string(),
            n_residues: output.n_residues,
            features: output.features,
            coords: coords.to_vec(),
            residue_names: vec!["UNK".to_string(); coords.len()],
            chain_ids: vec!["A".to_string(); coords.len()],
            seq_positions: (0..coords.len() as i32).collect(),
            extraction_time_us,
        })
    }

    /// Extract features from a PDB file
    pub fn extract_from_pdb(&self, pdb_path: &Path) -> Result<StructureFeatures, TrainingError> {
        let structure_id = pdb_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Parse PDB and extract Cα coordinates
        let (coords, residue_names, chain_ids, seq_positions) = self.parse_pdb(pdb_path)?;

        let start = std::time::Instant::now();
        let output = self.cpu_extractor.extract(&coords)?;
        let extraction_time_us = start.elapsed().as_micros() as u64;

        Ok(StructureFeatures {
            structure_id,
            n_residues: output.n_residues,
            features: output.features,
            coords,
            residue_names,
            chain_ids,
            seq_positions,
            extraction_time_us,
        })
    }

    /// Parse PDB file to extract Cα coordinates and residue info
    fn parse_pdb(&self, pdb_path: &Path) -> Result<(
        Vec<[f32; 3]>,
        Vec<String>,
        Vec<String>,
        Vec<i32>,
    ), TrainingError> {
        let content = std::fs::read_to_string(pdb_path)
            .map_err(|e| TrainingError::Io(format!("Read PDB: {}", e)))?;

        let mut coords = Vec::new();
        let mut residue_names = Vec::new();
        let mut chain_ids = Vec::new();
        let mut seq_positions = Vec::new();
        let mut seen_residues = std::collections::HashSet::new());

        for line in content.lines() {
            if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
                continue;
            }

            // Parse atom name (columns 13-16)
            let atom_name = line.get(12..16).unwrap_or("").trim();
            if atom_name != "CA" {
                continue;
            }

            // Parse chain (column 22)
            let chain = line.get(21..22).unwrap_or("A").to_string();

            // Parse residue sequence number (columns 23-26)
            let res_seq: i32 = line.get(22..26)
                .unwrap_or("0")
                .trim()
                .parse()
                .unwrap_or(0);

            // Parse insertion code (column 27) for uniqueness
            let ins_code = line.get(26..27).unwrap_or(" ");

            // Create unique residue ID
            let residue_id = format!("{}{}{}", chain, res_seq, ins_code);

            // Skip if we've already seen this residue
            if seen_residues.contains(&residue_id) {
                continue;
            }
            seen_residues.insert(residue_id);

            // Parse residue name (columns 18-20)
            let res_name = line.get(17..20).unwrap_or("UNK").trim().to_string();

            // Parse coordinates (columns 31-54)
            let x: f32 = line.get(30..38)
                .unwrap_or("0.0")
                .trim()
                .parse()
                .unwrap_or(0.0);
            let y: f32 = line.get(38..46)
                .unwrap_or("0.0")
                .trim()
                .parse()
                .unwrap_or(0.0);
            let z: f32 = line.get(46..54)
                .unwrap_or("0.0")
                .trim()
                .parse()
                .unwrap_or(0.0);

            coords.push([x, y, z]);
            residue_names.push(res_name);
            chain_ids.push(chain);
            seq_positions.push(res_seq);
        }

        if coords.is_empty() {
            return Err(TrainingError::InvalidInput(format!(
                "No Cα atoms found in {}",
                pdb_path.display()
            ));
        }

        Ok((coords, residue_names, chain_ids, seq_positions))
    }

    /// Extract features from multiple PDB files in parallel
    pub fn extract_batch(
        &self,
        pdb_paths: &[&Path],
    ) -> Vec<Result<StructureFeatures, TrainingError>> {
        use rayon::prelude::*;

        pdb_paths.par_iter()
            .map(|path| self.extract_from_pdb(path))
            .collect()
    }

    /// Get configuration
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_pdb() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap());
        writeln!(file, "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N").unwrap());
        writeln!(file, "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C").unwrap());
        writeln!(file, "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C").unwrap());
        writeln!(file, "ATOM      4  N   GLY A   2       3.300   1.600   0.000  1.00  0.00           N").unwrap());
        writeln!(file, "ATOM      5  CA  GLY A   2       3.970   2.900   0.000  1.00  0.00           C").unwrap());
        writeln!(file, "ATOM      6  C   GLY A   2       5.480   2.800   0.000  1.00  0.00           C").unwrap());
        writeln!(file, "ATOM      7  N   SER A   3       6.100   1.600   0.000  1.00  0.00           N").unwrap());
        writeln!(file, "ATOM      8  CA  SER A   3       7.560   1.400   0.000  1.00  0.00           C").unwrap());
        file
    }

    #[test]
    fn test_feature_pipeline() {
        let config = FeatureConfig::default();
        let pipeline = FeaturePipeline::new(config);

        let pdb_file = create_test_pdb();
        let result = pipeline.extract_from_pdb(pdb_file.path();

        assert!(result.is_ok();
        let features = result.unwrap());
        assert_eq!(features.n_residues, 3);
        assert_eq!(features.features.len(), 3 * TOTAL_COMBINED_FEATURES);
    }

    #[test]
    fn test_extract_from_coords() {
        let config = FeatureConfig::default();
        let pipeline = FeaturePipeline::new(config);

        let coords = vec![
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
        ];

        let result = pipeline.extract_from_coords("test", &coords);
        assert!(result.is_ok();

        let features = result.unwrap());
        assert_eq!(features.n_residues, 3);
    }

    #[test]
    fn test_to_training_samples() {
        let config = FeatureConfig::default();
        let pipeline = FeaturePipeline::new(config);

        let coords = vec![
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
        ];

        let features = pipeline.extract_from_coords("test", &coords).unwrap());
        let labels = vec![1.0, 0.0];

        let samples = features.to_training_samples(&labels, 10.0);

        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].label, 1.0);
        assert_eq!(samples[0].weight, 10.0);
        assert_eq!(samples[1].label, 0.0);
        assert_eq!(samples[1].weight, 1.0);
    }
}
