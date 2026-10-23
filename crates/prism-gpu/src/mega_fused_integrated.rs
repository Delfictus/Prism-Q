//! Integrated Mega-Fused + TDA Feature Pipeline
//!
//! Combines 32-dim base features from geometric analysis with
//! 48-dim TDA features from spatial neighborhood analysis.
//!
//! Total output: 80-dimensional feature vectors per residue.
//!
//! ## CPU-Only Implementation
//!
//! This version is CPU-only for reliability. GPU acceleration can be
//! added later via the separate MegaFusedGpu class.

use prism_core::PrismError;
use rayon::prelude::*;

use crate::batch_tda::{
    NeighborhoodBuilder, NeighborhoodData, HybridTdaConfig,
    BASE_FEATURES, TDA_FEATURE_COUNT, TOTAL_COMBINED_FEATURES,
    TDA_RADII, TDA_SCALES,
};
use crate::batch_tda::half_utils::f16_to_f32;

/// Configuration for integrated pipeline
#[derive(Clone, Debug)]
pub struct IntegratedConfig {
    /// TDA configuration
    pub tda: HybridTdaConfig,
    /// Apply z-score normalization
    pub normalize: bool,
    /// Normalization statistics (mean, std per feature)
    pub norm_stats: Option<NormalizationStats>,
}

impl Default for IntegratedConfig {
    fn default() -> Self {
        Self {
            tda: HybridTdaConfig::default(),
            normalize: false,
            norm_stats: None,
        }
    }
}

/// Normalization statistics for z-score
#[derive(Clone, Debug)]
pub struct NormalizationStats {
    /// Mean per feature [TOTAL_COMBINED_FEATURES]
    pub mean: Vec<f32>,
    /// Standard deviation per feature [TOTAL_COMBINED_FEATURES]
    pub std: Vec<f32>,
}

impl NormalizationStats {
    /// Create with zeros
    pub fn zeros() -> Self {
        Self {
            mean: vec![0.0; TOTAL_COMBINED_FEATURES],
            std: vec![1.0; TOTAL_COMBINED_FEATURES],
        }
    }

    /// Validate dimensions
    pub fn validate(&self) -> Result<(), PrismError> {
        if self.mean.len() != TOTAL_COMBINED_FEATURES {
            return Err(PrismError::validation(format!(
                "Mean length {} != {}", self.mean.len(), TOTAL_COMBINED_FEATURES
            ));
        }
        if self.std.len() != TOTAL_COMBINED_FEATURES {
            return Err(PrismError::validation(format!(
                "Std length {} != {}", self.std.len(), TOTAL_COMBINED_FEATURES
            ));
        }
        Ok(())
    }
}

/// Output from integrated pipeline
#[derive(Clone, Debug)]
pub struct IntegratedOutput {
    /// Number of residues
    pub n_residues: usize,
    /// Combined features [n_residues × 80]
    pub features: Vec<f32>,
    /// Base feature extraction time (μs)
    pub base_time_us: u64,
    /// TDA extraction time (μs)
    pub tda_time_us: u64,
    /// Merge time (μs)
    pub merge_time_us: u64,
    /// Total time (μs)
    pub total_time_us: u64,
}

impl IntegratedOutput {
    /// Get features for a specific residue
    pub fn get_residue(&self, idx: usize) -> &[f32] {
        let start = idx * TOTAL_COMBINED_FEATURES;
        &self.features[start..start + TOTAL_COMBINED_FEATURES]
    }

    /// Get base features (first 32) for a residue
    pub fn get_base_features(&self, idx: usize) -> &[f32] {
        let start = idx * TOTAL_COMBINED_FEATURES;
        &self.features[start..start + BASE_FEATURES]
    }

    /// Get TDA features (last 48) for a residue
    pub fn get_tda_features(&self, idx: usize) -> &[f32] {
        let start = idx * TOTAL_COMBINED_FEATURES + BASE_FEATURES;
        &self.features[start..start + TDA_FEATURE_COUNT]
    }
}

/// CPU-only implementation for feature extraction
pub struct IntegratedCpu {
    /// Neighborhood builder
    nb_builder: NeighborhoodBuilder,
    /// Configuration
    config: IntegratedConfig,
}

impl IntegratedCpu {
    /// Create CPU-only integrated pipeline
    pub fn new() -> Self {
        let config = IntegratedConfig::default();
        let nb_builder = NeighborhoodBuilder::new()
            .with_radii(config.tda.radii.to_vec())
            .with_max_neighbors(config.tda.max_neighbors);

        Self { nb_builder, config }
    }

    /// Set configuration
    pub fn with_config(mut self, config: IntegratedConfig) -> Self {
        self.config = config.clone();
        self.nb_builder = NeighborhoodBuilder::new()
            .with_radii(config.tda.radii.to_vec())
            .with_max_neighbors(config.tda.max_neighbors);
        self
    }

    /// Extract combined features (CPU implementation)
    pub fn extract(&self, coords: &[[f32; 3]]) -> Result<IntegratedOutput, PrismError> {
        let total_start = std::time::Instant::now();
        let n_residues = coords.len();

        if n_residues == 0 {
            return Ok(IntegratedOutput {
                n_residues: 0,
                features: vec![],
                base_time_us: 0,
                tda_time_us: 0,
                merge_time_us: 0,
                total_time_us: 0,
            });
        }

        // Extract base features (simplified CPU version)
        let base_start = std::time::Instant::now();
        let base_features = self.extract_base_cpu(coords);
        let base_time_us = base_start.elapsed().as_micros() as u64;

        // Extract TDA features
        let tda_start = std::time::Instant::now();
        let neighborhood = self.nb_builder.build(coords);
        let tda_features = self.extract_tda_cpu(coords, &neighborhood);
        let tda_time_us = tda_start.elapsed().as_micros() as u64;

        // Merge
        let merge_start = std::time::Instant::now();
        let mut features = vec![0.0f32; n_residues * TOTAL_COMBINED_FEATURES];

        for i in 0..n_residues {
            let base_start = i * BASE_FEATURES;
            let tda_start = i * TDA_FEATURE_COUNT;
            let out_start = i * TOTAL_COMBINED_FEATURES;

            features[out_start..out_start + BASE_FEATURES]
                .copy_from_slice(&base_features[base_start..base_start + BASE_FEATURES]);
            features[out_start + BASE_FEATURES..out_start + TOTAL_COMBINED_FEATURES]
                .copy_from_slice(&tda_features[tda_start..tda_start + TDA_FEATURE_COUNT]);
        }
        let merge_time_us = merge_start.elapsed().as_micros() as u64;

        // Apply normalization if enabled
        if self.config.normalize {
            if let Some(ref stats) = self.config.norm_stats {
                for i in 0..n_residues {
                    for f in 0..TOTAL_COMBINED_FEATURES {
                        let idx = i * TOTAL_COMBINED_FEATURES + f;
                        features[idx] = (features[idx] - stats.mean[f]) / stats.std[f].max(1e-8);
                    }
                }
            }
        }

        let total_time_us = total_start.elapsed().as_micros() as u64;

        Ok(IntegratedOutput {
            n_residues,
            features,
            base_time_us,
            tda_time_us,
            merge_time_us,
            total_time_us,
        })
    }

    /// Extract simplified base features on CPU
    /// 32-dimensional geometric features per residue
    fn extract_base_cpu(&self, coords: &[[f32; 3]]) -> Vec<f32> {
        let n = coords.len();
        let mut features = vec![0.0f32; n * BASE_FEATURES];

        // Compute centroid
        let mut centroid = [0.0f32; 3];
        for c in coords {
            centroid[0] += c[0];
            centroid[1] += c[1];
            centroid[2] += c[2];
        }
        let inv_n = 1.0 / n as f32;
        centroid[0] *= inv_n;
        centroid[1] *= inv_n;
        centroid[2] *= inv_n;

        // Compute max distance from centroid for normalization
        let max_dist = coords.iter()
            .map(|c| {
                let dx = c[0] - centroid[0];
                let dy = c[1] - centroid[1];
                let dz = c[2] - centroid[2];
                (dx*dx + dy*dy + dz*dz).sqrt()
            })
            .fold(0.0f32, f32::max)
            .max(1.0);

        // Extract features per residue (parallelized)
        features.par_chunks_mut(BASE_FEATURES)
            .enumerate()
            .for_each(|(i, f)| {
                let coord = coords[i];

                // Features 0-2: Normalized position relative to centroid
                f[0] = (coord[0] - centroid[0]) / max_dist;
                f[1] = (coord[1] - centroid[1]) / max_dist;
                f[2] = (coord[2] - centroid[2]) / max_dist;

                // Feature 3: Distance from centroid (normalized)
                let dist_from_centroid = ((coord[0] - centroid[0]).powi(2) +
                                          (coord[1] - centroid[1]).powi(2) +
                                          (coord[2] - centroid[2]).powi(2)).sqrt();
                f[3] = dist_from_centroid / max_dist;

                // Features 4-7: Local density at different radii
                let mut density_8 = 0.0f32;
                let mut density_12 = 0.0f32;
                let mut density_16 = 0.0f32;
                let mut density_20 = 0.0f32;

                for (j, other) in coords.iter().enumerate() {
                    if i != j {
                        let dx = coord[0] - other[0];
                        let dy = coord[1] - other[1];
                        let dz = coord[2] - other[2];
                        let dist = (dx*dx + dy*dy + dz*dz).sqrt();

                        if dist < 8.0 { density_8 += 1.0; }
                        if dist < 12.0 { density_12 += 1.0; }
                        if dist < 16.0 { density_16 += 1.0; }
                        if dist < 20.0 { density_20 += 1.0; }
                    }
                }

                f[4] = density_8 / 20.0;   // Normalize by typical max
                f[5] = density_12 / 40.0;
                f[6] = density_16 / 60.0;
                f[7] = density_20 / 80.0;

                // Features 8-10: Nearest neighbor distances
                let mut min_dist = f32::MAX;
                let mut second_min = f32::MAX;
                let mut third_min = f32::MAX;

                for (j, other) in coords.iter().enumerate() {
                    if i != j {
                        let dx = coord[0] - other[0];
                        let dy = coord[1] - other[1];
                        let dz = coord[2] - other[2];
                        let dist = (dx*dx + dy*dy + dz*dz).sqrt();

                        if dist < min_dist {
                            third_min = second_min;
                            second_min = min_dist;
                            min_dist = dist;
                        } else if dist < second_min {
                            third_min = second_min;
                            second_min = dist;
                        } else if dist < third_min {
                            third_min = dist;
                        }
                    }
                }

                f[8] = min_dist / 10.0;        // Normalize by typical Cα-Cα distance
                f[9] = second_min / 10.0;
                f[10] = third_min / 10.0;

                // Features 11-13: Sequence position features
                let seq_frac = i as f32 / n.max(1) as f32;
                f[11] = seq_frac;                           // Relative position
                f[12] = (seq_frac * std::f32::consts::PI).sin(); // Sinusoidal
                f[13] = (seq_frac * std::f32::consts::PI).cos(); // Cosinusoidal

                // Features 14-16: Local curvature (using prev/next residues)
                if i > 0 && i < n - 1 {
                    let prev = coords[i - 1];
                    let next = coords[i + 1];

                    // Vector from prev to current
                    let v1 = [coord[0] - prev[0], coord[1] - prev[1], coord[2] - prev[2]];
                    // Vector from current to next
                    let v2 = [next[0] - coord[0], next[1] - coord[1], next[2] - coord[2]];

                    // Angle between vectors (curvature indicator)
                    let dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
                    let len1 = (v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]).sqrt();
                    let len2 = (v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]).sqrt();
                    let cos_angle = dot / (len1 * len2 + 1e-8);

                    f[14] = cos_angle;  // Curvature
                    f[15] = len1 / 4.0; // Bond length 1
                    f[16] = len2 / 4.0; // Bond length 2
                } else {
                    f[14] = 0.0;
                    f[15] = 0.0;
                    f[16] = 0.0;
                }

                // Features 17-23: Directional density (hemisphere analysis)
                let mut dir_pos_x = 0.0f32;
                let mut dir_neg_x = 0.0f32;
                let mut dir_pos_y = 0.0f32;
                let mut dir_neg_y = 0.0f32;
                let mut dir_pos_z = 0.0f32;
                let mut dir_neg_z = 0.0f32;
                let mut total_in_range = 0.0f32;

                for (j, other) in coords.iter().enumerate() {
                    if i != j {
                        let dx = other[0] - coord[0];
                        let dy = other[1] - coord[1];
                        let dz = other[2] - coord[2];
                        let dist = (dx*dx + dy*dy + dz*dz).sqrt();

                        if dist < 12.0 {
                            total_in_range += 1.0;
                            if dx > 0.0 { dir_pos_x += 1.0; } else { dir_neg_x += 1.0; }
                            if dy > 0.0 { dir_pos_y += 1.0; } else { dir_neg_y += 1.0; }
                            if dz > 0.0 { dir_pos_z += 1.0; } else { dir_neg_z += 1.0; }
                        }
                    }
                }

                let total = total_in_range.max(1.0);
                f[17] = dir_pos_x / total;
                f[18] = dir_neg_x / total;
                f[19] = dir_pos_y / total;
                f[20] = dir_neg_y / total;
                f[21] = dir_pos_z / total;
                f[22] = dir_neg_z / total;

                // Feature 23: Anisotropy
                let balance_x = (dir_pos_x - dir_neg_x).abs() / total;
                let balance_y = (dir_pos_y - dir_neg_y).abs() / total;
                let balance_z = (dir_pos_z - dir_neg_z).abs() / total;
                f[23] = (balance_x + balance_y + balance_z) / 3.0;

                // Features 24-27: Contact order features
                let mut contact_order = 0.0f32;
                let mut num_contacts = 0.0f32;

                for (j, other) in coords.iter().enumerate() {
                    if i != j {
                        let dx = coord[0] - other[0];
                        let dy = coord[1] - other[1];
                        let dz = coord[2] - other[2];
                        let dist = (dx*dx + dy*dy + dz*dz).sqrt();

                        if dist < 10.0 {
                            let seq_dist = (i as i32 - j as i32).abs() as f32;
                            contact_order += seq_dist;
                            num_contacts += 1.0;
                        }
                    }
                }

                f[24] = if num_contacts > 0.0 { contact_order / num_contacts / n as f32 } else { 0.0 };
                f[25] = num_contacts / 30.0;  // Normalized contact count

                // Features 26-27: Surface exposure proxy
                f[26] = 1.0 - f[4].min(1.0);  // Inverse of local density = exposure proxy
                f[27] = f[3];  // Distance from centroid (surface residues are further)

                // Features 28-31: Reserved for future use (zero-filled)
                f[28] = 0.0;
                f[29] = 0.0;
                f[30] = 0.0;
                f[31] = 0.0;
            });

        features
    }

    /// Extract TDA features on CPU
    fn extract_tda_cpu(&self, coords: &[[f32; 3]], neighborhood: &NeighborhoodData) -> Vec<f32> {
        use crate::batch_tda::executor::cpu_fallback;

        let n_residues = neighborhood.n_residues;
        let n_radii = neighborhood.n_radii;
        let scales = self.config.tda.scales;

        let features: Vec<[f32; TDA_FEATURE_COUNT]> = (0..n_residues)
            .into_par_iter()
            .map(|i| {
                let mut all_features = [0.0f32; TDA_FEATURE_COUNT];
                let center = coords[i];

                for r in 0..n_radii {
                    let offset_idx = i * n_radii + r;
                    let start = neighborhood.offsets[offset_idx] as usize;
                    let end = neighborhood.offsets[offset_idx + 1] as usize;

                    if start >= end {
                        continue;
                    }

                    let neighbor_indices = &neighborhood.neighbor_indices[start..end];
                    let neighbor_distances_f16 = &neighborhood.neighbor_distances[start..end];

                    let neighbor_coords: Vec<[f32; 3]> = neighbor_indices
                        .iter()
                        .map(|&idx| coords[idx as usize])
                        .collect();

                    let neighbor_distances: Vec<f32> = neighbor_distances_f16
                        .iter()
                        .map(|&d| f16_to_f32(d))
                        .collect();

                    let radius_features = cpu_fallback::extract_tda_features_cpu(
                        center,
                        &neighbor_coords,
                        &neighbor_distances,
                        &scales,
                    );

                    let offset = r * 16;
                    all_features[offset..offset + 16].copy_from_slice(&radius_features);
                }

                all_features
            })
            .collect();

        features.into_iter().flatten().collect()
    }
}

impl Default for IntegratedCpu {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_helix_coords(n: usize) -> Vec<[f32; 3]> {
        (0..n).map(|i| {
            let t = i as f32 * 0.3;
            [t.cos() * 5.0, t.sin() * 5.0, i as f32 * 1.5]
        }).collect()
    }

    #[test]
    fn test_integrated_cpu() {
        let pipeline = IntegratedCpu::new());
        let coords = make_helix_coords(50);

        let output = pipeline.extract(&coords).unwrap());

        assert_eq!(output.n_residues, 50);
        assert_eq!(output.features.len(), 50 * TOTAL_COMBINED_FEATURES);
    }

    #[test]
    fn test_output_accessors() {
        let pipeline = IntegratedCpu::new());
        let coords = make_helix_coords(10);

        let output = pipeline.extract(&coords).unwrap());

        // Check accessor methods
        for i in 0..10 {
            let residue = output.get_residue(i);
            assert_eq!(residue.len(), TOTAL_COMBINED_FEATURES);

            let base = output.get_base_features(i);
            assert_eq!(base.len(), BASE_FEATURES);

            let tda = output.get_tda_features(i);
            assert_eq!(tda.len(), TDA_FEATURE_COUNT);
        }
    }

    #[test]
    fn test_empty_coords() {
        let pipeline = IntegratedCpu::new());
        let coords: Vec<[f32; 3]> = vec![];

        let output = pipeline.extract(&coords).unwrap());

        assert_eq!(output.n_residues, 0);
        assert!(output.features.is_empty();
    }
}
