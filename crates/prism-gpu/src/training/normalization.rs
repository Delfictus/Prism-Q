//! Feature Normalization with Welford's Algorithm
//!
//! Provides numerically stable online computation of mean and variance
//! for z-score normalization of feature vectors.

use crate::batch_tda::TOTAL_COMBINED_FEATURES;
use serde::{Serialize, Deserialize};

/// Normalization statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormStats {
    /// Mean per feature
    pub mean: Vec<f32>,
    /// Standard deviation per feature
    pub std: Vec<f32>,
    /// Number of samples used to compute statistics
    pub n_samples: usize,
}

impl NormStats {
    /// Create with zeros (identity normalization)
    pub fn identity(n_features: usize) -> Self {
        Self {
            mean: vec![0.0; n_features],
            std: vec![1.0; n_features],
            n_samples: 0,
        }
    }

    /// Apply z-score normalization in-place
    pub fn normalize_inplace(&self, features: &mut [f32]) {
        assert_eq!(features.len() % self.mean.len(), 0);
        let n_features = self.mean.len();

        for chunk in features.chunks_exact_mut(n_features) {
            for (i, val) in chunk.iter_mut().enumerate() {
                *val = (*val - self.mean[i]) / self.std[i].max(1e-8);
            }
        }
    }

    /// Apply z-score normalization, returning new vector
    pub fn normalize(&self, features: &[f32]) -> Vec<f32> {
        let mut result = features.to_vec();
        self.normalize_inplace(&mut result);
        result
    }

    /// Denormalize features (inverse operation)
    pub fn denormalize(&self, features: &[f32]) -> Vec<f32> {
        let n_features = self.mean.len();
        features.iter()
            .enumerate()
            .map(|(i, &val)| {
                let f = i % n_features;
                val * self.std[f] + self.mean[f]
            })
            .collect()
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

impl Default for NormStats {
    fn default() -> Self {
        Self::identity(TOTAL_COMBINED_FEATURES)
    }
}

/// Welford's online algorithm for numerically stable variance computation
///
/// Computes running mean and variance in a single pass with O(1) memory.
#[derive(Clone, Debug)]
pub struct WelfordStats {
    /// Number of features
    n_features: usize,
    /// Number of samples seen
    count: u64,
    /// Running mean per feature
    mean: Vec<f64>,
    /// Running M2 (sum of squared differences from mean)
    m2: Vec<f64>,
}

impl WelfordStats {
    /// Create a new Welford accumulator
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            count: 0,
            mean: vec![0.0; n_features],
            m2: vec![0.0; n_features],
        }
    }

    /// Add a single sample
    pub fn update(&mut self, sample: &[f32]) {
        assert_eq!(sample.len(), self.n_features);

        self.count += 1;
        let n = self.count as f64;

        for (i, &x) in sample.iter().enumerate() {
            let x = x as f64;
            let delta = x - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = x - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Add a batch of samples
    pub fn update_batch(&mut self, samples: &[f32]) {
        assert_eq!(samples.len() % self.n_features, 0);
        let n_samples = samples.len() / self.n_features;

        for i in 0..n_samples {
            let start = i * self.n_features;
            let sample = &samples[start..start + self.n_features];
            self.update(sample);
        }
    }

    /// Merge another WelfordStats into this one
    ///
    /// Uses parallel algorithm for combining statistics.
    pub fn merge(&mut self, other: &WelfordStats) {
        assert_eq!(self.n_features, other.n_features);

        if other.count == 0 {
            return;
        }

        if self.count == 0 {
            self.count = other.count;
            self.mean = other.mean.clone();
            self.m2 = other.m2.clone();
            return;
        }

        let n_a = self.count as f64;
        let n_b = other.count as f64;
        let n_ab = n_a + n_b;

        for i in 0..self.n_features {
            let delta = other.mean[i] - self.mean[i];

            // Combined mean
            let new_mean = (n_a * self.mean[i] + n_b * other.mean[i]) / n_ab;

            // Combined M2
            let new_m2 = self.m2[i] + other.m2[i] +
                delta * delta * n_a * n_b / n_ab;

            self.mean[i] = new_mean;
            self.m2[i] = new_m2;
        }

        self.count = self.count + other.count;
    }

    /// Finalize and get normalization statistics
    pub fn finalize(&self) -> NormStats {
        let mean: Vec<f32> = self.mean.iter().map(|&x| x as f32).collect();

        let std: Vec<f32> = if self.count > 1 {
            self.m2.iter()
                .map(|&m2| (m2 / (self.count - 1) as f64).sqrt() as f32)
                .collect()
        } else {
            vec![1.0; self.n_features]
        };

        NormStats {
            mean,
            std,
            n_samples: self.count as usize,
        }
    }

    /// Get sample count
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get current mean (before finalization)
    pub fn current_mean(&self) -> Vec<f32> {
        self.mean.iter().map(|&x| x as f32).collect()
    }

    /// Get current variance (before finalization)
    pub fn current_variance(&self) -> Vec<f32> {
        if self.count > 1 {
            self.m2.iter()
                .map(|&m2| (m2 / (self.count - 1) as f64) as f32)
                .collect()
        } else {
            vec![0.0; self.n_features]
        }
    }
}

/// Feature normalizer that computes statistics online
pub struct Normalizer {
    /// Welford accumulator
    stats: WelfordStats,
    /// Finalized statistics (cached after finalize())
    finalized: Option<NormStats>,
}

impl Normalizer {
    /// Create a new normalizer
    pub fn new(n_features: usize) -> Self {
        Self {
            stats: WelfordStats::new(n_features),
            finalized: None,
        }
    }

    /// Add samples to the normalizer
    pub fn fit(&mut self, samples: &[f32]) {
        self.stats.update_batch(samples);
        self.finalized = None; // Invalidate cached stats
    }

    /// Add a single sample
    pub fn fit_one(&mut self, sample: &[f32]) {
        self.stats.update(sample);
        self.finalized = None;
    }

    /// Finalize and get normalization statistics
    pub fn finalize(&mut self) -> &NormStats {
        if self.finalized.is_none() {
            self.finalized = Some(self.stats.finalize();
        }
        self.finalized.as_ref().unwrap()
    }

    /// Transform features using computed statistics
    pub fn transform(&mut self, features: &[f32]) -> Vec<f32> {
        self.finalize().normalize(features)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, features: &[f32]) -> Vec<f32> {
        self.fit(features);
        self.transform(features)
    }

    /// Get number of samples seen
    pub fn n_samples(&self) -> usize {
        self.stats.count() as usize
    }

    /// Merge another normalizer's statistics
    pub fn merge(&mut self, other: &Normalizer) {
        self.stats.merge(&other.stats);
        self.finalized = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_welford_basic() {
        let mut stats = WelfordStats::new(2);

        // Add samples: [1, 2], [3, 4], [5, 6]
        stats.update(&[1.0, 2.0]);
        stats.update(&[3.0, 4.0]);
        stats.update(&[5.0, 6.0]);

        let result = stats.finalize();

        // Mean should be [3, 4]
        assert!((result.mean[0] - 3.0).abs() < 1e-6);
        assert!((result.mean[1] - 4.0).abs() < 1e-6);

        // Std should be [2, 2] (sample std)
        assert!((result.std[0] - 2.0).abs() < 1e-6);
        assert!((result.std[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_welford_merge() {
        let mut stats1 = WelfordStats::new(2);
        let mut stats2 = WelfordStats::new(2);

        // Split data between two accumulators
        stats1.update(&[1.0, 2.0]);
        stats1.update(&[3.0, 4.0]);

        stats2.update(&[5.0, 6.0]);
        stats2.update(&[7.0, 8.0]);

        // Merge
        stats1.merge(&stats2);

        let result = stats1.finalize();

        // Mean should be [4, 5]
        assert!((result.mean[0] - 4.0).abs() < 1e-6);
        assert!((result.mean[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalization() {
        let stats = NormStats {
            mean: vec![2.0, 4.0],
            std: vec![2.0, 2.0],
            n_samples: 100,
        };

        let features = vec![4.0, 6.0];
        let normalized = stats.normalize(&features);

        // (4 - 2) / 2 = 1, (6 - 4) / 2 = 1
        assert!((normalized[0] - 1.0).abs() < 1e-6);
        assert!((normalized[1] - 1.0).abs() < 1e-6);

        // Test denormalize
        let denormalized = stats.denormalize(&normalized);
        assert!((denormalized[0] - 4.0).abs() < 1e-6);
        assert!((denormalized[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalizer() {
        let mut normalizer = Normalizer::new(2);

        normalizer.fit(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let n_samples = normalizer.n_samples();
        let stats = normalizer.finalize();
        assert_eq!(n_samples, 3);
        assert!((stats.mean[0] - 3.0).abs() < 1e-6);
        assert!((stats.mean[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_single_sample() {
        let mut stats = WelfordStats::new(2);
        stats.update(&[5.0, 10.0]);

        let result = stats.finalize();

        // With single sample, std should default to 1.0
        assert!((result.mean[0] - 5.0).abs() < 1e-6);
        assert!((result.std[0] - 1.0).abs() < 1e-6);
    }
}
