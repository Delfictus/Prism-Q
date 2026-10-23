//! Feature Reorthogonalization via PCA Whitening
//! Practical implementation for 92-dim features

use serde::{Deserialize, Serialize};

pub const N_FEATURES: usize = 92;

#[derive(Clone, Serialize, Deserialize)]
pub struct WhiteningParams {
    pub means: Vec<f32>,
    pub stds: Vec<f32>,
    pub eigenvalues: Vec<f32>,
    pub variance_explained: Vec<f32>,
}

impl WhiteningParams {
    /// Compute PCA whitening from training features
    pub fn fit(features: &[Vec<f32>]) -> Self {
        let n_samples = features.len();
        if n_samples == 0 {
            return Self {
                means: vec![0.0; N_FEATURES],
                stds: vec![1.0; N_FEATURES],
                eigenvalues: vec![1.0; N_FEATURES],
                variance_explained: (0..N_FEATURES).map(|i| i as f32 / N_FEATURES as f32).collect(),
            };
        }

        log::info!("Computing PCA whitening for {} samples, {} features", n_samples, N_FEATURES);

        // 1. Compute means
        let mut means = vec![0.0f64; N_FEATURES];
        for sample in features {
            for (f, &val) in sample.iter().take(N_FEATURES).enumerate() {
                means[f] += val as f64;
            }
        }
        for m in &mut means {
            *m /= n_samples as f64;
        }

        // 2. Compute standard deviations (for diagonal whitening)
        let mut stds = vec![0.0f64; N_FEATURES];
        for sample in features {
            for (f, &val) in sample.iter().take(N_FEATURES).enumerate() {
                let diff = val as f64 - means[f];
                stds[f] += diff * diff;
            }
        }
        
        let mut eigenvalues = Vec::with_capacity(N_FEATURES);
        for s in &mut stds {
            let variance = *s / n_samples as f64;
            eigenvalues.push(variance);
            *s = variance.sqrt().max(1e-6);
        }

        // 3. Compute variance explained (sorted eigenvalues)
        let total_var: f64 = eigenvalues.iter().sum();
        let mut sorted_eigs: Vec<f64> = eigenvalues.clone();
        sorted_eigs.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let mut variance_explained = Vec::with_capacity(N_FEATURES);
        let mut cumulative = 0.0;
        for &eig in &sorted_eigs {
            cumulative += eig / total_var;
            variance_explained.push(cumulative as f32);
        }

        log::info!(
            "PCA fit: {} features, variance explained by top 50: {:.1}%, top 80: {:.1}%",
            N_FEATURES,
            variance_explained[49.min(N_FEATURES-1)] * 100.0,
            variance_explained[79.min(N_FEATURES-1)] * 100.0
        );

        Self {
            means: means.iter().map(|&m| m as f32).collect(),
            stds: stds.iter().map(|&s| s as f32).collect(),
            eigenvalues: eigenvalues.iter().map(|&e| e as f32).collect(),
            variance_explained,
        }
    }

    /// Transform single sample (diagonal whitening)
    pub fn transform(&self, features: &[f32]) -> Vec<f32> {
        features.iter()
            .zip(self.means.iter())
            .zip(self.stds.iter())
            .map(|((&f, &m), &s)| (f - m) / s)
            .collect()
    }

    /// Transform batch in place
    pub fn transform_batch(&self, features: &mut [Vec<f32>]) {
        for sample in features.iter_mut() {
            let transformed = self.transform(sample);
            *sample = transformed;
        }
    }
}
