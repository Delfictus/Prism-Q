//! Simulated Annealing Classifier
//! Adapted from qubo.rs solve_cpu_sa() (lines 110-150)

use super::reorthogonalization::N_FEATURES;
use rand::prelude::*;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct SAClassifier {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub threshold: f32,
}

#[derive(Clone)]
pub struct SAConfig {
    pub max_iterations: usize,
    pub initial_temp: f64,
    pub cooling_rate: f64,
    pub perturbation_scale: f64,
    pub seed: u64,
}

impl Default for SAConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50000,
            initial_temp: 10.0,
            cooling_rate: 0.0,  // Will be computed from initial_temp
            perturbation_scale: 0.1,
            seed: 42,
        }
    }
}

impl SAClassifier {
    /// Train using simulated annealing (adapted from qubo.rs)
    pub fn train(
        features: &[Vec<f32>],
        labels: &[f32],
        pos_weight: f32,
        config: &SAConfig,
    ) -> Self {
        let mut rng = rand::thread_rng();
        
        log::info!("SA classifier training: {} samples, {} features", features.len(), N_FEATURES);
        log::info!("SA config: {} iterations, temp {:.1}, pos_weight {:.2}", 
                   config.max_iterations, config.initial_temp, pos_weight);

        // Initialize weights randomly
        let mut weights = vec![0.0f32; N_FEATURES];
        for w in &mut weights {
            *w = rng.gen_range(-0.1..0.1);
        }
        let mut bias = 0.0f32;

        let mut best_weights = weights.clone();
        let mut best_bias = bias;
        let mut best_loss = Self::compute_loss(features, labels, &weights, bias, pos_weight);
        let mut current_loss = best_loss;

        // Cooling schedule (from qubo.rs line 118)
        let mut temperature = config.initial_temp;
        let cooling_rate = (config.initial_temp / 0.01).powf(1.0 / config.max_iterations as f64);

        for iteration in 0..config.max_iterations {
            // Random perturbation (adapt from bit flip to weight update)
            let perturb_idx = rng.gen_range(0..N_FEATURES + 1);
            let perturb_amount = rng.gen_range(-config.perturbation_scale..config.perturbation_scale) as f32;

            let old_value = if perturb_idx < N_FEATURES {
                let old = weights[perturb_idx];
                weights[perturb_idx] += perturb_amount;
                old
            } else {
                let old = bias;
                bias += perturb_amount;
                old
            };

            // Calculate new loss (energy)
            let new_loss = Self::compute_loss(features, labels, &weights, bias, pos_weight);
            let delta_loss = new_loss - current_loss;

            // Metropolis criterion (EXACT from qubo.rs line 131)
            if delta_loss < 0.0 || rng.gen::<f64>() < (-delta_loss / temperature).exp() {
                current_loss = new_loss;
                if current_loss < best_loss {
                    best_loss = current_loss;
                    best_weights = weights.clone();
                    best_bias = bias;
                }
            } else {
                // Revert (from qubo.rs lines 139-140)
                if perturb_idx < N_FEATURES {
                    weights[perturb_idx] = old_value;
                } else {
                    bias = old_value;
                }
            }

            // Cool down (from qubo.rs line 144)
            temperature *= cooling_rate;

            if iteration % 10000 == 0 && iteration > 0 {
                log::debug!("SA iter {}: loss={:.4}, best={:.4}, temp={:.2e}", 
                           iteration, current_loss, best_loss, temperature);
            }
        }

        // Find optimal threshold on training data
        let predictions: Vec<f32> = features.iter()
            .map(|f| Self::sigmoid(Self::dot(&best_weights, best_bias, f)))
            .collect();
        let threshold = Self::optimize_threshold(&predictions, labels);

        log::info!("SA complete: best_loss={:.4}, threshold={:.3}", best_loss, threshold);

        Self {
            weights: best_weights,
            bias: best_bias,
            threshold,
        }
    }

    fn compute_loss(
        features: &[Vec<f32>],
        labels: &[f32],
        weights: &[f32],
        bias: f32,
        pos_weight: f32,
    ) -> f64 {
        let mut loss = 0.0f64;
        for (feat, &label) in features.iter().zip(labels) {
            let logit = Self::dot(weights, bias, feat);
            let prob = Self::sigmoid(logit) as f64;

            let weight = if label > 0.5 { pos_weight as f64 } else { 1.0 };
            let eps = 1e-7;
            loss -= weight * (
                label as f64 * (prob + eps).ln() +
                (1.0 - label as f64) * (1.0 - prob + eps).ln()
            );
        }
        loss / features.len() as f64
    }

    fn dot(weights: &[f32], bias: f32, features: &[f32]) -> f32 {
        let mut sum = bias;
        for (w, f) in weights.iter().zip(features) {
            sum += w * f;
        }
        sum
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    fn optimize_threshold(predictions: &[f32], labels: &[f32]) -> f32 {
        let mut best_threshold = 0.5;
        let mut best_f1 = 0.0;

        for i in 1..100 {
            let threshold = i as f32 / 100.0;
            let (_, _, f1) = Self::compute_metrics(predictions, labels, threshold);
            if f1 > best_f1 {
                best_f1 = f1;
                best_threshold = threshold;
            }
        }
        best_threshold
    }

    fn compute_metrics(predictions: &[f32], labels: &[f32], threshold: f32) -> (f32, f32, f32) {
        let mut tp = 0u32;
        let mut fp = 0u32;
        let mut fn_ = 0u32;

        for (&pred, &label) in predictions.iter().zip(labels) {
            let pred_pos = pred >= threshold;
            let actual_pos = label > 0.5;
            match (pred_pos, actual_pos) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                _ => {}
            }
        }

        let precision = tp as f32 / (tp + fp).max(1) as f32;
        let recall = tp as f32 / (tp + fn_).max(1) as f32;
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        (precision, recall, f1)
    }

    pub fn predict(&self, features: &[f32]) -> f32 {
        Self::sigmoid(Self::dot(&self.weights, self.bias, features))
    }

    pub fn predict_label(&self, features: &[f32]) -> bool {
        self.predict(features) >= self.threshold
    }
}
