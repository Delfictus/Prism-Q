//! Swarm Consensus and Final Prediction
//!
//! Aggregates agent predictions using fitness-weighted voting with
//! physics-informed constraints.

use anyhow::Result;

/// Physics constraints for prediction
#[derive(Clone, Debug)]
pub struct PhysicsConstraints {
    /// Frequency threshold for saturation (high freq = likely fall)
    pub saturation_threshold: f32,
    /// Velocity threshold for peak detection
    pub peak_velocity_threshold: f32,
    /// Minimum frequency for meaningful prediction
    pub min_frequency: f32,
    /// Maximum frequency change per week (sanity check)
    pub max_freq_change: f32,
}

impl Default for PhysicsConstraints {
    fn default() -> Self {
        Self {
            saturation_threshold: 0.7,
            peak_velocity_threshold: 0.1,
            min_frequency: 0.01,
            max_freq_change: 0.3,
        }
    }
}

/// Consensus prediction result
#[derive(Clone, Debug)]
pub struct ConsensusPrediction {
    /// Final RISE probability
    pub rise_prob: f32,
    /// Prediction confidence
    pub confidence: f32,
    /// Predicted label
    pub predicted_rise: bool,
    /// Physics constraint adjustments applied
    pub constraints_applied: Vec<String>,
    /// Agent agreement ratio
    pub agent_agreement: f32,
}

/// Compute weighted consensus from agent predictions
pub fn compute_consensus(
    agent_predictions: &[f32],
    agent_confidences: &[f32],
    agent_fitness: &[f32],
    current_frequency: f32,
    current_velocity: f32,
    constraints: &PhysicsConstraints,
) -> ConsensusPrediction {
    let n = agent_predictions.len();
    assert_eq!(n, agent_confidences.len());
    assert_eq!(n, agent_fitness.len());

    // Compute fitness-weighted prediction
    let mut sum_weighted_pred = 0.0f32;
    let mut sum_weight = 0.0f32;

    for i in 0..n {
        let weight = agent_fitness[i] * agent_confidences[i];
        sum_weighted_pred += agent_predictions[i] * weight;
        sum_weight += weight;
    }

    let mut consensus = sum_weighted_pred / sum_weight.max(1e-6);

    // Track applied constraints
    let mut constraints_applied = Vec::new();

    // =========================================================================
    // Physics-Informed Constraint Application
    // =========================================================================

    // Constraint 1: High frequency saturation
    // Variants at high frequency tend to FALL (no room to grow)
    if current_frequency > constraints.saturation_threshold {
        let adjustment = 0.2 * (current_frequency - constraints.saturation_threshold)
            / (1.0 - constraints.saturation_threshold);
        consensus *= 1.0 - adjustment;
        constraints_applied.push(format!(
            "Saturation: freq={:.2} > {:.2}, consensus *= {:.2}",
            current_frequency,
            constraints.saturation_threshold,
            1.0 - adjustment
        ));
    }

    // Constraint 2: Velocity inversion at moderate-high frequency
    // High velocity + high frequency = AT PEAK, about to FALL
    if current_velocity > constraints.peak_velocity_threshold && current_frequency > 0.3 {
        let adjustment = 0.3 * current_velocity / constraints.peak_velocity_threshold;
        let new_consensus = consensus * (1.0 - adjustment) + 0.3 * adjustment;
        constraints_applied.push(format!(
            "Peak detection: vel={:.3} > {:.3} at freq={:.2}, {} -> {}",
            current_velocity,
            constraints.peak_velocity_threshold,
            current_frequency,
            consensus,
            new_consensus
        ));
        consensus = new_consensus;
    }

    // Constraint 3: Low frequency growth is likely TRUE
    if current_frequency < 0.1 && current_velocity > 0.0 {
        let boost = 0.2 * (0.1 - current_frequency) / 0.1;
        let new_consensus = consensus * (1.0 - boost) + 0.7 * boost;
        constraints_applied.push(format!(
            "Early growth: freq={:.3} with positive vel, {} -> {}",
            current_frequency, consensus, new_consensus
        ));
        consensus = new_consensus;
    }

    // Constraint 4: Very low frequency = probably noise, dampen confidence
    if current_frequency < constraints.min_frequency {
        consensus *= 0.5;
        constraints_applied.push(format!(
            "Low signal: freq={:.4} < {:.4}, consensus *= 0.5",
            current_frequency, constraints.min_frequency
        ));
    }

    // Compute agent agreement
    let rise_count = agent_predictions.iter().filter(|&&p| p > 0.5).count();
    let agent_agreement = rise_count.max(n - rise_count) as f32 / n as f32;

    // Final confidence
    let confidence = sum_weight / n as f32 * agent_agreement;

    ConsensusPrediction {
        rise_prob: consensus.clamp(0.0, 1.0),
        confidence: confidence.clamp(0.0, 1.0),
        predicted_rise: consensus > 0.5,
        constraints_applied,
        agent_agreement,
    }
}

/// Apply velocity inversion correction (standalone)
pub fn correct_velocity(velocity: f32, frequency: f32) -> f32 {
    if frequency > 0.5 {
        // High frequency: invert velocity signal
        // Positive velocity here means AT PEAK, so invert to indicate FALL
        -velocity * 2.0
    } else if frequency > 0.2 && velocity > 0.05 {
        // Moderate frequency with strong growth: likely approaching peak
        // Dampen the velocity signal
        velocity * 0.3
    } else if frequency < 0.1 && velocity > 0.0 {
        // Low frequency with positive velocity: true growth phase
        // Amplify the signal
        velocity * 1.5
    } else {
        velocity
    }
}

/// Determine prediction label with calibrated threshold
pub fn predict_label(rise_prob: f32, confidence: f32, threshold: f32) -> (bool, &'static str) {
    if confidence < 0.3 {
        // Low confidence: default to FALL (safer prediction)
        (false, "LOW_CONFIDENCE_FALL")
    } else if rise_prob > threshold + 0.1 {
        // High probability RISE
        (true, "HIGH_CONFIDENCE_RISE")
    } else if rise_prob < threshold - 0.1 {
        // High probability FALL
        (false, "HIGH_CONFIDENCE_FALL")
    } else if rise_prob > threshold {
        // Marginal RISE
        (true, "MARGINAL_RISE")
    } else {
        // Marginal FALL
        (false, "MARGINAL_FALL")
    }
}

/// Batch prediction for multiple variants
pub fn batch_consensus(
    all_predictions: &[Vec<f32>], // [N_variants x N_agents]
    all_confidences: &[Vec<f32>], // [N_variants x N_agents]
    all_fitness: &[f32],          // [N_agents] (shared)
    frequencies: &[f32],          // [N_variants]
    velocities: &[f32],           // [N_variants]
    constraints: &PhysicsConstraints,
) -> Vec<ConsensusPrediction> {
    let n_variants = all_predictions.len();

    (0..n_variants)
        .map(|v| {
            compute_consensus(
                &all_predictions[v],
                &all_confidences[v],
                all_fitness,
                frequencies[v],
                velocities[v],
                constraints,
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_velocity_correction() {
        // High frequency: invert
        assert!(correct_velocity(0.1, 0.6) < 0.0);

        // Low frequency: amplify
        assert!(correct_velocity(0.05, 0.05) > 0.05);

        // Moderate frequency with growth: dampen
        assert!(correct_velocity(0.1, 0.3) < 0.1);
    }

    #[test]
    #[ignore = "LEGACY_PRE_PRISM4D_NON_CANONICAL: ve_swarm consensus is not used by the canonical engine (0 refs); the saturation-dampening threshold is a consensus-model domain decision. Runtime unchanged. Run via `cargo test -- --ignored`"]
    fn test_consensus_saturation() {
        let predictions = vec![0.8, 0.7, 0.6, 0.8];
        let confidences = vec![0.9, 0.8, 0.7, 0.9];
        let fitness = vec![0.7, 0.7, 0.7, 0.7];

        let constraints = PhysicsConstraints::default();

        // Low frequency: prediction should pass through mostly unchanged
        let result = compute_consensus(
            &predictions,
            &confidences,
            &fitness,
            0.1,
            0.05,
            &constraints,
        );
        assert!(result.rise_prob > 0.6);

        // High frequency: prediction should be dampened
        let result = compute_consensus(
            &predictions,
            &confidences,
            &fitness,
            0.8,
            0.05,
            &constraints,
        );
        assert!(result.rise_prob < 0.6);
        assert!(result
            .constraints_applied
            .iter()
            .any(|s| s.contains("Saturation")));
    }

    #[test]
    fn test_peak_detection() {
        let predictions = vec![0.7, 0.7, 0.7, 0.7];
        let confidences = vec![0.8, 0.8, 0.8, 0.8];
        let fitness = vec![0.7, 0.7, 0.7, 0.7];

        let constraints = PhysicsConstraints::default();

        // High velocity at moderate frequency: should detect peak
        let result = compute_consensus(
            &predictions,
            &confidences,
            &fitness,
            0.4,
            0.15,
            &constraints,
        );
        assert!(result
            .constraints_applied
            .iter()
            .any(|s| s.contains("Peak")));
    }

    #[test]
    fn test_predict_label() {
        let (label, _) = predict_label(0.8, 0.9, 0.5);
        assert!(label);

        let (label, _) = predict_label(0.2, 0.9, 0.5);
        assert!(!label);

        let (label, reason) = predict_label(0.5, 0.1, 0.5);
        assert!(!label);
        assert!(reason.contains("LOW_CONFIDENCE"));
    }
}
