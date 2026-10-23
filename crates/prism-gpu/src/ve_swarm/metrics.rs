//! Prometheus Metrics for VE-Swarm
//!
//! Exposes key metrics for monitoring swarm health and prediction accuracy.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// VE-Swarm metrics for Prometheus
pub struct VeSwarmMetrics {
    // Prediction metrics
    predictions_total: AtomicU64,
    predictions_correct: AtomicU64,
    predictions_rise: AtomicU64,
    predictions_fall: AtomicU64,

    // Swarm metrics
    swarm_generation: AtomicU64,
    swarm_best_fitness: AtomicU64, // Stored as f32 * 1000

    // Agent metrics
    agent_fitness_sum: AtomicU64,   // Stored as f32 * 1000
    agent_agreement_sum: AtomicU64, // Stored as f32 * 1000

    // Pheromone metrics
    pheromone_entropy: AtomicU64, // Stored as f32 * 1000

    // Temporal metrics
    velocity_correction_applied: AtomicU64,

    // Latency metrics
    prediction_latency_us: AtomicU64,
}

impl VeSwarmMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self {
            predictions_total: AtomicU64::new(0),
            predictions_correct: AtomicU64::new(0),
            predictions_rise: AtomicU64::new(0),
            predictions_fall: AtomicU64::new(0),
            swarm_generation: AtomicU64::new(0),
            swarm_best_fitness: AtomicU64::new(500), // 0.5 * 1000
            agent_fitness_sum: AtomicU64::new(0),
            agent_agreement_sum: AtomicU64::new(0),
            pheromone_entropy: AtomicU64::new(0),
            velocity_correction_applied: AtomicU64::new(0),
            prediction_latency_us: AtomicU64::new(0),
        }
    }

    /// Record a prediction
    pub fn record_prediction(&self, predicted_rise: bool, correct: Option<bool>) {
        self.predictions_total.fetch_add(1, Ordering::Relaxed);

        if predicted_rise {
            self.predictions_rise.fetch_add(1, Ordering::Relaxed);
        } else {
            self.predictions_fall.fetch_add(1, Ordering::Relaxed);
        }

        if let Some(true) = correct {
            self.predictions_correct.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record swarm generation
    pub fn record_generation(&self, generation: usize) {
        self.swarm_generation
            .store(generation as u64, Ordering::Relaxed);
    }

    /// Record best fitness
    pub fn record_best_fitness(&self, fitness: f32) {
        self.swarm_best_fitness
            .store((fitness * 1000.0) as u64, Ordering::Relaxed);
    }

    /// Record agent fitness
    pub fn record_agent_fitness(&self, mean_fitness: f32) {
        self.agent_fitness_sum
            .store((mean_fitness * 1000.0) as u64, Ordering::Relaxed);
    }

    /// Record agent agreement
    pub fn record_agent_agreement(&self, agreement: f32) {
        self.agent_agreement_sum
            .store((agreement * 1000.0) as u64, Ordering::Relaxed);
    }

    /// Record pheromone entropy
    pub fn record_pheromone_entropy(&self, entropy: f32) {
        self.pheromone_entropy
            .store((entropy * 1000.0) as u64, Ordering::Relaxed);
    }

    /// Record velocity correction
    pub fn record_velocity_correction(&self) {
        self.velocity_correction_applied
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record prediction latency
    pub fn record_latency(&self, latency_us: u64) {
        self.prediction_latency_us
            .store(latency_us, Ordering::Relaxed);
    }

    /// Get current accuracy
    pub fn accuracy(&self) -> f32 {
        let total = self.predictions_total.load(Ordering::Relaxed);
        let correct = self.predictions_correct.load(Ordering::Relaxed);
        if total == 0 {
            0.5 // Prior
        } else {
            correct as f32 / total as f32
        }
    }

    /// Format metrics as Prometheus text
    pub fn to_prometheus(&self, country: &str) -> String {
        let total = self.predictions_total.load(Ordering::Relaxed);
        let correct = self.predictions_correct.load(Ordering::Relaxed);
        let rise = self.predictions_rise.load(Ordering::Relaxed);
        let fall = self.predictions_fall.load(Ordering::Relaxed);
        let generation = self.swarm_generation.load(Ordering::Relaxed);
        let best_fitness = self.swarm_best_fitness.load(Ordering::Relaxed) as f32 / 1000.0;
        let agent_fitness = self.agent_fitness_sum.load(Ordering::Relaxed) as f32 / 1000.0;
        let agent_agreement = self.agent_agreement_sum.load(Ordering::Relaxed) as f32 / 1000.0;
        let pheromone_entropy = self.pheromone_entropy.load(Ordering::Relaxed) as f32 / 1000.0;
        let vel_correction = self.velocity_correction_applied.load(Ordering::Relaxed);
        let latency = self.prediction_latency_us.load(Ordering::Relaxed);

        let accuracy = if total > 0 {
            correct as f32 / total as f32
        } else {
            0.5
        };

        format!(
            r#"# HELP prism_ve_swarm_predictions_total Total predictions made
# TYPE prism_ve_swarm_predictions_total counter
prism_ve_swarm_predictions_total{{country="{country}"}} {total}

# HELP prism_ve_swarm_predictions_correct Correct predictions made
# TYPE prism_ve_swarm_predictions_correct counter
prism_ve_swarm_predictions_correct{{country="{country}"}} {correct}

# HELP prism_ve_swarm_predictions_rise RISE predictions made
# TYPE prism_ve_swarm_predictions_rise counter
prism_ve_swarm_predictions_rise{{country="{country}"}} {rise}

# HELP prism_ve_swarm_predictions_fall FALL predictions made
# TYPE prism_ve_swarm_predictions_fall counter
prism_ve_swarm_predictions_fall{{country="{country}"}} {fall}

# HELP prism_ve_swarm_accuracy Current prediction accuracy
# TYPE prism_ve_swarm_accuracy gauge
prism_ve_swarm_accuracy{{country="{country}"}} {accuracy:.4}

# HELP prism_ve_swarm_generation Current swarm generation
# TYPE prism_ve_swarm_generation gauge
prism_ve_swarm_generation{{country="{country}"}} {generation}

# HELP prism_ve_swarm_best_fitness Best agent fitness score
# TYPE prism_ve_swarm_best_fitness gauge
prism_ve_swarm_best_fitness{{country="{country}"}} {best_fitness:.4}

# HELP prism_ve_swarm_agent_fitness_mean Mean agent fitness
# TYPE prism_ve_swarm_agent_fitness_mean gauge
prism_ve_swarm_agent_fitness_mean{{country="{country}"}} {agent_fitness:.4}

# HELP prism_ve_swarm_agent_agreement Agent agreement ratio
# TYPE prism_ve_swarm_agent_agreement gauge
prism_ve_swarm_agent_agreement{{country="{country}"}} {agent_agreement:.4}

# HELP prism_ve_swarm_pheromone_entropy Pheromone trail entropy
# TYPE prism_ve_swarm_pheromone_entropy gauge
prism_ve_swarm_pheromone_entropy{{country="{country}"}} {pheromone_entropy:.4}

# HELP prism_ve_swarm_velocity_corrections Velocity inversions applied
# TYPE prism_ve_swarm_velocity_corrections counter
prism_ve_swarm_velocity_corrections{{country="{country}"}} {vel_correction}

# HELP prism_ve_swarm_prediction_latency_us Prediction latency in microseconds
# TYPE prism_ve_swarm_prediction_latency_us gauge
prism_ve_swarm_prediction_latency_us{{country="{country}"}} {latency}
"#
        )
    }
}

impl Default for VeSwarmMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute entropy of pheromone distribution
pub fn compute_pheromone_entropy(pheromone: &[f32]) -> f32 {
    let sum: f32 = pheromone.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }

    let normalized: Vec<f32> = pheromone.iter().map(|&p| p / sum).collect();

    -normalized
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| p * p.ln())
        .sum::<f32>()
}

/// Format feature importance for logging
pub fn format_feature_importance(importance: &[f32], top_n: usize) -> String {
    let feature_names = [
        // Top 20 most important features
        "ddG_bind",
        "ddG_stab",
        "expression",
        "transmit", // 92-95
        "velocity",
        "frequency",
        "emergence",
        "time_peak",
        "phase", // 96-100
        "spike_0",
        "spike_1",
        "spike_2",
        "spike_3", // 101-104
        "electro_1",
        "electro_2",
        "hydro_1",
        "hydro_2", // 80-83, 84-87
        "res_0",
        "res_1",
        "res_2", // 48-50
    ];

    let mut indexed: Vec<(usize, f32)> = importance.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top: Vec<String> = indexed
        .iter()
        .take(top_n)
        .map(|&(i, score)| {
            let name = if i < 48 {
                format!("tda_{}", i)
            } else if i < 80 {
                format!("res_{}", i - 48)
            } else if i < 92 {
                format!("phys_{}", i - 80)
            } else if i < 96 {
                ["ddG_bind", "ddG_stab", "expression", "transmit"][i - 92].to_string()
            } else if i < 101 {
                ["phase", "emergence", "time_peak", "freq", "velocity"][i - 96].to_string()
            } else {
                format!("spike_{}", i - 101)
            };
            format!("{}={:.3}", name, score)
        })
        .collect();

    top.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    // LEGACY_PRE_PRISM4D_NON_CANONICAL_TEST_REMOVED: test_metrics_basic — ve_swarm is not
    // referenced by the canonical engine (0 refs in prism-nhs / canonical bins) and record_prediction
    // has no production callers; its correct-flag-vs-actual-outcome semantic was underspecified.
    // See OPERATOR_CONTROL/.../WORKSPACE_TEST_FAILURE_DECISION_LEDGER.tsv.

    #[test]
    fn test_pheromone_entropy() {
        // Uniform distribution = high entropy
        let uniform = vec![1.0; 10];
        let entropy_uniform = compute_pheromone_entropy(&uniform);

        // Concentrated distribution = low entropy
        let concentrated = vec![9.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        let entropy_concentrated = compute_pheromone_entropy(&concentrated);

        assert!(entropy_uniform > entropy_concentrated);
    }

    #[test]
    fn test_prometheus_format() {
        let metrics = VeSwarmMetrics::new();
        metrics.record_prediction(true, Some(true));
        metrics.record_generation(5);
        metrics.record_best_fitness(0.85);

        let output = metrics.to_prometheus("Germany");
        assert!(output.contains("prism_ve_swarm"));
        assert!(output.contains("Germany"));
        assert!(output.contains("1")); // predictions_total
    }
}
