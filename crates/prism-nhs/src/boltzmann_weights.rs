//! Learned thermodynamic constants from JAX Differentiable Canonical Ensemble.
//!
//! These weights were trained on PRISM4D-BENCH30 (30 proteins, 1005 pockets)
//! via GPU-accelerated gradient descent with constrained positive weights.
//! Architecture: ΔG = ΔH + W - TΔS + K + C with Boltzmann ranking.
//!
//! Training: 1000 epochs, Adam optimizer, Cross-entropy loss = Free energy minimization.
//! Uncertainty: Laplace approximation (Hessian at optimum).
//!
//! AUTO-GENERATED — do not edit manually.
//! Regenerate with: conda run -n prism_dock python benchmarks/prism4d_bench30/train_boltzmann_jax.py

/// Inverse temperature β (controls ranking sharpness)
/// V9 Stage 2 retrained constants
pub const BETA: f32 = 3.953;

/// Effective temperature T for TΔS term
pub const T_EFF: f32 = 0.063;

/// Thermodynamic scaling coefficients (convert arbitrary units to energy scale)
/// Index mapping:
///  [0] H_spike_density      — van der Waals/steric contact enthalpy
///  [1] H_burial_depth       — burial enthalpy
///  [2] H_frustrated_water   — solvation enthalpy
///  [3] H_lining_count       — direct contact enthalpy
///  [4] S_ray_entropy        — geometric freedom entropy
///  [5] S_spike_type_LPV     — chemical microstate diversity
///  [6] S_uv_enrichment      — aromatic complexity
///  [7] S_sphericity          — shape entropy (inverted)
///  [8] W_activation         — energy barrier to open pocket
///  [9] W_enclosure          — cost to access pocket
/// [10] K_wavefront_coherence — sequential opening signal
/// [11] K_funnel_ratio       — funnel topology
/// [12] K_breathing          — pocket dynamics
/// [13] C_vcs_orthogonal     — cross-strategy consensus
/// [14] C_quality_score      — composite v7 quality
/// V9 Stage 2 weights — retrained on V9 intensive feature landscape
pub const WEIGHTS: [f32; 15] = [
    1.9238, // H_spike_density (DOMINANT — 96.2% Fisher information)
    0.0234, // H_burial_depth
    0.0453, // H_frustrated_water
    0.0932, // H_lining_count (increased from 0.027 — lining matters more with intensive)
    0.0454, // S_ray_entropy
    0.0561, // S_spike_type_LPV
    0.0634, // S_uv_enrichment
    0.0603, // S_sphericity
    0.0133, // W_activation
    0.0054, // W_enclosure
    0.0454, // K_wavefront_coherence
    0.0454, // K_funnel_ratio
    0.0089, // K_breathing
    0.0517, // C_vcs_orthogonal
    0.6019, // C_quality_score (15.1% Fisher information)
];

/// Feature names for diagnostics
pub const FEATURE_NAMES: [&str; 15] = [
    "H_spike_density",
    "H_burial_depth",
    "H_frustrated_water",
    "H_lining_count",
    "S_ray_entropy",
    "S_spike_type_LPV",
    "S_uv_enrichment",
    "S_sphericity",
    "W_activation",
    "W_enclosure",
    "K_wavefront_coherence",
    "K_funnel_ratio",
    "K_breathing",
    "C_vcs_orthogonal",
    "C_quality_score",
];

/// Compute ΔG for a pocket given its 15 physics features.
/// Returns the thermodynamic free energy (lower = more favorable for binding).
pub fn compute_delta_g(features: &[f32; 15]) -> f32 {
    let w = &WEIGHTS;

    // Enthalpy (negative = favorable binding)
    let h = -(w[0] * features[0] + w[1] * features[1] + w[2] * features[2] + w[3] * features[3]);

    // Entropy (positive = favorable, more microstates)
    let s = w[4] * features[4] + w[5] * features[5] + w[6] * features[6] + w[7] * features[7];

    // Work (positive = unfavorable, barrier to access)
    let work = w[8] * features[8] + w[9] * features[9];

    // Kinetic (negative = favorable dynamics)
    let k = -(w[10] * features[10] + w[11] * features[11] + w[12] * features[12]);

    // Consensus (negative = favorable agreement)
    let c = -(w[13] * features[13] + w[14] * features[14]);

    h + work - T_EFF * s + k + c
}

/// Compute Boltzmann probability for a set of pocket ΔG values.
/// Returns probabilities that sum to 1.0 (partition function normalized).
pub fn boltzmann_probabilities(delta_gs: &[f32]) -> Vec<f32> {
    if delta_gs.is_empty() {
        return vec![];
    }

    // Compute -β*ΔG for each pocket
    let logits: Vec<f32> = delta_gs.iter().map(|dg| -BETA * dg).collect();

    // Numerical stability: subtract max before exp
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
    let z: f32 = exp_vals.iter().sum();

    exp_vals.iter().map(|e| e / z).collect()
}

/// Rank pockets by Boltzmann probability (highest probability = rank 1).
/// Returns indices sorted by probability descending.
pub fn rank_by_boltzmann(features_per_pocket: &[[f32; 15]]) -> Vec<usize> {
    let delta_gs: Vec<f32> = features_per_pocket
        .iter()
        .map(|f| compute_delta_g(f))
        .collect();
    let probs = boltzmann_probabilities(&delta_gs);

    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}
