//! Dendritic Neuromorphic Agent with RLS Learning
//!
//! Replaces the DQN (PyTorch) agent with a neuromorphic architecture:
//! - **Reservoir**: GPU-accelerated LIF spiking neural network
//! - **Readout**: Linear layer trained via Recursive Least Squares
//! - **Stability**: Target network pattern with Polyak averaging
//!
//! ## Architecture
//!
//! ```text
//! 23 features → [SNN Reservoir] → 512 filtered rates → [Linear Readout] → 4×5 Q-values
//!                  (GPU/LIF)         (smooth state)       (RLS trained)    (factorized)
//! ```
//!
//! ## Key Benefits
//!
//! - **No PyTorch**: Pure Rust + CUDA (~5MB vs ~100MB binary)
//! - **Faster Inference**: Microseconds vs milliseconds
//! - **Online Learning**: RLS updates in real-time
//! - **Stable Training**: Target network + regularization
//!
//! ## RLS Update Rule (Regularized)
//!
//! For readout weights W_out and state vector x:
//! ```text
//! k = P @ x / (λ + x^T @ P @ x)      // Kalman gain
//! P = (1/λ) * (P - k @ x^T @ P)      // Precision update
//! W = W + (target - W @ x) @ k^T     // Weight update
//! ```
//!
//! where λ is the regularization parameter (forgetting factor).

use anyhow::{Context, Result};
use cudarc::driver::CudaContext;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

use prism_gpu::DendriticSNNReservoir;

/// Number of bins per physics parameter (same as DQN)
pub const BINS_PER_PARAM: usize = 5;
/// Number of physics parameters (temp, friction, spring_k, bias)
pub const NUM_PARAMS: usize = 4;
/// Total output Q-values (4 × 5 = 20)
pub const NUM_OUTPUTS: usize = NUM_PARAMS * BINS_PER_PARAM;
/// Default reservoir size
pub const RESERVOIR_SIZE: usize = 512;
/// Default regularization parameter (forgetting factor)
pub const DEFAULT_LAMBDA: f32 = 0.99;
/// Default Polyak averaging coefficient
pub const DEFAULT_TAU: f32 = 0.005;
/// Initial P matrix diagonal value
pub const INITIAL_P_DIAG: f32 = 100.0;

// ============================================================================
// REWARD-MODULATED PLASTICITY PARAMETERS
// ============================================================================

/// Minimum reward modulation (prevents zero learning)
pub const MIN_REWARD_MODULATION: f32 = 0.1;
/// Maximum reward modulation (prevents explosive learning)
pub const MAX_REWARD_MODULATION: f32 = 2.0;
/// Reward scale factor for modulation (higher = more sensitive to reward)
pub const REWARD_MODULATION_SCALE: f32 = 0.5;

/// Factorized action (same interface as DQN)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FactorizedAction {
    pub temp_idx: usize,
    pub friction_idx: usize,
    pub spring_idx: usize,
    pub bias_idx: usize,
}

impl FactorizedAction {
    pub fn new(temp: usize, friction: usize, spring: usize, bias: usize) -> Self {
        Self {
            temp_idx: temp,
            friction_idx: friction,
            spring_idx: spring,
            bias_idx: bias,
        }
    }

    pub fn to_flat(&self) -> usize {
        self.temp_idx * 125 + self.friction_idx * 25 + self.spring_idx * 5 + self.bias_idx
    }

    pub fn from_flat(flat: usize) -> Self {
        Self {
            temp_idx: (flat / 125) % 5,
            friction_idx: (flat / 25) % 5,
            spring_idx: (flat / 5) % 5,
            bias_idx: flat % 5,
        }
    }

    pub fn to_array(&self) -> [usize; 4] {
        [
            self.temp_idx,
            self.friction_idx,
            self.spring_idx,
            self.bias_idx,
        ]
    }
}

/// Configuration for the Dendritic Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DendriticAgentConfig {
    /// Reservoir size (number of LIF neurons)
    pub reservoir_size: usize,
    /// RLS forgetting factor (λ) - higher = slower adaptation
    pub lambda: f32,
    /// Polyak averaging coefficient (τ) for target weights
    pub tau: f32,
    /// Initial exploration rate
    pub epsilon_start: f64,
    /// Minimum exploration rate
    pub epsilon_min: f64,
    /// Exploration decay rate
    pub epsilon_decay: f64,
    /// Discount factor (γ)
    pub gamma: f32,
    /// Target update frequency (in steps)
    pub target_update_freq: u64,
}

impl Default for DendriticAgentConfig {
    fn default() -> Self {
        Self {
            reservoir_size: RESERVOIR_SIZE,
            lambda: DEFAULT_LAMBDA,
            tau: DEFAULT_TAU,
            epsilon_start: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.995,
            gamma: 0.99,
            target_update_freq: 100,
        }
    }
}

/// RLS Readout Layer
///
/// Maintains weights and precision matrix for Recursive Least Squares learning.
/// Uses 4 separate readout heads (one per physics parameter).
struct RLSReadout {
    /// Active weights [NUM_OUTPUTS × reservoir_size]
    w_active: Vec<Vec<f32>>,
    /// Target weights (slow-moving copy for stability)
    w_target: Vec<Vec<f32>>,
    /// Precision matrices (one per output) [NUM_OUTPUTS × reservoir_size × reservoir_size]
    /// Stored as flattened vectors for efficiency
    p_matrices: Vec<Vec<f32>>,
    /// Regularization parameter
    lambda: f32,
    /// Polyak averaging coefficient
    tau: f32,
    /// Reservoir size
    reservoir_size: usize,
}

impl RLSReadout {
    /// Create new RLS readout layer
    fn new(reservoir_size: usize, lambda: f32, tau: f32) -> Self {
        // Initialize weights to small random values
        let mut rng = rand::thread_rng();
        let w_active: Vec<Vec<f32>> = (0..NUM_OUTPUTS)
            .map(|_| {
                (0..reservoir_size)
                    .map(|_| (rand::random::<f32>() - 0.5) * 0.01)
                    .collect()
            })
            .collect();

        let w_target = w_active.clone();

        // Initialize P matrices to λ * I (diagonal)
        let p_matrices: Vec<Vec<f32>> = (0..NUM_OUTPUTS)
            .map(|_| {
                let mut p = vec![0.0f32; reservoir_size * reservoir_size];
                for i in 0..reservoir_size {
                    p[i * reservoir_size + i] = INITIAL_P_DIAG;
                }
                p
            })
            .collect();

        info!(
            "RLS Readout initialized: {} outputs × {} reservoir neurons",
            NUM_OUTPUTS, reservoir_size
        );

        Self {
            w_active,
            w_target,
            p_matrices,
            lambda,
            tau,
            reservoir_size,
        }
    }

    /// Compute Q-values from reservoir state
    fn compute_q_values(&self, state: &[f32], use_target: bool) -> [f32; NUM_OUTPUTS] {
        let weights = if use_target {
            &self.w_target
        } else {
            &self.w_active
        };

        let mut q_values = [0.0f32; NUM_OUTPUTS];
        for (i, w) in weights.iter().enumerate() {
            q_values[i] = w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum();
        }
        q_values
    }

    /// Standard RLS update for a single output head
    ///
    /// Updates weights to minimize (target - w @ x)^2
    fn rls_update(&mut self, output_idx: usize, state: &[f32], target: f32) {
        self.rls_update_modulated(output_idx, state, target, 1.0);
    }

    /// Reward-Modulated RLS update (Feature Adapter Protocol)
    ///
    /// Biologically-inspired plasticity: learning rate scales with reward magnitude.
    /// - High reward → more learning (strengthen successful actions)
    /// - Low/negative reward → less learning (but don't ignore failures)
    ///
    /// This mimics dopaminergic modulation of synaptic plasticity in real brains.
    ///
    /// # Arguments
    /// * `output_idx` - Which output head to update
    /// * `state` - Reservoir state vector
    /// * `target` - TD target value
    /// * `reward_modulation` - Reward-based learning rate scaling [MIN..MAX]
    fn rls_update_modulated(
        &mut self,
        output_idx: usize,
        state: &[f32],
        target: f32,
        reward_modulation: f32,
    ) {
        let n = self.reservoir_size;
        let w = &mut self.w_active[output_idx];
        let p = &mut self.p_matrices[output_idx];

        // Skip update if state contains NaN/Inf
        if state.iter().any(|x| !x.is_finite()) {
            log::warn!("⚠️ RLS: Skipping update due to non-finite state values");
            return;
        }

        // Skip if target is NaN/Inf
        if !target.is_finite() {
            log::warn!("⚠️ RLS: Skipping update due to non-finite target");
            return;
        }

        // Clamp reward modulation to safe range
        let modulation = reward_modulation
            .max(MIN_REWARD_MODULATION)
            .min(MAX_REWARD_MODULATION);

        // Step 1: Compute P @ x
        let mut px = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                px[i] += p[i * n + j] * state[j];
            }
        }

        // Step 2: Compute x^T @ P @ x
        let xtpx: f32 = state.iter().zip(px.iter()).map(|(xi, pxi)| xi * pxi).sum();

        // Step 3: Compute Kalman gain k = P @ x / (λ + x^T @ P @ x)
        let denom = (self.lambda + xtpx).max(1e-8); // Prevent division by zero
        let mut k = vec![0.0f32; n];
        for i in 0..n {
            k[i] = px[i] / denom;
            // Clamp Kalman gain to prevent explosions
            k[i] = k[i].clamp(-1e6, 1e6);
        }

        // Step 4: Compute prediction error
        let prediction: f32 = w.iter().zip(state.iter()).map(|(wi, si)| wi * si).sum();
        let error = target - prediction;

        // Step 5: Update weights with REWARD MODULATION
        // w = w + modulation * error * k
        // Clamp error to prevent explosive updates
        let clamped_error = error.clamp(-100.0, 100.0);
        for i in 0..n {
            w[i] += modulation * clamped_error * k[i];
            // Clamp weights to prevent overflow
            w[i] = w[i].clamp(-1e4, 1e4);
        }

        // Step 6: Update P matrix: P = (1/λ) * (P - k @ x^T @ P)
        // Sherman-Morrison update (also modulated for consistency)
        let inv_lambda = 1.0 / self.lambda;
        let p_modulation = 0.5 + 0.5 * modulation; // P update less aggressive
        for i in 0..n {
            for j in 0..n {
                let delta = p_modulation * k[i] * px[j];
                p[i * n + j] = inv_lambda * (p[i * n + j] - delta);
                // Clamp P matrix elements
                p[i * n + j] = p[i * n + j].clamp(-1e8, 1e8);
            }
        }

        // Step 7: Regularization - ensure P diagonal doesn't collapse or explode
        for i in 0..n {
            p[i * n + i] = p[i * n + i].clamp(1e-6, 1e6);
        }
    }

    /// Compute reward modulation factor from raw reward
    ///
    /// Maps reward to [MIN_REWARD_MODULATION, MAX_REWARD_MODULATION]
    /// using a soft transformation that preserves sign information.
    fn compute_reward_modulation(reward: f32) -> f32 {
        // Sigmoid-like transformation centered at 0
        // reward = 0 → modulation = 1.0 (baseline)
        // reward > 0 → modulation > 1.0 (more learning)
        // reward < 0 → modulation < 1.0 (less learning, but not zero)
        let scaled = reward * REWARD_MODULATION_SCALE;
        let modulation = 1.0 + scaled.tanh(); // Range: [0, 2]

        // Map to [MIN, MAX] range
        MIN_REWARD_MODULATION + modulation * (MAX_REWARD_MODULATION - MIN_REWARD_MODULATION) / 2.0
    }

    /// Update target weights using Polyak averaging
    fn update_target_weights(&mut self) {
        for (w_t, w_a) in self.w_target.iter_mut().zip(self.w_active.iter()) {
            for (wt, wa) in w_t.iter_mut().zip(w_a.iter()) {
                *wt = self.tau * *wa + (1.0 - self.tau) * *wt;
            }
        }
    }

    /// Hard copy active weights to target
    fn hard_update_target(&mut self) {
        for (w_t, w_a) in self.w_target.iter_mut().zip(self.w_active.iter()) {
            w_t.copy_from_slice(w_a);
        }
    }

    /// Reset P matrices to initial state
    fn reset_precision_matrices(&mut self) {
        for p in &mut self.p_matrices {
            p.fill(0.0);
            for i in 0..self.reservoir_size {
                p[i * self.reservoir_size + i] = INITIAL_P_DIAG;
            }
        }
    }
}

/// Dendritic Neuromorphic Agent
///
/// Combines GPU-accelerated SNN reservoir with CPU-based RLS readout.
/// Drop-in replacement for DQNAgent with identical interface.
pub struct DendriticAgent {
    /// GPU SNN reservoir
    reservoir: DendriticSNNReservoir,
    /// RLS readout layer
    readout: RLSReadout,
    /// Configuration
    config: DendriticAgentConfig,
    /// Current exploration rate
    epsilon: f64,
    /// Training step counter
    step_count: u64,
    /// Last reservoir state (cached for training)
    last_state: Option<Vec<f32>>,
    /// Learning rate multiplier (HIL control)
    learning_rate_multiplier: f32,
}

impl DendriticAgent {
    /// Create new Dendritic Agent
    ///
    /// # Arguments
    /// * `input_dim` - Feature vector size (must be 23)
    /// * `_output_dim` - Ignored (kept for API compatibility)
    /// * `device_idx` - CUDA device index
    pub fn new(input_dim: i64, _output_dim: i64, device_idx: usize) -> Result<Self> {
        Self::new_with_config(input_dim, device_idx, DendriticAgentConfig::default())
    }

    /// Create new Dendritic Agent with custom configuration
    ///
    /// # Feature Dimensions (v3.1.1 Meta-Learning + Bio-Chemistry)
    /// - Standard: 23 features (Global + Target + Stability + Family + Temporal)
    /// - With difficulty: 27 features (+4 difficulty one-hot)
    /// - With glycan awareness: 31 features (+4 glycan features)
    /// - Full meta-learning: 37 features (+6 mechanism one-hot)
    /// - Full atomic-aware: 40 features (+3 bio-chemistry: grease, hinge, frustration)
    pub fn new_with_config(
        input_dim: i64,
        device_idx: usize,
        config: DendriticAgentConfig,
    ) -> Result<Self> {
        // Accept valid feature dimensions for different configurations
        // 23: standard, 27: +difficulty, 31: +glycan, 37: meta-learning, 40: +bio-chemistry
        anyhow::ensure!(
            input_dim == 23
                || input_dim == 27
                || input_dim == 31
                || input_dim == 37
                || input_dim == 40,
            "DendriticAgent requires 23, 27, 31, 37, or 40 input features, got {}. \
             23=standard, 27=+difficulty, 31=+glycan, 37=meta-learning, 40=+bio-chemistry",
            input_dim
        );

        // Initialize CUDA context
        let context = CudaContext::new(device_idx).context("Failed to create CUDA context")?;

        // Create SNN reservoir
        let mut reservoir = DendriticSNNReservoir::new(context, config.reservoir_size)
            .context("Failed to create SNN reservoir")?;

        // Initialize reservoir weights
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        reservoir.initialize(seed)?;

        // Create RLS readout
        let readout = RLSReadout::new(config.reservoir_size, config.lambda, config.tau);

        info!("🧠 Dendritic Agent initialized on CUDA:{}", device_idx);
        info!(
            "   Architecture: 23→SNN({})→RLS({})",
            config.reservoir_size, NUM_OUTPUTS
        );
        info!("   Lambda: {}, Tau: {}", config.lambda, config.tau);
        info!("   No PyTorch dependency - Pure neuromorphic!");

        Ok(Self {
            reservoir,
            readout,
            config: config.clone(),
            epsilon: config.epsilon_start,
            step_count: 0,
            last_state: None,
            learning_rate_multiplier: 1.0,
        })
    }

    // ========================================================================
    // ACTION SELECTION
    // ========================================================================

    /// Select action (flat index) using epsilon-greedy policy
    pub fn select_action(&mut self, features: &[f32]) -> usize {
        let action = self.select_factorized_action(features);
        action.to_flat()
    }

    /// Select factorized action
    pub fn select_factorized_action(&mut self, features: &[f32]) -> FactorizedAction {
        // Epsilon-greedy exploration
        if rand::random::<f64>() < self.epsilon {
            return FactorizedAction::new(
                rand::random::<usize>() % BINS_PER_PARAM,
                rand::random::<usize>() % BINS_PER_PARAM,
                rand::random::<usize>() % BINS_PER_PARAM,
                rand::random::<usize>() % BINS_PER_PARAM,
            );
        }

        // Process through reservoir
        let state = match self.reservoir.process_features(features) {
            Ok(s) => s,
            Err(e) => {
                warn!("Reservoir error: {}, using random action", e);
                return FactorizedAction::new(
                    rand::random::<usize>() % BINS_PER_PARAM,
                    rand::random::<usize>() % BINS_PER_PARAM,
                    rand::random::<usize>() % BINS_PER_PARAM,
                    rand::random::<usize>() % BINS_PER_PARAM,
                );
            }
        };

        // Cache state for potential training
        self.last_state = Some(state.clone());

        // Compute Q-values using active weights
        let q_values = self.readout.compute_q_values(&state, false);

        // Check for NaN in Q-values (indicates numerical instability)
        let nan_count = q_values.iter().filter(|x| x.is_nan()).count();
        if nan_count > 0 {
            log::warn!(
                "⚠️ NaN detected in Q-values: {}/{} values are NaN. Falling back to random action.",
                nan_count,
                q_values.len()
            );
            // Return random action when Q-values are corrupted
            use rand::Rng;
            let mut rng = rand::thread_rng();
            return FactorizedAction::new(
                rng.gen_range(0..BINS_PER_PARAM),
                rng.gen_range(0..BINS_PER_PARAM),
                rng.gen_range(0..BINS_PER_PARAM),
                rng.gen_range(0..BINS_PER_PARAM),
            );
        }

        // Select best action from each head (handle NaN gracefully)
        let safe_cmp = |a: f32, b: f32| -> std::cmp::Ordering {
            // Treat NaN as less than any real number
            match (a.is_nan(), b.is_nan()) {
                (true, true) => std::cmp::Ordering::Equal,
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                (false, false) => a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal),
            }
        };

        let temp_idx = (0..BINS_PER_PARAM)
            .max_by(|&a, &b| safe_cmp(q_values[a], q_values[b]))
            .unwrap_or(0);
        let fric_idx = (0..BINS_PER_PARAM)
            .max_by(|&a, &b| safe_cmp(q_values[BINS_PER_PARAM + a], q_values[BINS_PER_PARAM + b]))
            .unwrap_or(0);
        let spring_idx = (0..BINS_PER_PARAM)
            .max_by(|&a, &b| {
                safe_cmp(
                    q_values[2 * BINS_PER_PARAM + a],
                    q_values[2 * BINS_PER_PARAM + b],
                )
            })
            .unwrap_or(0);
        let bias_idx = (0..BINS_PER_PARAM)
            .max_by(|&a, &b| {
                safe_cmp(
                    q_values[3 * BINS_PER_PARAM + a],
                    q_values[3 * BINS_PER_PARAM + b],
                )
            })
            .unwrap_or(0);

        FactorizedAction::new(temp_idx, fric_idx, spring_idx, bias_idx)
    }

    /// Select greedy action (no exploration)
    pub fn select_action_greedy(&mut self, features: &[f32]) -> usize {
        let old_epsilon = self.epsilon;
        self.epsilon = 0.0;
        let action = self.select_action(features);
        self.epsilon = old_epsilon;
        action
    }

    // ========================================================================
    // TRAINING (RLS)
    // ========================================================================

    /// Train on batch using Reward-Modulated RLS updates
    ///
    /// Feature Adapter Protocol: Reward-Modulated Plasticity
    /// - Learning rate scales with reward magnitude
    /// - High rewards → stronger weight updates
    /// - Low/negative rewards → weaker updates (but not zero)
    ///
    /// For each transition (s, a, r, s', done):
    /// 1. Compute reservoir state for s and s'
    /// 2. Compute TD target using target weights
    /// 3. Compute reward modulation factor
    /// 4. Update active weights via modulated RLS
    /// 5. Polyak-average target weights
    pub fn train(&mut self, batch: Vec<(Vec<f32>, usize, f32, Vec<f32>, bool)>) -> Result<f32> {
        if batch.is_empty() {
            return Ok(0.0);
        }

        let mut total_error = 0.0f32;
        let gamma = self.config.gamma;
        let batch_len = batch.len();

        for (state_features, action, reward, next_features, done) in batch {
            // Process states through reservoir
            let state = self.reservoir.process_features(&state_features)?;
            let next_state = self.reservoir.process_features(&next_features)?;

            // Compute TD target using TARGET weights
            let next_q = self.readout.compute_q_values(&next_state, true);

            // Extract action indices
            let action_fact = FactorizedAction::from_flat(action);
            let action_indices = [
                action_fact.temp_idx,
                action_fact.friction_idx + BINS_PER_PARAM,
                action_fact.spring_idx + 2 * BINS_PER_PARAM,
                action_fact.bias_idx + 3 * BINS_PER_PARAM,
            ];

            // ================================================================
            // REWARD-MODULATED PLASTICITY: Scale learning by reward magnitude
            // Apply learning rate multiplier (HIL control)
            // ================================================================
            let base_modulation = RLSReadout::compute_reward_modulation(reward);
            let reward_modulation = base_modulation * self.learning_rate_multiplier;

            // For each head, compute max Q and update with modulation
            for head in 0..NUM_PARAMS {
                let head_start = head * BINS_PER_PARAM;
                let head_end = head_start + BINS_PER_PARAM;

                // Max Q for this head from next state
                let max_next_q = next_q[head_start..head_end]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);

                // TD target
                let target = if done {
                    reward
                } else {
                    reward + gamma * max_next_q
                };

                // Current Q-value
                let current_q = self.readout.compute_q_values(&state, false)[action_indices[head]];
                let error = (target - current_q).abs();
                total_error += error;

                // RLS update with REWARD MODULATION
                self.readout.rls_update_modulated(
                    action_indices[head],
                    &state,
                    target,
                    reward_modulation,
                );
            }

            // Polyak averaging of target weights (every step)
            self.readout.update_target_weights();
        }

        // Update state
        self.step_count += 1;
        self.decay_epsilon();

        let avg_error = total_error / (batch_len * NUM_PARAMS) as f32;
        Ok(avg_error)
    }

    // ========================================================================
    // EPSILON MANAGEMENT
    // ========================================================================

    pub fn decay_epsilon(&mut self) {
        if self.epsilon > self.config.epsilon_min {
            self.epsilon *= self.config.epsilon_decay;
        }
    }

    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon.max(0.0).min(1.0);
    }

    pub fn get_epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn get_step_count(&self) -> u64 {
        self.step_count
    }

    /// Set learning rate multiplier (HIL control)
    /// Values > 1.0 speed up learning, < 1.0 slow it down
    pub fn set_learning_rate_multiplier(&mut self, multiplier: f32) {
        self.learning_rate_multiplier = multiplier.clamp(0.1, 10.0);
        log::info!(
            "📈 Learning rate multiplier set to {:.2}x",
            self.learning_rate_multiplier
        );
    }

    pub fn get_learning_rate_multiplier(&self) -> f32 {
        self.learning_rate_multiplier
    }

    /// Export neural network state for visualization dashboard
    /// Returns a JSON-serializable struct with all internal states
    pub fn export_neural_state(&self, features: Option<&[f32]>) -> NeuralStateExport {
        // Get reservoir state (last cached or zeros)
        let reservoir_state: Vec<f32> = self
            .last_state
            .clone()
            .unwrap_or_else(|| vec![0.0; self.config.reservoir_size]);

        // Compute Q-values from current state
        let q_values: Vec<f32> = if self.last_state.is_some() {
            self.readout
                .compute_q_values(self.last_state.as_ref().unwrap(), false)
                .to_vec()
        } else {
            vec![0.0; NUM_OUTPUTS]
        };

        // Get weight statistics per output head
        let weight_stats: Vec<WeightStats> = self
            .readout
            .w_active
            .iter()
            .enumerate()
            .map(|(i, weights)| {
                let sum: f32 = weights.iter().sum();
                let mean = sum / weights.len() as f32;
                let variance: f32 =
                    weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / weights.len() as f32;
                let min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                WeightStats {
                    head_idx: i,
                    mean,
                    std: variance.sqrt(),
                    min,
                    max,
                    l2_norm: weights.iter().map(|w| w * w).sum::<f32>().sqrt(),
                }
            })
            .collect();

        // Subsample weights for visualization (full matrix too large)
        // Take every Nth weight to get ~64 values per head
        let subsample_rate = (self.config.reservoir_size / 64).max(1);
        let weight_heatmap: Vec<Vec<f32>> = self
            .readout
            .w_active
            .iter()
            .map(|weights| {
                weights
                    .iter()
                    .step_by(subsample_rate)
                    .take(64)
                    .cloned()
                    .collect()
            })
            .collect();

        // Subsample reservoir state for heatmap (reshape to 2D grid)
        let reservoir_heatmap: Vec<f32> = reservoir_state
            .iter()
            .step_by(subsample_rate)
            .take(256)
            .cloned()
            .collect();

        // Feature importance (approximate via weight magnitudes)
        let feature_importance: Vec<f32> = if let Some(feats) = features {
            feats.to_vec()
        } else {
            vec![0.0; 23]
        };

        NeuralStateExport {
            timestamp: chrono::Utc::now().to_rfc3339(),
            architecture: format!(
                "{}→SNN({})→RLS({})",
                feature_importance.len(),
                self.config.reservoir_size,
                NUM_OUTPUTS
            ),
            reservoir_size: self.config.reservoir_size,
            num_outputs: NUM_OUTPUTS,
            epsilon: self.epsilon,
            step_count: self.step_count,
            learning_rate_multiplier: self.learning_rate_multiplier,

            // Neural states
            reservoir_heatmap,
            reservoir_stats: ReservoirStats {
                mean: reservoir_state.iter().sum::<f32>() / reservoir_state.len() as f32,
                std: {
                    let mean = reservoir_state.iter().sum::<f32>() / reservoir_state.len() as f32;
                    (reservoir_state
                        .iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f32>()
                        / reservoir_state.len() as f32)
                        .sqrt()
                },
                sparsity: reservoir_state.iter().filter(|&&x| x.abs() < 0.01).count() as f32
                    / reservoir_state.len() as f32,
                max_activation: reservoir_state
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max),
            },

            // Q-values
            q_values: q_values.clone(),
            q_values_by_param: vec![
                q_values[0..BINS_PER_PARAM].to_vec(),
                q_values[BINS_PER_PARAM..2 * BINS_PER_PARAM].to_vec(),
                q_values[2 * BINS_PER_PARAM..3 * BINS_PER_PARAM].to_vec(),
                q_values[3 * BINS_PER_PARAM..4 * BINS_PER_PARAM].to_vec(),
            ],
            param_names: vec![
                "temperature".to_string(),
                "friction".to_string(),
                "spring_k".to_string(),
                "bias".to_string(),
            ],

            // Weights
            weight_heatmap,
            weight_stats,

            // Features
            feature_values: feature_importance,
            feature_names: vec![
                "rgyr".into(),
                "sasa".into(),
                "hbonds".into(),
                "contacts".into(),
                "rmsd".into(),
                "temp".into(),
                "friction".into(),
                "spring_k".into(),
                "bias".into(),
                "energy_pot".into(),
                "energy_kin".into(),
                "energy_tot".into(),
                "d_rgyr".into(),
                "d_sasa".into(),
                "d_hbonds".into(),
                "d_contacts".into(),
                "d_rmsd".into(),
                "d_energy".into(),
                "step_norm".into(),
                "episode_frac".into(),
                "stuck_frac".into(),
                "reward_ma".into(),
                "explore_bonus".into(),
            ],
        }
    }

    pub fn eval_mode(&mut self) {
        self.epsilon = 0.0;
    }

    pub fn train_mode(&mut self) {
        if self.epsilon < self.config.epsilon_min {
            self.epsilon = self.config.epsilon_min;
        }
    }

    // ========================================================================
    // PERSISTENCE
    // ========================================================================

    /// Save agent state to disk
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Save weights
        let weights_data = DendriticWeights {
            w_active: self.readout.w_active.clone(),
            w_target: self.readout.w_target.clone(),
            reservoir_size: self.readout.reservoir_size,
        };
        let weights_json = serde_json::to_string(&weights_data)?;
        std::fs::write(path, &weights_json)?;

        // Save metadata
        let meta_path = path.with_extension("meta.json");
        let metadata = serde_json::json!({
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "architecture": "dendritic_snn_rls",
            "reservoir_size": self.config.reservoir_size,
            "lambda": self.config.lambda,
            "tau": self.config.tau,
            "num_outputs": NUM_OUTPUTS,
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)?;

        info!(
            "💾 Dendritic Agent saved to {:?} (ε={:.4}, steps={})",
            path, self.epsilon, self.step_count
        );
        Ok(())
    }

    /// Load agent state from disk
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();

        // Load weights
        let weights_json = std::fs::read_to_string(path)?;
        let weights_data: DendriticWeights = serde_json::from_str(&weights_json)?;

        anyhow::ensure!(
            weights_data.reservoir_size == self.readout.reservoir_size,
            "Reservoir size mismatch: saved={}, current={}",
            weights_data.reservoir_size,
            self.readout.reservoir_size
        );

        self.readout.w_active = weights_data.w_active;
        self.readout.w_target = weights_data.w_target;

        // Load metadata
        let meta_path = path.with_extension("meta.json");
        if meta_path.exists() {
            let meta_content = std::fs::read_to_string(&meta_path)?;
            let metadata: serde_json::Value = serde_json::from_str(&meta_content)?;

            if let Some(eps) = metadata["epsilon"].as_f64() {
                self.epsilon = eps;
            }
            if let Some(steps) = metadata["step_count"].as_u64() {
                self.step_count = steps;
            }
        }

        // Reset precision matrices for fresh learning
        self.readout.reset_precision_matrices();

        info!(
            "📂 Dendritic Agent loaded from {:?} (ε={:.4}, steps={})",
            path, self.epsilon, self.step_count
        );
        Ok(())
    }

    /// Check if checkpoint exists
    pub fn checkpoint_exists<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists()
    }

    // ========================================================================
    // EPISODE MANAGEMENT
    // ========================================================================

    /// Reset reservoir state for new episode
    pub fn reset_episode(&mut self) -> Result<()> {
        self.reservoir.reset_state()?;
        self.last_state = None;
        Ok(())
    }

    /// Hard update target network (for periodic sync)
    pub fn update_target_network(&mut self) -> Result<()> {
        self.readout.hard_update_target();
        Ok(())
    }
}

/// Serializable weights structure
#[derive(Serialize, Deserialize)]
struct DendriticWeights {
    w_active: Vec<Vec<f32>>,
    w_target: Vec<Vec<f32>>,
    reservoir_size: usize,
}

/// Neural network state export for visualization dashboard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralStateExport {
    pub timestamp: String,
    pub architecture: String,
    pub reservoir_size: usize,
    pub num_outputs: usize,
    pub epsilon: f64,
    pub step_count: u64,
    pub learning_rate_multiplier: f32,

    /// Subsampled reservoir activations for heatmap (256 values)
    pub reservoir_heatmap: Vec<f32>,
    pub reservoir_stats: ReservoirStats,

    /// Q-values for all outputs
    pub q_values: Vec<f32>,
    /// Q-values grouped by parameter (4 groups of 5)
    pub q_values_by_param: Vec<Vec<f32>>,
    pub param_names: Vec<String>,

    /// Weight heatmap (20 heads × 64 subsampled weights)
    pub weight_heatmap: Vec<Vec<f32>>,
    pub weight_stats: Vec<WeightStats>,

    /// Input features
    pub feature_values: Vec<f32>,
    pub feature_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirStats {
    pub mean: f32,
    pub std: f32,
    pub sparsity: f32,
    pub max_activation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightStats {
    pub head_idx: usize,
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub l2_norm: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorized_action_compat() {
        // Ensure our FactorizedAction matches DQN's
        let action = FactorizedAction::new(2, 3, 1, 4);
        let flat = action.to_flat();
        let recovered = FactorizedAction::from_flat(flat);
        assert_eq!(action, recovered);
    }

    #[test]
    fn test_rls_readout_basic() {
        let readout = RLSReadout::new(64, 0.99, 0.005);
        let state = vec![0.1f32; 64];
        let q_values = readout.compute_q_values(&state, false);
        assert_eq!(q_values.len(), NUM_OUTPUTS);
    }
}
