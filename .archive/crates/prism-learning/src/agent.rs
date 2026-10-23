//! DQN Agent with Factorized Multi-Discrete Action Space
//!
//! Uses 4 separate heads for physics parameters instead of flat 625 actions:
//! - Temperature head (5 bins)
//! - Friction head (5 bins)
//! - Spring_k head (5 bins)
//! - Bias_strength head (5 bins)
//!
//! Total: 4 heads × 5 bins = 20 logits (vs 625 for flat)
//!
//! ## Benefits
//! - **Faster learning**: Each parameter learned independently
//! - **Better generalization**: Factorized structure captures parameter independence
//! - **VRAM efficient**: 20 output neurons vs 625

use anyhow::{Context, Result};
use log::{debug, info};
use std::path::Path;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

/// Number of bins per physics parameter
pub const BINS_PER_PARAM: i64 = 5;
/// Number of physics parameters (temp, friction, spring_k, bias)
pub const NUM_PARAMS: i64 = 4;
/// Total output logits (4 × 5 = 20)
pub const TOTAL_LOGITS: i64 = NUM_PARAMS * BINS_PER_PARAM;

/// Factorized action: indices for each physics parameter
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FactorizedAction {
    pub temp_idx: usize,
    pub friction_idx: usize,
    pub spring_idx: usize,
    pub bias_idx: usize,
}

impl FactorizedAction {
    /// Create from individual indices
    pub fn new(temp: usize, friction: usize, spring: usize, bias: usize) -> Self {
        Self {
            temp_idx: temp,
            friction_idx: friction,
            spring_idx: spring,
            bias_idx: bias,
        }
    }

    /// Convert to flat action index (for backward compatibility)
    pub fn to_flat(&self) -> usize {
        self.temp_idx * 125 + self.friction_idx * 25 + self.spring_idx * 5 + self.bias_idx
    }

    /// Create from flat action index
    pub fn from_flat(flat: usize) -> Self {
        Self {
            temp_idx: (flat / 125) % 5,
            friction_idx: (flat / 25) % 5,
            spring_idx: (flat / 5) % 5,
            bias_idx: flat % 5,
        }
    }

    /// Convert to array for tensor operations
    pub fn to_array(&self) -> [usize; 4] {
        [
            self.temp_idx,
            self.friction_idx,
            self.spring_idx,
            self.bias_idx,
        ]
    }
}

/// DQN Agent with Factorized Multi-Discrete Action Space
pub struct DQNAgent {
    /// Variable store for main network
    vs: nn::VarStore,
    /// Shared feature encoder
    encoder: nn::Sequential,
    /// 4 separate Q-heads (one per physics parameter)
    heads: Vec<nn::Linear>,
    /// Target network variable store
    target_vs: nn::VarStore,
    target_encoder: nn::Sequential,
    target_heads: Vec<nn::Linear>,
    /// Optimizer
    optimizer: nn::Optimizer,
    /// Device
    device: Device,
    /// Exploration rate
    epsilon: f64,
    epsilon_min: f64,
    epsilon_decay: f64,
    /// Training state
    step_count: u64,
    target_update_freq: u64,
    /// Dimensions
    input_dim: i64,
}

impl DQNAgent {
    /// Create new factorized multi-discrete DQN agent
    ///
    /// # Arguments
    /// * `input_dim` - Feature vector size (23 for target-aware)
    /// * `_output_dim` - Ignored (kept for API compatibility, we use 4×5=20)
    /// * `device_idx` - CUDA device index
    pub fn new(input_dim: i64, _output_dim: i64, device_idx: usize) -> Result<Self> {
        let device = Device::Cuda(device_idx);

        // Main network
        let vs = nn::VarStore::new(device);
        let (encoder, heads) = Self::build_network(&vs.root(), input_dim);

        // Target network
        let target_vs = nn::VarStore::new(device);
        let (target_encoder, target_heads) = Self::build_network(&target_vs.root(), input_dim);

        let optimizer = nn::Adam::default().build(&vs, 1e-4)?;

        info!("🧠 Factorized DQN Agent initialized on CUDA:{}", device_idx);
        info!("   Input dim: {}", input_dim);
        info!("   Action space: 4 heads × 5 bins = 20 logits (factorized)");
        info!("   Architecture: {}→128→64→[4×5 heads]", input_dim);

        Ok(Self {
            vs,
            encoder,
            heads,
            target_vs,
            target_encoder,
            target_heads,
            optimizer,
            device,
            epsilon: 1.0,
            epsilon_min: 0.05,
            epsilon_decay: 0.995,
            step_count: 0,
            target_update_freq: 100,
            input_dim,
        })
    }

    /// Build factorized network with shared encoder and separate heads
    fn build_network(root: &nn::Path, input_dim: i64) -> (nn::Sequential, Vec<nn::Linear>) {
        // Shared encoder
        let encoder = nn::seq()
            .add(nn::linear(
                root / "enc_l1",
                input_dim,
                128,
                Default::default(),
            ))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(root / "enc_l2", 128, 64, Default::default()))
            .add_fn(|xs| xs.relu());

        // 4 separate heads, one per physics parameter
        let heads = vec![
            nn::linear(root / "head_temp", 64, BINS_PER_PARAM, Default::default()),
            nn::linear(root / "head_fric", 64, BINS_PER_PARAM, Default::default()),
            nn::linear(root / "head_spring", 64, BINS_PER_PARAM, Default::default()),
            nn::linear(root / "head_bias", 64, BINS_PER_PARAM, Default::default()),
        ];

        (encoder, heads)
    }

    // ========================================================================
    // PERSISTENCE
    // ========================================================================

    /// Save model weights and metadata
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        self.vs
            .save(path)
            .with_context(|| format!("Failed to save model to {:?}", path))?;

        let meta_path = path.with_extension("meta.json");
        let metadata = serde_json::json!({
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "input_dim": self.input_dim,
            "architecture": "factorized_multi_discrete",
            "num_heads": NUM_PARAMS,
            "bins_per_head": BINS_PER_PARAM,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        });
        std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)
            .with_context(|| format!("Failed to save metadata to {:?}", meta_path))?;

        info!(
            "💾 Model saved to {:?} (ε={:.4}, steps={})",
            path, self.epsilon, self.step_count
        );
        Ok(())
    }

    /// Load model weights and metadata
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        self.vs
            .load(path)
            .with_context(|| format!("Failed to load model from {:?}", path))?;

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

        self.update_target_network()?;
        info!(
            "📂 Model loaded from {:?} (ε={:.4}, steps={})",
            path, self.epsilon, self.step_count
        );
        Ok(())
    }

    /// Check if checkpoint exists
    pub fn checkpoint_exists<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists()
    }

    // ========================================================================
    // ACTION SELECTION (Factorized)
    // ========================================================================

    /// Select factorized action using epsilon-greedy policy
    pub fn select_action(&self, state: &[f32]) -> usize {
        let action = self.select_factorized_action(state);
        action.to_flat()
    }

    /// Select factorized action (returns structured action)
    pub fn select_factorized_action(&self, state: &[f32]) -> FactorizedAction {
        // Epsilon-greedy exploration
        if rand::random::<f64>() < self.epsilon {
            return FactorizedAction::new(
                rand::random::<usize>() % BINS_PER_PARAM as usize,
                rand::random::<usize>() % BINS_PER_PARAM as usize,
                rand::random::<usize>() % BINS_PER_PARAM as usize,
                rand::random::<usize>() % BINS_PER_PARAM as usize,
            );
        }

        // Exploit: select best action from each head independently
        let state_tensor = Tensor::from_slice(state)
            .to_kind(Kind::Float)
            .to(self.device)
            .unsqueeze(0);

        tch::no_grad(|| {
            let features = self.encoder.forward(&state_tensor);

            let temp_idx = self.heads[0]
                .forward(&features)
                .argmax(1, false)
                .int64_value(&[0]) as usize;
            let fric_idx = self.heads[1]
                .forward(&features)
                .argmax(1, false)
                .int64_value(&[0]) as usize;
            let spring_idx = self.heads[2]
                .forward(&features)
                .argmax(1, false)
                .int64_value(&[0]) as usize;
            let bias_idx = self.heads[3]
                .forward(&features)
                .argmax(1, false)
                .int64_value(&[0]) as usize;

            FactorizedAction::new(temp_idx, fric_idx, spring_idx, bias_idx)
        })
    }

    /// Select action without exploration (greedy)
    pub fn select_action_greedy(&self, state: &[f32]) -> usize {
        let old_epsilon = self.epsilon;
        // Temporarily set epsilon to 0 for greedy selection
        let action = tch::no_grad(|| {
            let state_tensor = Tensor::from_slice(state)
                .to_kind(Kind::Float)
                .to(self.device)
                .unsqueeze(0);

            let features = self.encoder.forward(&state_tensor);

            let temp_idx = self.heads[0]
                .forward(&features)
                .argmax(1, false)
                .int64_value(&[0]) as usize;
            let fric_idx = self.heads[1]
                .forward(&features)
                .argmax(1, false)
                .int64_value(&[0]) as usize;
            let spring_idx = self.heads[2]
                .forward(&features)
                .argmax(1, false)
                .int64_value(&[0]) as usize;
            let bias_idx = self.heads[3]
                .forward(&features)
                .argmax(1, false)
                .int64_value(&[0]) as usize;

            FactorizedAction::new(temp_idx, fric_idx, spring_idx, bias_idx)
        });

        action.to_flat()
    }

    // ========================================================================
    // TRAINING (Factorized Loss)
    // ========================================================================

    /// Train on batch with factorized loss (sum of per-head losses)
    pub fn train(&mut self, batch: Vec<(Vec<f32>, usize, f32, Vec<f32>, bool)>) -> Result<f32> {
        if batch.is_empty() {
            return Ok(0.0);
        }

        let batch_size = batch.len() as i64;

        // Prepare batch data
        let states: Vec<f32> = batch.iter().flat_map(|x| x.0.clone()).collect();
        let actions: Vec<FactorizedAction> = batch
            .iter()
            .map(|x| FactorizedAction::from_flat(x.1))
            .collect();
        let rewards: Vec<f32> = batch.iter().map(|x| x.2).collect();
        let next_states: Vec<f32> = batch.iter().flat_map(|x| x.3.clone()).collect();
        let dones: Vec<f32> = batch.iter().map(|x| if x.4 { 0.0 } else { 1.0 }).collect();

        // Convert to tensors
        let b_states = Tensor::from_slice(&states)
            .to_kind(Kind::Float)
            .to(self.device)
            .view([batch_size, self.input_dim]);
        let b_next_states = Tensor::from_slice(&next_states)
            .to_kind(Kind::Float)
            .to(self.device)
            .view([batch_size, self.input_dim]);
        let b_rewards = Tensor::from_slice(&rewards)
            .to_kind(Kind::Float)
            .to(self.device);
        let b_dones = Tensor::from_slice(&dones)
            .to_kind(Kind::Float)
            .to(self.device);

        // Extract action indices for each head
        let temp_actions: Vec<i64> = actions.iter().map(|a| a.temp_idx as i64).collect();
        let fric_actions: Vec<i64> = actions.iter().map(|a| a.friction_idx as i64).collect();
        let spring_actions: Vec<i64> = actions.iter().map(|a| a.spring_idx as i64).collect();
        let bias_actions: Vec<i64> = actions.iter().map(|a| a.bias_idx as i64).collect();

        let b_temp_actions = Tensor::from_slice(&temp_actions)
            .to(self.device)
            .unsqueeze(1);
        let b_fric_actions = Tensor::from_slice(&fric_actions)
            .to(self.device)
            .unsqueeze(1);
        let b_spring_actions = Tensor::from_slice(&spring_actions)
            .to(self.device)
            .unsqueeze(1);
        let b_bias_actions = Tensor::from_slice(&bias_actions)
            .to(self.device)
            .unsqueeze(1);

        // Forward pass through encoder
        let features = self.encoder.forward(&b_states);

        // Q-values for each head (selected actions)
        let q_temp = self.heads[0]
            .forward(&features)
            .gather(1, &b_temp_actions, false)
            .squeeze_dim(1);
        let q_fric = self.heads[1]
            .forward(&features)
            .gather(1, &b_fric_actions, false)
            .squeeze_dim(1);
        let q_spring = self.heads[2]
            .forward(&features)
            .gather(1, &b_spring_actions, false)
            .squeeze_dim(1);
        let q_bias = self.heads[3]
            .forward(&features)
            .gather(1, &b_bias_actions, false)
            .squeeze_dim(1);

        // Target Q-values (max over each head independently)
        let (target_q_temp, target_q_fric, target_q_spring, target_q_bias) = tch::no_grad(|| {
            let next_features = self.target_encoder.forward(&b_next_states);

            let t_temp = self.target_heads[0]
                .forward(&next_features)
                .max_dim(1, false)
                .0;
            let t_fric = self.target_heads[1]
                .forward(&next_features)
                .max_dim(1, false)
                .0;
            let t_spring = self.target_heads[2]
                .forward(&next_features)
                .max_dim(1, false)
                .0;
            let t_bias = self.target_heads[3]
                .forward(&next_features)
                .max_dim(1, false)
                .0;

            (t_temp, t_fric, t_spring, t_bias)
        });

        // TD targets: r + γ * max_Q * (1 - done)
        // Distribute reward equally across heads (use full reward for each head)
        let gamma = 0.99;
        let target_temp: Tensor = &b_rewards + gamma * &target_q_temp * &b_dones;
        let target_fric: Tensor = &b_rewards + gamma * &target_q_fric * &b_dones;
        let target_spring: Tensor = &b_rewards + gamma * &target_q_spring * &b_dones;
        let target_bias: Tensor = &b_rewards + gamma * &target_q_bias * &b_dones;

        // Compute loss for each head (MSE between Q-values and TD targets)
        let loss_temp = q_temp.mse_loss(&target_temp.detach(), tch::Reduction::Mean);
        let loss_fric = q_fric.mse_loss(&target_fric.detach(), tch::Reduction::Mean);
        let loss_spring = q_spring.mse_loss(&target_spring.detach(), tch::Reduction::Mean);
        let loss_bias = q_bias.mse_loss(&target_bias.detach(), tch::Reduction::Mean);

        // Total loss = sum of head losses
        let total_loss = &loss_temp + &loss_fric + &loss_spring + &loss_bias;
        let loss_value = f32::try_from(&total_loss).unwrap_or(0.0);

        // Backprop
        self.optimizer.backward_step(&total_loss);

        // Update state
        self.step_count += 1;
        self.decay_epsilon();

        // Periodic target update
        if self.step_count % self.target_update_freq == 0 {
            self.update_target_network()?;
            debug!("Target network updated at step {}", self.step_count);
        }

        Ok(loss_value)
    }

    /// Update target network (hard copy)
    pub fn update_target_network(&mut self) -> Result<()> {
        self.target_vs.copy(&self.vs)?;
        Ok(())
    }

    // ========================================================================
    // EPSILON MANAGEMENT
    // ========================================================================

    pub fn decay_epsilon(&mut self) {
        if self.epsilon > self.epsilon_min {
            self.epsilon *= self.epsilon_decay;
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

    pub fn eval_mode(&mut self) {
        self.epsilon = 0.0;
    }

    pub fn train_mode(&mut self) {
        if self.epsilon < self.epsilon_min {
            self.epsilon = self.epsilon_min;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorized_action_roundtrip() {
        let action = FactorizedAction::new(2, 3, 1, 4);
        let flat = action.to_flat();
        let recovered = FactorizedAction::from_flat(flat);
        assert_eq!(action, recovered);
    }

    #[test]
    fn test_factorized_action_range() {
        for t in 0..5 {
            for f in 0..5 {
                for s in 0..5 {
                    for b in 0..5 {
                        let action = FactorizedAction::new(t, f, s, b);
                        let flat = action.to_flat();
                        assert!(flat < 625);
                        let recovered = FactorizedAction::from_flat(flat);
                        assert_eq!(action, recovered);
                    }
                }
            }
        }
    }
}
