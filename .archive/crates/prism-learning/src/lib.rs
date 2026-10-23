//! PRISM-Zero v3.1: Self-calibrating engine with macro-step training
//!
//! This crate implements a reinforcement learning system that automatically tunes
//! molecular dynamics parameters for cryptic site prediction. The system uses:
//!
//! - **JSON-Configurable Rewards**: Tune the reward function from JSON without recompiling
//! - **Macro-Step Training**: Collect multiple transitions per episode (10x more signal)
//! - **Target-Aware Features**: 23-dimensional feature vector with semantic meaning
//! - **Intrinsic Reward System**: Physics-based evaluation without ground truth
//!
//! ## Architecture Overview
//!
//! ```text
//! Manifest (JSON) → Feature Extractor → Agent → Physics → Weighted Reward
//!     ↓                   ↓               ↓        ↓            ↓
//! Config/Weights     23-dim vector    Action    MD sim    JSON weights
//!
//! Agent Backends:
//!   - DendriticAgent (default): GPU SNN Reservoir + RLS Readout (no PyTorch!)
//!   - DQNAgent (optional): PyTorch MLP (requires `rl` feature)
//! ```
//!
//! ## Key Innovations
//!
//! 1. **Macro-Step Training** (trainer.rs:232):
//!    Instead of running 1M steps → 1 transition, we chunk into 10 × 100K steps
//!    and collect 10 transitions. This dramatically improves sample efficiency.
//!
//! 2. **Target-Aware Features** (features.rs):
//!    The agent "sees" proteins through semantically meaningful features:
//!    - Global: Size, Rg, Density
//!    - Target Neighborhood: Exposure, burial, contacts
//!    - Stability: RMSD proxies, clash counts
//!    - Family Flags: Protein family one-hot
//!    - Temporal: Changes from initial state
//!
//! 3. **JSON → Math Connection** (manifest.rs + rewards.rs):
//!    Reward weights flow directly from JSON config into reward calculation,
//!    allowing hyperparameter tuning without recompilation.
//!
//! ## Module Structure
//!
//! - [`manifest`]: JSON-based dataset and reward configuration
//! - [`features`]: Target-aware feature extraction (23 dimensions)
//! - [`buffers`]: Float4 SIMD-aligned memory layouts
//! - [`rewards`]: Intrinsic reward computation with configurable weights
//! - [`trainer`]: Macro-step parallel training pipeline
//! - [`agent`]: DQN agent for physics parameter selection (PyTorch, optional)
//! - [`dendritic_agent`]: Neuromorphic agent with SNN reservoir (default)

pub mod atomic_chemistry;
pub mod buffers;
pub mod features;
pub mod manifest;
pub mod persistence;
pub mod rewards;
pub mod trainer;

// Dendritic Agent (default - no PyTorch dependency)
pub mod dendritic_agent;

// DQN Agent (optional - requires PyTorch/libtorch)
#[cfg(feature = "rl")]
pub mod agent;

// Re-exports for convenience
pub use buffers::SimulationBuffers;
pub use features::{FeatureExtractor, FeatureVector};
pub use manifest::{
    CalibrationManifest, FeatureConfig, MacroStepConfig, PhysicsParameterRanges, ProteinTarget,
    RewardWeighting, TrainingParameters,
};
pub use rewards::{
    calculate_macro_step_reward, evaluate_simulation, evaluate_simulation_weighted,
    EvaluationResult, RewardBreakdown,
};
pub use trainer::{PrismTrainer, TrainingConfig, TrainingSession, Transition};

// Dendritic Agent exports (default)
pub use dendritic_agent::{
    DendriticAgent, DendriticAgentConfig, FactorizedAction, NeuralStateExport, ReservoirStats,
    WeightStats,
};

// DQN Agent exports (optional)
#[cfg(feature = "rl")]
pub use agent::DQNAgent;

/// Result type for PRISM-Learning operations
pub type Result<T> = std::result::Result<T, anyhow::Error>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PRISM_ZERO_VERSION: &str = "3.1";

/// Feature dimension for target-aware extraction
/// Breakdown: Global(3) + Target(8) + Stability(4) + Family(4) + Temporal(4) = 23
pub const FEATURE_DIM: usize = 23;

/// Action dimension for physics parameters (5×5×5 = 125)
pub const ACTION_DIM: usize = 125;

/// Default macro-steps per episode
pub const DEFAULT_MACRO_STEPS: usize = 10;

/// Default MD steps per macro-step
pub const DEFAULT_STEPS_PER_MACRO: u64 = 100_000;
