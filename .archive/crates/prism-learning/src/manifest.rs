//! Dataset manifest system with JSON-configurable reward tuning
//!
//! This module connects JSON Configuration (Manifest) directly to the Math (Rewards),
//! allowing you to tune the "Reward Function" from the JSON file without recompiling.
//!
//! ## Key Components
//! - `RewardWeighting`: Exposure/RMSD trade-offs
//! - `PhysicsParameterRanges`: RL action space bounds
//! - `MacroStepConfig`: Chunked training configuration
//! - `FeatureConfig`: Target-aware feature extraction settings

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

// ============================================================================
// CORE MANIFEST STRUCTURE
// ============================================================================

/// Complete training dataset manifest with reward tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationManifest {
    pub dataset_name: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub description: String,
    pub targets: Vec<ProteinTarget>,
    pub training_parameters: TrainingParameters,
    pub physics_parameter_ranges: PhysicsParameterRanges,
    #[serde(default)]
    pub macro_step_config: MacroStepConfig,
    #[serde(default)]
    pub feature_config: FeatureConfig,
}

/// Individual protein target for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProteinTarget {
    pub name: String,
    #[serde(default)]
    pub family: String,
    #[serde(default)]
    pub description: String,
    pub apo_pdb: String,
    #[serde(default)]
    pub holo_pdb: String,
    pub target_residues: Vec<usize>,
    #[serde(default)]
    pub core_residues: Vec<usize>,
    #[serde(default)]
    pub difficulty: String,
    #[serde(default = "default_expected_sasa")]
    pub expected_sasa_gain: f32,
    #[serde(default)]
    pub is_multimer: bool,
    #[serde(default)]
    pub has_glycans: bool,

    // ========== v3.1.1 META-LEARNING ENHANCEMENTS ==========
    /// Residues with attached glycans (for glycan shield tracking)
    /// When these move away from target_residues, it indicates shield opening
    #[serde(default)]
    pub glycan_residues: Vec<usize>,

    /// Site value multiplier (0.1 = low priority, 1.0 = standard, 5.0 = high-value epitope)
    /// Used to weight rewards for known important sites
    #[serde(default = "default_site_value")]
    pub site_value: f32,

    /// Adjacent residues to primary target (secondary priority)
    /// Exposure of these indicates partial opening / approach to full exposure
    #[serde(default)]
    pub adjacent_residues: Vec<usize>,

    /// Cryptic site mechanism type for mechanism-aware learning
    /// Options: "glycan_shield", "allosteric", "induced_fit", "flap", "loop", "unknown"
    #[serde(default = "default_mechanism")]
    pub mechanism: String,

    /// Known binding partners or antibodies (for context)
    #[serde(default)]
    pub known_binders: Vec<String>,
}

fn default_expected_sasa() -> f32 {
    100.0
}
fn default_site_value() -> f32 {
    1.0
}
fn default_mechanism() -> String {
    "unknown".to_string()
}

// ============================================================================
// TRAINING PARAMETERS (DQN + Reward Weights)
// ============================================================================

/// Training hyperparameters including reward function weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParameters {
    #[serde(default = "default_lr")]
    pub learning_rate: f32,
    #[serde(default = "default_batch")]
    pub batch_size: usize,
    #[serde(default = "default_gamma")]
    pub gamma: f32,
    #[serde(default = "default_eps_start")]
    pub epsilon_start: f32,
    #[serde(default = "default_eps_end")]
    pub epsilon_end: f32,
    #[serde(default = "default_eps_decay")]
    pub epsilon_decay: f32,
    pub reward_weighting: RewardWeighting,
}

fn default_lr() -> f32 {
    1e-4
}
fn default_batch() -> usize {
    32
}
fn default_gamma() -> f32 {
    0.99
}
fn default_eps_start() -> f32 {
    1.0
}
fn default_eps_end() -> f32 {
    0.05
}
fn default_eps_decay() -> f32 {
    0.995
}

impl Default for TrainingParameters {
    fn default() -> Self {
        Self {
            learning_rate: default_lr(),
            batch_size: default_batch(),
            gamma: default_gamma(),
            epsilon_start: default_eps_start(),
            epsilon_end: default_eps_end(),
            epsilon_decay: default_eps_decay(),
            reward_weighting: RewardWeighting::default(),
        }
    }
}

// ============================================================================
// REWARD WEIGHTING (The Core of the Scoring Function)
// ============================================================================

/// Reward function weights - configurable via JSON
///
/// The final reward is:
/// ```text
/// reward = (exposure_gain * exposure_weight) - stability_penalty
///
/// where stability_penalty = max(0, core_rmsd - stability_threshold) * rmsd_weight
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardWeighting {
    /// Multiplier for Exposure Gain (The "Carrot")
    /// Higher = more reward for opening cryptic sites
    #[serde(default = "default_exposure_weight")]
    pub exposure_weight: f32,

    /// Multiplier for Core RMSD penalty (The "Stick")
    /// Higher = more penalty for destabilizing the protein core
    #[serde(default = "default_rmsd_weight")]
    pub rmsd_weight: f32,

    /// RMSD threshold (Angstroms) before penalty kicks in
    /// Allows small movements without penalty
    #[serde(default = "default_stability_threshold")]
    pub stability_threshold: f32,

    /// Bonus multiplier for achieving target exposure
    #[serde(default = "default_target_bonus")]
    pub target_bonus: f32,

    /// Penalty for atomic clashes (atoms < clash_distance apart)
    #[serde(default = "default_clash_penalty")]
    pub clash_penalty: f32,

    /// Distance threshold for clash detection (Angstroms)
    #[serde(default = "default_clash_distance")]
    pub clash_distance: f32,

    // ========== v3.1.1 META-LEARNING REWARD ENHANCEMENTS ==========
    /// Bonus multiplier for glycan-correlated discovery
    /// Rewards exposure that correlates with glycan shield displacement
    /// This teaches the agent that "moving glycan = biologically meaningful opening"
    #[serde(default = "default_glycan_discovery_bonus")]
    pub glycan_discovery_bonus: f32,

    /// Minimum glycan displacement (Angstroms) to qualify for discovery bonus
    #[serde(default = "default_glycan_displacement_threshold")]
    pub glycan_displacement_threshold: f32,

    /// Weight multiplier applied from target's site_value field
    /// Final reward *= site_value, so high-value sites get more reward
    #[serde(default = "default_site_value_weight")]
    pub site_value_weight: f32,

    /// Bonus for exposing adjacent residues (partial discovery)
    /// Smaller than main exposure but indicates progress
    #[serde(default = "default_adjacent_bonus")]
    pub adjacent_bonus: f32,
}

fn default_exposure_weight() -> f32 {
    10.0
}
fn default_rmsd_weight() -> f32 {
    5.0
}
fn default_stability_threshold() -> f32 {
    2.5
}
fn default_target_bonus() -> f32 {
    50.0
}
fn default_clash_penalty() -> f32 {
    1.0
}
fn default_clash_distance() -> f32 {
    1.5
}
fn default_glycan_discovery_bonus() -> f32 {
    25.0
}
fn default_glycan_displacement_threshold() -> f32 {
    3.0
}
fn default_site_value_weight() -> f32 {
    1.0
}
fn default_adjacent_bonus() -> f32 {
    2.0
}

impl Default for RewardWeighting {
    fn default() -> Self {
        Self {
            exposure_weight: default_exposure_weight(),
            rmsd_weight: default_rmsd_weight(),
            stability_threshold: default_stability_threshold(),
            target_bonus: default_target_bonus(),
            clash_penalty: default_clash_penalty(),
            clash_distance: default_clash_distance(),
            // v3.1.1 meta-learning defaults
            glycan_discovery_bonus: default_glycan_discovery_bonus(),
            glycan_displacement_threshold: default_glycan_displacement_threshold(),
            site_value_weight: default_site_value_weight(),
            adjacent_bonus: default_adjacent_bonus(),
        }
    }
}

// ============================================================================
// PHYSICS PARAMETER RANGES (RL Action Space)
// ============================================================================

/// Physics parameter search ranges for the RL action space (4D factorized)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsParameterRanges {
    /// [min, max] for Temperature (Energy injection)
    #[serde(default = "default_temp_range")]
    pub temperature: [f32; 2],

    /// [min, max] for Friction (Damping)
    #[serde(default = "default_friction_range")]
    pub friction: [f32; 2],

    /// [min, max] for Spring Constant (Rigidity)
    #[serde(default = "default_spring_range")]
    pub spring_k: [f32; 2],

    /// [min, max] for Bias Strength (Target residue pull force)
    #[serde(default = "default_bias_range")]
    pub bias_strength: [f32; 2],
}

fn default_temp_range() -> [f32; 2] {
    [0.1, 5.0]
}
fn default_friction_range() -> [f32; 2] {
    [0.01, 1.0]
}
fn default_spring_range() -> [f32; 2] {
    [0.1, 10.0]
}
fn default_bias_range() -> [f32; 2] {
    [0.0, 2.0]
}

impl Default for PhysicsParameterRanges {
    fn default() -> Self {
        Self {
            temperature: default_temp_range(),
            friction: default_friction_range(),
            spring_k: default_spring_range(),
            bias_strength: default_bias_range(),
        }
    }
}

/// Continuous physics parameters (output from action decoding)
#[derive(Debug, Clone, Copy)]
pub struct PhysicsParams {
    pub temperature: f32,
    pub friction: f32,
    pub spring_k: f32,
    pub bias_strength: f32,
}

impl PhysicsParameterRanges {
    /// Number of bins per parameter
    pub const BINS: usize = 5;

    /// Convert factorized action indices to continuous physics parameters
    ///
    /// # Arguments
    /// * `temp_idx` - Temperature bin [0-4]
    /// * `fric_idx` - Friction bin [0-4]
    /// * `spring_idx` - Spring constant bin [0-4]
    /// * `bias_idx` - Bias strength bin [0-4]
    pub fn indices_to_params(
        &self,
        temp_idx: usize,
        fric_idx: usize,
        spring_idx: usize,
        bias_idx: usize,
    ) -> PhysicsParams {
        let bins = Self::BINS as f32;

        PhysicsParams {
            temperature: self.temperature[0]
                + (temp_idx as f32 / (bins - 1.0)) * (self.temperature[1] - self.temperature[0]),
            friction: self.friction[0]
                + (fric_idx as f32 / (bins - 1.0)) * (self.friction[1] - self.friction[0]),
            spring_k: self.spring_k[0]
                + (spring_idx as f32 / (bins - 1.0)) * (self.spring_k[1] - self.spring_k[0]),
            bias_strength: self.bias_strength[0]
                + (bias_idx as f32 / (bins - 1.0))
                    * (self.bias_strength[1] - self.bias_strength[0]),
        }
    }

    /// Convert flat action index to continuous physics parameters (backward compatible)
    /// Flat action space: 5^4 = 625 actions
    pub fn action_to_params(&self, action: usize) -> (f32, f32, f32, f32) {
        let temp_idx = (action / 125) % 5;
        let fric_idx = (action / 25) % 5;
        let spring_idx = (action / 5) % 5;
        let bias_idx = action % 5;

        let params = self.indices_to_params(temp_idx, fric_idx, spring_idx, bias_idx);
        (
            params.temperature,
            params.friction,
            params.spring_k,
            params.bias_strength,
        )
    }

    /// Total number of discrete actions (5^4 = 625, but factorized = 4×5 = 20 logits)
    pub fn action_space_size(&self) -> usize {
        625 // Flat size (for backward compatibility)
    }

    /// Number of logits in factorized representation
    pub fn factorized_logits(&self) -> usize {
        4 * Self::BINS // 20
    }

    /// Get parameter names for logging
    pub fn param_names() -> [&'static str; 4] {
        ["temperature", "friction", "spring_k", "bias_strength"]
    }
}

// ============================================================================
// MACRO-STEP CONFIGURATION (Chunked Training)
// ============================================================================

/// Configuration for macro-step training
///
/// Instead of running 1M steps and getting one transition, run in chunks
/// and collect transitions at each macro-step boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroStepConfig {
    /// Number of macro-steps per episode
    #[serde(default = "default_macro_steps")]
    pub num_macro_steps: usize,

    /// MD steps per macro-step (total = num_macro_steps × steps_per_macro)
    #[serde(default = "default_steps_per_macro")]
    pub steps_per_macro: u64,

    /// Whether to allow action changes between macro-steps
    #[serde(default = "default_allow_action_change")]
    pub allow_action_change: bool,

    /// Discount factor between macro-steps (for multi-step returns)
    #[serde(default = "default_macro_gamma")]
    pub macro_gamma: f32,
}

fn default_macro_steps() -> usize {
    10
}
fn default_steps_per_macro() -> u64 {
    100_000
}
fn default_allow_action_change() -> bool {
    true
}
fn default_macro_gamma() -> f32 {
    0.95
}

impl Default for MacroStepConfig {
    fn default() -> Self {
        Self {
            num_macro_steps: default_macro_steps(),
            steps_per_macro: default_steps_per_macro(),
            allow_action_change: default_allow_action_change(),
            macro_gamma: default_macro_gamma(),
        }
    }
}

// ============================================================================
// FEATURE CONFIGURATION (Target-Aware Extraction)
// ============================================================================

/// Configuration for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Cutoff radius for neighbor counting (Angstroms)
    #[serde(default = "default_neighbor_cutoff")]
    pub neighbor_cutoff: f32,

    /// Cutoff radius for contact detection (Angstroms)
    #[serde(default = "default_contact_cutoff")]
    pub contact_cutoff: f32,

    /// Include global features (Size, Rg, Density)
    #[serde(default = "default_true")]
    pub include_global: bool,

    /// Include target-residue neighborhood features
    #[serde(default = "default_true")]
    pub include_target_neighborhood: bool,

    /// Include stability proxy features
    #[serde(default = "default_true")]
    pub include_stability: bool,

    /// Include family conditioning flags
    #[serde(default = "default_true")]
    pub include_family_flags: bool,

    /// Include temporal features (change from initial)
    #[serde(default = "default_true")]
    pub include_temporal: bool,

    /// Include difficulty encoding (4 one-hot dims: easy, medium, hard, expert)
    /// NEW in v3.1.1 - Enables difficulty-aware learning strategies
    /// Default: false for backward compatibility with existing models
    #[serde(default = "default_false")]
    pub include_difficulty: bool,

    /// Include glycan-aware features (4 dims: proximity, displacement, coverage, correlation)
    /// NEW in v3.1.1 - For glycan shield dynamics awareness
    /// Teaches agent that glycan displacement reveals cryptic sites
    #[serde(default = "default_false")]
    pub include_glycan_awareness: bool,

    /// Include mechanism encoding (6 one-hot: glycan_shield, allosteric, induced_fit, flap, loop, unknown)
    /// NEW in v3.1.1 - For mechanism-specific learning strategies
    #[serde(default = "default_false")]
    pub include_mechanism: bool,

    // ========== v3.1.1 BIO-CHEMISTRY FEATURES ==========
    /// Include bio-chemistry features (3 dims: hydrophobic exposure, anisotropy, frustration)
    /// NEW in v3.1.1 - Adds chemical intelligence to geometry
    /// - Hydrophobic ΔΔ SASA: Exposure of "greasy" residues (drug targets love these)
    /// - Local Anisotropy: Hinge detection (where the door opens)
    /// - Electrostatic Frustration: Spring-loaded regions wanting to pop open
    #[serde(default = "default_false")]
    pub include_bio_chemistry: bool,

    // ========== GPU ACCELERATION ==========
    /// Use GPU-accelerated feature extraction for bio-chemistry features
    /// NEW in v3.1.1 - Unlocks ~40% more GPU utilization by moving
    /// feature extraction loops from CPU to CUDA kernels.
    /// Falls back to CPU if GPU initialization fails.
    /// Requires include_bio_chemistry = true to have effect.
    #[serde(default = "default_false")]
    pub use_gpu_features: bool,
}

fn default_neighbor_cutoff() -> f32 {
    8.0
}
fn default_contact_cutoff() -> f32 {
    6.0
}
fn default_true() -> bool {
    true
}
fn default_false() -> bool {
    false
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            neighbor_cutoff: default_neighbor_cutoff(),
            contact_cutoff: default_contact_cutoff(),
            include_global: true,
            include_target_neighborhood: true,
            include_stability: true,
            include_family_flags: true,
            include_temporal: true,
            include_difficulty: false,       // Backward compatible
            include_glycan_awareness: false, // Backward compatible
            include_mechanism: false,        // Backward compatible
            include_bio_chemistry: false,    // Backward compatible
            use_gpu_features: false,         // Opt-in for GPU acceleration
        }
    }
}

impl FeatureConfig {
    /// Calculate total feature dimension based on enabled features
    ///
    /// Base dimensions: 23 (standard)
    /// With difficulty: +4 = 27
    /// With glycan awareness: +4 = 31
    /// With mechanism: +6 = 37
    /// With bio-chemistry: +3 = 40 (full atomic-aware)
    pub fn feature_dim(&self) -> usize {
        let mut dim = 0;
        if self.include_global {
            dim += 3;
        } // Size, Rg, Density
        if self.include_target_neighborhood {
            dim += 8;
        } // Target-aware features
        if self.include_stability {
            dim += 4;
        } // RMSD, clashes, displacement
        if self.include_family_flags {
            dim += 4;
        } // Family conditioning
        if self.include_temporal {
            dim += 4;
        } // Change features
        if self.include_difficulty {
            dim += 4;
        } // Difficulty one-hot
        if self.include_glycan_awareness {
            dim += 4;
        } // Glycan shield dynamics
        if self.include_mechanism {
            dim += 6;
        } // Mechanism one-hot
        if self.include_bio_chemistry {
            dim += 3;
        } // Hydrophobic, anisotropy, frustration
        dim
    }
}

// ============================================================================
// MANIFEST IMPLEMENTATION
// ============================================================================

impl CalibrationManifest {
    /// Load manifest from JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read manifest file: {}", path.display()))?;

        let manifest: CalibrationManifest = serde_json::from_str(&contents)
            .with_context(|| format!("Failed to parse manifest JSON: {}", path.display()))?;

        log::info!(
            "Loaded manifest '{}' with {} targets",
            manifest.dataset_name,
            manifest.targets.len()
        );
        log::info!(
            "  Reward weights: exposure={}, rmsd={}, threshold={}Å",
            manifest
                .training_parameters
                .reward_weighting
                .exposure_weight,
            manifest.training_parameters.reward_weighting.rmsd_weight,
            manifest
                .training_parameters
                .reward_weighting
                .stability_threshold
        );
        log::info!(
            "  Macro-steps: {} × {} steps",
            manifest.macro_step_config.num_macro_steps,
            manifest.macro_step_config.steps_per_macro
        );
        log::info!("  Feature dim: {}", manifest.feature_config.feature_dim());

        Ok(manifest)
    }

    /// Save manifest to JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        let contents =
            serde_json::to_string_pretty(self).context("Failed to serialize manifest")?;

        fs::write(path, contents)
            .with_context(|| format!("Failed to write manifest: {}", path.display()))?;

        Ok(())
    }

    /// Validate that all PDB files exist
    pub fn validate_files(&self) -> Result<()> {
        for target in &self.targets {
            if !Path::new(&target.apo_pdb).exists() {
                anyhow::bail!("Apo PDB file not found: {}", target.apo_pdb);
            }
        }
        log::info!(
            "All PDB files validated for manifest '{}'",
            self.dataset_name
        );
        Ok(())
    }

    /// Get total MD steps per episode
    pub fn total_steps_per_episode(&self) -> u64 {
        self.macro_step_config.num_macro_steps as u64 * self.macro_step_config.steps_per_macro
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_to_params() {
        let ranges = PhysicsParameterRanges::default();

        // Action 0 = min all (4D factorized)
        let (t, f, s, b) = ranges.action_to_params(0);
        assert!((t - 0.1).abs() < 0.01);
        assert!((f - 0.01).abs() < 0.01);
        assert!((s - 0.1).abs() < 0.01);
        assert!((b - 0.0).abs() < 0.01);

        // Action 624 = max all (5^4 - 1)
        let (t, f, s, b) = ranges.action_to_params(624);
        assert!((t - 5.0).abs() < 0.01);
        assert!((f - 1.0).abs() < 0.01);
        assert!((s - 10.0).abs() < 0.01);
        assert!((b - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_feature_dim() {
        let config = FeatureConfig::default();
        assert_eq!(config.feature_dim(), 23); // 3+8+4+4+4
    }

    #[test]
    fn test_feature_dim_full_bio_chemistry() {
        // Test with all features enabled including bio-chemistry
        let config = FeatureConfig {
            include_global: true,
            include_target_neighborhood: true,
            include_stability: true,
            include_family_flags: true,
            include_temporal: true,
            include_difficulty: true,
            include_glycan_awareness: true,
            include_mechanism: true,
            include_bio_chemistry: true,
            ..Default::default()
        };
        // 3 + 8 + 4 + 4 + 4 + 4 + 4 + 6 + 3 = 40
        assert_eq!(config.feature_dim(), 40);
    }

    #[test]
    fn test_default_reward_weights() {
        let weights = RewardWeighting::default();
        assert_eq!(weights.exposure_weight, 10.0);
        assert_eq!(weights.rmsd_weight, 5.0);
        assert_eq!(weights.stability_threshold, 2.5);
    }
}
