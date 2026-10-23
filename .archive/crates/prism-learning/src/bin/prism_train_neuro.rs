//! PRISM-Zero v3.1 Neuromorphic Training Binary
//!
//! Uses the Flashbulb Reservoir (E/I balanced SNN) instead of DQN.
//! NO PYTORCH REQUIRED - Pure Rust + CUDA.
//!
//! Architecture:
//! - 80/20 Excitatory/Inhibitory balance
//! - Adaptive time constants (5-50ms, matching protein dynamics)
//! - Reward-modulated RLS learning (Flashbulb plasticity)
//! - Velocity features for temporal dynamics (23→46 dim expansion)
//!
//! Recommended Config (vs DQN):
//! - DQN: 10 chunks × 100K steps = 1M total
//! - Dendritic: 100 chunks × 10K steps = 1M total (more learning signal)
//!
//! ## Human-in-the-Loop (HIL) Live Control
//!
//! The trainer watches `{output}/hil_control.json` for runtime commands:
//! - `spike_exploration`: Temporarily boost epsilon to 0.8
//! - `save_checkpoint`: Force immediate checkpoint save
//! - `set_epsilon`: Set epsilon to custom value (0.0-1.0)
//! - `pause`: Pause after current episode
//! - `resume`: Resume training
//!
//! Status is written to `{output}/hil_status.json` every episode.

use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use log::{info, warn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::Instant;

use prism_io::sovereign_types::Atom;
use prism_learning::{
    calculate_macro_step_reward, CalibrationManifest, DendriticAgent, DendriticAgentConfig,
    FactorizedAction, FeatureExtractor, NeuralStateExport, SimulationBuffers, Transition,
};
use prism_physics::molecular_dynamics::{MolecularDynamicsConfig, MolecularDynamicsEngine};

/// Command line arguments for Neuromorphic PRISM training
#[derive(Parser)]
#[command(name = "prism-train-neuro")]
#[command(about = "PRISM-Zero v3.1: Neuromorphic training with Flashbulb Reservoir (No PyTorch!)")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Args {
    /// Path to calibration manifest JSON file
    #[arg(short, long)]
    manifest: String,

    /// Maximum episodes per target
    #[arg(long, default_value = "1000")]
    max_episodes: usize,

    /// Output directory for checkpoints and results
    #[arg(short, long, default_value = "training_output_neuro")]
    output: String,

    /// CUDA device ID to use
    #[arg(long, default_value = "0")]
    device: usize,

    /// Checkpoint interval (episodes)
    #[arg(long, default_value = "100")]
    checkpoint_interval: usize,

    /// Early stopping patience (episodes)
    #[arg(long, default_value = "50")]
    patience: usize,

    /// Target reward for early stopping
    #[arg(long)]
    target_reward: Option<f32>,

    /// Reservoir size (number of LIF neurons)
    #[arg(long, default_value = "512")]
    reservoir_size: usize,

    /// RLS forgetting factor (lambda)
    #[arg(long, default_value = "0.99")]
    lambda: f32,

    /// Number of macro-steps per episode (Dendritic: use more, shorter chunks)
    #[arg(long, default_value = "100")]
    macro_steps: usize,

    /// Steps per macro-step (Dendritic: shorter bursts for more learning signal)
    #[arg(long, default_value = "10000")]
    steps_per_macro: u64,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,

    // ========== CONTINUAL LEARNING OPTIONS ==========
    /// Resume from checkpoint (load existing weights)
    #[arg(long)]
    resume: Option<String>,

    /// Number of full epochs (passes through all targets)
    #[arg(long, default_value = "1")]
    epochs: usize,

    /// Randomize target order within interleaved pattern each epoch
    #[arg(long)]
    randomize: bool,

    /// Initial learning rate multiplier (>1 for faster learning)
    #[arg(long, default_value = "1.0")]
    lr_multiplier: f32,

    /// Enable curriculum learning (progressive difficulty)
    #[arg(long)]
    curriculum: bool,

    /// Minimum epsilon (lower = more exploitation of learned policy)
    #[arg(long, default_value = "0.05")]
    epsilon_min: f64,

    /// Epsilon decay rate (higher = slower decay, more exploration)
    #[arg(long, default_value = "0.995")]
    epsilon_decay: f64,
}

/// Training statistics
#[derive(Debug, Default, Clone, Serialize)]
struct TrainingStats {
    targets_completed: usize,
    total_episodes: usize,
    total_steps: u64,
    total_transitions: usize,
    best_reward: f32,
    sum_best_rewards: f32,
}

// ============================================================================
// HIL (Human-in-the-Loop) Control System
// ============================================================================

/// HIL Control Commands (read from hil_control.json)
#[derive(Debug, Serialize, Deserialize, Default)]
struct HilControl {
    /// Spike exploration to 0.8 for next N episodes
    #[serde(default)]
    spike_exploration: usize,
    /// Set epsilon to custom value
    #[serde(default)]
    set_epsilon: Option<f32>,
    /// Learning rate multiplier (1.0 = default, >1 = faster learning, <1 = slower)
    #[serde(default)]
    learning_rate_multiplier: Option<f32>,
    /// Force checkpoint save
    #[serde(default)]
    save_checkpoint: bool,
    /// Pause training after current episode
    #[serde(default)]
    pause: bool,
    /// Command acknowledged flag (trainer sets this after processing)
    #[serde(default)]
    ack: u64,
}

/// HIL Status (written to hil_status.json every episode)
#[derive(Debug, Serialize)]
struct HilStatus {
    /// Current target name
    current_target: String,
    /// Current target family
    current_family: String,
    /// Current target index (1-indexed)
    target_idx: usize,
    /// Total targets
    total_targets: usize,
    /// Current episode (0-indexed)
    episode: usize,
    /// Max episodes per target
    max_episodes: usize,
    /// Current epsilon
    epsilon: f64,
    /// Current episode reward
    episode_reward: f32,
    /// Best reward for this target
    best_reward: f32,
    /// Episodes without improvement
    episodes_without_improvement: usize,
    /// Patience threshold
    patience: usize,
    /// RLS error from last training
    rls_error: f32,
    /// Episode duration in seconds
    episode_time_secs: f32,
    /// Total training time in seconds
    total_time_secs: f32,
    /// Estimated time remaining in seconds
    eta_secs: f32,
    /// Is paused
    paused: bool,
    /// Current learning rate multiplier
    learning_rate_multiplier: f32,
    /// Training stats
    stats: TrainingStats,
    /// Learning monitor - detailed live stats
    learning_monitor: LearningMonitor,
    /// Timestamp
    timestamp: String,
}

/// Detailed learning statistics for live monitoring
#[derive(Debug, Serialize, Default, Clone)]
struct LearningMonitor {
    /// Recent reward trend (last 10 episodes), positive = improving
    reward_trend: f32,
    /// RLS error trend (last 10 episodes), negative = improving
    error_trend: f32,
    /// Per-family performance summary
    family_performance: std::collections::HashMap<String, FamilyStats>,
    /// Rolling average of episode times
    avg_episode_time: f32,
    /// Recent rewards buffer for trend calculation
    recent_rewards: Vec<f32>,
    /// Recent errors buffer for trend calculation
    recent_errors: Vec<f32>,
    /// Current training mode (interleaved vs sequential)
    training_mode: String,
    /// Current family round (for interleaved mode)
    interleave_round: usize,
    /// Families in current round
    families_in_round: Vec<String>,
}

/// Per-family statistics
#[derive(Debug, Serialize, Default, Clone)]
struct FamilyStats {
    /// Number of targets in this family
    targets_count: usize,
    /// Targets completed
    targets_completed: usize,
    /// Sum of best rewards
    sum_best_rewards: f32,
    /// Average best reward
    avg_best_reward: f32,
    /// Total episodes trained
    total_episodes: usize,
}

/// Read HIL control file if it exists
fn read_hil_control(output_dir: &str) -> HilControl {
    let control_path = format!("{}/hil_control.json", output_dir);
    if let Ok(content) = fs::read_to_string(&control_path) {
        serde_json::from_str(&content).unwrap_or_default()
    } else {
        HilControl::default()
    }
}

/// Write HIL status file
fn write_hil_status(output_dir: &str, status: &HilStatus) {
    let status_path = format!("{}/hil_status.json", output_dir);
    if let Ok(json) = serde_json::to_string_pretty(status) {
        let _ = fs::write(&status_path, json);
    }
}

/// Write neural network state for visualization dashboard
fn write_neural_state(output_dir: &str, state: &NeuralStateExport) {
    let state_path = format!("{}/neural_state.json", output_dir);
    if let Ok(json) = serde_json::to_string_pretty(state) {
        let _ = fs::write(&state_path, json);
    }
}

/// Acknowledge HIL control command
fn ack_hil_control(output_dir: &str, ack_id: u64) {
    let control_path = format!("{}/hil_control.json", output_dir);
    let control = HilControl {
        ack: ack_id,
        ..Default::default()
    };
    if let Ok(json) = serde_json::to_string_pretty(&control) {
        let _ = fs::write(&control_path, json);
    }
}

/// Create initial HIL control file template
fn create_hil_control_template(output_dir: &str) {
    let control_path = format!("{}/hil_control.json", output_dir);
    let template = r#"{
  "spike_exploration": 0,
  "set_epsilon": null,
  "learning_rate_multiplier": null,
  "save_checkpoint": false,
  "pause": false,
  "ack": 0
}
"#;
    let _ = fs::write(&control_path, template);
    info!("📡 HIL control file created: {}", control_path);
    info!("   Edit this file to control training in real-time!");
    info!("   Commands:");
    info!("     spike_exploration: N   → Boost epsilon to 0.8 for N episodes");
    info!("     set_epsilon: 0.5       → Set epsilon to specific value");
    info!("     learning_rate_multiplier: 2.0 → Speed up learning (or <1 to slow)");
    info!("     save_checkpoint: true  → Force immediate checkpoint save");
    info!("     pause: true            → Pause training");
}

/// Calculate trend from recent values (positive = increasing)
fn calculate_trend(values: &[f32]) -> f32 {
    if values.len() < 3 {
        return 0.0;
    }
    let n = values.len();
    let half = n / 2;
    let first_half: f32 = values[..half].iter().sum::<f32>() / half as f32;
    let second_half: f32 = values[half..].iter().sum::<f32>() / (n - half) as f32;
    second_half - first_half
}

/// Interleave targets by family in round-robin fashion
/// This creates a "zipper" pattern: family1_t1, family2_t1, family3_t1, family1_t2, ...
/// If randomize=true, shuffles within each family for variety while maintaining interleaving
fn interleave_targets_by_family(
    targets: &[prism_learning::ProteinTarget],
    randomize: bool,
    seed: u64,
) -> Vec<usize> {
    use std::collections::HashMap;

    // Group target indices by family
    let mut family_indices: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, target) in targets.iter().enumerate() {
        family_indices
            .entry(target.family.clone())
            .or_default()
            .push(idx);
    }

    // Optionally randomize within each family
    if randomize {
        let mut rng = StdRng::seed_from_u64(seed);
        for indices in family_indices.values_mut() {
            indices.shuffle(&mut rng);
        }
    }

    // Get family names (randomize order too if requested)
    let mut family_names: Vec<_> = family_indices.keys().cloned().collect();
    if randomize {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(1));
        family_names.shuffle(&mut rng);
    } else {
        family_names.sort();
    }

    // Create interleaved order
    let mut interleaved: Vec<usize> = Vec::with_capacity(targets.len());
    let max_family_size = family_indices.values().map(|v| v.len()).max().unwrap_or(0);

    for round in 0..max_family_size {
        for family in &family_names {
            if let Some(indices) = family_indices.get(family) {
                if round < indices.len() {
                    interleaved.push(indices[round]);
                }
            }
        }
    }

    interleaved
}

/// Sort targets by difficulty for curriculum learning
/// Easy (low difficulty) first, hard (high difficulty) last
fn curriculum_sort_targets(targets: &[prism_learning::ProteinTarget]) -> Vec<usize> {
    let mut indexed: Vec<(usize, &str)> = targets
        .iter()
        .enumerate()
        .map(|(i, t)| (i, t.difficulty.as_str()))
        .collect();

    // Sort by difficulty: easy < medium < hard < expert
    indexed.sort_by_key(|(_, diff)| {
        match *diff {
            "easy" => 0,
            "medium" => 1,
            "hard" => 2,
            "expert" => 3,
            _ => 1, // Default to medium
        }
    });

    indexed.into_iter().map(|(i, _)| i).collect()
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    let log_level = if args.verbose { "debug" } else { "info" };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp_secs()
        .format_module_path(false)
        .init();

    // Print header
    println!();
    println!(
        "🧠 PRISM-Zero v{} NEUROMORPHIC Training Engine",
        prism_learning::PRISM_ZERO_VERSION
    );
    println!("⚡ Flashbulb Reservoir: E/I Balanced SNN + Reward-Modulated RLS");
    println!("🚀 NO PYTORCH REQUIRED - Pure Rust + CUDA");
    println!("{}", "═".repeat(70));
    println!();

    // Validate inputs
    if !Path::new(&args.manifest).exists() {
        anyhow::bail!("Manifest file not found: {}", args.manifest);
    }

    // Create output directory
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("Failed to create output directory: {}", args.output))?;

    // Initialize HIL control system
    create_hil_control_template(&args.output);

    // Load manifest
    info!("📋 Loading manifest: {}", args.manifest);
    let manifest =
        CalibrationManifest::load(&args.manifest).context("Failed to load calibration manifest")?;

    info!("   Targets: {}", manifest.targets.len());
    info!(
        "   Macro-steps: {} × {} steps = {}M total per episode",
        args.macro_steps,
        args.steps_per_macro,
        (args.macro_steps as u64 * args.steps_per_macro) / 1_000_000
    );

    // Create Dendritic Agent configuration
    // If resuming, start with lower epsilon (more exploitation)
    let epsilon_start = if args.resume.is_some() { 0.3 } else { 1.0 };

    let agent_config = DendriticAgentConfig {
        reservoir_size: args.reservoir_size,
        lambda: args.lambda,
        tau: 0.005, // Polyak averaging coefficient
        epsilon_start,
        epsilon_min: args.epsilon_min,
        epsilon_decay: args.epsilon_decay,
        gamma: 0.99,
        target_update_freq: 100,
    };

    info!("🧬 Initializing Dendritic Agent:");
    info!(
        "   Reservoir: {} neurons (80% E / 20% I)",
        agent_config.reservoir_size
    );
    info!("   Adaptive τ: 5-50ms (fast I, gradient E)");
    info!("   RLS lambda: {} (forgetting factor)", agent_config.lambda);
    info!("   Features: 23 raw → 46 expanded (+ velocity)");
    info!("   CUDA device: {}", args.device);
    if args.resume.is_some() {
        info!("   🔄 CONTINUAL LEARNING MODE: Starting from checkpoint");
    }

    // Create agent
    let mut agent = DendriticAgent::new_with_config(23, args.device, agent_config)
        .context("Failed to create Dendritic Agent")?;

    // Load checkpoint if resuming
    if let Some(ref checkpoint_path) = args.resume {
        info!("📂 Loading checkpoint: {}", checkpoint_path);
        agent
            .load(checkpoint_path)
            .with_context(|| format!("Failed to load checkpoint: {}", checkpoint_path))?;
        info!("   ✅ Weights loaded successfully!");
        info!(
            "   Starting epsilon: {:.3} (lower = more exploitation of learned policy)",
            epsilon_start
        );
    }

    // Set initial learning rate multiplier
    if args.lr_multiplier != 1.0 {
        agent.set_learning_rate_multiplier(args.lr_multiplier);
        info!("   📈 Learning rate multiplier: {}x", args.lr_multiplier);
    }

    // Training loop
    let mut stats = TrainingStats::default();
    let training_start = Instant::now();
    let mut hil_spike_episodes_remaining = 0usize;
    let mut hil_paused = false;
    let mut hil_ack_counter = 1u64;
    let mut hil_lr_multiplier = 1.0f32;
    let total_targets = manifest.targets.len();

    // Initialize learning monitor
    let mut learning_monitor = LearningMonitor {
        training_mode: "interleaved".to_string(),
        ..Default::default()
    };

    // Initialize family performance tracking
    for target in &manifest.targets {
        learning_monitor
            .family_performance
            .entry(target.family.clone())
            .or_insert_with(FamilyStats::default)
            .targets_count += 1;
    }

    // Training mode info
    let mode_str = if args.curriculum {
        "CURRICULUM (easy→hard)"
    } else if args.randomize {
        "RANDOMIZED INTERLEAVED"
    } else {
        "INTERLEAVED"
    };
    info!("📚 Training Mode: {}", mode_str);
    info!(
        "   Epochs: {} (full passes through all targets)",
        args.epochs
    );
    if args.randomize {
        info!("   🎲 Order will be randomized each epoch for better generalization");
    }
    info!("");

    let num_families = learning_monitor.family_performance.len();

    // ========================================================================
    // EPOCH LOOP - Multiple passes through all targets
    // ========================================================================
    for epoch in 0..args.epochs {
        println!();
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════╗"
        );
        info!(
            "🔄 EPOCH {}/{} {}",
            epoch + 1,
            args.epochs,
            if args.resume.is_some() {
                "(CONTINUAL LEARNING)"
            } else {
                ""
            }
        );
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════╝"
        );

        // Create target order for this epoch
        let epoch_seed = (epoch as u64).wrapping_mul(12345).wrapping_add(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        );

        let interleaved_order = if args.curriculum {
            // Curriculum: easy targets first, hard targets last
            curriculum_sort_targets(&manifest.targets)
        } else {
            // Interleaved with optional randomization
            interleave_targets_by_family(&manifest.targets, args.randomize, epoch_seed)
        };

        // Log the training order for this epoch
        if epoch == 0 || args.randomize {
            info!(
                "🔀 {} ORDER (Epoch {}):",
                mode_str.to_uppercase(),
                epoch + 1
            );
            let mut current_round = 0;
            for (i, &idx) in interleaved_order.iter().enumerate() {
                let target = &manifest.targets[idx];
                let round = i / num_families;
                if round != current_round && !args.curriculum {
                    current_round = round;
                    info!("   --- Round {} ---", round + 1);
                }
                info!(
                    "   {}: {} [{}] ({})",
                    i + 1,
                    target.name,
                    target.family,
                    target.difficulty
                );
            }
            info!("");
        }

        // Progressive learning rate adjustment per epoch
        if args.epochs > 1 && epoch > 0 {
            // Gradually decrease learning rate each epoch for fine-tuning
            let epoch_lr = args.lr_multiplier * (0.8_f32).powi(epoch as i32);
            agent.set_learning_rate_multiplier(epoch_lr);
            info!(
                "📈 Epoch {} learning rate: {:.2}x (progressive decay)",
                epoch + 1,
                epoch_lr
            );
        }

        for (order_idx, &target_idx) in interleaved_order.iter().enumerate() {
            let target = &manifest.targets[target_idx];
            // Calculate interleave round
            let interleave_round = order_idx / num_families;
            learning_monitor.interleave_round = interleave_round;

            println!();
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            info!(
                "🎯 Target {}/{}: {} (Interleave Round {})",
                order_idx + 1,
                total_targets,
                target.name,
                interleave_round + 1
            );
            info!(
                "   Family: {}, Difficulty: {}",
                target.family, target.difficulty
            );
            info!("   Target residues: {:?}", target.target_residues);
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

            // Load protein structure from PDB file
            let pdb_data = fs::read(&target.apo_pdb)
                .with_context(|| format!("Failed to read PDB: {}", target.apo_pdb))?;
            info!("   Loaded {} bytes from {}", pdb_data.len(), target.apo_pdb);

            // Reset agent for new target (clears neuronal state, keeps weights)
            agent.reset_episode()?;

            let mut best_episode_reward = f32::NEG_INFINITY;
            let mut episodes_without_improvement = 0;

            for episode in 0..args.max_episodes {
                let episode_start = Instant::now();

                // ================================================================
                // HIL Control: Check for commands before each episode
                // ================================================================
                let hil_control = read_hil_control(&args.output);

                // Handle pause command
                if hil_control.pause && !hil_paused {
                    hil_paused = true;
                    warn!("⏸️  HIL: Training PAUSED. Set 'pause: false' in hil_control.json to resume.");
                }

                // Wait while paused
                while hil_paused {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    let ctrl = read_hil_control(&args.output);
                    if !ctrl.pause {
                        hil_paused = false;
                        info!("▶️  HIL: Training RESUMED");
                    }
                }

                // Handle spike_exploration command
                if hil_control.spike_exploration > 0 {
                    hil_spike_episodes_remaining = hil_control.spike_exploration;
                    agent.set_epsilon(0.8f64);
                    warn!(
                        "🔥 HIL: SPIKE EXPLORATION activated for {} episodes (ε=0.8)",
                        hil_spike_episodes_remaining
                    );
                    ack_hil_control(&args.output, hil_ack_counter);
                    hil_ack_counter += 1;
                }

                // Handle set_epsilon command
                if let Some(new_eps) = hil_control.set_epsilon {
                    let clamped = (new_eps as f64).clamp(0.0, 1.0);
                    agent.set_epsilon(clamped);
                    warn!("🎚️  HIL: Epsilon set to {:.3}", clamped);
                    ack_hil_control(&args.output, hil_ack_counter);
                    hil_ack_counter += 1;
                }

                // Handle save_checkpoint command
                if hil_control.save_checkpoint {
                    let checkpoint_path = format!(
                        "{}/hil_checkpoint_{}_{}.json",
                        args.output, target.name, episode
                    );
                    agent.save(&checkpoint_path)?;
                    warn!("💾 HIL: Forced checkpoint saved: {}", checkpoint_path);
                    ack_hil_control(&args.output, hil_ack_counter);
                    hil_ack_counter += 1;
                }

                // Handle learning_rate_multiplier command
                if let Some(lr_mult) = hil_control.learning_rate_multiplier {
                    let clamped = lr_mult.clamp(0.1, 10.0);
                    if (clamped - hil_lr_multiplier).abs() > 0.001 {
                        hil_lr_multiplier = clamped;
                        agent.set_learning_rate_multiplier(clamped);
                        warn!("📈 HIL: Learning rate multiplier set to {:.2}x", clamped);
                        ack_hil_control(&args.output, hil_ack_counter);
                        hil_ack_counter += 1;
                    }
                }

                // Decay spike exploration counter
                if hil_spike_episodes_remaining > 0 {
                    hil_spike_episodes_remaining -= 1;
                    if hil_spike_episodes_remaining == 0 {
                        info!("🔥 HIL: Spike exploration ended, resuming normal decay");
                    }
                }
                // ================================================================

                // Run one episode with macro-step training
                let (episode_reward, transitions, steps) = run_macro_step_episode(
                    &mut agent,
                    &manifest,
                    target,
                    &pdb_data,
                    args.macro_steps,
                    args.steps_per_macro,
                )?;

                stats.total_steps += steps;
                stats.total_transitions += transitions.len();

                // Train on episode transitions (Reward-Modulated RLS)
                let batch: Vec<_> = transitions
                    .iter()
                    .map(|t| {
                        (
                            t.state.clone(),
                            t.action,
                            t.reward,
                            t.next_state.clone(),
                            t.done,
                        )
                    })
                    .collect();

                let avg_error = agent.train(batch)?;

                // Track best reward
                if episode_reward > best_episode_reward {
                    best_episode_reward = episode_reward;
                    episodes_without_improvement = 0;
                } else {
                    episodes_without_improvement += 1;
                }

                // Logging
                let episode_time = episode_start.elapsed();
                if episode % 10 == 0 || episode == args.max_episodes - 1 {
                    info!(
                    "   Episode {:4}: reward={:+.3}, best={:+.3}, ε={:.3}, RLS_err={:.4}, time={:.1}s",
                    episode, episode_reward, best_episode_reward,
                    agent.get_epsilon(), avg_error, episode_time.as_secs_f32()
                );
                }

                stats.total_episodes += 1;

                // ================================================================
                // HIL Status: Write status file for monitoring
                // ================================================================
                let total_time = training_start.elapsed().as_secs_f32();
                let episodes_done = stats.total_episodes;
                let episodes_remaining = (total_targets - target_idx - 1) * args.max_episodes
                    + (args.max_episodes - episode - 1);
                let avg_episode_time = if episodes_done > 0 {
                    total_time / episodes_done as f32
                } else {
                    0.0
                };
                let eta_secs = episodes_remaining as f32 * avg_episode_time;

                // Update learning monitor with recent data
                learning_monitor.recent_rewards.push(episode_reward);
                if learning_monitor.recent_rewards.len() > 20 {
                    learning_monitor.recent_rewards.remove(0);
                }
                learning_monitor.recent_errors.push(avg_error);
                if learning_monitor.recent_errors.len() > 20 {
                    learning_monitor.recent_errors.remove(0);
                }
                learning_monitor.reward_trend = calculate_trend(&learning_monitor.recent_rewards);
                learning_monitor.error_trend = calculate_trend(&learning_monitor.recent_errors);
                learning_monitor.avg_episode_time = if stats.total_episodes > 0 {
                    total_time / stats.total_episodes as f32
                } else {
                    0.0
                };

                // Update families in current round
                learning_monitor.families_in_round = learning_monitor
                    .family_performance
                    .keys()
                    .cloned()
                    .collect();
                learning_monitor.families_in_round.sort();

                let status = HilStatus {
                    current_target: target.name.clone(),
                    current_family: target.family.clone(),
                    target_idx: order_idx + 1,
                    total_targets,
                    episode,
                    max_episodes: args.max_episodes,
                    epsilon: agent.get_epsilon(),
                    episode_reward,
                    best_reward: best_episode_reward,
                    episodes_without_improvement,
                    patience: args.patience,
                    rls_error: avg_error,
                    episode_time_secs: episode_time.as_secs_f32(),
                    total_time_secs: total_time,
                    eta_secs,
                    paused: hil_paused,
                    learning_rate_multiplier: hil_lr_multiplier,
                    stats: stats.clone(),
                    learning_monitor: learning_monitor.clone(),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                };
                write_hil_status(&args.output, &status);

                // Export neural network state for visualization dashboard
                let neural_state = agent.export_neural_state(None);
                write_neural_state(&args.output, &neural_state);
                // ================================================================

                // Early stopping on target reward
                if let Some(target_reward) = args.target_reward {
                    if best_episode_reward >= target_reward {
                        info!(
                            "🎉 Target reward reached: {:.3} >= {:.3}",
                            best_episode_reward, target_reward
                        );
                        break;
                    }
                }

                // Early stopping on patience
                if episodes_without_improvement >= args.patience {
                    info!(
                        "⏹️  Early stopping: {} episodes without improvement",
                        args.patience
                    );
                    break;
                }

                // Checkpoint
                if episode > 0 && episode % args.checkpoint_interval == 0 {
                    let checkpoint_path = format!(
                        "{}/checkpoint_{}_{}.json",
                        args.output, target.name, episode
                    );
                    agent.save(&checkpoint_path)?;
                    info!("💾 Checkpoint saved: {}", checkpoint_path);
                }
            }

            stats.targets_completed += 1;
            stats.sum_best_rewards += best_episode_reward;
            if best_episode_reward > stats.best_reward {
                stats.best_reward = best_episode_reward;
            }

            // Update family performance
            if let Some(family_stats) = learning_monitor.family_performance.get_mut(&target.family)
            {
                family_stats.targets_completed += 1;
                family_stats.sum_best_rewards += best_episode_reward;
                family_stats.avg_best_reward = if family_stats.targets_completed > 0 {
                    family_stats.sum_best_rewards / family_stats.targets_completed as f32
                } else {
                    0.0
                };
            }

            info!(
                "✅ Target {} [{}] completed: best_reward={:+.3}",
                target.name, target.family, best_episode_reward
            );

            // Log family progress
            if let Some(fs) = learning_monitor.family_performance.get(&target.family) {
                info!(
                    "   📊 Family '{}' progress: {}/{} targets, avg_reward={:+.4}",
                    target.family, fs.targets_completed, fs.targets_count, fs.avg_best_reward
                );
            }

            // Save per-target checkpoint
            let target_path = format!("{}/agent_after_{}.json", args.output, target.name);
            agent.save(&target_path)?;
        }

        // End of epoch - save epoch checkpoint
        let epoch_path = format!("{}/agent_epoch_{}.json", args.output, epoch + 1);
        agent.save(&epoch_path)?;
        info!("💾 Epoch {} checkpoint saved: {}", epoch + 1, epoch_path);

        // Epoch summary
        println!();
        println!(
            "{}",
            "╔══════════════════════════════════════════════════════════════════════════╗"
        );
        info!("📊 EPOCH {}/{} COMPLETE", epoch + 1, args.epochs);
        info!("   Targets this epoch: {}", manifest.targets.len());
        info!("   Total episodes: {}", stats.total_episodes);
        info!("   Best reward: {:+.6}", stats.best_reward);
        println!(
            "{}",
            "╚══════════════════════════════════════════════════════════════════════════╝"
        );
    } // End of epoch loop

    // Final summary
    let training_time = training_start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("✅ NEUROMORPHIC TRAINING COMPLETED");
    println!("═══════════════════════════════════════════════════════════════════════");
    println!("📊 Statistics:");
    println!("   Targets completed: {}", stats.targets_completed);
    println!("   Total episodes: {}", stats.total_episodes);
    println!(
        "   Total simulation steps: {} ({:.1}B)",
        stats.total_steps,
        stats.total_steps as f64 / 1e9
    );
    println!("   Total transitions: {}", stats.total_transitions);
    println!(
        "   Average best reward: {:+.3}",
        stats.sum_best_rewards / stats.targets_completed.max(1) as f32
    );
    println!("   Best overall reward: {:+.3}", stats.best_reward);
    println!(
        "   Training time: {:.1} minutes",
        training_time.as_secs_f32() / 60.0
    );
    println!();
    println!("📁 Results saved to: {}/", args.output);
    println!();

    // Save final agent
    let final_path = format!("{}/dendritic_agent_final.json", args.output);
    agent.save(&final_path)?;
    info!("💾 Final agent saved: {}", final_path);

    Ok(())
}

/// Run a single episode with macro-step chunking
///
/// This is the key innovation: instead of 1M steps → 1 transition,
/// we do 100 × 10K steps → 100 transitions. More learning signal!
fn run_macro_step_episode(
    agent: &mut DendriticAgent,
    manifest: &CalibrationManifest,
    target: &prism_learning::ProteinTarget,
    pdb_data: &[u8],
    num_macro_steps: usize,
    steps_per_macro: u64,
) -> Result<(f32, Vec<Transition>, u64)> {
    let reward_weights = &manifest.training_parameters.reward_weighting;
    let feature_config = &manifest.feature_config;

    // 1. Configure MD engine
    let md_config = MolecularDynamicsConfig {
        max_steps: steps_per_macro * num_macro_steps as u64,
        dt: 0.002,         // 2 femtoseconds
        friction: 1.0,     // Default, action will modify
        temp_start: 300.0, // Default, action will modify
        temp_end: 150.0,
        annealing_steps: steps_per_macro / 2,
        cutoff_dist: 10.0,
        spring_k: 500.0,
        bias_strength: 0.0,
        target_mode: 7,
        use_gpu: true,
        max_trajectory_memory: 256 * 1024 * 1024,
        max_workspace_memory: 128 * 1024 * 1024,
    };

    // 2. Initialize MD engine from PDB data
    let mut engine = MolecularDynamicsEngine::from_sovereign_buffer(md_config, pdb_data)
        .context("Failed to initialize MD engine")?;

    // 3. Get initial atoms and create feature extractor
    let initial_atoms = engine.get_initial_atoms().to_vec();
    let mut feature_extractor = FeatureExtractor::new(feature_config.clone(), target);
    feature_extractor.initialize(&initial_atoms);

    // 4. Create SimulationBuffers for reward calculation
    let initial_buffers = SimulationBuffers::from_atoms(&initial_atoms);
    let mut current_buffers = initial_buffers.clone();

    // 5. Extract initial features
    let initial_features = feature_extractor.extract(&initial_atoms, None);
    let mut current_features = initial_features.as_slice().to_vec();

    // 6. Select initial action
    let mut current_action = agent.select_action(&current_features);

    // 7. MACRO-STEP LOOP - The heart of the training
    let mut transitions: Vec<Transition> = Vec::new();
    let mut cumulative_reward = 0.0f32;
    let mut total_steps = 0u64;

    for macro_step in 0..num_macro_steps {
        // A. Run physics simulation for this chunk
        engine
            .run_nlnm_breathing(steps_per_macro)
            .context("Physics simulation failed")?;
        total_steps += steps_per_macro;

        // B. Get current atom positions from GPU
        let current_atoms = engine
            .get_current_atoms()
            .context("Failed to get current atoms")?;

        // C. Update SimulationBuffers with new positions
        for (i, atom) in current_atoms.iter().enumerate() {
            if i < current_buffers.num_atoms {
                let base_idx = i * 4;
                current_buffers.positions[base_idx] = atom.coords[0];
                current_buffers.positions[base_idx + 1] = atom.coords[1];
                current_buffers.positions[base_idx + 2] = atom.coords[2];
            }
        }
        current_buffers.global_step = (macro_step as u64 + 1) * steps_per_macro;

        // D. Calculate macro-step reward (now with stability + clash penalties)
        let step_reward = calculate_macro_step_reward(
            &initial_buffers,
            &current_buffers,
            &target.target_residues,
            &target.core_residues,
            reward_weights,
            macro_step,
            num_macro_steps,
        );
        cumulative_reward += step_reward;

        // E. Extract new features
        let next_features_vec = feature_extractor.extract(&current_atoms, None);
        let next_features = next_features_vec.as_slice().to_vec();

        // F. Select next action (while we have current state)
        let next_action = agent.select_action(&next_features);

        // G. Store transition
        let done = macro_step == num_macro_steps - 1;
        transitions.push(Transition {
            state: current_features.clone(),
            action: current_action,
            reward: step_reward,
            next_state: next_features.clone(),
            done,
        });

        // H. Advance state
        current_features = next_features;
        current_action = next_action;
    }

    Ok((cumulative_reward, transitions, total_steps))
}
