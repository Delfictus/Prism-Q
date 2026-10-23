//! PRISM-Zero v3.1 Macro-Step Training Pipeline
//!
//! ## Key Innovation: Macro-Step Training
//! Instead of running 1M steps and getting ONE transition, we chunk the simulation
//! into macro-steps (e.g., 10 chunks of 100K steps) and collect a transition at
//! each boundary. This gives us 10x more training signal per episode.
//!
//! ## Target-Aware Feature Extraction
//! The AI "sees" each protein through a 23-dimensional feature vector that includes:
//! - **Global**: Size, Radius of Gyration, Density
//! - **Target Neighborhood**: Exposure, burial depth, contacts
//! - **Stability**: RMSD proxies, clash counts
//! - **Family Flags**: Protein family one-hot encoding
//! - **Temporal**: Change from initial state
//!
//! ## Architecture
//! ```text
//! Manifest (JSON) → Feature Extractor → DQN Agent → Physics → Reward
//!     ↓                   ↓                ↓          ↓         ↓
//! Weights/Config    23-dim vector   4×5 factorized MD sim   JSON weights
//!                                   (temp,fric,k,bias)
//! ```

use anyhow::{Context, Result};
use log::{debug, info, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(feature = "rl")]
use crate::agent::DQNAgent;
use crate::buffers::SimulationBuffers;
use crate::features::FeatureExtractor;
use crate::manifest::{CalibrationManifest, ProteinTarget};
use crate::rewards::{calculate_macro_step_reward, evaluate_simulation_weighted};
use prism_io::sovereign_types::Atom;
use prism_physics::molecular_dynamics::{MolecularDynamicsConfig, MolecularDynamicsEngine};

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Hyper-Q Parallel Trainer with Macro-Step Training
#[cfg(feature = "rl")]
pub struct PrismTrainer {
    /// Thread-Safe Agent (Shared across parallel simulations)
    agent: Arc<Mutex<DQNAgent>>,
    manifest: CalibrationManifest,
    config: TrainingConfig,
    device_id: usize,
    sessions: Arc<Mutex<Vec<TrainingSession>>>,
}

/// Trainer stub when RL feature is disabled
#[cfg(not(feature = "rl"))]
pub struct PrismTrainer {
    manifest: CalibrationManifest,
    config: TrainingConfig,
    device_id: usize,
    sessions: Arc<Mutex<Vec<TrainingSession>>>,
}

/// Training configuration (extends manifest with runtime options)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub max_episodes: usize,
    pub output_dir: String,
    pub parallel_jobs: usize,
    pub checkpoint_interval: usize,
    pub early_stopping_patience: usize,
    pub target_reward: Option<f32>,
    pub parallel_targets: bool,
    /// Enable macro-step training (chunked episodes)
    pub use_macro_steps: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_episodes: 100,
            output_dir: "training_results".to_string(),
            parallel_jobs: 4,
            checkpoint_interval: 100,
            early_stopping_patience: 50,
            target_reward: None,
            parallel_targets: true,
            use_macro_steps: true,
        }
    }
}

/// Training session state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSession {
    pub target_name: String,
    pub episode: usize,
    pub best_reward: f32,
    pub total_steps: u64,
    pub transitions_collected: usize,
}

/// Transition for replay buffer
#[derive(Clone)]
pub struct Transition {
    pub state: Vec<f32>,
    pub action: usize,
    pub reward: f32,
    pub next_state: Vec<f32>,
    pub done: bool,
}

// ============================================================================
// TRAINER IMPLEMENTATION
// ============================================================================

#[cfg(feature = "rl")]
impl PrismTrainer {
    /// Create new Macro-Step Trainer with target-aware features
    ///
    /// Uses 23-dimensional feature vector and 4×5 factorized action space
    pub fn new(manifest_path: &str, config: TrainingConfig, device_id: usize) -> Result<Self> {
        info!("⚡ Initializing PRISM-Zero v3.1 Macro-Step Trainer");
        info!("   Parallel jobs: {} CUDA streams", config.parallel_jobs);
        info!("   Feature vector: 23 dimensions (target-aware)");
        info!("   Action space: 4×5 factorized (temp, friction, spring_k, bias_strength)");

        let manifest = CalibrationManifest::load(manifest_path)
            .context("Failed to load calibration manifest")?;

        // Feature dim from manifest config, action dim = 125 (5^3)
        let feature_dim = manifest.feature_config.feature_dim();
        let action_dim = manifest.physics_parameter_ranges.action_space_size();

        info!("   Computed feature dim: {}", feature_dim);
        info!(
            "   Macro-steps: {} × {} steps",
            manifest.macro_step_config.num_macro_steps, manifest.macro_step_config.steps_per_macro
        );

        let agent = DQNAgent::new(feature_dim as i64, action_dim as i64, device_id)
            .context("Failed to create DQN agent")?;

        std::fs::create_dir_all(&config.output_dir).context("Failed to create output directory")?;

        info!("📋 Loaded {} targets from manifest", manifest.targets.len());

        Ok(Self {
            agent: Arc::new(Mutex::new(agent)),
            manifest,
            config,
            device_id,
            sessions: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Run training on all targets
    pub fn train_all_targets(&mut self) -> Result<()> {
        let start_time = Instant::now();

        info!(
            "🚀 Starting MACRO-STEP Training ({} jobs)...",
            self.config.parallel_jobs
        );
        info!(
            "   Mode: {} training",
            if self.config.use_macro_steps {
                "Macro-step"
            } else {
                "Single-step"
            }
        );

        if self.config.parallel_targets {
            self.train_hyperq_parallel()?;
        } else {
            self.train_sequential()?;
        }

        self.save_final_results()?;

        let elapsed = start_time.elapsed();
        info!(
            "🏁 Training completed in {:.1} minutes",
            elapsed.as_secs_f32() / 60.0
        );

        Ok(())
    }

    /// Sequential training (one episode at a time)
    fn train_sequential(&mut self) -> Result<()> {
        let targets = self.manifest.targets.clone();

        for episode in 0..self.config.max_episodes {
            let target = &targets[episode % targets.len()];
            info!("Episode {}: Training on {}", episode, target.name);

            if self.config.use_macro_steps {
                if let Err(e) = self.run_macro_step_episode(target, episode) {
                    warn!("❌ Episode {} failed: {}", episode, e);
                }
            } else {
                if let Err(e) = self.run_single_step_episode(target, episode) {
                    warn!("❌ Episode {} failed: {}", episode, e);
                }
            }

            // Periodic checkpoint saving
            if (episode + 1) % self.config.checkpoint_interval == 0 {
                let checkpoint_path =
                    format!("{}/checkpoint_ep{}.pt", self.config.output_dir, episode + 1);
                if let Err(e) = self.save_checkpoint(&checkpoint_path) {
                    warn!("Failed to save checkpoint: {}", e);
                }
                // Also save as latest
                let latest_path = format!("{}/latest_checkpoint.pt", self.config.output_dir);
                let _ = self.save_checkpoint(&latest_path);
            }
        }
        Ok(())
    }

    /// Hyper-Q Parallel Training - Multiple CUDA Streams
    fn train_hyperq_parallel(&mut self) -> Result<()> {
        info!(
            "⚡ HYPER-Q MODE: {} parallel CUDA streams",
            self.config.parallel_jobs
        );

        let targets = self.manifest.targets.clone();
        let num_batches =
            (self.config.max_episodes as f32 / self.config.parallel_jobs as f32).ceil() as usize;

        for batch_idx in 0..num_batches {
            info!("⚡ Processing Batch {}/{}", batch_idx + 1, num_batches);

            // Prepare work items for this batch
            let mut work_items = Vec::new();
            for i in 0..self.config.parallel_jobs {
                let global_episode = batch_idx * self.config.parallel_jobs + i;
                if global_episode >= self.config.max_episodes {
                    break;
                }

                let target = targets[global_episode % targets.len()].clone();
                work_items.push((global_episode, target));
            }

            // EXECUTE PARALLEL BATCH
            if self.config.use_macro_steps {
                work_items.par_iter().for_each(|(episode, target)| {
                    if let Err(e) = self.run_macro_step_episode(target, *episode) {
                        warn!("Episode {} failed: {}", episode, e);
                    }
                });
            } else {
                work_items.par_iter().for_each(|(episode, target)| {
                    if let Err(e) = self.run_single_step_episode(target, *episode) {
                        warn!("Episode {} failed: {}", episode, e);
                    }
                });
            }

            // Checkpoint after each batch
            let current_episode = (batch_idx + 1) * self.config.parallel_jobs;
            if current_episode % self.config.checkpoint_interval == 0 {
                let checkpoint_path = format!(
                    "{}/checkpoint_ep{}.pt",
                    self.config.output_dir, current_episode
                );
                if let Err(e) = self.save_checkpoint(&checkpoint_path) {
                    warn!("Failed to save checkpoint: {}", e);
                }
                let latest_path = format!("{}/latest_checkpoint.pt", self.config.output_dir);
                let _ = self.save_checkpoint(&latest_path);
            }
        }

        info!("🏁 Hyper-Q parallel training complete");
        Ok(())
    }

    // ========================================================================
    // MACRO-STEP EPISODE (The Key Innovation)
    // ========================================================================

    /// Run episode with macro-step chunking - collects MULTIPLE transitions per episode
    fn run_macro_step_episode(&self, target: &ProteinTarget, episode: usize) -> Result<()> {
        let macro_config = &self.manifest.macro_step_config;
        let reward_weights = &self.manifest.training_parameters.reward_weighting;
        let feature_config = &self.manifest.feature_config;

        // 1. Load and parse PDB
        let pdb_data = std::fs::read(&target.apo_pdb)
            .with_context(|| format!("Failed to read PDB: {}", target.apo_pdb))?;
        let initial_atoms = self.parse_pdb_to_atoms(&pdb_data)?;

        // 2. Initialize Feature Extractor (target-aware)
        let mut feature_extractor = FeatureExtractor::new(feature_config.clone(), target);
        feature_extractor.initialize(&initial_atoms);

        // 3. Extract initial features
        let initial_features = feature_extractor.extract(&initial_atoms, None);
        let mut current_features = initial_features.as_slice().to_vec();

        // 4. Build initial buffers
        let initial_buffers = SimulationBuffers::from_atoms(&initial_atoms);
        let mut current_buffers = initial_buffers.clone();

        // 5. Select initial action
        let mut current_action = {
            let agent = self.agent.lock().unwrap();
            agent.select_action(&current_features)
        };

        // 6. Configure Physics Engine (4D action space)
        let (temp, fric, spring, bias) = self
            .manifest
            .physics_parameter_ranges
            .action_to_params(current_action);
        let md_config = MolecularDynamicsConfig {
            max_steps: macro_config.steps_per_macro,
            dt: 0.001,
            friction: fric,
            temp_start: temp,
            temp_end: temp * 0.5,
            annealing_steps: macro_config.steps_per_macro / 2,
            cutoff_dist: 10.0,
            spring_k: spring,
            bias_strength: bias,
            target_mode: 7,
            use_gpu: true,
            max_trajectory_memory: 256 * 1024 * 1024,
            max_workspace_memory: 128 * 1024 * 1024,
        };

        let mut engine =
            MolecularDynamicsEngine::from_sovereign_buffer(md_config.clone(), &pdb_data)
                .context("Failed to initialize physics engine")?;

        // 7. MACRO-STEP LOOP - Collect transitions at each boundary
        let mut transitions: Vec<Transition> = Vec::new();
        let mut cumulative_reward = 0.0;

        for macro_step in 0..macro_config.num_macro_steps {
            // Run one macro-step of simulation
            engine
                .run_nlnm_breathing(macro_config.steps_per_macro)
                .context("Physics simulation failed")?;

            // Get current state from GPU
            let current_atoms = engine
                .get_current_atoms()
                .context("Failed to download atoms from GPU")?;

            // Update buffers
            for (i, atom) in current_atoms.iter().enumerate() {
                if i < current_buffers.num_atoms {
                    let base_idx = i * 4;
                    current_buffers.positions[base_idx] = atom.coords[0];
                    current_buffers.positions[base_idx + 1] = atom.coords[1];
                    current_buffers.positions[base_idx + 2] = atom.coords[2];
                }
            }
            current_buffers.global_step = (macro_step as u64 + 1) * macro_config.steps_per_macro;

            // Calculate intermediate reward (with stability + clash penalties)
            let step_reward = calculate_macro_step_reward(
                &initial_buffers,
                &current_buffers,
                &target.target_residues,
                &target.core_residues,
                reward_weights,
                macro_step,
                macro_config.num_macro_steps,
            );

            // Extract next state features
            let next_features = feature_extractor.extract(&current_atoms, Some(&initial_atoms));
            let next_features_vec = next_features.as_slice().to_vec();

            // Determine if episode is done
            let is_last = macro_step == macro_config.num_macro_steps - 1;

            // Store transition
            transitions.push(Transition {
                state: current_features.clone(),
                action: current_action,
                reward: step_reward,
                next_state: next_features_vec.clone(),
                done: is_last,
            });

            cumulative_reward += step_reward;

            // Optionally track action changes between macro-steps
            // Note: Actual physics parameter changes require engine reconstruction
            // For now, we track the intended action for the transition but continue
            // with the original parameters. Future: serialize atoms to PDB buffer.
            if macro_config.allow_action_change && !is_last {
                let new_action = {
                    let agent = self.agent.lock().unwrap();
                    agent.select_action(&next_features_vec)
                };

                if new_action != current_action {
                    let (new_temp, new_fric, new_spring, new_bias) = self
                        .manifest
                        .physics_parameter_ranges
                        .action_to_params(new_action);

                    // Log the intended action change (engine continues with original params)
                    debug!("Macro-step {}: Would change action to {} (T={:.1}, F={:.2}, K={:.1}, B={:.2})",
                           macro_step, new_action, new_temp, new_fric, new_spring, new_bias);

                    // Track the action change for the next transition
                    current_action = new_action;
                }
            }

            current_features = next_features_vec;
        }

        // 8. Calculate final evaluation with full reward weights
        let final_result = evaluate_simulation_weighted(
            &initial_buffers,
            &current_buffers,
            &target.target_residues,
            &target.core_residues,
            reward_weights,
            target.expected_sasa_gain,
        )?;

        info!("✅ Ep {} | {} | {} transitions | Reward: {:.4} (Exposure: {:.2}, RMSD: {:.2}Å, Clashes: {})",
              episode, target.name, transitions.len(),
              final_result.intrinsic_reward, final_result.total_sasa_gain,
              final_result.core_rmsd, final_result.clash_count);

        // 9. Train agent with collected transitions
        {
            let mut agent = self.agent.lock().unwrap();
            let batch: Vec<(Vec<f32>, usize, f32, Vec<f32>, bool)> = transitions
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
            let _ = agent.train(batch);
        }

        // 10. Update session
        {
            let mut sessions = self.sessions.lock().unwrap();
            let session = sessions.iter_mut().find(|s| s.target_name == target.name);

            if let Some(s) = session {
                s.episode += 1;
                s.total_steps += self.manifest.total_steps_per_episode();
                s.transitions_collected += transitions.len();
                if final_result.intrinsic_reward > s.best_reward {
                    s.best_reward = final_result.intrinsic_reward;
                }
            } else {
                sessions.push(TrainingSession {
                    target_name: target.name.clone(),
                    episode: 1,
                    best_reward: final_result.intrinsic_reward,
                    total_steps: self.manifest.total_steps_per_episode(),
                    transitions_collected: transitions.len(),
                });
            }
        }

        Ok(())
    }

    // ========================================================================
    // SINGLE-STEP EPISODE (Legacy mode)
    // ========================================================================

    /// Run episode without macro-stepping (single transition per episode)
    fn run_single_step_episode(&self, target: &ProteinTarget, episode: usize) -> Result<()> {
        let total_steps = self.manifest.total_steps_per_episode();
        let reward_weights = &self.manifest.training_parameters.reward_weighting;
        let feature_config = &self.manifest.feature_config;

        // 1. Load and parse PDB
        let pdb_data = std::fs::read(&target.apo_pdb)
            .with_context(|| format!("Failed to read PDB: {}", target.apo_pdb))?;
        let initial_atoms = self.parse_pdb_to_atoms(&pdb_data)?;

        // 2. Initialize Feature Extractor
        let mut feature_extractor = FeatureExtractor::new(feature_config.clone(), target);
        feature_extractor.initialize(&initial_atoms);

        // 3. Extract features
        let features = feature_extractor.extract(&initial_atoms, None);
        let features_vec = features.as_slice().to_vec();

        // 4. Agent selects action
        let action = {
            let agent = self.agent.lock().unwrap();
            agent.select_action(&features_vec)
        };

        let (temp, fric, spring, bias) = self
            .manifest
            .physics_parameter_ranges
            .action_to_params(action);

        debug!(
            "Ep {} ({}): Action {} (T={:.1}, F={:.2}, K={:.1}, B={:.2})",
            episode, target.name, action, temp, fric, spring, bias
        );

        // 5. Configure and run physics (4D action space)
        let md_config = MolecularDynamicsConfig {
            max_steps: total_steps,
            dt: 0.001,
            friction: fric,
            temp_start: temp,
            temp_end: temp * 0.5,
            annealing_steps: total_steps / 2,
            cutoff_dist: 10.0,
            spring_k: spring,
            bias_strength: bias,
            target_mode: 7,
            use_gpu: true,
            max_trajectory_memory: 256 * 1024 * 1024,
            max_workspace_memory: 128 * 1024 * 1024,
        };

        let mut engine = MolecularDynamicsEngine::from_sovereign_buffer(md_config, &pdb_data)
            .context("Failed to initialize physics engine")?;

        engine
            .run_nlnm_breathing(total_steps)
            .context("Physics simulation failed")?;

        // 6. Get results
        let initial_buffers = SimulationBuffers::from_atoms(&initial_atoms);
        let final_atoms = engine
            .get_current_atoms()
            .context("Failed to download atoms from GPU")?;

        let mut final_buffers = initial_buffers.clone();
        for (i, atom) in final_atoms.iter().enumerate() {
            if i < final_buffers.num_atoms {
                let base_idx = i * 4;
                final_buffers.positions[base_idx] = atom.coords[0];
                final_buffers.positions[base_idx + 1] = atom.coords[1];
                final_buffers.positions[base_idx + 2] = atom.coords[2];
            }
        }
        final_buffers.global_step = total_steps;

        // 7. Evaluate with manifest weights
        let result = evaluate_simulation_weighted(
            &initial_buffers,
            &final_buffers,
            &target.target_residues,
            &target.core_residues,
            reward_weights,
            target.expected_sasa_gain,
        )?;

        info!(
            "✅ Ep {} | {} | Reward: {:.4} (Exposure: {:.2}, RMSD: {:.2}Å)",
            episode, target.name, result.intrinsic_reward, result.total_sasa_gain, result.core_rmsd
        );

        // 8. Train agent
        {
            let next_features = feature_extractor.extract(&final_atoms, Some(&initial_atoms));
            let mut agent = self.agent.lock().unwrap();
            let _ = agent.train(vec![(
                features_vec,
                action,
                result.intrinsic_reward,
                next_features.as_slice().to_vec(),
                true, // Single-step episodes are always "done"
            )]);
        }

        // 9. Update session
        {
            let mut sessions = self.sessions.lock().unwrap();
            let session = sessions.iter_mut().find(|s| s.target_name == target.name);

            if let Some(s) = session {
                s.episode += 1;
                s.total_steps += total_steps;
                s.transitions_collected += 1;
                if result.intrinsic_reward > s.best_reward {
                    s.best_reward = result.intrinsic_reward;
                }
            } else {
                sessions.push(TrainingSession {
                    target_name: target.name.clone(),
                    episode: 1,
                    best_reward: result.intrinsic_reward,
                    total_steps: total_steps,
                    transitions_collected: 1,
                });
            }
        }

        Ok(())
    }

    // ========================================================================
    // UTILITY METHODS
    // ========================================================================

    /// Parse PDB bytes to sovereign Atom array
    fn parse_pdb_to_atoms(&self, pdb_data: &[u8]) -> Result<Vec<Atom>> {
        let content = String::from_utf8_lossy(pdb_data);
        let mut atoms = Vec::new();

        for line in content.lines() {
            if line.starts_with("ATOM") || line.starts_with("HETATM") {
                if line.len() < 54 {
                    continue;
                }

                let residue_seq: u16 = line[22..26].trim().parse().unwrap_or(0);
                let x: f32 = line[30..38].trim().parse().unwrap_or(0.0);
                let y: f32 = line[38..46].trim().parse().unwrap_or(0.0);
                let z: f32 = line[46..54].trim().parse().unwrap_or(0.0);

                // Parse element from columns 77-78
                let element: u8 = if line.len() >= 78 {
                    let elem_str = line[76..78].trim();
                    match elem_str {
                        "C" => 6,
                        "N" => 7,
                        "O" => 8,
                        "S" => 16,
                        "H" => 1,
                        "P" => 15,
                        "CA" | "Ca" => 20,
                        "FE" | "Fe" => 26,
                        "ZN" | "Zn" => 30,
                        "MG" | "Mg" => 12,
                        _ => 6,
                    }
                } else {
                    6
                };

                atoms.push(Atom {
                    coords: [x, y, z],
                    element,
                    residue_id: residue_seq,
                    atom_type: 0,
                    charge: 0.0,
                    radius: 1.7,
                    _reserved: [0; 4],
                });
            }
        }

        if atoms.is_empty() {
            anyhow::bail!("No atoms found in PDB data");
        }

        debug!("Parsed {} atoms from PDB", atoms.len());
        Ok(atoms)
    }

    /// Save checkpoint (model weights + training state)
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        let agent = self.agent.lock().unwrap();
        agent.save(path)?;
        info!("💾 Checkpoint saved to {}", path);
        Ok(())
    }

    /// Load checkpoint and resume training
    pub fn load_checkpoint(&self, path: &str) -> Result<()> {
        let mut agent = self.agent.lock().unwrap();
        agent.load(path)?;
        info!("📂 Checkpoint loaded from {}", path);
        Ok(())
    }

    /// Try to resume from latest checkpoint in output directory
    pub fn try_resume(&self) -> Result<bool> {
        let checkpoint_path = format!("{}/latest_checkpoint.pt", self.config.output_dir);
        if std::path::Path::new(&checkpoint_path).exists() {
            self.load_checkpoint(&checkpoint_path)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Save final training results
    fn save_final_results(&self) -> Result<()> {
        // Save final checkpoint
        let checkpoint_path = format!("{}/final_model.pt", self.config.output_dir);
        self.save_checkpoint(&checkpoint_path)?;

        let sessions = self.sessions.lock().unwrap();
        let results_path = format!("{}/training_results.json", self.config.output_dir);

        let total_transitions: usize = sessions.iter().map(|s| s.transitions_collected).sum();

        let results = serde_json::json!({
            "prism_zero_version": crate::PRISM_ZERO_VERSION,
            "manifest": self.manifest.dataset_name,
            "config": self.config,
            "macro_step_config": self.manifest.macro_step_config,
            "reward_weights": self.manifest.training_parameters.reward_weighting,
            "feature_config": self.manifest.feature_config,
            "sessions": *sessions,
            "total_transitions_collected": total_transitions,
            "feature_description": {
                "global": ["size", "radius_of_gyration", "density"],
                "target_neighborhood": ["exposure", "burial_depth", "neighbor_count", "contact_count",
                                        "distance_to_core", "local_density", "mobility_proxy", "exposure_delta"],
                "stability": ["core_rmsd_proxy", "clash_count", "max_displacement", "avg_displacement"],
                "family_flags": ["is_cytokine", "is_gtpase", "is_phosphatase", "is_viral"],
                "temporal": ["exposure_change", "rg_change", "density_change", "displacement_magnitude"]
            },
            "completion_time": chrono::Utc::now(),
        });

        std::fs::write(&results_path, serde_json::to_string_pretty(&results)?)
            .context("Failed to save results")?;

        info!("📁 Results saved to {}", results_path);
        info!("   Total transitions collected: {}", total_transitions);
        Ok(())
    }

    /// Get training statistics
    pub fn get_stats(&self) -> serde_json::Value {
        let sessions = self.sessions.lock().unwrap();
        let total_episodes: usize = sessions.iter().map(|s| s.episode).sum();
        let total_steps: u64 = sessions.iter().map(|s| s.total_steps).sum();
        let total_transitions: usize = sessions.iter().map(|s| s.transitions_collected).sum();
        let best_rewards: Vec<f32> = sessions.iter().map(|s| s.best_reward).collect();

        let avg_reward = if !best_rewards.is_empty() {
            best_rewards.iter().sum::<f32>() / best_rewards.len() as f32
        } else {
            0.0
        };

        serde_json::json!({
            "targets_completed": sessions.len(),
            "targets_total": self.manifest.targets.len(),
            "total_episodes": total_episodes,
            "total_simulation_steps": total_steps,
            "total_transitions": total_transitions,
            "transitions_per_episode": if total_episodes > 0 { total_transitions / total_episodes } else { 0 },
            "average_best_reward": avg_reward,
            "best_overall_reward": best_rewards.into_iter().fold(f32::NEG_INFINITY, f32::max),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.max_episodes, 100);
        assert_eq!(config.parallel_jobs, 4);
        assert!(config.use_macro_steps);
    }

    #[test]
    fn test_transition_clone() {
        let t = Transition {
            state: vec![1.0, 2.0, 3.0],
            action: 42,
            reward: 0.5,
            next_state: vec![4.0, 5.0, 6.0],
            done: false,
        };
        let t2 = t.clone();
        assert_eq!(t2.action, 42);
        assert_eq!(t2.reward, 0.5);
    }
}
