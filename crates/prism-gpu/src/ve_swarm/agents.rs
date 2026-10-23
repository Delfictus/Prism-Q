//! Swarm Agent Intelligence
//!
//! 32 GPU-accelerated agents that compete and cooperate to discover
//! optimal feature combinations for RISE/FALL prediction.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Number of features in input (136-dim: TDA + base + physics + fitness + cycle + spike + immunity + epi)
pub const N_FEATURES: usize = 136;

/// Number of swarm agents
pub const N_AGENTS: usize = 32;

/// Configuration for swarm agents
#[derive(Clone, Debug)]
pub struct SwarmConfig {
    /// Evolution interval (predictions between evolutions)
    pub evolution_interval: usize,
    /// Mutation rate for genetic algorithm
    pub mutation_rate: f32,
    /// Crossover rate
    pub crossover_rate: f32,
    /// Number of elite agents that survive unchanged
    pub elite_count: usize,
    /// Number of agents that die each generation
    pub death_count: usize,
    /// Pheromone decay rate
    pub pheromone_decay: f32,
    /// Pheromone deposit amount on success
    pub pheromone_deposit: f32,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            evolution_interval: 100,
            mutation_rate: 0.05,
            crossover_rate: 0.3,
            elite_count: 8,
            death_count: 8,
            pheromone_decay: 0.95,
            pheromone_deposit: 0.1,
        }
    }
}

/// Statistics for a single agent
#[derive(Clone, Debug, Default)]
pub struct AgentStats {
    pub fitness: f32,
    pub correct_count: usize,
    pub total_count: usize,
    pub age: usize,
    pub active_features: usize,
}

/// Swarm-level statistics
#[derive(Clone, Debug, Default)]
pub struct SwarmStats {
    pub generation: usize,
    pub prediction_count: usize,
    pub swarm_accuracy: f32,
    pub best_agent_id: usize,
    pub best_fitness: f32,
    pub pheromone_entropy: f32,
}

/// Swarm agent manager
pub struct SwarmAgents {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    config: SwarmConfig,

    // Kernel functions
    fn_init: CudaFunction,
    fn_predict: CudaFunction,
    fn_consensus: CudaFunction,
    fn_update: CudaFunction,
    fn_evolve: CudaFunction,
    fn_importance: CudaFunction,

    // GPU buffers
    agent_states: CudaSlice<u8>,
    swarm_state: CudaSlice<u8>,
    pheromone: CudaSlice<f32>,

    // Local tracking
    prediction_count: usize,
    generation: usize,
}

impl SwarmAgents {
    /// Create new swarm agents
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &str, config: SwarmConfig) -> Result<Self> {
        let stream = ctx.default_stream();
        let ptx = Ptx::from_file(ptx_path);

        let module = ctx.load_module(ptx)?;

        let fn_init = module.load_function("ve_swarm_init_agents").unwrap();
        let fn_predict = module.load_function("ve_swarm_agent_predict").unwrap();
        let fn_consensus = module.load_function("ve_swarm_consensus").unwrap();
        let fn_update = module.load_function("ve_swarm_update_stats").unwrap();
        let fn_evolve = module.load_function("ve_swarm_evolve").unwrap();
        let fn_importance = module.load_function("ve_swarm_feature_importance").unwrap();

        // Allocate GPU buffers
        // AgentState is ~600 bytes, SwarmState is ~500 bytes
        let agent_states: CudaSlice<u8> = stream.alloc_zeros(N_AGENTS * 600)?;
        let swarm_state: CudaSlice<u8> = stream.alloc_zeros(500)?;
        let pheromone: CudaSlice<f32> = stream.alloc_zeros(N_FEATURES)?;

        let mut agents = Self {
            ctx,
            stream,
            config,
            fn_init,
            fn_predict,
            fn_consensus,
            fn_update,
            fn_evolve,
            fn_importance,
            agent_states,
            swarm_state,
            pheromone,
            prediction_count: 0,
            generation: 0,
        };

        agents.initialize()?;

        Ok(agents)
    }

    /// Initialize swarm with diverse agents
    fn initialize(&mut self) -> Result<()> {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N_AGENTS as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_init)
                .arg(&mut self.agent_states)
                .arg(&mut self.swarm_state)
                .arg(&seed)
                .launch(cfg)?;
        }

        self.ctx.synchronize()?;

        log::info!("Swarm initialized with {} agents", N_AGENTS);

        Ok(())
    }

    /// Get agent predictions for given features
    pub fn predict(
        &self,
        attended_features: &CudaSlice<f32>,  // [125]
        temporal_embedding: &CudaSlice<f32>, // [64]
        reservoir_summary: &CudaSlice<f32>,  // [32]
    ) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
        let mut predictions: CudaSlice<f32> = self.stream.alloc_zeros(N_AGENTS)?;
        let mut confidences: CudaSlice<f32> = self.stream.alloc_zeros(N_AGENTS)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N_AGENTS as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_predict)
                .arg(&self.agent_states)
                .arg(attended_features)
                .arg(temporal_embedding)
                .arg(reservoir_summary)
                .arg(&mut predictions)
                .arg(&mut confidences)
                .arg(&self.swarm_state)
                .launch(cfg)?;
        }

        self.ctx.synchronize()?;

        Ok((predictions, confidences))
    }

    /// Compute swarm consensus prediction
    pub fn consensus(
        &self,
        agent_predictions: &CudaSlice<f32>,
        agent_confidences: &CudaSlice<f32>,
        current_freq: f32,
        current_velocity: f32,
    ) -> Result<(f32, f32)> {
        let mut final_prediction: CudaSlice<f32> = self.stream.alloc_zeros(1)?;
        let mut final_confidence: CudaSlice<f32> = self.stream.alloc_zeros(1)?;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N_AGENTS as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_consensus)
                .arg(agent_predictions)
                .arg(agent_confidences)
                .arg(&self.agent_states)
                .arg(&mut final_prediction)
                .arg(&mut final_confidence)
                .arg(&current_freq)
                .arg(&current_velocity)
                .launch(cfg)?;
        }

        self.ctx.synchronize()?;

        let pred = self.stream.clone_dtoh(&final_prediction)?;
        let conf = self.stream.clone_dtoh(&final_confidence)?;

        Ok((pred[0], conf[0]))
    }

    /// Update agent statistics after observing true label
    pub fn update(&mut self, agent_predictions: &CudaSlice<f32>, true_label: bool) -> Result<()> {
        let label = if true_label { 1i32 } else { 0i32 };
        let threshold = 0.5f32;

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N_AGENTS as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_update)
                .arg(&mut self.agent_states)
                .arg(&mut self.swarm_state)
                .arg(agent_predictions)
                .arg(&label)
                .arg(&threshold)
                .launch(cfg)?;
        }

        self.ctx.synchronize()?;

        self.prediction_count += 1;

        // Evolution check
        if self.prediction_count % self.config.evolution_interval == 0 {
            self.evolve()?;
        }

        Ok(())
    }

    /// Evolve the swarm
    pub fn evolve(&mut self) -> Result<()> {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (N_AGENTS as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_evolve)
                .arg(&mut self.agent_states)
                .arg(&mut self.swarm_state)
                .arg(&seed)
                .launch(cfg)?;
        }

        self.ctx.synchronize()?;

        self.generation += 1;
        log::info!("Swarm evolved to generation {}", self.generation);

        Ok(())
    }

    /// Get feature importance from pheromone trail
    pub fn feature_importance(&self) -> Result<Vec<f32>> {
        let mut importance: CudaSlice<f32> = self.stream.alloc_zeros(N_FEATURES)?;
        let mut usage: CudaSlice<f32> = self.stream.alloc_zeros(N_FEATURES)?;

        let cfg = LaunchConfig {
            grid_dim: ((N_FEATURES as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_importance)
                .arg(&self.agent_states)
                .arg(&self.swarm_state)
                .arg(&mut importance)
                .arg(&mut usage)
                .launch(cfg)?;
        }

        self.ctx.synchronize()?;

        let result = self.stream.clone_dtoh(&importance)?;
        Ok(result)
    }

    /// Get current pheromone trail
    pub fn pheromone(&self) -> Result<Vec<f32>> {
        let result = self.stream.clone_dtoh(&self.pheromone)?;
        Ok(result)
    }

    /// Get current generation
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Get prediction count
    pub fn prediction_count(&self) -> usize {
        self.prediction_count
    }
}

/// Feature groups for analysis
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureGroup {
    Tda,       // 0-47
    Reservoir, // 48-79
    Physics,   // 80-91
    Fitness,   // 92-95
    Cycle,     // 96-100
    Spike,     // 101-108
}

impl FeatureGroup {
    /// Get feature range for this group
    pub fn range(&self) -> std::ops::Range<usize> {
        match self {
            Self::Tda => 0..48,
            Self::Reservoir => 48..80,
            Self::Physics => 80..92,
            Self::Fitness => 92..96,
            Self::Cycle => 96..101,
            Self::Spike => 101..109,
        }
    }

    /// Get group from feature index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0..=47 => Self::Tda,
            48..=79 => Self::Reservoir,
            80..=91 => Self::Physics,
            92..=95 => Self::Fitness,
            96..=100 => Self::Cycle,
            _ => Self::Spike,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tda => "TDA Topology",
            Self::Reservoir => "Reservoir State",
            Self::Physics => "Physics (Electrostatics/Hydrophobicity)",
            Self::Fitness => "Fitness (ddG, Expression)",
            Self::Cycle => "Cycle (Dynamics)",
            Self::Spike => "Spike (LIF Neurons)",
        }
    }
}

/// Compute group-level importance from feature importance
pub fn group_importance(feature_importance: &[f32]) -> Vec<(FeatureGroup, f32)> {
    let groups = [
        FeatureGroup::Tda,
        FeatureGroup::Reservoir,
        FeatureGroup::Physics,
        FeatureGroup::Fitness,
        FeatureGroup::Cycle,
        FeatureGroup::Spike,
    ];

    groups
        .iter()
        .map(|group| {
            let range = group.range();
            let sum: f32 = feature_importance[range.clone()].iter().sum();
            let mean = sum / range.len() as f32;
            (*group, mean)
        })
        .collect()
}
