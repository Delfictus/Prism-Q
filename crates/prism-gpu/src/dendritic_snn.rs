//! Dendritic Spiking Neural Network Reservoir for RL Agent
//!
//! GPU-accelerated LIF (Leaky Integrate-and-Fire) reservoir computing
//! for processing RL feature vectors. Implements the **Feature Adapter Protocol**
//! for high-quality neuromorphic reinforcement learning.
//!
//! ## Architecture (Feature Adapter Protocol)
//!
//! ```text
//! 23 raw features ──┬──→ [Velocity Computation] ──→ 46 expanded features
//!                   │           ↓
//!                   └──→ [Tanh Normalization] ──→ Bounded inputs
//!                              ↓
//!                    [Structured Sparse Topology] ──→ ~10% connectivity
//!                              ↓
//!                    [Adaptive LIF Neurons (512)] ──→ Fast/Medium/Slow
//!                              ↓
//!                    [Filtered Rates] ──→ 512-dim smooth output
//! ```
//!
//! ## Key Features (Feature Adapter Protocol)
//!
//! - **Input Expansion**: 23 raw → 46 (raw + velocity/delta) features
//! - **Tanh Scaling**: Normalizes inputs to [-1, 1] for LIF compatibility
//! - **Structured Sparsity**: Hash-based deterministic connectivity (~10%)
//! - **Adaptive Time Constants**: Fast (5ms) → Slow (50ms) neuron gradient
//! - **Persistent State**: Neurons carry memory between inference steps
//!
//! ## Usage
//!
//! ```rust,no_run
//! use prism_gpu::dendritic_snn::DendriticSNNReservoir;
//! # use cudarc::driver::CudaContext;
//! # fn demo() -> anyhow::Result<()> {
//! # let device = CudaContext::new(0)?;
//! # let features_t0 = vec![0.0f32; 40];
//! # let features_t1 = vec![0.0f32; 40];
//!
//! let mut reservoir = DendriticSNNReservoir::new(device, 512)?;
//! // First call computes velocity as zero
//! let state1 = reservoir.process_features(&features_t0)?;
//! // Subsequent calls compute velocity as (current - previous)
//! let state2 = reservoir.process_features(&features_t1)?; // 512-dim output
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Raw input feature dimension (matches PRISM-Learning with bio-chemistry)
/// Supports: 23 (legacy), 27, 31, 37, or 40 (full bio-chemistry)
pub const INPUT_DIM: usize = 40;

/// Expanded input dimension (raw + velocity features)
pub const EXPANDED_INPUT_DIM: usize = 80;

/// Default reservoir size
pub const DEFAULT_RESERVOIR_SIZE: usize = 512;

/// Maximum supported reservoir size
pub const MAX_RESERVOIR_SIZE: usize = 2048;

/// Number of LIF timesteps per feature processing
pub const DEFAULT_STEPS_PER_INFERENCE: usize = 10;

/// Block size for CUDA kernels
const BLOCK_SIZE: usize = 256;

/// GPU-accelerated Spiking Neural Network Reservoir
///
/// Implements the Feature Adapter Protocol with:
/// - Input expansion (40 raw → 80 with velocities)
/// - Tanh normalization (bounded inputs)
/// - Structured sparse topology
/// - Adaptive time constants
/// - Persistent neuronal state
///
/// Maintains all state (membrane potentials, filtered rates, feature history) in VRAM.
pub struct DendriticSNNReservoir {
    /// CUDA device handle
    context: Arc<CudaContext>,
    /// CUDA stream for async operations
    stream: Arc<CudaStream>,
    /// PTX module
    _module: Arc<CudaModule>,

    // Kernel functions
    init_reservoir: CudaFunction,
    lif_multistep: CudaFunction,
    extract_state: CudaFunction,
    reset_state_fn: CudaFunction,
    compute_q_values: CudaFunction,

    // Persistent GPU buffers (survive across calls)
    input_weights: CudaSlice<f32>, // [reservoir_size × EXPANDED_INPUT_DIM]
    recurrent_weights: CudaSlice<f32>, // [reservoir_size × reservoir_size]
    membrane: CudaSlice<f32>,      // [reservoir_size]
    filtered_rates: CudaSlice<f32>, // [reservoir_size]
    tau_mem_array: CudaSlice<f32>, // [reservoir_size] - adaptive time constants
    neuron_signs: CudaSlice<f32>,  // [reservoir_size] - E/I type (+1/-1)

    // Feature history for velocity computation (CPU-side)
    prev_features: Option<Vec<f32>>, // Previous 23-dim raw features

    /// Reservoir size
    reservoir_size: usize,
    /// Steps per inference
    steps_per_inference: usize,
    /// Whether weights have been initialized
    initialized: bool,
}

impl DendriticSNNReservoir {
    /// Creates a new SNN reservoir with default parameters
    ///
    /// # Arguments
    /// * `context` - CUDA device context
    /// * `reservoir_size` - Number of LIF neurons (default: 512)
    ///
    /// # Returns
    /// Initialized reservoir with random weights
    pub fn new(context: Arc<CudaContext>, reservoir_size: usize) -> Result<Self> {
        Self::new_with_params(context, reservoir_size, DEFAULT_STEPS_PER_INFERENCE)
    }

    /// Creates a new SNN reservoir with custom parameters
    ///
    /// Implements Feature Adapter Protocol with:
    /// - Expanded input weights for 80-dim features (40 raw + 40 velocity)
    /// - Per-neuron adaptive time constants
    /// - Feature history tracking for velocity computation
    ///
    /// # Arguments
    /// * `context` - CUDA device context
    /// * `reservoir_size` - Number of LIF neurons
    /// * `steps_per_inference` - LIF timesteps per feature processing
    pub fn new_with_params(
        context: Arc<CudaContext>,
        reservoir_size: usize,
        steps_per_inference: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            reservoir_size > 0 && reservoir_size <= MAX_RESERVOIR_SIZE,
            "reservoir_size must be in [1, {}], got {}",
            MAX_RESERVOIR_SIZE,
            reservoir_size
        );
        anyhow::ensure!(
            steps_per_inference > 0 && steps_per_inference <= 100,
            "steps_per_inference must be in [1, 100], got {}",
            steps_per_inference
        );

        log::info!(
            "Creating Dendritic SNN Reservoir (Feature Adapter Protocol): size={}, steps={}",
            reservoir_size,
            steps_per_inference
        );
        log::info!(
            "  Input expansion: {} raw → {} (raw + velocity)",
            INPUT_DIM,
            EXPANDED_INPUT_DIM
        );

        let stream = context.default_stream();

        // Load PTX module
        let ptx_path = Self::find_ptx_path()?;
        log::info!("Loading SNN PTX from: {}", ptx_path);

        let module = context
            .load_module(Ptx::from_file(&ptx_path))
            .context("Failed to load Dendritic SNN PTX module")?;

        // Load kernel functions
        let init_reservoir = module.load_function("init_snn_reservoir")?;
        let lif_multistep = module.load_function("lif_multistep")?;
        let extract_state = module.load_function("extract_state")?;
        let reset_state_fn = module.load_function("reset_state")?;
        let compute_q_values = module.load_function("compute_q_values")?;

        // Allocate persistent GPU buffers (Feature Adapter Protocol)
        // Input weights: [reservoir_size × EXPANDED_INPUT_DIM] for 46-dim inputs
        let input_weights: CudaSlice<f32> = stream
            .alloc_zeros(reservoir_size * EXPANDED_INPUT_DIM)
            .context("Failed to allocate input weights")?;

        let recurrent_weights: CudaSlice<f32> = stream
            .alloc_zeros(reservoir_size * reservoir_size)
            .context("Failed to allocate recurrent weights")?;

        let membrane: CudaSlice<f32> = stream
            .alloc_zeros(reservoir_size)
            .context("Failed to allocate membrane potentials")?;

        let filtered_rates: CudaSlice<f32> = stream
            .alloc_zeros(reservoir_size)
            .context("Failed to allocate filtered rates")?;

        // Per-neuron adaptive time constants (initialized in initialize())
        let tau_mem_array: CudaSlice<f32> = stream
            .alloc_zeros(reservoir_size)
            .context("Failed to allocate tau_mem array")?;

        // E/I neuron signs (+1 excitatory, -1 inhibitory)
        let neuron_signs: CudaSlice<f32> = stream
            .alloc_zeros(reservoir_size)
            .context("Failed to allocate neuron_signs array")?;

        let bytes_allocated = (reservoir_size * EXPANDED_INPUT_DIM
            + reservoir_size * reservoir_size
            + reservoir_size * 5)  // membrane, filtered_rates, tau_mem, neuron_signs, output
            * 4;
        log::info!(
            "SNN GPU buffers allocated: {:.2}MB total (E/I balanced reservoir)",
            bytes_allocated as f64 / 1024.0 / 1024.0
        );

        Ok(Self {
            context,
            stream,
            _module: module,
            init_reservoir,
            lif_multistep,
            extract_state,
            reset_state_fn,
            compute_q_values,
            input_weights,
            recurrent_weights,
            membrane,
            filtered_rates,
            tau_mem_array,
            neuron_signs,
            prev_features: None, // Will be set on first process_features call
            reservoir_size,
            steps_per_inference,
            initialized: false,
        })
    }

    /// Finds the PTX file path
    fn find_ptx_path() -> Result<String> {
        let paths = [
            "target/ptx/dendritic_snn_reservoir.ptx",
            "crates/prism-gpu/target/ptx/dendritic_snn_reservoir.ptx",
            "../prism-gpu/target/ptx/dendritic_snn_reservoir.ptx",
        ];

        for path in &paths {
            if std::path::Path::new(path).exists() {
                return Ok(path.to_string());
            }
        }

        // Try OUT_DIR from build
        if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let path = format!("{}/ptx/dendritic_snn_reservoir.ptx", out_dir);
            if std::path::Path::new(&path).exists() {
                return Ok(path);
            }
        }

        Err(anyhow::anyhow!(
            "dendritic_snn_reservoir.ptx not found. Run `cargo build -p prism-gpu --features cuda`"
        ))
    }

    /// Initializes reservoir weights with E/I balanced sparse connectivity
    ///
    /// "Flashbulb Reservoir" initialization:
    /// - 80% Excitatory / 20% Inhibitory neuron balance
    /// - Structured sparse topology (hash-based deterministic connectivity)
    /// - Adaptive time constants per neuron (fast→slow gradient)
    /// - Inhibitory neurons are fast (like PV+ interneurons)
    /// - Velocity features get boosted weights (2.5×)
    ///
    /// Must be called once before processing features. Can be called again
    /// to re-randomize weights (e.g., for different random seeds).
    pub fn initialize(&mut self, seed: u64) -> Result<()> {
        log::info!(
            "Initializing E/I balanced SNN reservoir (Flashbulb) with seed={}",
            seed
        );

        // Calculate total elements to initialize
        let total_weights = self.reservoir_size * EXPANDED_INPUT_DIM  // Input weights (46-dim)
            + self.reservoir_size * self.reservoir_size               // Recurrent weights
            + self.reservoir_size * 4; // membrane, filtered_rates, tau_mem, neuron_signs
        let grid_dim = (total_weights as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.init_reservoir)
                .arg(&self.input_weights)
                .arg(&self.recurrent_weights)
                .arg(&self.membrane)
                .arg(&self.filtered_rates)
                .arg(&self.tau_mem_array)
                .arg(&self.neuron_signs)
                .arg(&(self.reservoir_size as i32))
                .arg(&seed)
                .launch(cfg)
        }
        .context("Failed to launch init_snn_reservoir kernel")?;

        self.context.synchronize()?;
        self.initialized = true;
        self.prev_features = None; // Reset feature history

        let excitatory_count = (self.reservoir_size as f32 * 0.8) as usize;
        let inhibitory_count = self.reservoir_size - excitatory_count;
        log::info!("SNN reservoir initialized (E/I balanced):");
        log::info!("  Excitatory: {} neurons (80%)", excitatory_count);
        log::info!(
            "  Inhibitory: {} neurons (20%, 2× strength)",
            inhibitory_count
        );
        log::info!("  τ_mem: fast(5-10ms) for I, gradient(5-50ms) for E");
        Ok(())
    }

    /// Processes a feature vector and returns the reservoir state
    ///
    /// Feature Adapter Protocol implementation:
    /// 1. Computes velocity features from previous call (delta = current - previous)
    /// 2. Expands 40-dim input to 80-dim (raw + velocity)
    /// 3. Applies tanh normalization in kernel
    /// 4. Runs LIF with adaptive time constants
    /// 5. Returns filtered firing rates (persistent across calls)
    ///
    /// # Arguments
    /// * `features` - 40-dimensional raw feature vector (with bio-chemistry)
    ///
    /// # Returns
    /// Reservoir state vector (size = reservoir_size)
    pub fn process_features(&mut self, features: &[f32]) -> Result<Vec<f32>> {
        anyhow::ensure!(
            features.len() == INPUT_DIM,
            "Expected {} raw features, got {}",
            INPUT_DIM,
            features.len()
        );

        if !self.initialized {
            self.initialize(42)?; // Default seed if not initialized
        }

        // =====================================================================
        // FEATURE ADAPTER PROTOCOL: Compute velocity and expand to 80-dim
        // =====================================================================
        let mut expanded_features = vec![0.0f32; EXPANDED_INPUT_DIM];

        // First 40 dims: raw features
        expanded_features[..INPUT_DIM].copy_from_slice(features);

        // Next 40 dims: velocity features (current - previous)
        if let Some(ref prev) = self.prev_features {
            for i in 0..INPUT_DIM {
                expanded_features[INPUT_DIM + i] = features[i] - prev[i];
            }
        } else {
            // First call: velocity is zero
            for i in 0..INPUT_DIM {
                expanded_features[INPUT_DIM + i] = 0.0;
            }
        }

        // Store current features for next velocity computation
        self.prev_features = Some(features.to_vec());

        // Copy expanded features to GPU
        let features_device: CudaSlice<f32> = self
            .stream
            .clone_htod(&expanded_features)
            .context("Failed to copy expanded features to GPU")?;

        // Run LIF multistep kernel with adaptive time constants
        let grid_dim = (self.reservoir_size as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: (self.reservoir_size * 4) as u32, // Shared memory for rates
        };

        unsafe {
            self.stream
                .launch_builder(&self.lif_multistep)
                .arg(&features_device)
                .arg(&self.membrane)
                .arg(&self.filtered_rates)
                .arg(&self.input_weights)
                .arg(&self.recurrent_weights)
                .arg(&self.tau_mem_array)
                .arg(&(self.reservoir_size as i32))
                .arg(&(self.steps_per_inference as i32))
                .launch(cfg)
        }
        .context("Failed to launch lif_multistep kernel")?;

        // Extract filtered rates to output buffer
        let mut output: CudaSlice<f32> = self
            .stream
            .alloc_zeros(self.reservoir_size)
            .context("Failed to allocate output buffer")?;

        unsafe {
            self.stream
                .launch_builder(&self.extract_state)
                .arg(&self.filtered_rates)
                .arg(&output)
                .arg(&(self.reservoir_size as i32))
                .launch(cfg)
        }
        .context("Failed to launch extract_state kernel")?;

        // Copy result back to host
        let state = self
            .stream
            .clone_dtoh(&output)
            .context("Failed to copy state from GPU")?;

        Ok(state)
    }

    /// Resets reservoir state for new episode
    ///
    /// Clears membrane potentials, filtered rates, AND feature history.
    /// Preserves weights and adaptive time constants.
    ///
    /// NOTE: Per Feature Adapter Protocol, this should only be called at
    /// episode boundaries, not between steps within an episode. The persistent
    /// state is critical for capturing temporal dynamics.
    pub fn reset_state(&mut self) -> Result<()> {
        let grid_dim = (self.reservoir_size as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        unsafe {
            self.stream
                .launch_builder(&self.reset_state_fn)
                .arg(&self.membrane)
                .arg(&self.filtered_rates)
                .arg(&(self.reservoir_size as i32))
                .arg(&seed)
                .launch(cfg)
        }
        .context("Failed to launch reset_state kernel")?;

        // Clear feature history for velocity computation
        self.prev_features = None;

        self.context.synchronize()?;
        Ok(())
    }

    /// Clears only feature history without resetting neuronal state
    ///
    /// Use this when you want to break the velocity computation chain
    /// but preserve the reservoir's learned temporal state.
    pub fn clear_feature_history(&mut self) {
        self.prev_features = None;
    }

    /// Computes Q-values directly on GPU (optional optimization)
    ///
    /// For small action spaces, CPU matmul is fine. For larger spaces,
    /// this can compute Q = W_out @ state entirely on GPU.
    ///
    /// # Arguments
    /// * `output_weights` - Readout weight matrix [num_actions × reservoir_size]
    /// * `num_actions` - Number of output actions
    ///
    /// # Returns
    /// Q-values for each action
    pub fn compute_q_values_gpu(
        &self,
        output_weights: &[f32],
        num_actions: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(
            output_weights.len() == num_actions * self.reservoir_size,
            "output_weights size mismatch: expected {}, got {}",
            num_actions * self.reservoir_size,
            output_weights.len()
        );

        // Copy weights to GPU
        let weights_device: CudaSlice<f32> = self
            .stream
            .clone_htod(output_weights)
            .context("Failed to copy output weights to GPU")?;

        // Allocate output buffer
        let mut q_values_device: CudaSlice<f32> = self
            .stream
            .alloc_zeros(num_actions)
            .context("Failed to allocate Q-values buffer")?;

        let grid_dim = (num_actions as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.compute_q_values)
                .arg(&self.filtered_rates)
                .arg(&weights_device)
                .arg(&q_values_device)
                .arg(&(self.reservoir_size as i32))
                .arg(&(num_actions as i32))
                .launch(cfg)
        }
        .context("Failed to launch compute_q_values kernel")?;

        // Copy result back
        let q_values = self
            .stream
            .clone_dtoh(&q_values_device)
            .context("Failed to copy Q-values from GPU")?;

        Ok(q_values)
    }

    /// Returns the reservoir size
    pub fn reservoir_size(&self) -> usize {
        self.reservoir_size
    }

    /// Returns reference to CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Returns reference to CUDA stream
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_snn_reservoir_basic() {
        env_logger::builder().is_test(true).try_init().ok();

        let context = CudaContext::new(0).expect("CUDA not available");
        let mut reservoir =
            DendriticSNNReservoir::new(context, 512).expect("Failed to create reservoir");

        reservoir.initialize(42).expect("Failed to initialize");

        // Create dummy features
        let features = vec![0.5f32; INPUT_DIM];
        let state = reservoir
            .process_features(&features)
            .expect("Failed to process");

        assert_eq!(state.len(), 512);
        // Verify values are in reasonable range (filtered rates should be ~[0, 1])
        for &val in &state {
            assert!(val >= 0.0 && val <= 1.0, "State value {} out of range", val);
        }
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_snn_reset() {
        let context = CudaContext::new(0).expect("CUDA not available");
        let mut reservoir =
            DendriticSNNReservoir::new(context, 256).expect("Failed to create reservoir");

        reservoir.initialize(42).expect("Failed to initialize");

        // Process some features
        let features = vec![1.0f32; INPUT_DIM];
        let _ = reservoir
            .process_features(&features)
            .expect("Failed to process");

        // Reset
        reservoir.reset_state().expect("Failed to reset");

        // State should be near zero after reset
        let features_zero = vec![0.0f32; INPUT_DIM];
        let state = reservoir
            .process_features(&features_zero)
            .expect("Failed to process");

        let sum: f32 = state.iter().sum();
        assert!(sum < 50.0, "State sum {} too high after reset", sum);
    }
}
