//! Dendritic Reservoir GPU Acceleration for Phase 0 Warmstart
//!
//! ASSUMPTIONS:
//! - Input graph: Adjacency list representation
//! - Output: Difficulty[n] and uncertainty[n] vectors (f32, normalized to [0,1])
//! - MAX_VERTICES = 100,000 (enforced at runtime)
//! - Memory layout: Row-major for state matrix (n × branches)
//! - Default branches: 8 dendritic branches per vertex
//! - Default leak rate: 0.1 (temporal dynamics parameter)
//! - Default iterations: 50 (propagation steps)
//! - Block size: 256 threads (defined in CUDA kernel)
//! - Requires: CUDA compute capability sm_86 (RTX 3060)
//!
//! ALGORITHM:
//! 1. Convert adjacency list to CSR format (row_ptr, col_idx)
//! 2. Initialize reservoir state with random weights on GPU
//! 3. Propagate graph structure through dendritic branches (N iterations)
//! 4. Compute difficulty (mean activation) and uncertainty (variance)
//! 5. Copy results back to host
//!
//! DENDRITIC RESERVOIR THEORY:
//! - Multi-branch neuromorphic model inspired by biological neurons
//! - Difficulty: High activation indicates structural complexity (hard to color)
//! - Uncertainty: High variance indicates exploration need (unpredictable behavior)
//! - Used as prior for warmstart softmax distribution in Phase 0
//!
//! PERFORMANCE TARGETS:
//! - DSJC250 (250 vertices, 50 iterations): < 100ms end-to-end
//! - H2D/D2H transfer: < 5% of total time
//! - GPU utilization: > 70%
//!
//! SECURITY:
//! - Validates PTX module loading
//! - Checks for CUDA errors after each operation
//! - Enforces MAX_VERTICES and MAX_BRANCHES limits
//!
//! REFERENCE: PRISM GPU Plan §4.1 (Phase 0 Dendritic Reservoir Kernel)

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Maximum supported graph size (enforced at runtime)
const MAX_VERTICES: usize = 100_000;

/// Maximum number of dendritic branches
const MAX_BRANCHES: usize = 32;

/// Default number of dendritic branches per vertex
const DEFAULT_BRANCHES: usize = 8;

/// Default leak rate for temporal dynamics
const DEFAULT_LEAK_RATE: f32 = 0.1;

/// Default number of propagation iterations
const DEFAULT_ITERATIONS: usize = 50;

/// Block size for CUDA kernels (threads per block)
const BLOCK_SIZE: usize = 256;

/// GPU-accelerated dendritic reservoir for Phase 0 warmstart
///
/// Maintains CUDA device context and compiled PTX module.
/// Thread-safe via Arc<CudaContext>.
pub struct DendriticReservoirGpu {
    /// CUDA device handle
    context: Arc<CudaContext>,
    /// CUDA stream
    stream: Arc<CudaStream>,
    /// PTX module
    _module: Arc<CudaModule>,
    /// Kernel functions
    init_reservoir: CudaFunction,
    propagate_dendritic: CudaFunction,
    compute_metrics_combined: CudaFunction,
    /// Number of dendritic branches per vertex
    num_branches: usize,
    /// Leak rate for temporal dynamics [0, 1]
    leak_rate: f32,
    /// Number of propagation iterations
    iterations: usize,
}

impl DendriticReservoirGpu {
    /// Creates a new GPU dendritic reservoir
    ///
    /// # Arguments
    /// * `device` - CUDA device handle
    /// * `ptx_path` - Path to compiled PTX module
    ///
    /// # Errors
    /// Returns error if:
    /// - PTX module fails to load
    /// - Required kernel functions not found in module
    /// - CUDA device initialization fails
    ///
    /// # Example
    /// ```rust,no_run
    /// use cudarc::driver::CudaContext;
    /// use prism_gpu::dendritic_reservoir::DendriticReservoirGpu;
    /// use std::sync::Arc;
    ///
    /// let device = CudaContext::new(0).unwrap();
    /// let reservoir = DendriticReservoirGpu::new(
    ///     device,
    ///     "kernels/dendritic_reservoir.ptx"
    /// ).unwrap();
    /// ```
    pub fn new(context: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        log::info!("Loading Dendritic Reservoir PTX module from: {}", ptx_path);

        let stream = context.default_stream();

        // Load PTX module
        let module = context
            .load_module(Ptx::from_file(ptx_path))
            .context("Failed to load Dendritic Reservoir PTX module")?;

        // Load kernel functions
        let init_reservoir = module.load_function("init_reservoir")?;
        let propagate_dendritic = module.load_function("propagate_dendritic")?;
        let compute_metrics_combined = module.load_function("compute_metrics_combined")?;

        log::info!("Dendritic Reservoir GPU module loaded successfully");

        Ok(Self {
            context,
            stream,
            _module: module,
            init_reservoir,
            propagate_dendritic,
            compute_metrics_combined,
            num_branches: DEFAULT_BRANCHES,
            leak_rate: DEFAULT_LEAK_RATE,
            iterations: DEFAULT_ITERATIONS,
        })
    }

    /// Creates reservoir with custom parameters
    ///
    /// # Arguments
    /// * `device` - CUDA device handle
    /// * `ptx_path` - Path to compiled PTX module
    /// * `num_branches` - Number of dendritic branches (must be <= MAX_BRANCHES)
    /// * `leak_rate` - Leak rate for temporal dynamics [0, 1]
    /// * `iterations` - Number of propagation iterations
    ///
    /// # Errors
    /// Returns error if:
    /// - PTX loading fails
    /// - Parameters out of valid range
    pub fn new_with_params(
        context: Arc<CudaContext>,
        ptx_path: &str,
        num_branches: usize,
        leak_rate: f32,
        iterations: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            num_branches > 0 && num_branches <= MAX_BRANCHES,
            "num_branches must be in range [1, {}], got {}",
            MAX_BRANCHES,
            num_branches
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&leak_rate),
            "leak_rate must be in range [0, 1], got {}",
            leak_rate
        );
        anyhow::ensure!(
            iterations > 0,
            "iterations must be positive, got {}",
            iterations
        );

        let mut reservoir = Self::new(context, ptx_path)?;
        reservoir.num_branches = num_branches;
        reservoir.leak_rate = leak_rate;
        reservoir.iterations = iterations;

        log::info!(
            "Dendritic Reservoir configured: branches={}, leak_rate={}, iterations={}",
            num_branches,
            leak_rate,
            iterations
        );

        Ok(reservoir)
    }

    /// Computes difficulty and uncertainty metrics from graph structure
    ///
    /// # Arguments
    /// * `adjacency` - Adjacency list representation (vertex -> neighbors)
    /// * `num_vertices` - Total number of vertices in graph
    ///
    /// # Returns
    /// Tuple of (difficulty, uncertainty) vectors:
    /// - difficulty[i]: Coloring hardness for vertex i [0, 1]
    /// - uncertainty[i]: Exploration need for vertex i [0, 1]
    ///
    /// # Errors
    /// Returns error if:
    /// - num_vertices exceeds MAX_VERTICES
    /// - GPU memory allocation fails
    /// - Kernel launch fails
    /// - Data transfer fails
    ///
    /// # Performance
    /// - Time complexity: O(iterations × edges) on GPU
    /// - Space complexity: O(n × branches) for state matrix
    /// - Target: 250 vertices in < 100ms on RTX 3060
    ///
    /// # Example
    /// ```rust,no_run
    /// # use prism_gpu::dendritic_reservoir::DendriticReservoirGpu;
    /// # use cudarc::driver::CudaContext;
    /// # use std::sync::Arc;
    /// # let device = CudaContext::new(0).unwrap();
    /// # let reservoir = DendriticReservoirGpu::new(
    /// #     device,
    /// #     "kernels/dendritic_reservoir.ptx"
    /// # ).unwrap();
    /// let adjacency = vec![
    ///     vec![1, 2],  // vertex 0 -> neighbors 1, 2
    ///     vec![0, 2],  // vertex 1 -> neighbors 0, 2
    ///     vec![0, 1],  // vertex 2 -> neighbors 0, 1
    /// ];
    /// let (difficulty, uncertainty) = reservoir
    ///     .compute_metrics(&adjacency, 3)
    ///     .unwrap();
    /// assert_eq!(difficulty.len(), 3);
    /// assert_eq!(uncertainty.len(), 3);
    /// ```
    pub fn compute_metrics(
        &self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // Validation
        anyhow::ensure!(num_vertices > 0, "Graph must have at least one vertex");
        anyhow::ensure!(
            num_vertices <= MAX_VERTICES,
            "Graph exceeds MAX_VERTICES limit: {} > {}",
            num_vertices,
            MAX_VERTICES
        );
        anyhow::ensure!(
            adjacency.len() == num_vertices,
            "Adjacency list size mismatch: expected {}, got {}",
            num_vertices,
            adjacency.len()
        );

        log::info!(
            "Computing dendritic reservoir metrics for graph with {} vertices",
            num_vertices
        );

        // Step 1: Convert adjacency list to CSR format
        let (row_ptr, col_idx) = self.adjacency_to_csr(adjacency, num_vertices);

        // Step 2: Allocate and initialize GPU memory
        log::debug!("Allocating GPU memory for reservoir state");
        let state_size = num_vertices * self.num_branches;

        // Allocate state buffers (double-buffered for ping-pong)
        let mut state_a: CudaSlice<f32> = self
            .stream
            .alloc_zeros(state_size)
            .context("Failed to allocate state buffer A")?;
        let mut state_b: CudaSlice<f32> = self
            .stream
            .alloc_zeros(state_size)
            .context("Failed to allocate state buffer B")?;

        // Copy CSR data to GPU
        let row_ptr_device: CudaSlice<i32> = self
            .stream
            .clone_htod(&row_ptr)
            .context("Failed to copy row_ptr to GPU")?;
        let col_idx_device: CudaSlice<i32> = self
            .stream
            .clone_htod(&col_idx)
            .context("Failed to copy col_idx to GPU")?;

        // Step 3: Initialize reservoir state
        self.initialize_state(&mut state_a, num_vertices)?;

        // Step 4: Propagate through iterations (ping-pong between buffers)
        for iter in 0..self.iterations {
            if iter % 2 == 0 {
                self.propagate_step(
                    &mut state_b,
                    &state_a,
                    &row_ptr_device,
                    &col_idx_device,
                    num_vertices,
                )?;
            } else {
                self.propagate_step(
                    &mut state_a,
                    &state_b,
                    &row_ptr_device,
                    &col_idx_device,
                    num_vertices,
                )?;
            }

            // Log progress every 10% of iterations
            if iter % (self.iterations / 10).max(1) == 0 {
                log::debug!(
                    "Dendritic propagation progress: {}/{} iterations",
                    iter + 1,
                    self.iterations
                );
            }
        }

        // Final state is in state_a if iterations is even, else state_b
        let final_state = if self.iterations.is_multiple_of(2) {
            &state_b
        } else {
            &state_a
        };

        // Step 5: Compute difficulty and uncertainty from final state
        let (difficulty, uncertainty) =
            self.compute_metrics_from_state(final_state, num_vertices)?;

        log::info!("Dendritic reservoir computation completed successfully");

        Ok((difficulty, uncertainty))
    }

    /// Converts adjacency list to CSR (Compressed Sparse Row) format
    ///
    /// CSR format stores:
    /// - row_ptr[i]: Start index of neighbors for vertex i
    /// - col_idx[row_ptr[i]..row_ptr[i+1]]: Neighbors of vertex i
    ///
    /// Efficient for GPU memory access patterns.
    fn adjacency_to_csr(
        &self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
    ) -> (Vec<i32>, Vec<i32>) {
        let mut row_ptr = Vec::with_capacity(num_vertices + 1);
        let mut col_idx = Vec::new();

        row_ptr.push(0);
        for neighbors in adjacency {
            for &neighbor in neighbors {
                if neighbor < num_vertices {
                    col_idx.push(neighbor as i32);
                }
            }
            row_ptr.push(col_idx.len() as i32);
        }

        log::debug!(
            "CSR conversion: {} vertices, {} edges",
            num_vertices,
            col_idx.len()
        );

        (row_ptr, col_idx)
    }

    /// Initializes reservoir state with random weights
    fn initialize_state(&self, state: &mut CudaSlice<f32>, num_vertices: usize) -> Result<()> {
        let state_size = num_vertices * self.num_branches;
        let grid_dim = (state_size as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Use current time as seed for randomness
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        log::debug!("Initializing reservoir state with seed={}", seed);

        unsafe {
            self.stream
                .launch_builder(&self.init_reservoir)
                .arg(state)
                .arg(&(num_vertices as i32))
                .arg(&(self.num_branches as i32))
                .arg(&seed)
                .launch(cfg)
        }
        .context("Failed to launch init_reservoir kernel")?;

        self.context
            .synchronize()
            .context("Initialization synchronization failed")?;

        Ok(())
    }

    /// Executes one propagation step
    fn propagate_step(
        &self,
        state_out: &mut CudaSlice<f32>,
        state_in: &CudaSlice<f32>,
        row_ptr: &CudaSlice<i32>,
        col_idx: &CudaSlice<i32>,
        num_vertices: usize,
    ) -> Result<()> {
        let grid_dim = (num_vertices as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.propagate_dendritic)
                .arg(state_out)
                .arg(state_in)
                .arg(row_ptr)
                .arg(col_idx)
                .arg(&(num_vertices as i32))
                .arg(&(self.num_branches as i32))
                .arg(&self.leak_rate)
                .launch(cfg)
        }
        .context("Failed to launch propagate_dendritic kernel")?;

        self.context
            .synchronize()
            .context("Propagation synchronization failed")?;

        Ok(())
    }

    /// Computes difficulty and uncertainty from final reservoir state
    fn compute_metrics_from_state(
        &self,
        state: &CudaSlice<f32>,
        num_vertices: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        log::debug!("Computing metrics from final reservoir state");

        // Allocate output buffers on GPU
        let mut difficulty_device: CudaSlice<f32> = self
            .stream
            .alloc_zeros(num_vertices)
            .context("Failed to allocate difficulty buffer")?;
        let mut uncertainty_device: CudaSlice<f32> = self
            .stream
            .alloc_zeros(num_vertices)
            .context("Failed to allocate uncertainty buffer")?;

        let grid_dim = (num_vertices as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.compute_metrics_combined)
                .arg(state)
                .arg(&mut difficulty_device)
                .arg(&mut uncertainty_device)
                .arg(&(num_vertices as i32))
                .arg(&(self.num_branches as i32))
                .launch(cfg)
        }
        .context("Failed to launch compute_metrics_combined kernel")?;

        self.context
            .synchronize()
            .context("Metrics computation synchronization failed")?;

        // Copy results back to host
        log::debug!("Copying metrics back to host");
        let difficulty = self
            .stream
            .clone_dtoh(&difficulty_device)
            .context("Failed to copy difficulty from GPU")?;
        let uncertainty = self
            .stream
            .clone_dtoh(&uncertainty_device)
            .context("Failed to copy uncertainty from GPU")?;

        Ok((difficulty, uncertainty))
    }

    /// Returns reference to underlying CUDA device
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Returns number of dendritic branches
    pub fn num_branches(&self) -> usize {
        self.num_branches
    }

    /// Returns leak rate parameter
    pub fn leak_rate(&self) -> f32 {
        self.leak_rate
    }

    /// Returns number of propagation iterations
    pub fn iterations(&self) -> usize {
        self.iterations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a simple test graph for unit testing
    fn create_test_graph() -> (Vec<Vec<usize>>, usize) {
        // Complete graph K5 (5 vertices, all-to-all connections)
        let adjacency = vec![
            vec![1, 2, 3, 4], // 0 -> all others
            vec![0, 2, 3, 4], // 1 -> all others
            vec![0, 1, 3, 4], // 2 -> all others
            vec![0, 1, 2, 4], // 3 -> all others
            vec![0, 1, 2, 3], // 4 -> all others
        ];
        (adjacency, 5)
    }

    #[test]
    #[ignore] // Requires GPU hardware
    fn test_dendritic_reservoir_small_graph() {
        env_logger::builder().is_test(true).try_init().ok();

        let device = CudaContext::new(0).expect("CUDA device not available");
        let reservoir = DendriticReservoirGpu::new(device, "target/ptx/dendritic_reservoir.ptx")
            .expect("Failed to create DendriticReservoirGpu");

        let (adjacency, num_vertices) = create_test_graph();
        let (difficulty, uncertainty) = reservoir
            .compute_metrics(&adjacency, num_vertices)
            .expect("Metrics computation failed");

        // Verify output dimensions
        assert_eq!(difficulty.len(), num_vertices);
        assert_eq!(uncertainty.len(), num_vertices);

        // Verify values are in [0, 1] range
        for i in 0..num_vertices {
            assert!(
                difficulty[i] >= 0.0 && difficulty[i] <= 1.0,
                "difficulty[{}] = {} out of range",
                i,
                difficulty[i]
            );
            assert!(
                uncertainty[i] >= 0.0 && uncertainty[i] <= 1.0,
                "uncertainty[{}] = {} out of range",
                i,
                uncertainty[i]
            );
        }

        // For complete graph K5, all vertices should have similar difficulty
        let mean_difficulty: f32 = difficulty.iter().sum::<f32>() / num_vertices as f32;
        for &d in &difficulty {
            assert!(
                (d - mean_difficulty).abs() < 0.2,
                "Difficulty variance too high for symmetric graph"
            );
        }

        log::info!("Test passed: difficulty={:?}", difficulty);
        log::info!("Test passed: uncertainty={:?}", uncertainty);
    }

    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn test_validation_max_vertices() {
        let device = CudaContext::new(0).expect("CUDA device not available");
        let reservoir = DendriticReservoirGpu::new(device, "target/ptx/dendritic_reservoir.ptx")
            .expect("Failed to create DendriticReservoirGpu");

        let large_adjacency = vec![vec![]; MAX_VERTICES + 1];
        let result = reservoir.compute_metrics(&large_adjacency, MAX_VERTICES + 1);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("MAX_VERTICES"));
    }

    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn test_validation_empty_graph() {
        let device = CudaContext::new(0).expect("CUDA device not available");
        let reservoir = DendriticReservoirGpu::new(device, "target/ptx/dendritic_reservoir.ptx")
            .expect("Failed to create DendriticReservoirGpu");

        let result = reservoir.compute_metrics(&[], 0);
        assert!(result.is_err());
    }

    #[ignore = "GPU_RUNTIME_REQUIRED: needs real GPU + built PTX; run on the RTX workstation via `cargo test -- --ignored`"]
    #[test]
    fn test_validation_custom_params() {
        let device = CudaContext::new(0).expect("CUDA device not available");

        // Valid parameters
        let result = DendriticReservoirGpu::new_with_params(
            device.clone(),
            "target/ptx/dendritic_reservoir.ptx",
            16,
            0.2,
            100,
        );
        assert!(result.is_ok());

        // Invalid num_branches
        let result = DendriticReservoirGpu::new_with_params(
            device.clone(),
            "target/ptx/dendritic_reservoir.ptx",
            0,
            0.2,
            100,
        );
        assert!(result.is_err());

        // Invalid leak_rate
        let result = DendriticReservoirGpu::new_with_params(
            device.clone(),
            "target/ptx/dendritic_reservoir.ptx",
            16,
            1.5,
            100,
        );
        assert!(result.is_err());

        // Invalid iterations
        let result = DendriticReservoirGpu::new_with_params(
            device,
            "target/ptx/dendritic_reservoir.ptx",
            16,
            0.2,
            0,
        );
        assert!(result.is_err());
    }
}
