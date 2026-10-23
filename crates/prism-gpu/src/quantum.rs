//! Quantum Evolution GPU Acceleration for Phase 3
//!
//! ASSUMPTIONS:
//! - Input graph: Adjacency list or matrix representation
//! - Output: ColoringSolution with color assignments and conflict count
//! - MAX_VERTICES = 10,000 (enforced at runtime)
//! - MAX_COLORS = 64 (enforced at runtime, matches CUDA kernel limit)
//! - Memory layout: Row-major for coalesced GPU access
//! - Default evolution time: 1.0 (radians)
//! - Default coupling strength: 1.0 (normalized energy scale)
//! - Block size: 256 threads (defined in CUDA kernel)
//! - Requires: CUDA compute capability sm_86 (RTX 3060)
//!
//! ALGORITHM:
//! 1. Convert adjacency list to dense matrix (if needed)
//! 2. Initialize probability amplitudes in equal superposition
//! 3. Execute quantum evolution kernel (Trotterized Hamiltonian)
//! 4. Execute measurement kernel (collapse to color assignment)
//! 5. Validate coloring and compute conflicts
//! 6. Copy results back to host
//!
//! QUANTUM-INSPIRED MODEL:
//! - Not true quantum computing (classical GPU simulation)
//! - Explores color space via superposition metaphor
//! - Conflict energy drives evolution dynamics
//! - Measurement collapses to deterministic or stochastic color
//!
//! PERFORMANCE TARGETS:
//! - DSJC500 (500 vertices): < 500ms end-to-end
//! - DSJC1000 (1000 vertices): < 2 seconds end-to-end
//! - H2D/D2H transfer: < 10% of total time
//! - GPU utilization: > 70%
//!
//! SECURITY:
//! - Validates PTX module loading
//! - Checks for CUDA errors after each operation
//! - Enforces MAX_VERTICES and MAX_COLORS limits
//! - Bounds checking on all host-side operations
//!
//! REFERENCE: PRISM GPU Plan §4.3 (Phase 3 Quantum Kernel)

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, DeviceSlice};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Maximum supported graph size (enforced at runtime)
const MAX_VERTICES: usize = 10_000;

/// Maximum number of colors (matches CUDA kernel stack allocation limit)
const MAX_COLORS: usize = 64;

/// Block size for CUDA kernels (threads per block)
const BLOCK_SIZE: usize = 256;

/// Default evolution time parameter (radians)
const DEFAULT_EVOLUTION_TIME: f32 = 1.0;

/// Default coupling strength
const DEFAULT_COUPLING_STRENGTH: f32 = 1.0;

/// GPU-accelerated quantum evolution for Phase 3 coloring
///
/// Maintains CUDA device context and compiled PTX module.
/// Thread-safe via Arc<CudaContext>.
///
/// STAGE 3 ENHANCEMENTS:
/// - Complex-valued quantum amplitudes (real + imaginary components)
/// - Multi-iteration annealing with configurable schedule
/// - Stochastic measurement using GPU RNG states
/// - Telemetry for coherence, purity, entanglement tracking
pub struct QuantumEvolutionGpu {
    /// CUDA device handle
    context: Arc<CudaContext>,

    /// CUDA stream for operations
    stream: Arc<CudaStream>,

    /// Loaded PTX module
    module: Arc<CudaModule>,

    /// Evolution time parameter (controls exploration)
    evolution_time: f32,

    /// Coupling strength (controls conflict penalty)
    coupling_strength: f32,

    /// Quantum purity metric (updated during evolution)
    purity: f32,

    /// Quantum entanglement metric (updated during evolution)
    entanglement: f32,

    /// Amplitude variance (Stage 5: complex quantum telemetry)
    amplitude_variance: f32,

    /// Phase coherence (Stage 5: complex quantum telemetry)
    coherence: f32,
}

impl QuantumEvolutionGpu {
    /// Creates a new GPU quantum evolution solver
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
    /// use prism_gpu::quantum::QuantumEvolutionGpu;
    /// use std::sync::Arc;
    ///
    /// let device = CudaContext::new(0).unwrap());
    /// let quantum = QuantumEvolutionGpu::new(Arc::new(device), "kernels/quantum.ptx").unwrap());
    /// ```
    pub fn new(context: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        log::info!("Loading Quantum Evolution PTX module from: {}", ptx_path);

        // Get default stream
        let stream = context.default_stream();

        // Load PTX module
        let module = context.load_module(Ptx::from_file(ptx_path))
            .context("Failed to load Quantum Evolution PTX module")?;

        // Verify kernel functions exist
        let required_kernels = [
            "quantum_evolve_kernel",
            "quantum_measure_kernel",
            "quantum_evolve_measure_fused_kernel",
            "init_amplitudes_kernel",
            "quantum_evolve_complex_kernel",
            "init_complex_amplitudes_kernel",
            "quantum_measure_stochastic_kernel",
            "init_rng_states_kernel",
        ];

        for kernel_name in &required_kernels {
            module.load_function(kernel_name)
                .with_context(|| format!("Failed to find kernel function: {}", kernel_name))?;
        }

        log::info!("Quantum Evolution GPU module loaded successfully");

        Ok(Self {
            context,
            stream,
            module,
            evolution_time: DEFAULT_EVOLUTION_TIME,
            coupling_strength: DEFAULT_COUPLING_STRENGTH,
            purity: 1.0,             // Initially pure state
            entanglement: 0.0,       // Initially unentangled
            amplitude_variance: 0.0, // Updated after complex evolution
            coherence: 1.0,          // Initially fully coherent
        })
    }

    /// Creates quantum solver with custom parameters
    ///
    /// # Arguments
    /// * `device` - CUDA device handle
    /// * `ptx_path` - Path to compiled PTX module
    /// * `evolution_time` - Evolution time parameter (radians, > 0)
    /// * `coupling_strength` - Coupling strength (> 0)
    ///
    /// # Errors
    /// Returns error if:
    /// - PTX loading fails
    /// - Parameters out of valid range
    pub fn new_with_params(
        context: Arc<CudaContext>,
        ptx_path: &str,
        evolution_time: f32,
        coupling_strength: f32,
    ) -> Result<Self> {
        anyhow::ensure!(
            evolution_time > 0.0,
            "evolution_time must be positive, got {}",
            evolution_time
        );
        anyhow::ensure!(
            coupling_strength > 0.0,
            "coupling_strength must be positive, got {}",
            coupling_strength
        );

        let mut quantum = Self::new(context, ptx_path)?;
        quantum.evolution_time = evolution_time;
        quantum.coupling_strength = coupling_strength;

        log::info!(
            "Quantum Evolution configured: evolution_time={}, coupling_strength={}",
            evolution_time,
            coupling_strength
        );

        Ok(quantum)
    }

    /// Evolves quantum state and measures coloring
    ///
    /// # Arguments
    /// * `adjacency` - Adjacency list representation (vertex -> neighbors)
    /// * `num_vertices` - Total number of vertices in graph
    /// * `max_colors` - Maximum number of colors to use
    ///
    /// # Returns
    /// Color assignment vector where colors[i] is the color of vertex i.
    /// Color values are in range [0, max_colors).
    ///
    /// # Errors
    /// Returns error if:
    /// - num_vertices exceeds MAX_VERTICES
    /// - max_colors exceeds MAX_COLORS
    /// - GPU memory allocation fails
    /// - Kernel launch fails
    /// - Data transfer fails
    ///
    /// # Performance
    /// - Time complexity: O(iterations × n × max_colors) on GPU
    /// - Space complexity: O(n²) for adjacency matrix + O(n × max_colors) for amplitudes
    /// - Target: 500 vertices in < 500ms on RTX 3060
    ///
    /// # Example
    /// ```rust,no_run
    /// # use prism_gpu::quantum::QuantumEvolutionGpu;
    /// # use cudarc::driver::CudaContext;
    /// # use std::sync::Arc;
    /// # let device = CudaContext::new(0).unwrap());
    /// # let quantum = QuantumEvolutionGpu::new(Arc::new(device), "target/ptx/quantum.ptx").unwrap());
    /// let adjacency = vec![
    ///     vec![1, 2],  // vertex 0 -> neighbors 1, 2
    ///     vec![0, 2],  // vertex 1 -> neighbors 0, 2
    ///     vec![0, 1],  // vertex 2 -> neighbors 0, 1
    /// ];
    /// let colors = quantum.evolve_and_measure(&adjacency, 3, 10).unwrap());
    /// assert_eq!(colors.len(), 3);
    /// ```
    pub fn evolve_and_measure(
        &self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
        max_colors: usize,
    ) -> Result<Vec<usize>> {
        // Validation
        anyhow::ensure!(num_vertices > 0, "Graph must have at least one vertex");
        anyhow::ensure!(
            num_vertices <= MAX_VERTICES,
            "Graph exceeds MAX_VERTICES limit: {} > {}",
            num_vertices,
            MAX_VERTICES
        );
        anyhow::ensure!(
            max_colors > 0 && max_colors <= MAX_COLORS,
            "max_colors must be in range [1, {}], got {}",
            MAX_COLORS,
            max_colors
        );
        anyhow::ensure!(
            adjacency.len() == num_vertices,
            "Adjacency list size mismatch: expected {}, got {}",
            num_vertices,
            adjacency.len()
        );

        log::info!(
            "Computing quantum evolution for graph with {} vertices, {} max colors",
            num_vertices,
            max_colors
        );

        // Step 1: Convert adjacency list to dense matrix
        let adjacency_matrix = self.adjacency_to_matrix(adjacency, num_vertices);

        // Step 2: Prepare coupling strengths (uniform for all vertices)
        let couplings = vec![self.coupling_strength; num_vertices];

        // Step 3: Allocate and initialize GPU memory
        log::debug!("Allocating GPU memory");

        let d_adjacency = self
            .stream
            .clone_htod(&adjacency_matrix)
            .context("Failed to copy adjacency matrix to GPU")?;

        let d_couplings = self
            .stream
            .clone_htod(&couplings)
            .context("Failed to copy couplings to GPU")?;

        // Initialize amplitudes in equal superposition
        let amplitude_size = num_vertices * max_colors;
        let mut d_amplitudes = self
            .stream
            .alloc_zeros(amplitude_size)
            .context("Failed to allocate amplitudes on GPU")?;

        self.init_amplitudes(&mut d_amplitudes, num_vertices, max_colors)?;

        // Step 4: Execute evolution kernel
        log::debug!("Launching quantum evolution kernel");
        self.launch_evolution_kernel(
            &d_adjacency,
            &mut d_amplitudes,
            &d_couplings,
            num_vertices,
            max_colors,
        )?;

        // Step 5: Execute measurement kernel
        log::debug!("Launching quantum measurement kernel");
        let d_colors = self.launch_measurement_kernel(&d_amplitudes, num_vertices, max_colors)?;

        // Step 6: Copy results back to host
        log::debug!("Copying results back to host");
        let colors_i32 = self
            .stream
            .clone_dtoh(&d_colors)
            .context("Failed to copy colors from GPU")?;

        // Convert i32 to usize
        let colors: Vec<usize> = colors_i32.iter().map(|&c| c as usize).collect();

        log::info!("Quantum evolution completed successfully");
        Ok(colors)
    }

    /// Converts adjacency list to dense matrix (flattened row-major)
    fn adjacency_to_matrix(&self, adjacency: &[Vec<usize>], num_vertices: usize) -> Vec<i32> {
        let mut matrix = vec![0i32; num_vertices * num_vertices];

        for (i, neighbors) in adjacency.iter().enumerate() {
            for &j in neighbors {
                if j < num_vertices {
                    matrix[i * num_vertices + j] = 1;
                }
            }
        }

        log::debug!(
            "Converted adjacency list to dense matrix ({} bytes)",
            matrix.len() * 4
        );

        matrix
    }

    /// Initializes amplitudes in equal superposition state
    fn init_amplitudes(
        &self,
        amplitudes: &mut CudaSlice<f32>,
        num_vertices: usize,
        max_colors: usize,
    ) -> Result<()> {
        let init_func = self.module.load_function("init_amplitudes_kernel")
            .context("Failed to get init_amplitudes_kernel function")?;

        let total_size = num_vertices * max_colors;
        let grid_dim = (total_size as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&init_func)
                .arg(amplitudes)
                .arg(&(num_vertices as i32))
                .arg(&(max_colors as i32))
                .launch(cfg)?
        };

        self.stream
            .synchronize()
            .context("Amplitude initialization synchronization failed")?;

        log::debug!("Amplitudes initialized in equal superposition");
        Ok(())
    }

    // ========================================================================
    // STAGE 3: COMPLEX AMPLITUDE SUPPORT
    // ========================================================================

    /// Initializes complex amplitudes in equal superposition state
    ///
    /// SUBSTEP 1: Complex-valued quantum state initialization
    ///
    /// # Arguments
    /// * `real_amplitudes` - Buffer for real components (size: num_vertices * max_colors)
    /// * `imag_amplitudes` - Buffer for imaginary components (size: num_vertices * max_colors)
    /// * `num_vertices` - Number of vertices in graph
    /// * `max_colors` - Maximum number of colors
    ///
    /// # Memory Layout
    /// - Each buffer: num_vertices * max_colors * 4 bytes (f32)
    /// - Total memory: 2x real-only mode
    /// - Initialization: real = 1/sqrt(max_colors), imag = 0 (equal superposition)
    ///
    /// # Errors
    /// Returns error if:
    /// - GPU memory insufficient (requires 2x real-only mode)
    /// - Kernel launch fails
    /// - Synchronization fails
    ///
    /// # CUDA Kernel Required
    /// Signature: `init_complex_amplitudes_kernel(float* real_amps, float* imag_amps, int num_vertices, int max_colors)`
    /// - Thread per amplitude element
    /// - Each thread computes: real[tid] = 1.0f / sqrtf((float)max_colors), imag[tid] = 0.0f
    /// - Grid: ceil((num_vertices * max_colors) / BLOCK_SIZE)
    ///
    /// # Reference
    /// PRISM Quantum Evolution Stage 3.1
    pub fn init_complex_amplitudes(
        &self,
        real_amplitudes: &mut CudaSlice<f32>,
        imag_amplitudes: &mut CudaSlice<f32>,
        num_vertices: usize,
        max_colors: usize,
    ) -> Result<()> {
        // Validate buffer sizes
        let expected_size = num_vertices * max_colors;
        anyhow::ensure!(
            real_amplitudes.len() == expected_size,
            "Real amplitude buffer size mismatch: expected {}, got {}",
            expected_size,
            real_amplitudes.len()
        );
        anyhow::ensure!(
            imag_amplitudes.len() == expected_size,
            "Imaginary amplitude buffer size mismatch: expected {}, got {}",
            expected_size,
            imag_amplitudes.len()
        );

        // Launch GPU kernel for stochastic initialization with random perturbations
        let init_func = self.module.load_function("init_complex_amplitudes_kernel")
            .context("Failed to get init_complex_amplitudes_kernel function")?;

        // Use dynamic seed for stochastic amplitude initialization
        let dynamic_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345);

        let total_size = num_vertices * max_colors;
        let grid_dim = (total_size as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&init_func)
                .arg(real_amplitudes)
                .arg(imag_amplitudes)
                .arg(&(num_vertices as i32))
                .arg(&(max_colors as i32))
                .arg(&dynamic_seed)
                .launch(cfg)?
        };

        self.stream
            .synchronize()
            .context("Complex amplitude initialization synchronization failed")?;

        let amplitude_value = 1.0 / (max_colors as f32).sqrt();
        log::debug!(
            "Complex amplitudes initialized (GPU kernel with symmetry breaking): {} vertices, {} colors, base_amplitude={:.6}",
            num_vertices,
            max_colors,
            amplitude_value
        );

        Ok(())
    }

    // ========================================================================
    // STAGE 3: RNG INITIALIZATION FOR STOCHASTIC MEASUREMENT
    // ========================================================================

    /// Initializes GPU RNG states for stochastic quantum measurement
    ///
    /// SUBSTEP 2: RNG state buffer allocation and seeding
    ///
    /// # Arguments
    /// * `num_vertices` - Number of vertices (one RNG state per vertex)
    /// * `seed` - Random seed for reproducibility
    ///
    /// # Returns
    /// GPU buffer containing initialized RNG states (curandStatePhilox4_32_10_t)
    ///
    /// # Memory Layout
    /// - RNG state size: 40 bytes per vertex (curandStatePhilox4_32_10_t)
    /// - Total allocation: num_vertices * 40 bytes
    /// - Each state initialized with seed + vertex_id for uniqueness
    ///
    /// # Errors
    /// Returns error if:
    /// - GPU memory allocation fails
    /// - Kernel launch fails
    /// - num_vertices * 40 bytes exceeds available GPU memory
    ///
    /// # CUDA Kernel Required
    /// Signature: `init_rng_states_kernel(curandStatePhilox4_32_10_t* states, unsigned long long seed, int num_vertices)`
    /// - Each thread initializes one RNG state
    /// - Uses curand_init(seed, vertex_id, 0, &states[vertex_id])
    /// - Grid: ceil(num_vertices / BLOCK_SIZE)
    /// - Block: BLOCK_SIZE threads
    ///
    /// # Performance
    /// - Initialization overhead: ~1-5ms for 1000 vertices on RTX 3060
    /// - Amortized over multiple measurements
    /// - One-time cost per quantum evolution session
    ///
    /// # Reference
    /// PRISM Quantum Evolution Stage 3.2
    /// cuRAND documentation: https://docs.nvidia.com/cuda/curand/index.html
    pub fn init_rng_states(&self, num_vertices: usize, seed: u64) -> Result<CudaSlice<u8>> {
        // curandStatePhilox4_32_10_t is 40 bytes
        const RNG_STATE_SIZE: usize = 40;

        let total_bytes = num_vertices * RNG_STATE_SIZE;

        // Memory validation
        anyhow::ensure!(
            num_vertices <= MAX_VERTICES,
            "RNG state allocation exceeds MAX_VERTICES: {} > {}",
            num_vertices,
            MAX_VERTICES
        );

        log::debug!(
            "Allocating RNG states: {} vertices, {} bytes total",
            num_vertices,
            total_bytes
        );

        // Allocate GPU buffer for RNG states
        let mut rng_states = self
            .stream
            .alloc_zeros(total_bytes)
            .context("Failed to allocate RNG states on GPU")?;

        // Initialize RNG states with init_rng_states_kernel
        let init_rng_func = self.module.load_function("init_rng_states_kernel")
            .context("Failed to get init_rng_states_kernel function")?;

        let grid_dim = (num_vertices as u32).div_ceil(BLOCK_SIZE as u32);
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&init_rng_func)
                .arg(&mut rng_states)
                .arg(&seed)
                .arg(&(num_vertices as i32))
                .launch(cfg)?
        };

        self.stream
            .synchronize()
            .context("RNG state initialization synchronization failed")?;

        log::debug!("RNG states allocated and initialized with seed={}", seed);

        Ok(rng_states)
    }

    // ========================================================================
    // STAGE 3: MULTI-ITERATION EVOLUTION WITH ANNEALING
    // ========================================================================

    /// Executes multi-iteration quantum evolution with annealing schedule
    ///
    /// SUBSTEP 3: Complex quantum evolution with gradual classical transition
    ///
    /// # Arguments
    /// * `real_amplitudes` - Real components of quantum state
    /// * `imag_amplitudes` - Imaginary components of quantum state
    /// * `adjacency` - Graph adjacency matrix
    /// * `couplings` - Vertex-specific coupling strengths
    /// * `num_vertices` - Number of vertices
    /// * `max_colors` - Maximum colors
    /// * `config` - Phase3Config with evolution parameters
    ///
    /// # Algorithm
    /// Implements quantum annealing schedule:
    /// ```text
    /// for iter in 0..evolution_iterations:
    ///     t = iter / evolution_iterations  // Annealing progress [0, 1]
    ///     transverse_field = schedule_function(t)
    ///     launch quantum_evolve_complex_kernel(transverse_field, interference_decay)
    /// ```
    ///
    /// # Schedule Types
    /// - "linear": transverse_field * (1 - t)
    /// - "exponential": transverse_field * exp(-3*t)
    /// - "custom": User-defined in future versions
    ///
    /// # Physics Interpretation
    /// - High transverse field: Strong quantum tunneling (exploration)
    /// - Low transverse field: Classical behavior (exploitation)
    /// - Interference decay: Gradual decoherence rate
    ///
    /// # Errors
    /// Returns error if:
    /// - Buffer size mismatches
    /// - Kernel launch fails at any iteration
    /// - Invalid schedule_type specified
    ///
    /// # CUDA Kernel Required
    /// Signature: `quantum_evolve_complex_kernel(
    ///     const int* adjacency,
    ///     float* real_amps,
    ///     float* imag_amps,
    ///     const float* couplings,
    ///     float transverse_field,
    ///     float interference_decay,
    ///     int num_vertices,
    ///     int max_colors
    /// )`
    /// - Implements Trotterized Hamiltonian with complex arithmetic
    /// - Applies transverse field term (quantum fluctuations)
    /// - Applies Ising term (conflict penalty)
    /// - Updates amplitudes: psi(t+dt) = exp(-iH*dt) * psi(t)
    /// - Grid: ceil(num_vertices / BLOCK_SIZE)
    ///
    /// # Performance
    /// - Target: DSJC500 with 100 iterations < 1.5s on RTX 3060
    /// - Linear scaling with evolution_iterations
    /// - Memory bandwidth bottleneck: 2 * num_vertices * max_colors * 4 bytes read/write per iteration
    ///
    /// # Reference
    /// PRISM Quantum Evolution Stage 3.3
    /// Quantum Annealing: D-Wave Systems documentation
    pub fn evolve_complex(
        &mut self,
        real_amplitudes: &mut CudaSlice<f32>,
        imag_amplitudes: &mut CudaSlice<f32>,
        adjacency: &CudaSlice<i32>,
        couplings: &CudaSlice<f32>,
        num_vertices: usize,
        max_colors: usize,
        config: &prism_core::types::Phase3Config,
    ) -> Result<()> {
        // Validate inputs
        let expected_amp_size = num_vertices * max_colors;
        anyhow::ensure!(
            real_amplitudes.len() == expected_amp_size,
            "Real amplitude buffer size mismatch"
        );
        anyhow::ensure!(
            imag_amplitudes.len() == expected_amp_size,
            "Imaginary amplitude buffer size mismatch"
        );

        log::info!(
            "Starting complex quantum evolution: {} iterations, schedule={}",
            config.evolution_iterations,
            config.schedule_type
        );

        // Multi-iteration annealing loop
        for iter in 0..config.evolution_iterations {
            let t = iter as f32 / config.evolution_iterations as f32;

            // Compute annealing schedule
            let transverse_field = match config.schedule_type.as_str() {
                "linear" => config.transverse_field * (1.0 - t),
                "exponential" => config.transverse_field * (-3.0 * t).exp(),
                _ => {
                    log::warn!(
                        "Unknown schedule_type '{}', using constant field",
                        config.schedule_type
                    );
                    config.transverse_field
                }
            };

            // Launch quantum_evolve_complex_kernel
            let evolve_func = self.module.load_function("quantum_evolve_complex_kernel")
                .context("Failed to get quantum_evolve_complex_kernel function")?;

            let grid_dim = (num_vertices as u32).div_ceil(BLOCK_SIZE as u32);
            let cfg = LaunchConfig {
                grid_dim: (grid_dim, 1, 1),
                block_dim: (BLOCK_SIZE as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream.launch_builder(&evolve_func)
                    .arg(adjacency as &CudaSlice<i32>)
                    .arg(real_amplitudes as &CudaSlice<f32>)
                    .arg(imag_amplitudes as &CudaSlice<f32>)
                    .arg(couplings as &CudaSlice<f32>)
                    .arg(&config.evolution_time)
                    .arg(&transverse_field)
                    .arg(&config.interference_decay)
                    .arg(&(num_vertices as i32))
                    .arg(&(max_colors as i32))
                    .launch(cfg)?
            };

            // Log progress every 10 iterations
            if iter % 10 == 0 || iter == config.evolution_iterations - 1 {
                log::debug!(
                    "Evolution iter {}/{}: t={:.3}, transverse_field={:.4}",
                    iter + 1,
                    config.evolution_iterations,
                    t,
                    transverse_field
                );
            }

            // Update telemetry metrics (simplified tracking)
            self.purity = 1.0 - t * config.interference_decay;
            self.entanglement = t * 0.5;
        }

        log::info!("Complex quantum evolution completed");
        Ok(())
    }

    // ========================================================================
    // STAGE 3: STOCHASTIC MEASUREMENT
    // ========================================================================

    /// Launches stochastic quantum measurement kernel
    ///
    /// SUBSTEP 4: Probabilistic collapse to color assignment
    ///
    /// # Arguments
    /// * `real_amplitudes` - Real components of quantum state
    /// * `imag_amplitudes` - Imaginary components of quantum state
    /// * `rng_states` - GPU RNG states (from init_rng_states)
    /// * `num_vertices` - Number of vertices
    /// * `max_colors` - Maximum colors
    ///
    /// # Returns
    /// Color assignment vector (device buffer)
    ///
    /// # Algorithm
    /// For each vertex:
    /// 1. Compute probability distribution: P(color) = |psi_real[color]|^2 + |psi_imag[color]|^2
    /// 2. Normalize probabilities
    /// 3. Sample color using curand_uniform and cumulative distribution
    ///
    /// # Errors
    /// Returns error if:
    /// - Buffer size mismatches
    /// - RNG state buffer invalid
    /// - Kernel launch fails
    ///
    /// # CUDA Kernel Required
    /// Signature: `quantum_measure_stochastic_kernel(
    ///     const float* real_amps,
    ///     const float* imag_amps,
    ///     curandStatePhilox4_32_10_t* rng_states,
    ///     int* colors_out,
    ///     int num_vertices,
    ///     int max_colors
    /// )`
    /// - One thread per vertex
    /// - Computes |psi|^2 probability distribution
    /// - Samples color via curand_uniform
    /// - Grid: ceil(num_vertices / BLOCK_SIZE)
    ///
    /// # Performance
    /// - Overhead vs deterministic: ~5-10% slower
    /// - Benefits: Explores multiple solutions via repeated measurements
    /// - Use case: RL exploration vs exploitation tradeoff
    ///
    /// # Reference
    /// PRISM Quantum Evolution Stage 3.4
    pub fn launch_measurement_stochastic(
        &self,
        real_amplitudes: &CudaSlice<f32>,
        imag_amplitudes: &CudaSlice<f32>,
        rng_states: &mut CudaSlice<u8>,
        num_vertices: usize,
        max_colors: usize,
    ) -> Result<CudaSlice<i32>> {
        // Validate inputs
        let expected_amp_size = num_vertices * max_colors;
        anyhow::ensure!(
            real_amplitudes.len() == expected_amp_size,
            "Real amplitude buffer size mismatch"
        );
        anyhow::ensure!(
            imag_amplitudes.len() == expected_amp_size,
            "Imaginary amplitude buffer size mismatch"
        );
        anyhow::ensure!(
            rng_states.len() == num_vertices * 40,
            "RNG state buffer size mismatch: expected {} bytes",
            num_vertices * 40
        );

        log::debug!("Launching stochastic measurement kernel");

        // Allocate output buffer
        let d_colors = self
            .stream
            .alloc_zeros(num_vertices)
            .context("Failed to allocate colors buffer")?;

        // Get kernel function
        let measure_func = self.module.load_function("quantum_measure_stochastic_kernel")
            .context("Failed to get quantum_measure_stochastic_kernel function")?;

        // Launch configuration: 1 thread per vertex
        let grid_dim = (num_vertices as u32).div_ceil(BLOCK_SIZE as u32);
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&measure_func)
                .arg(real_amplitudes)
                .arg(imag_amplitudes)
                .arg(&d_colors)
                .arg(rng_states)
                .arg(&(num_vertices as i32))
                .arg(&(max_colors as i32))
                .launch(cfg)?
        };

        log::debug!(
            "Stochastic measurement kernel launched: {} vertices, grid_dim={}",
            num_vertices,
            grid_dim
        );

        Ok(d_colors)
    }

    // ========================================================================
    // STAGE 3: TELEMETRY GETTERS FOR QUANTUM METRICS
    // ========================================================================

    /// Computes amplitude variance across all vertices and colors
    ///
    /// SUBSTEP 5: Quantum state telemetry for FluxNet RL
    ///
    /// # Arguments
    /// * `real_amps` - Real amplitude buffer
    /// * `imag_amps` - Imaginary amplitude buffer
    ///
    /// # Returns
    /// Variance of complex amplitude magnitudes |psi|^2
    ///
    /// # Algorithm
    /// 1. Compute magnitude: mag[i] = real[i]^2 + imag[i]^2
    /// 2. Compute mean: mean = sum(mag) / N
    /// 3. Compute variance: var = sum((mag - mean)^2) / N
    ///
    /// # Use Case
    /// High variance indicates diverse probability distribution (quantum-like).
    /// Low variance indicates peaked distribution (classical-like).
    ///
    /// # Performance
    /// - GPU reduction kernel recommended for large graphs
    /// - Current implementation: CPU-side reduction (development phase)
    ///
    /// # Reference
    /// PRISM Quantum Evolution Stage 3.5
    pub fn compute_amplitude_variance(
        &self,
        real_amps: &CudaSlice<f32>,
        imag_amps: &CudaSlice<f32>,
    ) -> Result<f32> {
        // Copy to host for now (production will use GPU reduction)
        let real_host = self
            .stream
            .clone_dtoh(real_amps)
            .context("Failed to copy real amplitudes")?;
        let imag_host = self
            .stream
            .clone_dtoh(imag_amps)
            .context("Failed to copy imaginary amplitudes")?;

        let n = real_host.len() as f32;
        let magnitudes: Vec<f32> = real_host
            .iter()
            .zip(imag_host.iter())
            .map(|(r, i)| r * r + i * i)
            .collect();

        let mean: f32 = magnitudes.iter().sum::<f32>() / n;
        let variance: f32 = magnitudes
            .iter()
            .map(|mag| (mag - mean).powi(2))
            .sum::<f32>()
            / n;

        Ok(variance)
    }

    /// Computes quantum coherence metric
    ///
    /// # Returns
    /// Coherence value in range [0, 1]
    /// - 1.0: Fully coherent (pure quantum state)
    /// - 0.0: Fully decoherent (classical state)
    ///
    /// # Algorithm
    /// Coherence = sqrt(sum(real[i] * imag[i])^2) / sum(|psi[i]|^2)
    ///
    /// # Use Case
    /// Tracks quantum-to-classical transition during annealing.
    /// FluxNet RL uses this to adapt exploration strategy.
    ///
    /// # Reference
    /// PRISM Quantum Evolution Stage 3.5
    pub fn compute_coherence(
        &self,
        real_amps: &CudaSlice<f32>,
        imag_amps: &CudaSlice<f32>,
    ) -> Result<f32> {
        let real_host = self.stream.clone_dtoh(real_amps)?;
        let imag_host = self.stream.clone_dtoh(imag_amps)?;

        let cross_term: f32 = real_host
            .iter()
            .zip(imag_host.iter())
            .map(|(r, i)| r * i)
            .sum();

        let norm: f32 = real_host
            .iter()
            .zip(imag_host.iter())
            .map(|(r, i)| r * r + i * i)
            .sum();

        if norm < 1e-9 {
            return Ok(0.0);
        }

        let coherence = (cross_term.abs() / norm).min(1.0);
        Ok(coherence)
    }

    /// Returns current quantum purity metric
    ///
    /// # Returns
    /// Purity in range [0, 1]
    /// - 1.0: Pure state (no decoherence)
    /// - 0.0: Maximally mixed state (fully classical)
    ///
    /// # Notes
    /// Updated during evolve_complex() based on interference_decay.
    /// In production, computed via GPU trace(rho^2) where rho is density matrix.
    ///
    /// # Reference
    /// PRISM Quantum Evolution Stage 3.5
    pub fn get_purity(&self) -> f32 {
        self.purity
    }

    /// Returns current quantum entanglement metric
    ///
    /// # Returns
    /// Entanglement measure in range [0, 1]
    /// - 0.0: Separable state (no entanglement)
    /// - 1.0: Maximally entangled
    ///
    /// # Notes
    /// Updated during evolve_complex() as placeholder.
    /// Production implementation: von Neumann entropy of reduced density matrix.
    ///
    /// # Reference
    /// PRISM Quantum Evolution Stage 3.5
    pub fn get_entanglement(&self) -> f32 {
        self.entanglement
    }

    /// Returns the amplitude variance metric (Stage 5)
    ///
    /// Measures the spread of quantum amplitude magnitudes.
    ///
    /// # Returns
    /// Amplitude variance in range [0, ∞):
    /// - 0.0: Uniform distribution (all amplitudes equal)
    /// - Higher values: More concentrated amplitude distribution
    ///
    /// # Notes
    /// Updated after complex evolution completes via compute_amplitude_variance().
    /// Enables RL to track quantum state exploration breadth.
    ///
    /// # Reference
    /// PRISM Complex Quantum Evolution Stage 5 (FluxNet RL Wiring)
    pub fn get_amplitude_variance(&self) -> f32 {
        self.amplitude_variance
    }

    /// Returns the phase coherence metric (Stage 5)
    ///
    /// Measures quantum phase alignment quality.
    ///
    /// # Returns
    /// Coherence in range [0, 1]:
    /// - 1.0: Fully coherent (phases perfectly aligned)
    /// - 0.0: Completely dephased (random phases, no interference)
    ///
    /// # Notes
    /// Updated after complex evolution completes via compute_coherence().
    /// High coherence enables quantum interference effects for better exploration.
    ///
    /// # Reference
    /// PRISM Complex Quantum Evolution Stage 5 (FluxNet RL Wiring)
    pub fn get_coherence(&self) -> f32 {
        self.coherence
    }

    /// Launches quantum evolution kernel
    fn launch_evolution_kernel(
        &self,
        adjacency: &CudaSlice<i32>,
        amplitudes: &mut CudaSlice<f32>,
        couplings: &CudaSlice<f32>,
        num_vertices: usize,
        max_colors: usize,
    ) -> Result<()> {
        let evolve_func = self.module.load_function("quantum_evolve_kernel")
            .context("Failed to get quantum_evolve_kernel function")?;

        let grid_dim = (num_vertices as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&evolve_func)
                .arg(adjacency)
                .arg(amplitudes)
                .arg(couplings)
                .arg(&self.evolution_time)
                .arg(&(num_vertices as i32))
                .arg(&(max_colors as i32))
                .launch(cfg)?
        };

        self.stream
            .synchronize()
            .context("Evolution kernel synchronization failed")?;

        log::debug!("Quantum evolution completed (time={})", self.evolution_time);
        Ok(())
    }

    /// Launches quantum measurement kernel
    fn launch_measurement_kernel(
        &self,
        amplitudes: &CudaSlice<f32>,
        num_vertices: usize,
        max_colors: usize,
    ) -> Result<CudaSlice<i32>> {
        let measure_func = self.module.load_function("quantum_measure_kernel")
            .context("Failed to get quantum_measure_kernel function")?;

        // Allocate output buffer
        let d_colors = self
            .stream
            .alloc_zeros(num_vertices)
            .context("Failed to allocate colors buffer on GPU")?;

        let grid_dim = (num_vertices as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Random seed for stochastic measurement
        let dynamic_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| (d.as_nanos() as u64) & 0xFFFFFFFF)
            .unwrap_or(12345);

        unsafe {
            self.stream.launch_builder(&measure_func)
                .arg(amplitudes)
                .arg(&d_colors)
                .arg(&dynamic_seed)
                .arg(&(num_vertices as i32))
                .arg(&(max_colors as i32))
                .launch(cfg)?
        };

        self.stream
            .synchronize()
            .context("Measurement kernel synchronization failed")?;

        log::debug!("Quantum measurement completed");
        Ok(d_colors)
    }

    /// Returns reference to underlying CUDA device
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Returns current evolution time parameter
    pub fn evolution_time(&self) -> f32 {
        self.evolution_time
    }

    /// Sets evolution time parameter
    pub fn set_evolution_time(&mut self, time: f32) {
        assert!(time > 0.0, "evolution_time must be positive");
        self.evolution_time = time;
        log::debug!("Evolution time updated to {}", time);
    }

    /// Returns current coupling strength parameter
    pub fn coupling_strength(&self) -> f32 {
        self.coupling_strength
    }

    /// Sets coupling strength parameter
    pub fn set_coupling_strength(&mut self, strength: f32) {
        assert!(strength > 0.0, "coupling_strength must be positive");
        self.coupling_strength = strength;
        log::debug!("Coupling strength updated to {}", strength);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a simple test graph for unit testing
    fn create_test_graph() -> (Vec<Vec<usize>>, usize) {
        // Triangle graph (3 vertices, all connected)
        let adjacency = vec![
            vec![1, 2], // 0 -> 1, 2
            vec![0, 2], // 1 -> 0, 2
            vec![0, 1], // 2 -> 0, 1
        ];
        (adjacency, 3)
    }

    #[test]
    #[ignore] // Requires GPU hardware
    fn test_quantum_evolution_small_graph() {
        env_logger::builder().is_test(true).try_init().ok();

        let device = CudaContext::new(0).expect("CUDA device not available");
        let quantum = QuantumEvolutionGpu::new(Arc::new(device), "target/ptx/quantum.ptx")
            .expect("Failed to create QuantumEvolutionGpu");

        let (adjacency, num_vertices) = create_test_graph();
        let colors = quantum
            .evolve_and_measure(&adjacency, num_vertices, 10)
            .expect("Quantum evolution failed");

        // Verify output dimensions
        assert_eq!(colors.len(), num_vertices);

        // Verify colors are in valid range
        for &color in &colors {
            assert!(color < 10, "Color out of range: {}", color);
        }

        // For triangle graph, we need at least 3 colors (no conflicts)
        // But quantum algorithm may not guarantee optimal coloring
        log::info!("Test passed: colors={:?}", colors);
    }

    #[test]
    fn test_validation_max_vertices() {
        let device = CudaContext::new(0).expect("CUDA device not available");
        let quantum = QuantumEvolutionGpu::new(Arc::new(device), "target/ptx/quantum.ptx")
            .expect("Failed to create QuantumEvolutionGpu");

        let large_adjacency = vec![vec![]; MAX_VERTICES + 1];
        let result = quantum.evolve_and_measure(&large_adjacency, MAX_VERTICES + 1, 10);

        assert!(result.is_err();
        assert!(result.unwrap_err().to_string().contains("MAX_VERTICES");
    }

    #[test]
    fn test_validation_max_colors() {
        let device = CudaContext::new(0).expect("CUDA device not available");
        let quantum = QuantumEvolutionGpu::new(Arc::new(device), "target/ptx/quantum.ptx")
            .expect("Failed to create QuantumEvolutionGpu");

        let adjacency = vec![vec![]; 10];
        let result = quantum.evolve_and_measure(&adjacency, 10, MAX_COLORS + 1);

        assert!(result.is_err();
        assert!(result.unwrap_err().to_string().contains("max_colors");
    }

    #[test]
    fn test_parameter_validation() {
        let device = CudaContext::new(0).expect("CUDA device not available");

        // Valid parameters
        let result = QuantumEvolutionGpu::new_with_params(
            Arc::new(device.clone()),
            "target/ptx/quantum.ptx",
            2.0,
            1.5,
        );
        assert!(result.is_ok();

        // Invalid evolution_time
        let result = QuantumEvolutionGpu::new_with_params(
            Arc::new(device.clone()),
            "target/ptx/quantum.ptx",
            0.0,
            1.0,
        );
        assert!(result.is_err();

        // Invalid coupling_strength
        let result = QuantumEvolutionGpu::new_with_params(
            Arc::new(device),
            "target/ptx/quantum.ptx",
            1.0,
            -1.0,
        );
        assert!(result.is_err();
    }
}

// ============================================================================
// HIGH-LEVEL COMPLEX QUANTUM EVOLUTION API (Stage 4)
// ============================================================================

impl QuantumEvolutionGpu {
    /// High-level complex quantum evolution with measurement
    ///
    /// Convenience method that handles all buffer management internally.
    /// Allocates complex amplitude buffers, runs multi-iteration evolution,
    /// performs measurement, and returns color assignments.
    ///
    /// This is the primary interface for Phase 3 controller integration.
    ///
    /// # Arguments
    /// * `adjacency` - Adjacency list representation (vertex -> neighbors)
    /// * `num_vertices` - Number of vertices in graph
    /// * `config` - Phase 3 configuration with complex evolution parameters
    ///
    /// # Returns
    /// `Vec<usize>` of color assignments (length = num_vertices)
    ///
    /// # Errors
    /// Returns error if:
    /// - GPU operations fail
    /// - Buffers exceed MAX_VERTICES or MAX_COLORS limits
    /// - Memory allocation fails
    ///
    /// # Example
    /// ```rust,no_run
    /// # use prism_gpu::QuantumEvolutionGpu;
    /// # use cudarc::driver::CudaContext;
    /// # use prism_core::Phase3Config;
    /// # use std::sync::Arc;
    /// let device = CudaContext::new(0).unwrap());
    /// let mut quantum = QuantumEvolutionGpu::new(Arc::new(device), "target/ptx/quantum.ptx").unwrap());
    /// let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    /// let config = Phase3Config {
    ///     use_complex_amplitudes: true,
    ///     evolution_iterations: 100,
    ///     max_colors: 10,
    ///     ..Default::default()
    /// };
    /// let colors = quantum.evolve_complex_and_measure(&adjacency, 3, &config).unwrap());
    /// ```
    pub fn evolve_complex_and_measure(
        &mut self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
        config: &prism_core::Phase3Config,
    ) -> Result<Vec<usize>> {
        // 1. Validation
        anyhow::ensure!(num_vertices > 0, "Graph must have at least one vertex");
        anyhow::ensure!(
            num_vertices <= MAX_VERTICES,
            "Graph exceeds MAX_VERTICES limit: {} > {}",
            num_vertices,
            MAX_VERTICES
        );
        anyhow::ensure!(
            config.max_colors > 0 && config.max_colors <= MAX_COLORS,
            "max_colors must be in range [1, {}], got {}",
            MAX_COLORS,
            config.max_colors
        );
        anyhow::ensure!(
            adjacency.len() == num_vertices,
            "Adjacency list size mismatch: expected {}, got {}",
            num_vertices,
            adjacency.len()
        );

        log::info!(
            "Complex quantum evolution: {} vertices, {} colors, {} iterations, schedule={}",
            num_vertices,
            config.max_colors,
            config.evolution_iterations,
            config.schedule_type
        );

        // 2. Convert adjacency list to dense matrix
        let adjacency_matrix = self.adjacency_to_matrix(adjacency, num_vertices);
        let d_adjacency = self
            .stream
            .clone_htod(&adjacency_matrix)
            .context("Failed to copy adjacency matrix to GPU")?;

        // 3. Prepare coupling strengths (uniform for all vertices)
        let couplings = vec![config.coupling_strength; num_vertices];
        let d_couplings = self
            .stream
            .clone_htod(&couplings)
            .context("Failed to copy couplings to GPU")?;

        // 4. Initialize complex amplitude buffers
        let amplitude_size = num_vertices * config.max_colors;
        let mut d_real_amplitudes = self
            .stream
            .alloc_zeros(amplitude_size)
            .context("Failed to allocate real amplitudes")?;
        let mut d_imag_amplitudes = self
            .stream
            .alloc_zeros(amplitude_size)
            .context("Failed to allocate imaginary amplitudes")?;

        self.init_complex_amplitudes(
            &mut d_real_amplitudes,
            &mut d_imag_amplitudes,
            num_vertices,
            config.max_colors,
        )?;

        // 5. Run multi-iteration complex evolution
        log::debug!("Starting multi-iteration complex evolution");
        self.evolve_complex(
            &mut d_real_amplitudes,
            &mut d_imag_amplitudes,
            &d_adjacency,
            &d_couplings,
            num_vertices,
            config.max_colors,
            config,
        )?;

        // 6. Perform measurement
        log::debug!(
            "Measuring quantum state (stochastic={})",
            config.stochastic_measurement
        );
        let d_colors = if config.stochastic_measurement {
            // Stochastic measurement (requires RNG states)
            // Use dynamic seed from system time to ensure different results per attempt
            let dynamic_seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345);
            let mut d_rng_states = self.init_rng_states(num_vertices, dynamic_seed)?;
            self.launch_measurement_stochastic(
                &d_real_amplitudes,
                &d_imag_amplitudes,
                &mut d_rng_states,
                num_vertices,
                config.max_colors,
            )?
        } else {
            // Deterministic measurement (argmax of |amplitude|²)
            self.launch_measurement_deterministic(
                &d_real_amplitudes,
                &d_imag_amplitudes,
                num_vertices,
                config.max_colors,
            )?
        };

        // 6.5. Compute and store telemetry metrics (Stage 5)
        // Must be done BEFORE buffers are deallocated
        log::debug!("Computing complex quantum telemetry metrics");
        self.amplitude_variance =
            self.compute_amplitude_variance(&d_real_amplitudes, &d_imag_amplitudes)?;
        self.coherence = self.compute_coherence(&d_real_amplitudes, &d_imag_amplitudes)?;
        log::debug!(
            "Telemetry: amplitude_variance={:.4}, coherence={:.4}",
            self.amplitude_variance,
            self.coherence
        );

        // 7. Copy results back to host
        log::debug!("Copying results back to host");
        let colors_i32 = self
            .stream
            .clone_dtoh(&d_colors)
            .context("Failed to copy colors from GPU")?;

        // 8. Convert i32 to usize
        let colors: Vec<usize> = colors_i32.iter().map(|&c| c as usize).collect();

        log::info!(
            "Complex quantum evolution completed: {} colors assigned",
            colors.len()
        );

        Ok(colors)
    }

    /// Launches deterministic measurement kernel for complex amplitudes
    ///
    /// Selects color with maximum |amplitude|² for each vertex.
    /// This is the deterministic (non-stochastic) measurement mode.
    ///
    /// # Arguments
    /// * `real_amplitudes` - Real parts of quantum amplitudes
    /// * `imag_amplitudes` - Imaginary parts of quantum amplitudes
    /// * `num_vertices` - Number of vertices
    /// * `max_colors` - Maximum colors
    ///
    /// # Returns
    /// `CudaSlice<i32>` containing color assignments
    ///
    /// # Implementation Note
    /// Currently uses the existing `quantum_measure_kernel` which computes
    /// |amplitude|² = real² + imag² internally (though the current kernel
    /// only uses real part). When CUDA kernels are implemented in Stage 7,
    /// this will properly handle complex amplitudes.
    fn launch_measurement_deterministic(
        &self,
        real_amplitudes: &CudaSlice<f32>,
        imag_amplitudes: &CudaSlice<f32>,
        num_vertices: usize,
        max_colors: usize,
    ) -> Result<CudaSlice<i32>> {
        // TODO(Stage 7): This currently uses the legacy real-only measurement kernel
        // The complex measurement kernel should compute |ψ|² = real² + imag²
        // For now, we approximate by using real amplitudes only

        log::debug!("Launching deterministic measurement (legacy kernel approximation)");

        let measure_func = self.module.load_function("quantum_measure_kernel")
            .context("Failed to get quantum_measure_kernel function")?;

        // Allocate output buffer
        let d_colors = self
            .stream
            .alloc_zeros(num_vertices)
            .context("Failed to allocate colors buffer on GPU")?;

        let grid_dim = (num_vertices as u32).div_ceil(BLOCK_SIZE as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Random seed for stochastic measurement
        let dynamic_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| (d.as_nanos() as u64) & 0xFFFFFFFF)
            .unwrap_or(12345);

        // Launch with real amplitudes (legacy path)
        // TODO(Stage 7): Replace with complex-aware kernel
        unsafe {
            self.stream.launch_builder(&measure_func)
                .arg(real_amplitudes)
                .arg(&d_colors)
                .arg(&dynamic_seed)
                .arg(&(num_vertices as i32))
                .arg(&(max_colors as i32))
                .launch(cfg)?
        };

        self.stream
            .synchronize()
            .context("Measurement kernel synchronization failed")?;

        log::debug!("Deterministic measurement completed");
        Ok(d_colors)
    }
}
