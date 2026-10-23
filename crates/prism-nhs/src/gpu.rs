//! NHS GPU Engine - CUDA-accelerated Hydrophobic Exclusion Mapping
//!
//! Full GPU implementation of the NHS pipeline components:
//!
//! 1. **Exclusion Field Computation** - Parallel HEM from classified atoms
//! 2. **Water Inference** - Holographic negative computation
//! 3. **Neuromorphic Network** - LIF dewetting detection
//! 4. **Spike Processing** - Extraction and residue mapping
//!
//! ## Performance Targets
//!
//! - <0.5ms for exclusion field (100³ grid, 10k atoms)
//! - <0.3ms for LIF network step
//! - <2ms total per frame on RTX 3060
//!
//! ## Usage
//!
//! ```ignore
//! use prism_nhs::gpu::NhsGpuEngine;
//!
//! let mut engine = NhsGpuEngine::new(device, grid_dim)?;
//! engine.update_exclusion_field(&atom_positions, &atom_types, &atom_charges)?;
//! engine.infer_water_density()?;
//! let spikes = engine.lif_step()?;
//! ```

use anyhow::{Context, Result};
#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;
use std::io::Write;
use std::sync::Arc;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default grid dimension (64³ = 262,144 voxels)
pub const DEFAULT_GRID_DIM: usize = 64;

/// Maximum supported grid dimension (160³ needed for very large proteins >5000 atoms)
pub const MAX_GRID_DIM: usize = 160;

/// Default grid spacing in Angstroms
pub const DEFAULT_GRID_SPACING: f32 = 0.5;

/// Block sizes for CUDA kernels
const BLOCK_SIZE_1D: usize = 256;
const BLOCK_SIZE_3D: usize = 8; // 8³ = 512 threads

/// Maximum spikes per frame for output buffers
const MAX_SPIKES_PER_FRAME: usize = 10000;

// ============================================================================
// NHS GPU ENGINE
// ============================================================================

/// GPU-accelerated NHS Engine
///
/// Manages all GPU buffers and kernel execution for the NHS pipeline.
/// All persistent state lives in VRAM for minimal transfer overhead.
#[cfg(feature = "gpu")]
pub struct NhsGpuEngine {
    // CUDA handles
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _exclusion_module: Arc<CudaModule>,
    _neuromorphic_module: Arc<CudaModule>,

    // Exclusion kernel functions
    compute_exclusion_field: CudaFunction,
    compute_exclusion_field_cell_list: CudaFunction,
    infer_water_density: CudaFunction,
    accumulate_pocket_probability: CudaFunction,
    finalize_pocket_probability: CudaFunction,
    detect_aromatic_targets: CudaFunction,
    reset_grid: CudaFunction,
    reset_grid_int: CudaFunction,

    // Neuromorphic kernel functions
    lif_dewetting_step: CudaFunction,
    lif_dewetting_batch: CudaFunction,
    apply_lateral_inhibition: CudaFunction,
    extract_spike_indices: CudaFunction,
    map_spikes_to_residues: CudaFunction,
    init_lif_state: CudaFunction,
    reset_lif_state: CudaFunction,

    // Grid buffers (persistent in VRAM)
    exclusion_field: CudaSlice<f32>,    // [grid_dim³]
    water_density: CudaSlice<f32>,      // [grid_dim³]
    prev_water_density: CudaSlice<f32>, // [grid_dim³] for dewetting delta
    water_gradient: CudaSlice<f32>,     // [grid_dim³]
    pocket_mean: CudaSlice<f32>,        // [grid_dim³] for variance tracking
    pocket_m2: CudaSlice<f32>,          // [grid_dim³] for Welford's algorithm
    pocket_probability: CudaSlice<f32>, // [grid_dim³]

    // LIF neuron state (persistent in VRAM)
    membrane_potential: CudaSlice<f32>, // [grid_dim³]
    refractory_counter: CudaSlice<i32>, // [grid_dim³]
    spike_output: CudaSlice<i32>,       // [grid_dim³]

    // Spike extraction buffers
    spike_indices: CudaSlice<i32>,   // [MAX_SPIKES_PER_FRAME]
    spike_positions: CudaSlice<f32>, // [MAX_SPIKES_PER_FRAME * 3]
    spike_count: CudaSlice<i32>,     // [1] atomic counter

    // Atom data buffers (updated per frame)
    atom_positions_gpu: CudaSlice<f32>, // [max_atoms * 3]
    atom_types_gpu: CudaSlice<i32>,     // [max_atoms]
    atom_charges_gpu: CudaSlice<f32>,   // [max_atoms]
    atom_residues_gpu: CudaSlice<i32>,  // [max_atoms]

    // Configuration
    grid_dim: usize,
    grid_spacing: f32,
    grid_origin: [f32; 3],
    max_atoms: usize,
    current_atoms: usize,

    // LIF parameters
    tau_mem: f32,
    sensitivity: f32,

    // State tracking
    frame_count: usize,
    initialized: bool,
}

#[cfg(feature = "gpu")]
impl NhsGpuEngine {
    /// Creates a new NHS GPU Engine
    ///
    /// # Arguments
    /// * `context` - CUDA device context
    /// * `grid_dim` - Grid dimension (cube: dim × dim × dim)
    /// * `max_atoms` - Maximum number of atoms to support
    ///
    /// # Returns
    /// Initialized GPU engine with allocated buffers
    pub fn new(context: Arc<CudaContext>, grid_dim: usize, max_atoms: usize) -> Result<Self> {
        Self::new_with_params(context, grid_dim, max_atoms, DEFAULT_GRID_SPACING)
    }

    /// Creates a new NHS GPU Engine with custom parameters
    pub fn new_with_params(
        context: Arc<CudaContext>,
        grid_dim: usize,
        max_atoms: usize,
        grid_spacing: f32,
    ) -> Result<Self> {
        anyhow::ensure!(
            grid_dim > 0 && grid_dim <= MAX_GRID_DIM,
            "grid_dim must be in [1, {}], got {}",
            MAX_GRID_DIM,
            grid_dim
        );

        log::info!(
            "Creating NHS GPU Engine: grid={}³ ({}M voxels), max_atoms={}, spacing={:.2}Å",
            grid_dim,
            (grid_dim * grid_dim * grid_dim) as f64 / 1e6,
            max_atoms,
            grid_spacing
        );

        let stream = context.default_stream();
        let grid_size = grid_dim * grid_dim * grid_dim;

        // Load PTX modules
        let exclusion_ptx_path = Self::find_ptx_path("nhs_exclusion")?;
        let neuromorphic_ptx_path = Self::find_ptx_path("nhs_neuromorphic")?;

        log::info!("Loading exclusion PTX from: {}", exclusion_ptx_path);
        log::info!("Loading neuromorphic PTX from: {}", neuromorphic_ptx_path);

        let exclusion_module = context
            .load_module(Ptx::from_file(&exclusion_ptx_path))
            .context("Failed to load NHS exclusion PTX module")?;

        let neuromorphic_module = context
            .load_module(Ptx::from_file(&neuromorphic_ptx_path))
            .context("Failed to load NHS neuromorphic PTX module")?;

        // Load exclusion kernel functions
        let compute_exclusion_field = exclusion_module.load_function("compute_exclusion_field")?;
        let compute_exclusion_field_cell_list =
            exclusion_module.load_function("compute_exclusion_field_cell_list")?;
        let infer_water_density = exclusion_module.load_function("infer_water_density")?;
        let accumulate_pocket_probability =
            exclusion_module.load_function("accumulate_pocket_probability")?;
        let finalize_pocket_probability =
            exclusion_module.load_function("finalize_pocket_probability")?;
        let detect_aromatic_targets = exclusion_module.load_function("detect_aromatic_targets")?;
        let reset_grid = exclusion_module.load_function("reset_grid")?;
        let reset_grid_int = exclusion_module.load_function("reset_grid_int")?;

        // Load neuromorphic kernel functions
        let lif_dewetting_step = neuromorphic_module.load_function("lif_dewetting_step")?;
        let lif_dewetting_batch = neuromorphic_module.load_function("lif_dewetting_batch")?;
        let apply_lateral_inhibition =
            neuromorphic_module.load_function("apply_lateral_inhibition")?;
        let extract_spike_indices = neuromorphic_module.load_function("extract_spike_indices")?;
        let map_spikes_to_residues = neuromorphic_module.load_function("map_spikes_to_residues")?;
        let init_lif_state = neuromorphic_module.load_function("init_lif_state")?;
        let reset_lif_state = neuromorphic_module.load_function("reset_lif_state")?;

        // Allocate grid buffers
        let exclusion_field: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate exclusion_field")?;

        let water_density: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate water_density")?;

        let prev_water_density: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate prev_water_density")?;

        let water_gradient: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate water_gradient")?;

        let pocket_mean: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate pocket_mean")?;

        let pocket_m2: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate pocket_m2")?;

        let pocket_probability: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate pocket_probability")?;

        // Allocate LIF neuron state
        let membrane_potential: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate membrane_potential")?;

        let refractory_counter: CudaSlice<i32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate refractory_counter")?;

        let spike_output: CudaSlice<i32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate spike_output")?;

        // Allocate spike extraction buffers
        let spike_indices: CudaSlice<i32> = stream
            .alloc_zeros(MAX_SPIKES_PER_FRAME)
            .context("Failed to allocate spike_indices")?;

        let spike_positions: CudaSlice<f32> = stream
            .alloc_zeros(MAX_SPIKES_PER_FRAME * 3)
            .context("Failed to allocate spike_positions")?;

        let spike_count: CudaSlice<i32> = stream
            .alloc_zeros(1)
            .context("Failed to allocate spike_count")?;

        // Allocate atom data buffers
        let atom_positions_gpu: CudaSlice<f32> = stream
            .alloc_zeros(max_atoms * 3)
            .context("Failed to allocate atom_positions_gpu")?;

        let atom_types_gpu: CudaSlice<i32> = stream
            .alloc_zeros(max_atoms)
            .context("Failed to allocate atom_types_gpu")?;

        let atom_charges_gpu: CudaSlice<f32> = stream
            .alloc_zeros(max_atoms)
            .context("Failed to allocate atom_charges_gpu")?;

        let atom_residues_gpu: CudaSlice<i32> = stream
            .alloc_zeros(max_atoms)
            .context("Failed to allocate atom_residues_gpu")?;

        // Calculate memory usage
        let grid_bytes = grid_size * 4 * 10; // 10 float/int grids
        let spike_bytes = MAX_SPIKES_PER_FRAME * 4 * 4; // indices, positions, count
        let atom_bytes = max_atoms * 4 * 4; // positions, types, charges, residues
        let total_mb = (grid_bytes + spike_bytes + atom_bytes) as f64 / 1024.0 / 1024.0;

        log::info!("NHS GPU buffers allocated: {:.2}MB total", total_mb);
        log::info!(
            "  Grid buffers: {:.2}MB",
            grid_bytes as f64 / 1024.0 / 1024.0
        );
        log::info!(
            "  Spike buffers: {:.2}MB",
            spike_bytes as f64 / 1024.0 / 1024.0
        );
        log::info!(
            "  Atom buffers: {:.2}MB",
            atom_bytes as f64 / 1024.0 / 1024.0
        );

        Ok(Self {
            context,
            stream,
            _exclusion_module: exclusion_module,
            _neuromorphic_module: neuromorphic_module,

            compute_exclusion_field,
            compute_exclusion_field_cell_list,
            infer_water_density,
            accumulate_pocket_probability,
            finalize_pocket_probability,
            detect_aromatic_targets,
            reset_grid,
            reset_grid_int,

            lif_dewetting_step,
            lif_dewetting_batch,
            apply_lateral_inhibition,
            extract_spike_indices,
            map_spikes_to_residues,
            init_lif_state,
            reset_lif_state,

            exclusion_field,
            water_density,
            prev_water_density,
            water_gradient,
            pocket_mean,
            pocket_m2,
            pocket_probability,

            membrane_potential,
            refractory_counter,
            spike_output,

            spike_indices,
            spike_positions,
            spike_count,

            atom_positions_gpu,
            atom_types_gpu,
            atom_charges_gpu,
            atom_residues_gpu,

            grid_dim,
            grid_spacing,
            grid_origin: [0.0, 0.0, 0.0],
            max_atoms,
            current_atoms: 0,

            tau_mem: 10.0,
            sensitivity: 1.0,

            frame_count: 0,
            initialized: false,
        })
    }

    /// Find PTX file path
    fn find_ptx_path(kernel_name: &str) -> Result<String> {
        let filename = format!("{}.ptx", kernel_name);
        // Check PRISM4D_PTX_DIR / PRISM_PTX_DIR first
        for env_key in &["PRISM4D_PTX_DIR", "PRISM_PTX_DIR"] {
            if let Ok(dir) = std::env::var(env_key) {
                let p = std::path::PathBuf::from(&dir).join(&filename);
                if p.exists() {
                    return Ok(p.display().to_string());
                }
            }
        }
        let paths = [
            format!("target/ptx/{}", filename),
            format!("crates/prism-gpu/target/ptx/{}", filename),
            format!("../prism-gpu/target/ptx/{}", filename),
        ];

        for path in &paths {
            if std::path::Path::new(path).exists() {
                return Ok(path.clone());
            }
        }

        // Try OUT_DIR from build
        if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let path = format!("{}/ptx/{}", out_dir, filename);
            if std::path::Path::new(&path).exists() {
                return Ok(path);
            }
        }

        Err(anyhow::anyhow!(
            "{}.ptx not found. Run `cargo build -p prism-gpu --features cuda` to compile kernels",
            kernel_name
        ))
    }

    /// Initialize the LIF network state
    pub fn initialize(&mut self, grid_origin: [f32; 3]) -> Result<()> {
        log::info!("Initializing NHS GPU Engine at origin {:?}", grid_origin);
        self.grid_origin = grid_origin;

        // Initialize LIF state
        let grid_size = self.grid_dim * self.grid_dim * self.grid_dim;
        let grid_blocks = (grid_size as u32).div_ceil(BLOCK_SIZE_1D as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.init_lif_state)
                .arg(&self.membrane_potential)
                .arg(&self.refractory_counter)
                .arg(&(self.grid_dim as i32))
                .launch(cfg)
        }
        .context("Failed to launch init_lif_state kernel")?;

        // Reset grids
        self.reset_grids()?;

        self.frame_count = 0;
        self.initialized = true;
        log::info!("NHS GPU Engine initialized");
        Ok(())
    }

    /// Reset all grid buffers to zero
    pub fn reset_grids(&mut self) -> Result<()> {
        let grid_size = self.grid_dim * self.grid_dim * self.grid_dim;
        let grid_blocks = (grid_size as u32).div_ceil(BLOCK_SIZE_1D as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Reset float grids
        for grid in [
            &self.exclusion_field,
            &self.water_density,
            &self.prev_water_density,
            &self.water_gradient,
            &self.pocket_mean,
            &self.pocket_m2,
            &self.pocket_probability,
            &self.membrane_potential,
        ] {
            unsafe {
                self.stream
                    .launch_builder(&self.reset_grid)
                    .arg(grid)
                    .arg(&(grid_size as i32))
                    .launch(cfg.clone())
            }
            .context("Failed to launch reset_grid kernel")?;
        }

        // Reset int grids
        for grid in [&self.refractory_counter, &self.spike_output] {
            unsafe {
                self.stream
                    .launch_builder(&self.reset_grid_int)
                    .arg(grid)
                    .arg(&(grid_size as i32))
                    .launch(cfg.clone())
            }
            .context("Failed to launch reset_grid_int kernel")?;
        }

        self.context.synchronize()?;
        Ok(())
    }

    /// Update atom data on GPU
    ///
    /// # Arguments
    /// * `positions` - Flat array of atom positions [n_atoms * 3]
    /// * `types` - Atom type classification [n_atoms]
    /// * `charges` - Partial charges [n_atoms]
    /// * `residues` - Residue indices [n_atoms]
    pub fn update_atom_data(
        &mut self,
        positions: &[f32],
        types: &[i32],
        charges: &[f32],
        residues: &[i32],
    ) -> Result<()> {
        let n_atoms = positions.len() / 3;
        anyhow::ensure!(
            n_atoms <= self.max_atoms,
            "Too many atoms: {} > max {}",
            n_atoms,
            self.max_atoms
        );
        anyhow::ensure!(
            types.len() == n_atoms && charges.len() == n_atoms && residues.len() == n_atoms,
            "Atom data length mismatch"
        );

        // Copy to GPU
        self.stream
            .memcpy_htod(&positions[..n_atoms * 3], &mut self.atom_positions_gpu)
            .context("Failed to copy atom positions to GPU")?;

        self.stream
            .memcpy_htod(&types[..n_atoms], &mut self.atom_types_gpu)
            .context("Failed to copy atom types to GPU")?;

        self.stream
            .memcpy_htod(&charges[..n_atoms], &mut self.atom_charges_gpu)
            .context("Failed to copy atom charges to GPU")?;

        self.stream
            .memcpy_htod(&residues[..n_atoms], &mut self.atom_residues_gpu)
            .context("Failed to copy atom residues to GPU")?;

        self.current_atoms = n_atoms;
        Ok(())
    }

    /// Compute exclusion field from current atom positions
    ///
    /// Runs the HEM kernel to compute 3D exclusion field based on
    /// atom classifications (hydrophobic, polar, charged, aromatic).
    pub fn compute_exclusion(&mut self) -> Result<()> {
        anyhow::ensure!(self.initialized, "Engine not initialized");
        anyhow::ensure!(self.current_atoms > 0, "No atoms loaded");

        // 3D grid launch configuration
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
            ),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.compute_exclusion_field)
                .arg(&self.atom_positions_gpu)
                .arg(&self.atom_types_gpu)
                .arg(&self.atom_charges_gpu)
                .arg(&self.exclusion_field)
                .arg(&(self.current_atoms as i32))
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&self.grid_spacing)
                .launch(cfg)
        }
        .context("Failed to launch compute_exclusion_field kernel")?;

        Ok(())
    }

    /// Infer water density from exclusion field (holographic negative)
    ///
    /// Also computes gradient magnitude for dewetting edge detection.
    pub fn infer_water(&mut self) -> Result<()> {
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
            ),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.infer_water_density)
                .arg(&self.exclusion_field)
                .arg(&self.water_density)
                .arg(&self.water_gradient)
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_spacing)
                .launch(cfg)
        }
        .context("Failed to launch infer_water_density kernel")?;

        Ok(())
    }

    /// Run one LIF network step for dewetting detection
    ///
    /// Returns the number of spikes detected.
    pub fn lif_step(&mut self) -> Result<usize> {
        // Swap water density buffers for delta computation
        std::mem::swap(&mut self.water_density, &mut self.prev_water_density);

        // Recompute current water density (we just swapped, so need fresh computation)
        self.infer_water()?;

        // Reset spike count
        let zero = [0i32];
        self.stream
            .memcpy_htod(&zero, &mut self.spike_count)
            .context("Failed to reset spike count")?;

        // LIF step kernel
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
            ),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.lif_dewetting_step)
                .arg(&self.prev_water_density)
                .arg(&self.water_density)
                .arg(&self.membrane_potential)
                .arg(&self.refractory_counter)
                .arg(&self.spike_output)
                .arg(&self.spike_count)
                .arg(&(self.grid_dim as i32))
                .arg(&self.tau_mem)
                .arg(&self.sensitivity)
                .launch(cfg)
        }
        .context("Failed to launch lif_dewetting_step kernel")?;

        // Apply lateral inhibition
        unsafe {
            self.stream
                .launch_builder(&self.apply_lateral_inhibition)
                .arg(&self.spike_output)
                .arg(&self.membrane_potential)
                .arg(&(self.grid_dim as i32))
                .arg(&0.1f32) // inhibition strength
                .launch(cfg)
        }
        .context("Failed to launch apply_lateral_inhibition kernel")?;

        // Get spike count
        let spike_counts = self
            .stream
            .clone_dtoh(&self.spike_count)
            .context("Failed to read spike count")?;

        self.frame_count += 1;
        Ok(spike_counts[0] as usize)
    }

    /// Extract spike positions and map to residues
    ///
    /// # Returns
    /// Tuple of (spike_positions, spike_residues, spike_distances)
    pub fn extract_spikes(&mut self) -> Result<(Vec<[f32; 3]>, Vec<i32>, Vec<f32>)> {
        // Reset spike count for extraction
        let zero = [0i32];
        self.stream
            .memcpy_htod(&zero, &mut self.spike_count)
            .context("Failed to reset spike count for extraction")?;

        // Extract spike indices and positions
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
            ),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.extract_spike_indices)
                .arg(&self.spike_output)
                .arg(&self.spike_indices)
                .arg(&self.spike_positions)
                .arg(&self.spike_count)
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&self.grid_spacing)
                .arg(&(MAX_SPIKES_PER_FRAME as i32))
                .launch(cfg)
        }
        .context("Failed to launch extract_spike_indices kernel")?;

        // Get spike count
        let spike_counts = self
            .stream
            .clone_dtoh(&self.spike_count)
            .context("Failed to read spike count")?;

        let n_spikes = (spike_counts[0] as usize).min(MAX_SPIKES_PER_FRAME);

        if n_spikes == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        // Copy spike positions from GPU
        let positions_flat = self
            .stream
            .clone_dtoh(&self.spike_positions)
            .context("Failed to read spike positions")?;

        let mut positions = Vec::with_capacity(n_spikes);
        for i in 0..n_spikes {
            positions.push([
                positions_flat[i * 3],
                positions_flat[i * 3 + 1],
                positions_flat[i * 3 + 2],
            ]);
        }

        // Allocate residue mapping buffers
        let mut spike_residues_gpu: CudaSlice<i32> = self
            .stream
            .alloc_zeros(n_spikes)
            .context("Failed to allocate spike_residues")?;

        let mut spike_distances_gpu: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_spikes)
            .context("Failed to allocate spike_distances")?;

        // Map spikes to residues
        let spike_blocks = (n_spikes as u32).div_ceil(BLOCK_SIZE_1D as u32);

        let cfg_1d = LaunchConfig {
            grid_dim: (spike_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.map_spikes_to_residues)
                .arg(&self.spike_positions)
                .arg(&self.atom_positions_gpu)
                .arg(&self.atom_residues_gpu)
                .arg(&spike_residues_gpu)
                .arg(&spike_distances_gpu)
                .arg(&(n_spikes as i32))
                .arg(&(self.current_atoms as i32))
                .arg(&10.0f32) // max_distance
                .launch(cfg_1d)
        }
        .context("Failed to launch map_spikes_to_residues kernel")?;

        // Copy results
        let residues = self
            .stream
            .clone_dtoh(&spike_residues_gpu)
            .context("Failed to read spike residues")?;

        let distances = self
            .stream
            .clone_dtoh(&spike_distances_gpu)
            .context("Failed to read spike distances")?;

        Ok((
            positions,
            residues[..n_spikes].to_vec(),
            distances[..n_spikes].to_vec(),
        ))
    }

    /// Accumulate pocket probability from temporal variance
    pub fn accumulate_pocket_variance(&mut self) -> Result<()> {
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
            ),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.accumulate_pocket_probability)
                .arg(&self.water_density)
                .arg(&self.pocket_mean)
                .arg(&self.pocket_m2)
                .arg(&(self.frame_count as i32))
                .arg(&(self.grid_dim as i32))
                .launch(cfg)
        }
        .context("Failed to launch accumulate_pocket_probability kernel")?;

        Ok(())
    }

    /// Finalize pocket probability and copy to host
    pub fn finalize_pocket_probability(&mut self, variance_threshold: f32) -> Result<Vec<f32>> {
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
                BLOCK_SIZE_3D as u32,
            ),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.finalize_pocket_probability)
                .arg(&self.pocket_mean)
                .arg(&self.pocket_m2)
                .arg(&self.pocket_probability)
                .arg(&(self.frame_count as i32))
                .arg(&(self.grid_dim as i32))
                .arg(&variance_threshold)
                .launch(cfg)
        }
        .context("Failed to launch finalize_pocket_probability kernel")?;

        let probability = self
            .stream
            .clone_dtoh(&self.pocket_probability)
            .context("Failed to read pocket probability")?;

        Ok(probability)
    }

    /// Get water density grid from GPU
    pub fn get_water_density(&self) -> Result<Vec<f32>> {
        let density = self
            .stream
            .clone_dtoh(&self.water_density)
            .context("Failed to read water density")?;
        Ok(density)
    }

    /// Get exclusion field from GPU
    pub fn get_exclusion_field(&self) -> Result<Vec<f32>> {
        let field = self
            .stream
            .clone_dtoh(&self.exclusion_field)
            .context("Failed to read exclusion field")?;
        Ok(field)
    }

    /// Process a complete frame: update atoms, compute exclusion, infer water, run LIF
    ///
    /// This is the main entry point for frame-by-frame processing.
    pub fn process_frame(
        &mut self,
        positions: &[f32],
        types: &[i32],
        charges: &[f32],
        residues: &[i32],
    ) -> Result<FrameResult> {
        // Update atom data
        self.update_atom_data(positions, types, charges, residues)?;

        // Compute exclusion field
        self.compute_exclusion()?;

        // Run LIF step (includes water inference)
        let spike_count = self.lif_step()?;

        // Accumulate pocket variance
        self.accumulate_pocket_variance()?;

        // Extract spikes if any
        let (spike_positions, spike_residues, spike_distances) = if spike_count > 0 {
            self.extract_spikes()?
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        Ok(FrameResult {
            frame: self.frame_count,
            spike_count,
            spike_positions,
            spike_residues,
            spike_distances,
        })
    }

    /// Set LIF parameters
    pub fn set_lif_params(&mut self, tau_mem: f32, sensitivity: f32) {
        self.tau_mem = tau_mem;
        self.sensitivity = sensitivity;
    }

    /// Get grid dimension
    pub fn grid_dim(&self) -> usize {
        self.grid_dim
    }

    /// Get grid spacing
    pub fn grid_spacing(&self) -> f32 {
        self.grid_spacing
    }

    /// Get frame count
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get CUDA stream
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

// ============================================================================
// Gate G2 — TrajectoryWriter (binary frames.bin emit)
// ============================================================================
//
// Produces a sidecar `<output>.frames.bin` alongside the existing
// multi-model PDB ensemble trajectory (`fn write_ensemble_trajectory`
// at `crates/prism-nhs/src/bin/nhs_rt_full.rs:14339`). The binary
// format is designed for fast `numpy.fromfile` (or `np.memmap`)
// loading from Python post-processing tools — the multi-model PDB
// is human-readable / MDAnalysis-compatible but slow to parse on
// large ensembles.
//
// File format (little-endian throughout):
//
//     bytes  0..8  : magic "PRISM4D\0"        ([u8; 8])
//     bytes  8..12 : version                   (u32, currently 1)
//     bytes 12..16 : n_atoms                   (u32)
//     bytes 16..20 : save_interval             (u32; in MD steps)
//     bytes 20..24 : dt_ps                     (f32)
//     bytes 24..   : raw frames, [n_frames, n_atoms, 3] f32 LE
//
// `n_frames` is implicit from file size:
//     n_frames = (file_size - 24) / (n_atoms * 3 * 4)
//
// Python loader (reference):
//
//     import numpy as np
//     header_dt = np.dtype([
//         ('magic', 'S8'), ('version', '<u4'), ('n_atoms', '<u4'),
//         ('save_interval', '<u4'), ('dt_ps', '<f4'),
//     ])
//     with open("4lpk.frames.bin", "rb") as fh:
//         hdr = np.fromfile(fh, dtype=header_dt, count=1)[0]
//         n_atoms = int(hdr['n_atoms'])
//         frames = np.fromfile(fh, dtype='<f4').reshape(-1, n_atoms, 3)

/// Magic header for the `frames.bin` format. Pinned by
/// [`TrajectoryWriter::write_header`] and the Python decoder reference
/// in the module-level documentation.
pub const FRAMES_BIN_MAGIC: [u8; 8] = *b"PRISM4D\0";

/// Format version emitted by [`TrajectoryWriter`]. Bump in the same
/// commit as any header-layout change. Consumers MUST refuse to load
/// files whose version does not match their expected version.
pub const FRAMES_BIN_VERSION: u32 = 1;

/// Buffered writer for the binary `<output>.frames.bin` ensemble
/// trajectory format. One writer per output stream. Writes the
/// 24-byte header on construction; `write_frame` appends a single
/// `[n_atoms × 3] f32 LE` frame per call. `finish` flushes the
/// underlying [`std::io::BufWriter`].
///
/// # Determinism
///
/// Each `write_frame` call emits `n_atoms * 3 * 4` bytes — same input,
/// same output, byte-for-byte. No timestamps in the file body, no
/// hash. The header is stamped once at construction and immutable
/// thereafter.
///
/// # Errors
///
/// Returns the underlying [`std::io::Error`] on any I/O failure
/// (file-create, write, flush). The struct holds the file handle for
/// the lifetime of the writer; `finish` flushes and closes.
pub struct TrajectoryWriter {
    writer: std::io::BufWriter<std::fs::File>,
    n_atoms: u32,
    save_interval: u32,
    dt_ps: f32,
    frames_written: u64,
    expected_floats_per_frame: usize,
}

impl TrajectoryWriter {
    /// Open `path` for writing, write the 24-byte header, and return
    /// a writer ready to accept frames via [`Self::write_frame`].
    ///
    /// `n_atoms` MUST match every subsequent frame's atom count.
    /// `save_interval` is recorded in the header for downstream tools
    /// (Python post-processing typically needs it to recover wall-time
    /// stride between frames). `dt_ps` is the simulator's per-step time
    /// in picoseconds at the time of capture; for engines with adaptive
    /// timestepping, callers SHOULD record the dt at the start of
    /// capture and accept that downstream-time scaling assumes constant
    /// dt over the captured window.
    pub fn create(
        path: impl AsRef<std::path::Path>,
        n_atoms: u32,
        save_interval: u32,
        dt_ps: f32,
    ) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(true).open(path)?;
        let mut writer = std::io::BufWriter::new(file);
        Self::write_header(&mut writer, n_atoms, save_interval, dt_ps)?;
        Ok(Self {
            writer,
            n_atoms,
            save_interval,
            dt_ps,
            frames_written: 0,
            expected_floats_per_frame: (n_atoms as usize) * 3,
        })
    }

    /// Internal: write the 24-byte header to a fresh writer. Pinned
    /// by `tests::trajectory_writer_header_layout`.
    fn write_header(
        w: &mut impl std::io::Write,
        n_atoms: u32,
        save_interval: u32,
        dt_ps: f32,
    ) -> std::io::Result<()> {
        w.write_all(&FRAMES_BIN_MAGIC)?;
        w.write_all(&FRAMES_BIN_VERSION.to_le_bytes())?;
        w.write_all(&n_atoms.to_le_bytes())?;
        w.write_all(&save_interval.to_le_bytes())?;
        w.write_all(&dt_ps.to_le_bytes())?;
        Ok(())
    }

    /// Append a single frame's worth of atom positions. `positions`
    /// MUST be flat `[x0, y0, z0, x1, y1, z1, …]` of length
    /// `n_atoms * 3` — same convention as
    /// `crate::fused_engine::EnsembleSnapshot::positions` and
    /// `crate::input::PrismPrepTopology::positions`. Mismatched length
    /// returns [`std::io::ErrorKind::InvalidInput`] without writing
    /// any partial frame.
    pub fn write_frame(&mut self, positions: &[f32]) -> std::io::Result<()> {
        if positions.len() != self.expected_floats_per_frame {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "TrajectoryWriter expects {} floats per frame ({} atoms × 3), got {}",
                    self.expected_floats_per_frame,
                    self.n_atoms,
                    positions.len()
                ),
            ));
        }
        // Per-float `to_le_bytes` write. The BufWriter batches these
        // small writes into a single underlying syscall per fill of
        // the 8 KB default buffer. Avoids an external bytemuck dep
        // and is portable across endianness without ifdef.
        for &f in positions {
            self.writer.write_all(&f.to_le_bytes())?;
        }
        self.frames_written += 1;
        Ok(())
    }

    /// Number of frames written so far (excluding the header).
    pub fn frames_written(&self) -> u64 {
        self.frames_written
    }

    /// Atoms-per-frame this writer expects.
    pub fn n_atoms(&self) -> u32 {
        self.n_atoms
    }

    /// Save interval in MD steps that the writer was constructed with.
    pub fn save_interval(&self) -> u32 {
        self.save_interval
    }

    /// Per-step time in picoseconds recorded in the header.
    pub fn dt_ps(&self) -> f32 {
        self.dt_ps
    }

    /// Flush the underlying buffer and consume the writer. Call this
    /// at the end of the run; the file is closed when the writer is
    /// dropped.
    pub fn finish(mut self) -> std::io::Result<u64> {
        self.writer.flush()?;
        Ok(self.frames_written)
    }
}

/// Result from processing a single frame
#[derive(Debug, Clone)]
pub struct FrameResult {
    /// Frame number
    pub frame: usize,
    /// Number of spikes detected
    pub spike_count: usize,
    /// Spike positions in world coordinates
    pub spike_positions: Vec<[f32; 3]>,
    /// Nearest residue for each spike
    pub spike_residues: Vec<i32>,
    /// Distance to nearest atom for each spike
    pub spike_distances: Vec<f32>,
}

// ============================================================================
// CPU FALLBACK (NO GPU)
// ============================================================================

#[cfg(not(feature = "gpu"))]
pub struct NhsGpuEngine;

#[cfg(not(feature = "gpu"))]
impl NhsGpuEngine {
    pub fn new(_grid_dim: usize, _max_atoms: usize) -> Result<Self> {
        Err(anyhow::anyhow!(
            "NHS GPU Engine requires GPU feature. Compile with --features gpu"
        ))
    }
}

// ============================================================================
// TESTS — TrajectoryWriter (Gate G2). Not GPU-gated; pure I/O.
// ============================================================================

#[cfg(test)]
mod trajectory_writer_tests {
    use super::*;
    use std::io::Read;

    /// Build a trivial 3-atom topology: 3 atoms × 3 floats = 9 f32 = 36 bytes/frame.
    fn synth_positions(seed: f32, n_atoms: u32) -> Vec<f32> {
        (0..(n_atoms as usize) * 3)
            .map(|i| seed + (i as f32) * 0.1)
            .collect()
    }

    #[test]
    fn header_layout_pinned() {
        // Pin every byte of the 24-byte header against the documented
        // schema. Any change to FRAMES_BIN_MAGIC, FRAMES_BIN_VERSION,
        // or the field order is a schema-drift bug and will fail this
        // test.
        let path = std::env::temp_dir().join(format!(
            "g2_trajectory_writer_header_{}.bin",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let writer = TrajectoryWriter::create(&path, 2467, 100, 0.004).unwrap();
        // Drop without writing frames; just header.
        let frames = writer.finish().unwrap();
        assert_eq!(frames, 0);

        let mut bytes = Vec::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        assert_eq!(bytes.len(), 24, "header must be exactly 24 bytes");
        assert_eq!(&bytes[0..8], FRAMES_BIN_MAGIC.as_slice());
        assert_eq!(&bytes[8..12], &FRAMES_BIN_VERSION.to_le_bytes());
        assert_eq!(&bytes[12..16], &2467u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &100u32.to_le_bytes());
        assert_eq!(&bytes[20..24], &0.004f32.to_le_bytes());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_frame_roundtrip() {
        // Write 3 frames of a 5-atom system (15 floats per frame =
        // 60 bytes). Verify file size, verify each frame's bytes
        // round-trip back to the original f32 values.
        let path = std::env::temp_dir().join(format!(
            "g2_trajectory_writer_roundtrip_{}.bin",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let n_atoms = 5u32;
        let mut writer = TrajectoryWriter::create(&path, n_atoms, 200, 0.002).unwrap();
        let f0 = synth_positions(0.0, n_atoms);
        let f1 = synth_positions(10.0, n_atoms);
        let f2 = synth_positions(-5.5, n_atoms);
        writer.write_frame(&f0).unwrap();
        writer.write_frame(&f1).unwrap();
        writer.write_frame(&f2).unwrap();
        let frames = writer.finish().unwrap();
        assert_eq!(frames, 3);

        let bytes = std::fs::read(&path).unwrap();
        // 24 (header) + 3 frames × 5 atoms × 3 × 4 bytes = 24 + 180 = 204
        assert_eq!(bytes.len(), 24 + 3 * (n_atoms as usize) * 3 * 4);

        // Decode each frame.
        let frame_bytes_each = (n_atoms as usize) * 3 * 4;
        let mut cursor = 24usize;
        for expected in [&f0, &f1, &f2] {
            let mut decoded = Vec::with_capacity(expected.len());
            for i in 0..expected.len() {
                let off = cursor + i * 4;
                let arr = [bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]];
                decoded.push(f32::from_le_bytes(arr));
            }
            assert_eq!(&decoded, expected);
            cursor += frame_bytes_each;
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn write_frame_rejects_wrong_length() {
        let path = std::env::temp_dir().join(format!(
            "g2_trajectory_writer_wrong_len_{}.bin",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let mut writer = TrajectoryWriter::create(&path, 4, 1, 0.002).unwrap();
        // Expect 12 floats; supply 11.
        let bad = vec![0.0f32; 11];
        let err = writer.write_frame(&bad).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(writer.frames_written(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writer_introspectors_pin_constructor_args() {
        let path = std::env::temp_dir().join(format!(
            "g2_trajectory_writer_introspect_{}.bin",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&path);
        let writer = TrajectoryWriter::create(&path, 12345, 999, 0.0040).unwrap();
        assert_eq!(writer.n_atoms(), 12345);
        assert_eq!(writer.save_interval(), 999);
        assert_eq!(writer.dt_ps(), 0.0040);
        assert_eq!(writer.frames_written(), 0);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn frames_bin_constants_pinned() {
        // Schema-drift guard. Any change to magic / version requires
        // updating docs/phase_bits_schema.md (no, wrong doc) — actually
        // requires updating the format-decoder reference in the
        // module-level docstring of the TrajectoryWriter region, plus
        // the Python loader pseudocode there.
        assert_eq!(FRAMES_BIN_MAGIC, *b"PRISM4D\0");
        assert_eq!(FRAMES_BIN_VERSION, 1);
    }
}

// ============================================================================
// TESTS — GPU engine (require GPU; ignored in CI)
// ============================================================================

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_engine_creation() {
        env_logger::builder().is_test(true).try_init().ok();

        let context = CudaContext::new(0).expect("CUDA not available");
        let engine = NhsGpuEngine::new(context, 32, 1000);

        assert!(
            engine.is_ok(),
            "Failed to create GPU engine: {:?}",
            engine.err()
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_engine_basic_workflow() {
        env_logger::builder().is_test(true).try_init().ok();

        let context = CudaContext::new(0).expect("CUDA not available");
        let mut engine = NhsGpuEngine::new(context, 32, 100).expect("Failed to create engine");

        // Initialize
        engine
            .initialize([0.0, 0.0, 0.0])
            .expect("Failed to initialize");

        // Create dummy atom data
        let positions = vec![0.0f32; 30]; // 10 atoms
        let types = vec![0i32; 10]; // All hydrophobic
        let charges = vec![0.0f32; 10];
        let residues = vec![0i32; 10];

        // Process frame
        let result = engine
            .process_frame(&positions, &types, &charges, &residues)
            .expect("Failed to process frame");

        assert_eq!(result.frame, 1);
    }
}


#[derive(Clone,Debug,serde::Serialize,serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrajectoryContinuationV1 {
    schema_version:u32,cursor:crate::scientific_checkpoint::AppendCursor,
    n_atoms:u32,save_interval:u32,dt_ps_bits:u32,frames_written:u64,
}
impl TrajectoryWriter {
    pub fn checkpoint_continuation(&mut self,relative_path:&str)->anyhow::Result<TrajectoryContinuationV1> {
        use std::io::Write;
        self.writer.flush()?;
        let cursor=crate::scientific_checkpoint::seal_append_file(self.writer.get_ref(),relative_path)?;
        let expected=24u64.checked_add(self.frames_written.checked_mul(self.n_atoms as u64)
            .and_then(|n|n.checked_mul(12)).ok_or_else(||anyhow::anyhow!("trajectory size overflow"))?)
            .ok_or_else(||anyhow::anyhow!("trajectory size overflow"))?;
        anyhow::ensure!(cursor.committed_bytes==expected,"trajectory file/frame population mismatch");
        Ok(TrajectoryContinuationV1{schema_version:1,cursor,n_atoms:self.n_atoms,
            save_interval:self.save_interval,dt_ps_bits:self.dt_ps.to_bits(),frames_written:self.frames_written})
    }
    pub fn reopen_continuation(root:&std::path::Path,saved:&TrajectoryContinuationV1)->anyhow::Result<Self> {
        use std::io::{Read,Seek,SeekFrom};
        anyhow::ensure!(saved.schema_version==1 && saved.n_atoms>0 && saved.save_interval>0,
            "trajectory restart schema/population invalid");
        let mut file=crate::scientific_checkpoint::reopen_append_file(root,&saved.cursor)?;
        file.seek(SeekFrom::Start(0))?;let mut header=[0u8;24];file.read_exact(&mut header)?;
        let mut expected=Vec::new();Self::write_header(&mut expected,saved.n_atoms,saved.save_interval,f32::from_bits(saved.dt_ps_bits))?;
        anyhow::ensure!(header.as_slice()==expected,"trajectory header differs from continuation identity");
        let expected_bytes=24u64.checked_add(saved.frames_written.checked_mul(saved.n_atoms as u64)
            .and_then(|n|n.checked_mul(12)).ok_or_else(||anyhow::anyhow!("trajectory size overflow"))?)
            .ok_or_else(||anyhow::anyhow!("trajectory size overflow"))?;
        anyhow::ensure!(saved.cursor.committed_bytes==expected_bytes,"trajectory frame/cursor mismatch");
        file.seek(SeekFrom::Start(saved.cursor.committed_bytes))?;
        Ok(Self{writer:std::io::BufWriter::new(file),n_atoms:saved.n_atoms,save_interval:saved.save_interval,
            dt_ps:f32::from_bits(saved.dt_ps_bits),frames_written:saved.frames_written,
            expected_floats_per_frame:saved.n_atoms as usize*3})
    }
}

impl TrajectoryContinuationV1 {
    pub fn branch_copy_allocation_bound(&self)->anyhow::Result<u64> {
        crate::scientific_checkpoint::restart_transaction::branch_copy_member_bound(self.cursor.committed_bytes)
    }

    pub fn fork_outputs(&self,source:&std::path::Path,destination:&std::path::Path)->anyhow::Result<()> {
        anyhow::ensure!(self.schema_version==1,"unsupported historical resident trajectory schema");
        crate::scientific_checkpoint::restart_transaction::copy_committed_append_prefix(source,destination,&self.cursor)
    }
}
