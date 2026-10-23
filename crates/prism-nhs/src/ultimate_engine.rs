//! Ultimate Hyperoptimized MD Engine Integration
//!
//! This module provides a wrapper for the ultimate_md.cu kernel that implements
//! all 14 GPU optimizations for maximum performance:
//!
//! 1. Occupancy tuning (__launch_bounds__)
//! 2. Texture memory for cached reads
//! 3. Constant memory for parameters
//! 4. Double buffering
//! 5. Cooperative groups
//! 6. Dynamic parallelism
//! 7. Multi-GPU P2P
//! 8. Mixed precision (FP16 compute, FP32 accumulate)
//! 9. ILP unrolling
//! 10. Memory coalescing (SoA layout)
//! 11. L2 persistence hints
//! 12. Async memcpy
//! 13. Graph optimization
//! 14. Template specialization
//!
//! Expected speedup: 2-4x over standard kernel

use anyhow::{bail, Context, Result};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut,
    LaunchConfig, PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

use crate::input::PrismPrepTopology;

/// Optimization level for kernel selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum OptimizationLevel {
    /// Standard kernel - compatible with all GPUs
    Standard,
    /// Hyperoptimized kernel - requires SM86+ (Ampere/Ada)
    Ultimate,
    /// Auto-select based on GPU capability
    Auto,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::Auto
    }
}

/// Configuration for the ultimate engine
#[derive(Debug, Clone)]
pub struct UltimateEngineConfig {
    /// Optimization level to use
    pub optimization_level: OptimizationLevel,
    /// Enable multi-GPU if available
    pub enable_multi_gpu: bool,
    /// Enable mixed precision (FP16)
    pub enable_mixed_precision: bool,
    /// Block size for compute-heavy kernels
    pub compute_block_size: u32,
    /// Neighbor list rebuild interval
    pub neighbor_rebuild_interval: i32,
}

impl Default for UltimateEngineConfig {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Auto,
            enable_multi_gpu: false,
            enable_mixed_precision: true,
            compute_block_size: 128,
            neighbor_rebuild_interval: 20,
        }
    }
}

/// Simulation parameters that go into constant memory
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SimulationParams {
    // Physical constants
    pub coulomb_const: f32,  // 332.0636 kcal*Å/(mol*e²)
    pub kb: f32,             // 0.001987204 kcal/(mol*K)
    pub force_to_accel: f32, // 4.184e-4 (AKMA units)

    // Cutoffs
    pub nb_cutoff: f32,          // 12.0 Å
    pub nb_cutoff_sq: f32,       // 144.0 Å²
    pub switch_start: f32,       // 10.0 Å
    pub switch_start_sq: f32,    // 100.0 Å²
    pub soft_core_delta_sq: f32, // 0.01 Å²

    // Implicit solvent
    pub dielectric_scale: f32, // 0.25 (ε=4r)

    // Integration
    pub dt: f32,          // Timestep (fs)
    pub half_dt: f32,     // dt/2
    pub temperature: f32, // Target temperature (K)
    pub gamma: f32,       // Langevin friction coefficient
    pub noise_scale: f32, // sqrt(2*gamma*kT*dt)

    // Limits
    pub max_velocity: f32, // 0.2 Å/fs
    pub max_force: f32,    // 300 kcal/(mol*Å)

    // Grid dimensions
    pub grid_dim: i32,         // Voxel grid dimension
    pub grid_spacing: f32,     // Å per voxel
    pub grid_origin: [f32; 3], // Grid origin coordinates

    // System size
    pub n_atoms: i32,
    pub n_bonds: i32,
    pub n_angles: i32,
    pub n_dihedrals: i32,
    pub max_exclusions: i32,

    // PME parameters
    pub use_pme: i32,
    pub ewald_beta: f32,

    // Current step
    pub step: u32,
}

impl Default for SimulationParams {
    fn default() -> Self {
        Self {
            coulomb_const: 332.0636,
            kb: 0.001987204,
            force_to_accel: 4.184e-4,
            nb_cutoff: 12.0,
            nb_cutoff_sq: 144.0,
            switch_start: 10.0,
            switch_start_sq: 100.0,
            soft_core_delta_sq: 0.01,
            dielectric_scale: 0.25,
            dt: 0.002,
            half_dt: 0.001,
            temperature: 300.0,
            gamma: 1.0,
            noise_scale: 0.0,
            max_velocity: 0.2,
            max_force: 300.0,
            grid_dim: 64,
            grid_spacing: 1.0,
            grid_origin: [0.0, 0.0, 0.0],
            n_atoms: 0,
            n_bonds: 0,
            n_angles: 0,
            n_dihedrals: 0,
            max_exclusions: 32,
            use_pme: 0,
            ewald_beta: 0.0,
            step: 0,
        }
    }
}

/// Result from a single MD step
#[derive(Debug, Clone)]
pub struct UltimateStepResult {
    pub timestep: i32,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub temperature: f32,
}

#[cfg(feature = "gpu")]
pub struct UltimateEngine {
    // CUDA handles
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _ultimate_module: Arc<CudaModule>,

    // Kernel function
    ultimate_step_kernel: CudaFunction,

    // SoA position buffers (for coalesced access)
    d_pos_x: CudaSlice<f32>,
    d_pos_y: CudaSlice<f32>,
    d_pos_z: CudaSlice<f32>,

    // SoA velocity buffers
    d_vel_x: CudaSlice<f32>,
    d_vel_y: CudaSlice<f32>,
    d_vel_z: CudaSlice<f32>,

    // SoA force buffers
    d_force_x: CudaSlice<f32>,
    d_force_y: CudaSlice<f32>,
    d_force_z: CudaSlice<f32>,

    // Energy accumulators
    d_potential_energy: CudaSlice<f32>,
    d_kinetic_energy: CudaSlice<f32>,

    // Parameters (SoA)
    d_sigma: CudaSlice<f32>,
    d_epsilon: CudaSlice<f32>,
    d_charge: CudaSlice<f32>,
    d_mass: CudaSlice<f32>,

    // Bond topology (SoA)
    d_bond_i: CudaSlice<i32>,
    d_bond_j: CudaSlice<i32>,
    d_bond_r0: CudaSlice<f32>,
    d_bond_k: CudaSlice<f32>,

    // Exclusions
    d_excl_list: CudaSlice<i32>,
    d_n_excl: CudaSlice<i32>,

    // System info
    n_atoms: usize,
    n_bonds: usize,
    max_exclusions: usize,

    // Configuration
    config: UltimateEngineConfig,
    params: SimulationParams,
    timestep: i32,
}

#[cfg(feature = "gpu")]
impl UltimateEngine {
    /// Create a new ultimate engine from topology
    pub fn new(
        context: Arc<CudaContext>,
        topology: &PrismPrepTopology,
        config: UltimateEngineConfig,
    ) -> Result<Self> {
        log::info!(
            "Creating Ultimate Hyperoptimized Engine: {} atoms",
            topology.n_atoms
        );

        let stream = context.default_stream();
        let n_atoms = topology.n_atoms;
        let n_bonds = topology.bonds.len();
        let max_exclusions = 32;

        // Load PTX module
        let (ultimate_module, loaded_path) = Self::load_ultimate_ptx(&context)?;
        log::info!("Loaded ultimate_md.ptx from: {}", loaded_path);

        // Get kernel function
        let ultimate_step_kernel = ultimate_module.load_function("ultimate_md_step")?;

        // Allocate SoA buffers for positions
        let d_pos_x: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_pos_y: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_pos_z: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;

        // Allocate SoA buffers for velocities
        let d_vel_x: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_vel_y: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_vel_z: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;

        // Allocate SoA buffers for forces
        let d_force_x: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_force_y: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_force_z: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;

        // Energy accumulators
        let d_potential_energy: CudaSlice<f32> = stream.alloc_zeros(1)?;
        let d_kinetic_energy: CudaSlice<f32> = stream.alloc_zeros(1)?;

        // Parameters (SoA)
        let d_sigma: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_epsilon: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_charge: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_mass: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;

        // Bond topology (SoA)
        let d_bond_i: CudaSlice<i32> = stream.alloc_zeros(n_bonds.max(1))?;
        let d_bond_j: CudaSlice<i32> = stream.alloc_zeros(n_bonds.max(1))?;
        let d_bond_r0: CudaSlice<f32> = stream.alloc_zeros(n_bonds.max(1))?;
        let d_bond_k: CudaSlice<f32> = stream.alloc_zeros(n_bonds.max(1))?;

        // Exclusions (flat array: n_atoms * max_exclusions)
        let d_excl_list: CudaSlice<i32> = stream.alloc_zeros(n_atoms * max_exclusions)?;
        let d_n_excl: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;

        // Initialize parameters
        let mut params = SimulationParams::default();
        params.n_atoms = n_atoms as i32;
        params.n_bonds = n_bonds as i32;
        params.max_exclusions = max_exclusions as i32;

        let mut engine = Self {
            context,
            stream,
            _ultimate_module: ultimate_module,
            ultimate_step_kernel,
            d_pos_x,
            d_pos_y,
            d_pos_z,
            d_vel_x,
            d_vel_y,
            d_vel_z,
            d_force_x,
            d_force_y,
            d_force_z,
            d_potential_energy,
            d_kinetic_energy,
            d_sigma,
            d_epsilon,
            d_charge,
            d_mass,
            d_bond_i,
            d_bond_j,
            d_bond_r0,
            d_bond_k,
            d_excl_list,
            d_n_excl,
            n_atoms,
            n_bonds,
            max_exclusions,
            config,
            params,
            timestep: 0,
        };

        // Upload topology data
        engine.upload_topology(topology)?;

        Ok(engine)
    }

    /// Load the ultimate_md.ptx module
    fn load_ultimate_ptx(context: &Arc<CudaContext>) -> Result<(Arc<CudaModule>, String)> {
        // Check env var first
        if let Ok(env_dir) = std::env::var("PRISM4D_PTX_DIR") {
            let ptx_path = std::path::PathBuf::from(&env_dir).join("ultimate_md.ptx");
            let path_str = ptx_path.display().to_string();

            if ptx_path.exists() {
                let module = context
                    .load_module(Ptx::from_file(&path_str))
                    .with_context(|| format!("Failed to load ultimate_md.ptx from {}", path_str))?;
                return Ok((module, path_str));
            }
        }

        // Try fallback paths
        let mut ptx_paths: Vec<std::path::PathBuf> = Vec::new();

        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                ptx_paths.push(exe_dir.join("../assets/ptx/ultimate_md.ptx"));
                ptx_paths.push(exe_dir.join("assets/ptx/ultimate_md.ptx"));
            }
        }

        ptx_paths.push(std::path::PathBuf::from("target/ptx/ultimate_md.ptx"));
        ptx_paths.push(std::path::PathBuf::from("../../target/ptx/ultimate_md.ptx"));
        ptx_paths.push(std::path::PathBuf::from(
            "crates/prism-gpu/src/kernels/ultimate_md.ptx",
        ));

        for path in &ptx_paths {
            let path_str = path.display().to_string();
            if path.exists() {
                if let Ok(module) = context.load_module(Ptx::from_file(&path_str)) {
                    return Ok((module, path_str));
                }
            }
        }

        bail!("Failed to load ultimate_md.ptx from any location")
    }

    /// Upload topology data to GPU in SoA format
    fn upload_topology(&mut self, topology: &PrismPrepTopology) -> Result<()> {
        let harmonic = topology.lower_harmonic_parameters(
            crate::input::HarmonicEnergyConvention::OpenmmFullKHalfEnergy,
        ).map_err(anyhow::Error::msg)?;
        // Convert positions from AoS to SoA
        let mut pos_x = vec![0.0f32; self.n_atoms];
        let mut pos_y = vec![0.0f32; self.n_atoms];
        let mut pos_z = vec![0.0f32; self.n_atoms];

        for i in 0..self.n_atoms {
            pos_x[i] = topology.positions[i * 3] as f32;
            pos_y[i] = topology.positions[i * 3 + 1] as f32;
            pos_z[i] = topology.positions[i * 3 + 2] as f32;
        }

        self.stream.memcpy_htod(&pos_x, &mut self.d_pos_x)?;
        self.stream.memcpy_htod(&pos_y, &mut self.d_pos_y)?;
        self.stream.memcpy_htod(&pos_z, &mut self.d_pos_z)?;

        // Upload parameters
        let sigma: Vec<f32> = topology
            .lj_params
            .iter()
            .map(|lj| lj.sigma as f32)
            .collect();
        let epsilon: Vec<f32> = topology
            .lj_params
            .iter()
            .map(|lj| lj.epsilon as f32)
            .collect();
        let charge: Vec<f32> = topology.charges.iter().map(|&c| c as f32).collect();
        let mass: Vec<f32> = topology.masses.iter().map(|&m| m as f32).collect();

        self.stream.memcpy_htod(&sigma, &mut self.d_sigma)?;
        self.stream.memcpy_htod(&epsilon, &mut self.d_epsilon)?;
        self.stream.memcpy_htod(&charge, &mut self.d_charge)?;
        self.stream.memcpy_htod(&mass, &mut self.d_mass)?;

        // Upload bonds in SoA format
        if !topology.bonds.is_empty() {
            let bond_i: Vec<i32> = topology.bonds.iter().map(|b| b.i as i32).collect();
            let bond_j: Vec<i32> = topology.bonds.iter().map(|b| b.j as i32).collect();
            let bond_r0: Vec<f32> = topology.bonds.iter().map(|b| b.r0 as f32).collect();
            let bond_k = harmonic.bond_k;

            self.stream.memcpy_htod(&bond_i, &mut self.d_bond_i)?;
            self.stream.memcpy_htod(&bond_j, &mut self.d_bond_j)?;
            self.stream.memcpy_htod(&bond_r0, &mut self.d_bond_r0)?;
            self.stream.memcpy_htod(&bond_k, &mut self.d_bond_k)?;
        }

        // Upload exclusions (flatten to fixed-size array per atom)
        let mut excl_flat = vec![-1i32; self.n_atoms * self.max_exclusions];
        let mut n_excl = vec![0i32; self.n_atoms];

        for (atom_idx, excl_list) in topology.exclusions.iter().enumerate() {
            let count = excl_list.len().min(self.max_exclusions);
            n_excl[atom_idx] = count as i32;
            for (j, &excl_atom) in excl_list.iter().take(count).enumerate() {
                excl_flat[atom_idx * self.max_exclusions + j] = excl_atom as i32;
            }
        }

        self.stream.memcpy_htod(&excl_flat, &mut self.d_excl_list)?;
        self.stream.memcpy_htod(&n_excl, &mut self.d_n_excl)?;

        Ok(())
    }

    /// Set simulation parameters
    pub fn set_params(&mut self, params: SimulationParams) {
        self.params = params;
        self.params.n_atoms = self.n_atoms as i32;
        self.params.n_bonds = self.n_bonds as i32;
        self.params.max_exclusions = self.max_exclusions as i32;
    }

    /// Update positions from AoS array
    pub fn update_positions(&mut self, positions: &[f64]) -> Result<()> {
        let mut pos_x = vec![0.0f32; self.n_atoms];
        let mut pos_y = vec![0.0f32; self.n_atoms];
        let mut pos_z = vec![0.0f32; self.n_atoms];

        for i in 0..self.n_atoms {
            pos_x[i] = positions[i * 3] as f32;
            pos_y[i] = positions[i * 3 + 1] as f32;
            pos_z[i] = positions[i * 3 + 2] as f32;
        }

        self.stream.memcpy_htod(&pos_x, &mut self.d_pos_x)?;
        self.stream.memcpy_htod(&pos_y, &mut self.d_pos_y)?;
        self.stream.memcpy_htod(&pos_z, &mut self.d_pos_z)?;

        Ok(())
    }

    /// Get positions as AoS array
    pub fn get_positions(&self) -> Result<Vec<f32>> {
        let mut pos_x = vec![0.0f32; self.n_atoms];
        let mut pos_y = vec![0.0f32; self.n_atoms];
        let mut pos_z = vec![0.0f32; self.n_atoms];

        self.stream.memcpy_dtoh(&self.d_pos_x, &mut pos_x)?;
        self.stream.memcpy_dtoh(&self.d_pos_y, &mut pos_y)?;
        self.stream.memcpy_dtoh(&self.d_pos_z, &mut pos_z)?;

        // Convert back to AoS
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        for i in 0..self.n_atoms {
            positions[i * 3] = pos_x[i];
            positions[i * 3 + 1] = pos_y[i];
            positions[i * 3 + 2] = pos_z[i];
        }

        Ok(positions)
    }

    /// Run a single timestep with the ultimate kernel
    pub fn step(&mut self) -> Result<UltimateStepResult> {
        let block_size = self.config.compute_block_size;
        let n_blocks = (self.n_atoms as u32).div_ceil(block_size);

        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;

        unsafe {
            self.stream
                .launch_builder(&self.ultimate_step_kernel)
                // Positions (SoA)
                .arg(&mut self.d_pos_x)
                .arg(&mut self.d_pos_y)
                .arg(&mut self.d_pos_z)
                // Velocities (SoA)
                .arg(&mut self.d_vel_x)
                .arg(&mut self.d_vel_y)
                .arg(&mut self.d_vel_z)
                // Forces (SoA)
                .arg(&mut self.d_force_x)
                .arg(&mut self.d_force_y)
                .arg(&mut self.d_force_z)
                // Energies
                .arg(&mut self.d_potential_energy)
                .arg(&mut self.d_kinetic_energy)
                // Parameters (SoA)
                .arg(&self.d_sigma)
                .arg(&self.d_epsilon)
                .arg(&self.d_charge)
                .arg(&self.d_mass)
                // Bonds (SoA)
                .arg(&self.d_bond_i)
                .arg(&self.d_bond_j)
                .arg(&self.d_bond_r0)
                .arg(&self.d_bond_k)
                // Exclusions
                .arg(&self.d_excl_list)
                .arg(&self.d_n_excl)
                // Sizes
                .arg(&n_atoms_i32)
                .arg(&n_bonds_i32)
                .launch(cfg)
        }
        .context("Failed to launch ultimate_md_step kernel")?;

        self.timestep += 1;

        // Read energies
        let mut pe = [0.0f32];
        let mut ke = [0.0f32];
        self.stream.memcpy_dtoh(&self.d_potential_energy, &mut pe)?;
        self.stream.memcpy_dtoh(&self.d_kinetic_energy, &mut ke)?;

        Ok(UltimateStepResult {
            timestep: self.timestep,
            potential_energy: pe[0] as f64,
            kinetic_energy: ke[0] as f64,
            temperature: self.params.temperature,
        })
    }

    /// Run multiple steps without intermediate sync
    pub fn step_batch(&mut self, n_steps: i32) -> Result<UltimateStepResult> {
        for _ in 0..n_steps {
            let block_size = self.config.compute_block_size;
            let n_blocks = (self.n_atoms as u32).div_ceil(block_size);

            let cfg = LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            let n_atoms_i32 = self.n_atoms as i32;
            let n_bonds_i32 = self.n_bonds as i32;

            unsafe {
                self.stream
                    .launch_builder(&self.ultimate_step_kernel)
                    .arg(&mut self.d_pos_x)
                    .arg(&mut self.d_pos_y)
                    .arg(&mut self.d_pos_z)
                    .arg(&mut self.d_vel_x)
                    .arg(&mut self.d_vel_y)
                    .arg(&mut self.d_vel_z)
                    .arg(&mut self.d_force_x)
                    .arg(&mut self.d_force_y)
                    .arg(&mut self.d_force_z)
                    .arg(&mut self.d_potential_energy)
                    .arg(&mut self.d_kinetic_energy)
                    .arg(&self.d_sigma)
                    .arg(&self.d_epsilon)
                    .arg(&self.d_charge)
                    .arg(&self.d_mass)
                    .arg(&self.d_bond_i)
                    .arg(&self.d_bond_j)
                    .arg(&self.d_bond_r0)
                    .arg(&self.d_bond_k)
                    .arg(&self.d_excl_list)
                    .arg(&self.d_n_excl)
                    .arg(&n_atoms_i32)
                    .arg(&n_bonds_i32)
                    .launch(cfg)
            }
            .context("Failed to launch ultimate_md_step kernel")?;

            self.timestep += 1;
        }

        // Single sync at end
        self.context.synchronize()?;

        let mut pe = [0.0f32];
        let mut ke = [0.0f32];
        self.stream.memcpy_dtoh(&self.d_potential_energy, &mut pe)?;
        self.stream.memcpy_dtoh(&self.d_kinetic_energy, &mut ke)?;

        Ok(UltimateStepResult {
            timestep: self.timestep,
            potential_energy: pe[0] as f64,
            kinetic_energy: ke[0] as f64,
            temperature: self.params.temperature,
        })
    }

    /// Get number of atoms
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Get current timestep
    pub fn timestep(&self) -> i32 {
        self.timestep
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_params_size() {
        // Ensure struct matches CUDA constant memory layout
        let size = std::mem::size_of::<SimulationParams>();
        assert!(
            size < 64 * 1024,
            "SimulationParams must fit in constant memory"
        );
    }

    #[test]
    fn test_config_defaults() {
        let config = UltimateEngineConfig::default();
        assert_eq!(config.compute_block_size, 128);
        assert!(config.enable_mixed_precision);
    }
}
