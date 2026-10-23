//! ENSEMBLE WARP MD - Revolutionary Parallel MD for Conformational Sampling
//!
//! Key Innovation: Each WARP (32 threads) processes ONE CLONE independently
//! - Topology loaded ONCE into shared memory (broadcast to all warps)
//! - Warp shuffle operations for fast force reduction
//! - NO cross-clone synchronization = perfect scaling!
//!
//! Expected: N× speedup for N clones (theoretical)

use anyhow::{bail, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::collections::HashSet;
use std::sync::Arc;

/// Maximum atoms per structure for warp-based processing
/// Reduced from 512 to fit within 48KB shared memory budget
pub const MAX_ATOMS_WARP: usize = 128;

/// Warp size (NVIDIA GPUs)
pub const WARP_SIZE: usize = 32;

/// Boltzmann constant in kcal/(mol·K)
pub const KB_KCAL_MOL_K: f64 = 0.001987204;

/// Ensemble MD Result
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    pub clone_id: usize,
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub temperature: f64,
}

/// Topology for ensemble (shared across all clones)
#[derive(Debug, Clone)]
pub struct EnsembleTopology {
    pub n_atoms: usize,
    pub masses: Vec<f32>,
    pub charges: Vec<f32>,
    pub sigmas: Vec<f32>,
    pub epsilons: Vec<f32>,
    pub bonds: Vec<(usize, usize, f32, f32)>, // (i, j, k, r0)
    pub angles: Vec<(usize, usize, usize, f32, f32)>, // (i, j, k, k_angle, theta0)
}

/// Ensemble Warp MD Engine
pub struct EnsembleWarpMd {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,

    // Kernels
    md_kernel: CudaFunction,
    init_vel_kernel: CudaFunction,

    // Topology (shared - read-only)
    n_atoms: usize,
    d_masses: CudaSlice<f32>,
    d_charges: CudaSlice<f32>,
    d_sigmas: CudaSlice<f32>,
    d_epsilons: CudaSlice<f32>,

    // Bond topology
    n_bonds: usize,
    d_bond_atoms: CudaSlice<i32>,
    d_bond_params: CudaSlice<f32>,

    // Angle topology
    n_angles: usize,
    d_angle_atoms: CudaSlice<i32>,
    d_angle_params: CudaSlice<f32>,

    // Per-clone state
    n_clones: usize,
    d_positions: CudaSlice<f32>,
    d_velocities: CudaSlice<f32>,
    d_energies: CudaSlice<f32>,

    // Configuration
    initialized: bool,
}

impl EnsembleWarpMd {
    /// Create new ensemble MD engine
    pub fn new(
        context: Arc<CudaContext>,
        topology: &EnsembleTopology,
        n_clones: usize,
    ) -> Result<Self> {
        if topology.n_atoms > MAX_ATOMS_WARP {
            bail!(
                "Structure too large for warp-based MD: {} atoms (max {})",
                topology.n_atoms,
                MAX_ATOMS_WARP
            );
        }

        let stream = context.default_stream();

        // Compile the kernel
        let kernel_source = include_str!("kernels/ensemble_warp_md.cu");
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            kernel_source,
            cudarc::nvrtc::CompileOptions {
                arch: Some("sm_86"),
                include_paths: vec![],
                ftz: Some(true),
                prec_div: Some(false),
                prec_sqrt: Some(false),
                fmad: Some(true),
                ..Default::default()
            },
        )
        .context("Failed to compile ensemble_warp_md.cu")?;

        let module = context.load_module(ptx)?;
        let md_kernel = module
            .load_function("ensemble_warp_md_kernel")
            .context("ensemble_warp_md_kernel not found")?;
        let init_vel_kernel = module
            .load_function("ensemble_init_velocities_kernel")
            .context("ensemble_init_velocities_kernel not found")?;

        let n_atoms = topology.n_atoms;

        // Upload shared topology
        let mut d_masses = stream.alloc_zeros::<f32>(n_atoms)?;
        let mut d_charges = stream.alloc_zeros::<f32>(n_atoms)?;
        let mut d_sigmas = stream.alloc_zeros::<f32>(n_atoms)?;
        let mut d_epsilons = stream.alloc_zeros::<f32>(n_atoms)?;
        stream.memcpy_htod(&topology.masses, &mut d_masses)?;
        stream.memcpy_htod(&topology.charges, &mut d_charges)?;
        stream.memcpy_htod(&topology.sigmas, &mut d_sigmas)?;
        stream.memcpy_htod(&topology.epsilons, &mut d_epsilons)?;

        // Upload bonds
        let n_bonds = topology.bonds.len(); // Actual bond count for kernel
        let mut bond_atoms: Vec<i32> = Vec::with_capacity(n_bonds * 2);
        let mut bond_params: Vec<f32> = Vec::with_capacity(n_bonds * 2);
        for &(i, j, k, r0) in &topology.bonds {
            bond_atoms.push(i as i32);
            bond_atoms.push(j as i32);
            bond_params.push(k);
            bond_params.push(r0);
        }
        // Ensure we have at least some data
        if bond_atoms.is_empty() {
            bond_atoms.push(0);
            bond_atoms.push(0);
            bond_params.push(0.0);
            bond_params.push(0.0);
        }
        let mut d_bond_atoms = stream.alloc_zeros::<i32>(bond_atoms.len())?;
        let mut d_bond_params = stream.alloc_zeros::<f32>(bond_params.len())?;
        stream.memcpy_htod(&bond_atoms, &mut d_bond_atoms)?;
        stream.memcpy_htod(&bond_params, &mut d_bond_params)?;

        // Upload angles
        let n_angles = topology.angles.len(); // Actual angle count for kernel
        let mut angle_atoms: Vec<i32> = Vec::with_capacity(n_angles * 4);
        let mut angle_params: Vec<f32> = Vec::with_capacity(n_angles * 2);
        for &(i, j, k, k_angle, theta0) in &topology.angles {
            angle_atoms.push(i as i32);
            angle_atoms.push(j as i32);
            angle_atoms.push(k as i32);
            angle_atoms.push(0); // padding
            angle_params.push(k_angle);
            angle_params.push(theta0);
        }
        // Ensure we have at least some data
        if angle_atoms.is_empty() {
            angle_atoms.extend_from_slice(&[0, 0, 0, 0]);
            angle_params.push(0.0);
            angle_params.push(0.0);
        }
        let mut d_angle_atoms = stream.alloc_zeros::<i32>(angle_atoms.len())?;
        let mut d_angle_params = stream.alloc_zeros::<f32>(angle_params.len())?;
        stream.memcpy_htod(&angle_atoms, &mut d_angle_atoms)?;
        stream.memcpy_htod(&angle_params, &mut d_angle_params)?;

        // Allocate per-clone state
        let total_floats = n_clones * n_atoms * 3;
        let d_positions = stream.alloc_zeros::<f32>(total_floats)?;
        let d_velocities = stream.alloc_zeros::<f32>(total_floats)?;
        let d_energies = stream.alloc_zeros::<f32>(n_clones * 4)?;

        Ok(Self {
            context,
            stream,
            _module: module,
            md_kernel,
            init_vel_kernel,
            n_atoms,
            d_masses,
            d_charges,
            d_sigmas,
            d_epsilons,
            n_bonds,
            d_bond_atoms,
            d_bond_params,
            n_angles,
            d_angle_atoms,
            d_angle_params,
            n_clones,
            d_positions,
            d_velocities,
            d_energies,
            initialized: false,
        })
    }

    /// Set initial positions for all clones from a template
    pub fn set_positions(&mut self, template_positions: &[f32]) -> Result<()> {
        if template_positions.len() != self.n_atoms * 3 {
            bail!(
                "Position count mismatch: expected {}, got {}",
                self.n_atoms * 3,
                template_positions.len()
            );
        }

        // Replicate positions for all clones
        let mut all_positions = Vec::with_capacity(self.n_clones * self.n_atoms * 3);
        for _ in 0..self.n_clones {
            all_positions.extend_from_slice(template_positions);
        }

        self.stream
            .memcpy_htod(&all_positions, &mut self.d_positions)?;
        Ok(())
    }

    /// Initialize velocities from Maxwell-Boltzmann distribution
    pub fn initialize_velocities(&mut self, temperature: f32) -> Result<()> {
        let n_clones_i32 = self.n_clones as i32;
        let n_atoms_i32 = self.n_atoms as i32;
        let seed = 0x12345678u32;

        // Launch: enough warps for all clones
        let total_threads = self.n_clones * WARP_SIZE;
        let block_size = 128;
        let n_blocks = (total_threads + block_size - 1) / block_size;

        let cfg = LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            let mut builder = self.stream.launch_builder(&self.init_vel_kernel);
            builder.arg(&n_clones_i32);
            builder.arg(&n_atoms_i32);
            builder.arg(&temperature);
            builder.arg(&self.d_velocities);
            builder.arg(&self.d_masses);
            builder.arg(&seed);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;
        self.initialized = true;
        Ok(())
    }

    /// Run MD simulation for all clones in parallel
    pub fn run(&mut self, n_steps: usize, dt: f32, temperature: f32, gamma: f32) -> Result<()> {
        if !self.initialized {
            bail!("Velocities not initialized - call initialize_velocities first");
        }

        let n_clones_i32 = self.n_clones as i32;
        let n_atoms_i32 = self.n_atoms as i32;
        let n_steps_i32 = n_steps as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let seed = 0x87654321u32;

        // Launch configuration: 4 warps per block, each warp = 1 clone
        let warps_per_block = 4;
        let threads_per_block = warps_per_block * WARP_SIZE;
        let n_blocks = (self.n_clones + warps_per_block - 1) / warps_per_block;

        // Shared memory for force arrays: 4 warps × n_atoms × 3 × sizeof(float)
        let shared_mem = (warps_per_block * self.n_atoms * 3 * 4) as u32;

        let cfg = LaunchConfig {
            grid_dim: (n_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: shared_mem,
        };

        log::info!(
            "Launching ensemble MD: {} clones × {} atoms × {} steps",
            self.n_clones,
            self.n_atoms,
            n_steps
        );
        log::info!(
            "  Grid: {}×1×1, Block: {}×1×1, Shared: {} bytes",
            n_blocks,
            threads_per_block,
            shared_mem
        );

        unsafe {
            let mut builder = self.stream.launch_builder(&self.md_kernel);
            builder.arg(&n_clones_i32);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_steps_i32);
            builder.arg(&dt);
            builder.arg(&temperature);
            builder.arg(&gamma);
            builder.arg(&self.d_positions);
            builder.arg(&self.d_velocities);
            builder.arg(&self.d_masses);
            builder.arg(&self.d_charges);
            builder.arg(&self.d_sigmas);
            builder.arg(&self.d_epsilons);
            builder.arg(&self.d_bond_atoms);
            builder.arg(&self.d_bond_params);
            builder.arg(&self.d_angle_atoms);
            builder.arg(&self.d_angle_params);
            builder.arg(&n_bonds_i32);
            builder.arg(&n_angles_i32);
            builder.arg(&self.d_energies);
            builder.arg(&seed);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;
        Ok(())
    }

    /// Get results for all clones
    pub fn get_results(&self) -> Result<Vec<EnsembleResult>> {
        // Download positions
        let total_pos = self.n_clones * self.n_atoms * 3;
        let mut positions = vec![0.0f32; total_pos];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        // Download velocities
        let mut velocities = vec![0.0f32; total_pos];
        self.stream
            .memcpy_dtoh(&self.d_velocities, &mut velocities)?;

        // Download energies
        let mut energies = vec![0.0f32; self.n_clones * 4];
        self.stream.memcpy_dtoh(&self.d_energies, &mut energies)?;

        let mut results = Vec::with_capacity(self.n_clones);
        let stride = self.n_atoms * 3;

        for i in 0..self.n_clones {
            let pos_start = i * stride;
            let pos_end = pos_start + stride;

            results.push(EnsembleResult {
                clone_id: i,
                positions: positions[pos_start..pos_end].to_vec(),
                velocities: velocities[pos_start..pos_end].to_vec(),
                potential_energy: energies[i * 4] as f64,
                kinetic_energy: energies[i * 4 + 1] as f64,
                temperature: energies[i * 4 + 2] as f64,
            });
        }

        Ok(results)
    }

    /// Get number of clones
    pub fn n_clones(&self) -> usize {
        self.n_clones
    }

    /// Get number of atoms
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }
}

/// Convert prism-prep topology to EnsembleTopology
pub fn topology_from_prism_prep(
    n_atoms: usize,
    masses: Vec<f32>,
    charges: Vec<f32>,
    sigmas: Vec<f32>,
    epsilons: Vec<f32>,
    bonds: Vec<(usize, usize, f32, f32)>,
    angles: Vec<(usize, usize, usize, f32, f32)>,
) -> EnsembleTopology {
    EnsembleTopology {
        n_atoms,
        masses,
        charges,
        sigmas,
        epsilons,
        bonds,
        angles,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_creation() {
        // Simple 3-atom water-like topology
        let topo = EnsembleTopology {
            n_atoms: 3,
            masses: vec![16.0, 1.0, 1.0],
            charges: vec![-0.82, 0.41, 0.41],
            sigmas: vec![3.15, 0.4, 0.4],
            epsilons: vec![0.152, 0.046, 0.046],
            bonds: vec![(0, 1, 450.0, 0.96), (0, 2, 450.0, 0.96)],
            angles: vec![(1, 0, 2, 55.0, 1.824)],
        };

        let context = CudaContext::new(0).expect("CUDA context");
        let engine = EnsembleWarpMd::new(context, &topo, 32);
        assert!(engine.is_ok());
    }
}
