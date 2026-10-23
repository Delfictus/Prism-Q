//! GPU Non-bonded Force Calculator
//!
//! Provides GPU-accelerated LJ + Coulomb non-bonded forces with cutoff.
//! Uses tiled algorithm with shared memory for O(n²/tile_size) efficiency.
//!
//! This module supplements amber_forces.rs (which handles bonded forces)
//! to provide complete GPU-accelerated AMBER force calculation.
//!
//! # Usage
//!
//! ```rust,ignore
//! let nb_gpu = AmberNonbondedGpu::new(context, n_atoms)?;
//! nb_gpu.upload_parameters(&sigmas, &epsilons, &charges, &exclusions)?;
//! nb_gpu.compute(&positions, &mut forces)?;
//! ```

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use std::collections::HashSet;

/// Maximum exclusions per atom (1-2, 1-3 bonded pairs)
const MAX_EXCLUSIONS: usize = 32;

/// GPU non-bonded force calculator
pub struct AmberNonbondedGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    // Kernel
    nonbonded_kernel: CudaFunction,

    // Device buffers
    d_lj_sigma: CudaSlice<f32>,
    d_lj_epsilon: CudaSlice<f32>,
    d_charges: CudaSlice<f32>,
    d_exclusion_list: CudaSlice<i32>,
    d_n_exclusions: CudaSlice<i32>,

    // Computation buffers
    d_positions: CudaSlice<f32>,
    d_forces: CudaSlice<f32>,
    d_energy: CudaSlice<f32>,

    n_atoms: usize,
    params_uploaded: bool,
}

impl AmberNonbondedGpu {
    /// Create a new GPU non-bonded force calculator
    pub fn new(context: Arc<CudaContext>, n_atoms: usize) -> Result<Self> {
        log::info!("🔌 Initializing GPU Non-bonded Forces for {} atoms", n_atoms);

        let stream = context.default_stream();

        // Load PTX module (same file as bonded forces)
        let ptx_path = "crates/prism-gpu/target/ptx/amber_bonded.ptx";
        let ptx = Ptx::from_file(ptx_path);
        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load AMBER PTX from {}", ptx_path))?;

        // Load non-bonded kernel
        let nonbonded_kernel = module
            .load_function("compute_nonbonded_forces")
            .context("Failed to load compute_nonbonded_forces kernel")?;

        // Allocate device buffers
        let d_lj_sigma = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_lj_epsilon = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_charges = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_exclusion_list = stream.alloc_zeros::<i32>(n_atoms * MAX_EXCLUSIONS)?;
        let d_n_exclusions = stream.alloc_zeros::<i32>(n_atoms)?;

        let d_positions = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let d_forces = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let d_energy = stream.alloc_zeros::<f32>(1)?;

        log::info!("✅ GPU Non-bonded initialized: {} atoms, {:.2} MB",
                   n_atoms,
                   (n_atoms * (4 + 4 + 4 + MAX_EXCLUSIONS * 4 + 4 + 12 + 12) + 4) as f64 / 1e6);

        Ok(Self {
            context,
            stream,
            module,
            nonbonded_kernel,
            d_lj_sigma,
            d_lj_epsilon,
            d_charges,
            d_exclusion_list,
            d_n_exclusions,
            d_positions,
            d_forces,
            d_energy,
            n_atoms,
            params_uploaded: false,
        })
    }

    /// Upload LJ parameters, charges, and exclusion lists
    ///
    /// # Arguments
    /// * `sigmas` - LJ sigma per atom (Å)
    /// * `epsilons` - LJ epsilon per atom (kcal/mol)
    /// * `charges` - Partial charges per atom (elementary charges)
    /// * `exclusions` - Per-atom exclusion sets (bonded atoms to exclude)
    pub fn upload_parameters(
        &mut self,
        sigmas: &[f32],
        epsilons: &[f32],
        charges: &[f32],
        exclusions: &[HashSet<usize>],
    ) -> Result<()> {
        if sigmas.len() != self.n_atoms {
            return Err(anyhow::anyhow!("sigma count {} != n_atoms {}", sigmas.len(), self.n_atoms));
        }

        // Upload LJ parameters
        self.stream.memcpy_htod(&sigmas, &mut self.d_lj_sigma)?;
        self.stream.memcpy_htod(&epsilons, &mut self.d_lj_epsilon)?;
        self.stream.memcpy_htod(&charges, &mut self.d_charges)?;

        // Flatten exclusion lists
        let mut excl_flat = vec![-1i32; self.n_atoms * MAX_EXCLUSIONS];
        let mut n_excl = vec![0i32; self.n_atoms];

        for (i, excl_set) in exclusions.iter().enumerate() {
            n_excl[i] = excl_set.len().min(MAX_EXCLUSIONS) as i32;
            for (j, &excl_idx) in excl_set.iter().take(MAX_EXCLUSIONS).enumerate() {
                excl_flat[i * MAX_EXCLUSIONS + j] = excl_idx as i32;
            }
        }

        self.stream.memcpy_htod(&excl_flat, &mut self.d_exclusion_list)?;
        self.stream.memcpy_htod(&n_excl, &mut self.d_n_exclusions)?;

        self.params_uploaded = true;
        log::debug!("Non-bonded parameters uploaded: {} atoms", self.n_atoms);

        Ok(())
    }

    /// Compute non-bonded forces on GPU
    ///
    /// # Arguments
    /// * `positions` - Atom positions [x0, y0, z0, x1, y1, z1, ...]
    /// * `forces` - Force output (accumulated, not zeroed)
    ///
    /// # Returns
    /// Total non-bonded energy (kcal/mol)
    pub fn compute(&mut self, positions: &[f32], forces: &mut [f32]) -> Result<f64> {
        if !self.params_uploaded {
            return Err(anyhow::anyhow!("Parameters not uploaded - call upload_parameters first"));
        }

        if positions.len() != self.n_atoms * 3 {
            return Err(anyhow::anyhow!("Position size {} != expected {}",
                                       positions.len(), self.n_atoms * 3));
        }

        // Upload positions
        self.stream.memcpy_htod(&positions, &mut self.d_positions)?;

        // Zero energy
        let zero_energy = vec![0.0f32; 1];
        self.stream.memcpy_htod(&zero_energy, &mut self.d_energy)?;

        // Download current forces to accumulate
        let mut h_forces = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_forces, &mut h_forces)?;

        // Zero GPU forces before kernel
        let zero_forces = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_htod(&zero_forces, &mut self.d_forces)?;

        // Launch kernel
        let threads_per_block = 128;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.nonbonded_kernel.launch(
                cfg,
                (
                    self.d_positions.device_ptr(),
                    self.d_forces.device_ptr_mut(),
                    self.d_energy.device_ptr_mut(),
                    self.d_lj_sigma.device_ptr(),
                    self.d_lj_epsilon.device_ptr(),
                    self.d_charges.device_ptr(),
                    self.d_exclusion_list.device_ptr(),
                    self.d_n_exclusions.device_ptr(),
                    MAX_EXCLUSIONS as i32,
                    self.n_atoms as i32,
                ),
            )?;
        }

        // Synchronize and download results
        self.stream.synchronize()?;

        // Download forces and add to output
        let mut gpu_forces = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_forces, &mut gpu_forces)?;

        for i in 0..forces.len() {
            forces[i] += gpu_forces[i];
        }

        // Download energy
        let mut h_energy = vec![0.0f32; 1];
        self.stream.memcpy_dtoh(&self.d_energy, &mut h_energy)?;

        Ok(h_energy[0] as f64)
    }

    /// Get the number of atoms
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }
}

/// Build exclusion lists from bond topology
///
/// Excludes 1-2 (bonded) and 1-3 (angle) pairs from non-bonded calculation.
/// 1-4 pairs are handled separately with scaled interactions.
pub fn build_exclusion_lists(
    bonds: &[(usize, usize)],
    angles: &[(usize, usize, usize)],
    n_atoms: usize,
) -> Vec<HashSet<usize>> {
    let mut exclusions = vec![HashSet::new(); n_atoms];

    // 1-2 exclusions (bonded pairs)
    for &(i, j) in bonds {
        if i < n_atoms && j < n_atoms {
            exclusions[i].insert(j);
            exclusions[j].insert(i);
        }
    }

    // 1-3 exclusions (angle endpoints)
    for &(i, _j, k) in angles {
        if i < n_atoms && k < n_atoms {
            exclusions[i].insert(k);
            exclusions[k].insert(i);
        }
    }

    exclusions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_exclusion_lists() {
        // Simple 3-atom system: 0-1-2 (linear)
        let bonds = vec![(0, 1), (1, 2)];
        let angles = vec![(0, 1, 2)];

        let exclusions = build_exclusion_lists(&bonds, &angles, 3);

        // Atom 0 excludes 1 (bonded) and 2 (1-3)
        assert!(exclusions[0].contains(&1));
        assert!(exclusions[0].contains(&2));

        // Atom 1 excludes 0 and 2 (both bonded)
        assert!(exclusions[1].contains(&0));
        assert!(exclusions[1].contains(&2));

        // Atom 2 excludes 1 (bonded) and 0 (1-3)
        assert!(exclusions[2].contains(&1));
        assert!(exclusions[2].contains(&0));
    }
}
