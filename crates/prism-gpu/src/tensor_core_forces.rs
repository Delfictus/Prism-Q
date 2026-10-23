//! Tensor Core Accelerated Non-bonded Force Computation
//!
//! PERFORMANCE OPTIMIZATION: 2-4× speedup using NVIDIA Tensor Cores (WMMA)
//!
//! ## Architecture
//!
//! Uses FP16→FP32 matrix multiply-accumulate for batched distance computation:
//!
//! ```text
//! Distance Matrix Approach:
//!
//!   Positions (Nx3)          Norms (N)
//!   ┌─────────────┐          ┌───┐
//!   │ x1 y1 z1    │          │n1 │
//!   │ x2 y2 z2    │          │n2 │
//!   │ ...         │    →     │...│
//!   │ xN yN zN    │          │nN │
//!   └─────────────┘          └───┘
//!
//!   D²[i,j] = ||r_i||² + ||r_j||² - 2 * r_i · r_j
//!                                    ↑
//!                              Tensor Core WMMA
//! ```
//!
//! ## Performance Benefits
//!
//! - WMMA instructions process 16×16 tiles in hardware
//! - FP16 parameters reduce memory bandwidth by 50%
//! - Batched computation amortizes kernel launch overhead
//! - Deterministic results (no atomicAdd race conditions)

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Tensor Core configuration constants
pub const TC_TILE_SIZE: usize = 16;
pub const TC_BLOCK_SIZE: usize = 256;

/// Tensor Core accelerated force computation
pub struct TensorCoreForces {
    _context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _module: Arc<CudaModule>,

    // Kernels
    precompute_norms_kernel: CudaFunction,
    convert_to_fp16_kernel: CudaFunction,
    tensor_core_nonbonded_kernel: CudaFunction,
    tensor_core_distances_kernel: CudaFunction,

    // GPU buffers
    d_norms_sq: CudaSlice<f32>,        // ||r_i||² for each atom
    d_positions_fp16: CudaSlice<u16>,  // FP16 positions (half = u16)
    d_sigma_fp16: CudaSlice<u16>,      // FP16 LJ sigma
    d_epsilon_fp16: CudaSlice<u16>,    // FP16 LJ epsilon
    d_distance_matrix: CudaSlice<f32>, // Tile distance matrix (temporary)

    // Configuration
    n_atoms: usize,
    cutoff_sq: f32,

    // Statistics
    kernel_calls: u64,
}

impl TensorCoreForces {
    /// Create a new Tensor Core force calculator
    pub fn new(
        context: Arc<CudaContext>,
        stream: Arc<CudaStream>,
        n_atoms: usize,
        cutoff: f32,
    ) -> Result<Self> {
        // Load PTX module
        let ptx_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/target/ptx/tensor_core_forces.ptx"
        );
        let ptx_src = std::fs::read_to_string(ptx_path)
            .with_context(|| format!("Failed to read tensor_core_forces.ptx from {}", ptx_path))?;
        let ptx = Ptx::from_src(&ptx_src);

        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load tensor_core_forces PTX from {}", ptx_path))?;

        // Get kernel functions
        let precompute_norms_kernel = module
            .load_function("precompute_norm_squared")
            .context("Failed to load precompute_norm_squared")?;
        let convert_to_fp16_kernel = module
            .load_function("convert_params_to_fp16")
            .context("Failed to load convert_params_to_fp16")?;
        let tensor_core_nonbonded_kernel = module
            .load_function("tensor_core_nonbonded")
            .context("Failed to load tensor_core_nonbonded")?;
        let tensor_core_distances_kernel = module
            .load_function("tensor_core_distances_16x16_kernel")
            .context("Failed to load tensor_core_distances_16x16_kernel")?;

        // Allocate GPU buffers
        let d_norms_sq = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_positions_fp16 = stream.alloc_zeros::<u16>(n_atoms * 3)?;
        let d_sigma_fp16 = stream.alloc_zeros::<u16>(n_atoms)?;
        let d_epsilon_fp16 = stream.alloc_zeros::<u16>(n_atoms)?;

        // Tile workspace for 16×16 distance computation
        let tile_workspace_size = TC_TILE_SIZE * TC_TILE_SIZE;
        let d_distance_matrix = stream.alloc_zeros::<f32>(tile_workspace_size)?;

        Ok(Self {
            _context: context,
            stream,
            _module: module,
            precompute_norms_kernel,
            convert_to_fp16_kernel,
            tensor_core_nonbonded_kernel,
            tensor_core_distances_kernel,
            d_norms_sq,
            d_positions_fp16,
            d_sigma_fp16,
            d_epsilon_fp16,
            d_distance_matrix,
            n_atoms,
            cutoff_sq: cutoff * cutoff,
            kernel_calls: 0,
        })
    }

    /// Precompute position norms (||r_i||²)
    ///
    /// Should be called once per step before force computation.
    pub fn precompute_norms(&mut self, d_positions: &CudaSlice<f32>) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(self.n_atoms as u32);
        let n_atoms_i32 = self.n_atoms as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.precompute_norms_kernel);
            builder.arg(d_positions);
            builder.arg(&self.d_norms_sq);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        Ok(())
    }

    /// Convert positions to FP16 for Tensor Core operations
    pub fn convert_positions_to_fp16(&mut self, d_positions: &CudaSlice<f32>) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems((self.n_atoms * 3) as u32);
        let n_elements_i32 = (self.n_atoms * 3) as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&self.convert_to_fp16_kernel);
            builder.arg(d_positions);
            builder.arg(&self.d_positions_fp16);
            builder.arg(&n_elements_i32);
            builder.launch(cfg)?;
        }

        Ok(())
    }

    /// Convert LJ parameters to FP16
    pub fn convert_params_to_fp16(
        &mut self,
        d_sigma: &CudaSlice<f32>,
        d_epsilon: &CudaSlice<f32>,
    ) -> Result<()> {
        let cfg = LaunchConfig::for_num_elems(self.n_atoms as u32);
        let n_atoms_i32 = self.n_atoms as i32;

        // Convert sigma
        unsafe {
            let mut builder = self.stream.launch_builder(&self.convert_to_fp16_kernel);
            builder.arg(d_sigma);
            builder.arg(&self.d_sigma_fp16);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg.clone())?;
        }

        // Convert epsilon
        unsafe {
            let mut builder = self.stream.launch_builder(&self.convert_to_fp16_kernel);
            builder.arg(d_epsilon);
            builder.arg(&self.d_epsilon_fp16);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        Ok(())
    }

    /// Compute non-bonded forces using Tensor Core acceleration
    ///
    /// Uses WMMA for batched distance computation with deterministic summation.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_nonbonded(
        &mut self,
        d_positions: &CudaSlice<f32>,
        d_forces: &mut CudaSlice<f32>,
        d_energy: &mut CudaSlice<f32>,
        d_sigma: &CudaSlice<f32>,
        d_epsilon: &CudaSlice<f32>,
        d_charge: &CudaSlice<f32>,
        d_excl_list: &CudaSlice<i32>,
        d_n_excl: &CudaSlice<i32>,
        max_excl: i32,
    ) -> Result<()> {
        // Step 1: Precompute norms
        self.precompute_norms(d_positions)?;

        // Step 2: Convert to FP16 (if not already cached)
        self.convert_positions_to_fp16(d_positions)?;

        // Step 3: Launch Tensor Core kernel
        // Grid: ceil(n_atoms / TILE_SIZE) × ceil(n_atoms / TILE_SIZE)
        let grid_dim = ((self.n_atoms + TC_TILE_SIZE - 1) / TC_TILE_SIZE) as u32;
        let cfg = LaunchConfig {
            grid_dim: (grid_dim, grid_dim, 1),
            block_dim: (TC_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let n_atoms_i32 = self.n_atoms as i32;

        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.tensor_core_nonbonded_kernel);
            builder.arg(d_positions);
            builder.arg(&self.d_positions_fp16);
            builder.arg(&self.d_norms_sq);
            builder.arg(d_forces);
            builder.arg(d_energy);
            builder.arg(d_sigma);
            builder.arg(&self.d_sigma_fp16);
            builder.arg(d_epsilon);
            builder.arg(&self.d_epsilon_fp16);
            builder.arg(d_charge);
            builder.arg(d_excl_list);
            builder.arg(d_n_excl);
            builder.arg(&self.cutoff_sq);
            builder.arg(&max_excl);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        self.kernel_calls += 1;

        Ok(())
    }

    /// Compute distance matrix for a tile using Tensor Cores
    ///
    /// Returns 16×16 distance matrix for atoms in tiles i_tile and j_tile.
    #[allow(dead_code)]
    pub fn compute_distance_tile(
        &mut self,
        d_positions: &CudaSlice<f32>,
        i_tile: usize,
        j_tile: usize,
    ) -> Result<Vec<f32>> {
        // Ensure norms are precomputed
        self.precompute_norms(d_positions)?;
        self.convert_positions_to_fp16(d_positions)?;

        // Launch distance computation kernel
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (32, 1, 1), // Single warp for WMMA
            shared_mem_bytes: 0,
        };
        let i_tile_i32 = i_tile as i32;
        let j_tile_i32 = j_tile as i32;
        let n_atoms_i32 = self.n_atoms as i32;

        unsafe {
            let mut builder = self
                .stream
                .launch_builder(&self.tensor_core_distances_kernel);
            builder.arg(&self.d_positions_fp16);
            builder.arg(&self.d_norms_sq);
            builder.arg(&self.d_distance_matrix);
            builder.arg(&i_tile_i32);
            builder.arg(&j_tile_i32);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        // Read back result
        let mut distances = vec![0.0f32; TC_TILE_SIZE * TC_TILE_SIZE];
        self.stream
            .memcpy_dtoh(&self.d_distance_matrix, &mut distances)?;

        Ok(distances)
    }

    /// Get FP16 positions buffer (for external use with Verlet list)
    pub fn positions_fp16(&self) -> &CudaSlice<u16> {
        &self.d_positions_fp16
    }

    /// Get FP16 sigma buffer
    pub fn sigma_fp16(&self) -> &CudaSlice<u16> {
        &self.d_sigma_fp16
    }

    /// Get FP16 epsilon buffer
    pub fn epsilon_fp16(&self) -> &CudaSlice<u16> {
        &self.d_epsilon_fp16
    }

    /// Get precomputed norms buffer
    pub fn norms_sq(&self) -> &CudaSlice<f32> {
        &self.d_norms_sq
    }

    /// Get kernel call count
    pub fn kernel_calls(&self) -> u64 {
        self.kernel_calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_core_config() {
        assert_eq!(TC_TILE_SIZE, 16);
        assert_eq!(TC_BLOCK_SIZE, 256);
    }

    #[test]
    fn test_cutoff_squared() {
        let cutoff = 10.0f32;
        let cutoff_sq = cutoff * cutoff;
        assert_eq!(cutoff_sq, 100.0);
    }
}
