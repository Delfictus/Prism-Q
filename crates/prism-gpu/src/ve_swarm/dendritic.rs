//! Dendritic Residue Graph Reservoir
//!
//! GPU-accelerated multi-branch neuromorphic computation on the protein
//! contact graph. Preserves full 136-dim feature tensor across residues.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Configuration for dendritic reservoir
#[derive(Clone, Debug)]
pub struct DendriticConfig {
    /// Number of propagation iterations
    pub iterations: usize,
    /// Number of dendritic branches (local, neighbor, global, recurrent)
    pub n_branches: usize,
    /// Reservoir dimension (neurons per residue)
    pub reservoir_dim: usize,
    /// Leak rate for temporal dynamics
    pub leak_rate: f32,
}

impl Default for DendriticConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            n_branches: 4,
            reservoir_dim: 32,
            leak_rate: 0.1,
        }
    }
}

/// Dendritic reservoir compute state
pub struct DendriticReservoir {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    config: DendriticConfig,

    // Kernel functions
    fn_init: CudaFunction,
    fn_propagate: CudaFunction,
    fn_attention: CudaFunction,
}

impl DendriticReservoir {
    /// Create new dendritic reservoir
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &str, config: DendriticConfig) -> Result<Self> {
        let stream = ctx.default_stream();
        let ptx = Ptx::from_file(ptx_path);

        let module = ctx.load_module(ptx)?;

        let fn_init = module.load_function("ve_swarm_init_reservoir").unwrap();
        let fn_propagate = module
            .load_function("ve_swarm_dendritic_reservoir")
            .unwrap();
        let fn_attention = module.load_function("ve_swarm_compute_attention").unwrap();

        Ok(Self {
            ctx,
            stream,
            config,
            fn_init,
            fn_propagate,
            fn_attention,
        })
    }

    /// Compute reservoir state from features and contact graph
    pub fn compute(
        &self,
        features: &CudaSlice<f32>,
        csr_row: &CudaSlice<i32>,
        csr_col: &CudaSlice<i32>,
        csr_weight: &CudaSlice<f32>,
        eigenvector: &CudaSlice<f32>,
        n_residues: usize,
    ) -> Result<CudaSlice<f32>> {
        // Allocate reservoir buffers
        let state_size = n_residues * self.config.reservoir_dim;
        let mut state_a: CudaSlice<f32> = self.stream.alloc_zeros(state_size)?;
        let mut state_b: CudaSlice<f32> = self.stream.alloc_zeros(state_size)?;

        // Initialize reservoir
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let init_cfg = LaunchConfig {
            grid_dim: ((state_size as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_init)
                .arg(&mut state_a)
                .arg(&(n_residues as i32))
                .arg(&seed)
                .launch(init_cfg)?;
        }

        // Propagate
        for iter in 0..self.config.iterations {
            let cfg = LaunchConfig {
                grid_dim: (n_residues as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };

            if iter % 2 == 0 {
                unsafe {
                    self.stream
                        .launch_builder(&self.fn_propagate)
                        .arg(features)
                        .arg(csr_row)
                        .arg(csr_col)
                        .arg(csr_weight)
                        .arg(eigenvector)
                        .arg(&mut state_b)
                        .arg(&state_a)
                        .arg(&(n_residues as i32))
                        .arg(&(iter as i32))
                        .launch(cfg)?;
                }
            } else {
                unsafe {
                    self.stream
                        .launch_builder(&self.fn_propagate)
                        .arg(features)
                        .arg(csr_row)
                        .arg(csr_col)
                        .arg(csr_weight)
                        .arg(eigenvector)
                        .arg(&mut state_a)
                        .arg(&state_b)
                        .arg(&(n_residues as i32))
                        .arg(&(iter as i32))
                        .launch(cfg)?;
                }
            }
        }

        self.stream.synchronize()?;

        // Return final state
        if self.config.iterations % 2 == 0 {
            Ok(state_a)
        } else {
            Ok(state_b)
        }
    }

    /// Compute attention weights over residues
    pub fn compute_attention(
        &self,
        reservoir_state: &CudaSlice<f32>,
        eigenvector: &CudaSlice<f32>,
        csr_row: &CudaSlice<i32>,
        n_residues: usize,
        temperature: f32,
    ) -> Result<CudaSlice<f32>> {
        let mut attention: CudaSlice<f32> = self.stream.alloc_zeros(n_residues)?;
        let mut attended_features: CudaSlice<f32> = self.stream.alloc_zeros(136)?;

        // Create dummy features for this call (actual features would be passed)
        let dummy_features: CudaSlice<f32> = self.stream.alloc_zeros(n_residues * 136)?;

        let cfg = LaunchConfig {
            grid_dim: ((n_residues as u32 + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n_residues * 4) as u32,
        };

        unsafe {
            self.stream
                .launch_builder(&self.fn_attention)
                .arg(reservoir_state)
                .arg(eigenvector)
                .arg(csr_row)
                .arg(&mut attention)
                .arg(&mut attended_features)
                .arg(&dummy_features)
                .arg(&(n_residues as i32))
                .arg(&temperature)
                .launch(cfg)?;
        }

        self.stream.synchronize()?;

        Ok(attention)
    }
}

/// Build contact graph CSR from distance matrix
pub fn build_contact_graph(ca_coords: &[[f32; 3]], cutoff: f32) -> (Vec<i32>, Vec<i32>, Vec<f32>) {
    let n = ca_coords.len();
    let mut row_ptr = vec![0i32; n + 1];
    let mut col_idx = Vec::new();
    let mut weights = Vec::new();

    for i in 0..n {
        row_ptr[i] = col_idx.len() as i32;

        for j in 0..n {
            if i != j {
                let dx = ca_coords[i][0] - ca_coords[j][0];
                let dy = ca_coords[i][1] - ca_coords[j][1];
                let dz = ca_coords[i][2] - ca_coords[j][2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < cutoff {
                    col_idx.push(j as i32);
                    // Gaussian weight: closer = stronger
                    let weight = (-dist * dist / (2.0 * 4.0 * 4.0)).exp();
                    weights.push(weight);
                }
            }
        }
    }
    row_ptr[n] = col_idx.len() as i32;

    (row_ptr, col_idx, weights)
}

/// Compute eigenvector centrality using power iteration
pub fn compute_eigenvector_centrality(
    row_ptr: &[i32],
    col_idx: &[i32],
    weights: &[f32],
    iterations: usize,
) -> Vec<f32> {
    let n = row_ptr.len() - 1;
    let mut eigenvector = vec![1.0 / n as f32; n];
    let mut new_eigenvector = vec![0.0f32; n];

    for _ in 0..iterations {
        // Matrix-vector multiply
        for i in 0..n {
            let start = row_ptr[i] as usize;
            let end = row_ptr[i + 1] as usize;

            let mut sum = 0.0f32;
            for e in start..end {
                let j = col_idx[e] as usize;
                let w = weights[e];
                sum += w * eigenvector[j];
            }
            new_eigenvector[i] = sum;
        }

        // Normalize
        let norm: f32 = new_eigenvector.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in new_eigenvector.iter_mut() {
            *x /= norm.max(1e-6);
        }

        std::mem::swap(&mut eigenvector, &mut new_eigenvector);
    }

    eigenvector
}
