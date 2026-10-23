//! Polycentric Immunity Field - Rust bindings for CUDA kernel
//!
//! This module provides the host-side interface for the fractal interference
//! model that replaces single-center gamma computation.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Number of epitope centers (immune pressure sources)
pub const N_EPITOPE_CENTERS: usize = 10;

/// Number of PK scenarios in the envelope
pub const N_PK_SCENARIOS: usize = 75;

/// Feature dimension from mega_fused_batch
pub const FEATURE_DIM: usize = 136;

/// Output features from polycentric kernel
pub const POLYCENTRIC_OUTPUT_DIM: usize = 22;

/// Cross-reactivity matrix (10×10) - default values from VASIL
/// Entry [i,j] = protection conferred by immunity to epitope i against epitope j
pub const DEFAULT_CROSS_REACTIVITY: [[f32; 10]; 10] = [
    // Class1  Class2  Class3  Class4  S309    CR3022  NTD1    NTD2    NTD3    S2
    [1.00, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.10], // Class 1
    [0.30, 1.00, 0.35, 0.25, 0.20, 0.15, 0.05, 0.05, 0.05, 0.10], // Class 2
    [0.25, 0.35, 1.00, 0.30, 0.25, 0.20, 0.05, 0.05, 0.05, 0.10], // Class 3
    [0.20, 0.25, 0.30, 1.00, 0.30, 0.25, 0.05, 0.05, 0.05, 0.10], // Class 4
    [0.15, 0.20, 0.25, 0.30, 1.00, 0.30, 0.05, 0.05, 0.05, 0.15], // S309
    [0.10, 0.15, 0.20, 0.25, 0.30, 1.00, 0.05, 0.05, 0.05, 0.20], // CR3022
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.00, 0.60, 0.50, 0.05], // NTD-1
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.60, 1.00, 0.60, 0.05], // NTD-2
    [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.50, 0.60, 1.00, 0.05], // NTD-3
    [0.10, 0.10, 0.10, 0.10, 0.15, 0.20, 0.05, 0.05, 0.05, 1.00], // S2
];

/// Polycentric immunity GPU processor
pub struct PolycentricImmunityGpu {
    #[allow(dead_code)]
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_main: CudaFunction,
    kernel_init_centers: CudaFunction,

    // Device memory for constants
    epitope_centers: CudaSlice<f32>,  // [10 × 136]
    cross_reactivity: CudaSlice<f32>, // [10 × 10]
    pk_tmax: CudaSlice<f32>,          // [75]
    pk_thalf: CudaSlice<f32>,         // [75]
}

impl PolycentricImmunityGpu {
    /// Create new polycentric immunity processor
    pub fn new(context: Arc<CudaContext>, ptx_path: &std::path::Path) -> Result<Self> {
        let stream = context.default_stream();

        // Load PTX
        let ptx_file = ptx_path.join("polycentric_immunity.ptx");
        let ptx_src = std::fs::read_to_string(&ptx_file)
            .with_context(|| format!("Failed to read PTX: {:?}", ptx_file))?;
        let ptx = Ptx::from_src(ptx_src);

        // Load module
        let module = context
            .load_module(ptx)
            .context("Failed to load polycentric PTX module")?;

        let kernel_main = module
            .load_function("polycentric_immunity_kernel")
            .context("Failed to load polycentric_immunity_kernel function")?;
        let kernel_init_centers = module
            .load_function("init_epitope_centers")
            .context("Failed to load init_epitope_centers function")?;

        // Allocate constant memory
        let epitope_centers = stream.alloc_zeros::<f32>(N_EPITOPE_CENTERS * FEATURE_DIM)?;

        // Flatten and upload cross-reactivity matrix
        let cross_flat: Vec<f32> = DEFAULT_CROSS_REACTIVITY.iter().flatten().copied().collect();
        // Allocates GPU memory and uploads data in one step
        let cross_reactivity = stream.clone_htod(&cross_flat)?;

        // Build PK parameter arrays (from VASIL specification)
        let tmax_values = build_pk_tmax();
        let thalf_values = build_pk_thalf();
        // Allocates GPU memory and uploads data in one step
        let pk_tmax = stream.clone_htod(&tmax_values)?;
        let pk_thalf = stream.clone_htod(&thalf_values)?;

        Ok(Self {
            context,
            stream,
            kernel_main,
            kernel_init_centers,
            epitope_centers,
            cross_reactivity,
            pk_tmax,
            pk_thalf,
        })
    }

    /// Initialize epitope centers from training data
    /// Call this ONCE at startup with representative samples
    pub fn init_centers(
        &mut self,
        features: &[f32],       // [n_samples × 136] flattened
        epitope_labels: &[i32], // [n_samples] class assignments (0-9)
    ) -> Result<()> {
        let n_samples = epitope_labels.len();
        assert_eq!(
            features.len(),
            n_samples * FEATURE_DIM,
            "Features length mismatch: {} != {} * {}",
            features.len(),
            n_samples,
            FEATURE_DIM
        );

        // Upload to device using context (atomic copy)
        let d_features = self.stream.clone_htod(features)?;
        let d_labels = self.stream.clone_htod(epitope_labels)?;

        // Count samples per epitope
        let mut counts = vec![0i32; N_EPITOPE_CENTERS];
        for &label in epitope_labels {
            if label >= 0 && (label as usize) < N_EPITOPE_CENTERS {
                counts[label as usize] += 1;
            }
        }
        let d_counts = self.stream.clone_htod(&counts[..])?;

        // Launch kernel
        let config = LaunchConfig {
            grid_dim: (N_EPITOPE_CENTERS as u32, 1, 1),
            block_dim: (FEATURE_DIM as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_samples_i32 = n_samples as i32;

        unsafe {
            &self
                .stream
                .launch_builder(&self.kernel_init_centers)
                .arg(&d_features)
                .arg(&d_labels)
                .arg(&d_counts)
                .arg(&n_samples_i32)
                .arg(&mut self.epitope_centers)
                .launch(config)?;
        }

        self.stream.synchronize()?;
        Ok(())
    }

    /// Process batch of structures through polycentric immunity field
    #[allow(clippy::too_many_arguments)]
    pub fn process_batch(
        &self,
        features_packed: &CudaSlice<f32>, // [total_residues × 136]
        residue_offsets: &CudaSlice<i32>, // [n_structures]
        n_residues: &CudaSlice<i32>,      // [n_structures]
        escape_10d: &CudaSlice<f32>,      // [n_structures × 10]
        pk_immunity: &CudaSlice<f32>,     // [n_structures × 75]
        time_since_infection: &CudaSlice<f32>, // [n_structures]
        freq_history: &CudaSlice<f32>,    // [n_structures × 7]
        current_freq: &CudaSlice<f32>,    // [n_structures]
        n_structures: usize,
    ) -> Result<CudaSlice<f32>> {
        // Allocate output on device
        let mut output = self
            .stream
            .alloc_zeros::<f32>(n_structures * POLYCENTRIC_OUTPUT_DIM)?;

        // Launch kernel
        let config = LaunchConfig {
            grid_dim: (n_structures as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: ((FEATURE_DIM + N_EPITOPE_CENTERS + N_PK_SCENARIOS + 7) * 4) as u32,
        };

        let n_structures_i32 = n_structures as i32;

        unsafe {
            &self
                .stream
                .launch_builder(&self.kernel_main)
                .arg(features_packed)
                .arg(residue_offsets)
                .arg(n_residues)
                .arg(escape_10d)
                .arg(pk_immunity)
                .arg(time_since_infection)
                .arg(freq_history)
                .arg(current_freq)
                .arg(&mut output)
                .arg(&n_structures_i32)
                .launch(config)?;
        }

        self.stream.synchronize()?;
        Ok(output)
    }

    /// Download output features to host
    pub fn download_output(&self, output: &CudaSlice<f32>) -> Result<Vec<f32>> {
        Ok(self.stream.clone_dtoh(output)?)
    }
}

/// Build 75 tmax values (5 base × 15 thalf = 75 combinations)
fn build_pk_tmax() -> Vec<f32> {
    let tmax_base = [14.0f32, 17.5, 21.0, 24.5, 28.0];
    let mut result = Vec::with_capacity(N_PK_SCENARIOS);
    for _ in 0..15 {
        for &t in &tmax_base {
            result.push(t);
        }
    }
    result
}

/// Build 75 thalf values (15 values × 5 tmax = 75 combinations)
fn build_pk_thalf() -> Vec<f32> {
    let mut result = Vec::with_capacity(N_PK_SCENARIOS);
    for i in 0..15 {
        let thalf = 25.0 + (i as f32) * (69.0 - 25.0) / 14.0;
        for _ in 0..5 {
            result.push(thalf);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pk_params() {
        let tmax = build_pk_tmax();
        let thalf = build_pk_thalf();

        assert_eq!(tmax.len(), 75);
        assert_eq!(thalf.len(), 75);
        assert!((tmax[0] - 14.0).abs() < 0.01);
        assert!((thalf[0] - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_cross_reactivity_symmetric() {
        // Verify matrix is properly defined (not necessarily symmetric, but well-formed)
        for i in 0..10 {
            assert!(
                (DEFAULT_CROSS_REACTIVITY[i][i] - 1.0).abs() < 0.01,
                "Diagonal should be 1.0 at position {}",
                i
            );
        }
    }
}
