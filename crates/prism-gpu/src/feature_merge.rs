//! Feature Merge GPU Module
//!
//! ARCHITECT DIRECTIVE: PHASE 1.5 - FEATURE MERGE KERNEL
//!
//! Rust bindings for the Feature Merge "Zipper" kernel that combines
//! 136-dimensional main features with 4-dimensional cryptic features
//! into 140-dimensional output for DQN consumption.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Configuration for Feature Merge GPU operations
#[derive(Debug, Clone)]
pub struct FeatureMergeConfig {
    /// Block size for CUDA kernel (default: 128 for high occupancy)
    pub block_size: u32,
    /// Maximum number of residues per batch
    pub max_residues: usize,
}

impl Default for FeatureMergeConfig {
    fn default() -> Self {
        Self {
            block_size: 128,         // __launch_bounds__(128) as specified
            max_residues: 1_048_576, // 1M residues max
        }
    }
}

/// GPU-accelerated Feature Merge operations
pub struct FeatureMergeGpu {
    device: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    feature_merge_kernel: CudaFunction,
    config: FeatureMergeConfig,
}

impl FeatureMergeGpu {
    /// Create new FeatureMergeGpu instance
    ///
    /// # Arguments
    /// * `device` - CUDA device context
    /// * `config` - Feature merge configuration
    pub fn new(device: Arc<CudaContext>, config: FeatureMergeConfig) -> Result<Self> {
        // Embed the governed build output, not a separately generated source copy.
        #[cfg(feature = "cuda")]
        let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/ptx/feature_merge.ptx"));
        #[cfg(not(feature = "cuda"))]
        let ptx_src: &str = anyhow::bail!("FeatureMergeGpu requires prism-gpu's cuda feature");
        let ptx = Ptx::from_src(ptx_src);

        // cudarc 0.18.1: load_ptx registers the module and functions
        let module = device
            .load_module(ptx)
            .context("Failed to load feature_merge PTX module")?;

        let feature_merge_kernel = module
            .load_function("feature_merge_kernel")
            .context("Failed to load feature_merge_kernel function")?;

        // Create stream for kernel execution
        let stream = device.default_stream();

        Ok(Self {
            device,
            stream,
            feature_merge_kernel,
            config,
        })
    }

    /// Merge main features (136-dim) with cryptic features (4-dim) to produce combined (140-dim)
    ///
    /// # Arguments
    /// * `main_features` - Input main features [n_residues * 136]
    /// * `cryptic_features` - Input cryptic features [n_residues * 4]
    /// * `n_residues` - Number of residues to process
    ///
    /// # Returns
    /// Combined features [n_residues * 140] on GPU
    pub fn merge_features(
        &self,
        main_features: &CudaSlice<f32>,
        cryptic_features: &CudaSlice<f32>,
        n_residues: usize,
    ) -> Result<CudaSlice<f32>> {
        // Validate inputs
        if n_residues == 0 {
            return Err(anyhow::anyhow!("n_residues must be > 0"));
        }
        if n_residues > self.config.max_residues {
            return Err(anyhow::anyhow!(
                "n_residues {} exceeds max_residues {}",
                n_residues,
                self.config.max_residues
            ));
        }

        let expected_main_size = n_residues * 136;
        let expected_cryptic_size = n_residues * 4;

        if main_features.len() != expected_main_size {
            return Err(anyhow::anyhow!(
                "main_features size mismatch: expected {}, got {}",
                expected_main_size,
                main_features.len()
            ));
        }
        if cryptic_features.len() != expected_cryptic_size {
            return Err(anyhow::anyhow!(
                "cryptic_features size mismatch: expected {}, got {}",
                expected_cryptic_size,
                cryptic_features.len()
            ));
        }

        // Allocate output buffer using device (Memory belongs to Device)
        let output_size = n_residues * 140;
        let mut combined_features = self.stream.alloc_zeros::<f32>(output_size)?;

        // Calculate launch configuration
        let block_size = self.config.block_size;
        let grid_size = (n_residues as u32 + block_size - 1) / block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            self.stream
                .launch_builder(&self.feature_merge_kernel)
                .arg(main_features)
                .arg(cryptic_features)
                .arg(&mut combined_features)
                .arg(&(n_residues as i32))
                .launch(launch_config)?;
        }

        // Synchronize to ensure completion
        self.stream.synchronize()?;

        Ok(combined_features)
    }

    /// Merge features with pre-allocated output buffer (zero-copy)
    ///
    /// # Arguments
    /// * `main_features` - Input main features [n_residues * 136]
    /// * `cryptic_features` - Input cryptic features [n_residues * 4]
    /// * `output` - Pre-allocated output buffer [n_residues * 140]
    /// * `n_residues` - Number of residues to process
    pub fn merge_features_inplace(
        &self,
        main_features: &CudaSlice<f32>,
        cryptic_features: &CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        n_residues: usize,
    ) -> Result<()> {
        // Validate inputs (same as above)
        if n_residues == 0 {
            return Err(anyhow::anyhow!("n_residues must be > 0"));
        }
        if n_residues > self.config.max_residues {
            return Err(anyhow::anyhow!(
                "n_residues {} exceeds max_residues {}",
                n_residues,
                self.config.max_residues
            ));
        }

        let expected_main_size = n_residues * 136;
        let expected_cryptic_size = n_residues * 4;
        let expected_output_size = n_residues * 140;

        if main_features.len() != expected_main_size {
            return Err(anyhow::anyhow!(
                "main_features size mismatch: expected {}, got {}",
                expected_main_size,
                main_features.len()
            ));
        }
        if cryptic_features.len() != expected_cryptic_size {
            return Err(anyhow::anyhow!(
                "cryptic_features size mismatch: expected {}, got {}",
                expected_cryptic_size,
                cryptic_features.len()
            ));
        }
        if output.len() != expected_output_size {
            return Err(anyhow::anyhow!(
                "output size mismatch: expected {}, got {}",
                expected_output_size,
                output.len()
            ));
        }

        // Calculate launch configuration
        let block_size = self.config.block_size;
        let grid_size = (n_residues as u32 + block_size - 1) / block_size;

        let launch_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel
        unsafe {
            self.stream
                .launch_builder(&self.feature_merge_kernel)
                .arg(main_features)
                .arg(cryptic_features)
                .arg(output)
                .arg(&(n_residues as i32))
                .launch(launch_config)?;
        }

        // Synchronize to ensure completion
        self.stream.synchronize()?;

        Ok(())
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.device
    }

    /// Get configuration
    pub fn config(&self) -> &FeatureMergeConfig {
        &self.config
    }
}

/// Feature Merge operation result
#[derive(Debug, Clone)]
pub struct FeatureMergeResult {
    /// Combined features [n_residues * 140]
    pub combined_features: Vec<f32>,
    /// Number of residues processed
    pub n_residues: usize,
    /// Kernel execution time (microseconds)
    pub execution_time_us: u64,
}

impl FeatureMergeResult {
    /// Get feature vector for specific residue
    ///
    /// # Arguments
    /// * `residue_idx` - Residue index (0-based)
    ///
    /// # Returns
    /// Feature slice [140 elements] or None if index out of bounds
    pub fn get_residue_features(&self, residue_idx: usize) -> Option<&[f32]> {
        if residue_idx >= self.n_residues {
            return None;
        }

        let start = residue_idx * 140;
        let end = start + 140;
        Some(&self.combined_features[start..end])
    }

    /// Validate that features are correctly merged
    ///
    /// Checks that the main features are in positions [0..135] and
    /// cryptic features are in positions [136..139].
    pub fn validate_merge(&self, main_features: &[f32], cryptic_features: &[f32]) -> bool {
        for residue_idx in 0..self.n_residues {
            if let Some(combined) = self.get_residue_features(residue_idx) {
                // Check main features [0..135]
                let main_start = residue_idx * 136;
                let main_end = main_start + 136;
                if main_features.len() >= main_end {
                    for i in 0..136 {
                        if (combined[i] - main_features[main_start + i]).abs() > 1e-6 {
                            return false;
                        }
                    }
                }

                // Check cryptic features [136..139]
                let cryptic_start = residue_idx * 4;
                let cryptic_end = cryptic_start + 4;
                if cryptic_features.len() >= cryptic_end {
                    for i in 0..4 {
                        if (combined[136 + i] - cryptic_features[cryptic_start + i]).abs() > 1e-6 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_merge_config_default() {
        let config = FeatureMergeConfig::default();
        assert_eq!(config.block_size, 128);
        assert_eq!(config.max_residues, 1_048_576);
    }

    #[test]
    fn test_feature_merge_result_get_residue_features() {
        let result = FeatureMergeResult {
            combined_features: vec![1.0; 280], // 2 residues * 140 features
            n_residues: 2,
            execution_time_us: 100,
        };

        assert!(result.get_residue_features(0).is_some());
        assert!(result.get_residue_features(1).is_some());
        assert!(result.get_residue_features(2).is_none());

        let features = result.get_residue_features(0).unwrap();
        assert_eq!(features.len(), 140);
    }
}
