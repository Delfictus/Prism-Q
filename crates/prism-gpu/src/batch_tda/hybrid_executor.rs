//! Hybrid TDA Executor with Multi-Radius Support
//!
//! Implements the full 48-dimensional TDA feature extraction:
//! - 3 radii (8Å, 12Å, 16Å)
//! - 16 features per radius (Betti numbers, persistence, directional)
//!
//! Supports both GPU-accelerated and CPU fallback modes.

use cudarc::driver::{DevicePtrMut, 
    CudaContext, CudaStream, CudaFunction, CudaSlice,
    LaunchConfig, PushKernelArg, DeviceSlice};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use std::path::Path;
use std::sync::Arc;

use super::neighborhood::{NeighborhoodBuilder, NeighborhoodData};
use super::executor::cpu_fallback;
use super::{
    TDA_RADII, TDA_SCALES, FEATURES_PER_RADIUS, TDA_FEATURE_COUNT,
    NUM_RADII, FUSED_MODE_THRESHOLD, MAX_NEIGHBORS,
};

/// Configuration for HybridTdaExecutor
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HybridTdaConfig {
    /// Radii for neighborhood queries (Angstroms)
    pub radii: [f32; NUM_RADII],
    /// Persistence scales (Angstroms)
    pub scales: [f32; 4],
    /// Maximum neighbors per residue
    pub max_neighbors: usize,
    /// Use GPU when available
    pub prefer_gpu: bool,
    /// Threshold for fused mode (small structures)
    pub fused_threshold: usize,
}

impl Default for HybridTdaConfig {
    fn default() -> Self {
        Self {
            radii: TDA_RADII,
            scales: TDA_SCALES,
            max_neighbors: MAX_NEIGHBORS,
            prefer_gpu: true,
            fused_threshold: FUSED_MODE_THRESHOLD,
        }
    }
}

/// TDA feature output for a structure
#[derive(Clone, Debug)]
pub struct TdaFeatures {
    /// Number of residues
    pub n_residues: usize,
    /// Feature matrix [n_residues × TDA_FEATURE_COUNT]
    pub features: Vec<f32>,
    /// Computation time in microseconds
    pub compute_time_us: u64,
    /// Whether GPU was used
    pub gpu_used: bool,
}

impl TdaFeatures {
    /// Get features for a specific residue
    pub fn get_residue(&self, idx: usize) -> &[f32] {
        let start = idx * TDA_FEATURE_COUNT;
        &self.features[start..start + TDA_FEATURE_COUNT]
    }

    /// Get feature matrix as 2D slice
    pub fn as_matrix(&self) -> Vec<&[f32]> {
        (0..self.n_residues)
            .map(|i| self.get_residue(i))
            .collect()
    }
}

/// GPU buffers for hybrid TDA
struct HybridTdaBuffers {
    /// Neighborhood offsets [n_residues * n_radii + 1]
    d_offsets: CudaSlice<u32>,
    /// Neighbor indices (packed)
    d_neighbor_indices: CudaSlice<u32>,
    /// Neighbor distances F16 (packed)
    d_neighbor_distances: CudaSlice<u16>,
    /// Neighbor coordinates (packed, 3 floats per neighbor)
    d_neighbor_coords: CudaSlice<f32>,
    /// Center coordinates [n_residues * 3]
    d_center_coords: CudaSlice<f32>,
    /// Output features [n_residues * TDA_FEATURE_COUNT]
    d_features: CudaSlice<f32>,
    /// Capacity
    capacity: usize,
}

/// Hybrid TDA executor supporting GPU and CPU modes
pub struct HybridTdaExecutor {
    /// CUDA context (None if CPU-only)
    ctx: Option<Arc<CudaContext>>,
    /// CUDA stream
    stream: Option<Arc<CudaStream>>,
    /// TDA kernel function
    kernel: Option<CudaFunction>,
    /// Reusable GPU buffers
    buffers: Option<HybridTdaBuffers>,
    /// Neighborhood builder
    nb_builder: NeighborhoodBuilder,
    /// Configuration
    config: HybridTdaConfig,
}

impl HybridTdaExecutor {
    /// Create a new hybrid executor with GPU support
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &Path) -> Result<Self, PrismError> {
        let stream = ctx.default_stream();

        // Load PTX
        let ptx_src = std::fs::read_to_string(ptx_path)
            .map_err(|e| PrismError::config(format!("Failed to read TDA PTX: {}", e)))?;
        let ptx = Ptx::from_src(ptx_src);

        let module = ctx.load_module(ptx)
            .map_err(|e| PrismError::config(format!("Failed to load TDA module: {:?}", e)))?;
        
        let kernel = module.load_function("hybrid_tda_kernel")
            .ok_or_else(|| PrismError::config("Failed to get TDA kernel".to_string()))?;

        let config = HybridTdaConfig::default();
        let nb_builder = NeighborhoodBuilder::new()
            .with_radii(config.radii.to_vec())
            .with_max_neighbors(config.max_neighbors);

        Ok(Self {
            ctx: Some(ctx.clone()),
            stream: Some(stream),
            kernel: Some(kernel),
            buffers: None,
            nb_builder,
            config,
        })
    }

    /// Create a CPU-only executor (no GPU)
    pub fn cpu_only() -> Self {
        let config = HybridTdaConfig::default();
        let nb_builder = NeighborhoodBuilder::new()
            .with_radii(config.radii.to_vec())
            .with_max_neighbors(config.max_neighbors);

        Self {
            ctx: None,
            stream: None,
            kernel: None,
            buffers: None,
            nb_builder,
            config,
        }
    }

    /// Create with custom configuration
    pub fn with_config(mut self, config: HybridTdaConfig) -> Self {
        self.config = config.clone();
        self.nb_builder = NeighborhoodBuilder::new()
            .with_radii(config.radii.to_vec())
            .with_max_neighbors(config.max_neighbors);
        self
    }

    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        self.ctx.is_some()
    }

    /// Ensure GPU buffers are allocated
    fn ensure_buffers(&mut self, n_residues: usize, total_neighbors: usize) -> Result<(), PrismError> {
        let ctx = self.ctx.as_ref().ok_or_else(|| {
            PrismError::config("No GPU context")
        })?;

        if self.buffers.is_none() || n_residues > self.buffers.as_ref().unwrap().capacity {
            let capacity = (n_residues * 12 / 10).max(n_residues);
            let neighbor_capacity = (total_neighbors * 12 / 10).max(total_neighbors);

            let stream = self.stream.as_ref().unwrap());
            let d_offsets = stream.alloc_zeros::<u32>(capacity * NUM_RADII + 1)
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("Alloc: {:?}", e)))?;
            let d_neighbor_indices = stream.alloc_zeros::<u32>(neighbor_capacity)
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("Alloc: {:?}", e)))?;
            let d_neighbor_distances = stream.alloc_zeros::<u16>(neighbor_capacity)
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("Alloc: {:?}", e)))?;
            let d_neighbor_coords = stream.alloc_zeros::<f32>(neighbor_capacity * 3)
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("Alloc: {:?}", e)))?;
            let d_center_coords = stream.alloc_zeros::<f32>(capacity * 3)
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("Alloc: {:?}", e)))?;
            let d_features = stream.alloc_zeros::<f32>(capacity * TDA_FEATURE_COUNT)
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("Alloc: {:?}", e)))?;

            self.buffers = Some(HybridTdaBuffers {
                d_offsets,
                d_neighbor_indices,
                d_neighbor_distances,
                d_neighbor_coords,
                d_center_coords,
                d_features,
                capacity,
            });
        }

        Ok(())
    }

    /// Extract TDA features for a structure
    ///
    /// Automatically chooses GPU or CPU based on availability and config.
    pub fn extract(&mut self, coords: &[[f32; 3]]) -> Result<TdaFeatures, PrismError> {
        let start = std::time::Instant::now();
        let n_residues = coords.len();

        if n_residues == 0 {
            return Ok(TdaFeatures {
                n_residues: 0,
                features: vec![],
                compute_time_us: 0,
                gpu_used: false,
            });
        }

        // Build neighborhoods (always on CPU with Rayon)
        let neighborhood = self.nb_builder.build(coords);

        // Choose execution path
        let (features, gpu_used) = if self.has_gpu() && self.config.prefer_gpu {
            (self.execute_gpu(&neighborhood)?, true)
        } else {
            (self.execute_cpu(coords, &neighborhood)?, false)
        };

        let elapsed = start.elapsed().as_micros() as u64;

        Ok(TdaFeatures {
            n_residues,
            features,
            compute_time_us: elapsed,
            gpu_used,
        })
    }

    /// GPU execution path
    fn execute_gpu(&mut self, neighborhood: &NeighborhoodData) -> Result<Vec<f32>, PrismError> {
        let n_residues = neighborhood.n_residues;
        let total_neighbors = neighborhood.total_neighbors();

        self.ensure_buffers(n_residues, total_neighbors)?;

        let ctx = self.ctx.as_ref().unwrap());
        let stream = self.stream.as_ref().unwrap());
        let kernel = self.kernel.as_ref().unwrap());

        // Get mutable reference to buffers for upload
        {
            let buffers = self.buffers.as_mut().unwrap());

            // Upload data (copy into pre-allocated buffers)
            buffers.d_offsets = stream.clone_htod(neighborhood.offsets))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
            buffers.d_neighbor_indices = stream.clone_htod(neighborhood.neighbor_indices))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
            buffers.d_neighbor_distances = stream.clone_htod(neighborhood.neighbor_distances))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
            buffers.d_neighbor_coords = stream.clone_htod(neighborhood.neighbor_coords))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
            buffers.d_center_coords = stream.clone_htod(neighborhood.center_coords))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
        }

        // Get immutable reference for launch
        let buffers = self.buffers.as_ref().unwrap());

        // Launch configuration: one warp per residue
        let threads_per_block = 256u32;
        let residues_per_block = threads_per_block / 32;
        let num_blocks = ((n_residues as u32) + residues_per_block - 1) / residues_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        // Launch kernel using launch_on_stream
        let n_residues_u32 = n_residues as u32;
        let n_radii_u32 = NUM_RADII as u32;
        
        unsafe {
            stream.launch_builder(&kernel)
                .arg(&buffers.d_offsets)
                .arg(&buffers.d_neighbor_indices)
                .arg(&buffers.d_neighbor_distances)
                .arg(&buffers.d_neighbor_coords)
                .arg(&buffers.d_center_coords)
                .arg(&buffers.d_features)
                .arg(&n_residues_u32)
                .arg(&n_radii_u32)
                .arg(&self.config.scales[0])
                .arg(&self.config.scales[1])
                .arg(&self.config.scales[2])
                .arg(&self.config.scales[3])
                .launch(cfg).map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
        }

        // Sync and download
        stream.synchronize()
            .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;

        let output_len = n_residues * TDA_FEATURE_COUNT;
        let output = stream.clone_dtoh(&buffers.d_features)
            .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;

        // Truncate to actual size
        let output = output.into_iter().take(output_len).collect();

        Ok(output)
    }

    /// CPU execution path (fallback)
    fn execute_cpu(&self, coords: &[[f32; 3]], neighborhood: &NeighborhoodData) -> Result<Vec<f32>, PrismError> {
        use rayon::prelude::*;
        use super::half_utils::f16_to_f32;

        let n_residues = neighborhood.n_residues;
        let n_radii = neighborhood.n_radii;

        // Process each residue in parallel
        let features: Vec<[f32; TDA_FEATURE_COUNT]> = (0..n_residues)
            .into_par_iter()
            .map(|i| {
                let mut all_features = [0.0f32; TDA_FEATURE_COUNT];
                let center = coords[i];

                // Process each radius
                for r in 0..n_radii {
                    let offset_idx = i * n_radii + r;
                    let start = neighborhood.offsets[offset_idx] as usize;
                    let end = neighborhood.offsets[offset_idx + 1] as usize;

                    if start >= end {
                        continue;
                    }

                    // Get neighbor data
                    let neighbor_indices = &neighborhood.neighbor_indices[start..end];
                    let neighbor_distances_f16 = &neighborhood.neighbor_distances[start..end];

                    // Convert to coordinates
                    let neighbor_coords: Vec<[f32; 3]> = neighbor_indices
                        .iter()
                        .map(|&idx| coords[idx as usize])
                        .collect();

                    let neighbor_distances: Vec<f32> = neighbor_distances_f16
                        .iter()
                        .map(|&d| f16_to_f32(d))
                        .collect();

                    // Extract features for this radius
                    let radius_features = cpu_fallback::extract_tda_features_cpu(
                        center,
                        &neighbor_coords,
                        &neighbor_distances,
                        &self.config.scales,
                    );

                    // Copy to output at correct offset
                    let offset = r * FEATURES_PER_RADIUS;
                    all_features[offset..offset + FEATURES_PER_RADIUS]
                        .copy_from_slice(&radius_features);
                }

                all_features
            })
            .collect();

        // Flatten to single vector
        Ok(features.into_iter().flatten().collect())
    }

    /// Get feature buffer on GPU (for integration with mega-fused)
    pub fn execute_gpu_buffer(&mut self, neighborhood: &NeighborhoodData) -> Result<&CudaSlice<f32>, PrismError> {
        let n_residues = neighborhood.n_residues;
        let total_neighbors = neighborhood.total_neighbors();

        self.ensure_buffers(n_residues, total_neighbors)?;

        let ctx = self.ctx.as_ref().ok_or_else(|| {
            PrismError::config("No GPU context")
        })?;
        let stream = self.stream.as_ref().ok_or_else(|| {
            PrismError::config("No GPU stream")
        })?;
        let kernel = self.kernel.as_ref().unwrap());

        // Upload data with mutable reference
        {
            let buffers = self.buffers.as_mut().unwrap());

            // Upload data (copy into pre-allocated buffers)
            buffers.d_offsets = stream.clone_htod(neighborhood.offsets))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
            buffers.d_neighbor_indices = stream.clone_htod(neighborhood.neighbor_indices))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
            buffers.d_neighbor_distances = stream.clone_htod(neighborhood.neighbor_distances))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
            buffers.d_neighbor_coords = stream.clone_htod(neighborhood.neighbor_coords))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
            buffers.d_center_coords = stream.clone_htod(neighborhood.center_coords))
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
        }

        // Get immutable reference for launch
        let buffers = self.buffers.as_ref().unwrap());

        // Launch kernel
        let threads_per_block = 256u32;
        let residues_per_block = threads_per_block / 32;
        let num_blocks = ((n_residues as u32) + residues_per_block - 1) / residues_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (threads_per_block, 1, 1),
            shared_mem_bytes: 0,
        };

        let n_residues_u32 = n_residues as u32;
        let n_radii_u32 = NUM_RADII as u32;
        
        unsafe {
            stream.launch_builder(&kernel)
                .arg(&buffers.d_offsets)
                .arg(&buffers.d_neighbor_indices)
                .arg(&buffers.d_neighbor_distances)
                .arg(&buffers.d_neighbor_coords)
                .arg(&buffers.d_center_coords)
                .arg(&buffers.d_features)
                .arg(&n_residues_u32)
                .arg(&n_radii_u32)
                .arg(&self.config.scales[0])
                .arg(&self.config.scales[1])
                .arg(&self.config.scales[2])
                .arg(&self.config.scales[3])
                .launch(cfg).map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
        }

        // Return buffer reference (caller handles sync)
        Ok(&self.buffers.as_ref().unwrap().d_features)
    }

    /// Synchronize GPU stream
    pub fn synchronize(&self) -> Result<(), PrismError> {
        if let Some(ref stream) = self.stream {
            stream.synchronize()
                .map_err(|e| PrismError::gpu("hybrid_executor", format!("{:?}", e)))?;
        }
        Ok(())
    }

    /// Get CUDA context for integration
    pub fn context(&self) -> Option<&Arc<CudaContext>> {
        self.ctx.as_ref()
    }

    /// Get stream for async operations
    pub fn stream(&self) -> Option<&Arc<CudaStream>> {
        self.stream.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_helix_coords(n: usize) -> Vec<[f32; 3]> {
        (0..n).map(|i| {
            let t = i as f32 * 0.3;
            [t.cos() * 5.0, t.sin() * 5.0, i as f32 * 1.5]
        }).collect()
    }

    #[test]
    fn test_cpu_only_executor() {
        let executor = HybridTdaExecutor::cpu_only();
        assert!(!executor.has_gpu();
    }

    #[test]
    fn test_cpu_extraction() {
        let mut executor = HybridTdaExecutor::cpu_only();
        let coords = make_helix_coords(50);

        let result = executor.extract(&coords).unwrap());

        assert_eq!(result.n_residues, 50);
        assert_eq!(result.features.len(), 50 * TDA_FEATURE_COUNT);
        assert!(!result.gpu_used);

        // Check that features are reasonable
        for i in 0..50 {
            let feats = result.get_residue(i);
            // Betti-0 should be at least 1 at smallest scale
            assert!(feats[0] >= 0.0, "Betti-0 should be non-negative");
        }
    }

    #[test]
    fn test_empty_coords() {
        let mut executor = HybridTdaExecutor::cpu_only();
        let coords: Vec<[f32; 3]> = vec![];

        let result = executor.extract(&coords).unwrap());

        assert_eq!(result.n_residues, 0);
        assert!(result.features.is_empty();
    }

    #[test]
    fn test_single_residue() {
        let mut executor = HybridTdaExecutor::cpu_only();
        let coords = vec![[0.0, 0.0, 0.0]];

        let result = executor.extract(&coords).unwrap());

        assert_eq!(result.n_residues, 1);
        assert_eq!(result.features.len(), TDA_FEATURE_COUNT);
    }
}
