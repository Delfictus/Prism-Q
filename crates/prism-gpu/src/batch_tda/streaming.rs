//! Double-Buffered Streaming TDA Pipeline
//!
//! Overlaps CPU neighborhood building with GPU TDA computation for maximum throughput.
//! Uses separate CUDA streams for independent execution.
//!
//! Pipeline stages:
//! 1. CPU: Build neighborhoods for batch N+1
//! 2. GPU: Compute TDA for batch N
//! 3. CPU: Download results for batch N-1

use cudarc::driver::{DevicePtrMut, 
    CudaContext, CudaStream, CudaFunction, CudaSlice,
    LaunchConfig, PushKernelArg, DeviceSlice};
use cudarc::nvrtc::Ptx;
use prism_core::PrismError;
use std::path::Path;
use std::sync::Arc;
use std::collections::VecDeque;

use super::neighborhood::{NeighborhoodBuilder, NeighborhoodData};
use super::hybrid_executor::TdaFeatures;
use super::{
    TDA_RADII, TDA_SCALES, TDA_FEATURE_COUNT,
    NUM_RADII, MAX_NEIGHBORS, MAX_PREFETCH, STREAMING_THRESHOLD,
};

/// Configuration for streaming pipeline
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// Maximum structures to prefetch
    pub max_prefetch: usize,
    /// Minimum batch size for streaming mode
    pub min_batch_size: usize,
    /// TDA radii
    pub radii: [f32; NUM_RADII],
    /// Persistence scales
    pub scales: [f32; 4],
    /// Maximum neighbors per residue
    pub max_neighbors: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_prefetch: MAX_PREFETCH,
            min_batch_size: STREAMING_THRESHOLD,
            radii: TDA_RADII,
            scales: TDA_SCALES,
            max_neighbors: MAX_NEIGHBORS,
        }
    }
}

/// Double-buffered GPU buffers
struct DoubleBuffer {
    /// Buffer A
    a: StreamBuffer,
    /// Buffer B
    b: StreamBuffer,
    /// Which buffer is currently being used for GPU compute
    gpu_active: bool,
}

struct StreamBuffer {
    d_offsets: CudaSlice<u32>,
    d_neighbor_indices: CudaSlice<u32>,
    d_neighbor_distances: CudaSlice<u16>,
    d_neighbor_coords: CudaSlice<f32>,
    d_center_coords: CudaSlice<f32>,
    d_features: CudaSlice<f32>,
    capacity: usize,
}

/// Streaming TDA Pipeline
///
/// Provides high-throughput TDA extraction for large structure batches.
pub struct StreamingTdaPipeline {
    /// CUDA context
    ctx: Arc<CudaContext>,
    /// Compute stream
    compute_stream: Arc<CudaStream>,
    /// Transfer stream
    transfer_stream: Arc<CudaStream>,
    /// TDA kernel
    kernel: CudaFunction,
    /// Double buffers
    buffers: DoubleBuffer,
    /// Neighborhood builder
    nb_builder: NeighborhoodBuilder,
    /// Configuration
    config: StreamingConfig,
}

impl StreamingTdaPipeline {
    /// Create a new streaming pipeline
    pub fn new(ctx: Arc<CudaContext>, ptx_path: &Path) -> Result<Self, PrismError> {
        // Create separate streams for compute and transfer
        let compute_stream = ctx.default_stream();
        let transfer_stream = ctx.new_stream()
            .map_err(|e| PrismError::gpu("streaming", format!("Create transfer stream: {:?}", e)))?;

        // Load kernel
        let ptx_src = std::fs::read_to_string(ptx_path)
            .map_err(|e| PrismError::config(format!("Read PTX: {}", e)))?;
        let ptx = Ptx::from_src(ptx_src);
        
        let module = ctx.load_module(ptx)
            .map_err(|e| PrismError::config(format!("Load module: {:?}", e)))?;
            
        let kernel = module.load_function("hybrid_tda_kernel")
            .ok_or_else(|| PrismError::config("Failed to get TDA kernel".to_string()))?;

        let config = StreamingConfig::default();
        let nb_builder = NeighborhoodBuilder::new()
            .with_radii(config.radii.to_vec())
            .with_max_neighbors(config.max_neighbors);

        // Allocate initial buffers (will resize as needed)
        let initial_capacity = 1000;
        let initial_neighbors = 50000;

        let alloc_buffer = |stream: &Arc<CudaStream>| -> Result<StreamBuffer, PrismError> {
            Ok(StreamBuffer {
                d_offsets: stream.alloc_zeros::<u32>(initial_capacity * NUM_RADII + 1)
                    .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                d_neighbor_indices: stream.alloc_zeros::<u32>(initial_neighbors)
                    .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                d_neighbor_distances: stream.alloc_zeros::<u16>(initial_neighbors)
                    .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                d_neighbor_coords: stream.alloc_zeros::<f32>(initial_neighbors * 3)
                    .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                d_center_coords: stream.alloc_zeros::<f32>(initial_capacity * 3)
                    .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                d_features: stream.alloc_zeros::<f32>(initial_capacity * TDA_FEATURE_COUNT)
                    .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                capacity: initial_capacity,
            })
        };

        let buffers = DoubleBuffer {
            a: alloc_buffer(&compute_stream)?,
            b: alloc_buffer(&compute_stream)?,
            gpu_active: false,
        };

        Ok(Self {
            ctx,
            compute_stream,
            transfer_stream,
            kernel,
            buffers,
            nb_builder,
            config,
        })
    }

    /// Process a batch of structures with pipelining
    ///
    /// Returns TDA features for each structure in order.
    pub fn process_batch(&mut self, structures: Vec<Vec<[f32; 3]>>) -> Result<Vec<TdaFeatures>, PrismError> {
        let n_structures = structures.len();

        if n_structures == 0 {
            return Ok(vec![]);
        }

        // For small batches, use simple sequential processing
        if n_structures < self.config.min_batch_size {
            return self.process_sequential(structures);
        }

        // Pipeline processing
        self.process_pipelined(structures)
    }

    /// Sequential processing (for small batches)
    fn process_sequential(&mut self, structures: Vec<Vec<[f32; 3]>>) -> Result<Vec<TdaFeatures>, PrismError> {
        let mut results = Vec::with_capacity(structures.len();

        for (_idx, coords) in structures.into_iter().enumerate() {
            let start = std::time::Instant::now();
            let n_residues = coords.len();

            if n_residues == 0 {
                results.push(TdaFeatures {
                    n_residues: 0,
                    features: vec![],
                    compute_time_us: 0,
                    gpu_used: false,
                });
                continue;
            }

            // Build neighborhood
            let neighborhood = self.nb_builder.build(&coords);

            // Process on GPU
            let features = self.execute_single(&neighborhood)?;

            results.push(TdaFeatures {
                n_residues,
                features,
                compute_time_us: start.elapsed().as_micros() as u64,
                gpu_used: true,
            });
        }

        Ok(results)
    }

    /// Pipelined processing (for large batches)
    fn process_pipelined(&mut self, structures: Vec<Vec<[f32; 3]>>) -> Result<Vec<TdaFeatures>, PrismError> {
        let n_structures = structures.len();
        let mut results = vec![None; n_structures];
        let mut pending: VecDeque<(usize, NeighborhoodData, usize)> = VecDeque::new());

        // Stage 1: Prepare first batch of neighborhoods (CPU)
        let prefetch_count = self.config.max_prefetch.min(n_structures);
        for idx in 0..prefetch_count {
            let coords = &structures[idx];
            let n_residues = coords.len();
            if n_residues > 0 {
                let neighborhood = self.nb_builder.build(coords);
                pending.push_back((idx, neighborhood, n_residues);
            } else {
                results[idx] = Some(TdaFeatures {
                    n_residues: 0,
                    features: vec![],
                    compute_time_us: 0,
                    gpu_used: false,
                });
            }
        }

        let mut next_to_prepare = prefetch_count;
        let mut completed = 0;

        // Process pipeline
        while completed < n_structures {
            // Process pending items
            if let Some((idx, neighborhood, n_residues)) = pending.pop_front() {
                let start = std::time::Instant::now();

                // Execute on GPU
                let features = self.execute_single(&neighborhood)?;

                results[idx] = Some(TdaFeatures {
                    n_residues,
                    features,
                    compute_time_us: start.elapsed().as_micros() as u64,
                    gpu_used: true,
                });
                completed += 1;

                // Prepare next structure while GPU is busy (if available)
                if next_to_prepare < n_structures {
                    let coords = &structures[next_to_prepare];
                    let n_residues = coords.len();
                    if n_residues > 0 {
                        let neighborhood = self.nb_builder.build(coords);
                        pending.push_back((next_to_prepare, neighborhood, n_residues);
                    } else {
                        results[next_to_prepare] = Some(TdaFeatures {
                            n_residues: 0,
                            features: vec![],
                            compute_time_us: 0,
                            gpu_used: false,
                        });
                        completed += 1;
                    }
                    next_to_prepare += 1;
                }
            } else {
                // No more pending items
                break;
            }
        }

        // Convert Option<TdaFeatures> to TdaFeatures
        Ok(results.into_iter().map(|r| r.unwrap_or(TdaFeatures {
            n_residues: 0,
            features: vec![],
            compute_time_us: 0,
            gpu_used: false,
        })).collect())
    }

    /// Execute TDA for a single structure
    fn execute_single(&mut self, neighborhood: &NeighborhoodData) -> Result<Vec<f32>, PrismError> {
        let n_residues = neighborhood.n_residues;
        let total_neighbors = neighborhood.total_neighbors();

        // Ensure buffer capacity
        self.ensure_buffer_capacity(n_residues, total_neighbors)?;

        // Get mutable reference to active buffer for upload
        let gpu_active = self.buffers.gpu_active;
        {
            let buffer = if gpu_active {
                &mut self.buffers.b
            } else {
                &mut self.buffers.a
            };

            // Upload data (copy into pre-allocated buffers)
            buffer.d_offsets = self.compute_stream.clone_htod(neighborhood.offsets))
                .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?;
            buffer.d_neighbor_indices = self.compute_stream.clone_htod(neighborhood.neighbor_indices))
                .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?;
            buffer.d_neighbor_distances = self.compute_stream.clone_htod(neighborhood.neighbor_distances))
                .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?;
            buffer.d_neighbor_coords = self.compute_stream.clone_htod(neighborhood.neighbor_coords))
                .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?;
            buffer.d_center_coords = self.compute_stream.clone_htod(neighborhood.center_coords))
                .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?;
        }

        // Get immutable reference for launch
        let buffer = if gpu_active {
            &self.buffers.b
        } else {
            &self.buffers.a
        };

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
            &self.compute_stream.launch_builder(&self.kernel)
                .arg(&buffer.d_offsets)
                .arg(&buffer.d_neighbor_indices)
                .arg(&buffer.d_neighbor_distances)
                .arg(&buffer.d_neighbor_coords)
                .arg(&buffer.d_center_coords)
                .arg(&buffer.d_features)
                .arg(&n_residues_u32)
                .arg(&n_radii_u32)
                .arg(&self.config.scales[0])
                .arg(&self.config.scales[1])
                .arg(&self.config.scales[2])
                .arg(&self.config.scales[3])
                .launch(cfg).map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?;
        }

        // Sync and download
        self.compute_stream.synchronize()
            .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?;

        let output_len = n_residues * TDA_FEATURE_COUNT;
        let output = self.compute_stream.clone_dtoh(&buffer.d_features)
            .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?;

        // Toggle active buffer
        self.buffers.gpu_active = !self.buffers.gpu_active;

        // Truncate to actual size
        let output = output.into_iter().take(output_len).collect();

        Ok(output)
    }

    /// Ensure buffer capacity
    fn ensure_buffer_capacity(&mut self, n_residues: usize, total_neighbors: usize) -> Result<(), PrismError> {
        let needed_capacity = (n_residues * 12 / 10).max(n_residues);
        let needed_neighbors = (total_neighbors * 12 / 10).max(total_neighbors);

        // Check if reallocation needed
        let need_realloc = self.buffers.a.capacity < needed_capacity;

        if need_realloc {
            log::debug!("Reallocating stream buffers: {} -> {} residues",
                       self.buffers.a.capacity, needed_capacity);

            let alloc_buffer = |stream: &Arc<CudaStream>| -> Result<StreamBuffer, PrismError> {
                Ok(StreamBuffer {
                    d_offsets: stream.alloc_zeros::<u32>(needed_capacity * NUM_RADII + 1)
                        .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                    d_neighbor_indices: stream.alloc_zeros::<u32>(needed_neighbors)
                        .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                    d_neighbor_distances: stream.alloc_zeros::<u16>(needed_neighbors)
                        .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                    d_neighbor_coords: stream.alloc_zeros::<f32>(needed_neighbors * 3)
                        .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                    d_center_coords: stream.alloc_zeros::<f32>(needed_capacity * 3)
                        .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                    d_features: stream.alloc_zeros::<f32>(needed_capacity * TDA_FEATURE_COUNT)
                        .map_err(|e| PrismError::gpu("streaming", format!("{:?}", e)))?,
                    capacity: needed_capacity,
                })
            };

            self.buffers.a = alloc_buffer(&self.compute_stream)?;
            self.buffers.b = alloc_buffer(&self.compute_stream)?;
        }

        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &StreamingConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: StreamingConfig) {
        self.config = config.clone();
        self.nb_builder = NeighborhoodBuilder::new()
            .with_radii(config.radii.to_vec())
            .with_max_neighbors(config.max_neighbors);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.max_prefetch, MAX_PREFETCH);
        assert_eq!(config.min_batch_size, STREAMING_THRESHOLD);
    }
}
