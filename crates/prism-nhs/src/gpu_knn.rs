//! GPU-Accelerated k-Nearest Neighbor Distance Computation
//!
//! Provides fast k-NN distance computation for adaptive epsilon selection.
//! Uses CUDA kernels for parallel distance computation.
//!
//! Performance:
//! - CPU brute force: ~5 seconds for 1000 queries / 40000 points
//! - GPU parallel: ~50ms (100x speedup)

use anyhow::{Context, Result};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

/// Result from GPU k-NN computation
#[derive(Debug, Clone)]
pub struct GpuKnnResult {
    /// k-th nearest neighbor distances for each query point
    pub kth_distances: Vec<f32>,
    /// Computed epsilon values (percentiles of k-th distances)
    pub epsilon_values: Vec<f32>,
    /// Number of query points processed
    pub num_queries: usize,
    /// GPU computation time in milliseconds
    pub gpu_time_ms: f64,
}

/// GPU k-NN engine for adaptive epsilon computation
#[cfg(feature = "gpu")]
pub struct GpuKnnEngine {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,
    fn_knn_simple: CudaFunction,
    fn_knn_tiled: CudaFunction,
    fn_extract_percentiles: CudaFunction,
    fn_adaptive_full: CudaFunction,
}

#[cfg(feature = "gpu")]
impl GpuKnnEngine {
    /// Create a new GPU k-NN engine
    pub fn new(context: Arc<CudaContext>) -> Result<Self> {
        let stream = context.default_stream();

        // Load PTX module
        let ptx_path = Self::find_ptx_path()?;
        log::info!("Loading k-NN CUDA kernels from: {}", ptx_path.display());

        let ptx_content = std::fs::read_to_string(&ptx_path)
            .with_context(|| format!("Failed to read k-NN PTX: {}", ptx_path.display()))?;

        let module = context
            .load_module(cudarc::nvrtc::Ptx::from_src(&ptx_content))
            .context("Failed to load k-NN CUDA module")?;

        // Load functions
        let fn_knn_simple = module.load_function("compute_knn_distances")?;
        let fn_knn_tiled = module.load_function("compute_knn_distances_tiled")?;
        let fn_extract_percentiles = module.load_function("extract_percentiles")?;
        let fn_adaptive_full = module.load_function("adaptive_epsilon_full")?;

        log::info!("GPU k-NN engine initialized");

        Ok(Self {
            context,
            stream,
            module,
            fn_knn_simple,
            fn_knn_tiled,
            fn_extract_percentiles,
            fn_adaptive_full,
        })
    }

    /// Find the PTX file path
    fn find_ptx_path() -> Result<std::path::PathBuf> {
        // Reuse native module discovery: installed environment authority first,
        // then the governed target/ptx and build output locations.
        crate::twin_kernels::find_twin_ptx("knn_cuda.ptx").map(std::path::PathBuf::from)
    }

    /// Compute adaptive epsilon values using GPU k-NN
    ///
    /// This is the main entry point for adaptive epsilon selection.
    /// It samples positions, computes k-NN distances on GPU, and returns
    /// epsilon values at key percentiles.
    ///
    /// # Arguments
    /// * `positions` - Flat array of [x, y, z, x, y, z, ...] coordinates
    /// * `k` - Number of nearest neighbors (typically 4 for DBSCAN min_points)
    /// * `sample_size` - Number of points to sample (typically 1000)
    ///
    /// # Returns
    /// GpuKnnResult with epsilon values and diagnostics
    pub fn compute_adaptive_epsilon(
        &self,
        positions: &[f32],
        k: usize,
        sample_size: usize,
    ) -> Result<GpuKnnResult> {
        let n_points = positions.len() / 3;
        if n_points < k + 1 {
            // Not enough points, return defaults
            return Ok(GpuKnnResult {
                kth_distances: vec![],
                epsilon_values: vec![5.0, 7.0, 10.0, 14.0],
                num_queries: 0,
                gpu_time_ms: 0.0,
            });
        }

        let start = std::time::Instant::now();

        // Upload positions to GPU
        let d_positions: CudaSlice<f32> = self.stream.clone_htod(positions)?;

        // Allocate output buffer for percentiles
        let d_percentiles: CudaSlice<f32> = self.stream.alloc_zeros(4)?;

        // Determine actual sample size
        let actual_samples = sample_size.min(n_points).min(1024);

        // Launch adaptive_epsilon_full kernel (combined k-NN + percentile)
        let block_size = actual_samples.min(256) as u32;
        let grid_size = 1u32; // Single block for combined kernel

        unsafe {
            self.stream
                .launch_builder(&self.fn_adaptive_full)
                .arg(&d_positions)
                .arg(&d_percentiles)
                .arg(&(n_points as u32))
                .arg(&(actual_samples as u32))
                .arg(&(k as u32))
                .arg(&0u32) // seed (not used currently)
                .launch(LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size.max(actual_samples as u32), 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("Failed to launch adaptive_epsilon_full kernel")?;
        }

        // Download results
        let mut percentiles = vec![0.0f32; 4];
        self.stream.memcpy_dtoh(&d_percentiles, &mut percentiles)?;

        let gpu_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Ensure monotonically increasing with minimum spacing
        let mut epsilons = vec![percentiles[0]];
        for &e in &percentiles[1..] {
            if e > epsilons.last().unwrap() + 1.5 {
                epsilons.push(e);
            }
        }

        // Ensure at least 3 scales
        while epsilons.len() < 3 {
            let last = *epsilons.last().unwrap();
            epsilons.push((last * 1.4).min(25.0));
        }

        log::info!(
            "GPU adaptive epsilon: {:?} (k={}, sampled={}/{}, {:.1}ms)",
            epsilons,
            k,
            actual_samples,
            n_points,
            gpu_time_ms
        );

        Ok(GpuKnnResult {
            kth_distances: vec![], // Not stored in combined kernel
            epsilon_values: epsilons,
            num_queries: actual_samples,
            gpu_time_ms,
        })
    }

    /// Compute k-NN distances for specific query points
    ///
    /// Lower-level API for custom k-NN queries.
    ///
    /// # Arguments
    /// * `all_positions` - All spike positions
    /// * `query_indices` - Indices of points to query k-NN for
    /// * `k` - Number of nearest neighbors
    pub fn compute_knn_distances(
        &self,
        all_positions: &[f32],
        query_indices: &[u32],
        k: usize,
    ) -> Result<Vec<f32>> {
        let n_total = all_positions.len() / 3;
        let n_queries = query_indices.len();

        if n_queries == 0 || n_total < k + 1 {
            return Ok(vec![]);
        }

        // Upload to GPU
        let d_positions: CudaSlice<f32> = self.stream.clone_htod(all_positions)?;
        let d_query_indices: CudaSlice<u32> = self.stream.clone_htod(query_indices)?;
        let d_kth_distances: CudaSlice<f32> = self.stream.alloc_zeros(n_queries)?;

        // Launch k-NN kernel
        let block_size = 256u32;
        let grid_size = ((n_queries as u32) + block_size - 1) / block_size;
        let max_compare = n_total.min(5000) as u32;

        unsafe {
            self.stream
                .launch_builder(&self.fn_knn_simple)
                .arg(&d_positions)
                .arg(&d_query_indices)
                .arg(&d_kth_distances)
                .arg(&(n_total as u32))
                .arg(&(n_queries as u32))
                .arg(&(k as u32))
                .arg(&max_compare)
                .launch(LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                })
                .context("Failed to launch compute_knn_distances kernel")?;
        }

        // Download results
        let mut kth_distances = vec![0.0f32; n_queries];
        self.stream
            .memcpy_dtoh(&d_kth_distances, &mut kth_distances)?;

        Ok(kth_distances)
    }
}

/// Compute adaptive epsilon using GPU if available, fallback to CPU
///
/// This is the recommended entry point - it automatically uses GPU
/// if available and falls back to CPU otherwise.
pub fn compute_adaptive_epsilon_auto(
    positions: &[f32],
    k: usize,
    sample_size: usize,
    #[cfg(feature = "gpu")] context: Option<Arc<CudaContext>>,
) -> Result<(Vec<f32>, usize)> {
    #[cfg(feature = "gpu")]
    if let Some(ctx) = context {
        match GpuKnnEngine::new(ctx) {
            Ok(engine) => match engine.compute_adaptive_epsilon(positions, k, sample_size) {
                Ok(result) => {
                    return Ok((result.epsilon_values, result.num_queries));
                }
                Err(e) => {
                    log::warn!("GPU k-NN failed, falling back to CPU: {}", e);
                }
            },
            Err(e) => {
                log::warn!("Could not initialize GPU k-NN, falling back to CPU: {}", e);
            }
        }
    }

    // CPU fallback
    Ok(compute_adaptive_epsilon_cpu(positions, k, sample_size))
}

/// CPU implementation of adaptive epsilon (fallback)
fn compute_adaptive_epsilon_cpu(
    positions: &[f32],
    k: usize,
    sample_size: usize,
) -> (Vec<f32>, usize) {
    let n_points = positions.len() / 3;
    if n_points < k + 1 {
        return (vec![5.0, 7.0, 10.0, 14.0], 0);
    }

    let actual_sample_count = n_points.min(sample_size);
    let sample_indices: Vec<usize> = if n_points <= sample_size {
        (0..n_points).collect()
    } else {
        let step = n_points / sample_size;
        (0..sample_size).map(|i| i * step).collect()
    };

    let mut knn_distances: Vec<f32> = Vec::with_capacity(sample_indices.len());

    for &i in &sample_indices {
        let xi = positions[i * 3];
        let yi = positions[i * 3 + 1];
        let zi = positions[i * 3 + 2];

        let mut distances: Vec<f32> = Vec::with_capacity(n_points.min(5000));
        for j in 0..n_points.min(5000) {
            if i == j {
                continue;
            }
            let xj = positions[j * 3];
            let yj = positions[j * 3 + 1];
            let zj = positions[j * 3 + 2];
            let d = ((xi - xj).powi(2) + (yi - yj).powi(2) + (zi - zj).powi(2)).sqrt();
            distances.push(d);
        }

        if distances.len() >= k {
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            knn_distances.push(distances[k - 1]);
        }
    }

    if knn_distances.is_empty() {
        return (vec![5.0, 7.0, 10.0, 14.0], 0);
    }

    knn_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = knn_distances.len();
    let p25 = knn_distances[n / 4].clamp(3.0, 8.0);
    let p50 = knn_distances[n / 2].clamp(5.0, 12.0);
    let p75 = knn_distances[3 * n / 4].clamp(7.0, 18.0);
    let p90 = knn_distances[9 * n / 10].clamp(10.0, 25.0);

    let mut epsilons = vec![p25];
    for &e in &[p50, p75, p90] {
        if e > epsilons.last().unwrap() + 1.5 {
            epsilons.push(e);
        }
    }

    while epsilons.len() < 3 {
        let last = *epsilons.last().unwrap();
        epsilons.push((last * 1.4).min(25.0));
    }

    (epsilons, actual_sample_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback() {
        // Create some random positions
        let positions: Vec<f32> = (0..3000).map(|i| (i as f32 * 0.1) % 100.0).collect();

        let (epsilons, samples) = compute_adaptive_epsilon_cpu(&positions, 4, 100);

        assert!(epsilons.len() >= 3);
        assert!(samples > 0);
        assert!(epsilons[0] >= 3.0);
        assert!(epsilons[0] <= 8.0);
    }
}
