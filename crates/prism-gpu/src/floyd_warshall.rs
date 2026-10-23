//! Floyd-Warshall All-Pairs Shortest Paths GPU Acceleration
//!
//! ASSUMPTIONS:
//! - Input graph: Adjacency list representation
//! - Output: Dense distance matrix (f32, n×n)
//! - MAX_VERTICES = 100,000 (enforced at runtime)
//! - Memory layout: Row-major for coalesced GPU access
//! - Block size: 32×32 (defined in CUDA kernel)
//! - Requires: CUDA compute capability sm_86 (RTX 3060)
//!
//! ALGORITHM:
//! 1. Convert adjacency list to dense distance matrix on host
//! 2. Copy matrix to GPU device memory (H2D transfer)
//! 3. Execute blocked Floyd-Warshall kernel (3 phases per pivot)
//! 4. Copy result back to host (D2H transfer)
//!
//! PERFORMANCE TARGETS:
//! - DSJC500 (500 vertices): < 1.5 seconds end-to-end
//! - H2D/D2H transfer: < 10% of total time
//! - GPU utilization: > 80%
//!
//! SECURITY:
//! - Validates PTX module loading
//! - Checks for CUDA errors after each operation
//! - Enforces MAX_VERTICES limit
//!
//! REFERENCE: PRISM GPU Plan §4.4 (Phase 4 APSP Kernel)

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, DeviceSlice};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Maximum supported graph size (enforced at runtime)
const MAX_VERTICES: usize = 100_000;

/// Block size for tiled Floyd-Warshall (must match CUDA kernel)
const BLOCK_SIZE: usize = 32;

/// GPU-accelerated Floyd-Warshall APSP solver
///
/// Maintains CUDA device context and compiled PTX module.
/// Thread-safe via Arc<CudaContext>.
pub struct FloydWarshallGpu {
    /// CUDA device handle
    context: Arc<CudaContext>,
    /// CUDA stream for operations
    stream: Arc<CudaStream>,
    /// Phase 1 kernel function
    phase1_func: CudaFunction,
    /// Phase 2 row kernel function
    phase2_row_func: CudaFunction,
    /// Phase 2 col kernel function
    phase2_col_func: CudaFunction,
    /// Phase 3 kernel function
    phase3_func: CudaFunction,
}

impl FloydWarshallGpu {
    /// Creates a new GPU Floyd-Warshall solver
    ///
    /// # Arguments
    /// * `device` - CUDA device handle
    /// * `ptx_path` - Path to compiled PTX module
    ///
    /// # Errors
    /// Returns error if:
    /// - PTX module fails to load
    /// - Required kernel functions not found in module
    /// - CUDA device initialization fails
    ///
    /// # Example
    /// ```rust,no_run
    /// use cudarc::driver::CudaContext;
    /// use prism_gpu::floyd_warshall::FloydWarshallGpu;
    /// use std::sync::Arc;
    ///
    /// let device = CudaContext::new(0).unwrap());
    /// let fw = FloydWarshallGpu::new(Arc::new(device), "kernels/floyd_warshall.ptx").unwrap());
    /// ```
    pub fn new(device: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        log::info!("Loading Floyd-Warshall PTX module from: {}", ptx_path);

        // Load PTX module
        let ptx = Ptx::from_file(ptx_path);
        let module = device
            .load_module(ptx)
            .context("Failed to load Floyd-Warshall PTX module")?;

        // Load kernel functions
        let phase1_func = module
            .load_function("floyd_warshall_phase1")
            .context("Failed to load phase1 kernel function")?;
        let phase2_row_func = module
            .load_function("floyd_warshall_phase2_row")
            .context("Failed to load phase2_row kernel function")?;
        let phase2_col_func = module
            .load_function("floyd_warshall_phase2_col")
            .context("Failed to load phase2_col kernel function")?;
        let phase3_func = module
            .load_function("floyd_warshall_phase3")
            .context("Failed to load phase3 kernel function")?;

        log::info!("Floyd-Warshall GPU module loaded successfully");

        // Get default stream
        let stream = device.default_stream();

        Ok(Self {
            context: device,
            stream,
            phase1_func,
            phase2_row_func,
            phase2_col_func,
            phase3_func,
        })
    }

    /// Computes all-pairs shortest paths on GPU
    ///
    /// # Arguments
    /// * `adjacency` - Adjacency list representation (vertex -> neighbors)
    /// * `num_vertices` - Total number of vertices in graph
    ///
    /// # Returns
    /// Dense distance matrix (n×n) with shortest path distances.
    /// - dist[i][j] = shortest path distance from vertex i to j
    /// - dist[i][i] = 0.0 (distance to self)
    /// - dist[i][j] = f32::INFINITY if no path exists
    ///
    /// # Errors
    /// Returns error if:
    /// - num_vertices exceeds MAX_VERTICES
    /// - GPU memory allocation fails
    /// - Kernel launch fails
    /// - Data transfer fails
    ///
    /// # Performance
    /// - Time complexity: O(n³) operations, but highly parallelized on GPU
    /// - Space complexity: O(n²) for distance matrix
    /// - Target: 500 vertices in < 1.5 seconds on RTX 3060
    ///
    /// # Example
    /// ```rust,no_run
    /// # use prism_gpu::floyd_warshall::FloydWarshallGpu;
    /// # use cudarc::driver::CudaContext;
    /// # use std::sync::Arc;
    /// # let device = CudaContext::new(0).unwrap());
    /// # let fw = FloydWarshallGpu::new(Arc::new(device), "kernels/floyd_warshall.ptx").unwrap());
    /// let adjacency = vec![
    ///     vec![1, 2],  // vertex 0 -> neighbors 1, 2
    ///     vec![2],     // vertex 1 -> neighbor 2
    ///     vec![],      // vertex 2 -> no neighbors
    /// ];
    /// let distances = fw.compute_apsp(&adjacency, 3).unwrap());
    /// assert_eq!(distances[0][2], 1.0); // Direct edge
    /// ```
    pub fn compute_apsp(
        &self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
    ) -> Result<Vec<Vec<f32>>> {
        // Validation
        anyhow::ensure!(num_vertices > 0, "Graph must have at least one vertex");
        anyhow::ensure!(
            num_vertices <= MAX_VERTICES,
            "Graph exceeds MAX_VERTICES limit: {} > {}",
            num_vertices,
            MAX_VERTICES
        );
        anyhow::ensure!(
            adjacency.len() == num_vertices,
            "Adjacency list size mismatch: expected {}, got {}",
            num_vertices,
            adjacency.len()
        );

        log::info!(
            "Computing APSP on GPU for graph with {} vertices",
            num_vertices
        );

        // Step 1: Initialize distance matrix on host
        let n = num_vertices;
        let mut dist_host = vec![f32::INFINITY; n * n];

        // Initialize diagonal to 0 and edges to 1
        for i in 0..n {
            dist_host[i * n + i] = 0.0;
            for &j in &adjacency[i] {
                if j < n {
                    dist_host[i * n + j] = 1.0;
                }
            }
        }

        // Step 2: Allocate device memory and copy data (H2D)
        log::debug!(
            "Copying distance matrix to GPU ({} bytes)",
            dist_host.len() * 4
        );
        let dist_device: CudaSlice<f32> = self
            .stream
            .clone_htod(&dist_host)
            .context("Failed to copy distance matrix to GPU")?;

        // Step 3: Execute blocked Floyd-Warshall algorithm
        let dist_device = self
            .run_blocked_floyd_warshall(dist_device, n)
            .context("Floyd-Warshall kernel execution failed")?;

        // Step 4: Copy result back to host (D2H)
        log::debug!("Copying result back to host");
        let result_flat = self
            .stream
            .clone_dtoh(&dist_device)
            .context("Failed to copy result from GPU")?;

        // Step 5: Convert flat array to 2D matrix
        let mut result = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                result[i][j] = result_flat[i * n + j];
            }
        }

        log::info!("APSP computation completed successfully");
        Ok(result)
    }

    /// Executes blocked Floyd-Warshall algorithm on GPU
    ///
    /// Runs n iterations (one per pivot), each with 3 kernel phases:
    /// - Phase 1: Update diagonal block containing pivot
    /// - Phase 2: Update row/column blocks dependent on pivot
    /// - Phase 3: Update remaining independent blocks
    ///
    /// # Arguments
    /// * `dist_device` - Distance matrix on GPU (n×n, row-major)
    /// * `n` - Number of vertices
    ///
    /// # Errors
    /// Returns error if kernel launch or synchronization fails
    fn run_blocked_floyd_warshall(
        &self,
        dist_device: CudaSlice<f32>,
        n: usize,
    ) -> Result<CudaSlice<f32>> {
        let num_blocks = n.div_ceil(BLOCK_SIZE);

        log::debug!(
            "Launching blocked Floyd-Warshall: {} vertices, {} blocks, block_size={}",
            n,
            num_blocks,
            BLOCK_SIZE
        );

        // Iterate through all pivots
        for k in 0..n {
            let pivot_block = k / BLOCK_SIZE;

            // Phase 1: Update diagonal block
            let cfg1 = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (BLOCK_SIZE as u32, BLOCK_SIZE as u32, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream
                    .launch_builder(&self.phase1_func)
                    .arg(&dist_device)
                    .arg(&(n as i32))
                    .arg(&(k as i32))
                    .arg(&(pivot_block as i32))
                    .launch(cfg1)
                    .with_context(|| format!("Phase 1 kernel launch failed at pivot {}", k))?;
            }

            self.stream
                .synchronize()
                .context("Phase 1 synchronization failed")?;

            // Phase 2: Update row and column blocks
            let cfg2 = LaunchConfig {
                grid_dim: (num_blocks as u32, 1, 1),
                block_dim: (BLOCK_SIZE as u32, BLOCK_SIZE as u32, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream
                    .launch_builder(&self.phase2_row_func)
                    .arg(&dist_device)
                    .arg(&(n as i32))
                    .arg(&(k as i32))
                    .arg(&(pivot_block as i32))
                    .launch(cfg2)
                    .with_context(|| format!("Phase 2 row kernel launch failed at pivot {}", k))?;
            }

            unsafe {
                self.stream
                    .launch_builder(&self.phase2_col_func)
                    .arg(&dist_device)
                    .arg(&(n as i32))
                    .arg(&(k as i32))
                    .arg(&(pivot_block as i32))
                    .launch(cfg2)
                    .with_context(|| format!("Phase 2 col kernel launch failed at pivot {}", k))?;
            }

            self.stream
                .synchronize()
                .context("Phase 2 synchronization failed")?;

            // Phase 3: Update remaining blocks
            let cfg3 = LaunchConfig {
                grid_dim: (num_blocks as u32, num_blocks as u32, 1),
                block_dim: (BLOCK_SIZE as u32, BLOCK_SIZE as u32, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream
                    .launch_builder(&self.phase3_func)
                    .arg(&dist_device)
                    .arg(&(n as i32))
                    .arg(&(k as i32))
                    .arg(&(pivot_block as i32))
                    .launch(cfg3)
                    .with_context(|| format!("Phase 3 kernel launch failed at pivot {}", k))?;
            }

            self.stream
                .synchronize()
                .context("Phase 3 synchronization failed")?;

            // Log progress every 10% of pivots
            if k % (n / 10).max(1) == 0 {
                log::debug!("Floyd-Warshall progress: {}/{} pivots", k + 1, n);
            }
        }

        log::debug!("Blocked Floyd-Warshall completed all {} pivots", n);
        Ok(dist_device)
    }

    /// Returns reference to underlying CUDA device
    pub fn device(&self) -> &Arc<CudaContext> {
        &self.context
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a simple test graph for unit testing
    fn create_test_graph() -> (Vec<Vec<usize>>, usize) {
        // Graph: 0 -> 1 -> 2
        //        |         ^
        //        +----3----+
        let adjacency = vec![
            vec![1, 3], // 0 -> 1, 3
            vec![2],    // 1 -> 2
            vec![],     // 2 -> (none)
            vec![2],    // 3 -> 2
        ];
        (adjacency, 4)
    }

    #[test]
    #[ignore] // Requires GPU hardware
    fn test_floyd_warshall_small_graph() {
        env_logger::builder().is_test(true).try_init().ok();

        let device = CudaContext::new(0).expect("CUDA device not available");
        let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
            .expect("Failed to create FloydWarshallGpu");

        let (adjacency, num_vertices) = create_test_graph();
        let distances = fw
            .compute_apsp(&adjacency, num_vertices)
            .expect("APSP computation failed");

        // Verify distances
        assert_eq!(distances[0][0], 0.0);
        assert_eq!(distances[0][1], 1.0);
        assert_eq!(distances[0][2], 2.0);
        assert_eq!(distances[0][3], 1.0);
        assert_eq!(distances[1][2], 1.0);
        assert_eq!(distances[3][2], 1.0);

        // Verify no path from 2 to others
        assert_eq!(distances[2][0], f32::INFINITY);
        assert_eq!(distances[2][1], f32::INFINITY);
        assert_eq!(distances[2][3], f32::INFINITY);
    }

    #[test]
    fn test_validation_max_vertices() {
        let device = CudaContext::new(0).expect("CUDA device not available");
        let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
            .expect("Failed to create FloydWarshallGpu");

        let large_adjacency = vec![vec![]; MAX_VERTICES + 1];
        let result = fw.compute_apsp(&large_adjacency, MAX_VERTICES + 1);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("MAX_VERTICES"));
    }

    #[test]
    fn test_validation_empty_graph() {
        let device = CudaContext::new(0).expect("CUDA device not available");
        let fw = FloydWarshallGpu::new(Arc::new(device), "target/ptx/floyd_warshall.ptx")
            .expect("Failed to create FloydWarshallGpu");

        let result = fw.compute_apsp(&[], 0);
        assert!(result.is_err());
    }
}
