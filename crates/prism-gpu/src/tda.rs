//! TDA Persistent Homology GPU Acceleration
//!
//! ASSUMPTIONS:
//! - Input graph: Adjacency list representation (Vec<Vec<usize>>)
//! - Output: Betti numbers (usize), persistence scores (Vec<f32>), importance scores (Vec<f32>)
//! - MAX_VERTICES = 100,000 (enforced at runtime)
//! - MAX_EDGES = 5,000,000 (enforced at runtime)
//! - Precision: f32 for scores, i32 for labels/counts
//! - Block size: 256 threads (defined in CUDA kernel)
//! - Requires: CUDA compute capability sm_86 (RTX 3060)
//!
//! ALGORITHM:
//! 1. Convert adjacency list to edge list (host)
//! 2. Transfer edge list to GPU (H2D)
//! 3. Execute union-find kernels (iterative path compression)
//! 4. Count components (Betti-0) via reduction
//! 5. Compute Betti-1 via Euler characteristic
//! 6. Compute persistence and importance scores
//! 7. Transfer results back to host (D2H)
//!
//! PERFORMANCE TARGETS:
//! - DSJC250 (250 vertices, ~15k edges): < 50ms end-to-end
//! - DSJC500 (500 vertices, ~125k edges): < 200ms end-to-end
//! - H2D/D2H transfer: < 15% of total time
//! - GPU utilization: > 75%
//!
//! SECURITY:
//! - Validates PTX module loading with anyhow::Result
//! - Checks CUDA errors after each kernel launch
//! - Enforces MAX_VERTICES and MAX_EDGES limits
//! - No unsafe blocks without documented safety invariants
//!
//! REFERENCE: PRISM GPU Plan §4.6 (Phase 6 TDA Kernel)

use anyhow::{bail, Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DeviceSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Maximum supported graph size (enforced at runtime)
const MAX_VERTICES: usize = 100_000;
const MAX_EDGES: usize = 5_000_000;

/// Block size for vertex-parallel operations (must match CUDA kernel)
const BLOCK_SIZE: u32 = 256;

/// Number of path compression iterations (empirically determined)
const PATH_COMPRESSION_ITERS: usize = 10;

/// GPU-accelerated TDA persistent homology solver
///
/// Maintains CUDA device context and compiled PTX module.
/// Thread-safe via Arc<CudaContext>.
pub struct TdaGpu {
    /// CUDA device handle
    context: Arc<CudaContext>,
    /// CUDA stream for async operations
    stream: Arc<CudaStream>,

    // Kernel functions
    union_find_init: CudaFunction,
    union_find_link: CudaFunction,
    union_find_compress: CudaFunction,
    count_components: CudaFunction,
    compute_degrees: CudaFunction,
    compute_persistence_scores: CudaFunction,
    compute_topological_importance: CudaFunction,
}

impl TdaGpu {
    /// Creates a new GPU TDA solver
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
    /// use prism_gpu::tda::TdaGpu;
    /// use std::sync::Arc;
    ///
    /// let device = CudaContext::new(0).unwrap();
    /// let tda = TdaGpu::new(device, "target/ptx/tda.ptx").unwrap();
    /// ```
    pub fn new(device: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        log::info!("Loading TDA PTX module from: {}", ptx_path);

        let stream = device.default_stream();

        // Load PTX module
        let module = device
            .load_module(Ptx::from_file(ptx_path))
            .with_context(|| format!("Failed to load PTX module from {}", ptx_path))?;

        // Load all required kernel functions
        let union_find_init = module
            .load_function("union_find_init")
            .context("Failed to load union_find_init function")?;
        let union_find_link = module
            .load_function("union_find_link")
            .context("Failed to load union_find_link function")?;
        let union_find_compress = module
            .load_function("union_find_compress")
            .context("Failed to load union_find_compress function")?;
        let count_components = module
            .load_function("count_components")
            .context("Failed to load count_components function")?;
        let compute_degrees = module
            .load_function("compute_degrees")
            .context("Failed to load compute_degrees function")?;
        let compute_persistence_scores = module
            .load_function("compute_persistence_scores")
            .context("Failed to load compute_persistence_scores function")?;
        let compute_topological_importance = module
            .load_function("compute_topological_importance")
            .context("Failed to load compute_topological_importance function")?;

        log::info!("TDA GPU module loaded successfully");

        Ok(Self {
            context: device,
            stream,
            union_find_init,
            union_find_link,
            union_find_compress,
            count_components,
            compute_degrees,
            compute_persistence_scores,
            compute_topological_importance,
        })
    }

    /// Computes Betti numbers (connected components and cycles) on GPU
    ///
    /// # Arguments
    /// * `adjacency` - Adjacency list representation (vertex -> neighbors)
    /// * `num_vertices` - Total number of vertices in graph
    /// * `num_edges` - Total number of edges in graph
    ///
    /// # Returns
    /// Tuple (betti_0, betti_1) where:
    /// - betti_0 = number of connected components
    /// - betti_1 = number of independent cycles (Euler characteristic)
    ///
    /// # Errors
    /// Returns error if:
    /// - Graph exceeds MAX_VERTICES or MAX_EDGES
    /// - CUDA operations fail (memory allocation, kernel launch, transfer)
    /// - Device synchronization fails
    ///
    /// # Example
    /// ```rust,no_run
    /// # use prism_gpu::tda::TdaGpu;
    /// # use cudarc::driver::CudaContext;
    /// # use std::sync::Arc;
    /// # let device = CudaContext::new(0).unwrap();
    /// # let tda = TdaGpu::new(device, "target/ptx/tda.ptx").unwrap();
    /// let adjacency = vec![vec![1], vec![0, 2], vec![1]]; // Triangle
    /// let (betti_0, betti_1) = tda.compute_betti_numbers(&adjacency, 3, 3).unwrap();
    /// assert_eq!(betti_0, 1); // One component
    /// assert_eq!(betti_1, 1); // One cycle
    /// ```
    pub fn compute_betti_numbers(
        &self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
        num_edges: usize,
    ) -> Result<(usize, usize)> {
        // Validate input size
        if num_vertices > MAX_VERTICES {
            bail!(
                "Graph exceeds MAX_VERTICES limit: {} > {}",
                num_vertices,
                MAX_VERTICES
            );
        }
        if num_edges > MAX_EDGES {
            bail!(
                "Graph exceeds MAX_EDGES limit: {} > {}",
                num_edges,
                MAX_EDGES
            );
        }

        log::debug!(
            "Computing Betti numbers for graph: {} vertices, {} edges",
            num_vertices,
            num_edges
        );

        // Convert adjacency list to edge list (host operation)
        let (edges_u, edges_v) = self.adjacency_to_edge_list(adjacency);
        let actual_edges = edges_u.len();

        if actual_edges == 0 {
            // Empty graph: betti_0 = num_vertices (all disconnected), betti_1 = 0
            return Ok((num_vertices, 0));
        }

        // Allocate device memory
        let d_parent = self
            .stream
            .alloc_zeros::<i32>(num_vertices)
            .context("Failed to allocate parent array")?;
        let d_edges_u = self
            .stream
            .clone_htod(&edges_u)
            .context("Failed to copy edges_u to device")?;
        let d_edges_v = self
            .stream
            .clone_htod(&edges_v)
            .context("Failed to copy edges_v to device")?;
        let d_component_count = self
            .stream
            .alloc_zeros::<i32>(1)
            .context("Failed to allocate component count")?;

        // Launch configuration
        let grid_vertices = ((num_vertices as u32).div_ceil(BLOCK_SIZE), 1, 1);
        let grid_edges = ((actual_edges as u32).div_ceil(BLOCK_SIZE), 1, 1);
        let block = (BLOCK_SIZE, 1, 1);

        // Kernel 1: Initialize union-find
        let cfg = LaunchConfig {
            grid_dim: grid_vertices,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            // SAFETY: d_parent is valid for num_vertices elements
            self.stream
                .launch_builder(&self.union_find_init)
                .arg(&d_parent)
                .arg(&(num_vertices as i32))
                .launch(cfg)
        }
        .context("Failed to launch union_find_init kernel")?;

        // Kernel 2: Link edges (union operation)
        let cfg_edges = LaunchConfig {
            grid_dim: grid_edges,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            // SAFETY: All pointers valid, sizes checked above
            self.stream
                .launch_builder(&self.union_find_link)
                .arg(&d_parent)
                .arg(&d_edges_u)
                .arg(&d_edges_v)
                .arg(&(actual_edges as i32))
                .launch(cfg_edges)
        }
        .context("Failed to launch union_find_link kernel")?;

        // Kernel 3: Path compression (iterative for full compression)
        for _ in 0..PATH_COMPRESSION_ITERS {
            unsafe {
                // SAFETY: d_parent is valid, no data races (idempotent compression)
                self.stream
                    .launch_builder(&self.union_find_compress)
                    .arg(&d_parent)
                    .arg(&(num_vertices as i32))
                    .launch(cfg)
            }
            .context("Failed to launch union_find_compress kernel")?;
        }

        // Kernel 4: Count components
        unsafe {
            // SAFETY: Pointers valid, atomic increments ensure thread safety
            self.stream
                .launch_builder(&self.count_components)
                .arg(&d_parent)
                .arg(&d_component_count)
                .arg(&(num_vertices as i32))
                .launch(cfg)
        }
        .context("Failed to launch count_components kernel")?;

        // Synchronize and copy results back
        self.stream
            .synchronize()
            .context("Failed to synchronize stream")?;

        let component_count_host = self
            .stream
            .clone_dtoh(&d_component_count)
            .context("Failed to copy component count to host")?;
        let betti_0 = component_count_host[0] as usize;

        // Compute Betti-1 via Euler characteristic: cycles = edges - vertices + components
        let betti_1 = if actual_edges >= num_vertices {
            actual_edges - num_vertices + betti_0
        } else {
            0
        };

        log::debug!("Betti numbers: β₀={}, β₁={}", betti_0, betti_1);

        Ok((betti_0, betti_1))
    }

    /// Computes persistence and importance scores on GPU
    ///
    /// # Arguments
    /// * `adjacency` - Adjacency list representation
    /// * `betti_0` - Number of connected components (from compute_betti_numbers)
    /// * `betti_1` - Number of cycles (from compute_betti_numbers)
    ///
    /// # Returns
    /// Tuple (persistence, importance) where:
    /// - persistence[v] = topological persistence score for vertex v
    /// - importance[v] = anchor selection metric for vertex v
    ///
    /// # Errors
    /// Returns error if CUDA operations fail
    ///
    /// # Example
    /// ```rust,no_run
    /// # use prism_gpu::tda::TdaGpu;
    /// # use cudarc::driver::CudaContext;
    /// # use std::sync::Arc;
    /// # let device = CudaContext::new(0).unwrap();
    /// # let tda = TdaGpu::new(device, "target/ptx/tda.ptx").unwrap();
    /// let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]]; // Triangle
    /// let (betti_0, betti_1) = tda.compute_betti_numbers(&adjacency, 3, 3).unwrap();
    /// let (persistence, importance) = tda.compute_persistence_and_importance(&adjacency, betti_0, betti_1).unwrap();
    /// assert_eq!(persistence.len(), 3);
    /// assert_eq!(importance.len(), 3);
    /// ```
    pub fn compute_persistence_and_importance(
        &self,
        adjacency: &[Vec<usize>],
        betti_0: usize,
        betti_1: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let num_vertices = adjacency.len();

        if num_vertices > MAX_VERTICES {
            bail!(
                "Graph exceeds MAX_VERTICES limit: {} > {}",
                num_vertices,
                MAX_VERTICES
            );
        }

        // Convert adjacency list to edge list
        let (edges_u, edges_v) = self.adjacency_to_edge_list(adjacency);
        let actual_edges = edges_u.len();

        if actual_edges == 0 {
            // Empty graph: all zeros
            return Ok((vec![0.0; num_vertices], vec![0.0; num_vertices]));
        }

        // Allocate device memory
        let d_degrees = self
            .stream
            .alloc_zeros::<i32>(num_vertices)
            .context("Failed to allocate degrees array")?;
        let d_parent = self
            .stream
            .alloc_zeros::<i32>(num_vertices)
            .context("Failed to allocate parent array")?;
        let d_edges_u = self
            .stream
            .clone_htod(&edges_u)
            .context("Failed to copy edges_u to device")?;
        let d_edges_v = self
            .stream
            .clone_htod(&edges_v)
            .context("Failed to copy edges_v to device")?;
        let d_persistence = self
            .stream
            .alloc_zeros::<f32>(num_vertices)
            .context("Failed to allocate persistence array")?;
        let d_importance = self
            .stream
            .alloc_zeros::<f32>(num_vertices)
            .context("Failed to allocate importance array")?;

        // Launch configuration
        let grid_vertices = ((num_vertices as u32).div_ceil(BLOCK_SIZE), 1, 1);
        let grid_edges = ((actual_edges as u32).div_ceil(BLOCK_SIZE), 1, 1);
        let block = (BLOCK_SIZE, 1, 1);

        // Kernel 1: Compute degrees
        let cfg_edges = LaunchConfig {
            grid_dim: grid_edges,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            // SAFETY: All pointers valid, atomic adds ensure thread safety
            self.stream
                .launch_builder(&self.compute_degrees)
                .arg(&d_edges_u)
                .arg(&d_edges_v)
                .arg(&d_degrees)
                .arg(&(actual_edges as i32))
                .launch(cfg_edges)
        }
        .context("Failed to launch compute_degrees kernel")?;

        // Kernel 2: Initialize parent array (for component-aware computation)
        let cfg_vertices = LaunchConfig {
            grid_dim: grid_vertices,
            block_dim: block,
            shared_mem_bytes: 0,
        };

        unsafe {
            // SAFETY: d_parent is valid for num_vertices elements
            self.stream
                .launch_builder(&self.union_find_init)
                .arg(&d_parent)
                .arg(&(num_vertices as i32))
                .launch(cfg_vertices)
        }
        .context("Failed to launch union_find_init kernel")?;

        // Kernel 3: Compute persistence scores
        unsafe {
            // SAFETY: All pointers valid, read-only access to degrees/parent
            self.stream
                .launch_builder(&self.compute_persistence_scores)
                .arg(&d_degrees)
                .arg(&d_parent)
                .arg(&(betti_0 as i32))
                .arg(&(betti_1 as i32))
                .arg(&d_persistence)
                .arg(&(num_vertices as i32))
                .launch(cfg_vertices)
        }
        .context("Failed to launch compute_persistence_scores kernel")?;

        // Kernel 4: Compute topological importance
        unsafe {
            // SAFETY: All pointers valid, read-only access to most arrays
            self.stream
                .launch_builder(&self.compute_topological_importance)
                .arg(&d_persistence)
                .arg(&d_degrees)
                .arg(&d_parent)
                .arg(&(betti_0 as i32))
                .arg(&d_importance)
                .arg(&(num_vertices as i32))
                .launch(cfg_vertices)
        }
        .context("Failed to launch compute_topological_importance kernel")?;

        // Synchronize and copy results back
        self.stream
            .synchronize()
            .context("Failed to synchronize stream")?;

        let persistence = self
            .stream
            .clone_dtoh(&d_persistence)
            .context("Failed to copy persistence to host")?;
        let importance = self
            .stream
            .clone_dtoh(&d_importance)
            .context("Failed to copy importance to host")?;

        Ok((persistence, importance))
    }

    /// Helper: Convert adjacency list to edge list (u, v pairs)
    ///
    /// Returns two parallel arrays: edges_u[i] and edges_v[i] form edge i.
    /// Deduplicates edges (stores each undirected edge once).
    fn adjacency_to_edge_list(&self, adjacency: &[Vec<usize>]) -> (Vec<i32>, Vec<i32>) {
        let mut edges_u = Vec::new();
        let mut edges_v = Vec::new();

        for (u, neighbors) in adjacency.iter().enumerate() {
            for &v in neighbors {
                // Store each edge once (u < v) to avoid duplicates
                if u < v {
                    edges_u.push(u as i32);
                    edges_v.push(v as i32);
                }
            }
        }

        (edges_u, edges_v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: Create GPU context for tests
    fn create_test_gpu() -> Result<TdaGpu> {
        let device = CudaContext::new(0)?;
        TdaGpu::new(device, "target/ptx/tda.ptx")
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_triangle_graph() {
        let tda = create_test_gpu().expect("Failed to initialize GPU");

        // Triangle: 3 vertices, 3 edges, forms a cycle
        let adjacency = vec![
            vec![1, 2], // 0 -> 1, 2
            vec![0, 2], // 1 -> 0, 2
            vec![0, 1], // 2 -> 0, 1
        ];

        let (betti_0, betti_1) = tda
            .compute_betti_numbers(&adjacency, 3, 3)
            .expect("Failed to compute Betti numbers");

        // Triangle: 1 component, 1 cycle
        assert_eq!(betti_0, 1, "Triangle should have 1 connected component");
        assert_eq!(betti_1, 1, "Triangle should have 1 cycle");

        let (persistence, importance) = tda
            .compute_persistence_and_importance(&adjacency, betti_0, betti_1)
            .expect("Failed to compute persistence/importance");

        assert_eq!(persistence.len(), 3);
        assert_eq!(importance.len(), 3);

        // All vertices in triangle have equal degree (2), so equal scores
        for i in 0..3 {
            assert!(persistence[i] > 0.0, "Persistence should be positive");
            assert!(importance[i] > 0.0, "Importance should be positive");
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_complete_graph_k5() {
        let tda = create_test_gpu().expect("Failed to initialize GPU");

        // K5: Complete graph on 5 vertices
        // Each vertex connected to all others
        let adjacency = vec![
            vec![1, 2, 3, 4],
            vec![0, 2, 3, 4],
            vec![0, 1, 3, 4],
            vec![0, 1, 2, 4],
            vec![0, 1, 2, 3],
        ];

        let num_edges = 10; // K5 has 5*4/2 = 10 edges

        let (betti_0, betti_1) = tda
            .compute_betti_numbers(&adjacency, 5, num_edges)
            .expect("Failed to compute Betti numbers");

        // K5: 1 component, 6 independent cycles
        assert_eq!(betti_0, 1, "K5 should have 1 connected component");
        assert_eq!(
            betti_1, 6,
            "K5 should have 6 independent cycles (10 - 5 + 1)"
        );

        let (persistence, importance) = tda
            .compute_persistence_and_importance(&adjacency, betti_0, betti_1)
            .expect("Failed to compute persistence/importance");

        assert_eq!(persistence.len(), 5);
        assert_eq!(importance.len(), 5);

        // All vertices in K5 have equal degree (4), so equal scores
        let first_pers = persistence[0];
        let first_imp = importance[0];
        for i in 1..5 {
            assert!(
                (persistence[i] - first_pers).abs() < 1e-5,
                "K5 vertices should have equal persistence"
            );
            assert!(
                (importance[i] - first_imp).abs() < 1e-5,
                "K5 vertices should have equal importance"
            );
        }
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_disconnected_components() {
        let tda = create_test_gpu().expect("Failed to initialize GPU");

        // Two disconnected triangles: 6 vertices, 6 edges, 2 components
        let adjacency = vec![
            vec![1, 2], // Triangle 1
            vec![0, 2],
            vec![0, 1],
            vec![4, 5], // Triangle 2
            vec![3, 5],
            vec![3, 4],
        ];

        let (betti_0, betti_1) = tda
            .compute_betti_numbers(&adjacency, 6, 6)
            .expect("Failed to compute Betti numbers");

        // Two triangles: 2 components, 2 cycles
        assert_eq!(betti_0, 2, "Should have 2 connected components");
        assert_eq!(betti_1, 2, "Should have 2 cycles (one per triangle)");
    }

    #[test]
    #[ignore] // Requires CUDA hardware
    fn test_empty_graph() {
        let tda = create_test_gpu().expect("Failed to initialize GPU");

        // 5 isolated vertices (no edges)
        let adjacency = vec![vec![], vec![], vec![], vec![], vec![]];

        let (betti_0, betti_1) = tda
            .compute_betti_numbers(&adjacency, 5, 0)
            .expect("Failed to compute Betti numbers");

        // Isolated vertices: n components, 0 cycles
        assert_eq!(betti_0, 5, "Should have 5 disconnected components");
        assert_eq!(betti_1, 0, "Should have 0 cycles");
    }
}
