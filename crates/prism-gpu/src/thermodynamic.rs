//! GPU-accelerated thermodynamic simulated annealing for graph coloring.
//!
//! Implements parallel tempering with multiple temperature replicas running
//! concurrently on GPU. Uses CUDA kernel from kernels/thermodynamic.cu.
//!
//! Implements §4.2 (Phase 2: Thermodynamic) of the PRISM GPU Plan.

use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DeviceSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ThermodynamicError {
    #[error("CUDA error: {0}")]
    Cuda(#[from] cudarc::driver::DriverError),

    #[error("PTX load failed: {0}")]
    PTXLoadFailed(String),

    #[error("Kernel not found: {0}")]
    KernelNotFound(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// GPU-accelerated thermodynamic optimizer.
///
/// Runs simulated annealing with parallel tempering across multiple temperature
/// schedules simultaneously on GPU.
///
/// # Example
///
/// ```ignore
/// use prism_gpu::ThermodynamicGpu;
/// use cudarc::driver::CudaContext;
/// use std::sync::Arc;
///
/// let device = Arc::new(CudaContext::new(0)?);
/// let thermo = ThermodynamicGpu::new(device, "target/ptx/thermodynamic.ptx")?;
///
/// let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]]; // Triangle
/// let initial = vec![0, 1, 2];
///
/// let result = thermo.run(&adjacency, 3, &initial, 4, 1000, 0.01, 5.0)?;
/// ```
pub struct ThermodynamicGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    kernel_anneal: CudaFunction,
    kernel_swap_replicas: CudaFunction,
}

impl ThermodynamicGpu {
    /// Initialize GPU thermodynamic module from PTX.
    ///
    /// # Arguments
    /// * `device` - CUDA device handle
    /// * `ptx_path` - Path to thermodynamic.ptx module
    ///
    /// # Errors
    /// Returns error if PTX loading fails or kernels are not found.
    pub fn new(device: Arc<CudaContext>, ptx_path: &str) -> Result<Self, ThermodynamicError> {
        log::info!("Loading Thermodynamic PTX module from: {}", ptx_path);

        let stream = device.default_stream();

        let module = device.load_module(Ptx::from_file(ptx_path))?;

        let kernel_anneal = module.load_function("parallel_tempering_step")?;

        let kernel_swap_replicas = module.load_function("replica_swap")?;

        log::info!("Thermodynamic GPU module loaded successfully");

        Ok(Self {
            context: device,
            stream,
            kernel_anneal,
            kernel_swap_replicas,
        })
    }

    /// Run parallel tempering simulated annealing on GPU.
    ///
    /// # Arguments
    /// * `adjacency` - Adjacency list representation of graph
    /// * `num_vertices` - Number of vertices
    /// * `initial_colors` - Starting coloring (warmstart)
    /// * `num_replicas` - Number of parallel temperature replicas (default: 8)
    /// * `iterations` - Number of annealing iterations (default: 10000)
    /// * `temp_min` - Minimum temperature (default: 0.01)
    /// * `temp_max` - Maximum temperature (default: 10.0)
    /// * `stress_scalar` - Geometry stress scalar (0.0-1.0) for adaptive temperature
    ///
    /// # Returns
    /// Best coloring found (color assignment per vertex)
    ///
    /// # Errors
    /// Returns error if GPU operations fail or parameters are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
        initial_colors: &[usize],
        num_replicas: usize,
        iterations: usize,
        temp_min: f32,
        temp_max: f32,
        stress_scalar: f32,
    ) -> Result<Vec<usize>, ThermodynamicError> {
        // Validate inputs
        if num_vertices == 0 {
            return Err(ThermodynamicError::InvalidParameter(
                "num_vertices must be > 0".to_string(),
            ));
        }

        if initial_colors.len() != num_vertices {
            return Err(ThermodynamicError::InvalidParameter(format!(
                "initial_colors length {} != num_vertices {}",
                initial_colors.len(),
                num_vertices
            )));
        }

        if num_replicas == 0 {
            return Err(ThermodynamicError::InvalidParameter(
                "num_replicas must be > 0".to_string(),
            ));
        }

        // Count unique colors in initial solution to detect warmstart
        let mut unique_colors = std::collections::HashSet::new();
        for &color in initial_colors {
            unique_colors.insert(color);
        }
        let num_initial_colors = unique_colors.len();
        let is_warmstart = num_initial_colors < num_vertices / 2; // Heuristic: good coloring uses fewer colors

        // Apply stress-adaptive temperature scaling
        // Higher stress -> increase temperature range for more exploration
        let stress_multiplier = 1.0 + (stress_scalar * 0.5); // 1.0 to 1.5x

        // WARMSTART-AWARE TEMPERATURE SCALING
        // If we have a good warmstart (e.g., 23 colors), don't destroy it with high temp
        let warmstart_cap = if is_warmstart {
            // Cap temperature based on quality of warmstart
            // Better warmstart (fewer colors) = lower max temp
            let quality_ratio = num_initial_colors as f32 / num_vertices as f32;
            1.5 + quality_ratio * 3.0 // Range: 1.5 to 4.5 for good warmstarts
        } else {
            temp_max // Full temperature range for random init
        };

        let temp_max_adjusted = (temp_max * stress_multiplier).min(warmstart_cap);

        if is_warmstart {
            log::info!(
                "Warmstart detected: {} colors. Capping max temperature at {:.2} (was {:.2})",
                num_initial_colors,
                temp_max_adjusted,
                temp_max * stress_multiplier
            );
        }

        log::info!(
            "Running GPU thermodynamic annealing: {} vertices, {} replicas, {} iterations, stress={:.3}, temp_max_adjusted={:.2}",
            num_vertices,
            num_replicas,
            iterations,
            stress_scalar,
            temp_max_adjusted
        );

        log::info!("Using default CUDA stream for replica execution (cudarc 0.18.1)");

        // Convert adjacency list to CSR format
        let (row_ptr, col_idx) = self.adjacency_to_csr(adjacency, num_vertices);
        let num_edges = col_idx.len();

        // Compute independent sets for race-free parallel updates
        let (independent_sets, num_independent_sets) =
            self.compute_independent_sets(adjacency, num_vertices);

        // Allocate device memory for graph structure
        let d_row_ptr = self.stream.clone_htod(&row_ptr)?;
        let d_col_idx = self.stream.clone_htod(&col_idx)?;
        let d_independent_sets = self.stream.clone_htod(&independent_sets)?;

        // Initialize replicas with temperature schedule (geometric progression)
        // Temperature will be dynamically adjusted for simulated annealing
        // Use stress-adjusted max temperature for initial distribution
        let mut temperatures = (0..num_replicas)
            .map(|i| {
                let t = i as f32 / (num_replicas - 1).max(1) as f32;
                temp_min * (temp_max_adjusted / temp_min).powf(t)
            })
            .collect::<Vec<f32>>();

        log::info!(
            "Initial temperature schedule: [{:.3}, {:.3}, ..., {:.3}]",
            temperatures[0],
            temperatures[num_replicas / 2],
            temperatures[num_replicas - 1]
        );
        log::info!(
            "Temperature sweeping enabled: {} → {} over {} iterations",
            temp_max,
            temp_min,
            iterations
        );

        let mut d_temperatures = self.stream.clone_htod(&temperatures)?;

        // Initialize all replicas with the initial coloring
        let mut replica_colors = vec![0u32; num_vertices * num_replicas];
        for r in 0..num_replicas {
            let offset = r * num_vertices;
            for v in 0..num_vertices {
                replica_colors[offset + v] = initial_colors[v] as u32;
            }
        }
        let d_replica_colors = self.stream.clone_htod(&replica_colors)?;

        // Allocate conflict counts
        let conflicts = vec![0u32; num_replicas];
        let d_conflicts = self.stream.clone_htod(&conflicts)?;

        // Launch configuration
        let total_threads = num_vertices * num_replicas;
        let block_size = 256;
        let grid_size = total_threads.div_ceil(block_size) as u32;

        log::debug!(
            "Launch config: grid={}, block={}, total_threads={}",
            grid_size,
            block_size,
            total_threads
        );

        // Run parallel tempering iterations with temperature sweeping
        for iter in 0..iterations {
            // Dynamic temperature cooling (simulated annealing schedule)
            // Progress from exploration (high temp) to exploitation (low temp)
            let cooling_progress = iter as f32 / iterations.max(1) as f32;
            let cooling_factor = 1.0 - cooling_progress; // 1.0 → 0.0

            // Update temperature schedule every 100 iterations for efficiency
            if iter % 100 == 0 {
                for (r, temp) in temperatures.iter_mut().enumerate().take(num_replicas) {
                    let base_temp_ratio = r as f32 / (num_replicas - 1).max(1) as f32;
                    // Interpolate between current cooling state and min temp
                    // Use stress-adjusted max throughout cooling schedule
                    let current_max = temp_min + (temp_max_adjusted - temp_min) * cooling_factor;
                    *temp = temp_min * (current_max / temp_min).powf(base_temp_ratio);
                }

                // Update GPU temperature array
                d_temperatures = self.stream.clone_htod(&temperatures)?;

                if iter % 1000 == 0 && iter > 0 {
                    log::debug!(
                        "Iteration {}/{}: Temp range [{:.3}, {:.3}], cooling_factor={:.3}",
                        iter,
                        iterations,
                        temperatures[0],
                        temperatures[num_replicas - 1],
                        cooling_factor
                    );
                }
            }

            // Process each independent set sequentially to avoid race conditions
            for set_id in 0..num_independent_sets {
                // Annealing step for vertices in this independent set only
                let config = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size as u32, 1, 1),
                    shared_mem_bytes: 0,
                };

                // Launch kernel on stream (processes all replicas in one kernel)
                unsafe {
                    self.stream
                        .launch_builder(&self.kernel_anneal)
                        .arg(&d_row_ptr)
                        .arg(&d_col_idx)
                        .arg(&d_replica_colors)
                        .arg(&d_temperatures)
                        .arg(&d_conflicts)
                        .arg(&d_independent_sets)
                        .arg(&(num_vertices as u32))
                        .arg(&(num_edges as u32))
                        .arg(&(num_replicas as u32))
                        .arg(&(iter as u32))
                        .arg(&(set_id as u32))
                        .launch(config)?
                };

                // Synchronize after each independent set to prevent races
                self.stream.synchronize()?;
            }

            // Replica swap every 100 iterations
            if iter % 100 == 0 && iter > 0 {
                let swap_config = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: ((num_replicas / 2).max(1) as u32, 1, 1),
                    shared_mem_bytes: 0,
                };

                // Launch replica swap kernel on stream
                unsafe {
                    self.stream
                        .launch_builder(&self.kernel_swap_replicas)
                        .arg(&d_replica_colors)
                        .arg(&d_temperatures)
                        .arg(&d_conflicts)
                        .arg(&(num_vertices as u32))
                        .arg(&(num_replicas as u32))
                        .launch(swap_config)?
                };

                // Synchronize after replica swap
                self.stream.synchronize()?;
            }

            if iter % 1000 == 0 && iter > 0 {
                log::debug!(
                    "Thermodynamic annealing progress: {}/{} iterations",
                    iter,
                    iterations
                );
            }
        }

        // Copy results back
        replica_colors = self.stream.clone_dtoh(&d_replica_colors)?;
        let conflicts_host = self.stream.clone_dtoh(&d_conflicts)?;

        // Find best replica (lowest conflicts)
        let best_replica = conflicts_host
            .iter()
            .enumerate()
            .min_by_key(|(_, &c)| c)
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let best_colors = replica_colors
            [best_replica * num_vertices..(best_replica + 1) * num_vertices]
            .iter()
            .map(|&c| c as usize)
            .collect::<Vec<_>>();

        log::info!(
            "Thermodynamic annealing completed: best replica {} with {} conflicts",
            best_replica,
            conflicts_host[best_replica]
        );

        Ok(best_colors)
    }

    /// Compute independent sets using greedy graph coloring.
    ///
    /// Returns (independent_sets, num_sets) where:
    /// - independent_sets[v] = which independent set vertex v belongs to
    /// - num_sets = total number of independent sets
    ///
    /// Vertices in the same independent set have no edges between them,
    /// so they can be safely updated in parallel without race conditions.
    fn compute_independent_sets(
        &self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
    ) -> (Vec<u32>, usize) {
        let mut vertex_set = vec![u32::MAX; num_vertices]; // u32::MAX means unassigned
        let mut num_sets = 0;

        // Greedy graph coloring to find independent sets
        for v in 0..num_vertices {
            if vertex_set[v] == u32::MAX {
                // Find smallest available set that doesn't conflict with neighbors
                let mut used_sets = vec![false; num_sets + 1];

                // Mark sets used by neighbors
                if v < adjacency.len() {
                    for &neighbor in &adjacency[v] {
                        if neighbor < num_vertices && vertex_set[neighbor] != u32::MAX {
                            used_sets[vertex_set[neighbor] as usize] = true;
                        }
                    }
                }

                // Find first unused set
                let mut assigned_set = num_sets;
                for (set_id, &used) in used_sets.iter().enumerate() {
                    if !used {
                        assigned_set = set_id;
                        break;
                    }
                }

                vertex_set[v] = assigned_set as u32;
                if assigned_set == num_sets {
                    num_sets += 1;
                }
            }
        }

        log::info!(
            "Computed {} independent sets for {} vertices (avg {:.1} vertices/set)",
            num_sets,
            num_vertices,
            num_vertices as f64 / num_sets as f64
        );

        (vertex_set, num_sets)
    }

    /// Convert adjacency list to CSR (Compressed Sparse Row) format.
    ///
    /// CSR format uses two arrays:
    /// - row_ptr[v] = starting index of neighbors for vertex v in col_idx
    /// - col_idx = concatenated neighbor lists
    ///
    /// This format is memory-efficient and GPU-friendly for sparse graphs.
    fn adjacency_to_csr(
        &self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut row_ptr = vec![0u32];
        let mut col_idx = Vec::new();

        for v in 0..num_vertices {
            if v < adjacency.len() {
                for &neighbor in &adjacency[v] {
                    col_idx.push(neighbor as u32);
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }

        (row_ptr, col_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_thermodynamic_gpu_triangle() {
        env_logger::try_init().ok();

        let device = CudaContext::new(0).expect("CUDA not available");
        let thermo =
            ThermodynamicGpu::new(device, "target/ptx/thermodynamic.ptx").expect("GPU init failed");

        // Triangle graph (3 vertices, requires 3 colors)
        let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let initial = vec![1, 2, 3]; // Start with valid coloring

        let result = thermo
            .run(&adjacency, 3, &initial, 4, 1000, 0.01, 5.0, 0.0)
            .expect("Thermodynamic run failed");

        assert_eq!(result.len(), 3);
        // Check validity (no adjacent vertices same color)
        assert_ne!(result[0], result[1]);
        assert_ne!(result[1], result[2]);
        assert_ne!(result[0], result[2]);
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_thermodynamic_gpu_petersen() {
        env_logger::try_init().ok();

        let device = CudaContext::new(0).expect("CUDA not available");
        let thermo =
            ThermodynamicGpu::new(device, "target/ptx/thermodynamic.ptx").expect("GPU init failed");

        // Petersen graph (10 vertices, chromatic number = 3)
        let adjacency = vec![
            vec![1, 4, 5],
            vec![0, 2, 6],
            vec![1, 3, 7],
            vec![2, 4, 8],
            vec![0, 3, 9],
            vec![0, 7, 8],
            vec![1, 8, 9],
            vec![2, 5, 9],
            vec![3, 5, 6],
            vec![4, 6, 7],
        ];

        // Start with greedy coloring
        let initial = vec![1, 2, 3, 1, 2, 2, 3, 1, 1, 2];

        let result = thermo
            .run(&adjacency, 10, &initial, 8, 5000, 0.01, 10.0, 0.0)
            .expect("Thermodynamic run failed");

        assert_eq!(result.len(), 10);

        // Check validity
        let mut conflicts = 0;
        for (v, neighbors) in adjacency.iter().enumerate() {
            for &u in neighbors {
                if result[v] == result[u] {
                    conflicts += 1;
                }
            }
        }

        assert_eq!(conflicts, 0, "Thermodynamic should produce valid coloring");

        // Check that chromatic number is reasonable (should be 3 for Petersen)
        let max_color = *result.iter().max().unwrap();
        assert!(
            max_color <= 5,
            "Chromatic number should be <= 5 for Petersen (optimal: 3)"
        );
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_adjacency_to_csr() {
        let device = CudaContext::new(0).unwrap_or_else(|_| {
            panic!("Test requires GPU");
        });
        let thermo = ThermodynamicGpu::new(device, "target/ptx/thermodynamic.ptx").unwrap();

        let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let (row_ptr, col_idx) = thermo.adjacency_to_csr(&adjacency, 3);

        assert_eq!(row_ptr, vec![0, 2, 4, 6]);
        assert_eq!(col_idx, vec![1, 2, 0, 2, 0, 1]);
    }
}
