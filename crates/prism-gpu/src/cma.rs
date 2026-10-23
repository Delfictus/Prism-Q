//! CMA-ES Ensemble Exchange GPU acceleration
//!
//! ASSUMPTIONS:
//! - PTX module "ensemble_exchange" loaded in GPU context
//! - Population stored as contiguous f32 arrays
//! - MAX_POPULATION = 1024, MAX_DIMENSIONS = 2048, MAX_REPLICAS = 64
//! - Requires sm_80+ for efficient matrix operations
//!
//! PERFORMANCE TARGETS:
//! - 100-population evolution: < 20ms per generation
//! - Replica exchange: < 5ms per round
//! - Memory: < 500MB for typical configurations
//!
//! REFERENCE: PRISM Spec Section 2.4 "CMA-ES Ensemble Exchange"

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, CudaFunction, LaunchConfig, PushKernelArg, DeviceSlice};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// CMA-ES ensemble parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CmaEnsembleParams {
    pub num_replicas: i32,
    pub population_size: i32,
    pub parent_size: i32,
    pub dimensions: i32,
    pub sigma: f32,
    pub c_sigma: f32,
    pub d_sigma: f32,
    pub c_c: f32,
    pub c_1: f32,
    pub c_mu: f32,
    pub chi_n: f32,
    pub seed: u64,
    pub exchange_interval: i32,
    pub exchange_rate: f32,
}

impl Default for CmaEnsembleParams {
    fn default() -> Self {
        let dimensions = 100;
        let population_size = 4 + (3.0 * (dimensions as f32).ln()) as i32;
        let parent_size = population_size / 2;

        // CMA-ES default parameters
        let weights_sum = (1..=parent_size)
            .map(|i| ((parent_size as f32 + 0.5).ln() - (i as f32).ln()))
            .sum::<f32>();

        let mu_eff = weights_sum.powi(2)
            / (1..=parent_size)
                .map(|i| {
                    let w = ((parent_size as f32 + 0.5).ln() - (i as f32).ln()) / weights_sum;
                    w * w
                })
                .sum::<f32>();

        let c_sigma = (mu_eff + 2.0) / (dimensions as f32 + mu_eff + 5.0);
        let d_sigma =
            1.0 + c_sigma + 2.0 * ((mu_eff - 1.0) / (dimensions as f32 + 1.0)).max(0.0).sqrt();

        let c_c = (4.0 + mu_eff / dimensions as f32)
            / (dimensions as f32 + 4.0 + 2.0 * mu_eff / dimensions as f32);

        let c_1 = 2.0 / ((dimensions as f32 + 1.3).powi(2) + mu_eff);
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff))
            / ((dimensions as f32 + 2.0).powi(2) + mu_eff).min(1.0 - c_1);

        let chi_n = (dimensions as f32).sqrt()
            * (1.0 - 1.0 / (4.0 * dimensions as f32))
                + 1.0 / (21.0 * dimensions as f32 * dimensions as f32);

        Self {
            num_replicas: 4,
            population_size,
            parent_size,
            dimensions,
            sigma: 1.0,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
            seed: 42,
            exchange_interval: 10,
            exchange_rate: 0.1,
        }
    }
}

/// CMA-ES Ensemble GPU accelerator
pub struct CmaEnsembleGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // Kernel functions
    cma_evolution_kernel: CudaFunction,
    replica_exchange_kernel: CudaFunction,
    diversity_kernel: CudaFunction,
    adaptive_parameters_kernel: CudaFunction,
    convergence_detection_kernel: CudaFunction,
    cma_performance_metrics: CudaFunction,

    // Device memory
    populations: CudaSlice<f32>,
    fitness_values: CudaSlice<f32>,
    mean_vectors: CudaSlice<f32>,
    covariance_matrices: CudaSlice<f32>,
    evolution_paths_sigma: CudaSlice<f32>,
    evolution_paths_c: CudaSlice<f32>,
    sigmas: CudaSlice<f32>,
    generations: CudaSlice<i32>,
    exchange_matrix: CudaSlice<f32>,
    exchange_counts: CudaSlice<i32>,
    diversity_metrics: CudaSlice<f32>,

    // Configuration
    params: CmaEnsembleParams,
}

impl CmaEnsembleGpu {
    /// Creates a new CMA-ES ensemble GPU accelerator
    pub fn new(
        context: Arc<CudaContext>,
        num_replicas: usize,
        dimensions: usize,
        population_size: Option<usize>,
    ) -> Result<Self> {
        // Validation
        anyhow::ensure!(
            num_replicas <= 64,
            "Replicas {} exceed maximum 64",
            num_replicas
        );
        anyhow::ensure!(
            dimensions <= 2048,
            "Dimensions {} exceed maximum 2048",
            dimensions
        );

        let pop_size =
            population_size.unwrap_or_else(|| 4 + (3.0 * (dimensions as f32).ln()) as usize);

        anyhow::ensure!(
            pop_size <= 1024,
            "Population size {} exceeds maximum 1024",
            pop_size
        );

        log::info!(
            "Initializing CMA-ES Ensemble GPU: {} replicas, {} dimensions, {} population",
            num_replicas,
            dimensions,
            pop_size
        );

        // Get stream from context
        let stream = context.default_stream();

        // Load PTX module with explicit kernel list from file system
        let ptx_path = "target/ptx/ensemble_exchange.ptx";
        let ptx = Ptx::from_file(ptx_path);

        let module = context.load_module(ptx)
            .with_context(|| format!("Failed to load CMA Ensemble PTX module from {}", ptx_path))?;

        // Retrieve functions using module.load_function
        let cma_evolution_kernel = module.load_function("cma_evolution_kernel")?;
        let replica_exchange_kernel = module.load_function("replica_exchange_kernel")?;
        let diversity_kernel = module.load_function("diversity_kernel")?;
        let adaptive_parameters_kernel = module.load_function("adaptive_parameters_kernel")?;
        let convergence_detection_kernel = module.load_function("convergence_detection_kernel")?;
        let cma_performance_metrics = module.load_function("cma_performance_metrics")?;

        log::debug!("CMA Ensemble PTX module loaded with 6 kernels");

        // Initialize parameters
        let mut params = CmaEnsembleParams::default();
        params.num_replicas = num_replicas as i32;
        params.dimensions = dimensions as i32;
        params.population_size = pop_size as i32;
        params.parent_size = (pop_size / 2) as i32;

        // Allocate device memory using STREAM
        let pop_total_size = num_replicas * pop_size * dimensions;
        let populations = stream
            .alloc_zeros::<f32>(pop_total_size)
            .context("Failed to allocate populations")?;

        let fitness_size = num_replicas * pop_size;
        let fitness_values = stream
            .alloc_zeros::<f32>(fitness_size)
            .context("Failed to allocate fitness values")?;

        let mean_size = num_replicas * dimensions;
        let mean_vectors = stream
            .alloc_zeros::<f32>(mean_size)
            .context("Failed to allocate mean vectors")?;

        let cov_size = num_replicas * dimensions * dimensions;
        let _covariance_matrices_temp = stream
            .alloc_zeros::<f32>(cov_size)
            .context("Failed to allocate covariance matrices")?;

        let evolution_paths_sigma = stream
            .alloc_zeros::<f32>(mean_size)
            .context("Failed to allocate evolution paths sigma")?;

        let evolution_paths_c = stream
            .alloc_zeros::<f32>(mean_size)
            .context("Failed to allocate evolution paths c")?;

        let _sigmas_temp = stream
            .alloc_zeros::<f32>(num_replicas)
            .context("Failed to allocate sigmas")?;

        let generations = stream
            .alloc_zeros::<i32>(num_replicas)
            .context("Failed to allocate generation counters")?;

        let exchange_matrix = stream
            .alloc_zeros::<f32>(num_replicas * num_replicas)
            .context("Failed to allocate exchange matrix")?;

        let num_pairs = num_replicas * (num_replicas - 1) / 2;
        let exchange_counts = stream
            .alloc_zeros::<i32>(num_pairs)
            .context("Failed to allocate exchange counts")?;

        let diversity_metrics = stream
            .alloc_zeros::<f32>(num_replicas)
            .context("Failed to allocate diversity metrics")?;

        // Initialize covariance to identity and sigma to 1.0
        let mut init_cov = vec![0.0f32; cov_size];
        for r in 0..num_replicas {
            for i in 0..dimensions {
                init_cov[r * dimensions * dimensions + i * dimensions + i] = 1.0;
            }
        }
        let covariance_matrices = stream.clone_htod(&init_cov)?;

        let init_sigmas = vec![1.0f32; num_replicas];
        let sigmas = stream.clone_htod(&init_sigmas)?;

        Ok(Self {
            context,
            stream,
            cma_evolution_kernel,
            replica_exchange_kernel,
            diversity_kernel,
            adaptive_parameters_kernel,
            convergence_detection_kernel,
            cma_performance_metrics,
            populations,
            fitness_values,
            mean_vectors,
            covariance_matrices,
            evolution_paths_sigma,
            evolution_paths_c,
            sigmas,
            generations,
            exchange_matrix,
            exchange_counts,
            diversity_metrics,
            params,
        })
    }

    /// Runs one generation of CMA-ES evolution for all replicas
    pub fn evolve_generation(&mut self) -> Result<()> {
        // Launch evolution kernel
        let blocks = self.params.num_replicas as u32;
        let threads = self.params.population_size.min(256) as u32;
        let shared_size =
            (self.params.population_size as usize * 4) + (self.params.population_size as usize * 4); // fitness + indices

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_size as u32,
        };

        // Pass individual parameters instead of struct
        unsafe {
            self.stream.launch_builder(&self.cma_evolution_kernel)
                .arg(&self.populations)
                .arg(&self.fitness_values)
                .arg(&self.mean_vectors)
                .arg(&self.covariance_matrices)
                .arg(&self.evolution_paths_sigma)
                .arg(&self.evolution_paths_c)
                .arg(&self.sigmas)
                .arg(&self.generations)
                .arg(&self.params.num_replicas)
                .arg(&self.params.population_size)
                .arg(&self.params.dimensions)
                .arg(&self.params.sigma)
                .launch(config)
        }
        .context("Failed to launch evolution kernel")?;

        self.stream.synchronize()?;
        Ok(())
    }

    /// Performs replica exchange between CMA-ES instances
    pub fn exchange_replicas(&mut self) -> Result<()> {
        let threads_per_block = 128;
        let num_pairs = (self.params.num_replicas * (self.params.num_replicas - 1) / 2) as u32;
        let blocks = (num_pairs + threads_per_block - 1) / threads_per_block;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&self.replica_exchange_kernel)
                .arg(&self.populations)
                .arg(&self.fitness_values)
                .arg(&self.mean_vectors)
                .arg(&self.sigmas)
                .arg(&self.exchange_matrix)
                .arg(&self.exchange_counts)
                .arg(&self.params.num_replicas)
                .arg(&self.params.exchange_rate)
                .launch(config)
        }
        .context("Failed to launch exchange kernel")?;

        self.stream.synchronize()?;
        Ok(())
    }

    /// Computes diversity metrics for all replicas
    pub fn compute_diversity(&mut self) -> Result<Vec<f32>> {
        let blocks = self.params.num_replicas as u32;
        let threads = self.params.population_size.min(256) as u32;
        let shared_size = threads as usize * 4;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared_size as u32,
        };

        // Allocate crowding distances
        let crowding_size = (self.params.num_replicas * self.params.population_size) as usize;
        let mut crowding_distances = self.stream.alloc_zeros::<f32>(crowding_size)?;

        unsafe {
            self.stream.launch_builder(&self.diversity_kernel)
                .arg(&self.populations)
                .arg(&self.diversity_metrics)
                .arg(&mut crowding_distances)
                .arg(&self.params.num_replicas)
                .arg(&self.params.population_size)
                .arg(&self.params.dimensions)
                .launch(config)
        }
        .context("Failed to launch diversity kernel")?;

        self.stream.synchronize()?;

        let result = self.stream.clone_dtoh(&self.diversity_metrics)?;
        Ok(result)
    }

    /// Adapts parameters based on performance metrics
    pub fn adapt_parameters(&mut self) -> Result<()> {
        // Compute success rates
        let success_rates = self.compute_success_rates()?;

        let threads_per_block = 256;
        let blocks =
            ((self.params.num_replicas + threads_per_block - 1) / threads_per_block) as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Get current generation
        let gens = self.stream.clone_dtoh(&self.generations)?;
        let avg_gen = gens.iter().sum::<i32>() / self.params.num_replicas;

        unsafe {
            self.stream.launch_builder(&self.adaptive_parameters_kernel)
                .arg(&self.sigmas)
                .arg(&self.fitness_values)
                .arg(&self.diversity_metrics)
                .arg(&success_rates)
                .arg(&avg_gen)
                .arg(&self.params.num_replicas)
                .launch(config)
        }
        .context("Failed to launch adaptive parameters kernel")?;

        self.stream.synchronize()?;

        // Note: In production, we would download adapted parameters from GPU
        // For now, simplified version without struct passing

        Ok(())
    }

    /// Checks convergence status of all replicas
    pub fn check_convergence(&self, fitness_tol: f32, diversity_tol: f32) -> Result<Vec<bool>> {
        let mut convergence_flags = self
            .stream
            .alloc_zeros::<i32>(self.params.num_replicas as usize)?;
        let mut convergence_metrics = self
            .stream
            .alloc_zeros::<f32>(self.params.num_replicas as usize)?;

        let threads_per_block = 256;
        let blocks =
            ((self.params.num_replicas + threads_per_block - 1) / threads_per_block) as u32;

        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&self.convergence_detection_kernel)
                .arg(&self.fitness_values)
                .arg(&self.diversity_metrics)
                .arg(&self.sigmas)
                .arg(&mut convergence_flags)
                .arg(&mut convergence_metrics)
                .arg(&self.params.num_replicas)
                .arg(&fitness_tol)
                .arg(&diversity_tol)
                .launch(config)
        }
        .context("Failed to launch convergence kernel")?;

        self.stream.synchronize()?;

        let flags = self.stream.clone_dtoh(&convergence_flags)?;
        Ok(flags.iter().map(|&f| f != 0).collect())
    }

    /// Gets the best solution across all replicas
    pub fn get_best_solution(&self) -> Result<(Vec<f32>, f32)> {
        // Download fitness values
        let fitness = self.stream.clone_dtoh(&self.fitness_values)?;

        // Find global best
        let (best_idx, &best_fitness) = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        // Calculate replica and individual indices
        let replica_idx = best_idx / self.params.population_size as usize;
        let individual_idx = best_idx % self.params.population_size as usize;

        // Download best individual
        let populations = self.stream.clone_dtoh(&self.populations)?;
        let start = (replica_idx * self.params.population_size as usize + individual_idx)
            * self.params.dimensions as usize;
        let end = start + self.params.dimensions as usize;

        Ok((populations[start..end].to_vec(), best_fitness))
    }

    /// Gets performance metrics
    pub fn get_performance_metrics(&self) -> Result<CmaMetrics> {
        let mut metrics = self.stream.alloc_zeros::<f32>(4)?;

        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream.launch_builder(&self.cma_performance_metrics)
                .arg(&self.fitness_values)
                .arg(&self.diversity_metrics)
                .arg(&self.exchange_counts)
                .arg(&self.generations)
                .arg(&mut metrics)
                .arg(&self.params.num_replicas)
                .launch(config)
        }
        .context("Failed to launch metrics kernel")?;

        self.stream.synchronize()?;

        let metrics_vec = self.stream.clone_dtoh(&metrics)?;

        Ok(CmaMetrics {
            best_fitness: metrics_vec[0],
            avg_diversity: metrics_vec[1],
            exchange_rate: metrics_vec[2],
            convergence_speed: metrics_vec[3],
        })
    }

    /// Sets fitness evaluation function (placeholder - would integrate with problem)
    pub fn evaluate_fitness<F>(&mut self, eval_fn: F) -> Result<()>
    where
        F: Fn(&[f32]) -> f32,
    {
        // Download populations
        let populations = self.stream.clone_dtoh(&self.populations)?;

        let fitness_size = (self.params.num_replicas * self.params.population_size) as usize;
        let mut fitness = vec![0.0f32; fitness_size];

        // Evaluate each individual
        for i in 0..fitness.len() {
            let start = i * self.params.dimensions as usize;
            let end = start + self.params.dimensions as usize;
            fitness[i] = eval_fn(&populations[start..end]);
        }

        // Upload fitness values - allocate new slice from host data
        self.fitness_values = self.stream.clone_htod(&fitness)?;

        Ok(())
    }

    /// Computes success rates for adaptive parameter control
    fn compute_success_rates(&self) -> Result<CudaSlice<f32>> {
        // Simplified: count improving offspring
        // In practice, would track parent-offspring comparisons

        // Placeholder: set to 0.2 (target rate)
        let rates = vec![0.2f32; self.params.num_replicas as usize];
        let success_rates_slice = self.stream.clone_htod(&rates)?;

        Ok(success_rates_slice)
    }

    /// Updates parameters
    pub fn set_params(&mut self, params: CmaEnsembleParams) {
        self.params = params;
    }

    /// Gets current parameters
    pub fn params(&self) -> &CmaEnsembleParams {
        &self.params
    }
}

/// CMA-ES performance metrics
#[derive(Debug, Clone)]
pub struct CmaMetrics {
    pub best_fitness: f32,
    pub avg_diversity: f32,
    pub exchange_rate: f32,
    pub convergence_speed: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_cma_initialization() {
        let device = CudaContext::new(0).unwrap();
        let cma = CmaEnsembleGpu::new(Arc::new(device), 4, 50, None);
        assert!(cma.is_ok());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_cma_evolution() {
        let device = Arc::new(CudaContext::new(0).unwrap());
        let mut cma = CmaEnsembleGpu::new(device, 2, 30, Some(20)).unwrap();

        // Simple sphere function
        cma.evaluate_fitness(|x| x.iter().map(|&v| v * v).sum())
            .unwrap();
        cma.evolve_generation().unwrap();

        let metrics = cma.get_performance_metrics().unwrap();
        assert!(metrics.best_fitness >= 0.0);
    }
}