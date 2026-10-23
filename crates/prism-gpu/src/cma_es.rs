//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy) GPU acceleration
//!
//! ASSUMPTIONS:
//! - PTX module "cma_es" loaded in GPU context
//! - Population stored as contiguous f32 arrays
//! - MAX_POPULATION = 1024, MAX_DIMENSIONS = 2048
//! - Requires sm_80+ for efficient matrix operations
//!
//! PERFORMANCE TARGETS:
//! - 100-population evolution: < 20ms per generation
//! - Memory: < 500MB for typical configurations
//! - Convergence: < 1000 generations for sphere function
//!
//! REFERENCE: PRISM Spec Section 2.4 "CMA-ES Optimization"

use anyhow::{Context, Result};
use cudarc::driver::{DevicePtrMut, CudaContext, CudaSlice, CudaStream, LaunchConfig, PushKernelArg, DeviceSlice};
use cudarc::nvrtc::Ptx;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// CMA-ES parameters for GPU kernel
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CmaParams {
    pub population_size: i32,
    pub parent_size: i32,
    pub dimensions: i32,
    pub sigma: f32,   // Step size
    pub c_sigma: f32, // Cumulation for step size control
    pub d_sigma: f32, // Damping for step size
    pub c_c: f32,     // Cumulation for covariance matrix
    pub c_1: f32,     // Learning rate for rank-one update
    pub c_mu: f32,    // Learning rate for rank-mu update
    pub chi_n: f32,   // Expected norm of N(0,I)
    pub mu_eff: f32,  // Variance effective selection mass
    pub seed: u64,
}

impl CmaParams {
    /// Create default CMA-ES parameters for given dimension
    pub fn new(dimensions: usize) -> Self {
        let dim = dimensions as f32;
        let population_size = (4.0 + 3.0 * dim.ln()).floor() as i32;
        let parent_size = population_size / 2;

        // Compute recombination weights
        let mut weights = Vec::new();
        let mut weights_sum = 0.0;
        let mut weights_sum2 = 0.0;

        for i in 0..parent_size {
            let w = ((parent_size as f32 + 0.5).ln() - ((i + 1) as f32).ln()).max(0.0);
            weights.push(w);
            weights_sum += w;
            weights_sum2 += w * w;
        }

        let mu_eff = weights_sum * weights_sum / weights_sum2;

        // Step size control parameters
        let c_sigma = (mu_eff + 2.0) / (dim + mu_eff + 5.0);
        let d_sigma = 1.0 + 2.0 * ((mu_eff - 1.0) / (dim + 1.0)).sqrt().max(0.0) + c_sigma;

        // Covariance matrix adaptation parameters
        let c_c = (4.0 + mu_eff / dim) / (dim + 4.0 + 2.0 * mu_eff / dim);
        let c_1 = 2.0 / ((dim + 1.3).powi(2) + mu_eff);
        let c_mu = ((1.0 - c_1)
            .min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((dim + 2.0).powi(2) + mu_eff)))
        .max(0.0);

        // Expected chi value
        let chi_n = dim.sqrt() * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim));

        Self {
            population_size,
            parent_size,
            dimensions: dimensions as i32,
            sigma: 0.5,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
            mu_eff,
            seed: 42,
        }
    }
}

/// CMA-ES state maintained during optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmaState {
    pub best_solution: Vec<f32>,
    pub best_fitness: f32,
    pub covariance_condition: f32,
    pub generation: usize,
    pub convergence_metric: f32,
    pub sigma: f32,
    pub mean: Vec<f32>,
}

impl CmaState {
    pub fn new(dimensions: usize) -> Self {
        Self {
            best_solution: vec![0.0; dimensions],
            best_fitness: f32::INFINITY,
            covariance_condition: 1.0,
            generation: 0,
            convergence_metric: 1.0,
            sigma: 0.5,
            mean: vec![0.0; dimensions],
        }
    }

    /// Check if converged based on various criteria
    pub fn is_converged(&self, tol_fun: f32, tol_x: f32, max_condition: f32) -> bool {
        self.best_fitness < tol_fun
            || self.convergence_metric < tol_x
            || self.covariance_condition > max_condition
            || self.sigma < 1e-10
    }
}

/// GPU-accelerated CMA-ES optimizer
pub struct CmaOptimizer {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    params: CmaParams,

    // GPU buffers
    d_population: CudaSlice<f32>,
    d_fitness: CudaSlice<f32>,
    d_mean: CudaSlice<f32>,
    d_covariance: CudaSlice<f32>,
    d_bd_matrix: CudaSlice<f32>,
    d_ps: CudaSlice<f32>,
    d_pc: CudaSlice<f32>,
    d_eigenvalues: CudaSlice<f32>,
    d_eigenvectors: CudaSlice<f32>,
    d_weights: CudaSlice<f32>,
    d_selected: CudaSlice<f32>,
    d_old_mean: CudaSlice<f32>,
    d_bd_inv: CudaSlice<f32>,
    d_sigma: CudaSlice<f32>,
    d_best_fitness: CudaSlice<f32>,
    d_condition: CudaSlice<f32>,
    d_ranks: CudaSlice<i32>,

    state: CmaState,
}

impl CmaOptimizer {
    /// Create new CMA-ES optimizer
    pub fn new(device: Arc<CudaContext>, dimensions: usize, seed: u64) -> Result<Self> {
        let mut params = CmaParams::new(dimensions);
        params.seed = seed;

        let pop_size = params.population_size as usize;
        let parent_size = params.parent_size as usize;
        let dim = dimensions;

        // Get default stream from context
        let stream = device.default_stream();

        // Allocate GPU memory using stream
        let d_population = stream
            .alloc_zeros::<f32>(pop_size * dim)
            .context("Failed to allocate population buffer")?;
        let d_fitness = stream
            .alloc_zeros::<f32>(pop_size)
            .context("Failed to allocate fitness buffer")?;
        let d_mean = stream
            .alloc_zeros::<f32>(dim)
            .context("Failed to allocate mean buffer")?;
        let mut d_covariance = stream
            .alloc_zeros::<f32>(dim * dim)
            .context("Failed to allocate covariance buffer")?;
        let mut d_bd_matrix = stream
            .alloc_zeros::<f32>(dim * dim)
            .context("Failed to allocate BD matrix buffer")?;
        let d_ps = stream
            .alloc_zeros::<f32>(dim)
            .context("Failed to allocate ps buffer")?;
        let d_pc = stream
            .alloc_zeros::<f32>(dim)
            .context("Failed to allocate pc buffer")?;
        let d_eigenvalues = stream
            .alloc_zeros::<f32>(dim)
            .context("Failed to allocate eigenvalues buffer")?;
        let d_eigenvectors = stream
            .alloc_zeros::<f32>(dim * dim)
            .context("Failed to allocate eigenvectors buffer")?;
        let mut d_weights = stream
            .alloc_zeros::<f32>(parent_size)
            .context("Failed to allocate weights buffer")?;
        let d_selected = stream
            .alloc_zeros::<f32>(parent_size * dim)
            .context("Failed to allocate selected buffer")?;
        let d_old_mean = stream
            .alloc_zeros::<f32>(dim)
            .context("Failed to allocate old mean buffer")?;
        let d_bd_inv = stream
            .alloc_zeros::<f32>(dim * dim)
            .context("Failed to allocate BD inverse buffer")?;
        let mut d_sigma = stream
            .alloc_zeros::<f32>(1)
            .context("Failed to allocate sigma buffer")?;
        let mut d_best_fitness = stream
            .alloc_zeros::<f32>(1)
            .context("Failed to allocate best fitness buffer")?;
        let mut d_condition = stream
            .alloc_zeros::<f32>(1)
            .context("Failed to allocate condition buffer")?;
        let d_ranks = stream
            .alloc_zeros::<i32>(pop_size)
            .context("Failed to allocate ranks buffer")?;

        // Initialize sigma
        d_sigma = stream.clone_htod(vec![params.sigma])?;
        d_best_fitness = stream.clone_htod(vec![f32::INFINITY])?;
        d_condition = stream.clone_htod(vec![1.0f32])?;

        // Initialize weights
        let mut weights = Vec::new();
        let mut weights_sum = 0.0;
        for i in 0..parent_size {
            let w = ((parent_size as f32 + 0.5).ln() - ((i + 1) as f32).ln()).max(0.0);
            weights.push(w);
            weights_sum += w;
        }
        for w in &mut weights {
            *w /= weights_sum;
        }
        d_weights = stream.clone_htod(weights)?;

        // Initialize covariance to identity
        let mut identity = vec![0.0f32; dim * dim];
        for i in 0..dim {
            identity[i * dim + i] = 1.0;
        }
        d_covariance = stream.clone_htod(identity)?;
        d_bd_matrix = stream.clone_htod(identity)?;

        // Load PTX module once during initialization
        log::info!("Loading CMA-ES PTX module...");
        let ptx_path = "target/ptx/cma_es.ptx";

        let _module = device
            .load_module(Ptx::from_file(ptx_path))
            .context("Failed to load CMA-ES PTX module")?;

        log::info!("CMA-ES GPU module loaded successfully");

        let state = CmaState::new(dimensions);

        Ok(Self {
            context: device,
            stream,
            params,
            d_population,
            d_fitness,
            d_mean,
            d_covariance,
            d_bd_matrix,
            d_ps,
            d_pc,
            d_eigenvalues,
            d_eigenvectors,
            d_weights,
            d_selected,
            d_old_mean,
            d_bd_inv,
            d_sigma,
            d_best_fitness,
            d_condition,
            d_ranks,
            state,
        })
    }

    /// Run one generation of CMA-ES
    pub fn step(&mut self, fitness_fn: impl Fn(&[f32]) -> f32) -> Result<()> {
        let dim = self.params.dimensions as usize;
        let pop_size = self.params.population_size as usize;

        // Get current sigma
        let sigma_vec = self.stream.clone_dtoh(&self.d_sigma)?;
        let sigma = sigma_vec[0];

        // Load kernel functions
        let ptx_path = "target/ptx/cma_es.ptx";
        let module = self.context.load_module(Ptx::from_file(ptx_path))?;

        // Sample population
        let sample_fn = module.load_function("sample_population")?;
        let cfg = LaunchConfig::for_num_elems((pop_size * dim) as u32);
        unsafe {
            self.stream
                .launch_builder(&sample_fn)
                .arg(&self.d_population)
                .arg(&self.d_mean)
                .arg(&self.d_bd_matrix)
                .arg(&sigma)
                .arg(&self.params.population_size)
                .arg(&self.params.dimensions)
                .arg(&self.params.seed)
                .arg(&(self.state.generation as i32))
                .launch(cfg)?;
        }

        // Evaluate fitness (copy to host for custom fitness function)
        let population = self.stream.clone_dtoh(&self.d_population)?;
        let mut fitness = vec![0.0f32; pop_size];

        for i in 0..pop_size {
            let individual = &population[i * dim..(i + 1) * dim];
            fitness[i] = fitness_fn(individual);
        }

        self.d_fitness = self.stream.clone_htod(&fitness)?;

        // Rank and select parents
        let rank_fn = module.load_function("rank_and_select")?;
        let cfg = LaunchConfig::for_num_elems(1);
        unsafe {
            self.stream
                .launch_builder(&rank_fn)
                .arg(&self.d_population)
                .arg(&self.d_fitness)
                .arg(&self.d_ranks)
                .arg(&self.d_selected)
                .arg(&self.params.population_size)
                .arg(&self.params.parent_size)
                .arg(&self.params.dimensions)
                .launch(cfg)?;
        }

        // Update mean
        let update_mean_fn = module.load_function("update_mean")?;
        let cfg = LaunchConfig::for_num_elems(dim as u32);
        unsafe {
            self.stream
                .launch_builder(&update_mean_fn)
                .arg(&self.d_mean)
                .arg(&self.d_old_mean)
                .arg(&self.d_selected)
                .arg(&self.d_weights)
                .arg(&self.params.parent_size)
                .arg(&self.params.dimensions)
                .launch(cfg)?;
        }

        // Update evolution paths
        let update_paths_fn = module.load_function("update_evolution_paths")?;
        let cfg = LaunchConfig::for_num_elems(dim as u32);
        unsafe {
            self.stream
                .launch_builder(&update_paths_fn)
                .arg(&self.d_ps)
                .arg(&self.d_pc)
                .arg(&self.d_mean)
                .arg(&self.d_old_mean)
                .arg(&self.d_bd_inv)
                .arg(&sigma)
                .arg(&self.params.c_sigma)
                .arg(&self.params.c_c)
                .arg(&self.params.mu_eff)
                .arg(&self.params.dimensions)
                .arg(&(self.state.generation as i32))
                .launch(cfg)?;
        }

        // Update covariance
        let update_cov_fn = module.load_function("update_covariance")?;
        let cfg = LaunchConfig::for_num_elems((dim * dim) as u32);
        unsafe {
            self.stream
                .launch_builder(&update_cov_fn)
                .arg(&self.d_covariance)
                .arg(&self.d_pc)
                .arg(&self.d_selected)
                .arg(&self.d_mean)
                .arg(&self.d_old_mean)
                .arg(&self.d_weights)
                .arg(&sigma)
                .arg(&self.params.c_1)
                .arg(&self.params.c_mu)
                .arg(&self.params.parent_size)
                .arg(&self.params.dimensions)
                .launch(cfg)?;
        }

        // Update sigma
        let update_sigma_fn = module.load_function("update_sigma")?;
        let cfg = LaunchConfig::for_num_elems(1);
        unsafe {
            self.stream
                .launch_builder(&update_sigma_fn)
                .arg(&self.d_sigma)
                .arg(&self.d_ps)
                .arg(&self.params.c_sigma)
                .arg(&self.params.d_sigma)
                .arg(&self.params.chi_n)
                .arg(&self.params.dimensions)
                .launch(cfg)?;
        }

        // Eigendecomposition
        let eigendecomp_fn = module.load_function("eigendecompose_covariance")?;
        let cfg = LaunchConfig::for_num_elems(dim as u32);
        unsafe {
            self.stream
                .launch_builder(&eigendecomp_fn)
                .arg(&self.d_covariance)
                .arg(&self.d_eigenvalues)
                .arg(&self.d_eigenvectors)
                .arg(&self.d_bd_matrix)
                .arg(&self.params.dimensions)
                .arg(&100i32)
                .launch(cfg)?;
        }

        // Compute condition number
        let condition_fn = module.load_function("compute_condition_number")?;
        let cfg = LaunchConfig::for_num_elems(1);
        unsafe {
            self.stream
                .launch_builder(&condition_fn)
                .arg(&self.d_eigenvalues)
                .arg(&self.d_condition)
                .arg(&self.params.dimensions)
                .launch(cfg)?;
        }

        // Update state
        let best_idx = fitness
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        if fitness[best_idx] < self.state.best_fitness {
            self.state.best_fitness = fitness[best_idx];
            self.state.best_solution = population[best_idx * dim..(best_idx + 1) * dim].to_vec();
        }

        let sigma_new = self.stream.clone_dtoh(&self.d_sigma)?[0];
        let condition = self.stream.clone_dtoh(&self.d_condition)?[0];
        let mean = self.stream.clone_dtoh(&self.d_mean)?;

        self.state.sigma = sigma_new;
        self.state.covariance_condition = condition;
        self.state.mean = mean;
        self.state.generation += 1;

        // Compute convergence metric (change in mean)
        let mut change = 0.0f32;
        let old_mean = self.stream.clone_dtoh(&self.d_old_mean)?;
        for i in 0..dim {
            change += (self.state.mean[i] - old_mean[i]).powi(2);
        }
        self.state.convergence_metric = change.sqrt();

        Ok(())
    }

    /// Get current optimization state
    pub fn get_state(&self) -> &CmaState {
        &self.state
    }

    /// Run optimization until convergence
    pub fn optimize(
        &mut self,
        fitness_fn: impl Fn(&[f32]) -> f32,
        max_generations: usize,
        tol_fun: f32,
        tol_x: f32,
        max_condition: f32,
    ) -> Result<CmaState> {
        println!("Starting CMA-ES optimization...");
        println!(
            "Dimensions: {}, Population: {}, Parents: {}",
            self.params.dimensions, self.params.population_size, self.params.parent_size
        );

        for gen in 0..max_generations {
            self.step(&fitness_fn)?;

            if gen % 10 == 0 {
                println!(
                    "Generation {}: fitness={:.6e}, sigma={:.6e}, condition={:.2e}",
                    self.state.generation,
                    self.state.best_fitness,
                    self.state.sigma,
                    self.state.covariance_condition
                );
            }

            if self.state.is_converged(tol_fun, tol_x, max_condition) {
                println!("Converged at generation {}", self.state.generation);
                break;
            }
        }

        Ok(self.state.clone())
    }

    /// Emit telemetry metrics
    pub fn emit_telemetry(&self) -> HashMap<String, serde_json::Value> {
        let mut telemetry = HashMap::new();
        telemetry.insert(
            "cma_generation".to_string(),
            serde_json::json!(self.state.generation),
        );
        telemetry.insert(
            "cma_best_fitness".to_string(),
            serde_json::json!(self.state.best_fitness),
        );
        telemetry.insert("cma_sigma".to_string(), serde_json::json!(self.state.sigma));
        telemetry.insert(
            "cma_condition".to_string(),
            serde_json::json!(self.state.covariance_condition),
        );
        telemetry.insert(
            "cma_convergence".to_string(),
            serde_json::json!(self.state.convergence_metric),
        );
        telemetry
    }
}
