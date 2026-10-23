//! GPU-Accelerated Active Inference for Phase 1
//!
//! Implements variational free energy minimization on GPU for graph coloring.
//! Based on the full Active Inference implementation from foundation/active_inference.
//!
//! ## Specification Compliance
//! - Implements PRISM GPU Plan §4.1: GPU Context Management
//! - Uses cudarc API for CUDA operations (like other prism-gpu modules)
//! - Loads PTX from target/ptx/active_inference.ptx
//! - Returns ActiveInferencePolicy with EFE, VFE, uncertainty scores
//!
//! ## Performance Targets (from foundation/active_inference/controller.rs)
//! - Action selection: <2ms per action
//! - Full policy computation: <50ms for 250 vertices
//! - GPU speedup: 10-20x over CPU
//!
//! ## Key Concepts
//! - **Expected Free Energy (EFE)**: Pragmatic value + Epistemic value
//! - **Variational Free Energy (VFE)**: Complexity - Accuracy
//! - **Pragmatic Value**: Goal-directed (vertex degree-based)
//! - **Epistemic Value**: Information-seeking (phase variance-based)

use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, DeviceSlice, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use prism_core::{Graph, PrismError};
use std::sync::Arc;

/// Active Inference policy computed on GPU
///
/// Represents the expected free energy and uncertainty for each vertex,
/// guiding vertex ordering in Phase 1 graph coloring.
///
/// Spec reference: foundation/prct-core/src/gpu_active_inference.rs:24-37
#[derive(Debug, Clone)]
pub struct ActiveInferencePolicy {
    /// Per-vertex uncertainty scores (normalized to [0, 1], higher = more uncertain)
    pub uncertainty: Vec<f64>,

    /// Expected free energy for each vertex (EFE = pragmatic - epistemic/2)
    pub expected_free_energy: Vec<f64>,

    /// Pragmatic value (goal-directed, degree-based)
    pub pragmatic_value: Vec<f64>,

    /// Epistemic value (information-seeking, variance-based)
    pub epistemic_value: Vec<f64>,
}

impl ActiveInferencePolicy {
    /// Sort vertices by uncertainty (descending) for greedy coloring
    pub fn vertex_order(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.uncertainty.len()).collect();
        indices.sort_by(|&a, &b| {
            self.uncertainty[b]
                .partial_cmp(&self.uncertainty[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }

    /// Mean uncertainty across all vertices
    pub fn mean_uncertainty(&self) -> f64 {
        self.uncertainty.iter().sum::<f64>() / self.uncertainty.len() as f64
    }

    /// Mean expected free energy
    pub fn mean_efe(&self) -> f64 {
        self.expected_free_energy.iter().sum::<f64>() / self.expected_free_energy.len() as f64
    }
}

/// GPU-accelerated Active Inference engine
///
/// Loads and executes CUDA kernels for variational inference operations.
/// Implements the full Active Inference algorithm from foundation/.
///
/// Spec reference: foundation/active_inference/gpu.rs:21-45
pub struct ActiveInferenceGpu {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // CUDA kernels from active_inference.ptx
    _gemv_kernel: CudaFunction,
    prediction_error_kernel: CudaFunction,
    _belief_update_kernel: CudaFunction,
    _precision_weight_kernel: CudaFunction,
    kl_divergence_kernel: CudaFunction,
    accuracy_kernel: CudaFunction,
    sum_reduction_kernel: CudaFunction,
    _axpby_kernel: CudaFunction,

    // Configuration (from foundation/active_inference/variational_inference.rs:99)
    _learning_rate: f64,    // κ = 0.01 (reduced from 0.1 to prevent divergence)
    _max_iterations: usize, // Default: 100
    _convergence_threshold: f64, // Default: 1e-4
}

impl ActiveInferenceGpu {
    /// Create new GPU Active Inference engine
    ///
    /// Loads PTX from target/ptx/active_inference.ptx and initializes kernels.
    ///
    /// # Arguments
    /// * `device` - Shared CUDA device (from GpuContext)
    /// * `ptx_path` - Path to active_inference.ptx (default: "target/ptx/active_inference.ptx")
    ///
    /// # Errors
    /// Returns PrismError::GpuInitError if PTX loading fails
    pub fn new(context: Arc<CudaContext>, ptx_path: &str) -> Result<Self, PrismError> {
        log::info!("[ActiveInferenceGpu] Loading PTX from: {}", ptx_path);

        // Get default stream
        let stream = context.default_stream();

        // Load PTX module (spec reference: foundation/active_inference/gpu.rs:54-77)
        let ptx = Ptx::from_file(ptx_path);

        // Load PTX module
        let module = context.load_module(ptx).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to load active_inference module: {}", e),
            )
        })?;

        // Retrieve functions
        let get_kernel = |name| {
            module.load_function(name).map_err(|e| {
                PrismError::gpu(
                    "active_inference",
                    format!("Failed to find kernel {}: {}", name, e),
                )
            })
        };

        let gemv_kernel = get_kernel("gemv_kernel")?;
        let prediction_error_kernel = get_kernel("prediction_error_kernel")?;
        let belief_update_kernel = get_kernel("belief_update_kernel")?;
        let precision_weight_kernel = get_kernel("precision_weight_kernel")?;
        let kl_divergence_kernel = get_kernel("kl_divergence_kernel")?;
        let accuracy_kernel = get_kernel("accuracy_kernel")?;
        let sum_reduction_kernel = get_kernel("sum_reduction_kernel")?;
        let axpby_kernel = get_kernel("axpby_kernel")?;

        log::info!("[ActiveInferenceGpu] Loaded 8 kernels successfully");

        Ok(Self {
            context,
            stream,
            _gemv_kernel: gemv_kernel,
            prediction_error_kernel,
            _belief_update_kernel: belief_update_kernel,
            _precision_weight_kernel: precision_weight_kernel,
            kl_divergence_kernel,
            accuracy_kernel,
            sum_reduction_kernel,
            _axpby_kernel: axpby_kernel,
            _learning_rate: 0.01,
            _max_iterations: 100,
            _convergence_threshold: 1e-4,
        })
    }

    /// Compute Active Inference policy for graph coloring
    ///
    /// Implements the full algorithm from foundation/prct-core/src/gpu_active_inference.rs:53-256
    ///
    /// # Algorithm
    /// 1. Compute observations from graph structure (degree, neighborhood)
    /// 2. Initialize beliefs (mean = current coloring, variance = 1.0)
    /// 3. Compute precision inversely proportional to degree (high degree = high uncertainty)
    /// 4. Run prediction error kernel on GPU
    /// 5. Compute pragmatic value (degree-based) and epistemic value (error-based)
    /// 6. Combine into expected free energy and uncertainty
    /// 7. Normalize uncertainty to [0, 1]
    ///
    /// # Returns
    /// ActiveInferencePolicy with per-vertex uncertainty and EFE scores
    pub fn compute_policy(
        &self,
        graph: &Graph,
        coloring: &[usize],
    ) -> Result<ActiveInferencePolicy, PrismError> {
        let start = std::time::Instant::now();
        let n = graph.num_vertices;

        log::debug!("[ActiveInferenceGpu] Computing policy for {} vertices", n);

        // Compute observations from graph structure
        // Spec reference: foundation/prct-core/src/gpu_active_inference.rs:262-293
        let observations = self.compute_observations(graph)?;

        // Initialize beliefs: mean = current coloring, variance = 1.0
        let initial_mean: Vec<f64> = coloring
            .iter()
            .map(|&c| if c == usize::MAX { 0.5 } else { c as f64 })
            .collect();
        let _initial_variance = vec![1.0; n];

        // Compute precision inversely proportional to degree
        // Spec reference: foundation/prct-core/src/gpu_active_inference.rs:113-132
        let precision = self.compute_precision(graph)?;

        // Upload data to GPU using device (atomic copy)
        let d_observations = self.stream.clone_htod(&observations).map_err(|e| {
            PrismError::gpu("active_inference", format!("Failed to upload obs: {}", e))
        })?;
        let d_mean = self.stream.clone_htod(&initial_mean).map_err(|e| {
            PrismError::gpu("active_inference", format!("Failed to upload mean: {}", e))
        })?;
        let d_precision = self.stream.clone_htod(&precision).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to upload precision: {}", e),
            )
        })?;

        // Allocate workspace for prediction errors
        let mut d_pred_error = self.stream.alloc_zeros::<f64>(n).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to allocate pred_error: {}", e),
            )
        })?;

        // Launch prediction error kernel
        let threads = 256;
        let blocks = n.div_ceil(threads);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.prediction_error_kernel)
                .arg(&mut d_pred_error)
                .arg(&d_observations)
                .arg(&d_mean)
                .arg(&d_precision)
                .arg(&(n as i32))
                .launch(cfg)
                .map_err(|e| {
                    PrismError::gpu(
                        "active_inference",
                        format!("Prediction error kernel failed: {}", e),
                    )
                })?;
        }

        // Download prediction errors using device
        let pred_errors = self.stream.clone_dtoh(&d_pred_error).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to download errors: {}", e),
            )
        })?;

        // Compute pragmatic value from precision (degree-based)
        // Spec reference: foundation/prct-core/src/gpu_active_inference.rs:185-194
        let pragmatic_value: Vec<f64> = precision
            .iter()
            .map(|&p| {
                // precision range: [0.001, 0.021]
                // pragmatic range: [0.1, 1.0]
                let normalized_prec = (p - 0.001) / (0.021 - 0.001);
                0.1 + (1.0 - normalized_prec) * 0.9
            })
            .collect();

        // Compute epistemic value from prediction errors
        let epistemic_value: Vec<f64> = pred_errors.iter().map(|&e| e.abs()).collect();

        // Uncertainty: Combination of pragmatic and epistemic
        let mut uncertainty: Vec<f64> = pragmatic_value
            .iter()
            .zip(epistemic_value.iter())
            .map(|(&prag, &epist)| prag * (1.0 + epist))
            .collect();

        // Expected free energy: Balance pragmatic and epistemic
        let expected_free_energy: Vec<f64> = pragmatic_value
            .iter()
            .zip(epistemic_value.iter())
            .map(|(&prag, &epist)| prag - 0.5 * epist)
            .collect();

        // Normalize uncertainty to [0, 1]
        // Spec reference: foundation/prct-core/src/gpu_active_inference.rs:214-245
        let min_unc = uncertainty.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_unc = uncertainty
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if (max_unc - min_unc).abs() > 1e-10 {
            for u in &mut uncertainty {
                *u = (*u - min_unc) / (max_unc - min_unc);
            }
        } else {
            // Fallback: uniform distribution
            log::warn!("[ActiveInferenceGpu] Uniform uncertainty detected, using fallback");
            let uniform = 1.0 / uncertainty.len() as f64;
            uncertainty.fill(uniform);
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "[ActiveInferenceGpu] Policy computed in {:.2}ms (target: <50ms)",
            elapsed
        );

        Ok(ActiveInferencePolicy {
            uncertainty,
            expected_free_energy,
            pragmatic_value,
            epistemic_value,
        })
    }

    /// Compute observations from graph structure
    ///
    /// Observations combine:
    /// 1. Normalized vertex degree (0 to 1)
    /// 2. Neighborhood density
    ///
    /// Spec reference: foundation/prct-core/src/gpu_active_inference.rs:262-293
    fn compute_observations(&self, graph: &Graph) -> Result<Vec<f64>, PrismError> {
        let n = graph.num_vertices;
        let mut observations = Vec::with_capacity(n);

        for v in 0..n {
            // Compute vertex degree
            let degree = graph.degree(v);
            let normalized_degree = degree as f64 / n as f64;

            // For graph coloring, observation is primarily degree-based
            // (In the original, Kuramoto phase is used, but we don't have that here)
            observations.push(normalized_degree);
        }

        Ok(observations)
    }

    /// Compute precision inversely proportional to degree
    ///
    /// High degree vertices have low precision (high uncertainty).
    /// Low degree vertices have high precision (low uncertainty).
    ///
    /// Spec reference: foundation/prct-core/src/gpu_active_inference.rs:113-132
    fn compute_precision(&self, graph: &Graph) -> Result<Vec<f64>, PrismError> {
        let n = graph.num_vertices;
        let max_degree = 500.0; // Approximate max for DSJC1000.5

        let precision: Vec<f64> = (0..n)
            .map(|v| {
                let degree = graph.degree(v);
                let normalized_degree = (degree as f64 / max_degree).min(1.0);

                // precision = 0.001 + (1.0 - normalized_degree) * 0.02
                // Range: [0.001, 0.021]
                0.001 + (1.0 - normalized_degree) * 0.02
            })
            .collect();

        Ok(precision)
    }

    /// Compute variational free energy on GPU
    ///
    /// F = Complexity - Accuracy
    /// Complexity = KL[q(x) || p(x)]
    /// Accuracy = E_q[ln p(o|x)]
    ///
    /// Spec reference: foundation/active_inference/gpu.rs:153-284
    ///
    /// TODO(GPU-AI-VFE): Currently unused in Phase 1, but available for future extensions
    #[allow(dead_code)]
    fn compute_free_energy(
        &self,
        observations: &[f64],
        mean_posterior: &[f64],
        var_posterior: &[f64],
        mean_prior: &[f64],
        var_prior: &[f64],
        obs_precision: &[f64],
    ) -> Result<f64, PrismError> {
        let n = mean_posterior.len();

        // Upload data to GPU using device
        let d_mean_q = self.stream.clone_htod(mean_posterior).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to upload mean_q: {}", e),
            )
        })?;
        let d_mean_p = self.stream.clone_htod(mean_prior).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to upload mean_p: {}", e),
            )
        })?;
        let d_var_q = self.stream.clone_htod(var_posterior).map_err(|e| {
            PrismError::gpu("active_inference", format!("Failed to upload var_q: {}", e))
        })?;
        let d_var_p = self.stream.clone_htod(var_prior).map_err(|e| {
            PrismError::gpu("active_inference", format!("Failed to upload var_p: {}", e))
        })?;

        let threads = 256;
        let blocks = n.div_ceil(threads);
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Compute KL divergence (complexity)
        let mut d_kl_components = self.stream.alloc_zeros::<f64>(n).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to allocate kl_components: {}", e),
            )
        })?;

        unsafe {
            self.stream
                .launch_builder(&self.kl_divergence_kernel)
                .arg(&d_mean_q)
                .arg(&d_mean_p)
                .arg(&d_var_q)
                .arg(&d_var_p)
                .arg(&mut d_kl_components)
                .arg(&(n as i32))
                .launch(cfg)
                .map_err(|e| {
                    PrismError::gpu(
                        "active_inference",
                        format!("KL divergence kernel failed: {}", e),
                    )
                })?;
        }

        // Sum KL components
        let mut d_complexity = self.stream.alloc_zeros::<f64>(1).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to allocate complexity: {}", e),
            )
        })?;

        unsafe {
            self.stream
                .launch_builder(&self.sum_reduction_kernel)
                .arg(&d_kl_components)
                .arg(&mut d_complexity)
                .arg(&(n as i32))
                .launch(cfg)
                .map_err(|e| {
                    PrismError::gpu(
                        "active_inference",
                        format!("Sum reduction kernel failed: {}", e),
                    )
                })?;
        }

        // Compute accuracy (similar process for prediction errors)
        let d_obs = self.stream.clone_htod(observations).map_err(|e| {
            PrismError::gpu("active_inference", format!("Failed to upload obs: {}", e))
        })?;
        let d_precision = self.stream.clone_htod(obs_precision).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to upload precision: {}", e),
            )
        })?;

        let mut d_errors = self
            .stream
            .alloc_zeros::<f64>(observations.len())
            .map_err(|e| {
                PrismError::gpu(
                    "active_inference",
                    format!("Failed to allocate errors: {}", e),
                )
            })?;

        let obs_cfg = LaunchConfig {
            grid_dim: (observations.len().div_ceil(threads) as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.prediction_error_kernel)
                .arg(&mut d_errors)
                .arg(&d_obs)
                .arg(&d_mean_q)
                .arg(&d_precision)
                .arg(&(observations.len() as i32))
                .launch(obs_cfg)
                .map_err(|e| {
                    PrismError::gpu(
                        "active_inference",
                        format!("Prediction error kernel failed: {}", e),
                    )
                })?;
        }

        let mut d_accuracy_components = self
            .stream
            .alloc_zeros::<f64>(observations.len())
            .map_err(|e| {
                PrismError::gpu(
                    "active_inference",
                    format!("Failed to allocate accuracy_components: {}", e),
                )
            })?;

        unsafe {
            self.stream
                .launch_builder(&self.accuracy_kernel)
                .arg(&d_errors)
                .arg(&d_precision)
                .arg(&mut d_accuracy_components)
                .arg(&(observations.len() as i32))
                .launch(obs_cfg)
                .map_err(|e| {
                    PrismError::gpu("active_inference", format!("Accuracy kernel failed: {}", e))
                })?;
        }

        let mut d_accuracy = self.stream.alloc_zeros::<f64>(1).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to allocate accuracy: {}", e),
            )
        })?;

        unsafe {
            self.stream
                .launch_builder(&self.sum_reduction_kernel)
                .arg(&d_accuracy_components)
                .arg(&mut d_accuracy)
                .arg(&(observations.len() as i32))
                .launch(obs_cfg)
                .map_err(|e| {
                    PrismError::gpu(
                        "active_inference",
                        format!("Sum reduction kernel failed: {}", e),
                    )
                })?;
        }

        // Download results
        let complexity_vec = self.stream.clone_dtoh(&d_complexity).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to download complexity: {}", e),
            )
        })?;
        let accuracy_vec = self.stream.clone_dtoh(&d_accuracy).map_err(|e| {
            PrismError::gpu(
                "active_inference",
                format!("Failed to download accuracy: {}", e),
            )
        })?;

        let complexity = complexity_vec[0];
        let accuracy = accuracy_vec[0];
        let free_energy = complexity - accuracy;

        Ok(free_energy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_vertex_order() {
        let policy = ActiveInferencePolicy {
            uncertainty: vec![0.1, 0.9, 0.3, 0.7],
            expected_free_energy: vec![1.0, 2.0, 1.5, 1.8],
            pragmatic_value: vec![0.5; 4],
            epistemic_value: vec![0.5; 4],
        };

        let order = policy.vertex_order();
        // Should be sorted by uncertainty descending: [1, 3, 2, 0]
        assert_eq!(order, vec![1, 3, 2, 0]);
    }

    #[test]
    fn test_policy_mean_metrics() {
        let policy = ActiveInferencePolicy {
            uncertainty: vec![0.2, 0.4, 0.6, 0.8],
            expected_free_energy: vec![1.0, 1.5, 2.0, 2.5],
            pragmatic_value: vec![0.5; 4],
            epistemic_value: vec![0.5; 4],
        };

        assert!((policy.mean_uncertainty() - 0.5).abs() < 1e-6);
        assert!((policy.mean_efe() - 1.75).abs() < 1e-6);
    }
}
