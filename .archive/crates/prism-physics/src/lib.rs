//! # PRISM Physics Module
//!
//! Implements physical simulation components including CMA-ES optimization.
//! Provides Phase X (CMA-ES) and materials physics computations.

use prism_core::{
    ColoringSolution, Graph, PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry,
    PrismError,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod cma_cpu;
pub mod cma_es;
pub mod dynamics;
pub mod fields;
pub mod materials;

// Molecular Dynamics - PIMC/NLNM Solvers for protein structures
pub mod molecular_dynamics;

// AMBER ff14SB force field implementation
pub mod amber_ff14sb;

// TIP3P water model for explicit solvent
pub mod water_model;
pub use water_model::{Ion, IonType, TIP3PWater, WaterMolecule};

// Solvation box builder for explicit solvent simulations
pub mod solvation;
pub use solvation::{SolvationBox, SolvationConfig};

// AMBER all-atom dynamics (HMC/Langevin integration)
pub mod amber_dynamics;
pub use amber_dynamics::{
    AmberSimConfig, AmberSimResult, AmberSimulator, TrajectoryFrame as AmberTrajectoryFrame,
};

// Gaussian Network Model for protein flexibility prediction
pub mod gnm;
pub use gnm::{AnisotropicNetworkModel, GaussianNetworkModel, GnmResult};

// GPU-accelerated GNM (uses Lanczos for large matrices)
pub mod gnm_gpu;
pub use gnm_gpu::{GpuGnm, GpuGnmResult};

// Structural analysis modules for enhanced RMSF prediction
pub mod gnm_enhanced;
pub mod secondary_structure;
pub mod sidechain_analysis;
pub mod tertiary_analysis;

// Chemistry-Aware GNM for improved RMSF prediction
pub mod gnm_chemistry;
pub mod residue_chemistry;

// Unified dynamics engine - togglable interface for all dynamics modes
pub mod dynamics_engine;

// NMR ensemble analysis for experimental grounding (positional variability, NOT "RMSF")
pub mod nmr_ensemble;

// Heterogeneous evaluation framework (ATLAS MD + NMR + Functional)
pub mod evaluation_framework;

// Pocket-centric functional metrics for drug discovery (Layer 3)
pub mod pocket_metrics;
pub use pocket_metrics::{
    AllostericSiteCandidate, CrypticPocketCandidate, FlexibilityClass, Layer3Result,
    PocketFlexibilityAnalysis, PocketMetricsCalculator,
};

// Re-export NMR ensemble types
pub use nmr_ensemble::{
    load_nmr_ensemble, parse_nmr_ensemble, NmrEnsemble, NmrModel, TrueRmsf, CURATED_NMR_PDBS,
};

// Re-export evaluation framework types
pub use evaluation_framework::{
    calculate_defensibility_score, create_defensibility_summary, get_published_baselines,
    BaselineComparison, ExperimentalGroundingResult, FunctionalRelevanceResult,
    HeterogeneousEvaluationResult, MdComparabilityResult, SotaComparison,
};

// Re-export enhanced GNM types
pub use gnm_enhanced::{EnhancedGnm, EnhancedGnmConfig, EnhancedGnmResult};
pub use secondary_structure::{SecondaryStructure, SecondaryStructureAnalyzer};
pub use sidechain_analysis::{flexibility_factor, SidechainAnalyzer};
pub use tertiary_analysis::{TertiaryAnalyzer, TertiarySummary};

// Re-export Chemistry-Aware GNM types
pub use gnm_chemistry::{ChemistryGnm, ChemistryGnmConfig, ChemistryGnmResult, ContactType};
pub use residue_chemistry::{
    enhanced_pair_stiffness, get_flexibility_factor, ResidueClass, AA_ORDER,
};

// Re-export dynamics engine types
pub use dynamics_engine::{
    DynamicsConfig, DynamicsEngine, DynamicsMode, DynamicsResult, StructureInput, TrajectoryFrame,
};

/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmaEsConfig {
    /// Population size
    pub population_size: usize,

    /// Initial step size (sigma)
    pub initial_sigma: f32,

    /// Maximum iterations
    pub max_iterations: usize,

    /// Target fitness value
    pub target_fitness: Option<f32>,

    /// Enable GPU acceleration
    pub use_gpu: bool,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            initial_sigma: 0.5,
            max_iterations: 1000,
            target_fitness: None,
            use_gpu: false,
        }
    }
}

/// CMA-ES phase controller with real optimization functionality
pub struct CmaEsPhaseController {
    config: CmaEsConfig,
    optimizer: Option<cma_cpu::CmaOptimizer>,
    best_solution: Vec<f32>,
    best_fitness: f32,
    generation: usize,
    fitness_history: Vec<f32>,
    telemetry: CmaEsTelemetry,
}

// Ensure thread safety for pipeline execution
unsafe impl Send for CmaEsPhaseController {}
unsafe impl Sync for CmaEsPhaseController {}

impl CmaEsPhaseController {
    pub fn new(config: CmaEsConfig) -> Self {
        let telemetry = CmaEsTelemetry {
            best_fitness: f32::INFINITY,
            mean_fitness: f32::INFINITY,
            fitness_std: 0.0,
            sigma: config.initial_sigma,
            generation: 0,
            condition_number: 1.0,
            convergence_metric: 0.0,
        };
        Self {
            config,
            optimizer: None,
            best_solution: Vec::new(),
            best_fitness: f32::INFINITY,
            generation: 0,
            fitness_history: Vec::new(),
            telemetry,
        }
    }

    pub fn validate_config(&self) -> Result<(), PrismError> {
        if self.config.population_size < 4 {
            return Err(PrismError::ValidationError(
                "Population size must be at least 4".to_string(),
            ));
        }
        if self.config.initial_sigma <= 0.0 {
            return Err(PrismError::ValidationError(
                "Initial sigma must be positive".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute fitness for a graph coloring solution (reserved for direct fitness eval)
    #[allow(dead_code)]
    fn compute_fitness(&self, solution: &[f32], graph: &Graph) -> f32 {
        // Convert continuous parameters to discrete colors
        let mut coloring = ColoringSolution::new(graph.num_vertices);
        let max_colors = 50; // Maximum color bound

        for (v, &param) in solution.iter().enumerate() {
            if v >= graph.num_vertices {
                break;
            }
            // Map continuous value to discrete color
            let color = ((param.abs() * max_colors as f32) as usize) % max_colors;
            coloring.colors[v] = color;
        }

        // Validate coloring and compute fitness
        let conflicts = coloring.validate(graph);
        coloring.compute_chromatic_number();
        let chromatic = coloring.chromatic_number;

        // Fitness = weighted combination of conflicts and chromatic number
        // Lower is better (minimization problem)
        conflicts as f32 * 1000.0 + chromatic as f32
    }

    /// Apply transfer entropy minimization (reserved for TE integration)
    #[allow(dead_code)]
    fn minimize_transfer_entropy(&self, solution: &[f32]) -> f32 {
        // Compute transfer entropy between solution components
        // This is a simplified version - real implementation would use
        // information-theoretic measures
        let mut te_sum = 0.0;
        for i in 1..solution.len() {
            let diff = (solution[i] - solution[i - 1]).abs();
            te_sum += diff * diff; // Quadratic penalty for differences
        }
        te_sum / solution.len() as f32
    }

    /// Update CMA-ES optimizer with new solutions and fitness values (reserved for batch updates)
    #[allow(dead_code)]
    fn update_optimizer(
        &mut self,
        solutions: Vec<Vec<f32>>,
        fitness_values: Vec<f32>,
    ) -> Result<(), PrismError> {
        // Track best solution
        if let Some((min_idx, &min_fitness)) = fitness_values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        {
            if min_fitness < self.best_fitness {
                self.best_fitness = min_fitness;
                self.best_solution = solutions[min_idx].clone();
            }
        }

        self.fitness_history.push(self.best_fitness);
        self.generation += 1;

        // TODO(GPU-CMA-04): Update covariance matrix on GPU
        // Optimizer state updated internally by step() method

        Ok(())
    }

    /// Compute convergence metric based on fitness history
    fn compute_convergence(&self) -> f32 {
        if self.fitness_history.len() < 10 {
            return 0.0; // Not enough history
        }

        // Check fitness improvement over last 10 generations
        let recent = &self.fitness_history[self.fitness_history.len() - 10..];
        let improvement = (recent[0] - recent[9]).abs() / recent[0].abs().max(1.0);

        // Convergence metric: 1.0 if no improvement, 0.0 if high improvement
        1.0 - improvement.min(1.0)
    }
}

/// Telemetry implementation for CMA-ES phase
pub struct CmaEsTelemetry {
    best_fitness: f32,
    mean_fitness: f32,
    fitness_std: f32,
    sigma: f32,
    generation: usize,
    condition_number: f32,
    convergence_metric: f32,
}

impl CmaEsTelemetry {
    fn from_controller(controller: &CmaEsPhaseController) -> Self {
        Self {
            best_fitness: controller.best_fitness,
            mean_fitness: if controller.fitness_history.is_empty() {
                f32::INFINITY
            } else {
                let sum: f32 = controller.fitness_history.iter().sum();
                sum / controller.fitness_history.len() as f32
            },
            fitness_std: if controller.fitness_history.len() > 1 {
                let mean = controller.fitness_history.iter().sum::<f32>()
                    / controller.fitness_history.len() as f32;
                let variance = controller
                    .fitness_history
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>()
                    / controller.fitness_history.len() as f32;
                variance.sqrt()
            } else {
                0.0
            },
            sigma: controller
                .optimizer
                .as_ref()
                .map_or(0.5, |opt| opt.get_state().sigma),
            generation: controller.generation,
            condition_number: 1.0, // Would be computed from covariance matrix
            convergence_metric: controller.compute_convergence(),
        }
    }
}

impl PhaseTelemetry for CmaEsTelemetry {
    fn metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        metrics.insert("best_fitness".to_string(), self.best_fitness as f64);
        metrics.insert("mean_fitness".to_string(), self.mean_fitness as f64);
        metrics.insert("fitness_std".to_string(), self.fitness_std as f64);
        metrics.insert("sigma".to_string(), self.sigma as f64);
        metrics.insert("generation".to_string(), self.generation as f64);
        metrics.insert("condition_number".to_string(), self.condition_number as f64);
        metrics.insert(
            "convergence_metric".to_string(),
            self.convergence_metric as f64,
        );
        metrics
    }
}

impl PhaseController for CmaEsPhaseController {
    fn execute(
        &mut self,
        graph: &Graph,
        context: &mut PhaseContext,
    ) -> Result<PhaseOutcome, PrismError> {
        log::info!("Executing CMA-ES Phase (Phase X) - Real Optimization");

        // Initialize optimizer if not already created
        if self.optimizer.is_none() {
            let dim = graph.num_vertices;
            self.optimizer = Some(cma_cpu::CmaOptimizer::new(
                dim,
                Some(self.config.population_size),
                self.config.initial_sigma,
            ));
            log::info!(
                "Initialized CMA-ES (CPU): dim={}, pop_size={}, sigma={}",
                dim,
                self.config.population_size,
                self.config.initial_sigma
            );
        }

        // Run optimization using real CMA-ES
        let max_iters = self.config.max_iterations.min(100); // Limit for demo

        // Clone graph for use in closure to avoid borrow conflicts
        let graph_clone = graph.clone();

        // Run CMA-ES optimization
        for iter in 0..max_iters {
            // Define fitness function for this iteration (captures graph_clone, not self)
            let fitness_fn = |solution: &[f32]| -> f32 {
                // Compute fitness directly inline to avoid self borrow
                let mut coloring = ColoringSolution::new(graph_clone.num_vertices);
                for (v, &param) in solution.iter().enumerate().take(graph_clone.num_vertices) {
                    let color = ((param.abs() * 50.0) as usize) % 50;
                    coloring.colors[v] = color;
                }
                let conflicts = coloring.validate(&graph_clone);
                coloring.compute_chromatic_number();
                let chromatic = coloring.chromatic_number;
                let fitness = conflicts as f32 * 1000.0 + chromatic as f32;

                // Simple TE penalty (simplified)
                let te_penalty = solution.iter().map(|x| x.abs()).sum::<f32>() * 0.01;
                fitness + te_penalty
            };

            // Perform one CMA-ES step
            self.optimizer
                .as_mut()
                .ok_or_else(|| {
                    PrismError::Internal("CMA-ES optimizer not initialized".to_string())
                })?
                .step(fitness_fn)
                .map_err(|e| PrismError::Internal(e.to_string()))?;

            // Get current state
            let state = self
                .optimizer
                .as_ref()
                .ok_or_else(|| {
                    PrismError::Internal("CMA-ES optimizer not initialized".to_string())
                })?
                .get_state();

            // Update our tracking
            self.best_fitness = state.best_fitness;
            self.best_solution = state.best_solution.clone();
            self.generation = state.generation;
            self.fitness_history.push(state.best_fitness);

            // Log progress every 10 generations
            if iter % 10 == 0 {
                log::info!(
                    "CMA-ES Gen {}: best_fitness={:.3}, mean={:.3}, sigma={:.4}",
                    self.generation,
                    self.best_fitness,
                    state.mean_fitness,
                    state.sigma
                );
            }

            // Check for convergence
            if self.compute_convergence() > 0.95 {
                log::info!("CMA-ES converged at generation {}", self.generation);
                break;
            }

            // Check target fitness if specified
            if let Some(target) = self.config.target_fitness {
                if self.best_fitness <= target {
                    log::info!(
                        "CMA-ES reached target fitness {} at generation {}",
                        target,
                        self.generation
                    );
                    break;
                }
            }
        }

        // Get final state from optimizer
        let cpu_state = self
            .optimizer
            .as_ref()
            .ok_or_else(|| PrismError::Internal("CMA-ES optimizer not initialized".to_string()))?
            .get_state();

        // Convert cma_cpu::CmaState to prism_core::CmaState
        let cma_state = prism_core::CmaState {
            best_solution: cpu_state.best_solution.clone(),
            best_fitness: cpu_state.best_fitness,
            covariance_condition: cpu_state.covariance_condition,
            generation: cpu_state.generation,
            convergence_metric: cpu_state.convergence_metric,
            mean_fitness: cpu_state.mean_fitness,
            fitness_std: cpu_state.fitness_std,
            sigma: cpu_state.sigma,
            effective_size: cpu_state.effective_size,
        };

        // Update context with CMA state
        context.update_cma_state(cma_state);

        // Update telemetry manually
        self.telemetry.best_fitness = self.best_fitness;
        self.telemetry.generation = self.generation;
        if !self.fitness_history.is_empty() {
            let sum: f32 = self.fitness_history.iter().sum();
            self.telemetry.mean_fitness = sum / self.fitness_history.len() as f32;

            if self.fitness_history.len() > 1 {
                let mean = self.telemetry.mean_fitness;
                let variance = self
                    .fitness_history
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>()
                    / self.fitness_history.len() as f32;
                self.telemetry.fitness_std = variance.sqrt();
            }
        }
        if let Some(ref opt) = self.optimizer {
            self.telemetry.sigma = opt.get_state().sigma;
        }
        self.telemetry.convergence_metric = self.compute_convergence();

        // Apply optimized solution to improve graph coloring
        if !self.best_solution.is_empty() {
            let mut improved_solution = ColoringSolution::new(graph.num_vertices);
            let max_colors = 50;

            for (v, &param) in self.best_solution.iter().enumerate() {
                if v >= graph.num_vertices {
                    break;
                }
                let color = ((param.abs() * max_colors as f32) as usize) % max_colors;
                improved_solution.colors[v] = color;
            }

            let conflicts = improved_solution.validate(graph);
            improved_solution.conflicts = conflicts;
            improved_solution.compute_chromatic_number();

            log::info!(
                "CMA-ES improved solution: {} colors, {} conflicts",
                improved_solution.chromatic_number,
                improved_solution.conflicts
            );

            // Update best solution if improved
            context.update_best_solution(improved_solution);
        }

        // Create telemetry
        let telemetry = CmaEsTelemetry::from_controller(self);
        let mut telemetry_map = HashMap::new();
        for (key, value) in telemetry.metrics() {
            telemetry_map.insert(key, serde_json::json!(value));
        }

        Ok(PhaseOutcome::Success {
            message: format!(
                "CMA-ES completed: {} generations, best_fitness={:.3}, convergence={:.3}",
                self.generation,
                self.best_fitness,
                self.compute_convergence()
            ),
            telemetry: telemetry_map,
        })
    }

    fn name(&self) -> &'static str {
        "PhaseX-CMA"
    }

    fn telemetry(&self) -> &dyn PhaseTelemetry {
        &self.telemetry
    }
}

/// CMA-ES optimization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmaEsMetrics {
    pub best_fitness: f32,
    pub mean_fitness: f32,
    pub generation: usize,
    pub sigma: f32,
    pub condition_number: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cma_es_config_validation() {
        let mut config = CmaEsConfig::default();
        let controller = CmaEsPhaseController::new(config.clone());
        assert!(controller.validate_config().is_ok());

        config.population_size = 2;
        let controller = CmaEsPhaseController::new(config);
        assert!(controller.validate_config().is_err());
    }

    #[test]
    fn test_cma_es_real_optimization() {
        // Create a small test graph
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 0);
        graph.add_edge(0, 2); // Add diagonal

        // Configure CMA-ES with small population for fast test
        let config = CmaEsConfig {
            population_size: 10,
            initial_sigma: 0.5,
            max_iterations: 20,
            target_fitness: None,
            use_gpu: false,
        };

        // Create controller and context
        let mut controller = CmaEsPhaseController::new(config);
        let mut context = PhaseContext::new();

        // Execute optimization
        let outcome = controller
            .execute(&graph, &mut context)
            .expect("Test execution should succeed");
        assert!(outcome.is_success());

        // Verify CMA state was created
        let cma_state = context.get_cma_state().expect("CMA state should exist");

        // Verify optimization ran
        assert!(cma_state.generation > 0);
        assert!(cma_state.best_fitness < f32::INFINITY);
        assert_eq!(cma_state.best_solution.len(), 4);

        // Print results for demonstration
        println!("\nCMA-ES Test Results:");
        println!("  Generations: {}", cma_state.generation);
        println!("  Best fitness: {:.3}", cma_state.best_fitness);
        println!("  Convergence: {:.3}", cma_state.convergence_metric);
    }
}
