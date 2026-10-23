//! Core traits for the PRISM pipeline architecture.
//!
//! Implements PRISM GPU Plan §2.2: Core Traits.

use crate::errors::PrismError;
use crate::types::{CmaState, ColoringSolution, Graph};
use std::collections::HashMap;
use std::sync::Arc;

/// Outcome of a phase execution.
///
/// Implements the Phase Transition Policy (PRISM GPU Plan §4.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseOutcome {
    /// Phase completed successfully
    Success {
        /// Success message describing what was accomplished
        message: String,
        /// Telemetry data from the phase execution
        telemetry: HashMap<String, serde_json::Value>,
    },

    /// Phase should be retried with adjusted parameters
    Retry {
        /// Reason for retry
        reason: String,
        /// Suggested wait time before retry (milliseconds)
        backoff_ms: u64,
    },

    /// Escalate to next phase (this phase failed irrecoverably)
    Escalate {
        /// Reason for escalation
        reason: String,
    },
}

impl PhaseOutcome {
    /// Creates a Success outcome.
    pub fn success() -> Self {
        PhaseOutcome::Success {
            message: "Phase completed successfully".to_string(),
            telemetry: HashMap::new(),
        }
    }

    /// Creates a Retry outcome with exponential backoff.
    pub fn retry(reason: impl Into<String>, attempt: u32) -> Self {
        let backoff_ms = 100 * 2u64.pow(attempt.min(10));
        PhaseOutcome::Retry {
            reason: reason.into(),
            backoff_ms,
        }
    }

    /// Creates an Escalate outcome.
    pub fn escalate(reason: impl Into<String>) -> Self {
        PhaseOutcome::Escalate {
            reason: reason.into(),
        }
    }

    /// Checks if the outcome is successful.
    pub fn is_success(&self) -> bool {
        matches!(self, PhaseOutcome::Success { .. })
    }
}

/// Execution context shared across all phases.
///
/// Implements PRISM GPU Plan §2.2: PhaseContext.
pub struct PhaseContext {
    /// Current best coloring solution (updated across phases)
    pub best_solution: Option<ColoringSolution>,

    /// CHECKPOINT: Best solution with ZERO conflicts (locked once achieved)
    /// Prevents downstream phases from expanding colors unnecessarily
    pub checkpoint_zero_conflicts: Option<ColoringSolution>,

    /// GPU context handle (opaque, managed by prism-gpu)
    /// TODO(GPU-Context): Initialize CudaContext and load PTX modules
    /// Stored as Arc<dyn Any> to avoid circular dependencies with prism-gpu
    pub gpu_context: Option<Arc<dyn std::any::Any + Send + Sync>>,

    /// Reinforcement learning state (managed by prism-fluxnet)
    pub rl_state: Option<Box<dyn std::any::Any>>,

    /// Phase-specific scratch space (for inter-phase communication)
    pub scratch: HashMap<String, Box<dyn std::any::Any>>,

    /// Global iteration counter
    pub iteration: usize,

    /// Geometric stress telemetry for metaphysical coupling.
    ///
    /// Populated by Phase 4 (Geodesic) and Phase 6 (TDA) after computing
    /// geometric/topological metrics. Read by Phase 1/2/3/7 to adjust
    /// parameters in response to geometric stress.
    ///
    /// When stress_scalar > 0.5, phases react:
    /// - Phase 1: Increase active inference exploration (prediction error)
    /// - Phase 2: Raise thermodynamic temperature (reheat)
    /// - Phase 3: Adjust quantum coupling strength
    /// - Phase 7: Intensify memetic local search
    ///
    /// Implements Metaphysical Telemetry Coupling feedback loop.
    pub geometry_metrics: Option<crate::types::GeometryTelemetry>,

    /// Previous chromatic number (for geometry telemetry growth rate tracking)
    pub previous_chromatic: Option<usize>,

    /// Flag indicating WHCR will be invoked after this phase
    /// When true, allows conflicted solutions to pass through checkpoint
    pub whcr_pending: bool,
}

impl PhaseContext {
    /// Creates a new empty phase context.
    pub fn new() -> Self {
        Self {
            best_solution: None,
            checkpoint_zero_conflicts: None,
            gpu_context: None,
            rl_state: None,
            scratch: HashMap::new(),
            iteration: 0,
            geometry_metrics: None,
            previous_chromatic: None,
            whcr_pending: false,
        }
    }

    /// Updates the best solution if the new solution is better.
    ///
    /// CHECKPOINT LOCKING LOGIC:
    /// - Once a 0-conflict solution is found, it's locked as checkpoint_zero_conflicts
    /// - Downstream phases can only improve upon checkpoint by:
    /// * Reducing colors while maintaining 0 conflicts
    /// * Achieving same colors with 0 conflicts
    /// - Phases cannot expand colors if checkpoint exists with 0 conflicts
    pub fn update_best_solution(&mut self, solution: ColoringSolution) {
        // Check if we have a zero-conflict checkpoint
        if let Some(ref checkpoint) = self.checkpoint_zero_conflicts {
            // WHCR bypass: Allow conflicted solutions if WHCR will be invoked
            if solution.conflicts > 0 && !self.whcr_pending {
                log::warn!(
 "CHECKPOINT LOCK: Rejecting {} colors, {} conflicts (checkpoint: {} colors, 0 conflicts)",
 solution.chromatic_number, solution.conflicts, checkpoint.chromatic_number
 );
                return;
            } else if solution.conflicts > 0 && self.whcr_pending {
                log::info!(
 "WHCR BYPASS: Allowing {} colors, {} conflicts for WHCR repair (checkpoint: {} colors)",
 solution.chromatic_number, solution.conflicts, checkpoint.chromatic_number
 );
            }

            // If new solution has 0 conflicts, allow update only if it's an improvement
            if solution.chromatic_number < checkpoint.chromatic_number {
                log::info!(
                    "CHECKPOINT IMPROVEMENT: {} colors → {} colors (0 conflicts locked)",
                    checkpoint.chromatic_number,
                    solution.chromatic_number
                );
                self.checkpoint_zero_conflicts = Some(solution.clone());
                self.best_solution = Some(solution);
            } else if solution.chromatic_number == checkpoint.chromatic_number {
                // Same colors, same conflicts - no need to update
                log::debug!(
                    "Checkpoint maintained at {} colors, 0 conflicts",
                    checkpoint.chromatic_number
                );
            } else {
                // More colors than checkpoint - reject
                log::warn!(
                    "CHECKPOINT LOCK: Rejecting {} colors (checkpoint: {} colors, 0 conflicts)",
                    solution.chromatic_number,
                    checkpoint.chromatic_number
                );
            }
            return;
        }

        // No checkpoint yet - use standard comparison logic
        let should_update = match &self.best_solution {
            None => true,
            Some(current) => {
                solution.conflicts < current.conflicts
                    || (solution.conflicts == current.conflicts
                        && solution.chromatic_number < current.chromatic_number)
            }
        };

        if should_update {
            // Track previous chromatic number for geometry telemetry
            if let Some(ref current) = self.best_solution {
                self.previous_chromatic = Some(current.chromatic_number);
            }

            // If this solution has 0 conflicts, lock it as checkpoint
            if solution.conflicts == 0 {
                log::info!(
                    "🔒 ZERO-CONFLICT CHECKPOINT LOCKED: {} colors, 0 conflicts",
                    solution.chromatic_number
                );
                self.checkpoint_zero_conflicts = Some(solution.clone());
            }

            self.best_solution = Some(solution);
        }
    }

    /// Updates geometry metrics (called by Phase 4/6 after computing geometry).
    ///
    /// # Arguments
    /// * `metrics` - New geometry telemetry to propagate to downstream phases
    ///
    /// # Side Effects
    /// - Logs coupling decisions if stress thresholds are crossed
    pub fn update_geometry_metrics(&mut self, metrics: crate::types::GeometryTelemetry) {
        if metrics.is_critical_stress() {
            log::warn!(
 "Geometry telemetry: CRITICAL stress detected (stress_scalar={:.3}, overlap_density={:.3})",
 metrics.stress_scalar,
 metrics.overlap_density
 );
        } else if metrics.is_high_stress() {
            log::info!(
                "Geometry telemetry: High stress detected (stress_scalar={:.3}), coupling enabled",
                metrics.stress_scalar
            );
        } else {
            log::debug!(
                "Geometry telemetry: Normal stress (stress_scalar={:.3})",
                metrics.stress_scalar
            );
        }

        self.geometry_metrics = Some(metrics);
    }

    /// Retrieves current geometry stress level (0.0 = no stress, 1.0 = maximum).
    ///
    /// Returns 0.0 if geometry metrics are not yet available.
    pub fn geometry_stress_level(&self) -> f32 {
        self.geometry_metrics
            .as_ref()
            .map(|m| m.stress_scalar)
            .unwrap_or(0.0)
    }

    /// Checks if geometric stress requires phase intervention.
    pub fn has_high_geometry_stress(&self) -> bool {
        self.geometry_metrics
            .as_ref()
            .map(|m| m.is_high_stress())
            .unwrap_or(false)
    }

    // === Dendritic Reservoir Metrics (Phase 0 → Phases 1-7 Coupling) ===

    /// Returns per-vertex difficulty metrics from Phase 0 dendritic reservoir.
    ///
    /// High difficulty vertices (close to 1.0) are hard to color and need
    /// more computational effort. Downstream phases use this for:
    /// - Phase 1: Increase prediction error tolerance
    /// - Phase 2: Start at higher temperature
    /// - Phase 4: Weight geodesic distance computation
    /// - Phase 7: Focus local search on difficult regions
    pub fn dendritic_difficulty(&self) -> Option<&Vec<f32>> {
        self.scratch
            .get("phase0_difficulty")
            .and_then(|v| v.downcast_ref::<Vec<f32>>())
    }

    /// Returns per-vertex uncertainty metrics from Phase 0 dendritic reservoir.
    ///
    /// High uncertainty vertices (close to 1.0) have unpredictable neighborhood
    /// structure. Downstream phases use this for:
    /// - Phase 3: Increase quantum tunneling probability
    /// - Phase 6: Adjust persistence thresholds in TDA
    pub fn dendritic_uncertainty(&self) -> Option<&Vec<f32>> {
        self.scratch
            .get("phase0_uncertainty")
            .and_then(|v| v.downcast_ref::<Vec<f32>>())
    }

    /// Returns Phase 0 telemetry summary (mean/variance/entropy stats).
    pub fn dendritic_telemetry(&self) -> Option<&crate::types::Phase0Telemetry> {
        self.scratch
            .get("phase0_telemetry")
            .and_then(|v| v.downcast_ref::<crate::types::Phase0Telemetry>())
    }

    /// Returns mean difficulty across all vertices (0.5 if not computed).
    ///
    /// Quick access for phases that don't need per-vertex granularity.
    pub fn mean_difficulty(&self) -> f32 {
        self.dendritic_telemetry()
            .map(|t| t.difficulty_mean)
            .unwrap_or(0.5)
    }

    /// Returns mean uncertainty across all vertices (0.5 if not computed).
    ///
    /// Quick access for phases that don't need per-vertex granularity.
    pub fn mean_uncertainty(&self) -> f32 {
        self.dendritic_telemetry()
            .map(|t| t.uncertainty_mean)
            .unwrap_or(0.5)
    }

    /// Returns difficulty for a specific vertex (0.5 default if not available).
    pub fn vertex_difficulty(&self, vertex: usize) -> f32 {
        self.dendritic_difficulty()
            .and_then(|d| d.get(vertex))
            .copied()
            .unwrap_or(0.5)
    }

    /// Returns uncertainty for a specific vertex (0.5 default if not available).
    pub fn vertex_uncertainty(&self, vertex: usize) -> f32 {
        self.dendritic_uncertainty()
            .and_then(|u| u.get(vertex))
            .copied()
            .unwrap_or(0.5)
    }

    /// Checks if dendritic reservoir metrics are available.
    pub fn has_dendritic_metrics(&self) -> bool {
        self.scratch.contains_key("phase0_difficulty")
    }

    /// Sets the WHCR pending flag, allowing conflicted solutions to bypass checkpoint.
    pub fn set_whcr_pending(&mut self, pending: bool) {
        self.whcr_pending = pending;
        if pending {
            log::debug!(
                "WHCR pending flag set - checkpoint bypass enabled for conflicted solutions"
            );
        } else {
            log::debug!("WHCR pending flag cleared - checkpoint enforcement restored");
        }
    }

    /// Checks if WHCR is pending for the current phase.
    pub fn is_whcr_pending(&self) -> bool {
        self.whcr_pending
    }

    /// Checks if a solution is allowed under checkpoint lock.
    ///
    /// Returns:
    /// - Ok(true) if solution is allowed (no checkpoint or solution improves upon it)
    /// - Ok(false) if solution violates checkpoint lock (should be rejected)
    pub fn is_solution_allowed(&self, solution: &ColoringSolution) -> bool {
        if let Some(ref checkpoint) = self.checkpoint_zero_conflicts {
            // If checkpoint exists, only allow solutions that:
            // 1. Have fewer colors (with 0 conflicts), OR
            // 2. Have same colors with 0 conflicts (tied)
            // 3. NEVER allow solutions with conflicts if checkpoint has 0
            if solution.conflicts > 0 {
                return false; // Checkpoint has 0 conflicts, new has conflicts → REJECT
            }
            if solution.chromatic_number > checkpoint.chromatic_number {
                return false; // Checkpoint has fewer colors → REJECT expansion
            }
            true // Either same colors or fewer colors → ALLOW
        } else {
            true // No checkpoint → always allow
        }
    }

    /// Gets the best color/conflict state that must be maintained.
    pub fn get_checkpoint(&self) -> Option<(usize, usize)> {
        self.checkpoint_zero_conflicts
            .as_ref()
            .map(|s| (s.chromatic_number, s.conflicts))
    }

    /// Sets a metadata value in the scratch space.
    pub fn set_metadata(&mut self, key: &str, value: serde_json::Value) {
        self.scratch.insert(key.to_string(), Box::new(value));
    }

    /// Gets a metadata value from the scratch space.
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.scratch.get(key).and_then(|v| v.downcast_ref())
    }

    // === Subsystem State Update Methods (PRISM GPU Plan §7.3) ===

    /// Updates ontology state from semantic grounding phase.
    ///
    /// Called by OntologyPhaseController after semantic analysis.
    pub fn update_ontology_state(&mut self, semantic_conflicts: u32, coherence_score: f64) {
        self.set_metadata("ontology_conflicts", serde_json::json!(semantic_conflicts));
        self.set_metadata("ontology_coherence", serde_json::json!(coherence_score));
        log::debug!(
            "Ontology state updated: conflicts={}, coherence={:.3}",
            semantic_conflicts,
            coherence_score
        );
    }

    /// Updates MEC (Molecular Emergent Computing) state.
    ///
    /// Called by MecPhaseController after molecular simulation.
    pub fn update_mec_state(&mut self, free_energy: f64, entropy: f64, temperature: f64) {
        self.set_metadata("mec_free_energy", serde_json::json!(free_energy));
        self.set_metadata("mec_entropy", serde_json::json!(entropy));
        self.set_metadata("mec_temperature", serde_json::json!(temperature));
        log::debug!(
            "MEC state updated: free_energy={:.3}, entropy={:.3}, temp={}K",
            free_energy,
            entropy,
            temperature
        );
    }

    /// Updates CMA-ES optimization state.
    ///
    /// Called by CmaEsPhaseController after transfer entropy computation.
    pub fn update_cma_state(&mut self, cma_state: CmaState) {
        // Store individual metrics for telemetry
        self.set_metadata(
            "cma_best_fitness",
            serde_json::json!(cma_state.best_fitness),
        );
        self.set_metadata(
            "cma_covariance_condition",
            serde_json::json!(cma_state.covariance_condition),
        );
        self.set_metadata("cma_generation", serde_json::json!(cma_state.generation));
        self.set_metadata(
            "cma_convergence_metric",
            serde_json::json!(cma_state.convergence_metric),
        );

        // Store best solution vector
        self.set_metadata(
            "cma_best_solution",
            serde_json::json!(cma_state.best_solution),
        );

        // Store full state for retrieval
        self.scratch
            .insert("cma_state".to_string(), Box::new(cma_state.clone()));

        log::info!(
            "CMA-ES state updated: gen={}, best_fitness={:.6}, condition={:.2e}, convergence={:.4}",
            cma_state.generation,
            cma_state.best_fitness,
            cma_state.covariance_condition,
            cma_state.convergence_metric
        );
    }

    /// Retrieves the current CMA-ES state if available.
    pub fn get_cma_state(&self) -> Option<&CmaState> {
        self.scratch
            .get("cma_state")
            .and_then(|v| v.downcast_ref::<CmaState>())
    }

    /// Checks if CMA-ES optimization has converged.
    ///
    /// Returns false if CMA-ES state is not available.
    pub fn is_cma_converged(&self) -> bool {
        self.get_cma_state()
            .map(|s| s.convergence_metric > 0.95 || s.best_fitness < 1.0)
            .unwrap_or(false)
    }

    /// Updates biomolecular prediction state.
    ///
    /// Called by BiomolecularAdapter after structure prediction.
    pub fn update_biomolecular_state(&mut self, confidence: f64, rmsd: f64, binding_affinity: f64) {
        self.set_metadata("bio_confidence", serde_json::json!(confidence));
        self.set_metadata("bio_rmsd", serde_json::json!(rmsd));
        self.set_metadata("bio_binding_affinity", serde_json::json!(binding_affinity));
        log::debug!(
            "Biomolecular state updated: confidence={:.3}, RMSD={:.3}, affinity={:.3}",
            confidence,
            rmsd,
            binding_affinity
        );
    }

    /// Updates materials prediction state.
    ///
    /// Called by MaterialsAdapter after property prediction.
    pub fn update_materials_state(&mut self, band_gap: f64, formation_energy: f64, stability: f64) {
        self.set_metadata("mat_band_gap", serde_json::json!(band_gap));
        self.set_metadata("mat_formation_energy", serde_json::json!(formation_energy));
        self.set_metadata("mat_stability", serde_json::json!(stability));
        log::debug!(
            "Materials state updated: band_gap={:.3}eV, formation_E={:.3}, stability={:.3}",
            band_gap,
            formation_energy,
            stability
        );
    }

    /// Updates GNN (Graph Neural Network) state.
    ///
    /// Called by GNN phase after graph embedding computation.
    pub fn update_gnn_state(&mut self, state: crate::domain::GnnState) {
        self.set_metadata(
            "gnn_predicted_chromatic",
            serde_json::json!(state.predicted_chromatic),
        );
        self.set_metadata("gnn_embedding_dim", serde_json::json!(state.embedding_dim));
        self.set_metadata("gnn_confidence", serde_json::json!(state.confidence));
        self.set_metadata(
            "gnn_manifold_dimension",
            serde_json::json!(state.manifold_dimension),
        );
        self.set_metadata(
            "gnn_manifold_curvature",
            serde_json::json!(state.manifold_curvature),
        );
        self.set_metadata(
            "gnn_geodesic_complexity",
            serde_json::json!(state.geodesic_complexity),
        );
        self.set_metadata("gnn_betti_count", serde_json::json!(state.betti_count));
        self.set_metadata("gnn_model_type", serde_json::json!(state.model_type));
        log::debug!(
            "GNN state updated: chromatic={}, dim={}, confidence={:.3}, manifold_dim={:.2}",
            state.predicted_chromatic,
            state.embedding_dim,
            state.confidence,
            state.manifold_dimension
        );
    }

    /// Updates molecular dynamics state.
    ///
    /// Called by MD simulation components.
    pub fn update_md_state(&mut self, timesteps: usize, energy_drift: f64, temperature_avg: f64) {
        self.set_metadata("md_timesteps", serde_json::json!(timesteps));
        self.set_metadata("md_energy_drift", serde_json::json!(energy_drift));
        self.set_metadata("md_temperature_avg", serde_json::json!(temperature_avg));
        log::debug!(
            "MD state updated: timesteps={}, drift={:.3}, temp_avg={:.3}K",
            timesteps,
            energy_drift,
            temperature_avg
        );
    }
}

impl Default for PhaseContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Controller for a single phase in the PRISM pipeline.
///
/// Implements PRISM GPU Plan §2.2: PhaseController trait.
pub trait PhaseController: Send + Sync {
    /// Executes this phase on the given graph.
    ///
    /// ## Parameters
    /// - `graph`: Input graph to color
    /// - `context`: Shared execution context (mutable for state updates)
    ///
    /// ## Returns
    /// - `PhaseOutcome`: Success, Retry, or Escalate
    ///
    /// ## Errors
    /// Returns `PrismError` if the phase encounters an unrecoverable error.
    fn execute(
        &mut self,
        graph: &Graph,
        context: &mut PhaseContext,
    ) -> Result<PhaseOutcome, PrismError>;

    /// Returns the human-readable name of this phase.
    fn name(&self) -> &'static str;

    /// Returns telemetry interface for this phase.
    fn telemetry(&self) -> &dyn PhaseTelemetry;
}

/// Telemetry interface for phase monitoring.
///
/// Implements PRISM GPU Plan §2.2: PhaseTelemetry trait.
pub trait PhaseTelemetry {
    /// Returns current phase metrics as key-value pairs.
    ///
    /// Common metrics:
    /// - `temperature`: Thermodynamic temperature
    /// - `entropy`: Reservoir entropy
    /// - `convergence`: Convergence metric (0.0 - 1.0)
    /// - `gpu_utilization`: GPU utilization (0.0 - 1.0)
    fn metrics(&self) -> HashMap<String, f64>;

    /// Returns a human-readable status string.
    fn status(&self) -> String {
        format!("{} metrics available", self.metrics().len())
    }
}

/// Phase executor trait (for advanced phase management).
///
/// This trait extends `PhaseController` with lifecycle hooks.
pub trait PhaseExecutor: PhaseController {
    /// Called once before the first execution.
    fn initialize(&mut self, _context: &mut PhaseContext) -> Result<(), PrismError> {
        Ok(())
    }

    /// Called once after all executions complete.
    fn finalize(&mut self, _context: &mut PhaseContext) -> Result<(), PrismError> {
        Ok(())
    }

    /// Estimates the cost (in arbitrary units) of executing this phase.
    ///
    /// Used by the orchestrator for scheduling and load balancing.
    fn estimated_cost(&self, graph: &Graph) -> f64 {
        graph.num_vertices as f64 * graph.num_edges as f64
    }
}

/// Trait for pluggable warmstart prior contributors.
///
/// Enables custom prior sources beyond built-in methods (flux reservoir,
/// structural anchors). Examples: degree-based priors, clustering-based priors,
/// ML model predictions, external solver hints.
///
/// ## Example
/// ```rust,ignore
/// use prism_core::{WarmstartContributor, Graph, WarmstartConfig, WarmstartPrior};
///
/// struct DegreeBasedContributor;
///
/// impl WarmstartContributor for DegreeBasedContributor {
/// fn contribute(&self, graph: &Graph, config: &WarmstartConfig)
/// -> Result<Vec<WarmstartPrior>, PrismError> {
/// // Generate priors based on vertex degrees
/// Ok(vec![])
/// }
///
/// fn name(&self) -> &'static str { "degree_based" }
/// }
/// ```
///
/// Implements Warmstart Plan Step 3: Extensible Prior Sources
pub trait WarmstartContributor: Send + Sync {
    /// Generates warmstart priors for the given graph.
    ///
    /// # Arguments
    /// * `graph` - Input graph
    /// * `config` - Warmstart configuration (max_colors, min_prob, etc.)
    ///
    /// # Returns
    /// Vector of WarmstartPrior (one per vertex), or error if generation fails.
    ///
    /// # Requirements
    /// - Must return exactly `graph.num_vertices` priors
    /// - Each prior must have valid probability distribution (sum = 1.0)
    /// - Probabilities must be >= config.min_prob
    fn contribute(
        &self,
        graph: &Graph,
        config: &crate::types::WarmstartConfig,
    ) -> Result<Vec<crate::types::WarmstartPrior>, PrismError>;

    /// Returns contributor name for telemetry/logging.
    ///
    /// Used in telemetry events and debug logging to identify prior source.
    fn name(&self) -> &'static str;

    /// Returns contributor weight for ensemble fusion.
    ///
    /// Weight determines influence in weighted averaging during fusion.
    /// Default: 1.0 (equal weight with other contributors)
    ///
    /// # Range
    /// Must be in (0.0, 1.0]. Weights are normalized before fusion.
    fn weight(&self) -> f32 {
        1.0
    }

    /// Returns whether contributor requires GPU acceleration.
    ///
    /// Used by orchestrator for scheduling (GPU contributors run sequentially
    /// to avoid resource contention).
    ///
    /// Default: false (CPU-only)
    fn requires_gpu(&self) -> bool {
        false
    }

    /// Returns estimated execution time in milliseconds.
    ///
    /// Used for performance profiling and scheduling decisions.
    /// Default: 0.0 (unknown)
    fn estimated_time_ms(&self, graph: &Graph) -> f64 {
        let _ = graph; // Suppress unused warning
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_outcome() {
        let success = PhaseOutcome::success();
        assert!(success.is_success());

        let retry = PhaseOutcome::retry("Too hot", 2);
        assert!(!retry.is_success());
        if let PhaseOutcome::Retry { backoff_ms, .. } = retry {
            assert_eq!(backoff_ms, 400); // 100 * 2^2
        }

        let escalate = PhaseOutcome::escalate("Failed convergence");
        assert!(!escalate.is_success());
    }

    #[test]
    fn test_phase_context_update() {
        let mut context = PhaseContext::new();

        let mut solution1 = ColoringSolution::new(10);
        solution1.chromatic_number = 5;
        solution1.conflicts = 0;

        context.update_best_solution(solution1.clone());
        assert_eq!(context.best_solution.as_ref().unwrap().chromatic_number, 5);

        let mut solution2 = ColoringSolution::new(10);
        solution2.chromatic_number = 4; // Better!
        solution2.conflicts = 0;

        context.update_best_solution(solution2.clone());
        assert_eq!(context.best_solution.as_ref().unwrap().chromatic_number, 4);
    }
}
