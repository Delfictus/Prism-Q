//! Core data types for graph coloring and PRISM pipeline.
//!
//! Implements PRISM GPU Plan §2.1: Core Data Types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Graph representation with adjacency structure.
///
/// Supports both sparse (CSR) and dense representations.
/// Designed for efficient GPU memory transfer.
///
/// ## Memory Layout
/// - Row-major adjacency matrix for dense graphs
/// - CSR (Compressed Sparse Row) for sparse graphs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    /// Number of vertices in the graph
    pub num_vertices: usize,

    /// Number of edges in the graph
    pub num_edges: usize,

    /// Adjacency list representation: vertex -> list of neighbors
    pub adjacency: Vec<Vec<usize>>,

    /// Optional: Vertex degrees (cached for performance)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub degrees: Option<Vec<usize>>,

    /// Optional: Edge weights (for weighted graph coloring)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edge_weights: Option<HashMap<(usize, usize), f64>>,
}

impl Graph {
    /// Creates a new empty graph with the specified number of vertices.
    pub fn new(num_vertices: usize) -> Self {
        Self {
            num_vertices,
            num_edges: 0,
            adjacency: vec![Vec::new(); num_vertices],
            degrees: None,
            edge_weights: None,
        }
    }

    /// Adds an undirected edge between two vertices.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        if u < self.num_vertices && v < self.num_vertices && u != v {
            self.adjacency[u].push(v);
            self.adjacency[v].push(u);
            self.num_edges += 1;
        }
    }

    /// Computes and caches vertex degrees.
    pub fn compute_degrees(&mut self) {
        self.degrees = Some(
            self.adjacency
                .iter()
                .map(|neighbors| neighbors.len())
                .collect(),
        );
    }

    /// Returns the degree of a vertex.
    pub fn degree(&self, vertex: usize) -> usize {
        if let Some(ref degrees) = self.degrees {
            degrees[vertex]
        } else {
            self.adjacency[vertex].len()
        }
    }

    /// Graph density: |E| / (|V| * (|V| - 1) / 2)
    pub fn density(&self) -> f64 {
        if self.num_vertices <= 1 {
            return 0.0;
        }
        (2.0 * self.num_edges as f64) / (self.num_vertices * (self.num_vertices - 1)) as f64
    }
}

/// Solution to a graph coloring problem.
///
/// Implements PRISM GPU Plan §2.1: ColoringSolution type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColoringSolution {
    /// Color assignment for each vertex (vertex index -> color)
    pub colors: Vec<usize>,

    /// Chromatic number (number of distinct colors used)
    pub chromatic_number: usize,

    /// Number of edge conflicts (0 = valid coloring)
    pub conflicts: usize,

    /// Quality score (higher = better, algorithm-specific)
    pub quality_score: f64,

    /// Computation time in milliseconds
    pub computation_time_ms: f64,
}

impl ColoringSolution {
    /// Creates a new empty solution.
    pub fn new(num_vertices: usize) -> Self {
        Self {
            colors: vec![0; num_vertices],
            chromatic_number: 0,
            conflicts: 0,
            quality_score: 0.0,
            computation_time_ms: 0.0,
        }
    }

    /// Validates the solution against the graph.
    ///
    /// Returns the number of conflicts (edges where both endpoints have the same color).
    pub fn validate(&self, graph: &Graph) -> usize {
        let mut conflicts = 0;
        for (u, neighbors) in graph.adjacency.iter().enumerate() {
            for &v in neighbors {
                if u < v && self.colors[u] == self.colors[v] {
                    conflicts += 1;
                }
            }
        }
        conflicts
    }

    /// Checks if the solution is valid (no conflicts).
    pub fn is_valid(&self) -> bool {
        self.conflicts == 0
    }

    /// Computes the chromatic number (number of distinct colors used).
    pub fn compute_chromatic_number(&mut self) {
        let mut color_set = std::collections::HashSet::new();
        for &color in &self.colors {
            color_set.insert(color);
        }
        self.chromatic_number = color_set.len();
    }

    /// Creates a solution from a color vector (for memetic algorithm).
    ///
    /// Initializes a ColoringSolution with the given colors, setting
    /// chromatic_number and conflicts to 0 (call recompute_metrics() to update).
    pub fn from_colors(colors: Vec<usize>) -> Self {
        Self {
            colors,
            chromatic_number: 0,
            conflicts: 0,
            quality_score: 0.0,
            computation_time_ms: 0.0,
        }
    }

    /// Recomputes chromatic number and conflicts from current color assignment.
    ///
    /// Used by memetic algorithm after crossover/mutation to update metrics.
    pub fn recompute_metrics(&mut self, adjacency: &[Vec<usize>]) {
        // Compute chromatic number
        let mut color_set = std::collections::HashSet::new();
        for &color in &self.colors {
            if color > 0 {
                // Skip uncolored vertices (color 0)
                color_set.insert(color);
            }
        }
        self.chromatic_number = color_set.len();

        // Compute conflicts
        let mut conflicts = 0;
        for (u, neighbors) in adjacency.iter().enumerate() {
            for &v in neighbors {
                if u < v && self.colors[u] == self.colors[v] && self.colors[u] > 0 {
                    conflicts += 1;
                }
            }
        }
        self.conflicts = conflicts;
    }
}

/// Configuration for a single phase in the PRISM pipeline.
///
/// Implements PRISM GPU Plan §2.1: PhaseConfig type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseConfig {
    /// Phase name (e.g., "Phase0-DendriticReservoir")
    pub name: String,

    /// Enable GPU acceleration for this phase
    pub gpu_enabled: bool,

    /// Maximum number of iterations
    pub max_iterations: usize,

    /// Convergence threshold
    pub convergence_threshold: f64,

    /// Phase-specific parameters (flexible key-value storage)
    #[serde(flatten)]
    pub parameters: HashMap<String, serde_json::Value>,
}

impl PhaseConfig {
    /// Creates a new phase configuration with default values.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            gpu_enabled: true,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            parameters: HashMap::new(),
        }
    }

    /// Sets a phase-specific parameter.
    pub fn set_parameter<T: Serialize>(&mut self, key: impl Into<String>, value: T) {
        self.parameters
            .insert(key.into(), serde_json::to_value(value).unwrap());
    }

    /// Gets a phase-specific parameter.
    pub fn get_parameter<T: for<'de> Deserialize<'de>>(&self, key: &str) -> Option<T> {
        self.parameters
            .get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }
}

// ============================================================================
// Warmstart Types (PRISM GPU Plan §6: Warmstart Upgrade)
// ============================================================================

/// Vertex ID type alias for clarity.
pub type VertexId = usize;

/// Warmstart prior for a single vertex.
///
/// Encodes probabilistic color preferences and structural anchors
/// derived from Phase 0 (Flux Reservoir), ensemble methods, and curriculum learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmstartPrior {
    /// Vertex identifier
    pub vertex: VertexId,

    /// Probability distribution over colors (length = max_colors)
    /// Sum should equal 1.0 after normalization
    pub color_probabilities: Vec<f32>,

    /// Whether this vertex is a structural anchor (deterministic assignment)
    pub is_anchor: bool,

    /// Optional: Anchor color (if is_anchor = true)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anchor_color: Option<usize>,
}

impl WarmstartPrior {
    /// Creates a new uniform prior for a vertex.
    pub fn uniform(vertex: VertexId, max_colors: usize) -> Self {
        let prob = 1.0 / max_colors as f32;
        Self {
            vertex,
            color_probabilities: vec![prob; max_colors],
            is_anchor: false,
            anchor_color: None,
        }
    }

    /// Creates an anchor prior with deterministic color assignment.
    pub fn anchor(vertex: VertexId, color: usize, max_colors: usize) -> Self {
        let mut probs = vec![0.0; max_colors];
        if color < max_colors {
            probs[color] = 1.0;
        }
        Self {
            vertex,
            color_probabilities: probs,
            is_anchor: true,
            anchor_color: Some(color),
        }
    }

    /// Validates that probabilities sum to ~1.0.
    pub fn validate(&self) -> Result<(), String> {
        let sum: f32 = self.color_probabilities.iter().sum();
        if (sum - 1.0).abs() > 0.01 {
            return Err(format!(
                "Vertex {} probabilities sum to {} (expected 1.0)",
                self.vertex, sum
            ));
        }
        Ok(())
    }

    /// Returns Shannon entropy of the color distribution.
    pub fn entropy(&self) -> f32 {
        self.color_probabilities
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }
}

/// Complete warmstart plan combining multiple prior sources.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmstartPlan {
    /// Per-vertex priors
    pub vertex_priors: Vec<WarmstartPrior>,

    /// Metadata about prior sources
    pub metadata: WarmstartMetadata,
}

impl WarmstartPlan {
    /// Creates an empty warmstart plan.
    pub fn empty(num_vertices: usize, max_colors: usize) -> Self {
        Self {
            vertex_priors: (0..num_vertices)
                .map(|v| WarmstartPrior::uniform(v, max_colors))
                .collect(),
            metadata: WarmstartMetadata::default(),
        }
    }

    /// Validates all priors in the plan.
    pub fn validate(&self) -> Result<(), String> {
        for prior in &self.vertex_priors {
            prior.validate()?;
        }
        Ok(())
    }

    /// Returns overall entropy of the plan.
    pub fn mean_entropy(&self) -> f32 {
        let sum: f32 = self.vertex_priors.iter().map(|p| p.entropy()).sum();
        sum / self.vertex_priors.len() as f32
    }
}

/// Metadata tracking warmstart plan construction.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WarmstartMetadata {
    /// Source weights used in fusion
    pub source_weights: HashMap<String, f32>,

    /// Number of structural anchors
    pub anchor_count: usize,

    /// Curriculum profile used (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curriculum_profile: Option<String>,

    /// Overall prior entropy
    pub prior_entropy: f32,

    /// Expected initial conflicts (predicted)
    pub expected_conflicts: usize,
}

/// Graph statistics for curriculum profile selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Graph density: |E| / (|V| * (|V| - 1) / 2)
    pub density: f64,

    /// Average vertex degree
    pub avg_degree: f64,

    /// Clustering coefficient (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clustering: Option<f64>,

    /// Number of vertices
    pub num_vertices: usize,

    /// Number of edges
    pub num_edges: usize,
}

impl GraphStats {
    /// Computes statistics from a graph.
    pub fn from_graph(graph: &Graph) -> Self {
        let n = graph.num_vertices;
        let m = graph.num_edges;

        let density = if n > 1 {
            (2.0 * m as f64) / (n * (n - 1)) as f64
        } else {
            0.0
        };

        let avg_degree = if n > 0 {
            (2.0 * m as f64) / n as f64
        } else {
            0.0
        };

        Self {
            density,
            avg_degree,
            clustering: None, // TODO: Implement clustering coefficient
            num_vertices: n,
            num_edges: m,
        }
    }
}

// ============================================================================
// Phase 0 & Warmstart Telemetry (Warmstart Plan Steps 2 & 7)
// ============================================================================

/// Telemetry for Phase 0 dendritic reservoir computation.
///
/// Captures comprehensive metrics from the reservoir computation including
/// difficulty/uncertainty statistics, convergence information, and execution time.
///
/// Implements Warmstart Plan Step 2: Phase 0 Telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase0Telemetry {
    /// Mean difficulty across all vertices
    pub difficulty_mean: f32,

    /// Variance of difficulty distribution
    pub difficulty_variance: f32,

    /// Shannon entropy of difficulty distribution
    pub difficulty_entropy: f32,

    /// Mean uncertainty across all vertices
    pub uncertainty_mean: f32,

    /// Variance of uncertainty distribution
    pub uncertainty_variance: f32,

    /// Shannon entropy of uncertainty distribution
    pub uncertainty_entropy: f32,

    /// Number of reservoir iterations executed
    pub reservoir_iterations: usize,

    /// Final convergence loss
    pub convergence_loss: f32,

    /// Execution time in milliseconds
    pub execution_time_ms: f64,

    /// Whether GPU acceleration was used
    pub used_gpu: bool,
}

impl Phase0Telemetry {
    /// Computes telemetry from difficulty and uncertainty vectors.
    ///
    /// # Arguments
    /// * `difficulty` - Per-vertex difficulty metrics
    /// * `uncertainty` - Per-vertex uncertainty metrics
    /// * `iterations` - Number of reservoir iterations
    /// * `loss` - Final convergence loss
    /// * `time_ms` - Execution time in milliseconds
    /// * `gpu` - Whether GPU was used
    pub fn from_metrics(
        difficulty: &[f32],
        uncertainty: &[f32],
        iterations: usize,
        loss: f32,
        time_ms: f64,
        gpu: bool,
    ) -> Self {
        let n = difficulty.len() as f32;

        // Compute difficulty statistics
        let difficulty_mean = difficulty.iter().sum::<f32>() / n;
        let difficulty_variance = difficulty
            .iter()
            .map(|&d| (d - difficulty_mean).powi(2))
            .sum::<f32>()
            / n;
        let difficulty_entropy = Self::compute_entropy(difficulty);

        // Compute uncertainty statistics
        let uncertainty_mean = uncertainty.iter().sum::<f32>() / n;
        let uncertainty_variance = uncertainty
            .iter()
            .map(|&u| (u - uncertainty_mean).powi(2))
            .sum::<f32>()
            / n;
        let uncertainty_entropy = Self::compute_entropy(uncertainty);

        Self {
            difficulty_mean,
            difficulty_variance,
            difficulty_entropy,
            uncertainty_mean,
            uncertainty_variance,
            uncertainty_entropy,
            reservoir_iterations: iterations,
            convergence_loss: loss,
            execution_time_ms: time_ms,
            used_gpu: gpu,
        }
    }

    /// Computes Shannon entropy of a distribution.
    fn compute_entropy(values: &[f32]) -> f32 {
        let sum: f32 = values.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }

        values
            .iter()
            .filter(|&&v| v > 0.0)
            .map(|&v| {
                let p = v / sum;
                -p * p.log2()
            })
            .sum()
    }

    /// Converts telemetry to HashMap for backwards compatibility with PhaseTelemetry trait.
    pub fn to_hashmap(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();
        map.insert("difficulty_mean".to_string(), self.difficulty_mean as f64);
        map.insert(
            "difficulty_variance".to_string(),
            self.difficulty_variance as f64,
        );
        map.insert(
            "difficulty_entropy".to_string(),
            self.difficulty_entropy as f64,
        );
        map.insert("uncertainty_mean".to_string(), self.uncertainty_mean as f64);
        map.insert(
            "uncertainty_variance".to_string(),
            self.uncertainty_variance as f64,
        );
        map.insert(
            "uncertainty_entropy".to_string(),
            self.uncertainty_entropy as f64,
        );
        map.insert(
            "reservoir_iterations".to_string(),
            self.reservoir_iterations as f64,
        );
        map.insert("convergence_loss".to_string(), self.convergence_loss as f64);
        map.insert("execution_time_ms".to_string(), self.execution_time_ms);
        map.insert(
            "used_gpu".to_string(),
            if self.used_gpu { 1.0 } else { 0.0 },
        );
        map
    }
}

/// Comprehensive telemetry for warmstart stage execution.
///
/// Captures metrics about prior quality, anchor statistics, curriculum selection,
/// fusion weights, and warmstart effectiveness (expected vs actual conflicts).
///
/// Implements Warmstart Plan Step 7: Warmstart Telemetry & Storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmstartTelemetry {
    /// Mean entropy of prior distributions (target: H >= 1.5 for DSJC250)
    pub prior_entropy_mean: f32,

    /// Variance of entropy across all vertex priors
    pub prior_entropy_variance: f32,

    /// Entropy distribution histogram (10 bins, 0.0-5.0 range)
    pub prior_entropy_distribution: Vec<f32>,

    /// Total number of anchor vertices (geodesic + TDA)
    pub anchor_count: usize,

    /// Percentage of vertices designated as anchors
    pub anchor_coverage_percent: f32,

    /// Number of geodesic anchors (from Phase 4)
    pub geodesic_anchor_count: usize,

    /// Number of TDA anchors (from Phase 6)
    pub tda_anchor_count: usize,

    /// Curriculum profile selected (e.g., "Easy", "Medium", "Hard", "Extreme")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curriculum_profile: Option<String>,

    /// Source of curriculum Q-table (e.g., "catalog.json", "pretrained")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curriculum_q_table_source: Option<String>,

    /// Fusion weight for Flux reservoir priors
    pub fusion_flux_weight: f32,

    /// Fusion weight for ensemble method priors
    pub fusion_ensemble_weight: f32,

    /// Fusion weight for random/uniform priors
    pub fusion_random_weight: f32,

    /// Expected number of conflicts (predicted before phase execution)
    pub expected_conflicts: usize,

    /// Actual conflicts after phase execution (filled post-execution)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actual_conflicts: Option<usize>,

    /// Warmstart effectiveness: 1.0 - (actual/expected), higher = better
    /// Values > 1.0 indicate warmstart exceeded expectations
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmstart_effectiveness: Option<f32>,
}

impl WarmstartTelemetry {
    /// Creates a new warmstart telemetry record.
    ///
    /// # Arguments
    /// * `priors` - Vector of warmstart priors for all vertices
    /// * `geodesic_anchors` - Geodesic anchor vertices
    /// * `tda_anchors` - TDA anchor vertices
    /// * `curriculum_profile` - Selected curriculum profile
    /// * `curriculum_source` - Q-table source path
    /// * `flux_weight` - Flux fusion weight
    /// * `ensemble_weight` - Ensemble fusion weight
    /// * `random_weight` - Random fusion weight
    /// * `expected_conflicts` - Predicted initial conflicts
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        priors: &[WarmstartPrior],
        geodesic_anchors: &[VertexId],
        tda_anchors: &[VertexId],
        curriculum_profile: Option<String>,
        curriculum_source: Option<String>,
        flux_weight: f32,
        ensemble_weight: f32,
        random_weight: f32,
        expected_conflicts: usize,
    ) -> Self {
        let num_vertices = priors.len();

        // Compute entropy statistics
        let entropies: Vec<f32> = priors.iter().map(|p| p.entropy()).collect();
        let prior_entropy_mean = entropies.iter().sum::<f32>() / entropies.len() as f32;
        let prior_entropy_variance = entropies
            .iter()
            .map(|&e| (e - prior_entropy_mean).powi(2))
            .sum::<f32>()
            / entropies.len() as f32;

        // Build entropy histogram (10 bins, 0.0-5.0 range)
        let mut histogram = vec![0.0f32; 10];
        for &entropy in &entropies {
            let bin = ((entropy / 5.0) * 10.0).floor() as usize;
            if bin < 10 {
                histogram[bin] += 1.0;
            }
        }
        // Normalize histogram
        let total: f32 = histogram.iter().sum();
        if total > 0.0 {
            for count in &mut histogram {
                *count /= total;
            }
        }

        // Compute anchor statistics
        let total_anchors = geodesic_anchors.len() + tda_anchors.len();
        let anchor_coverage_percent = if num_vertices > 0 {
            (total_anchors as f32 / num_vertices as f32) * 100.0
        } else {
            0.0
        };

        Self {
            prior_entropy_mean,
            prior_entropy_variance,
            prior_entropy_distribution: histogram,
            anchor_count: total_anchors,
            anchor_coverage_percent,
            geodesic_anchor_count: geodesic_anchors.len(),
            tda_anchor_count: tda_anchors.len(),
            curriculum_profile,
            curriculum_q_table_source: curriculum_source,
            fusion_flux_weight: flux_weight,
            fusion_ensemble_weight: ensemble_weight,
            fusion_random_weight: random_weight,
            expected_conflicts,
            actual_conflicts: None,
            warmstart_effectiveness: None,
        }
    }

    /// Updates effectiveness metrics after phase execution completes.
    ///
    /// # Arguments
    /// * `actual_conflicts` - Conflicts from best solution after phase execution
    pub fn update_effectiveness(&mut self, actual_conflicts: usize) {
        self.actual_conflicts = Some(actual_conflicts);

        if self.expected_conflicts > 0 {
            self.warmstart_effectiveness =
                Some(1.0 - (actual_conflicts as f32 / self.expected_conflicts as f32));
        } else {
            // If expected conflicts was 0, set effectiveness based on whether we achieved 0 conflicts
            self.warmstart_effectiveness = Some(if actual_conflicts == 0 { 1.0 } else { 0.0 });
        }
    }

    /// Returns mean prior entropy.
    pub fn mean_entropy(&self) -> f32 {
        self.prior_entropy_mean
    }

    /// Returns anchor coverage percentage.
    pub fn anchor_coverage(&self) -> f32 {
        self.anchor_coverage_percent
    }

    /// Returns curriculum profile name (if available).
    pub fn profile_name(&self) -> Option<&str> {
        self.curriculum_profile.as_deref()
    }
}

/// Geometric stress telemetry for metaphysical coupling feedback loop.
///
/// Captures geometric metrics from Phase 4 (Geodesic) and Phase 6 (TDA) to
/// influence subsequent phase behavior (Phase 1/2/3/7 adjustments).
///
/// ## Metaphysical Coupling (Geometric Stress Telemetry Extension)
/// When geometric stress is high (overlap_density > threshold, stress_scalar > 0.5),
/// downstream phases react:
/// - Phase 1: Increase prediction error (active inference exploration)
/// - Phase 2: Raise temperature (thermodynamic reheat)
/// - Phase 3: Adjust coupling strength (quantum entanglement)
/// - Phase 7: Intensify local search (memetic mutation)
///
/// Implements the feedback loop where geometry telemetry influences all phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryTelemetry {
    /// Bounding box area of the coloring solution (normalized)
    /// Higher values indicate solutions spread across color space
    pub bounding_box_area: f32,

    /// Rate of growth in chromatic number over iterations
    /// Negative values indicate shrinkage (good), positive = expansion (bad)
    pub growth_rate: f32,

    /// Density of color overlaps (0.0 = clean, 1.0 = highly conflicted)
    /// Computed as fraction of edges with same-color endpoints
    pub overlap_density: f32,

    /// Composite stress scalar: weighted combination of all metrics (0.0 - 1.0)
    /// Formula: stress_scalar = 0.4*overlap_density + 0.3*growth_rate + 0.3*bounding_box_area
    pub stress_scalar: f32,

    /// Hotspot vertices with high local stress (anchor candidates)
    /// These vertices have high conflict counts and should be prioritized
    pub anchor_hotspots: Vec<usize>,
}

impl GeometryTelemetry {
    /// Creates a new GeometryTelemetry with zero stress (ideal state).
    pub fn zero_stress() -> Self {
        Self {
            bounding_box_area: 0.0,
            growth_rate: 0.0,
            overlap_density: 0.0,
            stress_scalar: 0.0,
            anchor_hotspots: Vec::new(),
        }
    }

    /// Computes stress scalar from individual metrics.
    ///
    /// # Formula
    /// stress_scalar = 0.4 * overlap_density + 0.3 * |growth_rate| + 0.3 * bounding_box_area
    ///
    /// # Range
    /// [0.0, 1.0] where 0.0 = no stress, 1.0 = maximum stress
    pub fn compute_stress_scalar(&mut self) {
        self.stress_scalar = 0.4 * self.overlap_density
            + 0.3 * self.growth_rate.abs()
            + 0.3 * self.bounding_box_area.min(1.0);
        self.stress_scalar = self.stress_scalar.clamp(0.0, 1.0);
    }

    /// Checks if stress level is high (requires phase intervention).
    ///
    /// High stress threshold: stress_scalar > 0.5
    pub fn is_high_stress(&self) -> bool {
        self.stress_scalar > 0.5
    }

    /// Checks if stress level is critical (requires aggressive intervention).
    ///
    /// Critical stress threshold: stress_scalar > 0.8
    pub fn is_critical_stress(&self) -> bool {
        self.stress_scalar > 0.8
    }

    /// Returns the number of anchor hotspots.
    pub fn hotspot_count(&self) -> usize {
        self.anchor_hotspots.len()
    }

    /// Creates synthetic geometry telemetry from early-phase signals.
    ///
    /// This enables metaphysical coupling before Phase 4/6 by using proxy metrics
    /// from Phase 0 (dendritic reservoir) and Phase 1 (active inference).
    ///
    /// # Arguments
    /// * `mean_uncertainty` - Mean uncertainty from Active Inference policy (Phase 1)
    /// * `mean_difficulty` - Mean difficulty from reservoir computation (Phase 0)
    /// * `num_vertices` - Graph size
    ///
    /// # Proxy Mapping
    /// - `overlap_density` ≈ mean_uncertainty (high uncertainty suggests conflicts)
    /// - `bounding_box_area` ≈ mean_difficulty (harder graphs use more colors)
    /// - `growth_rate` = 0.0 (no history yet)
    /// - `anchor_hotspots` = top 10% most uncertain/difficult vertices
    ///
    /// # Returns
    /// Synthetic geometry telemetry for early-phase coupling
    pub fn from_early_phase_signals(
        mean_uncertainty: f32,
        mean_difficulty: f32,
        vertex_uncertainties: &[f32],
    ) -> Self {
        // Map uncertainty to overlap density (proxy for conflicts)
        let overlap_density = mean_uncertainty.clamp(0.0, 1.0);

        // Map difficulty to bounding box area (proxy for color spread)
        let bounding_box_area = mean_difficulty.clamp(0.0, 1.0);

        // No growth rate available yet (requires iteration history)
        let growth_rate = 0.0;

        // Identify top 10% most uncertain vertices as hotspots
        let mut vertex_scores: Vec<(usize, f32)> = vertex_uncertainties
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        vertex_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let hotspot_count = (vertex_scores.len() / 10).max(1);
        let anchor_hotspots: Vec<usize> = vertex_scores
            .iter()
            .take(hotspot_count)
            .map(|(idx, _)| *idx)
            .collect();

        let mut telemetry = Self {
            bounding_box_area,
            growth_rate,
            overlap_density,
            stress_scalar: 0.0, // Will be computed below
            anchor_hotspots,
        };

        telemetry.compute_stress_scalar();
        telemetry
    }

    /// Creates geometry telemetry from a coloring solution and graph.
    ///
    /// # Arguments
    /// * `solution` - Current coloring solution
    /// * `graph` - Input graph
    /// * `previous_chromatic` - Chromatic number from previous iteration (for growth rate)
    pub fn from_solution(
        solution: &ColoringSolution,
        graph: &Graph,
        previous_chromatic: Option<usize>,
    ) -> Self {
        // Compute overlap density (fraction of conflicted edges)
        let overlap_density = if graph.num_edges > 0 {
            solution.conflicts as f32 / graph.num_edges as f32
        } else {
            0.0
        };

        // Compute growth rate (change in chromatic number)
        let growth_rate = if let Some(prev) = previous_chromatic {
            if prev > 0 {
                (solution.chromatic_number as f32 - prev as f32) / prev as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Compute bounding box area (normalized by max colors)
        let max_color = solution.colors.iter().max().copied().unwrap_or(0);
        let bounding_box_area = if max_color > 0 {
            solution.chromatic_number as f32 / (max_color + 1) as f32
        } else {
            0.0
        };

        // Identify anchor hotspots (vertices with most conflicts)
        let mut vertex_conflicts: Vec<(usize, usize)> = (0..graph.num_vertices)
            .map(|v| {
                let conflicts = graph.adjacency[v]
                    .iter()
                    .filter(|&&neighbor| solution.colors[v] == solution.colors[neighbor])
                    .count();
                (v, conflicts)
            })
            .collect();

        // Sort by conflict count (descending) and take top 10%
        vertex_conflicts.sort_by(|a, b| b.1.cmp(&a.1));
        let hotspot_count = (graph.num_vertices / 10).max(1);
        let anchor_hotspots: Vec<usize> = vertex_conflicts
            .iter()
            .take(hotspot_count)
            .map(|(v, _)| *v)
            .collect();

        let mut telemetry = Self {
            bounding_box_area,
            growth_rate,
            overlap_density,
            stress_scalar: 0.0, // Computed below
            anchor_hotspots,
        };

        telemetry.compute_stress_scalar();
        telemetry
    }
}

/// Configuration for warmstart system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmstartConfig {
    /// Maximum number of colors in prior distribution
    pub max_colors: usize,

    /// Minimum probability for any color (prevents zero probabilities)
    pub min_prob: f32,

    /// Fraction of vertices to designate as anchors (0.0 - 1.0)
    pub anchor_fraction: f32,

    /// Source weights for prior fusion
    pub flux_weight: f32,
    pub ensemble_weight: f32,
    pub random_weight: f32,

    /// Path to curriculum profile catalog
    #[serde(skip_serializing_if = "Option::is_none")]
    pub curriculum_catalog_path: Option<String>,
}

impl Default for WarmstartConfig {
    fn default() -> Self {
        Self {
            max_colors: 50,
            min_prob: 0.001,
            anchor_fraction: 0.10, // 10% of vertices as anchors
            flux_weight: 0.4,
            ensemble_weight: 0.4,
            random_weight: 0.2,
            curriculum_catalog_path: Some("profiles/curriculum/catalog.json".to_string()),
        }
    }
}

/// Phase 3 (Quantum Evolution) hyperparameters.
///
/// Controls quantum-inspired annealing parameters for GPU evolution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase3Config {
    /// Evolution time parameter (controls quantum fluctuation magnitude)
    pub evolution_time: f32,

    /// Coupling strength (conflict penalty weight)
    pub coupling_strength: f32,

    /// Maximum number of colors to use in quantum evolution
    pub max_colors: usize,

    /// Number of qubits (typically equals number of vertices)
    pub num_qubits: usize,

    /// Use complex-valued quantum amplitudes (enables interference effects)
    #[serde(default = "default_use_complex")]
    pub use_complex_amplitudes: bool,

    /// Number of evolution iterations (annealing schedule steps)
    #[serde(default = "default_evolution_iterations")]
    pub evolution_iterations: usize,

    /// Transverse field strength (quantum tunneling parameter)
    #[serde(default = "default_transverse_field")]
    pub transverse_field: f32,

    /// Interference decay rate (controls decoherence)
    #[serde(default = "default_interference_decay")]
    pub interference_decay: f32,

    /// Evolution schedule type: "linear", "exponential", or "custom"
    #[serde(default = "default_schedule_type")]
    pub schedule_type: String,

    /// Use stochastic quantum measurement (requires RNG initialization)
    #[serde(default = "default_stochastic_measurement")]
    pub stochastic_measurement: bool,
}

// Default value functions for serde
fn default_use_complex() -> bool {
    true
}
fn default_evolution_iterations() -> usize {
    100
}
fn default_transverse_field() -> f32 {
    1.0
}
fn default_interference_decay() -> f32 {
    0.01
}
fn default_schedule_type() -> String {
    "linear".to_string()
}
fn default_stochastic_measurement() -> bool {
    false
}

impl Default for Phase3Config {
    fn default() -> Self {
        Self {
            evolution_time: 1.0,
            coupling_strength: 1.0,
            max_colors: 50,
            num_qubits: 500,
            use_complex_amplitudes: default_use_complex(),
            evolution_iterations: default_evolution_iterations(),
            transverse_field: default_transverse_field(),
            interference_decay: default_interference_decay(),
            schedule_type: default_schedule_type(),
            stochastic_measurement: default_stochastic_measurement(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);

        assert_eq!(graph.num_vertices, 5);
        assert_eq!(graph.num_edges, 3);
        assert_eq!(graph.adjacency[0], vec![1]);
        assert_eq!(graph.adjacency[1], vec![0, 2]);
    }

    #[test]
    fn test_solution_validation() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);

        let mut solution = ColoringSolution::new(3);
        solution.colors = vec![0, 1, 0]; // Valid 2-coloring

        let conflicts = solution.validate(&graph);
        assert_eq!(conflicts, 0);

        solution.colors = vec![0, 0, 1]; // Invalid: edge (0,1) conflict
        let conflicts = solution.validate(&graph);
        assert_eq!(conflicts, 1);
    }

    #[test]
    fn test_phase_config_parameters() {
        let mut config = PhaseConfig::new("Phase0");
        config.set_parameter("temperature", 1.5);
        config.set_parameter("learning_rate", 0.01);

        let temp: f64 = config.get_parameter("temperature").unwrap();
        assert_eq!(temp, 1.5);

        let lr: f64 = config.get_parameter("learning_rate").unwrap();
        assert_eq!(lr, 0.01);
    }

    #[test]
    fn test_phase0_telemetry_computation() {
        let difficulty = vec![0.1, 0.5, 0.8, 0.3, 0.6];
        let uncertainty = vec![0.2, 0.4, 0.1, 0.7, 0.5];

        let telemetry =
            Phase0Telemetry::from_metrics(&difficulty, &uncertainty, 100, 0.01, 123.45, true);

        // Verify basic fields
        assert_eq!(telemetry.reservoir_iterations, 100);
        assert_eq!(telemetry.convergence_loss, 0.01);
        assert_eq!(telemetry.execution_time_ms, 123.45);
        assert!(telemetry.used_gpu);

        // Verify statistics are reasonable
        assert!(telemetry.difficulty_mean > 0.0 && telemetry.difficulty_mean < 1.0);
        assert!(telemetry.uncertainty_mean > 0.0 && telemetry.uncertainty_mean < 1.0);
        assert!(telemetry.difficulty_variance >= 0.0);
        assert!(telemetry.uncertainty_variance >= 0.0);
        assert!(telemetry.difficulty_entropy >= 0.0);
        assert!(telemetry.uncertainty_entropy >= 0.0);

        // Test to_hashmap conversion
        let map = telemetry.to_hashmap();
        assert_eq!(map.len(), 10);
        assert_eq!(map.get("used_gpu"), Some(&1.0));
        assert_eq!(map.get("reservoir_iterations"), Some(&100.0));
    }

    #[test]
    fn test_phase0_telemetry_serialization() {
        let difficulty = vec![0.5, 0.5];
        let uncertainty = vec![0.5, 0.5];

        let telemetry =
            Phase0Telemetry::from_metrics(&difficulty, &uncertainty, 50, 0.001, 50.0, false);

        // Test JSON serialization
        let json = serde_json::to_string(&telemetry).unwrap();
        assert!(json.contains("difficulty_mean"));
        assert!(json.contains("used_gpu"));

        // Test deserialization round-trip
        let deserialized: Phase0Telemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.reservoir_iterations, 50);
        assert!(!deserialized.used_gpu);
    }

    #[test]
    fn test_warmstart_telemetry_creation() {
        let priors = vec![
            WarmstartPrior::uniform(0, 5),
            WarmstartPrior::uniform(1, 5),
            WarmstartPrior::anchor(2, 0, 5),
        ];

        let geodesic_anchors = vec![0];
        let tda_anchors = vec![2];

        let telemetry = WarmstartTelemetry::new(
            &priors,
            &geodesic_anchors,
            &tda_anchors,
            Some("Medium".to_string()),
            Some("catalog.json".to_string()),
            0.4,
            0.4,
            0.2,
            10,
        );

        assert_eq!(telemetry.anchor_count, 2);
        assert_eq!(telemetry.geodesic_anchor_count, 1);
        assert_eq!(telemetry.tda_anchor_count, 1);
        assert!(telemetry.anchor_coverage_percent > 0.0);
        assert_eq!(telemetry.fusion_flux_weight, 0.4);
        assert_eq!(telemetry.fusion_ensemble_weight, 0.4);
        assert_eq!(telemetry.fusion_random_weight, 0.2);
        assert_eq!(telemetry.expected_conflicts, 10);
        assert!(telemetry.actual_conflicts.is_none());
        assert!(telemetry.warmstart_effectiveness.is_none());
        assert_eq!(telemetry.curriculum_profile.as_deref(), Some("Medium"));
    }

    #[test]
    fn test_warmstart_telemetry_effectiveness() {
        let priors = vec![WarmstartPrior::uniform(0, 3)];
        let mut telemetry = WarmstartTelemetry::new(
            &priors,
            &[],
            &[],
            None,
            None,
            0.5,
            0.3,
            0.2,
            20, // Expected 20 conflicts
        );

        // Update with actual conflicts: 10 (better than expected)
        telemetry.update_effectiveness(10);
        assert_eq!(telemetry.actual_conflicts, Some(10));
        assert_eq!(telemetry.warmstart_effectiveness, Some(0.5)); // 1.0 - (10/20) = 0.5

        // Test exceeded expectations case
        let mut telemetry2 =
            WarmstartTelemetry::new(&priors, &[], &[], None, None, 0.5, 0.3, 0.2, 10);
        telemetry2.update_effectiveness(5);
        assert_eq!(telemetry2.warmstart_effectiveness, Some(0.5)); // 1.0 - (5/10)

        // Test zero expected conflicts case
        let mut telemetry3 =
            WarmstartTelemetry::new(&priors, &[], &[], None, None, 0.5, 0.3, 0.2, 0);
        telemetry3.update_effectiveness(0);
        assert_eq!(telemetry3.warmstart_effectiveness, Some(1.0)); // Perfect!

        telemetry3.update_effectiveness(5);
        assert_eq!(telemetry3.warmstart_effectiveness, Some(0.0)); // Failed
    }

    #[test]
    fn test_warmstart_telemetry_serialization() {
        let priors = vec![WarmstartPrior::uniform(0, 5)];
        let mut telemetry = WarmstartTelemetry::new(
            &priors,
            &[0],
            &[],
            Some("Hard".to_string()),
            Some("test.json".to_string()),
            0.5,
            0.3,
            0.2,
            15,
        );
        telemetry.update_effectiveness(8);

        // Test JSON serialization round-trip
        let json = serde_json::to_string(&telemetry).unwrap();
        assert!(json.contains("prior_entropy_mean"));
        assert!(json.contains("anchor_count"));
        assert!(json.contains("warmstart_effectiveness"));

        let deserialized: WarmstartTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.anchor_count, 1);
        assert_eq!(deserialized.actual_conflicts, Some(8));
        assert_eq!(deserialized.curriculum_profile.as_deref(), Some("Hard"));
    }

    #[test]
    fn test_warmstart_telemetry_entropy_distribution() {
        // Create priors with varying entropy
        let mut priors = Vec::new();
        for i in 0..10 {
            let mut prior = WarmstartPrior::uniform(i, 5);
            // Modify to create different entropy values
            if i < 3 {
                prior.color_probabilities = vec![1.0, 0.0, 0.0, 0.0, 0.0]; // Low entropy
            } else if i < 7 {
                prior.color_probabilities = vec![0.4, 0.3, 0.2, 0.1, 0.0]; // Medium entropy
            }
            // else keep uniform (high entropy)
            priors.push(prior);
        }

        let telemetry = WarmstartTelemetry::new(&priors, &[], &[], None, None, 0.5, 0.3, 0.2, 0);

        // Verify histogram has 10 bins
        assert_eq!(telemetry.prior_entropy_distribution.len(), 10);

        // Verify histogram sums to ~1.0 (normalized)
        let sum: f32 = telemetry.prior_entropy_distribution.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Histogram sum: {}", sum);

        // Verify entropy statistics
        assert!(telemetry.prior_entropy_mean >= 0.0);
        assert!(telemetry.prior_entropy_variance >= 0.0);
    }

    #[test]
    fn test_geometry_telemetry_creation() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);

        let mut solution = ColoringSolution::new(5);
        solution.colors = vec![0, 0, 1, 1, 2]; // 2 conflicts: (0,1) and (2,3)
        solution.conflicts = 2;
        solution.chromatic_number = 3;

        let telemetry = GeometryTelemetry::from_solution(&solution, &graph, Some(2));

        // Verify overlap density
        assert_eq!(telemetry.overlap_density, 0.5); // 2 conflicts / 4 edges

        // Verify growth rate
        assert_eq!(telemetry.growth_rate, 0.5); // (3 - 2) / 2 = 0.5

        // Verify stress scalar was computed
        assert!(telemetry.stress_scalar > 0.0);

        // Verify hotspots were identified
        assert!(telemetry.anchor_hotspots.len() > 0);
    }

    #[test]
    fn test_geometry_telemetry_zero_stress() {
        let telemetry = GeometryTelemetry::zero_stress();
        assert_eq!(telemetry.stress_scalar, 0.0);
        assert!(!telemetry.is_high_stress());
        assert!(!telemetry.is_critical_stress());
        assert_eq!(telemetry.hotspot_count(), 0);
    }

    #[test]
    fn test_geometry_telemetry_stress_thresholds() {
        let mut telemetry = GeometryTelemetry::zero_stress();

        // Test high stress threshold
        // stress_scalar = 0.4 * 0.7 + 0.3 * 0 + 0.3 * 0 = 0.28 (below 0.5, not high stress)
        // Need higher values to cross 0.5 threshold
        telemetry.overlap_density = 0.9;
        telemetry.growth_rate = 0.5;
        telemetry.bounding_box_area = 0.5;
        telemetry.compute_stress_scalar();
        // stress_scalar = 0.4 * 0.9 + 0.3 * 0.5 + 0.3 * 0.5 = 0.36 + 0.15 + 0.15 = 0.66
        assert!(telemetry.is_high_stress());
        assert!(!telemetry.is_critical_stress());

        // Test critical stress threshold (>0.8)
        telemetry.overlap_density = 1.0;
        telemetry.growth_rate = 1.0;
        telemetry.bounding_box_area = 1.0;
        telemetry.compute_stress_scalar();
        // stress_scalar = 0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 1.0 = 0.4 + 0.3 + 0.3 = 1.0
        assert!(telemetry.is_critical_stress());
    }

    #[test]
    fn test_geometry_telemetry_serialization() {
        let mut telemetry = GeometryTelemetry::zero_stress();
        telemetry.overlap_density = 0.5;
        telemetry.growth_rate = 0.1;
        telemetry.bounding_box_area = 0.3;
        telemetry.anchor_hotspots = vec![0, 5, 10];
        telemetry.compute_stress_scalar();

        // Test JSON serialization round-trip
        let json = serde_json::to_string(&telemetry).unwrap();
        assert!(json.contains("overlap_density"));
        assert!(json.contains("stress_scalar"));
        assert!(json.contains("anchor_hotspots"));

        let deserialized: GeometryTelemetry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.overlap_density, 0.5);
        assert_eq!(deserialized.anchor_hotspots, vec![0, 5, 10]);
    }
}

/// CMA-ES optimization state for tracking evolutionary progress.
///
/// Stores key metrics from the CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
/// algorithm used for optimizing graph coloring via transfer entropy minimization.
///
/// Implements PRISM GPU Plan §7.3: Subsystem State Updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CmaState {
    /// Best solution found (parameter vector)
    pub best_solution: Vec<f32>,

    /// Fitness value of best solution (lower is better for minimization)
    pub best_fitness: f32,

    /// Condition number of the covariance matrix
    /// High values (>1e14) indicate degeneration
    pub covariance_condition: f32,

    /// Current generation/iteration number
    pub generation: usize,

    /// Convergence metric (0.0 = not converged, 1.0 = fully converged)
    /// Based on change in best fitness over recent generations
    pub convergence_metric: f32,

    /// Mean fitness of current population
    pub mean_fitness: f32,

    /// Standard deviation of fitness in current population
    pub fitness_std: f32,

    /// Step size (sigma) parameter
    pub sigma: f32,

    /// Effective sample size
    pub effective_size: f32,
}
