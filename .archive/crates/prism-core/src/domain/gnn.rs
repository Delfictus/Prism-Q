//! GNN (Graph Neural Network) domain module
//!
//! Provides state and configuration for GNN-based graph analysis.

use serde::{Deserialize, Serialize};

/// GNN state for telemetry and pipeline tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnState {
    /// Predicted chromatic number
    pub predicted_chromatic: usize,

    /// Embedding dimension used
    pub embedding_dim: usize,

    /// Prediction confidence (0.0-1.0)
    pub confidence: f32,

    /// Estimated manifold dimension
    pub manifold_dimension: f32,

    /// Manifold curvature estimate
    pub manifold_curvature: f32,

    /// Geodesic complexity metric
    pub geodesic_complexity: f32,

    /// Number of Betti numbers computed
    pub betti_count: usize,

    /// GNN model type used ("e3_equivariant" or "onnx")
    pub model_type: String,
}

impl Default for GnnState {
    fn default() -> Self {
        Self {
            predicted_chromatic: 0,
            embedding_dim: 0,
            confidence: 0.0,
            manifold_dimension: 0.0,
            manifold_curvature: 0.0,
            geodesic_complexity: 0.0,
            betti_count: 0,
            model_type: "e3_equivariant".to_string(),
        }
    }
}
