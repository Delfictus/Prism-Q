//! Spatial Neighbor Index — backend-agnostic neighbor-finding contract
//!
//! Phase 1 of the LBVH lane (2026-04-21 canonical lockdown).
//!
//! This trait is the single API boundary between Layer 1 (spatial indexing)
//! and Layer 2 (clustering / persistence analysis). Any backend — OptiX RT,
//! CUDA LBVH, or grid-LIGSITE debug — implements this trait and is drop-in
//! compatible with `hierarchical_clustering.rs`.
//!
//! **Production rule #10 (2026-04-21):**
//!   - LBVH is the default on SM120+.
//!   - Grid-LIGSITE is debug-only behind `--clustering-backend=grid`.
//!   - `--clustering-backend=grid` combined with `--multi-differential`
//!     is a hard ERROR in production (see nhs_rt_full.rs dispatch).
//!
//! Source of truth: `crates/prism-nhs/src/bin/nhs_rt_full.rs` dispatch region.

use anyhow::Result;

/// Result of a spatial neighbor query at a single epsilon level.
///
/// This is the EXACT contract that `hierarchical_clustering.rs::cluster_spikes`
/// consumes. Every backend (OptiX / LBVH / grid-debug) MUST produce this
/// shape with these semantics.
#[derive(Debug, Clone)]
pub struct NeighborQueryResult {
    /// Cluster ID per point after GPU union-find.
    /// `-1` = noise (no neighbors above min_points threshold).
    /// Non-negative = cluster id (dense small integers).
    pub cluster_ids: Vec<i32>,
    /// Number of distinct clusters (excluding noise).
    pub num_clusters: usize,
    /// Total number of neighbor pairs found (sum of neighbor_count).
    pub total_neighbors: usize,
    /// Wall-clock GPU time for this query in milliseconds.
    pub gpu_time_ms: f64,
}

/// Configuration common to every backend.
#[derive(Debug, Clone)]
pub struct NeighborIndexConfig {
    /// Neighborhood radius in Å.
    pub epsilon: f32,
    /// Minimum points required for a core point (DBSCAN-style).
    pub min_points: u32,
    /// Minimum cluster size to keep (smaller clusters → noise).
    pub min_cluster_size: u32,
}

impl Default for NeighborIndexConfig {
    fn default() -> Self {
        Self {
            epsilon: 5.0,
            min_points: 3,
            min_cluster_size: 100,
        }
    }
}

/// Backend selection — persisted across the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialBackend {
    /// OptiX RT cores (platform-gated; currently unavailable on SM120).
    OptixRt,
    /// CUDA LBVH (default on SM120+; implemented in Phase 2).
    Lbvh,
    /// Grid-LIGSITE debug fallback. Degrades detection quality.
    /// FORBIDDEN in production when combined with --multi-differential.
    GridDebug,
}

impl SpatialBackend {
    /// Human-readable name for logs.
    pub fn name(&self) -> &'static str {
        match self {
            SpatialBackend::OptixRt => "optix-rt",
            SpatialBackend::Lbvh => "lbvh",
            SpatialBackend::GridDebug => "grid-debug",
        }
    }

    /// Is this backend permitted in a production (multi-differential) run?
    pub fn is_production_permitted(&self) -> bool {
        matches!(self, SpatialBackend::OptixRt | SpatialBackend::Lbvh)
    }
}

/// The backend-agnostic neighbor-finding trait.
///
/// Every backend implements exactly this and nothing more. Consumers
/// (`hierarchical_clustering.rs`) hold `Box<dyn SpatialNeighborIndex>`
/// and never name a concrete type.
///
/// Not `Send`: backends may hold thread-affinity resources (OptiX
/// contexts are thread-bound; CUDA contexts are device-bound). The
/// engine is single-threaded within clustering; multi-stream
/// parallelism is above this layer at the PersistentNhsEngine level.
pub trait SpatialNeighborIndex {
    /// Identity of this backend (for logs and policy decisions).
    fn backend(&self) -> SpatialBackend;

    /// Prepare internal state for a given position set at `max_epsilon`.
    ///
    /// For BVH-based backends (OptiX, LBVH) this builds the tree ONCE so
    /// `query_at_epsilon` can be called repeatedly for ε-sweep without
    /// rebuilding. For grid backends this is a no-op or grid-resize.
    ///
    /// `positions` is a flattened f32 array: `[x0, y0, z0, x1, y1, z1, ...]`.
    fn prepare(&mut self, positions: &[f32], max_epsilon: f32) -> Result<()>;

    /// Query all neighbors within `epsilon` of every point, then run the
    /// shared union-find clustering path. Reuses the structure built in
    /// `prepare()` — caller is responsible for calling `prepare()` first
    /// if switching to a different position set.
    fn query_at_epsilon(&mut self, positions: &[f32], epsilon: f32) -> Result<NeighborQueryResult>;

    /// Convenience: single query with no persistent prepare.
    /// Default impl calls `prepare()` then `query_at_epsilon()`.
    fn query_once(&mut self, positions: &[f32], epsilon: f32) -> Result<NeighborQueryResult> {
        self.prepare(positions, epsilon)?;
        self.query_at_epsilon(positions, epsilon)
    }
}
