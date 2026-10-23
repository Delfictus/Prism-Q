//! Grid-LIGSITE Debug Backend
//!
//! **DEBUG ONLY.** This backend degrades detection quality by skipping
//! multi-scale persistence analysis. It exists solely as a bisection
//! fallback when debugging new backends (e.g., LBVH in Phase 2).
//!
//! Production rule #10 (2026-04-21): this backend is FORBIDDEN when
//! combined with --multi-differential. The dispatch in nhs_rt_full.rs
//! MUST emit a hard ERROR exit if a user attempts this combination.
//!
//! Historical context: the existing grid path in nhs_rt_full.rs
//! (Dynamic LIGSITE, line ~5900+ region) was the silent fallback when
//! OptiX became unavailable on SM120. It produces geometric pockets via
//! LIGSITE + GPU Gaussian splat + Eikonal watershed, but does NOT run
//! the multi-scale ε-sweep that hierarchical_clustering.rs requires for
//! proper persistence-based noise rejection. This file exposes that
//! degraded path explicitly behind the trait so it is no longer silent.

use anyhow::{bail, Result};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;

use crate::spatial_index::{
    NeighborIndexConfig, NeighborQueryResult, SpatialBackend, SpatialNeighborIndex,
};

/// Grid-debug neighbor index — wraps the existing grid-LIGSITE path.
///
/// Owns no GPU resources itself; it delegates to the existing grid
/// code in nhs_rt_full.rs via the legacy function signatures. Phase 1
/// does NOT refactor the grid code — that would be out of scope. The
/// goal here is only to make the degraded behavior *explicit and
/// flag-gated*.
#[cfg(feature = "gpu")]
pub struct GridDebugIndex {
    #[allow(dead_code)]
    context: Arc<CudaContext>,
    config: NeighborIndexConfig,
    // Cached last positions so query_at_epsilon doesn't need to re-receive them
    last_positions: Option<Vec<f32>>,
}

#[cfg(feature = "gpu")]
impl GridDebugIndex {
    pub fn new(context: Arc<CudaContext>, config: NeighborIndexConfig) -> Result<Self> {
        log::warn!(
            "  [SPATIAL-INDEX] Grid-debug backend selected. Detection quality DEGRADED \
             (skips multi-scale persistence). Production rule #10: forbidden with \
             --multi-differential. This MUST only be used for debug/bisection."
        );
        Ok(Self {
            context,
            config,
            last_positions: None,
        })
    }
}

#[cfg(feature = "gpu")]
impl SpatialNeighborIndex for GridDebugIndex {
    fn backend(&self) -> SpatialBackend {
        SpatialBackend::GridDebug
    }

    fn prepare(&mut self, positions: &[f32], _max_epsilon: f32) -> Result<()> {
        // Grid path rebuilds per-query; just cache positions for query_at_epsilon.
        self.last_positions = Some(positions.to_vec());
        Ok(())
    }

    fn query_at_epsilon(
        &mut self,
        _positions: &[f32],
        _epsilon: f32,
    ) -> Result<NeighborQueryResult> {
        // Phase 1 stub: the grid-LIGSITE path inside nhs_rt_full.rs consumes
        // spikes directly into PocketInfo — it does NOT produce the
        // neighbor_list/cluster_ids contract that the trait demands.
        //
        // Rather than retrofit the grid code to emit the contract (which
        // would be real algorithmic work), Phase 1 returns an explicit
        // error. The nhs_rt_full.rs dispatch calls the grid path directly
        // in the --clustering-backend=grid case; this trait impl only
        // exists to expose the backend enum for policy checks.
        bail!(
            "GridDebugIndex::query_at_epsilon is intentionally unimplemented. \
             The grid-LIGSITE path in nhs_rt_full.rs does not produce the \
             SpatialNeighborIndex neighbor_list contract. When \
             --clustering-backend=grid is set, nhs_rt_full.rs routes \
             directly to the legacy grid path without going through this \
             trait. See production rule #10."
        )
    }
}
