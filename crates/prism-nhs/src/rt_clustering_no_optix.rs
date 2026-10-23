//! Fail-closed OptiX clustering surface for CUDA-only production builds.
//!
//! The SM120 runtime uses the native CUDA spatial-hash backend. This module
//! preserves the clustering result contract used by that backend while making
//! every attempt to instantiate or invoke OptiX clustering fail explicitly.

use anyhow::{bail, Result};
use cudarc::driver::CudaContext;
use std::path::Path;
use std::sync::Arc;

const OPTIX_DISABLED: &str =
    "OptiX clustering requested, but this binary was built without the explicit `optix` feature";

/// RT clustering configuration retained for API compatibility.
#[derive(Debug, Clone)]
pub struct RtClusteringConfig {
    /// Neighborhood radius in angstroms.
    pub epsilon: f32,
    /// Minimum points required to form a core point.
    pub min_points: u32,
    /// Minimum cluster size retained in the result.
    pub min_cluster_size: u32,
    /// Rays per event for the OptiX implementation.
    pub rays_per_event: u32,
}

impl Default for RtClusteringConfig {
    fn default() -> Self {
        Self {
            epsilon: 5.0,
            min_points: 3,
            min_cluster_size: 100,
            rays_per_event: 16,
        }
    }
}

/// Result contract shared with the production CUDA clustering backend.
#[derive(Debug, Clone)]
pub struct RtClusteringResult {
    /// Cluster ID for each event, with `-1` denoting noise.
    pub cluster_ids: Vec<i32>,
    /// Number of clusters found.
    pub num_clusters: usize,
    /// Total neighbor pairs found.
    pub total_neighbors: usize,
    /// GPU time in milliseconds.
    pub gpu_time_ms: f64,
}

/// Unavailable OptiX engine handle for a CUDA-only build.
pub struct RtClusteringEngine {
    /// Requested OptiX clustering configuration.
    pub config: RtClusteringConfig,
}

impl RtClusteringEngine {
    /// Reject construction because OptiX was not compiled into this binary.
    pub fn new(_context: Arc<CudaContext>, _config: RtClusteringConfig) -> Result<Self> {
        bail!(OPTIX_DISABLED)
    }

    /// Reject loading an OptiX IR pipeline in a CUDA-only binary.
    pub fn load_pipeline(&mut self, _optixir_path: impl AsRef<Path>) -> Result<()> {
        bail!(OPTIX_DISABLED)
    }

    /// Reject OptiX BVH preparation in a CUDA-only binary.
    pub fn prepare_bvh(&mut self, _positions: &[f32], _max_epsilon: f32) -> Result<()> {
        bail!(OPTIX_DISABLED)
    }

    /// Reject OptiX clustering in a CUDA-only binary.
    pub fn cluster(&mut self, _positions: &[f32]) -> Result<RtClusteringResult> {
        bail!(OPTIX_DISABLED)
    }

    /// Reject OptiX clustering in a CUDA-only binary.
    pub fn cluster_at_epsilon(
        &mut self,
        _positions: &[f32],
        _epsilon: f32,
    ) -> Result<RtClusteringResult> {
        bail!(OPTIX_DISABLED)
    }
}

/// No OptiX IR is selectable when the `optix` feature is absent.
pub fn find_optixir_path() -> Option<std::path::PathBuf> {
    None
}

impl crate::spatial_index::SpatialNeighborIndex for RtClusteringEngine {
    fn backend(&self) -> crate::spatial_index::SpatialBackend {
        crate::spatial_index::SpatialBackend::OptixRt
    }

    fn prepare(&mut self, _positions: &[f32], _max_epsilon: f32) -> Result<()> {
        bail!(OPTIX_DISABLED)
    }

    fn query_at_epsilon(
        &mut self,
        _positions: &[f32],
        _epsilon: f32,
    ) -> Result<crate::spatial_index::NeighborQueryResult> {
        bail!(OPTIX_DISABLED)
    }
}
