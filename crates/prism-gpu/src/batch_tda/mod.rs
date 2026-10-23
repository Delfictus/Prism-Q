//! Batch TDA (Topological Data Analysis) with Spatial Neighborhoods
//!
//! This module implements TDA feature extraction using spatial neighborhoods
//! (KD-tree based) rather than sequential tile-based approaches. Key insight:
//! binding sites span 50-200+ residues spatially but are scattered in sequence.
//!
//! ## Architecture
//!
//! 1. **KD-tree Construction**: O(n log n) spatial indexing of Cα coordinates
//! 2. **Neighborhood Mapping**: Parallel radius queries (8Å, 12Å, 16Å)
//! 3. **GPU TDA Computation**: Warp-cooperative Betti number calculation
//! 4. **Streaming Pipeline**: Double-buffered overlapping of CPU prep + GPU compute
//!
//! ## Feature Dimensions
//!
//! - Base features from MegaFused: 32
//! - TDA features: 48 (3 radii × 16 features per radius)
//! - Total combined: 80

pub mod kdtree;
pub mod half_utils;
pub mod neighborhood;
pub mod executor;
pub mod hybrid_executor;
pub mod streaming;

// Re-exports
pub use kdtree::KdTree;
pub use half_utils::{f32_to_f16, f16_to_f32, F16};
pub use neighborhood::{NeighborhoodBuilder, SpatialNeighborhood, NeighborhoodData};
pub use executor::BatchTdaExecutor;
pub use hybrid_executor::{HybridTdaExecutor, HybridTdaConfig, TdaFeatures};
pub use streaming::{StreamingTdaPipeline, StreamingConfig};

// ============================================================================
// Constants
// ============================================================================

/// Base feature dimension from MegaFused kernel
pub const BASE_FEATURES: usize = 32;

/// TDA feature count: 3 radii × 16 features per radius
pub const TDA_FEATURE_COUNT: usize = 48;

/// Total combined feature dimension
pub const TOTAL_COMBINED_FEATURES: usize = BASE_FEATURES + TDA_FEATURE_COUNT;

/// Number of TDA radii for multi-resolution analysis
pub const NUM_RADII: usize = 3;

/// TDA neighborhood radii in Angstroms
pub const TDA_RADII: [f32; NUM_RADII] = [8.0, 12.0, 16.0];

/// Features extracted per radius
pub const FEATURES_PER_RADIUS: usize = 16;

/// TDA persistence scales in Angstroms (for filtration)
pub const TDA_SCALES: [f32; 4] = [3.0, 5.0, 7.0, 9.0];

/// Maximum neighbors per residue (memory limit)
pub const MAX_NEIGHBORS: usize = 64;

/// Threshold for fused mode (small structures fit in shared memory)
pub const FUSED_MODE_THRESHOLD: usize = 500;

/// Maximum prefetch depth for streaming pipeline
pub const MAX_PREFETCH: usize = 4;

/// Streaming threshold (structures in batch)
pub const STREAMING_THRESHOLD: usize = 10;

// Feature indices within the 16-feature-per-radius block
pub mod feature_idx {
    /// Betti-0 at scale 0 (connected components at 3Å)
    pub const BETTI0_SCALE0: usize = 0;
    /// Betti-0 at scale 1 (connected components at 5Å)
    pub const BETTI0_SCALE1: usize = 1;
    /// Betti-0 at scale 2 (connected components at 7Å)
    pub const BETTI0_SCALE2: usize = 2;
    /// Betti-0 at scale 3 (connected components at 9Å)
    pub const BETTI0_SCALE3: usize = 3;

    /// Betti-1 at scale 0 (loops at 3Å)
    pub const BETTI1_SCALE0: usize = 4;
    /// Betti-1 at scale 1 (loops at 5Å)
    pub const BETTI1_SCALE1: usize = 5;
    /// Betti-1 at scale 2 (loops at 7Å)
    pub const BETTI1_SCALE2: usize = 6;
    /// Betti-1 at scale 3 (loops at 9Å)
    pub const BETTI1_SCALE3: usize = 7;

    /// Total persistence (sum of death - birth)
    pub const TOTAL_PERSISTENCE: usize = 8;
    /// Maximum persistence gap
    pub const MAX_PERSISTENCE: usize = 9;
    /// Persistence entropy
    pub const PERSISTENCE_ENTROPY: usize = 10;
    /// Number of significant features (persistence > threshold)
    pub const SIGNIFICANT_FEATURES: usize = 11;

    /// Directional: +X hemisphere density
    pub const DIR_PLUS_X: usize = 12;
    /// Directional: +Y hemisphere density
    pub const DIR_PLUS_Y: usize = 13;
    /// Directional: +Z hemisphere density
    pub const DIR_PLUS_Z: usize = 14;
    /// Anisotropy score (variance in directional densities)
    pub const ANISOTROPY: usize = 15;
}

/// Get the global feature index for a TDA feature
#[inline]
pub fn tda_feature_index(radius_idx: usize, feature_idx: usize) -> usize {
    BASE_FEATURES + radius_idx * FEATURES_PER_RADIUS + feature_idx
}

/// Validate that all constants are consistent
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_consistency() {
        assert_eq!(TDA_FEATURE_COUNT, NUM_RADII * FEATURES_PER_RADIUS);
        assert_eq!(TOTAL_COMBINED_FEATURES, BASE_FEATURES + TDA_FEATURE_COUNT);
        assert_eq!(TDA_RADII.len(), NUM_RADII);
        assert_eq!(TDA_SCALES.len(), 4); // 4 scales per radius
    }

    #[test]
    fn test_feature_index() {
        // First radius, first feature = 32
        assert_eq!(tda_feature_index(0, 0), 32);
        // Second radius, first feature = 32 + 16 = 48
        assert_eq!(tda_feature_index(1, 0), 48);
        // Last feature = 32 + 48 - 1 = 79
        assert_eq!(tda_feature_index(2, 15), 79);
    }
}
