//! CPU pre-mapping of spatial neighborhoods using KD-tree
//!
//! This module builds neighborhood mappings on CPU using Rayon parallelism,
//! then uploads the compact representation to GPU for TDA computation.

use super::kdtree::KdTree;
use super::half_utils::{f32_to_f16, PackedDistanceMatrix};
use super::{MAX_NEIGHBORS, TDA_RADII, NUM_RADII};
use rayon::prelude::*;
use std::sync::Arc;

/// A single residue's spatial neighborhood at a specific radius
#[derive(Clone, Debug)]
pub struct SpatialNeighborhood {
    /// Center residue index
    pub center_idx: usize,
    /// Neighbor indices (up to MAX_NEIGHBORS)
    pub neighbor_indices: Vec<usize>,
    /// Neighbor distances in F16 format
    pub neighbor_distances_f16: Vec<u16>,
    /// Neighbor 3D coordinates (for directional features)
    pub neighbor_coords: Vec<[f32; 3]>,
    /// Radius used for this neighborhood
    pub radius: f32,
}

impl SpatialNeighborhood {
    /// Number of neighbors
    #[inline]
    pub fn len(&self) -> usize {
        self.neighbor_indices.len()
    }

    /// Check if empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.neighbor_indices.is_empty()
    }
}

/// Multi-radius neighborhood data for a single residue
#[derive(Clone, Debug)]
pub struct MultiRadiusNeighborhood {
    /// Center residue index
    pub center_idx: usize,
    /// Center coordinates
    pub center_coords: [f32; 3],
    /// Neighborhoods at each radius (indexed by radius_idx 0..NUM_RADII)
    pub by_radius: [SpatialNeighborhood; NUM_RADII],
}

/// Compact GPU-uploadable format for neighborhood data
#[derive(Clone, Debug)]
pub struct NeighborhoodData {
    /// Number of residues
    pub n_residues: usize,
    /// Number of radii
    pub n_radii: usize,
    /// Offsets into neighbor_indices for each (residue, radius) pair
    /// Shape: [n_residues * n_radii + 1]
    pub offsets: Vec<u32>,
    /// Packed neighbor indices
    pub neighbor_indices: Vec<u32>,
    /// Packed neighbor distances (F16)
    pub neighbor_distances: Vec<u16>,
    /// Packed neighbor coordinates (3 floats per neighbor)
    pub neighbor_coords: Vec<f32>,
    /// Center coordinates for all residues
    pub center_coords: Vec<f32>,
}

impl NeighborhoodData {
    /// Get neighbors for a specific residue and radius
    pub fn get_neighbors(&self, residue_idx: usize, radius_idx: usize) -> &[u32] {
        let offset_idx = residue_idx * self.n_radii + radius_idx;
        let start = self.offsets[offset_idx] as usize;
        let end = self.offsets[offset_idx + 1] as usize;
        &self.neighbor_indices[start..end]
    }

    /// Total neighbor count across all residues and radii
    pub fn total_neighbors(&self) -> usize {
        self.neighbor_indices.len()
    }

    /// Memory usage in bytes
    pub fn memory_bytes(&self) -> usize {
        self.offsets.len() * 4
            + self.neighbor_indices.len() * 4
            + self.neighbor_distances.len() * 2
            + self.neighbor_coords.len() * 4
            + self.center_coords.len() * 4
    }
}

/// Builder for spatial neighborhoods using KD-tree
pub struct NeighborhoodBuilder {
    /// Maximum neighbors per residue (per radius)
    max_neighbors: usize,
    /// Radii to use for neighborhood queries
    radii: Vec<f32>,
}

impl NeighborhoodBuilder {
    /// Create a new builder with default settings
    pub fn new() -> Self {
        Self {
            max_neighbors: MAX_NEIGHBORS,
            radii: TDA_RADII.to_vec(),
        }
    }

    /// Set maximum neighbors per radius
    pub fn with_max_neighbors(mut self, max: usize) -> Self {
        self.max_neighbors = max;
        self
    }

    /// Set custom radii
    pub fn with_radii(mut self, radii: Vec<f32>) -> Self {
        self.radii = radii;
        self
    }

    /// Build neighborhood data from Cα coordinates
    ///
    /// Uses Rayon for parallel neighborhood queries.
    pub fn build(&self, coords: &[[f32; 3]]) -> NeighborhoodData {
        let n_residues = coords.len();
        let n_radii = self.radii.len();

        if n_residues == 0 {
            return NeighborhoodData {
                n_residues: 0,
                n_radii,
                offsets: vec![0],
                neighbor_indices: vec![],
                neighbor_distances: vec![],
                neighbor_coords: vec![],
                center_coords: vec![],
            };
        }

        // Build KD-tree
        let tree = Arc::new(KdTree::build(coords);

        // Parallel query for all residues
        let max_radius = self.radii.iter().cloned().fold(0.0f32, f32::max);
        let max_neighbors = self.max_neighbors;
        let radii = self.radii.clone();

        // Collect all neighborhoods in parallel
        let neighborhoods: Vec<Vec<(Vec<usize>, Vec<u16>, Vec<[f32; 3]>)>> = (0..n_residues)
            .into_par_iter()
            .map(|i| {
                let center = coords[i];

                // Query at maximum radius first
                let all_neighbors = tree.radius_search(center, max_radius);

                // Filter for each radius
                radii.iter().map(|&radius| {
                    let mut indices = Vec::with_capacity(max_neighbors);
                    let mut distances = Vec::with_capacity(max_neighbors);
                    let mut neighbor_coords = Vec::with_capacity(max_neighbors);

                    for &(idx, dist) in &all_neighbors {
                        if idx == i {
                            continue; // Skip self
                        }
                        if dist <= radius && indices.len() < max_neighbors {
                            indices.push(idx);
                            distances.push(f32_to_f16(dist);
                            neighbor_coords.push(coords[idx]);
                        }
                    }

                    (indices, distances, neighbor_coords)
                }).collect()
            })
            .collect();

        // Flatten into compact format
        let total_entries = n_residues * n_radii;
        let mut offsets = Vec::with_capacity(total_entries + 1);
        let mut all_indices = Vec::new();
        let mut all_distances = Vec::new();
        let mut all_coords = Vec::new();

        offsets.push(0u32);

        for i in 0..n_residues {
            for r in 0..n_radii {
                let (ref indices, ref distances, ref ncoords) = neighborhoods[i][r];

                for &idx in indices {
                    all_indices.push(idx as u32);
                }
                all_distances.extend_from_slice(distances);
                for coord in ncoords {
                    all_coords.extend_from_slice(coord);
                }

                offsets.push(all_indices.len() as u32);
            }
        }

        // Center coordinates (flattened)
        let center_coords: Vec<f32> = coords.iter().flat_map(|c| c.iter().copied()).collect();

        NeighborhoodData {
            n_residues,
            n_radii,
            offsets,
            neighbor_indices: all_indices,
            neighbor_distances: all_distances,
            neighbor_coords: all_coords,
            center_coords,
        }
    }

    /// Build neighborhoods for a single radius (optimized path)
    pub fn build_single_radius(&self, coords: &[[f32; 3]], radius: f32) -> NeighborhoodData {
        let n_residues = coords.len();
        let n_radii = 1;

        if n_residues == 0 {
            return NeighborhoodData {
                n_residues: 0,
                n_radii: 1,
                offsets: vec![0],
                neighbor_indices: vec![],
                neighbor_distances: vec![],
                neighbor_coords: vec![],
                center_coords: vec![],
            };
        }

        let tree = Arc::new(KdTree::build(coords);
        let max_neighbors = self.max_neighbors;

        // Parallel query
        let neighborhoods: Vec<_> = (0..n_residues)
            .into_par_iter()
            .map(|i| {
                let center = coords[i];
                let neighbors = tree.radius_search(center, radius);

                let mut indices = Vec::with_capacity(max_neighbors);
                let mut distances = Vec::with_capacity(max_neighbors);
                let mut neighbor_coords = Vec::with_capacity(max_neighbors);

                for &(idx, dist) in &neighbors {
                    if idx == i {
                        continue;
                    }
                    if indices.len() >= max_neighbors {
                        break;
                    }
                    indices.push(idx as u32);
                    distances.push(f32_to_f16(dist);
                    neighbor_coords.push(coords[idx]);
                }

                (indices, distances, neighbor_coords)
            })
            .collect();

        // Flatten
        let mut offsets = Vec::with_capacity(n_residues + 1);
        let mut all_indices = Vec::new();
        let mut all_distances = Vec::new();
        let mut all_coords = Vec::new();

        offsets.push(0u32);

        for (indices, distances, ncoords) in neighborhoods {
            all_indices.extend(indices);
            all_distances.extend(distances);
            for coord in ncoords {
                all_coords.extend_from_slice(&coord);
            }
            offsets.push(all_indices.len() as u32);
        }

        let center_coords: Vec<f32> = coords.iter().flat_map(|c| c.iter().copied()).collect();

        NeighborhoodData {
            n_residues,
            n_radii,
            offsets,
            neighbor_indices: all_indices,
            neighbor_distances: all_distances,
            neighbor_coords: all_coords,
            center_coords,
        }
    }
}

impl Default for NeighborhoodBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute pairwise distance matrix for a small neighborhood
///
/// Returns upper triangle as a flat array in F16 format.
pub fn compute_local_distances(coords: &[[f32; 3]]) -> Vec<u16> {
    let n = coords.len();
    let size = n * (n - 1) / 2;
    let mut distances = Vec::with_capacity(size);

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = coords[i][0] - coords[j][0];
            let dy = coords[i][1] - coords[j][1];
            let dz = coords[i][2] - coords[j][2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            distances.push(f32_to_f16(dist);
        }
    }

    distances
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_coords() -> Vec<[f32; 3]> {
        // Simple helix-like structure
        let mut coords = Vec::new();
        for i in 0..20 {
            let t = i as f32 * 0.5;
            coords.push([
                t.cos() * 5.0,
                t.sin() * 5.0,
                i as f32 * 1.5,
            ]);
        }
        coords
    }

    #[test]
    fn test_neighborhood_builder() {
        let coords = make_test_coords();
        let builder = NeighborhoodBuilder::new());
        let data = builder.build(&coords);

        assert_eq!(data.n_residues, 20);
        assert_eq!(data.n_radii, 3); // Default: 8, 12, 16 Å

        // Check offsets are monotonically increasing
        for i in 1..data.offsets.len() {
            assert!(data.offsets[i] >= data.offsets[i - 1]);
        }

        // Check that each residue has some neighbors at the largest radius
        for i in 0..20 {
            let neighbors = data.get_neighbors(i, 2); // 16Å radius
            assert!(!neighbors.is_empty(), "Residue {} should have neighbors at 16Å", i);
        }
    }

    #[test]
    fn test_single_radius() {
        let coords = make_test_coords();
        let builder = NeighborhoodBuilder::new());
        let data = builder.build_single_radius(&coords, 8.0);

        assert_eq!(data.n_residues, 20);
        assert_eq!(data.n_radii, 1);
    }

    #[test]
    fn test_local_distances() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];

        let distances = compute_local_distances(&coords);
        assert_eq!(distances.len(), 3); // 3 pairs

        // Check distances are reasonable (F16 precision)
        use super::super::half_utils::f16_to_f32;
        let d01 = f16_to_f32(distances[0]);
        let d02 = f16_to_f32(distances[1]);
        let d12 = f16_to_f32(distances[2]);

        assert!((d01 - 1.0).abs() < 0.01);
        assert!((d02 - 1.0).abs() < 0.01);
        assert!((d12 - 1.414).abs() < 0.01);
    }

    #[test]
    fn test_empty_coords() {
        let coords: Vec<[f32; 3]> = vec![];
        let builder = NeighborhoodBuilder::new());
        let data = builder.build(&coords);

        assert_eq!(data.n_residues, 0);
        assert!(data.neighbor_indices.is_empty();
    }
}
