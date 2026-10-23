//! KD-tree for O(log n) spatial neighbor queries
//!
//! Optimized for 3D point clouds (Cα coordinates). Uses a median-split
//! construction for balanced trees. Thread-safe for concurrent queries.

use std::cmp::Ordering;

/// A 3D point with an associated index
#[derive(Clone, Copy, Debug)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub idx: usize,
}

impl Point3D {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32, idx: usize) -> Self {
        Self { x, y, z, idx }
    }

    #[inline]
    pub fn coords(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    #[inline]
    pub fn get(&self, dim: usize) -> f32 {
        match dim {
            0 => self.x,
            1 => self.y,
            2 => self.z,
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn distance_sq(&self, other: &Point3D) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        dx * dx + dy * dy + dz * dz
    }
}

/// KD-tree node
#[derive(Debug)]
struct KdNode {
    /// The point stored at this node
    point: Point3D,
    /// Split dimension (0, 1, or 2)
    split_dim: usize,
    /// Left subtree (points with smaller coordinate in split dimension)
    left: Option<Box<KdNode>>,
    /// Right subtree (points with larger coordinate in split dimension)
    right: Option<Box<KdNode>>,
}

/// 3D KD-tree for efficient spatial queries
///
/// Construction: O(n log n)
/// Range query: O(√n + k) where k is the number of points returned
/// Nearest neighbor: O(log n) average
#[derive(Debug)]
pub struct KdTree {
    root: Option<Box<KdNode>>,
    size: usize,
}

impl KdTree {
    /// Build a KD-tree from a slice of 3D coordinates
    ///
    /// Each coordinate is (x, y, z). The index is the position in the input slice.
    pub fn build(coords: &[[f32; 3]]) -> Self {
        let mut points: Vec<Point3D> = coords
            .iter()
            .enumerate()
            .map(|(idx, &[x, y, z])| Point3D::new(x, y, z, idx))
            .collect();

        let size = points.len();
        let root = Self::build_recursive(&mut points, 0);

        KdTree { root, size }
    }

    /// Build from explicit (x, y, z, idx) tuples
    pub fn build_with_indices(points: &[(f32, f32, f32, usize)]) -> Self {
        let mut points: Vec<Point3D> = points
            .iter()
            .map(|&(x, y, z, idx)| Point3D::new(x, y, z, idx))
            .collect();

        let size = points.len();
        let root = Self::build_recursive(&mut points, 0);

        KdTree { root, size }
    }

    fn build_recursive(points: &mut [Point3D], depth: usize) -> Option<Box<KdNode>> {
        if points.is_empty() {
            return None;
        }

        let dim = depth % 3;

        // Sort by the current dimension
        points.sort_by(|a, b| {
            a.get(dim)
                .partial_cmp(&b.get(dim))
                .unwrap_or(Ordering::Equal)
        });

        let mid = points.len() / 2;

        // Find the actual median (handle duplicates)
        let median_val = points[mid].get(dim);

        // Ensure we don't split equal values incorrectly
        let split_idx = mid;

        let (left_slice, right_slice) = points.split_at_mut(split_idx);
        let (pivot, right_slice) = right_slice.split_first_mut().unwrap());

        Some(Box::new(KdNode {
            point: *pivot,
            split_dim: dim,
            left: Self::build_recursive(left_slice, depth + 1),
            right: Self::build_recursive(right_slice, depth + 1),
        }))
    }

    /// Returns the number of points in the tree
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Returns true if the tree is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Find all points within radius of the query point
    ///
    /// Returns a vector of (index, distance) pairs, sorted by distance.
    pub fn radius_search(&self, query: [f32; 3], radius: f32) -> Vec<(usize, f32)> {
        let mut results = Vec::new();
        let query_point = Point3D::new(query[0], query[1], query[2], usize::MAX);
        let radius_sq = radius * radius;

        if let Some(ref root) = self.root {
            self.radius_search_recursive(root, &query_point, radius_sq, &mut results);
        }

        // Sort by distance and return
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal);
        results
    }

    fn radius_search_recursive(
        &self,
        node: &KdNode,
        query: &Point3D,
        radius_sq: f32,
        results: &mut Vec<(usize, f32)>,
    ) {
        let dist_sq = query.distance_sq(&node.point);
        if dist_sq <= radius_sq {
            results.push((node.point.idx, dist_sq.sqrt());
        }

        let dim = node.split_dim;
        let diff = query.get(dim) - node.point.get(dim);
        let diff_sq = diff * diff;

        // Determine which subtree to search first
        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search the closer subtree
        if let Some(ref child) = first {
            self.radius_search_recursive(child, query, radius_sq, results);
        }

        // Search the farther subtree if the splitting plane is within radius
        if diff_sq <= radius_sq {
            if let Some(ref child) = second {
                self.radius_search_recursive(child, query, radius_sq, results);
            }
        }
    }

    /// Find the k nearest neighbors to the query point
    ///
    /// Returns a vector of (index, distance) pairs, sorted by distance (nearest first).
    pub fn knn(&self, query: [f32; 3], k: usize) -> Vec<(usize, f32)> {
        use std::collections::BinaryHeap;

        if k == 0 || self.is_empty() {
            return Vec::new();
        }

        let query_point = Point3D::new(query[0], query[1], query[2], usize::MAX);

        // Max-heap to keep track of k nearest (we want to easily remove the farthest)
        let mut heap: BinaryHeap<(ordered_float::OrderedFloat<f32>, usize)> = BinaryHeap::new());

        if let Some(ref root) = self.root {
            self.knn_recursive(root, &query_point, k, &mut heap);
        }

        // Convert to sorted vector (nearest first)
        let mut results: Vec<_> = heap
            .into_iter()
            .map(|(d, idx)| (idx, d.0))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal);
        results
    }

    fn knn_recursive(
        &self,
        node: &KdNode,
        query: &Point3D,
        k: usize,
        heap: &mut std::collections::BinaryHeap<(ordered_float::OrderedFloat<f32>, usize)>,
    ) {
        let dist_sq = query.distance_sq(&node.point);
        let dist = dist_sq.sqrt();

        // If we have fewer than k points, or this point is closer than the farthest
        if heap.len() < k {
            heap.push((ordered_float::OrderedFloat(dist), node.point.idx);
        } else if let Some(&(max_dist, _)) = heap.peek() {
            if dist < max_dist.0 {
                heap.pop();
                heap.push((ordered_float::OrderedFloat(dist), node.point.idx);
            }
        }

        let dim = node.split_dim;
        let diff = query.get(dim) - node.point.get(dim);
        let diff_sq = diff * diff;

        // Determine which subtree to search first
        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        // Search the closer subtree first
        if let Some(ref child) = first {
            self.knn_recursive(child, query, k, heap);
        }

        // Only search the farther subtree if it could contain closer points
        let should_search_second = heap.len() < k || {
            if let Some(&(max_dist, _)) = heap.peek() {
                diff_sq < max_dist.0 * max_dist.0
            } else {
                true
            }
        };

        if should_search_second {
            if let Some(ref child) = second {
                self.knn_recursive(child, query, k, heap);
            }
        }
    }

    /// Find all points within radius, limited to max_results
    ///
    /// Returns indices only (no distances) for memory efficiency.
    pub fn radius_search_indices(&self, query: [f32; 3], radius: f32, max_results: usize) -> Vec<usize> {
        let mut results = Vec::with_capacity(max_results.min(64);
        let query_point = Point3D::new(query[0], query[1], query[2], usize::MAX);
        let radius_sq = radius * radius;

        if let Some(ref root) = self.root {
            self.radius_search_indices_recursive(root, &query_point, radius_sq, max_results, &mut results);
        }

        results
    }

    fn radius_search_indices_recursive(
        &self,
        node: &KdNode,
        query: &Point3D,
        radius_sq: f32,
        max_results: usize,
        results: &mut Vec<usize>,
    ) {
        if results.len() >= max_results {
            return;
        }

        let dist_sq = query.distance_sq(&node.point);
        if dist_sq <= radius_sq {
            results.push(node.point.idx);
            if results.len() >= max_results {
                return;
            }
        }

        let dim = node.split_dim;
        let diff = query.get(dim) - node.point.get(dim);
        let diff_sq = diff * diff;

        let (first, second) = if diff < 0.0 {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        if let Some(ref child) = first {
            self.radius_search_indices_recursive(child, query, radius_sq, max_results, results);
        }

        if diff_sq <= radius_sq && results.len() < max_results {
            if let Some(ref child) = second {
                self.radius_search_indices_recursive(child, query, radius_sq, max_results, results);
            }
        }
    }
}

// Thread-safe: KdTree is immutable after construction
unsafe impl Send for KdTree {}
unsafe impl Sync for KdTree {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_search() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [5.0, 5.0, 5.0],
        ];

        let tree = KdTree::build(&coords);
        assert_eq!(tree.len(), 5);

        // Search around origin with radius 1.5
        let results = tree.radius_search([0.0, 0.0, 0.0], 1.5);
        assert_eq!(results.len(), 4); // All but (5,5,5)

        // Verify distances
        for (idx, dist) in &results {
            assert!(dist <= &1.5);
            assert!(*idx < 5);
        }
    }

    #[test]
    fn test_knn() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ];

        let tree = KdTree::build(&coords);

        // Find 3 nearest to origin
        let results = tree.knn([0.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);

        // Should be indices 0, 1, 2 (distances 0, 1, 2)
        assert_eq!(results[0].0, 0);
        assert!((results[0].1 - 0.0).abs() < 1e-6);
        assert_eq!(results[1].0, 1);
        assert!((results[1].1 - 1.0).abs() < 1e-6);
        assert_eq!(results[2].0, 2);
        assert!((results[2].1 - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_tree() {
        let coords: Vec<[f32; 3]> = vec![];
        let tree = KdTree::build(&coords);
        assert!(tree.is_empty();
        assert!(tree.radius_search([0.0, 0.0, 0.0], 10.0).is_empty();
        assert!(tree.knn([0.0, 0.0, 0.0], 5).is_empty();
    }

    #[test]
    fn test_single_point() {
        let coords = vec![[1.0, 2.0, 3.0]];
        let tree = KdTree::build(&coords);

        let results = tree.radius_search([1.0, 2.0, 3.0], 0.1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }
}
