//! Spike Avalanche Detection and Cryptic Site Classification
//!
//! Detects spatially and temporally coherent spike clusters that
//! indicate cooperative dewetting events (cryptic pocket opening).
//!
//! # Avalanche Criteria
//!
//! A valid cryptic site avalanche must:
//! 1. Have ≥ MIN_AVALANCHE_SIZE spikes within SPATIAL_THRESHOLD
//! 2. Persist for ≥ TEMPORAL_WINDOW frames
//! 3. Form a contiguous spatial cluster
//! 4. Have estimated volume ≥ MIN_DRUGGABLE_VOLUME

use crate::config::NhsConfig;
use crate::exclusion::ExclusionGrid;
use crate::neuromorphic::DewettingNetwork;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// =============================================================================
// CRYPTIC SITE EVENT
// =============================================================================

/// Detected cryptic site event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSiteEvent {
    /// Unique site identifier
    pub site_id: u64,

    /// Frame when first detected
    pub first_frame: usize,

    /// Most recent frame
    pub last_frame: usize,

    /// Centroid position (Å)
    pub centroid: [f32; 3],

    /// Estimated pocket volume (Å³)
    pub volume: f32,

    /// Residue indices involved
    pub residues: Vec<u32>,

    /// Voxel indices in cluster
    pub voxels: Vec<usize>,

    /// Number of spikes in avalanche
    pub spike_count: usize,

    /// Confidence score [0, 1]
    pub confidence: f32,

    /// Is this a druggable pocket?
    pub is_druggable: bool,

    /// Open frequency (fraction of frames with activity)
    pub open_frequency: f32,

    /// Was this causally validated by UV bias?
    pub uv_validated: bool,

    // === CCNS (Conformational Crackling Noise Spectroscopy) fields ===
    /// Per-spike amplitudes (volume delta or intensity at each spike event)
    pub spike_amplitudes: Vec<f32>,

    /// Frame index of each spike event
    pub spike_frames: Vec<usize>,

    /// Inter-spike intervals (waiting times between successive spikes, in frames)
    pub inter_spike_intervals: Vec<f32>,
}

// =============================================================================
// SPIKE CLUSTER (INTERMEDIATE)
// =============================================================================

/// Intermediate spike cluster before classification
#[derive(Debug, Clone)]
struct SpikeCluster {
    /// Neuron indices in cluster
    spike_indices: Vec<usize>,

    /// World positions of spikes
    positions: Vec<[f32; 3]>,

    /// Voxel indices
    voxels: Vec<usize>,

    /// Cluster centroid
    centroid: [f32; 3],

    /// Estimated volume
    volume: f32,

    /// Frame first observed
    frame_first_seen: usize,

    /// Number of frames with activity
    frame_count: usize,

    // CCNS tracking fields
    /// Per-spike amplitudes accumulated across frames
    spike_amplitudes: Vec<f32>,

    /// Frame index for each spike event
    spike_frames: Vec<usize>,
}

// =============================================================================
// AVALANCHE DETECTOR
// =============================================================================

/// Avalanche detector with temporal tracking
pub struct AvalancheDetector {
    config: NhsConfig,

    /// Current frame index
    current_frame: usize,

    /// Active clusters being tracked
    active_clusters: HashMap<u64, SpikeCluster>,

    /// Next cluster ID
    next_cluster_id: u64,

    /// Completed events ready for output
    completed_events: VecDeque<CrypticSiteEvent>,

    /// Site ID counter
    next_site_id: u64,

    /// UV-validated residue pairs (for marking events)
    uv_validated_residues: HashSet<u32>,
}

impl AvalancheDetector {
    pub fn new(config: NhsConfig) -> Self {
        Self {
            config,
            current_frame: 0,
            active_clusters: HashMap::new(),
            next_cluster_id: 0,
            completed_events: VecDeque::new(),
            next_site_id: 0,
            uv_validated_residues: HashSet::new(),
        }
    }

    /// Process spikes from current frame
    ///
    /// Returns any newly completed cryptic site events
    pub fn process_spikes(
        &mut self,
        network: &DewettingNetwork,
        grid: &ExclusionGrid,
    ) -> Vec<CrypticSiteEvent> {
        self.current_frame += 1;

        // Get spike data
        let spike_positions = network.get_spike_positions(grid);
        let spike_voxels = network.get_spike_voxels();
        let spike_residues = network.get_spike_residues();

        if spike_positions.is_empty() {
            // No spikes: age out old clusters
            self.age_clusters(&spike_residues);
            return self.drain_completed();
        }

        // Spatial clustering of current spikes
        let frame_clusters = self.spatial_clustering(&spike_positions, &spike_voxels);

        // Match with existing tracked clusters
        self.match_and_update_clusters(frame_clusters, &spike_residues);

        // Age out old clusters and promote mature ones
        self.age_clusters(&spike_residues);

        self.drain_completed()
    }

    /// Process spikes from pre-extracted data
    ///
    /// Alternative to process_spikes that takes data directly to avoid borrow issues
    pub fn process_spikes_data(
        &mut self,
        spike_positions: &[[f32; 3]],
        spike_voxels: &[usize],
        spike_residues: &[u32],
    ) -> Vec<CrypticSiteEvent> {
        self.current_frame += 1;

        if spike_positions.is_empty() {
            // No spikes: age out old clusters
            self.age_clusters(spike_residues);
            return self.drain_completed();
        }

        // Spatial clustering of current spikes
        let frame_clusters = self.spatial_clustering(spike_positions, spike_voxels);

        // Match with existing tracked clusters
        self.match_and_update_clusters(frame_clusters, spike_residues);

        // Age out old clusters and promote mature ones
        self.age_clusters(spike_residues);

        self.drain_completed()
    }

    /// Mark residues as UV-validated (causal link established)
    pub fn mark_uv_validated(&mut self, residues: &[u32]) {
        self.uv_validated_residues.extend(residues.iter().copied());
    }

    /// DBSCAN-like spatial clustering
    fn spatial_clustering(&self, positions: &[[f32; 3]], voxels: &[usize]) -> Vec<SpikeCluster> {
        let eps = self.config.avalanche_spatial_threshold;
        let min_pts = self.config.min_avalanche_spikes;

        let mut visited = vec![false; positions.len()];
        let mut clusters = Vec::new();

        for i in 0..positions.len() {
            if visited[i] {
                continue;
            }

            // Find neighbors within eps
            let neighbors = self.region_query(positions, i, eps);

            if neighbors.len() < min_pts {
                visited[i] = true; // Noise point
                continue;
            }

            // Start new cluster
            let mut cluster_indices = Vec::new();
            let mut cluster_positions = Vec::new();
            let mut cluster_voxels = Vec::new();
            let mut seeds: VecDeque<usize> = neighbors.into_iter().collect();

            while let Some(j) = seeds.pop_front() {
                if visited[j] {
                    continue;
                }
                visited[j] = true;

                cluster_indices.push(j);
                cluster_positions.push(positions[j]);
                if j < voxels.len() {
                    cluster_voxels.push(voxels[j]);
                }

                let j_neighbors = self.region_query(positions, j, eps);
                if j_neighbors.len() >= min_pts {
                    for n in j_neighbors {
                        if !visited[n] {
                            seeds.push_back(n);
                        }
                    }
                }
            }

            if cluster_positions.len() >= min_pts {
                let centroid = compute_centroid(&cluster_positions);
                let volume = estimate_volume(&cluster_positions, self.config.grid_spacing);
                clusters.push(SpikeCluster {
                    spike_indices: cluster_indices,
                    positions: cluster_positions,
                    voxels: cluster_voxels,
                    centroid,
                    volume,
                    frame_first_seen: self.current_frame,
                    frame_count: 1,
                    spike_amplitudes: vec![volume; 1], // first frame: volume as amplitude proxy
                    spike_frames: vec![self.current_frame],
                });
            }
        }

        clusters
    }

    /// Find all points within eps distance
    fn region_query(&self, positions: &[[f32; 3]], center_idx: usize, eps: f32) -> Vec<usize> {
        let center = positions[center_idx];
        let eps_sq = eps * eps;

        positions
            .iter()
            .enumerate()
            .filter(|(i, pos)| {
                if *i == center_idx {
                    return true;
                }
                let dx = pos[0] - center[0];
                let dy = pos[1] - center[1];
                let dz = pos[2] - center[2];
                dx * dx + dy * dy + dz * dz <= eps_sq
            })
            .map(|(i, _)| i)
            .collect()
    }

    /// Match new clusters with existing tracked clusters
    fn match_and_update_clusters(&mut self, new_clusters: Vec<SpikeCluster>, _residues: &[u32]) {
        let match_threshold = self.config.avalanche_spatial_threshold * 2.0;

        for new_cluster in new_clusters {
            // Find best matching existing cluster
            let mut best_match: Option<u64> = None;
            let mut best_dist = f32::MAX;

            for (&cluster_id, existing) in &self.active_clusters {
                let dist = distance(&new_cluster.centroid, &existing.centroid);
                if dist < match_threshold && dist < best_dist {
                    best_match = Some(cluster_id);
                    best_dist = dist;
                }
            }

            if let Some(cluster_id) = best_match {
                // Update existing cluster
                let existing = self.active_clusters.get_mut(&cluster_id).unwrap();
                existing.centroid = new_cluster.centroid;
                existing.positions = new_cluster.positions;
                existing.spike_indices = new_cluster.spike_indices;
                existing.voxels = new_cluster.voxels;
                // Track per-frame amplitude (volume as proxy) and frame index
                existing.spike_amplitudes.push(new_cluster.volume);
                existing.spike_frames.push(self.current_frame);
                existing.volume = (existing.volume + new_cluster.volume) / 2.0; // Running average
                existing.frame_count += 1;
            } else {
                // Start tracking new cluster
                let cluster_id = self.next_cluster_id;
                self.next_cluster_id += 1;
                self.active_clusters.insert(cluster_id, new_cluster);
            }
        }
    }

    /// Age out old clusters, promote mature ones to events
    fn age_clusters(&mut self, residues: &[u32]) {
        let temporal_window = self.config.avalanche_temporal_window;
        let min_volume = self.config.min_volume;

        let mut to_remove = Vec::new();
        let mut to_complete = Vec::new();

        for (&cluster_id, cluster) in &self.active_clusters {
            let age = self.current_frame - cluster.frame_first_seen;

            if age >= temporal_window {
                // Cluster is mature
                if cluster.frame_count >= temporal_window / 2 && cluster.volume >= min_volume {
                    to_complete.push(cluster_id);
                } else {
                    // Didn't persist
                    to_remove.push(cluster_id);
                }
            } else if self.current_frame > cluster.frame_first_seen + temporal_window * 3 {
                // Too old without maturing
                to_remove.push(cluster_id);
            }
        }

        // Complete mature clusters
        for cluster_id in to_complete {
            if let Some(cluster) = self.active_clusters.remove(&cluster_id) {
                let event = self.cluster_to_event(cluster, residues);
                self.completed_events.push_back(event);
            }
        }

        // Remove stale clusters
        for cluster_id in to_remove {
            self.active_clusters.remove(&cluster_id);
        }
    }

    /// Convert cluster to output event
    fn cluster_to_event(&mut self, cluster: SpikeCluster, _residues: &[u32]) -> CrypticSiteEvent {
        let site_id = self.next_site_id;
        self.next_site_id += 1;

        // Compute confidence based on persistence and size
        let persistence = cluster.frame_count as f32 / self.config.avalanche_temporal_window as f32;
        let size_score = (cluster.spike_indices.len() as f32 / 20.0).min(1.0);
        let volume_score = (cluster.volume / 500.0).min(1.0);
        let confidence = (persistence * 0.4 + size_score * 0.3 + volume_score * 0.3).min(1.0);

        let is_druggable = cluster.volume >= self.config.min_volume && confidence > 0.5;

        let open_frequency = cluster.frame_count as f32
            / (self.current_frame - cluster.frame_first_seen).max(1) as f32;

        // Check UV validation
        let uv_validated = false; // Will be updated by pipeline

        // Compute inter-spike intervals from frame indices
        let inter_spike_intervals = if cluster.spike_frames.len() >= 2 {
            cluster
                .spike_frames
                .windows(2)
                .map(|w| (w[1] as f32) - (w[0] as f32))
                .collect()
        } else {
            Vec::new()
        };

        CrypticSiteEvent {
            site_id,
            first_frame: cluster.frame_first_seen,
            last_frame: self.current_frame,
            centroid: cluster.centroid,
            volume: cluster.volume,
            residues: Vec::new(), // Populated by caller
            voxels: cluster.voxels,
            spike_count: cluster.spike_indices.len(),
            confidence,
            is_druggable,
            open_frequency,
            uv_validated,
            spike_amplitudes: cluster.spike_amplitudes,
            spike_frames: cluster.spike_frames,
            inter_spike_intervals,
        }
    }

    /// Drain completed events
    fn drain_completed(&mut self) -> Vec<CrypticSiteEvent> {
        self.completed_events.drain(..).collect()
    }

    /// Get number of active clusters being tracked
    pub fn num_active_clusters(&self) -> usize {
        self.active_clusters.len()
    }

    /// Reset for new trajectory
    pub fn reset(&mut self) {
        self.current_frame = 0;
        self.active_clusters.clear();
        self.completed_events.clear();
        self.uv_validated_residues.clear();
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Compute centroid of positions
fn compute_centroid(positions: &[[f32; 3]]) -> [f32; 3] {
    if positions.is_empty() {
        return [0.0; 3];
    }

    let n = positions.len() as f32;
    let mut centroid = [0.0f32; 3];

    for pos in positions {
        centroid[0] += pos[0];
        centroid[1] += pos[1];
        centroid[2] += pos[2];
    }

    [centroid[0] / n, centroid[1] / n, centroid[2] / n]
}

/// Estimate pocket volume from spike positions
fn estimate_volume(positions: &[[f32; 3]], grid_spacing: f32) -> f32 {
    // Simple estimation: points × voxel volume × expansion factor
    let voxel_volume = grid_spacing.powi(3);
    let expansion = 2.0; // Account for gaps between spikes
    positions.len() as f32 * voxel_volume * expansion
}

/// Euclidean distance
fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_centroid() {
        let positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, 0.0]];
        let c = compute_centroid(&positions);
        assert!((c[0] - 1.0).abs() < 0.001);
        assert!((c[1] - 1.0).abs() < 0.001);
        assert!((c[2] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_distance() {
        assert!((distance(&[0.0, 0.0, 0.0], &[3.0, 4.0, 0.0]) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_volume_estimation() {
        let positions = vec![[0.0; 3]; 10];
        let vol = estimate_volume(&positions, 0.5);
        // 10 × 0.125 × 2.0 = 2.5 Å³
        assert!((vol - 2.5).abs() < 0.01);
    }
}
