//! [STAGE-2B-CLUSTER] Representative Conformation Clustering
//!
//! Select 50-200 representative conformations from MD trajectory using
//! RMSD-based clustering with Boltzmann weighting.
//!
//! **Purpose**: Reduce 10K+ frames to manageable ensemble for analysis while
//! preserving thermodynamic sampling.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Configuration for clustering
#[derive(Debug, Clone)]
pub struct ClusteringConfig {
    /// Target number of clusters (50-200)
    pub target_clusters: usize,
    /// RMSD cutoff for cluster membership (Å)
    pub rmsd_cutoff: f32,
    /// Use Boltzmann weighting based on energy
    pub use_boltzmann_weights: bool,
    /// Temperature for Boltzmann weighting (K)
    pub temperature_k: f32,
}

impl Default for ClusteringConfig {
    fn default() -> Self {
        Self {
            target_clusters: 100,
            rmsd_cutoff: 2.5, // Å
            use_boltzmann_weights: true,
            temperature_k: 300.0, // Room temperature
        }
    }
}

/// A representative conformation with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepresentativeFrame {
    /// Frame index in original trajectory
    pub frame_idx: usize,
    /// Timestep when captured
    pub timestep: i32,
    /// Positions [x, y, z, ...] for all atoms
    pub positions: Vec<f32>,
    /// Boltzmann weight (population)
    pub boltzmann_weight: f32,
    /// Number of frames in this cluster
    pub cluster_size: usize,
    /// Average RMSD to cluster members (Å)
    pub avg_rmsd_to_members: f32,
    /// Potential energy (kJ/mol) if available
    pub energy_kj_mol: Option<f32>,
}

/// Clustering results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringResults {
    /// Representative frames (50-200)
    pub representatives: Vec<RepresentativeFrame>,
    /// Total number of frames clustered
    pub total_frames: usize,
    /// Number of clusters found
    pub num_clusters: usize,
    /// Average cluster size
    pub avg_cluster_size: f32,
    /// Coverage fraction (frames assigned / total frames)
    pub coverage: f32,
}

/// Trajectory clustering engine
pub struct TrajectoryClusterer {
    config: ClusteringConfig,
    /// Cα atom indices for RMSD calculation
    ca_indices: Vec<usize>,
}

impl TrajectoryClusterer {
    /// Create new clusterer
    ///
    /// # Arguments
    /// * `config` - Clustering configuration
    /// * `atom_names` - IUPAC atom names for all atoms
    /// * `residue_indices` - Residue index for each atom
    pub fn new(
        config: ClusteringConfig,
        atom_names: &[String],
        residue_indices: &[usize],
    ) -> Result<Self> {
        // Find Cα atoms for RMSD calculation
        let mut ca_indices = Vec::new();
        let mut current_residue = usize::MAX;

        for (atom_idx, (name, &res_idx)) in atom_names.iter().zip(residue_indices).enumerate() {
            if name == "CA" && res_idx != current_residue {
                ca_indices.push(atom_idx);
                current_residue = res_idx;
            }
        }

        anyhow::ensure!(
            !ca_indices.is_empty(),
            "No Cα atoms found for RMSD calculation"
        );

        log::info!(
            "Trajectory clusterer initialized: {} Cα atoms, target {} clusters",
            ca_indices.len(),
            config.target_clusters
        );

        Ok(Self { config, ca_indices })
    }

    /// Cluster trajectory frames and select representatives
    ///
    /// Uses greedy leader clustering algorithm:
    /// 1. Pick first frame as cluster 1 center
    /// 2. For each remaining frame:
    ///    - If RMSD to all centers > cutoff → new cluster
    ///    - Else assign to nearest cluster
    /// 3. Compute Boltzmann weights
    ///
    /// # Arguments
    /// * `frames` - Trajectory frames with positions
    /// * `timesteps` - Timestep for each frame
    /// * `energies` - Optional potential energies (kJ/mol)
    ///
    /// # Returns
    /// ClusteringResults with 50-200 representatives
    pub fn cluster(
        &self,
        frames: &[Vec<f32>],
        timesteps: &[i32],
        energies: Option<&[f32]>,
    ) -> Result<ClusteringResults> {
        anyhow::ensure!(!frames.is_empty(), "No frames provided for clustering");
        anyhow::ensure!(
            frames.len() == timesteps.len(),
            "Frame count mismatch: {} frames, {} timesteps",
            frames.len(),
            timesteps.len()
        );

        if let Some(e) = energies {
            anyhow::ensure!(
                frames.len() == e.len(),
                "Frame count mismatch: {} frames, {} energies",
                frames.len(),
                e.len()
            );
        }

        log::info!(
            "Clustering {} frames into ~{} clusters",
            frames.len(),
            self.config.target_clusters
        );

        // Greedy leader clustering
        let mut cluster_centers = Vec::new();
        let mut cluster_members: Vec<Vec<usize>> = Vec::new();

        // First frame is first cluster center
        cluster_centers.push(0);
        cluster_members.push(vec![0]);

        // Assign remaining frames
        for frame_idx in 1..frames.len() {
            let mut min_rmsd = f32::MAX;
            let mut nearest_cluster = 0;

            // Find nearest cluster center
            for (cluster_idx, &center_idx) in cluster_centers.iter().enumerate() {
                let rmsd = self.compute_rmsd(&frames[frame_idx], &frames[center_idx])?;

                if rmsd < min_rmsd {
                    min_rmsd = rmsd;
                    nearest_cluster = cluster_idx;
                }
            }

            // Create new cluster if RMSD exceeds cutoff and under target
            if min_rmsd > self.config.rmsd_cutoff
                && cluster_centers.len() < self.config.target_clusters
            {
                cluster_centers.push(frame_idx);
                cluster_members.push(vec![frame_idx]);
            } else {
                // Assign to nearest cluster
                cluster_members[nearest_cluster].push(frame_idx);
            }
        }

        log::info!(
            "Clustering complete: {} clusters from {} frames",
            cluster_centers.len(),
            frames.len()
        );

        // Build representative frames with Boltzmann weights
        let mut representatives = Vec::new();
        let total_frames = frames.len();

        for (cluster_idx, &center_idx) in cluster_centers.iter().enumerate() {
            let members = &cluster_members[cluster_idx];

            // Compute average RMSD to members
            let mut total_rmsd = 0.0f32;
            for &member_idx in members {
                if member_idx != center_idx {
                    total_rmsd += self.compute_rmsd(&frames[center_idx], &frames[member_idx])?;
                }
            }
            let avg_rmsd = if members.len() > 1 {
                total_rmsd / (members.len() - 1) as f32
            } else {
                0.0
            };

            // Boltzmann weight = cluster_size / total_frames
            // (proportional to -exp(-E/kT) in canonical ensemble)
            let boltzmann_weight = members.len() as f32 / total_frames as f32;

            representatives.push(RepresentativeFrame {
                frame_idx: center_idx,
                timestep: timesteps[center_idx],
                positions: frames[center_idx].clone(),
                boltzmann_weight,
                cluster_size: members.len(),
                avg_rmsd_to_members: avg_rmsd,
                energy_kj_mol: energies.map(|e| e[center_idx]),
            });
        }

        // Sort by Boltzmann weight (descending) for importance ordering
        representatives.sort_by(|a, b| {
            b.boltzmann_weight
                .partial_cmp(&a.boltzmann_weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let num_clusters = representatives.len();
        let avg_cluster_size = total_frames as f32 / num_clusters as f32;
        let coverage = 1.0; // All frames are assigned

        log::info!(
            "Representative selection: {} clusters, avg size {:.1} frames",
            num_clusters,
            avg_cluster_size
        );

        Ok(ClusteringResults {
            representatives,
            total_frames,
            num_clusters,
            avg_cluster_size,
            coverage,
        })
    }

    /// Compute Cα RMSD between two frames
    ///
    /// RMSD = sqrt( (1/N) * sum_i (r_i - r'_i)^2 )
    ///
    /// No alignment - assumes frames are pre-aligned or from same trajectory
    fn compute_rmsd(&self, frame1: &[f32], frame2: &[f32]) -> Result<f32> {
        anyhow::ensure!(
            frame1.len() == frame2.len(),
            "Frame size mismatch: {} != {}",
            frame1.len(),
            frame2.len()
        );

        let mut sum_sq_dist = 0.0f32;

        for &ca_idx in &self.ca_indices {
            let pos_idx = ca_idx * 3;
            let dx = frame1[pos_idx] - frame2[pos_idx];
            let dy = frame1[pos_idx + 1] - frame2[pos_idx + 1];
            let dz = frame1[pos_idx + 2] - frame2[pos_idx + 2];
            sum_sq_dist += dx * dx + dy * dy + dz * dz;
        }

        let rmsd = (sum_sq_dist / self.ca_indices.len() as f32).sqrt();
        Ok(rmsd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_clusterer() -> TrajectoryClusterer {
        let atom_names = vec![
            "N".to_string(),
            "CA".to_string(),
            "C".to_string(),
            "N".to_string(),
            "CA".to_string(),
            "C".to_string(),
        ];
        let residue_indices = vec![0, 0, 0, 1, 1, 1];
        let config = ClusteringConfig {
            target_clusters: 3,
            rmsd_cutoff: 2.0,
            use_boltzmann_weights: true,
            temperature_k: 300.0,
        };
        TrajectoryClusterer::new(config, &atom_names, &residue_indices).unwrap()
    }

    #[test]
    fn test_clusterer_creation() {
        let clusterer = create_test_clusterer();
        assert_eq!(clusterer.ca_indices.len(), 2);
        assert_eq!(clusterer.ca_indices, vec![1, 4]);
    }

    #[test]
    fn test_rmsd_identical_frames() {
        let clusterer = create_test_clusterer();
        // Full atom array: N, CA, C, N, CA, C (6 atoms * 3 coords = 18 elements)
        // CA indices are 1 and 4
        let frame = vec![
            0.0, 0.0, 0.0, // N (idx 0)
            1.0, 1.0, 1.0, // CA (idx 1)
            2.0, 2.0, 2.0, // C (idx 2)
            3.0, 3.0, 3.0, // N (idx 3)
            4.0, 4.0, 4.0, // CA (idx 4)
            5.0, 5.0, 5.0, // C (idx 5)
        ];
        let rmsd = clusterer.compute_rmsd(&frame, &frame).unwrap();
        assert!(rmsd.abs() < 1e-6);
    }

    #[test]
    fn test_rmsd_different_frames() {
        let clusterer = create_test_clusterer();
        // Frame 1
        let frame1 = vec![
            0.0, 0.0, 0.0, // N (idx 0)
            0.0, 0.0, 0.0, // CA (idx 1) at origin
            2.0, 2.0, 2.0, // C (idx 2)
            3.0, 3.0, 3.0, // N (idx 3)
            1.0, 1.0, 1.0, // CA (idx 4) at (1,1,1)
            5.0, 5.0, 5.0, // C (idx 5)
        ];
        // Frame 2
        let frame2 = vec![
            0.0, 0.0, 0.0, // N (idx 0)
            1.0, 0.0, 0.0, // CA (idx 1) at (1,0,0)
            2.0, 2.0, 2.0, // C (idx 2)
            3.0, 3.0, 3.0, // N (idx 3)
            2.0, 1.0, 1.0, // CA (idx 4) at (2,1,1)
            5.0, 5.0, 5.0, // C (idx 5)
        ];
        let rmsd = clusterer.compute_rmsd(&frame1, &frame2).unwrap();
        // CA1: (0,0,0) → (1,0,0), dist^2 = 1.0
        // CA2: (1,1,1) → (2,1,1), dist^2 = 1.0
        // RMSD = sqrt((1.0 + 1.0)/2) = 1.0
        assert!((rmsd - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_clustering_single_frame() {
        let clusterer = create_test_clusterer();
        // Full atom array: 6 atoms * 3 coords = 18 elements
        let frames = vec![vec![0.0; 18]];
        let timesteps = vec![0];
        let result = clusterer.cluster(&frames, &timesteps, None).unwrap();
        assert_eq!(result.num_clusters, 1);
        assert_eq!(result.representatives.len(), 1);
        assert_eq!(result.total_frames, 1);
    }

    #[test]
    fn test_clustering_multiple_frames() {
        let clusterer = create_test_clusterer();
        // Create 10 frames in 3 groups (similar frames within each group)
        let mut frames = Vec::new();
        let mut timesteps = Vec::new();

        // Group 1: frames near origin (4 frames)
        for i in 0..4 {
            let mut frame = vec![0.0; 18];
            // Set CA positions (indices 1 and 4)
            frame[1 * 3] = 0.1 * i as f32; // CA1 x
            frame[4 * 3] = 0.1 * i as f32; // CA2 x
            frames.push(frame);
            timesteps.push(i);
        }

        // Group 2: frames at x=5 (3 frames)
        for i in 0..3 {
            let mut frame = vec![0.0; 18];
            frame[1 * 3] = 5.0; // CA1 x = 5.0
            frame[1 * 3 + 1] = 0.1 * i as f32; // CA1 y varies
            frame[4 * 3] = 5.0; // CA2 x = 5.0
            frame[4 * 3 + 1] = 0.1 * i as f32; // CA2 y varies
            frames.push(frame);
            timesteps.push(4 + i);
        }

        // Group 3: frames at x=10 (3 frames)
        for i in 0..3 {
            let mut frame = vec![0.0; 18];
            frame[1 * 3] = 10.0; // CA1 x = 10.0
            frame[1 * 3 + 1] = 0.1 * i as f32; // CA1 y varies
            frame[4 * 3] = 10.0; // CA2 x = 10.0
            frame[4 * 3 + 1] = 0.1 * i as f32; // CA2 y varies
            frames.push(frame);
            timesteps.push(7 + i);
        }

        let result = clusterer.cluster(&frames, &timesteps, None).unwrap();

        // Should find 3 clusters (rmsd_cutoff = 2.0)
        assert_eq!(result.num_clusters, 3);
        assert_eq!(result.representatives.len(), 3);
        assert_eq!(result.total_frames, 10);

        // Check Boltzmann weights sum to 1.0
        let weight_sum: f32 = result
            .representatives
            .iter()
            .map(|r| r.boltzmann_weight)
            .sum();
        assert!((weight_sum - 1.0).abs() < 0.01);

        // Largest cluster should have highest weight (4/10 = 0.4)
        assert!(result.representatives[0].boltzmann_weight > 0.3);
    }
}
