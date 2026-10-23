//! Hierarchical RT-Core Clustering — SNDC Stage 3
//!
//! Multi-scale persistence clustering on GPU. Sweeps epsilon from tight (2Å)
//! to loose (8Å), running RT-core DBSCAN at each level. Clusters that persist
//! across many epsilon levels are robust binding sites; transient clusters are noise.
//!
//! Uses the existing [`RtClusteringEngine`] infrastructure (BVH build + ray casting)
//! via the new [`cluster_at_epsilon()`] method added in this stage.
//!
//! The output is a list of [`PersistentCluster`] structs with intensity²-weighted
//! centroids, persistence counts, and density scores sampled from the
//! [`SpikeDensityGrid`] (Stage 2).

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::CudaContext;

use crate::fused_engine::GpuSpikeEvent;
#[cfg(feature = "gpu")]
use crate::rt_clustering::{RtClusteringConfig, RtClusteringEngine};
#[cfg(feature = "gpu")]
use crate::spike_density::SpikeDensityGrid;

/// Configuration for hierarchical persistence clustering.
#[derive(Debug, Clone)]
pub struct HierarchicalConfig {
    /// Epsilon sweep: (start, end, step) in Å.
    /// e.g., (2.0, 8.0, 0.5) → 13 levels
    pub eps_range: (f32, f32, f32),
    /// Minimum persistence: number of epsilon levels a cluster must survive
    /// to be considered a real binding site.
    pub min_persistence: u32,
    /// Whether to weight spikes by intensity² during centroid computation.
    pub intensity_weighted: bool,
    /// Minimum spike count per cluster at any epsilon level.
    pub min_cluster_spikes: u32,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            eps_range: (2.0, 8.0, 0.5),
            min_persistence: 3,
            intensity_weighted: true,
            min_cluster_spikes: 10,
        }
    }
}

/// A cluster that has been tracked across multiple epsilon levels.
#[derive(Debug, Clone)]
pub struct PersistentCluster {
    /// Cluster ID (arbitrary, sequential)
    pub id: u32,
    /// Intensity²-weighted centroid [x, y, z] in Å
    pub centroid: [f64; 3],
    /// Number of spikes in cluster (at best epsilon level)
    pub spike_count: u32,
    /// Mean spike intensity
    pub mean_intensity: f64,
    /// Coefficient of variation of spike intensities
    pub intensity_cv: f64,
    /// Number of epsilon levels this cluster survived
    pub persistence: u32,
    /// Peak density sampled from SpikeDensityGrid at centroid
    pub peak_density: f64,
    /// Unique aromatic probe types (TRP=0, TYR=1, PHE=2, SS=3)
    pub probe_diversity: u32,
    /// Mean water density at spike positions
    pub mean_water_density: f64,
    /// Spatial spread: RMS distance of spikes from centroid (Å)
    pub cluster_radius: f64,
    /// Indices into the original spike array belonging to this cluster
    pub spike_indices: Vec<usize>,
}

/// Result of hierarchical clustering.
#[derive(Debug, Clone)]
pub struct HierarchicalResult {
    /// Persistent clusters sorted by peak_density descending.
    pub clusters: Vec<PersistentCluster>,
    /// Total epsilon levels swept.
    pub n_levels: u32,
    /// Total GPU time across all epsilon levels (ms).
    pub total_gpu_time_ms: f64,
}

/// Internal: a cluster at a single epsilon level, used for tracking.
#[derive(Debug, Clone)]
struct EpsilonCluster {
    /// Spike indices (into the original spike array)
    spike_indices: Vec<usize>,
    /// Centroid [x, y, z]
    centroid: [f64; 3],
}

/// Internal: a chain tracking cluster identity across levels.
#[derive(Debug)]
struct ClusterChain {
    /// Best epsilon level's spike indices (the one with max coverage)
    best_spike_indices: Vec<usize>,
    /// Best centroid
    best_centroid: [f64; 3],
    /// Best spike count
    best_spike_count: usize,
    /// Number of epsilon levels this chain has survived
    persistence: u32,
}

/// Hierarchical density-based clustering using RT cores.
///
/// Wraps [`RtClusteringEngine`] and sweeps epsilon to find clusters
/// that persist across spatial scales.
#[cfg(feature = "gpu")]
pub struct HierarchicalRtClustering {
    config: HierarchicalConfig,
    rt_engine: RtClusteringEngine,
}

#[cfg(feature = "gpu")]
impl HierarchicalRtClustering {
    /// Create a new hierarchical clustering engine.
    ///
    /// # Arguments
    /// * `config` — hierarchical clustering parameters
    /// * `rt_config` — base RT clustering config (epsilon will be overridden per-level)
    /// * `context` — shared CUDA context
    pub fn new(
        config: HierarchicalConfig,
        rt_config: RtClusteringConfig,
        context: Arc<CudaContext>,
    ) -> Result<Self> {
        let rt_engine = RtClusteringEngine::new(context, rt_config)
            .context("Failed to create RT clustering engine")?;
        Ok(Self { config, rt_engine })
    }

    /// Load the OptiX pipeline. Must be called before `cluster_spikes()`.
    ///
    /// # Arguments
    /// * `optixir_path` — path to the compiled RT clustering OptiXIR module
    pub fn load_pipeline(&mut self, optixir_path: impl AsRef<std::path::Path>) -> Result<()> {
        self.rt_engine.load_pipeline(optixir_path)
    }

    /// Cluster spikes using hierarchical persistence analysis.
    ///
    /// 1. Extract spike positions + intensities
    /// 2. Sweep epsilon from tight to loose, running RT-DBSCAN at each level
    /// 3. Track cluster persistence using spike-overlap chain matching
    /// 4. Compute intensity²-weighted centroids for persistent clusters
    /// 5. Sample density grid at centroids
    /// 6. Merge overlapping clusters (centroid distance < 4Å)
    /// 7. Return sorted by peak_density descending
    pub fn cluster_spikes(
        &mut self,
        spikes: &[GpuSpikeEvent],
        density_grid: &mut SpikeDensityGrid,
    ) -> Result<HierarchicalResult> {
        let n_spikes = spikes.len();
        if n_spikes == 0 {
            return Ok(HierarchicalResult {
                clusters: vec![],
                n_levels: 0,
                total_gpu_time_ms: 0.0,
            });
        }

        // ── 1. Extract flattened positions for RT engine ─────────────────
        let mut flat_positions = Vec::with_capacity(n_spikes * 3);
        for s in spikes {
            flat_positions.push(s.position[0]);
            flat_positions.push(s.position[1]);
            flat_positions.push(s.position[2]);
        }

        // ── 2. Compute epsilon levels ───────────────────────────────────
        let (eps_start, eps_end, eps_step) = self.config.eps_range;
        let mut epsilon_levels = Vec::new();
        let mut eps = eps_start;
        while eps <= eps_end + 1e-6 {
            epsilon_levels.push(eps);
            eps += eps_step;
        }
        let n_levels = epsilon_levels.len() as u32;

        log::info!(
            "Hierarchical clustering: {} spikes, {} epsilon levels [{:.1}..{:.1}]",
            n_spikes,
            n_levels,
            eps_start,
            eps_end
        );

        // ── 3. Sweep epsilon levels, track clusters ─────────────────────
        // Active chains: key = chain_id, value = ClusterChain
        let mut chains: Vec<ClusterChain> = Vec::new();
        let mut total_gpu_time = 0.0;

        // Previous level's clusters (for chain matching)
        let mut prev_clusters: Vec<EpsilonCluster> = Vec::new();
        // Map: prev_cluster_index → chain_id
        let mut prev_chain_map: Vec<usize> = Vec::new();

        for (level_idx, &epsilon) in epsilon_levels.iter().enumerate() {
            // Run RT-DBSCAN at this epsilon
            let result = self
                .rt_engine
                .cluster_at_epsilon(&flat_positions, epsilon)
                .with_context(|| format!("RT clustering failed at epsilon={:.1}", epsilon))?;
            total_gpu_time += result.gpu_time_ms;

            // Group spike indices by cluster_id (ignoring noise = -1)
            let mut cluster_groups: HashMap<i32, Vec<usize>> = HashMap::new();
            for (spike_idx, &cid) in result.cluster_ids.iter().enumerate() {
                if cid >= 0 {
                    cluster_groups.entry(cid).or_default().push(spike_idx);
                }
            }

            // Filter by minimum spike count
            let min_spikes = self.config.min_cluster_spikes as usize;
            let current_clusters: Vec<EpsilonCluster> = cluster_groups
                .into_values()
                .filter(|indices| indices.len() >= min_spikes)
                .map(|indices| {
                    let centroid =
                        compute_centroid(spikes, &indices, self.config.intensity_weighted);
                    EpsilonCluster {
                        spike_indices: indices,
                        centroid,
                    }
                })
                .collect();

            log::debug!(
                "  ε={:.1}Å: {} raw clusters → {} after size filter (min={})",
                epsilon,
                result.num_clusters,
                current_clusters.len(),
                min_spikes
            );

            if level_idx == 0 {
                // First level: each cluster starts a new chain
                for c in &current_clusters {
                    chains.push(ClusterChain {
                        best_spike_indices: c.spike_indices.clone(),
                        best_centroid: c.centroid,
                        best_spike_count: c.spike_indices.len(),
                        persistence: 1,
                    });
                }
                prev_chain_map = (0..current_clusters.len()).collect();
            } else {
                // Match current clusters to previous clusters via spike overlap
                let mut current_chain_map = Vec::with_capacity(current_clusters.len());
                let mut matched_chains: Vec<bool> = vec![false; chains.len()];

                for curr in &current_clusters {
                    let best_match = find_best_overlap(curr, &prev_clusters, &prev_chain_map);

                    if let Some(chain_id) = best_match {
                        // Extend existing chain
                        matched_chains[chain_id] = true;
                        chains[chain_id].persistence += 1;
                        // Update best if this level has more spikes
                        if curr.spike_indices.len() > chains[chain_id].best_spike_count {
                            chains[chain_id].best_spike_indices = curr.spike_indices.clone();
                            chains[chain_id].best_centroid = curr.centroid;
                            chains[chain_id].best_spike_count = curr.spike_indices.len();
                        }
                        current_chain_map.push(chain_id);
                    } else {
                        // New chain
                        let chain_id = chains.len();
                        chains.push(ClusterChain {
                            best_spike_indices: curr.spike_indices.clone(),
                            best_centroid: curr.centroid,
                            best_spike_count: curr.spike_indices.len(),
                            persistence: 1,
                        });
                        current_chain_map.push(chain_id);
                    }
                }

                prev_chain_map = current_chain_map;
            }

            prev_clusters = current_clusters;
        }

        // ── 4. Filter by minimum persistence ────────────────────────────
        let min_persistence = self.config.min_persistence;
        let persistent_chains: Vec<&ClusterChain> = chains
            .iter()
            .filter(|c| c.persistence >= min_persistence)
            .collect();

        log::info!(
            "  {} total chains, {} with persistence >= {}",
            chains.len(),
            persistent_chains.len(),
            min_persistence
        );

        // ── 5. Build PersistentCluster from each chain ──────────────────
        let mut clusters: Vec<PersistentCluster> = Vec::new();
        for (idx, chain) in persistent_chains.iter().enumerate() {
            let indices = &chain.best_spike_indices;
            let centroid = chain.best_centroid;

            // Mean intensity
            let intensities: Vec<f32> = indices.iter().map(|&i| spikes[i].intensity).collect();
            let mean_intensity =
                intensities.iter().map(|&x| x as f64).sum::<f64>() / intensities.len() as f64;

            // Intensity CV (coefficient of variation)
            let variance = intensities
                .iter()
                .map(|&x| {
                    let d = x as f64 - mean_intensity;
                    d * d
                })
                .sum::<f64>()
                / intensities.len() as f64;
            let std_dev = variance.sqrt();
            let intensity_cv = if mean_intensity > 1e-12 {
                std_dev / mean_intensity
            } else {
                0.0
            };

            // Probe diversity: count unique aromatic types (0..3)
            let mut probe_set = [false; 4]; // TRP=0, TYR=1, PHE=2, SS=3
            for &i in indices {
                let at = spikes[i].aromatic_type;
                if at >= 0 && at <= 3 {
                    probe_set[at as usize] = true;
                }
            }
            let probe_diversity = probe_set.iter().filter(|&&b| b).count() as u32;

            // Mean water density
            let mean_water_density = indices
                .iter()
                .map(|&i| spikes[i].water_density as f64)
                .sum::<f64>()
                / indices.len() as f64;

            // Cluster radius: RMS distance from centroid
            let rms_sq = indices
                .iter()
                .map(|&i| {
                    let dx = spikes[i].position[0] as f64 - centroid[0];
                    let dy = spikes[i].position[1] as f64 - centroid[1];
                    let dz = spikes[i].position[2] as f64 - centroid[2];
                    dx * dx + dy * dy + dz * dz
                })
                .sum::<f64>()
                / indices.len() as f64;
            let cluster_radius = rms_sq.sqrt();

            // Sample density grid at centroid
            let peak_density = density_grid
                .sample_at([centroid[0] as f32, centroid[1] as f32, centroid[2] as f32])
                .unwrap_or(0.0) as f64;

            clusters.push(PersistentCluster {
                id: idx as u32,
                centroid,
                spike_count: indices.len() as u32,
                mean_intensity,
                intensity_cv,
                persistence: chain.persistence,
                peak_density,
                probe_diversity,
                mean_water_density,
                cluster_radius,
                spike_indices: indices.clone(),
            });
        }

        // ── 6. Merge overlapping clusters (centroid distance < 4Å) ──────
        clusters = merge_overlapping(clusters, 4.0);

        // ── 7. Sort by peak_density descending ──────────────────────────
        clusters.sort_by(|a, b| {
            b.peak_density
                .partial_cmp(&a.peak_density)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Re-assign sequential IDs after merge + sort
        for (i, c) in clusters.iter_mut().enumerate() {
            c.id = i as u32;
        }

        log::info!(
            "Hierarchical clustering complete: {} persistent sites (total GPU time: {:.1}ms)",
            clusters.len(),
            total_gpu_time
        );

        Ok(HierarchicalResult {
            clusters,
            n_levels,
            total_gpu_time_ms: total_gpu_time,
        })
    }
}

// ─── Helper functions ──────────────────────────────────────────────────────

/// Compute intensity²-weighted or unweighted centroid for a set of spike indices.
fn compute_centroid(
    spikes: &[GpuSpikeEvent],
    indices: &[usize],
    intensity_weighted: bool,
) -> [f64; 3] {
    if indices.is_empty() {
        return [0.0; 3];
    }

    if intensity_weighted {
        let mut wx_sum = 0.0f64;
        let mut wy_sum = 0.0f64;
        let mut wz_sum = 0.0f64;
        let mut w_sum = 0.0f64;

        for &i in indices {
            let s = &spikes[i];
            let w = (s.intensity as f64) * (s.intensity as f64); // intensity²
            wx_sum += w * s.position[0] as f64;
            wy_sum += w * s.position[1] as f64;
            wz_sum += w * s.position[2] as f64;
            w_sum += w;
        }

        if w_sum > 1e-30 {
            [wx_sum / w_sum, wy_sum / w_sum, wz_sum / w_sum]
        } else {
            // Fallback to unweighted
            let n = indices.len() as f64;
            let cx = indices
                .iter()
                .map(|&i| spikes[i].position[0] as f64)
                .sum::<f64>()
                / n;
            let cy = indices
                .iter()
                .map(|&i| spikes[i].position[1] as f64)
                .sum::<f64>()
                / n;
            let cz = indices
                .iter()
                .map(|&i| spikes[i].position[2] as f64)
                .sum::<f64>()
                / n;
            [cx, cy, cz]
        }
    } else {
        let n = indices.len() as f64;
        let cx = indices
            .iter()
            .map(|&i| spikes[i].position[0] as f64)
            .sum::<f64>()
            / n;
        let cy = indices
            .iter()
            .map(|&i| spikes[i].position[1] as f64)
            .sum::<f64>()
            / n;
        let cz = indices
            .iter()
            .map(|&i| spikes[i].position[2] as f64)
            .sum::<f64>()
            / n;
        [cx, cy, cz]
    }
}

/// Find the best matching previous cluster by spike overlap.
///
/// Returns the chain_id of the previous cluster with >50% Jaccard overlap,
/// or None if no match.
fn find_best_overlap(
    current: &EpsilonCluster,
    prev_clusters: &[EpsilonCluster],
    prev_chain_map: &[usize],
) -> Option<usize> {
    if prev_clusters.is_empty() {
        return None;
    }

    // Build a set of current spike indices for fast lookup
    let current_set: std::collections::HashSet<usize> =
        current.spike_indices.iter().copied().collect();

    let mut best_chain: Option<usize> = None;
    let mut best_jaccard = 0.0f64;

    for (prev_idx, prev) in prev_clusters.iter().enumerate() {
        // Intersection: spikes in both current and previous cluster
        let intersection = prev
            .spike_indices
            .iter()
            .filter(|idx| current_set.contains(idx))
            .count();

        if intersection == 0 {
            continue;
        }

        // Jaccard similarity: |A ∩ B| / |A ∪ B|
        let union = current.spike_indices.len() + prev.spike_indices.len() - intersection;
        let jaccard = intersection as f64 / union as f64;

        // Require >30% overlap (clusters grow as epsilon increases, so strict
        // 50% would miss valid chains where a cluster absorbs nearby spikes)
        if jaccard > 0.3 && jaccard > best_jaccard {
            best_jaccard = jaccard;
            best_chain = Some(prev_chain_map[prev_idx]);
        }
    }

    best_chain
}

/// Merge clusters whose centroids are within `merge_radius` Å.
///
/// When two clusters merge, we keep the one with higher peak_density and
/// add the other's persistence. Spike indices are combined.
fn merge_overlapping(
    mut clusters: Vec<PersistentCluster>,
    merge_radius: f64,
) -> Vec<PersistentCluster> {
    if clusters.len() <= 1 {
        return clusters;
    }

    let merge_radius_sq = merge_radius * merge_radius;

    // Iteratively merge until no more pairs are within merge_radius
    let mut changed = true;
    while changed {
        changed = false;
        let mut merged_into: Vec<Option<usize>> = vec![None; clusters.len()];

        for i in 0..clusters.len() {
            if merged_into[i].is_some() {
                continue;
            }
            for j in (i + 1)..clusters.len() {
                if merged_into[j].is_some() {
                    continue;
                }

                let dx = clusters[i].centroid[0] - clusters[j].centroid[0];
                let dy = clusters[i].centroid[1] - clusters[j].centroid[1];
                let dz = clusters[i].centroid[2] - clusters[j].centroid[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < merge_radius_sq {
                    merged_into[j] = Some(i);
                    changed = true;
                }
            }
        }

        if changed {
            // Apply merges: collect indices to merge, build new list
            let mut new_clusters: Vec<PersistentCluster> = Vec::new();
            for i in 0..clusters.len() {
                if merged_into[i].is_some() {
                    continue;
                }

                let mut merged = clusters[i].clone();

                // Absorb all clusters merged into this one
                for j in 0..clusters.len() {
                    if merged_into[j] == Some(i) {
                        // Keep higher persistence
                        merged.persistence = merged.persistence.max(clusters[j].persistence);
                        // Sum spike counts
                        merged.spike_count += clusters[j].spike_count;
                        // Combine spike indices
                        merged
                            .spike_indices
                            .extend_from_slice(&clusters[j].spike_indices);
                        // Keep higher peak_density
                        if clusters[j].peak_density > merged.peak_density {
                            merged.peak_density = clusters[j].peak_density;
                            merged.centroid = clusters[j].centroid;
                        }
                        // Max probe diversity
                        merged.probe_diversity =
                            merged.probe_diversity.max(clusters[j].probe_diversity);
                    }
                }

                new_clusters.push(merged);
            }
            clusters = new_clusters;
        }
    }

    clusters
}

// ─── Unit tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_spike(
        pos: [f32; 3],
        intensity: f32,
        aromatic_type: i32,
        water_density: f32,
    ) -> GpuSpikeEvent {
        GpuSpikeEvent {
            timestep: 0,
            voxel_idx: 0,
            position: pos,
            intensity,
            nearby_residues: [0; 8],
            n_residues: 0,
            spike_source: 1,
            wavelength_nm: 280.0,
            aromatic_type,
            aromatic_residue_id: -1,
            water_density,
            vibrational_energy: 0.0,
            n_nearby_excited: 0,
            wd_change: 0.0,
            phase_bits: 0,
            ..GpuSpikeEvent::default()
        }
    }

    #[test]
    fn test_compute_centroid_unweighted() {
        let spikes = vec![
            make_spike([0.0, 0.0, 0.0], 1.0, 0, 0.5),
            make_spike([10.0, 0.0, 0.0], 1.0, 1, 0.3),
        ];
        let indices = vec![0, 1];
        let c = compute_centroid(&spikes, &indices, false);
        assert!(
            (c[0] - 5.0).abs() < 1e-6,
            "x centroid should be 5.0, got {}",
            c[0]
        );
        assert!(c[1].abs() < 1e-6);
        assert!(c[2].abs() < 1e-6);
    }

    #[test]
    fn test_compute_centroid_intensity_weighted() {
        // Spike at origin has intensity 2.0 → weight 4.0
        // Spike at (10,0,0) has intensity 1.0 → weight 1.0
        // Weighted centroid: 4*0 + 1*10 / (4+1) = 2.0
        let spikes = vec![
            make_spike([0.0, 0.0, 0.0], 2.0, 0, 0.5),
            make_spike([10.0, 0.0, 0.0], 1.0, 1, 0.3),
        ];
        let indices = vec![0, 1];
        let c = compute_centroid(&spikes, &indices, true);
        assert!(
            (c[0] - 2.0).abs() < 1e-6,
            "weighted x centroid should be 2.0, got {}",
            c[0]
        );
    }

    #[test]
    fn test_compute_centroid_empty() {
        let spikes: Vec<GpuSpikeEvent> = vec![];
        let indices: Vec<usize> = vec![];
        let c = compute_centroid(&spikes, &indices, true);
        assert_eq!(c, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_find_best_overlap_no_prev() {
        let current = EpsilonCluster {
            spike_indices: vec![0, 1, 2, 3, 4],
            centroid: [0.0; 3],
        };
        assert!(find_best_overlap(&current, &[], &[]).is_none());
    }

    #[test]
    fn test_find_best_overlap_strong_match() {
        let current = EpsilonCluster {
            spike_indices: vec![0, 1, 2, 3, 4, 5, 6, 7],
            centroid: [0.0; 3],
        };
        let prev = vec![
            EpsilonCluster {
                spike_indices: vec![0, 1, 2, 3, 4],
                centroid: [0.0; 3],
            },
            EpsilonCluster {
                spike_indices: vec![100, 101, 102],
                centroid: [50.0, 0.0, 0.0],
            },
        ];
        let chain_map = vec![0, 1]; // chain 0, chain 1

        let result = find_best_overlap(&current, &prev, &chain_map);
        assert_eq!(result, Some(0), "should match first prev cluster (chain 0)");
    }

    #[test]
    fn test_find_best_overlap_no_match() {
        let current = EpsilonCluster {
            spike_indices: vec![100, 101, 102, 103, 104],
            centroid: [50.0, 0.0, 0.0],
        };
        let prev = vec![EpsilonCluster {
            spike_indices: vec![0, 1, 2, 3, 4],
            centroid: [0.0; 3],
        }];
        let chain_map = vec![0];

        let result = find_best_overlap(&current, &prev, &chain_map);
        assert!(result.is_none(), "disjoint clusters should not match");
    }

    #[test]
    fn test_merge_overlapping_close_clusters() {
        let clusters = vec![
            PersistentCluster {
                id: 0,
                centroid: [10.0, 10.0, 10.0],
                spike_count: 100,
                mean_intensity: 1.5,
                intensity_cv: 0.3,
                persistence: 5,
                peak_density: 50.0,
                probe_diversity: 3,
                mean_water_density: 0.4,
                cluster_radius: 3.0,
                spike_indices: (0..100).collect(),
            },
            PersistentCluster {
                id: 1,
                centroid: [12.0, 10.0, 10.0], // 2Å away → within 4Å merge radius
                spike_count: 50,
                mean_intensity: 1.2,
                intensity_cv: 0.2,
                persistence: 3,
                peak_density: 30.0,
                probe_diversity: 2,
                mean_water_density: 0.5,
                cluster_radius: 2.5,
                spike_indices: (100..150).collect(),
            },
        ];

        let merged = merge_overlapping(clusters, 4.0);
        assert_eq!(merged.len(), 1, "two close clusters should merge into one");
        assert_eq!(merged[0].spike_count, 150, "spike counts should sum");
        assert_eq!(merged[0].persistence, 5, "persistence should be max");
    }

    #[test]
    fn test_merge_overlapping_distant_clusters() {
        let clusters = vec![
            PersistentCluster {
                id: 0,
                centroid: [10.0, 10.0, 10.0],
                spike_count: 100,
                mean_intensity: 1.5,
                intensity_cv: 0.3,
                persistence: 5,
                peak_density: 50.0,
                probe_diversity: 3,
                mean_water_density: 0.4,
                cluster_radius: 3.0,
                spike_indices: (0..100).collect(),
            },
            PersistentCluster {
                id: 1,
                centroid: [30.0, 30.0, 30.0], // ~34.6Å away → well beyond merge radius
                spike_count: 80,
                mean_intensity: 1.8,
                intensity_cv: 0.2,
                persistence: 4,
                peak_density: 60.0,
                probe_diversity: 4,
                mean_water_density: 0.3,
                cluster_radius: 4.0,
                spike_indices: (100..180).collect(),
            },
        ];

        let merged = merge_overlapping(clusters, 4.0);
        assert_eq!(merged.len(), 2, "distant clusters should NOT merge");
    }

    #[test]
    fn test_hierarchical_config_default() {
        let config = HierarchicalConfig::default();
        assert_eq!(config.eps_range, (2.0, 8.0, 0.5));
        assert_eq!(config.min_persistence, 3);
        assert!(config.intensity_weighted);
        assert_eq!(config.min_cluster_spikes, 10);

        // Count levels
        let mut count = 0u32;
        let mut eps = config.eps_range.0;
        while eps <= config.eps_range.1 + 1e-6 {
            count += 1;
            eps += config.eps_range.2;
        }
        assert_eq!(count, 13, "default range should produce 13 levels");
    }

    #[test]
    fn test_persistent_cluster_fields() {
        let pc = PersistentCluster {
            id: 0,
            centroid: [1.0, 2.0, 3.0],
            spike_count: 42,
            mean_intensity: 1.5,
            intensity_cv: 0.3,
            persistence: 7,
            peak_density: 55.0,
            probe_diversity: 3,
            mean_water_density: 0.45,
            cluster_radius: 4.2,
            spike_indices: vec![0, 1, 2],
        };
        assert_eq!(pc.spike_count, 42);
        assert_eq!(pc.persistence, 7);
        assert!((pc.peak_density - 55.0).abs() < 1e-6);
    }
}
