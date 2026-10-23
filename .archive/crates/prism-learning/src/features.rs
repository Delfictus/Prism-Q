//! Target-Aware Feature Extraction for Cryptic Site Prediction
//!
//! This module implements VRAM-efficient, target-aware feature extraction that enables
//! the RL agent to generalize across diverse protein targets.
//!
//! ## Feature Categories (23 total dimensions by default)
//!
//! | Category | Dims | Description |
//! |----------|------|-------------|
//! | Global | 3 | Size, Radius of Gyration, Density |
//! | Target Neighborhood | 8 | Target-specific exposure and contacts |
//! | Stability | 4 | RMSD, clashes, max displacement |
//! | Family Flags | 4 | Multimer, glycan, size class |
//! | Temporal | 4 | Change from initial state |
//!
//! ## Design Principles
//! - O(N) complexity using spatial hashing
//! - No large matrix allocations (VRAM-safe)
//! - Invariant to protein identity (generalizable)
//! - Computed on CPU with simple reductions

use crate::atomic_chemistry::{
    calculate_anisotropy, calculate_electrostatic_frustration, calculate_hydrophobic_exposure,
    AtomicMetadata, ResidueType,
};
use crate::buffers::SimulationBuffers;
use crate::manifest::{FeatureConfig, ProteinTarget};
use prism_io::sovereign_types::Atom;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

// GPU-accelerated bio-chemistry feature extraction (optional)
#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
#[cfg(feature = "cuda")]
use prism_gpu::{BiochemistryGpu, GpuAtomicMetadata};
#[cfg(feature = "cuda")]
use std::sync::Arc;

// ============================================================================
// FEATURE VECTOR
// ============================================================================

/// Complete feature vector for RL agent input
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Raw feature values
    pub values: Vec<f32>,
    /// Feature dimension
    pub dim: usize,
}

impl FeatureVector {
    /// Create new feature vector with given dimension
    pub fn new(dim: usize) -> Self {
        Self {
            values: vec![0.0; dim],
            dim,
        }
    }

    /// Get slice for agent input
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Clone into owned Vec
    pub fn to_vec(&self) -> Vec<f32> {
        self.values.clone()
    }
}

// ============================================================================
// FEATURE EXTRACTOR
// ============================================================================

/// Target-aware feature extractor
///
/// Maintains reference to initial state for temporal features
pub struct FeatureExtractor {
    config: FeatureConfig,
    target_residues: HashSet<usize>,
    core_residues: HashSet<usize>,
    is_multimer: bool,
    has_glycans: bool,
    initial_exposure: f32,
    initial_rg: f32,
    /// Difficulty encoding for difficulty-aware learning (v3.1.1)
    difficulty: String,
    /// Glycan residue positions for shield tracking (v3.1.1)
    glycan_residues: HashSet<usize>,
    /// Adjacent residues for partial discovery tracking (v3.1.1)
    adjacent_residues: HashSet<usize>,
    /// Cryptic site mechanism type (v3.1.1)
    mechanism: String,
    /// Initial glycan positions for displacement calculation (v3.1.1)
    initial_glycan_positions: Vec<f32>,
    /// Atomic-level metadata for bio-chemistry features (v3.1.1)
    /// Contains residue types, hydrophobicity, partial charges, Cα indices
    atomic_metadata: Option<AtomicMetadata>,
    /// Initial positions for bio-chemistry delta calculations
    initial_positions: Vec<f32>,
    /// Target residue indices as Vec for bio-chemistry calculations
    target_residue_vec: Vec<usize>,
    /// GPU-accelerated bio-chemistry extractor (v3.1.2)
    /// Uses RefCell for interior mutability (GPU compute modifies buffers)
    #[cfg(feature = "cuda")]
    gpu_biochemistry: RefCell<Option<BiochemistryGpu>>,
    /// Whether GPU features were requested (for non-cuda builds)
    #[cfg(not(feature = "cuda"))]
    _gpu_features_requested: bool,
}

impl FeatureExtractor {
    /// Create new feature extractor for a target
    pub fn new(config: FeatureConfig, target: &ProteinTarget) -> Self {
        // Try to initialize GPU bio-chemistry extractor if requested
        #[cfg(feature = "cuda")]
        let gpu_biochemistry = if config.use_gpu_features && config.include_bio_chemistry {
            Self::try_init_gpu_biochemistry(config.neighbor_cutoff)
        } else {
            None
        };

        Self {
            config,
            target_residues: target.target_residues.iter().cloned().collect(),
            core_residues: target.core_residues.iter().cloned().collect(),
            is_multimer: target.is_multimer,
            has_glycans: target.has_glycans,
            initial_exposure: 0.0,
            initial_rg: 0.0,
            difficulty: target.difficulty.clone(),
            glycan_residues: target.glycan_residues.iter().cloned().collect(),
            adjacent_residues: target.adjacent_residues.iter().cloned().collect(),
            mechanism: target.mechanism.clone(),
            initial_glycan_positions: Vec::new(),
            atomic_metadata: None,
            initial_positions: Vec::new(),
            target_residue_vec: target.target_residues.clone(),
            #[cfg(feature = "cuda")]
            gpu_biochemistry: RefCell::new(gpu_biochemistry),
            #[cfg(not(feature = "cuda"))]
            _gpu_features_requested: config.use_gpu_features,
        }
    }

    /// Try to initialize GPU bio-chemistry extractor
    #[cfg(feature = "cuda")]
    fn try_init_gpu_biochemistry(neighbor_cutoff: f32) -> Option<BiochemistryGpu> {
        match CudaContext::new(0) {
            Ok(ctx) => match BiochemistryGpu::new(ctx, 10000, neighbor_cutoff) {
                Ok(gpu) => {
                    log::info!("GPU bio-chemistry extractor initialized successfully");
                    Some(gpu)
                }
                Err(e) => {
                    log::warn!("GPU bio-chemistry init failed (using CPU fallback): {}", e);
                    None
                }
            },
            Err(e) => {
                log::warn!("CUDA context creation failed (using CPU fallback): {:?}", e);
                None
            }
        }
    }

    /// Initialize with reference to initial state
    pub fn initialize(&mut self, initial_atoms: &[Atom]) {
        if !initial_atoms.is_empty() {
            let positions = atoms_to_flat_positions(initial_atoms);
            self.initial_rg = calculate_radius_of_gyration(&positions);

            // Calculate initial target exposure
            let target_indices: Vec<usize> = initial_atoms
                .iter()
                .enumerate()
                .filter(|(_, a)| self.target_residues.contains(&(a.residue_id as usize)))
                .map(|(i, _)| i)
                .collect();

            self.initial_exposure =
                calculate_exposure_fast(&positions, &target_indices, self.config.neighbor_cutoff);

            // Store initial glycan positions for displacement tracking (v3.1.1)
            if !self.glycan_residues.is_empty() {
                self.initial_glycan_positions.clear();
                for (i, atom) in initial_atoms.iter().enumerate() {
                    if self.glycan_residues.contains(&(atom.residue_id as usize)) {
                        self.initial_glycan_positions.push(positions[i * 3]);
                        self.initial_glycan_positions.push(positions[i * 3 + 1]);
                        self.initial_glycan_positions.push(positions[i * 3 + 2]);
                    }
                }
            }

            // Build atomic metadata for bio-chemistry features (v3.1.1)
            if self.config.include_bio_chemistry {
                let metadata = AtomicMetadata::from_atoms(initial_atoms);
                self.initial_positions = positions.clone();

                // Upload metadata to GPU if GPU extractor is available
                #[cfg(feature = "cuda")]
                {
                    let mut gpu_failed = false;
                    if let Some(ref mut gpu) = *self.gpu_biochemistry.borrow_mut() {
                        if let Err(e) =
                            Self::upload_gpu_metadata(gpu, &metadata, &self.target_residue_vec)
                        {
                            log::warn!("GPU metadata upload failed (will use CPU): {}", e);
                            gpu_failed = true;
                        } else if let Err(e) = gpu.upload_initial_positions(&self.initial_positions)
                        {
                            log::warn!("GPU initial positions upload failed (will use CPU): {}", e);
                            gpu_failed = true;
                        }
                    }
                    if gpu_failed {
                        *self.gpu_biochemistry.borrow_mut() = None;
                    }
                }

                self.atomic_metadata = Some(metadata);
            }
        }
    }

    /// Upload atomic metadata to GPU extractor
    #[cfg(feature = "cuda")]
    fn upload_gpu_metadata(
        gpu: &mut BiochemistryGpu,
        metadata: &AtomicMetadata,
        target_residues: &[usize],
    ) -> anyhow::Result<()> {
        let gpu_metadata = GpuAtomicMetadata {
            hydrophobicity: metadata.hydrophobicity.clone(),
            atom_to_residue: metadata.atom_to_residue.iter().map(|&r| r as i32).collect(),
            target_residues: target_residues.iter().map(|&r| r as i32).collect(),
            ca_indices: metadata.ca_indices.iter().map(|&i| i as i32).collect(),
            charges: metadata.atom_charges.clone(),
            charged_indices: metadata
                .atom_charges
                .iter()
                .enumerate()
                .filter(|(_, &c)| c.abs() > 0.25) // Significant charge
                .map(|(i, _)| i as i32)
                .collect(),
        };
        gpu.upload_metadata(&gpu_metadata)
    }

    /// Extract complete feature vector from current state
    pub fn extract(&self, atoms: &[Atom], initial_atoms: Option<&[Atom]>) -> FeatureVector {
        let mut features = FeatureVector::new(self.config.feature_dim());
        let mut idx = 0;

        let positions = atoms_to_flat_positions(atoms);
        let n = atoms.len() as f32;

        // Identify target and core atom indices
        let (target_indices, core_indices) = self.identify_atom_groups(atoms);

        // 1. Global Features (3)
        if self.config.include_global {
            let (size, rg, density) = self.extract_global(&positions, n);
            features.values[idx] = size;
            features.values[idx + 1] = rg;
            features.values[idx + 2] = density;
            idx += 3;
        }

        // 2. Target Neighborhood Features (8)
        if self.config.include_target_neighborhood {
            let neighborhood =
                self.extract_target_neighborhood(&positions, &target_indices, &core_indices);
            for v in neighborhood {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 3. Stability Features (4)
        if self.config.include_stability {
            let initial_pos = initial_atoms.map(|a| atoms_to_flat_positions(a));
            let stability = self.extract_stability(&positions, initial_pos.as_ref(), &core_indices);
            for v in stability {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 4. Family Flags (4)
        if self.config.include_family_flags {
            let flags = self.extract_family_flags(n);
            for v in flags {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 5. Temporal Features (4)
        if self.config.include_temporal {
            let temporal = self.extract_temporal(&positions, &target_indices);
            for v in temporal {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 6. Difficulty Features (4) - NEW in v3.1.1
        if self.config.include_difficulty {
            let difficulty_flags = self.extract_difficulty();
            for v in difficulty_flags {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 7. Glycan Awareness Features (4) - NEW in v3.1.1
        // Teaches agent that glycan displacement reveals cryptic sites
        if self.config.include_glycan_awareness {
            let glycan_features = self.extract_glycan_features(atoms, &positions, &target_indices);
            for v in glycan_features {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 8. Mechanism Features (6) - NEW in v3.1.1
        // One-hot encoding for cryptic site mechanism type
        if self.config.include_mechanism {
            let mechanism_flags = self.extract_mechanism();
            for v in mechanism_flags {
                features.values[idx] = v;
                idx += 1;
            }
        }

        // 9. Bio-Chemistry Features (3) - NEW in v3.1.1
        // Atomic-level chemistry: hydrophobic exposure, hinge detection, electrostatic frustration
        if self.config.include_bio_chemistry {
            let bio_features = self.extract_bio_chemistry(&positions);
            for v in bio_features {
                features.values[idx] = v;
                idx += 1;
            }
        }

        features
    }

    /// Identify target and core atom indices
    fn identify_atom_groups(&self, atoms: &[Atom]) -> (Vec<usize>, Vec<usize>) {
        let mut target_indices = Vec::new();
        let mut core_indices = Vec::new();

        for (i, atom) in atoms.iter().enumerate() {
            let res_id = atom.residue_id as usize;
            if self.target_residues.contains(&res_id) {
                target_indices.push(i);
            } else if self.core_residues.is_empty() || self.core_residues.contains(&res_id) {
                core_indices.push(i);
            }
        }

        (target_indices, core_indices)
    }

    // ========================================================================
    // GLOBAL FEATURES (3 dims)
    // ========================================================================

    fn extract_global(&self, positions: &[f32], n: f32) -> (f32, f32, f32) {
        // Size: normalized atom count (0-1 for typical proteins)
        let size = (n / 10000.0).min(1.0);

        // Radius of Gyration
        let rg = calculate_radius_of_gyration(positions);

        // Density: atoms per unit volume (normalized)
        let density = if rg > 0.0 {
            n / (rg.powi(3) + 1e-6)
        } else {
            0.0
        };
        let density_norm = (density / 100.0).min(1.0);

        (size, rg / 50.0, density_norm) // Normalize Rg to ~[0,1]
    }

    // ========================================================================
    // TARGET NEIGHBORHOOD FEATURES (8 dims)
    // ========================================================================

    fn extract_target_neighborhood(
        &self,
        positions: &[f32],
        target_indices: &[usize],
        _core_indices: &[usize],
    ) -> [f32; 8] {
        let cutoff = self.config.neighbor_cutoff;
        let contact_cutoff = self.config.contact_cutoff;

        if target_indices.is_empty() {
            return [0.0; 8];
        }

        let grid = SpatialGrid::new(positions, cutoff);

        // 1. Mean exposure of target residues (0 = buried, 1 = exposed)
        let mut total_exposure = 0.0;
        let mut min_exposure = f32::MAX;
        let mut max_exposure = 0.0f32;

        // 2. Contact counts
        let mut total_contacts = 0.0;
        let mut min_contacts = f32::MAX;
        let mut max_contacts = 0.0f32;

        // 3. Distance to nearest non-target
        let mut min_dist_to_non_target = f32::MAX;
        let mut mean_dist_to_non_target = 0.0;

        let target_set: HashSet<usize> = target_indices.iter().cloned().collect();

        for &idx in target_indices {
            let (neighbors, non_target_dist) =
                grid.count_neighbors_detailed(positions, idx, &target_set, cutoff);

            // Exposure proxy: fewer neighbors = more exposed
            let exposure = 1.0 / (1.0 + neighbors);
            total_exposure += exposure;
            min_exposure = min_exposure.min(exposure);
            max_exposure = max_exposure.max(exposure);

            // Contact counts within tighter cutoff
            let contacts = grid.count_within_cutoff(positions, idx, contact_cutoff);
            total_contacts += contacts;
            min_contacts = min_contacts.min(contacts);
            max_contacts = max_contacts.max(contacts);

            // Distance to nearest non-target
            if non_target_dist < min_dist_to_non_target {
                min_dist_to_non_target = non_target_dist;
            }
            mean_dist_to_non_target += non_target_dist;
        }

        let n = target_indices.len() as f32;
        let mean_exposure = total_exposure / n;
        let mean_contacts = total_contacts / n;
        mean_dist_to_non_target /= n;

        [
            mean_exposure,                                  // 0: Mean target exposure [0,1]
            min_exposure.min(1.0),                          // 1: Min target exposure (bottleneck)
            max_exposure.min(1.0),                          // 2: Max target exposure
            (mean_contacts / 20.0).min(1.0),                // 3: Mean contact count (normalized)
            (max_contacts / 30.0).min(1.0),                 // 4: Max contacts (crowding)
            (min_dist_to_non_target / 10.0).min(1.0),       // 5: Min dist to non-target (boundary)
            (mean_dist_to_non_target / 15.0).min(1.0),      // 6: Mean dist to non-target
            (target_indices.len() as f32 / 100.0).min(1.0), // 7: Target region size
        ]
    }

    // ========================================================================
    // STABILITY FEATURES (4 dims)
    // ========================================================================

    fn extract_stability(
        &self,
        positions: &[f32],
        initial_positions: Option<&Vec<f32>>,
        core_indices: &[usize],
    ) -> [f32; 4] {
        let clash_dist_sq = (self.config.contact_cutoff * 0.25).powi(2); // ~1.5Å

        // 1. Core RMSD from initial (if available)
        let core_rmsd = if let Some(init_pos) = initial_positions {
            calculate_core_rmsd_fast(positions, init_pos, core_indices)
        } else {
            0.0
        };

        // 2. Clash count (atoms too close together)
        let clash_count = count_clashes(positions, clash_dist_sq);

        // 3. Max displacement from initial
        let max_displacement = if let Some(init_pos) = initial_positions {
            calculate_max_displacement(positions, init_pos)
        } else {
            0.0
        };

        // 4. Local RMSD variance (stability heterogeneity)
        let rmsd_variance = if let Some(init_pos) = initial_positions {
            calculate_displacement_variance(positions, init_pos)
        } else {
            0.0
        };

        [
            (core_rmsd / 5.0).min(1.0),           // Normalized RMSD
            (clash_count as f32 / 10.0).min(1.0), // Clash severity
            (max_displacement / 10.0).min(1.0),   // Max displacement
            (rmsd_variance / 5.0).min(1.0),       // Displacement variance
        ]
    }

    // ========================================================================
    // FAMILY FLAGS (4 dims)
    // ========================================================================

    fn extract_family_flags(&self, n: f32) -> [f32; 4] {
        // 1. Is multimer flag
        let multimer_flag = if self.is_multimer { 1.0 } else { 0.0 };

        // 2. Has glycans flag
        let glycan_flag = if self.has_glycans { 1.0 } else { 0.0 };

        // 3. Size class (small/medium/large/huge)
        let size_class = if n < 1000.0 {
            0.0 // Small
        } else if n < 5000.0 {
            0.33 // Medium
        } else if n < 15000.0 {
            0.66 // Large
        } else {
            1.0 // Huge (spike-like)
        };

        // 4. Target fraction (what % of protein is target)
        let target_fraction = (self.target_residues.len() as f32 / (n / 10.0).max(1.0)).min(1.0);

        [multimer_flag, glycan_flag, size_class, target_fraction]
    }

    // ========================================================================
    // TEMPORAL FEATURES (4 dims)
    // ========================================================================

    fn extract_temporal(&self, positions: &[f32], target_indices: &[usize]) -> [f32; 4] {
        // Current exposure
        let current_exposure =
            calculate_exposure_fast(positions, target_indices, self.config.neighbor_cutoff);

        // Exposure change from initial
        let exposure_delta = current_exposure - self.initial_exposure;

        // Current Rg
        let current_rg = calculate_radius_of_gyration(positions);

        // Rg change from initial
        let rg_delta = current_rg - self.initial_rg;

        [
            current_exposure,                            // 0: Current exposure
            (exposure_delta + 0.5).max(0.0).min(1.0),    // 1: Exposure change (centered)
            (current_rg / 50.0).min(1.0),                // 2: Current Rg (normalized)
            ((rg_delta / 10.0) + 0.5).max(0.0).min(1.0), // 3: Rg change (centered)
        ]
    }

    // ========================================================================
    // DIFFICULTY FEATURES (4 dims) - NEW in v3.1.1
    // ========================================================================

    /// Extract difficulty one-hot encoding
    /// Enables agent to learn different strategies for different difficulty levels:
    /// - Easy: More aggressive exploration
    /// - Medium: Balanced approach
    /// - Hard: More cautious, longer simulations
    /// - Expert: Ultra-careful, prioritize stability
    fn extract_difficulty(&self) -> [f32; 4] {
        match self.difficulty.to_lowercase().as_str() {
            "easy" => [1.0, 0.0, 0.0, 0.0],
            "medium" => [0.0, 1.0, 0.0, 0.0],
            "hard" => [0.0, 0.0, 1.0, 0.0],
            "expert" => [0.0, 0.0, 0.0, 1.0],
            _ => [0.0, 1.0, 0.0, 0.0], // Default to medium
        }
    }

    // ========================================================================
    // GLYCAN AWARENESS FEATURES (4 dims) - NEW in v3.1.1
    // ========================================================================

    /// Extract glycan shield awareness features
    ///
    /// Key insight from 6VXX discovery: glycan shield displacement correlates
    /// with cryptic site exposure. When glycans move away from target residues,
    /// it indicates biologically meaningful opening (not just destabilization).
    ///
    /// Features:
    /// 0. Glycan-target proximity: How close are glycans to target residues?
    /// 1. Glycan displacement: How much have glycans moved from initial positions?
    /// 2. Glycan coverage: What fraction of target residues have nearby glycans?
    /// 3. Displacement-exposure correlation: Does glycan movement correlate with exposure?
    fn extract_glycan_features(
        &self,
        atoms: &[Atom],
        positions: &[f32],
        target_indices: &[usize],
    ) -> [f32; 4] {
        // If no glycan residues defined, return zeros
        if self.glycan_residues.is_empty() {
            return [0.0; 4];
        }

        // Identify glycan atom indices
        let glycan_indices: Vec<usize> = atoms
            .iter()
            .enumerate()
            .filter(|(_, a)| self.glycan_residues.contains(&(a.residue_id as usize)))
            .map(|(i, _)| i)
            .collect();

        if glycan_indices.is_empty() {
            return [0.0; 4];
        }

        // 1. Glycan-target proximity: average min distance from glycan to target
        let mut total_min_dist = 0.0;
        for &gi in &glycan_indices {
            let gx = positions[gi * 3];
            let gy = positions[gi * 3 + 1];
            let gz = positions[gi * 3 + 2];

            let mut min_dist_sq = f32::MAX;
            for &ti in target_indices {
                let tx = positions[ti * 3];
                let ty = positions[ti * 3 + 1];
                let tz = positions[ti * 3 + 2];
                let dist_sq = (gx - tx).powi(2) + (gy - ty).powi(2) + (gz - tz).powi(2);
                min_dist_sq = min_dist_sq.min(dist_sq);
            }
            total_min_dist += min_dist_sq.sqrt();
        }
        let avg_proximity = total_min_dist / glycan_indices.len() as f32;
        // Invert and normalize: closer = higher value
        let proximity_feature = 1.0 / (1.0 + avg_proximity / 10.0);

        // 2. Glycan displacement from initial positions
        let mut total_displacement = 0.0;
        if !self.initial_glycan_positions.is_empty() {
            let mut glycan_atom_idx = 0;
            for (i, atom) in atoms.iter().enumerate() {
                if self.glycan_residues.contains(&(atom.residue_id as usize)) {
                    let init_idx = glycan_atom_idx * 3;
                    if init_idx + 2 < self.initial_glycan_positions.len() {
                        let dx = positions[i * 3] - self.initial_glycan_positions[init_idx];
                        let dy = positions[i * 3 + 1] - self.initial_glycan_positions[init_idx + 1];
                        let dz = positions[i * 3 + 2] - self.initial_glycan_positions[init_idx + 2];
                        total_displacement += (dx * dx + dy * dy + dz * dz).sqrt();
                    }
                    glycan_atom_idx += 1;
                }
            }
        }
        let avg_displacement = if glycan_indices.is_empty() {
            0.0
        } else {
            total_displacement / glycan_indices.len() as f32
        };
        // Normalize: 0-10Å range
        let displacement_feature = (avg_displacement / 10.0).min(1.0);

        // 3. Glycan coverage: fraction of target residues with glycan within cutoff
        let cutoff_sq = self.config.neighbor_cutoff.powi(2);
        let mut covered_targets = 0;
        for &ti in target_indices {
            let tx = positions[ti * 3];
            let ty = positions[ti * 3 + 1];
            let tz = positions[ti * 3 + 2];

            for &gi in &glycan_indices {
                let gx = positions[gi * 3];
                let gy = positions[gi * 3 + 1];
                let gz = positions[gi * 3 + 2];
                let dist_sq = (gx - tx).powi(2) + (gy - ty).powi(2) + (gz - tz).powi(2);
                if dist_sq < cutoff_sq {
                    covered_targets += 1;
                    break; // Only count once per target
                }
            }
        }
        let coverage_feature = if target_indices.is_empty() {
            0.0
        } else {
            covered_targets as f32 / target_indices.len() as f32
        };

        // 4. Displacement-exposure correlation
        // Higher value when glycan displacement correlates with target exposure
        let current_exposure =
            calculate_exposure_fast(positions, target_indices, self.config.neighbor_cutoff);
        let exposure_gain = current_exposure - self.initial_exposure;

        // Correlation proxy: if both displacement and exposure increase together
        let correlation_feature = if avg_displacement > 0.0 && exposure_gain > 0.0 {
            // Both positive = good correlation
            ((displacement_feature * 0.5 + exposure_gain) / 1.5).min(1.0)
        } else if avg_displacement > 0.0 && exposure_gain <= 0.0 {
            // Displacement without exposure = glycan moved but site not opening
            0.2
        } else {
            0.0
        };

        [
            proximity_feature,      // 0: Glycan-target proximity [0,1]
            displacement_feature,   // 1: Glycan displacement [0,1]
            1.0 - coverage_feature, // 2: Exposure potential (inverted coverage) [0,1]
            correlation_feature,    // 3: Displacement-exposure correlation [0,1]
        ]
    }

    // ========================================================================
    // MECHANISM FEATURES (6 dims) - NEW in v3.1.1
    // ========================================================================

    /// Extract mechanism one-hot encoding
    /// Enables agent to learn different strategies for different cryptic site mechanisms:
    /// - glycan_shield: Look for glycan displacement patterns (spike proteins)
    /// - allosteric: Distant site coupling, look for hinge motions
    /// - induced_fit: Ligand-induced conformational changes
    /// - flap: Protease flap dynamics (HIV protease)
    /// - loop: Omega-loop movements (beta-lactamases)
    /// - unknown: General strategy
    fn extract_mechanism(&self) -> [f32; 6] {
        match self.mechanism.to_lowercase().as_str() {
            "glycan_shield" => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "allosteric" => [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "induced_fit" => [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "flap" | "flap_cryptic" => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "loop" | "omega_loop" => [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            _ => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], // Unknown
        }
    }

    // ========================================================================
    // BIO-CHEMISTRY FEATURES (3 dims) - NEW in v3.1.1
    // ========================================================================

    /// Extract atomic-level bio-chemistry features
    ///
    /// These features give the SNN "chemical intelligence" beyond geometry:
    ///
    /// 1. **Hydrophobic Exposure Delta** (The "Grease Signal")
    ///    - Drug binding sites are usually hydrophobic
    ///    - High value = exposed "greasy" residues = prime drug target
    ///    - Weighted by Kyte-Doolittle hydrophobicity scale
    ///
    /// 2. **Local Displacement Anisotropy** (The "Hinge Signal")
    ///    - Detects where backbone moves relative to neighbors
    ///    - High value = mechanical hinge = cryptic pocket entrance
    ///    - Computed from Cα atom trajectories
    ///
    /// 3. **Electrostatic Frustration** (The "Spring Signal")
    ///    - Like-charges forced close together want to separate
    ///    - High value = thermodynamic stress = wants to open
    ///    - Computed from partial atomic charges
    fn extract_bio_chemistry(&self, positions: &[f32]) -> [f32; 3] {
        // Check if atomic metadata is available
        let metadata = match &self.atomic_metadata {
            Some(m) => m,
            None => return [0.0, 0.0, 0.0], // No metadata, return zeros
        };

        // Check if initial positions are available
        if self.initial_positions.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        // Try GPU-accelerated path first
        #[cfg(feature = "cuda")]
        {
            let mut gpu_ref = self.gpu_biochemistry.borrow_mut();
            if let Some(ref mut gpu) = *gpu_ref {
                match gpu.compute(positions) {
                    Ok(result) => {
                        // GPU returns already normalized [0,1] values
                        return result;
                    }
                    Err(e) => {
                        log::warn!(
                            "GPU bio-chemistry compute failed, falling back to CPU: {}",
                            e
                        );
                        // Disable GPU for subsequent calls
                        *gpu_ref = None;
                    }
                }
            }
        }

        // CPU fallback path
        self.extract_bio_chemistry_cpu(positions, metadata)
    }

    /// CPU implementation of bio-chemistry features
    fn extract_bio_chemistry_cpu(&self, positions: &[f32], metadata: &AtomicMetadata) -> [f32; 3] {
        // 1. Hydrophobic Exposure Delta
        let hydrophobic_delta = calculate_hydrophobic_exposure(
            positions,
            &self.initial_positions,
            metadata,
            &self.target_residue_vec,
            self.config.neighbor_cutoff,
        );
        // Normalize to [0, 1] - typical range is [-1, 1] for delta
        let hydrophobic_feature = ((hydrophobic_delta / 10.0) + 0.5).max(0.0).min(1.0);

        // 2. Local Displacement Anisotropy (Hinge Detection)
        let (max_anisotropy, _mean_anisotropy) =
            calculate_anisotropy(positions, &self.initial_positions, metadata);
        // Normalize: typical hinge motion is 2-5Å differential
        let anisotropy_feature = (max_anisotropy / 5.0).min(1.0);

        // 3. Electrostatic Frustration
        let frustration =
            calculate_electrostatic_frustration(positions, metadata, self.config.neighbor_cutoff);
        // Normalize: frustration is usually 0-10 energy units
        let frustration_feature = (frustration / 10.0).min(1.0);

        [
            hydrophobic_feature, // 0: "Grease" - exposed hydrophobic surface
            anisotropy_feature,  // 1: "Hinge" - local displacement differential
            frustration_feature, // 2: "Spring" - electrostatic stress
        ]
    }
}

// ============================================================================
// SPATIAL GRID (O(N) Neighbor Search)
// ============================================================================

struct SpatialGrid {
    cell_size: f32,
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    fn new(positions: &[f32], cell_size: f32) -> Self {
        let mut cells = HashMap::new();
        let n = positions.len() / 3;

        for i in 0..n {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            let key = (
                (x / cell_size).floor() as i32,
                (y / cell_size).floor() as i32,
                (z / cell_size).floor() as i32,
            );

            cells.entry(key).or_insert_with(Vec::new).push(i);
        }

        Self { cell_size, cells }
    }

    fn count_neighbors_detailed(
        &self,
        positions: &[f32],
        atom_idx: usize,
        target_set: &HashSet<usize>,
        cutoff: f32,
    ) -> (f32, f32) {
        let cutoff_sq = cutoff * cutoff;
        let x = positions[atom_idx * 3];
        let y = positions[atom_idx * 3 + 1];
        let z = positions[atom_idx * 3 + 2];

        let cx = (x / self.cell_size).floor() as i32;
        let cy = (y / self.cell_size).floor() as i32;
        let cz = (z / self.cell_size).floor() as i32;

        let mut count = 0.0;
        let mut min_non_target_dist = f32::MAX;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if atom_idx == j {
                                continue;
                            }

                            let x2 = positions[j * 3];
                            let y2 = positions[j * 3 + 1];
                            let z2 = positions[j * 3 + 2];

                            let dist_sq = (x - x2).powi(2) + (y - y2).powi(2) + (z - z2).powi(2);

                            if dist_sq < cutoff_sq {
                                count += 1.0;
                            }

                            // Track distance to non-target atoms
                            if !target_set.contains(&j) && dist_sq < min_non_target_dist {
                                min_non_target_dist = dist_sq;
                            }
                        }
                    }
                }
            }
        }

        (count, min_non_target_dist.sqrt())
    }

    fn count_within_cutoff(&self, positions: &[f32], atom_idx: usize, cutoff: f32) -> f32 {
        let cutoff_sq = cutoff * cutoff;
        let x = positions[atom_idx * 3];
        let y = positions[atom_idx * 3 + 1];
        let z = positions[atom_idx * 3 + 2];

        let cx = (x / self.cell_size).floor() as i32;
        let cy = (y / self.cell_size).floor() as i32;
        let cz = (z / self.cell_size).floor() as i32;

        let mut count = 0.0;

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(indices) = self.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if atom_idx == j {
                                continue;
                            }

                            let x2 = positions[j * 3];
                            let y2 = positions[j * 3 + 1];
                            let z2 = positions[j * 3 + 2];

                            let dist_sq = (x - x2).powi(2) + (y - y2).powi(2) + (z - z2).powi(2);
                            if dist_sq < cutoff_sq {
                                count += 1.0;
                            }
                        }
                    }
                }
            }
        }

        count
    }
}

// ============================================================================
// HELPER FUNCTIONS (VRAM-Safe, O(N))
// ============================================================================

/// Convert atoms to flat position array [x0,y0,z0,x1,y1,z1,...]
fn atoms_to_flat_positions(atoms: &[Atom]) -> Vec<f32> {
    let mut positions = Vec::with_capacity(atoms.len() * 3);
    for atom in atoms {
        positions.push(atom.coords[0]);
        positions.push(atom.coords[1]);
        positions.push(atom.coords[2]);
    }
    positions
}

/// Calculate radius of gyration
fn calculate_radius_of_gyration(positions: &[f32]) -> f32 {
    let n = positions.len() / 3;
    if n == 0 {
        return 0.0;
    }

    // Calculate center of mass
    let mut com = [0.0f32; 3];
    for i in 0..n {
        com[0] += positions[i * 3];
        com[1] += positions[i * 3 + 1];
        com[2] += positions[i * 3 + 2];
    }
    let n_f = n as f32;
    com[0] /= n_f;
    com[1] /= n_f;
    com[2] /= n_f;

    // Calculate Rg
    let mut rg_sq = 0.0;
    for i in 0..n {
        let dx = positions[i * 3] - com[0];
        let dy = positions[i * 3 + 1] - com[1];
        let dz = positions[i * 3 + 2] - com[2];
        rg_sq += dx * dx + dy * dy + dz * dz;
    }

    (rg_sq / n_f).sqrt()
}

/// Fast exposure calculation using spatial grid
fn calculate_exposure_fast(positions: &[f32], target_indices: &[usize], cutoff: f32) -> f32 {
    if target_indices.is_empty() {
        return 0.0;
    }

    let grid = SpatialGrid::new(positions, cutoff);
    let mut total_exposure = 0.0;

    for &idx in target_indices {
        let neighbors = grid.count_within_cutoff(positions, idx, cutoff);
        total_exposure += 1.0 / (1.0 + neighbors);
    }

    total_exposure / target_indices.len() as f32
}

/// Fast core RMSD calculation
fn calculate_core_rmsd_fast(positions: &[f32], initial: &[f32], core_indices: &[usize]) -> f32 {
    if core_indices.is_empty() {
        return 0.0;
    }

    let mut sum_sq = 0.0;
    for &idx in core_indices {
        let base = idx * 3;
        if base + 2 < positions.len() && base + 2 < initial.len() {
            let dx = positions[base] - initial[base];
            let dy = positions[base + 1] - initial[base + 1];
            let dz = positions[base + 2] - initial[base + 2];
            sum_sq += dx * dx + dy * dy + dz * dz;
        }
    }

    (sum_sq / core_indices.len() as f32).sqrt()
}

/// Count atomic clashes (pairs closer than threshold)
fn count_clashes(positions: &[f32], clash_dist_sq: f32) -> usize {
    let n = positions.len() / 3;
    let mut clashes = 0;

    // Use grid for efficiency
    let grid = SpatialGrid::new(positions, clash_dist_sq.sqrt() * 2.0);

    for i in 0..n {
        let x = positions[i * 3];
        let y = positions[i * 3 + 1];
        let z = positions[i * 3 + 2];

        let cx = (x / (clash_dist_sq.sqrt() * 2.0)).floor() as i32;
        let cy = (y / (clash_dist_sq.sqrt() * 2.0)).floor() as i32;
        let cz = (z / (clash_dist_sq.sqrt() * 2.0)).floor() as i32;

        for dx in 0..=1 {
            for dy in 0..=1 {
                for dz in 0..=1 {
                    if let Some(indices) = grid.cells.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &j in indices {
                            if j <= i {
                                continue;
                            }

                            let x2 = positions[j * 3];
                            let y2 = positions[j * 3 + 1];
                            let z2 = positions[j * 3 + 2];

                            let dist_sq = (x - x2).powi(2) + (y - y2).powi(2) + (z - z2).powi(2);
                            if dist_sq < clash_dist_sq {
                                clashes += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    clashes
}

/// Calculate maximum displacement from initial
fn calculate_max_displacement(positions: &[f32], initial: &[f32]) -> f32 {
    let n = positions.len().min(initial.len()) / 3;
    let mut max_disp = 0.0f32;

    for i in 0..n {
        let base = i * 3;
        let dx = positions[base] - initial[base];
        let dy = positions[base + 1] - initial[base + 1];
        let dz = positions[base + 2] - initial[base + 2];
        let disp = (dx * dx + dy * dy + dz * dz).sqrt();
        max_disp = max_disp.max(disp);
    }

    max_disp
}

/// Calculate variance in displacement (heterogeneity)
fn calculate_displacement_variance(positions: &[f32], initial: &[f32]) -> f32 {
    let n = positions.len().min(initial.len()) / 3;
    if n == 0 {
        return 0.0;
    }

    let mut displacements = Vec::with_capacity(n);
    let mut sum = 0.0;

    for i in 0..n {
        let base = i * 3;
        let dx = positions[base] - initial[base];
        let dy = positions[base + 1] - initial[base + 1];
        let dz = positions[base + 2] - initial[base + 2];
        let disp = (dx * dx + dy * dy + dz * dz).sqrt();
        displacements.push(disp);
        sum += disp;
    }

    let mean = sum / n as f32;
    let variance: f32 = displacements
        .iter()
        .map(|&d| (d - mean).powi(2))
        .sum::<f32>()
        / n as f32;

    variance.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_dim() {
        let config = FeatureConfig::default();
        assert_eq!(config.feature_dim(), 23);
    }

    #[test]
    fn test_radius_of_gyration() {
        // Simple cube of atoms
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];
        let rg = calculate_radius_of_gyration(&positions);
        assert!(rg > 0.0 && rg < 2.0);
    }
}
