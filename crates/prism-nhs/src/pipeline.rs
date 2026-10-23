//! Unified NHS Pipeline
//!
//! Orchestrates the complete Neuromorphic Holographic Stream pipeline:
//!
//! 1. **Atom Classification** - Identify hydrophobic/polar atoms
//! 2. **Exclusion Field** - Compute 3D water exclusion map
//! 3. **Water Inference** - Infer water density from exclusion
//! 4. **Neuromorphic Detection** - Spike on dewetting events
//! 5. **Avalanche Detection** - Cluster spikes into pocket events
//! 6. **UV Bias Perturbation** - Targeted aromatic excitation (optional)
//!
//! # Performance Target
//!
//! - <2ms per frame on RTX 3060
//! - 30,000× faster than explicit solvent
//! - 85-90% accuracy vs explicit solvent ground truth

use crate::avalanche::{AvalancheDetector, CrypticSiteEvent};
use crate::config::NhsConfig;
use crate::exclusion::{ClassifiedAtom, ExclusionComputer, ExclusionGrid};
use crate::neuromorphic::DewettingNetwork;
use crate::uv_bias::{PerturbationResult, UvBiasEngine};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::sync::mpsc;

// =============================================================================
// PIPELINE
// =============================================================================

/// NHS Pipeline - Unified cryptic site detection engine
pub struct NhsPipeline {
    config: NhsConfig,

    // Components
    exclusion_computer: ExclusionComputer,
    network: Option<DewettingNetwork>,
    avalanche_detector: AvalancheDetector,
    uv_bias_engine: Option<UvBiasEngine>,

    // State
    grid: Option<ExclusionGrid>,
    classified_atoms: Vec<ClassifiedAtom>,

    // Streaming output
    event_sender: Option<mpsc::Sender<CrypticSiteEvent>>,

    // Metrics
    frames_processed: u64,
    total_spikes: u64,
    sites_detected: u64,
    total_processing_time_ms: f64,
}

impl NhsPipeline {
    /// Create new NHS pipeline
    pub fn new(config: NhsConfig) -> Self {
        let exclusion_computer = ExclusionComputer::new(config.clone());
        let avalanche_detector = AvalancheDetector::new(config.clone());

        let uv_bias_engine = if config.uv_bias_enabled {
            Some(UvBiasEngine::new(config.uv_bias.clone()))
        } else {
            None
        };

        Self {
            config,
            exclusion_computer,
            network: None,
            avalanche_detector,
            uv_bias_engine,
            grid: None,
            classified_atoms: Vec::new(),
            event_sender: None,
            frames_processed: 0,
            total_spikes: 0,
            sites_detected: 0,
            total_processing_time_ms: 0.0,
        }
    }

    /// Set streaming output channel
    pub fn set_output_channel(&mut self, sender: mpsc::Sender<CrypticSiteEvent>) {
        self.event_sender = Some(sender);
    }

    /// Initialize pipeline for a protein structure
    ///
    /// Must be called before processing frames.
    ///
    /// # Arguments
    /// * `positions` - Flat array [x0, y0, z0, x1, y1, z1, ...]
    /// * `elements` - Atomic number for each atom
    /// * `charges` - Partial charges for each atom
    /// * `residue_names` - Residue name indexed by residue index
    /// * `atom_names` - IUPAC atom names for each atom (e.g., "CA", "CG", "CD1")
    /// * `atom_residues` - Residue index for each atom
    pub fn initialize(
        &mut self,
        positions: &[f32],
        elements: &[u8],
        charges: &[f32],
        residue_names: &[String],
        atom_names: &[String],
        atom_residues: &[usize],
    ) -> Result<()> {
        let n_atoms = positions.len() / 3;
        log::info!("NHS Pipeline: Initializing for {} atoms", n_atoms);

        // 1. Classify atoms
        self.classified_atoms = self.exclusion_computer.classify_atoms(
            positions,
            elements,
            charges,
            residue_names,
            atom_residues,
        );

        // Store atom names for UV bias ring detection
        let atom_names_owned = atom_names.to_vec();

        // 2. Create exclusion grid
        let grid = ExclusionGrid::from_atoms(&self.classified_atoms, &self.config);

        // 3. Compute initial exclusion field
        let mut grid_mut = grid;
        self.exclusion_computer
            .compute(&self.classified_atoms, &mut grid_mut)
            .context("Failed to compute initial exclusion field")?;

        // 4. Create neuromorphic network
        let network = DewettingNetwork::from_grid(&grid_mut, self.config.clone());

        // 5. Initialize UV bias engine if enabled
        if let Some(uv_engine) = &mut self.uv_bias_engine {
            uv_engine
                .initialize_targets(residue_names, &atom_names_owned, atom_residues, positions)
                .context("Failed to initialize UV bias targets")?;
        }

        self.grid = Some(grid_mut);
        self.network = Some(network);

        // Reset state
        self.avalanche_detector.reset();
        self.frames_processed = 0;
        self.total_spikes = 0;
        self.sites_detected = 0;
        self.total_processing_time_ms = 0.0;

        log::info!("NHS Pipeline: Initialization complete");
        Ok(())
    }

    /// Process a single frame
    ///
    /// Returns detected cryptic site events and optional UV perturbation to apply.
    pub fn process_frame(
        &mut self,
        positions: &[f32],
    ) -> Result<(Vec<CrypticSiteEvent>, Option<PerturbationResult>)> {
        let start = Instant::now();

        // Check initialization first
        if self.grid.is_none() || self.network.is_none() {
            anyhow::bail!("Pipeline not initialized");
        }

        // 1. Update atom positions
        self.update_atom_positions(positions);

        // 2. Recompute exclusion field
        self.exclusion_computer
            .compute(&self.classified_atoms, self.grid.as_mut().unwrap())
            .context("Failed to compute exclusion field")?;

        // 3. Run neuromorphic network - get water density first
        let water_density = self.grid.as_ref().unwrap().water_density.clone();
        let spikes = self.network.as_mut().unwrap().step(&water_density);
        self.total_spikes += spikes.len() as u64;

        // 4. Get spike data for avalanche detection
        let spike_positions = self
            .network
            .as_ref()
            .unwrap()
            .get_spike_positions(self.grid.as_ref().unwrap());
        let spike_voxels = self.network.as_ref().unwrap().get_spike_voxels();
        let spike_residues = self.network.as_ref().unwrap().get_spike_residues();

        // 5. Detect avalanches
        let events = self.avalanche_detector.process_spikes_data(
            &spike_positions,
            &spike_voxels,
            &spike_residues,
        );
        self.sites_detected += events.len() as u64;

        // 6. UV bias perturbation (if enabled)
        let perturbation = if let Some(uv_engine) = &mut self.uv_bias_engine {
            let grid = self.grid.as_ref().unwrap();

            // Update target selection based on pocket probability
            uv_engine.update_target_selection(&grid.pocket_probability, grid.spacing);

            // Record spikes for causal correlation
            uv_engine.record_spikes(
                spike_voxels.clone(),
                spike_residues.iter().map(|&r| r as usize).collect(),
            );

            // Get perturbation (may be None if not in burst phase)
            uv_engine.step(positions)
        } else {
            None
        };

        // 7. Stream events if channel set
        if let Some(sender) = &self.event_sender {
            for event in &events {
                let _ = sender.try_send(event.clone());
            }
        }

        self.frames_processed += 1;
        self.total_processing_time_ms += start.elapsed().as_secs_f64() * 1000.0;

        Ok((events, perturbation))
    }

    /// Update classified atom positions
    fn update_atom_positions(&mut self, positions: &[f32]) {
        for (i, atom) in self.classified_atoms.iter_mut().enumerate() {
            let base = i * 3;
            if base + 2 < positions.len() {
                atom.x = positions[base];
                atom.y = positions[base + 1];
                atom.z = positions[base + 2];
            }
        }
    }

    /// Get pipeline statistics
    pub fn stats(&self) -> NhsStats {
        let avg_time = if self.frames_processed > 0 {
            self.total_processing_time_ms / self.frames_processed as f64
        } else {
            0.0
        };

        let avg_spikes = if self.frames_processed > 0 {
            self.total_spikes as f32 / self.frames_processed as f32
        } else {
            0.0
        };

        let uv_stats = self.uv_bias_engine.as_ref().map(|e| e.stats());

        NhsStats {
            frames_processed: self.frames_processed,
            total_spikes: self.total_spikes,
            sites_detected: self.sites_detected,
            avg_spikes_per_frame: avg_spikes,
            avg_frame_time_ms: avg_time,
            active_clusters: self.avalanche_detector.num_active_clusters(),
            uv_bias_stats: uv_stats,
        }
    }

    /// Get causal links from UV bias engine
    pub fn get_causal_links(&self) -> Vec<crate::uv_bias::CausalCorrelation> {
        self.uv_bias_engine
            .as_ref()
            .map(|e| e.get_causal_links().into_iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Reset for new trajectory
    pub fn reset(&mut self) {
        if let Some(network) = &mut self.network {
            network.reset();
        }
        if let Some(uv_engine) = &mut self.uv_bias_engine {
            uv_engine.reset();
        }
        self.avalanche_detector.reset();
        self.frames_processed = 0;
        self.total_spikes = 0;
        self.sites_detected = 0;
        self.total_processing_time_ms = 0.0;
    }

    /// Check if pipeline is initialized
    pub fn is_initialized(&self) -> bool {
        self.grid.is_some() && self.network.is_some()
    }

    /// Get reference to exclusion grid
    pub fn grid(&self) -> Option<&ExclusionGrid> {
        self.grid.as_ref()
    }
}

// =============================================================================
// STATISTICS
// =============================================================================

/// Pipeline statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NhsStats {
    pub frames_processed: u64,
    pub total_spikes: u64,
    pub sites_detected: u64,
    pub avg_spikes_per_frame: f32,
    pub avg_frame_time_ms: f64,
    pub active_clusters: usize,
    pub uv_bias_stats: Option<crate::uv_bias::UvBiasStats>,
}

impl std::fmt::Display for NhsStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== NHS Pipeline Statistics ===")?;
        writeln!(f, "Frames processed:     {}", self.frames_processed)?;
        writeln!(f, "Total spikes:         {}", self.total_spikes)?;
        writeln!(f, "Sites detected:       {}", self.sites_detected)?;
        writeln!(f, "Avg spikes/frame:     {:.2}", self.avg_spikes_per_frame)?;
        writeln!(f, "Avg frame time:       {:.2} ms", self.avg_frame_time_ms)?;
        writeln!(f, "Active clusters:      {}", self.active_clusters)?;

        if let Some(uv) = &self.uv_bias_stats {
            writeln!(f, "\n--- UV Bias ---")?;
            writeln!(f, "Targets:              {}", uv.total_targets)?;
            writeln!(f, "Active targets:       {}", uv.active_targets)?;
            writeln!(f, "Bursts applied:       {}", uv.bursts_applied)?;
            writeln!(f, "Causal links:         {}", uv.causal_links)?;
        }

        Ok(())
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let config = NhsConfig::default();
        let pipeline = NhsPipeline::new(config);

        assert!(!pipeline.is_initialized());
        assert_eq!(pipeline.frames_processed, 0);
    }

    #[test]
    fn test_stats_display() {
        let stats = NhsStats {
            frames_processed: 100,
            total_spikes: 500,
            sites_detected: 3,
            avg_spikes_per_frame: 5.0,
            avg_frame_time_ms: 1.5,
            active_clusters: 2,
            uv_bias_stats: None,
        };

        let display = format!("{}", stats);
        assert!(display.contains("100"));
        assert!(display.contains("500"));
    }
}
