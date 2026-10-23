//! Pocket Persistence Tracking for Cryptic Site Validation
//!
//! A pocket that opens for 1 femtosecond isn't druggable. Drugs need time to bind.
//! This module tracks whether cryptic pockets remain open long enough for chemistry.
//!
//! ## Key Metrics
//! - **Persistence Ratio**: Fraction of frames where pocket is open (target: >50%)
//! - **Mean Residence Time**: Average duration pocket stays open
//! - **Opening Events**: Number of open→close transitions (fewer = more stable)
//! - **Max Continuous Open**: Longest stretch of continuous exposure

use prism_io::sovereign_types::Atom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for persistence tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Number of samples to collect during simulation
    pub num_samples: usize,
    /// Displacement threshold (Å) to consider a residue "exposed"
    pub exposure_threshold: f32,
    /// Minimum fraction of target residues that must be exposed for pocket to be "open"
    pub pocket_open_fraction: f32,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            num_samples: 20,           // Sample 20 times during simulation
            exposure_threshold: 2.0,   // 2Å displacement = exposed
            pocket_open_fraction: 0.3, // 30% of target residues exposed = pocket open
        }
    }
}

/// Per-residue exposure tracking across time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueExposureHistory {
    pub residue_id: u32,
    /// Displacement at each sample point (Å)
    pub displacement_timeline: Vec<f32>,
    /// Binary: was residue exposed at each sample?
    pub exposed_timeline: Vec<bool>,
    /// Fraction of time this residue was exposed
    pub exposure_fraction: f32,
    /// Maximum displacement observed
    pub max_displacement: f32,
    /// Mean displacement across all samples
    pub mean_displacement: f32,
}

/// Pocket-level persistence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketPersistenceMetrics {
    /// Fraction of frames where pocket was considered "open"
    pub persistence_ratio: f32,
    /// Number of distinct opening events (open→close transitions)
    pub opening_events: usize,
    /// Longest continuous stretch of open frames
    pub max_continuous_open: usize,
    /// Mean duration of open periods (in sample intervals)
    pub mean_open_duration: f32,
    /// Frame indices where pocket was open
    pub open_frames: Vec<usize>,
    /// Per-residue exposure histories for target residues
    pub residue_histories: Vec<ResidueExposureHistory>,
    /// Overall assessment
    pub assessment: PersistenceAssessment,
}

/// Qualitative assessment of persistence
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PersistenceAssessment {
    /// Pocket stays open >70% of time - excellent druggability
    Stable,
    /// Pocket open 50-70% - good candidate
    Moderate,
    /// Pocket open 30-50% - marginal, may need optimization
    Transient,
    /// Pocket open <30% - likely not druggable
    Unstable,
}

impl PersistenceAssessment {
    pub fn from_ratio(ratio: f32) -> Self {
        if ratio >= 0.7 {
            Self::Stable
        } else if ratio >= 0.5 {
            Self::Moderate
        } else if ratio >= 0.3 {
            Self::Transient
        } else {
            Self::Unstable
        }
    }

    pub fn is_druggable(&self) -> bool {
        matches!(self, Self::Stable | Self::Moderate)
    }
}

/// Tracks pocket persistence across simulation trajectory
pub struct PersistenceTracker {
    config: PersistenceConfig,
    /// Initial atom positions (reference state)
    initial_atoms: Vec<Atom>,
    /// Target residue IDs to track
    target_residues: Vec<usize>,
    /// Core residue IDs (for stability check)
    core_residues: Vec<usize>,
    /// Snapshots: (sample_index, atoms_at_that_time)
    snapshots: Vec<(usize, Vec<Atom>)>,
    /// Steps between samples
    sample_interval: u64,
    /// Current sample index
    current_sample: usize,
}

impl PersistenceTracker {
    /// Create a new persistence tracker
    pub fn new(
        config: PersistenceConfig,
        initial_atoms: Vec<Atom>,
        target_residues: Vec<usize>,
        core_residues: Vec<usize>,
        total_steps: u64,
    ) -> Self {
        let sample_interval = total_steps / config.num_samples as u64;

        Self {
            config,
            initial_atoms,
            target_residues,
            core_residues,
            snapshots: Vec::with_capacity(20),
            sample_interval,
            current_sample: 0,
        }
    }

    /// Record a snapshot of current atom positions
    pub fn record_snapshot(&mut self, sample_index: usize, atoms: Vec<Atom>) {
        self.snapshots.push((sample_index, atoms));
        self.current_sample = sample_index;
    }

    /// Get the step interval between samples
    pub fn get_sample_interval(&self) -> u64 {
        self.sample_interval
    }

    /// Get number of samples to collect
    pub fn get_num_samples(&self) -> usize {
        self.config.num_samples
    }

    /// Calculate per-residue displacement at a given snapshot
    fn calculate_residue_displacements(&self, atoms: &[Atom]) -> HashMap<u32, f32> {
        let mut max_displacements: HashMap<u32, f32> = HashMap::new();

        for (initial, current) in self.initial_atoms.iter().zip(atoms.iter()) {
            let dx = current.coords[0] - initial.coords[0];
            let dy = current.coords[1] - initial.coords[1];
            let dz = current.coords[2] - initial.coords[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            let res_id = u32::from(initial.residue_id);
            let entry = max_displacements.entry(res_id).or_insert(0.0);
            if dist > *entry {
                *entry = dist;
            }
        }

        max_displacements
    }

    /// Determine if pocket is "open" at a given snapshot
    fn is_pocket_open(&self, displacements: &HashMap<u32, f32>) -> bool {
        if self.target_residues.is_empty() {
            return false;
        }

        let exposed_count = self
            .target_residues
            .iter()
            .filter(|&&res_id| {
                displacements
                    .get(&(res_id as u32))
                    .map(|&d| d >= self.config.exposure_threshold)
                    .unwrap_or(false)
            })
            .count();

        let exposure_ratio = exposed_count as f32 / self.target_residues.len() as f32;
        exposure_ratio >= self.config.pocket_open_fraction
    }

    /// Analyze all snapshots and compute persistence metrics
    pub fn analyze(&self) -> PocketPersistenceMetrics {
        if self.snapshots.is_empty() {
            return PocketPersistenceMetrics {
                persistence_ratio: 0.0,
                opening_events: 0,
                max_continuous_open: 0,
                mean_open_duration: 0.0,
                open_frames: Vec::new(),
                residue_histories: Vec::new(),
                assessment: PersistenceAssessment::Unstable,
            };
        }

        // Build per-residue timelines
        let mut residue_timelines: HashMap<u32, Vec<f32>> = HashMap::new();
        for res_id in &self.target_residues {
            residue_timelines.insert(*res_id as u32, Vec::with_capacity(self.snapshots.len()));
        }

        // Track pocket open/close state per frame
        let mut pocket_open_timeline: Vec<bool> = Vec::with_capacity(self.snapshots.len());
        let mut open_frames: Vec<usize> = Vec::new();

        // Process each snapshot
        for (sample_idx, atoms) in &self.snapshots {
            let displacements = self.calculate_residue_displacements(atoms);

            // Record per-residue displacements
            for res_id in &self.target_residues {
                let disp = displacements.get(&(*res_id as u32)).copied().unwrap_or(0.0);
                if let Some(timeline) = residue_timelines.get_mut(&(*res_id as u32)) {
                    timeline.push(disp);
                }
            }

            // Check if pocket is open
            let is_open = self.is_pocket_open(&displacements);
            pocket_open_timeline.push(is_open);
            if is_open {
                open_frames.push(*sample_idx);
            }
        }

        // Calculate persistence ratio
        let open_count = pocket_open_timeline.iter().filter(|&&x| x).count();
        let persistence_ratio = open_count as f32 / pocket_open_timeline.len() as f32;

        // Count opening events and find max continuous open
        let (opening_events, max_continuous_open, open_durations) =
            self.analyze_transitions(&pocket_open_timeline);

        // Mean open duration
        let mean_open_duration = if !open_durations.is_empty() {
            open_durations.iter().sum::<usize>() as f32 / open_durations.len() as f32
        } else {
            0.0
        };

        // Build per-residue histories
        let residue_histories: Vec<ResidueExposureHistory> = self
            .target_residues
            .iter()
            .filter_map(|&res_id| {
                let timeline = residue_timelines.get(&(res_id as u32))?;
                if timeline.is_empty() {
                    return None;
                }

                let exposed_timeline: Vec<bool> = timeline
                    .iter()
                    .map(|&d| d >= self.config.exposure_threshold)
                    .collect();

                let exposure_fraction = exposed_timeline.iter().filter(|&&x| x).count() as f32
                    / exposed_timeline.len() as f32;

                let max_displacement = timeline.iter().cloned().fold(0.0f32, f32::max);

                let mean_displacement = timeline.iter().sum::<f32>() / timeline.len() as f32;

                Some(ResidueExposureHistory {
                    residue_id: res_id as u32,
                    displacement_timeline: timeline.clone(),
                    exposed_timeline,
                    exposure_fraction,
                    max_displacement,
                    mean_displacement,
                })
            })
            .collect();

        let assessment = PersistenceAssessment::from_ratio(persistence_ratio);

        PocketPersistenceMetrics {
            persistence_ratio,
            opening_events,
            max_continuous_open,
            mean_open_duration,
            open_frames,
            residue_histories,
            assessment,
        }
    }

    /// Analyze open/close transitions
    fn analyze_transitions(&self, timeline: &[bool]) -> (usize, usize, Vec<usize>) {
        if timeline.is_empty() {
            return (0, 0, Vec::new());
        }

        let mut opening_events = 0;
        let mut max_continuous_open = 0;
        let mut current_open_stretch = 0;
        let mut open_durations: Vec<usize> = Vec::new();
        let mut was_open = false;

        for &is_open in timeline {
            if is_open {
                current_open_stretch += 1;
                if !was_open {
                    opening_events += 1; // Transition from closed to open
                }
            } else {
                if was_open && current_open_stretch > 0 {
                    open_durations.push(current_open_stretch);
                    max_continuous_open = max_continuous_open.max(current_open_stretch);
                }
                current_open_stretch = 0;
            }
            was_open = is_open;
        }

        // Handle case where simulation ends with pocket open
        if was_open && current_open_stretch > 0 {
            open_durations.push(current_open_stretch);
            max_continuous_open = max_continuous_open.max(current_open_stretch);
        }

        (opening_events, max_continuous_open, open_durations)
    }

    /// Get a summary suitable for logging
    pub fn get_summary(&self) -> String {
        let metrics = self.analyze();
        format!(
            "Persistence: {:.1}% | Events: {} | MaxOpen: {} frames | Assessment: {:?}",
            metrics.persistence_ratio * 100.0,
            metrics.opening_events,
            metrics.max_continuous_open,
            metrics.assessment
        )
    }
}

/// Convenience function to run simulation with persistence tracking
/// Returns (final_atoms, persistence_metrics)
pub fn run_simulation_with_persistence<F>(
    config: PersistenceConfig,
    initial_atoms: Vec<Atom>,
    target_residues: Vec<usize>,
    core_residues: Vec<usize>,
    total_steps: u64,
    mut run_chunk: F,
) -> anyhow::Result<(Vec<Atom>, PocketPersistenceMetrics)>
where
    F: FnMut(u64) -> anyhow::Result<Vec<Atom>>,
{
    let mut tracker = PersistenceTracker::new(
        config,
        initial_atoms,
        target_residues,
        core_residues,
        total_steps,
    );

    let sample_interval = tracker.get_sample_interval();
    let num_samples = tracker.get_num_samples();

    let mut final_atoms = Vec::new();

    for sample_idx in 0..num_samples {
        // Run simulation chunk
        final_atoms = run_chunk(sample_interval)?;

        // Record snapshot
        tracker.record_snapshot(sample_idx, final_atoms.clone());

        log::debug!(
            "Persistence sample {}/{}: {}",
            sample_idx + 1,
            num_samples,
            tracker.get_summary()
        );
    }

    let metrics = tracker.analyze();
    Ok((final_atoms, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistence_assessment() {
        assert_eq!(
            PersistenceAssessment::from_ratio(0.8),
            PersistenceAssessment::Stable
        );
        assert_eq!(
            PersistenceAssessment::from_ratio(0.6),
            PersistenceAssessment::Moderate
        );
        assert_eq!(
            PersistenceAssessment::from_ratio(0.4),
            PersistenceAssessment::Transient
        );
        assert_eq!(
            PersistenceAssessment::from_ratio(0.2),
            PersistenceAssessment::Unstable
        );
    }

    #[test]
    fn test_transition_analysis() {
        let tracker = PersistenceTracker::new(
            PersistenceConfig::default(),
            Vec::new(),
            vec![1, 2, 3],
            vec![10, 11],
            1000,
        );

        // Test: open, open, close, open, open, open, close
        let timeline = vec![true, true, false, true, true, true, false];
        let (events, max_open, durations) = tracker.analyze_transitions(&timeline);

        assert_eq!(events, 2); // Two opening events
        assert_eq!(max_open, 3); // Longest open stretch is 3
        assert_eq!(durations, vec![2, 3]); // Two open periods: 2 frames and 3 frames
    }

    #[test]
    fn test_druggability() {
        assert!(PersistenceAssessment::Stable.is_druggable());
        assert!(PersistenceAssessment::Moderate.is_druggable());
        assert!(!PersistenceAssessment::Transient.is_druggable());
        assert!(!PersistenceAssessment::Unstable.is_druggable());
    }
}
