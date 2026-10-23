//! [STAGE-2B-RT] RT Probe Data Analysis
//!
//! Process RT probe data from Stage 2a to detect cryptic site formation signals.
//!
//! **Signals**:
//! - **Geometric void formation**: Hit distance time series → void opening rate
//! - **Solvation disruption**: Variance in hit distances → water reorganization
//!
//! These are LEADING INDICATORS that occur before full pocket opening.

use crate::rt_probe::RtProbeSnapshot;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// RT probe analysis configuration
#[derive(Debug, Clone)]
pub struct RtAnalysisConfig {
    /// Window size for variance calculation (timesteps)
    pub variance_window: usize,
    /// Void formation threshold (Å increase in hit distance)
    pub void_threshold: f32,
    /// Solvation disruption threshold (Å variance)
    pub disruption_threshold: f32,
    /// Minimum persistence (consecutive timesteps)
    pub min_persistence: usize,
}

impl Default for RtAnalysisConfig {
    fn default() -> Self {
        Self {
            variance_window: 20,       // 20 timesteps
            void_threshold: 2.0,       // 2Å void formation
            disruption_threshold: 0.5, // 0.5Å variance
            min_persistence: 5,        // 5 consecutive timesteps
        }
    }
}

/// Void formation event detected by RT probes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoidFormationEvent {
    /// Timestep when void formed
    pub timestep: i32,
    /// Spatial position of void centroid [x, y, z] in Å
    pub position: [f32; 3],
    /// Average hit distance increase (Å)
    pub distance_increase: f32,
    /// Aromatic LIF count in region
    pub aromatic_lif_count: usize,
    /// Persistence (consecutive timesteps detected)
    pub persistence: usize,
}

/// Solvation disruption event (water reorganization)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolvationDisruptionEvent {
    /// Timestep when disruption detected
    pub timestep: i32,
    /// Spatial position of disruption [x, y, z] in Å
    pub position: [f32; 3],
    /// Solvation variance (Å)
    pub variance: f32,
    /// Is this a leading signal? (occurs before geometric void)
    pub is_leading: bool,
    /// Timesteps until void formation (if leading)
    pub timesteps_until_void: Option<i32>,
}

/// Complete RT probe analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtAnalysisResults {
    /// Void formation events
    pub void_events: Vec<VoidFormationEvent>,
    /// Solvation disruption events (explicit solvent only)
    pub disruption_events: Vec<SolvationDisruptionEvent>,
    /// Total snapshots analyzed
    pub total_snapshots: usize,
    /// Average hit distance (Å)
    pub avg_hit_distance: f32,
    /// Average solvation variance (Å) if available
    pub avg_solvation_variance: Option<f32>,
}

/// RT probe data analyzer
pub struct RtProbeAnalyzer {
    config: RtAnalysisConfig,
}

impl RtProbeAnalyzer {
    /// Create new RT probe analyzer
    pub fn new(config: RtAnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze RT probe snapshots from Stage 2a
    ///
    /// # Arguments
    /// * `snapshots` - RT probe snapshots from simulation
    ///
    /// # Returns
    /// RtAnalysisResults with void formation and disruption events
    pub fn analyze(&self, snapshots: &[RtProbeSnapshot]) -> Result<RtAnalysisResults> {
        anyhow::ensure!(!snapshots.is_empty(), "No RT probe snapshots provided");

        log::info!("Analyzing {} RT probe snapshots", snapshots.len());

        // Detect void formation events
        let void_events = self.detect_void_formation(snapshots)?;

        // Detect solvation disruption events (if solvation tracking enabled)
        let disruption_events = self.detect_solvation_disruption(snapshots)?;

        // Mark leading signals (disruption before void)
        let disruption_events = self.identify_leading_signals(&void_events, disruption_events);

        // Compute statistics
        let avg_hit_distance = self.compute_avg_hit_distance(snapshots);
        let avg_solvation_variance = self.compute_avg_solvation_variance(snapshots);

        log::info!(
            "RT analysis: {} void events, {} disruption events",
            void_events.len(),
            disruption_events.len()
        );

        Ok(RtAnalysisResults {
            void_events,
            disruption_events,
            total_snapshots: snapshots.len(),
            avg_hit_distance,
            avg_solvation_variance,
        })
    }

    /// Detect void formation from hit distance time series
    fn detect_void_formation(
        &self,
        snapshots: &[RtProbeSnapshot],
    ) -> Result<Vec<VoidFormationEvent>> {
        let mut events = Vec::new();

        if snapshots.len() < 2 {
            return Ok(events);
        }

        // Compute baseline (first snapshot)
        let baseline_distance = Self::avg_hit_distance(&snapshots[0]);

        let mut persistence_count = 0;
        let mut event_start_idx = 0;

        for (i, snapshot) in snapshots.iter().enumerate().skip(1) {
            let current_distance = Self::avg_hit_distance(snapshot);
            let distance_increase = current_distance - baseline_distance;

            if distance_increase >= self.config.void_threshold {
                if persistence_count == 0 {
                    event_start_idx = i;
                }
                persistence_count += 1;

                // If persistence threshold met, record event
                if persistence_count >= self.config.min_persistence {
                    events.push(VoidFormationEvent {
                        timestep: snapshot.timestep,
                        position: snapshot.probe_position,
                        distance_increase,
                        aromatic_lif_count: snapshot.aromatic_lif_count,
                        persistence: persistence_count,
                    });
                    // Reset for next event
                    persistence_count = 0;
                }
            } else {
                persistence_count = 0;
            }
        }

        Ok(events)
    }

    /// Detect solvation disruption from variance time series
    fn detect_solvation_disruption(
        &self,
        snapshots: &[RtProbeSnapshot],
    ) -> Result<Vec<SolvationDisruptionEvent>> {
        let mut events = Vec::new();

        if snapshots.len() < self.config.variance_window {
            return Ok(events);
        }

        // Sliding window variance calculation
        for i in self.config.variance_window..snapshots.len() {
            let window = &snapshots[i - self.config.variance_window..i];

            if let Some(variance) = Self::compute_window_variance(window) {
                if variance >= self.config.disruption_threshold {
                    events.push(SolvationDisruptionEvent {
                        timestep: snapshots[i].timestep,
                        position: snapshots[i].probe_position,
                        variance,
                        is_leading: false, // Will be set by identify_leading_signals
                        timesteps_until_void: None,
                    });
                }
            }
        }

        Ok(events)
    }

    /// Identify leading signals (disruption events that occur before void formation)
    fn identify_leading_signals(
        &self,
        void_events: &[VoidFormationEvent],
        mut disruption_events: Vec<SolvationDisruptionEvent>,
    ) -> Vec<SolvationDisruptionEvent> {
        for disruption in &mut disruption_events {
            // Find nearest future void event
            if let Some(void_event) = void_events
                .iter()
                .filter(|v| v.timestep > disruption.timestep)
                .min_by_key(|v| v.timestep - disruption.timestep)
            {
                let dt = void_event.timestep - disruption.timestep;
                if dt > 0 && dt <= 500 {
                    // Within 500 timesteps (1ps @ 2fs)
                    disruption.is_leading = true;
                    disruption.timesteps_until_void = Some(dt);
                }
            }
        }

        disruption_events
    }

    /// Compute average hit distance for a snapshot
    fn avg_hit_distance(snapshot: &RtProbeSnapshot) -> f32 {
        if snapshot.hit_distances.is_empty() {
            0.0
        } else {
            snapshot.hit_distances.iter().sum::<f32>() / snapshot.hit_distances.len() as f32
        }
    }

    /// Compute average hit distance across all snapshots
    fn compute_avg_hit_distance(&self, snapshots: &[RtProbeSnapshot]) -> f32 {
        let total: f32 = snapshots.iter().map(|s| Self::avg_hit_distance(s)).sum();
        total / snapshots.len() as f32
    }

    /// Compute average solvation variance if available
    fn compute_avg_solvation_variance(&self, snapshots: &[RtProbeSnapshot]) -> Option<f32> {
        let variances: Vec<f32> = snapshots
            .iter()
            .filter_map(|s| s.solvation_variance)
            .collect();

        if variances.is_empty() {
            None
        } else {
            Some(variances.iter().sum::<f32>() / variances.len() as f32)
        }
    }

    /// Compute variance for a window of snapshots
    fn compute_window_variance(window: &[RtProbeSnapshot]) -> Option<f32> {
        // Check if solvation tracking is enabled
        if window.iter().all(|s| s.solvation_variance.is_some()) {
            let values: Vec<f32> = window.iter().filter_map(|s| s.solvation_variance).collect();
            Some(Self::variance(&values))
        } else {
            None
        }
    }

    /// Compute variance of values
    fn variance(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let sum_sq_diff: f32 = values.iter().map(|v| (v - mean).powi(2)).sum();
        sum_sq_diff / values.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_snapshot(
        timestep: i32,
        hit_distance: f32,
        aromatic_lif: usize,
    ) -> RtProbeSnapshot {
        RtProbeSnapshot {
            timestep,
            probe_position: [0.0, 0.0, 0.0], // Test probe at origin
            hit_distances: vec![hit_distance; 10], // 10 rays, same distance
            void_detected: hit_distance > 5.0,
            solvation_variance: Some(0.1),
            aromatic_lif_count: aromatic_lif,
        }
    }

    #[test]
    fn test_analyzer_creation() {
        let config = RtAnalysisConfig::default();
        let _analyzer = RtProbeAnalyzer::new(config);
    }

    #[test]
    fn test_void_detection_no_events() {
        let analyzer = RtProbeAnalyzer::new(RtAnalysisConfig::default());
        let snapshots = vec![
            create_test_snapshot(0, 3.0, 0),
            create_test_snapshot(100, 3.1, 0),
            create_test_snapshot(200, 3.0, 0),
        ];

        let result = analyzer.analyze(&snapshots).unwrap();
        assert_eq!(result.void_events.len(), 0);
        assert_eq!(result.total_snapshots, 3);
    }

    #[test]
    fn test_void_detection_with_event() {
        let mut config = RtAnalysisConfig::default();
        config.void_threshold = 2.0;
        config.min_persistence = 2;
        let analyzer = RtProbeAnalyzer::new(config);

        let snapshots = vec![
            create_test_snapshot(0, 3.0, 0),   // Baseline
            create_test_snapshot(100, 5.5, 1), // +2.5Å (exceeds threshold)
            create_test_snapshot(200, 5.6, 1), // Persistent
            create_test_snapshot(300, 3.0, 0), // Returns to baseline
        ];

        let result = analyzer.analyze(&snapshots).unwrap();
        assert!(result.void_events.len() >= 1);

        let event = &result.void_events[0];
        assert!(event.distance_increase >= 2.0);
        assert!(event.persistence >= 2);
    }

    #[test]
    fn test_statistics_computation() {
        let analyzer = RtProbeAnalyzer::new(RtAnalysisConfig::default());
        let snapshots = vec![
            create_test_snapshot(0, 3.0, 0),
            create_test_snapshot(100, 4.0, 1),
            create_test_snapshot(200, 5.0, 2),
        ];

        let result = analyzer.analyze(&snapshots).unwrap();
        assert!((result.avg_hit_distance - 4.0).abs() < 0.1);
        assert!(result.avg_solvation_variance.is_some());
    }
}
