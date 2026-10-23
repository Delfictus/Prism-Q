//! Adaptive Resolution NHS Protocol
//!
//! Implements intelligent exploration strategy for cryptic site detection:
//! - Phase 1: Broad survey with coarse grid (2Å) - fast coverage
//! - Phase 2: Signal-guided convergence (1Å) - focus on hot zones
//! - Phase 3: Precision mapping (0.5Å) - confirmation of validated sites
//!
//! ## Jitter Detection
//!
//! In quiet landscapes (equilibrated systems), small perturbations become
//! the meaningful signal. UV-induced "jitter" propagates through hidden
//! pathways, revealing cryptic binding sites through cascade effects.
//!
//! ## Scientific Principle
//!
//! This mimics experimental pump-probe spectroscopy:
//! 1. Establish quiet baseline (system at equilibrium)
//! 2. Apply controlled perturbation (UV pulse to aromatics)
//! 3. Observe jitter propagation (signal in quiet landscape)
//! 4. Detect cascade events (cooperative dewetting = cryptic site)

use std::collections::{HashMap, HashSet};

/// Exploration phase for adaptive protocol
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplorationPhase {
    /// Initial broad survey with coarse resolution
    Survey,
    /// Signal-guided convergence on hot zones
    Convergence,
    /// Precision mapping of validated sites
    Precision,
}

/// UV targeting strategy that evolves with exploration phase
#[derive(Debug, Clone)]
pub enum UvStrategy {
    /// Random uniform targeting - all aromatics equally
    Random { burst_interval: i32, energy: f32 },
    /// Signal-guided targeting - focus on responsive regions
    SignalGuided {
        hot_zone_probability: f32,
        random_probability: f32,
        burst_interval: i32,
        energy: f32,
        exclusion_radius: f32,
    },
    /// Precision targeting - only confirmed sites
    PrecisionMapping {
        confirmed_sites: Vec<i32>,
        burst_interval: i32,
        energy: f32,
        spatial_refinement: bool,
    },
}

/// Configuration for a single exploration phase
#[derive(Debug, Clone)]
pub struct GridPhase {
    /// Grid resolution in Angstroms
    pub resolution: f32,
    /// Duration in timesteps
    pub duration: i32,
    /// UV targeting strategy
    pub uv_strategy: UvStrategy,
    /// Detection threshold (lower = more sensitive)
    pub detection_threshold: f32,
}

/// Complete adaptive grid protocol with 3 phases
#[derive(Debug, Clone)]
pub struct AdaptiveGridProtocol {
    /// Phase 1: Broad survey
    pub phase1_survey: GridPhase,
    /// Phase 2: Signal-guided convergence
    pub phase2_convergence: GridPhase,
    /// Phase 3: Precision mapping
    pub phase3_precision: GridPhase,
    /// Current phase
    pub current_phase: ExplorationPhase,
    /// Current step within phase
    pub current_step: i32,
}

impl Default for AdaptiveGridProtocol {
    fn default() -> Self {
        Self {
            phase1_survey: GridPhase {
                resolution: 2.0, // Coarse 2Å grid - fast coverage
                duration: 20000, // 20k steps initial survey
                uv_strategy: UvStrategy::Random {
                    burst_interval: 2000, // Infrequent, broad sampling
                    energy: 4.0,          // Low energy, gentle probing
                },
                detection_threshold: 0.4, // High threshold - only strong signals
            },
            phase2_convergence: GridPhase {
                resolution: 1.0, // Standard 1Å grid
                duration: 40000, // Main exploration phase
                uv_strategy: UvStrategy::SignalGuided {
                    hot_zone_probability: 0.7, // 70% chance to target hot zones
                    random_probability: 0.3,   // 30% random exploration
                    burst_interval: 1000,      // More frequent probing
                    energy: 6.0,               // Increased energy
                    exclusion_radius: 5.0,     // Don't re-probe dead zones
                },
                detection_threshold: 0.25, // Lower threshold - more sensitive
            },
            phase3_precision: GridPhase {
                resolution: 0.5, // Fine 0.5Å grid around signals
                duration: 40000, // Detailed characterization
                uv_strategy: UvStrategy::PrecisionMapping {
                    confirmed_sites: Vec::new(),
                    burst_interval: 500,      // High frequency validation
                    energy: 8.0,              // Maximum energy for confirmation
                    spatial_refinement: true, // Sub-voxel precision
                },
                detection_threshold: 0.15, // Lowest threshold - maximum sensitivity
            },
            current_phase: ExplorationPhase::Survey,
            current_step: 0,
        }
    }
}

impl AdaptiveGridProtocol {
    /// Create new adaptive protocol
    pub fn new() -> Self {
        Self::default()
    }

    /// Get current phase configuration
    pub fn current_config(&self) -> &GridPhase {
        match self.current_phase {
            ExplorationPhase::Survey => &self.phase1_survey,
            ExplorationPhase::Convergence => &self.phase2_convergence,
            ExplorationPhase::Precision => &self.phase3_precision,
        }
    }

    /// Get current grid resolution
    pub fn current_resolution(&self) -> f32 {
        self.current_config().resolution
    }

    /// Get current detection threshold
    pub fn current_threshold(&self) -> f32 {
        self.current_config().detection_threshold
    }

    /// Get current UV strategy
    pub fn current_uv_strategy(&self) -> &UvStrategy {
        &self.current_config().uv_strategy
    }

    /// Total steps in protocol
    pub fn total_steps(&self) -> i32 {
        self.phase1_survey.duration
            + self.phase2_convergence.duration
            + self.phase3_precision.duration
    }

    /// Advance to next step, return true if phase transitioned
    pub fn advance(&mut self) -> bool {
        self.current_step += 1;

        let phase_duration = self.current_config().duration;
        if self.current_step >= phase_duration {
            self.current_step = 0;
            match self.current_phase {
                ExplorationPhase::Survey => {
                    self.current_phase = ExplorationPhase::Convergence;
                    return true;
                }
                ExplorationPhase::Convergence => {
                    self.current_phase = ExplorationPhase::Precision;
                    return true;
                }
                ExplorationPhase::Precision => {
                    // Stay in precision phase
                    self.current_step = phase_duration - 1;
                }
            }
        }
        false
    }

    /// Check if protocol is complete
    pub fn is_complete(&self) -> bool {
        self.current_phase == ExplorationPhase::Precision
            && self.current_step >= self.phase3_precision.duration - 1
    }

    /// Update confirmed sites for precision phase
    pub fn set_confirmed_sites(&mut self, sites: Vec<i32>) {
        if let UvStrategy::PrecisionMapping {
            confirmed_sites, ..
        } = &mut self.phase3_precision.uv_strategy
        {
            *confirmed_sites = sites;
        }
    }
}

// ============================================================================
// JITTER DETECTION
// ============================================================================

/// A detected jitter signal
#[derive(Debug, Clone)]
pub struct JitterSignal {
    /// Voxel location
    pub voxel_idx: i32,
    /// 3D position
    pub position: [f32; 3],
    /// Jitter intensity relative to baseline
    pub intensity: f32,
    /// Signal-to-noise ratio
    pub confidence: f32,
    /// Timestep when detected
    pub timestep: i32,
    /// Which UV burst triggered this (if known)
    pub trigger_burst: Option<i32>,
}

/// Quiet baseline measurement
#[derive(Debug, Clone)]
pub struct QuietBaseline {
    /// Steps used for equilibration
    pub equilibration_steps: i32,
    /// Measured thermal noise floor
    pub thermal_noise_floor: f32,
    /// Spatial correlation length of background
    pub spatial_correlation: f32,
    /// Temporal correlation of background
    pub temporal_correlation: f32,
    /// Per-voxel baseline variance
    pub voxel_variances: Vec<f32>,
    /// Is baseline established?
    pub is_established: bool,
}

impl Default for QuietBaseline {
    fn default() -> Self {
        Self {
            equilibration_steps: 10000,
            thermal_noise_floor: 0.05,
            spatial_correlation: 2.0,
            temporal_correlation: 50.0,
            voxel_variances: Vec::new(),
            is_established: false,
        }
    }
}

/// Jitter detection configuration
#[derive(Debug, Clone)]
pub struct JitterConfig {
    /// UV pulse energy for jitter induction
    pub pulse_energy: f32,
    /// Duration of pulse in steps
    pub pulse_duration: i32,
    /// Recovery time to wait after pulse
    pub recovery_time: i32,
    /// Spatial radius to monitor for jitter
    pub detection_radius: f32,
    /// Temporal window for jitter detection
    pub detection_window: i32,
    /// Sensitivity amplification during detection
    pub amplification_factor: f32,
    /// Minimum amplitude for meaningful jitter
    pub amplitude_threshold: f32,
    /// Minimum spatial coherence
    pub coherence_threshold: f32,
    /// Minimum temporal persistence
    pub persistence_threshold: f32,
}

impl Default for JitterConfig {
    fn default() -> Self {
        Self {
            pulse_energy: 6.0,
            pulse_duration: 5,
            recovery_time: 100,
            detection_radius: 15.0,
            detection_window: 200,
            amplification_factor: 3.0,
            amplitude_threshold: 0.1,
            coherence_threshold: 0.3,
            persistence_threshold: 10.0,
        }
    }
}

/// Main jitter detector for quiet landscape signal detection
#[derive(Debug, Clone)]
pub struct JitterDetector {
    /// Configuration
    pub config: JitterConfig,
    /// Quiet baseline
    pub baseline: QuietBaseline,
    /// History of jitter signals
    pub signals: Vec<JitterSignal>,
    /// Current sensitivity level (increases as system gets quieter)
    pub sensitivity_boost: f32,
    /// Duration of quiet period
    pub quiet_duration: i32,
    /// Recent variance per voxel (for signal detection)
    recent_variances: Vec<f32>,
    /// Voxel spike history (for hot zone identification)
    spike_history: Vec<Vec<f32>>,
    /// History length in timesteps
    history_length: usize,
}

impl JitterDetector {
    /// Create new jitter detector
    pub fn new(config: JitterConfig, n_voxels: usize) -> Self {
        let history_length = 1000;
        Self {
            config,
            baseline: QuietBaseline::default(),
            signals: Vec::new(),
            sensitivity_boost: 1.0,
            quiet_duration: 0,
            recent_variances: vec![0.0; n_voxels],
            spike_history: vec![vec![0.0; history_length]; n_voxels],
            history_length,
        }
    }

    /// Update detector with current field state
    pub fn update(&mut self, water_density: &[f32], timestep: i32) {
        // Update spike history
        let history_idx = (timestep as usize) % self.history_length;
        for (i, &density) in water_density.iter().enumerate() {
            if i < self.spike_history.len() {
                self.spike_history[i][history_idx] = density;
            }
        }

        // Update sensitivity based on quiet duration
        self.update_sensitivity();
    }

    /// Update sensitivity based on how long the system has been quiet
    fn update_sensitivity(&mut self) {
        self.sensitivity_boost = match self.quiet_duration {
            0..=1000 => 1.0,     // Standard sensitivity
            1001..=5000 => 1.5,  // System settling - moderate boost
            5001..=10000 => 2.0, // Quiet achieved - high sensitivity
            _ => 3.0,            // Ultra-quiet - maximum sensitivity
        };
    }

    /// Establish quiet baseline from equilibrated system
    pub fn establish_baseline(&mut self, water_density_history: &[Vec<f32>]) {
        if water_density_history.is_empty() || water_density_history[0].is_empty() {
            return;
        }

        let n_voxels = water_density_history[0].len();
        let n_frames = water_density_history.len();

        // Calculate per-voxel variance
        self.baseline.voxel_variances = vec![0.0; n_voxels];
        for voxel in 0..n_voxels {
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            for frame in water_density_history {
                let val = frame[voxel];
                sum += val;
                sum_sq += val * val;
            }
            let mean = sum / n_frames as f32;
            let variance = (sum_sq / n_frames as f32) - mean * mean;
            self.baseline.voxel_variances[voxel] = variance.max(0.001); // Avoid zero
        }

        // Calculate overall noise floor
        let total_variance: f32 = self.baseline.voxel_variances.iter().sum();
        self.baseline.thermal_noise_floor = (total_variance / n_voxels as f32).sqrt();
        self.baseline.is_established = true;
    }

    /// Detect jitter signals in current field state
    pub fn detect_jitter(&mut self, water_density: &[f32], timestep: i32) -> Vec<JitterSignal> {
        if !self.baseline.is_established {
            return Vec::new();
        }

        let mut signals = Vec::new();
        let threshold = self.config.amplitude_threshold / self.sensitivity_boost;

        for (voxel, &current) in water_density.iter().enumerate() {
            if voxel >= self.baseline.voxel_variances.len() {
                continue;
            }

            let baseline_var = self.baseline.voxel_variances[voxel];

            // Calculate current variance from recent history
            let current_var = self.calculate_recent_variance(voxel);
            self.recent_variances[voxel] = current_var;

            // Check for significant variance increase (jitter)
            let variance_ratio = current_var / baseline_var;
            if variance_ratio > 1.0 + threshold {
                // Check spatial coherence
                let coherence = self.measure_spatial_coherence(voxel, water_density);
                if coherence < self.config.coherence_threshold {
                    continue;
                }

                // Check temporal persistence
                let persistence = self.measure_persistence(voxel);
                if persistence < self.config.persistence_threshold {
                    continue;
                }

                // This is meaningful jitter!
                let intensity = variance_ratio - 1.0;
                let confidence = (coherence + persistence / 100.0) / 2.0;

                signals.push(JitterSignal {
                    voxel_idx: voxel as i32,
                    position: [0.0, 0.0, 0.0], // Would be filled by caller
                    intensity,
                    confidence,
                    timestep,
                    trigger_burst: None,
                });
            }
        }

        // Store signals
        self.signals.extend(signals.clone());
        signals
    }

    /// Calculate recent variance for a voxel
    fn calculate_recent_variance(&self, voxel: usize) -> f32 {
        if voxel >= self.spike_history.len() {
            return 0.0;
        }

        let history = &self.spike_history[voxel];
        let n = history.len() as f32;

        let sum: f32 = history.iter().sum();
        let sum_sq: f32 = history.iter().map(|x| x * x).sum();

        let mean = sum / n;
        let variance = (sum_sq / n) - mean * mean;
        variance.max(0.0)
    }

    /// Measure spatial coherence around a voxel
    fn measure_spatial_coherence(&self, voxel: usize, water_density: &[f32]) -> f32 {
        // Simplified: check if neighbors have similar variance changes
        let neighbors = self.get_neighbor_voxels(voxel, water_density.len());
        if neighbors.is_empty() {
            return 0.0;
        }

        let my_var = self.recent_variances.get(voxel).copied().unwrap_or(0.0);
        let mut similar_count = 0;

        for &neighbor in &neighbors {
            let neighbor_var = self.recent_variances.get(neighbor).copied().unwrap_or(0.0);
            if (my_var - neighbor_var).abs() < my_var * 0.5 {
                similar_count += 1;
            }
        }

        similar_count as f32 / neighbors.len() as f32
    }

    /// Measure temporal persistence of signal at voxel
    fn measure_persistence(&self, voxel: usize) -> f32 {
        if voxel >= self.spike_history.len() {
            return 0.0;
        }

        let history = &self.spike_history[voxel];

        // Count consecutive frames above baseline
        let mut max_streak = 0;
        let mut current_streak = 0;
        let baseline_var = self
            .baseline
            .voxel_variances
            .get(voxel)
            .copied()
            .unwrap_or(0.1);
        let threshold = baseline_var * 1.2;

        for &val in history {
            if val > threshold {
                current_streak += 1;
                max_streak = max_streak.max(current_streak);
            } else {
                current_streak = 0;
            }
        }

        max_streak as f32
    }

    /// Get neighbor voxels (simplified 6-connected)
    fn get_neighbor_voxels(&self, voxel: usize, n_voxels: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();
        if voxel > 0 {
            neighbors.push(voxel - 1);
        }
        if voxel + 1 < n_voxels {
            neighbors.push(voxel + 1);
        }
        neighbors
    }

    /// Identify hot zones that show emerging activity
    pub fn identify_hot_zones(&self) -> Vec<i32> {
        let mut hot_zones = Vec::new();

        for (voxel, history) in self.spike_history.iter().enumerate() {
            // Compare recent vs baseline activity
            let baseline_len = history.len() / 2;
            let baseline_activity: f32 = history[..baseline_len].iter().sum();
            let recent_activity: f32 = history[baseline_len..].iter().sum();

            // Hot zone if recent > 3x baseline
            if recent_activity > 3.0 * baseline_activity && recent_activity > 0.1 {
                hot_zones.push(voxel as i32);
            }
        }

        hot_zones
    }

    /// Get all detected signals
    pub fn get_signals(&self) -> &[JitterSignal] {
        &self.signals
    }

    /// Clear signal history
    pub fn clear_signals(&mut self) {
        self.signals.clear();
    }
}

// ============================================================================
// CASCADE DETECTION
// ============================================================================

/// A cascade event (cooperative dewetting indicating cryptic site)
#[derive(Debug, Clone)]
pub struct CascadeEvent {
    /// Initial trigger signal
    pub trigger: JitterSignal,
    /// Cascade propagation sites
    pub cascade_sites: Vec<JitterSignal>,
    /// Overall confidence
    pub confidence: f32,
    /// Estimated cryptic site center
    pub site_center: [f32; 3],
    /// Estimated site radius
    pub site_radius: f32,
}

/// Cascade detector for cryptic site opening
#[derive(Debug, Clone)]
pub struct CascadeDetector {
    /// Minimum cascade size to report
    pub min_cascade_size: usize,
    /// Time window for cascade propagation
    pub cascade_window: i32,
    /// Spatial radius for cascade connection
    pub cascade_radius: f32,
    /// Detected cascade events
    pub events: Vec<CascadeEvent>,
    /// Validated cryptic sites
    pub validated_sites: Vec<[f32; 3]>,
}

impl Default for CascadeDetector {
    fn default() -> Self {
        Self {
            min_cascade_size: 3,
            cascade_window: 50,
            cascade_radius: 8.0,
            events: Vec::new(),
            validated_sites: Vec::new(),
        }
    }
}

impl CascadeDetector {
    /// Create new cascade detector
    pub fn new(min_cascade_size: usize, cascade_window: i32, cascade_radius: f32) -> Self {
        Self {
            min_cascade_size,
            cascade_window,
            cascade_radius,
            events: Vec::new(),
            validated_sites: Vec::new(),
        }
    }

    /// Detect cascade events from jitter signals
    pub fn detect_cascades(&mut self, signals: &[JitterSignal]) -> Vec<CascadeEvent> {
        if signals.len() < self.min_cascade_size {
            return Vec::new();
        }

        let mut new_cascades = Vec::new();

        // Group signals by temporal proximity
        let mut time_groups: HashMap<i32, Vec<&JitterSignal>> = HashMap::new();
        for signal in signals {
            let time_bucket = signal.timestep / self.cascade_window;
            time_groups.entry(time_bucket).or_default().push(signal);
        }

        // Look for spatially connected groups
        for (_, group) in time_groups {
            let cascades = self.find_spatial_clusters(&group);
            for cascade in cascades {
                if cascade.len() >= self.min_cascade_size {
                    let event = self.build_cascade_event(cascade);
                    new_cascades.push(event);
                }
            }
        }

        // Store events
        self.events.extend(new_cascades.clone());
        new_cascades
    }

    /// Find spatially connected clusters of signals
    fn find_spatial_clusters<'a>(
        &self,
        signals: &[&'a JitterSignal],
    ) -> Vec<Vec<&'a JitterSignal>> {
        let mut clusters: Vec<Vec<&'a JitterSignal>> = Vec::new();
        let mut used: HashSet<i32> = HashSet::new();

        for signal in signals {
            if used.contains(&signal.voxel_idx) {
                continue;
            }

            // Start new cluster
            let mut cluster = vec![*signal];
            used.insert(signal.voxel_idx);

            // Grow cluster by finding connected signals
            let mut frontier = vec![signal.voxel_idx];
            while let Some(current) = frontier.pop() {
                for &other_signal in signals {
                    if used.contains(&other_signal.voxel_idx) {
                        continue;
                    }

                    // Check spatial proximity (simplified using voxel indices)
                    let dist = (other_signal.voxel_idx - current).abs();
                    if dist < (self.cascade_radius as i32) {
                        cluster.push(other_signal);
                        used.insert(other_signal.voxel_idx);
                        frontier.push(other_signal.voxel_idx);
                    }
                }
            }

            clusters.push(cluster);
        }

        clusters
    }

    /// Build cascade event from cluster
    fn build_cascade_event(&self, signals: Vec<&JitterSignal>) -> CascadeEvent {
        // Find trigger (earliest signal)
        let trigger = signals
            .iter()
            .min_by_key(|s| s.timestep)
            .cloned()
            .cloned()
            .unwrap_or_else(|| signals[0].clone());

        // Calculate center and radius
        let mut center = [0.0f32; 3];
        let mut total_intensity = 0.0f32;
        for signal in &signals {
            center[0] += signal.position[0] * signal.intensity;
            center[1] += signal.position[1] * signal.intensity;
            center[2] += signal.position[2] * signal.intensity;
            total_intensity += signal.intensity;
        }
        if total_intensity > 0.0 {
            center[0] /= total_intensity;
            center[1] /= total_intensity;
            center[2] /= total_intensity;
        }

        // Calculate radius
        let mut max_dist = 0.0f32;
        for signal in &signals {
            let dx = signal.position[0] - center[0];
            let dy = signal.position[1] - center[1];
            let dz = signal.position[2] - center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            max_dist = max_dist.max(dist);
        }

        // Calculate confidence
        let confidence = (signals.len() as f32 / self.min_cascade_size as f32).min(1.0)
            * signals.iter().map(|s| s.confidence).sum::<f32>()
            / signals.len() as f32;

        CascadeEvent {
            trigger,
            cascade_sites: signals.into_iter().cloned().collect(),
            confidence,
            site_center: center,
            site_radius: max_dist + 2.0, // Add buffer
        }
    }

    /// Validate cascade as cryptic site
    pub fn validate_cascade(&mut self, event: &CascadeEvent) -> bool {
        if event.confidence > 0.5 && event.cascade_sites.len() >= self.min_cascade_size {
            self.validated_sites.push(event.site_center);
            return true;
        }
        false
    }

    /// Get validated cryptic sites
    pub fn get_validated_sites(&self) -> &[[f32; 3]] {
        &self.validated_sites
    }
}

// ============================================================================
// ADAPTIVE NHS ENGINE
// ============================================================================

/// Complete adaptive NHS engine combining all components
pub struct AdaptiveNhsEngine {
    /// Adaptive grid protocol
    pub protocol: AdaptiveGridProtocol,
    /// Jitter detector
    pub jitter_detector: JitterDetector,
    /// Cascade detector
    pub cascade_detector: CascadeDetector,
    /// Current timestep
    pub timestep: i32,
    /// Water density history for baseline
    water_density_history: Vec<Vec<f32>>,
    /// History capacity
    history_capacity: usize,
}

impl AdaptiveNhsEngine {
    /// Create new adaptive NHS engine
    pub fn new(n_voxels: usize) -> Self {
        Self {
            protocol: AdaptiveGridProtocol::new(),
            jitter_detector: JitterDetector::new(JitterConfig::default(), n_voxels),
            cascade_detector: CascadeDetector::default(),
            timestep: 0,
            water_density_history: Vec::new(),
            history_capacity: 1000,
        }
    }

    /// Process one timestep
    pub fn step(&mut self, water_density: &[f32]) -> AdaptiveStepResult {
        self.timestep += 1;

        // Store history for baseline
        if self.water_density_history.len() < self.history_capacity {
            self.water_density_history.push(water_density.to_vec());
        } else {
            let idx = (self.timestep as usize) % self.history_capacity;
            self.water_density_history[idx] = water_density.to_vec();
        }

        // Update jitter detector
        self.jitter_detector.update(water_density, self.timestep);

        // Phase-specific processing
        let result = match self.protocol.current_phase {
            ExplorationPhase::Survey => self.process_survey_phase(water_density),
            ExplorationPhase::Convergence => self.process_convergence_phase(water_density),
            ExplorationPhase::Precision => self.process_precision_phase(water_density),
        };

        // Check for phase transition
        let phase_changed = self.protocol.advance();
        if phase_changed {
            self.handle_phase_transition();
        }

        result
    }

    /// Process survey phase - establish baseline and detect strong signals
    fn process_survey_phase(&mut self, water_density: &[f32]) -> AdaptiveStepResult {
        // Establish baseline after equilibration
        if self.timestep == self.jitter_detector.baseline.equilibration_steps {
            self.jitter_detector
                .establish_baseline(&self.water_density_history);
        }

        AdaptiveStepResult {
            phase: ExplorationPhase::Survey,
            resolution: self.protocol.current_resolution(),
            jitter_signals: Vec::new(),
            cascade_events: Vec::new(),
            hot_zones: Vec::new(),
        }
    }

    /// Process convergence phase - signal-guided exploration
    fn process_convergence_phase(&mut self, water_density: &[f32]) -> AdaptiveStepResult {
        // Detect jitter signals
        let jitter_signals = self
            .jitter_detector
            .detect_jitter(water_density, self.timestep);

        // Identify hot zones
        let hot_zones = self.jitter_detector.identify_hot_zones();

        // Detect cascades
        let cascade_events = self.cascade_detector.detect_cascades(&jitter_signals);

        AdaptiveStepResult {
            phase: ExplorationPhase::Convergence,
            resolution: self.protocol.current_resolution(),
            jitter_signals,
            cascade_events,
            hot_zones,
        }
    }

    /// Process precision phase - validate and refine
    fn process_precision_phase(&mut self, water_density: &[f32]) -> AdaptiveStepResult {
        // Continue jitter detection with maximum sensitivity
        let jitter_signals = self
            .jitter_detector
            .detect_jitter(water_density, self.timestep);

        // Detect cascades
        let cascade_events = self.cascade_detector.detect_cascades(&jitter_signals);

        // Validate new cascades
        for event in &cascade_events {
            self.cascade_detector.validate_cascade(event);
        }

        AdaptiveStepResult {
            phase: ExplorationPhase::Precision,
            resolution: self.protocol.current_resolution(),
            jitter_signals,
            cascade_events,
            hot_zones: Vec::new(),
        }
    }

    /// Handle phase transition
    fn handle_phase_transition(&mut self) {
        match self.protocol.current_phase {
            ExplorationPhase::Convergence => {
                // Entering convergence - baseline should be established
                log::info!(
                    "Entering convergence phase - baseline noise: {:.4}",
                    self.jitter_detector.baseline.thermal_noise_floor
                );
            }
            ExplorationPhase::Precision => {
                // Entering precision - set confirmed sites
                let hot_zones = self.jitter_detector.identify_hot_zones();
                self.protocol.set_confirmed_sites(hot_zones);
                log::info!(
                    "Entering precision phase - {} hot zones identified",
                    self.cascade_detector.validated_sites.len()
                );
            }
            _ => {}
        }
    }

    /// Get current UV targeting parameters
    pub fn get_uv_params(&self) -> (f32, i32, Option<i32>) {
        match self.protocol.current_uv_strategy() {
            UvStrategy::Random {
                energy,
                burst_interval,
            } => (*energy, *burst_interval, None),
            UvStrategy::SignalGuided {
                energy,
                burst_interval,
                ..
            } => {
                // Select target based on hot zones
                let hot_zones = self.jitter_detector.identify_hot_zones();
                let target = if !hot_zones.is_empty() {
                    Some(hot_zones[0])
                } else {
                    None
                };
                (*energy, *burst_interval, target)
            }
            UvStrategy::PrecisionMapping {
                energy,
                burst_interval,
                confirmed_sites,
                ..
            } => {
                let target = confirmed_sites.first().copied();
                (*energy, *burst_interval, target)
            }
        }
    }

    /// Get detection summary
    pub fn get_summary(&self) -> AdaptiveSummary {
        AdaptiveSummary {
            total_steps: self.timestep,
            current_phase: self.protocol.current_phase,
            jitter_signals_detected: self.jitter_detector.signals.len(),
            cascade_events_detected: self.cascade_detector.events.len(),
            validated_sites: self.cascade_detector.validated_sites.len(),
            baseline_noise: self.jitter_detector.baseline.thermal_noise_floor,
            current_sensitivity: self.jitter_detector.sensitivity_boost,
        }
    }
}

/// Result of one adaptive step
#[derive(Debug, Clone)]
pub struct AdaptiveStepResult {
    /// Current phase
    pub phase: ExplorationPhase,
    /// Current grid resolution
    pub resolution: f32,
    /// Jitter signals detected this step
    pub jitter_signals: Vec<JitterSignal>,
    /// Cascade events detected this step
    pub cascade_events: Vec<CascadeEvent>,
    /// Currently identified hot zones
    pub hot_zones: Vec<i32>,
}

/// Summary of adaptive detection
#[derive(Debug, Clone)]
pub struct AdaptiveSummary {
    /// Total timesteps run
    pub total_steps: i32,
    /// Current exploration phase
    pub current_phase: ExplorationPhase,
    /// Total jitter signals detected
    pub jitter_signals_detected: usize,
    /// Total cascade events detected
    pub cascade_events_detected: usize,
    /// Validated cryptic sites
    pub validated_sites: usize,
    /// Baseline noise floor
    pub baseline_noise: f32,
    /// Current sensitivity boost
    pub current_sensitivity: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_protocol_phases() {
        let mut protocol = AdaptiveGridProtocol::new();

        assert_eq!(protocol.current_phase, ExplorationPhase::Survey);
        assert_eq!(protocol.current_resolution(), 2.0);

        // Advance through survey
        for _ in 0..20000 {
            protocol.advance();
        }

        assert_eq!(protocol.current_phase, ExplorationPhase::Convergence);
        assert_eq!(protocol.current_resolution(), 1.0);
    }

    #[test]
    fn test_jitter_detector() {
        let detector = JitterDetector::new(JitterConfig::default(), 100);
        assert!(!detector.baseline.is_established);
    }

    #[test]
    fn test_cascade_detector() {
        let detector = CascadeDetector::default();
        assert_eq!(detector.min_cascade_size, 3);
    }
}
