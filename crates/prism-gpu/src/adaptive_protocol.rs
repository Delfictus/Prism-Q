//! PRISM-NHS Adaptive Protocol Controller
//!
//! Manages the three-phase cryptic site detection:
//! 1. Cryo Burst - aggressive exploration in frozen state
//! 2. Thermal Ramp - validation during warming
//! 3. Focused Dig - exploitation of best candidates
//!
//! This controller coordinates:
//! - Temperature schedules
//! - UV probe intensity and targeting
//! - Exploration vs exploitation balance
//! - Candidate tracking and validation

use std::collections::HashMap;

/// Phase identifiers matching CUDA constants
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ExplorationPhase {
    CryoBurst = 0,
    ThermalRamp = 1,
    FocusedDig = 2,
}

impl ExplorationPhase {
    pub fn name(&self) -> &'static str {
        match self {
            Self::CryoBurst => "CRYO BURST",
            Self::ThermalRamp => "THERMAL RAMP",
            Self::FocusedDig => "FOCUSED DIG",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            Self::CryoBurst => "❄️",
            Self::ThermalRamp => "🌡️",
            Self::FocusedDig => "🎯",
        }
    }
}

/// Hot spot candidate tracked across phases
#[derive(Debug, Clone)]
pub struct HotSpotCandidate {
    pub id: usize,
    pub center: [f32; 3],
    pub confidence: f32,
    pub resonance_frequency: f32, // THz
    pub spike_density: f32,
    pub aromatic_count: usize,
    pub phase_discovered: ExplorationPhase,
    pub validation_count: usize,
    pub is_active: bool,
    pub contributing_aromatics: Vec<usize>,
    pub confidence_history: Vec<f32>,
}

impl HotSpotCandidate {
    pub fn new(id: usize, center: [f32; 3], phase: ExplorationPhase) -> Self {
        Self {
            id,
            center,
            confidence: 0.1,
            resonance_frequency: 10.0, // Will be updated to lowest observed
            spike_density: 0.0,
            aromatic_count: 0,
            phase_discovered: phase,
            validation_count: 0,
            is_active: true,
            contributing_aromatics: Vec::new(),
            confidence_history: Vec::new(),
        }
    }

    /// Check if candidate is near a position
    pub fn is_near(&self, position: &[f32; 3], threshold: f32) -> bool {
        let dx = position[0] - self.center[0];
        let dy = position[1] - self.center[1];
        let dz = position[2] - self.center[2];
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();
        dist < threshold
    }

    /// Update center with weighted average
    pub fn update_center(&mut self, position: &[f32; 3], weight: f32) {
        let total_weight = self.spike_density + weight;
        let w = weight / (total_weight + 1e-6);
        self.center[0] = self.center[0] * (1.0 - w) + position[0] * w;
        self.center[1] = self.center[1] * (1.0 - w) + position[1] * w;
        self.center[2] = self.center[2] * (1.0 - w) + position[2] * w;
        self.spike_density = total_weight;
    }
}

/// Protocol configuration
#[derive(Debug, Clone)]
pub struct AdaptiveProtocolConfig {
    // Phase durations
    pub cryo_steps: usize,
    pub ramp_steps: usize,
    pub dig_steps: usize,

    // Temperature settings
    pub cryo_temp: f32,
    pub ramp_start_temp: f32,
    pub ramp_end_temp: f32,
    pub physiological_temp: f32,

    // UV intensities
    pub cryo_uv_intensity: f32,
    pub ramp_uv_intensity: f32,
    pub dig_uv_intensity: f32,

    // Exploration radii (Angstroms)
    pub cryo_exploration_radius: f32,
    pub ramp_exploration_radius: f32,
    pub dig_exploration_radius: f32,

    // Thresholds
    pub min_candidates_for_ramp: usize,
    pub min_confidence_for_dig: f32,
    pub hot_spot_threshold: f32,
    pub site_confirmation_threshold: f32,
}

impl Default for AdaptiveProtocolConfig {
    fn default() -> Self {
        Self {
            cryo_steps: 100,
            ramp_steps: 400,
            dig_steps: 9500,

            cryo_temp: 80.0,
            ramp_start_temp: 80.0,
            ramp_end_temp: 300.0,
            physiological_temp: 300.0,

            cryo_uv_intensity: 10.0,
            ramp_uv_intensity: 5.0,
            dig_uv_intensity: 2.0,

            cryo_exploration_radius: 50.0,
            ramp_exploration_radius: 20.0,
            dig_exploration_radius: 8.0,

            min_candidates_for_ramp: 3,
            min_confidence_for_dig: 0.5,
            hot_spot_threshold: 0.3,
            site_confirmation_threshold: 0.8,
        }
    }
}

impl AdaptiveProtocolConfig {
    /// Create config for quick validation run
    pub fn quick_validation() -> Self {
        Self {
            cryo_steps: 50,
            ramp_steps: 100,
            dig_steps: 350,
            ..Default::default()
        }
    }

    /// Create config for deep analysis
    pub fn deep_analysis() -> Self {
        Self {
            cryo_steps: 200,
            ramp_steps: 800,
            dig_steps: 19000,
            ..Default::default()
        }
    }

    /// Create config for cryo-only analysis (no dynamics)
    pub fn cryo_only(steps: usize) -> Self {
        Self {
            cryo_steps: steps,
            ramp_steps: 0,
            dig_steps: 0,
            min_candidates_for_ramp: usize::MAX, // Never transition
            ..Default::default()
        }
    }

    pub fn total_steps(&self) -> usize {
        self.cryo_steps + self.ramp_steps + self.dig_steps
    }
}

/// Protocol state machine
#[derive(Debug)]
pub struct AdaptiveProtocolState {
    pub config: AdaptiveProtocolConfig,

    // Current state
    pub current_phase: ExplorationPhase,
    pub phase_step: usize,
    pub total_step: usize,

    // Temperature
    pub current_temp: f32,
    pub target_temp: f32,
    pub temp_ramp_rate: f32,

    // UV control
    pub current_uv_intensity: f32,
    pub probe_sweep_idx: usize,

    // Exploration
    pub exploration_radius: f32,
    pub exploration_center: [f32; 3],

    // Candidates
    pub candidates: Vec<HotSpotCandidate>,
    pub best_candidate_idx: Option<usize>,

    // Phase completion
    pub cryo_complete: bool,
    pub ramp_complete: bool,
    pub dig_complete: bool,
    pub site_confirmed: bool,

    // Statistics
    pub total_spikes: usize,
    pub spikes_this_phase: usize,
    pub peak_confidence: f32,
    pub best_site_location: [f32; 3],

    // Logging
    pub phase_transitions: Vec<(usize, ExplorationPhase, String)>,
}

impl AdaptiveProtocolState {
    pub fn new(config: AdaptiveProtocolConfig) -> Self {
        Self {
            current_phase: ExplorationPhase::CryoBurst,
            phase_step: 0,
            total_step: 0,

            current_temp: config.cryo_temp,
            target_temp: config.cryo_temp,
            temp_ramp_rate: 0.0,

            current_uv_intensity: config.cryo_uv_intensity,
            probe_sweep_idx: 0,

            exploration_radius: config.cryo_exploration_radius,
            exploration_center: [0.0, 0.0, 0.0],

            candidates: Vec::new(),
            best_candidate_idx: None,

            cryo_complete: false,
            ramp_complete: false,
            dig_complete: false,
            site_confirmed: false,

            total_spikes: 0,
            spikes_this_phase: 0,
            peak_confidence: 0.0,
            best_site_location: [0.0, 0.0, 0.0],

            phase_transitions: Vec::new(),

            config,
        }
    }

    /// Advance one simulation step
    pub fn step(&mut self) {
        self.total_step += 1;
        self.phase_step += 1;

        // Update temperature during ramp
        if self.current_phase == ExplorationPhase::ThermalRamp {
            self.current_temp += self.temp_ramp_rate;
            if self.current_temp > self.target_temp {
                self.current_temp = self.target_temp;
            }
        }

        // Check phase transitions
        self.check_phase_transition();
    }

    fn check_phase_transition(&mut self) {
        match self.current_phase {
            ExplorationPhase::CryoBurst => {
                if self.phase_step >= self.config.cryo_steps {
                    let n_active = self.candidates.iter().filter(|c| c.is_active).count();

                    if n_active >= self.config.min_candidates_for_ramp {
                        self.transition_to_ramp();
                    } else {
                        // Extend cryo phase
                        self.config.cryo_steps += 50;
                        log::warn!(
                            "Extending cryo phase: only {} candidates (need {})",
                            n_active,
                            self.config.min_candidates_for_ramp
                        );
                    }
                }
            }

            ExplorationPhase::ThermalRamp => {
                if self.phase_step >= self.config.ramp_steps {
                    let has_validated = self.candidates.iter().any(|c| {
                        c.is_active
                            && c.confidence >= self.config.min_confidence_for_dig
                            && c.validation_count >= 2
                    });

                    if has_validated {
                        self.transition_to_dig();
                    } else {
                        // Proceed anyway but log warning
                        log::warn!("No validated candidates - proceeding to dig phase anyway");
                        self.transition_to_dig();
                    }
                }
            }

            ExplorationPhase::FocusedDig => {
                // Check for site confirmation
                if self.peak_confidence >= self.config.site_confirmation_threshold {
                    self.site_confirmed = true;
                }

                if self.phase_step >= self.config.dig_steps {
                    self.dig_complete = true;
                }
            }
        }
    }

    fn transition_to_ramp(&mut self) {
        self.current_phase = ExplorationPhase::ThermalRamp;
        self.phase_step = 0;
        self.spikes_this_phase = 0;
        self.cryo_complete = true;

        // Set up temperature ramp
        self.temp_ramp_rate = (self.config.ramp_end_temp - self.config.ramp_start_temp)
            / self.config.ramp_steps as f32;
        self.target_temp = self.config.ramp_end_temp;

        // Adjust probe settings
        self.current_uv_intensity = self.config.ramp_uv_intensity;
        self.exploration_radius = self.config.ramp_exploration_radius;

        // Focus on best candidate
        if let Some(idx) = self.best_candidate_idx {
            self.exploration_center = self.candidates[idx].center;
        }

        self.phase_transitions.push((
            self.total_step,
            ExplorationPhase::ThermalRamp,
            format!(
                "Cryo complete: {} candidates, best conf={:.2}",
                self.candidates.len(),
                self.peak_confidence
            ),
        ));

        log::info!(
            "🌡️ PHASE TRANSITION: Cryo → Ramp | {} candidates | Peak confidence: {:.2}",
            self.candidates.len(),
            self.peak_confidence
        );
    }

    fn transition_to_dig(&mut self) {
        self.current_phase = ExplorationPhase::FocusedDig;
        self.phase_step = 0;
        self.spikes_this_phase = 0;
        self.ramp_complete = true;

        // Physiological temperature
        self.current_temp = self.config.physiological_temp;
        self.temp_ramp_rate = 0.0;

        // Focused probe settings
        self.current_uv_intensity = self.config.dig_uv_intensity;
        self.exploration_radius = self.config.dig_exploration_radius;

        // Prune inactive candidates
        let active_count = self.candidates.iter().filter(|c| c.is_active).count();

        self.phase_transitions.push((
            self.total_step,
            ExplorationPhase::FocusedDig,
            format!(
                "Ramp complete: {} active candidates, temp={:.0}K",
                active_count, self.current_temp
            ),
        ));

        log::info!(
            "🎯 PHASE TRANSITION: Ramp → Dig | {} active candidates | Focusing on best site",
            active_count
        );
    }

    /// Record a spike response
    pub fn record_spike(&mut self, position: [f32; 3], intensity: f32, aromatic_idx: usize) {
        if intensity < self.config.hot_spot_threshold {
            return;
        }

        self.total_spikes += 1;
        self.spikes_this_phase += 1;

        match self.current_phase {
            ExplorationPhase::CryoBurst => {
                self.record_cryo_spike(position, intensity, aromatic_idx);
            }
            ExplorationPhase::ThermalRamp => {
                self.record_ramp_spike(position, intensity);
            }
            ExplorationPhase::FocusedDig => {
                self.record_dig_spike(position, intensity);
            }
        }
    }

    fn record_cryo_spike(&mut self, position: [f32; 3], intensity: f32, aromatic_idx: usize) {
        // Check if near existing candidate
        let merge_distance = 5.0; // Angstroms

        let mut matched_idx = None;
        for (i, cand) in self.candidates.iter().enumerate() {
            if cand.is_active && cand.is_near(&position, merge_distance) {
                matched_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = matched_idx {
            // Update existing candidate
            let cand = &mut self.candidates[idx];
            cand.update_center(&position, intensity);
            cand.confidence = (cand.confidence + 0.05).min(1.0);

            if !cand.contributing_aromatics.contains(&aromatic_idx) {
                cand.contributing_aromatics.push(aromatic_idx);
                cand.aromatic_count += 1;
            }

            // Update best
            if cand.confidence > self.peak_confidence {
                self.peak_confidence = cand.confidence;
                self.best_candidate_idx = Some(idx);
                self.best_site_location = cand.center;
            }
        } else if self.candidates.len() < 32 {
            // Create new candidate
            let mut cand =
                HotSpotCandidate::new(self.candidates.len(), position, ExplorationPhase::CryoBurst);
            cand.spike_density = intensity;
            cand.contributing_aromatics.push(aromatic_idx);
            cand.aromatic_count = 1;

            self.candidates.push(cand);
        }
    }

    fn record_ramp_spike(&mut self, position: [f32; 3], intensity: f32) {
        // Find nearest candidate and validate
        let mut best_dist = f32::MAX;
        let mut nearest_idx = None;

        for (i, cand) in self.candidates.iter().enumerate() {
            if !cand.is_active {
                continue;
            }

            let dx = position[0] - cand.center[0];
            let dy = position[1] - cand.center[1];
            let dz = position[2] - cand.center[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            if dist < best_dist && dist < self.exploration_radius {
                best_dist = dist;
                nearest_idx = Some(i);
            }
        }

        if let Some(idx) = nearest_idx {
            // Validate - real sites respond MORE as temp increases
            let temp_progress = (self.current_temp - self.config.ramp_start_temp)
                / (self.config.ramp_end_temp - self.config.ramp_start_temp);
            let thermal_factor = 1.0 + temp_progress; // Higher bonus at high temp

            let cand = &mut self.candidates[idx];
            cand.validation_count += 1;
            cand.confidence = (cand.confidence + 0.1 * thermal_factor).min(1.0);
            cand.confidence_history.push(cand.confidence);

            if cand.confidence > self.peak_confidence {
                self.peak_confidence = cand.confidence;
                self.best_candidate_idx = Some(idx);
                self.best_site_location = cand.center;
            }
        }

        // Decay candidates that aren't responding
        for cand in &mut self.candidates {
            if cand.is_active && !cand.is_near(&position, self.exploration_radius * 2.0) {
                cand.confidence = (cand.confidence - 0.01).max(0.0);
                if cand.confidence < 0.1 {
                    cand.is_active = false;
                }
            }
        }
    }

    fn record_dig_spike(&mut self, position: [f32; 3], intensity: f32) {
        if let Some(idx) = self.best_candidate_idx {
            let cand = &mut self.candidates[idx];

            if cand.is_near(&position, self.exploration_radius) {
                cand.spike_density += intensity;
                cand.validation_count += 1;
                cand.confidence = (cand.confidence + 0.02).min(1.0);
                cand.confidence_history.push(cand.confidence);

                if cand.confidence > self.peak_confidence {
                    self.peak_confidence = cand.confidence;
                    self.best_site_location = cand.center;
                }

                // Check for confirmation
                if cand.confidence >= self.config.site_confirmation_threshold
                    && cand.validation_count >= 10
                {
                    self.site_confirmed = true;
                }
            }
        }
    }

    /// Get probe intensity for an aromatic
    pub fn get_probe_intensity(
        &self,
        aromatic_idx: usize,
        aromatic_center: [f32; 3],
        n_aromatics: usize,
    ) -> f32 {
        match self.current_phase {
            ExplorationPhase::CryoBurst => {
                // Sweep through all aromatics
                let sweep_period = self.config.cryo_steps.max(1) / n_aromatics.max(1);
                let current_target = (self.phase_step / sweep_period.max(1)) % n_aromatics;

                if aromatic_idx == current_target {
                    self.current_uv_intensity
                } else if aromatic_idx == (current_target + n_aromatics - 1) % n_aromatics
                    || aromatic_idx == (current_target + 1) % n_aromatics
                {
                    self.current_uv_intensity * 0.3
                } else {
                    0.0
                }
            }

            ExplorationPhase::ThermalRamp => {
                // Focus on candidates
                let mut intensity = 0.0f32;

                for cand in &self.candidates {
                    if !cand.is_active {
                        continue;
                    }

                    let dx = aromatic_center[0] - cand.center[0];
                    let dy = aromatic_center[1] - cand.center[1];
                    let dz = aromatic_center[2] - cand.center[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    if dist < self.exploration_radius {
                        let proximity_weight = 1.0 - dist / self.exploration_radius;
                        let conf_weight = cand.confidence;
                        intensity = intensity
                            .max(self.current_uv_intensity * proximity_weight * conf_weight);
                    }
                }

                intensity
            }

            ExplorationPhase::FocusedDig => {
                // Gaussian focus on best candidate
                if let Some(idx) = self.best_candidate_idx {
                    let cand = &self.candidates[idx];

                    let dx = aromatic_center[0] - cand.center[0];
                    let dy = aromatic_center[1] - cand.center[1];
                    let dz = aromatic_center[2] - cand.center[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                    if dist < self.exploration_radius {
                        let sigma = self.exploration_radius / 2.0;
                        let weight = (-0.5 * dist * dist / (sigma * sigma)).exp();
                        self.current_uv_intensity * weight
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
        }
    }

    /// Get Langevin damping coefficient
    pub fn get_langevin_gamma(&self) -> f32 {
        match self.current_phase {
            ExplorationPhase::CryoBurst => 50.0, // Heavy damping
            ExplorationPhase::ThermalRamp => {
                let progress = self.phase_step as f32 / self.config.ramp_steps as f32;
                50.0 * (1.0 - progress) + 5.0 * progress
            }
            ExplorationPhase::FocusedDig => 5.0, // Normal
        }
    }

    /// Get current status summary
    pub fn status_summary(&self) -> String {
        let phase_name = self.current_phase.name();
        let phase_emoji = self.current_phase.emoji();

        let n_active = self.candidates.iter().filter(|c| c.is_active).count();

        format!(
            "{} {} | Step {}/{} | T={:.0}K | {} candidates | Conf={:.2}{}",
            phase_emoji,
            phase_name,
            self.total_step,
            self.config.total_steps(),
            self.current_temp,
            n_active,
            self.peak_confidence,
            if self.site_confirmed {
                " ✓ CONFIRMED"
            } else {
                ""
            }
        )
    }

    /// Get best candidate if any
    pub fn get_best_candidate(&self) -> Option<&HotSpotCandidate> {
        self.best_candidate_idx.map(|idx| &self.candidates[idx])
    }

    /// Check if protocol is complete
    pub fn is_complete(&self) -> bool {
        self.dig_complete || self.site_confirmed
    }

    /// Get active candidates sorted by confidence
    pub fn get_active_candidates(&self) -> Vec<&HotSpotCandidate> {
        let mut active: Vec<_> = self.candidates.iter().filter(|c| c.is_active).collect();
        active.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        active
    }

    /// Generate final report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("╔══════════════════════════════════════════════════════════════════╗\n");
        report.push_str("║           PRISM-NHS ADAPTIVE PROTOCOL REPORT                     ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        report.push_str(&format!(
            "║ Total Steps: {:>10}                                         ║\n",
            self.total_step
        ));
        report.push_str(&format!(
            "║ Total Spikes: {:>9}                                         ║\n",
            self.total_spikes
        ));
        report.push_str(&format!(
            "║ Peak Confidence: {:>6.2}                                         ║\n",
            self.peak_confidence
        ));
        report.push_str(&format!(
            "║ Site Confirmed: {:>7}                                         ║\n",
            if self.site_confirmed { "YES ✓" } else { "NO" }
        ));

        report.push_str("╠══════════════════════════════════════════════════════════════════╣\n");
        report.push_str("║ PHASE TRANSITIONS                                                ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        for (step, phase, note) in &self.phase_transitions {
            report.push_str(&format!(
                "║ Step {:>6}: {} {} - {}                   \n",
                step,
                phase.emoji(),
                phase.name(),
                note
            ));
        }

        report.push_str("╠══════════════════════════════════════════════════════════════════╣\n");
        report.push_str("║ TOP CANDIDATES                                                   ║\n");
        report.push_str("╠══════════════════════════════════════════════════════════════════╣\n");

        let active = self.get_active_candidates();
        for (i, cand) in active.iter().take(5).enumerate() {
            let status = if Some(cand.id) == self.best_candidate_idx {
                "★ BEST"
            } else {
                ""
            };
            report.push_str(&format!(
                "║ #{}: ({:>6.1}, {:>6.1}, {:>6.1}) conf={:.2} val={:>2} {}        \n",
                i + 1,
                cand.center[0],
                cand.center[1],
                cand.center[2],
                cand.confidence,
                cand.validation_count,
                status
            ));
        }

        if let Some(best) = self.get_best_candidate() {
            report
                .push_str("╠══════════════════════════════════════════════════════════════════╣\n");
            report
                .push_str("║ BEST SITE DETAILS                                                ║\n");
            report
                .push_str("╠══════════════════════════════════════════════════════════════════╣\n");
            report.push_str(&format!(
                "║ Center: ({:.2}, {:.2}, {:.2}) Å                                 \n",
                best.center[0], best.center[1], best.center[2]
            ));
            report.push_str(&format!(
                "║ Confidence: {:.3}                                              \n",
                best.confidence
            ));
            report.push_str(&format!(
                "║ Resonance: {:.2} THz                                           \n",
                best.resonance_frequency
            ));
            report.push_str(&format!(
                "║ Validations: {}                                                \n",
                best.validation_count
            ));
            report.push_str(&format!(
                "║ Contributing aromatics: {:?}                                   \n",
                best.contributing_aromatics
            ));
        }

        report.push_str("╚══════════════════════════════════════════════════════════════════╝\n");

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_phases() {
        let config = AdaptiveProtocolConfig::quick_validation();
        let mut state = AdaptiveProtocolState::new(config);

        assert_eq!(state.current_phase, ExplorationPhase::CryoBurst);
        assert_eq!(state.current_temp, 80.0);

        // Add some candidates
        for i in 0..5 {
            state.record_spike([i as f32 * 10.0, 0.0, 0.0], 0.5, i);
        }

        // Run through cryo phase
        for _ in 0..60 {
            state.step();
        }

        assert!(state.candidates.len() >= 3);
    }

    #[test]
    fn test_candidate_tracking() {
        let config = AdaptiveProtocolConfig::default();
        let mut state = AdaptiveProtocolState::new(config);

        // Add spikes at same location - should merge
        state.record_spike([10.0, 10.0, 10.0], 0.5, 0);
        state.record_spike([11.0, 10.0, 10.0], 0.5, 0); // Within 5A
        state.record_spike([12.0, 10.0, 10.0], 0.5, 1); // Within 5A

        assert_eq!(state.candidates.len(), 1);
        assert!(state.candidates[0].confidence > 0.1);

        // Add spike far away - should create new candidate
        state.record_spike([50.0, 50.0, 50.0], 0.5, 2);
        assert_eq!(state.candidates.len(), 2);
    }
}
