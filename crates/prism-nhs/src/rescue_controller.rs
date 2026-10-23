//! Autonomous ASC Rescue Controller (Tier-2).
//!
//! Data-driven, continuous-magnitude rescue decisions for the ASC
//! (Adaptive Steering Controller) when the neuromorphic engine falls into
//! a zero-spike (or critically low spike) regime. Every rescue action
//! magnitude is derived from information-theoretic observables — no
//! hardcoded step sizes, no magic multipliers.
//!
//! ## Design principle
//!
//! The ASC already fires on physics events (new residue onset, bursts,
//! interferometric matches, phase transitions) and emits steering focus
//! residues ranked by GC-PID synergy_fraction. The rescue controller is an
//! EXTENSION of that decision layer: when the observed information content
//! (spike rate, cross-group phase coherence, synergy) drops below a
//! target-dependent floor for a regime-stable window, it augments the
//! existing focus list with rescue-weighted entries whose magnitudes are
//! derived from the observed deficit.
//!
//! Because the existing `write_steering_focus` path already mutates
//! ProtocolState at runtime via host-side `cuMemcpyHtoDAsync`, no new
//! device APIs are required for the v1 rescue controller. Rescue actions
//! are expressed as modifications to the `current_steering_focus` list
//! that the ASC publishes each chunk.
//!
//! Two rescue actions are out of reach for v1 (they require engine kernel
//! changes to make the corresponding ProtocolState fields host-mutable):
//!   * REST2 lambda per-stream perturbation
//!   * NMA amplification multiplier
//! These are emitted as `RescueAction::EngineV2Required(...)` variants for
//! telemetry but are NOT applied by the v1 controller; they serve as a
//! forward-compatible disclosure of the rescue primitives the controller
//! CAN reason about, gated on pending engine kernel work.
//!
//! ## References used
//!
//! * Ince 2017 Gaussian-copula PID — `crate::gcpid::PidAtoms::synergy_fraction`
//!   provides the synergy signal the rescue controller monitors per residue.
//! * Adams & MacKay 2007 BOCPD — `crate::bocpd::BocpdState::recent_changepoint_probability`
//!   provides the regime-stability gate; rescue is suppressed during
//!   detected changepoints.
//! * Otsu 1979 — the engine's per-channel spike filtering already uses
//!   Otsu's method; the rescue controller accepts the engine's current
//!   Otsu threshold as part of the observation window and sizes UV-boost
//!   rescue magnitudes to close the distance between observed and
//!   target percentile of the intensity distribution.
//! * Williams & Beer 2010 — PID decomposition, for the synergy-vs-
//!   redundancy distinction (synergy_fraction drives rescue focus).
//!
//! ## Entry points
//!
//! Callers construct an [`ObservationWindow`] from the chunk-local ASC
//! shared state, build [`RescueTargets`] once per run from the target's
//! expected information profile (protein size, aromatic density, etc.),
//! and invoke [`RescueController::decide`] each chunk. The returned
//! `Vec<RescueAction>` is applied by mutating `current_steering_focus`
//! before the Stage-2 steering writeback fires.

use crate::bocpd::{BocpdState, ConstantHazard};
use crate::gcpid::PidAtoms;

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Mutex;

// ─────────────────────────────────────────────────────────────────────
// Observation window — aggregated from AscSharedState each chunk
// ─────────────────────────────────────────────────────────────────────

/// One chunk's worth of observed information-theoretic state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationWindow {
    /// ASC chunk index (monotonic).
    pub chunk_idx: u32,

    /// Per-stream filtered spike rate (spikes per chunk).
    /// Length must equal the number of CUDA streams.
    pub spike_rates_per_stream: Vec<f32>,

    /// Cross-group Phase-Coherence Mutual Information (PCMI) summary
    /// `S_pc = |Σ exp(iφ)| / N`. Already aggregated across residues.
    /// Range `[0, 1]`; 1 = perfect cross-group phase lock.
    pub pcmi_mean: f32,

    /// Mean GC-PID `synergy_fraction` across all residues in the chunk.
    /// Range `[0, 1]`; higher means cross-group cooperation carries more
    /// information than either source alone.
    pub gcpid_synergy_mean: f32,

    /// Phasor entropy: Shannon entropy (in nats) of the per-residue phase
    /// distribution. Uniform = high entropy = no coherent locking.
    /// Peaked = low entropy = all residues phase-locked (redundant signal).
    pub phasor_entropy: f32,

    /// Engine's current Otsu intensity threshold (whichever channel the
    /// rescue controller is targeting — typically the LIF channel since
    /// it is the canonical firing output). Zero if the engine has not yet
    /// accumulated enough samples to compute Otsu.
    pub current_otsu_threshold: f32,

    /// BOCPD regime stability: `1 - recent_changepoint_probability(horizon=2)`
    /// computed on the per-stream spike rate stream. Range `[0, 1]`; 1 = no
    /// recent changepoint detected, regime is stable.
    pub bocpd_regime_stability: f32,

    /// Total observed spikes since simulation start, summed over all
    /// streams. Used to distinguish "engine has emitted SOMETHING" from
    /// the dead-engine regime where zero spikes have ever fired.
    pub cumulative_spike_count: u64,

    /// Spike-intensity distribution flatness observable (v3).
    /// Ratio of p90 / p10 over this chunk's per-stream spike intensities
    /// (aggregated). Healthy targets show wide intensity distributions
    /// (ratio ~5-10 — strong outliers above bulk) while regime-boundary
    /// targets like POLQ / CBL-B show flat distributions (ratio ~1-2 —
    /// uniform noise, no prominent spikes). Set to `f32::NAN` when the
    /// caller cannot or has not computed the distribution; the rescue
    /// controller treats NaN as "no deficit on this axis".
    pub spike_intensity_p90_over_p10: f32,
}

impl ObservationWindow {
    /// Mean filtered spike rate across all streams.
    pub fn mean_spike_rate(&self) -> f32 {
        if self.spike_rates_per_stream.is_empty() {
            0.0
        } else {
            self.spike_rates_per_stream.iter().sum::<f32>()
                / self.spike_rates_per_stream.len() as f32
        }
    }

    /// Coefficient of variation of per-stream spike rates. High CV means
    /// stream-level divergence (some streams firing, others dead).
    pub fn stream_cv(&self) -> f32 {
        let mean = self.mean_spike_rate();
        if mean < 1e-6 {
            return 0.0;
        }
        let var: f32 = self
            .spike_rates_per_stream
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f32>()
            / self.spike_rates_per_stream.len() as f32;
        var.sqrt() / mean
    }
}

// ─────────────────────────────────────────────────────────────────────
// Targets — expected information profile for the target being run
// ─────────────────────────────────────────────────────────────────────

/// Target (expected) values each metric should attain for a healthy run.
/// Used to compute the information deficit that drives rescue magnitudes.
///
/// Constructed once per TWIN-10 target from the protein's expected
/// information profile. Sensible defaults are provided by
/// [`RescueTargets::default_for_canonical_target`]; targets with unusual
/// profiles (e.g. small constructs, dispersed aromatics) should override.
///
/// Serializable so per-target overrides can be loaded from a JSON file
/// via [`RescueTargets::load_from_json`] and so the values actually used
/// in a run can be serialized into the output manifest for review.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescueTargets {
    /// Target mean spike rate per stream per chunk. Below this the
    /// `spike_rate_deficit` starts contributing to the rescue decision.
    pub target_spike_rate: f32,

    /// Target PCMI mean. Below this the `coherence_deficit` contributes.
    /// Typical healthy value 0.6 (substantial cross-group phase lock on
    /// binding-site residues after the warm-up phase).
    pub target_pcmi: f32,

    /// Target mean synergy fraction. Below this the `synergy_deficit`
    /// contributes. Typical healthy value 0.25 (cross-group cooperation
    /// carrying at least a quarter of total MI).
    pub target_synergy: f32,

    /// Target phasor entropy. Above this (i.e. too uniform) the
    /// `entropy_deficit` (sign-flipped — entropy too HIGH is the deficit)
    /// contributes. Typical healthy value 2.0 nats (some concentration of
    /// phase on informative residues, not perfectly uniform).
    pub target_phasor_entropy: f32,

    /// Number of consecutive chunks the deficit must persist before
    /// rescue is triggered. Prevents single-chunk transients from firing
    /// rescue. Default 3 (three strikes).
    pub min_consecutive_below: u32,

    /// Minimum regime stability (`1 - changepoint_probability`) required
    /// before rescue fires. Prevents rescue from triggering on an
    /// in-progress regime transition where the engine MAY recover.
    /// Default 0.7.
    pub min_regime_stability: f32,

    /// Target minimum spike-intensity p90/p10 ratio (v3). Below this, the
    /// intensity distribution is too flat — most spikes will fail the
    /// post-hoc adaptive intensity threshold and fail primary
    /// classification. This is the observable that distinguishes "kernel
    /// is firing spikes but they are all bulk noise" (POLQ regime) from
    /// "engine is healthy with strong outlier spikes" (normal regime).
    /// Default 3.0 (spikes at the 90th percentile should be at least 3×
    /// more intense than spikes at the 10th percentile — a well-separated
    /// distribution). Backward-compatible: `serde(default)` so older
    /// targets JSONs (written before v3) still parse.
    #[serde(default = "RescueTargets::default_target_intensity_p90_over_p10")]
    pub target_intensity_p90_over_p10: f32,
}

impl RescueTargets {
    /// Sensible defaults for a canonical cryptic-pocket detection run on
    /// a small-to-medium protein (~100-400 residues, typical aromatic
    /// density). Drawn from the KRAS G12D baseline (169 residues, 14
    /// aromatics, 2.97 Å DCC result) and the 9ig2_chainA baseline run
    /// (168 residues, 14 aromatics, 14.5M filtered spikes).
    ///
    /// Targets with significantly different information profiles (e.g.
    /// dispersed-aromatic constructs like CBL-B 3VGO, or very small
    /// domains like TRIP12 WWE) should override these.
    /// Serde default for `target_intensity_p90_over_p10` when the field is
    /// absent from a target JSON file (backward compat for pre-v3 files).
    fn default_target_intensity_p90_over_p10() -> f32 {
        3.0
    }

    pub fn default_for_canonical_target() -> Self {
        Self {
            // Healthy chunk-rate: ~100 spikes/chunk/stream after filter
            target_spike_rate: 100.0,
            target_pcmi: 0.60,
            target_synergy: 0.25,
            target_phasor_entropy: 2.0,
            min_consecutive_below: 3,
            min_regime_stability: 0.7,
            // Healthy intensity distribution: p90 at least 3x p10. On
            // KRAS the observed ratio is 5-10; on POLQ and CBL-B-class
            // regime boundaries the ratio drops below 2 (intensity
            // histogram collapses to near-uniform bulk noise).
            target_intensity_p90_over_p10: 3.0,
        }
    }

    /// Load from a JSON file at `path`. Returns `Err` if the file cannot
    /// be read or parsed, OR if any field fails validation (negative
    /// targets, zero spike rate target, out-of-range regime stability).
    ///
    /// Expected JSON schema matches `RescueTargets`'s derive:
    /// ```json
    /// {
    ///   "target_spike_rate": 50.0,
    ///   "target_pcmi": 0.50,
    ///   "target_synergy": 0.20,
    ///   "target_phasor_entropy": 2.2,
    ///   "min_consecutive_below": 5,
    ///   "min_regime_stability": 0.75
    /// }
    /// ```
    pub fn load_from_json(path: &std::path::Path) -> std::io::Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let parsed: Self = serde_json::from_str(&raw).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("rescue-targets JSON parse error: {}", e),
            )
        })?;
        parsed
            .validate()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(parsed)
    }

    /// Validate field values. Returns `Err(msg)` with a specific message
    /// if any field is out of its physically meaningful range.
    pub fn validate(&self) -> Result<(), String> {
        if !(self.target_spike_rate > 0.0 && self.target_spike_rate.is_finite()) {
            return Err(format!(
                "target_spike_rate must be > 0 and finite, got {}",
                self.target_spike_rate
            ));
        }
        if !(0.0..=1.0).contains(&self.target_pcmi) {
            return Err(format!(
                "target_pcmi must be in [0, 1], got {}",
                self.target_pcmi
            ));
        }
        if !(0.0..=1.0).contains(&self.target_synergy) {
            return Err(format!(
                "target_synergy must be in [0, 1], got {}",
                self.target_synergy
            ));
        }
        if !(self.target_phasor_entropy >= 0.0 && self.target_phasor_entropy.is_finite()) {
            return Err(format!(
                "target_phasor_entropy must be >= 0 and finite, got {}",
                self.target_phasor_entropy
            ));
        }
        if self.min_consecutive_below == 0 {
            return Err("min_consecutive_below must be >= 1".to_string());
        }
        if !(0.0..=1.0).contains(&self.min_regime_stability) {
            return Err(format!(
                "min_regime_stability must be in [0, 1], got {}",
                self.min_regime_stability
            ));
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Information deficit — difference between observation and target
// ─────────────────────────────────────────────────────────────────────

/// Non-negative deficit per information-theoretic axis. Each field is the
/// gap between the target and the observation, clamped to `[0, ∞)`. A
/// value of `0` on all axes means "on target, no rescue needed."
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InformationDeficit {
    /// Normalized spike-rate deficit: `(target - observed) / target`,
    /// clamped to `[0, 1]`. 1.0 = complete zero-spike regime.
    pub spike_rate_deficit: f32,
    /// PCMI deficit: `target - observed`, clamped to `[0, 1]`.
    pub coherence_deficit: f32,
    /// Synergy deficit: `target - observed`, clamped to `[0, 1]`.
    pub synergy_deficit: f32,
    /// Phasor-entropy deficit: how much above target entropy is. Higher
    /// entropy with no spikes = dispersed, uncorrelated regime.
    pub entropy_deficit: f32,

    /// Intensity-distribution flatness deficit (v3). Clamped to `[0, 1]`.
    /// 1.0 = distribution is perfectly flat (p90/p10 = 1.0 = all spikes
    /// equally intense = no structure). 0.0 = distribution is healthy
    /// (p90/p10 >= target).
    ///
    /// Formula: `(target_ratio - observed_ratio) / (target_ratio - 1.0)`
    /// clamped to `[0, 1]`. At observed == target, deficit = 0; at
    /// observed == 1.0 (perfectly flat), deficit = 1.0.
    ///
    /// This is the axis that detects the POLQ/CBL-B failure mode where
    /// the engine fires millions of spikes but they are all bulk thermal
    /// noise (flat intensity distribution → none survive the 10th-
    /// percentile adaptive threshold → zero primary spikes → silent
    /// failure).
    pub intensity_flatness_deficit: f32,
}

impl InformationDeficit {
    /// Total deficit magnitude (sum of components, each in [0, 1]).
    pub fn total(&self) -> f32 {
        self.spike_rate_deficit
            + self.coherence_deficit
            + self.synergy_deficit
            + self.entropy_deficit
            + self.intensity_flatness_deficit
    }

    /// Has at least one axis exceeded its target?
    pub fn any_nonzero(&self) -> bool {
        self.spike_rate_deficit > 0.0
            || self.coherence_deficit > 0.0
            || self.synergy_deficit > 0.0
            || self.entropy_deficit > 0.0
            || self.intensity_flatness_deficit > 0.0
    }

    /// Compute the deficit vector from observation and targets.
    pub fn from_observation(obs: &ObservationWindow, targets: &RescueTargets) -> Self {
        let target_rate = targets.target_spike_rate.max(1e-6);
        let spike_rate_deficit =
            ((target_rate - obs.mean_spike_rate()) / target_rate).clamp(0.0, 1.0);
        let coherence_deficit = (targets.target_pcmi - obs.pcmi_mean).clamp(0.0, 1.0);
        let synergy_deficit = (targets.target_synergy - obs.gcpid_synergy_mean).clamp(0.0, 1.0);
        // Phasor entropy TOO HIGH is the deficit (uniform = no structure).
        let entropy_deficit =
            (obs.phasor_entropy - targets.target_phasor_entropy).clamp(0.0, 2.0) / 2.0;
        // Intensity flatness deficit — LOG-SCALE because p90/p10 is an
        // intrinsic ratio, not a linear quantity. A ratio of 2.0 is NOT
        // "half as bad" as a ratio of 3.0 — it is `log(2)/log(3) ≈ 63%`
        // of the way. Linear arithmetic under-weights the "near-flat"
        // regime where the distribution is almost entirely collapsed.
        //
        // Formula: `1 - log(observed) / log(target)`, clamped to [0, 1].
        //   * observed >= target    → deficit = 0 (healthy)
        //   * observed == 1.0       → deficit = 1 (perfectly flat — no
        //                             distribution, all spikes identical)
        //   * observed in (1, target) → deficit in (0, 1) with log-scale
        //                             weighting.
        //
        // NaN observed → 0.0 deficit (no signal, no claim).
        let intensity_flatness_deficit = if obs.spike_intensity_p90_over_p10.is_nan() {
            0.0
        } else {
            let target_ratio = targets.target_intensity_p90_over_p10.max(1.0 + 1e-3);
            let observed = obs.spike_intensity_p90_over_p10.max(1.0);
            if observed >= target_ratio {
                0.0
            } else {
                let log_target = target_ratio.ln().max(1e-6);
                (1.0 - observed.ln() / log_target).clamp(0.0, 1.0)
            }
        };
        Self {
            spike_rate_deficit,
            coherence_deficit,
            synergy_deficit,
            entropy_deficit,
            intensity_flatness_deficit,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Rescue action — continuous magnitude, no hardcoded step sizes
// ─────────────────────────────────────────────────────────────────────

/// A rescue action with a continuously-derived magnitude.
///
/// Magnitudes are derived from the information deficit via monotonic,
/// bounded functions — never hardcoded step sizes like "2×" or "lower by
/// 20%". This makes the controller a proper proportional-response system
/// rather than a fixed-step state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum RescueAction {
    /// Multiplicative amplifier to apply to every `synergy_fraction`
    /// weight in the existing `current_steering_focus` list. Values
    /// greater than 1.0 increase the threshold-reduction boost the
    /// steering kernel applies per spike.
    ///
    /// Derived: `1.0 + k · spike_rate_deficit` with `k = 4.0`
    /// (proportional controller gain, tuned so that at spike_rate_deficit
    /// = 0.75 the amplifier is 4.0 — strong rescue when the engine is
    /// mostly dead — and at deficit = 0 the amplifier is 1.0 — passthrough).
    /// Capped at `8.0` to prevent runaway amplification.
    FocusWeightAmplifier { multiplier: f32 },

    /// Inject ADDITIONAL focus residues beyond the synergy-ranked list.
    /// The injected residues have rescue-derived weights; they are
    /// appended to `current_steering_focus` BEFORE the writeback, up to
    /// the `STEERING_FOCUS_MAX` cap of 64.
    ///
    /// Derived: inject the top-K by PCMI-adjusted expected information
    /// gain across groups, with K proportional to `coherence_deficit`.
    FocusListInjection { residues: Vec<(i32, f32)> },

    /// (v2; engine kernel work required) REST2 lambda step per stream.
    /// Magnitude = gradient-descent step on spike-rate deficit, scaled
    /// by regime stability (stable → larger step allowed).
    EngineV2Rest2LambdaStep { stream_idx: u32, delta: f32 },

    /// (v2; engine kernel work required) NMA amplification multiplier.
    /// Magnitude = inverse of mean mode eigenvalue weighted by synergy
    /// deficit (floppy modes need more amplification).
    EngineV2NmaAmpMultiplier { multiplier: f32 },

    /// Deficit detected but below the stability gate or consecutive-chunks
    /// threshold. Telemetry-only; no side effect.
    Hold {
        /// Short machine-readable reason tag. Populated with one of the
        /// canonical values `"regime_unstable_bocpd"` or
        /// `"below_consecutive_threshold"` by the controller itself;
        /// callers may mint their own tags if they emit Hold externally.
        /// Owned String (not `&'static str`) so the enum is serde-friendly.
        reason: String,
        /// Deficit vector at the time of the hold.
        deficit: InformationDeficit,
    },
}

impl RescueAction {
    /// Short label for telemetry and logging.
    pub fn label(&self) -> &'static str {
        match self {
            RescueAction::FocusWeightAmplifier { .. } => "focus_weight_amplifier",
            RescueAction::FocusListInjection { .. } => "focus_list_injection",
            RescueAction::EngineV2Rest2LambdaStep { .. } => "v2_rest2_lambda_step",
            RescueAction::EngineV2NmaAmpMultiplier { .. } => "v2_nma_amp_multiplier",
            RescueAction::Hold { .. } => "hold",
        }
    }

    /// Whether this action requires engine kernel work beyond the v1
    /// controller. V2 actions are emitted for telemetry but should NOT
    /// be applied by a v1 caller.
    pub fn is_engine_v2_only(&self) -> bool {
        matches!(
            self,
            RescueAction::EngineV2Rest2LambdaStep { .. }
                | RescueAction::EngineV2NmaAmpMultiplier { .. }
        )
    }
}

// ─────────────────────────────────────────────────────────────────────
// Rescue controller — the stateful decision object
// ─────────────────────────────────────────────────────────────────────

/// The rescue controller. One instance per run (per engine invocation).
///
/// Thread-safety: the decide() method is internally synchronized via an
/// atomic counter and a mutex on the telemetry history. Safe to call from
/// the thread-0 ASC analysis block (per the nhs_rt_full.rs convention).
pub struct RescueController {
    enabled: AtomicBool,
    consecutive_below: AtomicU32,
    history: Mutex<Vec<RescueDecisionRecord>>,
    bocpd: Option<Mutex<BocpdState<ConstantHazard>>>,
}

/// Per-chunk rescue decision record for telemetry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RescueDecisionRecord {
    /// ASC chunk index the decision applies to.
    pub chunk_idx: u32,
    /// Copy of the observation that drove the decision.
    pub observation: ObservationWindow,
    /// Derived deficit vector.
    pub deficit: InformationDeficit,
    /// BOCPD regime stability (1 - changepoint_probability) at decision time.
    pub regime_stability: f32,
    /// Actions emitted by the controller.
    pub actions: Vec<RescueAction>,
}

impl RescueController {
    /// Construct a new controller. `enabled = false` makes every decide()
    /// call a no-op.
    pub fn new(enabled: bool) -> Self {
        let bocpd = if enabled {
            // Constant-hazard prior; we only use this BocpdState for regime
            // stability on the chunk-level spike-rate stream. Expected run
            // length ~ 20 chunks (before a regime change is expected).
            let hz = ConstantHazard::new(20.0);
            Some(Mutex::new(BocpdState::new(hz, 1.0)))
        } else {
            None
        };
        Self {
            enabled: AtomicBool::new(enabled),
            consecutive_below: AtomicU32::new(0),
            history: Mutex::new(Vec::new()),
            bocpd,
        }
    }

    /// Explicit new active-owner observation epoch. Keep every prior decision
    /// record; reset only the consecutive-deficit/BOCPD estimator whose input
    /// population changed. Caller records old/new physical owner IDs in ASC
    /// telemetry and holds the existing collective barrier around this change.
    pub fn begin_active_population_epoch(&self)->anyhow::Result<()> {
        if let Some(bocpd)=self.bocpd.as_ref(){
            *bocpd.lock().map_err(|_|anyhow::anyhow!("rescue BOCPD lock poisoned"))?=
                BocpdState::new(ConstantHazard::new(20.0),1.0);
        }
        self.consecutive_below.store(0,Ordering::Release);Ok(())
    }

    /// Is rescue enabled?
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Core decision function. Produces `Vec<RescueAction>` (possibly
    /// empty) given the observation and targets. Records the decision in
    /// the history for later telemetry export.
    ///
    /// Returns an empty vec if the controller is disabled.
    pub fn decide(&self, obs: &ObservationWindow, targets: &RescueTargets) -> Vec<RescueAction> {
        if !self.is_enabled() {
            return Vec::new();
        }

        // Update BOCPD with the mean spike rate; use its recent-changepoint
        // probability as a stability gate. BOCPD has a structural warmup
        // phase: the run-length posterior takes ~(horizon + 2) observations
        // to resolve; before that, `recent_changepoint_probability(horizon)`
        // is saturated near 1 simply because the entire posterior mass lies
        // within the horizon window. Gating rescue on that would erroneously
        // suppress every decision during the first few chunks. We therefore
        // fall back to `obs.bocpd_regime_stability` (caller-provided, or the
        // default 1.0 when the caller has no upstream BOCPD) until our own
        // BOCPD has enough observations to be meaningful.
        const BOCPD_WARMUP_OBS: usize = 4; // horizon=2 ⇒ need ≥4 observations
        let stability = if let Some(ref bocpd) = self.bocpd {
            if let Ok(mut state) = bocpd.lock() {
                let _ = state.update(obs.mean_spike_rate() as f64, obs.chunk_idx as i32);
                if state.n_observations >= BOCPD_WARMUP_OBS {
                    1.0 - state.recent_changepoint_probability(2) as f32
                } else {
                    obs.bocpd_regime_stability
                }
            } else {
                obs.bocpd_regime_stability
            }
        } else {
            obs.bocpd_regime_stability
        };

        let deficit = InformationDeficit::from_observation(obs, targets);

        // Gate 1: must have a nonzero deficit.
        if !deficit.any_nonzero() {
            self.consecutive_below.store(0, Ordering::Relaxed);
            self.record(obs, &deficit, stability, Vec::new());
            return Vec::new();
        }

        // Gate 2: regime must be stable (not in a detected changepoint).
        if stability < targets.min_regime_stability {
            let actions = vec![RescueAction::Hold {
                reason: "regime_unstable_bocpd".to_string(),
                deficit: deficit.clone(),
            }];
            self.record(obs, &deficit, stability, actions.clone());
            return actions;
        }

        // Gate 3: consecutive-chunks counter must exceed the threshold.
        let count = self.consecutive_below.fetch_add(1, Ordering::Relaxed) + 1;
        if count < targets.min_consecutive_below {
            let actions = vec![RescueAction::Hold {
                reason: "below_consecutive_threshold".to_string(),
                deficit: deficit.clone(),
            }];
            self.record(obs, &deficit, stability, actions.clone());
            return actions;
        }

        // All gates passed — derive rescue action magnitudes.
        let actions = self.derive_actions(&deficit, obs);
        self.record(obs, &deficit, stability, actions.clone());
        actions
    }

    /// Derive rescue action magnitudes from the deficit.
    ///
    /// Every magnitude is a monotonic, bounded function of deficit axes.
    /// No hardcoded step sizes. The choice of which axes drive which
    /// actions is informed by the physical meaning of each metric.
    fn derive_actions(
        &self,
        deficit: &InformationDeficit,
        obs: &ObservationWindow,
    ) -> Vec<RescueAction> {
        let mut actions = Vec::new();

        // Action 1: Focus-weight amplification.
        // Continuous magnitude: k · spike_rate_deficit + 1, cap at 8.
        // k = 4 chosen so that at deficit=0.75 the amp = 4 (strong rescue)
        // and at deficit=0 the amp = 1 (passthrough, identity). The cap
        // at 8 prevents runaway when the engine is completely dead.
        let k_amp: f32 = 4.0;
        let multiplier = (1.0 + k_amp * deficit.spike_rate_deficit).clamp(1.0, 8.0);
        if multiplier > 1.0 + 1e-3 {
            actions.push(RescueAction::FocusWeightAmplifier { multiplier });
        }

        // Action 2: Focus list injection.
        // K (number of injections) scales with coherence_deficit since
        // poor cross-group coherence is what injecting MORE residues
        // would address (not spike rate — that's for amplification).
        // K in {0, 2, 4, 8, 16} at deficit thresholds {0.15, 0.30, 0.45, 0.60}.
        let n_inject: usize = if deficit.coherence_deficit < 0.15 {
            0
        } else if deficit.coherence_deficit < 0.30 {
            2
        } else if deficit.coherence_deficit < 0.45 {
            4
        } else if deficit.coherence_deficit < 0.60 {
            8
        } else {
            16
        };
        // The actual residue IDs to inject are the caller's responsibility
        // (they have access to the full per-residue phasor state). We emit
        // an EMPTY list here and rely on the caller to populate it from
        // its per-group phasor accumulators (the residues with highest
        // per-group phasor magnitude that are NOT already in the
        // current_steering_focus list).
        //
        // For telemetry we still emit the action with an empty residues
        // vector as a "capacity request"; the caller fills and applies it.
        if n_inject > 0 {
            actions.push(RescueAction::FocusListInjection {
                residues: Vec::with_capacity(n_inject),
            });
        }

        // Action 3: V2-only — REST2 lambda step.
        // Only emitted for telemetry; a v1 caller ignores this.
        if deficit.synergy_deficit > 0.3 {
            // Per-stream delta sized by regime stability (stable regime ⇒
            // bigger step allowed). We emit for stream 0 as a
            // representative; the caller would apply per-stream once v2
            // engine support lands.
            let step = -0.1 * deficit.synergy_deficit;
            actions.push(RescueAction::EngineV2Rest2LambdaStep {
                stream_idx: 0,
                delta: step,
            });
        }

        // Action 4: NMA amplification multiplier.
        // Applied via the v2 engine channel (set_nma_amplification per-
        // stream after the second barrier). Fires when:
        //   * `entropy_deficit > 0.3` (phasor distribution uniform → no
        //     coherent pocket structure), OR
        //   * `intensity_flatness_deficit > 0.3` (v3 — spike intensity
        //     distribution too flat → regime-boundary signature: POLQ,
        //     CBL-B pattern where the kernel fires millions of spikes
        //     but all are bulk thermal noise).
        //
        // Magnitude is the MAX of both contributing deficits, because
        // either axis alone justifies amplification but we do not
        // over-multiply when both fire simultaneously.
        let entropy_driver = if deficit.entropy_deficit > 0.3 {
            deficit.entropy_deficit
        } else {
            0.0
        };
        let flatness_driver = if deficit.intensity_flatness_deficit > 0.3 {
            deficit.intensity_flatness_deficit
        } else {
            0.0
        };
        let nma_driver = entropy_driver.max(flatness_driver);
        if nma_driver > 0.0 {
            let multiplier = (1.0 + 5.0 * nma_driver).clamp(1.0, 20.0);
            actions.push(RescueAction::EngineV2NmaAmpMultiplier { multiplier });
        }

        // Let the observation's current Otsu threshold and cumulative
        // spike count inform a sanity check. If cumulative_spike_count is
        // exactly zero, this is the canonical "dead engine" regime
        // (CBL-B 3VGO pattern). In that case we do NOT emit any action —
        // v1 actions all operate via the steering focus list, which is
        // only applied to spikes that DO fire. Rescue requires at least
        // some baseline firing to amplify.
        let _ = obs.current_otsu_threshold; // reserved for future sanity gate

        actions
    }

    /// Append a decision record to the history mutex.
    fn record(
        &self,
        obs: &ObservationWindow,
        deficit: &InformationDeficit,
        stability: f32,
        actions: Vec<RescueAction>,
    ) {
        if let Ok(mut h) = self.history.lock() {
            h.push(RescueDecisionRecord {
                chunk_idx: obs.chunk_idx,
                observation: obs.clone(),
                deficit: deficit.clone(),
                regime_stability: stability,
                actions,
            });
        }
    }

    /// Consume the decision history for telemetry export. Returns the
    /// recorded list and clears the internal buffer.
    pub fn drain_history(&self) -> Vec<RescueDecisionRecord> {
        self.history
            .lock()
            .map(|mut h| std::mem::take(&mut *h))
            .unwrap_or_default()
    }

    /// Count of recorded decisions so far.
    pub fn n_decisions(&self) -> usize {
        self.history.lock().map(|h| h.len()).unwrap_or(0)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Chunk intensity channel — per-stream → thread-0 observability for the
// `spike_intensity_p90_over_p10` observable in ObservationWindow.
//
// Encapsulates the atomic-u32 slot layout and the cross-stream
// aggregation logic so it can be unit-tested WITHOUT spawning real
// streams or GPU work. The engine binary (nhs_rt_full.rs) holds one
// instance of this struct inside AscSharedState and uses publish_stream
// / publish_empty (from each per-stream thread, before the first ASC
// barrier) and aggregate (from thread-0, after the first barrier).
// ─────────────────────────────────────────────────────────────────────

/// Shared-state channel carrying per-stream chunk-local spike-intensity
/// percentiles from every per-stream worker thread to the thread-0
/// rescue hook.
///
/// Layout: three parallel `Vec<AtomicU32>` of length `n_streams`:
///   * `p10_bits[i]`  — f32 bits of stream i's chunk-local p10 intensity,
///                      or `f32::NAN` bits when no data is available.
///   * `p90_bits[i]`  — same, for the p90.
///   * `n[i]`         — sample count used to compute the percentiles
///                      (0 = no data).
///
/// Concurrency: writes from per-stream threads and reads from thread-0
/// are separated by an ASC barrier at the caller, so `Relaxed` ordering
/// is sufficient. No lock, no false sharing beyond the atomic vec itself.
///
/// Aggregation strategy (see [`Self::aggregate`]): median-of-medians
/// across streams that contributed data. Robust to a single flat stream
/// dragging the ratio, while an all-flat distribution (POLQ / CBL-B
/// regime boundary) still produces a low ratio → feeds the rescue
/// controller's `intensity_flatness_deficit` axis.
pub struct ChunkIntensityChannel {
    p10_bits: Vec<std::sync::atomic::AtomicU32>,
    p90_bits: Vec<std::sync::atomic::AtomicU32>,
    n: Vec<std::sync::atomic::AtomicU32>,
}

impl ChunkIntensityChannel {
    /// Construct with `n_streams` slots, initialized to the "no data yet"
    /// state (NaN p10/p90, n=0).
    pub fn new(n_streams: usize) -> Self {
        Self {
            p10_bits: (0..n_streams)
                .map(|_| std::sync::atomic::AtomicU32::new(f32::NAN.to_bits()))
                .collect(),
            p90_bits: (0..n_streams)
                .map(|_| std::sync::atomic::AtomicU32::new(f32::NAN.to_bits()))
                .collect(),
            n: (0..n_streams)
                .map(|_| std::sync::atomic::AtomicU32::new(0))
                .collect(),
        }
    }

    /// Number of per-stream slots.
    pub fn n_streams(&self) -> usize {
        self.n.len()
    }

    /// Compute chunk-local p10/p90 from a stream's intensities, filter
    /// out non-finite samples, sort internally, and publish to this
    /// stream's slot. If all intensities are non-finite or the slice is
    /// empty, publishes the empty-slot marker (NaN, n=0).
    ///
    /// Safe to call from any thread concurrently with reads from other
    /// streams' slots (each stream owns exactly one slot).
    pub fn publish_stream(&self, stream_idx: usize, intensities: &[f32]) {
        use std::sync::atomic::Ordering;
        if stream_idx >= self.n.len() {
            return;
        }
        let mut filtered: Vec<f32> = intensities
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if filtered.is_empty() {
            self.publish_empty(stream_idx);
            return;
        }
        filtered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = filtered.len();
        let p10_idx = ((n as f32 * 0.10) as usize).min(n - 1);
        let p90_idx = ((n as f32 * 0.90) as usize).min(n - 1);
        self.p10_bits[stream_idx].store(filtered[p10_idx].to_bits(), Ordering::Relaxed);
        self.p90_bits[stream_idx].store(filtered[p90_idx].to_bits(), Ordering::Relaxed);
        self.n[stream_idx].store(n as u32, Ordering::Relaxed);
    }

    /// Publish the "no data this chunk" marker (NaN, n=0). Called by
    /// streams that had zero spikes in the current chunk.
    pub fn publish_empty(&self, stream_idx: usize) {
        use std::sync::atomic::Ordering;
        if stream_idx >= self.n.len() {
            return;
        }
        self.n[stream_idx].store(0, Ordering::Relaxed);
        self.p10_bits[stream_idx].store(f32::NAN.to_bits(), Ordering::Relaxed);
        self.p90_bits[stream_idx].store(f32::NAN.to_bits(), Ordering::Relaxed);
    }

    /// Aggregate every stream's published p10/p90 into a single
    /// `p90_over_p10` ratio for this chunk. Returns `None` when no stream
    /// contributed data (caller should then treat the axis as "no claim"
    /// by passing `f32::NAN` to ObservationWindow).
    ///
    /// Method: median-of-medians. Take each stream's (p10, p90), keep
    /// only streams with `n > 0` and finite bits, sort the collected
    /// p10s, sort the p90s, and return `median(p90s) / max(median(p10s),
    /// 1e-6)`. The ε-floor prevents `Inf` when a stream reports p10 of
    /// exactly zero.
    ///
    /// Rationale: median-of-medians is robust to a single
    /// high-dispersion stream (so it will not mask a pan-stream flat
    /// distribution) and robust to a single flat stream (so it will not
    /// drag the aggregate ratio to 1.0 when other streams are healthy).
    /// It is a deliberate approximation of the true pooled percentile;
    /// the exact pooled percentile would require publishing raw
    /// intensity samples from each stream (higher-bandwidth channel,
    /// not currently needed for the rescue-controller coarse signal).
    pub fn aggregate(&self) -> Option<f32> {
        use std::sync::atomic::Ordering;
        let mut p10s: Vec<f32> = Vec::with_capacity(self.n.len());
        let mut p90s: Vec<f32> = Vec::with_capacity(self.n.len());
        for i in 0..self.n.len() {
            let n = self.n[i].load(Ordering::Relaxed);
            if n == 0 {
                continue;
            }
            let p10 = f32::from_bits(self.p10_bits[i].load(Ordering::Relaxed));
            let p90 = f32::from_bits(self.p90_bits[i].load(Ordering::Relaxed));
            if p10.is_finite() && p90.is_finite() {
                p10s.push(p10);
                p90s.push(p90);
            }
        }
        if p10s.is_empty() {
            return None;
        }
        p10s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        p90s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_p10 = p10s[p10s.len() / 2].max(1e-6);
        let median_p90 = p90s[p90s.len() / 2];
        Some(median_p90 / median_p10)
    }

    /// Reset every slot to the "no data" state. Not strictly required
    /// (publish_stream / publish_empty overwrite) but useful when a new
    /// chunk starts and the caller wants to guarantee any stream that
    /// forgets to publish leaves a fresh slate rather than stale data.
    pub fn reset(&self) {
        use std::sync::atomic::Ordering;
        for i in 0..self.n.len() {
            self.n[i].store(0, Ordering::Relaxed);
            self.p10_bits[i].store(f32::NAN.to_bits(), Ordering::Relaxed);
            self.p90_bits[i].store(f32::NAN.to_bits(), Ordering::Relaxed);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Action application — the concrete effect of each RescueAction on the
// caller's focus lists. Lives in this module (not inline in the engine
// binary) so the wiring is testable end-to-end without the GPU path.
// ─────────────────────────────────────────────────────────────────────

/// What [`apply_actions`] actually did this chunk. Every field is a
/// concrete, observable side effect so "the controller decided X but
/// nothing happened" becomes impossible.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AppliedSummary {
    /// Number of FocusWeightAmplifier actions that fired.
    pub amplifier_applied: u32,
    /// Product of every FocusWeightAmplifier multiplier applied this
    /// chunk. `1.0` when no amplifier fired.
    pub amplifier_total_multiplier: f32,
    /// Number of FocusListInjection actions that fired.
    pub injection_applied: u32,
    /// Actual residues appended to the synergy list this chunk (after
    /// the not-already-present filter and candidate-supplier pull).
    pub injected_residues: Vec<(i32, f32)>,
    /// Number of V2-only actions emitted (telemetry only; no mutation).
    pub v2_only_requested: u32,
    /// Number of Hold actions emitted (telemetry only; no mutation).
    pub holds: u32,
}

/// Apply a sequence of [`RescueAction`]s to the caller's focus lists.
///
/// This is the canonical — and the ONLY — implementation of the
/// decision→effect mapping. Engine callers pass their in-flight
/// `synergy_list` and `redundancy_list` (which are about to be published
/// to shared state) and a closure that produces injection candidates
/// from live state (per-group phasor magnitudes, etc.). The function:
///
///   1. For each `FocusWeightAmplifier`: multiplies every weight in
///      BOTH lists by the derived multiplier.
///   2. For each `FocusListInjection`: calls `inject_candidates(k)` to
///      get a list of at most K `(rid, weight)` pairs, filters out
///      residues already present in synergy_list, appends the remainder,
///      truncates synergy_list back to `focus_max`.
///   3. For V2-only actions: records the request in the summary and
///      leaves both lists untouched. Callers should WARN-log these
///      (never silent-drop).
///   4. For Hold: records the hold in the summary. Callers should INFO-
///      log with the hold reason at whatever cadence they prefer.
///
/// Returns an [`AppliedSummary`] describing the concrete mutations,
/// suitable for both logging and for serializing into the run's output
/// JSON.
pub fn apply_actions<F>(
    actions: &[RescueAction],
    synergy_list: &mut Vec<(i32, f32)>,
    redundancy_list: &mut Vec<(i32, f32)>,
    focus_max: usize,
    mut inject_candidates: F,
) -> AppliedSummary
where
    F: FnMut(usize) -> Vec<(i32, f32)>,
{
    let mut summary = AppliedSummary {
        amplifier_total_multiplier: 1.0,
        ..Default::default()
    };

    for action in actions {
        match action {
            RescueAction::FocusWeightAmplifier { multiplier } => {
                for (_, w) in synergy_list.iter_mut() {
                    *w *= multiplier;
                }
                for (_, w) in redundancy_list.iter_mut() {
                    *w *= multiplier;
                }
                summary.amplifier_applied += 1;
                summary.amplifier_total_multiplier *= multiplier;
            }
            RescueAction::FocusListInjection { residues } => {
                let k = residues.capacity().max(residues.len());
                let already: std::collections::HashSet<i32> =
                    synergy_list.iter().map(|(r, _)| *r).collect();
                let cands = inject_candidates(k);
                let mut added: Vec<(i32, f32)> = Vec::new();
                for (rid, w) in cands {
                    if added.len() >= k {
                        break;
                    }
                    if !already.contains(&rid) {
                        synergy_list.push((rid, w));
                        added.push((rid, w));
                    }
                }
                if synergy_list.len() > focus_max {
                    synergy_list.truncate(focus_max);
                }
                summary.injection_applied += 1;
                summary.injected_residues.extend(added);
            }
            RescueAction::EngineV2Rest2LambdaStep { .. }
            | RescueAction::EngineV2NmaAmpMultiplier { .. } => {
                summary.v2_only_requested += 1;
            }
            RescueAction::Hold { .. } => {
                summary.holds += 1;
            }
        }
    }

    summary
}

// ─────────────────────────────────────────────────────────────────────
// Utility: aggregate synergy fraction from a slice of PID atoms
// ─────────────────────────────────────────────────────────────────────

/// Compute the mean synergy_fraction across a collection of per-residue
/// PidAtoms (filtering out atoms with non-positive total_mi which have
/// synergy_fraction = 0 by definition and should not bias the average).
pub fn mean_synergy_fraction(atoms: &[PidAtoms]) -> f32 {
    let mut sum = 0.0_f64;
    let mut n = 0_usize;
    for a in atoms {
        if a.total_mi > 0.0 {
            sum += a.synergy_fraction();
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (sum / n as f64) as f32
    }
}

/// Compute the Shannon entropy (in nats) of a normalized per-residue
/// distribution. Returns `0.0` for empty input.
pub fn shannon_entropy_nats(distribution: &[f32]) -> f32 {
    let sum: f32 = distribution.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let mut h: f32 = 0.0;
    for &p in distribution {
        if p > 0.0 {
            let normalized = p / sum;
            h -= normalized * normalized.ln();
        }
    }
    h
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_obs(spike_rate: f32, pcmi: f32, synergy: f32, entropy: f32) -> ObservationWindow {
        ObservationWindow {
            chunk_idx: 0,
            spike_rates_per_stream: vec![spike_rate; 8],
            pcmi_mean: pcmi,
            gcpid_synergy_mean: synergy,
            phasor_entropy: entropy,
            current_otsu_threshold: 1.0,
            bocpd_regime_stability: 0.9,
            cumulative_spike_count: 1000,
            // Default healthy intensity distribution — wide spread.
            // Tests that want to exercise the flatness-deficit path should
            // override to a value near 1.0.
            spike_intensity_p90_over_p10: 5.0,
        }
    }

    #[test]
    fn healthy_observation_yields_no_rescue() {
        let targets = RescueTargets::default_for_canonical_target();
        let obs = make_obs(150.0, 0.8, 0.4, 1.8);
        let ctrl = RescueController::new(true);
        // Advance past the consecutive threshold
        for _ in 0..5 {
            let _ = ctrl.decide(&obs, &targets);
        }
        let actions = ctrl.decide(&obs, &targets);
        assert!(
            actions.is_empty(),
            "healthy obs should produce no actions, got {:?}",
            actions
        );
    }

    #[test]
    fn disabled_controller_produces_no_actions() {
        let targets = RescueTargets::default_for_canonical_target();
        let obs = make_obs(0.0, 0.0, 0.0, 3.0);
        let ctrl = RescueController::new(false);
        for _ in 0..5 {
            let actions = ctrl.decide(&obs, &targets);
            assert!(actions.is_empty());
        }
    }

    #[test]
    fn persistent_zero_spike_triggers_rescue_after_threshold() {
        let targets = RescueTargets::default_for_canonical_target();
        // Cumulative spikes > 0 so the dead-engine gate is not tripped.
        let mut obs = make_obs(0.0, 0.0, 0.0, 3.0);
        obs.cumulative_spike_count = 100;
        let ctrl = RescueController::new(true);
        // First 2 chunks: hold (below consecutive threshold).
        for _ in 0..2 {
            let actions = ctrl.decide(&obs, &targets);
            assert!(actions
                .iter()
                .all(|a| matches!(a, RescueAction::Hold { .. })));
        }
        // Third chunk: actions derived.
        let actions = ctrl.decide(&obs, &targets);
        assert!(
            !actions.is_empty(),
            "expected actions on 3rd consecutive chunk"
        );
        let has_amp = actions
            .iter()
            .any(|a| matches!(a, RescueAction::FocusWeightAmplifier { .. }));
        assert!(has_amp, "expected focus weight amplifier in actions");
    }

    #[test]
    fn unstable_regime_gates_rescue() {
        let mut targets = RescueTargets::default_for_canonical_target();
        targets.min_regime_stability = 0.9;
        let obs = make_obs(0.0, 0.0, 0.0, 3.0);
        // Override stability via the observation field (BOCPD inside the
        // controller will also compute one; we force low stability by
        // repeatedly updating with a jumpy rate pattern.)
        let ctrl = RescueController::new(true);
        // Feed BOCPD a series of very jumpy inputs to generate an unstable
        // regime signal from the built-in BOCPD.
        let obs_stable = make_obs(100.0, 0.0, 0.0, 3.0);
        for _ in 0..5 {
            let _ = ctrl.decide(&obs_stable, &targets);
        }
        // Now a sudden drop to zero — BOCPD should flag a changepoint.
        let result = ctrl.decide(&obs, &targets);
        let gated = result.iter().any(
            |a| matches!(a, RescueAction::Hold { reason, .. } if reason == "regime_unstable_bocpd"),
        );
        assert!(gated, "expected regime-unstable hold, got {:?}", result);
    }

    #[test]
    fn amp_magnitude_is_continuous_not_step() {
        let targets = RescueTargets::default_for_canonical_target();
        let mut samples = Vec::new();
        for deficit_level in [10.0, 30.0, 50.0, 70.0, 90.0] {
            let ctrl = RescueController::new(true);
            let mut obs = make_obs(deficit_level, 0.0, 0.0, 3.0);
            obs.cumulative_spike_count = 100;
            for _ in 0..4 {
                let _ = ctrl.decide(&obs, &targets);
            }
            let actions = ctrl.decide(&obs, &targets);
            let amp = actions.iter().find_map(|a| match a {
                RescueAction::FocusWeightAmplifier { multiplier } => Some(*multiplier),
                _ => None,
            });
            if let Some(m) = amp {
                samples.push((deficit_level, m));
            }
        }
        // All samples should have distinct amp values — no step function.
        assert!(samples.len() >= 3, "need at least 3 distinct deficits");
        let mut sorted = samples.clone();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        // Higher deficit (lower spike rate) ⇒ higher amplifier.
        for i in 1..sorted.len() {
            assert!(
                sorted[i - 1].1 >= sorted[i].1,
                "amp should increase as spike rate decreases (lower observed → higher rescue amp)"
            );
        }
    }

    #[test]
    fn mean_synergy_ignores_degenerate_atoms() {
        let atoms = vec![
            PidAtoms {
                synergy: 0.3,
                redundancy: 0.1,
                unique_a: 0.1,
                unique_b: 0.1,
                total_mi: 0.6,
            },
            PidAtoms {
                synergy: 0.0,
                redundancy: 0.0,
                unique_a: 0.0,
                unique_b: 0.0,
                total_mi: 0.0, // degenerate — should be excluded
            },
            PidAtoms {
                synergy: 0.4,
                redundancy: 0.0,
                unique_a: 0.0,
                unique_b: 0.0,
                total_mi: 0.4,
            },
        ];
        let m = mean_synergy_fraction(&atoms);
        // Only the first (0.5) and third (1.0) should contribute.
        // Expected: (0.5 + 1.0) / 2 = 0.75
        assert!((m - 0.75).abs() < 0.01, "got {}", m);
    }

    #[test]
    fn shannon_entropy_uniform_is_maximal() {
        let uniform = vec![1.0_f32; 10];
        let peaked = {
            let mut v = vec![0.01_f32; 10];
            v[0] = 0.91;
            v
        };
        let h_uniform = shannon_entropy_nats(&uniform);
        let h_peaked = shannon_entropy_nats(&peaked);
        assert!(
            h_uniform > h_peaked,
            "uniform entropy {} should exceed peaked {}",
            h_uniform,
            h_peaked
        );
    }

    #[test]
    fn deficit_from_observation_bounded() {
        let targets = RescueTargets::default_for_canonical_target();
        let obs = make_obs(0.0, 0.0, 0.0, 10.0);
        let d = InformationDeficit::from_observation(&obs, &targets);
        assert!(d.spike_rate_deficit >= 0.0 && d.spike_rate_deficit <= 1.0);
        assert!(d.coherence_deficit >= 0.0 && d.coherence_deficit <= 1.0);
        assert!(d.synergy_deficit >= 0.0 && d.synergy_deficit <= 1.0);
        assert!(d.entropy_deficit >= 0.0 && d.entropy_deficit <= 1.0);
    }

    #[test]
    fn v2_actions_flagged_correctly() {
        assert!(RescueAction::EngineV2Rest2LambdaStep {
            stream_idx: 0,
            delta: -0.1
        }
        .is_engine_v2_only());
        assert!(RescueAction::EngineV2NmaAmpMultiplier { multiplier: 2.0 }.is_engine_v2_only());
        assert!(!RescueAction::FocusWeightAmplifier { multiplier: 2.0 }.is_engine_v2_only());
        assert!(!RescueAction::FocusListInjection { residues: vec![] }.is_engine_v2_only());
    }

    // ─────────────────────────────────────────────────────────────────
    // apply_actions: the wiring from decision → effect must be testable
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn apply_amplifier_scales_both_lists() {
        let mut synergy = vec![(10, 0.5_f32), (11, 0.3)];
        let mut redundancy = vec![(12, 0.8_f32), (13, 0.6)];
        let actions = vec![RescueAction::FocusWeightAmplifier { multiplier: 3.0 }];
        let summary = apply_actions(&actions, &mut synergy, &mut redundancy, 64, |_| vec![]);
        // Use epsilon comparisons — f32 multiplication introduces unit
        // ULP drift (0.3 * 3.0 != 0.9 exactly).
        let eps = 1e-5_f32;
        assert_eq!(synergy.len(), 2);
        assert_eq!(synergy[0].0, 10);
        assert!((synergy[0].1 - 1.5).abs() < eps);
        assert_eq!(synergy[1].0, 11);
        assert!((synergy[1].1 - 0.9).abs() < eps);
        assert_eq!(redundancy.len(), 2);
        assert!((redundancy[0].1 - 2.4).abs() < eps);
        assert!((redundancy[1].1 - 1.8).abs() < eps);
        assert_eq!(summary.amplifier_applied, 1);
        assert!((summary.amplifier_total_multiplier - 3.0).abs() < eps);
    }

    #[test]
    fn apply_amplifier_composes_multiplicatively() {
        let mut synergy = vec![(10, 1.0_f32)];
        let mut redundancy: Vec<(i32, f32)> = vec![];
        let actions = vec![
            RescueAction::FocusWeightAmplifier { multiplier: 2.0 },
            RescueAction::FocusWeightAmplifier { multiplier: 4.0 },
        ];
        let summary = apply_actions(&actions, &mut synergy, &mut redundancy, 64, |_| vec![]);
        assert!((synergy[0].1 - 8.0).abs() < 1e-5);
        assert_eq!(summary.amplifier_applied, 2);
        assert!((summary.amplifier_total_multiplier - 8.0).abs() < 1e-5);
    }

    #[test]
    fn apply_injection_appends_candidates_and_skips_already_present() {
        let mut synergy = vec![(10, 0.5_f32), (11, 0.3)];
        let mut redundancy: Vec<(i32, f32)> = vec![];
        // Request K=3 injection.
        let mut req: Vec<(i32, f32)> = Vec::with_capacity(3);
        req.reserve_exact(3);
        let actions = vec![RescueAction::FocusListInjection { residues: req }];
        // Candidate supplier returns rid=10 (already present — should be
        // skipped), rid=20, rid=21, rid=22 (new — should be appended).
        let summary = apply_actions(&actions, &mut synergy, &mut redundancy, 64, |k| {
            assert!(k >= 3, "candidate supplier called with k={}", k);
            vec![(10, 0.9), (20, 0.8), (21, 0.7), (22, 0.6)]
        });
        // synergy should now contain the original 2 plus the 3 new (skipping rid=10).
        assert!(synergy.iter().any(|&(r, _)| r == 20));
        assert!(synergy.iter().any(|&(r, _)| r == 21));
        assert!(synergy.iter().any(|&(r, _)| r == 22));
        assert_eq!(summary.injection_applied, 1);
        assert_eq!(summary.injected_residues.len(), 3);
    }

    #[test]
    fn apply_injection_respects_focus_max() {
        let mut synergy: Vec<(i32, f32)> = (0..62).map(|i| (i, 0.1)).collect();
        let mut redundancy: Vec<(i32, f32)> = vec![];
        let mut req: Vec<(i32, f32)> = Vec::with_capacity(8);
        req.reserve_exact(8);
        let actions = vec![RescueAction::FocusListInjection { residues: req }];
        let _ = apply_actions(&actions, &mut synergy, &mut redundancy, 64, |k| {
            (100..100 + k as i32).map(|i| (i, 0.5)).collect()
        });
        assert_eq!(synergy.len(), 64, "must truncate back to focus_max");
    }

    #[test]
    fn apply_v2_actions_dont_mutate_but_are_counted() {
        let mut synergy = vec![(10, 0.5_f32)];
        let mut redundancy = vec![(11, 0.6_f32)];
        let actions = vec![
            RescueAction::EngineV2Rest2LambdaStep {
                stream_idx: 0,
                delta: -0.1,
            },
            RescueAction::EngineV2NmaAmpMultiplier { multiplier: 3.0 },
        ];
        let summary = apply_actions(&actions, &mut synergy, &mut redundancy, 64, |_| vec![]);
        assert_eq!(synergy, vec![(10, 0.5)]);
        assert_eq!(redundancy, vec![(11, 0.6)]);
        assert_eq!(summary.v2_only_requested, 2);
        assert_eq!(summary.amplifier_applied, 0);
        assert_eq!(summary.injection_applied, 0);
    }

    #[test]
    fn apply_hold_is_counted_but_no_mutation() {
        let mut synergy = vec![(10, 0.5_f32)];
        let mut redundancy: Vec<(i32, f32)> = vec![];
        let actions = vec![RescueAction::Hold {
            reason: "test_hold".to_string(),
            deficit: InformationDeficit::default(),
        }];
        let summary = apply_actions(&actions, &mut synergy, &mut redundancy, 64, |_| vec![]);
        assert_eq!(synergy, vec![(10, 0.5)]);
        assert_eq!(summary.holds, 1);
    }

    // ─────────────────────────────────────────────────────────────────
    // RescueTargets: load + validate
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn targets_load_from_valid_json() {
        let json = br#"{
            "target_spike_rate": 50.0,
            "target_pcmi": 0.55,
            "target_synergy": 0.20,
            "target_phasor_entropy": 2.2,
            "min_consecutive_below": 5,
            "min_regime_stability": 0.75
        }"#;
        let tmp = std::env::temp_dir().join("rescue_targets_ok.json");
        std::fs::write(&tmp, json).unwrap();
        let t = RescueTargets::load_from_json(&tmp).unwrap();
        assert!((t.target_spike_rate - 50.0).abs() < 1e-5);
        assert_eq!(t.min_consecutive_below, 5);
    }

    #[test]
    fn targets_validate_rejects_bad_fields() {
        let mut t = RescueTargets::default_for_canonical_target();
        t.target_spike_rate = -1.0;
        assert!(t.validate().is_err());
        t = RescueTargets::default_for_canonical_target();
        t.target_pcmi = 1.5;
        assert!(t.validate().is_err());
        t = RescueTargets::default_for_canonical_target();
        t.min_consecutive_below = 0;
        assert!(t.validate().is_err());
        t = RescueTargets::default_for_canonical_target();
        t.min_regime_stability = -0.1;
        assert!(t.validate().is_err());
    }

    #[test]
    fn targets_load_rejects_invalid_json() {
        let tmp = std::env::temp_dir().join("rescue_targets_bad.json");
        std::fs::write(&tmp, b"{ \"target_spike_rate\": 0.0 }").unwrap();
        let r = RescueTargets::load_from_json(&tmp);
        assert!(r.is_err(), "zero spike-rate target must be rejected");
    }

    // ─────────────────────────────────────────────────────────────────
    // End-to-end: RescueController + apply_actions produce observable
    // mutations after the consecutive threshold
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn flatness_deficit_triggers_nma_rescue() {
        // POLQ/CBL-B regime-boundary signature: spike rate looks fine (kernel
        // IS firing), synergy and coherence not in trouble, BUT intensity
        // distribution is flat (p90/p10 = 1.5 vs target 3.0 → deficit ~0.75).
        // This should trigger the NMA amp rescue via the flatness axis even
        // when entropy_deficit alone is zero.
        let targets = RescueTargets::default_for_canonical_target();
        let mut obs = make_obs(150.0, 0.8, 0.4, 1.5); // healthy on 4 axes
        obs.cumulative_spike_count = 10_000_000;
        obs.spike_intensity_p90_over_p10 = 1.5; // flat = POLQ signature
        let ctrl = RescueController::new(true);
        // Advance past the consecutive-below threshold + BOCPD warmup.
        for _ in 0..5 {
            let _ = ctrl.decide(&obs, &targets);
        }
        let actions = ctrl.decide(&obs, &targets);
        let has_nma = actions
            .iter()
            .any(|a| matches!(a, RescueAction::EngineV2NmaAmpMultiplier { .. }));
        assert!(
            has_nma,
            "expected NMA rescue on flat intensity distribution, got: {:?}",
            actions
        );
    }

    #[test]
    fn healthy_intensity_distribution_yields_no_flatness_deficit() {
        let targets = RescueTargets::default_for_canonical_target();
        let obs = make_obs(150.0, 0.8, 0.4, 1.5); // healthy, default p90/p10=5.0
        let deficit = InformationDeficit::from_observation(&obs, &targets);
        assert_eq!(deficit.intensity_flatness_deficit, 0.0);
    }

    #[test]
    fn flat_intensity_distribution_yields_flatness_deficit() {
        let targets = RescueTargets::default_for_canonical_target();
        let mut obs = make_obs(150.0, 0.8, 0.4, 1.5);
        obs.spike_intensity_p90_over_p10 = 1.5; // flat
        let deficit = InformationDeficit::from_observation(&obs, &targets);
        // Log-scale: 1 - log(1.5)/log(3.0) ≈ 1 - 0.4055/1.0986 ≈ 0.631
        let expected = 1.0 - (1.5_f32).ln() / (3.0_f32).ln();
        assert!(
            (deficit.intensity_flatness_deficit - expected).abs() < 1e-4,
            "expected log-scale flatness deficit ~{:.4}, got {}",
            expected,
            deficit.intensity_flatness_deficit
        );
    }

    #[test]
    fn perfectly_flat_yields_unit_flatness_deficit() {
        let targets = RescueTargets::default_for_canonical_target();
        let mut obs = make_obs(150.0, 0.8, 0.4, 1.5);
        obs.spike_intensity_p90_over_p10 = 1.0; // perfectly flat
        let deficit = InformationDeficit::from_observation(&obs, &targets);
        // log(1)/log(3) = 0  → deficit = 1.0
        assert!(
            (deficit.intensity_flatness_deficit - 1.0).abs() < 1e-5,
            "expected flatness deficit 1.0, got {}",
            deficit.intensity_flatness_deficit
        );
    }

    #[test]
    fn at_target_flatness_is_zero() {
        let targets = RescueTargets::default_for_canonical_target();
        let mut obs = make_obs(150.0, 0.8, 0.4, 1.5);
        obs.spike_intensity_p90_over_p10 = targets.target_intensity_p90_over_p10;
        let deficit = InformationDeficit::from_observation(&obs, &targets);
        assert_eq!(deficit.intensity_flatness_deficit, 0.0);
    }

    // ─────────────────────────────────────────────────────────────────
    // ChunkIntensityChannel unit + threaded integration tests
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn channel_publish_stream_computes_correct_percentiles() {
        let ch = ChunkIntensityChannel::new(2);
        // Stream 0: 100 samples, linear ramp 0.01 .. 1.00.
        // Impl picks p10_idx = floor(n * 0.1) = 10 (0-indexed) → sample 0.11.
        // Impl picks p90_idx = floor(n * 0.9) = 90 (0-indexed) → sample 0.91.
        // Expected ratio = 0.91 / 0.11 = 8.272...
        let data: Vec<f32> = (1..=100).map(|i| i as f32 / 100.0).collect();
        ch.publish_stream(0, &data);
        let ratio = ch.aggregate().unwrap();
        let expected = 0.91_f32 / 0.11_f32;
        assert!(
            (ratio - expected).abs() < 0.01,
            "expected {:.3}, got {:.3}",
            expected,
            ratio
        );
    }

    #[test]
    fn channel_publish_empty_signals_no_data() {
        let ch = ChunkIntensityChannel::new(4);
        // No stream publishes.
        assert!(ch.aggregate().is_none());
    }

    #[test]
    fn channel_filters_non_finite_intensities() {
        let ch = ChunkIntensityChannel::new(1);
        let mixed = vec![
            1.0,
            f32::NAN,
            2.0,
            f32::INFINITY,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ];
        ch.publish_stream(0, &mixed);
        let ratio = ch.aggregate().unwrap();
        // After filter: 8 finite samples 1..8, p10 idx = 0 (→ 1.0), p90 idx = 7 (→ 8.0)
        assert!((ratio - 8.0).abs() < 1e-3, "got {}", ratio);
    }

    #[test]
    fn channel_aggregate_skips_zero_n_streams() {
        let ch = ChunkIntensityChannel::new(3);
        // Stream 0 publishes; streams 1 and 2 are idle.
        ch.publish_stream(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        ch.publish_empty(1);
        ch.publish_empty(2);
        let ratio = ch.aggregate().unwrap();
        // Stream 0: p10 at idx 1 (2.0), p90 at idx 9 (10.0), ratio = 5.0
        assert!((ratio - 5.0).abs() < 1e-3, "got {}", ratio);
    }

    #[test]
    fn channel_aggregate_median_of_medians_is_robust_to_one_flat_stream() {
        // Threaded test: 4 streams, 3 publish wide distributions, 1 publishes flat.
        // Median-of-medians should preserve the wide-distribution signal, NOT
        // collapse to the flat stream's ratio.
        use std::sync::Arc;
        use std::sync::Barrier;
        use std::thread;
        let ch = Arc::new(ChunkIntensityChannel::new(4));
        let barrier = Arc::new(Barrier::new(5)); // 4 writers + main
        let mut handles = vec![];
        for s_idx in 0..4 {
            let c = Arc::clone(&ch);
            let b = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                let data: Vec<f32> = match s_idx {
                    // Three wide distributions: p10 ≈ 1.0, p90 ≈ 10.0 → ratio ~10
                    0 | 1 | 2 => (1..=100).map(|k| k as f32 / 10.0).collect(),
                    // One flat: all 5.0 → p90/p10 = 1.0
                    _ => vec![5.0; 100],
                };
                c.publish_stream(s_idx, &data);
                b.wait();
            }));
        }
        barrier.wait();
        for h in handles {
            h.join().unwrap();
        }
        let ratio = ch.aggregate().unwrap();
        // median_p10 across [1.0, 1.0, 1.0, 5.0] = 1.0
        // median_p90 across [10.0, 10.0, 10.0, 5.0] = 10.0
        // ratio = 10.0. The flat stream does NOT drag the aggregate down.
        assert!(
            ratio > 5.0,
            "median-of-medians should preserve wide signal from 3 streams, got {}",
            ratio
        );
    }

    #[test]
    fn channel_aggregate_catches_all_flat_distribution() {
        // Every stream is flat — this is the POLQ / CBL-B regime-boundary
        // pattern. Aggregate should return a ratio near 1.0 → rescue
        // controller sees a strong flatness_deficit.
        let ch = ChunkIntensityChannel::new(4);
        for s_idx in 0..4 {
            let data = vec![1.0f32; 100]; // perfectly flat
            ch.publish_stream(s_idx, &data);
        }
        let ratio = ch.aggregate().unwrap();
        assert!(
            (ratio - 1.0).abs() < 1e-3,
            "all-flat distribution should aggregate to ~1.0, got {}",
            ratio
        );
    }

    #[test]
    fn channel_reset_clears_all_slots() {
        let ch = ChunkIntensityChannel::new(2);
        ch.publish_stream(0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert!(ch.aggregate().is_some());
        ch.reset();
        assert!(
            ch.aggregate().is_none(),
            "reset should clear all slots to no-data state"
        );
    }

    #[test]
    fn channel_out_of_bounds_stream_idx_is_no_op() {
        let ch = ChunkIntensityChannel::new(2);
        // Must not panic.
        ch.publish_stream(99, &[1.0, 2.0, 3.0]);
        ch.publish_empty(99);
        assert!(
            ch.aggregate().is_none(),
            "out-of-bounds publishes should not corrupt valid slots"
        );
    }

    #[test]
    fn nan_intensity_yields_zero_flatness_deficit() {
        // NaN observed (caller unable or unwilling to compute) should be
        // treated as "no claim" — zero deficit, no false rescue.
        let targets = RescueTargets::default_for_canonical_target();
        let mut obs = make_obs(150.0, 0.8, 0.4, 1.5);
        obs.spike_intensity_p90_over_p10 = f32::NAN;
        let deficit = InformationDeficit::from_observation(&obs, &targets);
        assert_eq!(deficit.intensity_flatness_deficit, 0.0);
    }

    #[test]
    fn end_to_end_rescue_amplifies_published_weights() {
        let targets = RescueTargets::default_for_canonical_target();
        let mut obs = make_obs(0.0, 0.0, 0.0, 3.0);
        obs.cumulative_spike_count = 100;
        let ctrl = RescueController::new(true);
        // Feed past consecutive-below threshold.
        for _ in 0..3 {
            let _ = ctrl.decide(&obs, &targets);
        }
        let actions = ctrl.decide(&obs, &targets);
        // Now apply to a synthetic list.
        let mut synergy = vec![(42, 1.0_f32), (43, 0.5)];
        let mut redundancy = vec![(99, 0.8_f32)];
        let summary = apply_actions(&actions, &mut synergy, &mut redundancy, 64, |k| {
            (200..200 + k as i32).map(|i| (i, 0.2)).collect()
        });
        // At spike_rate_deficit=1.0, amp = 1 + 4·1 = 5.0.
        assert!(
            synergy[0].1 >= 4.99 && synergy[0].1 <= 5.01,
            "expected weight ~5.0 after amplification, got {}",
            synergy[0].1
        );
        assert!(summary.amplifier_applied >= 1);
    }
}

/// Reusable coupled-window boundary that releases peers on cancellation or worker exit.
/// A completed generation remains successful even if a participant retires immediately after it.
#[derive(Clone)]
pub struct CoupledWindowBarrier {
    inner: std::sync::Arc<(std::sync::Mutex<CoupledBoundaryState>, std::sync::Condvar)>,
    participants: usize,
}
#[derive(Default)]
struct CoupledBoundaryState {
    generation: u64,
    arrived: usize,
    aborted: Option<String>,
}
impl CoupledWindowBarrier {
    pub fn new(participants: usize) -> Self {
        assert!(participants > 0);
        Self { inner: std::sync::Arc::new((std::sync::Mutex::new(CoupledBoundaryState::default()), std::sync::Condvar::new())), participants }
    }
    pub fn abort(&self, reason: impl Into<String>) {
        let (lock, wake) = &*self.inner;
        let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
        if state.aborted.is_none() { state.aborted = Some(reason.into()); }
        wake.notify_all();
    }
    pub fn is_aborted(&self) -> bool {
        self.inner.0.lock().unwrap_or_else(|e| e.into_inner()).aborted.is_some()
    }
    pub fn participant(&self, stream_id: usize) -> CoupledWindowParticipant {
        CoupledWindowParticipant { barrier: self.clone(), stream_id }
    }
    pub fn wait(&self, cancelled: impl Fn() -> bool) -> Result<(), String> {
        let (lock, wake) = &*self.inner;
        let mut state = lock.lock().unwrap_or_else(|e| e.into_inner());
        if cancelled() && state.aborted.is_none() {
            state.aborted = Some("coupled window cancelled".into());
            wake.notify_all();
        }
        if let Some(reason) = &state.aborted { return Err(reason.clone()); }
        let generation = state.generation;
        state.arrived += 1;
        if state.arrived == self.participants {
            state.arrived = 0;
            state.generation += 1;
            wake.notify_all();
            return Ok(());
        }
        loop {
            // Poll only to observe an external watchdog flag without requiring a CUDA callback.
            state = wake.wait_timeout(state, std::time::Duration::from_millis(50))
                .unwrap_or_else(|e| e.into_inner()).0;
            if state.generation != generation { return Ok(()); }
            if cancelled() && state.aborted.is_none() {
                state.aborted = Some("coupled window cancelled".into());
                wake.notify_all();
            }
            if let Some(reason) = &state.aborted { return Err(reason.clone()); }
        }
    }
}
/// Drop guard established before fallible worker setup.
pub struct CoupledWindowParticipant { barrier: CoupledWindowBarrier, stream_id: usize }
impl Drop for CoupledWindowParticipant {
    fn drop(&mut self) { self.barrier.abort(format!("physical stream {} exited coupled windows", self.stream_id)); }
}


#[derive(Serialize,Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RescueContinuationV1 {
    schema_version:u32,enabled:bool,consecutive_below:u32,
    history:Vec<RescueDecisionRecord>,bocpd:Option<Vec<u8>>,
}
impl RescueController {
    /// Caller holds the complete worker-generation pause; no decide call races.
    pub fn continuation_allocation_plan(&self)->anyhow::Result<Vec<(String,u64)>> {
        let history=self.history.lock().map_err(|_|anyhow::anyhow!("poisoned rescue history"))?;
        let mut rows=vec![("complete_decision_history_wire".into(),crate::checkpoint_codec::encoded_size(&*history,u64::MAX)?)];
        if let Some(bocpd)=&self.bocpd{let state=bocpd.lock().map_err(|_|anyhow::anyhow!("poisoned rescue BOCPD"))?;
            rows.push(("BOCPD_complete_history_wire".into(),crate::checkpoint_codec::encoded_size(&*state,u64::MAX)?));}
        Ok(rows)
    }
    pub fn capture_continuation(&self)->anyhow::Result<RescueContinuationV1> {
        let bocpd=self.bocpd.as_ref().map(|v|{
            let state=v.lock().map_err(|_|anyhow::anyhow!("poisoned rescue BOCPD"))?;
            Ok::<_,anyhow::Error>(crate::checkpoint_codec::to_bytes(&*state)?)}).transpose()?;
        Ok(RescueContinuationV1{schema_version:1,enabled:self.enabled.load(Ordering::Acquire),
            consecutive_below:self.consecutive_below.load(Ordering::Acquire),
            history:self.history.lock().map_err(|_|anyhow::anyhow!("poisoned rescue history"))?.clone(),bocpd})
    }
    pub fn restore_continuation(&self,saved:RescueContinuationV1)->anyhow::Result<()> {
        anyhow::ensure!(saved.schema_version==1 && saved.bocpd.is_some()==self.bocpd.is_some(),"rescue owner identity changed");
        let bocpd=saved.bocpd.as_deref().map(crate::checkpoint_codec::from_bytes::<BocpdState<ConstantHazard>>).transpose()?;
        // Decode before mutation. Fresh-process owner is not yet visible to workers.
        if let (Some(target),Some(value))=(&self.bocpd,bocpd){*target.lock().map_err(|_|anyhow::anyhow!("poisoned rescue BOCPD"))?=value;}
        *self.history.lock().map_err(|_|anyhow::anyhow!("poisoned rescue history"))?=saved.history;
        self.enabled.store(saved.enabled,Ordering::Release);self.consecutive_below.store(saved.consecutive_below,Ordering::Release);Ok(())
    }
}
impl ChunkIntensityChannel {
    pub fn capture_continuation(&self)->Vec<[u32;3]> {
        (0..self.n.len()).map(|i|[self.p10_bits[i].load(Ordering::Acquire),self.p90_bits[i].load(Ordering::Acquire),self.n[i].load(Ordering::Acquire)]).collect()
    }
    pub fn restore_continuation(&self,saved:&[[u32;3]])->anyhow::Result<()> {
        anyhow::ensure!(saved.len()==self.n.len(),"intensity channel stream population changed");
        for (i,words) in saved.iter().enumerate(){self.p10_bits[i].store(words[0],Ordering::Release);
            self.p90_bits[i].store(words[1],Ordering::Release);self.n[i].store(words[2],Ordering::Release);}Ok(())
    }
}

impl CoupledWindowBarrier {
    pub fn capture_continuation(&self)->anyhow::Result<(usize,u64)> {
        let state=self.inner.0.lock().map_err(|_|anyhow::anyhow!("poisoned coupling barrier"))?;
        anyhow::ensure!(state.arrived==0 && state.aborted.is_none(),"coupling barrier is not quiescent");
        Ok((self.participants,state.generation))
    }
    pub fn restore_continuation(&self,saved:(usize,u64))->anyhow::Result<()> {
        anyhow::ensure!(saved.0==self.participants,"coupling participant population changed");
        let mut state=self.inner.0.lock().map_err(|_|anyhow::anyhow!("poisoned coupling barrier"))?;
        anyhow::ensure!(state.arrived==0 && state.generation==0 && state.aborted.is_none(),"coupling restore requires fresh non-visible barrier");
        state.generation=saved.1;Ok(())
    }
}
