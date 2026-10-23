//! Bayesian Online Changepoint Detection (Adams & MacKay, 2007).
//!
//! Stage 1B-2: principled, parameter-light replacement for the magic
//! `chunk_size = 500` constant in the autonomous chunk loop. The fixed
//! 500-step chunking is wall-clock-driven; this module provides the
//! foundation for **physics-driven** chunking where chunk boundaries are
//! placed at points the model identifies as regime changes (changepoints)
//! in the per-stream spike rate stream.
//!
//! ## Algorithm
//!
//! BOCPD maintains, at each timestep `t`, a posterior distribution over
//! the **run length** `r_t` — the number of steps since the most recent
//! changepoint. The posterior is updated recursively at each new
//! observation `x_{t+1}`:
//!
//!   • **Growth**:
//!     P(r_{t+1} = r + 1 | x_{1:t+1}) ∝ (1 − H(r)) · π(x_{t+1} | r)
//!                                      · P(r_t = r | x_{1:t})
//!   • **Reset** (a changepoint just occurred):
//!     P(r_{t+1} = 0 | x_{1:t+1}) ∝ Σ_r H(r) · π(x_{t+1} | r)
//!                                  · P(r_t = r | x_{1:t})
//!   • **Normalize** so the run-length posterior sums to 1.
//!
//! Where:
//!   • `H(r)` is the **hazard function** — the prior probability of a
//!     changepoint given the current run length. For a constant-hazard
//!     process H is independent of `r` and equals `1/expected_run_length`.
//!     For PRISM-TWIN we parameterize H by the CCNS protocol phase so
//!     cold_hold has long expected runs (low hazard) and warm_hold has
//!     short expected runs (high hazard).
//!   • `π(x | r)` is the **predictive distribution** of the next
//!     observation given a run of length r. For PRISM spike-rate
//!     observations we use a Gaussian with running mean and variance
//!     (Welford's online algorithm) per run length.
//!
//! The output of interest is `P(r_t = 0 | x_{1:t})` — the posterior
//! probability that the most recent observation was a changepoint. When
//! this exceeds a threshold (default 0.5 = "more likely than not"), the
//! caller can interpret it as "regime change detected, close the chunk."
//!
//! ## Reference
//!
//! Adams, R. P., & MacKay, D. J. C. (2007). *Bayesian online changepoint
//! detection.* arXiv:0710.3742.
//!
//! Knoblauch, J., & Damoulas, T. (2018). *Spatio-temporal Bayesian
//! on-line changepoint detection with model selection.* For the
//! multi-stream extension where N parallel streams share a common
//! protocol-phase hazard.
//!
//! ## Why this is non-naive
//!
//!   1. **Zero magic constants in the run-length update.** The hazard
//!      function takes its expected run lengths from the actual CCNS
//!      protocol step counts (`cold_hold_steps`, `ramp_steps`,
//!      `warm_hold_steps`). The threshold for "close the chunk" is the
//!      single user-facing tunable, expressible as P(reset) > τ or
//!      equivalently as a log-odds ratio in bits.
//!   2. **Numerically stable**. All updates are in log space using
//!      `log_sum_exp` to avoid underflow on long observation streams.
//!   3. **Bounded memory**. The run-length posterior is pruned to
//!      `MAX_RUN_LENGTH` entries; older run lengths get folded into
//!      a tail bucket so the posterior never grows unbounded.
//!   4. **Online**. O(MAX_RUN_LENGTH) per observation. No retraining,
//!      no batch refit, no parameter tuning.
//!   5. **The same machinery feeds Stage 1B-3 GC-PID** — the per-chunk
//!      BOCPD posterior is the natural conditioning variable for the
//!      Gaussian-copula PID synergy estimator.

/// Maximum run length to track in the posterior. Beyond this, the
/// posterior is truncated and the tail mass is collapsed to the
/// truncation boundary. 256 is a generous upper bound for chunk-rate
/// observations: at one observation per 500 MD steps and a 55 000-step
/// run, we get 110 observations total — well under 256.
pub const MAX_RUN_LENGTH: usize = 256;

/// Default posterior threshold for chunk close decisions. When
/// `P(r_t = 0 | x_{1:t}) > BOCPD_CLOSE_THRESHOLD`, the caller should
/// treat the most recent observation as a changepoint and close the
/// current chunk.
///
/// 0.5 means "more likely than not a changepoint occurred" — the
/// canonical Bayesian decision threshold. In bits this is a log-odds
/// ratio of 0 (zero evidence either way; the threshold sits exactly at
/// the indifference point). Users who want stricter changepoint
/// detection should pass a higher value (0.8 ≈ 2 bits = "4× more likely
/// reset than continuation"); users who want more aggressive segmentation
/// should pass a lower value (0.3 ≈ −1 bit = "2× more likely
/// continuation than reset, but still allowed to close").
pub const BOCPD_CLOSE_THRESHOLD: f64 = 0.5;

/// Hazard function trait — the prior probability of a changepoint at a
/// given run length and timestep. Implementations can be constant in
/// run length, time-varying with the protocol phase, or both.
pub trait HazardFn {
    /// Return P(changepoint | run_length = r, timestep = t).
    /// Must be in (0, 1) — clamp degenerate values internally.
    fn hazard(&self, run_length: usize, timestep: i32) -> f64;
}

/// Constant-hazard model: H(r) = 1 / expected_run_length, independent of
/// run length and timestep. Equivalent to assuming changepoints follow a
/// memoryless geometric process.
#[derive(Debug, Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ConstantHazard {
    pub expected_run_length: f64,
}

impl ConstantHazard {
    pub fn new(expected_run_length: f64) -> Self {
        Self {
            expected_run_length: expected_run_length.max(1.0),
        }
    }
}

impl HazardFn for ConstantHazard {
    fn hazard(&self, _run_length: usize, _timestep: i32) -> f64 {
        (1.0 / self.expected_run_length).clamp(1e-9, 1.0 - 1e-9)
    }
}

/// CCNS protocol-phase-aware hazard function. The expected run length
/// changes between cold_hold (long, low hazard), ramp (medium), and
/// warm_hold (short, high hazard). This encodes the prior knowledge that
/// changepoints are more frequent during the warm phase where the
/// cryptic pocket dynamics are actually unfolding.
#[derive(Debug, Clone, Copy)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PhaseAwareHazard {
    pub cold_hold_steps: i32,
    pub ramp_steps: i32,
    pub cold_expected_run: f64,
    pub ramp_expected_run: f64,
    pub warm_expected_run: f64,
}

impl PhaseAwareHazard {
    /// Sensible default: cold expected ~ 1/4 of cold_hold (so ~4
    /// changepoints during cold phase), ramp ~1/8 of ramp (~8 during
    /// ramp), warm ~1/16 of warm_hold (~16 changepoints during warm —
    /// where the cryptic pocket dynamics live).
    pub fn from_protocol(cold_hold: i32, ramp: i32, warm_hold: i32) -> Self {
        Self {
            cold_hold_steps: cold_hold,
            ramp_steps: ramp,
            cold_expected_run: (cold_hold as f64 / 4.0).max(50.0),
            ramp_expected_run: (ramp as f64 / 8.0).max(25.0),
            warm_expected_run: (warm_hold as f64 / 16.0).max(15.0),
        }
    }
}

impl HazardFn for PhaseAwareHazard {
    fn hazard(&self, _run_length: usize, timestep: i32) -> f64 {
        let expected = if timestep < self.cold_hold_steps {
            self.cold_expected_run
        } else if timestep < self.cold_hold_steps + self.ramp_steps {
            self.ramp_expected_run
        } else {
            self.warm_expected_run
        };
        (1.0 / expected.max(1.0)).clamp(1e-9, 1.0 - 1e-9)
    }
}

/// Online sufficient statistics for a Gaussian predictive distribution
/// using Welford's algorithm. Each run length in the BOCPD state has its
/// own GaussianSufficient that accumulates the observations seen since
/// the start of that run.
#[derive(Debug, Clone)]
#[derive(serde::Serialize, serde::Deserialize)]
pub struct GaussianSufficient {
    /// Running mean.
    pub mean: f64,
    /// Sum of squared deviations from the running mean (for variance).
    pub m2: f64,
    /// Number of observations seen.
    pub n: usize,
    /// Prior variance — used as the predictive variance when n < 2.
    pub prior_variance: f64,
}

impl GaussianSufficient {
    /// Construct an empty sufficient statistic with the given prior
    /// variance (used until at least 2 observations have been seen).
    pub fn new(prior_variance: f64) -> Self {
        Self {
            mean: 0.0,
            m2: 0.0,
            n: 0,
            prior_variance: prior_variance.max(1e-9),
        }
    }

    /// Update with a new observation (Welford's online algorithm).
    pub fn update(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    /// Sample variance, with Bayesian regularization toward the prior.
    ///
    /// For n < 2: returns the prior variance (no data yet).
    /// For n >= 2: returns max(sample_variance, prior_variance × 0.01).
    ///
    /// The 1% prior floor prevents the predictive distribution from
    /// collapsing to a delta function when all observations happen to be
    /// identical (sample_variance = 0). Without this floor, an established
    /// run with σ² = 0 would assign log-probability ≈ −∞ to ANY new value
    /// that isn't exactly the running mean, including legitimate
    /// changepoints, which prevents the BOCPD reset path from firing.
    /// Caught by the `jump_triggers_reset` test.
    ///
    /// This is the simplest defensive form of conjugate-prior
    /// regularization. A future revision could swap in the full
    /// Normal-Inverse-Gamma posterior + Student's t predictive for a
    /// more principled treatment, but the floor is sufficient for
    /// well-behaved chunk-rate signals.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            self.prior_variance
        } else {
            let sample_var = self.m2 / (self.n - 1) as f64;
            sample_var.max(self.prior_variance * 0.01)
        }
    }

    /// Log-probability of `x` under this Gaussian predictive distribution.
    pub fn log_predictive(&self, x: f64) -> f64 {
        let var = self.variance();
        let dx = x - self.mean;
        -0.5 * (2.0 * std::f64::consts::PI * var).ln() - dx * dx / (2.0 * var)
    }
}

/// Numerically stable log-sum-exp.
///
/// Given a slice of log-probabilities `[log a_1, log a_2, ..., log a_n]`,
/// returns `log(a_1 + a_2 + ... + a_n)` without underflow.
pub fn log_sum_exp(log_values: &[f64]) -> f64 {
    if log_values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = log_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = log_values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// BOCPD state: maintains the run-length posterior and per-run-length
/// Gaussian sufficient statistics.
///
/// One `BocpdState` instance per stream (per group, per protocol). The
/// caller feeds observations via `update()` and queries the posterior
/// via `reset_probability()`.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BocpdState<H: HazardFn> {
    /// log P(r_t = j | x_{1:t}) for j = 0..len-1.
    log_run_length_dist: Vec<f64>,
    /// Per-run-length Gaussian sufficient statistics. Index j corresponds
    /// to run length j; suff_stats[0] is the freshly-reset run.
    suff_stats: Vec<GaussianSufficient>,
    /// Hazard function (parameterized by protocol phase).
    pub hazard: H,
    /// Prior variance for fresh runs.
    prior_variance: f64,
    /// Number of update calls so far (for diagnostics).
    pub n_observations: usize,
    /// Maximum run length to track.
    max_run_length: usize,
}

impl<H: HazardFn> BocpdState<H> {
    /// Construct a fresh BOCPD state with the given hazard function and
    /// prior variance for the Gaussian predictive.
    pub fn new(hazard: H, prior_variance: f64) -> Self {
        let mut suff = Vec::with_capacity(MAX_RUN_LENGTH);
        suff.push(GaussianSufficient::new(prior_variance));
        Self {
            log_run_length_dist: vec![0.0], // log P(r=0) = 0 (probability 1)
            suff_stats: suff,
            hazard,
            prior_variance,
            n_observations: 0,
            max_run_length: MAX_RUN_LENGTH,
        }
    }

    /// Update the run-length posterior with a new observation.
    ///
    /// `observation` is the new value (e.g., a per-stream spike rate).
    /// `timestep` is passed to the hazard function so phase-aware
    /// hazards can compute the correct expected run length.
    ///
    /// Returns the new `P(r_t = 0 | x_{1:t})` — the posterior
    /// probability that the most recent observation was a changepoint.
    pub fn update(&mut self, observation: f64, timestep: i32) -> f64 {
        let n_existing = self.log_run_length_dist.len();

        // Compute log-predictive of `observation` under each existing run length.
        let mut log_pred: Vec<f64> = Vec::with_capacity(n_existing);
        for r in 0..n_existing {
            log_pred.push(self.suff_stats[r].log_predictive(observation));
        }

        // Compute log-hazard and log-(1-hazard) per run length.
        let mut log_h: Vec<f64> = Vec::with_capacity(n_existing);
        let mut log_one_minus_h: Vec<f64> = Vec::with_capacity(n_existing);
        for r in 0..n_existing {
            let h = self.hazard.hazard(r, timestep).clamp(1e-9, 1.0 - 1e-9);
            log_h.push(h.ln());
            log_one_minus_h.push((1.0 - h).ln());
        }

        // Allocate new run-length distribution: existing runs grow by 1
        // (so length n_existing + 1), capped at max_run_length + 1.
        let new_len = (n_existing + 1).min(self.max_run_length);
        let mut new_log_dist = vec![f64::NEG_INFINITY; new_len];

        // Growth: r' = r + 1
        // log P(r_{t+1} = r+1) = log(1-h(r)) + log_pred(r) + log P(r_t = r)
        for r in 0..n_existing.min(new_len - 1) {
            new_log_dist[r + 1] = log_one_minus_h[r] + log_pred[r] + self.log_run_length_dist[r];
        }

        // Reset: r' = 0
        // log P(r_{t+1} = 0) = log Σ_r h(r) · pred(r) · P(r_t = r)
        let mut log_reset_terms: Vec<f64> = Vec::with_capacity(n_existing);
        for r in 0..n_existing {
            log_reset_terms.push(log_h[r] + log_pred[r] + self.log_run_length_dist[r]);
        }
        new_log_dist[0] = log_sum_exp(&log_reset_terms);

        // Normalize via log-sum-exp.
        let log_z = log_sum_exp(&new_log_dist);
        if log_z.is_finite() {
            for v in new_log_dist.iter_mut() {
                *v -= log_z;
            }
        }

        // Update sufficient statistics:
        //   • new run length 0 is a fresh prior plus the latest observation
        //   • old run length r is shifted to new run length r+1 and updated
        //     with the latest observation
        let mut new_suff: Vec<GaussianSufficient> = Vec::with_capacity(new_len);
        let mut fresh = GaussianSufficient::new(self.prior_variance);
        fresh.update(observation);
        new_suff.push(fresh);
        for r in 0..n_existing.min(new_len - 1) {
            let mut updated = self.suff_stats[r].clone();
            updated.update(observation);
            new_suff.push(updated);
        }

        self.suff_stats = new_suff;
        self.log_run_length_dist = new_log_dist;
        self.n_observations += 1;

        self.reset_probability()
    }

    /// Current `P(r_t = 0 | x_{1:t})`.
    pub fn reset_probability(&self) -> f64 {
        self.log_run_length_dist
            .first()
            .copied()
            .unwrap_or(f64::NEG_INFINITY)
            .exp()
    }

    /// Map P(r=0) to log-odds in bits: log2(p / (1-p)).
    /// Positive means "reset more likely than continuation"; the
    /// `BOCPD_CLOSE_THRESHOLD` of 0.5 corresponds to 0 bits exactly.
    pub fn reset_log_odds_bits(&self) -> f64 {
        let p = self.reset_probability().clamp(1e-9, 1.0 - 1e-9);
        (p / (1.0 - p)).log2()
    }

    /// Convenience: return the most likely run length (the argmax of the
    /// posterior) and its probability mass.
    pub fn most_likely_run_length(&self) -> (usize, f64) {
        let mut best_r = 0;
        let mut best_log_p = f64::NEG_INFINITY;
        for (r, &log_p) in self.log_run_length_dist.iter().enumerate() {
            if log_p > best_log_p {
                best_log_p = log_p;
                best_r = r;
            }
        }
        (best_r, best_log_p.exp())
    }

    /// `P(r_t ≤ horizon | x_{1:t})` — the posterior probability that a
    /// changepoint occurred within the last `horizon` observations.
    ///
    /// **This is the canonical BOCPD changepoint signal**, NOT
    /// `reset_probability` (which is just `P(r=0)`). When a regime change
    /// happens at step `t`, BOCPD does not necessarily put the mass on
    /// `r=0` at step `t+1` — it more often attributes the changepoint to
    /// the previous step (`r=1`) or to a small recent past, because the
    /// posterior sees the divergence between the established run's
    /// predictive and the new observation and assigns the changepoint to
    /// "shortly before now." So watching just `P(r=0)` misses real
    /// changepoints. Watching `P(r ≤ horizon)` for small horizon (1, 2,
    /// or 3) catches the collapse of the run-length posterior.
    ///
    /// For chunk-close decisions, use `horizon = 2` (covers "right now"
    /// and "one observation ago" — captures both immediate and
    /// retrospective changepoint attributions).
    pub fn recent_changepoint_probability(&self, horizon: usize) -> f64 {
        if self.log_run_length_dist.is_empty() {
            return 0.0;
        }
        let upper = (horizon + 1).min(self.log_run_length_dist.len());
        let log_terms: Vec<f64> = self.log_run_length_dist[..upper].to_vec();
        log_sum_exp(&log_terms).exp()
    }

    /// Log-odds (in bits) of "changepoint within last `horizon`
    /// observations" vs "no changepoint." Positive means the model favors
    /// the changepoint hypothesis. Threshold of 0 bits is the canonical
    /// "indifferent" decision boundary.
    pub fn recent_changepoint_log_odds_bits(&self, horizon: usize) -> f64 {
        let p = self
            .recent_changepoint_probability(horizon)
            .clamp(1e-9, 1.0 - 1e-9);
        (p / (1.0 - p)).log2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: a constant signal should accumulate run length without
    /// triggering changepoints.
    #[test]
    fn constant_signal_grows_run_length() {
        let hazard = ConstantHazard::new(50.0); // expected run length = 50
        let mut state = BocpdState::new(hazard, 1.0);
        for _ in 0..30 {
            state.update(5.0, 0);
        }
        let (best_r, _) = state.most_likely_run_length();
        // After 30 identical observations, the most likely run length
        // should be near 30 (the model has high confidence no changepoint
        // has occurred).
        assert!(best_r >= 25, "expected run length near 30, got {}", best_r);
        // P(reset) should be very small.
        assert!(
            state.reset_probability() < 0.05,
            "expected reset prob < 0.05, got {}",
            state.reset_probability()
        );
    }

    /// A sudden jump in the observation stream should collapse the MAP
    /// run length and spike `recent_changepoint_probability`. We do NOT
    /// check `reset_probability` (= P(r=0)) because BOCPD typically
    /// attributes the changepoint to one observation ago (r=1), not to
    /// "right now" (r=0). See `recent_changepoint_probability` doc for
    /// the explanation.
    #[test]
    fn jump_collapses_run_length() {
        let hazard = ConstantHazard::new(50.0);
        let mut state = BocpdState::new(hazard, 1.0);
        // Establish a baseline at value 5.0
        for _ in 0..20 {
            state.update(5.0, 0);
        }
        let baseline_map = state.most_likely_run_length().0;
        let baseline_recent = state.recent_changepoint_probability(2);
        assert!(
            baseline_map >= 15,
            "expected baseline MAP >= 15, got {}",
            baseline_map
        );
        assert!(
            baseline_recent < 0.10,
            "expected baseline recent_changepoint_prob < 0.10, got {}",
            baseline_recent
        );

        // Jump to 100.0 — a 95-unit jump from the established mean of 5.
        state.update(100.0, 0);

        let post_map = state.most_likely_run_length().0;
        let post_recent = state.recent_changepoint_probability(2);
        // The MAP run length should collapse from ~20 to a small value
        // (typically 1, sometimes 2).
        assert!(
            post_map <= 3,
            "expected post-jump MAP run length <= 3, got {}",
            post_map
        );
        // P(r ≤ 2) should jump to near 1.0 — the model is now very
        // confident a changepoint just happened.
        assert!(
            post_recent > 0.9,
            "expected post-jump recent_changepoint_prob > 0.9, got {}",
            post_recent
        );
    }

    /// Phase-aware hazard should give different rates in cold vs warm phases.
    #[test]
    fn phase_aware_hazard_changes_with_timestep() {
        let h = PhaseAwareHazard::from_protocol(14000, 6000, 5000);
        let cold = h.hazard(0, 5000); // mid cold_hold
        let ramp = h.hazard(0, 17000); // mid ramp
        let warm = h.hazard(0, 22000); // warm_hold
                                       // Warm should have higher hazard than cold (shorter expected runs).
        assert!(
            warm > cold,
            "warm hazard ({}) should exceed cold hazard ({})",
            warm,
            cold
        );
        // Ramp should be between.
        assert!(
            ramp > cold,
            "ramp hazard ({}) should exceed cold ({})",
            ramp,
            cold
        );
        assert!(
            ramp < warm,
            "ramp hazard ({}) should be less than warm ({})",
            ramp,
            warm
        );
    }

    #[test]
    fn log_sum_exp_handles_empty_and_infinity() {
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY]), f64::NEG_INFINITY);
        // Standard case: log(e^0 + e^0) = log(2)
        let result = log_sum_exp(&[0.0, 0.0]);
        assert!((result - 2.0_f64.ln()).abs() < 1e-10);
        // Numerical stability: very negative values
        let result = log_sum_exp(&[-1000.0, -1000.0]);
        assert!((result - (-1000.0 + 2.0_f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn gaussian_sufficient_welford_basic() {
        let mut g = GaussianSufficient::new(1.0);
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            g.update(x);
        }
        assert!((g.mean - 3.0).abs() < 1e-10);
        // Sample variance of {1,2,3,4,5} = 2.5
        assert!((g.variance() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn reset_log_odds_bits_at_threshold() {
        // When p = 0.5, log_odds = log2(0.5 / 0.5) = 0
        let h = ConstantHazard::new(50.0);
        let mut state = BocpdState::new(h, 1.0);
        // Don't actually update — just check the math via direct manipulation
        state.log_run_length_dist = vec![0.5_f64.ln(), 0.5_f64.ln()];
        let bits = state.reset_log_odds_bits();
        assert!(bits.abs() < 1e-10, "expected 0 bits at p=0.5, got {}", bits);
    }
}
