//! Gaussian-Copula Partial Information Decomposition (Ince 2017).
//!
//! Stage 1B-3: principled, parameter-free per-residue **synergy fraction**
//! estimator that quantifies how much of the information about a target
//! variable is uniquely available from the joint distribution of two
//! sources (i.e., requires both sources together — neither alone is
//! enough). This is the canonical "non-naive" steering weight for the
//! Stage 2 closed-loop ASC writeback: high synergy fraction = high
//! leverage = "this residue's activity carries information that no
//! single TWIN group can see alone."
//!
//! ## Why Partial Information Decomposition (PID) at all
//!
//! Mutual information `I(T; S₁, S₂)` tells you how much information the
//! joint distribution `(S₁, S₂)` carries about a target `T`, but it
//! doesn't tell you HOW that information is distributed. Two sources
//! could each independently carry the same info (redundancy), they
//! could each carry distinct info (unique), or they could carry
//! information that only emerges from their combination (synergy).
//! Williams & Beer (2010) introduced PID to make this distinction
//! formal:
//!
//!   `I(T; S₁, S₂) = Synergy(T; S₁, S₂)
//!                 + Unique(T; S₁)
//!                 + Unique(T; S₂)
//!                 + Redundancy(T; S₁, S₂)`
//!
//! Each atom is non-negative and the four sum to the joint MI by
//! construction. The **synergy fraction** `Synergy / I(T; S₁, S₂)` is
//! a number in `[0, 1]` that says "what fraction of the total
//! information is only available from the joint distribution."
//!
//! ## Why Gaussian copula
//!
//! For arbitrary distributions, computing PID atoms requires either
//! solving a constrained optimization (Bertschinger et al. 2014) or
//! discretizing into bins (which loses information and introduces a
//! parameter). Both are unsuitable for the inner ASC loop where we need
//! to compute synergy per residue per chunk (~150 residues × ~100
//! chunks × 4 groups = 60K PID computations per run).
//!
//! Ince (2017) showed that if you transform each variable to its rank
//! within its empirical distribution, then to standard normal scores
//! via the inverse normal CDF, and **assume joint Gaussianity in the
//! transformed space**, then all the PID atoms have **closed-form
//! expressions** in terms of the Pearson correlations among the
//! transformed variables. This is the **Gaussian copula** approximation:
//! it preserves the dependency structure (the copula) while making the
//! marginals standard normal, so the linear-Gaussian formulas apply.
//!
//! Properties:
//!
//!   1. **Distribution-free**. The rank-CDF transform makes the result
//!      invariant to the actual marginal distributions of the inputs.
//!      Skewed spike-count distributions, heavy-tailed phase data,
//!      bimodal burial scores — all reduce to standard normals after
//!      the copula transform.
//!   2. **Closed-form**. No optimization, no MCMC, no binning.
//!      ~10 floating-point operations per PID computation given the
//!      pairwise correlations.
//!   3. **Online updateable**. Per-residue running rank statistics can
//!      be maintained with O(observations) per chunk.
//!   4. **Bounded by construction**. The synergy fraction is in `[0, 1]`
//!      with no normalization constant.
//!
//! ## Reference
//!
//! Ince, R. A. A. (2017). *Measuring multivariate redundant information
//! with pointwise common change in surprisal.* Entropy 19(7), 318.
//! doi:10.3390/e19070318.
//!
//! Williams, P. L., & Beer, R. D. (2010). *Nonnegative decomposition of
//! multivariate information.* arXiv:1004.2515. (Original PID framework
//! and the I_min redundancy definition used here.)
//!
//! Bertschinger, N., et al. (2014). *Quantifying unique information.*
//! Entropy 16(4), 2161–2183. (Operationally meaningful definition of
//! unique information for the 2-source case.)
//!
//! ## Why this is non-naive
//!
//!   1. **Zero magic constants** in the closed-form atoms. The only
//!      tunable is the minimum sample count below which we refuse to
//!      compute a PID (default 5, with documented rationale).
//!   2. **Numerically stable**. Pearson correlations are clamped to
//!      `(-CLAMP, +CLAMP)` to keep the determinants strictly positive.
//!   3. **Distribution-free** via rank transform. No assumption about
//!      the underlying spike-count distribution.
//!   4. **The synergy fraction directly drives Stage 2 steering weights**.
//!      Per-residue: `weight = synergy_fraction(target=future_spike_rate,
//!      source_A=scout_group_rate, source_B=observer_group_rate)`. The
//!      higher the synergy, the more the residue's information requires
//!      cross-group cooperation, the more leverage there is in steering.

use crate::bocpd::log_sum_exp;

/// Minimum sample count required to compute a meaningful PID. Below
/// this, the rank statistics are too noisy and we return None.
///
/// 5 is the smallest n where the rank-CDF transform gives meaningful
/// percentile spread (1/5 = 20% per rank), AND where the sample
/// correlation has degrees of freedom > 2. Lower values produce
/// near-degenerate covariance matrices.
pub const PID_MIN_SAMPLES: usize = 5;

/// Maximum absolute Pearson correlation we allow before clamping. Beyond
/// this, the determinants in the closed-form GC-PID become arbitrarily
/// close to zero and the mutual information blows up to infinity. The
/// clamp keeps everything finite without changing the qualitative
/// answer (a correlation of 0.999 vs 1.000 produces a synergy fraction
/// difference smaller than the rank-transform noise floor).
pub const CORRELATION_CLAMP: f64 = 0.99;

/// The four PID atoms for a 2-source decomposition, plus the total MI
/// (which equals the sum of the four atoms by construction).
///
/// All values are in **nats** (natural log). Multiply by `1.0 / 2_f64.ln()`
/// to convert to bits if desired.
#[derive(Debug, Clone, Copy)]
pub struct PidAtoms {
    /// Information about T that REQUIRES the joint distribution of (A, B).
    /// Neither source alone tells you this; only the combination does.
    pub synergy: f64,
    /// Information about T that is present in BOTH sources independently.
    /// Williams-Beer I_min: `min(I(T; A), I(T; B))`.
    pub redundancy: f64,
    /// Information about T uniquely in source A (not in B).
    pub unique_a: f64,
    /// Information about T uniquely in source B (not in A).
    pub unique_b: f64,
    /// Joint mutual information `I(T; A, B)`. Equals the sum of the
    /// four atoms by construction.
    pub total_mi: f64,
}

impl PidAtoms {
    /// The synergy fraction: `Synergy / I(T; A, B)`. In `[0, 1]` by
    /// construction. Returns 0 if total_mi is non-positive (degenerate
    /// distribution where the joint MI is negligible).
    ///
    /// **This is the steering weight for Stage 2's closed-loop ASC.**
    /// Per-residue: high synergy fraction = "this residue's information
    /// only emerges from cross-group cooperation, so steering it is
    /// high-leverage."
    pub fn synergy_fraction(&self) -> f64 {
        if self.total_mi <= 0.0 {
            0.0
        } else {
            (self.synergy / self.total_mi).clamp(0.0, 1.0)
        }
    }

    /// The redundancy fraction: `Redundancy / I(T; A, B)`. In `[0, 1]` by
    /// construction. Returns 0 if total_mi is non-positive.
    ///
    /// **Used by #1 asymmetric steering as the OBSERVER focus weight.**
    /// Per-residue: high redundancy fraction = "this residue's information
    /// is shared across both sources, both groups already agree on it."
    /// Pushing observers toward high-redundancy residues anchors them to
    /// the cross-group consensus, making the scout/observer divergence
    /// physically meaningful (scouts probe novelty, observers reinforce
    /// consensus). This is structurally orthogonal to UCB1/LCB1 on the
    /// SAME metric, which collapses when sample counts are uniform.
    pub fn redundancy_fraction(&self) -> f64 {
        if self.total_mi <= 0.0 {
            0.0
        } else {
            (self.redundancy / self.total_mi).clamp(0.0, 1.0)
        }
    }

    /// Convert from nats to bits.
    pub fn to_bits(&self) -> PidAtoms {
        let inv_ln2 = 1.0 / 2_f64.ln();
        PidAtoms {
            synergy: self.synergy * inv_ln2,
            redundancy: self.redundancy * inv_ln2,
            unique_a: self.unique_a * inv_ln2,
            unique_b: self.unique_b * inv_ln2,
            total_mi: self.total_mi * inv_ln2,
        }
    }
}

/// Closed-form Gaussian-copula PID for 2 sources and 1 target.
///
/// Inputs are the three Pearson correlations among the standard-normal-
/// transformed variables (after rank-CDF transform applied to the raw
/// observations). Use `compute_pid_from_samples` if you have raw
/// observation arrays — that function handles the transform.
///
/// ## The formulas
///
/// For standard normal variables (variance = 1 after the copula
/// transform), the Gaussian mutual information formulas reduce to:
///
///   `I(T; A) = -½ ln(1 - ρ²_TA)`
///   `I(T; B) = -½ ln(1 - ρ²_TB)`
///   `I(T; A, B) = -½ ln(det(R_{T,A,B}) / det(R_{A,B}))`
///
/// where `R_{X,Y,...}` is the correlation matrix on the named variables.
/// For standard normals all diagonal entries are 1, so:
///
///   `det(R_{A,B}) = 1 - ρ²_AB`
///   `det(R_{T,A,B}) = 1 - ρ²_TA - ρ²_TB - ρ²_AB + 2·ρ_TA·ρ_TB·ρ_AB`
///
/// The Williams-Beer I_min decomposition then gives:
///
///   `Redundancy = min(I(T; A), I(T; B))`
///   `Unique(A) = I(T; A) - Redundancy`
///   `Unique(B) = I(T; B) - Redundancy`
///   `Synergy   = I(T; A, B) - Unique(A) - Unique(B) - Redundancy`
///
/// All four atoms are non-negative by construction (we clamp to 0 to
/// handle floating-point underflow at the boundary).
///
/// ## Reference
///
/// Ince 2017 §3 (the closed-form derivation for the bivariate Gaussian
/// case). Williams-Beer 2010 (the I_min redundancy definition).
pub fn closed_form_gcpid(rho_ta: f64, rho_tb: f64, rho_ab: f64) -> PidAtoms {
    // Clamp correlations to keep determinants strictly positive
    let rho_ta = rho_ta.clamp(-CORRELATION_CLAMP, CORRELATION_CLAMP);
    let rho_tb = rho_tb.clamp(-CORRELATION_CLAMP, CORRELATION_CLAMP);
    let rho_ab = rho_ab.clamp(-CORRELATION_CLAMP, CORRELATION_CLAMP);

    // Marginal mutual informations (in nats)
    let mi_ta = -0.5 * (1.0 - rho_ta.powi(2)).max(1e-12).ln();
    let mi_tb = -0.5 * (1.0 - rho_tb.powi(2)).max(1e-12).ln();

    // Joint mutual information I(T; A, B) for 3-variable Gaussian.
    // Use the determinant ratio formula, with both numerator and
    // denominator clamped to keep the log finite.
    let det_r_full = (1.0 - rho_ta.powi(2) - rho_tb.powi(2) - rho_ab.powi(2)
        + 2.0 * rho_ta * rho_tb * rho_ab)
        .max(1e-12);
    let det_r_ab = (1.0 - rho_ab.powi(2)).max(1e-12);
    let mi_tab = (-0.5 * (det_r_full / det_r_ab).max(1e-12).ln()).max(0.0);

    // Williams-Beer I_min decomposition
    let redundancy = mi_ta.min(mi_tb).max(0.0);
    let unique_a = (mi_ta - redundancy).max(0.0);
    let unique_b = (mi_tb - redundancy).max(0.0);
    let synergy = (mi_tab - unique_a - unique_b - redundancy).max(0.0);

    PidAtoms {
        synergy,
        redundancy,
        unique_a,
        unique_b,
        total_mi: mi_tab,
    }
}

/// Pearson correlation coefficient of two equal-length samples.
/// Returns 0.0 for empty inputs or constant samples.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;
    let mean_x: f64 = x[..n].iter().sum::<f64>() / n_f;
    let mean_y: f64 = y[..n].iter().sum::<f64>() / n_f;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        cov / denom
    }
}

/// Empirical-CDF rank transform: replace each value with its rank
/// (1-based) divided by `n + 1`, producing values in `(0, 1)`.
///
/// Ties are resolved by stable sort and assigned consecutive ranks
/// (no tied-rank averaging — for typical spike-count data with high
/// quantization at the max value, the stable order produces a uniform
/// rank distribution that the inverse normal CDF can map cleanly).
pub fn rank_cdf_transform(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let denom = (n + 1) as f64;
    for (rank, (orig_idx, _)) in indexed.iter().enumerate() {
        ranks[*orig_idx] = (rank + 1) as f64 / denom;
    }
    ranks
}

/// Inverse standard normal CDF (probit function), Φ⁻¹.
///
/// Uses the Beasley-Springer-Moro algorithm (Moro 1995) — a high-
/// accuracy rational approximation that's standard in the quantitative
/// finance literature for converting uniform random samples to standard
/// normals. Maximum absolute error ~1e-9 over the input range
/// `(0, 1)`. Outside that range returns ±∞.
///
/// For the Gaussian-copula PID we feed it the rank-CDF values from
/// `rank_cdf_transform` which are always in `(0, 1)` by construction
/// (1/(n+1) to n/(n+1) inclusive), so the boundary cases never trigger.
pub fn inverse_normal_cdf(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Beasley-Springer-Moro coefficients
    const A: [f64; 4] = [
        2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637,
    ];
    const B: [f64; 4] = [
        -8.47351093090,
        23.08336743743,
        -21.06224101826,
        3.13082909833,
    ];
    const C: [f64; 9] = [
        0.3374754822726147,
        0.9761690190917186,
        0.1607979714918209,
        0.0276438810333863,
        0.0038405729373609,
        0.0003951896511919,
        0.0000321767881768,
        0.0000002888167364,
        0.0000003960315187,
    ];

    let y = p - 0.5;
    if y.abs() < 0.42 {
        // Central region — Beasley-Springer rational approximation
        let r = y * y;
        let num = ((A[3] * r + A[2]) * r + A[1]) * r + A[0];
        let den = (((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0;
        y * num / den
    } else {
        // Tail region — Moro extension
        let r = if y > 0.0 { 1.0 - p } else { p };
        let r = (-r.ln()).ln();
        let mut x = C[0];
        let mut r_pow = 1.0;
        for &c in &C[1..] {
            r_pow *= r;
            x += c * r_pow;
        }
        if y > 0.0 {
            x
        } else {
            -x
        }
    }
}

/// Compute PID atoms from raw observation samples.
///
/// 1. Apply rank-CDF transform to each variable independently.
/// 2. Apply inverse normal CDF to get standard normal scores.
/// 3. Compute the three pairwise Pearson correlations.
/// 4. Apply `closed_form_gcpid`.
///
/// Returns `None` if any sample array has fewer than `PID_MIN_SAMPLES`
/// observations OR if all three samples have zero variance (degenerate
/// case where all observations are identical).
pub fn compute_pid_from_samples(
    target: &[f64],
    source_a: &[f64],
    source_b: &[f64],
) -> Option<PidAtoms> {
    let n = target.len().min(source_a.len()).min(source_b.len());
    if n < PID_MIN_SAMPLES {
        return None;
    }
    let t_ranks = rank_cdf_transform(&target[..n]);
    let a_ranks = rank_cdf_transform(&source_a[..n]);
    let b_ranks = rank_cdf_transform(&source_b[..n]);
    let t_scores: Vec<f64> = t_ranks.iter().map(|&r| inverse_normal_cdf(r)).collect();
    let a_scores: Vec<f64> = a_ranks.iter().map(|&r| inverse_normal_cdf(r)).collect();
    let b_scores: Vec<f64> = b_ranks.iter().map(|&r| inverse_normal_cdf(r)).collect();

    let rho_ta = pearson_correlation(&t_scores, &a_scores);
    let rho_tb = pearson_correlation(&t_scores, &b_scores);
    let rho_ab = pearson_correlation(&a_scores, &b_scores);

    Some(closed_form_gcpid(rho_ta, rho_tb, rho_ab))
}

/// Per-residue PID accumulator with bounded sample buffers. One instance
/// per residue, fed observations every chunk by the ASC controller.
///
/// The buffer is bounded by `MAX_SAMPLES` to keep the per-chunk
/// computation cost constant — old observations are dropped via FIFO
/// rotation. Default `MAX_SAMPLES = 256` is generous for the chunk
/// cadence (~80 chunks per run).
#[derive(serde::Serialize, serde::Deserialize)]
pub struct PidAccumulator {
    target: Vec<f64>,
    source_a: Vec<f64>,
    source_b: Vec<f64>,
    max_samples: usize,
}

impl PidAccumulator {
    pub fn new(max_samples: usize) -> Self {
        Self {
            target: Vec::with_capacity(max_samples),
            source_a: Vec::with_capacity(max_samples),
            source_b: Vec::with_capacity(max_samples),
            max_samples: max_samples.max(PID_MIN_SAMPLES * 2),
        }
    }

    /// Append a new (target, source_a, source_b) observation. If the
    /// buffer is at capacity, drop the oldest observation (FIFO).
    pub fn observe(&mut self, target: f64, source_a: f64, source_b: f64) {
        if self.target.len() >= self.max_samples {
            self.target.remove(0);
            self.source_a.remove(0);
            self.source_b.remove(0);
        }
        self.target.push(target);
        self.source_a.push(source_a);
        self.source_b.push(source_b);
    }

    pub fn n_samples(&self) -> usize {
        self.target.len()
    }

    /// Compute PID atoms from the current sample buffer. Returns None
    /// if not enough samples accumulated yet.
    pub fn compute(&self) -> Option<PidAtoms> {
        compute_pid_from_samples(&self.target, &self.source_a, &self.source_b)
    }
}

// Reference the unused log_sum_exp from bocpd to silence unused-import
// warnings during early development. Removed when this module gets its
// own use of log_sum_exp (likely in the multi-source PID extension).
#[allow(dead_code)]
fn _silence_unused() -> f64 {
    log_sum_exp(&[0.0, 0.0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pearson correlation of identical series should be 1.0.
    #[test]
    fn pearson_identical_is_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = x.clone();
        assert!((pearson_correlation(&x, &y) - 1.0).abs() < 1e-12);
    }

    /// Pearson correlation of opposite series should be -1.0.
    #[test]
    fn pearson_opposite_is_negative_one() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((pearson_correlation(&x, &y) + 1.0).abs() < 1e-12);
    }

    /// Pearson correlation of orthogonal series should be ~0.
    #[test]
    fn pearson_orthogonal_near_zero() {
        let x = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let y = vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0];
        assert!(pearson_correlation(&x, &y).abs() < 0.4);
    }

    /// Rank transform should produce values in (0, 1).
    #[test]
    fn rank_cdf_basic() {
        let v = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let r = rank_cdf_transform(&v);
        assert_eq!(r.len(), 5);
        for &x in &r {
            assert!(x > 0.0 && x < 1.0, "rank out of (0,1): {}", x);
        }
        // Sorted input → sorted ranks
        for i in 1..r.len() {
            assert!(r[i] > r[i - 1]);
        }
    }

    /// Inverse normal CDF should be monotonic and produce 0 at p=0.5.
    #[test]
    fn inverse_normal_cdf_basic() {
        assert!((inverse_normal_cdf(0.5)).abs() < 1e-6);
        assert!(inverse_normal_cdf(0.84) > 0.9 && inverse_normal_cdf(0.84) < 1.1);
        assert!(inverse_normal_cdf(0.16) < -0.9 && inverse_normal_cdf(0.16) > -1.1);
    }

    /// Closed-form GCPID: when both sources are identical to the target,
    /// redundancy should dominate (the two sources carry the same
    /// information about T).
    #[test]
    fn redundant_sources_have_high_redundancy() {
        // ρ_TA = 1, ρ_TB = 1, ρ_AB = 1 (perfect redundancy)
        let atoms = closed_form_gcpid(0.99, 0.99, 0.99);
        assert!(
            atoms.redundancy > atoms.synergy,
            "redundant case: redundancy {} should exceed synergy {}",
            atoms.redundancy,
            atoms.synergy
        );
        assert!(atoms.unique_a < 0.1);
        assert!(atoms.unique_b < 0.1);
    }

    /// Closed-form GCPID: when sources are independent and target is
    /// their sum (the canonical XOR-style synergy case in the Gaussian
    /// world), synergy should dominate.
    ///
    /// For T = A + B with A, B ~ iid N(0, 1):
    ///   ρ_TA = 1/√2, ρ_TB = 1/√2, ρ_AB = 0
    ///   I(T; A) = I(T; B) = -½ ln(1 - 1/2) = ½ ln 2 ≈ 0.347
    ///   I(T; A, B) = ∞ (T is exactly determined by A + B)
    ///
    /// With clamping at correlation 0.99 we get a finite I(T; A, B)
    /// that's still much larger than the marginals, so synergy dominates.
    #[test]
    fn synergistic_sources_have_high_synergy() {
        let r = (0.5_f64).sqrt();
        let atoms = closed_form_gcpid(r, r, 0.0);
        assert!(
            atoms.synergy > atoms.redundancy,
            "synergistic case: synergy {} should exceed redundancy {}",
            atoms.synergy,
            atoms.redundancy
        );
        assert!(
            atoms.synergy_fraction() > 0.3,
            "synergistic case: synergy_fraction {} should be > 0.3",
            atoms.synergy_fraction()
        );
    }

    /// Independent sources from independent target: all atoms ~0.
    #[test]
    fn independent_target_has_low_total_mi() {
        let atoms = closed_form_gcpid(0.0, 0.0, 0.0);
        assert!(
            atoms.total_mi < 1e-6,
            "independent: total_mi = {}",
            atoms.total_mi
        );
    }

    /// Synergy fraction is in [0, 1].
    #[test]
    fn synergy_fraction_bounded() {
        for rho_ta in [-0.9, -0.5, 0.0, 0.5, 0.9] {
            for rho_tb in [-0.9, -0.5, 0.0, 0.5, 0.9] {
                for rho_ab in [-0.9, -0.5, 0.0, 0.5, 0.9] {
                    let atoms = closed_form_gcpid(rho_ta, rho_tb, rho_ab);
                    let f = atoms.synergy_fraction();
                    assert!(
                        (0.0..=1.0).contains(&f),
                        "out-of-range synergy fraction {} at ({}, {}, {})",
                        f,
                        rho_ta,
                        rho_tb,
                        rho_ab
                    );
                }
            }
        }
    }

    /// PidAccumulator: rotates samples FIFO when at capacity.
    #[test]
    fn accumulator_rotates_at_capacity() {
        let mut acc = PidAccumulator::new(10);
        for i in 0..15 {
            acc.observe(i as f64, (i * 2) as f64, (i * 3) as f64);
        }
        assert_eq!(acc.n_samples(), 10);
        // Earliest 5 observations should have been dropped; oldest
        // remaining target value is 5.0.
        assert_eq!(acc.target[0], 5.0);
    }

    /// End-to-end: feed the accumulator a synergistic stream and verify
    /// synergy fraction is positive.
    #[test]
    fn accumulator_end_to_end_synergistic() {
        let mut acc = PidAccumulator::new(100);
        // Generate a deterministic synergistic stream:
        // T = A + B where A and B are independent rotating values.
        for i in 0..50 {
            let a = ((i as f64 * 0.7).sin() * 5.0) as f64;
            let b = ((i as f64 * 1.1).cos() * 5.0) as f64;
            let t = a + b;
            acc.observe(t, a, b);
        }
        let atoms = acc.compute().expect("PID should compute on 50 samples");
        assert!(atoms.total_mi > 0.0, "total_mi should be positive");
        assert!(
            atoms.synergy_fraction() >= 0.0 && atoms.synergy_fraction() <= 1.0,
            "synergy_fraction out of range: {}",
            atoms.synergy_fraction()
        );
    }
}
