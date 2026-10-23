//! §8 — STAGE 3: information-proportional bit allocation (reverse water-filling).
//!
//! Codebook resolution is allocated per field **in proportion to measured information, not
//! frequency** (§8) — "unique tokens, 1:1 representative" made quantitative. For a field `f`
//! with information score `s_f` (bits), reverse water-filling against a global budget `B`:
//!
//! ```text
//!   b_f = clamp( s_f − level , b_min , b_max ) ,   choose `level` s.t.  Σ_f b_f = B
//! ```
//!
//! where `s_f = ½·log2(σ_f²)` for the engine variance channels (`te_windowed_variance`,
//! `trajectory_entropy`) and `s_f = H_f` (empirical code entropy) for the non-Gaussian fields —
//! the two §8 forms unified by the choice of `s_f` (the only difference is the `level` offset:
//! ½·log2(D) vs log2(D')).
//!
//! Two §8 guards ride on top:
//!   * **Decision-critical fields are PINNED** at full native resolution (`b_min = b_max`)
//!     regardless of entropy — a low-entropy `rhat`/`ess`/`label_confidence` must never be
//!     coarsened across its threshold (INV-8). Metadata-only fields, conversely, collapse to
//!     `b_min` so they never dominate the budget.
//!   * **Confidence gate** (§8): `g = label_confidence · student_weight`, with `is_bimodal ⇒
//!     g ← min(g, G_BIMODAL_CAP)`, so high-consensus records sharpen and bimodal / low-confidence
//!     records coarsen (merge). Never applied to pinned fields.
//!
//! This is a pure ANALYSIS layer: it produces an [`AllocationPlan`] and never mutates encode
//! behaviour by default (no regression — existing tokens are byte-identical). §14.5 alignment
//! (`corr(bits_f, H_f) > 0`) is exposed as [`AllocationPlan::bits_entropy_correlation`].

use std::collections::HashMap;

use crate::encoder::Encoder;
use crate::quant::build_field_quantizer;
use crate::types::{ArtifactFamily, DecisionSpec, InfoSource, Quantizer};

/// Bimodal cap on the confidence gate `g` (§8): a bimodal record never sharpens past this.
pub const G_BIMODAL_CAP: f64 = 0.5;

/// Default floor for a non-decision-critical field (always keep at least a binary distinction;
/// metadata fields collapse here so they never dominate the budget).
pub const B_MIN_DEFAULT: u8 = 1;

/// One field's measured-information summary + its native budget bounds.
#[derive(Clone, Debug)]
pub struct FieldInfo {
    pub family: ArtifactFamily,
    pub field: &'static str,
    /// Empirical Shannon entropy `H_f` (bits) of the fitted code distribution on the samples.
    pub h_bits: f64,
    /// Population variance `σ_f²` of the raw samples (the Gaussian-channel proxy).
    pub variance: f64,
    /// The §8 information score `s_f` used in water-filling (`H_f` or `½·log2 σ²`).
    pub score_bits: f64,
    pub b_min: u8,
    pub b_max: u8,
    /// `DecisionSpec::Threshold` ⇒ pinned at `b_max` regardless of entropy (INV-8).
    pub decision_critical: bool,
    pub n_samples: usize,
}

/// Allocation result for one field.
#[derive(Clone, Debug, PartialEq)]
pub struct Allocation {
    pub family: ArtifactFamily,
    pub field: &'static str,
    /// `b_f` from water-filling (pre confidence-gate).
    pub bits: u8,
    pub h_bits: f64,
    /// Decision-critical field pinned at full native resolution.
    pub pinned: bool,
}

/// The realized allocation across a field set (deterministic: follows the input order).
#[derive(Clone, Debug)]
pub struct AllocationPlan {
    pub allocations: Vec<Allocation>,
    /// The water level (`½·log2 D` / `log2 D'`) the bisection settled on.
    pub water_level: f64,
    pub total_bits: u32,
}

impl AllocationPlan {
    pub fn bits_for(&self, family: ArtifactFamily, field: &str) -> Option<u8> {
        self.allocations
            .iter()
            .find(|a| a.family == family && a.field == field)
            .map(|a| a.bits)
    }

    /// §14.5 information-alignment statistic: Pearson correlation between allocated bits and the
    /// empirical entropy `H_f`, over the NON-pinned fields (pinned fields carry a constant budget
    /// and would only add degenerate variance). Returns 0.0 when it is undefined (< 2 fields, or
    /// either series constant).
    pub fn bits_entropy_correlation(&self) -> f64 {
        let pairs: Vec<(f64, f64)> = self
            .allocations
            .iter()
            .filter(|a| !a.pinned)
            .map(|a| (a.bits as f64, a.h_bits))
            .collect();
        pearson(&pairs)
    }

    /// One line per field — the realized allocation, for telemetry (§8 "log the realized
    /// allocation").
    pub fn summary(&self) -> String {
        let mut s = format!(
            "alloc: {} fields, {} bits total, level {:.3}, corr(b,H)={:.3}\n",
            self.allocations.len(),
            self.total_bits,
            self.water_level,
            self.bits_entropy_correlation()
        );
        for a in &self.allocations {
            s += &format!(
                "  {:?}/{:<26} b={:<2} H={:.3}{}\n",
                a.family,
                a.field,
                a.bits,
                a.h_bits,
                if a.pinned { " [pinned]" } else { "" }
            );
        }
        s
    }
}

/// The confidence gate `g` (§8). `is_bimodal` caps it; the result is clamped to [0, 1].
pub fn confidence_gate(label_confidence: f64, student_weight: f64, is_bimodal: bool) -> f64 {
    let mut g = (label_confidence * student_weight).clamp(0.0, 1.0);
    if is_bimodal {
        g = g.min(G_BIMODAL_CAP);
    }
    g
}

/// Apply the confidence gate to a field's allocated bits: `b_eff = round(b_f · g)`, never below
/// `b_min` (a non-degenerate codebook) and never above `b_f` (the gate only coarsens). Decision-
/// critical fields must bypass this (the caller does not gate pinned fields).
pub fn gated_bits(b_f: u8, b_min: u8, g: f64) -> u8 {
    let v = ((b_f as f64) * g).round();
    (v as u8).clamp(b_min.min(b_f), b_f)
}

/// Measure one scalar field's information from its real routed samples. `None` for non-scalar
/// geometries (lattice / SMILES / opaque have no `f64`-scalar codebook to budget) or unknown
/// fields or empty samples.
pub fn measure_field(
    enc: &Encoder,
    family: ArtifactFamily,
    field: &'static str,
    samples: &[f64],
) -> Option<FieldInfo> {
    if samples.is_empty() {
        return None;
    }
    let spec = enc.registry().lookup(family, field)?;
    // build a fresh quantizer for measurement so the Encoder's own state is never mutated.
    let mut q = build_field_quantizer(spec)?;
    q.fit(samples);
    let h_bits = code_entropy_bits(q.as_ref(), samples);
    let variance = population_variance(samples);
    let decision_critical = matches!(spec.decision_relevance, DecisionSpec::Threshold { .. });
    let b_max = q.bit_budget();
    let b_min = if decision_critical {
        b_max
    } else {
        B_MIN_DEFAULT.min(b_max)
    };
    let score_bits = match spec.info_source {
        // engine variance channels → Gaussian rate ½·log2(σ²); guard tiny/zero variance.
        InfoSource::TeWindowedVariance | InfoSource::TrajectoryEntropy => {
            0.5 * (variance.max(1e-12)).log2()
        }
        // everything else → non-Gaussian empirical entropy.
        _ => h_bits,
    };
    Some(FieldInfo {
        family,
        field,
        h_bits,
        variance,
        score_bits,
        b_min,
        b_max,
        decision_critical,
        n_samples: samples.len(),
    })
}

/// Measure every `(family, field) → samples` entry into a deterministic `FieldInfo` list (input
/// iteration order is the caller's; sort upstream if a stable order across maps is needed).
pub fn measure_corpus(
    enc: &Encoder,
    family: ArtifactFamily,
    samples: &HashMap<&'static str, Vec<f64>>,
) -> Vec<FieldInfo> {
    // deterministic field order: registry slot order for this family.
    let mut out = Vec::new();
    for spec in enc.registry().iter().filter(|s| s.family == family) {
        if let Some(s) = samples.get(spec.field) {
            if let Some(fi) = measure_field(enc, family, spec.field, s) {
                out.push(fi);
            }
        }
    }
    out
}

/// Reverse water-filling: allocate `total_budget_bits` across `infos`. Decision-critical fields
/// are pinned at `b_max` first; the remainder is water-filled over the rest by bisecting the
/// `level` so the non-pinned bit sum meets the residual budget.
pub fn allocate(infos: &[FieldInfo], total_budget_bits: u32) -> AllocationPlan {
    let pinned_bits: u32 = infos
        .iter()
        .filter(|fi| fi.decision_critical)
        .map(|fi| fi.b_max as u32)
        .sum();
    let residual = total_budget_bits.saturating_sub(pinned_bits);

    let free: Vec<&FieldInfo> = infos.iter().filter(|fi| !fi.decision_critical).collect();
    let (free_bits, level) = water_fill(&free, residual);

    // reassemble in the original field order.
    let mut free_iter = free_bits.into_iter();
    let mut allocations = Vec::with_capacity(infos.len());
    let mut total = 0u32;
    for fi in infos {
        let (bits, pinned) = if fi.decision_critical {
            (fi.b_max, true)
        } else {
            (free_iter.next().unwrap_or(fi.b_min), false)
        };
        total += bits as u32;
        allocations.push(Allocation {
            family: fi.family,
            field: fi.field,
            bits,
            h_bits: fi.h_bits,
            pinned,
        });
    }
    AllocationPlan {
        allocations,
        water_level: level,
        total_bits: total,
    }
}

/// Bisection on the water level for the free (non-pinned) fields. `bits(level)` is monotonically
/// non-increasing in `level`, so we find the smallest level whose bit sum does not exceed
/// `budget` (the most generous in-budget allocation).
fn water_fill(infos: &[&FieldInfo], budget: u32) -> (Vec<u8>, f64) {
    if infos.is_empty() {
        return (Vec::new(), 0.0);
    }
    let bits_at = |level: f64| -> Vec<u8> {
        infos
            .iter()
            .map(|fi| {
                // Information cap (standard rate-distortion): never allocate a field more
                // resolution than its own information `ceil(score)` — otherwise a generous budget
                // drives `level` negative and a zero-entropy metadata field (e.g. a constant
                // total_steps) would inflate above `b_min` and dominate. This enforces the §8
                // principle (resolution ∝ measured information) as a hard per-field ceiling on top
                // of the geometry's `b_max`.
                let cap =
                    (fi.score_bits.ceil() as i64).clamp(fi.b_min as i64, fi.b_max as i64) as u8;
                let b = (fi.score_bits - level).round();
                b.clamp(fi.b_min as f64, cap as f64) as u8
            })
            .collect()
    };
    let sum_at = |level: f64| -> u32 { bits_at(level).iter().map(|&b| b as u32).sum() };

    let s_max = infos
        .iter()
        .map(|fi| fi.score_bits)
        .fold(f64::NEG_INFINITY, f64::max);
    let s_min = infos
        .iter()
        .map(|fi| fi.score_bits)
        .fold(f64::INFINITY, f64::min);
    // level ≤ lo ⇒ every field at b_max (max sum); level ≥ hi ⇒ every field at b_min (min sum).
    let mut lo = s_min - 64.0;
    let mut hi = s_max + 64.0;
    // if even the max-resolution sum fits the budget, no coarsening is needed.
    if sum_at(lo) <= budget {
        return (bits_at(lo), lo);
    }
    // smallest level with sum ≤ budget (sum is non-increasing in level).
    for _ in 0..96 {
        let mid = 0.5 * (lo + hi);
        if sum_at(mid) > budget {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (bits_at(hi), hi)
}

// --- small numeric helpers ------------------------------------------------------------------

fn population_variance(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    if n == 0.0 {
        return 0.0;
    }
    let mean = samples.iter().sum::<f64>() / n;
    samples
        .iter()
        .map(|&x| (x - mean) * (x - mean))
        .sum::<f64>()
        / n
}

/// Shannon entropy (bits) of the code distribution the (fitted) quantizer assigns to the samples.
fn code_entropy_bits(q: &dyn Quantizer, samples: &[f64]) -> f64 {
    let mut hist: HashMap<u32, u64> = HashMap::new();
    for &v in samples {
        *hist.entry(q.encode(v).0).or_insert(0) += 1;
    }
    let n = samples.len() as f64;
    let mut h = 0.0;
    for &c in hist.values() {
        let p = c as f64 / n;
        h -= p * p.log2();
    }
    h.max(0.0)
}

/// Pearson correlation of `(x, y)` pairs; 0.0 if undefined (< 2 points or a constant series).
fn pearson(pairs: &[(f64, f64)]) -> f64 {
    let n = pairs.len() as f64;
    if pairs.len() < 2 {
        return 0.0;
    }
    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for &(x, y) in pairs {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx) * (x - mx);
        syy += (y - my) * (y - my);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return 0.0;
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn enc() -> Encoder {
        Encoder::new()
    }

    #[test]
    fn confidence_gate_coarsens_bimodal_and_low_confidence() {
        // high consensus sharpens; low-confidence and bimodal coarsen (§8).
        let g_hi = confidence_gate(0.95, 0.9, false);
        let g_lo = confidence_gate(0.30, 0.9, false);
        let g_bi = confidence_gate(0.95, 0.9, true);
        assert!(g_hi > g_lo, "low confidence must coarsen: {g_hi} !> {g_lo}");
        assert!(
            g_bi <= G_BIMODAL_CAP,
            "bimodal capped at {G_BIMODAL_CAP}, got {g_bi}"
        );
        assert!(g_bi < g_hi, "bimodal must coarsen vs high-consensus");
        // gated bits never exceed the input and never drop below the floor.
        assert_eq!(gated_bits(8, 1, g_hi), (8.0 * g_hi).round() as u8);
        assert!(gated_bits(8, 1, g_bi) < 8);
        assert!(
            gated_bits(8, 1, 0.0) >= 1,
            "never a degenerate 0-bit codebook"
        );
    }

    #[test]
    fn water_filling_respects_budget_and_bounds() {
        // three synthetic non-critical fields with distinct entropies → bits track entropy and the
        // total stays within budget.
        let mk = |field: &'static str, h: f64| FieldInfo {
            family: ArtifactFamily::KccDescriptor,
            field,
            h_bits: h,
            variance: 0.0,
            score_bits: h,
            b_min: 1,
            b_max: 10,
            decision_critical: false,
            n_samples: 100,
        };
        let infos = vec![mk("lo", 1.0), mk("mid", 4.0), mk("hi", 8.0)];
        let plan = allocate(&infos, 12);
        assert!(
            plan.total_bits <= 12,
            "must not exceed budget, got {}",
            plan.total_bits
        );
        let b_lo = plan.bits_for(ArtifactFamily::KccDescriptor, "lo").unwrap();
        let b_hi = plan.bits_for(ArtifactFamily::KccDescriptor, "hi").unwrap();
        assert!(b_hi >= b_lo, "higher entropy ⇒ ≥ bits: {b_hi} !>= {b_lo}");
        assert!(
            plan.bits_entropy_correlation() > 0.0,
            "§14.5 information alignment"
        );
        // generous budget ⇒ each field rises to its INFORMATION cap (ceil(H)), not b_max — a
        // field never gets more resolution than its own information (§8 ∝-information principle).
        let plan2 = allocate(&infos, 100);
        assert_eq!(
            plan2.bits_for(ArtifactFamily::KccDescriptor, "lo").unwrap(),
            1,
            "ceil(H=1)=1"
        );
        assert_eq!(
            plan2.bits_for(ArtifactFamily::KccDescriptor, "hi").unwrap(),
            8,
            "ceil(H=8)=8"
        );
    }

    #[test]
    fn decision_critical_fields_are_pinned_regardless_of_entropy() {
        // a near-constant decision-critical field keeps its full native budget even under a tight
        // global budget — never coarsened across its threshold (INV-8 / handoff).
        let critical = FieldInfo {
            family: ArtifactFamily::TeacherStudentConsensus,
            field: "rhat",
            h_bits: 0.05, // almost no entropy
            variance: 1e-6,
            score_bits: 0.05,
            b_min: 10,
            b_max: 10,
            decision_critical: true,
            n_samples: 100,
        };
        let metadata = FieldInfo {
            family: ArtifactFamily::TeacherStudentConsensus,
            field: "student_weight",
            h_bits: 3.0,
            variance: 0.1,
            score_bits: 3.0,
            b_min: 1,
            b_max: 8,
            decision_critical: false,
            n_samples: 100,
        };
        let plan = allocate(&[critical, metadata], 4); // tiny budget
        let a_rhat = plan.allocations.iter().find(|a| a.field == "rhat").unwrap();
        assert!(
            a_rhat.pinned && a_rhat.bits == 10,
            "rhat must stay pinned at 10 bits"
        );
    }

    fn real_kcc_residues() -> Option<serde_json::Value> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
            "../../kv31_validation_frozen_20260512T215852Z/engine_manifest/Kv31_chainA.kcc_visualization.json",
        );
        if !p.exists() {
            return None;
        }
        serde_json::from_slice(&std::fs::read(&p).ok()?).ok()
    }

    #[test]
    fn allocates_real_kcc_fields_with_information_alignment() {
        // §14.5 on REAL routed payloads: drive the allocator with the 396 real residue values per
        // field; allocated bits must positively correlate with entropy, the near-constant
        // total_steps (9500 for every residue, H≈0) must collapse to b_min, and a spread field
        // (sum_motion) must get strictly more.
        let Some(json) = real_kcc_residues() else {
            eprintln!("skip: no real KCC fixture present");
            return;
        };
        let residues = json["residues"].as_array().unwrap();
        let collect = |k: &str| -> Vec<f64> {
            residues
                .iter()
                .filter_map(|r| r.get(k).and_then(|v| v.as_f64()))
                .collect()
        };
        let mut samples: HashMap<&'static str, Vec<f64>> = HashMap::new();
        for k in [
            "kcc_score",
            "direction_score",
            "motion_efficiency",
            "burst_motion",
            "causal_lag",
            "lag_corr_peak",
            "local_cov",
            "sum_motion",
            "active_causal_steps",
            "total_steps",
            "net_dx",
            "net_dy",
            "net_dz",
        ] {
            samples.insert(k, collect(k));
        }
        let e = enc();
        let infos = measure_corpus(&e, ArtifactFamily::KccDescriptor, &samples);
        assert_eq!(infos.len(), 13, "all 13 scalar KCC fields measured");

        // budget ≈ half the native ceiling so water-filling actually binds.
        let ceiling: u32 = infos.iter().map(|fi| fi.b_max as u32).sum();
        let plan = allocate(&infos, ceiling / 2);
        assert!(
            plan.bits_entropy_correlation() > 0.0,
            "§14.5: corr(bits, H) must be positive, got {}\n{}",
            plan.bits_entropy_correlation(),
            plan.summary()
        );
        // total_steps is constant (9500) ⇒ H≈0 ⇒ b_min; sum_motion is spread ⇒ more bits.
        let b_const = plan
            .bits_for(ArtifactFamily::KccDescriptor, "total_steps")
            .unwrap();
        let b_spread = plan
            .bits_for(ArtifactFamily::KccDescriptor, "sum_motion")
            .unwrap();
        assert_eq!(
            b_const, B_MIN_DEFAULT,
            "constant total_steps must collapse to b_min"
        );
        assert!(
            b_spread > b_const,
            "a spread field must out-bid a constant one: {b_spread} !> {b_const}"
        );

        // determinism (§13/§16): identical inputs ⇒ identical plan.
        let plan2 = allocate(&infos, ceiling / 2);
        assert_eq!(
            plan.allocations, plan2.allocations,
            "allocation must be deterministic"
        );
    }

    #[test]
    fn kcc_site_decision_fields_pinned_under_tight_budget() {
        // Phase 5/9: the KccSite signed-lag AND burst_motion (both DecisionSpec::Threshold) are
        // decision-critical → held at b_max under the EXISTING M3 allocator (no allocator rewrite),
        // even with near-constant samples + a tiny global budget, while spread metadata collapses.
        let e = enc();
        let lag: Vec<f64> = (0..100).map(|i| -24.0 + (i as f64) * 0.001).collect(); // near-constant
        let burst: Vec<f64> = (0..100).map(|i| 1.0 + (i as f64) * 0.0001).collect(); // near-constant ≈1.0
        let rank: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect(); // spread metadata
        let mut samples: HashMap<&'static str, Vec<f64>> = HashMap::new();
        samples.insert("site_causal_lag", lag);
        samples.insert("burst_motion", burst);
        samples.insert("rank_C", rank);
        let infos = measure_corpus(&e, ArtifactFamily::KccSite, &samples);
        let plan = allocate(&infos, 2); // tiny budget
        for field in ["site_causal_lag", "burst_motion"] {
            let info = infos.iter().find(|fi| fi.field == field).unwrap();
            assert!(
                info.decision_critical,
                "{field} must be decision-critical (Threshold)"
            );
            let a = plan.allocations.iter().find(|a| a.field == field).unwrap();
            assert!(
                a.pinned && a.bits == info.b_max,
                "{field} must stay pinned at b_max"
            );
        }
        // a spread non-critical metadata field is NOT pinned (it can be coarsened).
        let rank_a = plan
            .allocations
            .iter()
            .find(|a| a.field == "rank_C")
            .unwrap();
        assert!(!rank_a.pinned, "metadata rank_C must not be pinned");
    }

    #[test]
    fn variance_channel_uses_gaussian_rate() {
        // a TeWindowedVariance field (adaptive_dt) scores via ½·log2(σ²), not code entropy.
        let e = enc();
        let samples: Vec<f64> = (0..200)
            .map(|i| 1e-3 * (1.0 + (i as f64 * 0.01).sin()))
            .collect();
        let fi =
            measure_field(&e, ArtifactFamily::FiberBundleDstw, "adaptive_dt", &samples).unwrap();
        let expected = 0.5 * population_variance(&samples).max(1e-12).log2();
        assert!(
            (fi.score_bits - expected).abs() < 1e-9,
            "variance-channel score must be ½log2σ²"
        );
    }
}
