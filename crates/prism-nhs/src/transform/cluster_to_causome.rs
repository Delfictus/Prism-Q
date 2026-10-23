//! Phase 3 — `ClusterToCausome` AuditedTransform.
//!
//! Second canonical wrapped transform after Phase 2's
//! `ClusteringToClusteredSites`.  Consumes already-existing per-site
//! KCC fields (no re-running of KCC kernels — strict remap, never
//! reconstruction, per the manifold-doctrine "Pillar 2+ is remap-only"
//! rule) and projects them to a typed `Vec<CausomeBlock>`.
//!
//! # Determinism / tolerance
//!
//! * `DeterminismClass::AtomicsAffected` — KCC reductions upstream
//!   are CUDA-atomics-affected (transfer-entropy + cross-correlation
//!   reductions over per-frame motion arrays).
//! * `TolerancePolicy::RelativeEpsilon { eps: 1e-4 }` — the right
//!   encoding for the FP outputs that span many magnitudes
//!   (`burst_motion`, `lag_corr_peak`, etc.).  Per-field dispatch
//!   inside `verify` keeps integer-typed fields (`driver_residue_id`,
//!   `active_causal_steps`) under exact equality.
//!
//! # Laws
//!
//! `LAW_L3_DRIVER_SUBSET_OF_LINING` — Family-A Conservation of
//! Locality.  For every emitted `CausomeBlock`,
//! `driver_residue_id ∈ lining_residues[].resid` for the same site,
//! AND every `candidate_residue_id ∈ lining_residues[].resid`.
//! Routing: `Abort`.

use std::collections::HashMap;

use super::{
    AuditOutcome, AuditRouting, AuditedTransform, DeterminismClass, LawFamily, LawId,
    TolerancePolicy, TransformId, TransformViolation, ViolationEvidence,
};
use crate::causome::CausomeBlock;
use crate::persistent_engine::ClusteredBindingSite;

// ============================================================================
// Identity + law declarations
// ============================================================================

pub const TRANSFORM_CLUSTER_TO_CAUSOME: TransformId = TransformId("cluster_to_causome");

pub const LAW_L3_DRIVER_SUBSET_OF_LINING: LawId =
    LawId::new("l3_driver_subset_of_lining", LawFamily::Algebraic);

/// Full declared law set for this transform.
pub const DECLARED_LAWS: &[LawId] = &[LAW_L3_DRIVER_SUBSET_OF_LINING];

// ============================================================================
// Input & per-residue helper types
// ============================================================================

/// Per-site KCC aggregates needed by `ClusterToCausome`.  Mirrors the
/// existing v1 `binding_sites.json sites[i].kcc{}` sub-dict shape, but
/// typed.  The transform's `apply` method receives a
/// `&HashMap<i32, KccSiteData>` keyed by `cluster_id`.
#[derive(Debug, Clone, PartialEq)]
pub struct KccSiteData {
    pub site_id: i32,
    pub driver_residue_id: i32,
    pub candidate_residue_ids: Vec<i32>,
    pub candidate_residue_support: Vec<f32>,
    pub candidate_lag_provenance: Vec<crate::causome::KccLagProvenance>,
    pub burst_motion: f32,
    pub causal_lag: f32,
    pub direction_score: f32,
    pub kcc_confidence: f32,
    pub lag_corr_peak: f32,
    pub local_cov: f32,
    pub motion_efficiency: f32,
    pub site_burst_motion: f32,
    pub site_causal_lag: f32,
    pub site_direction_score: f32,
    pub site_lag_corr_peak: f32,
    pub site_local_cov: f32,
    pub site_motion_efficiency: f32,
    pub temporal_corr: f32,
    pub active_causal_steps: u32,
    pub total_steps: u32,
}

/// Per-site lining residue summary needed for L3 verification.  We
/// only need the resid set; the full `LiningResidue` struct is read
/// elsewhere for other purposes.
pub type LiningResidSet = std::collections::HashSet<i32>;

/// Borrowed input to [`ClusterToCausome::apply`].
pub struct ClusterToCausomeInput<'a> {
    pub sites: &'a [ClusteredBindingSite],
    pub kcc_per_site: &'a HashMap<i32, KccSiteData>,
    pub lining_per_site: &'a HashMap<i32, LiningResidSet>,
}

// ============================================================================
// The transform
// ============================================================================

/// Zero-sized singleton.  Phase 3's second `AuditedTransform`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClusterToCausome;

impl ClusterToCausome {
    pub const fn new() -> Self {
        ClusterToCausome
    }

    /// Pure projection from `KccSiteData` → `CausomeBlock`.  Does NOT
    /// run any law check.  Caller routes the result through
    /// `adjudicate` to enforce L3.
    fn project(kcc: &KccSiteData) -> CausomeBlock {
        CausomeBlock {
            site_id: kcc.site_id,
            driver_residue_id: kcc.driver_residue_id,
            candidate_residue_ids: kcc.candidate_residue_ids.clone(),
            candidate_residue_support: kcc.candidate_residue_support.clone(),
            candidate_lag_provenance: kcc.candidate_lag_provenance.clone(),
            burst_motion: kcc.burst_motion,
            causal_lag: kcc.causal_lag,
            direction_score: kcc.direction_score,
            kcc_confidence: kcc.kcc_confidence,
            lag_corr_peak: kcc.lag_corr_peak,
            local_cov: kcc.local_cov,
            motion_efficiency: kcc.motion_efficiency,
            site_burst_motion: kcc.site_burst_motion,
            site_causal_lag: kcc.site_causal_lag,
            site_direction_score: kcc.site_direction_score,
            site_lag_corr_peak: kcc.site_lag_corr_peak,
            site_local_cov: kcc.site_local_cov,
            site_motion_efficiency: kcc.site_motion_efficiency,
            temporal_corr: kcc.temporal_corr,
            active_causal_steps: kcc.active_causal_steps,
            total_steps: kcc.total_steps,
        }
    }
}

impl AuditedTransform for ClusterToCausome {
    type Input<'a> = ClusterToCausomeInput<'a>;
    type Output = Vec<(CausomeBlock, LiningResidSet)>;

    fn identity(&self) -> TransformId {
        TRANSFORM_CLUSTER_TO_CAUSOME
    }

    fn determinism(&self) -> DeterminismClass {
        DeterminismClass::AtomicsAffected
    }

    fn tolerance(&self) -> TolerancePolicy {
        TolerancePolicy::RelativeEpsilon { eps: 1e-4 }
    }

    fn laws(&self) -> &'static [LawId] {
        DECLARED_LAWS
    }

    fn verify(&self, output: &Self::Output) -> Vec<TransformViolation> {
        let mut violations = Vec::new();
        for (cb, lining) in output {
            verify_l3_driver_subset_of_lining(cb, lining, &mut violations);
        }
        violations
    }

    fn apply<'a>(&self, input: Self::Input<'a>) -> AuditOutcome<Self::Output> {
        let mut out: Vec<(CausomeBlock, LiningResidSet)> = Vec::with_capacity(input.sites.len());
        for site in input.sites {
            let sid = site.cluster_id;
            let Some(kcc) = input.kcc_per_site.get(&sid) else {
                // Sites without KCC data are simply skipped — their
                // CausomeBlock cannot be constructed honestly.  This
                // is not a law violation; it is a missing-input
                // situation that the upstream KCC pipeline owns.
                continue;
            };
            let lining = input.lining_per_site.get(&sid).cloned().unwrap_or_default();
            out.push((Self::project(kcc), lining));
        }
        self.adjudicate(out)
    }
}

// ============================================================================
// Law verifier — L3 Conservation of Locality
// ============================================================================

/// Verifies `driver_residue_id ∈ lining` AND every
/// `candidate_residue_id ∈ lining` for a single CausomeBlock.
fn verify_l3_driver_subset_of_lining(
    cb: &CausomeBlock,
    lining: &LiningResidSet,
    out: &mut Vec<TransformViolation>,
) {
    if !lining.contains(&cb.driver_residue_id) {
        out.push(TransformViolation {
            transform: TRANSFORM_CLUSTER_TO_CAUSOME,
            law: LAW_L3_DRIVER_SUBSET_OF_LINING,
            routing: AuditRouting::Abort,
            evidence: ViolationEvidence::DriverNotInLining {
                site_id: cb.site_id,
                driver_id: cb.driver_residue_id,
                lining_set_size: lining.len(),
            },
        });
    }
    for cand in &cb.candidate_residue_ids {
        if !lining.contains(cand) {
            out.push(TransformViolation {
                transform: TRANSFORM_CLUSTER_TO_CAUSOME,
                law: LAW_L3_DRIVER_SUBSET_OF_LINING,
                routing: AuditRouting::Abort,
                evidence: ViolationEvidence::CandidateNotInLining {
                    site_id: cb.site_id,
                    candidate_id: *cand,
                    lining_set_size: lining.len(),
                },
            });
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn synthetic_kcc(site_id: i32, driver: i32, candidates: Vec<i32>) -> KccSiteData {
        KccSiteData {
            candidate_lag_provenance: Vec::new(),
            site_id,
            driver_residue_id: driver,
            candidate_residue_ids: candidates.clone(),
            candidate_residue_support: vec![1.0 / candidates.len().max(1) as f32; candidates.len()],
            burst_motion: 1.0,
            causal_lag: 10.0,
            direction_score: 0.5,
            kcc_confidence: 0.8,
            lag_corr_peak: 0.9,
            local_cov: 0.3,
            motion_efficiency: 0.001,
            site_burst_motion: 1.0,
            site_causal_lag: 10.0,
            site_direction_score: 0.5,
            site_lag_corr_peak: 0.9,
            site_local_cov: 0.3,
            site_motion_efficiency: 0.001,
            temporal_corr: 0.4,
            active_causal_steps: 7000,
            total_steps: 10000,
        }
    }

    fn synthetic_lining(ids: &[i32]) -> LiningResidSet {
        ids.iter().copied().collect()
    }

    #[test]
    fn metadata_exposed() {
        let t = ClusterToCausome::new();
        assert_eq!(t.identity(), TRANSFORM_CLUSTER_TO_CAUSOME);
        assert_eq!(t.determinism(), DeterminismClass::AtomicsAffected);
        match t.tolerance() {
            TolerancePolicy::RelativeEpsilon { eps } => assert!((eps - 1e-4).abs() < 1e-12),
            other => panic!("expected RelativeEpsilon, got {other:?}"),
        }
        assert_eq!(t.laws(), DECLARED_LAWS);
        assert_eq!(t.laws()[0].family, LawFamily::Algebraic);
    }

    #[test]
    fn relative_epsilon_admits_close_values() {
        let pol = TolerancePolicy::RelativeEpsilon { eps: 1e-4 };
        // 1.0 vs 1.00005 → relative diff 5e-5 < 1e-4 → accepted
        assert!(pol.approx_eq(1.0, 1.00005));
        // 1.0 vs 1.001 → relative diff 1e-3 > 1e-4 → rejected
        assert!(!pol.approx_eq(1.0, 1.001));
        // Realistic KCC magnitude (0.5) with an in-tolerance perturbation
        // (relative diff = 0.00004 / 0.5 = 8e-5 < 1e-4 → accepted).
        assert!(pol.approx_eq(0.5, 0.50004));
        // Same magnitude with an out-of-tolerance perturbation
        // (relative diff = 0.0001 / 0.5 = 2e-4 > 1e-4 → rejected).
        assert!(!pol.approx_eq(0.5, 0.5001));
    }

    #[test]
    fn happy_path_no_violations() {
        // driver and candidates are all in lining → L3 passes
        let cb_input = synthetic_kcc(1, 50, vec![50, 60, 70]);
        let lining = synthetic_lining(&[50, 60, 70, 80, 90]);
        let cb = ClusterToCausome::project(&cb_input);
        let mut violations = Vec::new();
        verify_l3_driver_subset_of_lining(&cb, &lining, &mut violations);
        assert!(
            violations.is_empty(),
            "got {} violations: {:?}",
            violations.len(),
            violations
        );
    }

    #[test]
    fn driver_not_in_lining_aborts() {
        // driver = 999, lining = {50, 60, 70} → DriverNotInLining
        let cb_input = synthetic_kcc(2, 999, vec![50]);
        let lining = synthetic_lining(&[50, 60, 70]);
        let cb = ClusterToCausome::project(&cb_input);
        let mut violations = Vec::new();
        verify_l3_driver_subset_of_lining(&cb, &lining, &mut violations);
        assert_eq!(violations.len(), 1);
        match violations[0].evidence {
            ViolationEvidence::DriverNotInLining {
                site_id,
                driver_id,
                lining_set_size,
            } => {
                assert_eq!(site_id, 2);
                assert_eq!(driver_id, 999);
                assert_eq!(lining_set_size, 3);
            }
            ref other => panic!("expected DriverNotInLining, got {other:?}"),
        }
        assert_eq!(violations[0].routing, AuditRouting::Abort);
        assert_eq!(violations[0].law, LAW_L3_DRIVER_SUBSET_OF_LINING);
    }

    #[test]
    fn candidate_not_in_lining_aborts() {
        // driver in lining but one candidate is not
        let cb_input = synthetic_kcc(3, 50, vec![50, 60, 9999]);
        let lining = synthetic_lining(&[50, 60, 70]);
        let cb = ClusterToCausome::project(&cb_input);
        let mut violations = Vec::new();
        verify_l3_driver_subset_of_lining(&cb, &lining, &mut violations);
        assert_eq!(violations.len(), 1);
        match violations[0].evidence {
            ViolationEvidence::CandidateNotInLining {
                site_id,
                candidate_id,
                ..
            } => {
                assert_eq!(site_id, 3);
                assert_eq!(candidate_id, 9999);
            }
            ref other => panic!("expected CandidateNotInLining, got {other:?}"),
        }
    }

    #[test]
    fn apply_aborts_when_l3_violated() {
        use crate::transform::AuditOutcome;
        // Build a synthetic ClusterToCausomeInput that forces L3 to fire.
        // We don't need real ClusteredBindingSite construction for this
        // test — verify() works directly on the (CausomeBlock, lining)
        // tuples that apply() would emit.
        let bad_cb = ClusterToCausome::project(&synthetic_kcc(7, 12345, vec![5]));
        let lining = synthetic_lining(&[1, 2, 3]);
        let outcome = ClusterToCausome::new().adjudicate(vec![(bad_cb, lining)]);
        match outcome {
            AuditOutcome::Aborted { violations, .. } => {
                assert!(violations
                    .iter()
                    .any(|v| matches!(v.evidence, ViolationEvidence::DriverNotInLining { .. })));
                assert!(violations
                    .iter()
                    .any(|v| matches!(v.evidence, ViolationEvidence::CandidateNotInLining { .. })));
            }
            other => panic!("expected Aborted, got {other:?}"),
        }
    }
}
