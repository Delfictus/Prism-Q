//! Phase 3.5 — `CausalTruthingAudit` AuditedTransform.
//!
//! Emitted as part of the producer-repair-causal-truthing lane.  Sits
//! downstream of KCC + TIDE: takes the per-site causal summaries that
//! the engine has already constructed and asks one question for each
//! site:
//!
//! > "If this site is not inert (it has real spike support), is at
//! > least one of its causal-truthing fields honestly computable?"
//!
//! "Honestly computable" means *finite* (not NaN, not Inf), per the
//! producer-repair contract:
//!
//! * `candidate_kcc_causal_lag[i]` is `NaN` ⇔ the kernel could not
//!   honestly compute lag for that residue (insufficient cross-paired
//!   events with `j != i`).
//! * `transfer_entropy[i]` is `NaN` ⇔ TIDE found degenerate (saturated
//!   or constant) source/target trains.
//!
//! The audit fires when **every** candidate's lag is `NaN` AND
//! **every** TIDE projection (or its absence) is `NaN`/empty, on a
//! site whose spike support exceeds [`MIN_MEANINGFUL_SPIKE_SUPPORT`].
//! That is the precise signature that previously hid behind sentinel
//! `-5.0` lags and `0.0` TE values: the runtime claimed meaningful
//! causal signal, but every producer was dead.
//!
//! # Routing
//!
//! `AuditRouting::Quarantine`.  The site is preserved in the JSON
//! output (so consumers can see which sites the audit caught), but
//! the violation is logged and consumers may down-weight or drop
//! quarantined sites.
//!
//! # Determinism / tolerance
//!
//! * `DeterminismClass::BitExact` — pure inspection of already-
//!   computed scalars; no atomics, no reductions.
//! * `TolerancePolicy::BitExact` — `is_finite()` is a bit-exact
//!   predicate.

use super::{
    AuditOutcome, AuditRouting, AuditedTransform, DeterminismClass, LawFamily, LawId,
    TolerancePolicy, TransformId, TransformViolation, ViolationEvidence,
};

// ============================================================================
// Identity + law declarations
// ============================================================================

pub const TRANSFORM_CAUSAL_TRUTHING_AUDIT: TransformId = TransformId("causal_truthing_audit");

pub const LAW_L4_CAUSAL_MATH_MEANINGFUL_OR_ABSENT: LawId =
    LawId::new("l4_causal_math_meaningful_or_absent", LawFamily::Algebraic);

pub const DECLARED_LAWS: &[LawId] = &[LAW_L4_CAUSAL_MATH_MEANINGFUL_OR_ABSENT];

/// Spike-support threshold above which a site is considered "meaningful"
/// for the purposes of L4.  Below this, a site has so little signal
/// that "no causal lag computable" is honestly the right answer and
/// should not raise a violation.
pub const MIN_MEANINGFUL_SPIKE_SUPPORT: usize = 32;

// ============================================================================
// Input / output types
// ============================================================================

/// Per-site causal-truthing summary.  All `f32` fields use the
/// producer-repair contract: `NaN` = not honestly computable.
#[derive(Debug, Clone, PartialEq)]
pub struct SiteCausalSummary {
    pub site_id: i32,
    /// Number of spike events attributed to this site.
    pub spike_support: usize,
    /// Per-candidate causal-lag values.  NaN entries mean the kernel
    /// could not honestly compute lag for that candidate.
    pub candidate_causal_lag: Vec<f32>,
    /// Per-candidate transfer entropies, if available.  NaN entries
    /// mean TIDE detected degenerate trains.  Empty vector means the
    /// site had no TIDE projection at all (also "all undefined").
    pub candidate_transfer_entropy: Vec<f32>,
}

impl SiteCausalSummary {
    /// True if every candidate's `causal_lag` is NaN/Inf.
    pub fn all_lag_undefined(&self) -> bool {
        self.candidate_causal_lag.iter().all(|v| !v.is_finite())
    }

    /// True if there is no TIDE projection or every entry is NaN/Inf.
    pub fn all_te_undefined(&self) -> bool {
        self.candidate_transfer_entropy.is_empty()
            || self
                .candidate_transfer_entropy
                .iter()
                .all(|v| !v.is_finite())
    }

    /// True if the site has enough spike support to be "meaningful"
    /// under L4.
    pub fn is_meaningful(&self) -> bool {
        self.spike_support >= MIN_MEANINGFUL_SPIKE_SUPPORT
    }
}

// ============================================================================
// The transform
// ============================================================================

#[derive(Debug, Clone, Copy, Default)]
pub struct CausalTruthingAudit;

impl CausalTruthingAudit {
    pub const fn new() -> Self {
        CausalTruthingAudit
    }
}

impl AuditedTransform for CausalTruthingAudit {
    type Input<'a> = &'a [SiteCausalSummary];
    type Output = Vec<SiteCausalSummary>;

    fn identity(&self) -> TransformId {
        TRANSFORM_CAUSAL_TRUTHING_AUDIT
    }

    fn determinism(&self) -> DeterminismClass {
        DeterminismClass::BitExact
    }

    fn tolerance(&self) -> TolerancePolicy {
        TolerancePolicy::BitExact
    }

    fn laws(&self) -> &'static [LawId] {
        DECLARED_LAWS
    }

    fn verify(&self, output: &Self::Output) -> Vec<TransformViolation> {
        let mut violations = Vec::new();
        for s in output {
            if !s.is_meaningful() {
                continue;
            }
            let lag_dead = s.all_lag_undefined();
            let te_dead = s.all_te_undefined();
            if lag_dead && te_dead {
                violations.push(TransformViolation {
                    transform: TRANSFORM_CAUSAL_TRUTHING_AUDIT,
                    law: LAW_L4_CAUSAL_MATH_MEANINGFUL_OR_ABSENT,
                    routing: AuditRouting::Quarantine,
                    evidence: ViolationEvidence::DeadCausalMathOnMeaningfulSite {
                        site_id: s.site_id,
                        spike_support: s.spike_support,
                        candidate_count: s.candidate_causal_lag.len(),
                        all_lag_undefined: lag_dead,
                        all_te_undefined: te_dead,
                    },
                });
            }
        }
        violations
    }

    fn apply<'a>(&self, input: Self::Input<'a>) -> AuditOutcome<Self::Output> {
        // Pass-through transform: the audit observes but does not
        // mutate.  `adjudicate` walks the violations and returns
        // `Quarantined` when they are present.
        let out = input.to_vec();
        self.adjudicate(out)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn nan() -> f32 {
        f32::NAN
    }

    #[test]
    fn inert_site_with_dead_math_does_not_violate() {
        let summary = SiteCausalSummary {
            site_id: 0,
            spike_support: 4, // below MIN_MEANINGFUL_SPIKE_SUPPORT
            candidate_causal_lag: vec![nan(), nan(), nan()],
            candidate_transfer_entropy: vec![nan(), nan(), nan()],
        };
        let audit = CausalTruthingAudit::new();
        let outcome = audit.apply(std::slice::from_ref(&summary));
        assert!(
            matches!(outcome, AuditOutcome::Accepted { .. }),
            "inert site (low spike_support) should be accepted regardless of dead math"
        );
    }

    #[test]
    fn meaningful_site_with_dead_math_quarantines() {
        let summary = SiteCausalSummary {
            site_id: 7,
            spike_support: 1024,
            candidate_causal_lag: vec![nan(), nan(), nan()],
            candidate_transfer_entropy: vec![nan(), nan()],
        };
        let audit = CausalTruthingAudit::new();
        let outcome = audit.apply(std::slice::from_ref(&summary));
        match outcome {
            AuditOutcome::Quarantined { violations, .. } => {
                assert_eq!(violations.len(), 1);
                let v = &violations[0];
                assert_eq!(v.law, LAW_L4_CAUSAL_MATH_MEANINGFUL_OR_ABSENT);
                assert_eq!(v.routing, AuditRouting::Quarantine);
                match &v.evidence {
                    ViolationEvidence::DeadCausalMathOnMeaningfulSite {
                        site_id,
                        spike_support,
                        candidate_count,
                        all_lag_undefined,
                        all_te_undefined,
                    } => {
                        assert_eq!(*site_id, 7);
                        assert_eq!(*spike_support, 1024);
                        assert_eq!(*candidate_count, 3);
                        assert!(*all_lag_undefined);
                        assert!(*all_te_undefined);
                    }
                    other => panic!("expected DeadCausalMathOnMeaningfulSite, got {:?}", other),
                }
            }
            other => panic!("expected Quarantined, got {:?}", other),
        }
    }

    #[test]
    fn meaningful_site_with_one_finite_lag_accepts() {
        let summary = SiteCausalSummary {
            site_id: 1,
            spike_support: 512,
            candidate_causal_lag: vec![nan(), 5.0, nan()],
            candidate_transfer_entropy: vec![nan(), nan()],
        };
        let audit = CausalTruthingAudit::new();
        let outcome = audit.apply(std::slice::from_ref(&summary));
        assert!(
            matches!(outcome, AuditOutcome::Accepted { .. }),
            "one finite lag value is enough to defuse L4 — TE may still be dead"
        );
    }

    #[test]
    fn meaningful_site_with_one_finite_te_accepts() {
        let summary = SiteCausalSummary {
            site_id: 1,
            spike_support: 512,
            candidate_causal_lag: vec![nan(), nan()],
            candidate_transfer_entropy: vec![0.0035, nan()],
        };
        let audit = CausalTruthingAudit::new();
        let outcome = audit.apply(std::slice::from_ref(&summary));
        assert!(
            matches!(outcome, AuditOutcome::Accepted { .. }),
            "one finite TE value is enough to defuse L4 — lag may still be dead"
        );
    }

    #[test]
    fn empty_te_with_finite_lag_accepts() {
        // No TIDE projection (TE vec empty) but finite lag values:
        // L4 is satisfied because lag is honestly computable.
        let summary = SiteCausalSummary {
            site_id: 2,
            spike_support: 512,
            candidate_causal_lag: vec![5.0, -10.0, 15.0],
            candidate_transfer_entropy: vec![],
        };
        let audit = CausalTruthingAudit::new();
        let outcome = audit.apply(std::slice::from_ref(&summary));
        assert!(matches!(outcome, AuditOutcome::Accepted { .. }));
    }

    #[test]
    fn audit_identity_and_metadata() {
        let audit = CausalTruthingAudit::new();
        assert_eq!(audit.identity(), TRANSFORM_CAUSAL_TRUTHING_AUDIT);
        assert_eq!(audit.determinism(), DeterminismClass::BitExact);
        assert_eq!(audit.tolerance(), TolerancePolicy::BitExact);
        assert_eq!(audit.laws(), DECLARED_LAWS);
    }

    #[test]
    fn multi_site_partial_violation() {
        let summaries = vec![
            // meaningful site, dead math → quarantine
            SiteCausalSummary {
                site_id: 0,
                spike_support: 1000,
                candidate_causal_lag: vec![nan()],
                candidate_transfer_entropy: vec![],
            },
            // meaningful site, healthy lag → no violation
            SiteCausalSummary {
                site_id: 1,
                spike_support: 1000,
                candidate_causal_lag: vec![5.0],
                candidate_transfer_entropy: vec![],
            },
            // inert site, dead math → no violation (not meaningful)
            SiteCausalSummary {
                site_id: 2,
                spike_support: 4,
                candidate_causal_lag: vec![nan()],
                candidate_transfer_entropy: vec![],
            },
        ];
        let audit = CausalTruthingAudit::new();
        let outcome = audit.apply(&summaries);
        match outcome {
            AuditOutcome::Quarantined { violations, .. } => {
                assert_eq!(violations.len(), 1);
                if let ViolationEvidence::DeadCausalMathOnMeaningfulSite { site_id, .. } =
                    &violations[0].evidence
                {
                    assert_eq!(*site_id, 0);
                } else {
                    panic!("unexpected evidence variant");
                }
            }
            other => panic!("expected Quarantined, got {:?}", other),
        }
    }
}
