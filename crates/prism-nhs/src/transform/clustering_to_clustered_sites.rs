//! First wrapped transform: `ClusteringToClusteredSites`.
//!
//! Wraps the canonical site-construction boundary in the engine —
//! `crate::persistent_engine::build_clustered_sites` — under the Phase 2
//! audit spine. The transform is the only public path through which
//! external callers may obtain a `Vec<ClusteredBindingSite>` from a
//! clustering result; the underlying function is crate-private.
//!
//! # Laws enforced
//!
//! * **L1 — `l1_gvm_populated`** (formalizes Phase 1 residual R5): every
//!   emitted site must have `localization.geometric_voxel_mass` populated
//!   as `Some(_)`. Absence is structural corruption — the canonical
//!   constructor `ClusteredBindingSite::new_with_geometric_voxel_mass`
//!   is the only legal construction path and it sets the GVM view from
//!   the same `[f32; 3]` as the legacy emission scalar, so `None` here
//!   means something has bypassed the constructor and produced an
//!   invalid object. **Routing: [`AuditRouting::Abort`]**.
//!
//! * **L2 — `l2_emission_compat_matches_gvm`** (formalizes Phase 1
//!   residual R4): for every emitted site,
//!   `emission_compat_centroid()` must bit-exactly equal
//!   `localization.geometric_voxel_mass`. Divergence is structural
//!   corruption — both values are set from the same input `[f32; 3]` at
//!   construction and updated atomically by `set_geometric_voxel_mass_centroid`.
//!   **Routing: [`AuditRouting::Abort`]**.
//!
//! Neither law emits `Quarantine` in Phase 2 — both are Family A
//! algebraic conservation laws on structural invariants of a single
//! output. `Quarantine` is reserved for site-local semantic violations
//! that will be introduced by later transforms (e.g., `LiningResidues`
//! view absence on sites classified as requiring it).
//!
//! # Determinism and tolerance
//!
//! * `DeterminismClass::AtomicsAffected` — the underlying clustering
//!   backend is gpu-hash ECL-CC union-find (Phase 1 residual R1). GPU
//!   atomic-order dependence induces bounded sub-Å centroid drift at
//!   fixed inputs across runs. Downstream consumers that need
//!   cross-run bit-identity must not assume this transform provides it.
//! * `TolerancePolicy::BitExact` — the two laws compare values that
//!   are set from the same input `[f32; 3]` with no floating-point
//!   arithmetic intervening. Bit-equal comparison is the correct
//!   semantics; there is no tolerance knob to tune here.

use super::{
    AuditOutcome, AuditRouting, AuditedTransform, DeterminismClass, LawFamily, LawId,
    TolerancePolicy, TransformId, TransformViolation, ViolationEvidence,
};
use crate::fused_engine::{NhsAmberFusedEngine, SpikeEvent};
use crate::persistent_engine::{build_clustered_sites, ClusteredBindingSite};
use crate::rt_clustering::RtClusteringResult;
use crate::spatial_view::SpatialView;

// ============================================================================
// Identity + law declarations
// ============================================================================

pub const TRANSFORM_CLUSTERING_TO_CLUSTERED_SITES: TransformId =
    TransformId("clustering_to_clustered_sites");

pub const LAW_L1_GVM_POPULATED: LawId = LawId::new("l1_gvm_populated", LawFamily::Algebraic);
pub const LAW_L2_EMISSION_COMPAT_MATCHES_GVM: LawId =
    LawId::new("l2_emission_compat_matches_gvm", LawFamily::Algebraic);

/// Full declared law set for this transform. Returned from
/// `AuditedTransform::laws`. Extending this array is a deliberate API
/// change — every law listed here must have a matching branch in
/// `verify`, and every violation `verify` emits must reference a
/// `LawId` listed here.
pub const DECLARED_LAWS: &[LawId] = &[LAW_L1_GVM_POPULATED, LAW_L2_EMISSION_COMPAT_MATCHES_GVM];

// ============================================================================
// Input type
// ============================================================================

/// Borrowed input to [`ClusteringToClusteredSites::apply`].
///
/// Holds references to the caller's data so the transform does not
/// force a clone at the spine boundary. The three fields mirror the
/// signature of the underlying `build_clustered_sites` function.
///
/// No `Debug` derive: `NhsAmberFusedEngine` does not implement `Debug`
/// and the engine ref is unavoidably part of this struct. Callers that
/// need to dump the input should log the two sliceable fields directly.
pub struct ClusteringInput<'a> {
    /// All spike events in the run. Zipped 1:1 with
    /// `clustering_result.cluster_ids`.
    pub spike_events: &'a [SpikeEvent],
    /// Result of the clustering backend
    /// (`gpu-hash` ECL-CC under the Phase 1 AUTO dispatch).
    pub clustering_result: &'a RtClusteringResult,
    /// Optional borrow of the fused engine, needed by
    /// `build_clustered_sites` for KDE density-peak refinement on GPU.
    /// `None` is acceptable and triggers the CPU-fallback path inside
    /// the underlying function.
    pub engine: Option<&'a NhsAmberFusedEngine>,
}

// ============================================================================
// The transform
// ============================================================================

/// Zero-sized singleton. State-free — all governance lives in the
/// trait impl. Construct with `ClusteringToClusteredSites::new()` or
/// the `Default` impl.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClusteringToClusteredSites;

impl ClusteringToClusteredSites {
    pub const fn new() -> Self {
        ClusteringToClusteredSites
    }
}

impl AuditedTransform for ClusteringToClusteredSites {
    type Input<'a> = ClusteringInput<'a>;
    type Output = Vec<ClusteredBindingSite>;

    fn identity(&self) -> TransformId {
        TRANSFORM_CLUSTERING_TO_CLUSTERED_SITES
    }

    fn determinism(&self) -> DeterminismClass {
        DeterminismClass::AtomicsAffected
    }

    fn tolerance(&self) -> TolerancePolicy {
        TolerancePolicy::BitExact
    }

    fn laws(&self) -> &'static [LawId] {
        DECLARED_LAWS
    }

    fn verify(&self, output: &Self::Output) -> Vec<TransformViolation> {
        let mut violations = Vec::new();
        verify_l1_gvm_populated(output, &mut violations);
        verify_l2_emission_compat_matches_gvm(output, &mut violations);
        violations
    }

    fn apply<'a>(&self, input: Self::Input<'a>) -> AuditOutcome<Self::Output> {
        let sites =
            build_clustered_sites(input.spike_events, input.clustering_result, input.engine);
        self.adjudicate(sites)
    }
}

// ============================================================================
// Law verifiers
// ============================================================================

/// L1 — every emitted site must have `GeometricVoxelMass` populated.
///
/// Pure; appends to `out`. Iterating produces violations in cluster_id
/// order of the input slice, which mirrors construction order.
fn verify_l1_gvm_populated(sites: &[ClusteredBindingSite], out: &mut Vec<TransformViolation>) {
    for site in sites {
        if site.view(SpatialView::GeometricVoxelMass).is_none() {
            out.push(TransformViolation {
                transform: TRANSFORM_CLUSTERING_TO_CLUSTERED_SITES,
                law: LAW_L1_GVM_POPULATED,
                routing: AuditRouting::Abort,
                evidence: ViolationEvidence::GvmViewAbsent {
                    cluster_id: site.cluster_id,
                },
            });
        }
    }
}

/// L2 — `emission_compat_centroid()` must bit-exactly equal the
/// `GeometricVoxelMass` view on every emitted site.
///
/// Sites that fail L1 (GVM absent) are skipped here — L1 already
/// produced an abort-routed violation for them and L2 has no basis to
/// compare against `None`. This avoids double-reporting the same
/// structural corruption under two law ids.
fn verify_l2_emission_compat_matches_gvm(
    sites: &[ClusteredBindingSite],
    out: &mut Vec<TransformViolation>,
) {
    for site in sites {
        let Some(gvm) = site.view(SpatialView::GeometricVoxelMass) else {
            continue;
        };
        let emission = site.emission_compat_centroid();
        // TolerancePolicy::BitExact — both values were set from the
        // same input `[f32; 3]`. No FP arithmetic intervenes, so
        // anything other than bit-equality is a structural fault.
        if gvm != emission {
            out.push(TransformViolation {
                transform: TRANSFORM_CLUSTERING_TO_CLUSTERED_SITES,
                law: LAW_L2_EMISSION_COMPAT_MATCHES_GVM,
                routing: AuditRouting::Abort,
                evidence: ViolationEvidence::EmissionCompatDivergent {
                    cluster_id: site.cluster_id,
                    emission_compat: emission,
                    gvm_view: gvm,
                },
            });
        }
    }
}

// ============================================================================
// Tests
// ============================================================================
//
// Strategy per the Phase 2 handoff: failure paths use a **mock transform with
// synthetic bad output**, NOT a `#[cfg(test)]` mutator that weakens
// `ClusteredBindingSite`. The goal is to prove the framework catches invalid
// transform output, not that the core production type can be manually broken.
//
// Tests live as lib unit tests (not in `tests/`) so `cargo test --lib` runs
// them without triggering cargo's package-wide build of unrelated bin targets.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistent_engine::{DruggabilityScore, SiteClassification};
    use crate::transform::{AuditRecord, AuditedTransform};

    fn zero_druggability() -> DruggabilityScore {
        DruggabilityScore {
            overall: 0.0,
            volume_score: 0.0,
            enclosure_score: 0.0,
            hydrophobicity_score: 0.0,
            aromatic_score: 0.0,
            catalytic_score: 0.0,
            is_druggable: false,
        }
    }

    fn make_site(cluster_id: i32, centroid: [f32; 3]) -> ClusteredBindingSite {
        ClusteredBindingSite::new_with_geometric_voxel_mass(
            cluster_id,
            centroid,
            0,
            Vec::new(),
            0.0,
            0.0,
            [0.0, 0.0, 0.0],
            0.0,
            zero_druggability(),
            SiteClassification::Unknown,
        )
    }

    // -----------------------------------------------------------------------
    // Mock transform for framework-spine tests
    // -----------------------------------------------------------------------

    struct MockTransform {
        violations_to_emit: Vec<TransformViolation>,
    }
    impl MockTransform {
        const ID: TransformId = TransformId("mock_transform_for_test");
        const LAW: LawId = LawId::new("mock_law", LawFamily::Algebraic);
        const LAWS: &'static [LawId] = &[Self::LAW];
        fn with_violations(violations: Vec<TransformViolation>) -> Self {
            MockTransform {
                violations_to_emit: violations,
            }
        }
    }
    impl AuditedTransform for MockTransform {
        type Input<'a>
            = ()
        where
            Self: 'a;
        type Output = u8;
        fn identity(&self) -> TransformId {
            Self::ID
        }
        fn determinism(&self) -> DeterminismClass {
            DeterminismClass::BitExact
        }
        fn tolerance(&self) -> TolerancePolicy {
            TolerancePolicy::BitExact
        }
        fn laws(&self) -> &'static [LawId] {
            Self::LAWS
        }
        fn verify(&self, _output: &Self::Output) -> Vec<TransformViolation> {
            self.violations_to_emit.clone()
        }
        fn apply<'a>(&self, _input: Self::Input<'a>) -> AuditOutcome<Self::Output> {
            self.adjudicate(42u8)
        }
    }
    fn mock_violation(routing: AuditRouting, tag: &'static str) -> TransformViolation {
        TransformViolation {
            transform: MockTransform::ID,
            law: MockTransform::LAW,
            routing,
            evidence: ViolationEvidence::SyntheticForTesting { tag },
        }
    }

    // -----------------------------------------------------------------------
    // Concrete-transform metadata + real happy path
    // -----------------------------------------------------------------------

    #[test]
    fn metadata_exposed() {
        let t = ClusteringToClusteredSites::new();
        assert_eq!(t.identity(), TRANSFORM_CLUSTERING_TO_CLUSTERED_SITES);
        assert_eq!(t.determinism(), DeterminismClass::AtomicsAffected);
        assert_eq!(t.tolerance(), TolerancePolicy::BitExact);
        assert_eq!(t.laws(), DECLARED_LAWS);
        assert_eq!(
            t.laws(),
            &[LAW_L1_GVM_POPULATED, LAW_L2_EMISSION_COMPAT_MATCHES_GVM]
        );
    }

    #[test]
    fn real_happy_path_verify_is_empty() {
        let sites = vec![
            make_site(0, [1.0, 2.0, 3.0]),
            make_site(1, [4.0, 5.0, 6.0]),
            make_site(2, [-7.5, 0.25, 12.0]),
        ];
        for s in &sites {
            let gvm = s
                .view(SpatialView::GeometricVoxelMass)
                .expect("fixture invariant: GVM populated");
            assert_eq!(s.emission_compat_centroid(), gvm);
        }
        let violations = ClusteringToClusteredSites::new().verify(&sites);
        assert!(
            violations.is_empty(),
            "happy-path produced violations: {:?}",
            violations
        );
    }

    #[test]
    fn real_happy_path_adjudicate_is_accepted() {
        let sites = vec![make_site(7, [0.0, 0.0, 0.0]), make_site(9, [1.0, 1.0, 1.0])];
        let n = sites.len();
        let outcome = ClusteringToClusteredSites::new().adjudicate(sites);
        match outcome {
            AuditOutcome::Accepted { output, record } => {
                assert_eq!(output.len(), n);
                let expected = AuditRecord {
                    transform: TRANSFORM_CLUSTERING_TO_CLUSTERED_SITES,
                    determinism: DeterminismClass::AtomicsAffected,
                    tolerance: TolerancePolicy::BitExact,
                    laws_declared: DECLARED_LAWS,
                };
                assert_eq!(record, expected);
            }
            other => panic!("expected Accepted, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Framework spine via mock transform
    // -----------------------------------------------------------------------

    #[test]
    fn mock_no_violations_accepted() {
        let mock = MockTransform::with_violations(Vec::new());
        match mock.apply(()) {
            AuditOutcome::Accepted { output, record } => {
                assert_eq!(output, 42);
                assert_eq!(record.transform, MockTransform::ID);
            }
            other => panic!("expected Accepted, got {:?}", other),
        }
    }

    #[test]
    fn mock_abort_violation_aborts() {
        let v = mock_violation(AuditRouting::Abort, "abort_only");
        let mock = MockTransform::with_violations(vec![v.clone()]);
        match mock.apply(()) {
            AuditOutcome::Aborted { record, violations } => {
                assert_eq!(record.transform, MockTransform::ID);
                assert_eq!(violations, vec![v]);
            }
            other => panic!("expected Aborted, got {:?}", other),
        }
    }

    #[test]
    fn mock_quarantine_violation_quarantines() {
        let v = mock_violation(AuditRouting::Quarantine, "quarantine_only");
        let mock = MockTransform::with_violations(vec![v.clone()]);
        match mock.apply(()) {
            AuditOutcome::Quarantined {
                output,
                record,
                violations,
            } => {
                assert_eq!(output, 42, "quarantine must retain the output");
                assert_eq!(record.transform, MockTransform::ID);
                assert_eq!(violations, vec![v]);
            }
            other => panic!("expected Quarantined, got {:?}", other),
        }
    }

    #[test]
    fn mock_mixed_routing_prefers_abort() {
        let vs = vec![
            mock_violation(AuditRouting::Quarantine, "q1"),
            mock_violation(AuditRouting::Quarantine, "q2"),
            mock_violation(AuditRouting::Abort, "abort_priority"),
        ];
        let mock = MockTransform::with_violations(vs);
        match mock.apply(()) {
            AuditOutcome::Aborted { violations, .. } => {
                assert_eq!(violations.len(), 3);
                let abort_count = violations
                    .iter()
                    .filter(|v| v.routing == AuditRouting::Abort)
                    .count();
                assert_eq!(abort_count, 1);
            }
            other => panic!("expected Aborted, got {:?}", other),
        }
    }

    #[test]
    fn aborted_audit_error_roundtrip() {
        let v = mock_violation(AuditRouting::Abort, "display_roundtrip");
        let mock = MockTransform::with_violations(vec![v]);
        let err = match mock.apply(()).into_result() {
            Err(e) => e,
            Ok(_) => panic!("expected Err from Aborted outcome"),
        };
        let rendered = format!("{err}");
        assert!(
            rendered.contains(MockTransform::ID.0),
            "missing transform id: {rendered}"
        );
        assert!(rendered.contains("abort"), "missing routing: {rendered}");
        assert!(
            rendered.contains("display_roundtrip"),
            "missing evidence tag: {rendered}"
        );
        let _: &dyn std::error::Error = &err;
    }
}
