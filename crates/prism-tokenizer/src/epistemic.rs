//! §10 — epistemic bits: leverage tier + the stream-level firewall predicate.
//!
//! Every token carries its Part-I ceiling class (`leverage_tier`), `verified` (seal integrity,
//! INV-6), and `provenance` (sidecar entry present). The generative↔canonical firewall (INV-3,
//! IMMUTABLE-#1/#3) is enforceable from these bits alone — downstream-independent.

use crate::types::{ArtifactFamily, Token};

/// Leverage tier ≥ this is GENERATIVE / aspirational and is firewalled from the canonical path:
/// such a token is never canonical-eligible regardless of seal integrity (INV-3).
pub const GENERATIVE_TIER: u8 = 5;

/// Part-I epistemic ceiling class for a family. 0 = source-of-truth (engine facts, receipts);
/// higher = more aspirational ceiling (proxies, then generative preference).
pub fn leverage_tier(family: ArtifactFamily) -> u8 {
    use ArtifactFamily::*;
    match family {
        // source-of-truth: receipts + engine source generators. ValidationState is the aggregate
        // MD-ISA execution VERDICT (receipt-grade verification evidence, by-value invariant
        // native_md_receipt_count == len(streams)) — tier 0 alongside ReceiptStatus.
        ReceiptStatus | SrcCuda | SrcEngine | SrcCodec | SrcTwin | SrcGraph | SrcGflownet
        | ValidationState => 0,
        // direct observables + provenance containers + engine control/QC state.
        // CapturedGraphGpuTelemetry is runtime captured-graph GPU evidence (ghost temporal frames +
        // lattice routing status) with a self-contained by-value invariant (physical_time_fs ==
        // timestep_max * dt_fs) — a direct GPU observable, tier 1 alongside GpuTelemetry/ProtocolState.
        // PhaseManifoldEdgeValidation is the GhostPhaseLattice4D 4D-intersection kernel output over
        // the captured Ghost v2 tiles (real edge counts + SO(3) coherence) with self-contained by-value
        // conservation invariants (Σ component.n_nodes == stats.n_nodes; Σ component.n_intra_edges ==
        // stats.n_directed_edges) — runtime captured-graph GPU evidence from the same v2-capture surface
        // as CapturedGraphGpuTelemetry, tier 1 alongside it.
        SpikeEvent | NanlSpikePacket | GpuTelemetry | QnanlArtifact | ProtocolState | NoiseFloor
        | CapturedGraphGpuTelemetry
        | PhaseManifoldEdgeValidation
        | ScientificChronologyEvent
        | ScientificStateEpoch
        | T4EpochOverlay
        | ChronologyAcceptanceDecision
        | ScientificPocketOpeningEvent
        | LigandCausalActivityEvent
        | ResidueActivityEvent
        | RuntimeNumericalUncertaintyTensor
        | ThermalUncertaintyTensor
        | IntegratorUncertaintyTensor
        | ObservableUncertaintyTensor
        | EpochUncertaintyTensor
        | ThermodynamicSalienceEvent
        | ThermodynamicAction => 1,
        // reproducible labels / consensus / advisory descriptors / lexicographic rank.
        // DenseTensorSemanticSummary is a DETERMINISTIC reduction over an engine observable (real
        // source_hash + decoded-summary dimensions/finite-scan) — reproducible descriptor, not raw.
        // PathBBindingSiteMaterialized is a reproducible lexicographic materialized-site descriptor
        // (site_count == receipt.materialized_site_count, rank == 0..N-1 total order) — canonical-
        // eligible evidence, same class/ceiling as KccSite/RankerRow, NOT generative.
        // KccDescriptorNative is the RAW PRKCC001 kcc_v2full.bin per-residue descriptor decoded BY
        // VALUE (deterministic transpose of the self-describing envelope, real per-residue scalars) —
        // a reproducible descriptor, same class/ceiling as KccDescriptor, NOT raw-observable or
        // generative.
        // SpikeSummary is the RAW PRSPK001 _spikes.bin decoded BY VALUE into a per-residue BOUNDED
        // aggregate (deterministic reduction over the direct spike observable: Σ spike_count ==
        // payload record_count) — a reproducible descriptor over an engine observable, same class/
        // ceiling as KccDescriptorNative / DenseTensorSemanticSummary, NOT the raw per-spike stream
        // (SpikeEvent, tier 1) and NOT generative.
        // SpectralExtractor is a reproducible spectral-geometry descriptor over native trajectory
        // frame contacts + Path-B materialized sites + TWIN16 residuals. It is not raw GPU evidence
        // and not generative; it is canonical descriptor evidence when its input/output hashes route.
        TeacherStudentConsensus
        | KccDescriptor
        | KccDescriptorNative
        | SpikeSummary
        | KccSite
        | RankerRow
        | PathBBindingSiteMaterialized
        | DenseTensorSemanticSummary
        | SpectralExtractor
        | NegativeScientificEvidenceEvent
        | ScientificUncertaintyTensor
        | SignalToUncertaintyTensor
        | TransitionMotif
        | PersistentTransitionObject
        | TransitionConsensusEvent
        | TransitionEvidenceLedger
        | ScientificTransitionConsensusEvent
        | TransitionFalsificationProfile
        | RejectedPersistentTransitionCandidate
        | PersistentTransitionAcceptanceLedger
        | EpistemicUncertainty
        | TeacherConsensus => 2,
        // kinetic / dynamics with an honest ceiling (MSM proxy, DSTW, BOCPD, TWIN, spike-transition)
        MsmKineticSummary
        | FiberBundleDstw
        | BocpdEvent
        | TwinCoupled
        | SpikeTransitionEvent
        | BocpdLog
        | TwinTransitionResidual
        | TwinConditionEdge
        // ThermoPhaseManifold is the engine bounded thermo/temporal CCNS/SDST AGGREGATE SUMMARY over
        // the SPIKE_TRANSITION_EVENT view anchors — a thermo/dynamics rollup with an honest ceiling
        // and a self-contained by-value invariant (event_count == Σ cells == Σ event_sources ==
        // Σ phase_bucket_counts), same class as SpikeTransitionEvent/TwinTransitionResidual.
        | ThermoPhaseManifold
        | NmaCoupledEdge => 3,
        // DSTW-T4 lifecycle provenance: vertices + transitions are control/provenance containers
        // carrying receipts + VALUE-verified hashes + the honest runtime_origin (like ProtocolState).
        // OpcodeExecutionLedger: prismos-authored, but every `count` is tied BY-VALUE to the
        // engine md_native_receipt opcode_counts/kernel_counts (source-of-truth receipts) with the
        // source_receipt provenance join — a receipt-verified provenance container, like ProtocolState.
        // LineageDag is the merge-manifest lineage DAG carrier: a control/provenance container
        // carrying VALUE-verified per-artifact sha256 + the honest runtime_origin + the DAG
        // topology (parent→child edges), same posture as DstwStateVertex/ProtocolState.
        // IsaRegistrySummary is the POST_RUN_DERIVED registry×receipt coverage matrix: every `count`
        // is the verification's known_md_prismop_observation[].count (itself md_native_receipt-derived)
        // and every opcode is the plan's md_prismop_known_instruction_set — a receipt-verified,
        // deterministic provenance container (by-value invariants distinct(opcode)==registry_size,
        // required_core⊆observed), same posture as OpcodeExecutionLedger.
        DstwStateVertex
        | DstwStateTransition
        | OpcodeExecutionLedger
        | IsaRegistrySummary
        | LineageDag => 1,
        // bound-state / benchmark / prediction (apo-scored oracle, manifest gaps). The next-operator
        // recommendation is advisory engine-owner guidance (aspirational, not an observable).
        // PredictionState is a FIRST-CLASS prediction record (a hypothesis ABOUT a binding site with
        // by-value confidence/uncertainty + evidence lineage) — an aspirational prediction ceiling,
        // same class as FalsifiablePrediction; DISTINCT from (and NOT an alias of) DstwNextOperator.
        HoloBoundState | CapabilityGate | FalsifiablePrediction | DstwNextOperator
        | PredictionState => 4,
        // generative preference — firewalled (GENERATIVE_TIER)
        GflownetTrajectory | MoleculeGeneration => GENERATIVE_TIER,
        // unknown family — maximum uncertainty
        Opaque => 6,
    }
}

/// INV-3 firewall: a token may enter a canonical path ONLY if its seal integrity is verified
/// AND it is not generative. Computed from the token bits alone. NOTE: `verified` is integrity
/// only (INV-6) — this does NOT assert origin authenticity (use `canonical_eligible_strict`).
pub fn canonical_eligible(t: Token) -> bool {
    t.verified() && t.leverage_tier() < GENERATIVE_TIER
}

/// INV-9 strict gate for paths that REQUIRE verified origin (DoD insider/corrupted-pipeline
/// threat model): canonical-eligible AND keyed-signature `authenticated`. With signing deferred
/// (§7 A-SCOPE), `authenticated` is always 0, so this gate is intentionally closed until a keyed
/// provider sets it — integrity (`verified`) is never silently read as trust.
pub fn canonical_eligible_strict(t: Token) -> bool {
    canonical_eligible(t) && t.authenticated()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Token;

    #[test]
    fn receipts_and_source_are_source_of_truth() {
        assert_eq!(leverage_tier(ArtifactFamily::ReceiptStatus), 0);
        assert_eq!(leverage_tier(ArtifactFamily::SrcCuda), 0);
    }

    #[test]
    fn gflownet_is_generative_tier() {
        assert_eq!(
            leverage_tier(ArtifactFamily::GflownetTrajectory),
            GENERATIVE_TIER
        );
    }

    #[test]
    fn scientific_uncertainty_families_are_not_generative() {
        assert_eq!(
            leverage_tier(ArtifactFamily::RuntimeNumericalUncertaintyTensor),
            1
        );
        assert_eq!(leverage_tier(ArtifactFamily::ThermalUncertaintyTensor), 1);
        assert_eq!(
            leverage_tier(ArtifactFamily::IntegratorUncertaintyTensor),
            1
        );
        assert_eq!(
            leverage_tier(ArtifactFamily::ObservableUncertaintyTensor),
            1
        );
        assert_eq!(leverage_tier(ArtifactFamily::EpochUncertaintyTensor), 1);
        assert_eq!(
            leverage_tier(ArtifactFamily::ScientificUncertaintyTensor),
            2
        );
        assert_eq!(leverage_tier(ArtifactFamily::SignalToUncertaintyTensor), 2);
        assert_eq!(leverage_tier(ArtifactFamily::ThermodynamicSalienceEvent), 1);
        assert_eq!(leverage_tier(ArtifactFamily::TransitionMotif), 2);
        assert_eq!(leverage_tier(ArtifactFamily::PersistentTransitionObject), 2);
        assert_eq!(leverage_tier(ArtifactFamily::TransitionConsensusEvent), 2);
        assert_eq!(leverage_tier(ArtifactFamily::TransitionEvidenceLedger), 2);
        assert_eq!(leverage_tier(ArtifactFamily::EpistemicUncertainty), 2);
        assert_eq!(leverage_tier(ArtifactFamily::TeacherConsensus), 2);
        assert_eq!(leverage_tier(ArtifactFamily::ThermodynamicAction), 1);
        assert_eq!(
            leverage_tier(ArtifactFamily::ScientificTransitionConsensusEvent),
            2
        );
        assert_eq!(
            leverage_tier(ArtifactFamily::TransitionFalsificationProfile),
            2
        );
        assert_eq!(
            leverage_tier(ArtifactFamily::RejectedPersistentTransitionCandidate),
            2
        );
        assert_eq!(
            leverage_tier(ArtifactFamily::PersistentTransitionAcceptanceLedger),
            2
        );
    }

    #[test]
    fn authenticated_defaults_false_and_strict_gate_requires_it() {
        // INV-9: a verified, non-generative token is canonical-eligible but NOT strict-eligible
        // until a keyed signature sets authenticated; the bit defaults to 0.
        let t = Token::new(1, true, true, 0, 0);
        assert!(!t.authenticated());
        assert!(canonical_eligible(t));
        assert!(!canonical_eligible_strict(t));
        let signed = t.with_authenticated(true);
        assert!(signed.authenticated());
        assert!(canonical_eligible_strict(signed));
    }

    #[test]
    fn firewall_blocks_generative_and_unverified() {
        // verified spike (tier 1) → eligible
        let spike = Token::new(1, true, true, 0, 0);
        assert!(canonical_eligible(spike));
        // verified gflownet (tier 5) → NOT eligible (generative firewall, even with good seal)
        let gen = Token::new(GENERATIVE_TIER, true, true, 0, 0);
        assert!(!canonical_eligible(gen));
        // unverified spike (verified=0) → NOT eligible (quarantine)
        let unver = Token::new(1, false, true, 0, 0);
        assert!(!canonical_eligible(unver));
    }
}
