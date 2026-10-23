//! §7 — the per-field geometry table + §2 per-field distortion-bound contract (INV-7).
//! Every field gets: a geometry matched to its physics, its measured-information channel (§8),
//! `canonical_safe` (INV-3), an explicit `max_distortion` (DistortionBound), and a
//! `decision_relevance` (DecisionSpec). Bounds are geometry-defaults unless a field is
//! decision-critical (§2), in which case they are explicit (and decision-aware, §3).

use crate::types::{
    ArtifactFamily as F, CircPeriod, DecisionSpec as DS, DistortionBound as DB, FieldGeometry as G,
    FieldSpec, InfoSource as I, Lattice,
};

/// Canonical UV excitation lines (CLAUDE.md `--uv-wavelengths`): Trp/Tyr at 280, plus 274/258/
/// 254/211 nm.
pub const UV_LINES: &[f32] = &[280.0, 274.0, 258.0, 254.0, 211.0];

/// Canonical amino-acid three-letter codes for KCC `residue_name`. The engine prep pipeline
/// writes AMBER residue names, so the protonation / disulfide variants (HID/HIE/HIP, CYX, ASH,
/// GLH, LYN) are included alongside the 20 standard residues. Any name not in this set escapes
/// losslessly to the sidecar (INV-1/INV-4) — it is never silently dropped or mis-coded. Ground
/// truth: 20 distinct names in `Kv31_chainA.kcc_visualization.json` (19 standard + HID).
pub const AA_NAMES: &[&str] = &[
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", // 20 standard
    "HID", "HIE", "HIP", "CYX", "ASH", "GLH", "LYN", // AMBER protonation/disulfide variants
];

/// Operator-term -> tokenizer source-schema field aliases for the three first-class scientific
/// surface events. Top-level operator fields stay unchanged. Aliases exist only where nested JSON
/// objects/arrays need flat tokenizer field slots; generated data remains traceable to the external
/// operator schema.
pub const SCIENTIFIC_SURFACE_FIELD_ALIASES: &[(&str, &str, &str)] = &[
    (
        "ScientificPocketOpeningEvent",
        "pocket_volume_a3",
        "pocket_volume",
    ),
    (
        "ScientificPocketOpeningEvent",
        "pocket_surface_area_a2",
        "pocket_surface_area",
    ),
    (
        "ScientificPocketOpeningEvent",
        "apo_holo_contrast_status",
        "apo_holo_contrast",
    ),
    (
        "LigandCausalActivityEvent",
        "contacts[].residue_id",
        "contact_residue_id",
    ),
    (
        "LigandCausalActivityEvent",
        "contacts[].contact_type",
        "contact_type",
    ),
    (
        "LigandCausalActivityEvent",
        "contacts[].distance_a",
        "distance_a",
    ),
    (
        "LigandCausalActivityEvent",
        "contacts[].duration_frames",
        "duration_frames",
    ),
    (
        "LigandCausalActivityEvent",
        "causal_signals.contact_delta",
        "contact_delta",
    ),
    (
        "LigandCausalActivityEvent",
        "causal_signals.pocket_volume_delta",
        "pocket_volume_delta",
    ),
    (
        "LigandCausalActivityEvent",
        "causal_signals.kcc_delta",
        "kcc_delta",
    ),
    (
        "LigandCausalActivityEvent",
        "causal_signals.warp_tensor_delta",
        "warp_tensor_delta",
    ),
    (
        "LigandCausalActivityEvent",
        "causal_signals.force_delta",
        "force_delta",
    ),
    (
        "LigandCausalActivityEvent",
        "causal_signals.residue_motion_delta",
        "residue_motion_delta",
    ),
    ("ResidueActivityEvent", "metrics.sasa_delta", "sasa_delta"),
    ("ResidueActivityEvent", "metrics.rmsf_delta", "rmsf_delta"),
    (
        "ResidueActivityEvent",
        "metrics.distance_gate_delta",
        "distance_gate_delta",
    ),
    (
        "ResidueActivityEvent",
        "metrics.contact_map_delta",
        "contact_map_delta",
    ),
    ("ResidueActivityEvent", "metrics.kcc_score", "kcc_score"),
    ("ResidueActivityEvent", "metrics.force_norm", "force_norm"),
    (
        "ResidueActivityEvent",
        "metrics.displacement_a",
        "displacement_a",
    ),
];

// Threshold point sets must be 'static (DecisionSpec holds &'static [f32]).
const RHAT_PTS: &[f32] = &[1.01, 1.05, 1.10];
const ESS_PTS: &[f32] = &[100.0, 400.0, 1000.0];
const CONF_PTS: &[f32] = &[0.5]; // M3 confidence-gate cut (placeholder until M3 fixes it)
const TE_PTS: &[f32] = &[0.0]; // sign = causal direction; the zero crossing must be preserved
const LAG_PTS: &[f32] = &[0.0]; // signed causal lag: sign = lead/lag direction; zero crossing is the decision
                                // burst_motion decision thresholds straight from the fixture's interpretation_guidelines:
                                // persistent_site "burst_motion ≈ 1.0", transient_site "burst_motion > 1.5". Visual domain is
                                // burst_motion/5 (vector_field_definition) ⇒ represent [0,5].
const BURST_PTS: &[f32] = &[1.0, 1.5];

// Geometry-default distortion bound (applied at authoring time; INV-7 still requires every field
// to carry an explicit value — `spec` makes it explicit here, not silently at runtime).
fn default_bound(g: &G) -> DB {
    match g {
        G::Enum { .. }
        | G::SpectralLine { .. }
        | G::Boolean
        | G::JoinKey
        | G::SmilesGrammar
        | G::OpcodeIsa
        | G::Opaque => DB::Exact,
        G::Sentinel { inner, .. } => default_bound(inner),
        G::Circular { .. } => DB::AbsoluteAngular(1.0), // 1 unit of the field's circular period
        G::LogMagnitude { .. } => DB::RelativeFraction(0.05),
        G::LinearBounded { .. } => DB::AbsoluteLinear(0.01),
        G::LatticeVq3d { .. } => DB::AbsoluteEuclidean(0.25),
    }
}

fn default_decision(g: &G) -> DS {
    match g {
        G::Enum { .. } | G::SpectralLine { .. } => DS::Categorical,
        _ => DS::None,
    }
}

fn spec(family: F, field: &'static str, geometry: G, info: I, canonical_safe: bool) -> FieldSpec {
    let max_distortion = default_bound(&geometry);
    let decision_relevance = default_decision(&geometry);
    FieldSpec {
        family,
        field,
        geometry,
        info_source: info,
        canonical_safe,
        max_distortion,
        decision_relevance,
    }
}

#[allow(clippy::too_many_arguments)]
fn spec_b(
    family: F,
    field: &'static str,
    geometry: G,
    info: I,
    canonical_safe: bool,
    max_distortion: DB,
    decision_relevance: DS,
) -> FieldSpec {
    FieldSpec {
        family,
        field,
        geometry,
        info_source: info,
        canonical_safe,
        max_distortion,
        decision_relevance,
    }
}

fn lin(lo: f32, hi: f32) -> G {
    G::LinearBounded { bits: 8, lo, hi }
}
fn unit() -> G {
    G::LinearBounded {
        bits: 8,
        lo: 0.0,
        hi: 1.0,
    }
}
fn logm(floor: f32, ceil: f32) -> G {
    G::LogMagnitude {
        bits: 10,
        floor,
        ceil,
    }
}
fn zero_logm(floor: f32, ceil: f32) -> G {
    G::Sentinel {
        value: 0.0,
        inner: Box::new(logm(floor, ceil)),
    }
}
fn enom(cardinality: u32) -> G {
    G::Enum { cardinality }
}

const UNCERTAINTY_TENSOR_FAMILIES: &[F] = &[
    F::ScientificUncertaintyTensor,
    F::RuntimeNumericalUncertaintyTensor,
    F::ThermalUncertaintyTensor,
    F::IntegratorUncertaintyTensor,
    F::ObservableUncertaintyTensor,
    F::SignalToUncertaintyTensor,
    F::EpochUncertaintyTensor,
];

const TSH_TRANSITION_FAMILIES: &[F] = &[
    F::ThermodynamicSalienceEvent,
    F::TransitionMotif,
    F::PersistentTransitionObject,
    F::TransitionConsensusEvent,
    F::TransitionEvidenceLedger,
];

const TSH_CONSENSUS_GATE_FAMILIES: &[F] = &[
    F::ScientificTransitionConsensusEvent,
    F::TransitionFalsificationProfile,
    F::RejectedPersistentTransitionCandidate,
    F::PersistentTransitionAcceptanceLedger,
];

const NATIVE_EPISTEMIC_SEQUENCE_FAMILIES: &[F] = &[
    F::EpistemicUncertainty,
    F::TeacherConsensus,
    F::ThermodynamicAction,
];

fn native_epistemic_sequence_specs(family: F) -> Vec<FieldSpec> {
    match family {
        F::EpistemicUncertainty => vec![
            spec_b(
                family,
                "sigma",
                G::LinearBounded {
                    bits: 16,
                    lo: 0.0,
                    hi: 1.0,
                },
                I::Empirical,
                true,
                DB::AbsoluteLinear(1.0 / ((1u32 << 16) as f32)),
                DS::Threshold {
                    points: &[0.05, 0.10, 0.25, 0.50],
                    region: (0.0, 1.0),
                },
            ),
            spec_b(
                family,
                "variance",
                G::LinearBounded {
                    bits: 16,
                    lo: 0.0,
                    hi: 1.0,
                },
                I::TeWindowedVariance,
                true,
                DB::AbsoluteLinear(1.0 / ((1u32 << 16) as f32)),
                DS::Threshold {
                    points: &[0.01, 0.05, 0.10, 0.25],
                    region: (0.0, 1.0),
                },
            ),
        ],
        F::TeacherConsensus => vec![
            spec(
                family,
                "entropy",
                G::LinearBounded {
                    bits: 16,
                    lo: 0.0,
                    hi: 8.0,
                },
                I::LabelConfidence,
                true,
            ),
            spec(
                family,
                "disagreement",
                G::LinearBounded {
                    bits: 16,
                    lo: 0.0,
                    hi: 1.0,
                },
                I::LabelConfidence,
                true,
            ),
            spec(family, "vote_state", enom(64), I::Fixed(6), true),
        ],
        F::ThermodynamicAction => vec![
            spec(family, "tsh_state", enom(64), I::Fixed(6), true),
            spec(
                family,
                "dt_fs",
                G::LogMagnitude {
                    bits: 16,
                    floor: 1.0e-6,
                    ceil: 1.0e12,
                },
                I::Empirical,
                true,
            ),
        ],
        _ => vec![],
    }
}

fn negative_scientific_evidence_specs() -> Vec<FieldSpec> {
    let family = F::NegativeScientificEvidenceEvent;
    vec![
        spec(family, "schema_kind", G::JoinKey, I::Fixed(0), true),
        spec(family, "schema_version", lin(0.0, 64.0), I::Fixed(6), true),
        spec(family, "negative_event_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "object_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_hash", G::JoinKey, I::Fixed(0), true),
        spec(family, "chronology_ref", G::JoinKey, I::Fixed(0), true),
        spec(family, "run_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "row_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "target_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "receptor_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "ligand_mode", enom(16), I::Fixed(4), true),
        spec(family, "detection_mode", enom(16), I::Fixed(4), true),
        spec(family, "replica_seed", G::JoinKey, I::Fixed(0), true),
        spec(family, "stream_id", lin(0.0, 4096.0), I::Empirical, true),
        spec(family, "epoch_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "chronology_event_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "frame_event_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_family", enom(64), I::Fixed(6), true),
        spec(family, "source_artifact", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_sha256", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_record_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "topology_sha256", G::JoinKey, I::Fixed(0), true),
        spec(family, "topology_status", enom(16), I::Fixed(4), true),
        spec(
            family,
            "negative_evidence_type",
            enom(32),
            I::Fixed(5),
            true,
        ),
        spec(family, "rejection_class", enom(32), I::Fixed(5), true),
        spec(family, "control_family", enom(32), I::Fixed(5), true),
        spec(family, "failed_claim_type", enom(32), I::Fixed(5), true),
        spec(family, "motif_status", enom(32), I::Fixed(5), true),
        spec(family, "rejection_reason", enom(64), I::Fixed(6), true),
        spec(family, "discovery_condition", enom(32), I::Fixed(5), true),
        spec(family, "failure_mode", enom(32), I::Fixed(5), true),
        spec(family, "source_claim", G::JoinKey, I::Fixed(0), true),
        spec(family, "rejected_claim", G::JoinKey, I::Fixed(0), true),
        spec(family, "prismos_opcode", enom(16), I::Fixed(4), true),
        spec(family, "opcode_family", enom(16), I::Fixed(4), true),
        spec(family, "transition_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "pocket_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "site_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "residue_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "opcode_context", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "prismcode_instruction_class",
            enom(16),
            I::Fixed(4),
            true,
        ),
        spec(
            family,
            "prismcode_instruction_context",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(family, "qnanl_artifact_family", enom(16), I::Fixed(4), true),
        spec(
            family,
            "uncertainty_tensor_ref",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(
            family,
            "uncertainty_sigma",
            zero_logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "signal_to_uncertainty",
            lin(-1.0e6, 1.0e6),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "snr",
            zero_logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "observed_value",
            lin(-1.0e12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "threshold_value",
            lin(-1.0e12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "baseline_value",
            lin(-1.0e12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "delta_value",
            lin(-1.0e12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "persistence_frames",
            logm(1.0, 1.0e9),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "replicate_count",
            logm(1.0, 1.0e6),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "supporting_seed_count",
            lin(0.0, 1024.0),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "failing_seed_count",
            lin(0.0, 1024.0),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "occurrence_count",
            logm(1.0, 1.0e9),
            I::Empirical,
            true,
        ),
        spec(family, "qnanl_codec_context", G::JoinKey, I::Fixed(0), true),
        spec(family, "confidence", unit(), I::LabelConfidence, true),
        spec(family, "severity", unit(), I::LabelConfidence, true),
        spec(family, "false_positive_status", enom(16), I::Fixed(4), true),
        spec(family, "suppression_status", enom(16), I::Fixed(4), true),
        spec(
            family,
            "final_merge_eligibility",
            enom(8),
            I::Fixed(3),
            true,
        ),
        spec(
            family,
            "recoverable_remaining",
            lin(0.0, 1.0e6),
            I::Empirical,
            true,
        ),
    ]
}

fn uncertainty_tensor_specs(family: F) -> Vec<FieldSpec> {
    vec![
        spec(family, "schema_version", lin(0.0, 8.0), I::Fixed(3), true),
        spec(family, "tensor_id", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "uncertainty_tensor_id",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(family, "mapping_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "row_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "run_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "epoch_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "stream_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "mode", enom(32), I::Fixed(5), true),
        spec(family, "tensor_kind", enom(32), I::Fixed(5), true),
        spec(family, "uncertainty_kind", enom(32), I::Fixed(5), true),
        spec(family, "observable", enom(64), I::Fixed(6), true),
        spec(family, "observable_family", enom(64), I::Fixed(6), true),
        spec(family, "source_family", enom(64), I::Fixed(6), true),
        spec(family, "uncertainty_channel", enom(64), I::Fixed(6), true),
        spec(family, "runtime_origin", enom(16), I::Fixed(4), true),
        spec(family, "tokenizer_family", enom(16), I::Fixed(4), true),
        spec(family, "measurement_window", G::JoinKey, I::Fixed(0), true),
        spec(family, "method", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_event_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_artifact_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_signal_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "target_tensor_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_refs", G::JoinKey, I::Fixed(0), true),
        spec(family, "sha256", G::JoinKey, I::Fixed(0), true),
        spec(family, "provenance_ref", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "scientific_uncertainty_tensor_ref",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(family, "signal_tensor_ref", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "signal_to_uncertainty_tensor_ref",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(
            family,
            "physics_consistency_tensor_ref",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(family, "chronology_ref", G::JoinKey, I::Fixed(0), true),
        spec(family, "spectral_ref", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_status", enom(16), I::Fixed(4), true),
        spec(family, "mu", lin(-1.0e9, 1.0e9), I::Empirical, true),
        spec(
            family,
            "sigma",
            zero_logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "variance",
            zero_logm(1.0e-24, 1.0e24),
            I::Empirical,
            true,
        ),
        spec(family, "n_samples", logm(1.0, 1.0e12), I::Empirical, true),
        spec(
            family,
            "evidence_count",
            logm(1.0, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "baseline_value",
            lin(-1.0e9, 1.0e9),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "observed_value",
            lin(-1.0e9, 1.0e9),
            I::Empirical,
            true,
        ),
        spec(family, "signal_value", unit(), I::Empirical, true),
        spec(
            family,
            "uncertainty_transfer_weight",
            unit(),
            I::Empirical,
            true,
        ),
        spec(family, "propagated_uncertainty", unit(), I::Empirical, true),
        spec(family, "epistemic_uncertainty", unit(), I::Empirical, true),
        spec(family, "aleatoric_uncertainty", unit(), I::Empirical, true),
        spec(family, "total_uncertainty", unit(), I::Empirical, true),
        spec(family, "confidence", unit(), I::Empirical, true),
        spec(
            family,
            "signal_delta",
            lin(-1.0e9, 1.0e9),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "signal_to_uncertainty",
            zero_logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(family, "consistency_score", unit(), I::Empirical, true),
        spec(
            family,
            "coordinate_rms_deviation",
            logm(1.0e-12, 1.0e9),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "force_rms_deviation",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "energy_variance",
            logm(1.0e-24, 1.0e24),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "pressure_variance",
            logm(1.0e-24, 1.0e24),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "virial_variance",
            logm(1.0e-24, 1.0e24),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "pme_variance",
            logm(1.0e-24, 1.0e24),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "integration_residual",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "constraint_residual",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "trajectory_divergence",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "energy_drift",
            lin(-1.0e12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "force_drift",
            lin(-1.0e12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "coordinate_divergence",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "contact_persistence_drift",
            lin(-1.0, 1.0),
            I::Empirical,
            true,
        ),
        spec(family, "measurement_status", enom(16), I::Fixed(4), true),
    ]
}

fn tsh_transition_specs(family: F) -> Vec<FieldSpec> {
    vec![
        spec(family, "schema_version", lin(0.0, 8.0), I::Fixed(3), true),
        spec(family, "schema_kind", enom(32), I::Fixed(5), true),
        spec(family, "transition_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "candidate_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "motif_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "consensus_id", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "persistent_transition_object_id",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(family, "evidence_ledger_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "stable_identifier", G::JoinKey, I::Fixed(0), true),
        spec(family, "version", lin(0.0, 4096.0), I::Fixed(12), true),
        spec(family, "lineage_ref", G::JoinKey, I::Fixed(0), true),
        spec(family, "supersedes_ref", G::JoinKey, I::Fixed(0), true),
        spec(family, "run_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "target_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "epoch_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "stream_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "seed_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "chronology_event_id", G::JoinKey, I::Fixed(0), true),
        spec(family, "chronology_ref", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "chronology_interval_start",
            lin(0.0, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "chronology_interval_end",
            lin(0.0, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(family, "step_start", lin(0.0, 1.0e12), I::Empirical, true),
        spec(family, "step_end", lin(0.0, 1.0e12), I::Empirical, true),
        spec(
            family,
            "time_start_ps",
            lin(0.0, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(family, "time_end_ps", lin(0.0, 1.0e12), I::Empirical, true),
        spec(
            family,
            "scientific_uncertainty_tensor_ref",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(
            family,
            "uncertainty_tensor_refs",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(family, "source_refs", G::JoinKey, I::Fixed(0), true),
        spec(family, "source_sha256", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "topology_hash_before",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(family, "topology_hash_after", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "topology_hash_transition",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(family, "residue_set_hash", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "spectral_signature_hash",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(
            family,
            "negative_control_ref",
            G::JoinKey,
            I::Fixed(0),
            true,
        ),
        spec(family, "provenance_ref", G::JoinKey, I::Fixed(0), true),
        spec(
            family,
            "force_delta",
            lin(-1.0e12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "force_delta_sigma",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "force_delta_snr",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "pocket_volume_delta",
            lin(-1.0e12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "pocket_volume_delta_sigma",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "pocket_volume_delta_snr",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "contact_delta",
            lin(-1.0e9, 1.0e9),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "contact_delta_sigma",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "contact_delta_snr",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(family, "causal_lag", lin(-1.0e6, 1.0e6), I::Empirical, true),
        spec(family, "lag_corr_peak", lin(-1.0, 1.0), I::Empirical, true),
        spec(
            family,
            "spectral_signature_delta",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(family, "replicate_recurrence", unit(), I::Empirical, true),
        spec(
            family,
            "replicate_count",
            logm(1.0, 1.0e9),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "consensus_seed_count",
            logm(1.0, 1.0e9),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "transition_stability_score",
            unit(),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "transition_recurrence_score",
            unit(),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "topology_persistence_score",
            unit(),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "chronology_persistence_score",
            unit(),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "uncertainty_normalized_magnitude",
            logm(1.0e-12, 1.0e12),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "spectral_similarity_score",
            unit(),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "control_rejection_score",
            unit(),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "scientific_evidence_score",
            unit(),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "aggregate_salience_score",
            unit(),
            I::Empirical,
            true,
        ),
        spec(
            family,
            "negative_control_status",
            enom(16),
            I::Fixed(4),
            true,
        ),
        spec(family, "acceptance_status", enom(16), I::Fixed(4), true),
        spec(family, "runtime_origin", enom(16), I::Fixed(4), true),
        spec(family, "tokenizer_family", enom(16), I::Fixed(4), true),
    ]
}

fn tsh_consensus_gate_specs(family: F) -> Vec<FieldSpec> {
    match family {
        F::ScientificTransitionConsensusEvent => vec![
            spec(family, "schema_kind", enom(16), I::Fixed(4), true),
            spec(family, "schema_version", lin(0.0, 16.0), I::Fixed(4), true),
            spec(family, "tokenizer_family", enom(16), I::Fixed(4), true),
            spec(family, "motif_id", G::JoinKey, I::Fixed(0), true),
            spec(family, "transition_id", G::JoinKey, I::Fixed(0), true),
            spec(family, "row_id", G::JoinKey, I::Fixed(0), true),
            spec(family, "mode", enom(32), I::Fixed(5), true),
            spec(family, "seed_set", G::JoinKey, I::Fixed(0), true),
            spec(family, "target_id", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "positive_evidence_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "negative_evidence_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "uncertainty_tensor_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "chronology_refs", G::JoinKey, I::Fixed(0), true),
            spec(family, "topology_hashes", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "spectral_signature_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "causal_geometry_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "residue_network_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "contact_support_observable",
                enom(16),
                I::Fixed(4),
                true,
            ),
            spec(
                family,
                "contact_support_semantic_equivalence",
                G::Boolean,
                I::Fixed(1),
                true,
            ),
            spec(
                family,
                "positive_evidence_score",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "negative_evidence_score",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "uncertainty_normalized_magnitude",
                logm(1.0e-12, 1.0e12),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "topology_persistence_score",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "chronology_persistence_score",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "spectral_recurrence_score",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "replicate_consensus_score",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "falsification_survival_score",
                unit(),
                I::Empirical,
                true,
            ),
            spec(family, "decision", enom(16), I::Fixed(4), true),
            spec(family, "decision_reason", enom(64), I::Fixed(6), true),
            spec(family, "confidence", unit(), I::LabelConfidence, true),
            spec(family, "provenance_ref", G::JoinKey, I::Fixed(0), true),
            spec(family, "opcode_context", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "prismcode_instruction_context",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "qnanl_codec_context", G::JoinKey, I::Fixed(0), true),
            spec(family, "stage_iii_eligible", G::Boolean, I::Fixed(1), true),
        ],
        F::TransitionFalsificationProfile => vec![
            spec(family, "schema_kind", enom(16), I::Fixed(4), true),
            spec(family, "schema_version", lin(0.0, 16.0), I::Fixed(4), true),
            spec(family, "tokenizer_family", enom(16), I::Fixed(4), true),
            spec(family, "motif_id", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "matched_negative_evidence_count",
                logm(1.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "strongest_negative_match",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "negative_match_classes",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "negative_match_reasons",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "negative_match_topology_overlap",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "negative_match_residue_overlap",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "negative_match_chronology_overlap",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "negative_match_spectral_overlap",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "rejected_by_falsification",
                G::Boolean,
                I::Fixed(1),
                true,
            ),
            spec(
                family,
                "survived_falsification",
                G::Boolean,
                I::Fixed(1),
                true,
            ),
            spec(
                family,
                "falsification_decision_reason",
                enom(64),
                I::Fixed(6),
                true,
            ),
            spec(
                family,
                "negative_evidence_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "provenance_ref", G::JoinKey, I::Fixed(0), true),
            spec(family, "source_sha256", G::JoinKey, I::Fixed(0), true),
            spec(family, "opcode_context", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "prismcode_instruction_context",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "qnanl_codec_context", G::JoinKey, I::Fixed(0), true),
        ],
        F::RejectedPersistentTransitionCandidate => vec![
            spec(family, "schema_kind", enom(16), I::Fixed(4), true),
            spec(family, "schema_version", lin(0.0, 16.0), I::Fixed(4), true),
            spec(family, "tokenizer_family", enom(16), I::Fixed(4), true),
            spec(family, "candidate_id", G::JoinKey, I::Fixed(0), true),
            spec(family, "source_motif_id", G::JoinKey, I::Fixed(0), true),
            spec(family, "rejection_status", enom(16), I::Fixed(4), true),
            spec(family, "rejection_reason", enom(64), I::Fixed(6), true),
            spec(family, "failed_gate", enom(32), I::Fixed(5), true),
            spec(
                family,
                "strongest_negative_evidence_ref",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "positive_evidence_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "negative_evidence_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "uncertainty_tensor_refs",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "chronology_refs", G::JoinKey, I::Fixed(0), true),
            spec(family, "topology_hashes", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "tokenizable_failure_mode",
                enom(64),
                I::Fixed(6),
                true,
            ),
            spec(family, "provenance_ref", G::JoinKey, I::Fixed(0), true),
            spec(family, "opcode_context", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "prismcode_instruction_context",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "qnanl_codec_context", G::JoinKey, I::Fixed(0), true),
            spec(family, "confidence", unit(), I::LabelConfidence, true),
        ],
        F::PersistentTransitionAcceptanceLedger => vec![
            spec(family, "schema_kind", enom(16), I::Fixed(4), true),
            spec(family, "schema_version", lin(0.0, 16.0), I::Fixed(4), true),
            spec(family, "tokenizer_family", enom(16), I::Fixed(4), true),
            spec(
                family,
                "candidate_count",
                logm(1.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "accepted_count",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "rejected_count",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(family, "partial_count", lin(0.0, 1.0e9), I::Empirical, true),
            spec(family, "promotion_rate", unit(), I::Empirical, true),
            spec(family, "rejection_rate", unit(), I::Empirical, true),
            spec(
                family,
                "false_positive_suppression_rate",
                unit(),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "motifs_rejected_by_negative_evidence",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "motifs_rejected_by_uncertainty",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "motifs_rejected_by_chronology",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "motifs_rejected_by_topology",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "motifs_rejected_by_consensus",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "motifs_promoted_to_persistent",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "stage_iii_eligible_count",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "stage_iii_blocked_count",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "corpus_merge_blockers",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "final_decision", enom(16), I::Fixed(4), true),
            spec(family, "provenance_ref", G::JoinKey, I::Fixed(0), true),
        ],
        _ => Vec::new(),
    }
}

fn scientific_surface_runtime_field_specs() -> Vec<FieldSpec> {
    let mut fields = Vec::new();
    for family in [
        F::ScientificPocketOpeningEvent,
        F::LigandCausalActivityEvent,
        F::ResidueActivityEvent,
    ] {
        fields.extend([
            spec(family, "event_id", G::JoinKey, I::Fixed(0), true),
            spec(family, "event_kind", G::JoinKey, I::Fixed(0), true),
            spec(family, "event_type", G::JoinKey, I::Fixed(0), true),
            spec(family, "chronology_event_id", G::JoinKey, I::Fixed(0), true),
            spec(family, "run_id", G::JoinKey, I::Fixed(0), true),
            spec(family, "target", G::JoinKey, I::Fixed(0), true),
            spec(family, "stream_id", lin(0.0, 4096.0), I::Empirical, true),
            spec(family, "frame_idx", lin(0.0, 1.0e9), I::Empirical, true),
            spec(family, "frame_index", lin(0.0, 1.0e9), I::Empirical, true),
            spec(family, "step", lin(0.0, 1.0e12), I::Empirical, true),
            spec(
                family,
                "physical_time_fs",
                lin(0.0, 1.0e15),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "emitted_unix_ns",
                lin(0.0, 1.0e18),
                I::Empirical,
                true,
            ),
            spec(family, "runtime_origin", enom(8), I::Fixed(3), true),
            spec(family, "runtime_native", G::Boolean, I::Fixed(1), true),
            spec(
                family,
                "chronology_fidelity_level",
                enom(16),
                I::Fixed(4),
                true,
            ),
            spec(
                family,
                "capture_cadence_class",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "chronology_marker_scope",
                enom(8),
                I::Fixed(3),
                true,
            ),
            spec(
                family,
                "interferometric_marker_kind",
                enom(16),
                I::Fixed(4),
                true,
            ),
            spec(family, "interferometric_mode", enom(8), I::Fixed(3), true),
            spec(
                family,
                "geometry_authority_status",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "physical_time_binding_status",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "step_binding_status", G::JoinKey, I::Fixed(0), true),
            spec(family, "status", G::JoinKey, I::Fixed(0), true),
            spec(family, "schema_kind", G::JoinKey, I::Fixed(0), true),
            spec(family, "schema_version", lin(0.0, 64.0), I::Fixed(6), true),
            spec(family, "source_kind", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "source_artifact_kind",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "record_kind", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "scientific_event_state",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(family, "artifact_family", G::JoinKey, I::Fixed(0), true),
            spec(family, "opcode_name", G::JoinKey, I::Fixed(0), true),
            spec(family, "operator_name", G::JoinKey, I::Fixed(0), true),
            spec(family, "topology_path", G::JoinKey, I::Fixed(0), true),
            spec(family, "topology_sha256", G::JoinKey, I::Fixed(0), true),
            spec(
                family,
                "trajectory_frame_path",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "trajectory_frame_byte_start",
                lin(0.0, 1.0e12),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "trajectory_frame_byte_end",
                lin(0.0, 1.0e12),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "trajectory_frame_byte_end_exclusive",
                lin(0.0, 1.0e12),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "trajectory_frame_n_floats",
                lin(0.0, 1.0e9),
                I::Empirical,
                true,
            ),
            spec(
                family,
                "trajectory_frame_hash64",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "trajectory_frame_rolling_hash64",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "trajectory_frame_sha256_status",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "trajectory_origin_receipt_path",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "prismcode_substrate_receipt_path",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
            spec(
                family,
                "spectral_geometry_authority_status",
                G::JoinKey,
                I::Fixed(0),
                true,
            ),
        ]);
    }
    fields
}

/// The full §7 field table. Deterministic order (field-slot assignment follows this order).
pub fn field_table() -> Vec<FieldSpec> {
    use I::*;
    let mut fields = vec![
        // --- A1 SPIKE_EVENT ---------------------------------------------------------------
        spec(
            F::SpikeEvent,
            "wavelength_nm",
            G::SpectralLine {
                lines: UV_LINES,
                residual_bits: 4,
            },
            Empirical,
            true,
        ),
        spec(F::SpikeEvent, "spike_source", enom(7), Fixed(3), true), // 0=NONE 1=UV 2=LIF 3=EFP 4=LADD 5=COFIRE 6=VORONOI_LIF
        spec(
            F::SpikeEvent,
            "position",
            G::LatticeVq3d {
                codebook_bits: 12,
                lattice: Lattice::LearnedVqEma,
            },
            Empirical,
            true,
        ),
        spec_b(
            F::SpikeEvent,
            "ccns_phase",
            G::Circular {
                bits: 10,
                period: CircPeriod::PhaseBits(10),
            },
            Empirical,
            true,
            DB::AbsoluteAngular(1.0),
            DS::None,
        ),
        spec_b(
            F::SpikeEvent,
            "phase_bits",
            G::Circular {
                bits: 10,
                period: CircPeriod::PhaseBits(10),
            },
            Empirical,
            true,
            DB::AbsoluteAngular(1.0),
            DS::None,
        ),
        spec(F::SpikeEvent, "intensity", logm(1e-4, 1e4), Empirical, true),
        spec(
            F::SpikeEvent,
            "aromatic_residue_id",
            G::Sentinel {
                value: -1.0,
                inner: Box::new(G::JoinKey),
            },
            Fixed(0),
            true,
        ),
        spec(F::SpikeEvent, "aromatic_type", enom(5), Fixed(3), true),
        spec(
            F::SpikeEvent,
            "water_density",
            logm(1e-3, 1e2),
            Empirical,
            true,
        ),
        spec(
            F::SpikeEvent,
            "n_nearby_excited",
            lin(0.0, 64.0),
            Empirical,
            true,
        ),
        // --- V2 SCHEMA ELEVATION (Oracle V2 manifold) --------------------------------------
        // Hoist the n80 spike physics OUT of the opaque lossless sidecar into first-class
        // typed channels. APPENDED at the SpikeEvent tail so every existing field_slot is
        // UNCHANGED (INV-2: same family keeps its discriminants; only new slots are added).
        // Ranges grounded in the REAL spike_events_snr_masked.parquet (row-group sampled,
        // 60,035 rgs): time_ps 0.024→~12ps and GROWING across the trajectory (valid dt_fs
        // source); wd_change ~0.0198–0.0334 (signed headroom for the delta); vibrational_energy
        // bimodal 0 / ~47 (p50=0, p99≈47) — a naturally sparse thermodynamic fire signal.
        spec(F::SpikeEvent, "time_ps", logm(1e-3, 1e6), Empirical, true),
        spec(F::SpikeEvent, "wd_change", lin(-0.1, 0.1), Empirical, true),
        // spec_b with an EXPLICIT distortion bound: lin(0,100)'s 8-bit step (~0.39) exceeds the
        // tight default_bound() that suits narrow fields (wd_change), so plain spec() escapes via
        // INV-7. AbsoluteLinear(0.5) sizes the bound to this range (step 0.39 < 0.5 → never escapes)
        // and — unlike a RelativeFraction — tolerates the 0-valued baseline spikes.
        spec_b(
            F::SpikeEvent,
            "vibrational_energy",
            lin(0.0, 100.0),
            Empirical,
            true,
            DB::AbsoluteLinear(0.5),
            DS::None,
        ),
        // --- A4 TEACHER/STUDENT CONSENSUS (decision-critical convergence stats) ------------
        // Field names are the EXACT parquet column names (teacher_consensus_residue_labels.parquet,
        // ensemble_aggregates.parquet, student_label_bundle.parquet; corpus shard sha 2146c1ee…03a5cc).
        spec(
            F::TeacherStudentConsensus,
            "cryptic_likelihood_label",
            unit(),
            LabelConfidence,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "phase_manifold_score_mean",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "phase_manifold_score_std",
            unit(),
            Empirical,
            true,
        ),
        spec_b(
            F::TeacherStudentConsensus,
            "rhat",
            logm(1.0, 4.0),
            Empirical,
            true,
            DB::AbsoluteLinear(0.0025),
            DS::Threshold {
                points: RHAT_PTS,
                region: (1.0, 1.2),
            },
        ),
        spec_b(
            F::TeacherStudentConsensus,
            "ess",
            logm(1.0, 1e4),
            Empirical,
            true,
            DB::RelativeFraction(0.05),
            DS::Threshold {
                points: ESS_PTS,
                region: (0.0, 4000.0),
            },
        ),
        spec_b(
            F::TeacherStudentConsensus,
            "label_confidence",
            unit(),
            LabelConfidence,
            true,
            DB::AbsoluteLinear(0.01),
            DS::Threshold {
                points: CONF_PTS,
                region: (0.0, 1.0),
            },
        ),
        spec(
            F::TeacherStudentConsensus,
            "is_bimodal",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "hard_negative_mask",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "train_mask",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "student_weight",
            unit(),
            Empirical,
            true,
        ), // real range [0, 0.52]
        spec(
            F::TeacherStudentConsensus,
            "residue_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        // v11 — Directive v6.1 Track B: the PER-PAIR teacher/student BRIDGE training labels
        // (PRISM_V2_ROW_MOLECULE_BRIDGE_TRAINING_TABLE.parquet, 1500 rows × 226 cols). These are
        // APPENDED LAST (existing 11 fields keep field-slots 0..10; new fields take 11..; INV-2
        // namespaces stay stable). Field names are the EXACT parquet columns; ranges MEASURED from
        // the real table (no guessing). All canonical_safe=true: the teacher head is decision/label
        // evidence (LabelConfidence/Empirical), NOT a generative firewall artifact. The molecule__*
        // enrichment block rides the GflownetTrajectory/MoleculeGeneration routers or the LOSSLESS
        // sidecar (push_opaque), NOT re-spec'd here (Agent 2 route_gap closed by reuse).
        // pair identity + training masks (pair_label 0/1; pair_weight 0.5..1.0; abstention 0/1)
        spec(
            F::TeacherStudentConsensus,
            "pair_label",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "pair_weight",
            unit(),
            Empirical,
            true,
        ), // [0.5, 1.0]
        spec(
            F::TeacherStudentConsensus,
            "abstention_target",
            G::Boolean,
            Fixed(1),
            true,
        ), // {0,1}
        // distillation / loss weights (measured: teacher 0.058..0.68; distillation 0.078..0.63)
        spec(
            F::TeacherStudentConsensus,
            "teacher_target",
            enom(8),
            Fixed(3),
            true,
        ), // ABSTAIN_ESCALATE/PROMOTE_COMPUTATIONAL_REVIEW/TEACHER_REVIEW
        spec(
            F::TeacherStudentConsensus,
            "teacher_loss_weight",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "distillation_loss_weight",
            unit(),
            Empirical,
            true,
        ),
        // soft targets (uncertainty 0.25..0.95; positive 0..0.55; negative 0..1)
        spec(
            F::TeacherStudentConsensus,
            "uncertainty_target",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "positive_target",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "negative_target",
            unit(),
            Empirical,
            true,
        ),
        // prism meta-reward model head (uncertainty 0.21..0.82; coverage 0.78..1.0)
        spec(
            F::TeacherStudentConsensus,
            "model_uncertainty",
            unit(),
            LabelConfidence,
            true,
        ), // == row__prism_meta_reward_model_uncertainty
        spec(
            F::TeacherStudentConsensus,
            "model_coverage",
            unit(),
            Empirical,
            true,
        ), // == row__prism_meta_reward_model_coverage
        spec(
            F::TeacherStudentConsensus,
            "evidence_quality_score",
            unit(),
            Empirical,
            true,
        ), // 0.91..1.0
        spec(
            F::TeacherStudentConsensus,
            "integrated_score",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ), // signed −0.26..0.17
        spec(
            F::TeacherStudentConsensus,
            "phase_coherence_score",
            unit(),
            Empirical,
            true,
        ), // 0.44..0.77
        spec(
            F::TeacherStudentConsensus,
            "gflownet_reward",
            unit(),
            Empirical,
            true,
        ), // 0.55..0.66 (label-side, NOT firewalled)
        spec(
            F::TeacherStudentConsensus,
            "gflownet_reward_fusion_target",
            unit(),
            Empirical,
            true,
        ),
        // multimodal FUSION evidence (each [0,1]-ish; warp_strain tiny → logm; measured ranges):
        spec(
            F::TeacherStudentConsensus,
            "fusion_kcc_coherence",
            unit(),
            Empirical,
            true,
        ), // 0.81..1.0
        spec(
            F::TeacherStudentConsensus,
            "fusion_kinetic_durability",
            unit(),
            Empirical,
            true,
        ), // 0.53..0.80
        spec(
            F::TeacherStudentConsensus,
            "fusion_hydration_gate_behavior",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_uncertainty",
            unit(),
            LabelConfidence,
            true,
        ), // 0.21..0.83
        spec(
            F::TeacherStudentConsensus,
            "fusion_evidence_coverage_score",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_force_stress",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_warp_strain",
            logm(1e-5, 1e0),
            Empirical,
            true,
        ), // 2e-4..8e-4 tiny → logm+fit
        spec(
            F::TeacherStudentConsensus,
            "fusion_receptor_recovery",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_raw_spike_activity",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_raw_spike_entropy",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_productive_activation_score",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_quiet_lock_risk",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_variant_rescue_signal",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_receptor_state_coherence",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_fiber_transport_coherence",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "fusion_chronology_alignment",
            unit(),
            Empirical,
            true,
        ),
        // signed delta evidence (each tiny, signed; measured ranges within ±0.13):
        spec(
            F::TeacherStudentConsensus,
            "delta_functional_score",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "delta_survival_auc",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "delta_temporal_score",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "delta_thermodynamic_score",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "delta_water_score_v2",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        // string-enum labels (vocab MEASURED; unknown escapes losslessly, INV-1/INV-4):
        spec(
            F::TeacherStudentConsensus,
            "evidence_role",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "phase_evidence_tier",
            enom(4),
            Fixed(2),
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "water_class_v2",
            enom(4),
            Fixed(2),
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "negative_family",
            enom(4),
            Fixed(2),
            true,
        ), // wrong_candidate/wrong_condition
        spec(
            F::TeacherStudentConsensus,
            "contradiction_flag",
            enom(4),
            Fixed(2),
            true,
        ),
        // OOD / discard flags (booleans)
        spec(
            F::TeacherStudentConsensus,
            "ood_or_missing_family_flag",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "do_not_discard",
            G::Boolean,
            Fixed(1),
            true,
        ),
        // pair / molecule join keys (string ids → escape token + exact sidecar Text; INV-1/INV-4)
        spec(
            F::TeacherStudentConsensus,
            "pair_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::TeacherStudentConsensus,
            "molecule_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        // --- A5 DSTW / FIBER --------------------------------------------------------------
        spec(
            F::FiberBundleDstw,
            "adaptive_dt",
            logm(1e-6, 1e0),
            TeWindowedVariance,
            true,
        ),
        spec_b(
            F::FiberBundleDstw,
            "failure_probability",
            unit(),
            Empirical,
            true,
            DB::AbsoluteLinear(0.01),
            DS::None,
        ),
        spec(
            F::FiberBundleDstw,
            "fisher_info",
            unit(),
            TeWindowedVariance,
            true,
        ), // == te_windowed_variance
        spec(
            F::FiberBundleDstw,
            "confidence_identity",
            unit(),
            Empirical,
            true,
        ),
        // --- A6 BOCPD ---------------------------------------------------------------------
        spec_b(
            F::BocpdEvent,
            "bocpd_probability",
            unit(),
            Empirical,
            true,
            DB::AbsoluteLinear(0.01),
            DS::None,
        ),
        spec(F::BocpdEvent, "phase_from", enom(16), Fixed(4), true),
        spec(F::BocpdEvent, "phase_to", enom(16), Fixed(4), true),
        spec(F::BocpdEvent, "trigger_kind", enom(16), Fixed(4), true),
        // --- A7 GFLOWNET (generative; canonical_safe=false ⇒ firewall, INV-3) -------------
        spec_b(
            F::GflownetTrajectory,
            "reward",
            unit(),
            TrajectoryEntropy,
            false,
            DB::RelativeFraction(0.02),
            DS::None,
        ),
        spec_b(
            F::GflownetTrajectory,
            "cryptic_bonus",
            unit(),
            TrajectoryEntropy,
            false,
            DB::RelativeFraction(0.02),
            DS::None,
        ),
        spec_b(
            F::GflownetTrajectory,
            "adjusted_pi_clash",
            unit(),
            TrajectoryEntropy,
            false,
            DB::RelativeFraction(0.02),
            DS::None,
        ),
        spec(
            F::GflownetTrajectory,
            "survival_tier",
            enom(8),
            Fixed(3),
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "trajectory_entropy",
            logm(1e-3, 1e2),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "policy_logprob",
            lin(-40.0, 0.0),
            TrajectoryEntropy,
            false,
        ),
        spec_b(
            F::GflownetTrajectory,
            "sampled_dihedral_deg",
            G::Circular {
                bits: 9,
                period: CircPeriod::Deg360,
            },
            Empirical,
            false,
            DB::AbsoluteAngular(1.0),
            DS::None,
        ),
        spec(
            F::GflownetTrajectory,
            "canonical_smiles",
            G::SmilesGrammar,
            Empirical,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "oracle_valid",
            G::Boolean,
            Fixed(1),
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "anchor_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        // v10 — Directive v6 Agent 3/4: the 52-col gflownet_raw_policy_samples reward-component +
        // sampling-control fields (real SAMPLER_OUTPUT shard, 1e6 rows). Ranges MEASURED from the
        // shard (regime∈{exploitation,focused,balanced,exploration,near_random}; validity_status
        // "valid"; temperature 0.1..1.0; policy_logprob −16.9..−5.7; trajectory_entropy 0.66..9.5;
        // u_pose −0..1.79; backward_logprob nan-prone ⇒ lin escapes a NaN losslessly). All
        // canonical_safe=false (generative firewall, INV-3). pathway/sigma/lock telemetry are NOT
        // canonical evidence — they ride the same firewall and never reach a canonical path.
        spec(F::GflownetTrajectory, "regime", enom(8), Fixed(3), false), // sampling temperature regime
        spec(
            F::GflownetTrajectory,
            "temperature",
            lin(0.0, 4.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "backward_logprob",
            lin(-40.0, 0.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "u_pose",
            lin(-4.0, 4.0),
            Empirical,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "sampled_action_idx",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "sampled_attachment_site",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "synthon_a_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "synthon_b_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "product_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "rotamers_evaluated",
            logm(1.0, 1e3),
            Empirical,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "best_rotamer_rank",
            enom(64),
            Fixed(6),
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "conformer_count",
            logm(1.0, 1e3),
            Empirical,
            false,
        ),
        // pathway / lock / sigma reward-component telemetry (generative firewall; canonical_safe=false)
        spec(
            F::GflownetTrajectory,
            "sigma_shear_mean",
            lin(-1e2, 1e2),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "hysteresis_mean",
            lin(-1e2, 1e2),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "reversibility_mean",
            lin(-1e2, 1e2),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "pathway_voxels_occupied",
            logm(1.0, 1e6),
            Empirical,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "pathway_neighborhood_contacts",
            logm(1.0, 1e6),
            Empirical,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "pathway_neighborhood_score_mean",
            lin(-1e2, 1e2),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "pathway_score_mean",
            lin(-1e2, 1e2),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "species_conservation_score_mean",
            lin(-1e2, 1e2),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "field_consensus_complement_bonus",
            lin(-1e2, 1e2),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::GflownetTrajectory,
            "charge_feature_mean",
            lin(-1e2, 1e2),
            TrajectoryEntropy,
            false,
        ),
        // --- A26 MOLECULE_GENERATION (Directive v6 Agent 3/4: enamine synthons + V-space products).
        //     Had ZERO field specs before v10. Geometries + enum vocab + numeric ranges MEASURED from
        //     enamine_115k_synthons_3d.parquet + vspace_survivors_{scaffold,population}_consensus_
        //     action_corpus.parquet (the Agent-1 eligibility-ledger MoleculeGeneration sources).
        //     ALL canonical_safe=false: these are GENERATIVE molecule-construction artifacts (firewall,
        //     INV-3) — never PRISM canonical lexicographic ranking truth (CLAUDE.md rule #1; `score`
        //     is the generative composite, NOT a persistence/pass_fraction/stability/quality tuple).
        //     JSON-blob columns (conformer_atoms_json / partial_charges_json / *_json / coordinates_json)
        //     ride the dedicated router's LOSSLESS capture (push_opaque, §10/INV-4), NOT a typed spec. -
        // identity / structure
        spec(
            F::MoleculeGeneration,
            "canonical_smiles",
            G::SmilesGrammar,
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "smiles",
            G::SmilesGrammar,
            Empirical,
            false,
        ), // raw SMILES, distinct
        spec(
            F::MoleculeGeneration,
            "is_valid_mol",
            G::Boolean,
            Fixed(1),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "oracle_valid",
            G::Boolean,
            Fixed(1),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "z_matrix_active",
            G::Boolean,
            Fixed(1),
            false,
        ),
        // join keys (string ids → escape token + exact sidecar Text; INV-1/INV-4)
        spec(
            F::MoleculeGeneration,
            "synthon_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "vendor_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "anchor_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "product_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "synthon_a_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "synthon_b_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "reaction_rule_id",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        // atom-index pointers into the SMILES (relational anchors, not magnitudes)
        spec(
            F::MoleculeGeneration,
            "attachment_atom_idx",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "leaving_group_atom_idx",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "dihedral_reference_atom_idx",
            G::JoinKey,
            Fixed(0),
            false,
        ),
        // counts / magnitudes (measured: heavy_atom_count 2..19; molecular_weight 59..252; charges 0..2)
        spec(
            F::MoleculeGeneration,
            "formal_charge",
            lin(-8.0, 8.0),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "leaving_group_formal_charge",
            lin(-8.0, 8.0),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "heavy_atom_count",
            logm(1.0, 1e3),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "molecular_weight",
            logm(1e1, 1e4),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "conformer_count",
            logm(1.0, 1e3),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "cryptic_bonus_atoms",
            logm(1.0, 1e3),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "rotamers_evaluated",
            logm(1.0, 1e3),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "best_rotamer_rank",
            enom(64),
            Fixed(6),
            false,
        ),
        // generative reward components (measured ranges: score −0.1..2.65; pi_complement 1..7;
        // pi_clash 0.35..3.5; cryptic_bonus 0..14). NOT unit() — real ranges exceed [0,1].
        spec(
            F::MoleculeGeneration,
            "score",
            lin(-1.0, 4.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "pi_complement",
            lin(0.0, 16.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "pi_clash",
            lin(0.0, 8.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "fragment_pi_complement",
            lin(0.0, 16.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "fragment_pi_clash_adjusted",
            lin(0.0, 8.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "cryptic_bonus",
            lin(0.0, 32.0),
            TrajectoryEntropy,
            false,
        ),
        spec_b(
            F::MoleculeGeneration,
            "selected_dihedral_deg",
            G::Circular {
                bits: 9,
                period: CircPeriod::Deg360,
            },
            Empirical,
            false,
            DB::AbsoluteAngular(1.0),
            DS::None,
        ),
        // scaffold / population consensus bonuses (measured: liability 51..113; atoms_scored 102..143;
        // bonus 0.13..0.52; bonus_scaled 0.27..1.04)
        spec(
            F::MoleculeGeneration,
            "scaffold_consensus_bonus",
            lin(-4.0, 4.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "consensus_complement_bonus",
            lin(-4.0, 4.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "scaffold_consensus_liability",
            lin(0.0, 256.0),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "scaffold_consensus_atoms_scored",
            logm(1.0, 1e3),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "scaffold_consensus_bonus_scaled",
            lin(-4.0, 4.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "population_consensus_bonus",
            lin(-4.0, 4.0),
            TrajectoryEntropy,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "population_consensus_liability",
            lin(0.0, 256.0),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "population_consensus_atoms_scored",
            logm(1.0, 1e3),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "population_consensus_bonus_scaled",
            lin(-4.0, 4.0),
            TrajectoryEntropy,
            false,
        ),
        // enums (string vocab MEASURED; unknown labels escape losslessly, INV-1/INV-4)
        spec(
            F::MoleculeGeneration,
            "survival_tier",
            enom(8),
            Fixed(3),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "reaction_role",
            enom(4),
            Fixed(2),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "ingest_status",
            enom(4),
            Fixed(2),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "charge_method",
            enom(8),
            Fixed(3),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "partial_charge_method",
            enom(8),
            Fixed(3),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "assembly_mode",
            enom(4),
            Fixed(2),
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "pose_reconciliation_method",
            enom(4),
            Fixed(2),
            false,
        ),
        // v13 — decoded Enamine/AM1-BCC and conformer summaries. The raw JSON arrays
        // (`partial_charges_json`, `conformer_atoms_json`, `coordinates_json`) remain lossless in
        // the sidecar; these compact derived fields give the packed-token channel direct charge and
        // 3-D geometry signal without making the tokenizer ingest rowstore blobs as opaque bytes.
        spec(
            F::MoleculeGeneration,
            "partial_charge_mean",
            lin(-2.0, 2.0),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "partial_charge_abs_mean",
            logm(1e-5, 4.0),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "partial_charge_min",
            lin(-2.0, 2.0),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "partial_charge_max",
            lin(-2.0, 2.0),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "partial_charge_l2",
            logm(1e-5, 32.0),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "conformer_atom_count",
            logm(1.0, 1e3),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "conformer_radius_gyration",
            logm(1e-3, 1e2),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "conformer_extent_max",
            logm(1e-3, 1e2),
            Empirical,
            false,
        ),
        spec(
            F::MoleculeGeneration,
            "attachment_vector_norm",
            logm(1e-5, 1e2),
            Empirical,
            false,
        ),
        // --- A9 GPU_TELEMETRY -------------------------------------------------------------
        spec(
            F::GpuTelemetry,
            "cuda_graph_replay_ms",
            logm(1e-2, 1e5),
            Empirical,
            true,
        ),
        spec(F::GpuTelemetry, "tb_loss", logm(1e-4, 1e2), Empirical, true),
        spec(
            F::GpuTelemetry,
            "gpu_solve_count",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        // --- A10 RECEIPT_STATUS (INV-5) ----------------------------------------------------
        spec(F::ReceiptStatus, "status", enom(3), Fixed(2), true),
        spec(
            F::ReceiptStatus,
            "structural_verified",
            G::Boolean,
            Fixed(1),
            true,
        ),
        // --- A13 TWIN_COUPLED -------------------------------------------------------------
        spec(F::TwinCoupled, "agreement", unit(), Empirical, true),
        spec(F::TwinCoupled, "lag", lin(-512.0, 512.0), Empirical, true),
        spec(F::TwinCoupled, "barrier", logm(1e-3, 1e2), Empirical, true),
        spec(
            F::TwinCoupled,
            "ccf_width",
            logm(1e-2, 1e3),
            Empirical,
            true,
        ),
        spec_b(
            F::TwinCoupled,
            "te_asymmetry",
            lin(-1.0, 1.0),
            Empirical,
            true,
            DB::RelativeFraction(0.05),
            DS::Threshold {
                points: TE_PTS,
                region: (-1.0, 1.0),
            },
        ),
        // REAL per-residue TWIN coupling evidence (twin_per_residue.json from a memory-safe 8-engine
        // coupled-TWIN run; 422 residues, joinable by resid). Populates the reserved TwinCoupled
        // manifest-gap family with measured fields (v6→v7).
        spec(F::TwinCoupled, "resid", G::JoinKey, Fixed(0), true),
        spec(
            F::TwinCoupled,
            "resname",
            enom(AA_NAMES.len() as u32),
            Fixed(5),
            true,
        ),
        spec(
            F::TwinCoupled,
            "ca_xyz",
            G::LatticeVq3d {
                codebook_bits: 12,
                lattice: Lattice::LearnedVqEma,
            },
            Empirical,
            true,
        ),
        spec(
            F::TwinCoupled,
            "b_over_a_ratio",
            lin(0.0, 2.0),
            Empirical,
            true,
        ), // A↔B coupling asymmetry (≈1 symmetric)
        spec(F::TwinCoupled, "barrier_class", enom(8), Fixed(3), true), // LOW/MEDIUM/HIGH
        spec(F::TwinCoupled, "ccf_max", lin(-1.0, 1.0), Empirical, true), // peak cross-correlation
        spec(F::TwinCoupled, "ccf_mean", lin(-1.0, 1.0), Empirical, true),
        spec(
            F::TwinCoupled,
            "ccf_centrality",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::TwinCoupled,
            "consensus_score",
            logm(1.0, 1e7),
            Empirical,
            true,
        ),
        spec(
            F::TwinCoupled,
            "differential_score",
            unit(),
            Empirical,
            true,
        ),
        spec(F::TwinCoupled, "spikes_a", logm(1.0, 1e9), Empirical, true),
        spec(F::TwinCoupled, "spikes_b", logm(1.0, 1e9), Empirical, true),
        spec(
            F::TwinCoupled,
            "total_spikes",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::TwinCoupled,
            "warm_hold_fraction",
            unit(),
            Empirical,
            true,
        ),
        // --- A16 RANKER_ROW (ordered tuple, no composite scalar) ---------------------------
        spec(F::RankerRow, "persistence_rank", enom(64), Fixed(6), true),
        spec(F::RankerRow, "pass_fraction_rank", enom(64), Fixed(6), true),
        spec(F::RankerRow, "stability_rank", enom(64), Fixed(6), true),
        spec(F::RankerRow, "quality_rank", enom(64), Fixed(6), true),
        // --- A16 RANKER_ROW pathb candidate carrier: site_candidates.json /candidates[] rows
        //     (schema_kind=pathb_site_candidates) + ranking_ablation_report.json scheme rows.
        //     ADDITIVE: RankerRow field-slots 4-8; existing slots 0-3 (persistence..quality_rank)
        //     and every OTHER family are byte-stable (per-family slotting, registry.rs). route_object
        //     only encodes keys present per record, so the consensus-ranker rows (persistence_rank..)
        //     and the pathb voxel-peak rows (rank/peak_id/spike_count) coexist collision-free.
        //     json_key_for default (routing.rs `_ => field`) => json key == field name (by value).
        //     rank/peak_id/region_id are exact ordinals/ids (JoinKey => encode_scalar exact, same
        //     path as KccDescriptor residue_id). spike_count is the density SCORE (count magnitude
        //     spans ~1980..3105 -> logm); score is the ablation normalized SCORE (0..~0.1). ------
        spec(F::RankerRow, "rank", G::JoinKey, Fixed(0), true), // ordinal 0..top_k-1 (exact)
        spec(F::RankerRow, "peak_id", G::JoinKey, Fixed(0), true), // voxel peak identity (exact)
        spec(
            F::RankerRow,
            "spike_count",
            logm(1.0, 1.0e8),
            Empirical,
            true,
        ), // density SCORE
        spec(F::RankerRow, "region_id", G::JoinKey, Fixed(0), true), // ablation region id (exact)
        spec(F::RankerRow, "score", lin(0.0, 1.0), Empirical, true), // ablation normalized SCORE
        // --- A17 KCC (real engine kcc_visualization.json residues[]; fixture sha 51bff23b…71edbe).
        //     `coherence` was a fiction (no such field) — REPLACED by the 16 real per-residue
        //     descriptor fields. Field-slot order below = stable token order. -------------------
        spec(F::KccDescriptor, "residue_id", G::JoinKey, Fixed(0), true),
        spec(
            F::KccDescriptor,
            "residue_name",
            enom(AA_NAMES.len() as u32),
            Fixed(5),
            true,
        ),
        spec(
            F::KccDescriptor,
            "ca_position",
            G::LatticeVq3d {
                codebook_bits: 12,
                lattice: Lattice::LearnedVqEma,
            },
            Empirical,
            true,
        ),
        spec(F::KccDescriptor, "kcc_score", unit(), Empirical, true),
        spec(F::KccDescriptor, "direction_score", unit(), Empirical, true),
        spec(
            F::KccDescriptor,
            "motion_efficiency",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "burst_motion",
            lin(0.0, 4.0),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "causal_lag",
            lin(-128.0, 128.0),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "lag_corr_peak",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "local_cov",
            logm(1e-6, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "sum_motion",
            logm(1e-4, 1e4),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "active_causal_steps",
            logm(1.0, 1e7),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "total_steps",
            logm(1.0, 1e7),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "net_dx",
            lin(-10.0, 10.0),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "net_dy",
            lin(-10.0, 10.0),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptor,
            "net_dz",
            lin(-10.0, 10.0),
            Empirical,
            true,
        ),
        // --- A18 KCC_SITE (real kcc_visualization.json sites[]; fixture sha 51bff23b…71edbe).
        //     A site is a RELATION over residues + spatial/temporal/causal/rank structure — NOT a
        //     widened residue row. Geometry is from the fixture's OWN semantics + measured ranges
        //     (see OPERATOR_CONTROL/KCC_SITE_RIGOR_LEDGER.md), not by analogy. Distinct namespace
        //     from KccDescriptor; NOT binding_sites.json (the same 65 sites, joined by `id`). Member
        //     ids join residues[].residue_id (93/93 exact; no translation). -----------------------
        // spatial
        spec(
            F::KccSite,
            "centroid",
            G::LatticeVq3d {
                codebook_bits: 12,
                lattice: Lattice::LearnedVqEma,
            },
            Empirical,
            true,
        ),
        spec(F::KccSite, "volume", logm(1e1, 1e4), Empirical, true), // 50..2005 Å³, decade-spanning (corpus-fit)
        // identity + ordered rank tuple. rank_C/G/K/L/T/rank_score are ENGINE COMPOSITES (semantics:
        // C=sqrt(coupling+causality+voxel), G=volume×(1-hydrophilicity)×compactness, …) — routed as
        // decoded evidence, NOT PRISM canonical lexicographic ranking truth (CLAUDE.md rule #1).
        spec(F::KccSite, "id", G::JoinKey, Fixed(0), true),
        // gtck_rank: ordinal rank 1..sites.len(). enom uses IDENTITY codes (code==rank ⇒ order is
        // preserved in the token value + exact in the sidecar). Registry cardinality is a fixed
        // ceiling (128) — ranks beyond escape; the true cap is sites.len() (ledger §gtck_rank).
        spec(F::KccSite, "gtck_rank", enom(128), Fixed(7), true),
        spec(F::KccSite, "rank_C", unit(), Empirical, true),
        spec(F::KccSite, "rank_G", unit(), Empirical, true),
        spec(F::KccSite, "rank_K", unit(), Empirical, true),
        spec(F::KccSite, "rank_L", unit(), Empirical, true),
        spec(F::KccSite, "rank_T", unit(), Empirical, true),
        spec(F::KccSite, "rank_score", unit(), Empirical, true),
        // nested kcc.* — AGGREGATE causal/temporal site summaries (no per-frame chronology in this
        // artifact; per-frame time series lives in trajectory+spike_events — deferred, NOT faked).
        spec(
            F::KccSite,
            "active_causal_steps",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // corpus-fit
        spec(
            F::KccSite,
            "best_kcc_candidate_index",
            enom(8),
            Fixed(3),
            true,
        ), // engine ORIGINAL-array index
        // best_candidate_residue_id (DERIVED) = candidate_residue_ids[best_kcc_candidate_index]: an
        // order-INDEPENDENT stable pointer (== driver_residue_id) that survives residue_id-asc
        // member emission, where the raw index would go stale.
        spec(
            F::KccSite,
            "best_candidate_residue_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        // burst_motion: documented domain [0,5] (vector burst_motion/5); decision-critical at the
        // persistent≈1.0 / transient>1.5 thresholds (interpretation_guidelines, INV-8).
        spec_b(
            F::KccSite,
            "burst_motion",
            lin(0.0, 5.0),
            Empirical,
            true,
            DB::AbsoluteLinear(0.1),
            DS::Threshold {
                points: BURST_PTS,
                region: (0.0, 5.0),
            },
        ),
        spec(
            F::KccSite,
            "direction_score",
            logm(1e-6, 1e0),
            Empirical,
            true,
        ), // [0,1] ratio, tiny regime → logm+fit
        spec(F::KccSite, "driver_residue_id", G::JoinKey, Fixed(0), true), // primary member → residues[]
        spec(F::KccSite, "kcc_confidence", unit(), Empirical, true),
        spec(F::KccSite, "lag_corr_peak", lin(-1.0, 1.0), Empirical, true), // peak cross-correlation
        spec(F::KccSite, "local_cov", unit(), Empirical, true), // BOUNDED covariance [0,1] — NOT 12-decade log
        spec(
            F::KccSite,
            "motion_efficiency",
            logm(1e-4, 1e0),
            Empirical,
            true,
        ), // [0,1] saturating, tiny → logm+fit
        spec_b(
            F::KccSite,
            "site_burst_motion",
            lin(0.0, 5.0),
            Empirical,
            true,
            DB::AbsoluteLinear(0.1),
            DS::Threshold {
                points: BURST_PTS,
                region: (0.0, 5.0),
            },
        ),
        // signed causal lag: zero crossing = lead/lag decision boundary (INV-8); decision-critical.
        // ABSOLUTE bound (not relative): fine_step = 2·0.5 = 1.0 step tiles [-64,64] with a bin edge
        // AT 0, so a near-zero ±ε lag lands in opposite-sign bins WITHOUT escaping — the sign
        // survives in the TOKEN, not only the sidecar (a relative bound would escape near zero).
        spec_b(
            F::KccSite,
            "site_causal_lag",
            lin(-64.0, 64.0),
            Empirical,
            true,
            DB::AbsoluteLinear(0.5),
            DS::Threshold {
                points: LAG_PTS,
                region: (-64.0, 64.0),
            },
        ),
        spec(
            F::KccSite,
            "site_direction_score",
            logm(1e-6, 1e0),
            Empirical,
            true,
        ),
        spec(
            F::KccSite,
            "site_lag_corr_peak",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(F::KccSite, "site_local_cov", unit(), Empirical, true), // bounded [0,1]
        spec(
            F::KccSite,
            "site_motion_efficiency",
            logm(1e-4, 1e0),
            Empirical,
            true,
        ),
        spec(F::KccSite, "temporal_corr", lin(-1.0, 1.0), Empirical, true), // signed corr; low variance in-fixture
        spec(F::KccSite, "total_steps", logm(1.0, 1e7), Empirical, true),   // corpus-fit
        // per-candidate-member parallel arrays — one token per member, emitted residue_id-asc.
        spec(
            F::KccSite,
            "candidate_residue_ids",
            G::JoinKey,
            Fixed(0),
            true,
        ), // member id → residues[]
        // member ENRICHMENT joined from residues[] (residue_name + Cα geometry), so a member token is
        // a real (id, name, position) tuple, not a bare id.
        spec(
            F::KccSite,
            "candidate_residue_name",
            enom(AA_NAMES.len() as u32),
            Fixed(5),
            true,
        ),
        spec(
            F::KccSite,
            "candidate_ca_position",
            G::LatticeVq3d {
                codebook_bits: 12,
                lattice: Lattice::LearnedVqEma,
            },
            Empirical,
            true,
        ),
        spec(
            F::KccSite,
            "candidate_causal_weights",
            unit(),
            Empirical,
            true,
        ),
        spec_b(
            F::KccSite,
            "candidate_kcc_burst_motion",
            lin(0.0, 5.0),
            Empirical,
            true,
            DB::AbsoluteLinear(0.1),
            DS::Threshold {
                points: BURST_PTS,
                region: (0.0, 5.0),
            },
        ),
        spec_b(
            F::KccSite,
            "candidate_kcc_causal_lag",
            lin(-64.0, 64.0),
            Empirical,
            true,
            DB::AbsoluteLinear(0.5),
            DS::Threshold {
                points: LAG_PTS,
                region: (-64.0, 64.0),
            },
        ),
        spec(
            F::KccSite,
            "candidate_kcc_confidence",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::KccSite,
            "candidate_kcc_direction_score",
            logm(1e-6, 1e0),
            Empirical,
            true,
        ),
        spec(
            F::KccSite,
            "candidate_kcc_local_cov",
            unit(),
            Empirical,
            true,
        ), // bounded [0,1]
        spec(
            F::KccSite,
            "candidate_residue_support",
            unit(),
            Empirical,
            true,
        ),
        // AUTHORITATIVE per-residue TWIN/coherence evidence, TYPED-JOINED from sibling
        // prism_therm.json by residue_id (74/93 member coverage; 19 genuinely absent — see
        // KCC_SITE_JOIN_COVERAGE_LEDGER.json). This is the explicit cross-link the prior pass left
        // "unjoined". Pairwise residue_i↔residue_j te_asymmetry MATRIX still needs a canonical
        // --coupled-twin run (BLOCKED: no Kv31 topology/trajectory in-repo).
        spec(
            F::KccSite,
            "candidate_transfer_entropy",
            lin(0.0, 0.1),
            Empirical,
            true,
        ), // info ≥0, [0,0.036]
        spec(
            F::KccSite,
            "candidate_fisher_info",
            lin(0.0, 0.05),
            Empirical,
            true,
        ), // [0,0.0224]
        spec(
            F::KccSite,
            "candidate_causal_dg",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ), // signed ΔG kcal/mol
        spec(
            F::KccSite,
            "candidate_kl_divergence",
            unit(),
            Empirical,
            true,
        ), // KL ≥0 [0,0.31]
        spec(F::KccSite, "candidate_te_role", enom(8), Fixed(3), true), // GATEWAY/SPECTAT/STABIL./TRIGGER
        spec(
            F::KccSite,
            "candidate_n_causal_spikes",
            logm(1.0, 1e6),
            Empirical,
            true,
        ), // count, 0 escapes
        // binding_sites.json typed join (by site `id`, 65/65 exact; gap-closure). Site presentation
        // evidence the engine marks out_of_scope_current_teacher_training but IS tokenizable.
        spec(F::KccSite, "site_classification", enom(8), Fixed(3), true), // ActiveSite/Cryptic/Unknown
        spec(F::KccSite, "site_druggability", unit(), Empirical, true),   // [0, 0.89]
        spec(F::KccSite, "site_cryptic_score", unit(), Empirical, true),  // [0, 0.13]
        // --- A19 SPIKE_TRANSITION_EVENT (engine writer nhs_rt_full.rs ~7616; full 23-field schema).
        //     Distinct family from BocpdEvent (bocpd.jsonl per-frame log is a DIFFERENT artifact).
        //     Typed evidence fields below; the string ids / status / hash / nested-object fields
        //     (condition_id, protocol_name, *_status, *_sha256/hash, threshold_before_after,
        //     bridge_groups_observed) route LOSSLESS via the dedicated router (push_opaque, §10). ---
        spec(
            F::SpikeTransitionEvent,
            "stream_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpikeTransitionEvent,
            "group_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpikeTransitionEvent,
            "step",
            logm(1.0, 1e9),
            Empirical,
            true,
        ), // timestep, 0 escapes
        spec(
            F::SpikeTransitionEvent,
            "chunk_idx",
            logm(1.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::SpikeTransitionEvent,
            "bocpd_probability",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::SpikeTransitionEvent,
            "spike_count",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::SpikeTransitionEvent,
            "trigger_kind",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::SpikeTransitionEvent,
            "phase_from",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::SpikeTransitionEvent,
            "phase_to",
            enom(16),
            Fixed(4),
            true,
        ),
        // typed enums now grounded in REAL run vocab (v4→v5): condition_id = TWIN16 cell (16),
        // protocol_name (4 protocols), spike_count_status (5 device-sync states).
        spec(
            F::SpikeTransitionEvent,
            "condition_id",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::SpikeTransitionEvent,
            "protocol_name",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::SpikeTransitionEvent,
            "spike_count_status",
            enom(8),
            Fixed(3),
            true,
        ),
        // --- A20 BOCPD_LOG (real bocpd.jsonl per-frame changepoint log; gap-closure). 12 fields;
        //     DISTINCT from SpikeTransitionEvent + BocpdEvent. Ranges measured from the all-on run. -
        spec(F::BocpdLog, "frame_idx", logm(1.0, 1e9), Empirical, true),
        spec(F::BocpdLog, "chunk_idx", logm(1.0, 1e6), Empirical, true),
        spec(F::BocpdLog, "stream", G::JoinKey, Fixed(0), true),
        spec(F::BocpdLog, "spike_delta", logm(1.0, 1e9), Empirical, true),
        spec(F::BocpdLog, "observation", logm(1e-3, 1e3), Empirical, true),
        spec(F::BocpdLog, "posterior_max", unit(), Empirical, true),
        spec(
            F::BocpdLog,
            "map_run_length",
            logm(1.0, 1e4),
            Empirical,
            true,
        ),
        spec(F::BocpdLog, "recent_p", unit(), Empirical, true),
        spec(
            F::BocpdLog,
            "recent_log_odds_bits",
            lin(-64.0, 64.0),
            Empirical,
            true,
        ), // signed log-odds
        spec(F::BocpdLog, "reset_probability", unit(), Empirical, true),
        spec(
            F::BocpdLog,
            "chunk_close_signal",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::BocpdLog,
            "n_observations",
            logm(1.0, 1e4),
            Empirical,
            true,
        ),
        // --- A21 PROTOCOL_STATE (real protocol_state.json control state; gap-closure). Key typed
        //     control fields; remaining fields preserved LOSSLESS via capture. ---------------------
        spec(
            F::ProtocolState,
            "current_step",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::ProtocolState,
            "current_phase_bits",
            logm(1.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::ProtocolState,
            "current_temperature_K",
            lin(0.0, 1000.0),
            Empirical,
            true,
        ),
        spec(F::ProtocolState, "dt_ps", logm(1e-6, 1e0), Empirical, true),
        spec(F::ProtocolState, "gamma", lin(0.0, 100.0), Empirical, true),
        spec(
            F::ProtocolState,
            "effective_gamma",
            lin(0.0, 100.0),
            Empirical,
            true,
        ),
        spec(
            F::ProtocolState,
            "coupling_phase",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(
            F::ProtocolState,
            "processed_spike_count",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::ProtocolState,
            "n_wavelengths",
            lin(0.0, 64.0),
            Empirical,
            true,
        ),
        spec(
            F::ProtocolState,
            "fused_step_counter",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        // --- A22 NOISE_FLOOR (real noise_floor.json per-stream QC; gap-closure). Typed scalars;
        //     mu/sigma/samples arrays preserved LOSSLESS via capture. -----------------------------
        spec(F::NoiseFloor, "n_samples", logm(1.0, 1e9), Empirical, true),
        spec(F::NoiseFloor, "phase", enom(16), Fixed(4), true),
        // --- A23 TWIN16 TRANSITION-VIEW RESIDUALS (real TWIN16_INTERFEROMETRIC...json; engine
        //     nhs_rt_full.rs:2266/7403). 16 per-cell bounded thermo/temporal aggregates (/cells) +
        //     cell-pair L1 residuals (/pairwise_residuals). NOT a residue-pair TE matrix. ----------
        // per-cell aggregate fields:
        spec(
            F::TwinTransitionResidual,
            "cell_index",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "condition_id",
            enom(16),
            Fixed(4),
            true,
        ), // TWIN16 cell
        spec(
            F::TwinTransitionResidual,
            "protocol_name",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "stream_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "group_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "event_count",
            logm(1.0, 1e7),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "bocpd_count",
            logm(1.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "bocpd_probability_mean",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "bridge_event_count",
            logm(1.0, 1e7),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "bridge_group_unique_count",
            logm(1.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "spike_count_total",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "spike_count_mean",
            logm(1.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "max_step",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "min_step",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "max_chunk_idx",
            logm(1.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "min_chunk_idx",
            logm(1.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "final_teardown_seen",
            G::Boolean,
            Fixed(1),
            true,
        ),
        // cell-pair interferometric residual fields (/pairwise_residuals):
        spec(
            F::TwinTransitionResidual,
            "cell_a",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "cell_b",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "stream_a",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "stream_b",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "group_a",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "group_b",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "same_bridge_group",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::TwinTransitionResidual,
            "transition_view_l1_residual",
            lin(0.0, 1e3),
            Empirical,
            true,
        ),
        // --- A24 TWIN16 CONDITION EDGE (Directive v2 A5: per_cellpair_condition_edges.jsonl) ----
        //     The DirectedPerturbationDifferential edges over the 2×2×2×2 thermodynamic factorial.
        spec(F::TwinConditionEdge, "cell_a", enom(16), Fixed(4), true),
        spec(F::TwinConditionEdge, "cell_b", enom(16), Fixed(4), true),
        spec(F::TwinConditionEdge, "axis", enom(8), Fixed(3), true), // thermal/uv/ramp/return
        spec(
            F::TwinConditionEdge,
            "hamming",
            logm(1.0, 4.0),
            Fixed(2),
            true,
        ),
        spec(
            F::TwinConditionEdge,
            "transition_view_l1_residual",
            lin(0.0, 1e3),
            Empirical,
            true,
        ),
        spec(F::TwinConditionEdge, "material", G::Boolean, Fixed(1), true),
        // --- A25 NMA COUPLED EDGE (auxiliary coupled-A/B NMA differential, SEPARATE from TWIN16) -
        //     per_residue_condition_edges.jsonl from the --coupled-twin path (thermal_only vs +NMA).
        spec(F::NmaCoupledEdge, "resid", G::JoinKey, Fixed(0), true),
        spec(
            F::NmaCoupledEdge,
            "resname",
            enom(AA_NAMES.len() as u32),
            Fixed(5),
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "spikes_a",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "spikes_b",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "b_over_a_ratio",
            lin(0.0, 2.0),
            Empirical,
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "b_over_a_intensity_ratio",
            lin(0.0, 2.0),
            Empirical,
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "nma_exclusive_count",
            logm(1.0, 1e7),
            Empirical,
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "thermal_exclusive_count",
            logm(1.0, 1e7),
            Empirical,
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "warm_hold_differential",
            lin(0.0, 10.0),
            Empirical,
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "mechanical_sensitivity",
            logm(1e-3, 1e3),
            Empirical,
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "barrier_classification",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "edge_score",
            logm(1e-3, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::NmaCoupledEdge,
            "condition_exclusive",
            G::Boolean,
            Fixed(1),
            true,
        ),
        // --- A26 DSTW_STATE_VERTEX (real dstw_state_graph.rs DSTW_T4_VERTEX_SCHEMA_KIND; one
        //     lifecycle state of an MD evidence run). FLAT projection of the vertex JSON: nested
        //     ligand_state / detection_mode.posture scalars are lifted to top-level keys by the
        //     dstw_state_graph source-kind so route_object matches them. enum vocab + ranges are
        //     GROUNDED in the emitter (no guessing): lifecycle_stage/ligand_state/feasibility/
        //     runtime_origin are exactly the strings build_vertex / RuntimeOrigin::as_str emit.
        //     canonical_safe=true: lifecycle provenance/control evidence, NOT a generative artifact.
        spec(
            F::DstwStateVertex,
            "lifecycle_stage",
            enom(8),
            Fixed(3),
            true,
        ), // pre_md/post_md/post_*
        spec(F::DstwStateVertex, "vertex_id", G::JoinKey, Fixed(0), true), // blake3 vtx_ id (string→escape)
        spec(F::DstwStateVertex, "target_id", G::JoinKey, Fixed(0), true),
        spec(F::DstwStateVertex, "ligand_state", enom(4), Fixed(2), true), // apo / holo_gdp
        spec(F::DstwStateVertex, "feasibility", enom(4), Fixed(2), true), // POPULATABLE_NOW / BLOCKED
        spec(
            F::DstwStateVertex,
            "gdp_present",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::DstwStateVertex,
            "input_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ), // VALUE-verified topo sha
        spec(F::DstwStateVertex, "blocker", G::JoinKey, Fixed(0), true), // typed blocker string (BLOCKED only)
        spec(
            F::DstwStateVertex,
            "detection_mode",
            G::JoinKey,
            Fixed(0),
            true,
        ), // free-form posture string
        spec(
            F::DstwStateVertex,
            "site_ranker",
            G::JoinKey,
            Fixed(0),
            true,
        ), // e.g. phase-manifold
        spec(
            F::DstwStateVertex,
            "md_only_evidence",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::DstwStateVertex,
            "stage_family",
            G::JoinKey,
            Fixed(0),
            true,
        ), // EXTERNAL_DERIVED downstream
        spec(
            F::DstwStateVertex,
            "runtime_origin",
            enom(8),
            Fixed(3),
            true,
        ), // MANDATORY honest origin
        spec(
            F::DstwStateVertex,
            "runtime_native",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::DstwStateVertex,
            "missing_precondition_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(
            F::DstwStateVertex,
            "qnanl_receipt_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(
            F::DstwStateVertex,
            "populatable_tensor_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(
            F::DstwStateVertex,
            "blocked_tensor_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        // --- A27 DSTW_STATE_TRANSITION (DSTW_T4_TRANSITION_SCHEMA_KIND; one operator applied between
        //     two vertices). operator_name vocab = the literal operators build_transition emits. The
        //     opcode_total is the SUM of opcode_operand_refs.opcode_counts (real engine opcode COUNTS,
        //     never null-padded operands). canonical_safe=true (lifecycle provenance). -----------------
        spec(
            F::DstwStateTransition,
            "transition_id",
            G::JoinKey,
            Fixed(0),
            true,
        ), // blake3 txn_ id
        spec(
            F::DstwStateTransition,
            "source_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::DstwStateTransition,
            "target_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::DstwStateTransition,
            "operator_name",
            enom(16),
            Fixed(4),
            true,
        ), // MD.INTEGRATE_../STATE.CLASSIFY_BLOCKED/..
        spec(
            F::DstwStateTransition,
            "operator_version",
            G::JoinKey,
            Fixed(0),
            true,
        ), // engine blake3
        spec(
            F::DstwStateTransition,
            "runtime_origin",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::DstwStateTransition,
            "next_recommended_operator",
            G::JoinKey,
            Fixed(0),
            true,
        ), // free-form string
        spec(
            F::DstwStateTransition,
            "validation_feasibility",
            enom(4),
            Fixed(2),
            true,
        ), // POPULATABLE_NOW
        spec(
            F::DstwStateTransition,
            "is_opcode_trace_grounded",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::DstwStateTransition,
            "changed_tensor_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(
            F::DstwStateTransition,
            "unchanged_tensor_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(
            F::DstwStateTransition,
            "failed_tensor_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(
            F::DstwStateTransition,
            "blocked_precondition_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(
            F::DstwStateTransition,
            "opcode_total",
            logm(1.0, 1e12),
            Empirical,
            true,
        ), // sum of opcode COUNTS
        // --- A28 DSTW_NEXT_OPERATOR (DSTW_T4_NEXT_OPERATOR_SCHEMA_KIND; the rank-ordered engine-owner
        //     work list, runtime_origin RUNTIME_INLINE). One row per recommendation file; the full
        //     priority list rides the lossless sidecar (capture). canonical_safe=true (advisory but
        //     not generative). ---------------------------------------------------------------------
        spec(
            F::DstwNextOperator,
            "runtime_origin",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::DstwNextOperator,
            "runtime_native",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::DstwNextOperator,
            "recommendation_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(F::DstwNextOperator, "top_rank", enom(16), Fixed(4), true), // rank-1 entry rank (identity)
        spec(
            F::DstwNextOperator,
            "top_operator",
            G::JoinKey,
            Fixed(0),
            true,
        ), // e.g. ENGINE.HOLO_PHYSICS_INGEST
        spec(F::DstwNextOperator, "top_owner", enom(4), Fixed(2), true), // engine / prismos
        // --- A29 DENSE_TENSOR_SEMANTIC_SUMMARY (DSTW_T4_DENSE_TENSOR_SUMMARY_SCHEMA_KIND; one
        //     POST_RUN_DERIVED semantic record per dense engine tensor so the raw bytes never enter
        //     training). enum vocab is GROUNDED in the prismcode emitter (no guessing): tensor_role/
        //     modality are exactly the strings build_dense_tensor_semantic_summaries emits; runtime_origin
        //     is exactly RuntimeOrigin::as_str (operand_ledger.rs). Identity fields (tensor_role/modality/
        //     source_kind/source_hash/tokenizer_family/parent_vertex_id/runtime_origin) are ALWAYS present
        //     ⇒ matched_fields>0 even when the decoded summary is absent. Dimension/finite fields
        //     (row_count/column_count/all_finite/nan_count/inf_count/nonfinite_or_null_count/
        //     reducer_scalar) populate ONLY from a decoded reducer summary; otherwise the emitter records
        //     them with absent_reason in the lossless sidecar (never null-padded). canonical_safe=true
        //     (evidence summary, NOT a generative artifact). ----------------------------------------------
        spec(
            F::DenseTensorSemanticSummary,
            "tensor_role",
            enom(8),
            Fixed(3),
            true,
        ), // SIGNAL_GRID/WARP_FIELD/..
        spec(
            F::DenseTensorSemanticSummary,
            "modality",
            enom(8),
            Fixed(3),
            true,
        ), // thermodynamic_occupancy_grid/..
        spec(
            F::DenseTensorSemanticSummary,
            "source_kind",
            G::JoinKey,
            Fixed(0),
            true,
        ), // canonical artifact kind
        spec(
            F::DenseTensorSemanticSummary,
            "source_hash",
            G::JoinKey,
            Fixed(0),
            true,
        ), // REAL sha256 (present only)
        spec(
            F::DenseTensorSemanticSummary,
            "tokenizer_family",
            G::JoinKey,
            Fixed(0),
            true,
        ), // downstream family name
        spec(
            F::DenseTensorSemanticSummary,
            "runtime_origin",
            enom(8),
            Fixed(3),
            true,
        ), // POST_RUN_DERIVED
        spec(
            F::DenseTensorSemanticSummary,
            "parent_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ), // post-MD vertex id
        spec(
            F::DenseTensorSemanticSummary,
            "stream_id",
            lin(0.0, 256.0),
            Empirical,
            true,
        ), // parsed _streamNN_
        spec(
            F::DenseTensorSemanticSummary,
            "on_disk_bytes",
            logm(1.0, 1e12),
            Empirical,
            true,
        ), // real metadata len
        spec(
            F::DenseTensorSemanticSummary,
            "artifact_present",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::DenseTensorSemanticSummary,
            "summary_present",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::DenseTensorSemanticSummary,
            "row_count",
            logm(1.0, 1e9),
            Empirical,
            true,
        ), // dimension (decoded summary)
        spec(
            F::DenseTensorSemanticSummary,
            "column_count",
            lin(0.0, 4096.0),
            Empirical,
            true,
        ),
        spec(
            F::DenseTensorSemanticSummary,
            "all_finite",
            G::Boolean,
            Fixed(1),
            true,
        ), // finite/NaN/Inf gate
        spec(
            F::DenseTensorSemanticSummary,
            "nan_count",
            lin(0.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::DenseTensorSemanticSummary,
            "inf_count",
            lin(0.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::DenseTensorSemanticSummary,
            "nonfinite_or_null_count",
            lin(0.0, 1e6),
            Empirical,
            true,
        ),
        spec(
            F::DenseTensorSemanticSummary,
            "reducer_scalar",
            lin(-1e6, 1e6),
            Empirical,
            true,
        ), // representative reduced scalar
        // --- A11 HOLO (manifest gap) ------------------------------------------------------
        spec(
            F::HoloBoundState,
            "holo_ground_truth",
            G::Opaque,
            Fixed(0),
            true,
        ),
        // --- A12 CAPABILITY_GATE (manifest gap) -------------------------------------------
        spec(F::CapabilityGate, "gate_verdict", enom(8), Fixed(3), true),
        // --- A37 OPCODE_EXECUTION_LEDGER (opcode_operand_ledger.json, prismcode_typed_operand_ledger)
        //     One per-executed-opcode/kernel/seal row. `count` is the VALUE field: it is tied by-value
        //     to the source `*_md_native_receipt.json` opcode_counts (MD.*) / kernel_counts (KERNEL::)
        //     — the by-value invariant. Enum vocabularies (operator_class/operand_target/runtime_origin/
        //     disassembly_status/validation_status/replay_status) are registered in routing.rs
        //     enum_labels; the nested value_proof + seal-only fields escape losslessly to the sidecar
        //     (INV-4). JSON keys == field names (json_key_for default). canonical_safe=true (evidence). --
        spec(
            F::OpcodeExecutionLedger,
            "opcode_name",
            G::JoinKey,
            Fixed(0),
            true,
        ), // MD.* ISA opcode / KERNEL:: dispatch label / MD.QNANL_SEAL
        spec(
            F::OpcodeExecutionLedger,
            "opcode_id",
            G::JoinKey,
            Fixed(0),
            true,
        ), // numeric ISA id (present on MD.* rows only)
        spec(
            F::OpcodeExecutionLedger,
            "count",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // THE value field — tied to receipt opcode_counts/kernel_counts
        spec(
            F::OpcodeExecutionLedger,
            "operand_schema_version",
            lin(0.0, 16.0),
            Fixed(1),
            true,
        ),
        spec(
            F::OpcodeExecutionLedger,
            "operator_class",
            enom(3),
            Fixed(2),
            true,
        ), // Md / Kcc / Qnanl
        spec(
            F::OpcodeExecutionLedger,
            "operand_target",
            enom(2),
            Fixed(1),
            true,
        ), // Tensor / Artifact
        spec(
            F::OpcodeExecutionLedger,
            "runtime_origin",
            enom(2),
            Fixed(1),
            true,
        ), // RuntimeSubprocessOrchestratedByPrismos / PostRunSealOnly
        spec(
            F::OpcodeExecutionLedger,
            "runtime_native",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::OpcodeExecutionLedger,
            "disassembly_status",
            enom(3),
            Fixed(2),
            true,
        ), // REGISTRY_COVERED / TYPED_VALUE_VERIFIED / NOT_APPLICABLE
        spec(
            F::OpcodeExecutionLedger,
            "validation_status",
            enom(3),
            Fixed(2),
            true,
        ),
        spec(
            F::OpcodeExecutionLedger,
            "replay_status",
            enom(3),
            Fixed(2),
            true,
        ),
        spec(
            F::OpcodeExecutionLedger,
            "blocker_reason",
            G::JoinKey,
            Fixed(0),
            true,
        ), // engine-owner blocker text (present on 336/368 rows)
        spec(
            F::OpcodeExecutionLedger,
            "source_receipt",
            G::JoinKey,
            Fixed(0),
            true,
        ), // provenance join → md_native_receipt / qnanl seal receipt
        spec(
            F::OpcodeExecutionLedger,
            "input_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ), // seal rows only
        spec(
            F::OpcodeExecutionLedger,
            "output_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ), // seal rows only
        spec(
            F::OpcodeExecutionLedger,
            "qnanl_packet_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ), // seal rows only
        // --- A38 VALIDATION_STATE (prismos_prismcode_isa_execution_verification.json, v13) ---------
        //     The AGGREGATE MD-ISA execution VERDICT over one native-run row. Top-level object routed
        //     EXPLICITLY (json_artifact_entry) — the object IS the record; the data-dense streams[]
        //     descends losslessly to the sidecar (INV-1). Field names are EXACTLY the keys emitted by
        //     build_prismos_isa_execution_verification (prismos.rs:13045-13104). BY-VALUE invariant:
        //     native_md_receipt_count == len(streams) (real 8) and all_streams_pass == AND over
        //     streams of (status=="PASS") (real true). status/substrate_status/
        //     implementation_status_language enum vocabularies are registered in routing.rs
        //     enum_labels (substrate_status widens PASS/PARTIAL/FAIL with the MISSING fallback).
        //     canonical_safe=true (verification evidence, NOT generative). --------------------------
        spec(F::ValidationState, "status", enom(3), Fixed(2), true), // PASS/PARTIAL/FAIL
        spec(
            F::ValidationState,
            "substrate_status",
            enom(4),
            Fixed(2),
            true,
        ), // PASS/PARTIAL/FAIL/MISSING (native-receipt-absent fallback)
        spec(
            F::ValidationState,
            "implementation_status_language",
            enom(2),
            Fixed(1),
            true,
        ), // IMPLEMENTATION_VERIFIED / IMPLEMENTATION_PARTIAL
        spec(
            F::ValidationState,
            "schema_version",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(
            F::ValidationState,
            "native_md_receipt_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ), // == len(streams) BY-VALUE INVARIANT
        spec(
            F::ValidationState,
            "min_hotpath_launch_collapse",
            lin(0.0, 4096.0),
            Empirical,
            true,
        ),
        spec(
            F::ValidationState,
            "all_streams_pass",
            G::Boolean,
            Fixed(1),
            true,
        ), // == AND(streams[i].status=="PASS") BY-VALUE INVARIANT
        spec(
            F::ValidationState,
            "all_streams_bounded_or_passed",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ValidationState,
            "cuda_opcode_coverage_exists",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ValidationState,
            "hotpath_acceleration_required",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ValidationState,
            "metal_resident_md_required",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ValidationState,
            "plan_hash_blake3",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        // --- A39 CAPTURED_GRAPH_GPU_TELEMETRY (ghost_time_map.json streams[] + ------------------
        //     ghost_lattice_routing_status.json flat object, v13) --------------------------------
        //     route_object matches whichever keys are present per record: ghost_time_map streams[]
        //     records match the temporal block (9 fields), the ghost_lattice_routing_status flat
        //     object matches the routing/status block (8 fields). Field names are EXACTLY the real
        //     JSON keys. No enom() fields => no enum_labels registration needed. All canonical_safe
        //     =true (runtime GPU evidence, NOT generative). BY-VALUE invariant (ghost_time_map,
        //     exact): physical_time_fs == timestep_max*dt_fs AND physical_time_at_min_fs ==
        //     timestep_min*dt_fs for every stream; len(streams)==8; stream_id distinct set {0..7}.
        //     DISTINCT from GpuTelemetry (slot 8, GFlowNet receipt scalars) — its OWN namespace.
        //     ghost_time_map streams[] per-stream captured-graph temporal frame:
        spec(
            F::CapturedGraphGpuTelemetry,
            "stream_id",
            lin(0.0, 256.0),
            Empirical,
            true,
        ), // 0..7 lane anchor
        spec(
            F::CapturedGraphGpuTelemetry,
            "gear_id",
            lin(0.0, 64.0),
            Empirical,
            true,
        ), // gearbox id (wave-a default 0)
        spec(
            F::CapturedGraphGpuTelemetry,
            "dt_fs",
            lin(0.0, 16.0),
            Empirical,
            true,
        ), // 2.0/4.0 fs HMR timestep
        spec(
            F::CapturedGraphGpuTelemetry,
            "timestep_min",
            logm(1.0, 1e9),
            Empirical,
            true,
        ), // first captured step
        spec(
            F::CapturedGraphGpuTelemetry,
            "timestep_max",
            logm(1.0, 1e9),
            Empirical,
            true,
        ), // last captured step
        spec(
            F::CapturedGraphGpuTelemetry,
            "frame_idx_at_teardown",
            logm(1.0, 1e9),
            Empirical,
            true,
        ),
        spec(
            F::CapturedGraphGpuTelemetry,
            "n_spikes",
            logm(1.0, 1e12),
            Empirical,
            true,
        ), // per-stream spike count
        spec(
            F::CapturedGraphGpuTelemetry,
            "physical_time_fs",
            logm(1e-1, 1e12),
            Empirical,
            true,
        ),
        spec(
            F::CapturedGraphGpuTelemetry,
            "physical_time_at_min_fs",
            logm(1e-1, 1e12),
            Empirical,
            true,
        ),
        // ghost_lattice_routing_status flat object captured-graph/lattice routing status:
        spec(
            F::CapturedGraphGpuTelemetry,
            "n_components",
            lin(0.0, 4096.0),
            Empirical,
            true,
        ), // lattice components
        spec(
            F::CapturedGraphGpuTelemetry,
            "lattice_kernel_ms",
            lin(0.0, 1e6),
            Empirical,
            true,
        ), // lin (0.0 present) NOT logm
        spec(
            F::CapturedGraphGpuTelemetry,
            "v2_was_live",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::CapturedGraphGpuTelemetry,
            "v2_capture_requested",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::CapturedGraphGpuTelemetry,
            "auto_request",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::CapturedGraphGpuTelemetry,
            "explicit_request",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::CapturedGraphGpuTelemetry,
            "m1_monolithic_discovery",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::CapturedGraphGpuTelemetry,
            "mar_v2_telemetry",
            G::Boolean,
            Fixed(1),
            true,
        ),
        // (schema_version/run_id/target/notes/dt_source/gear_id_status/source/backend_request/
        //  lattice_run_status are string/meta -> preserved LOSSLESS in the §10 sidecar, not quantized.)
        // --- PATH-B FLAGSHIP materialized binding site (binding_sites.materialized.json) ---------
        // Per-site record in the wrapper's binding_sites[] array (schema_kind
        // pathb_binding_sites_materialized). 13 by-value scalar specs => matched_fields>0 per site.
        // canonical_safe=TRUE (MD-evidence lexicographic materialization descriptor, NOT a generative
        // firewall artifact). enom() fields are backed by grounded vocab in routing.rs enum_labels.
        spec(
            F::PathBBindingSiteMaterialized,
            "site_id",
            G::JoinKey,
            Fixed(0),
            true,
        ), // 'site_000'..'site_0NN'
        spec(
            F::PathBBindingSiteMaterialized,
            "rank",
            lin(0.0, 512.0),
            Empirical,
            true,
        ), // int 0..N-1 monotonic (INV-B)
        spec(
            F::PathBBindingSiteMaterialized,
            "new_prism_rank",
            lin(0.0, 512.0),
            Empirical,
            true,
        ), // permutation 0..N-1 (INV-D)
        spec(
            F::PathBBindingSiteMaterialized,
            "old_density_rank",
            lin(0.0, 512.0),
            Empirical,
            true,
        ), // distinct in [0,n_consolidated_regions) (INV-D corrected)
        spec(
            F::PathBBindingSiteMaterialized,
            "n_spikes",
            logm(1.0, 1e9),
            Empirical,
            true,
        ), // 470549 (decade-spanning)
        spec(
            F::PathBBindingSiteMaterialized,
            "n_raw_peaks_merged",
            lin(0.0, 256.0),
            Empirical,
            true,
        ), // 3
        spec(
            F::PathBBindingSiteMaterialized,
            "n_active_non_density_dimensions",
            lin(0.0, 16.0),
            Empirical,
            true,
        ), // 4
        spec(
            F::PathBBindingSiteMaterialized,
            "intensity_sum",
            logm(1.0, 1e12),
            Empirical,
            true,
        ), // 2.63e7
        spec(
            F::PathBBindingSiteMaterialized,
            "centroid_spread_a",
            lin(0.0, 50.0),
            Empirical,
            true,
        ), // 2.99 A
        spec(
            F::PathBBindingSiteMaterialized,
            "region_radius_a",
            lin(0.0, 50.0),
            Empirical,
            true,
        ), // 2.99 A
        spec(
            F::PathBBindingSiteMaterialized,
            "promoted",
            G::Boolean,
            Fixed(1),
            true,
        ), // true
        spec(
            F::PathBBindingSiteMaterialized,
            "materialization_status",
            enom(8),
            Fixed(3),
            true,
        ), // 'materialized'
        spec(
            F::PathBBindingSiteMaterialized,
            "materialization_level",
            enom(8),
            Fixed(3),
            true,
        ), // 'materialized_dynamic_site'
        // (lining_residues/driver_residues/representative_residues/centroid_xyz + nested dicts
        //  score_components/lexicographic_rank_key/provenance/... escape LOSSLESS to the §10 sidecar;
        //  the residue arrays are VALUE-cross-checked by the family by-value INVARIANT, not scalar-routed.)
        // --- A30 LINEAGE_DAG (merge_manifest.rs emit_state_graph_jsonl; PRISM_NATIVE_STATE_GRAPH.jsonl) ---
        //     Heterogeneous envelope stream: 8 `state_graph_node` records (one per emitted artifact
        //     vertex/edge) + 1 final `orphan_check` summary. Routed by-value as jsonl_records (one
        //     record/line). All enom vocab is GROUNDED in the emitter (no guessing): record_type/
        //     verdict literals from emit_state_graph_jsonl; state_object_class from STATE_OBJECT_CLASSES
        //     (11) + ORPHAN_NOT_PRISM_NATIVE; runtime_origin from RuntimeOrigin::as_str; operator_name
        //     from the transition builder; schema_kind/artifact_kind from dstw_state_graph.rs. EVERY
        //     enom field has an enum_labels table registered in routing.rs (else the string escapes
        //     losslessly but is not tokenized — the #1 review defect, fixed here). schema_version is
        //     a numeric int(=1) so it takes lin (not enom) — the reviewer's cleaner geometry. Identity
        //     JoinKeys (vertex_id/parent/source/target/transition_id/artifact_sha256) are the DAG
        //     topology carriers cross-checked by the by-value invariant. High-entropy machine paths
        //     (json_path/qnanl_packet_path/emission_unix_ns/orphan_note/tokenizer_family_absent_reason)
        //     are DELIBERATELY UNSPECCED and ride the lossless §10 sidecar (INV-1). BY-VALUE invariant:
        //     every non-null source/target/parent_vertex_id resolves to an existing vertex_id, and
        //     orphan_check.total_artifacts == count(state_graph_node). canonical_safe=true (provenance
        //     container, NOT generative). ----------------------------------------------------------------
        spec(F::LineageDag, "record_type", enom(4), Fixed(2), true), // state_graph_node | orphan_check
        spec(
            F::LineageDag,
            "schema_version",
            lin(0.0, 2.0),
            Fixed(1),
            true,
        ), // MERGE_MANIFEST_SCHEMA_VERSION (=1); numeric int ⇒ lin
        spec(F::LineageDag, "artifact_kind", enom(16), Fixed(4), true), // pre/post_md_state_vertex, md_state_transition, ..
        spec(F::LineageDag, "schema_kind", enom(16), Fixed(4), true), // dstw_t4_scientific_state_*
        spec(
            F::LineageDag,
            "state_object_class",
            enom(16),
            Fixed(4),
            true,
        ), // STATE_OBJECT_CLASSES(11) + ORPHAN_NOT_PRISM_NATIVE
        spec(F::LineageDag, "runtime_origin", enom(8), Fixed(3), true), // RuntimeOrigin::as_str (7)
        spec(F::LineageDag, "runtime_native", G::Boolean, Fixed(1), true),
        spec(
            F::LineageDag,
            "decode_proof_pass",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(F::LineageDag, "is_orphan", G::Boolean, Fixed(1), true),
        spec(F::LineageDag, "vertex_id", G::JoinKey, Fixed(0), true), // blake3 vtx_ id (nullable ⇒ string→escape)
        spec(
            F::LineageDag,
            "parent_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LineageDag,
            "source_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LineageDag,
            "target_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(F::LineageDag, "transition_id", G::JoinKey, Fixed(0), true), // blake3 txn_ id (nullable)
        spec(F::LineageDag, "operator_name", enom(16), Fixed(4), true), // MD.INTEGRATE_VELOCITY_VERLET | PATHB.MATERIALIZE_BINDING_SITE | null
        spec(F::LineageDag, "tokenizer_family", enom(16), Fixed(4), true), // DstwStateVertex/.. (nullable)
        spec(F::LineageDag, "artifact_sha256", G::JoinKey, Fixed(0), true), // VALUE-verified per-artifact sha256
        spec(
            F::LineageDag,
            "run_pid",
            lin(0.0, 4194304.0),
            Empirical,
            true,
        ),
        spec(
            F::LineageDag,
            "total_artifacts",
            lin(0.0, 4096.0),
            Empirical,
            true,
        ), // orphan_check: == n state_graph_node
        spec(
            F::LineageDag,
            "orphan_count",
            lin(0.0, 4096.0),
            Empirical,
            true,
        ), // orphan_check: == count(is_orphan)
        spec(F::LineageDag, "verdict", enom(4), Fixed(2), true), // ALL_ARTIFACTS_PRISM_NATIVE | ORPHANS_PRESENT_NOT_PRISM_NATIVE
        // --- A42 THERMO PHASE MANIFOLD (bounded_thermo_temporal_ccns_sdst_summary.json; ROUTE-MISSING closure) ---
        // Field names == the FLATTENED keys produced by flatten_thermo_phase_manifold() in
        // tok_export_full_prism_corpus.rs (top-level flat scalars pass through verbatim; the nested
        // ccns/sdst/phase_bucket/trigger rollups are lifted to the ccns_*/sdst_*/phase_bucket_*/
        // trigger_* keys below). Analytic logm/lin geometries encode from static params (no empirical
        // fit of the derived keys required); enom(status) has its grounded vocab in routing.rs.
        spec(
            F::ThermoPhaseManifold,
            "event_count",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // ==376: the by-value invariant anchor
        spec(
            F::ThermoPhaseManifold,
            "schema_version",
            logm(1.0, 1e3),
            Fixed(0),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "n_cells",
            logm(1.0, 1e3),
            Fixed(0),
            true,
        ), // derived == cells[].len == 4
        spec(
            F::ThermoPhaseManifold,
            "n_event_sources",
            logm(1.0, 1e3),
            Fixed(0),
            true,
        ), // derived == event_sources[].len == 8
        spec(
            F::ThermoPhaseManifold,
            "ccns_cold_hold_fraction",
            unit(),
            Fixed(6),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "ccns_ramp_fraction",
            unit(),
            Fixed(6),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "ccns_warm_hold_fraction",
            unit(),
            Fixed(6),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "ccns_cooling_fraction",
            unit(),
            Fixed(6),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "sdst_cold_fraction",
            unit(),
            Fixed(6),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "sdst_hot_fraction",
            unit(),
            Fixed(6),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "sdst_delta_hot_minus_cold",
            lin(-1.0, 1.0),
            Fixed(6),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "phase_bucket_cold_hold",
            logm(1.0, 1e6),
            Fixed(0),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "phase_bucket_ramp",
            logm(1.0, 1e6),
            Fixed(0),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "trigger_bocpd_changepoint",
            logm(1.0, 1e6),
            Fixed(0),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "trigger_ring_buffer_push",
            logm(1.0, 1e6),
            Fixed(0),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "trigger_ring_buffer_read_and_adapt",
            logm(1.0, 1e6),
            Fixed(0),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "trigger_cold_hold_start",
            logm(1.0, 1e6),
            Fixed(0),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "trigger_cold_hold_to_ramp",
            logm(1.0, 1e6),
            Fixed(0),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "trigger_final_teardown",
            logm(1.0, 1e6),
            Fixed(0),
            true,
        ),
        spec(F::ThermoPhaseManifold, "status", enom(2), Fixed(1), true), // EMITTED | NO_SPIKE_TRANSITION_EVENTS
        spec(F::ThermoPhaseManifold, "target", G::JoinKey, Fixed(0), true), // "4LPK_apo" (string id -> INV-7 escape)
        spec(F::ThermoPhaseManifold, "run_id", G::JoinKey, Fixed(0), true),
        spec(
            F::ThermoPhaseManifold,
            "source_surface",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ThermoPhaseManifold,
            "site_materialized_prism_therm_status",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        // --- A42 ISA_REGISTRY_SUMMARY (prismos_prismcode_isa_registry_summary.json records[], v13) --
        //     POST_RUN_DERIVED registry×receipt coverage matrix — one per-(stream,opcode) observation.
        //     Field names EXACTLY match the emitter build_prismos_isa_registry_summary record keys.
        //     `opcode` is a genuine categorical token (enum_labels ISA_MD_OPCODES — the 25 real
        //     md_prismop_known_instruction_set strings), NOT a filename/family label. `count` is the
        //     by-value observation (tied to the verification's known_md_prismop_observation[].count).
        //     stream_status/stream_coverage_status use the PASS/PARTIAL/FAIL STATUS vocab. BY-VALUE
        //     invariants: per stream distinct(opcode)==count(records)==registry_size==25; every
        //     required_core opcode has observed==true; distinct(stream_id)==native_md_receipt_count.
        //     canonical_safe=true (execution evidence, NOT a generative artifact). ----------------
        spec(
            F::IsaRegistrySummary,
            "stream_id",
            lin(0.0, 256.0),
            Empirical,
            true,
        ),
        spec(F::IsaRegistrySummary, "opcode", enom(25), Fixed(5), true), // enum_labels ISA_MD_OPCODES
        spec(
            F::IsaRegistrySummary,
            "opcode_index",
            lin(0.0, 64.0),
            Empirical,
            true,
        ),
        spec(
            F::IsaRegistrySummary,
            "count",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // BY-VALUE observation
        spec(
            F::IsaRegistrySummary,
            "observed",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(F::IsaRegistrySummary, "planned", G::Boolean, Fixed(1), true),
        spec(
            F::IsaRegistrySummary,
            "required_core",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::IsaRegistrySummary,
            "registry_size",
            lin(0.0, 64.0),
            Empirical,
            true,
        ), // == 25 (INV-A)
        spec(
            F::IsaRegistrySummary,
            "required_core_size",
            lin(0.0, 64.0),
            Empirical,
            true,
        ),
        spec(
            F::IsaRegistrySummary,
            "native_md_receipt_count",
            lin(0.0, 256.0),
            Empirical,
            true,
        ), // == #streams (INV-D)
        spec(
            F::IsaRegistrySummary,
            "stream_status",
            enom(4),
            Fixed(2),
            true,
        ), // enum_labels STATUS
        spec(
            F::IsaRegistrySummary,
            "stream_coverage_status",
            enom(4),
            Fixed(2),
            true,
        ), // enum_labels STATUS
        // --- KccDescriptorNative (RAW PRKCC001 kcc_v2full.bin per-residue descriptor; discriminant 43).
        //     16 slots = 2 synthesized JoinKeys + 14 self-describing binary columns, transposed one
        //     record PER RESIDUE (156 for 4LPK_apo). Ranges MEASURED from apo stream0+stream5 with
        //     headroom (spec grounding OPERATOR_CONTROL/PRISMOS_NATIVE_CORPUS_CLOSURE_20260701).
        //     REVIEW FIXES applied: burst_motion + active_causal have measured min==0 ⇒ lin(0,·) NOT
        //     logm (logm floor>0 would quantize a genuine 0 to a nonzero floor); net_dx/dy/dz KEPT at
        //     the existing KccDescriptor lin(-10,10) (NOT narrowed to ±1) for coupled-run headroom.
        //     DISTINCT namespace from KccDescriptor (the kcc_visualization.json residues[] JSON view). ---
        spec(
            F::KccDescriptorNative,
            "residue_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::KccDescriptorNative,
            "stream_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::KccDescriptorNative,
            "temporal_corr",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptorNative,
            "direction_score",
            unit(),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptorNative,
            "motion_efficiency",
            logm(1e-4, 1e-1),
            Empirical,
            true,
        ), // min measured 1.26e-3>0
        spec(
            F::KccDescriptorNative,
            "burst_motion",
            lin(0.0, 100.0),
            Empirical,
            true,
        ), // FIX: min==0 ⇒ lin, max measured 51.74
        spec(
            F::KccDescriptorNative,
            "phase_shift",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptorNative,
            "causal_lag",
            lin(-128.0, 128.0),
            Empirical,
            true,
        ), // NaN->null md-only sentinel
        spec(
            F::KccDescriptorNative,
            "lag_corr_peak",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ), // NaN->null md-only sentinel
        spec(F::KccDescriptorNative, "local_cov", unit(), Empirical, true),
        spec(
            F::KccDescriptorNative,
            "net_dx",
            lin(-10.0, 10.0),
            Empirical,
            true,
        ), // KEEP existing KccDescriptor range
        spec(
            F::KccDescriptorNative,
            "net_dy",
            lin(-10.0, 10.0),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptorNative,
            "net_dz",
            lin(-10.0, 10.0),
            Empirical,
            true,
        ),
        spec(
            F::KccDescriptorNative,
            "sum_m",
            logm(1e-2, 1e1),
            Empirical,
            true,
        ), // min measured 0.139>0
        spec(
            F::KccDescriptorNative,
            "residue_count",
            lin(0.0, 64.0),
            Fixed(5),
            true,
        ), // constant coherence window (=22)
        spec(
            F::KccDescriptorNative,
            "active_causal",
            lin(0.0, 64.0),
            Empirical,
            true,
        ), // FIX: min==0 ⇒ lin (0..residue_count)
        // --- SpikeSummary (RAW PRSPK001 _spikes.bin per-residue BOUNDED by-value aggregate; discriminant 44).
        //     Decoder prismos.rs decode_prspk001_spike_summary parses the FULL ~1.39M-spike PRSPK001
        //     payload (96 B GpuSpikeEvent records, the envelope-declared stride)
        //     and aggregates BY the primary nearby residue
        //     (nearby_residues[0]) into ONE record per residue (~146 for 4LPK_apo stream0). NOT a
        //     DenseTensorSemanticSummary: per-residue typed by-value surface whose COUNT-CONSERVATION
        //     invariant (Σ spike_count == payload record_count == receipt record_count == 1,394,939)
        //     proves every raw spike was decoded. Ranges MEASURED on real apo_cryptic stream0
        //     (spike_count 1..~95k, mean_amplitude ~48..52, max_amplitude 64.8, first/last_frame 6..1770).
        //     REVIEW FIX (aromatic_type): real domain is {-1,1,2,4,5,6} with -1('none', every LIF spike)
        //     the DOMINANT value ⇒ MUST be Sentinel(-1)-wrapped, never a bare enum; inner enom(7) covers
        //     the canonical chromophore codes 0=TRP 1=TYR 2=PHE 3=S-S 4=BNZ 5/6=UNK (fused_engine.rs:7003).
        //     canonical_safe=true (engine spike observable, deterministic reduction, NOT generative). ----
        spec(F::SpikeSummary, "residue_id", G::JoinKey, Fixed(0), true), // primary nearby residue 0..n_res-1; join key to KccDescriptorNative.residue_id
        spec(F::SpikeSummary, "stream_id", G::JoinKey, Fixed(0), true),
        spec(
            F::SpikeSummary,
            "spike_count",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // BY-VALUE aggregate; Σ==record_count (min 1>0)
        spec(
            F::SpikeSummary,
            "spike_rate",
            logm(1e-5, 1e5),
            Empirical,
            true,
        ), // spikes / frame-span (min 1/large>0)
        spec(
            F::SpikeSummary,
            "mean_amplitude",
            logm(1e-4, 1e4),
            Empirical,
            true,
        ), // mean spike intensity
        spec(
            F::SpikeSummary,
            "max_amplitude",
            logm(1e-4, 1e4),
            Empirical,
            true,
        ), // peak spike intensity
        spec(
            F::SpikeSummary,
            "first_frame",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // min timestep (measured min 6>0)
        spec(
            F::SpikeSummary,
            "last_frame",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // max timestep
        spec(
            F::SpikeSummary,
            "aromatic_type",
            G::Sentinel {
                value: -1.0,
                inner: Box::new(enom(7)),
            },
            Fixed(3),
            true,
        ), // dominant per-residue chromophore code; enum_labels AROMATIC_TYPE_LABELS
        // --- PredictionState (FIRST-CLASS prediction record per PROMOTED materialized binding site;
        //     schema_kind prismos_prediction_state; discriminant 45). Operator correction 2026-07-01:
        //     a PREDICTION STATE is DISTINCT from DstwNextOperator (an engine-owner work list) — this
        //     SUPERSEDES the DstwNextOperator-alias closure and routes INDEPENDENTLY. Each of the 18
        //     fields maps to a REAL by-value source (no placeholders): confidence is the REAL decoded
        //     manifold_consistency.manifold_consistency_score (∈[0,1]); uncertainty is the REAL
        //     centroid_spread_a normalized spread/(1+spread) (∈[0,1)); predicted_delta ==
        //     new_prism_rank − old_density_rank (signed, permutation vs density order); the vertex ids
        //     resolve to PRISM_NATIVE_STATE_GRAPH vertices. Identity JoinKeys (prediction_id/target_id/
        //     source_vertex_id/target_vertex_id/...) are ALWAYS present ⇒ matched_fields>0 for every
        //     record. canonical_safe=true (evidence-derived prediction, NOT a generative artifact). ----
        spec(
            F::PredictionState,
            "prediction_id",
            G::JoinKey,
            Fixed(0),
            true,
        ), // '<target>::pred_rNNN' (stable, unique per site)
        spec(F::PredictionState, "target_id", G::JoinKey, Fixed(0), true), // validation_state.target_id (4LPK_apo.topology)
        spec(
            F::PredictionState,
            "receptor_state",
            enom(2),
            Fixed(1),
            true,
        ), // enum_labels PREDICTION_RECEPTOR_STATE (apo/holo)
        spec(F::PredictionState, "ligand_state", enom(2), Fixed(1), true), // enum_labels DSTW_LIGAND_STATE (apo/holo_gdp)
        spec(
            F::PredictionState,
            "predicted_surface",
            G::JoinKey,
            Fixed(0),
            true,
        ), // centroid+lining-residue digest 'surf_<hash>'
        spec(
            F::PredictionState,
            "predicted_delta",
            lin(-512.0, 512.0),
            Empirical,
            true,
        ), // == new_prism_rank − old_density_rank (signed)
        spec(F::PredictionState, "confidence", unit(), Empirical, true), // REAL manifold_consistency_score ∈[0,1] (varies)
        spec(F::PredictionState, "uncertainty", unit(), Empirical, true), // REAL centroid_spread_a normalized ∈[0,1) (varies)
        spec(
            F::PredictionState,
            "epistemic_status",
            enom(4),
            Fixed(2),
            true,
        ), // enum_labels PREDICTION_EPISTEMIC_STATUS (materialized)
        spec(
            F::PredictionState,
            "evidence_parent_ids",
            G::JoinKey,
            Fixed(0),
            true,
        ), // [source_vtx, target_vtx] array -> §10 sidecar
        spec(
            F::PredictionState,
            "source_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ), // state-graph PATHB.MATERIALIZE_BINDING_SITE source vtx
        spec(
            F::PredictionState,
            "target_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ), // state-graph PATHB.MATERIALIZE_BINDING_SITE target vtx
        spec(
            F::PredictionState,
            "operator_recommendation_id",
            G::JoinKey,
            Fixed(0),
            true,
        ), // 'rankN::ENGINE.*' from next_operator_recommendation
        spec(
            F::PredictionState,
            "validation_status",
            enom(3),
            Fixed(2),
            true,
        ), // enum_labels PREDICTION_VALIDATION_STATUS
        spec(
            F::PredictionState,
            "falsification_status",
            enom(4),
            Fixed(2),
            true,
        ), // enum_labels PREDICTION_FALSIFICATION_STATUS
        spec(
            F::PredictionState,
            "qnanl_receipt_ref",
            G::JoinKey,
            Fixed(0),
            true,
        ), // pathb_site_records.sha256 (materialized artifact receipt)
        spec(
            F::PredictionState,
            "runtime_origin",
            enom(8),
            Fixed(3),
            true,
        ), // enum_labels DSTW_RUNTIME_ORIGIN (RUNTIME_SUBPROCESS_ORCHESTRATED_BY_PRISMOS)
        spec(
            F::PredictionState,
            "tokenizer_family",
            enom(2),
            Fixed(1),
            true,
        ), // enum_labels PREDICTION_TOKENIZER_FAMILY ('PredictionState')
        // --- A46 PHASE_MANIFOLD_EDGE_VALIDATION (GhostPhaseLattice4D {stem}_binding_sites_lattice.json;
        //     engine nhs_rt_full.rs ~16336; emitted ONLY when v2_was_live==true AND cluster() ok).
        //     ONE top-level record; flatten_phase_manifold_edge_validation lifts stats.*/config.* and
        //     derives component aggregates so route_object value-matches real EDGE COUNTS + SO(3)
        //     COHERENCE. Ranges bracket the measured 4LPK_apo v2-capture values (RED-FLAG-EXCEPTION
        //     corpus-closure lane, non-canonical); Empirical refits per-source where samples exist. ----
        spec(
            F::PhaseManifoldEdgeValidation,
            "schema_version",
            lin(0.0, 4.0),
            Fixed(1),
            true,
        ), // manifest schema_version (=1)
        // stats.* edge-funnel counts (lifted). Decade-spanning integer magnitudes ⇒ logm.
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_n_nodes",
            logm(1.0, 1e9),
            Empirical,
            true,
        ), // captured Ghost v2 tiles (60000)
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_n_lattice_cells",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // (spatial×phase×step) cells (232)
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_n_pairs_evaluated",
            logm(1.0, 1e12),
            Empirical,
            true,
        ), // candidate pairs (12,778,218)
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_n_pairs_phase_legal",
            logm(1.0, 1e12),
            Empirical,
            true,
        ), // phase-legal pairs (≤ evaluated)
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_n_pairs_aabb_overlap",
            logm(1.0, 1e12),
            Empirical,
            true,
        ), // AABB-overlap pairs (≤ evaluated)
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_n_directed_edges",
            logm(1.0, 1e12),
            Empirical,
            true,
        ), // accepted edges (≤ aabb_overlap; 10,047,212)
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_min_step_idx",
            logm(1.0, 1e10),
            Empirical,
            true,
        ), // producer step span lo (6101)
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_max_step_idx",
            logm(1.0, 1e10),
            Empirical,
            true,
        ), // producer step span hi (10273)
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_kernel_elapsed_ms",
            logm(1e-2, 1e7),
            Empirical,
            true,
        ), // 4D-intersection kernel ms (564.6)
        spec(
            F::PhaseManifoldEdgeValidation,
            "stats_host_elapsed_ms",
            logm(1e-2, 1e7),
            Empirical,
            true,
        ), // host-total ms (568.4)
        // config.* lattice parameters (lifted).
        spec(
            F::PhaseManifoldEdgeValidation,
            "config_so3_threshold",
            unit(),
            Empirical,
            true,
        ), // SO(3) acceptance threshold ∈[0,1] (0.75)
        spec(
            F::PhaseManifoldEdgeValidation,
            "config_spatial_cell_size_a",
            logm(1e-1, 1e3),
            Empirical,
            true,
        ), // spatial cell Å (5.0)
        spec(
            F::PhaseManifoldEdgeValidation,
            "config_max_temporal_edge_steps",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // max temporal edge step gap (500)
        spec(
            F::PhaseManifoldEdgeValidation,
            "config_step_bucket_size",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // temporal bucket size (500)
        // Derived component aggregates (invariant anchors).
        spec(
            F::PhaseManifoldEdgeValidation,
            "n_components",
            logm(1.0, 1e7),
            Empirical,
            true,
        ), // len(components) == routing-status n_components (2333)
        spec(
            F::PhaseManifoldEdgeValidation,
            "sum_component_n_nodes",
            logm(1.0, 1e9),
            Empirical,
            true,
        ), // Σ component.n_nodes == stats.n_nodes (node partition)
        spec(
            F::PhaseManifoldEdgeValidation,
            "total_intra_edges",
            logm(1.0, 1e12),
            Empirical,
            true,
        ), // Σ component.so3_manifold.n_intra_edges == stats.n_directed_edges
        // SO(3) coherence aggregates (real phase/coherence values ∈[-1,1]).
        spec(
            F::PhaseManifoldEdgeValidation,
            "mean_intra_component_mean_cosine",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::PhaseManifoldEdgeValidation,
            "min_intra_component_mean_cosine",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        spec(
            F::PhaseManifoldEdgeValidation,
            "max_intra_component_mean_cosine",
            lin(-1.0, 1.0),
            Empirical,
            true,
        ),
        // --- A47 SCIENTIFIC_CHRONOLOGY_EVENT (scientific_chronology_events.json records[]).
        //     First-class chronological event lane, not a DSTW transition wrapper.
        spec(
            F::ScientificChronologyEvent,
            "chronology_event_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "epoch_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "event_index",
            lin(0.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "event_kind",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "event_type",
            enom(32),
            Fixed(5),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "source_kind",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "artifact_family",
            enom(32),
            Fixed(5),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "parent_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "source_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "target_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "transition_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "stream_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "frame_idx",
            lin(0.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "timestep_min",
            lin(0.0, 1.0e12),
            Empirical,
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "timestep_max",
            lin(0.0, 1.0e12),
            Empirical,
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "physical_time_fs",
            lin(0.0, 1.0e15),
            Empirical,
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "delta_time_fs",
            lin(0.0, 1.0e15),
            Empirical,
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "operator_name",
            enom(32),
            Fixed(5),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "opcode_name",
            enom(32),
            Fixed(5),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "opcode_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "qnanl_packet_ref",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "operand_ref",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "state_hash",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "runtime_origin",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "runtime_native",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "chronology_fidelity_level",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "interferometric_mode",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "interferometric_marker_kind",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "chronology_marker_scope",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "perturbation_edge_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "twin_pair_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "spike_event_count",
            lin(0.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::ScientificChronologyEvent,
            "bocpd_event_count",
            lin(0.0, 1.0e9),
            Empirical,
            true,
        ),
        // --- A48 SCIENTIFIC_STATE_EPOCH (scientific_state_epochs.json records[]).
        //     First-class state interval lane, not a lifecycle vertex wrapper.
        spec(
            F::ScientificStateEpoch,
            "epoch_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "epoch_index",
            lin(0.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "epoch_kind",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "target_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "parent_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "source_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "target_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "start_event_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "end_event_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "lifecycle_stage",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "ligand_state",
            enom(4),
            Fixed(2),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "event_count",
            lin(0.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "stream_count",
            lin(0.0, 4096.0),
            Empirical,
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "timestep_min",
            lin(0.0, 1.0e12),
            Empirical,
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "timestep_max",
            lin(0.0, 1.0e12),
            Empirical,
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "physical_time_start_fs",
            lin(0.0, 1.0e15),
            Empirical,
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "physical_time_end_fs",
            lin(0.0, 1.0e15),
            Empirical,
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "physical_time_duration_fs",
            lin(0.0, 1.0e15),
            Empirical,
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "qnanl_packet_ref",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "operand_ref",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "state_hash",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "runtime_origin",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "runtime_native",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "chronology_fidelity_level",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "interferometric_mode",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "interferometric_marker_kind",
            enom(16),
            Fixed(4),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "chronology_marker_scope",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "closed_interval",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "perturbation_edge_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificStateEpoch,
            "twin_pair_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        // --- A49 T4_EPOCH_OVERLAY (dstw_t4_epoch_overlay.json).
        //     First-class DSTW-T4 epoch closeout overlay, routed as one top-level record.
        spec(F::T4EpochOverlay, "overlay_id", G::JoinKey, Fixed(0), true),
        spec(F::T4EpochOverlay, "epoch_id", G::JoinKey, Fixed(0), true),
        spec(F::T4EpochOverlay, "target_id", G::JoinKey, Fixed(0), true),
        spec(
            F::T4EpochOverlay,
            "source_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "target_vertex_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "transition_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "start_event_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "end_event_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(F::T4EpochOverlay, "overlay_kind", enom(8), Fixed(3), true),
        spec(F::T4EpochOverlay, "epoch_kind", enom(16), Fixed(4), true),
        spec(
            F::T4EpochOverlay,
            "lifecycle_stage",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(F::T4EpochOverlay, "ligand_state", enom(4), Fixed(2), true),
        spec(
            F::T4EpochOverlay,
            "event_count",
            lin(0.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "artifact_count",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "timestep_min",
            lin(0.0, 1.0e12),
            Empirical,
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "timestep_max",
            lin(0.0, 1.0e12),
            Empirical,
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "physical_time_start_fs",
            lin(0.0, 1.0e15),
            Empirical,
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "physical_time_end_fs",
            lin(0.0, 1.0e15),
            Empirical,
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "physical_time_duration_fs",
            lin(0.0, 1.0e15),
            Empirical,
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "qnanl_packet_ref",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(F::T4EpochOverlay, "operand_ref", G::JoinKey, Fixed(0), true),
        spec(F::T4EpochOverlay, "state_hash", G::JoinKey, Fixed(0), true),
        spec(F::T4EpochOverlay, "runtime_origin", enom(8), Fixed(3), true),
        spec(
            F::T4EpochOverlay,
            "runtime_native",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::T4EpochOverlay,
            "closed_interval",
            G::Boolean,
            Fixed(1),
            true,
        ),
        // --- A50 CHRONOLOGY_ACCEPTANCE_DECISION (chronology_acceptance_decision.json).
        //     First-class closeout gate result, routed as one top-level record.
        spec(
            F::ChronologyAcceptanceDecision,
            "acceptance_decision_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "epoch_overlay_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "epoch_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "target_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "decision",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "decision_basis",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "accepted",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "all_decode_proofs_pass",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "operand_ledger_record_count",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "artifact_count",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "qnanl_packet_count",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "decode_proof_count",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "decode_proof_pass_count",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "runtime_native_artifact_count",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "post_run_artifact_count",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "blocked_gate_count",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "lifecycle_stage",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "runtime_origin",
            enom(8),
            Fixed(3),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "runtime_native",
            G::Boolean,
            Fixed(1),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "qnanl_packet_ref",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "operand_ref",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ChronologyAcceptanceDecision,
            "state_hash",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        // --- A51 SCIENTIFIC_POCKET_OPENING_EVENT.
        //     Operator schema: ScientificPocketOpeningEvent. Top-level fields are preserved exactly;
        //     source unit aliases (`*_a3`, `*_a2`, `*_status`) are documented in
        //     SCIENTIFIC_SURFACE_FIELD_ALIASES and normalized before routing.
        spec(
            F::ScientificPocketOpeningEvent,
            "epoch_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificPocketOpeningEvent,
            "frame_event_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificPocketOpeningEvent,
            "residue_ids",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificPocketOpeningEvent,
            "pocket_volume",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ScientificPocketOpeningEvent,
            "pocket_surface_area",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ScientificPocketOpeningEvent,
            "lining_contacts",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificPocketOpeningEvent,
            "distance_gates",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificPocketOpeningEvent,
            "persistence_frames",
            lin(0.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::ScientificPocketOpeningEvent,
            "apo_holo_contrast",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ScientificPocketOpeningEvent,
            "confidence",
            unit(),
            LabelConfidence,
            true,
        ),
        // --- A52 LIGAND_CAUSAL_ACTIVITY_EVENT.
        //     Operator schema: LigandCausalActivityEvent. contacts[] and causal_signals{} are
        //     preserved as exact top-level fields and their named children route through the alias
        //     table above.
        spec(
            F::LigandCausalActivityEvent,
            "epoch_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "frame_event_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "ligand_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "atom_ids",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "pose_cluster_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "binding_site_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "contacts",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "contact_residue_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "contact_type",
            enom(5),
            Fixed(3),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "distance_a",
            lin(0.0, 64.0),
            Empirical,
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "duration_frames",
            lin(0.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "causal_signals",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "contact_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "pocket_volume_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "kcc_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "warp_tensor_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "force_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "residue_motion_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::LigandCausalActivityEvent,
            "confidence",
            unit(),
            LabelConfidence,
            true,
        ),
        // --- A53 RESIDUE_ACTIVITY_EVENT.
        //     Operator schema: ResidueActivityEvent. metrics{} is preserved exactly and its named
        //     children route through SCIENTIFIC_SURFACE_FIELD_ALIASES.
        spec(
            F::ResidueActivityEvent,
            "epoch_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "frame_event_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "chain_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "residue_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "residue_name",
            enom(AA_NAMES.len() as u32),
            Fixed(5),
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "activity_type",
            enom(7),
            Fixed(3),
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "metrics",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "sasa_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "rmsf_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "distance_gate_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "contact_map_delta",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "kcc_score",
            lin(-1.0e6, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "force_norm",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "displacement_a",
            lin(0.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "linked_ligand_event_id",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::ResidueActivityEvent,
            "confidence",
            unit(),
            LabelConfidence,
            true,
        ),
        // --- A54 SPECTRAL_EXTRACTOR.
        //     PRISMOS run-specific spectral geometry closure manifest
        //     (`DSTW_RUN_SPECIFIC_SPECTRAL_CLOSURE_MANIFEST.json`). The exporter flattens nested
        //     `summary`, `source_inputs`, `closure_outputs`, and `qnanl_source_state` members into
        //     these first-class fields. This is the by-value route for the multi-step spectral
        //     extractor: captured-frame contact graph -> physical profile -> normalized graph
        //     Laplacian eigenspectrum -> site-rank overlay / edge-validation / TWIN16 residual view.
        spec(
            F::SpectralExtractor,
            "schema_version",
            lin(0.0, 8.0),
            Fixed(2),
            true,
        ),
        spec(F::SpectralExtractor, "status", enom(2), Fixed(1), true),
        spec(F::SpectralExtractor, "run_root", G::JoinKey, Fixed(0), true),
        spec(
            F::SpectralExtractor,
            "materialization_root",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "output_root",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "spectral_basis",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "site_rank_policy",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "site_ranking_methodology",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "residue_count",
            logm(1.0, 1.0e7),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "physical_profile_rows",
            logm(1.0, 1.0e7),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "contact_edge_count",
            logm(1.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "contact_edges_rows",
            logm(1.0, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "materialized_site_count",
            logm(1.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "ranked_site_rows",
            logm(1.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "twin16_condition_cells",
            logm(1.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "twin16_pairwise_residuals",
            logm(1.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "twin16_nonzero_pairwise_residuals",
            logm(1.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "twin16_max_l1_residual",
            logm(1.0e-12, 1.0e9),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "blocking_issue_count",
            lin(0.0, 128.0),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "spectral_arrow_rows",
            logm(1.0, 1.0e6),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "laplacian_eigenvalue_0",
            lin(-1.0, 4.0),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "laplacian_eigenvalue_1",
            lin(-1.0, 4.0),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "laplacian_eigenvalue_2",
            lin(-1.0, 4.0),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "laplacian_eigenvalue_3",
            lin(-1.0, 4.0),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "laplacian_eigenvalue_4",
            lin(-1.0, 4.0),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "laplacian_eigenvalue_5",
            lin(-1.0, 4.0),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "laplacian_eigenvalue_6",
            lin(-1.0, 4.0),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "laplacian_eigenvalue_7",
            lin(-1.0, 4.0),
            Empirical,
            true,
        ),
        spec(
            F::SpectralExtractor,
            "spectral_arrow_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "physical_profile_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "contact_edges_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "materialized_sites_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "twin16_residuals_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "residue_coordinates_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "site_overlay_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "site_overlay_arrow_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "edge_validation_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "twin16_condition_view_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "twin16_pairwise_residuals_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
        spec(
            F::SpectralExtractor,
            "sha256_manifest_sha256",
            G::JoinKey,
            Fixed(0),
            true,
        ),
    ];
    fields.extend(scientific_surface_runtime_field_specs());
    for family in UNCERTAINTY_TENSOR_FAMILIES {
        fields.extend(uncertainty_tensor_specs(*family));
    }
    for family in TSH_TRANSITION_FAMILIES {
        fields.extend(tsh_transition_specs(*family));
    }
    for family in TSH_CONSENSUS_GATE_FAMILIES {
        fields.extend(tsh_consensus_gate_specs(*family));
    }
    for family in NATIVE_EPISTEMIC_SEQUENCE_FAMILIES {
        fields.extend(native_epistemic_sequence_specs(*family));
    }
    fields.extend(negative_scientific_evidence_specs());
    fields
}
