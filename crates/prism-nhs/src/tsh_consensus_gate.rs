//! Corpus-level TSH falsification consensus gate.
//!
//! This module is deliberately post-runtime and observational. It reads TSH Stage-II runtime
//! artifacts plus the canonical V6 negative-evidence bundle and materializes trainable
//! adjudication objects. It never mutates forces, timesteps, sampling, energy, or trajectories.

// Independent foliated ring contract. The active legacy adjudicator and
// corpus-level TSH gate below retain their existing representations.
pub use crate::biprism_adjudicator::{
    adjudicate_foliated_biprism_boundary, adjudicate_foliated_biprism_ring,
    validate_foliated_physical_acceptance, FoliatedBoundaryEvidence,
    FoliatedBoundaryExpectation, FoliatedConsensusError, FoliatedConsensusVerdict,
    FoliatedLeafEvidence, FoliatedPhysicalAcceptance, FoliatedPhysicalAcceptanceEvidence,
    FoliatedSiteContribution, NveEnergyWindow, PachnerTraceFluxObservation,
    CheckedFoliatedObservations, FoliatedEvidenceLevel,
};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Default minimum seed count required before persistent transition promotion can occur.
pub const DEFAULT_MIN_CONSENSUS_SEEDS: usize = 2;

/// Environment variable naming a spectral-recurrence policy document.
///
/// Resolution is explicit and receipted. When unset, the gate uses the
/// documented source literal
/// [`prism_nhs_default`](crate::tsh_stage_ii::DEFAULT_SPECTRAL_RECURRENCE_THRESHOLD)
/// and stamps `source_literal_default` — never `policy_file`.
pub const SPECTRAL_POLICY_ENV: &str = "PRISM_TSH_SPECTRAL_POLICY";

/// Schema version this build understands for a spectral policy document.
pub const SPECTRAL_POLICY_SCHEMA_VERSION: u64 = 1;

/// Where the resolved spectral-recurrence threshold came from.
///
/// The number alone is not auditable: `0.5` does not say whether it is the
/// documented source literal or an operator override.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpectralThresholdOrigin {
    /// [`crate::tsh_stage_ii::DEFAULT_SPECTRAL_RECURRENCE_THRESHOLD`].
    SourceLiteralDefault,
    /// A versioned policy document, sha256-recorded.
    PolicyFile,
    /// Set programmatically by the caller.
    Caller,
}

impl SpectralThresholdOrigin {
    /// Stable machine-readable label for receipts.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SourceLiteralDefault => "source_literal_default",
            Self::PolicyFile => "policy_file",
            Self::Caller => "caller",
        }
    }
}

/// Named terminal failures of spectral-policy resolution.
///
/// There is no variant meaning "carried on with a guessed value". A policy that
/// cannot be resolved is a hard stop, never a silent fallback to the default —
/// a silent fallback would let a receipt claim a policy origin for a value the
/// policy did not supply.
#[derive(Debug, thiserror::Error)]
pub enum SpectralPolicyError {
    /// The policy document could not be read.
    #[error("spectral policy file {path} could not be read: {source}")]
    Unreadable {
        /// The path that failed.
        path: String,
        /// The underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// The policy document is not valid JSON.
    #[error("spectral policy file {path} is not a valid policy document: {message}")]
    Malformed {
        /// The path that failed.
        path: String,
        /// The parser's message.
        message: String,
    },
    /// The document declares a schema version this build does not understand.
    #[error("spectral policy file {path} declares schema_version {found}, expected {expected}")]
    SchemaMismatch {
        /// The path that failed.
        path: String,
        /// What the document said.
        found: u64,
        /// What this build understands.
        expected: u64,
    },
    /// The document is present but carries no spectral threshold.
    #[error(
        "spectral policy file {path} declares no spectral_recurrence_threshold; \
         a policy that supplies nothing must not be recorded as the origin of anything"
    )]
    KeyAbsent {
        /// The path that failed.
        path: String,
    },
    /// The resolved threshold cannot bind any outcome.
    ///
    /// The score's measured range is [0, 1]. A threshold at or below 0 admits
    /// every measurement and a threshold above 1 rejects every one, so neither
    /// is a gate: the receipt would record a governing threshold that governs
    /// nothing.
    #[error(
        "spectral policy file {path} sets spectral_recurrence_threshold={value}, \
         which governs nothing: the score's measured range is [0, 1], so the \
         threshold must lie in (0, 1]"
    )]
    NonGoverningThreshold {
        /// The path that failed.
        path: String,
        /// The offending value.
        value: f64,
    },
}

/// A resolved spectral-recurrence threshold with its provenance.
#[derive(Clone, Debug)]
pub struct SpectralThresholdPolicy {
    /// The threshold a measured score must reach.
    pub threshold: f64,
    /// Where the value came from.
    pub origin: SpectralThresholdOrigin,
    /// Policy document path, when one was read.
    pub policy_path: Option<String>,
    /// sha256 of the policy document, when one was read.
    pub policy_sha256: Option<String>,
}

impl Default for SpectralThresholdPolicy {
    fn default() -> Self {
        Self {
            threshold: crate::tsh_stage_ii::DEFAULT_SPECTRAL_RECURRENCE_THRESHOLD,
            origin: SpectralThresholdOrigin::SourceLiteralDefault,
            policy_path: None,
            policy_sha256: None,
        }
    }
}

/// Resolve the spectral-recurrence threshold.
///
/// `path` of `None` yields the documented source literal with origin
/// `source_literal_default`. A supplied path must exist, parse, declare the
/// understood schema version, carry the key, and set a value that can actually
/// bind — otherwise this returns a named terminal error rather than quietly
/// reverting to the default under a `policy_file` label.
pub fn resolve_spectral_recurrence_threshold(
    path: Option<&Path>,
) -> Result<SpectralThresholdPolicy, SpectralPolicyError> {
    let Some(path) = path else {
        return Ok(SpectralThresholdPolicy::default());
    };
    let display = path.display().to_string();
    let bytes = std::fs::read(path).map_err(|source| SpectralPolicyError::Unreadable {
        path: display.clone(),
        source,
    })?;
    let document: Value =
        serde_json::from_slice(&bytes).map_err(|err| SpectralPolicyError::Malformed {
            path: display.clone(),
            message: err.to_string(),
        })?;
    let declared = document
        .get("schema_version")
        .and_then(Value::as_u64)
        .unwrap_or(SPECTRAL_POLICY_SCHEMA_VERSION);
    if declared != SPECTRAL_POLICY_SCHEMA_VERSION {
        return Err(SpectralPolicyError::SchemaMismatch {
            path: display,
            found: declared,
            expected: SPECTRAL_POLICY_SCHEMA_VERSION,
        });
    }
    let value = document
        .get("spectral_recurrence_threshold")
        .and_then(Value::as_f64)
        .ok_or_else(|| SpectralPolicyError::KeyAbsent {
            path: display.clone(),
        })?;
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(SpectralPolicyError::NonGoverningThreshold {
            path: display,
            value,
        });
    }
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(SpectralThresholdPolicy {
        threshold: value,
        origin: SpectralThresholdOrigin::PolicyFile,
        policy_path: Some(display),
        policy_sha256: Some(format!("{:x}", hasher.finalize())),
    })
}

/// Configuration for one consensus-gate materialization pass.
#[derive(Clone, Debug)]
pub struct ConsensusGateConfig {
    pub tsh_runtime_dir: PathBuf,
    pub negative_evidence_jsonl: PathBuf,
    pub output_dir: PathBuf,
    pub min_consensus_seeds: usize,
    pub positive_evidence_threshold: f64,
    pub uncertainty_magnitude_threshold: f64,
    pub topology_persistence_threshold: f64,
    pub chronology_persistence_threshold: f64,
    pub replicate_consensus_threshold: f64,
    pub falsification_survival_threshold: f64,
    /// Optional spectral-recurrence policy document. Defaults to whatever
    /// [`SPECTRAL_POLICY_ENV`] names, so the knob is reachable without a
    /// caller change and is still explicit.
    pub spectral_policy_path: Option<PathBuf>,
}

impl ConsensusGateConfig {
    pub fn new(
        tsh_runtime_dir: PathBuf,
        negative_evidence_jsonl: PathBuf,
        output_dir: PathBuf,
    ) -> Self {
        Self {
            tsh_runtime_dir,
            negative_evidence_jsonl,
            output_dir,
            min_consensus_seeds: DEFAULT_MIN_CONSENSUS_SEEDS,
            positive_evidence_threshold: 0.55,
            uncertainty_magnitude_threshold: 1.0,
            topology_persistence_threshold: 1.0,
            chronology_persistence_threshold: 1.0,
            replicate_consensus_threshold: 1.0,
            falsification_survival_threshold: 0.55,
            spectral_policy_path: std::env::var_os(SPECTRAL_POLICY_ENV).map(PathBuf::from),
        }
    }
}

/// Summary returned by a completed materialization pass.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsensusGateRunSummary {
    pub status: String,
    pub motif_count: usize,
    pub negative_event_count: usize,
    pub consensus_event_count: usize,
    pub falsification_profile_count: usize,
    pub persistent_transition_object_count: usize,
    pub rejected_candidate_count: usize,
    pub acceptance_ledger_path: PathBuf,
    pub tokenizer_manifest_path: PathBuf,
}

#[derive(Clone, Debug)]
struct Adjudication {
    consensus_event: Value,
    falsification_profile: Value,
    persistent_object: Option<Value>,
    rejected_candidate: Option<Value>,
    decision: String,
    decision_reason: String,
    rejected_by: String,
    /// The spectral threshold was actually compared against a measurement.
    spectral_evaluated: bool,
    /// That comparison bound the outcome.
    spectral_rejected: bool,
    /// No spectral measurement existed for this motif.
    spectral_not_measurable: bool,
}

/// Execute the post-runtime consensus/falsification materialization pass.
pub fn run_consensus_gate(config: &ConsensusGateConfig) -> Result<ConsensusGateRunSummary> {
    std::fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("create {}", config.output_dir.display()))?;

    let motifs_path = config.tsh_runtime_dir.join("transition_motifs.jsonl");
    let consensus_path = config
        .tsh_runtime_dir
        .join("transition_consensus_events.jsonl");
    let candidates_path = config
        .tsh_runtime_dir
        .join("transition_motif_candidates.jsonl");
    let spectral_policy =
        resolve_spectral_recurrence_threshold(config.spectral_policy_path.as_deref())
            .with_context(|| "resolve TSH spectral-recurrence gate policy")?;

    let mut motifs = load_jsonl(&motifs_path)?;
    if motifs.is_empty() {
        motifs = load_jsonl(&consensus_path)?;
    }
    if motifs.is_empty() {
        motifs = load_jsonl(&candidates_path)?;
    }
    if motifs.is_empty() {
        // C6 no-producer marker on the gate's own input side. "Zero motifs" is
        // ambiguous between "the Stage-II producer never ran" and "it ran and
        // promoted nothing", and the counts cannot carry the difference. Write
        // the distinction to disk before failing, so absence is self-describing
        // even on the failure path.
        write_input_producer_marker(config, &[&motifs_path, &consensus_path, &candidates_path])?;
        bail!(
            "no TSH motifs/candidates found in {}; salience producer status recorded in {}",
            config.tsh_runtime_dir.display(),
            config
                .output_dir
                .join("TSH_SALIENCE_PRODUCER_STATUS.json")
                .display()
        );
    }
    let negatives = load_jsonl(&config.negative_evidence_jsonl)?;
    if negatives.is_empty() {
        bail!(
            "no canonical negative evidence found in {}",
            config.negative_evidence_jsonl.display()
        );
    }

    let mut consensus_events = Vec::new();
    let mut falsification_profiles = Vec::new();
    let mut persistent_objects = Vec::new();
    let mut rejected_candidates = Vec::new();
    let mut rejected_by_negative = 0usize;
    let mut rejected_by_uncertainty = 0usize;
    let mut rejected_by_chronology = 0usize;
    let mut rejected_by_topology = 0usize;
    let mut rejected_by_consensus = 0usize;
    // Governance evidence for the spectral threshold. A threshold that is never
    // evaluated has not governed anything, and the receipt must say so rather
    // than claim a policy origin for an inert number.
    let mut spectral_evaluated = 0usize;
    let mut spectral_rejected = 0usize;
    let mut spectral_not_measurable = 0usize;

    for (idx, motif) in motifs.iter().enumerate() {
        let adjudication = adjudicate_motif(config, &spectral_policy, motif, &negatives, idx);
        match adjudication.rejected_by.as_str() {
            "negative_evidence" => rejected_by_negative += 1,
            "uncertainty" => rejected_by_uncertainty += 1,
            "chronology" => rejected_by_chronology += 1,
            "topology" => rejected_by_topology += 1,
            "consensus" => rejected_by_consensus += 1,
            _ => {}
        }
        if adjudication.spectral_evaluated {
            spectral_evaluated += 1;
        }
        if adjudication.spectral_rejected {
            spectral_rejected += 1;
        }
        if adjudication.spectral_not_measurable {
            spectral_not_measurable += 1;
        }
        consensus_events.push(adjudication.consensus_event);
        falsification_profiles.push(adjudication.falsification_profile);
        if let Some(object) = adjudication.persistent_object {
            persistent_objects.push(object);
        }
        if let Some(candidate) = adjudication.rejected_candidate {
            rejected_candidates.push(candidate);
        }
    }

    // A threshold that was never compared against a measurement has governed
    // nothing on this run. Recording `origin=policy_file` for such a value would
    // assert an authority the run never exercised, so the label is demoted and
    // the demotion is itself receipted.
    let spectral_gate_origin_label = if spectral_evaluated == 0 {
        format!("{}_declared_not_governing", spectral_policy.origin.as_str())
    } else {
        spectral_policy.origin.as_str().to_string()
    };

    let candidate_count = consensus_events.len();
    let accepted_count = persistent_objects.len();
    let rejected_count = rejected_candidates.len();
    let partial_count = consensus_events
        .iter()
        .filter(|event| str_value(event, "decision") == Some("PARTIAL".to_string()))
        .count();
    let final_decision = if accepted_count > 0 {
        "IMPLEMENTATION_VERIFIED"
    } else {
        "IMPLEMENTATION_PARTIAL"
    };
    let corpus_merge_blockers = if accepted_count == 0 {
        "no_persistent_transition_promoted"
    } else {
        "none"
    };
    let acceptance_ledger = json!({
        "schema_kind": "persistent_transition_acceptance_ledger",
        "schema_version": 1,
        "tokenizer_family": "PersistentTransitionAcceptanceLedger",
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "partial_count": partial_count,
        "promotion_rate": ratio(accepted_count, candidate_count),
        "rejection_rate": ratio(rejected_count, candidate_count),
        "false_positive_suppression_rate": ratio(rejected_count, candidate_count),
        "motifs_rejected_by_negative_evidence": rejected_by_negative,
        "motifs_rejected_by_uncertainty": rejected_by_uncertainty,
        "motifs_rejected_by_chronology": rejected_by_chronology,
        "motifs_rejected_by_topology": rejected_by_topology,
        "motifs_rejected_by_consensus": rejected_by_consensus,
        "motifs_promoted_to_persistent": accepted_count,
        "stage_iii_eligible_count": 0,
        "stage_iii_blocked_count": candidate_count,
        "corpus_merge_blockers": corpus_merge_blockers,
        "final_decision": final_decision,
        "spectral_recurrence_threshold": spectral_policy.threshold,
        "spectral_recurrence_threshold_origin": spectral_gate_origin_label,
        "spectral_recurrence_threshold_governed": spectral_evaluated > 0,
        "motifs_spectral_gate_evaluated": spectral_evaluated,
        "motifs_rejected_by_spectral_recurrence": spectral_rejected,
        "motifs_spectral_not_measurable": spectral_not_measurable,
        "provenance_ref": config.output_dir.display().to_string(),
    });
    let stage_iii_ledger = json!({
        "schema_kind": "stage_iii_eligibility_ledger",
        "schema_version": 1,
        "tokenizer_family": "PersistentTransitionAcceptanceLedger",
        "candidate_count": candidate_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "partial_count": partial_count,
        "promotion_rate": ratio(accepted_count, candidate_count),
        "rejection_rate": ratio(rejected_count, candidate_count),
        "false_positive_suppression_rate": ratio(rejected_count, candidate_count),
        "motifs_rejected_by_negative_evidence": rejected_by_negative,
        "motifs_rejected_by_uncertainty": rejected_by_uncertainty,
        "motifs_rejected_by_chronology": rejected_by_chronology,
        "motifs_rejected_by_topology": rejected_by_topology,
        "motifs_rejected_by_consensus": rejected_by_consensus,
        "motifs_promoted_to_persistent": accepted_count,
        "stage_iii_eligible_count": 0,
        "stage_iii_blocked_count": candidate_count,
        "corpus_merge_blockers": "stage_iii_not_enabled",
        "final_decision": "IMPLEMENTATION_PARTIAL",
        "provenance_ref": config.output_dir.display().to_string(),
    });

    let consensus_path = config
        .output_dir
        .join("SCIENTIFIC_TRANSITION_CONSENSUS_EVENTS.jsonl");
    let falsification_path = config
        .output_dir
        .join("TRANSITION_FALSIFICATION_PROFILES.jsonl");
    let persistent_path = config
        .output_dir
        .join("PERSISTENT_TRANSITION_OBJECTS.jsonl");
    let rejected_path = config
        .output_dir
        .join("REJECTED_PERSISTENT_TRANSITION_CANDIDATES.jsonl");
    let acceptance_path = config
        .output_dir
        .join("PERSISTENT_TRANSITION_ACCEPTANCE_LEDGER.json");
    let stage_iii_path = config.output_dir.join("STAGE_III_ELIGIBILITY_LEDGER.json");
    write_jsonl(&consensus_path, &consensus_events)?;
    write_jsonl(&falsification_path, &falsification_profiles)?;
    write_jsonl(&persistent_path, &persistent_objects)?;
    write_jsonl(&rejected_path, &rejected_candidates)?;
    write_json_pretty(&acceptance_path, &acceptance_ledger)?;
    write_json_pretty(&stage_iii_path, &stage_iii_ledger)?;

    let input_coverage = json!({
        "schema_kind": "tsh_falsification_consensus_input_coverage",
        "schema_version": 1,
        "tsh_runtime_dir": config.tsh_runtime_dir,
        "negative_evidence_jsonl": config.negative_evidence_jsonl,
        "motif_count": motifs.len(),
        "negative_event_count": negatives.len(),
        "required_inputs_present": true,
        "new_md_executed": false,
    });
    write_json_pretty(
        &config.output_dir.join("INPUT_COVERAGE_LEDGER.json"),
        &input_coverage,
    )?;

    // Standalone provenance for the one gate whose value is policy-resolved.
    let spectral_gate_policy = json!({
        "schema_kind": "tsh_spectral_recurrence_gate_policy",
        "schema_version": 1,
        "metric_kind": "normalized_autocorrelation_signature_l1",
        "metric_units": "dimensionless_ratio_of_second_moments",
        "metric_invariance": "invariant_under_affine_rescaling_x_to_a_x_plus_b_for_a_nonzero",
        "metric_range": [0.0, 1.0],
        "threshold": spectral_policy.threshold,
        "threshold_origin": spectral_gate_origin_label,
        "threshold_origin_resolved": spectral_policy.origin.as_str(),
        "threshold_source_literal_default":
            crate::tsh_stage_ii::DEFAULT_SPECTRAL_RECURRENCE_THRESHOLD,
        "threshold_policy_path": spectral_policy.policy_path,
        "threshold_policy_sha256": spectral_policy.policy_sha256,
        "threshold_policy_env": SPECTRAL_POLICY_ENV,
        "threshold_admissible_interval": "(0.0, 1.0]",
        "threshold_governed_this_run": spectral_evaluated > 0,
        "motifs_evaluated": spectral_evaluated,
        "motifs_rejected": spectral_rejected,
        "motifs_not_measurable": spectral_not_measurable,
        "motif_count": motifs.len(),
        "provenance_ref": config.output_dir.display().to_string(),
    });
    write_json_pretty(
        &config
            .output_dir
            .join("SPECTRAL_RECURRENCE_GATE_POLICY.json"),
        &spectral_gate_policy,
    )?;

    let manifest = json!([
        {"source_id":"scientific_transition_consensus_events","jsonl_records_path": consensus_path.display().to_string(),"family":"ScientificTransitionConsensusEvent"},
        {"source_id":"transition_falsification_profiles","jsonl_records_path": falsification_path.display().to_string(),"family":"TransitionFalsificationProfile"},
        {"source_id":"persistent_transition_objects","jsonl_records_path": persistent_path.display().to_string(),"family":"PersistentTransitionObject"},
        {"source_id":"rejected_persistent_transition_candidates","jsonl_records_path": rejected_path.display().to_string(),"family":"RejectedPersistentTransitionCandidate"},
        {"source_id":"persistent_transition_acceptance_ledger","json_path": acceptance_path.display().to_string(),"family":"PersistentTransitionAcceptanceLedger"},
        {"source_id":"stage_iii_eligibility_ledger","json_path": stage_iii_path.display().to_string(),"family":"PersistentTransitionAcceptanceLedger"}
    ]);
    let manifest_path = config.output_dir.join("TOKENIZER_MANIFEST.json");
    write_json_pretty(&manifest_path, &manifest)?;

    Ok(ConsensusGateRunSummary {
        status: final_decision.to_string(),
        motif_count: motifs.len(),
        negative_event_count: negatives.len(),
        consensus_event_count: consensus_events.len(),
        falsification_profile_count: falsification_profiles.len(),
        persistent_transition_object_count: persistent_objects.len(),
        rejected_candidate_count: rejected_candidates.len(),
        acceptance_ledger_path: acceptance_path,
        tokenizer_manifest_path: manifest_path,
    })
}

fn adjudicate_motif(
    config: &ConsensusGateConfig,
    spectral_policy: &SpectralThresholdPolicy,
    motif: &Value,
    negatives: &[Value],
    index: usize,
) -> Adjudication {
    let motif_id = nonempty(motif, "motif_id")
        .or_else(|| nonempty(motif, "transition_id"))
        .unwrap_or_else(|| format!("motif_{index:04}"));
    let transition_id = nonempty(motif, "transition_id").unwrap_or_else(|| motif_id.clone());
    let target_id = nonempty(motif, "target_id").unwrap_or_else(|| "unknown_target".to_string());
    let seed_id = nonempty(motif, "seed_id").unwrap_or_else(|| "unknown_seed".to_string());
    let seed_digits = digits_only(&seed_id);
    let mode = nonempty(motif, "mode")
        .or_else(|| infer_mode_from_refs(motif))
        .unwrap_or_else(|| "unknown_mode".to_string());
    let row_id = nonempty(motif, "row_id").unwrap_or_else(|| "unknown_row".to_string());
    let negative_matches =
        matching_negative_events(&motif_id, &target_id, &seed_digits, &mode, negatives);
    let strongest = negative_matches.iter().copied().max_by(|a, b| {
        negative_strength(a)
            .partial_cmp(&negative_strength(b))
            .unwrap()
    });
    let strongest_ref = strongest
        .and_then(|v| nonempty(v, "negative_event_id"))
        .unwrap_or_else(|| "NO_DIRECT_NEGATIVE_MATCH".to_string());
    let negative_refs = join_limited(
        negative_matches
            .iter()
            .filter_map(|v| nonempty(v, "negative_event_id")),
        48,
    );
    let negative_classes = join_unique_limited(
        negative_matches
            .iter()
            .filter_map(|v| nonempty(v, "rejection_class")),
        32,
    );
    let negative_reasons = join_unique_limited(
        negative_matches
            .iter()
            .filter_map(|v| nonempty(v, "rejection_reason")),
        32,
    );
    let negative_score = if negative_matches.is_empty() {
        0.0
    } else {
        let count_score = ((negative_matches.len() as f64 + 1.0).log10() / 3.0).clamp(0.0, 1.0);
        strongest
            .map(negative_strength)
            .unwrap_or(0.0)
            .max(count_score)
    };
    let positive_score = f64_value(motif, "scientific_evidence_score")
        .or_else(|| f64_value(motif, "aggregate_salience_score"))
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let uncertainty_mag = f64_value(motif, "uncertainty_normalized_magnitude").unwrap_or(0.0);
    let topology_score = f64_value(motif, "topology_persistence_score")
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let chronology_score = f64_value(motif, "chronology_persistence_score")
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    // Spectral recurrence is measured or it is not. A `null` score, a
    // non-`measured` status, or an absent field all mean "no measurement", and
    // none of them may be silently read as the number 0.0 — that would turn an
    // absence into a falsification.
    let spectral_status = nonempty(motif, "spectral_signature_status").unwrap_or_else(|| {
        match f64_value(motif, "spectral_similarity_score") {
            Some(value) if value.is_finite() => "measured_unlabeled_legacy_record".to_string(),
            _ => "not_measurable_absent_from_record".to_string(),
        }
    });
    let spectral_score = f64_value(motif, "spectral_similarity_score")
        .filter(|value| value.is_finite())
        .map(|value| value.clamp(0.0, 1.0));
    let spectral_measured = matches!(
        spectral_status.as_str(),
        "measured" | "measured_unlabeled_legacy_record"
    ) && spectral_score.is_some();
    let spectral_score_for_confidence = spectral_score.unwrap_or(0.0);
    let consensus_seed_count = f64_value(motif, "consensus_seed_count")
        .or_else(|| f64_value(motif, "replicate_count"))
        .unwrap_or(1.0);
    let replicate_consensus_score =
        (consensus_seed_count / config.min_consensus_seeds.max(1) as f64).clamp(0.0, 1.0);
    let falsification_survival_score = (1.0 - negative_score).clamp(0.0, 1.0);
    let uncertainty_refs = refs_string(
        motif,
        &[
            "uncertainty_tensor_refs",
            "scientific_uncertainty_tensor_ref",
        ],
    );
    let chronology_refs = refs_string(motif, &["chronology_ref", "chronology_event_id"]);
    let positive_refs = refs_string(motif, &["source_refs", "source_sha256", "transition_id"]);
    let source_sha256 =
        nonempty(motif, "source_sha256").unwrap_or_else(|| "source_sha256_unavailable".to_string());
    let topology_hash = nonempty(motif, "topology_hash_transition")
        .unwrap_or_else(|| "topology_hash_unavailable".to_string());
    let spectral_hash = nonempty(motif, "spectral_signature_hash")
        .unwrap_or_else(|| "spectral_signature_unavailable".to_string());
    let contact_support =
        nonempty(motif, "contact_support_observable").unwrap_or_else(|| "kcc".to_string());
    let opcode_context = strongest
        .and_then(|v| nonempty(v, "opcode_context"))
        .unwrap_or_else(|| "scientific_consensus_gate:TSH_FALSIFICATION_CONSENSUS".to_string());
    let prismcode_context = strongest
        .and_then(|v| nonempty(v, "prismcode_instruction_context"))
        .unwrap_or_else(|| {
            format!("ASSERT.TSH_FALSIFICATION_CONSENSUS|{motif_id}|{source_sha256}")
        });
    let qnanl_context = strongest
        .and_then(|v| nonempty(v, "qnanl_codec_context"))
        .unwrap_or_else(|| {
            "QnanlArtifactFamily::EventStream:ScientificTransitionConsensusEvent".to_string()
        });

    let missing_required_refs = uncertainty_refs.is_empty()
        || chronology_refs.is_empty()
        || positive_refs.is_empty()
        || source_sha256 == "source_sha256_unavailable";
    // Lexicographic order. Provenance and the physical/falsification gates are
    // strictly senior to the spectral gate, and a gate that cannot be evaluated
    // never promotes: it yields PARTIAL, which is neither a pass nor a claim of
    // falsification. No downstream salience can override a senior REJECT.
    let mut spectral_evaluated = false;
    let mut spectral_rejected = false;
    let mut spectral_not_measurable = false;
    // Senior band: provenance and the physical/falsification prohibitions. None
    // of these can be overridden by any downstream salience quantity.
    let senior = if missing_required_refs {
        Some(("PARTIAL", "missing_required_provenance", "provenance"))
    } else if negative_score >= positive_score && negative_score >= 0.55 {
        Some(("REJECT", "stronger_negative_evidence", "negative_evidence"))
    } else if uncertainty_mag < config.uncertainty_magnitude_threshold {
        Some(("REJECT", "below_uncertainty_floor", "uncertainty"))
    } else if topology_score < config.topology_persistence_threshold {
        Some(("REJECT", "topology_persistence_failed", "topology"))
    } else if chronology_score < config.chronology_persistence_threshold {
        Some(("REJECT", "chronology_persistence_failed", "chronology"))
    } else {
        None
    };
    let (decision, decision_reason, rejected_by) = match senior {
        Some(outcome) => outcome,
        // No measurement is not a falsification. PARTIAL withholds promotion
        // without asserting the motif failed a comparison that never happened.
        None if !spectral_measured => {
            spectral_not_measurable = true;
            (
                "PARTIAL",
                "spectral_recurrence_not_measurable",
                "spectral_not_measurable",
            )
        }
        None => {
            spectral_evaluated = true;
            if spectral_score_for_confidence < spectral_policy.threshold {
                spectral_rejected = true;
                ("REJECT", "spectral_recurrence_failed", "consensus")
            } else if replicate_consensus_score < config.replicate_consensus_threshold {
                ("REJECT", "insufficient_consensus", "consensus")
            } else if falsification_survival_score < config.falsification_survival_threshold {
                ("REJECT", "stronger_negative_evidence", "negative_evidence")
            } else if positive_score < config.positive_evidence_threshold {
                (
                    "PARTIAL",
                    "negative_evidence_gate_passed_no_promotion",
                    "positive_evidence",
                )
            } else {
                ("ACCEPT", "survived_falsification_and_consensus", "none")
            }
        }
    };
    // An unmeasured spectral component is omitted from the confidence mean
    // rather than entered as a number.
    let mut confidence_terms = vec![
        positive_score,
        topology_score,
        chronology_score,
        replicate_consensus_score,
        falsification_survival_score,
    ];
    if spectral_measured {
        confidence_terms.push(spectral_score_for_confidence);
    }
    let confidence = mean(&confidence_terms);
    let provenance_ref = format!("{}#motif={motif_id}", config.output_dir.display());
    // Everything a consumer needs to interpret the spectral verdict without
    // inferring anything: the status, whether it was a measurement, the metric's
    // declared units and invariance, and the threshold with its provenance.
    let spectral_evidence = json!({
        "score": spectral_score,
        "status": spectral_status,
        "measured": spectral_measured,
        "threshold": spectral_policy.threshold,
        "threshold_origin": spectral_policy.origin.as_str(),
        "threshold_policy_path": spectral_policy.policy_path,
        "threshold_policy_sha256": spectral_policy.policy_sha256,
        "threshold_evaluated": spectral_evaluated,
        "metric_kind": nonempty(motif, "spectral_metric_kind"),
        "metric_units": nonempty(motif, "spectral_metric_units"),
        "metric_invariance": nonempty(motif, "spectral_metric_invariance"),
        "lags": f64_value(motif, "spectral_signature_lags"),
    });
    let consensus_event = json!({
        "schema_kind": "scientific_transition_consensus_event",
        "schema_version": 1,
        "tokenizer_family": "ScientificTransitionConsensusEvent",
        "motif_id": motif_id,
        "transition_id": transition_id,
        "row_id": row_id,
        "mode": mode,
        "seed_set": seed_id,
        "target_id": target_id,
        "positive_evidence_refs": positive_refs,
        "negative_evidence_refs": negative_refs,
        "uncertainty_tensor_refs": uncertainty_refs,
        "chronology_refs": chronology_refs,
        "topology_hashes": topology_hash,
        "spectral_signature_refs": spectral_hash,
        "causal_geometry_refs": refs_string(motif, &["source_refs", "causal_geometry_refs"]),
        "residue_network_refs": nonempty(motif, "residue_set_hash").unwrap_or_else(|| "residue_network_unavailable".to_string()),
        "contact_support_observable": contact_support,
        "contact_support_semantic_equivalence": false,
        "positive_evidence_score": positive_score,
        "negative_evidence_score": negative_score,
        "uncertainty_normalized_magnitude": uncertainty_mag,
        "topology_persistence_score": topology_score,
        "chronology_persistence_score": chronology_score,
        // Non-finite/absent stays `null`: out of band for a [0, 1] score, so a
        // consumer can never mistake "not measured" for "measured low".
        "spectral_recurrence_score": spectral_score,
        "spectral_recurrence": spectral_evidence,
        "replicate_consensus_score": replicate_consensus_score,
        "falsification_survival_score": falsification_survival_score,
        "decision": decision,
        "decision_reason": decision_reason,
        "confidence": confidence,
        "provenance_ref": provenance_ref,
        "opcode_context": opcode_context,
        "prismcode_instruction_context": prismcode_context,
        "qnanl_codec_context": qnanl_context,
        "stage_iii_eligible": false,
    });

    let falsification_profile = json!({
        "schema_kind": "transition_falsification_profile",
        "schema_version": 1,
        "tokenizer_family": "TransitionFalsificationProfile",
        "motif_id": motif_id,
        "matched_negative_evidence_count": negative_matches.len(),
        "strongest_negative_match": strongest_ref,
        "negative_match_classes": negative_classes,
        "negative_match_reasons": negative_reasons,
        "negative_match_topology_overlap": overlap_score(&topology_hash, strongest, "topology_sha256"),
        "negative_match_residue_overlap": overlap_score(&nonempty(motif, "residue_set_hash").unwrap_or_default(), strongest, "residue_id"),
        "negative_match_chronology_overlap": overlap_score(&chronology_refs, strongest, "chronology_ref"),
        "negative_match_spectral_overlap": overlap_score(&spectral_hash, strongest, "spectral_state_id"),
        "rejected_by_falsification": decision == "REJECT" && rejected_by == "negative_evidence",
        "survived_falsification": decision == "ACCEPT",
        "falsification_decision_reason": decision_reason,
        "negative_evidence_refs": negative_refs,
        "provenance_ref": provenance_ref,
        "source_sha256": source_sha256,
        "opcode_context": opcode_context,
        "prismcode_instruction_context": prismcode_context,
        "qnanl_codec_context": qnanl_context,
    });

    let persistent_object = (decision == "ACCEPT").then(|| {
        json!({
            "schema_kind": "persistent_transition_object",
            "schema_version": 1,
            "tokenizer_family": "PersistentTransitionObject",
            "persistent_transition_object_id": format!("pto_consensus:{motif_id}"),
            "source_motif_id": motif_id,
            "transition_family": "TSH_STAGE_II_OBSERVATIONAL_CONSENSUS",
            "accepted_status": "ACCEPT",
            "accepted_reason": decision_reason,
            "support_seed_count": consensus_seed_count,
            "rejection_seed_count": distinct_negative_seeds(&negative_matches),
            "supporting_rows": row_id,
            "opposing_rows": join_unique_limited(negative_matches.iter().filter_map(|v| nonempty(v, "row_id")), 32),
            "uncertainty_envelope": uncertainty_refs,
            "topology_persistence": topology_score,
            "chronology_persistence": chronology_score,
            "spectral_signature": spectral_hash,
            "causal_pathway": refs_string(motif, &["causal_lag", "lag_corr_peak"]),
            "residue_network": nonempty(motif, "residue_set_hash").unwrap_or_else(|| "residue_network_unavailable".to_string()),
            "contact_support_observable": contact_support,
            "negative_evidence_refs_considered": negative_refs,
            "negative_evidence_refs_survived": negative_refs,
            "promotion_confidence": confidence,
            "stage_iii_eligible": false,
            "provenance_ref": provenance_ref,
        })
    });
    let rejected_candidate = (decision != "ACCEPT").then(|| {
        json!({
            "schema_kind": "rejected_persistent_transition_candidate",
            "schema_version": 1,
            "tokenizer_family": "RejectedPersistentTransitionCandidate",
            "candidate_id": format!("rejected_ptc:{motif_id}"),
            "source_motif_id": motif_id,
            "rejection_status": decision,
            "rejection_reason": decision_reason,
            "failed_gate": decision_reason,
            "strongest_negative_evidence_ref": strongest_ref,
            "positive_evidence_refs": positive_refs,
            "negative_evidence_refs": negative_refs,
            "uncertainty_tensor_refs": uncertainty_refs,
            "chronology_refs": chronology_refs,
            "topology_hashes": topology_hash,
            "tokenizable_failure_mode": decision_reason,
            "provenance_ref": provenance_ref,
            "opcode_context": opcode_context,
            "prismcode_instruction_context": prismcode_context,
            "qnanl_codec_context": qnanl_context,
            "confidence": confidence,
        })
    });

    Adjudication {
        consensus_event,
        falsification_profile,
        persistent_object,
        rejected_candidate,
        decision: decision.to_string(),
        decision_reason: decision_reason.to_string(),
        rejected_by: rejected_by.to_string(),
        spectral_evaluated,
        spectral_rejected,
        spectral_not_measurable,
    }
}

/// Write the C6 no-producer marker for the gate's own input side.
///
/// The three Stage-II input files are checked independently: all absent means
/// the producer never wrote anything, while present-but-empty means it ran and
/// promoted nothing. A count of zero motifs cannot distinguish those, so the
/// distinction is written as a typed status instead.
fn write_input_producer_marker(config: &ConsensusGateConfig, inputs: &[&Path]) -> Result<()> {
    let mut present = Vec::new();
    let mut absent = Vec::new();
    let mut present_empty = Vec::new();
    for path in inputs {
        let display = path.display().to_string();
        if path.exists() {
            let empty = load_jsonl(path)
                .map(|records| records.is_empty())
                .unwrap_or(true);
            if empty {
                present_empty.push(display.clone());
            }
            present.push(display);
        } else {
            absent.push(display);
        }
    }
    let (status, reason) = if present.is_empty() {
        (
            "producer_not_wired",
            "no TSH Stage-II motif, consensus, or candidate file exists: the salience producer never wrote to this directory",
        )
    } else if present.len() == present_empty.len() {
        (
            "producer_wired_zero_events",
            "TSH Stage-II files exist and are empty: the producer ran and promoted nothing; absence here is a result, not a wiring gap",
        )
    } else {
        (
            "producer_wired_unparsed_records",
            "TSH Stage-II files exist with records that did not parse as JSONL objects",
        )
    };
    let marker = json!({
        "schema_kind": "tsh_salience_producer_status",
        "schema_version": 1,
        "producer_status": status,
        "producer_wired": status != "producer_not_wired",
        "producer_status_reason": reason,
        "observed_scope": "tsh_consensus_gate_input",
        "tsh_runtime_dir": config.tsh_runtime_dir.display().to_string(),
        "input_files_present": present,
        "input_files_absent": absent,
        "input_files_present_but_empty": present_empty,
        "motif_count": 0,
    });
    write_json_pretty(
        &config.output_dir.join("TSH_SALIENCE_PRODUCER_STATUS.json"),
        &marker,
    )
}

fn matching_negative_events<'a>(
    motif_id: &str,
    target_id: &str,
    seed_digits: &str,
    mode: &str,
    negatives: &'a [Value],
) -> Vec<&'a Value> {
    let normalized_target = normalize_target_id(target_id);
    negatives
        .iter()
        .filter(|event| {
            let event_target = nonempty(event, "target_id")
                .map(|target| normalize_target_id(&target))
                .unwrap_or_default();
            let event_seed_digits = nonempty(event, "replica_seed")
                .map(|seed| digits_only(&seed))
                .unwrap_or_default();
            nonempty(event, "motif_id")
                .filter(|candidate| !candidate.is_empty() && candidate == motif_id)
                .is_some()
                || (!normalized_target.is_empty()
                    && event_target == normalized_target
                    && !seed_digits.is_empty()
                    && event_seed_digits == seed_digits)
                || (mode != "unknown_mode"
                    && (nonempty(event, "ligand_mode").as_deref() == Some(mode)
                        || nonempty(event, "detection_mode").as_deref() == Some(mode)))
        })
        .collect()
}

fn normalize_target_id(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .replace("_clean", "")
        .replace(".topology", "")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect()
}

fn digits_only(value: &str) -> String {
    value.chars().filter(|c| c.is_ascii_digit()).collect()
}

fn negative_strength(value: &Value) -> f64 {
    f64_value(value, "severity")
        .or_else(|| f64_value(value, "confidence"))
        .unwrap_or(0.5)
        .clamp(0.0, 1.0)
        .max((f64_value(value, "occurrence_count").unwrap_or(1.0).log10() / 3.0).clamp(0.0, 1.0))
}

fn distinct_negative_seeds(matches: &[&Value]) -> usize {
    matches
        .iter()
        .filter_map(|v| nonempty(v, "replica_seed"))
        .collect::<BTreeSet<_>>()
        .len()
}

fn infer_mode_from_refs(value: &Value) -> Option<String> {
    refs_string(value, &["source_refs", "chronology_ref"])
        .split(';')
        .find_map(|s| {
            let lower = s.to_ascii_lowercase();
            if lower.contains("holo") && lower.contains("cryptic") {
                Some("holo_gdp_cryptic".to_string())
            } else if lower.contains("holo") && lower.contains("orthosteric") {
                Some("holo_gdp_orthosteric".to_string())
            } else if lower.contains("apo") && lower.contains("cryptic") {
                Some("apo_cryptic".to_string())
            } else if lower.contains("apo") && lower.contains("orthosteric") {
                Some("apo_orthosteric".to_string())
            } else {
                None
            }
        })
}

fn overlap_score(probe: &str, value: Option<&Value>, field: &str) -> f64 {
    let Some(value) = value else { return 0.0 };
    let Some(candidate) = nonempty(value, field) else {
        return 0.0;
    };
    if !probe.is_empty() && probe.contains(&candidate)
        || !candidate.is_empty() && candidate.contains(probe)
    {
        1.0
    } else {
        0.0
    }
}

fn ratio(n: usize, d: usize) -> f64 {
    if d == 0 {
        0.0
    } else {
        n as f64 / d as f64
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().copied().sum::<f64>() / values.len().max(1) as f64
}

fn load_jsonl(path: &Path) -> Result<Vec<Value>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line.with_context(|| format!("read {}", path.display()))?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value =
            serde_json::from_str(&line).with_context(|| format!("parse {}", path.display()))?;
        values.push(value);
    }
    Ok(values)
}

fn write_jsonl(path: &Path, values: &[Value]) -> Result<()> {
    let mut file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    for value in values {
        serde_json::to_writer(&mut file, value)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

fn write_json_pretty(path: &Path, value: &Value) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

fn refs_string(value: &Value, keys: &[&str]) -> String {
    let mut refs = Vec::new();
    for key in keys {
        if let Some(v) = value.get(*key) {
            collect_refs(v, &mut refs);
        }
    }
    refs.sort();
    refs.dedup();
    refs.join(";")
}

fn collect_refs(value: &Value, refs: &mut Vec<String>) {
    match value {
        Value::String(s) if !s.trim().is_empty() => refs.push(s.clone()),
        Value::Number(n) => refs.push(n.to_string()),
        Value::Bool(b) => refs.push(b.to_string()),
        Value::Array(items) => {
            for item in items {
                collect_refs(item, refs);
            }
        }
        Value::Object(map) => {
            for key in ["id", "path", "sha256", "tensor_id", "record_id"] {
                if let Some(v) = map.get(key) {
                    collect_refs(v, refs);
                }
            }
        }
        _ => {}
    }
}

fn nonempty(value: &Value, key: &str) -> Option<String> {
    match value.get(key)? {
        Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

fn str_value(value: &Value, key: &str) -> Option<String> {
    nonempty(value, key)
}

fn f64_value(value: &Value, key: &str) -> Option<f64> {
    match value.get(key)? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    }
}

fn join_limited<I>(items: I, limit: usize) -> String
where
    I: IntoIterator<Item = String>,
{
    items.into_iter().take(limit).collect::<Vec<_>>().join(";")
}

fn join_unique_limited<I>(items: I, limit: usize) -> String
where
    I: IntoIterator<Item = String>,
{
    let mut values = items.into_iter().collect::<BTreeSet<_>>();
    values
        .iter()
        .take(limit)
        .cloned()
        .collect::<Vec<_>>()
        .join(";")
}

/// SHA256 helper for closure ledgers.
pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher).with_context(|| format!("hash {}", path.display()))?;
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn rejects_single_seed_motif_against_canonical_negative_event() {
        let tmp = tempdir().unwrap();
        let tsh_dir = tmp.path().join("tsh");
        std::fs::create_dir_all(&tsh_dir).unwrap();
        let motif = json!({
            "schema_kind": "transition_motif",
            "transition_id": "t1",
            "motif_id": "m1",
            "target_id": "4LPK",
            "seed_id": "seed_7101",
            "source_sha256": "abc",
            "source_refs": ["runtime.jsonl"],
            "uncertainty_tensor_refs": ["sut1"],
            "chronology_ref": "epoch1",
            "topology_hash_transition": "topo:a->a",
            "residue_set_hash": "residue:1",
            "spectral_signature_hash": "spec:1",
            "scientific_evidence_score": 0.9,
            "uncertainty_normalized_magnitude": 4.0,
            "topology_persistence_score": 1.0,
            "chronology_persistence_score": 1.0,
            "spectral_similarity_score": 0.8,
            "consensus_seed_count": 1
        });
        write_jsonl(&tsh_dir.join("transition_motifs.jsonl"), &[motif]).unwrap();
        write_jsonl(&tsh_dir.join("transition_consensus_events.jsonl"), &[]).unwrap();
        write_jsonl(&tsh_dir.join("transition_motif_candidates.jsonl"), &[]).unwrap();
        let neg_path = tmp.path().join("negative.jsonl");
        let negative = json!({
            "schema_kind": "negative_scientific_evidence_event",
            "negative_event_id": "n1",
            "motif_id": "m1",
            "target_id": "4LPK",
            "replica_seed": "7101",
            "severity": 0.95,
            "rejection_class": "uncertainty_rejection",
            "rejection_reason": "falls_below_empirical_uncertainty_threshold",
            "opcode_context": "scientific_falsification:SCIENTIFIC_UNCERTAINTY_GATE",
            "prismcode_instruction_context": "ASSERT.SCIENTIFIC_UNCERTAINTY|n1",
            "qnanl_codec_context": "QnanlArtifactFamily::ScientificUncertaintyTensor:NegativeScientificEvidenceEvent"
        });
        write_jsonl(&neg_path, &[negative]).unwrap();
        let out = tmp.path().join("out");
        let summary =
            run_consensus_gate(&ConsensusGateConfig::new(tsh_dir, neg_path, out.clone())).unwrap();
        assert_eq!(summary.status, "IMPLEMENTATION_PARTIAL");
        assert_eq!(summary.persistent_transition_object_count, 0);
        assert_eq!(summary.rejected_candidate_count, 1);
        let events = load_jsonl(&out.join("SCIENTIFIC_TRANSITION_CONSENSUS_EVENTS.jsonl")).unwrap();
        assert_eq!(str_value(&events[0], "decision").as_deref(), Some("REJECT"));
        assert_eq!(
            str_value(&events[0], "decision_reason").as_deref(),
            Some("stronger_negative_evidence")
        );
    }
}
